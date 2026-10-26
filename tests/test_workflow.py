"""Fixed budgets, frozen TEST evaluation, recovery and safe outputs."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.runtime import configure_runtime

configure_runtime()
import numpy as np
import pandas as pd
import io
import json
from contextlib import redirect_stdout
from unittest.mock import Mock
from src.config import load_config, output_paths, validate_search
from src.data import SPACE_NAMES
from src.experiments import StudyChangedError, _prune_models, run, tune


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix='.test-', dir=Path(__file__).resolve().parents[1]
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config('config/config.yaml')
        raw = self.root / 'raw.h5'
        raw.write_text('synthetic input', encoding='utf-8')
        self.config['data'].update(path=str(raw), pre=2, horizon=2, correct_power=False)
        self.config['tuning'].update(
            trials_per_space=2, startup_trials=1, pre_choices=[0, 2, 4]
        )
        self.config['study'].update(min_test_origins=5, bootstrap_resamples=1000)
        self.config['output_dir'] = str(self.root / 'outputs')
        n = 240
        self.frame = pd.DataFrame(
            {
                'P_Solar[kW]': np.linspace(0, 5, n),
                'POA Irr[kW1m2]': np.linspace(0.1, 1, n),
                'GHI[kW1m2]': np.linspace(0.1, 0.9, n),
                'TEMPERATURE[degC]': np.full(n, 20.0),
                'WIND_SPEED[m1s]': np.full(n, 2.0),
                'WIND_DIR[deg]': np.arange(n),
                'HUMIDITY[%]': np.linspace(40, 70, n),
            },
            index=pd.date_range('2020-06-01', periods=n, freq='5min'),
        )
        self.source = {
            'commit': 'fixture',
            'dirty': True,
            'diff': 'fixture source diff',
            'files': {'src/models.py': 'fixture'},
            'command': 'fixture prepare',
        }

    def test_stage_failures_are_saved_before_being_raised(self):
        with (
            patch('src.data.load_data', return_value=self.frame),
            patch('src.experiments.source_record', return_value=self.source),
            redirect_stdout(io.StringIO()),
        ):
            for error, status in (
                (OSError('fixture disk failure'), 'failed'),
                (KeyboardInterrupt('fixture interruption'), 'interrupted'),
            ):
                with self.subTest(status=status):
                    manifest = run('prepare', config=self.config)
                    with patch('src.experiments.tune', side_effect=error):
                        with self.assertRaises(type(error)):
                            run('train', manifest=manifest)
                    state = json.loads(manifest.read_text())
                    self.assertEqual(state['stage'], 'train')
                    self.assertEqual(state['status'], status)
                    self.assertEqual(state['error'], str(error))
                    self.assertEqual(state['final'], [])
                    self.assertFalse(state.get('parameters_frozen', False))

    def test_four_stages_freeze_test_once_and_report_from_saved_results(self):
        # Exercise actual preprocessing, scaler persistence and CSV reporting; mock only model fitting/inference.
        with (
            patch('src.data.load_data', return_value=self.frame) as read,
            patch('src.experiments.source_record', return_value=self.source),
            patch('src.experiments.train_model', return_value=(Mock(), {})) as fit,
            patch('src.experiments.save_model'),
            patch('src.experiments.load_model'),
            patch(
                'src.experiments.predict_scaled',
                side_effect=lambda model, name, x: np.repeat(x[:, -1, :1], 2, axis=1),
            ) as inference,
            redirect_stdout(io.StringIO()),
        ):
            manifest = run('prepare', config=self.config)
            self.assertEqual(json.loads(manifest.read_text())['status'], 'prepared')
            fit.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'freeze'):
                run('evaluate', manifest=manifest)
            run('train', manifest=manifest)
            self.assertEqual(fit.call_count, 16)
            state = json.loads(manifest.read_text())
            self.assertEqual(len(state['frozen_configs']), 8)
            self.assertTrue(
                all('TEST' not in call.args[2] for call in fit.call_args_list)
            )
            run('train', manifest=manifest)
            self.assertEqual(fit.call_count, 16)
            before = inference.call_count
            with patch(
                'sklearn.preprocessing.MinMaxScaler.fit',
                side_effect=AssertionError('refit'),
            ):
                run('evaluate', manifest=manifest)
                run('evaluate', manifest=manifest)
            self.assertEqual(inference.call_count - before, 8)
            run('report', manifest=manifest)
            self.assertEqual(inference.call_count - before, 8)
            state = json.loads(manifest.read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(len(state['final']), 8)
            self.assertEqual(len(state['attempts']), 24)
            self.assertTrue(
                all(item['status'] == 'completed' for item in state['attempts'])
            )
            self.assertEqual(read.call_count, 1)
            metrics = pd.read_csv(manifest.with_name('metrics.csv'))
            primary = metrics.query(
                'variant == "raw" and target == "corrected" and model != "Persistence"'
            )
            np.testing.assert_allclose(primary.nRMSE_cap, primary.RMSE * 10)
            self.assertIn('raw_measurement', metrics.target.unique())
            self.assertEqual(len(list((self.root / 'outputs/models').iterdir())), 8)
            self.assertEqual(
                {path.name for path in manifest.parent.glob('*.csv')},
                {'tuning_results.csv', 'metrics.csv', 'predictions.csv'},
            )
            figures = self.root / 'outputs/figures' / state['study_id']
            self.assertEqual(
                {path.name for path in figures.iterdir()},
                {'forecast_horizon_error.png', 'prediction_curves.png'},
            )
            run('report', manifest=manifest)
            self.assertEqual(inference.call_count - before, 8)

    def test_model_cleanup_preserves_winners_logs_and_unrelated_files(self):
        parent = self.root / 'outputs/models'
        parent.mkdir(parents=True)
        for name in ('winner', 'loser'):
            (parent / name).mkdir()
        outside = self.root / 'unrelated'
        outside.mkdir()
        state = {
            'config': self.config,
            'attempts': [
                {
                    'run_id': name,
                    'run_type': 'tuning',
                    'status': 'completed',
                    'artifact': str(parent / name),
                }
                for name in ('winner', 'loser')
            ],
            'tuning': {
                'fixture': {
                    'trials': [
                        {
                            'number': 0,
                            'status': 'completed',
                            'validation_rmse': 1.0,
                            'result': {'training_run_id': 'winner'},
                        }
                    ]
                }
            },
        }
        _prune_models(state, self.root / 'manifest.json')
        self.assertTrue((parent / 'winner').exists())
        self.assertFalse((parent / 'loser').exists())
        state['attempts'].append(
            {
                'run_id': 'unrelated',
                'run_type': 'tuning',
                'status': 'failed',
                'artifact': str(outside),
            }
        )
        with self.assertRaises(StudyChangedError):
            _prune_models(state, self.root / 'manifest.json')
        self.assertTrue(outside.exists())
        output_paths(self.config, 'fixture')
        with self.assertRaises(FileExistsError):
            output_paths(self.config, 'fixture')

    def test_failed_and_interrupted_attempts_keep_the_same_budget(self):
        spaces = dict(zip(SPACE_NAMES, (['POA Irr[kW1m2]'], ['Pac'], ['TempCell'], [])))
        state = {'feature_spaces': spaces}
        manifest = self.root / 'manifest.json'
        calls = []

        def fit(config, arm, model, *args):
            calls.append(arm['id'])
            if len(calls) == 1:
                raise KeyboardInterrupt()
            if len(calls) == 3:
                raise RuntimeError('fixture training failure')
            return {
                'model': model,
                'arm': arm['id'],
                'pre': config['data']['pre'],
                'subset': arm['subset'],
                'validation_rmse': 1.0,
                'validation_mae': 1.0,
                'training_run_id': str(len(calls)),
                'artifact': 'fixture',
            }

        with (
            patch('src.experiments._fit_trial', side_effect=fit),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(KeyboardInterrupt):
                tune(self.config, state, manifest, self.frame)
            self.assertEqual(
                state['tuning']['XGBoost/Original']['trials'][0]['status'],
                'interrupted',
            )
            tune(self.config, state, manifest, self.frame)
        self.assertEqual(len(calls), 16)
        self.assertTrue(
            all(len(record['trials']) == 2 for record in state['tuning'].values())
        )
        self.assertEqual(state['tuning']['XGBoost/Original']['best_trial'], 1)
        failed = state['tuning']['XGBoost/Expanded-Pearson']['trials'][0]
        self.assertEqual(
            (failed['status'], failed['error']), ('failed', 'fixture training failure')
        )
        self.assertTrue(state['parameters_frozen'])
        invalid = copy.deepcopy(self.config)
        invalid['tuning']['trials_per_space'] = 0
        with self.assertRaises(ValueError):
            validate_search(invalid)

    def test_changed_source_and_io_errors_stop_search(self):
        spaces = dict.fromkeys(SPACE_NAMES, [])
        for error in (StudyChangedError('changed'), OSError('disk failure')):
            state = {'feature_spaces': spaces}
            with (
                patch('src.experiments._fit_trial', side_effect=error) as fit,
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(type(error)):
                    tune(self.config, state, self.root / 'manifest.json', self.frame)
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(
                    state['tuning']['XGBoost/Original']['trials'][0]['status'], 'failed'
                )

    def test_resume_rejects_source_and_input_changes(self):
        with (
            patch('src.data.load_data', return_value=self.frame),
            patch('src.experiments.source_record', return_value=self.source),
            redirect_stdout(io.StringIO()),
        ):
            manifest = run('prepare', config=self.config)
            with self.assertRaisesRegex(ValueError, 'changed settings'):
                run('train', manifest=manifest, config=self.config)
            changed = {**self.source, 'files': {'src/models.py': 'changed'}}
            with patch('src.experiments.source_record', return_value=changed):
                with self.assertRaises(StudyChangedError):
                    run('train', manifest=manifest)
            cache = manifest.with_name('features.joblib')
            cache.write_bytes(cache.read_bytes() + b'changed')
            with self.assertRaisesRegex(StudyChangedError, 'Prepared features'):
                run('train', manifest=manifest)
            Path(self.config['data']['path']).write_text(
                'changed input size', encoding='utf-8'
            )
            with self.assertRaises(StudyChangedError):
                run('train', manifest=manifest)


if __name__ == '__main__':
    unittest.main()
