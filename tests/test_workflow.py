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
from src.experiments import (
    StudyChangedError, _attempt, _launch_worker, _prune_models, _resume_study, _worker,
    run, select_model, tune,
)
from src.runtime import GPUError, environment_metadata, write_json


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix='.test-', dir=Path(__file__).resolve().parents[1]
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config('config/config.yaml')
        self.config['data'].pop('time_split')
        self.config['data']['split'] = [0.7, 0.2, 0.1]
        self.config['runtime']['gpu_id'] = 0
        raw = self.root / 'raw.h5'
        raw.write_text('synthetic input', encoding='utf-8')
        self.config['data'].update(path=str(raw), pre=2, horizon=2, correct_power=False)
        self.config['tuning'].update(
            trials_per_space=2, startup_trials=1, pre_choices=[0, 2, 4]
        )
        self.config['study'].update(min_test_origins=5)
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

    def inline_worker(self, kind, state, manifest):
        """Exercise the tuning worker with mocked fitting, without training."""
        attempt = _attempt(state, manifest, state['selected_model'], 'all',
                           'search', state['config'])
        manifest.write_text(json.dumps(state), encoding='utf-8')
        try:
            _worker(kind, manifest, attempt['run_id'])
        finally:
            state.clear()
            state.update(json.loads(manifest.read_text()))

    @staticmethod
    def fake_fit(*args, **kwargs):
        measurement = kwargs.get('measurement')
        if measurement is not None:
            measurement.update(fit_time_s=.01, peak_cpu_ram_mb=50.,
                               peak_gpu_vram_mb=None, gpu_memory_scope=None,
                               gpu_memory_reason='fixture unavailable')
        return Mock(), {'fit_time_s': .01, 'epochs_completed': 3, 'best_epoch': 2}

    def test_stage_failures_are_saved_before_being_raised(self):
        with (
            patch('src.data.load_data', return_value=self.frame),
            patch('src.experiments.source_record', return_value=self.source),
            patch('src.experiments._launch_worker', side_effect=self.inline_worker),
            patch('src.experiments.check_gpu', return_value={'pci_bus_id': 'fixture'}),
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
                            run('train', manifest=manifest, model='XGBoost')
                    state = json.loads(manifest.read_text())
                    self.assertEqual(state['stage'], 'train')
                    self.assertEqual(state['status'], status)
                    self.assertEqual(state['error'], str(error))
                    self.assertEqual(state['final'], [])
                    self.assertFalse(state.get('parameters_frozen', False))

    def test_three_stages_freeze_and_evaluate_test_once(self):
        self.exercise_single_model('XGBoost')

    def test_cnn_lstm_single_model_completes_all_three_stages(self):
        self.exercise_single_model('CNN_LSTM')

    def exercise_single_model(self, model):
        # Exercise actual preprocessing, scaler persistence and CSV reporting; mock only model fitting/inference.
        with (
            patch('src.data.load_data', return_value=self.frame) as read,
            patch('src.experiments.source_record', return_value=self.source),
            patch('src.experiments.train_model', side_effect=self.fake_fit) as fit,
            patch('src.experiments._launch_worker', side_effect=self.inline_worker),
            patch('src.experiments.check_gpu', return_value={'pci_bus_id': 'fixture'}),
            patch('src.experiments.configure_gpu', return_value={'pci_bus_id': 'fixture'}),
            patch('src.experiments.verify_gpu_model', return_value='GPU:0'),
            patch('src.experiments.synchronize_gpu'),
            patch('src.experiments.save_model'),
            patch('src.experiments.load_model'),
            patch(
                'src.experiments.predict_scaled',
                side_effect=lambda model, name, x, **kwargs: np.repeat(x[:, -1, :1], 2, axis=1),
            ) as inference,
            redirect_stdout(io.StringIO()),
        ):
            manifest = run('prepare', config=self.config)
            self.assertEqual(json.loads(manifest.read_text())['status'], 'prepared')
            fit.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'freeze'):
                run('evaluate', manifest=manifest)
            run('train', manifest=manifest, model=model)
            self.assertEqual(fit.call_count, 8)
            state = json.loads(manifest.read_text())
            self.assertEqual(len(state['frozen_configs']), 4)
            self.assertTrue(
                all('TEST' not in call.args[2] for call in fit.call_args_list)
            )
            run('train', manifest=manifest)
            self.assertEqual(fit.call_count, 8)
            before = inference.call_count
            with patch(
                'sklearn.preprocessing.MinMaxScaler.fit',
                side_effect=AssertionError('refit'),
            ):
                run('evaluate', manifest=manifest)
                run('evaluate', manifest=manifest)
            self.assertEqual(inference.call_count - before, 4)
            self.assertEqual(inference.call_count - before, 4)
            state = json.loads(manifest.read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(len(state['final']), 4)
            self.assertEqual(len(state['attempts']), 13)
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
            self.assertEqual(len(list((self.root / 'outputs/models').iterdir())), 4)
            self.assertEqual(
                {path.name for path in manifest.parent.glob('*.csv')},
                {'tuning_results.csv', 'metrics.csv', 'predictions.csv'},
            )
            self.assertEqual(inference.call_count - before, 4)

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
        predecessor = parent / 'predecessor'
        predecessor.mkdir()
        state['attempts'].append({
            'run_id': 'predecessor', 'run_type': 'tuning', 'status': 'completed',
            'artifact': str(predecessor), 'inherited_from_study': 'prior_study',
        })
        _prune_models(state, self.root / 'manifest.json')
        self.assertTrue((parent / 'winner').exists())
        self.assertTrue(predecessor.exists())
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

    def test_cli_model_binding_overrides_yaml_and_cannot_switch(self):
        import main

        for model in ('XGBoost', 'CNN_LSTM'):
            with patch('sys.argv', ['main.py', 'train', '--resume', 'fixture',
                                    '--model', model]), patch('src.experiments.run') as dispatch:
                main.main()
                dispatch.assert_called_once_with('train', manifest='fixture', model=model)
        state = {'config': copy.deepcopy(self.config), 'selected_model': None}
        manifest = self.root / 'selection.json'
        with self.assertRaisesRegex(ValueError, 'First train requires'):
            select_model(state, manifest, None)
        select_model(state, manifest, 'XGBoost')
        self.assertEqual(state['config']['model'], 'XGBoost')
        select_model(state, manifest, None)
        with self.assertRaisesRegex(ValueError, 'bound to XGBoost'):
            select_model(state, manifest, 'CNN_LSTM')


    def test_gpu_preflight_failure_spends_no_trial_budget(self):
        with (patch('src.data.load_data', return_value=self.frame),
              patch('src.experiments.source_record', return_value=self.source),
              patch('src.experiments._launch_worker', side_effect=self.inline_worker),
              patch('src.experiments.check_gpu', side_effect=GPUError('No GPU')),
              patch('src.experiments.train_model') as fit,
              redirect_stdout(io.StringIO())):
            manifest = run('prepare', config=self.config)
            with self.assertRaises(GPUError):
                run('train', manifest=manifest, model='XGBoost')
            state = json.loads(manifest.read_text())
            self.assertNotIn('tuning', state)
            self.assertEqual(state['attempts'][0]['status'], 'failed')
            self.assertEqual(state['selected_model'], 'XGBoost')
            fit.assert_not_called()

    def test_worker_streams_output_before_process_completes(self):
        import subprocess

        state = {'config': self.config, 'source': self.source,
                 'selected_model': 'XGBoost', 'attempts': [],
                 'frozen_configs': {'XGBoost/Original': self.config}}
        manifest = self.root / 'manifest.json'
        process = Mock(returncode=0)
        terminal = io.StringIO()
        calls = 0

        def wait(timeout):
            nonlocal calls
            calls += 1
            saved = json.loads(manifest.read_text())
            attempt = saved['attempts'][-1]
            if calls == 1:
                with Path(attempt['log']).open('a', encoding='utf-8') as stream:
                    stream.write('XGBoost horizon 1/60\n')
                raise subprocess.TimeoutExpired('fixture', timeout)
            self.assertIn('XGBoost horizon 1/60', terminal.getvalue())
            attempt['status'] = 'completed'
            write_json(manifest, saved)

        process.wait.side_effect = wait
        with (patch('src.experiments.subprocess.Popen', return_value=process),
              redirect_stdout(terminal)):
            _launch_worker('tuning', state, manifest)
        self.assertIn('XGBoost horizon 1/60', Path(state['attempts'][0]['log']).read_text())


    def test_hard_exit_recovers_durable_elapsed_lower_bound(self):
        with (patch('src.data.load_data', return_value=self.frame),
              patch('src.experiments.source_record', return_value=self.source),
              redirect_stdout(io.StringIO())):
            manifest = run('prepare', config=self.config)
            state = json.loads(manifest.read_text())
            select_model(state, manifest, 'XGBoost')
            attempt = _attempt(state, manifest, 'XGBoost', 'all', 'search', state['config'])
            state['tuning'] = {'XGBoost/Original': {'model': 'XGBoost', 'space': 'Original',
                'best_trial': None, 'trials': [{'number': 0, 'status': 'started',
                    'params': {}, 'parameters': {}, 'subset': [], 'trial_wall_time_s': 0.}]}}
            log = Path(attempt['log'])
            log.parent.mkdir(parents=True)
            log.write_text('PhysPV progress ' + json.dumps({
                'run_id': attempt['run_id'], 'wall_time_s': 18.,
                'trial_key': 'XGBoost/Original', 'trial_number': 0,
                'trial_wall_time_s': 12.,
            }) + '\nPhysPV progress {incomplete', encoding='utf-8')
            write_json(manifest, state)
            _, recovered = _resume_study(manifest, None)
        trial = recovered['tuning']['XGBoost/Original']['trials'][0]
        self.assertEqual(trial['status'], 'interrupted')
        self.assertEqual(trial['trial_wall_time_s'], 12.)
        self.assertFalse(trial['wall_time_complete'])
        self.assertEqual(recovered['attempts'][0]['wall_time_s'], 18.)

    def test_val_rmse_winner_is_chosen_even_when_fit_is_slower(self):
        state = {'feature_spaces': dict.fromkeys(SPACE_NAMES, []),
                 'selected_model': 'XGBoost', 'config': self.config,
                 'partitions': {'VAL': {'n_origins': 10}}}
        counter = 0

        def fit(config, arm, model, *args):
            nonlocal counter
            counter += 1
            first = counter % 2 == 1
            return {'model': model, 'arm': arm['id'], 'pre': config['data']['pre'],
                    'subset': [], 'validation_rmse': .5 if first else 1.,
                    'validation_mae': 1., 'training_seconds': 1000. if first else .01,
                    'training_run_id': str(counter), 'artifact': 'fixture'}

        with (patch('src.experiments._fit_trial', side_effect=fit),
              redirect_stdout(io.StringIO())):
            tune(self.config, state, self.root / 'manifest.json', self.frame)
        self.assertTrue(all(record['best_trial'] == 0 for record in state['tuning'].values()))

    def test_resume_refuses_to_overlap_a_live_worker(self):
        import os
        import psutil

        with (patch('src.data.load_data', return_value=self.frame),
              patch('src.experiments.source_record', return_value=self.source),
              redirect_stdout(io.StringIO())):
            manifest = run('prepare', config=self.config)
            state = json.loads(manifest.read_text())
            attempt = _attempt(state, manifest, 'XGBoost', 'all', 'search', state['config'])
            attempt.update(worker_pid=os.getpid(), worker_created_at=psutil.Process().create_time())
            write_json(manifest, state)
            with self.assertRaisesRegex(RuntimeError, 'still running'):
                _resume_study(manifest, None)
        self.assertEqual(json.loads(manifest.read_text())['attempts'][0]['status'], 'started')

    def test_failed_and_interrupted_attempts_keep_the_same_budget(self):
        spaces = dict(zip(SPACE_NAMES, (['POA Irr[kW1m2]'], ['Pac'], ['TempCell'], [])))
        state = {'feature_spaces': spaces, 'selected_model': 'XGBoost',
                 'config': self.config, 'partitions': {'VAL': {'n_origins': 10}}}
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
        self.assertEqual(len(calls), 8)
        self.assertTrue(
            all(len(record['trials']) == 2 for record in state['tuning'].values())
        )
        self.assertEqual(state['tuning']['XGBoost/Original']['best_trial'], 1)
        failed = state['tuning']['XGBoost/Expanded-Pearson']['trials'][0]
        self.assertEqual(
            (failed['status'], failed['error']), ('failed', 'fixture training failure')
        )
        self.assertTrue(state['parameters_frozen'])
        self.assertTrue(state['tuning_completed'])
        self.assertNotIn('benchmark_protocol', state)
        invalid = copy.deepcopy(self.config)
        invalid['tuning']['trials_per_space'] = 0
        with self.assertRaises(ValueError):
            validate_search(invalid)

    def test_changed_source_and_io_errors_stop_search(self):
        spaces = dict.fromkeys(SPACE_NAMES, [])
        for error in (StudyChangedError('changed'), OSError('disk failure'),
                      GPUError('device unavailable'), RuntimeError('CUDA out of memory')):
            state = {'feature_spaces': spaces, 'selected_model': 'XGBoost'}
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
                self.assertIsNone(state['tuning']['XGBoost/Original']['trials'][0]['training_seconds'])

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
            state = json.loads(manifest.read_text())
            state['config']['runtime']['gpu_id'] += 1
            write_json(manifest, state)
            with self.assertRaisesRegex(StudyChangedError, 'Effective configuration'):
                run('train', manifest=manifest)
            state['config']['runtime']['gpu_id'] -= 1
            write_json(manifest, state)
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
