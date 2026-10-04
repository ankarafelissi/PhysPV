"""Joint optimization, failure history, freeze and closed-TEST regression tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from src.runtime import configure_runtime
configure_runtime()
from src.config import load_config
from src.tuning import tune
from src.data import SPACE_NAMES
from src.config import validate_search
from src.experiments import run


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = load_config('config/config.yaml')
        self.config['tuning'].update(trials_per_space=6, startup_trials=2)
        self.config['output_dir'] = str(Path(self.temp.name) / 'outputs')
        self.spaces = dict(zip(SPACE_NAMES, (['POA Irr[kW1m2]'], ['Pac', 'Pdc'], ['Pac', 'HoursOfDay'], [])))
        self.state = {'config': self.config, 'feature_spaces': self.spaces}
        self.manifest = Path(self.temp.name) / 'manifest.json'

    def test_equal_budget_joint_selection_and_no_refit_on_resume(self):
        seen = []
        def fit(config, arm, model, kind):
            index = len(seen) % 6
            seen.append((model, arm['id'], config['data']['pre'], arm['subset']))
            return {'model': model, 'arm': arm['id'], 'seed': config['seed'],
                    'subset': arm['subset'], 'pre': config['data']['pre'],
                    'validation_rmse': [6., 5., 4., 3., 1., 2.][index],
                    'validation_mae': float(index), 'training_run_id': f'fixture_{len(seen)}'}
        tune(self.config, self.state, self.manifest, fit)
        self.assertEqual(len(seen), 48)
        self.assertTrue(self.state['parameters_frozen'])
        for record in self.state['tuning'].values():
            self.assertEqual(record['best_trial'], 4)
            for trial in record['trials']:
                self.assertIn('pre', trial['params'])
                self.assertTrue(set(trial['subset']).issubset(self.spaces[record['space']]))
                if record['space'] == 'Intrinsic':
                    self.assertEqual(trial['subset'], [])
        self.assertGreater(len({pre for _, _, pre, _ in seen}), 1)
        tune(self.config, self.state, self.manifest, Mock(side_effect=AssertionError('Refit')))
        self.assertTrue(self.manifest.with_name('frozen_config.yaml').exists())

    def test_failed_trials_retained_and_consume_budget(self):
        calls = 0
        def fit(config, arm, model, kind):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError('fixture failure')
            return {'model': model, 'arm': arm['id'], 'pre': config['data']['pre'],
                    'subset': arm['subset'], 'validation_rmse': 1., 'validation_mae': 1., 'training_run_id': str(calls)}
        tune(self.config, self.state, self.manifest, fit)
        first = self.state['tuning']['XGBoost/Original']['trials'][0]
        self.assertEqual(first['status'], 'failed')
        self.assertEqual(first['error'], 'fixture failure')
        self.assertEqual(calls, 48)

    def test_invalid_budget_and_pre_rejected_before_fit(self):
        for invalid in (0, True):
            self.config['tuning']['trials_per_space'] = invalid
            with self.assertRaises(ValueError):
                validate_search(self.config)

    def test_workflow_resume_and_all_four_frozen_test_arms(self):
        self.config['tuning'].update(trials_per_space=2, startup_trials=1)
        calls = []
        def simulated(config, arm, model, seed, expected, run_id, directory):
            calls.append(config)
            return {'model': model, 'arm': arm['id'], 'seed': seed, 'subset': arm['subset'],
                    'pre': config['data']['pre'], 'validation_mae': 1., 'validation_rmse': 2.,
                    'training_run_id': run_id, 'artifact': str(Path(self.temp.name) / run_id), 'training_seconds': 0.}
        with patch('src.experiments._prepare_study', return_value=(self.spaces, {'TEST': {'n_origins': 1000}})), \
                patch('src.experiments._train_record', side_effect=simulated) as fit, \
                patch('src.predict.run', return_value={'metadata_path': Path(self.temp.name) / 'prediction.json'}) as predict, \
                patch('src.evaluation.report'):
            manifest = run(self.config)
            self.assertEqual(fit.call_count, 16)
            predict.assert_not_called()
            state = json.loads(manifest.read_text())
            self.assertEqual(state['status'], 'selection_frozen')
            self.assertEqual(len(state['frozen_configs']), 8)
            run(resume=manifest)
            self.assertEqual(fit.call_count, 16)
            run(resume=manifest, evaluate_test=True)
            self.assertEqual(predict.call_count, 8)
            self.assertEqual(json.loads(manifest.read_text())['status'], 'completed')
            run(resume=manifest, evaluate_test=True)
            self.assertEqual(predict.call_count, 8)
            self.assertEqual(fit.call_count, 16)
            changed = copy.deepcopy(self.config)
            changed['seed'] = 12
            with self.assertRaisesRegex(ValueError, 'Changed settings'):
                run(changed, resume=manifest)

    def test_interrupted_search_resume_uses_new_attempt(self):
        with self.assertRaises(KeyboardInterrupt):
            tune(self.config, self.state, self.manifest, Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(self.state['tuning']['XGBoost/Original']['trials'][0]['status'], 'interrupted')
        def fit(config, arm, model, kind):
            return {'model': model, 'arm': arm['id'], 'subset': arm['subset'], 'pre': config['data']['pre'],
                    'validation_rmse': 1., 'validation_mae': 1., 'training_run_id': 'resumed'}
        tune(self.config, self.state, self.manifest, fit)
        self.assertEqual(self.state['tuning']['XGBoost/Original']['trials'][0]['status'], 'interrupted')
