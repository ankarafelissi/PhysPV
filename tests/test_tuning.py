"""Exercise real TPE selection and resume using simulated fits, without training."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.runtime import configure_runtime
configure_runtime()
from src.config import load_config
from src.tuning import tune
from src.evaluation import classify_result
from src.experiments import run


class TuningTests(unittest.TestCase):
    def test_result_classes_preserve_negative_mixed_and_small_effects(self):
        self.assertEqual(classify_result(-0.4, -0.5, -1., -.1), 'small_effect')
        self.assertEqual(classify_result(-3., 2., -1., -.1), 'metric_tradeoff')
        self.assertEqual(classify_result(-3., -2., -1., .1), 'uncertain')
        self.assertEqual(classify_result(3., 2., .1, 1.), 'degradation')
        self.assertEqual(classify_result(-3., -2., -1., -.1), 'improvement')
    def setUp(self):
        self.config = load_config('config/config.yaml')
        self.config['tuning'].update(trials_per_model=6, startup_trials=2)
        self.state = {'config': copy.deepcopy(self.config),
                      'arms': [{'id': 'nonpi_reference', 'subset': []}]}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest = Path(self.temp.name) / 'manifest.json'

    def test_budget_frozen_selection_and_resume_do_not_refit(self):
        values = iter([5., 4., 3., 2., 1., 6.] * 2)
        calls = []
        def fit(config, reference, model, kind):
            calls.append((model, kind, config['seed']))
            return {'validation_mae': next(values), 'training_run_id': f'fixture_{len(calls)}'}
        tune(self.config, self.state, self.manifest, fit)
        self.assertEqual(len(calls), 12)
        self.assertTrue(all(seed == 11 and kind == 'tuning' for _, kind, seed in calls))
        for model in ('XGBoost', 'CNN_LSTM'):
            self.assertEqual(self.state['tuning'][model]['best_trial'], 4)
            self.assertEqual(self.config['models'][model],
                             self.state['tuning'][model]['trials'][4]['parameters'])
        self.assertTrue((self.manifest.parent / 'frozen_config.yaml').exists())
        frozen = copy.deepcopy(self.config)
        tune(self.config, self.state, self.manifest, Mock(side_effect=AssertionError('Refit')))
        self.assertEqual(self.config, frozen)

    def test_failure_is_retained_and_resume_uses_distinct_attempt(self):
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            tune(self.config, self.state, self.manifest,
                 Mock(side_effect=RuntimeError('fixture failure')))
        failed = self.state['tuning']['XGBoost']['trials'][0]
        self.assertEqual(failed['status'], 'failed')
        tune(self.config, self.state, self.manifest,
             Mock(return_value={'validation_mae': 1., 'training_run_id': 'fixture'}))
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(self.state['tuning']['XGBoost']['trials'][1]['number'], 1)

    def test_invalid_budget_is_rejected_before_fit(self):
        self.config['tuning']['trials_per_model'] = 0
        fit = Mock()
        with self.assertRaises(ValueError):
            tune(self.config, self.state, self.manifest, fit)
        fit.assert_not_called()

    def test_full_workflow_freezes_params_and_resumes_without_training(self):
        import json
        self.config['output_dir'] = str(Path(self.temp.name) / 'outputs')
        self.config['tuning']['trials_per_model'] = 2
        partitions = {'TEST': {'n_origins': 1088}}
        seen = []
        def simulated_fit(config, arm, model, seed, expected, run_id, study_dir):
            seen.append((model, copy.deepcopy(config['models'][model])))
            return {'model': model, 'arm': arm['id'], 'seed': seed,
                    'subset': arm['subset'], 'validation_mae': float(len(seen)),
                    'training_run_id': run_id, 'artifact': str(Path(self.temp.name) / run_id),
                    'training_seconds': 0.}
        with patch('src.experiments._prepare_study', return_value=partitions), \
                patch('src.experiments._train_record', side_effect=simulated_fit) as fit, \
                patch('src.predict.run', return_value={'metadata_path': Path(self.temp.name) / 'fixture.json'}) as predict, \
                patch('src.evaluation.report'):
            manifest = run(self.config)
            state = json.loads(manifest.read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(fit.call_count, 20)
            self.assertEqual(predict.call_count, 4)
            self.assertTrue(all(a['status'] == 'completed' for a in state['attempts']))
            for model in ('XGBoost', 'CNN_LSTM'):
                frozen = state['config']['models'][model]
                screened = [a for a in state['attempts'] if a['run_type'] == 'screening' and a['model'] == model]
                self.assertTrue(all(a['effective_config']['models'][model] == frozen for a in screened))
            run(resume=manifest)
            self.assertEqual(fit.call_count, 20)
            self.assertEqual(predict.call_count, 4)
            changed = copy.deepcopy(self.config)
            changed['seed'] = 12
            with self.assertRaisesRegex(ValueError, 'Changed settings'):
                run(changed, resume=manifest)
