"""Synthetic multi-seed evidence checks; fixture scores are not research results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.runtime import configure_runtime
configure_runtime()
import numpy as np
import pandas as pd

from src.config import arm_name, data_contract, load_config
from src.metrics import evaluate
from src.provenance import file_hash
from src.report import assemble_report, read_study
from src.config import validate_study


class StudyReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = load_config('config/config.yaml')
        config['output_dir'] = str(self.root / 'outputs')
        config['data']['horizon'] = 2
        config['study'].update(min_test_origins=1, bootstrap_resamples=1000)
        self.manifest = dict(config=config, study_id='synthetic_fixture', smoke=False,
                             seeds=config['study']['seeds'], splits=[config['data']['split']],
                             runs=[], status='complete')
        n, horizon = 12, 2
        y = np.arange(n*horizon).reshape(n, horizon) / 10
        origins = pd.date_range('2020-01-01', periods=n, freq='h')
        for seed_index, seed in enumerate(self.manifest['seeds']):
            for model in ('XGBoost', 'CNN_LSTM'):
                for physics in (False, True):
                    arm = copy.deepcopy(config)
                    arm.update(seed=seed, model=model)
                    arm['data']['physics'] = physics
                    label = arm_name(model, physics)
                    run_id = f'fixture_{seed}_{label}'
                    predicted = y + (0.5 if physics else 1.) + seed_index*.01
                    records = pd.DataFrame({
                        'run_id': run_id, 'forecast_origin': np.repeat(origins, horizon),
                        'target_time': (np.repeat(origins, horizon)
                                        + pd.to_timedelta(np.tile([1, 2], n), unit='h')),
                        'horizon': np.tile([1, 2], n), 'observed': y.ravel(),
                        'predicted_raw': predicted.ravel(), 'persistence': y.ravel()-.1,
                        'absolute_error': np.abs(predicted-y).ravel(),
                        'squared_error': ((predicted-y)**2).ravel(),
                        'sky_condition': 'clear', 'power_level_class': 'low', 'ramp_class': 'steady'})
                    csv = self.root / f'{run_id}.csv'
                    records.to_csv(csv, index=False)
                    metrics = evaluate(y, predicted, y-.1, 7.44, label)
                    metrics['variant'], metrics['run_id'] = 'raw', run_id
                    metric_path = self.root / f'{run_id}_metrics.csv'
                    metrics.to_csv(metric_path, index=False)
                    meta = dict(run_id=run_id, seed=seed, model=model, arm=label, config=arm,
                                data_contract=data_contract(arm['data']), smoke=False,
                                partitions={'fixture': True}, input={'fixture': True},
                                environment={'fixture': True}, scenario_thresholds={'fixture': True},
                                samples=n, csv_sha256=file_hash(csv), metrics_sha256=file_hash(metric_path))
                    path = self.root / f'{run_id}_metadata.json'
                    path.write_text(json.dumps(meta), encoding='utf-8')
                    self.manifest['runs'].append(dict(split_id=0, seed=seed, model=model,
                        physics=physics, run_id=run_id, metadata_path=str(path)))

    def test_multiseed_horizons_ablation_and_empty_strata(self):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(self.manifest), encoding='utf-8')
        with patch('src.report.plot_report'):
            report = assemble_report(path)
        main = pd.read_csv(report / 'main_table.csv')
        self.assertEqual(len(main), 5*3)
        self.assertTrue((main['n_seeds'] == 5).all())
        self.assertGreater(main.query('arm == "XGBoost"')['MAE_std'].iloc[0], 0)
        ablation = pd.read_csv(report / 'ablation_table.csv')
        np.testing.assert_allclose(ablation['delta_MAE'], -.5)
        np.testing.assert_allclose(ablation['ci_high'], -.5)
        np.testing.assert_allclose(ablation['delta_nRMSE_cap'], -50/7.44)
        scenario = pd.read_csv(report / 'scenario_table.csv')
        empty = scenario[scenario['n_origins'] == 0]
        self.assertFalse(empty.empty)
        self.assertTrue(empty['MAE_mean'].isna().all())
        self.assertTrue(empty['underpowered'].all())

    def test_missing_arm_rejected(self):
        self.manifest['runs'].pop()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            read_study(self.manifest)

    def test_collapsed_forecasts_cannot_pass_complete_report(self):
        entry = self.manifest['runs'][0]
        path = Path(entry['metadata_path'])
        meta = json.loads(path.read_text())
        csv = self.root / (entry['run_id']+'.csv')
        records = pd.read_csv(csv)
        records['predicted_raw'] = 0.
        records['absolute_error'] = records['observed'].abs()
        records['squared_error'] = records['observed']**2
        records.to_csv(csv, index=False)
        meta['csv_sha256'] = file_hash(csv)
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'Test forecasts are constant'):
            read_study(self.manifest)

    def test_changed_result_rejected(self):
        entry = self.manifest['runs'][0]
        with (self.root / (entry['run_id']+'.csv')).open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            read_study(self.manifest)

    def test_mismatched_origins_rejected_even_with_updated_hash(self):
        entry = self.manifest['runs'][1]
        csv = self.root / (entry['run_id']+'.csv')
        records = pd.read_csv(csv)
        records['forecast_origin'] = pd.to_datetime(records['forecast_origin']) + pd.Timedelta('1min')
        records.to_csv(csv, index=False)
        path = Path(entry['metadata_path'])
        meta = json.loads(path.read_text())
        meta['csv_sha256'] = file_hash(csv)
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'origins'):
            read_study(self.manifest)

    def _rewrite_first_csv(self, mutate):
        """Apply mutate to one fixture run, then keep its metrics and hashes honest."""
        entry = self.manifest['runs'][0]
        n, horizon = 12, 2
        csv = self.root / (entry['run_id'] + '.csv')
        records = pd.read_csv(csv)
        mutate(records)
        observed = records['observed'].to_numpy().reshape(n, horizon)
        predicted = records['predicted_raw'].to_numpy().reshape(n, horizon)
        persistence = records['persistence'].to_numpy().reshape(n, horizon)
        records.to_csv(csv, index=False)
        metrics = evaluate(observed, predicted, persistence, 7.44,
                           arm_name(entry['model'], entry['physics']))
        metrics['variant'], metrics['run_id'] = 'raw', entry['run_id']
        metric_path = self.root / f"{entry['run_id']}_metrics.csv"
        metrics.to_csv(metric_path, index=False)
        path = Path(entry['metadata_path'])
        meta = json.loads(path.read_text())
        meta['csv_sha256'], meta['metrics_sha256'] = file_hash(csv), file_hash(metric_path)
        path.write_text(json.dumps(meta))
        return pd.read_csv(csv)

    def test_float32_serialized_errors_are_accepted(self):
        """Forecast heads emit float32, which decimal CSV text cannot round-trip exactly.

        Stored errors are exact for those float32 values, so recomputing them from the
        file differs by a few float32 ulps of the operands. That is rounding, not
        tampering: a default relative tolerance collapses where the error itself is
        near zero and previously rejected every legitimate run.
        """
        def quantize(records):
            observed = records['observed'].to_numpy()
            predicted = (observed + np.random.default_rng(0).normal(0, .4, observed.size)
                         ).astype(np.float32)
            # Half the origins are forecast almost perfectly, so their absolute error is
            # small enough for float32 rounding to dominate it -- precisely where a
            # relative tolerance collapses and rejected valid runs.
            predicted[::2] = observed[::2].astype(np.float32)
            error = predicted.astype(float) - observed
            records['predicted_raw'] = predicted
            records['absolute_error'] = np.abs(error)
            records['squared_error'] = error**2

        stored = self._rewrite_first_csv(quantize)
        error = stored['predicted_raw'] - stored['observed']
        self.assertGreater(np.max(np.abs(stored['absolute_error'] - np.abs(error))), 0,
                           'fixture did not lose float32 precision')
        self.assertFalse(np.allclose(stored['absolute_error'], np.abs(error)))
        read_study(self.manifest)

    def test_altered_errors_still_rejected(self):
        """The float32 bound must not excuse a genuinely wrong stored error."""
        def shift(records):
            records['absolute_error'] = records['absolute_error'] + 1e-3

        self._rewrite_first_csv(shift)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            read_study(self.manifest)

    def test_completed_runs_are_reportable_before_the_report_exists(self):
        """A reporting failure must not make finished runs unrecoverable."""
        read_study(dict(self.manifest, status='runs_complete'))
        for status in ('running', 'failed'):
            with self.assertRaisesRegex(ValueError, 'completed runs'):
                read_study(dict(self.manifest, status=status))

    def test_smoke_cannot_be_presented_as_research(self):
        path = Path(self.manifest['runs'][0]['metadata_path'])
        meta = json.loads(path.read_text())
        meta['smoke'] = True
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'Smoke'):
            read_study(self.manifest)

    def test_seed_and_resample_minimums(self):
        settings = copy.deepcopy(self.manifest['config']['study'])
        settings['seeds'] = [11]
        with self.assertRaisesRegex(ValueError, 'five'):
            validate_study(settings)
        settings['seeds'] = [1, 2, 3, 4, 5]
        settings['bootstrap_resamples'] = 99
        with self.assertRaisesRegex(ValueError, '1000'):
            validate_study(settings)


if __name__ == '__main__':
    unittest.main()
