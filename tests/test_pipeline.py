"""Regression tests for forecast alignment, leakage and physical features."""
import unittest
from unittest.mock import patch
from src.runtime import configure_runtime
configure_runtime()
import numpy as np
import pandas as pd
from src.data import load_data, prepare_data
from src.features import build_features
from src.metrics import evaluate
from src.config import load_config


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'power': np.arange(100.)}, index=pd.date_range('2020', periods=100, freq='h'))
        self.config = dict(pre=3, horizon=4, features=['power'], target='power', split=[.7, .2, .1], scaler='MinMax01', physics=False)

    def test_alignment_and_disjoint_targets(self):
        parts, scalers = prepare_data(self.frame, self.config)
        train, val, test = [parts[k] for k in ['TRAIN', 'VAL', 'TEST']]
        np.testing.assert_array_equal(train['Y'][0], [4, 5, 6, 7])
        self.assertLess(train['target_times'].max(), val['target_times'].min())
        self.assertLess(val['target_times'].max(), test['target_times'].min())
        self.assertEqual(scalers['Y'].data_max_[0], 69)
        self.assertGreater(test['Y_scaled'].max(), 1)
        np.testing.assert_array_equal(test['persistence'][0], [89]*4)

    def test_equal_windows_and_missing_data(self):
        self.config.update(pre=4)
        self.frame.iloc[30] = np.nan
        parts, _ = prepare_data(self.frame, self.config)
        for part in parts.values():
            self.assertTrue(np.isfinite(part['X']).all())
            for origin in part['origins']:
                self.assertNotIn(self.frame.index[30], pd.date_range(origin-pd.Timedelta(hours=4), origin+pd.Timedelta(hours=4), freq='h'))

    def test_pre_zero_and_insufficient_samples(self):
        self.config.update(pre=0, horizon=1)
        parts, _ = prepare_data(self.frame, self.config)
        self.assertEqual(parts['TRAIN']['X'].shape[1], 1)
        self.config['horizon'] = 100
        with self.assertRaisesRegex(ValueError, 'no valid windows'):
            prepare_data(self.frame, self.config)

    def test_reuse_scalers(self):
        _, scalers = prepare_data(self.frame, self.config)
        old_max = scalers['Y'].data_max_.copy()
        parts, _ = prepare_data(self.frame * 10, self.config, scalers)
        np.testing.assert_array_equal(scalers['Y'].data_max_, old_max)
        self.assertGreater(parts['TRAIN']['Y_scaled'].max(), 1)

    def test_metrics(self):
        observed = np.array([[1., 2.], [3., 4.]])
        result = evaluate(observed, observed+2, observed, 7.44)
        np.testing.assert_array_equal(result.query('model == "Forecaster"')['RMSE'], [2, 2])
        np.testing.assert_array_equal(result.query('model == "Persistence"')['RMSE'], [0, 0])

    def test_loading_sorts_days_and_preserves_gaps(self):
        frame = self.frame.drop(self.frame.index[30]).iloc[::-1]
        with patch('src.data.pd.read_hdf', return_value=frame):
            result = load_data({'path': 'unused.h5', 'resolution': '60min'})
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertTrue(np.isnan(result.loc[self.frame.index[30], 'power']))
        with patch('src.data.pd.read_hdf', return_value=pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_data({'path': 'unused.h5', 'resolution': '60min'})

    def test_features_do_not_rewrite_observations(self):
        frame = pd.DataFrame({'P_Solar[kW]': [0., 1.], 'TEMPERATURE[degC]': [-5., 10.],
                              'POA Irr[kW1m2]': [0., 1.], 'WIND_SPEED[m1s]': [2., 2.]},
                             index=pd.date_range('2020', periods=2, freq='h'))
        original = frame.copy(deep=True)
        config = load_config('config/config.yaml')['data']
        config.update(features=['P_Solar[kW]'], physics=True)
        result = build_features(frame, config)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_series_equal(result['P_Solar[kW]'], frame['P_Solar[kW]'])
        self.assertEqual(result['TempModule'].iloc[0], -5.)


if __name__ == '__main__':
    unittest.main()
