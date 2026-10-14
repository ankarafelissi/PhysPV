"""Feature configuration, causal windows and physical target handling."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.runtime import configure_runtime
configure_runtime()
import numpy as np
import pandas as pd
import yaml
from src.config import arm_config, data_contract, load_config
from src.data import (EXPANDED_FEATURES, build_features, feature_names, load_data,
                      prepare_data, prepare_spaces, uses_physics)
from src.evaluate import constrain_power


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.test-', dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config('config/config.yaml')
        raw = self.root / 'raw.h5'
        raw.write_text('synthetic input', encoding='utf-8')
        self.config['data'].update(pre=2, horizon=2, correct_power=False)
        n = 240
        self.frame = pd.DataFrame({
            'P_Solar[kW]': np.linspace(0, 5, n), 'POA Irr[kW1m2]': np.linspace(.1, 1, n),
            'GHI[kW1m2]': np.linspace(.1, .9, n), 'TEMPERATURE[degC]': np.full(n, 20.),
            'WIND_SPEED[m1s]': np.full(n, 2.), 'WIND_DIR[deg]': np.arange(n),
            'HUMIDITY[%]': np.linspace(40, 70, n),
        }, index=pd.date_range('2020-06-01', periods=n, freq='5min'))


    def test_feature_config_and_legacy_loading(self):
        config = arm_config(self.config, ['TempCell', 'Pac'], 'XGBoost', 11)
        self.assertEqual(config['data']['features'], ['P_Solar[kW]', 'Pac', 'TempCell'])
        self.assertTrue(uses_physics(config['data']))
        self.assertNotEqual(data_contract(config['data']), data_contract(self.config['data']))
        with self.assertRaises(ValueError):
            feature_names({'features': ['Pac', 'Pac']})
        legacy = copy.deepcopy(self.config)
        legacy['data'].update(physics=True, candidate_subset=['TempCell', 'Pac'])
        path = self.root / 'legacy.yaml'
        path.write_text(yaml.safe_dump(legacy), encoding='utf-8')
        converted = load_config(path)['data']
        self.assertEqual(converted['features'], ['P_Solar[kW]', 'Pac', 'TempCell'])
        self.assertFalse({'physics', 'candidate_subset', 'selected_features'} & converted.keys())

    def test_window_alignment_train_scaling_and_reuse(self):
        frame = pd.DataFrame({'power': np.arange(100.)}, index=pd.date_range('2020', periods=100, freq='5min'))
        config = dict(pre=3, horizon=4, features=['power'], target='power', split=[.7, .2, .1], scaler='MinMax01')
        parts, scalers = prepare_data(frame, config)
        np.testing.assert_array_equal(parts['TRAIN']['Y'][0], [4, 5, 6, 7])
        self.assertLess(parts['TRAIN']['target_times'].max(), parts['VAL']['target_times'].min())
        self.assertLess(parts['VAL']['target_times'].max(), parts['TEST']['target_times'].min())
        self.assertEqual(scalers['Y'].data_max_[0], 69)
        self.assertGreater(parts['TEST']['Y_scaled'].max(), 1)
        np.testing.assert_array_equal(parts['TEST']['persistence'][0], [89] * 4)
        with patch('sklearn.preprocessing.MinMaxScaler.fit', side_effect=AssertionError('refit')):
            test, _ = prepare_data(frame, config, scalers=scalers, partitions=('TEST',))
        self.assertEqual(list(test), ['TEST'])
        np.testing.assert_array_equal(test['TEST']['X'], parts['TEST']['X'])

    def test_common_origins_and_missing_data(self):
        self.frame.iloc[30, self.frame.columns.get_loc('HUMIDITY[%]')] = np.nan
        common = None
        for pre, subset in [(0, []), (2, ['Pac']), (4, ['HUMIDITY[%]', 'TempCell'])]:
            config = arm_config(self.config, subset, 'XGBoost', 11)
            config['data'].update(pre=pre, origin_pre=4, eligibility_features=['HUMIDITY[%]'])
            parts, _ = prepare_data(build_features(self.frame, config['data']), config['data'])
            for name, part in parts.items():
                self.assertTrue(np.isfinite(part['X']).all())
                if common is not None:
                    np.testing.assert_array_equal(part['origins'], common[name]['origins'])
                    np.testing.assert_array_equal(part['Y'], common[name]['Y'])
            common = parts
        frame = self.frame.drop(self.frame.index[30]).iloc[::-1]
        with patch('src.data.pd.read_hdf', return_value=frame):
            loaded = load_data(self.config['data'])
        self.assertTrue(loaded.index.is_monotonic_increasing)
        self.assertTrue(loaded.iloc[30].isna().all())
        with patch('src.data.pd.read_hdf', return_value=pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_data(self.config['data'])

    def test_selectors_and_features_do_not_use_future_values(self):
        changed = self.frame.copy()
        changed.iloc[168:] *= 9
        self.config['data']['eligibility_features'] = ['P_Solar[kW]', *EXPANDED_FEATURES]
        with patch('src.data.load_data', return_value=self.frame):
            before, frame = prepare_spaces(self.config, self.root)
        with patch('src.data.load_data', return_value=changed):
            after, _ = prepare_spaces(self.config, self.root)
        self.assertEqual(before, after)
        self.assertEqual(before['Intrinsic'], [])
        full = arm_config(self.config, list(EXPANDED_FEATURES), 'XGBoost', 11)['data']
        parts, scalers = prepare_data(frame, full)
        new_parts, new_scalers = prepare_data(build_features(changed, full), full)
        np.testing.assert_array_equal(parts['TRAIN']['X'], new_parts['TRAIN']['X'])
        np.testing.assert_array_equal(scalers['X'].data_max_, new_scalers['X'].data_max_)

    def test_physics_correction_preserves_raw_and_is_common(self):
        original = self.frame.copy(deep=True)
        self.config['data']['correct_power'] = True
        plain = build_features(self.frame, self.config['data'])
        physics = build_features(self.frame, arm_config(self.config, ['Pac', 'TempCell'], 'XGBoost', 11)['data'])
        pd.testing.assert_frame_equal(self.frame, original)
        np.testing.assert_array_equal(plain['observed_power_raw'], original['P_Solar[kW]'])
        np.testing.assert_array_equal(plain['P_Solar[kW]'], physics['P_Solar[kW]'])
        self.assertGreater(plain.attrs['power_correction']['replaced_rows'], 0)
        self.assertNotIn('Pac', plain)
        self.assertIn('TempCell', physics)
        raw = np.array([[-1., 12.]])
        clipped = constrain_power(raw, self.config['data'])
        np.testing.assert_array_equal(raw, [[-1., 12.]])
        np.testing.assert_array_equal(clipped, [[0., 12.]])


if __name__ == '__main__':
    unittest.main()
