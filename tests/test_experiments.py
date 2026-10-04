"""Regression coverage for subset isolation, selection and corrected metrics."""
import copy
import unittest

from src.runtime import configure_runtime
configure_runtime()
import numpy as np
import pandas as pd

from src.config import arm_config, data_contract, load_config
from src.data import prepare_data
from src.data import EXPANDED_FEATURES, PHYSICS_SELECTED, SPACE_NAMES, prepare_spaces
from src.data import CANDIDATE_FEATURES, build_features, feature_names, safe_ratio, solar_features
from src.metrics import evaluate


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('config/config.yaml')
        self.config['data'].update(pre=3, horizon=2)
        self.config['data']['solar'].update(elevation_column='elevation_deg', clear_ghi_column='clear_ghi')
        n = 140
        self.frame = pd.DataFrame({
            'POA Irr[kW1m2]': np.linspace(0, 1, n), 'GHI[kW1m2]': np.linspace(0, .9, n),
            'TEMPERATURE[degC]': np.full(n, 20.), 'WIND_SPEED[m1s]': np.full(n, 2.),
            'HUMIDITY[%]': np.full(n, 50.), 'P_Solar[kW]': np.linspace(0, 5, n),
            'elevation_deg': np.full(n, 30.), 'clear_ghi': np.full(n, .8),
        }, index=pd.date_range('2020-01-01', periods=n, freq='h'))

    def test_candidate_order_and_contract(self):
        config = arm_config(self.config, ['TempCell', 'Pac'], 'XGBoost', 11)
        self.assertEqual(feature_names(config['data'])[-2:], ['Pac', 'TempCell'])
        self.assertNotEqual(data_contract(config['data']), data_contract(self.config['data']))
        with self.assertRaisesRegex(ValueError, 'unique supported'):
            arm_config(self.config, ['future_power'], 'XGBoost', 11)
        with self.assertRaisesRegex(ValueError, 'unique supported'):
            arm_config(self.config, ['Pac', 'Pac'], 'XGBoost', 11)

    def test_ratio_floors_and_unconfigured_station(self):
        np.testing.assert_allclose(safe_ratio([2., 2., 2., 2., np.nan],
                                             [0., .05, .1, np.nan, .1], .05),
                                   [0., 0., 20., np.nan, np.nan], equal_nan=True)
        with self.assertRaisesRegex(ValueError, 'positive'):
            safe_ratio([1.], [1.], 0)
        with self.assertRaisesRegex(ValueError, 'verified station metadata'):
            solar_features(self.frame, {**self.config['data'], 'solar': {}})

    def test_solar_geometry_clear_sky_units_and_timezone(self):
        config = copy.deepcopy(self.config['data'])
        config['solar'] = {'site': {'latitude': 0., 'longitude': 0., 'timezone': 'UTC'}}
        frame = pd.DataFrame({'GHI[kW1m2]': .5},
                             index=pd.date_range('2020-03-20', periods=24, freq='h'))
        sun = solar_features(frame, config)
        self.assertTrue(np.isfinite(sun.to_numpy()).all())
        self.assertLess(sun.solar_elevation.iloc[0], -80)
        self.assertGreater(sun.solar_elevation.iloc[12], 80)
        self.assertEqual(sun.clear_sky_index.iloc[0], 0)
        self.assertGreater(sun.clear_sky_index.iloc[12], .3)
        self.assertLess(sun.clear_sky_index.iloc[12], .8)
        aware = frame.copy()
        aware.index = aware.index.tz_localize('UTC').tz_convert('Europe/London')
        np.testing.assert_allclose(sun.to_numpy(), solar_features(aware, config).to_numpy())

    def test_state_formulas_night_and_missing_solar_common_mask(self):
        config = arm_config(self.config, list(CANDIDATE_FEATURES), 'XGBoost', 11)
        frame = build_features(self.frame, config['data'])
        np.testing.assert_allclose(frame.physics_residual, frame['P_Solar[kW]'] - frame.Pac)
        valid = frame.Pac > .05
        np.testing.assert_allclose(frame.loc[valid, 'performance_ratio'],
                                   frame.loc[valid, 'P_Solar[kW]'] / frame.loc[valid, 'Pac'])
        self.assertTrue((frame.loc[~valid, 'performance_ratio'] == 0).all())
        np.testing.assert_allclose(frame.clear_sky_index, self.frame['GHI[kW1m2]'] / .8)
        np.testing.assert_allclose(frame.solar_zenith + frame.solar_elevation, 90)
        self.frame.loc[self.frame.index[30], 'clear_ghi'] = np.nan
        masks = [build_features(self.frame, arm_config(self.config, subset, 'XGBoost', 11)['data']).attrs['eligible_rows']
                 for subset in ([], ['Pac'], ['clear_sky_index'])]
        for mask in masks[1:]:
            np.testing.assert_array_equal(mask, masks[0])

    def test_missing_unused_candidate_has_common_eligibility(self):
        self.frame.iloc[30, self.frame.columns.get_loc('GHI[kW1m2]')] = np.nan
        parts = []
        for subset in ([], ['Pac'], list(CANDIDATE_FEATURES)):
            config = arm_config(self.config, subset, 'XGBoost', 11)
            parts.append(prepare_data(build_features(self.frame, config['data']), config['data'])[0])
        for partition in ('TRAIN', 'VAL', 'TEST'):
            for other in parts[1:]:
                np.testing.assert_array_equal(parts[0][partition]['origins'], other[partition]['origins'])
                np.testing.assert_array_equal(parts[0][partition]['Y'], other[partition]['Y'])

    def test_future_changes_do_not_change_train_features_or_scalers(self):
        config = arm_config(self.config, list(CANDIDATE_FEATURES), 'XGBoost', 11)
        before, scalers = prepare_data(build_features(self.frame, config['data']), config['data'])
        modified = self.frame.copy()
        modified.iloc[126:] *= 3
        after, new_scalers = prepare_data(build_features(modified, config['data']), config['data'])
        np.testing.assert_array_equal(before['TRAIN']['X'], after['TRAIN']['X'])
        np.testing.assert_array_equal(scalers['X'].data_max_, new_scalers['X'].data_max_)

    def test_joint_search_has_identical_origins_for_different_pre_and_features(self):
        self.frame.loc[self.frame.index[45], 'HUMIDITY[%]'] = np.nan
        parts = []
        for pre, subset in [(0, []), (3, ['Pac']), (6, ['HUMIDITY[%]', 'HoursOfDay'])]:
            config = arm_config(self.config, subset, 'XGBoost', 11)
            config['data'].update(pre=pre, origin_pre=6, eligibility_features=['HUMIDITY[%]'])
            parts.append(prepare_data(build_features(self.frame, config['data']), config['data'])[0])
        for name in ('TRAIN', 'VAL', 'TEST'):
            for other in parts[1:]:
                np.testing.assert_array_equal(parts[0][name]['origins'], other[name]['origins'])
                np.testing.assert_array_equal(parts[0][name]['Y'], other[name]['Y'])
        self.assertEqual(parts[0]['TRAIN']['X'].shape[1], 1)
        self.assertEqual(parts[-1]['TRAIN']['X'].shape[1], 7)

    def test_feature_space_selectors_ignore_heldout_values(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        frame = self.frame.assign(**{'WIND_DIR[deg]': np.arange(len(self.frame)), 'Pressure[mbar]': 1000.})
        changed = frame.copy()
        changed.iloc[98:] *= 9
        original_config = copy.deepcopy(self.config)
        expected_data = arm_config(self.config, list(EXPANDED_FEATURES),
                                   'XGBoost', self.config['seed'])['data']
        with tempfile.TemporaryDirectory() as directory:
            with patch('src.data.load_data', return_value=frame) as load:
                before = prepare_spaces(self.config, Path(directory))
                load.assert_called_once_with(expected_data)
            with patch('src.data.load_data', return_value=changed):
                after = prepare_spaces(self.config, Path(directory))
        self.assertEqual(self.config, original_config)
        self.assertEqual(before, after)
        self.assertEqual(before['Expanded-Physics'], list(PHYSICS_SELECTED))
        self.assertEqual(before['Intrinsic'], [])

    def test_rmse_is_root_of_mean_square_per_horizon(self):
        y = np.array([[0., 2.], [2., 4.]])
        p = y + np.array([[0., 2.], [4., 0.]])
        scores = evaluate(y, p, y, 7.44).query('model == "Forecaster"')
        np.testing.assert_allclose(scores.RMSE, np.sqrt([8., 2.]))
        np.testing.assert_allclose(scores.R2, [-7., -1.])
        constant = evaluate(np.zeros((2, 1)), np.ones((2, 1)), np.zeros((2, 1)), 7.44)
        self.assertTrue(constant.R2.isna().all())


if __name__ == '__main__':
    unittest.main()
