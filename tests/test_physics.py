"""Physical validity, ablation comparability and causal scenario regression tests."""
import copy
import unittest

from src.runtime import configure_runtime
configure_runtime()
import numpy as np
import pandas as pd

from src.config import data_contract, load_config, validate_config
from src.constraints import constrain_power
from src.data import prepare_data
from src.features import PHYSICS_FEATURES, build_features, feature_names, pv_power_features
from src.metrics import evaluate, paired_bootstrap
from src.scenarios import fit_thresholds, scenario_labels


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('config/config.yaml')
        self.data = self.config['data']
        n = 120
        self.frame = pd.DataFrame({
            'POA Irr[kW1m2]': np.linspace(0, 1, n),
            'TEMPERATURE[degC]': np.full(n, 25.), 'WIND_SPEED[m1s]': np.full(n, 2.),
            'P_Solar[kW]': np.linspace(0, 5, n),
        }, index=pd.date_range('2020-01-01', periods=n, freq='h'))
        self.data.update(features=['POA Irr[kW1m2]', 'P_Solar[kW]', 'HoursOfDay'], pre=3, horizon=4)

    def test_physics_switch_and_fixed_order(self):
        off = build_features(self.frame.assign(Pac=-999), self.data)
        self.assertFalse(set(PHYSICS_FEATURES).intersection(off.columns))
        self.data['physics'] = True
        on = build_features(self.frame, self.data)
        self.assertEqual(list(on.columns[-4:]), list(PHYSICS_FEATURES))
        self.assertEqual(feature_names(self.data)[-4:], list(PHYSICS_FEATURES))
        self.data['features'].append('Pac')
        with self.assertRaisesRegex(ValueError, 'Do not list physics'):
            feature_names(self.data)

    def test_common_windows_even_if_physical_inputs_missing(self):
        self.frame.iloc[20, self.frame.columns.get_loc('WIND_SPEED[m1s]')] = np.nan
        parts = []
        for switch in (False, True):
            self.data['physics'] = switch
            parts.append(prepare_data(build_features(self.frame, self.data), self.data)[0])
        for partition in ('TRAIN', 'VAL', 'TEST'):
            np.testing.assert_array_equal(parts[0][partition]['origins'], parts[1][partition]['origins'])
            np.testing.assert_array_equal(parts[0][partition]['Y'], parts[1][partition]['Y'])

    def test_missing_power_stays_missing_and_clipping_is_isolated(self):
        frame = self.frame.iloc[:3].copy()
        frame['POA Irr[kW1m2]'] = [3., np.nan, 0.]
        raw = pv_power_features(frame, self.data['plant'])
        plant = copy.deepcopy(self.data['plant'])
        plant['ac_capacity_kw'] = 1.
        clipped = pv_power_features(frame, plant)
        self.assertLessEqual(clipped['Pac'].iloc[0], 1.)
        self.assertLess(clipped['TempModule'].iloc[0], 200.)
        pd.testing.assert_frame_equal(raw.drop(columns='Pac'), clipped.drop(columns='Pac'))
        self.assertTrue(clipped.iloc[1].isna().all())
        self.assertEqual(clipped['Pac'].iloc[2], 0.)

    def test_efficiency_is_used_and_observations_untouched(self):
        original = self.frame.copy(deep=True)
        plant = copy.deepcopy(self.data['plant'])
        plant['efficiency_fraction'] = [0.5] * len(plant['efficiency_fraction'])
        values = pv_power_features(self.frame, plant)
        np.testing.assert_allclose(values['Pac'], values['Pdc'] * .5)
        pd.testing.assert_frame_equal(original, self.frame)

    def test_feature_and_sky_labels_are_causal(self):
        self.data.update(physics=True)
        before = build_features(self.frame, self.data)
        changed = self.frame.copy()
        changed.iloc[80:] *= 2
        after = build_features(changed, self.data)
        pd.testing.assert_frame_equal(before.iloc[:80], after.iloc[:80])
        # TRAIN ends at 84; change TEST only so fitted thresholds cannot change.
        changed = self.frame.copy()
        changed.iloc[108:] *= 10
        original_parts, _ = prepare_data(before, self.data)
        new_parts, _ = prepare_data(build_features(changed, self.data), self.data)
        thresholds = fit_thresholds(original_parts['TRAIN'])
        self.assertEqual(thresholds, fit_thresholds(new_parts['TRAIN']))
        labels = scenario_labels(original_parts['TEST'], thresholds)
        self.assertEqual(labels['ramp_class'].shape, original_parts['TEST']['Y'].shape)

    def test_capacity_contract_and_target_protection(self):
        before = data_contract(self.data)
        self.data['physics'] = True
        self.assertNotEqual(before, data_contract(self.data))
        self.data['p_nom_kw'] = 10
        with self.assertRaisesRegex(ValueError, 'nameplate'):
            validate_config(self.config)
        self.data['p_nom_kw'] = 7.44
        self.data['correct_power'] = True
        with self.assertRaisesRegex(ValueError, 'correct_power'):
            validate_config(self.config)

    def test_constraints_preserve_raw_and_require_real_ac_capacity(self):
        raw = np.array([[-1., 9.]])
        result = constrain_power(raw, self.data)
        np.testing.assert_array_equal(raw, [[-1., 9.]])
        np.testing.assert_array_equal(result, [[0., 9.]])
        self.data['constraints']['capacity'] = True
        with self.assertRaisesRegex(ValueError, 'capacity'):
            constrain_power(raw, self.data)
        self.data['plant']['ac_capacity_kw'] = 8.
        np.testing.assert_array_equal(constrain_power(raw, self.data), [[0., 8.]])

    def test_nrmse_and_paired_bootstrap(self):
        y = np.zeros((20, 2))
        metrics = evaluate(y, y + 1., y, 7.44)
        np.testing.assert_allclose(metrics.query('model == "Forecaster"')['nRMSE_cap'], 100/7.44)
        with self.assertRaisesRegex(ValueError, 'positive'):
            evaluate(y, y, y, 0)
        interval = paired_bootstrap(np.full(30, -.5), 1000, 32, 4)
        self.assertEqual(interval['ci_low'], -.5)
        self.assertEqual(interval['ci_high'], -.5)


if __name__ == '__main__':
    unittest.main()
