"""Regression coverage for subset isolation, selection and corrected metrics."""
import copy
import unittest

from src.runtime import configure_runtime
configure_runtime()
import numpy as np
import pandas as pd

from src.config import arm_config, data_contract, load_config
from src.data import prepare_data
from src.experiments import _arms, _select
from src.data import CANDIDATE_FEATURES, build_features, feature_names
from src.metrics import evaluate


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('config/config.yaml')
        self.config['data'].update(pre=3, horizon=2)
        n = 140
        self.frame = pd.DataFrame({
            'POA Irr[kW1m2]': np.linspace(0, 1, n), 'GHI[kW1m2]': np.linspace(0, .9, n),
            'TEMPERATURE[degC]': np.full(n, 20.), 'WIND_SPEED[m1s]': np.full(n, 2.),
            'HUMIDITY[%]': np.full(n, 50.), 'P_Solar[kW]': np.linspace(0, 5, n),
        }, index=pd.date_range('2020-01-01', periods=n, freq='h'))

    def test_candidate_order_and_contract(self):
        config = arm_config(self.config, ['TempCell', 'Pac'], 'XGBoost', 11)
        self.assertEqual(feature_names(config['data'])[-2:], ['Pac', 'TempCell'])
        self.assertNotEqual(data_contract(config['data']), data_contract(self.config['data']))
        with self.assertRaisesRegex(ValueError, 'unique supported'):
            arm_config(self.config, ['future_power'], 'XGBoost', 11)
        with self.assertRaisesRegex(ValueError, 'unique supported'):
            arm_config(self.config, ['Pac', 'Pac'], 'XGBoost', 11)

    def test_physics_arms_extend_one_reference(self):
        reference = self.config['experiment']['reference_features']
        arms = _arms(self.config)
        self.assertEqual(arms[0]['id'], 'nonpi_reference')
        for arm in arms:
            self.assertEqual(arm['subset'][:len(reference)], reference)

    def test_selection_uses_validation_for_seed_11(self):
        arms = _arms(self.config)
        records = [{'model': model, 'arm': arm['id'], 'seed': 11,
                    'validation_mae': 1.0 + index / 100}
                   for model in ('XGBoost', 'CNN_LSTM')
                   for index, arm in enumerate(arms)]
        self.assertEqual(_select(records, arms, 11),
                         {'XGBoost': 'add_pac', 'CNN_LSTM': 'add_pac'})
        with self.assertRaisesRegex(ValueError, 'configured seed'):
            _select(records[:-1], arms, 11)

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
