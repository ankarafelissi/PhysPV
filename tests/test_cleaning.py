import unittest

import numpy as np
import pandas as pd

from src.cleaning import TRAINING_COLUMNS, clean_training_data


class CleaningTests(unittest.TestCase):

    def frame(self):
        values = [1., 15., .7, 2., 180., .2, .3]
        return pd.DataFrame(np.tile(values, (5, 1)), columns=TRAINING_COLUMNS,
                            index=pd.date_range('2018-06-01', periods=5, freq='5min'))

    def test_scoped_rules_and_review_do_not_rewrite_good_data(self):
        frame = self.frame()
        frame['Pressure[mbar]'] = 3000.
        frame.iloc[0, frame.columns.get_loc('TEMPERATURE[degC]')] = 44.
        cleaned, flags = clean_training_data(frame)
        pd.testing.assert_frame_equal(frame, cleaned, check_flags=False)
        self.assertTrue(flags.iloc[0]['TEMPERATURE[degC]'] & 16)
        self.assertNotIn('Pressure[mbar]', flags)

    def test_aggregate_wind_does_not_get_arbitrary_modulo_repair(self):
        frame = self.frame()
        frame['WIND_DIR[deg]'] = [360., 680., -1., 359., 0.]
        cleaned, flags = clean_training_data(frame)
        self.assertEqual(cleaned['WIND_DIR[deg]'].iloc[0], 0)
        self.assertTrue(cleaned['WIND_DIR[deg]'].iloc[1:3].isna().all())
        self.assertEqual(flags['WIND_DIR[deg]'].iloc[1] & 2, 2)

    def test_no_future_interpolation_or_model_fill(self):
        frame = self.frame()
        frame.loc[frame.index[1], ['HUMIDITY[%]', 'WIND_SPEED[m1s]']] = 0.
        frame.loc[frame.index[2], 'HUMIDITY[%]'] = 1.8
        frame.loc[frame.index[3], 'P_Solar[kW]'] = np.nan
        cleaned, flags = clean_training_data(frame)
        self.assertTrue(cleaned.loc[frame.index[1], ['HUMIDITY[%]', 'WIND_SPEED[m1s]']].isna().all())
        self.assertTrue(pd.isna(cleaned.loc[frame.index[3], 'P_Solar[kW]']))
        self.assertEqual(flags.loc[frame.index[2], 'HUMIDITY[%]'], 2)
        again, _ = clean_training_data(cleaned)
        pd.testing.assert_frame_equal(cleaned, again)

    def test_signed_offsets_and_standalone_calm_are_preserved(self):
        frame = self.frame()
        frame['GHI[kW1m2]'] = -.001
        frame['WIND_SPEED[m1s]'] = 0.
        cleaned, _ = clean_training_data(frame)
        pd.testing.assert_frame_equal(frame, cleaned)

    def test_gaps_are_not_compressed(self):
        with self.assertRaisesRegex(ValueError, 'continuous'):
            clean_training_data(self.frame().drop(self.frame().index[2]))

    def test_feature_construction_does_not_replace_missing_meter_value(self):
        from src.config import load_config
        from src.data import build_features
        config = load_config('config/config.yaml')['data']
        frame = self.frame()
        frame.loc[frame.index[2], 'P_Solar[kW]'] = np.nan
        cleaned, _ = clean_training_data(frame)
        features = build_features(cleaned, config)
        self.assertTrue(pd.isna(features.loc[frame.index[2], 'P_Solar[kW]']))


if __name__ == '__main__':
    unittest.main()
