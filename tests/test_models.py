"""Daylight scores, forecast health and model persistence; no fitting."""
import copy
import tempfile
import unittest
from pathlib import Path
from src.runtime import configure_runtime
configure_runtime()
import numpy as np
import pandas as pd
from src.config import load_config
from src.data import solar_features
from src.evaluate import evaluate, forecast_health, paired_bootstrap, require_healthy_forecasts
from src.models import build_model, load_model, save_model


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.test-', dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config('config/config.yaml')

    def test_daylight_metrics_and_health(self):
        truth = np.array([[1., 2.], [100., 200.], [3., 4.]])
        predicted = truth.copy()
        predicted[1] = 0
        mask = np.array([[True, True], [False, False], [True, True]])
        scores = evaluate(truth, predicted, truth, 10., mask=mask)
        self.assertTrue((scores.RMSE == 0).all())
        self.assertTrue((scores.n == 2).all())
        self.assertGreater(evaluate(truth, predicted, truth, 10.).RMSE.max(), 0)
        with self.assertRaises(ValueError):
            require_healthy_forecasts(forecast_health(truth, np.zeros_like(truth)), 'VAL')
        require_healthy_forecasts(forecast_health(np.zeros((3, 2)), np.zeros((3, 2))), 'night')
        interval = paired_bootstrap(np.full(20, -.5), 1000, 11, 4)
        self.assertEqual((interval['ci_low'], interval['ci_high']), (-.5, -.5))
        config = copy.deepcopy(self.config['data'])
        config['solar'] = {'site': {'latitude': 0., 'longitude': 0., 'timezone': 'UTC'}}
        frame = pd.DataFrame({'GHI[kW1m2]': .5}, index=pd.date_range('2020-03-20', periods=24, freq='h'))
        sun = solar_features(frame, config)
        self.assertLess(sun.solar_elevation.iloc[0], -80)
        self.assertGreater(sun.solar_elevation.iloc[12], 80)


    def test_model_persistence_and_linear_head_without_training(self):
        import tables  # Load HDF5 before TensorFlow on Windows.
        import tensorflow as tf
        params = self.config['models']['CNN_LSTM']
        model = build_model(params, (3, 2), 2)
        head = model.layers[-1]
        weights, bias = head.get_weights()
        head.set_weights([np.zeros_like(weights), -np.ones_like(bias)])
        with tf.GradientTape() as tape:
            prediction = model(tf.zeros((4, 3, 2)))
            loss = tf.reduce_mean(tf.abs(prediction - 1))
        self.assertTrue(np.all(tape.gradient(loss, head.bias).numpy() < 0))
        artifact = self.root / 'model'
        save_model(model, 'CNN_LSTM', artifact)
        np.testing.assert_allclose(load_model('CNN_LSTM', artifact)(tf.zeros((4, 3, 2))), prediction)



if __name__ == '__main__':
    unittest.main()
