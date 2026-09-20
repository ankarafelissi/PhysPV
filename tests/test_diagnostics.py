"""Regression tests for dead forecast heads and per-horizon health checks."""
import unittest

from src.runtime import configure_runtime
configure_runtime()
import numpy as np
from src.config import load_config, validate_config
from src.diagnostics import forecast_health, require_healthy_forecasts


class ForecastHealthTests(unittest.TestCase):
    def test_zero_and_nonzero_constant_heads_are_detected(self):
        observed = np.arange(12.).reshape(6, 2)
        health = forecast_health(observed, np.tile([0., 2.], (6, 1)))
        self.assertTrue(health['collapsed'].all())
        with self.assertRaisesRegex(ValueError, r'horizons \[1, 2\]'):
            require_healthy_forecasts(health, 'Validation')

    def test_night_only_targets_are_not_a_collapsed_head(self):
        health = forecast_health(np.zeros((8, 2)), np.zeros((8, 2)))
        require_healthy_forecasts(health, 'Validation')
        self.assertFalse(health['collapsed'].any())

    def test_varying_forecasts_and_negative_raw_values_are_valid(self):
        observed = np.arange(12.).reshape(6, 2)
        health = forecast_health(observed, observed-3.)
        require_healthy_forecasts(health, 'Validation')
        self.assertTrue((health['prediction_span'] > 0).all())
        self.assertGreater(health['negative_fraction'].max(), 0)

    def test_nonfinite_forecasts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            forecast_health(np.ones((2, 1)), np.array([[1.], [np.nan]]))

    def test_relu_config_is_rejected(self):
        config = load_config('config/full.yaml')
        config['models']['CNN_LSTM']['output_activation'] = 'relu'
        with self.assertRaisesRegex(ValueError, 'linear'):
            validate_config(config)

    def test_negative_linear_head_keeps_corrective_gradient(self):
        # Load PyTables before TensorFlow to avoid Windows HDF5 DLL collisions.
        import tables
        import tensorflow as tf
        from src.models.cnn_lstm import build_model
        params = load_config('config/full.yaml')['models']['CNN_LSTM']
        tf.keras.backend.clear_session()
        model = build_model(params, (3, 2), 2)
        head = model.layers[-1]
        weights, bias = head.get_weights()
        head.set_weights([np.zeros_like(weights), -np.ones_like(bias)])
        inputs = tf.zeros((4, 3, 2))
        with tf.GradientTape() as tape:
            prediction = model(inputs)
            loss = tf.reduce_mean(tf.abs(prediction-tf.ones((4, 2))))
        gradient = tape.gradient(loss, head.bias)
        np.testing.assert_allclose(prediction.numpy(), -1.)
        self.assertTrue(np.all(gradient.numpy() < 0))
        head.bias.assign_sub(.1 * gradient)
        self.assertTrue(np.all(model(inputs).numpy() > prediction.numpy()))
