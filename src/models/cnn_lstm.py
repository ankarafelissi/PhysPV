"""CNN-LSTM training and TensorFlow SavedModel persistence."""

def build_model(params, input_shape, horizon):
    """Use an unconstrained forecast head so negative scores retain gradients."""
    import tensorflow as tf
    if params.get('output_activation') not in (None, 'linear'):
        raise ValueError('CNN-LSTM requires a linear output; apply power constraints after inverse scaling.')
    neurons = params['neurons']
    if not neurons or any(type(n) is not int or n < 1 for n in neurons):
        raise ValueError('neurons must contain positive integers.')
    layers = tf.keras.layers
    model = tf.keras.Sequential([layers.Input(shape=input_shape)])
    model.add(layers.Conv1D(params['filters'], params['kernel_size'], padding='causal', activation=params['cnn_activation']))
    # Pooling is optional: with a short input window it can leave the LSTM with
    # fewer timesteps than it has layers. `pool_size: null` disables it.
    if params.get('pool_size'):
        model.add(layers.MaxPooling1D(params['pool_size'], padding='same'))
    for i, units in enumerate(neurons):
        model.add(layers.LSTM(units, activation=params['lstm_activation'], return_sequences=i < len(neurons)-1))
    model.add(layers.Dense(horizon, activation='linear'))
    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    return model


def train(params, splits, seed):
    train, val = splits['TRAIN'], splits['VAL']
    import numpy as np
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(params, train['X'].shape[1:], train['Y_scaled'].shape[1])

    class HorizonDiagnostics(tf.keras.callbacks.Callback):
        """Record each validation horizon without using TEST or changing selection."""
        def on_epoch_end(self, epoch, logs=None):
            predicted = self.model.predict(val['X'], verbose=0)
            if not np.isfinite(predicted).all():
                raise ValueError('CNN-LSTM produced non-finite validation forecasts.')
            for h in range(predicted.shape[1]):
                values = predicted[:, h]
                logs[f'val_mae_scaled_h{h+1}'] = float(np.mean(abs(values-val['Y_scaled'][:, h])))
                logs[f'val_prediction_span_scaled_h{h+1}'] = float(np.ptp(values))
                logs[f'val_zero_fraction_scaled_h{h+1}'] = float(np.mean(values == 0))

    history = model.fit(train['X'], train['Y_scaled'], validation_data=(val['X'], val['Y_scaled']),
                        epochs=params['epochs'], batch_size=params['batch_size'], shuffle=False, verbose=2,
                        callbacks=[HorizonDiagnostics(), tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=params['patience'], restore_best_weights=True)])
    return model, history.history


def save(model, directory):
    model.save(str(directory / "cnn_lstm"), save_format="tf")

def load(directory):
    from tensorflow.keras.models import load_model
    return load_model(str(directory / "cnn_lstm"))
