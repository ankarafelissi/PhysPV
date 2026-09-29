"""CNN-LSTM training and TensorFlow SavedModel persistence."""


def build_optimizer(spec):
    """Build a serializable optimizer specification from YAML."""
    import tensorflow as tf
    if isinstance(spec, str):
        return tf.keras.optimizers.get(spec)
    if not isinstance(spec, dict) or set(spec) - {'name', 'learning_rate', 'clipnorm'}:
        raise ValueError('optimizer must be a name or a mapping with name, learning_rate and optional clipnorm.')
    name = spec.get('name')
    if not isinstance(name, str) or not name:
        raise ValueError('optimizer.name must be a nonempty string.')
    kwargs = {}
    for key in ('learning_rate', 'clipnorm'):
        if key in spec:
            value = spec[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f'optimizer.{key} must be positive.')
            kwargs[key] = float(value)
    optimizer = tf.keras.optimizers.get(name)
    config = optimizer.get_config()
    config.update(kwargs)
    return optimizer.__class__.from_config(config)


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
    model.compile(optimizer=build_optimizer(params['optimizer']), loss=params['loss'])
    return model


def train(params, splits, seed):
    """Fit once in chronological order and restore the best validation checkpoint."""
    train, val = splits['TRAIN'], splits['VAL']
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(params, train['X'].shape[1:], train['Y_scaled'].shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=params['patience'], restore_best_weights=True)
    history = model.fit(train['X'], train['Y_scaled'], validation_data=(val['X'], val['Y_scaled']),
                        epochs=params['epochs'], batch_size=params['batch_size'], shuffle=False,
                        verbose=params.get('verbose', 2),
                        callbacks=[early_stopping])
    return model, history.history


def save(model, directory):
    model.save(str(directory / "cnn_lstm"), save_format="tf")


def load(directory):
    from tensorflow.keras.models import load_model
    return load_model(str(directory / "cnn_lstm"))
