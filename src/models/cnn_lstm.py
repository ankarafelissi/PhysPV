"""CNN-LSTM training and TensorFlow SavedModel persistence."""
import math


class ValidationLossGuard:
    """Track repeated severe validation-loss excursions."""

    def __init__(self, patience=3, factor=2.0, minimum_increase=0.05):
        self.patience = patience
        self.factor = factor
        self.minimum_increase = minimum_increase
        self.best = math.inf
        self.streak = 0

    def observe(self, value):
        if value is None or not math.isfinite(value):
            self.streak = self.patience
            return True
        if value < self.best:
            self.best = value
            self.streak = 0
            return False
        abnormal = value > self.best * self.factor and value > self.best + self.minimum_increase
        self.streak = self.streak + 1 if abnormal else 0
        return self.streak >= self.patience


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
    model.compile(
        optimizer=build_optimizer(params['optimizer']),
        loss=params['loss'],
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')],
    )
    return model


def _gpu_status(tf):
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        return 'GPU=unavailable (CPU)'
    try:
        memory = tf.config.experimental.get_memory_info('GPU:0')
        current = memory['current'] / 1024 ** 2
        peak = memory['peak'] / 1024 ** 2
        return f'GPU={devices[0].name}, memory={current:.0f} MiB, peak={peak:.0f} MiB'
    except Exception:
        return f'GPU={devices[0].name}'


def _callbacks(tf, checkpoint_path, logger, patience):
    guard = ValidationLossGuard()

    class TrainingMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            number = epoch + 1
            val_loss = logs.get('val_loss')
            if number == 1 or number % 10 == 0:
                message = (
                    f'Epoch {number}: loss={logs.get("loss", float("nan")):.6f}, '
                    f'val_loss={val_loss if val_loss is not None else float("nan"):.6f}, '
                    f'val_mae={logs.get("val_mae", float("nan")):.6f}, {_gpu_status(tf)}')
                print(message, flush=True)
                logger.info(message)
            if guard.observe(val_loss):
                message = (
                    f'Stopping at epoch {number}: validation loss was non-finite or severely '
                    f'abnormal for {guard.streak} consecutive epoch(s); best={guard.best:.6f}.')
                print(message, flush=True)
                logger.error(message)
                self.model.stop_training = True
            elif guard.streak:
                logger.warning(
                    'Abnormal validation loss at epoch %s (%s/%s): %.6f; best=%.6f',
                    number, guard.streak, guard.patience, val_loss, guard.best)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor='val_loss', mode='min',
            save_best_only=True, save_weights_only=True, verbose=0),
        TrainingMonitor(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=False),
    ]


def train(params, splits, seed, checkpoint_path, logger):
    """Fit once in chronological order and restore the best validation checkpoint."""
    train, val = splits['TRAIN'], splits['VAL']
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(params, train['X'].shape[1:], train['Y_scaled'].shape[1])
    callbacks = _callbacks(tf, checkpoint_path, logger, params['patience'])
    history = model.fit(train['X'], train['Y_scaled'], validation_data=(val['X'], val['Y_scaled']),
                        epochs=params['epochs'], batch_size=params['batch_size'], shuffle=False,
                        verbose=params.get('verbose', 2),
                        callbacks=callbacks)
    if not checkpoint_path.with_suffix('.index').exists():
        raise RuntimeError('Training ended without a valid best checkpoint.')
    model.load_weights(str(checkpoint_path))
    return model, history.history


def save(model, directory):
    model.save(str(directory / "cnn_lstm"), save_format="tf")


def load(directory):
    from tensorflow.keras.models import load_model
    return load_model(str(directory / "cnn_lstm"))
