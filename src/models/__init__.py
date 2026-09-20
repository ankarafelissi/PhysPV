"""Model dispatch for the CNN-LSTM versus XGBoost comparison."""
from . import cnn_lstm
from .xgboost import XGBoostForecaster

SUPPORTED_MODELS = ('CNN_LSTM', 'XGBoost')


def train_model(name, params, splits, seed):
    if name == 'CNN_LSTM':
        return cnn_lstm.train(params, splits, seed)
    if name == 'XGBoost':
        train, val = splits['TRAIN'], splits['VAL']
        model = XGBoostForecaster(params, seed).fit(
            train['X'], train['Y_scaled'], val['X'], val['Y_scaled'])
        return model, {str(i+1): m.evals_result() for i, m in enumerate(model.estimators)}
    raise ValueError(f'Unsupported model: {name}; choose {SUPPORTED_MODELS}')


def save_model(model, name, directory):
    if name == 'CNN_LSTM':
        cnn_lstm.save(model, directory)
    elif name == 'XGBoost':
        model.save(directory / 'xgboost')
    else:
        raise ValueError(f'Unsupported model: {name}')


def load_model(name, directory):
    if name == 'CNN_LSTM':
        return cnn_lstm.load(directory)
    if name == 'XGBoost':
        return XGBoostForecaster.load(directory / 'xgboost')
    raise ValueError(f'Unsupported model: {name}')


def predict_scaled(model, name, windows):
    if name == 'CNN_LSTM':
        return model.predict(windows, verbose=0)
    if name == 'XGBoost':
        return model.predict(windows)
    raise ValueError(f'Unsupported model: {name}')
