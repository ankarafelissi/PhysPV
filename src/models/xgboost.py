"""Direct multi-step XGBoost: one independently validated tree ensemble per step."""
import json
from pathlib import Path

import numpy as np


class XGBoostForecaster:
    def __init__(self, params=None, seed=32):
        self.params = dict(params or {})
        self.params.setdefault('random_state', seed)
        self.estimators = []

    @staticmethod
    def flatten(x):
        x = np.asarray(x)
        if x.ndim not in (2, 3):
            raise ValueError('X must be a 2D feature matrix or 3D history window.')
        return x.reshape(len(x), -1)

    def fit(self, x, y, x_val, y_val):
        from xgboost import XGBRegressor
        x, x_val = self.flatten(x), self.flatten(x_val)
        if len(x) != len(y) or len(x_val) != len(y_val) or x.shape[1] != x_val.shape[1]:
            raise ValueError('Training/validation sample counts and feature dimensions must match their targets.')
        if y.ndim != 2 or y_val.ndim != 2 or y.shape[1] != y_val.shape[1]:
            raise ValueError('Targets must have shape (samples, horizon).')
        self.estimators = []
        for step in range(y.shape[1]):
            estimator = XGBRegressor(**self.params)
            estimator.fit(x, y[:, step], eval_set=[(x_val, y_val[:, step])], verbose=False)
            self.estimators.append(estimator)
        return self

    def predict(self, x):
        if not self.estimators:
            raise ValueError('Fit or load the model before prediction.')
        return np.column_stack([model.predict(self.flatten(x)) for model in self.estimators])

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if not self.estimators:
            raise ValueError('Cannot save an unfitted model.')
        for step, estimator in enumerate(self.estimators):
            estimator.save_model(str(directory / f'step_{step+1}.json'))
        (directory / 'manifest.json').write_text(json.dumps({
            'horizon': len(self.estimators), 'params': self.params}, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, directory):
        from xgboost import XGBRegressor
        directory = Path(directory)
        metadata = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        result = cls(metadata['params'])
        for step in range(metadata['horizon']):
            estimator = XGBRegressor()
            estimator.load_model(str(directory / f'step_{step+1}.json'))
            estimator.set_params(n_jobs=result.params.get('n_jobs', 2))
            result.estimators.append(estimator)
        return result
