"""TRAIN-fitted scenario cutoffs; future ramp labels remain analysis-only."""
import numpy as np


def fit_thresholds(train):
    for key in ('sky_variability', 'power_level', 'ramp_magnitude'):
        if key not in train or not np.isfinite(train[key]).all():
            raise ValueError(f'Finite TRAIN {key} is required for scenario thresholds.')
    return {'sky_variability': float(np.median(train['sky_variability'])),
            'power_level': float(np.median(train['power_level'])),
            'ramp_magnitude': np.median(train['ramp_magnitude'], axis=0).tolist(),
            'method': 'TRAIN median; sky=trailing POA std(ddof=0); '
                      'power=trailing target mean; ramp=abs(y[t+h]-y[t]), post-hoc only'}


def scenario_labels(part, thresholds):
    horizon = part['Y'].shape[1]
    ramps = np.asarray(thresholds['ramp_magnitude'])
    if ramps.shape != (horizon,):
        raise ValueError('Stored ramp thresholds do not match the forecast horizon.')
    return {
        'sky_condition': np.repeat(np.where(part['sky_variability'] <= thresholds['sky_variability'],
                                            'clear', 'cloudy')[:, None], horizon, axis=1),
        'power_level_class': np.repeat(np.where(part['power_level'] <= thresholds['power_level'],
                                                'low', 'high')[:, None], horizon, axis=1),
        'ramp_class': np.where(part['ramp_magnitude'] <= ramps, 'steady', 'ramp'),
    }
