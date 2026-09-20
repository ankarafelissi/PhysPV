"""Read timestamped data and build aligned, leakage-safe forecast samples."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .features import feature_names
from .paths import project_path


def load_data(config):
    frame = pd.read_hdf(project_path(config['path']))
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError('Data must have a DatetimeIndex.')
    if frame.index.has_duplicates:
        raise ValueError('Timestamps must be unique; resolve duplicate observations first.')
    # The full SOLETE file stores some day blocks out of chronological order.
    frame = frame.sort_index()
    delta = pd.to_timedelta(config['resolution'])
    if delta <= pd.Timedelta(0):
        raise ValueError('resolution must be positive.')
    if len(frame) > 1 and ((frame.index - frame.index[0]) % delta != pd.Timedelta(0)).any():
        raise ValueError('Timestamps do not align with the configured resolution.')
    # Missing timestamps remain NaN; no window may jump over a data gap.
    frame = frame.asfreq(delta)
    limit = config.get('max_rows')
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError('max_rows must be a positive integer or null.')
        frame = frame.iloc[:limit].copy()
    return frame


def make_scaler(name):
    if name == 'MinMax01':
        return MinMaxScaler((0, 1))
    if name == 'MinMax11':
        return MinMaxScaler((-1, 1))
    if name == 'Standard':
        return StandardScaler()
    raise ValueError('scaler must be MinMax01, MinMax11 or Standard.')


def prepare_data(frame, config, scalers=None):
    """Split by target timestamps; validation/test may use earlier observed context.

    Each input contains t-PRE through t; labels contain t+1 through t+H.
    All labels of a sample must belong to one chronological partition.
    """
    pre, horizon = config['pre'], config['horizon']
    if type(pre) is not int or pre < 0 or type(horizon) is not int or horizon < 1:
        raise ValueError('pre must be a nonnegative integer; horizon a positive integer.')
    features, target = feature_names(config), config['target']
    if not isinstance(target, str):
        raise ValueError('This pipeline supports one target and multiple forecast steps.')
    if not features or len(features) != len(set(features)):
        raise ValueError('features must be nonempty and unique.')
    missing = set(features + [target]) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    ratios = np.asarray(config['split'], dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError('split must contain three positive fractions summing to one.')
    n = len(frame)
    cut1 = int(np.floor(n * ratios[0] + 1e-9))
    cut2 = int(np.floor(n * (ratios[0] + ratios[1]) + 1e-9))
    values = frame[features].to_numpy(dtype=float)
    labels = frame[target].to_numpy(dtype=float)
    eligibility = np.asarray(frame.attrs.get('eligible_rows', np.ones(n, dtype=bool)), dtype=bool)
    if eligibility.shape != (n,):
        raise ValueError('Feature eligibility mask does not match the time axis.')
    splits = {}
    for name, start, end in [('TRAIN', 0, cut1), ('VAL', cut1, cut2), ('TEST', cut2, n)]:
        origins = np.arange(max(pre, start - 1), end - horizon)
        origins = np.asarray([t for t in origins
                              if np.isfinite(values[t-pre:t+1]).all()
                              and eligibility[t-pre:t+1].all()
                              and np.isfinite(labels[t-pre:t+horizon+1]).all()], dtype=int)
        if len(origins) == 0:
            raise ValueError(f'{name} has no valid windows: rows={n}, pre={pre}, '
                             f'horizon={horizon}. Use more data or shorter windows.')
        splits[name] = {
            'X': np.stack([values[t-pre:t+1] for t in origins]),
            'Y': np.stack([labels[t+1:t+horizon+1] for t in origins]),
            'persistence': np.repeat(labels[origins, None], horizon, axis=1),
            'origins': frame.index[origins],
            'target_times': np.stack([frame.index[t+1:t+horizon+1].to_numpy() for t in origins]),
        }
        if 'POA Irr[kW1m2]' in frame:
            poa = frame['POA Irr[kW1m2]'].to_numpy(dtype=float)
            splits[name]['sky_variability'] = np.array([np.std(poa[t-pre:t+1]) for t in origins])
            splits[name]['power_level'] = np.array([np.mean(labels[t-pre:t+1]) for t in origins])
            # Used only for post-hoc stratification, never included in X.
            splits[name]['ramp_magnitude'] = np.abs(splits[name]['Y'] - labels[origins, None])
    if scalers is None:
        scalers = {'X': make_scaler(config['scaler']), 'Y': make_scaler(config['scaler'])}
        scalers['X'].fit(splits['TRAIN']['X'].reshape(-1, len(features)))
        scalers['Y'].fit(splits['TRAIN']['Y'].reshape(-1, 1))
    for part in splits.values():
        part['X'] = scalers['X'].transform(part['X'].reshape(-1, len(features))).reshape(part['X'].shape).astype('float32')
        part['Y_scaled'] = scalers['Y'].transform(part['Y'].reshape(-1, 1)).reshape(part['Y'].shape).astype('float32')
    return splits, scalers
