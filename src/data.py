"""PV features, timestamped inputs, chronological windows and train-only scaling."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .runtime import project_path


PHYSICS_FEATURES = ('Pac', 'Pdc', 'TempModule', 'TempCell')
RESERVED_PHYSICS = (*PHYSICS_FEATURES, 'TempModule_RP')
PHYSICAL_INPUTS = ('POA Irr[kW1m2]', 'TEMPERATURE[degC]', 'WIND_SPEED[m1s]')
CANDIDATE_FEATURES = ('POA Irr[kW1m2]', 'GHI[kW1m2]', *PHYSICS_FEATURES, 'HoursOfDay')


def feature_names(config):
    """YAML lists common features; the switch controls a fixed appended block."""
    if type(config.get('physics')) is not bool:
        raise ValueError('data.physics must be an explicit boolean.')
    base = config.get('features')
    if not isinstance(base, list) or not base or any(not isinstance(c, str) for c in base):
        raise ValueError('data.features must be a nonempty list of column names.')
    if len(base) != len(set(base)):
        raise ValueError('data.features must not contain duplicates.')
    if set(base).intersection(RESERVED_PHYSICS):
        raise ValueError('Do not list physics columns in data.features; use data.physics.')
    if 'candidate_subset' in config:
        subset = config['candidate_subset']
        if (not isinstance(subset, list) or len(subset) != len(set(subset))
                or not set(subset).issubset(CANDIDATE_FEATURES)
                or set(base).intersection(CANDIDATE_FEATURES)):
            raise ValueError('candidate_subset must contain unique supported candidates, separate from common features.')
        if config['physics'] != bool(set(subset).intersection(PHYSICS_FEATURES)):
            raise ValueError('data.physics must match the presence of derived physical candidates.')
        return list(base) + [name for name in CANDIDATE_FEATURES if name in subset]
    return base + list(PHYSICS_FEATURES) if config['physics'] else list(base)


def validate_plant(plant):
    """Validate reference units and return the configured DC nameplate in kW."""
    if not isinstance(plant, dict) or not plant.get('arrays'):
        raise ValueError('data.plant must define at least one PV array.')
    grid = np.asarray(plant.get('efficiency_input_w', []), dtype=float)
    efficiency = np.asarray(plant.get('efficiency_fraction', []), dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or grid.shape != efficiency.shape
            or not np.isfinite(grid).all() or not np.isfinite(efficiency).all()
            or grid[0] != 0 or np.any(np.diff(grid) <= 0)
            or np.any((efficiency < 0) | (efficiency > 1))):
        raise ValueError('Plant efficiency curve requires increasing W inputs from zero and fractions in [0, 1].')
    for key in ('reference_irradiance_w_m2', 'reference_temperature_c'):
        if not np.isfinite(plant.get(key, np.nan)):
            raise ValueError(f'Plant {key} must be finite.')
    if plant['reference_irradiance_w_m2'] <= 0:
        raise ValueError('Reference irradiance must be positive.')
    rating = plant.get('ac_capacity_kw')
    if rating is not None and (not np.isfinite(rating) or rating <= 0):
        raise ValueError('ac_capacity_kw must be a confirmed positive rating or null.')
    capacity_w = 0.0
    for array in plant['arrays']:
        for key in ('module_power_w', 'modules_series', 'strings_parallel',
                    'temperature_coefficient_per_c', 'module_a', 'module_b', 'cell_delta_c'):
            if not np.isfinite(array.get(key, np.nan)):
                raise ValueError(f'Array {key} must be finite.')
        if array['module_power_w'] <= 0 or array['cell_delta_c'] < 0:
            raise ValueError('Array power must be positive and cell delta nonnegative.')
        for key in ('modules_series', 'strings_parallel'):
            if type(array[key]) is not int or array[key] < 1:
                raise ValueError(f'Array {key} must be a positive integer.')
        capacity_w += array['module_power_w'] * array['modules_series'] * array['strings_parallel']
    return capacity_w / 1000.0


def pv_power_features(frame, plant):
    """Estimate temperatures (C) and power (kW), preserving invalid observations.

    Module/cell temperature follows the supplied exponential temperature model.
    Efficiency is interpolated per array; AC capacity, if known, clips only the
    total AC power. Curve endpoints are never interpreted as equipment ratings.
    """
    validate_plant(plant)
    missing = set(PHYSICAL_INPUTS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing physical inputs: {sorted(missing)}')
    inputs = frame[list(PHYSICAL_INPUTS)].to_numpy(dtype=float)
    irradiance, ambient, wind = inputs.T
    valid = np.isfinite(inputs).all(axis=1) & (ambient > -273.15) & (wind >= 0)
    # Negative irradiance sensor noise is not rewritten in the measured frame.
    poa = np.maximum(irradiance, 0) * 1000.0
    dc, ac, modules, cells = [], [], [], []
    with np.errstate(over='ignore', invalid='ignore'):
        for array in plant['arrays']:
            module = ambient + poa * np.exp(array['module_a'] + array['module_b'] * wind)
            cell = module + poa / plant['reference_irradiance_w_m2'] * array['cell_delta_c']
            rated_w = array['module_power_w'] * array['modules_series'] * array['strings_parallel']
            power = np.maximum(0, rated_w * poa / plant['reference_irradiance_w_m2'] * (
                1 + array['temperature_coefficient_per_c'] * (cell - plant['reference_temperature_c'])))
            efficiency = np.interp(power, plant['efficiency_input_w'], plant['efficiency_fraction'])
            dc.append(power)
            ac.append(power * efficiency)
            modules.append(module)
            cells.append(cell)
    values = np.column_stack((np.sum(ac, axis=0) / 1000, np.sum(dc, axis=0) / 1000,
                              np.mean(modules, axis=0), np.mean(cells, axis=0)))
    if plant.get('ac_capacity_kw') is not None:
        values[:, 0] = np.minimum(values[:, 0], plant['ac_capacity_kw'])
    valid &= np.isfinite(values).all(axis=1)
    values[~valid] = np.nan
    return pd.DataFrame(values, index=frame.index, columns=PHYSICS_FEATURES)


def build_features(frame, config):
    """Build both ablation arms on a common physical-input validity mask."""
    names = feature_names(config)
    if config.get('correct_power') is not False:
        raise ValueError('data.correct_power must remain false for the physics ablation.')
    result = frame.drop(columns=list(RESERVED_PHYSICS), errors='ignore').copy()
    physical = pv_power_features(result, config['plant'])
    # Compute eligibility in both arms, but expose no physics columns when off.
    result.attrs['eligible_rows'] = np.isfinite(physical.to_numpy()).all(axis=1)
    if 'candidate_subset' in config:
        # Every arm is eligible on the full pool, including an unused GHI sensor.
        result.attrs['eligible_rows'] &= np.isfinite(result[['GHI[kW1m2]']].to_numpy()).all(axis=1)
    if 'HoursOfDay' in names:
        result['HoursOfDay'] = result.index.hour
    for name, column, operation in (
        ('MeanPrevH', config['target'], 'mean'), ('StdPrevH', config['target'], 'std'),
        ('MeanWindSpeedPrevH', 'WIND_SPEED[m1s]', 'mean'),
        ('StdWindSpeedPrevH', 'WIND_SPEED[m1s]', 'std'),
    ):
        if name in names:
            rolling = result[column].rolling(config['horizon'])
            result[name] = rolling.mean() if operation == 'mean' else rolling.std(ddof=0)
    selected_physics = set(names).intersection(PHYSICS_FEATURES)
    if selected_physics:
        for name in PHYSICS_FEATURES:
            if name not in selected_physics:
                continue
            result[name] = physical[name]
    if set(result.columns).intersection(RESERVED_PHYSICS) != selected_physics:
        raise ValueError('Physics feature invariant violated.')
    return result


def input_identity(config):
    path = project_path(config['path']).resolve()
    info = path.stat()
    return {'path': str(path), 'size_bytes': info.st_size, 'modified_ns': info.st_mtime_ns}


def partition_metadata(splits):
    return {key: {'n_origins': len(part['origins']),
                  'first_origin': str(part['origins'][0]), 'last_origin': str(part['origins'][-1]),
                  'target_start': str(part['target_times'][0, 0]),
                  'target_end': str(part['target_times'][-1, -1])}
            for key, part in splits.items()}


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
