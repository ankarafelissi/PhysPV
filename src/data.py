"""PV features, timestamped inputs, chronological windows and train-only scaling."""

import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .runtime import project_path


PHYSICS_FEATURES = ('Pac', 'Pdc', 'TempModule', 'TempCell')
STATE_FEATURES = ('physics_residual', 'performance_ratio')
SOLAR_FEATURES = ('clear_sky_index', 'solar_elevation', 'solar_zenith')
DERIVED_FEATURES = (*PHYSICS_FEATURES, *STATE_FEATURES, *SOLAR_FEATURES)
RESERVED_PHYSICS = (*DERIVED_FEATURES, 'TempModule_RP')
PHYSICAL_INPUTS = ('POA Irr[kW1m2]', 'TEMPERATURE[degC]', 'WIND_SPEED[m1s]')
CANDIDATE_FEATURES = ('POA Irr[kW1m2]', 'GHI[kW1m2]', *DERIVED_FEATURES, 'HoursOfDay')
ORIGINAL_FEATURES = (
    'TEMPERATURE[degC]',
    'HUMIDITY[%]',
    'WIND_SPEED[m1s]',
    'WIND_DIR[deg]',
    'GHI[kW1m2]',
    'POA Irr[kW1m2]',
)
SUPPORTED_FEATURES = tuple(
    dict.fromkeys((*ORIGINAL_FEATURES, *CANDIDATE_FEATURES, 'MinutesOfDay'))
)
SPACE_NAMES = ('Original', 'Expanded-Pearson', 'Expanded-Physics', 'Intrinsic')
EXPANDED_FEATURES = (
    *ORIGINAL_FEATURES,
    *PHYSICS_FEATURES,
    'MinutesOfDay',
    'HoursOfDay',
)
PHYSICS_SELECTED = ('HUMIDITY[%]', 'POA Irr[kW1m2]', 'Pac', 'TempCell', 'HoursOfDay')


def feature_names(config):
    """Return the explicit input columns in their declared order."""
    names = config.get('features')
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(c, str) or not c for c in names)
    ):
        raise ValueError('data.features must be a nonempty list of column names.')
    if len(names) != len(set(names)):
        raise ValueError('data.features must not contain duplicates.')
    return list(names)


def uses_physics(config):
    """Whether the declared inputs contain a derived physical feature."""
    return bool(set(feature_names(config)).intersection(DERIVED_FEATURES))


def validate_plant(plant):
    """Validate reference units and return the configured DC nameplate in kW."""
    if not isinstance(plant, dict) or not plant.get('arrays'):
        raise ValueError('data.plant must define at least one PV array.')
    grid = np.asarray(plant.get('efficiency_input_w', []), dtype=float)
    efficiency = np.asarray(plant.get('efficiency_fraction', []), dtype=float)
    if (
        grid.ndim != 1
        or len(grid) < 2
        or grid.shape != efficiency.shape
        or not np.isfinite(grid).all()
        or not np.isfinite(efficiency).all()
        or grid[0] != 0
        or np.any(np.diff(grid) <= 0)
        or np.any((efficiency < 0) | (efficiency > 1))
    ):
        raise ValueError(
            'Plant efficiency curve requires increasing W inputs from zero and fractions in [0, 1].'
        )
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
        for key in (
            'module_power_w',
            'modules_series',
            'strings_parallel',
            'temperature_coefficient_per_c',
            'module_a',
            'module_b',
            'cell_delta_c',
        ):
            if not np.isfinite(array.get(key, np.nan)):
                raise ValueError(f'Array {key} must be finite.')
        if array['module_power_w'] <= 0 or array['cell_delta_c'] < 0:
            raise ValueError('Array power must be positive and cell delta nonnegative.')
        for key in ('modules_series', 'strings_parallel'):
            if type(array[key]) is not int or array[key] < 1:
                raise ValueError(f'Array {key} must be a positive integer.')
        capacity_w += (
            array['module_power_w']
            * array['modules_series']
            * array['strings_parallel']
        )
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
            module = ambient + poa * np.exp(
                array['module_a'] + array['module_b'] * wind
            )
            cell = (
                module
                + poa / plant['reference_irradiance_w_m2'] * array['cell_delta_c']
            )
            rated_w = (
                array['module_power_w']
                * array['modules_series']
                * array['strings_parallel']
            )
            power = np.maximum(
                0,
                rated_w
                * poa
                / plant['reference_irradiance_w_m2']
                * (
                    1
                    + array['temperature_coefficient_per_c']
                    * (cell - plant['reference_temperature_c'])
                ),
            )
            efficiency = np.interp(
                power, plant['efficiency_input_w'], plant['efficiency_fraction']
            )
            dc.append(power)
            ac.append(power * efficiency)
            modules.append(module)
            cells.append(cell)
    values = np.column_stack(
        (
            np.sum(ac, axis=0) / 1000,
            np.sum(dc, axis=0) / 1000,
            np.mean(modules, axis=0),
            np.mean(cells, axis=0),
        )
    )
    if plant.get('ac_capacity_kw') is not None:
        values[:, 0] = np.minimum(values[:, 0], plant['ac_capacity_kw'])
    valid &= np.isfinite(values).all(axis=1)
    values[~valid] = np.nan
    return pd.DataFrame(values, index=frame.index, columns=PHYSICS_FEATURES)


def safe_ratio(numerator, denominator, minimum):
    """Zero below the fixed denominator floor; preserve missing observations."""
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError('Ratio denominator floors must be finite and positive.')
    numerator, denominator = (
        np.asarray(numerator, float),
        np.asarray(denominator, float),
    )
    values = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=values, where=denominator > minimum)
    values[~np.isfinite(numerator) | ~np.isfinite(denominator)] = np.nan
    return values


def solar_features(frame, config, geometry_only=False):
    """Verified columns or NOAA geometry with a fixed Haurwitz clear-sky model."""
    settings = config.get('solar', {})
    if settings.get('elevation_column') and settings.get('clear_ghi_column'):
        elevation = frame[settings['elevation_column']].to_numpy(float)
        clear = frame[settings['clear_ghi_column']].to_numpy(float)
    else:
        site = settings.get('site') or {}
        for key, low, high in (('latitude', -90, 90), ('longitude', -180, 180)):
            value = site.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(
                    f'data.solar.site.{key} requires verified station metadata.'
                )
        if not site.get('timezone'):
            raise ValueError(
                'data.solar.site.timezone must identify the timestamp timezone.'
            )
        times = frame.index
        times = (
            times.tz_localize(site['timezone'])
            if times.tz is None
            else times.tz_convert(site['timezone'])
        )
        times = times.tz_convert('UTC')
        hours = np.asarray(times.hour + times.minute / 60 + times.second / 3600, float)
        # NOAA fractional-year approximation, evaluated in UTC (timezone offset zero).
        days = np.where(times.is_leap_year, 366, 365)
        gamma = 2 * np.pi / days * (np.asarray(times.dayofyear) - 1 + (hours - 12) / 24)
        equation = 229.18 * (
            0.000075
            + 0.001868 * np.cos(gamma)
            - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma)
            - 0.040849 * np.sin(2 * gamma)
        )
        declination = (
            0.006918
            - 0.399912 * np.cos(gamma)
            + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma)
            + 0.00148 * np.sin(3 * gamma)
        )
        hour_angle = np.deg2rad(
            (hours * 60 + equation + 4 * site['longitude']) / 4 - 180
        )
        latitude = np.deg2rad(site['latitude'])
        cosine = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(
            declination
        ) * np.cos(hour_angle)
        elevation = 90 - np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
        # Haurwitz: GHI_clear = 1098*cos(zenith)*exp(-0.059/cos(zenith)), W/m2.
        clear = np.zeros(len(frame))
        daylight = cosine > 0
        clear[daylight] = 1.098 * cosine[daylight] * np.exp(-0.059 / cosine[daylight])
    if np.any(np.isfinite(clear) & (clear < 0)):
        raise ValueError('Clear-sky GHI must be nonnegative and use kW/m2.')
    if geometry_only:
        return pd.DataFrame(
            {'solar_elevation': elevation, 'solar_zenith': 90 - elevation},
            index=frame.index,
        )
    return pd.DataFrame(
        {
            'clear_sky_index': safe_ratio(
                frame['GHI[kW1m2]'],
                clear,
                config.get('ratio_floors', {}).get('clear_ghi_kw_m2', 0.02),
            ),
            'solar_elevation': elevation,
            'solar_zenith': 90 - elevation,
        },
        index=frame.index,
    )


def build_features(frame, config):
    """Build all spaces with the same predeclared target processing."""
    names = feature_names(config)
    result = frame.drop(columns=list(RESERVED_PHYSICS), errors='ignore').copy()
    physical = pv_power_features(result, config['plant'])
    result['observed_power_raw'] = result[config['target']]
    if config.get('correct_power'):
        rules = config['power_correction']
        measured = result[config['target']].to_numpy(float)
        pac = physical['Pac'].to_numpy(float)
        active = pac > rules['minimum_pac_kw']
        replace = active & (np.abs(measured - pac) > rules['relative_deviation'] * pac)
        replace |= ~np.isfinite(measured) & np.isfinite(pac)
        corrected = np.where(replace, pac, measured)
        result[config['target']] = np.where(
            corrected <= rules['zero_floor_kw'], 0, corrected
        )
        physical['Pac'] = physical['Pac'].where(
            physical['Pac'] > rules['zero_floor_kw'], 0
        )
        result.attrs['power_correction'] = {
            'replaced_rows': int(replace.sum()),
            'finite_changed_rows': int(
                np.sum(np.isfinite(measured) & (measured != result[config['target']]))
            ),
            'mean_absolute_change_kw': float(
                np.nanmean(abs(result[config['target']] - measured))
            ),
        }
    if config.get('evaluation', {}).get('daylight_only'):
        result['evaluation_daylight'] = (
            solar_features(frame, config, geometry_only=True)['solar_elevation'] > 0
        )
    # Physical-input validity is common even when no physical feature is selected.
    result.attrs['eligible_rows'] = np.isfinite(physical.to_numpy()).all(axis=1)
    eligibility = config.get('eligibility_features', [])
    if 'HoursOfDay' in names:
        result['HoursOfDay'] = result.index.hour
    if 'MinutesOfDay' in names:
        result['MinutesOfDay'] = result.index.hour * 12 + result.index.minute // 5
    for name, column, operation in (
        ('MeanPrevH', config['target'], 'mean'),
        ('StdPrevH', config['target'], 'std'),
        ('MeanWindSpeedPrevH', 'WIND_SPEED[m1s]', 'mean'),
        ('StdWindSpeedPrevH', 'WIND_SPEED[m1s]', 'std'),
    ):
        if name in names:
            rolling = result[column].rolling(config['horizon'])
            result[name] = (
                rolling.mean() if operation == 'mean' else rolling.std(ddof=0)
            )
    derived = physical.copy()
    if set(names).intersection(STATE_FEATURES):
        derived['physics_residual'] = result[config['target']] - physical['Pac']
        derived['performance_ratio'] = safe_ratio(
            result[config['target']],
            physical['Pac'],
            config.get('ratio_floors', {}).get('pac_kw', 0.05),
        )
    if set(names + list(eligibility)).intersection(SOLAR_FEATURES):
        sun = solar_features(frame, config)
        derived = pd.concat([derived, sun], axis=1)
        result.attrs['eligible_rows'] &= np.isfinite(sun.to_numpy()).all(axis=1)
    selected_physics = set(names).intersection(DERIVED_FEATURES)
    for name in DERIVED_FEATURES:
        if name in selected_physics:
            result[name] = derived[name]
    if set(result.columns).intersection(RESERVED_PHYSICS) != selected_physics:
        raise ValueError('Physics feature invariant violated.')
    if eligibility:
        raw = [
            name
            for name in eligibility
            if name not in DERIVED_FEATURES
            and name not in ('HoursOfDay', 'MinutesOfDay')
        ]
        result.attrs['eligible_rows'] &= np.isfinite(result[raw].to_numpy(float)).all(
            axis=1
        )
    return result


def _correction_audit(frame, data):
    """Summarize target corrections and daylight counts for each time split."""
    train_end = int(len(frame) * data['split'][0])
    val_end = int(len(frame) * sum(data['split'][:2]))
    boundaries = (
        ('TRAIN', 0, train_end),
        ('VAL', train_end, val_end),
        ('TEST', val_end, len(frame)),
    )
    audit = []
    for label, start, end in boundaries:
        part = frame.iloc[start:end]
        raw = part['observed_power_raw']
        corrected = part[data['target']]
        daylight_rows = end - start
        if 'evaluation_daylight' in part:
            daylight_rows = int(part['evaluation_daylight'].sum())
        audit.append(
            {
                'partition': label,
                'rows': end - start,
                'changed_rows': int((raw != corrected).sum()),
                'mean_absolute_change_kw': float(abs(raw - corrected).mean()),
                'daylight_rows': daylight_rows,
            }
        )
    return audit


def prepare_spaces(config, directory):
    """Compute selectors before HPO, using only chronological TRAIN rows."""
    data = copy.deepcopy(config['data'])
    data['features'] = [
        data['target'],
        *[name for name in SUPPORTED_FEATURES if name in EXPANDED_FEATURES],
    ]
    frame = build_features(load_data(data), data)
    audit = _correction_audit(frame, data)
    end = int(np.floor(len(frame) * data['split'][0] + 1e-9))
    columns = [data['target'], *EXPANDED_FEATURES]
    train = frame.iloc[:end].loc[frame.attrs['eligible_rows'][:end], columns].dropna()
    if len(train) < 2:
        raise ValueError(
            'Insufficient finite TRAIN observations for feature selection.'
        )
    correlation = train.corr()
    target_correlation = correlation[data['target']]
    pearson = [
        name
        for name in EXPANDED_FEATURES
        if abs(target_correlation[name]) >= config['experiment']['pearson_threshold']
        or name in config['experiment'].get('pearson_force_include', [])
    ]
    spaces = {
        name: list(features)
        for name, features in zip(
            SPACE_NAMES, (ORIGINAL_FEATURES, pearson, PHYSICS_SELECTED, ())
        )
    }
    reasons = [
        (
            'GHI[kW1m2]',
            'POA Irr[kW1m2]',
            'POA measures irradiance on the actual array plane',
        ),
        ('Pdc', 'Pac', 'AC estimate incorporates inverter conversion efficiency'),
        ('TempModule', 'TempCell', 'Cell temperature enters the PV power equation'),
        (
            'TEMPERATURE[degC]',
            'TempCell',
            'Ambient temperature is incorporated in cell temperature',
        ),
        (
            'WIND_SPEED[m1s]',
            'TempCell',
            'Wind cooling is incorporated in cell temperature',
        ),
        ('WIND_DIR[deg]', None, 'Not an input to the configured PV performance model'),
        (
            'MinutesOfDay',
            'HoursOfDay',
            'Coarser time indicator reduces redundant time-of-day values',
        ),
    ]
    frame.attrs['preprocessing'] = {
        'audit': audit,
        'feature_selection': [
            {
                'feature': name,
                'train_target_pearson': float(target_correlation[name]),
                'pearson_selected': name in pearson,
                'physics_selected': name in PHYSICS_SELECTED,
                'n_train_rows': len(train),
            }
            for name in EXPANDED_FEATURES
        ],
        'physical_redundancy': [
            {
                'removed': left,
                'retained': right,
                'reason': reason,
                'train_pearson': float(correlation.loc[left, right]) if right else None,
            }
            for left, right, reason in reasons
        ],
    }
    return spaces, frame


def input_identity(config):
    """Record the resolved input path, byte size and modification time for resume checks."""
    path = project_path(config['path']).resolve()
    info = path.stat()
    return {
        'path': str(path),
        'size_bytes': info.st_size,
        'modified_ns': info.st_mtime_ns,
    }


def partition_metadata(splits):
    """Summarize origin counts and time bounds for each nonempty split."""
    return {
        key: {
            'n_origins': len(part['origins']),
            'first_origin': str(part['origins'][0]),
            'last_origin': str(part['origins'][-1]),
            'target_start': str(part['target_times'][0, 0]),
            'target_end': str(part['target_times'][-1, -1]),
        }
        for key, part in splits.items()
    }


def load_data(config):
    """Read HDF5 observations, sort timestamps and retain gaps as missing rows.

    Reject duplicate or off-grid timestamps. Apply max_rows only after restoring
    the continuous timeline; no interpolation or target correction occurs here.
    """
    frame = pd.read_hdf(project_path(config['path']))
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError('Data must have a DatetimeIndex.')
    if frame.index.has_duplicates:
        raise ValueError(
            'Timestamps must be unique; resolve duplicate observations first.'
        )
    # The full SOLETE file stores some day blocks out of chronological order.
    frame = frame.sort_index()
    delta = pd.to_timedelta(config['resolution'])
    if delta <= pd.Timedelta(0):
        raise ValueError('resolution must be positive.')
    if (
        len(frame) > 1
        and ((frame.index - frame.index[0]) % delta != pd.Timedelta(0)).any()
    ):
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
    """Build an unfitted scaler from its YAML name."""
    if name == 'MinMax01':
        return MinMaxScaler((0, 1))
    if name == 'MinMax11':
        return MinMaxScaler((-1, 1))
    if name == 'Standard':
        return StandardScaler()
    raise ValueError('scaler must be MinMax01, MinMax11 or Standard.')


def prepare_data(frame, config, scalers=None, partitions=('TRAIN', 'VAL', 'TEST')):
    """Split by target timestamps; validation/test may use earlier observed context.

    Each input contains t-PRE through t; labels contain t+1 through t+H.
    All labels of a sample must belong to one chronological partition.

    Return (splits, scalers). Each split contains X with shape
    (origins, PRE + 1, features) and Y/Y_scaled with shape (origins, H),
    plus raw targets, persistence, timestamps and a daylight scoring mask.
    Without supplied scalers, TRAIN must be requested to fit them. Supplied
    scalers are reused unchanged. Night observations remain in the histories.
    """
    pre, horizon = config['pre'], config['horizon']
    if type(pre) is not int or pre < 0 or type(horizon) is not int or horizon < 1:
        raise ValueError(
            'pre must be a nonnegative integer; horizon a positive integer.'
        )
    origin_pre = config.get('origin_pre', pre)
    if type(origin_pre) is not int or origin_pre < pre:
        raise ValueError('origin_pre must be an integer at least as large as pre.')
    features, target = feature_names(config), config['target']
    if not isinstance(target, str):
        raise ValueError(
            'This pipeline supports one target and multiple forecast steps.'
        )
    missing = set(features + [target]) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    ratios = np.asarray(config['split'], dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError('split must contain three positive fractions summing to one.')
    n = len(frame)
    train_end = int(np.floor(n * ratios[0] + 1e-9))
    val_end = int(np.floor(n * (ratios[0] + ratios[1]) + 1e-9))
    values = frame[features].to_numpy(dtype=float)
    labels = frame[target].to_numpy(dtype=float)
    eligibility = np.asarray(
        frame.attrs.get('eligible_rows', np.ones(n, dtype=bool)), dtype=bool
    )
    if eligibility.shape != (n,):
        raise ValueError('Feature eligibility mask does not match the time axis.')
    splits = {}
    boundaries = [
        ('TRAIN', 0, train_end),
        ('VAL', train_end, val_end),
        ('TEST', val_end, n),
    ]
    for name, start, end in boundaries:
        if name not in partitions:
            continue
        origins = np.arange(max(origin_pre, start - 1), end - horizon)
        origins = np.asarray(
            [
                t
                for t in origins
                if np.isfinite(values[t - pre : t + 1]).all()
                and eligibility[t - origin_pre : t + 1].all()
                and np.isfinite(labels[t - origin_pre : t + horizon + 1]).all()
            ],
            dtype=int,
        )
        if len(origins) == 0:
            raise ValueError(
                f'{name} has no valid windows: rows={n}, pre={pre}, '
                f'horizon={horizon}. Use more data or shorter windows.'
            )
        splits[name] = _window_arrays(frame, values, labels, origins, pre, horizon)
    if scalers is None:
        scalers = {
            'X': make_scaler(config['scaler']),
            'Y': make_scaler(config['scaler']),
        }
        scalers['X'].fit(splits['TRAIN']['X'].reshape(-1, len(features)))
        scalers['Y'].fit(splits['TRAIN']['Y'].reshape(-1, 1))
    for part in splits.values():
        part['X'] = (
            scalers['X']
            .transform(part['X'].reshape(-1, len(features)))
            .reshape(part['X'].shape)
            .astype('float32')
        )
        part['Y_scaled'] = (
            scalers['Y']
            .transform(part['Y'].reshape(-1, 1))
            .reshape(part['Y'].shape)
            .astype('float32')
        )
    return splits, scalers


def _window_arrays(frame, values, labels, origins, pre, horizon):
    """Build one partition's aligned histories, targets and scoring masks."""
    target_windows = [slice(t + 1, t + horizon + 1) for t in origins]
    targets = np.stack([labels[window] for window in target_windows])
    evaluation_mask = np.ones((len(origins), horizon), bool)
    if 'evaluation_daylight' in frame:
        daylight = frame['evaluation_daylight'].to_numpy(bool)
        evaluation_mask = np.stack([daylight[window] for window in target_windows])
    raw_targets = targets.copy()
    if 'observed_power_raw' in frame:
        raw = frame['observed_power_raw'].to_numpy(float)
        raw_targets = np.stack([raw[window] for window in target_windows])
    return {
        'X': np.stack([values[t - pre : t + 1] for t in origins]),
        'Y': targets,
        'persistence': np.repeat(labels[origins, None], horizon, axis=1),
        'origins': frame.index[origins],
        'target_times': np.stack(
            [frame.index[window].to_numpy() for window in target_windows]
        ),
        'evaluation_mask': evaluation_mask,
        'Y_raw': raw_targets,
    }
