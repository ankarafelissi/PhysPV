"""Causal feature construction and vectorized PV estimates; attribution in docs/data.md."""
import numpy as np
import pandas as pd

PHYSICS_FEATURES = ('Pac', 'Pdc', 'TempModule', 'TempCell')
RESERVED_PHYSICS = (*PHYSICS_FEATURES, 'TempModule_RP')
PHYSICAL_INPUTS = ('POA Irr[kW1m2]', 'TEMPERATURE[degC]', 'WIND_SPEED[m1s]')


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
    if config['physics']:
        for name in PHYSICS_FEATURES:
            result[name] = physical[name]
    if set(result.columns).intersection(RESERVED_PHYSICS) != (set(PHYSICS_FEATURES) if config['physics'] else set()):
        raise ValueError('Physics feature invariant violated.')
    return result
