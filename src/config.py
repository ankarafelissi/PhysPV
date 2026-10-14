"""Validated experiment configuration and versioned preprocessing contracts."""
import copy

import numpy as np
import yaml

from .runtime import project_path
from .data import (DERIVED_FEATURES, PHYSICS_FEATURES, RESERVED_PHYSICS, SOLAR_FEATURES,
                   CANDIDATE_FEATURES, SUPPORTED_FEATURES, feature_names, validate_plant)

CONTRACT_VERSION = 6
LEGACY_FEATURE_KEYS = {'physics', 'candidate_subset', 'selected_features'}


def upgrade_legacy_features(data):
    """Convert old YAML feature controls once, preserving ordering and eligibility."""
    if not LEGACY_FEATURE_KEYS.intersection(data):
        return
    base = feature_names(data)
    physics = data.get('physics')
    if type(physics) is not bool:
        raise ValueError('Legacy data.physics must be an explicit boolean.')
    if set(base).intersection(RESERVED_PHYSICS):
        raise ValueError('Legacy data.features cannot contain physics columns.')
    if 'selected_features' in data:
        selected = feature_names({'features': data['selected_features']})
        supported = (data['target'], *SUPPORTED_FEATURES)
        if not set(selected).issubset(supported) or data['target'] not in selected:
            raise ValueError('Legacy selected_features requires supported features and historical target.')
        names = list(dict.fromkeys(name for name in supported if name in selected))
        if physics != bool(set(names).intersection(DERIVED_FEATURES)):
            raise ValueError('Legacy data.physics does not match selected_features.')
    elif 'candidate_subset' in data:
        subset = data['candidate_subset']
        if (not isinstance(subset, list) or any(not isinstance(name, str) for name in subset)
                or len(subset) != len(set(subset)) or not set(subset).issubset(CANDIDATE_FEATURES)
                or set(base).intersection(CANDIDATE_FEATURES)):
            raise ValueError('Legacy candidate_subset requires unique supported candidates separate from features.')
        if physics != bool(set(subset).intersection(DERIVED_FEATURES)):
            raise ValueError('Legacy data.physics does not match candidate_subset.')
        names = base + [name for name in CANDIDATE_FEATURES if name in subset]
    else:
        names = base + list(PHYSICS_FEATURES) if physics else base
    if 'candidate_subset' in data:
        # The old subset interface screened the whole GHI/solar pool in every arm.
        common = ['GHI[kW1m2]']
        solar = data.get('solar', {})
        if ((solar.get('site') or {}).get('latitude') is not None
                or bool(solar.get('elevation_column') and solar.get('clear_ghi_column'))):
            common.extend(SOLAR_FEATURES)
        data['eligibility_features'] = list(dict.fromkeys([*data.get('eligibility_features', []), *common]))
    data['features'] = names
    for key in LEGACY_FEATURE_KEYS:
        data.pop(key, None)


def validate_config(config):
    for key in ('data', 'model', 'models', 'seed', 'output_dir'):
        if key not in config:
            raise ValueError(f'Missing config section: {key}')
    data = config['data']
    if LEGACY_FEATURE_KEYS.intersection(data):
        raise ValueError('Use explicit data.features; load legacy YAML through load_config.')
    feature_names(data)
    floors = data.get('ratio_floors', {})
    for key in ('pac_kw', 'clear_ghi_kw_m2'):
        value = floors.get(key, .05 if key == 'pac_kw' else .02)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f'data.ratio_floors.{key} must be finite and positive.')
    solar = data.get('solar', {})
    if bool(solar.get('elevation_column')) != bool(solar.get('clear_ghi_column')):
        raise ValueError('Solar input columns must specify both elevation and clear-sky GHI.')
    if type(data.get('correct_power')) is not bool:
        raise ValueError('data.correct_power must be explicit boolean.')
    if data['correct_power']:
        rules = data.get('power_correction', {})
        for key in ('relative_deviation', 'minimum_pac_kw', 'zero_floor_kw'):
            value = rules.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'power_correction.{key} must be finite and positive.')
    evaluation = data.get('evaluation', {})
    if evaluation:
        if type(evaluation.get('daylight_only')) is not bool or evaluation.get('normalization_kw', 0) <= 0:
            raise ValueError('evaluation requires daylight_only and positive normalization_kw.')
    if data.get('target') != 'P_Solar[kW]':
        raise ValueError('This research phase requires target P_Solar[kW].')
    p_nom = data.get('p_nom_kw')
    if isinstance(p_nom, bool) or not isinstance(p_nom, (int, float)) or not np.isfinite(p_nom) or p_nom <= 0:
        raise ValueError('data.p_nom_kw must be an explicit finite positive DC nameplate capacity.')
    if not np.isclose(p_nom, validate_plant(data['plant'])):
        raise ValueError('data.p_nom_kw does not match the configured PV array nameplate capacity.')
    constraints = data.get('constraints', {})
    if set(constraints) != {'nonnegative', 'capacity'} or any(type(v) is not bool for v in constraints.values()):
        raise ValueError('data.constraints requires boolean nonnegative and capacity fields.')
    if constraints['capacity'] and data['plant'].get('ac_capacity_kw') is None:
        raise ValueError('Capacity constraints require a confirmed plant.ac_capacity_kw; DC rating is not a substitute.')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer.')
    xgb_seed = config['models'].get('XGBoost', {}).get('random_state', config['seed'])
    if xgb_seed != config['seed']:
        raise ValueError('XGBoost random_state must match the experiment seed; omit the override.')
    if config['model'] not in ('CNN_LSTM', 'XGBoost'):
        raise ValueError('model must be CNN_LSTM or XGBoost.')
    if config['models'].get('CNN_LSTM', {}).get('output_activation') not in (None, 'linear'):
        raise ValueError('CNN-LSTM requires a linear output; configure physical bounds in data.constraints.')
    optimizer = config['models'].get('CNN_LSTM', {}).get('optimizer')
    if not isinstance(optimizer, (str, dict)):
        raise ValueError('CNN-LSTM optimizer must be a name or parameter mapping.')
    if isinstance(optimizer, dict):
        if set(optimizer) - {'name', 'learning_rate', 'clipnorm'} or not isinstance(optimizer.get('name'), str):
            raise ValueError('CNN-LSTM optimizer mapping supports name, learning_rate and clipnorm.')
        for key in ('learning_rate', 'clipnorm'):
            value = optimizer.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
                raise ValueError(f'CNN-LSTM optimizer.{key} must be positive.')
    return config


def validate_search(config):
    settings = config['tuning']
    for key in ('trials_per_space', 'startup_trials'):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f'tuning.{key} must be a positive integer.')
    if settings['startup_trials'] > settings['trials_per_space']:
        raise ValueError('Startup trials cannot exceed the per-space budget.')
    choices = settings.get('pre_choices')
    if (not isinstance(choices, list) or not choices or len(set(choices)) != len(choices)
            or any(type(value) is not int or value < 0 for value in choices)):
        raise ValueError('pre_choices requires distinct nonnegative integers.')
    if config['data']['pre'] not in choices:
        raise ValueError('Configured starting PRE must belong to pre_choices.')
    threshold = config['experiment'].get('pearson_threshold')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError('pearson_threshold must be in [0, 1].')
    if config['data']['features'] != [config['data']['target']]:
        raise ValueError('The joint search keeps only historical target mandatory.')


def load_config(path):
    config = yaml.safe_load(project_path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Config must be a YAML mapping.')
    data = config.get('data', {})
    upgrade_legacy_features(data)
    if 'plant' not in data:
        if 'plant_path' not in data:
            raise ValueError('data.plant_path or embedded data.plant is required.')
        data['plant'] = yaml.safe_load(project_path(data['plant_path']).read_text(encoding='utf-8'))
    return validate_config(config)


def arm_config(config, subset, model, seed):
    """Return a validated configuration for one model and feature subset."""
    result = copy.deepcopy(config)
    if len(subset) != len(set(subset)) or not set(subset).issubset(SUPPORTED_FEATURES):
        raise ValueError('subset must contain unique supported features.')
    result['data']['features'] = [result['data']['target'], *[name for name in SUPPORTED_FEATURES if name in subset]]
    result.update(model=model, seed=seed)
    return validate_config(result)


def data_contract(data):
    keys = ('features', 'target', 'pre', 'horizon', 'resolution', 'scaler',
            'correct_power', 'split', 'p_nom_kw', 'plant', 'constraints')
    return copy.deepcopy({**{key: data[key] for key in keys},
                          'solar': data.get('solar'), 'ratio_floors': data.get('ratio_floors'),
                          'origin_pre': data.get('origin_pre'),
                          'eligibility_features': data.get('eligibility_features'),
                          'evaluation': data.get('evaluation'),
                          'power_correction': data.get('power_correction'),
                          'max_rows': data.get('max_rows'), 'version': CONTRACT_VERSION})


def output_paths(config, run_id=None, study_dir=None):
    paths = {key: project_path(config['output_dir']) / key for key in ('models', 'figures', 'results')}
    if run_id:
        parent = project_path(study_dir) / 'runs' if study_dir else paths['results']
        paths['results'] = parent / run_id
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=key != 'results' or run_id is None)
    return paths


def validate_study(settings):
    if not isinstance(settings, dict):
        raise ValueError('study must be a configuration mapping.')
    for key in ('bootstrap_resamples', 'bootstrap_seed', 'bootstrap_block_length', 'min_test_origins'):
        if type(settings.get(key)) is not int or settings[key] < (0 if key == 'bootstrap_seed' else 1):
            raise ValueError(f'study.{key} must be a valid integer.')
    if settings['bootstrap_resamples'] < 1000:
        raise ValueError('study.bootstrap_resamples must be at least 1000.')


def smoke_config(config):
    """Copy an experiment and apply the shared execution-only YAML overrides."""
    result = copy.deepcopy(config)
    overrides = yaml.safe_load(project_path("config/smoke.yaml").read_text(encoding="utf-8"))
    result["data"].update(overrides["data"])
    for model, params in overrides["models"].items():
        result["models"][model].update(params)
    result["smoke_study"] = True
    if 'tuning' in result:
        result['tuning'].update(trials_per_space=1, startup_trials=1, pre_choices=[0, 2])
    result['data'].pop('origin_pre', None)
    return validate_config(result)
