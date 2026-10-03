"""Validated experiment configuration and versioned preprocessing contracts."""
import copy

import numpy as np
import yaml

from .runtime import project_path
from .data import PHYSICS_FEATURES, feature_names, validate_plant

CONTRACT_VERSION = 2


def validate_config(config):
    for key in ('data', 'model', 'models', 'seed', 'output_dir'):
        if key not in config:
            raise ValueError(f'Missing config section: {key}')
    data = config['data']
    feature_names(data)
    if data.get('correct_power') is not False:
        raise ValueError('data.correct_power must be false; observed targets cannot be replaced.')
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


def load_config(path):
    config = yaml.safe_load(project_path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Config must be a YAML mapping.')
    data = config.get('data', {})
    if 'plant' not in data:
        if 'plant_path' not in data:
            raise ValueError('data.plant_path or embedded data.plant is required.')
        data['plant'] = yaml.safe_load(project_path(data['plant_path']).read_text(encoding='utf-8'))
    return validate_config(config)


def arm_config(config, subset, model, seed):
    """Return a validated configuration for one model and feature subset."""
    result = copy.deepcopy(config)
    result['data']['candidate_subset'] = list(subset)
    result['data']['physics'] = bool(set(subset).intersection(PHYSICS_FEATURES))
    result.update(model=model, seed=seed)
    return validate_config(result)


def data_contract(data):
    keys = ('features', 'target', 'pre', 'horizon', 'resolution', 'scaler',
            'correct_power', 'split', 'physics', 'p_nom_kw', 'plant', 'constraints')
    return copy.deepcopy({**{key: data[key] for key in keys},
                          **({'candidate_subset': data['candidate_subset']} if 'candidate_subset' in data else {}),
                          'effective_features': feature_names(data),
                          'max_rows': data.get('max_rows'), 'version': CONTRACT_VERSION})


def arm_name(model, physics):
    return ('PI-' if physics else '') + model


def output_paths(config):
    paths = {key: project_path(config['output_dir']) / key for key in ('models', 'figures', 'results')}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
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
        result['tuning'].update(trials_per_model=1, startup_trials=1)
    return validate_config(result)
