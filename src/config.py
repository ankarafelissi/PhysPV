"""Validated experiment configuration and versioned preprocessing contracts."""
import copy

import numpy as np
import yaml

from .paths import project_path
from .features import feature_names, validate_plant

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


def data_contract(data):
    keys = ('features', 'target', 'pre', 'horizon', 'resolution', 'scaler',
            'correct_power', 'split', 'physics', 'p_nom_kw', 'plant', 'constraints')
    return copy.deepcopy({**{key: data[key] for key in keys},
                          'effective_features': feature_names(data),
                          'max_rows': data.get('max_rows'), 'version': CONTRACT_VERSION})


def comparison_contract(config):
    """Ignore only the ablation switch, its injected columns, and model choice."""
    contract = data_contract(config['data'])
    contract.pop('physics')
    contract.pop('effective_features')
    return {**contract, 'data_path': str(project_path(config['data']['path']).resolve()),
            'models': copy.deepcopy(config['models']), 'study': copy.deepcopy(config.get('study', {}))}


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
    seeds = settings.get('seeds', [])
    if (not isinstance(seeds, list) or len(seeds) < 5 or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError('study.seeds requires at least five distinct nonnegative integers.')
    for key in ('bootstrap_resamples', 'bootstrap_seed', 'bootstrap_block_length',
                'min_test_origins', 'min_stratum_origins', 'representative_origins', 'min_bootstrap_blocks'):
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
    return validate_config(result)
