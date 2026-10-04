"""Equal-budget TPE joint search of feature subset, PRE and hyperparameters."""
import copy
import math
from pathlib import Path
import pandas as pd
import yaml
from .runtime import write_json
from .data import SPACE_NAMES
from .config import validate_search


def sample_parameters(trial, model, defaults):
    params = copy.deepcopy(defaults)
    if model == 'XGBoost':
        params.update(
            max_depth=trial.suggest_int('max_depth', 2, 6),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 8),
            reg_lambda=trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
            subsample=trial.suggest_float('subsample', 0.7, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.7, 1.0))
    else:
        params['filters'] = trial.suggest_categorical('filters', [16, 32, 48])
        params['kernel_size'] = trial.suggest_int('kernel_size', 1, 7)
        params['pool_size'] = trial.suggest_int('pool_size', 1, 7)
        layers = trial.suggest_int('lstm_layers', 1, 3)
        params['neurons'] = [trial.suggest_int('lstm_units', 5, 20)] * layers
        dense_layers = trial.suggest_int('dense_layers', 0, 2)
        params['dense_neurons'] = [trial.suggest_int('dense_units', 1, 5)] * dense_layers
        params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32])
        params['optimizer']['learning_rate'] = trial.suggest_float('learning_rate', .0002, .002, log=True)
    return params


def starting_parameters(defaults, model, pool, pre):
    keys = ('max_depth', 'learning_rate', 'min_child_weight', 'reg_lambda', 'subsample', 'colsample_bytree')
    params = ({key: defaults[key] for key in keys} if model == 'XGBoost' else
              {'filters': defaults['filters'], 'lstm_units': defaults['neurons'][0],
               'lstm_layers': len(defaults['neurons']), 'kernel_size': defaults['kernel_size'],
               'pool_size': defaults.get('pool_size', 1), 'dense_layers': len(defaults.get('dense_neurons', [])),
               'dense_units': defaults.get('dense_neurons', [1])[0] if defaults.get('dense_neurons') else 1,
               'batch_size': defaults['batch_size'], 'learning_rate': defaults['optimizer']['learning_rate']})
    return {**params, 'pre': pre, **{'include_' + feature: True for feature in pool}}


def export_search(state, manifest):
    rows = []
    for record in state['tuning'].values():
        for trial in record['trials']:
            rows.append({'model': record['model'], 'space': record['space'], 'trial': trial['number'],
                         'status': trial['status'], 'validation_rmse': trial.get('validation_rmse'),
                         'validation_mae': trial.get('validation_mae'),
                         'training_seconds': trial.get('training_seconds'),
                         'training_run_id': trial.get('training_run_id'),
                         'selected': trial['number'] == record.get('best_trial'),
                         'subset': ';'.join(trial['subset']), **trial['params']})
    pd.DataFrame(rows).to_csv(Path(manifest).with_name('tuning_results.csv'), index=False)


def tune(config, state, manifest, fit):
    import optuna
    from optuna.distributions import distribution_to_json, json_to_distribution
    validate_search(config)
    settings = config['tuning']
    records = state.setdefault('tuning', {})
    for model in ('XGBoost', 'CNN_LSTM'):
        for space in SPACE_NAMES:
            key = model + '/' + space
            pool = state['feature_spaces'][space]
            record = records.setdefault(key, {'model': model, 'space': space, 'trials': [], 'best_trial': None})
            if record['best_trial'] is not None:
                continue
            defaults = config['models'][model]
            study = optuna.create_study(direction='minimize', study_name=key)
            for previous in record['trials']:
                if previous['status'] == 'completed':
                    study.add_trial(optuna.trial.create_trial(params=previous['params'],
                        value=previous['validation_rmse'], distributions={k: json_to_distribution(v)
                        for k, v in previous['distributions'].items()}))
            if not record['trials']:
                study.enqueue_trial(starting_parameters(defaults, model, pool, config['data']['pre']))
            while len(record['trials']) < settings['trials_per_space']:
                number = len(record['trials'])
                study.sampler = optuna.samplers.TPESampler(seed=config['seed'] + number,
                    n_startup_trials=settings['startup_trials'])
                trial = study.ask()
                params = sample_parameters(trial, model, defaults)
                pre = trial.suggest_categorical('pre', settings['pre_choices'])
                subset = [name for name in pool if trial.suggest_categorical('include_' + name, [False, True])]
                candidate = copy.deepcopy(config)
                candidate['models'][model] = params
                candidate['data']['pre'] = pre
                arm = {'id': space, 'subset': subset, 'category': 'feature_space'}
                item = {'number': number, 'status': 'started', 'params': dict(trial.params),
                        'parameters': params, 'subset': subset, 'pre': pre,
                        'distributions': {k: distribution_to_json(v) for k, v in trial.distributions.items()}}
                record['trials'].append(item)
                write_json(manifest, state)
                print(f'TPE {model} {space} trial={number + 1}/{settings["trials_per_space"]} PRE={pre} subset={subset}', flush=True)
                try:
                    result = fit(candidate, arm, model, 'tuning')
                    value = result['validation_rmse']
                    if not math.isfinite(value):
                        raise ValueError('Non-finite validation RMSE.')
                    item.update(status='completed', validation_rmse=value,
                                validation_mae=result['validation_mae'], result=result,
                                training_run_id=result['training_run_id'],
                                training_seconds=result.get('training_seconds'))
                    study.tell(trial, value)
                except (Exception, KeyboardInterrupt) as error:
                    item.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
                    if state.get('attempts'):
                        item['training_run_id'] = state['attempts'][-1]['run_id']
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    write_json(manifest, state)
                    export_search(state, manifest)
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    continue
                write_json(manifest, state)
                export_search(state, manifest)
            completed = [item for item in record['trials'] if item['status'] == 'completed']
            if not completed:
                raise ValueError(f'No successful tuning trials for {key}; exhausted budget is retained.')
            best = min(completed, key=lambda item: (item['validation_rmse'], item['number']))
            record.update(best_trial=best['number'], best_parameters=copy.deepcopy(best['parameters']),
                          best_subset=best['subset'], best_pre=best['pre'])
            write_json(manifest, state)
    state['selected'] = {}
    state['screening'] = []
    state['frozen_configs'] = {}
    from .config import arm_config
    for key, record in records.items():
        best = record['trials'][record['best_trial']]
        state['screening'].append(best['result'])
        frozen = arm_config(config, best['subset'], record['model'], config['seed'])
        frozen['data']['pre'] = best['pre']
        frozen['models'][record['model']] = best['parameters']
        state['frozen_configs'][key] = frozen
    for model in ('XGBoost', 'CNN_LSTM'):
        best = min((row for row in state['screening'] if row['model'] == model),
                   key=lambda row: (row['validation_rmse'], SPACE_NAMES.index(row['arm'])))
        state['selected'][model] = best['arm']
    state['parameters_frozen'] = True
    Path(manifest).with_name('frozen_config.yaml').write_text(
        yaml.safe_dump(state['frozen_configs'], sort_keys=False), encoding='utf-8')
    export_search(state, manifest)
    write_json(manifest, state)
