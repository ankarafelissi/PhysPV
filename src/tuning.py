"""Bounded TPE search on the non-PI validation reference only."""
import copy
import math


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
        params['filters'] = trial.suggest_categorical('filters', [16, 32, 64])
        params['neurons'] = [trial.suggest_categorical('lstm_units', [16, 32, 64])]
        params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32])
        params['optimizer']['learning_rate'] = trial.suggest_float(
            'learning_rate', 0.0002, 0.002, log=True)
    return params


def tune(config, state, manifest, fit):
    """Reconstruct completed trials on resume; preserve failures and attempt links."""
    import optuna
    from optuna.distributions import distribution_to_json, json_to_distribution
    from .provenance import write_json
    settings = config['tuning']
    if type(settings['trials_per_model']) is not int or settings['trials_per_model'] < 1:
        raise ValueError('tuning.trials_per_model must be a positive integer.')
    if type(settings['startup_trials']) is not int or settings['startup_trials'] < 1:
        raise ValueError('tuning.startup_trials must be a positive integer.')
    tuning = state.setdefault('tuning', {})
    reference = state['arms'][0]
    for model in ('XGBoost', 'CNN_LSTM'):
        record = tuning.setdefault(model, {'trials': [], 'best_parameters': None})
        if record['best_parameters'] is not None:
            config['models'][model] = copy.deepcopy(record['best_parameters'])
            continue
        defaults = copy.deepcopy(config['models'][model])
        study = optuna.create_study(direction='minimize', study_name=model)
        for previous in record['trials']:
            if previous['status'] != 'completed':
                continue
            study.add_trial(optuna.trial.create_trial(
                params=previous['params'], value=previous['validation_mae'],
                distributions={k: json_to_distribution(v) for k, v in previous['distributions'].items()}))
        # Include the existing reasonable configuration; TPE can never select worse VAL MAE.
        if not record['trials']:
            if model == 'XGBoost':
                baseline = {k: defaults[k] for k in ('max_depth', 'learning_rate',
                    'min_child_weight', 'reg_lambda', 'subsample', 'colsample_bytree')}
            else:
                baseline = {'filters': defaults['filters'], 'lstm_units': defaults['neurons'][0],
                            'batch_size': defaults['batch_size'],
                            'learning_rate': defaults['optimizer']['learning_rate']}
            study.enqueue_trial(baseline)
        while len(record['trials']) < settings['trials_per_model']:
            # Per-attempt deterministic sampler seed also makes resumed suggestions reproducible.
            number = len(record['trials'])
            study.sampler = optuna.samplers.TPESampler(
                seed=config['seed'] + number, n_startup_trials=settings['startup_trials'])
            trial = study.ask()
            params = sample_parameters(trial, model, defaults)
            item = {'number': number, 'status': 'started', 'params': dict(trial.params),
                    'parameters': params, 'distributions': {
                        k: distribution_to_json(v) for k, v in trial.distributions.items()}}
            record['trials'].append(item)
            write_json(manifest, state)
            candidate = copy.deepcopy(config)
            candidate['models'][model] = params
            try:
                result = fit(candidate, reference, model, 'tuning')
                value = result['validation_mae']
                if not math.isfinite(value):
                    raise ValueError('Non-finite validation MAE.')
                item.update(status='completed', validation_mae=value,
                            training_run_id=result['training_run_id'],
                            training_seconds=result.get('training_seconds'))
                study.tell(trial, value)
            except (Exception, KeyboardInterrupt) as error:
                item.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                            error=str(error))
                if state.get('attempts'):
                    item['training_run_id'] = state['attempts'][-1]['run_id']
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                write_json(manifest, state)
                raise
            write_json(manifest, state)
        completed = [t for t in record['trials'] if t['status'] == 'completed']
        if not completed:
            raise ValueError(f'No successful tuning trials for {model}.')
        best = min(completed, key=lambda t: t['validation_mae'])
        record['best_parameters'] = copy.deepcopy(best['parameters'])
        record['best_trial'] = best['number']
        config['models'][model] = copy.deepcopy(best['parameters'])
        write_json(manifest, state)
    state['config'] = copy.deepcopy(config)
    state['parameters_frozen'] = True
    write_json(manifest, state)
    from pathlib import Path
    import yaml
    Path(manifest).with_name('frozen_config.yaml').write_text(
        yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    import pandas as pd
    rows = []
    for model, record in tuning.items():
        for trial in record['trials']:
            rows.append({'model': model, 'trial': trial['number'], 'status': trial['status'],
                         'validation_mae': trial.get('validation_mae'),
                         'training_seconds': trial.get('training_seconds'),
                         'training_run_id': trial.get('training_run_id'),
                         'selected': trial['number'] == record['best_trial'],
                         **trial['params']})
    pd.DataFrame(rows).to_csv(Path(manifest).with_name('tuning_results.csv'), index=False)
