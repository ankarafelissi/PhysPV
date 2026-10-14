"""Stepwise study: prepare data, optimize on VAL, evaluate frozen models, report."""
import copy
import json
import logging
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
import joblib
import pandas as pd
import yaml
from .config import (arm_config, data_contract, load_config, output_paths,
                     validate_config, validate_search, validate_study)
from .data import (EXPANDED_FEATURES, SPACE_NAMES, input_identity, partition_metadata,
                   prepare_data, prepare_spaces)
from .evaluate import (consolidate_results, evaluate, export_results, forecast_health,
                       require_healthy_forecasts, report)
from .models import load_model, predict_scaled, save_model, train_model
from .runtime import configure_runtime, environment_metadata, project_path, write_json

WORKFLOW_VERSION = 6


class StudyChangedError(RuntimeError):
    """An immutable input or source changed; stop the whole study."""


def source_record():
    """Keep the command, Git state and readable source in the study manifest."""
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=project_path('.'), text=True, encoding='utf-8', stderr=subprocess.PIPE)
    files = [project_path('main.py'), project_path('requirements.txt')]
    files += sorted(project_path('src').rglob('*.py'))
    files += sorted(project_path('config').glob('*.yaml'))
    return {'commit': git('rev-parse', 'HEAD').strip(),
            'dirty': bool(git('status', '--porcelain').strip()),
            'diff': git('diff', 'HEAD', '--', 'src', 'config', 'main.py', 'requirements.txt'),
            'files': {p.relative_to(project_path('.')).as_posix(): p.read_text(encoding='utf-8')
                      for p in files},
            'command': subprocess.list2cmdline([sys.executable, *sys.argv])}


def validate_source(saved):
    current = source_record()
    if current['commit'] != saved['commit'] or current['files'] != saved['files']:
        raise StudyChangedError('Source or Git commit changed; create a new study instead of resuming.')


def _attempt(state, manifest, model, space, kind, config):
    """Record every attempt before creating its outputs."""
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + f'_{model}_{space}_{kind}_s{config["seed"]}'
    attempt = {'run_id': run_id, 'run_type': kind, 'model': model, 'arm': space,
               'seed': config['seed'], 'status': 'started', 'effective_config': config,
               'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
               'commit': state['source']['commit'], 'dirty': state['source']['dirty'],
               'log': str(manifest.parent / 'runs' / run_id / 'train.log') if kind == 'tuning' else None}
    state['attempts'].append(attempt)
    if kind == 'tuning':
        attempt['artifact'] = str(project_path(config['output_dir']) / 'models' / run_id)
    write_json(manifest, state)
    return attempt


def _prune_models(state, manifest):
    """Keep the current VAL winner in each space; retain all attempt records/logs."""
    keep = set()
    for record in state.get('tuning', {}).values():
        completed = [trial for trial in record['trials'] if trial['status'] == 'completed']
        if completed:
            best = min(completed, key=lambda trial: (trial['validation_rmse'], trial['number']))
            keep.add(best['result']['training_run_id'])
    for attempt in state.get('attempts', []):
        if attempt['run_type'] != 'tuning' or attempt.get('artifact_retained') is False:
            continue
        if attempt['run_id'] in keep or attempt['status'] == 'started':
            attempt['artifact_retained'] = True
            continue
        parent = (project_path(state['config']['output_dir']) / 'models').resolve()
        artifact = Path(attempt['artifact'])
        if artifact.resolve().parent != parent or artifact.name != attempt['run_id']:
            raise StudyChangedError('Refusing to remove a model outside its recorded run directory.')
        if artifact.exists():
            shutil.rmtree(artifact)
        attempt['artifact_retained'] = False
    write_json(manifest, state)


def train(config, space, run_id, directory, frame, expected_partitions):
    """Fit one trial on TRAIN/VAL and save its model, scaler and validation score."""
    validate_config(config)
    name = config['model']
    paths = output_paths(config, run_id, directory)
    artifact = paths['models'] / run_id
    artifact.mkdir()
    (artifact / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(paths['results'] / 'train.log', encoding='utf-8')
    logger.addHandler(handler)
    checkpoint = None
    try:
        splits, scalers = prepare_data(frame, config['data'], partitions=('TRAIN', 'VAL'))
        if partition_metadata(splits) != {key: expected_partitions[key] for key in splits}:
            raise StudyChangedError('Training origins changed during the experiment.')
        sizes = {key: len(part['X']) for key, part in splits.items()}
        logger.info('Rows=%s; samples=%s', len(frame), sizes)
        checkpoint = paths['results'] / 'checkpoint' / 'weights' if name == 'CNN_LSTM' else None
        if checkpoint:
            checkpoint.parent.mkdir()
        started = perf_counter()
        logger.info('Features=%s; PRE=%s; parameters=%s', config['data']['features'],
                    config['data']['pre'], config['models'][name])
        model, _ = train_model(name, config['models'][name], splits, config['seed'], checkpoint, logger)
        training_seconds = perf_counter() - started
        scaled = predict_scaled(model, name, splits['VAL']['X'])
        predicted = scalers['Y'].inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape)
        health = forecast_health(splits['VAL']['Y'], predicted)
        if not config.get('smoke_study'):
            require_healthy_forecasts(health, 'Validation')
        denominator = config['data']['evaluation']['normalization_kw']
        scores = evaluate(splits['VAL']['Y'], predicted, splits['VAL']['persistence'],
                          denominator, name, splits['VAL']['evaluation_mask'])
        scores = scores[scores['model'] == name]
        save_model(model, name, artifact)
        joblib.dump(scalers, artifact / 'scalers.joblib')
        metadata = {'model': name, 'arm': space, 'seed': config['seed'], 'run_id': run_id,
                    'data_contract': data_contract(config['data']), 'partitions': expected_partitions,
                    'input': input_identity(config['data']), 'environment': environment_metadata(),
                    'validation_mae': float(scores['MAE'].mean()), 'validation_rmse': float(scores['RMSE'].mean()),
                    'power_correction': frame.attrs.get('power_correction'),
                    'training_seconds': training_seconds, 'samples': sizes,
                    'smoke': bool(config.get('smoke_study'))}
        write_json(artifact / 'metadata.json', metadata)
        logger.info('VAL RMSE=%.6f; artifact=%s', metadata['validation_rmse'], artifact)
        return {'model': name, 'arm': space, 'seed': config['seed'],
                'subset': config['data']['features'][1:], 'pre': config['data']['pre'],
                'artifact': str(artifact), 'training_run_id': run_id,
                'validation_mae': metadata['validation_mae'], 'validation_rmse': metadata['validation_rmse'],
                'training_seconds': training_seconds}
    except (Exception, KeyboardInterrupt):
        logger.exception('Training failed or interrupted')
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
        if checkpoint and checkpoint.parent.exists():
            if checkpoint.parent.resolve().parent != paths['results'].resolve():
                raise StudyChangedError('Unexpected checkpoint cleanup path.')
            shutil.rmtree(checkpoint.parent)


def _fit_trial(candidate, arm, model, state, manifest, frame):
    validate_source(state['source'])
    config = arm_config(candidate, arm['subset'], model, candidate['seed'])
    attempt = _attempt(state, manifest, model, arm['id'], 'tuning', config)
    try:
        result = train(config, arm['id'], attempt['run_id'], manifest.parent, frame, state['partitions'])
        if not math.isfinite(result['validation_rmse']):
            raise ValueError('Non-finite validation RMSE.')
        validate_source(state['source'])
        attempt.update(status='completed', artifact=result['artifact'])
    except (Exception, KeyboardInterrupt) as error:
        attempt.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(manifest, state)
        raise
    write_json(manifest, state)
    return result


def predict(config, artifact, run_id, directory, frame):
    """Load a frozen model and scaler; build only TEST windows without fitting."""
    artifact = Path(artifact)
    saved = json.loads((artifact / 'metadata.json').read_text(encoding='utf-8'))
    if load_config(artifact / 'config.yaml') != config or saved['data_contract'] != data_contract(config['data']):
        raise StudyChangedError('Frozen model configuration changed.')
    if saved['seed'] != config['seed'] or saved['model'] != config['model']:
        raise StudyChangedError('Frozen model or seed changed.')
    if saved['environment'] != environment_metadata() or saved['input'] != input_identity(config['data']):
        raise StudyChangedError('Runtime or input changed since training.')
    scalers = joblib.load(artifact / 'scalers.joblib')
    splits, _ = prepare_data(frame, config['data'], scalers=scalers, partitions=('TEST',))
    if partition_metadata(splits)['TEST'] != saved['partitions']['TEST']:
        raise StudyChangedError('TEST origins changed since training.')
    paths = output_paths(config, run_id, directory)
    name = config['model']
    model = load_model(name, artifact)
    started = perf_counter()
    scaled = predict_scaled(model, name, splits['TEST']['X'])
    prediction_seconds = perf_counter() - started
    predicted = scalers['Y'].inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape)
    scores = export_results(splits['TEST'], predicted, name, saved['arm'], run_id, paths, config)
    metadata_path = paths['results'] / 'metadata.json'
    write_json(metadata_path, {'model_dir': str(artifact), 'model': name, 'arm': saved['arm'],
               'seed': config['seed'], 'run_id': run_id, 'training_run_id': saved['run_id'],
               'data_contract': saved['data_contract'], 'partitions': saved['partitions'],
               'input': saved['input'], 'environment': saved['environment'], 'smoke': saved['smoke'],
               'prediction_file': 'predictions.csv', 'metrics_file': 'metrics.csv',
               'prediction_seconds': prediction_seconds})
    primary = scores[(scores['variant'] == 'raw') & (scores['target'] == 'corrected') & (scores['model'] != 'Persistence')]
    print(f'{name} TEST mean-horizon RMSE: {primary.RMSE.mean():.6f}', flush=True)
    return metadata_path


def run(stage, config=None, manifest=None):
    """Execute exactly one stage and keep the same manifest across commands."""
    configure_runtime()
    if stage not in ('prepare', 'train', 'evaluate', 'report'):
        raise ValueError(f'Unknown stage: {stage}')
    if stage == 'prepare':
        config = copy.deepcopy(validate_config(config))
        validate_search(config)
        validate_study(config['study'])
        requested = copy.deepcopy(config)
        config['data']['origin_pre'] = max(config['tuning']['pre_choices'])
        config['data']['eligibility_features'] = list(dict.fromkeys([
            *config['data'].get('eligibility_features', []), config['data']['target'], *EXPANDED_FEATURES]))
        study_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_physics_tpe'
        directory = project_path(config['output_dir']) / 'results' / study_id
        directory.mkdir(parents=True)
        manifest = directory / 'manifest.json'
        state = {'workflow_version': WORKFLOW_VERSION, 'study_id': study_id, 'config': config,
                 'requested_config': requested, 'source': source_record(), 'attempts': [],
                 'environment': environment_metadata(), 'input': input_identity(config['data']),
                 'status': 'started', 'partitions': None, 'feature_spaces': None,
                 'screening': [], 'selected': None, 'final': []}
    else:
        if manifest is None or config is not None:
            raise ValueError('Continue with --resume and the saved config; changed settings require a new study.')
        manifest = project_path(manifest)
        directory = manifest.parent
        state = json.loads(manifest.read_text(encoding='utf-8'))
        if state.get('workflow_version') != WORKFLOW_VERSION:
            raise ValueError('Source layout changed; use the original code for an old study or prepare a new study.')
        validate_source(state['source'])
        config = state['config']
        if state['environment'] != environment_metadata() or state['input'] != input_identity(config['data']):
            raise StudyChangedError('Runtime or input changed; prepare a new study.')
        if state['processed_input'] != input_identity({'path': directory / 'features.joblib'}):
            raise StudyChangedError('Prepared features changed; prepare a new study.')
        for attempt in state['attempts']:
            if attempt['status'] == 'started':
                attempt.update(status='interrupted', error='Previous process ended before completion.')
        for record in state.get('tuning', {}).values():
            for trial in record['trials']:
                if trial['status'] == 'started':
                    trial.update(status='interrupted', error='Previous process ended before completion.')
        _prune_models(state, manifest)
    state['stage'] = stage
    write_json(manifest, state)
    try:
        if stage == 'prepare':
            spaces, frame = prepare_spaces(config, directory)
            full = arm_config(config, list(EXPANDED_FEATURES), 'XGBoost', config['seed'])
            full['data']['pre'] = full['data']['origin_pre']
            splits, _ = prepare_data(frame, full['data'])
            partitions = partition_metadata(splits)
            if partitions['TEST']['n_origins'] < config['study']['min_test_origins'] and not config.get('smoke_study'):
                raise ValueError('Insufficient common TEST origins for final comparison.')
            joblib.dump(frame, directory / 'features.joblib')
            state.update(feature_spaces=spaces, partitions=partitions, status='prepared',
                         preprocessing=frame.attrs.get('preprocessing', {}),
                         processed_input=input_identity({'path': directory / 'features.joblib'}))
        elif stage == 'train':
            if state['partitions'] is None:
                raise ValueError('Prepare data before training.')
            if not state.get('parameters_frozen'):
                state['status'] = 'started'
                write_json(manifest, state)
                tune(config, state, manifest, joblib.load(directory / 'features.joblib'))
                state.update(status='selection_frozen', selection_criterion='mean per-horizon validation RMSE in kW')
        elif stage == 'evaluate':
            if not state.get('parameters_frozen'):
                raise ValueError('Train and freeze every space winner before TEST evaluation.')
            frame = joblib.load(directory / 'features.joblib')
            for record in state['screening']:
                model, space = record['model'], record['arm']
                if any(row['model'] == model and row['arm'] == space for row in state['final']):
                    continue
                validate_source(state['source'])
                frozen = state['frozen_configs'][model + '/' + space]
                attempt = _attempt(state, manifest, model, space, 'prediction', frozen)
                attempt['artifact'] = record['artifact']
                write_json(manifest, state)
                try:
                    metadata = predict(frozen, record['artifact'], attempt['run_id'], directory, frame)
                    validate_source(state['source'])
                    attempt.update(status='completed', metadata=str(metadata))
                except (Exception, KeyboardInterrupt) as error:
                    attempt.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
                    raise
                state['final'].append({**record, 'prediction_metadata': str(metadata)})
                write_json(manifest, state)
            if state['status'] != 'completed':
                state['status'] = 'evaluated'
            consolidate_results(directory, state)
        else:
            if not state.get('parameters_frozen') or len(state['final']) != 2 * len(SPACE_NAMES):
                raise ValueError('Evaluate all frozen models before generating the report.')
            report(directory, state)
            state['status'] = 'completed'
        validate_source(state['source'])
        write_json(manifest, state)
        print(f'{stage} finished ({state["status"]}): {manifest}', flush=True)
        return manifest
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(manifest, state)
        raise


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
                         'error': trial.get('error'),
                         'parameters': json.dumps(trial['parameters'], ensure_ascii=False),
                         'training_run_id': trial.get('training_run_id'),
                         'selected': trial['number'] == record.get('best_trial'),
                         'subset': ';'.join(trial['subset']), **trial['params']})
    pd.DataFrame(rows).to_csv(Path(manifest).with_name('tuning_results.csv'), index=False)


def _print_trial(model, space, trial, record):
    completed = [item['validation_rmse'] for item in record['trials'] if item['status'] == 'completed']
    best = f'{min(completed):.6f} kW' if completed else 'none'
    print(f'[{model} | {space}] Trial {trial["number"] + 1}: {trial["status"]}', flush=True)
    print(f'  PRE={trial["pre"]}; features={trial["subset"]}', flush=True)
    print(f'  Parameters: {json.dumps(trial["parameters"], ensure_ascii=False)}', flush=True)
    if trial['status'] == 'completed':
        print(f'  VAL RMSE={trial["validation_rmse"]:.6f} kW; MAE={trial["validation_mae"]:.6f} kW; '
              f'time={trial["training_seconds"]:.1f}s; best VAL RMSE={best}', flush=True)
    else:
        print(f'  Reason: {trial["error"]}; time={trial["training_seconds"]:.1f}s; best VAL RMSE={best}', flush=True)


def tune(config, state, manifest, frame):
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
                attempt_count = len(state.get('attempts', []))
                started = perf_counter()
                try:
                    result = _fit_trial(candidate, arm, model, state, manifest, frame)
                    value = result['validation_rmse']
                    if not math.isfinite(value):
                        raise ValueError('Non-finite validation RMSE.')
                    item.update(status='completed', validation_rmse=value,
                                validation_mae=result['validation_mae'], result=result,
                                training_run_id=result['training_run_id'],
                                training_seconds=result.get('training_seconds', perf_counter() - started))
                    study.tell(trial, value)
                except (Exception, KeyboardInterrupt) as error:
                    item.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                error=str(error) or type(error).__name__, training_seconds=perf_counter() - started)
                    if len(state.get('attempts', [])) > attempt_count:
                        item['training_run_id'] = state['attempts'][-1]['run_id']
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    write_json(manifest, state)
                    export_search(state, manifest)
                    _prune_models(state, manifest)
                    _print_trial(model, space, item, record)
                    if isinstance(error, (KeyboardInterrupt, OSError, StudyChangedError)):
                        raise
                    continue
                write_json(manifest, state)
                export_search(state, manifest)
                _prune_models(state, manifest)
                _print_trial(model, space, item, record)
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
