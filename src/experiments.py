"""One-seed TRAIN/VAL feature screening followed by a frozen TEST comparison."""
import argparse
import copy
import json
import itertools
import math
import subprocess
import sys
from datetime import datetime
from .config import arm_config
from .runtime import configure_runtime, project_path, write_json


def source_record():
    """Keep the command, Git state and readable source in the study manifest."""
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=project_path('.'), text=True)
    files = [project_path('main.py'), project_path('requirements.txt')]
    files += sorted(project_path('src').rglob('*.py'))
    files += sorted(project_path('config').glob('*.yaml'))
    return {'commit': git('rev-parse', 'HEAD').strip(),
            'dirty': bool(git('status', '--porcelain').strip()),
            'diff': git('diff', 'HEAD', '--', 'src', 'config', 'main.py', 'requirements.txt'),
            'files': {str(p.relative_to(project_path('.'))): p.read_text(encoding='utf-8')
                      for p in files},
            'command': subprocess.list2cmdline([sys.executable, *sys.argv])}


def validate_source(saved):
    current = source_record()
    if current['commit'] != saved['commit'] or current['files'] != saved['files']:
        raise ValueError('Source or Git commit changed; create a new study instead of resuming.')


def _arms(config):
    reference = config['experiment']['reference_features']
    arms = [{'id': 'nonpi_reference', 'subset': reference, 'category': 'reference'}]
    settings = config['experiment']
    threshold = settings.get('min_improvement_percent', .5)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('min_improvement_percent must be finite and nonnegative.')
    if type(settings.get('max_combinations', 3)) is not int or not 2 <= settings.get('max_combinations', 3) <= 3:
        raise ValueError('max_combinations must be 2 or 3.')
    for item in config['experiment']['physics_candidates']:
        if len(item['add']) != 1:
            raise ValueError('Initial screening requires single-feature candidates.')
        subset = reference + item['add']
        if len(subset) != len(set(subset)):
            raise ValueError(f"Candidate {item['id']} duplicates a reference feature.")
        arms.append({'id': item['id'], 'subset': subset, 'category': 'single'})
    if len({arm['id'] for arm in arms}) != len(arms):
        raise ValueError('Experiment arm identifiers must be unique.')
    return arms


def _winners(screening, arms, model, seed, threshold):
    model_rows = [row for row in screening if row['model'] == model]
    rows = {row['arm']: row for row in model_rows}
    if len(rows) != len(model_rows):
        raise ValueError('Every screening arm requires the configured seed exactly once.')
    for arm in arms:
        if arm['category'] not in ('reference', 'single'):
            continue
        row = rows.get(arm['id'])
        if row is None or row['seed'] != seed or not math.isfinite(row['validation_mae']):
            raise ValueError('Every screening arm requires finite MAE and the configured seed exactly once.')
    baseline = rows['nonpi_reference']['validation_mae']
    if baseline <= 0:
        return []
    return sorted([arm for arm in arms if arm['category'] == 'single'
                   and rows[arm['id']]['validation_mae'] < baseline * (1 - threshold / 100)],
                  key=lambda arm: (rows[arm['id']]['validation_mae'], arm['id']))


def _followup_arms(config, winners, model):
    """Predeclared ranking: at most three pairs of the top three winners."""
    reference = config['experiment']['reference_features']
    names = [next(name for name in arm['subset'] if name not in reference) for arm in winners[:3]]
    combinations = list(itertools.combinations(names, 2))
    arms = [{'id': 'combine_' + '_'.join(group), 'subset': reference + list(group),
             'category': 'combination', 'model': model}
            for group in combinations[:config['experiment'].get('max_combinations', 3)]]
    replacement = config['experiment']['physics_replacement']
    remove, add = replacement['remove'], replacement['add']
    if not remove or not set(remove).issubset(reference):
        raise ValueError('physics_replacement.remove must identify reference features.')
    subset = [name for name in reference if name not in remove] + add
    if not add or len(subset) != len(set(subset)) or subset == reference:
        raise ValueError('physics_replacement must provide a distinct unique subset.')
    arms.append({'id': 'physics_replacement', 'subset': subset,
                 'category': 'replacement', 'model': model})
    return arms


def _prepare_study(config, arms, directory):
    """Create TRAIN-only diagnostics and verify common forecast origins."""
    import numpy as np
    import pandas as pd
    from .data import (load_data, prepare_data, partition_metadata,
                       CANDIDATE_FEATURES, build_features)

    full = arm_config(config, CANDIDATE_FEATURES, 'XGBoost', config['seed'])
    frame = build_features(load_data(full['data']), full['data'])
    splits, _ = prepare_data(frame, full['data'])
    expected = partition_metadata(splits)

    train_end = int(np.floor(len(frame) * config['data']['split'][0] + 1e-9))
    columns = config['data']['features'] + list(CANDIDATE_FEATURES)
    train = frame.iloc[:train_end].loc[
        frame.attrs['eligible_rows'][:train_end], columns
    ].replace([np.inf, -np.inf], np.nan).dropna()
    train.corr().to_csv(directory / 'train_pearson_correlation.csv')
    train.corr(method='spearman').to_csv(directory / 'train_spearman_correlation.csv')

    redundancy = []
    for left, right, reason in [
        ('GHI[kW1m2]', 'POA Irr[kW1m2]', 'Horizontal versus array-plane irradiance'),
        ('Pdc', 'Pac', 'DC versus inverter-converted AC estimate'),
        ('TempModule', 'TempCell', 'Module versus irradiance-adjusted cell temperature'),
    ]:
        redundancy.append({
            'left': left,
            'right': right,
            'reason': reason,
            'train_pearson': float(train[left].corr(train[right])),
            'train_spearman': float(train[left].corr(train[right], method='spearman')),
            'n_train_rows': len(train),
        })
    pd.DataFrame(redundancy).to_csv(directory / 'physical_redundancy.csv', index=False)

    for arm in arms:
        candidate = arm_config(config, arm['subset'], 'XGBoost', config['seed'])
        candidate_splits, _ = prepare_data(build_features(frame, candidate['data']), candidate['data'])
        if partition_metadata(candidate_splits) != expected:
            raise ValueError('Candidate arms do not share the same forecast origins.')
    return expected


def _train_record(config, arm, model, seed, expected_partitions, run_id=None, study_dir=None):
    from .train import run as train
    result = train(arm_config(config, arm['subset'], model, seed),
                   run_id=run_id, study_dir=study_dir)
    metadata_path = result['artifact'] / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata['partitions'] != expected_partitions:
        raise ValueError('Training partitions changed during the experiment.')
    history = result['result_dir'] / 'history.json'
    health = result['result_dir'] / 'validation_health.csv'
    return {'model': model, 'arm': arm['id'], 'seed': seed, 'subset': arm['subset'],
            'artifact': str(result['artifact']), 'training_run_id': result['run_id'],
            'validation_mae': metadata['validation_mae'],
            'training_seconds': metadata['training_seconds'],
            'history': str(history), 'validation_health': str(health)}


def _select(screening, arms, seed):
    """Choose lowest VAL MAE, including the reference when physics does not help."""
    selected = {}
    for model in ('XGBoost', 'CNN_LSTM'):
        ranked = []
        for arm in arms:
            if arm.get('model', model) != model:
                continue
            rows = [row for row in screening if row['model'] == model and row['arm'] == arm['id']]
            if len(rows) != 1 or rows[0]['seed'] != seed:
                raise ValueError('Every screening arm requires the configured seed exactly once.')
            if not math.isfinite(rows[0]['validation_mae']):
                raise ValueError('Selection requires finite validation MAE.')
            ranked.append((rows[0]['validation_mae'], arm['category'] != 'reference', len(arm['subset']), arm['id']))
        selected[model] = min(ranked)[3]
    return selected


def run(config=None, resume=None, evaluate_test=False):
    """Freeze HPO, stage VAL experiments, and keep TEST behind an explicit flag."""
    config = copy.deepcopy(config)
    configure_runtime()
    from .config import validate_study
    from .predict import run as predict
    from .runtime import environment_metadata
    from .data import input_identity

    if resume:
        manifest = project_path(resume)
        state = json.loads(manifest.read_text(encoding='utf-8'))
        if 'source' not in state:
            raise ValueError('Legacy studies cannot resume as v2.0; start a new study.')
        validate_source(state['source'])
        if config is not None and config != state['requested_config']:
            raise ValueError('Changed settings require a new study ID.')
        config, arms, directory = state['config'], state['arms'], manifest.parent
        if state['environment'] != environment_metadata() or state['input'] != input_identity(config['data']):
            raise ValueError('Runtime environment or configured input path changed; resume is not comparable.')
    else:
        validate_study(config['study'])
        arms = _arms(config)
        for model in ('XGBoost', 'CNN_LSTM'):
            for arm in _followup_arms(config, [], model):
                arm_config(config, arm['subset'], model, config['seed'])
        study_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_features'
        directory = project_path(config['output_dir']) / 'results' / study_id
        directory.mkdir(parents=True)
        state = {'study_id': study_id, 'config': copy.deepcopy(config),
                 'requested_config': copy.deepcopy(config), 'source': source_record(),
                 'attempts': [], 'arms': arms, 'environment': environment_metadata(),
                 'input': input_identity(config['data']), 'partitions': None,
                 'status': 'started', 'screening': [], 'selected': None, 'final': []}
        manifest = directory / 'manifest.json'
        write_json(manifest, state)
    if state['partitions'] is None:
        try:
            state['partitions'] = _prepare_study(config, arms, directory)
            if (state['partitions']['TEST']['n_origins'] < config['study']['min_test_origins']
                    and not config.get('smoke_study')):
                raise ValueError('Insufficient TEST origins for the final comparison.')
        except (Exception, KeyboardInterrupt) as error:
            state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                         error=str(error))
            write_json(manifest, state)
            raise
        write_json(manifest, state)
    partitions = state['partitions']
    if state['selected'] is None or (evaluate_test and state['status'] != 'completed'):
        state['status'] = 'started'
    for attempt in state.get('attempts', []):
        if attempt['status'] == 'started':
            attempt.update(status='interrupted', error='Previous process ended before completion.')
    for model_record in state.get('tuning', {}).values():
        for trial in model_record['trials']:
            if trial['status'] == 'started':
                trial.update(status='interrupted', error='Previous process ended before completion.')
    write_json(manifest, state)

    def fit(candidate, arm, model, run_type):
        validate_source(state['source'])
        run_id = (datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                  + f'_{model}_{run_type}_{arm["id"]}_s{candidate["seed"]}')
        attempt = {'run_id': run_id, 'run_type': run_type, 'model': model,
                   'arm': arm['id'], 'seed': candidate['seed'], 'status': 'started',
                   'command': state['source']['command'], 'commit': state['source']['commit'],
                   'dirty': state['source']['dirty'],
                   'effective_config': arm_config(candidate, arm['subset'], model, candidate['seed']),
                   'log': str(directory / 'runs' / run_id / 'train.log')}
        state['attempts'].append(attempt)
        write_json(manifest, state)
        try:
            result = _train_record(candidate, arm, model, candidate['seed'],
                                   partitions, run_id, directory)
            if not math.isfinite(result['validation_mae']):
                raise ValueError('Training produced non-finite validation MAE.')
            validate_source(state['source'])
            attempt.update(status='completed', artifact=result['artifact'])
        except (Exception, KeyboardInterrupt) as error:
            attempt.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                           error=str(error))
            state['status'] = attempt['status']
            write_json(manifest, state)
            raise
        write_json(manifest, state)
        return result

    if config.get('tuning', {}).get('enabled') and not state.get('parameters_frozen'):
        from .tuning import tune
        tune(config, state, manifest, fit)
    if not state.get('parameters_frozen'):
        import yaml
        state['parameters_frozen'] = True
        manifest.with_name('frozen_config.yaml').write_text(
            yaml.safe_dump(state['config'], sort_keys=False), encoding='utf-8')
        write_json(manifest, state)
    config = state['config']
    validate_study(config['study'])
    screening_seed = config['seed']
    if state['selected'] is None:
        for model in ('XGBoost', 'CNN_LSTM'):
            for arm in arms:
                if arm['category'] not in ('reference', 'single'):
                    continue
                if any(row['model'] == model and row['arm'] == arm['id'] for row in state['screening']):
                    continue
                print(f"SCREEN {model} {arm['id']} seed={screening_seed}", flush=True)
                state['screening'].append(fit(config, arm, model, 'screening'))
                write_json(manifest, state)
        eligible = []
        for model in ('XGBoost', 'CNN_LSTM'):
            winners = _winners(state['screening'], arms, model, screening_seed,
                               config['experiment'].get('min_improvement_percent', .5))
            state.setdefault('winners', {})[model] = [arm['id'] for arm in winners]
            planned = _followup_arms(config, winners, model)
            for arm in planned:
                if not any(item['id'] == arm['id'] and item.get('model') == model for item in arms):
                    arms.append(arm)
            state['arms'] = arms
            write_json(manifest, state)
            for arm in planned:
                if any(row['model'] == model and row['arm'] == arm['id'] for row in state['screening']):
                    continue
                state['screening'].append(fit(config, arm, model, arm['category']))
                write_json(manifest, state)
            eligible += [{**arm, 'model': model} for arm in [arms[0], *winners, *planned]]
        state['selected'] = _select(state['screening'], eligible, screening_seed)
        state['status'] = 'selection_frozen'
        state['selection_criterion'] = 'validation MAE for the fixed seed'
        write_json(manifest, state)
    from .evaluation import screening_summary
    screening_summary(state).to_csv(directory / 'validation_screening_summary.csv', index=False)
    if not evaluate_test:
        print(f'VAL selection frozen; TEST remains closed: {manifest}', flush=True)
        return manifest
    for model in ('XGBoost', 'CNN_LSTM'):
        final_ids = list(dict.fromkeys(['nonpi_reference', state['selected'][model]]))
        for arm_id in final_ids:
            validate_source(state['source'])
            arm = next(item for item in arms if item['id'] == arm_id and item.get('model', model) == model)
            if any(row['model'] == model and row['arm'] == arm_id for row in state['final']):
                continue
            existing = next(row for row in state['screening']
                            if row['model'] == model and row['arm'] == arm_id)
            record = copy.deepcopy(existing)
            prediction_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + f'_{model}_{arm_id}_predict'
            attempt = {'run_id': prediction_id, 'run_type': 'prediction', 'status': 'started',
                       'model': model, 'arm': arm_id, 'seed': config['seed'],
                       'command': state['source']['command'], 'commit': state['source']['commit'],
                       'dirty': state['source']['dirty'], 'artifact': record['artifact'],
                       'effective_config': arm_config(config, arm['subset'], model, config['seed'])}
            state['attempts'].append(attempt)
            write_json(manifest, state)
            try:
                prediction = predict(record['artifact'], run_id=prediction_id, study_dir=directory)
                attempt.update(status='completed', metadata=str(prediction['metadata_path']))
            except (Exception, KeyboardInterrupt) as error:
                attempt.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                               error=str(error))
                state['status'] = attempt['status']
                write_json(manifest, state)
                raise
            record['prediction_metadata'] = str(prediction['metadata_path'])
            state['final'].append(record)
            write_json(manifest, state)
    state['status'] = 'evaluated'
    write_json(manifest, state)
    from .evaluation import report
    try:
        report(directory, state)
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=str(error))
        write_json(manifest, state)
        raise
    state['status'] = 'completed'
    write_json(manifest, state)
    print(f'Completed: {manifest}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true', help='Open TEST after VAL selection is frozen')
    args = parser.parse_args()
    configure_runtime()
    from .config import load_config
    config = None if args.resume else load_config(args.config)
    if args.smoke:
        if args.resume:
            raise ValueError('--smoke cannot modify a resumed study.')
        from .config import smoke_config
        config = smoke_config(config)
        config['experiment']['physics_candidates'] = config['experiment']['physics_candidates'][:1]
    run(config, args.resume, evaluate_test=args.evaluate_test)


if __name__ == '__main__':
    main()
