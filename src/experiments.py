"""Physics-defined feature spaces with joint TPE and frozen TEST comparison."""
import argparse
import copy
import json
import math
import subprocess
import sys
from datetime import datetime
from .config import arm_config, validate_search
from .runtime import configure_runtime, project_path, write_json
from .data import SPACE_NAMES, EXPANDED_FEATURES, prepare_spaces


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
            'files': {str(p.relative_to(project_path('.'))): p.read_text(encoding='utf-8')
                      for p in files},
            'command': subprocess.list2cmdline([sys.executable, *sys.argv])}


def validate_source(saved):
    current = source_record()
    if current['commit'] != saved['commit'] or current['files'] != saved['files']:
        raise ValueError('Source or Git commit changed; create a new study instead of resuming.')


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
            'validation_rmse': metadata['validation_rmse'], 'pre': config['data']['pre'],
            'training_seconds': metadata['training_seconds'],
            'history': str(history), 'validation_health': str(health)}


def _prepare_study(config, directory):
    from .data import build_features, load_data, prepare_data, partition_metadata
    spaces = prepare_spaces(config, directory)
    full = arm_config(config, list(EXPANDED_FEATURES), 'XGBoost', config['seed'])
    full['data']['pre'] = full['data']['origin_pre']
    splits, _ = prepare_data(build_features(load_data(full['data']), full['data']), full['data'])
    return spaces, partition_metadata(splits)


def run(config=None, resume=None, evaluate_test=False):
    configure_runtime()
    from .config import validate_study
    from .runtime import environment_metadata
    from .data import input_identity
    from .predict import run as predict
    config = copy.deepcopy(config)
    if resume:
        manifest = project_path(resume)
        state = json.loads(manifest.read_text(encoding='utf-8'))
        if state.get('workflow_version') != 4:
            raise ValueError('Legacy ablation studies cannot resume under the joint-search protocol.')
        validate_source(state['source'])
        if config is not None and config != state['requested_config']:
            raise ValueError('Changed settings require a new study ID.')
        config, directory = state['config'], manifest.parent
        if state['environment'] != environment_metadata() or state['input'] != input_identity(config['data']):
            raise ValueError('Runtime or input changed; create a new study.')
    else:
        validate_search(config)
        validate_study(config['study'])
        requested = copy.deepcopy(config)
        config['data']['origin_pre'] = max(config['tuning']['pre_choices'])
        config['data']['eligibility_features'] = [config['data']['target'], *EXPANDED_FEATURES]
        study_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_physics_tpe'
        directory = project_path(config['output_dir']) / 'results' / study_id
        directory.mkdir(parents=True)
        state = {'workflow_version': 4, 'study_id': study_id, 'config': config,
                 'requested_config': requested, 'source': source_record(), 'attempts': [],
                 'environment': environment_metadata(), 'input': input_identity(config['data']),
                 'status': 'started', 'partitions': None, 'feature_spaces': None,
                 'screening': [], 'selected': None, 'final': []}
        manifest = directory / 'manifest.json'
    for attempt in state['attempts']:
        if attempt['status'] == 'started':
            attempt.update(status='interrupted', error='Previous process ended before completion.')
    for record in state.get('tuning', {}).values():
        for trial in record['trials']:
            if trial['status'] == 'started':
                trial.update(status='interrupted', error='Previous process ended before completion.')
    write_json(manifest, state)

    def fit(candidate, arm, model, run_type):
        validate_source(state['source'])
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + f'_{model}_{arm["id"]}_trial_s{candidate["seed"]}'
        effective = arm_config(candidate, arm['subset'], model, candidate['seed'])
        attempt = {'run_id': run_id, 'run_type': run_type, 'model': model, 'arm': arm['id'],
                   'seed': candidate['seed'], 'status': 'started', 'effective_config': effective,
                   'command': state['source']['command'], 'commit': state['source']['commit'],
                   'dirty': state['source']['dirty'], 'log': str(directory / 'runs' / run_id / 'train.log')}
        state['attempts'].append(attempt)
        write_json(manifest, state)
        try:
            result = _train_record(candidate, arm, model, candidate['seed'], state['partitions'], run_id, directory)
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

    try:
        if state['partitions'] is None:
            state['feature_spaces'], state['partitions'] = _prepare_study(config, directory)
            if state['partitions']['TEST']['n_origins'] < config['study']['min_test_origins'] and not config.get('smoke_study'):
                raise ValueError('Insufficient common TEST origins for final comparison.')
            write_json(manifest, state)
        if not state.get('parameters_frozen'):
            state['status'] = 'started'
            from .tuning import tune
            tune(config, state, manifest, fit)
            state.update(status='selection_frozen', selection_criterion='mean per-horizon validation RMSE in kW')
            write_json(manifest, state)
        from .evaluation import screening_summary
        screening_summary(state).to_csv(directory / 'validation_screening_summary.csv', index=False)
        if not evaluate_test:
            print(f'VAL selection frozen; TEST remains closed: {manifest}', flush=True)
            return manifest
        for model in ('XGBoost', 'CNN_LSTM'):
            for space in SPACE_NAMES:
                if any(row['model'] == model and row['arm'] == space for row in state['final']):
                    continue
                validate_source(state['source'])
                record = copy.deepcopy(next(row for row in state['screening'] if row['model'] == model and row['arm'] == space))
                run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + f'_{model}_{space}_predict'
                attempt = {'run_id': run_id, 'run_type': 'prediction', 'model': model, 'arm': space,
                           'seed': config['seed'], 'status': 'started', 'artifact': record['artifact'],
                           'effective_config': state['frozen_configs'][model + '/' + space],
                           'command': state['source']['command'], 'commit': state['source']['commit'],
                           'dirty': state['source']['dirty']}
                state['attempts'].append(attempt)
                write_json(manifest, state)
                try:
                    prediction = predict(record['artifact'], run_id=run_id, study_dir=directory)
                    attempt.update(status='completed', metadata=str(prediction['metadata_path']))
                except (Exception, KeyboardInterrupt) as error:
                    attempt.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
                    raise
                record['prediction_metadata'] = str(prediction['metadata_path'])
                state['final'].append(record)
                write_json(manifest, state)
        state['status'] = 'evaluated'
        write_json(manifest, state)
        from .evaluation import report
        report(directory, state)
        state['status'] = 'completed'
        write_json(manifest, state)
        print(f'Completed: {manifest}', flush=True)
        return manifest
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(manifest, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()
    from .config import load_config, smoke_config
    config = None if args.resume else load_config(args.config)
    if args.smoke:
        if args.resume:
            parser.error('--smoke cannot modify a resumed study.')
        config = smoke_config(config)
    run(config, args.resume, args.evaluate_test)


if __name__ == '__main__':
    main()
