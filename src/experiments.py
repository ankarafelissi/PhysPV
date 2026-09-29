"""One-seed TRAIN/VAL feature screening followed by a frozen TEST comparison."""
import argparse
import copy
import json
from datetime import datetime
from .config import arm_config
from .provenance import write_json
from .runtime import configure_runtime


def _score(metadata):
    return sum(row['MAE'] for row in metadata['validation_health']) / len(metadata['validation_health'])


def _arms(config):
    reference = config['experiment']['reference_features']
    arms = [{'id': 'nonpi_reference', 'subset': reference, 'category': 'reference'}]
    for item in config['experiment']['physics_candidates']:
        subset = reference + item['add']
        if len(subset) != len(set(subset)):
            raise ValueError(f"Candidate {item['id']} duplicates a reference feature.")
        arms.append({'id': item['id'], 'subset': subset, 'category': 'physics'})
    if len({arm['id'] for arm in arms}) != len(arms):
        raise ValueError('Experiment arm identifiers must be unique.')
    return arms


def _prepare_study(config, arms, directory):
    """Create TRAIN-only diagnostics and verify common forecast origins."""
    import numpy as np
    import pandas as pd
    from .data import load_data, prepare_data
    from .features import CANDIDATE_FEATURES, build_features
    from .provenance import partition_metadata

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


def _train_record(config, arm, model, seed, expected_partitions):
    from .paths import project_path
    from .train import run as train
    result = train(arm_config(config, arm['subset'], model, seed))
    metadata_path = result['artifact'] / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata['partitions'] != expected_partitions:
        raise ValueError('Training partitions changed during the experiment.')
    history = project_path(config['output_dir']) / 'results' / f"{result['run_id']}_history.json"
    health = project_path(config['output_dir']) / 'results' / f"{result['run_id']}_validation_health.csv"
    return {'model': model, 'arm': arm['id'], 'seed': seed, 'subset': arm['subset'],
            'artifact': str(result['artifact']), 'training_run_id': result['run_id'],
            'validation_mae': _score(metadata), 'training_seconds': metadata['training_seconds'],
            'history': str(history), 'validation_health': str(health)}


def _select(screening, arms, seed):
    """Choose the lowest validation-MAE physics arm for each model."""
    selected = {}
    for model in ('XGBoost', 'CNN_LSTM'):
        ranked = []
        for arm in arms:
            if arm['category'] != 'physics':
                continue
            rows = [row for row in screening if row['model'] == model and row['arm'] == arm['id']]
            if len(rows) != 1 or rows[0]['seed'] != seed:
                raise ValueError('Every screening arm requires the configured seed exactly once.')
            ranked.append((rows[0]['validation_mae'], len(arm['subset']), arm['id']))
        selected[model] = min(ranked)[2]
    return selected


def run(config=None, resume=None):
    configure_runtime()
    import tables  # noqa: F401
    from .config import validate_study
    from .paths import project_path
    from .predict import run as predict
    from .provenance import environment_metadata, input_identity

    if resume:
        manifest = project_path(resume)
        state = json.loads(manifest.read_text(encoding='utf-8'))
        config, arms, directory = state['config'], state['arms'], manifest.parent
        if state['environment'] != environment_metadata() or state['input'] != input_identity(config['data']):
            raise ValueError('Runtime environment or configured input path changed; resume is not comparable.')
        partitions = state['partitions']
    else:
        validate_study(config['study'])
        arms = _arms(config)
        study_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_features'
        directory = project_path(config['output_dir']) / 'results' / study_id
        directory.mkdir(parents=True)
        partitions = _prepare_study(config, arms, directory)
        if (partitions['TEST']['n_origins'] < config['study']['min_test_origins']
                and not config.get('smoke_study')):
            raise ValueError('Insufficient TEST origins for the final comparison.')
        state = {'study_id': study_id, 'config': config, 'arms': arms, 'environment': environment_metadata(),
                 'input': input_identity(config['data']), 'partitions': partitions,
                 'status': 'screening', 'screening': [], 'selected': None, 'final': []}
        manifest = directory / 'manifest.json'
        write_json(manifest, state)
    validate_study(config['study'])
    screening_seed = config['seed']
    if state['selected'] is None:
        for model in ('XGBoost', 'CNN_LSTM'):
            for arm in arms:
                if any(row['model'] == model and row['arm'] == arm['id'] for row in state['screening']):
                    continue
                print(f"SCREEN {model} {arm['id']} seed={screening_seed}", flush=True)
                state['screening'].append(_train_record(config, arm, model, screening_seed, partitions))
                write_json(manifest, state)
        state['selected'] = _select(state['screening'], arms, screening_seed)
        state['status'] = 'selection_frozen'
        write_json(directory / 'selection.json', {'selected': state['selected'], 'screening': state['screening'],
                                                    'criterion': 'validation MAE for the fixed seed'})
        write_json(manifest, state)
    for model in ('XGBoost', 'CNN_LSTM'):
        final_ids = ['nonpi_reference', state['selected'][model]]
        for arm_id in final_ids:
            arm = next(item for item in arms if item['id'] == arm_id)
            if any(row['model'] == model and row['arm'] == arm_id for row in state['final']):
                continue
            existing = next(row for row in state['screening']
                            if row['model'] == model and row['arm'] == arm_id)
            record = copy.deepcopy(existing)
            prediction = predict(record['artifact'])
            record['prediction_metadata'] = str(prediction['metadata_path'])
            state['final'].append(record)
            write_json(manifest, state)
    state['status'] = 'evaluated'
    write_json(manifest, state)
    from .evaluation import report
    report(directory, state)
    state['status'] = 'complete'
    write_json(manifest, state)
    print(f'Completed: {manifest}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
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
    run(config, args.resume)


if __name__ == '__main__':
    main()
