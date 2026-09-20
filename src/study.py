"""Run the predeclared four-arm, multi-seed study through existing entry points."""
import argparse
import copy
import json
from datetime import datetime

from .runtime import configure_runtime

# Keep native-library setup ahead of scientific configuration imports in the CLI.
if __name__ == '__main__':
    configure_runtime()

from .config import smoke_config, validate_study


def run(config, smoke=False):
    configure_runtime()
    from .config import output_paths
    from .data import load_data, prepare_data
    from .features import build_features
    from .train import run as train
    from .predict import run as predict
    from .provenance import partition_metadata
    from .report import assemble_report

    config = copy.deepcopy(config)
    validate_study(config['study'])
    if smoke:
        config = smoke_config(config)
    seeds = config['study']['seeds'][:1] if smoke else config['study']['seeds']
    splits = [config['data']['split']]
    probe = copy.deepcopy(config['data'])
    probe['physics'] = False
    probe_parts, _ = prepare_data(build_features(load_data(probe), probe), probe)
    if len(probe_parts['TEST']['X']) < config['study']['min_test_origins']:
        fallback = config['study'].get('fallback_split')
        if not fallback or fallback == splits[0]:
            raise ValueError('A distinct study.fallback_split is required for a small test partition.')
        splits.append(fallback)
    study_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + ('_study_smoke' if smoke else '_study')
    paths = output_paths(config)
    manifest_path = paths['results'] / f'{study_id}_manifest.json'
    manifest = {'study_id': study_id, 'config': config, 'smoke': smoke,
                'seeds': seeds, 'splits': splits, 'runs': [], 'status': 'running'}

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    save_manifest()
    try:
        for split_id, split in enumerate(splits):
            reference = None
            # Reject mismatched eligibility before spending time training any arm.
            for physics in (False, True):
                data = copy.deepcopy(config['data'])
                data.update(split=split, physics=physics)
                parts, _ = prepare_data(build_features(load_data(data), data), data)
                identity = partition_metadata(parts)
                if reference is not None and reference != identity:
                    raise ValueError('Physics arms have different forecast origins or target partitions.')
                reference = identity
            for seed in seeds:
                for model in ('XGBoost', 'CNN_LSTM'):
                    for physics in (False, True):
                        arm = copy.deepcopy(config)
                        arm.update(seed=seed, model=model)
                        arm['data'].update(split=split, physics=physics)
                        print(f'Study {study_id}: split={split_id}, seed={seed}, model={model}, physics={physics}')
                        trained = train(arm)
                        evaluated = predict(trained['artifact'])
                        manifest['runs'].append({'split_id': split_id, 'seed': seed, 'model': model,
                                                 'physics': physics, 'run_id': evaluated['run_id'],
                                                 'metadata_path': str(evaluated['metadata_path'])})
                        save_manifest()
        # Distinguish "every arm finished" from "the report was assembled": a reporting
        # failure must not discard a completed set of runs, which recover() can rebuild
        # without retraining.
        manifest['status'] = 'runs_complete'
        save_manifest()
        report_dir = assemble_report(manifest_path)
        manifest['status'] = 'complete'
        save_manifest()
        print(f'Study report: {report_dir}')
        return manifest_path
    except Exception:
        if manifest['status'] == 'running':
            manifest['status'] = 'failed'
        save_manifest()
        raise


def recover(manifest_path):
    """Rebuild a report for a study whose runs all completed.

    Nothing is retrained and nothing is hand-edited. assemble_report() revalidates
    every recorded run -- file hashes, configuration contract, partition identity and
    arm completeness -- so a manifest missing an arm is still rejected here.
    """
    from pathlib import Path
    from .report import assemble_report

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['status'] == 'running':
        raise ValueError('A study whose runs never finished cannot be recovered.')
    manifest['status'] = 'runs_complete'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Recovering study {manifest["study_id"]} from {len(manifest["runs"])} recorded runs')
    report_dir = assemble_report(path)
    manifest['status'] = 'complete'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return report_dir


def main():
    parser = argparse.ArgumentParser(description='Physics-feature ablation with paired multi-seed comparisons')
    parser.add_argument('--config', default='config/full.yaml')
    parser.add_argument('--smoke', action='store_true', help='One seed per arm, tiny training; diagnostics only')
    parser.add_argument('--recover', metavar='MANIFEST',
                        help='Rebuild the report for a completed study without retraining')
    args = parser.parse_args()
    configure_runtime()
    if args.recover:
        print(recover(args.recover))
        return
    from .config import load_config
    run(load_config(args.config), args.smoke)


if __name__ == '__main__':
    main()
