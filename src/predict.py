"""Predict and evaluate saved models without training or refitting scalers.

Run from the project root: python -m src.predict --model-dir PATH
"""
import argparse
import json
from time import perf_counter
from datetime import datetime

from .runtime import configure_runtime


def run(model_dir, config=None, run_id=None, study_dir=None):
    configure_runtime()
    import joblib
    from .config import arm_name, data_contract, load_config, output_paths, validate_config
    from .data import load_data, prepare_data, build_features, input_identity, partition_metadata
    from .metrics import export_results
    from .models import load_model, predict_scaled
    from .runtime import environment_metadata, project_path

    artifact = project_path(model_dir)
    saved = json.loads((artifact / 'metadata.json').read_text(encoding='utf-8'))
    config = load_config(artifact / 'config.yaml') if config is None else config
    validate_config(config)
    name = saved['model']
    if config['model'] != name or saved['data_contract'] != data_contract(config['data']):
        raise ValueError('Saved model and current data/model configuration are incompatible.')
    if config['seed'] != saved['seed']:
        raise ValueError('Evaluation seed must match the saved training seed.')
    current_environment = environment_metadata()
    if saved['environment'] != current_environment:
        raise ValueError('Runtime environment changed since training; retrain before comparison.')
    identity = input_identity(config['data'])
    if saved.get('input') != identity:
        raise ValueError('Configured input path changed; retrain for a reproducible evaluation.')
    scalers = joblib.load(artifact / 'scalers.joblib')
    frame = build_features(load_data(config['data']), config['data'])
    splits, _ = prepare_data(frame, config['data'], scalers=scalers)
    if saved.get('partitions') != partition_metadata(splits):
        raise ValueError('Evaluation origins or partitions changed since training.')
    rung = arm_name(name, config['data']['physics'])
    run_id = run_id or (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + rung + f'_s{config["seed"]}_predict')
    paths = output_paths(config, run_id, study_dir)
    model = load_model(name, artifact)
    started = perf_counter()
    scaled = predict_scaled(model, name, splits['TEST']['X'])
    prediction_seconds = perf_counter() - started
    predicted = scalers['Y'].inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape)
    scores, health = export_results(splits['TEST'], predicted, rung, run_id, paths, config, saved['scenario_thresholds'])
    (paths['results'] / 'metadata.json').write_text(json.dumps({
        'model_dir': str(artifact), 'model': name, 'arm': rung, 'config': config,
        'seed': config['seed'], 'run_id': run_id, 'training_run_id': saved['run_id'],
        'data_contract': data_contract(config['data']), 'partitions': saved['partitions'],
        'input': identity, 'environment': saved['environment'],
        'scenario_thresholds': saved['scenario_thresholds'], 'smoke': saved['smoke'],
        'samples': len(predicted),
        'prediction_file': 'predictions.csv', 'metrics_file': 'metrics.csv',
        'forecast_health': health.to_dict('records'),
        'prediction_seconds': prediction_seconds, 'training_seconds': saved.get('training_seconds'),
    }, indent=2), encoding='utf-8')
    print(scores.to_string(index=False))
    print(f'Results: {paths["results"]}')
    return {'run_id': run_id, 'scores': scores, 'predicted': predicted,
            'metadata_path': paths['results'] / 'metadata.json'}


def main():
    parser = argparse.ArgumentParser(description='Predict and evaluate an existing PV forecasting model')
    parser.add_argument('--model-dir', required=True, help='Saved model artifact directory')
    parser.add_argument('--config', help='Optional compatible configuration; defaults to the saved configuration')
    args = parser.parse_args()
    configure_runtime()
    from .config import load_config
    run(args.model_dir, load_config(args.config) if args.config else None)


if __name__ == '__main__':
    main()
