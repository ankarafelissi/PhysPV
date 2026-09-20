"""Train models and persist artifacts; test prediction lives in src.predict."""
import copy
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import yaml

from .config import arm_name, data_contract, output_paths, validate_config, smoke_config
from .data import load_data, prepare_data
from .paths import project_path
from .features import build_features
from .metrics import export_training_curve
from .models import SUPPORTED_MODELS, load_model, predict_scaled, save_model, train_model
from .provenance import environment_metadata, input_identity, partition_metadata
from .scenarios import fit_thresholds
from .diagnostics import forecast_health, require_healthy_forecasts


def run(config, model_name=None, smoke=False):
    config = copy.deepcopy(config)
    name = model_name or config['model']
    if name not in SUPPORTED_MODELS:
        raise ValueError(f'Choose one of {SUPPORTED_MODELS}.')
    config['model'] = name
    validate_config(config)
    if smoke:
        config = smoke_config(config)
    is_smoke = smoke or config.get('smoke_study', False)
    rung = arm_name(name, config['data']['physics'])
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + rung + f'_s{config["seed"]}' + ('_smoke' if is_smoke else '')
    paths = output_paths(config)
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(paths['results'] / f'{run_id}_train.log', encoding='utf-8')
    logger.addHandler(handler)
    try:
        frame = build_features(load_data(config['data']), config['data'])
        cache = project_path(config['data']['processed_path'])
        if is_smoke:
            cache = cache.with_name(cache.stem + '_smoke' + cache.suffix)
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_hdf(cache, key='features', mode='w')
        splits, scalers = prepare_data(frame, config['data'])
        thresholds = fit_thresholds(splits['TRAIN'])
        sizes = {key: len(part['X']) for key, part in splits.items()}
        logger.info('Rows=%s; samples=%s', len(frame), sizes)
        print(f'{name}: {sizes}')
        model, history = train_model(name, config['models'][name], splits, config['seed'])
        # Validate restored best weights on VAL, never use TEST to select a model.
        validation_scaled = predict_scaled(model, name, splits['VAL']['X'])
        validation_predicted = scalers['Y'].inverse_transform(
            validation_scaled.reshape(-1, 1)).reshape(validation_scaled.shape)
        health = forecast_health(splits['VAL']['Y'], validation_predicted)
        health.insert(0, 'run_id', run_id)
        health.to_csv(paths['results'] / f'{run_id}_validation_health.csv', index=False)
        (paths['results'] / f'{run_id}_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if not is_smoke:
            require_healthy_forecasts(health, 'Validation')
        artifact = paths['models'] / run_id
        artifact.mkdir()
        save_model(model, name, artifact)
        joblib.dump(scalers, artifact / 'scalers.joblib')
        (artifact / 'metadata.json').write_text(json.dumps({
            'model': name, 'arm': rung, 'seed': config['seed'], 'run_id': run_id,
            'data_contract': data_contract(config['data']),
            'partitions': partition_metadata(splits), 'input': input_identity(config['data']),
            'environment': environment_metadata(), 'scenario_thresholds': thresholds,
            'validation_health': health.to_dict('records'),
            'train_target_end': str(splits['TRAIN']['target_times'][-1, -1]),
            'validation_target_end': str(splits['VAL']['target_times'][-1, -1]),
            'samples': sizes, 'smoke': is_smoke,
        }, indent=2), encoding='utf-8')
        (artifact / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        windows = splits['VAL']['X'][:16]
        np.testing.assert_allclose(predict_scaled(model, name, windows),
                                   predict_scaled(load_model(name, artifact), name, windows), rtol=1e-5, atol=1e-6)
        export_training_curve(history, name, run_id, paths)
        logger.info('Reload predictions match; artifact=%s', artifact)
        print(f'Model: {artifact}')
        print(f'Predict: python -m src.predict --model-dir "{artifact}"')
        return {'artifact': artifact, 'run_id': run_id}
    except Exception:
        logger.exception('Training failed')
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
