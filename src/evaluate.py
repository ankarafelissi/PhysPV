"""Forecast metrics, diagnostic exports and frozen four-space comparison."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import write_json


def constrain_power(predicted, config):
    """Apply configured output constraints without changing raw predictions."""
    values = np.asarray(predicted, dtype=float).copy()
    if not np.isfinite(values).all():
        raise ValueError('Cannot constrain non-finite predictions.')
    rules = config['constraints']
    if rules['nonnegative']:
        values = np.maximum(values, 0)
    if rules['capacity']:
        limit = config['plant'].get('ac_capacity_kw')
        if limit is None or not np.isfinite(limit) or limit <= 0:
            raise ValueError(
                'A confirmed positive AC capacity is required for clipping.'
            )
        values = np.minimum(values, limit)
    return values


def forecast_health(observed, predicted):
    """Detect constant forecast heads against varying targets."""
    observed, predicted = (
        np.asarray(values, dtype=float) for values in (observed, predicted)
    )
    if observed.ndim != 2 or not observed.size or observed.shape != predicted.shape:
        raise ValueError(
            'Health checks require matching nonempty (origins, horizon) arrays.'
        )
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError('Health checks found non-finite observations or predictions.')
    target_span = np.ptp(observed, axis=0)
    prediction_span = np.ptp(predicted, axis=0)
    return pd.DataFrame(
        {
            'horizon': np.arange(1, observed.shape[1] + 1),
            'n': len(observed),
            'MAE': np.mean(abs(predicted - observed), axis=0),
            'target_span': target_span,
            'prediction_span': prediction_span,
            'prediction_min': predicted.min(axis=0),
            'prediction_max': predicted.max(axis=0),
            'prediction_std': predicted.std(axis=0),
            'zero_fraction': np.mean(predicted == 0, axis=0),
            'negative_fraction': np.mean(predicted < 0, axis=0),
            'collapsed': (prediction_span == 0) & (target_span > 0),
        }
    )


def require_healthy_forecasts(health, partition):
    """Raise ValueError if any forecast is constant while its targets vary."""
    failed = health.loc[health['collapsed'], 'horizon'].tolist()
    if failed:
        raise ValueError(
            f'{partition} forecasts are constant despite varying targets at horizons {failed}.'
        )


def evaluate(observed, predicted, persistence, p_nom_kw, name='Forecaster', mask=None):
    """Return per-horizon metrics for predictions and Persistence.

    Inputs have shape (origins, horizon) in kW. The optional Boolean mask
    selects scored target timestamps and must retain every horizon.
    p_nom_kw is the normalization capacity; the default workflow passes the
    10 kW AC rating. nRMSE_cap is a percentage, and constant targets yield NaN R2.
    """
    observed, predicted, persistence = [
        np.asarray(x, dtype=float) for x in (observed, predicted, persistence)
    ]
    if observed.ndim != 2 or observed.size == 0:
        raise ValueError('Evaluation requires a nonempty (origins, horizon) array.')
    if isinstance(p_nom_kw, bool) or not np.isfinite(p_nom_kw) or p_nom_kw <= 0:
        raise ValueError('p_nom_kw must be a finite positive DC nameplate capacity.')
    if observed.shape != predicted.shape or observed.shape != persistence.shape:
        raise ValueError('Prediction, observation and baseline shapes must match.')
    if not all(np.isfinite(a).all() for a in (observed, predicted, persistence)):
        raise ValueError('Evaluation contains non-finite values.')
    rows = []
    mask = np.ones(observed.shape, bool) if mask is None else np.asarray(mask, bool)
    if mask.shape != observed.shape or not mask.any(axis=0).all():
        raise ValueError(
            'Evaluation mask must match targets and retain samples at every horizon.'
        )
    for label, values in [(name, predicted), ('Persistence', persistence)]:
        error = values - observed
        for step in range(observed.shape[1]):
            valid = mask[:, step]
            mse = float(np.mean(error[valid, step] ** 2))
            truth = observed[valid, step]
            variance = float(np.mean((truth - truth.mean()) ** 2))
            rows.append(
                {
                    'model': label,
                    'horizon': step + 1,
                    'MAE': float(np.mean(abs(error[valid, step]))),
                    'R2': float(1 - mse / variance) if variance > 0 else np.nan,
                    'MSE': mse,
                    'RMSE': float(np.sqrt(mse)),
                    'nRMSE_cap': float(100 * np.sqrt(mse) / p_nom_kw),
                    'n': int(valid.sum()),
                }
            )
    return pd.DataFrame(rows)


def export_results(part, predicted, name, space, run_id, paths, config):
    """Write temporary per-attempt tables, later consolidated after TEST completes."""
    predicted = np.asarray(predicted, dtype=float)
    constrained = constrain_power(predicted, config['data'])
    horizon = predicted.shape[1]
    records = pd.DataFrame(
        {
            'run_id': run_id,
            'model': name,
            'arm': space,
            'forecast_origin': np.repeat(part['origins'].to_numpy(), horizon),
            'target_time': part['target_times'].reshape(-1),
            'horizon': np.tile(np.arange(1, horizon + 1), len(predicted)),
            'observed': part['Y'].reshape(-1),
            'observed_raw': part['Y_raw'].reshape(-1),
            'predicted': predicted.reshape(-1),
            'predicted_constrained': constrained.reshape(-1),
            'persistence': part['persistence'].reshape(-1),
            'evaluated': part['evaluation_mask'].reshape(-1),
        }
    )
    records.to_csv(paths['results'] / 'predictions.csv', index=False)
    denominator = config['data']['evaluation']['normalization_kw']
    tables = []
    for target, observed, mask in [
        ('corrected', part['Y'], part['evaluation_mask']),
        (
            'raw_measurement',
            np.nan_to_num(part['Y_raw']),
            part['evaluation_mask'] & np.isfinite(part['Y_raw']),
        ),
    ]:
        scores = evaluate(
            observed, predicted, part['persistence'], denominator, name, mask
        )
        scores['target'], scores['variant'] = target, 'raw'
        tables.append(scores)
        if not np.array_equal(predicted, constrained):
            limited = evaluate(
                observed, constrained, part['persistence'], denominator, name, mask
            )
            limited = limited[limited['model'] != 'Persistence'].copy()
            limited['target'], limited['variant'] = target, 'constrained'
            tables.append(limited)
    scores = pd.concat(tables, ignore_index=True)
    scores['run_id'], scores['arm'] = run_id, space
    scores.to_csv(paths['results'] / 'metrics.csv', index=False)
    return scores


def consolidate_results(directory, state):
    """Commit a single prediction table and metric table; then remove temporary copies."""
    combined = {
        name: pd.read_csv(directory / name) if (directory / name).exists() else None
        for name in ('predictions.csv', 'metrics.csv')
    }
    predictions, metrics = [], []
    for record in state['final']:
        metadata_path = Path(record['prediction_metadata'])
        run_id = metadata_path.parent.name
        for name, tables in [
            ('predictions.csv', predictions),
            ('metrics.csv', metrics),
        ]:
            temporary = metadata_path.parent / name
            if temporary.exists():
                frame = pd.read_csv(temporary)
            else:
                frame = combined[name].loc[combined[name]['run_id'] == run_id].copy()
            if name == 'metrics.csv':
                frame = frame[frame.horizon > 0]
            tables.append(frame)
    predictions = pd.concat(predictions, ignore_index=True)
    metrics = pd.concat(metrics, ignore_index=True)
    baseline = (
        metrics.query('model == "Persistence"')
        .drop_duplicates(['target', 'variant', 'horizon'])
        .copy()
    )
    baseline['arm'] = 'Persistence'
    metrics = pd.concat(
        [metrics.query('model != "Persistence"'), baseline], ignore_index=True
    )
    means = []
    for (model, arm, target, variant), group in metrics.groupby(
        ['model', 'arm', 'target', 'variant']
    ):
        means.append(
            {
                'model': model,
                'arm': arm,
                'target': target,
                'variant': variant,
                'horizon': 0,
                'run_id': group.run_id.iloc[0],
                'n': int(group.n.min()),
                **{
                    key: group[key].mean()
                    for key in ('MAE', 'RMSE', 'nRMSE_cap', 'R2', 'MSE')
                },
            }
        )
    metrics = pd.concat([metrics, pd.DataFrame(means)], ignore_index=True)
    for name, table in [('predictions.csv', predictions), ('metrics.csv', metrics)]:
        target = directory / name
        temporary = target.with_suffix('.tmp')
        table.to_csv(temporary, index=False)
        temporary.replace(target)
    for record in state['final']:
        metadata_path = Path(record['prediction_metadata'])
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.update(
            prediction_file='../../predictions.csv', metrics_file='../../metrics.csv'
        )
        write_json(metadata_path, metadata)
        for name in ('predictions.csv', 'metrics.csv'):
            (metadata_path.parent / name).unlink(missing_ok=True)
