"""Forecast metrics, diagnostic exports and frozen four-space comparison."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import SPACE_NAMES
from .runtime import project_path, write_json


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


def paired_bootstrap(delta_errors, resamples, seed, block_length):
    """Paired circular block bootstrap over sorted origins, not seed replicas.

    delta_errors contains PI absolute error minus baseline absolute error,
    averaged by forecast origin before resampling.
    """
    delta = np.asarray(delta_errors, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError(
            'Bootstrap requires finite paired per-origin error differences.'
        )
    if type(resamples) is not int or resamples < 1000:
        raise ValueError('At least 1000 bootstrap resamples are required.')
    if type(block_length) is not int or block_length < 1:
        raise ValueError('Bootstrap block length must be a positive integer.')
    length = min(block_length, len(delta))
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    blocks = int(np.ceil(len(delta) / length))
    for i in range(resamples):
        starts = rng.integers(0, len(delta), size=blocks)
        indices = ((starts[:, None] + np.arange(length)) % len(delta)).ravel()[
            : len(delta)
        ]
        means[i] = delta[indices].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        'delta_mae': float(delta.mean()),
        'ci_low': float(low),
        'ci_high': float(high),
        'n_origins': len(delta),
        'block_length': length,
        'effective_blocks': len(delta) / length,
    }


def _save(figure, directory, name):
    """Save one PNG and close its figure to release plotting resources."""
    figure.tight_layout()
    figure.savefig(directory / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(figure)


def create_figures(state, comparison, predictions, output):
    """Plot horizon errors and first TEST-day forecasts from saved predictions."""
    output.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(SPACE_NAMES, ('#777777', '#37799b', '#c26633', '#7d6594')))
    plt.rcParams.update(
        {
            'font.size': 8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'legend.frameon': False,
            'svg.fonttype': 'none',
            'pdf.fonttype': 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for axis, model in zip(axes, ('XGBoost', 'CNN_LSTM')):
        for space in SPACE_NAMES:
            group = comparison.query(
                'model == @model and arm == @space and horizon > 0'
            )
            hours = (
                group.horizon
                * pd.Timedelta(state['config']['data']['resolution']).total_seconds()
                / 3600
            )
            axis.plot(hours, group.RMSE, color=colors[space], label=space)
        persistence = comparison.query('model == "Persistence" and horizon > 0')
        hours = (
            persistence.horizon
            * pd.Timedelta(state['config']['data']['resolution']).total_seconds()
            / 3600
        )
        axis.plot(hours, persistence.RMSE, ':', color='black', label='Persistence')
        axis.set(title=model, xlabel='Forecast horizon (hours)')
        axis.legend(fontsize=7)
    axes[0].set_ylabel('TEST RMSE (kW)')
    _save(figure, output, 'forecast_horizon_error')
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for axis, model in zip(axes, ('XGBoost', 'CNN_LSTM')):
        for space in SPACE_NAMES:
            frame = predictions[model, space].query('horizon == 1').iloc[:864]
            axis.plot(
                pd.to_datetime(frame.target_time),
                frame.predicted,
                color=colors[space],
                label=space,
            )
        axis.plot(
            pd.to_datetime(frame.target_time),
            frame.observed,
            color='black',
            label='Observed',
        )
        axis.set(title=model, ylabel='Power (kW)')
        axis.legend(ncol=3, fontsize=7)
    axes[-1].tick_params(axis='x', rotation=25)
    _save(figure, output, 'prediction_curves')


def _compare_physics(predictions, config):
    """Compare paired absolute errors by origin, then bootstrap each horizon."""
    settings = config['study']
    paired = []
    for model in ('XGBoost', 'CNN_LSTM'):
        physics = predictions[model, 'Expanded-Physics'].query('evaluated')
        for baseline in ('Original', 'Expanded-Pearson', 'Intrinsic'):
            base = predictions[model, baseline].query('evaluated')
            for horizon in range(config['data']['horizon'] + 1):
                left = physics[physics.horizon == horizon] if horizon else physics
                right = base[base.horizon == horizon] if horizon else base
                delta = abs(left.predicted.to_numpy() - left.observed.to_numpy()) - abs(
                    right.predicted.to_numpy() - right.observed.to_numpy()
                )
                origin_delta = (
                    pd.DataFrame(
                        {'origin': left.forecast_origin.to_numpy(), 'delta': delta}
                    )
                    .groupby('origin')
                    .delta.mean()
                    .to_numpy()
                )
                interval = paired_bootstrap(
                    origin_delta,
                    settings['bootstrap_resamples'],
                    settings['bootstrap_seed'],
                    max(settings['bootstrap_block_length'], config['data']['horizon']),
                )
                paired.append(
                    {
                        'model': model,
                        'baseline': baseline,
                        'horizon': horizon,
                        **interval,
                    }
                )
    return paired


def report(directory, state):
    """Read the two consolidated tables; generate only the two requested PNGs."""
    config = state['config']
    metrics = pd.read_csv(directory / 'metrics.csv')
    all_predictions = pd.read_csv(directory / 'predictions.csv')
    comparison = metrics.query('target == "corrected" and variant == "raw"')
    predictions = {
        (record['model'], record['arm']): all_predictions[
            all_predictions.run_id == Path(record['prediction_metadata']).parent.name
        ].reset_index(drop=True)
        for record in state['final']
    }
    anchor = next(iter(predictions.values()))
    pairing = [
        'forecast_origin',
        'target_time',
        'horizon',
        'observed',
        'evaluated',
        'persistence',
    ]
    for frame in predictions.values():
        if not frame[pairing].equals(anchor[pairing]):
            raise ValueError(
                'All optimized configurations must share identical TEST targets.'
            )
    paired = _compare_physics(predictions, config)
    create_figures(
        state,
        comparison,
        predictions,
        project_path(config['output_dir']) / 'figures' / state['study_id'],
    )
    state['summary'] = {
        'seed': config['seed'],
        'smoke_diagnostic_only': bool(config.get('smoke_study')),
        'validation_selected_space': state['selected'],
        'test_mean_horizon_metrics': comparison.query('horizon == 0').to_dict(
            'records'
        ),
        'physics_vs_baselines': paired,
        'interpretation': 'Single-seed equal-attempt-budget joint optimization; TEST never selects configurations.',
    }
    write_json(directory / 'manifest.json', state)


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
