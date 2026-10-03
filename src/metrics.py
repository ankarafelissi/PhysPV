"""Evaluate physical-unit predictions and export timestamped results."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
            raise ValueError('A confirmed positive AC capacity is required for clipping.')
        values = np.minimum(values, limit)
    return values


def forecast_health(observed, predicted):
    """Detect constant forecast heads against varying targets."""
    observed, predicted = (np.asarray(values, dtype=float) for values in (observed, predicted))
    if observed.ndim != 2 or not observed.size or observed.shape != predicted.shape:
        raise ValueError('Health checks require matching nonempty (origins, horizon) arrays.')
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError('Health checks found non-finite observations or predictions.')
    target_span = np.ptp(observed, axis=0)
    prediction_span = np.ptp(predicted, axis=0)
    return pd.DataFrame({
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
    })


def require_healthy_forecasts(health, partition):
    failed = health.loc[health['collapsed'], 'horizon'].tolist()
    if failed:
        raise ValueError(
            f'{partition} forecasts are constant despite varying targets at horizons {failed}.')


def fit_scenario_thresholds(train):
    """Fit scenario cutoffs from TRAIN only."""
    for key in ('sky_variability', 'power_level', 'ramp_magnitude'):
        if key not in train or not np.isfinite(train[key]).all():
            raise ValueError(f'Finite TRAIN {key} is required for scenario thresholds.')
    return {
        'sky_variability': float(np.median(train['sky_variability'])),
        'power_level': float(np.median(train['power_level'])),
        'ramp_magnitude': np.median(train['ramp_magnitude'], axis=0).tolist(),
        'method': ('TRAIN median; sky=trailing POA std(ddof=0); '
                   'power=trailing target mean; ramp=abs(y[t+h]-y[t]), post-hoc only'),
    }


def scenario_labels(part, thresholds):
    horizon = part['Y'].shape[1]
    ramps = np.asarray(thresholds['ramp_magnitude'])
    if ramps.shape != (horizon,):
        raise ValueError('Stored ramp thresholds do not match the forecast horizon.')
    return {
        'sky_condition': np.repeat(
            np.where(part['sky_variability'] <= thresholds['sky_variability'],
                     'clear', 'cloudy')[:, None], horizon, axis=1),
        'power_level_class': np.repeat(
            np.where(part['power_level'] <= thresholds['power_level'],
                     'low', 'high')[:, None], horizon, axis=1),
        'ramp_class': np.where(part['ramp_magnitude'] <= ramps, 'steady', 'ramp'),
    }


def evaluate(observed, predicted, persistence, p_nom_kw, name='Forecaster'):
    observed, predicted, persistence = [np.asarray(x, dtype=float) for x in (observed, predicted, persistence)]
    if observed.ndim != 2 or observed.size == 0:
        raise ValueError('Evaluation requires a nonempty (origins, horizon) array.')
    if isinstance(p_nom_kw, bool) or not np.isfinite(p_nom_kw) or p_nom_kw <= 0:
        raise ValueError('p_nom_kw must be a finite positive DC nameplate capacity.')
    if observed.shape != predicted.shape or observed.shape != persistence.shape:
        raise ValueError('Prediction, observation and baseline shapes must match.')
    if not all(np.isfinite(a).all() for a in (observed, predicted, persistence)):
        raise ValueError('Evaluation contains non-finite values.')
    rows = []
    for label, values in [(name, predicted), ('Persistence', persistence)]:
        error = values - observed
        for h in range(observed.shape[1]):
            mse = float(np.mean(error[:, h] ** 2))
            variance = float(np.mean((observed[:, h] - observed[:, h].mean()) ** 2))
            rows.append({'model': label, 'horizon': h+1, 'MAE': float(np.mean(abs(error[:, h]))),
                         'R2': float(1 - mse / variance) if variance > 0 else np.nan,
                         'MSE': mse, 'RMSE': float(np.sqrt(mse)),
                         'nRMSE_cap': float(100 * np.sqrt(mse) / p_nom_kw), 'n': len(observed)})
    return pd.DataFrame(rows)


def export_results(part, predicted, name, run_id, paths, config, thresholds):
    constrained = constrain_power(predicted, config['data'])
    # Widen float32 forecasts before calculating errors for CSV export.
    predicted = np.asarray(predicted, dtype=float)
    constrained = np.asarray(constrained, dtype=float)
    horizon = predicted.shape[1]
    records = pd.DataFrame({
        'run_id': run_id, 'model': name, 'seed': config['seed'], 'physics': config['data']['physics'],
        'forecast_origin': np.repeat(part['origins'].to_numpy(), horizon),
        'target_time': part['target_times'].reshape(-1),
        'horizon': np.tile(np.arange(1, horizon+1), len(predicted)),
        'observed': part['Y'].reshape(-1), 'predicted': predicted.reshape(-1),
        'predicted_raw': predicted.reshape(-1), 'predicted_constrained': constrained.reshape(-1),
        'persistence': part['persistence'].reshape(-1),
        'sky_variability': np.repeat(part['sky_variability'], horizon),
        'power_level': np.repeat(part['power_level'], horizon),
        'ramp_magnitude': part['ramp_magnitude'].reshape(-1),
        'absolute_error': np.abs(predicted-part['Y']).reshape(-1),
        'squared_error': ((predicted-part['Y'])**2).reshape(-1),
    })
    for key, values in scenario_labels(part, thresholds).items():
        records[key] = values.reshape(-1)
    records.to_csv(paths['results'] / 'predictions.csv', index=False)
    health_tables = []
    for variant, values in [('raw', predicted), ('constrained', constrained)]:
        health = forecast_health(part['Y'], values)
        health.insert(0, 'run_id', run_id)
        health['variant'] = variant
        health_tables.append(health)
    health = pd.concat(health_tables, ignore_index=True)
    scores = evaluate(part['Y'], predicted, part['persistence'], config['data']['p_nom_kw'], name)
    scores['variant'] = 'raw'
    limited = evaluate(part['Y'], constrained, part['persistence'], config['data']['p_nom_kw'], name)
    limited = limited[limited['model'] != 'Persistence'].copy()
    limited['variant'] = 'constrained'
    scores = pd.concat([scores, limited], ignore_index=True)
    scores['run_id'], scores['seed'] = run_id, config['seed']
    scores['physics'] = (scores['model'] != 'Persistence') & config['data']['physics']
    scores['negative_fraction'] = [float(np.mean((part['persistence'] if row.model == 'Persistence' else
        constrained if row.variant == 'constrained' else predicted)[:, row.horizon-1] < 0))
        for row in scores.itertuples()]
    capacity = config['data']['plant'].get('ac_capacity_kw')
    scores['above_capacity_fraction'] = [float(np.mean((part['persistence'] if row.model == 'Persistence' else
        constrained if row.variant == 'constrained' else predicted)[:, row.horizon-1] > capacity))
        if capacity is not None else np.nan for row in scores.itertuples()]
    scores.to_csv(paths['results'] / 'metrics.csv', index=False)
    fig, ax = plt.subplots()
    for (label, variant), group in scores.groupby(['model', 'variant'], sort=False):
        ax.plot(group['horizon'], group['RMSE'], marker='o', label=f'{label} ({variant})')
    ax.set(xlabel='Forecast horizon (steps)', ylabel='RMSE (kW)', title=name)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(paths['figures'] / f'{run_id}_rmse.png', dpi=180)
    plt.close(fig)
    return scores, health


def paired_bootstrap(delta_errors, resamples, seed, block_length):
    """Paired circular block bootstrap over sorted origins, not seed replicas.

    delta_errors contains PI absolute error minus baseline absolute error,
    averaged by forecast origin before resampling.
    """
    delta = np.asarray(delta_errors, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Bootstrap requires finite paired per-origin error differences.')
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
        indices = ((starts[:, None] + np.arange(length)) % len(delta)).ravel()[:len(delta)]
        means[i] = delta[indices].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {'delta_mae': float(delta.mean()), 'ci_low': float(low), 'ci_high': float(high),
            'n_origins': len(delta), 'block_length': length,
            'effective_blocks': len(delta) / length}


def export_training_curve(history, name, run_id, paths):
    if history:
        fig, ax = plt.subplots()
        if name == 'CNN_LSTM':
            ax.plot(history['loss'], label='train')
            ax.plot(history['val_loss'], label='validation')
        else:
            for step, values in history.items():
                metric, curve = next(iter(values['validation_0'].items()))
                ax.plot(curve, label=f't+{step} validation {metric}')
        ax.set(xlabel='Training iteration', ylabel='Loss (scaled target)', title=name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths['figures'] / f'{run_id}_training.png', dpi=180)
        plt.close(fig)
