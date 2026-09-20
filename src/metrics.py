"""Evaluate physical-unit predictions and export timestamped results."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
            rows.append({'model': label, 'horizon': h+1, 'MAE': float(np.mean(abs(error[:, h]))),
                         'MSE': mse, 'RMSE': float(np.sqrt(mse)),
                         'nRMSE_cap': float(100 * np.sqrt(mse) / p_nom_kw), 'n': len(observed)})
    return pd.DataFrame(rows)


def export_results(part, predicted, name, run_id, paths, config, thresholds):
    from .constraints import constrain_power
    from .scenarios import scenario_labels
    from .diagnostics import forecast_health
    constrained = constrain_power(predicted, config['data'])
    # The forecast head returns float32, which decimal text does not round-trip
    # exactly. These CSVs are the report's source of truth, so widen them to float64
    # and let read_study() verify stored errors exactly rather than within a bound.
    predicted = np.asarray(predicted, dtype=float)
    constrained = np.asarray(constrained, dtype=float)
    horizon = predicted.shape[1]
    columns = [f't+{h+1}' for h in range(horizon)]
    with pd.HDFStore(paths['results'] / f'{run_id}.h5', mode='w') as store:
        for key, values in [('Forecasted', predicted), ('Constrained', constrained),
                            ('Observed', part['Y']), ('Persistence', part['persistence'])]:
            store.put(key, pd.DataFrame(values, index=part['origins'], columns=columns))
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
    records.to_csv(paths['results'] / f'{run_id}.csv', index=False)
    health_tables = []
    for variant, values in [('raw', predicted), ('constrained', constrained)]:
        health = forecast_health(part['Y'], values)
        health.insert(0, 'run_id', run_id)
        health['variant'] = variant
        health_tables.append(health)
    pd.concat(health_tables, ignore_index=True).to_csv(paths['results'] / f'{run_id}_health.csv', index=False)
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
    scores.to_csv(paths['results'] / f'{run_id}_metrics.csv', index=False)
    fig, ax = plt.subplots()
    for (label, variant), group in scores.groupby(['model', 'variant'], sort=False):
        ax.plot(group['horizon'], group['RMSE'], marker='o', label=f'{label} ({variant})')
    ax.set(xlabel='Forecast horizon (steps)', ylabel='RMSE (kW)', title=name)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(paths['figures'] / f'{run_id}_rmse.png', dpi=180)
    plt.close(fig)
    return scores


def paired_bootstrap(delta_errors, resamples, seed, block_length):
    """Paired circular block bootstrap over sorted origins, not seed replicas.

    delta_errors contains PI absolute error minus baseline absolute error,
    averaged over the predeclared seed set at each origin before resampling.
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
