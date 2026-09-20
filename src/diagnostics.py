"""Per-horizon forecast health checks, independent of model family and accuracy."""
import numpy as np
import pandas as pd


def forecast_health(observed, predicted):
    """Flag exactly constant forecasts against varying targets, not valid night-only data."""
    observed, predicted = (np.asarray(values, dtype=float) for values in (observed, predicted))
    if observed.ndim != 2 or not observed.size or observed.shape != predicted.shape:
        raise ValueError('Health checks require matching nonempty (origins, horizon) arrays.')
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError('Health checks found non-finite observations or predictions.')
    target_span, prediction_span = np.ptp(observed, axis=0), np.ptp(predicted, axis=0)
    return pd.DataFrame({
        'horizon': np.arange(1, observed.shape[1]+1), 'n': len(observed),
        'MAE': np.mean(abs(predicted-observed), axis=0),
        'target_span': target_span, 'prediction_span': prediction_span,
        'prediction_min': predicted.min(axis=0), 'prediction_max': predicted.max(axis=0),
        'prediction_std': predicted.std(axis=0),
        'zero_fraction': np.mean(predicted == 0, axis=0),
        'negative_fraction': np.mean(predicted < 0, axis=0),
        'collapsed': (prediction_span == 0) & (target_span > 0),
    })


def require_healthy_forecasts(health, partition):
    failed = health.loc[health['collapsed'], 'horizon'].tolist()
    if failed:
        raise ValueError(f'{partition} forecasts are constant despite varying targets at horizons {failed}.')
