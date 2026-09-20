"""Shared prediction constraints in kW, separate from physical input features."""
import numpy as np


def constrain_power(predicted, config):
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
