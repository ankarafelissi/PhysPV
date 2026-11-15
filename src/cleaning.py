"""Conservative QC for the seven measured inputs used by the PV study.

Policy and references: eda/cleaning_workflow.md. No interpolation or model imputation.
"""

import numpy as np
import pandas as pd

TRAINING_COLUMNS = (
    'P_Solar[kW]', 'TEMPERATURE[degC]', 'HUMIDITY[%]', 'WIND_SPEED[m1s]',
    'WIND_DIR[deg]', 'GHI[kW1m2]', 'POA Irr[kW1m2]',
)
POLICY = 'conservative_5min_v1'
QC_CODES = {
    1: 'nonfinite_to_nan', 2: 'invalid_domain_to_nan',
    4: 'joint_zero_dropout_to_nan', 8: '360_degrees_to_zero',
    16: 'midnight_review_only', 32: 'negative_measurement_review_only',
}


def clean_training_data(frame):
    """Return a copy and per-cell bit flags; unused columns remain untouched."""
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError('QC requires a unique DatetimeIndex.')
    if not frame.index.is_monotonic_increasing:
        raise ValueError('Sort observations before QC.')
    if len(frame) > 1 and not (np.diff(frame.index.asi8) == 300_000_000_000).all():
        raise ValueError('QC requires the continuous five-minute grid.')
    missing = set(TRAINING_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError('Missing training columns: ' + ', '.join(sorted(missing)))
    result = frame.copy()
    flags = pd.DataFrame(0, index=frame.index, columns=TRAINING_COLUMNS, dtype='uint8')

    def mark(column, mask, code, replace=False):
        flags.loc[mask, column] |= code
        if replace:
            result.loc[mask, column] = np.nan

    for column in TRAINING_COLUMNS:
        mark(column, ~np.isfinite(frame[column]), 1, replace=True)
    domains = {
        'HUMIDITY[%]': (frame['HUMIDITY[%]'] < 0) | (frame['HUMIDITY[%]'] > 1),
        'WIND_SPEED[m1s]': frame['WIND_SPEED[m1s]'] < 0,
        'TEMPERATURE[degC]': frame['TEMPERATURE[degC]'] <= -273.15,
        'WIND_DIR[deg]': (frame['WIND_DIR[deg]'] < 0) | (frame['WIND_DIR[deg]'] > 360),
    }
    for column, mask in domains.items():
        mark(column, mask & np.isfinite(frame[column]), 2, replace=True)
    # A coarse aggregate beyond 360 cannot be safely interpreted as a raw revolution.
    north = frame['WIND_DIR[deg]'] == 360
    mark('WIND_DIR[deg]', north, 8)
    result.loc[north, 'WIND_DIR[deg]'] = 0.0
    dropout = (frame['HUMIDITY[%]'] == 0) & (frame['WIND_SPEED[m1s]'] == 0)
    for column in ['HUMIDITY[%]', 'WIND_SPEED[m1s]']:
        mark(column, dropout, 4, replace=True)
    midnight = (frame.index.hour == 0) & (frame.index.minute == 0)
    for column in TRAINING_COLUMNS[1:5]:
        mark(column, midnight, 16)
    # Small negative radiometer offsets and signed meter readings need context.
    for column in ['P_Solar[kW]', 'GHI[kW1m2]', 'POA Irr[kW1m2]']:
        mark(column, frame[column] < 0, 32)
    result.attrs['sensor_cleaning'] = {
        'policy': POLICY,
        'invalidated_cells': int(((flags.to_numpy() & 7) != 0).sum()),
        'flag_counts': {
            str(code): {c: int(((flags[c] & code) != 0).sum()) for c in TRAINING_COLUMNS}
            for code in QC_CODES
        },
    }
    return result, flags
