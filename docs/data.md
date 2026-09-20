# Data and Features

## Input contract

Use a single numeric HDF5 table with a `DatetimeIndex`, readable by
`pandas.read_hdf(path)`. Timestamps are sorted; duplicates and interval misalignment
are rejected. Missing timestamps remain missing: no interpolation or filling.

| Column | Unit |
| --- | --- |
| `P_Solar[kW]` | kW; measured target |
| `POA Irr[kW1m2]`, `GHI[kW1m2]` | kW/m2 |
| `TEMPERATURE[degC]` | degrees Celsius |
| `WIND_SPEED[m1s]` | m/s |
| `HUMIDITY[%]` | percent |

`SOLETE_short.h5` contains 24 hourly rows for execution checks only. The full config
uses `SOLETE_Pombo_60min.h5`. Raw measurements are never rewritten;
`data.correct_power` must remain `false`.

## Windows and scaling

At origin `t`, inputs span `[t-pre, t]` and targets span `[t+1, t+horizon]`.
The current full configuration uses **25 hourly observations** (`pre=24`) to forecast
**10 hours**, with a chronological 70/20/10 split.

Each target window belongs entirely to one partition; observed history may precede
its boundary. Reject windows with invalid selected inputs, physical eligibility,
or target observations across the history and forecast window. Fit scalers on TRAIN
only and reuse them for VAL/TEST. Prediction never refits them.

## Physics switch

`data.features` lists ordered common inputs. With `data.physics: true`, code appends
`Pac`, `Pdc`, `TempModule`, `TempCell` in that order. Do not list them manually.
Both arms use the same physical-input validity mask and forecast origins.
Calendar and rolling-history features are shared, not classified as physics.
`TempModule_RP` is unsupported.

[Plant parameters](../config/plant.yaml) drive irradiance/temperature-based DC power,
interpolated AC efficiency, and module/cell temperature estimates. Physical inputs
must be finite, wind nonnegative, and ambient temperature above absolute zero.
Negative irradiance is floored inside the estimate only.

DC nameplate is **7.44 kW**, used for normalization, not AC clipping. The AC rating
is unknown by default. Upper prediction clipping requires a verified
`plant.ac_capacity_kw` and `constraints.capacity: true`.

## Prediction outputs

CNN-LSTM requires a **linear output head** so negative scores retain corrective
gradients. Physical bounds are applied after inverse scaling. CSVs preserve raw
and constrained predictions, observations, persistence, timestamps, and errors;
`predicted` aliases `predicted_raw`. Primary comparisons use raw forecasts.

Processed tables are inspection artifacts, not validated reusable caches. Always
construct experiments through the raw-data pipeline.

Source material: SOLETE [dataset](https://doi.org/10.11583/DTU.17040767) and
[physical-feature code](https://doi.org/10.11583/DTU.17040626), Daniel Vazquez Pombo
(dataset contributors: Oliver Gehrke and Henrik W. Bindner), CC BY 4.0.
The current physical-feature implementation is adapted from that source.
