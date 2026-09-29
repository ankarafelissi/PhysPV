# Data and Features

## Input contract

The pipeline reads one numeric HDF5 table with a `DatetimeIndex`. Timestamps are
sorted, duplicate timestamps are rejected, and missing intervals remain missing.
Raw observations are never rewritten.

| Column | Meaning |
| --- | --- |
| `P_Solar[kW]` | Measured PV-power target in kW |
| `POA Irr[kW1m2]` | Plane-of-array irradiance in kW/m2 |
| `GHI[kW1m2]` | Global horizontal irradiance in kW/m2 |
| `TEMPERATURE[degC]` | Ambient temperature |
| `WIND_SPEED[m1s]` | Wind speed |
| `HUMIDITY[%]` | Relative humidity |

The full run uses `data/raw/SOLETE_Pombo_60min.h5`. The tracked
`data/raw/SOLETE_short.h5` is only an execution fixture.

## Windows and scaling

At forecast origin `t`, inputs cover `[t-pre, t]` and targets cover
`[t+1, t+horizon]`. The default configuration uses `pre=24`, `horizon=10`, and a
chronological 70/20/10 TRAIN/VAL/TEST split. Target windows cannot cross partition
boundaries. Scalers are fit on TRAIN only and reused for VAL and TEST.

## Physics-derived features

All models share the EPOA/GHI/Hday reference. Candidate PI arms add `Pac`, `Pdc`,
`TempModule`, and `TempCell` in a fixed order. Their equations use the equipment
values in [`config/plant.yaml`](../config/plant.yaml). PI and non-PI arms retain the
same eligible forecast origins so that feature comparisons use the same targets.

The 7.44 kW DC nameplate is used to normalize RMSE. It is not an AC clipping limit.
Capacity clipping is disabled unless a verified positive `ac_capacity_kw` is supplied.
Raw and constrained predictions are stored separately; primary comparisons use raw
predictions.

## Source and license

The measurements come from the SOLETE
[dataset](https://doi.org/10.11583/DTU.17040767). The physical-feature implementation
is adapted from the associated
[source material](https://doi.org/10.11583/DTU.17040626) by Daniel Vazquez Pombo and
collaborators Oliver Gehrke and Henrik W. Bindner. The source material is licensed
under CC BY 4.0. SOLETE v3.0 documents corrections to the original split and RMSE
logic.
