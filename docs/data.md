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

All models share the EPOA/GHI/Hday reference during single-feature and combination
screening. Candidate PI arms add `Pac`, `Pdc`, `TempModule`, `TempCell`,
`physics_residual`, `performance_ratio`, `clear_sky_index`, or `solar_elevation`.
Their equations use the equipment
values in [`config/plant.yaml`](../config/plant.yaml). PI and non-PI arms retain the
same eligible forecast origins so that feature comparisons use the same targets.

| New feature | Definition and units |
| --- | --- |
| `physics_residual` | Observed `P_Solar[kW] - Pac`, in kW |
| `performance_ratio` | Observed `P_Solar[kW] / Pac`, dimensionless system-state proxy |
| `clear_sky_index` | Observed `GHI[kW1m2] / GHI_clear`, dimensionless weather-state proxy |
| `solar_elevation` | Geometric solar elevation in degrees |
| `solar_zenith` | `90 - solar_elevation`; supported alternative, excluded from default screening |

Residual and ratio inputs use only observations in `[t-pre, t]`; future measured
power never enters X. This performance ratio is an actual/theoretical power ratio,
not the conventional irradiance-normalized energy performance ratio.
Ratios are zero when Pac is at or below 0.05 kW or clear-sky GHI is at or below
0.02 kW/m2. These fixed YAML floors cover night and unstable low denominators;
missing observations remain missing, and daytime ratios above one are retained.

Solar calculations use the [NOAA fractional-year solar geometry equations](https://gml.noaa.gov/grad/solcalc/solareqns.PDF)
and the fixed [Haurwitz clear-sky GHI equation](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.clearsky.haurwitz.html)
at the timestamps, with verified latitude, longitude and timestamp timezone.
This simple clear-sky reference models geometry; it does not account for site-specific
aerosol, turbidity or altitude, so clear-sky index remains an approximate weather proxy.
Naive timestamps are localized with that timezone; timezone-aware timestamps are
converted. Ambiguous or nonexistent local times raise an error. Coordinates and
timezone are supplied in the default YAML as 55.6867 latitude, 12.0985 longitude,
and UTC for this study. These user-supplied values were checked for irradiance
alignment on TRAIN. Alternatively, input column names
can supply elevation (degrees) and clear-sky GHI (kW/m2). These columns must be
derived independently of held-out outcomes. Solar validity is shared across all arms.
The fixed replacement removes EPOA/GHI/Hday and inserts Pac/clear-sky index/elevation;
common measured weather and historical power inputs remain.

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
