# Data and features

## Input and timing

The current input is data/raw/SOLETE_Pombo_5min.h5, a numeric HDF5 table with
unique DatetimeIndex. It has 131617 rows, from 2018-06-01 to 2019-09-01.
The reference's date labels and count differ; actual timestamps are preserved.
The source file is never modified. Sort and reindex five-minute gaps as NaN.
The tracked SOLETE_short.h5 is only an execution fixture.

Target P_Solar[kW]; weather inputs POA Irr[kW1m2], GHI[kW1m2], TEMPERATURE[degC],
WIND_SPEED[m1s], WIND_DIR[deg], HUMIDITY[%]. Pressure, wind power and recorded
azimuth/elevation are outside the default feature spaces.

At origin t, input covers [t-PRE,t], targets [t+1,t+60]. PRE counts samples,
not hours; starting PRE=30 is 2.5 hours. TRAIN/VAL/TEST are chronological
70/20/10. No label window crosses a boundary. Maximum PRE=120 and full candidate
validity give common origins across all trials. Missing physical inputs invalidate
windows; scalers and selectors fit TRAIN only. Timeline remains continuous at night.

## Physics and targets

Pac/Pdc/TempModule/TempCell use config/plant.yaml, which now follows Energy Reports
Table 1 rather than the different public-example arrays: 200 W x 18 x 2 =7.2 kW
DC, 10 kW AC. Constant inverter efficiency 0.98 is a documented choice because
the paper leaves the numeric maximum unspecified. MinutesOfDay=hour*12+minute//5;
HoursOfDay=hour. Physics selection keeps humidity/POA/Pac/TempCell/HoursOfDay.
Pearson threshold 0.7 is TRAIN-fitted, with humidity explicitly retained.

With correct_power=true, finite Pac>0.05 kW replaces measured power when
abs(measured-Pac)/Pac>0.2; missing power is filled from finite Pac. Target and Pac
<=0.001 kW become zero. This explicit interpretation differs from public v3.0's
one-sided 1.5x rule. Raw observations remain in observed_power_raw and in
prediction CSVs; raw_target_metrics.csv reports the same forecasts against raw
measurements. This sensitivity is essential because the main corrected target
depends on the same physics model used to build features. No unpublished outlier
or detrending settings are invented; raw irradiance is retained.

## Daylight and metrics

Primary VAL/TEST metrics use geometric solar elevation>0 at each target timestamp,
computed independently of measured target power using the configured UTC site:
latitude 55.6867, longitude 12.0985. This is a declared interpretation of the
paper's night omission; its exact original mask is unavailable. All histories
and training windows retain night samples. Training MAE and early stopping use
all finite windows; final selection uses daylight per-horizon VAL RMSE.
All spaces use identical TEST origins and evaluation masks. MAE/RMSE are kW;
nRMSE uses explicit 10 kW AC denominator, separate from 7.2 kW DC nameplate.
Forecasts are raw in primary comparisons; constraints are secondary outputs.

Solar geometry follows [NOAA equations](https://gml.noaa.gov/grad/solcalc/solareqns.PDF).
The supported clear-sky utility uses Haurwitz; residual/performance-ratio and
clear-sky inputs remain utilities outside the default search. No future weather
is assumed available. Full choices and reproduction limitations are in
[research](research.md).

## Source and license

The measurements come from the SOLETE
[dataset](https://doi.org/10.11583/DTU.17040767). The physical-feature implementation
is adapted from the associated
[source material](https://doi.org/10.11583/DTU.17040626) by Daniel Vazquez Pombo and
collaborators Oliver Gehrke and Henrik W. Bindner. The source material is licensed
under CC BY 4.0. SOLETE v3.0 documents corrections to the original split and RMSE
logic.
