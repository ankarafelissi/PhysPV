# Research protocol: five-minute physics-guided TPE

## Objective and reference

Physics defines the admissible feature space; bounded TPE jointly chooses feature
subsets, PRE and model hyperparameters. Compare Original, Expanded-Pearson,
Expanded-Physics and Intrinsic for XGBoost and CNN_LSTM, plus Persistence.
Negative and mixed outcomes are valid. This estimates search-space design plus
model adaptation, not isolated feature causality or a guaranteed Physics ranking.

The primary reference is [Pombo et al., Energy Reports 2022](https://doi.org/10.1016/j.egyr.2022.05.006),
Sections 3–5. Use five-minute resolution, H=60 (five hours), chronological
70/20/10 and PRE candidates equal to the union of its coarse/fine grids:
0/5/10/15/20/25/30/35/40/45/50/60/70/80/90/100/110/120. PRE counts previous
samples in addition to the current sample; PRE=30 means 150 minutes.

## Feature spaces and model search

Historical target power is mandatory. Original has six optional weather inputs:
ambient temperature, humidity, wind speed/direction, GHI and POA. Pressure is
excluded. Expanded adds Pac, Pdc, TempModule, TempCell, MinutesOfDay (0–287)
and HoursOfDay (0–23). Pearson uses absolute TRAIN target correlation >=0.7,
with humidity explicitly retained as in the reference. It is refitted on TRAIN,
so its actual pool need not equal the paper's fixed list. Physics retains humidity,
POA, Pac, TempCell and HoursOfDay; Intrinsic has no optional inputs.

Seed 11; exactly 40 attempts per model/space, including ten startup attempts:
320 fits total. The first attempt uses all pool features, PRE=30 and YAML defaults.
Every optional input has a Boolean inclusion variable; empty subsets are allowed.
Failed/interrupted attempts consume the budget and remain recorded. There is no
TEST-driven budget extension or retuning. This remains a bounded search, not proof
of convergence or a global optimum.

XGBoost uses independent direct ensembles at each of 60 horizons: 700-tree ceiling,
depth 2–6, learning rate 0.01–0.1, child weight 1–8, lambda 0.5–10, row/column
sampling 0.7–1; VAL early stopping at 40 rounds. GPU hist is enabled after a local
compatibility check. CNN_LSTM searches filters 16/32/48, kernel/pool sizes 1–7,
1–3 LSTM layers with 5–20 equal-width units, 0–2 dense layers with 1–5 units,
batch 16/32 and Adam learning rate 0.0002–0.002. MAE training loss, 1000-epoch
ceiling and 20-epoch VAL-MAE patience are fixed. Kernel padding is causal.
TensorFlow 2.10 currently runs on CPU. TPE, early stopping, tied LSTM widths,
and XGBoost replacing RF are explicit adaptations; no additional model family.

## Data, target processing and evaluation

Read SOLETE_Pombo_5min.h5 without changing the source file. Its local timestamps
span 2018-06-01 to 2019-09-01 (131617 rows), whereas the paper states
2019-06-01 to 2020-08-31 (131616 rows). Retain actual timestamps and record this
discrepancy; no undocumented year shifting or claim of matching acquisition dates.
Sort/reindex the continuous timeline; reject duplicate times. Missing physical
inputs invalidate windows. Input/target windows never cross gaps or target split
boundaries. Maximum PRE=120 and full raw-candidate validity define common origins.
Scalers and Pearson selection use TRAIN only. VAL/TEST may use observed earlier
context. No future weather or power enters model inputs.

Plant Table 1: 200 W modules, 18 series x 2 parallel, DC nameplate 7.2 kW;
a=-3.56, b=-0.075, cell delta=3 C, gamma=-0.00478/C, AC capacity 10 kW.
Use constant inverter efficiency 0.98, predeclared from the public SMA curve;
the paper does not specify its numeric maximum. The public v3.0 example instead
has 165 W and 125 W arrays totalling 7.44 kW; these are not silently mixed.

The default target estimates available power. For finite Pac>0.05 kW, replace a
measurement if abs(measured-Pac)/Pac>0.2. Fill missing target from finite Pac and
set target/Pac <=0.001 kW to zero. This is an explicit interpretation of the
paper's >20% deviation prose, not its unpublished original cleaning code. The
public v3.0 example uses a different one-sided 1.5x threshold. Save untouched
measurements in observed_power_raw and export raw_target_metrics.csv to expose
sensitivity to target correction. Corrected outcomes are partly physics-derived;
any apparent Physics advantage must be interpreted with that dependence in mind.

Do not invent undocumented detrending/outlier thresholds. The paper mentions
clear-sky detrending and outlier removal without a fully executable specification;
this implementation keeps raw irradiance and finite continuous samples. Those
remain reproduction limitations alongside year labels and model differences.

Training/checkpoint MAE uses all finite windows. Primary selection uses mean
per-horizon VAL RMSE over geometric daylight target timestamps (solar elevation>0).
Freeze all eight winners before TEST. TEST uses exactly the same daylight rule,
not a threshold chosen from TEST power. Night history remains available and target
windows retain their true timing. Report kW MAE/RMSE, correct R2 and nRMSE divided
by the predeclared 10 kW AC rating, plus raw-target sensitivity and Persistence.
Normalization is explicitly stated because the paper does not fully specify its
percentage denominator. Horizon 0 in tables averages per-horizon scores.

Physics-minus-baseline paired MAE intervals use evaluated daylight records,
averaged by origin and circular block bootstrap (2000 resamples, seed 2026,
288 origins or at least H). These are conditional temporal intervals, not seed
or search uncertainty or multiple-comparison-adjusted inference. Equal attempts
do not imply equal time or convergence. Results must not be forced to match paper.

## Reproducibility and status

The existing manifest keeps effective configs, source text/diff, Git commit/dirty
state, command, input metadata, seed, attempts, logs/checkpoints and lifecycle.
Resume requires unchanged source/config/environment/input. Changes start new IDs;
old attempts and artifacts are not overwritten. No hashes or per-run Markdown.

The previous hourly study below is retained as historical evidence of the earlier
adaptation. It is not a result of the current five-minute protocol and cannot be
compared numerically with the reference. The new study's status and results will
be recorded here after execution; TEST stays closed until all configurations freeze.

### Current five-minute run (4 October 2026)

Study `20261004_163215_553398_physics_tpe`; fixed seed-11 budget of 320 attempts.
[Manifest](../outputs/results/20261004_163215_553398_physics_tpe/manifest.json),
[trial table](../outputs/results/20261004_163215_553398_physics_tpe/tuning_results.csv).
Status: **failed**, checked 2026-10-04 17:16:51 local.
Training attempts: 14 completed, 1 failed,
0 interrupted, 0 currently started.
TEST evaluations: 0/8. TRAIN/VAL/TEST common origins:
91,951 / 26,265 / 13,103. The TRAIN-fitted Pearson pool matches the reference's
named humidity/GHI/POA/Pac/Pdc/TempModule/TempCell list. Correction changed
5,653 rows (mean absolute change 0.00835 kW); preprocessing_audit.csv holds counts.
Compilation, 45 unit tests and the 8-fit/8-prediction smoke passed before launch.
Smoke outputs are execution diagnostics only. Source/config remain frozen.

Execution ended with status failed: [WinError 5] Access is denied: 'C:\\Users\\felix\\Github\\pv\\outputs\\results\\20261004_163215_553398_physics_tpe\\manifest.tmp' -> 'C:\\Users\\felix\\Github\\pv\\outputs\\results\\20261004_163215_553398_physics_tpe\\manifest.json'.
Failure history is preserved; no final result or budget extension is fabricated.

## Legacy hourly experiment (4 October 2026)

Study `20261004_140151_675271_physics_tpe` completed with seed 11: 96/96 successful training
attempts, eight successful frozen TEST evaluations, and no failed/interrupted
attempts. Unit verification passed 43 tests; compilation and the four-space
end-to-end diagnostic also passed. The final audit confirmed source immutability,
saved/frozen config equality, all log paths and identical TEST partitions.
TRAIN/VAL/TEST used 7,620 / 2,185 / 1,088 common forecast origins.

[Manifest](../outputs/results/20261004_140151_675271_physics_tpe/manifest.json), [all trials](../outputs/results/20261004_140151_675271_physics_tpe/tuning_results.csv),
[frozen configs](../outputs/results/20261004_140151_675271_physics_tpe/frozen_config.yaml), [full comparison](../outputs/results/20261004_140151_675271_physics_tpe/model_comparison.csv),
[paired differences](../outputs/results/20261004_140151_675271_physics_tpe/physics_comparison.csv).

### Frozen configurations and search cost

Historical measured PV power is mandatory and omitted from the optional-subset
column below. Trial numbers here are one-based; CSV/manifest numbers are zero-based.

| Model | Space | Trial | PRE | Optional subset | VAL RMSE (kW) | Search training time (min) |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| XGBoost | Original | 1 | 24 | TEMPERATURE[degC], HUMIDITY[%], WIND_SPEED[m1s], WIND_DIR[deg], GHI[kW1m2], POA Irr[kW1m2], Pressure[mbar] | 0.794919 | 2.17 |
| XGBoost | Expanded-Pearson | 6 | 48 | HUMIDITY[%] | 0.803950 | 1.65 |
| XGBoost | Expanded-Physics | 12 | 48 | HUMIDITY[%], POA Irr[kW1m2], Pac, TempCell, HoursOfDay | 0.795757 | 1.92 |
| XGBoost | Intrinsic | 8 | 48 | None | 0.791463 | 0.66 |
| CNN_LSTM | Original | 2 | 24 | TEMPERATURE[degC], POA Irr[kW1m2], Pressure[mbar] | 0.813407 | 15.87 |
| CNN_LSTM | Expanded-Pearson | 3 | 48 | TEMPERATURE[degC], HUMIDITY[%], GHI[kW1m2], Pac, Pdc, TempModule, TempCell | 0.811113 | 19.62 |
| CNN_LSTM | Expanded-Physics | 12 | 48 | HUMIDITY[%] | 0.834170 | 28.21 |
| CNN_LSTM | Intrinsic | 11 | 48 | None | 0.834517 | 24.48 |

Total measured model-fitting time was 94.58 minutes on CPU;
these times exclude preprocessing, checkpoint persistence and reporting.
The smaller physical feature space did not imply lower CNN_LSTM wall-clock cost:
its sampled configurations took longer to train than those of the larger spaces.

The overall **VAL-selected spaces remain XGBoost Intrinsic and CNN_LSTM
Expanded-Pearson**. They were not changed after TEST. XGBoost Original won
within its space using the initial configuration, so TPE did not improve every
space over its starting trial. Twelve attempts do not establish search convergence.

### TEST comparison

All entries are raw all-hour scores averaged over the ten individual horizons.

| Model | Space | MAE (kW) | RMSE (kW) | nRMSE (%) | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| XGBoost | Original | 0.384247 | 0.705127 | 9.478 | 0.809386 |
| XGBoost | Expanded-Pearson | 0.400313 | 0.727018 | 9.772 | 0.796832 |
| XGBoost | Expanded-Physics | 0.387190 | 0.713404 | 9.589 | 0.804806 |
| XGBoost | Intrinsic | 0.393561 | 0.726474 | 9.764 | 0.797114 |
| CNN_LSTM | Original | 0.428945 | 0.763137 | 10.257 | 0.777946 |
| CNN_LSTM | Expanded-Pearson | 0.430525 | 0.749693 | 10.077 | 0.782953 |
| CNN_LSTM | Expanded-Physics | 0.431375 | 0.813828 | 10.939 | 0.745493 |
| CNN_LSTM | Intrinsic | 0.421764 | 0.765808 | 10.293 | 0.776958 |
| Persistence | Persistence | 1.522309 | 2.061128 | 27.703 | -0.794566 |

### Physical-space comparisons

Differences below are Physics minus baseline mean MAE; negative favors Physics.

| Model | Baseline | MAE difference (kW) | 95% block-bootstrap interval (kW) |
| --- | --- | ---: | --- |
| XGBoost | Original | +0.002943 | [-0.011737, +0.018365] |
| XGBoost | Expanded-Pearson | -0.013124 | [-0.023345, -0.002217] |
| XGBoost | Intrinsic | -0.006371 | [-0.018188, +0.005459] |
| CNN_LSTM | Original | +0.002430 | [-0.026204, +0.029591] |
| CNN_LSTM | Expanded-Pearson | +0.000850 | [-0.035358, +0.035519] |
| CNN_LSTM | Intrinsic | +0.009612 | [-0.018210, +0.037861] |

XGBoost Expanded-Physics had lower TEST MAE/RMSE than Pearson and Intrinsic,
but higher errors than Original. Only its MAE contrast against Pearson had a
bootstrap interval excluding zero; this is conditional on these fitted models
and is not adjusted for multiple comparisons or search/seed variability.

CNN_LSTM Expanded-Physics had higher mean TEST RMSE than every alternative.
Its best subset contained **humidity only besides historical power**: no Pac,
TempCell, POA or HoursOfDay survived subset selection. It is therefore a winner
searched inside the physical space, not a fitted model using constructed physics
features. All its paired mean-MAE intervals included zero. Do not attribute its
scores to a successful or unsuccessful causal effect of Pac/TempCell inclusion.

The observed TEST RMSE minima were XGBoost Original and CNN_LSTM Pearson.
That descriptive ranking does not replace the pre-frozen VAL selection.
This run provides the requested SOLETE-style four-space result structure, with
mixed findings rather than a universal advantage for physics-guided screening.

The result is conditional on one seed, one chronological split, H=10, the local
data period, the original-space weather scope, fixed model families and this
bounded budget. TRAIN target correlation with Pdc was 0.999990; strong redundancy
is observable, but its acquisition-level origin and causal role are not established.

### Figures

[Horizon errors](../outputs/figures/20261004_140151_675271_physics_tpe/forecast_horizon_error.png),
[optimization progress](../outputs/figures/20261004_140151_675271_physics_tpe/optimization_progress.png),
[prediction curves](../outputs/figures/20261004_140151_675271_physics_tpe/prediction_curves.png).
The PNG exports were visually checked; matching editable SVG and vector PDF exports are available.
