# Research Protocol

## Question

The study tests whether physics-informed feature expansion and model-specific feature
selection improve short-term PV forecasting across XGBoost and CNN-LSTM. Persistence
is the deterministic reference. PI denotes additional physical inputs, not a new model
architecture or loss function.

## Experimental controls

Version 2.0 uses a bounded Optuna TPE search on the non-PI reference before feature
selection. Each family receives 12 attempts with identical seed and partitions;
the existing configuration is the first candidate. The objective is validation MAE
averaged over horizons in kW. Failed trials are retained and count toward the budget.
All physical feature arms subsequently use the winning parameters unchanged.
Tuning is a preparation step; the research question concerns physical features.

- Use one fixed seed: `11`.
- Preserve chronological TRAIN/VAL/TEST partitions.
- Fit scalers and scenario thresholds on TRAIN only.
- Train and early-stop with TRAIN/VAL only.
- Select hyperparameters and feature subsets with VAL only.
- Evaluate TEST only after the selected subset is frozen.
- Keep the reference features, forecast origins and targets identical across arms.
- Treat smoke runs as execution checks, never performance evidence.

The candidate pool contains single-feature additions, selected combinations and the
full physics set. XGBoost and CNN-LSTM may select different subsets. Selection never
requires matching cross-model trends.

## Metrics

Report MAE, RMSE, capacity-normalized RMSE and R2 for every forecast horizon. Horizon
`0` in summary tables is the arithmetic mean of the per-horizon scores. RMSE is
calculated separately at each horizon as `sqrt(mean(error^2))`.

Feature ablation reports selected PI minus non-PI MAE, so a negative value favors the
PI model. Paired circular block bootstrap intervals resample forecast origins and use
the configured 2,000 resamples and block length of 24. These intervals describe the
single fixed-seed run; they do not estimate seed-to-seed uncertainty.

## Diagnostics and interpretation

Forecast health checks reject non-finite output and constant prediction heads when the
target varies. Night-only constant-zero targets remain valid. Raw forecasts are the
primary evidence; constrained forecasts are supplementary.

Correlation and physical-redundancy tables are TRAIN-only diagnostics. Scenario labels
use TRAIN-fitted thresholds and are descriptive post-hoc analyses. Future ramp labels
never enter model inputs.

Positive, negative and mixed results must all be retained. A lower MAE with a higher
RMSE is a metric-dependent result, not a general improvement. A single seed supports
one reproducible comparison and must not be described as publication-level evidence
of stability.

Each study stores its frozen configuration, configured input path, runtime environment,
partition boundaries, selected features and run links in
`outputs/results/STUDY_ID/manifest.json`. Changing source code or input data requires a
new training study.

The manifest summary reports relative MAE/RMSE changes and paired MAE intervals.
MAE changes below 1% are labeled small effects; opposing MAE/RMSE directions are
metric tradeoffs; intervals crossing zero are uncertain. The threshold is descriptive,
not a selection criterion. No trend or practical improvement is guaranteed.
No per-run Markdown summary is created. Failed/interrupted attempts and retries remain
in the existing manifest with distinct readable run IDs. Source text and Git diff are
recorded without file digests or cryptographic checks.

## Initial v2.0 experiment (3 October 2026)

Study `20261003_005317_360657_features` completed from clean commit
`78580418f4ac9e20f6d9f1e3d98248b3e6ba3813` using
`python main.py --feature-study`. All 24 tuning attempts, 16 feature-screening fits
and four final predictions completed. Training used CPU and seed 11. The study's
frozen configuration and machine-readable diagnostics remain in its existing manifest.

The hourly dataset produced 7,644 TRAIN, 2,185 VAL and 1,088 TEST forecast origins,
with 25 historical observations and forecasts one to ten hours ahead. TEST targets
span 17 July to 1 September 2019. TPE reduced validation MAE by 1.70% for XGBoost
and 2.07% for CNN-LSTM relative to each family's initial configuration.

XGBoost selected depth 5, learning rate 0.01979, minimum child weight 6 and lambda
0.88370, with row/column sampling 0.82114/0.79227. The 700-tree ceiling and
40-round early stopping remained fixed. CNN-LSTM selected 16 convolution filters,
16 LSTM units, batch size 16 and Adam learning rate 0.00036457, with a 120-epoch
ceiling and patience 12. Exact settings are saved in `frozen_config.yaml`.

### Feature screening on VAL

Positive changes below mean higher MAE than the tuned non-PI reference.

| Feature addition | XGBoost MAE change | CNN-LSTM MAE change |
| --- | ---: | ---: |
| PAC | +0.39% | +2.88% |
| PDC | +0.35% | +3.06% |
| Tm | +1.58% | +1.61% |
| Tc | +0.53% | +1.70% |
| PAC + Tc | +0.69% | +0.82% |
| PDC + Tc | +0.51% | +0.72% |
| Full physics set | +1.70% | +9.91% |

Every tested PI arm was worse than its reference on VAL. PDC was the least harmful
PI candidate for XGBoost; PDC + Tc was the least harmful for CNN-LSTM. The PI-only
selection therefore does not imply that either candidate outperformed the baseline.

### Final TEST comparison

These are raw forecasts, with metrics averaged over the ten horizons. Normalized
RMSE uses the configured 7.44 kW DC nameplate, not a confirmed inverter AC limit.

| Model | Feature arm | MAE (kW) | RMSE (kW) | nRMSE (%) | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Persistence | Persistence | 1.5223 | 2.0611 | 27.7033 | -0.7946 |
| XGBoost | Non-PI reference | 0.3797 | 0.7069 | 9.5015 | 0.8083 |
| PI-XGBoost | PDC addition | 0.3788 | 0.7044 | 9.4684 | 0.8097 |
| CNN-LSTM | Non-PI reference | 0.4147 | 0.7630 | 10.2556 | 0.7767 |
| PI-CNN-LSTM | PDC + Tc addition | 0.4144 | 0.7669 | 10.3075 | 0.7750 |

The selected PI-XGBoost arm reduced MAE by 0.22% and RMSE by 0.35%; its paired
95% MAE-difference interval was [-0.00213, 0.00050] kW. PI-CNN-LSTM reduced MAE
by 0.08% but increased RMSE by 0.51%; its interval was [-0.00544, 0.00487] kW.
Both intervals include zero. The manifest's numerical `improvement` MAE direction
does not establish a reliable improvement: both effects are classified as small,
and CNN-LSTM has opposing MAE/RMSE directions.

TRAIN Pearson correlation was 0.99973 between PAC/PDC and 0.99912 between Tm/Tc.
This supports physical redundancy as a plausible explanation, rather than proving
it caused the observed degradation. Both model families already receive measured
irradiance, ambient temperature, wind speed and historical PV power. The derived
features re-express available information; they do not add future weather data.
All-at-once expansion was especially harmful to CNN-LSTM on VAL, while its selected
combination was less harmful than either corresponding single addition. Feature
effects therefore depend on the model and combination, even without a net benefit.

All 400 validation prediction heads passed collapse checks, all 20 CNN-LSTM best
checkpoints were verified, and all four final forecasts were finite with no all-zero
horizon. No validation-loss anomaly stop was recorded. Raw CNN-LSTM forecasts
included small negative values; the separately saved nonnegative forecasts reduced
baseline/PI MAE to 0.40618/0.40626 kW, respectively. The tiny PI advantage therefore
also changes direction under this supplementary constraint.

### Conclusion and limits

Under this dataset, split, seed and training budget, both ML families substantially
outperformed Persistence, and XGBoost had lower errors than CNN-LSTM. Physics-derived
feature expansion did not demonstrate a reliable additional benefit across the two
families. This experiment does not substantiate the benchmark's cross-model
improvement claim in this setting; it does not rule out gains under other conditions.

The run is exploratory and uses one seed. Fourteen of twenty CNN-LSTM fits reached
the 120-epoch ceiling, and the final reference/PI best checkpoints occurred at epochs
119/120. The result is conditional on this budget, not evidence of full convergence
or globally optimal hyperparameters. There was no runtime defect requiring a retry,
and TEST was not used to tune another run toward a positive feature result. A future
training-budget sensitivity study must be declared separately and preserve this run.

Local evidence: `outputs/results/20261003_005317_360657_features/` contains
`results.csv`, `model_comparison.csv`, `feature_ablation.csv`,
`validation_screening_summary.csv`, `tuning_results.csv`, `selection.json`,
`frozen_config.yaml` and `manifest.json`. The four plot families are available in
`outputs/figures/20261003_005317_360657_features/` as PNG, PDF and SVG. Generated
artifacts remain ignored by Git.

## Training-budget follow-up (3 October 2026)

Study `20261003_015328_145601_features` reused the initial study's tuned parameters,
seed, partitions and all eight feature arms. Only the CNN-LSTM epoch ceiling changed
from 120 to 240; patience remained 12. TPE was disabled. The prior TEST results had
already been observed, so this is an exploratory sensitivity study, not independent
confirmation. The initial study remains intact.

All 16 screening fits and four final predictions completed. The eight XGBoost VAL
scores reproduced exactly. All eight CNN-LSTM fits stopped normally before epoch
240, with no validation-loss anomaly stops. All 160 validation prediction heads
passed health checks, eight best checkpoints were verified, and final forecasts were
finite without all-zero horizons. Source layout and model code were unchanged.

### What changed with additional training

The CNN-LSTM reference stopped at epoch 208 and restored its epoch-196 checkpoint.
Validation MAE fell from 0.44441 to 0.43715 kW (-1.64%). TEST MAE fell from 0.41469
to 0.40691 kW (-1.88%), and RMSE fell from 0.76302 to 0.75042 kW (-1.65%). The
paired TEST MAE difference between budgets was -0.00778 kW, with a block-bootstrap
95% interval of [-0.01318, -0.00214] kW. This interval is conditional on the single
seed and observed TEST period. The initial epoch ceiling did restrict baseline
performance; extending it produced a measurable improvement in this comparison.

Every CNN-LSTM PI arm nevertheless remained worse than the extended reference on
VAL. PAC/PDC increased MAE by 4.24%/4.55%, Tm/Tc by 0.97%/0.92%, PAC + Tc/PDC + Tc
by 1.17%/1.30%, and the full set by 11.74%. The full-set fit stopped at epoch 39
in both studies and received no benefit from a higher ceiling. Its reported relative
degradation increased because the reference improved.

CNN-LSTM's selected PI arm changed from PDC + Tc to Tc, demonstrating that feature
ranking also depends on training budget. Tc restored its epoch-194 checkpoint after
stopping at epoch 206. XGBoost again selected PDC within the PI candidates. Neither
PI selection outperformed its reference on VAL.

### Final TEST results with the extended budget

| Model | Feature arm | MAE (kW) | RMSE (kW) | nRMSE (%) | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Persistence | Persistence | 1.5223 | 2.0611 | 27.7033 | -0.7946 |
| XGBoost | Non-PI reference | 0.3797 | 0.7069 | 9.5015 | 0.8083 |
| PI-XGBoost | PDC addition | 0.3788 | 0.7044 | 9.4684 | 0.8097 |
| CNN-LSTM | Non-PI reference | 0.4069 | 0.7504 | 10.0863 | 0.7835 |
| PI-CNN-LSTM | Tc addition | 0.4031 | 0.7641 | 10.2698 | 0.7761 |

The XGBoost PI comparison is unchanged: its -0.22% MAE change has an interval
including zero. CNN-LSTM's selected Tc arm reduced raw MAE by 0.94% but increased
RMSE by 1.82%; its paired MAE-difference interval was [-0.01363, 0.00567] kW.
It is a small, uncertain effect with a metric tradeoff. With the same supplementary
nonnegative constraint applied to both arms, CNN-LSTM baseline/PI MAE was
0.39950/0.40061 kW, reversing the small raw-MAE advantage.

### Working conclusion

The longer budget improved CNN-LSTM's baseline without establishing a reliable
physics-feature benefit. Across both budgets, every PI arm was worse on VAL, both
selected TEST MAE intervals included zero, and PI-CNN-LSTM increased TEST RMSE.
The appropriate conclusion is that physical feature expansion is not automatically
beneficial when its measured inputs already enter the model. Feature usefulness
depends on model class, combination, training budget and evaluation metric. High
TRAIN feature correlations support redundancy as a plausible explanation, not a
causal demonstration. These findings do not establish the benchmark's cross-model
improvement claim in this setting and do not refute its possibility elsewhere.

The tuned non-PI XGBoost is the strongest practical default supported by VAL and has
lower TEST errors than either CNN-LSTM arm. The extended CNN-LSTM budget is a better
comparison than the capped initial run. All CNN-LSTM fits now terminated through the
same early-stopping rule; no further TEST-guided retuning was performed. The one-seed,
one-site and one-TEST-period limits remain.

The new study directory contains its manifest, `frozen_config.yaml`, the standard
result tables and `training_budget_comparison.csv` (paired VAL scores for every arm
under both budgets). Its four figure families are in
`outputs/figures/20261003_015328_145601_features/`. To repeat this configuration as a
new run without another TPE search:

```bash
python main.py --feature-study --config outputs/results/20261003_015328_145601_features/frozen_config.yaml
```
