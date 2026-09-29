# Research Protocol

## Question

The study tests whether physics-informed feature expansion and model-specific feature
selection improve short-term PV forecasting across XGBoost and CNN-LSTM. Persistence
is the deterministic reference. PI denotes additional physical inputs, not a new model
architecture or loss function.

## Experimental controls

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
