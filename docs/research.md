# Evaluation Protocol

## Question and comparison

Do physics-informed features consistently improve PV forecasting across model families?
Compare **Persistence, XGBoost, PI-XGBoost, CNN-LSTM, and PI-CNN-LSTM**.
PI denotes physical inputs, not a physics-informed loss. Transformers remain out of scope.

Use identical data, target, windows, splits, common features, preprocessing, equipment
parameters, and seed sets across arms. Only model family and the physics switch vary.
Tune on TRAIN/VAL with equal treatment; never select settings or seeds on TEST.

The default study uses seeds **11, 23, 32, 47, 59**. It adds the configured second split
when the initial test partition has fewer than 500 origins. Smoke runs are diagnostics,
not performance evidence.

## Scores and uncertainty

Report MAE and RMSE in kW, and `nRMSE_cap = 100 * RMSE / p_nom_kw` in percent, at every
horizon. Tables contain seed mean and sample standard deviation. Horizon means average
per-horizon scores, rather than pooling residuals.

Ablation uses **PI minus baseline**: negative favors physics. Absolute nRMSE differences
are percentage points; relative differences divide by the baseline mean.

Paired intervals average seed-wise absolute-error differences at each origin, then
use circular block bootstrap: 2,000 resamples by default, block length at least the
forecast horizon (default 24 origins). Intervals condition on the seed set, are
pointwise, and have no multiple-comparison correction. Read them alongside seed spread.

The automated positive conclusion requires improvement across all reported
model/horizon/split pairs, intervals below zero, improvement exceeding paired seed
spread, and sufficient effective blocks. Otherwise consistent improvement is not established.

## Scenarios

Thresholds are TRAIN medians, frozen for evaluation:

| Stratum | Statistic |
| --- | --- |
| Clear/cloudy proxy | Trailing-window POA variability |
| Low/high power | Trailing-window mean observed power |
| Steady/ramp | Absolute future-minus-origin power, per horizon; post-hoc only |

These reuse the same test errors without retraining. Sky labels are variability proxies,
not verified weather. Future ramp labels never enter model inputs. Retain empty groups
and flag groups below 30 origins by default as underpowered. Scenario differences are
descriptive; no scenario-specific significance tests are provided.

## Health and evidence

Training records per-epoch, per-horizon validation diagnostics and checks restored best
weights. Formal training/reporting rejects constant raw predictions when targets vary;
night-only constant targets are allowed. This detects output collapse, not all accuracy
problems. Raw and constrained forecasts are reported separately.

Reports verify complete arms, matching origins/configurations, source identities, file
hashes, and reproducible metrics. Preserve run IDs and original records; never hand-edit
scores or remove unfavorable seeds. A completed report does not imply physics helped.
After inspecting TEST for debugging, treat reruns as follow-up analysis, not independent
confirmation.

Read `main_table.csv`, `ablation_table.csv`, `scenario_table.csv`, and `conclusion.txt`
under `outputs/results/<study_id>/`. Figures show a predetermined test period, all
forecast horizons, and last-horizon scenarios. See [architecture.md](architecture.md)
for commands and report recovery.
