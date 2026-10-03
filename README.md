# PhysPV 2.0: Physics-Informed PV Forecasting

This repository compares **Persistence, XGBoost, PI-XGBoost, CNN-LSTM, and
PI-CNN-LSTM** for short-term PV-power forecasting. PI models use additional
physics-derived input features; they do not use a physics-informed loss.

This repository compares CNN-LSTM with XGBoost.
A separate follow-up project will introduce a Transformer-based model for further comparison.

The study follows the experimental logic of
[Pombo et al. (2022)](https://doi.org/10.1016/j.egyr.2022.05.006) and the corrected
split and RMSE principles from SOLETE v3.0+. It tests the method without assuming
that physics-informed features must improve performance.

## Setup

Use Python 3.9 and the pinned dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Download `SOLETE_Pombo_60min.h5` from the
[SOLETE dataset](https://doi.org/10.11583/DTU.17040767) and place it in `data/raw/`.
The included `SOLETE_short.h5` is only for execution checks. Large data and generated
artifacts are excluded from Git.

## Run

```bash
# Fast pipeline check
python main.py --model all --smoke

# Complete feature study
python main.py --feature-study

# Resume an interrupted study
python -m src.experiments --resume outputs/results/STUDY_ID/manifest.json

# Evaluate one saved model
python -m src.predict --model-dir outputs/models/RUN_ID

# Rebuild verified tables and figures without training
python -m src.evaluation --manifest outputs/results/STUDY_ID/manifest.json
```

The default study uses seed `11`. Hyperparameters and physics-feature subsets are
selected from TRAIN/VAL only. TEST is used after selection is frozen. Scalers are fit
on TRAIN and reused for VAL and TEST. A single-seed result is one reproducible run,
not evidence of seed stability.

CNN-LSTM reports loss, validation loss, validation MAE and GPU availability every ten
epochs. It stops on repeated severe validation-loss anomalies and preserves the lowest
validation-loss weights under `best_checkpoint/` in the model directory.

The default path first runs 12 sequential Optuna TPE attempts per model, including
the configured starting parameters. Four successful trials form the startup phase;
later suggestions use TPE. The objective is mean per-horizon VAL MAE in kW on the
non-PI reference. No TEST scores enter tuning. Tree count, CNN epoch ceiling and
early-stopping settings remain fixed. Failed attempts consume the bounded budget.
Tuning is preparation, not a research objective.

The selected parameters are frozen in `frozen_config.yaml` inside the study directory,
then the feature study trains eight feature arms for each model. Every PI arm adds one or
more of `Pac`, `Pdc`, `TempModule`, and `TempCell` to the same EPOA/GHI/Hday reference.
Negative and mixed results remain valid.

The default budget is at most 24 tuning attempts plus 16 feature fits. Extra seeds
are only used for final selected experiments when explicitly requested. Change the
budget through `tuning` in `config/config.yaml`; disabling tuning reuses configured
model parameters. The default search spaces are intentionally bounded:

| Model | Tuned parameters |
| --- | --- |
| XGBoost | Depth 2-6, learning rate 0.01-0.1, min child weight 1-8, lambda 0.5-10, row/column sampling 0.7-1.0 |
| CNN-LSTM | Filters 16/32/64, LSTM units 16/32/64, batch 16/32, Adam learning rate 0.0002-0.002 |

Tuning can improve validation performance; it does not guarantee a larger physical
feature gain or better TEST generalization.

## Project structure

```text
config/                 Experiment, plant and smoke configurations
src/
  data.py               Physics features, loading, chronological windows and scaling
  models/               CNN-LSTM and XGBoost implementations
  train.py              TRAIN/VAL fitting and model persistence
  predict.py            Saved-model TEST inference
  experiments.py        Feature screening and validation-only selection
  tuning.py             Bounded TPE search before parameter freezing
  evaluation.py         Comparison tables, manifest summary and figures
  metrics.py            Metrics, constraints, health checks and bootstrap
  runtime.py            Runtime setup, project paths and JSON records
data/raw/               Local source data
data/processed/         Regenerated feature tables
outputs/models/         Saved models and scalers
outputs/results/        Predictions, metrics and study manifests
outputs/figures/        Training and comparison figures
tests/                  Leakage, physics, metrics and prediction checks
docs/                   Data contract and research protocol
```

The main study directory contains `manifest.json`, `selection.json`, `results.csv`,
`model_comparison.csv`, `feature_ablation.csv`, correlation tables and
the machine-readable `summary` in `manifest.json`. MAE and RMSE use kW; `nRMSE_cap` uses the configured 7.44 kW
DC nameplate. Horizon `0` means the average of per-horizon metrics.

`tuning_results.csv` lists all tuning attempts and the selected trial for each model;
`frozen_config.yaml` stores the parameters used by the subsequent feature experiments.

The summary labels small MAE effects (below 1%), metric tradeoffs, uncertain effects,
improvement and degradation. This descriptive threshold is fixed before training and
never affects parameter or feature selection. Conclusions remain single-seed results.
The manifest preserves commands, effective configurations, Git state, source text/diff,
checkpoints and failed/interrupted attempts. Resume rejects changed source or settings;
older studies should be read as archived results, not resumed through v2.0.

See [data and feature documentation](docs/data.md) for the input contract and source
attribution. See the [research protocol](docs/research.md) for selection, evaluation
and interpretation rules.

## Attribution

SOLETE data and the adapted physical-feature implementation originate from Daniel
Vazquez Pombo and collaborators. Full dataset and license details are preserved in
[docs/data.md](docs/data.md).
