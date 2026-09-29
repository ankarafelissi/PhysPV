# Physics-Informed PV Forecasting

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

The feature study trains eight feature arms for each model. Every PI arm adds one or
more of `Pac`, `Pdc`, `TempModule`, and `TempCell` to the same EPOA/GHI/Hday reference.
Negative and mixed results remain valid.

## Project structure

```text
config/                 Experiment, plant and smoke configurations
src/
  data.py               Loading, chronological splits, windows and scaling
  features.py           Calendar, irradiance and physics-derived features
  models/               CNN-LSTM and XGBoost implementations
  train.py              TRAIN/VAL fitting and model persistence
  predict.py            Saved-model TEST inference
  experiments.py        Feature screening and validation-only selection
  evaluation.py         Verified comparison tables and research summary
  visualization.py      Figures generated from verified results
  metrics.py            Metrics, constraints, health checks and bootstrap
  provenance.py         Input-path, partition and runtime metadata
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
`research_summary.md`. MAE and RMSE use kW; `nRMSE_cap` uses the configured 7.44 kW
DC nameplate. Horizon `0` means the average of per-horizon metrics.

See [data and feature documentation](docs/data.md) for the input contract and source
attribution. See the [research protocol](docs/research.md) for selection, evaluation
and interpretation rules.

## Attribution

SOLETE data and the adapted physical-feature implementation originate from Daniel
Vazquez Pombo and collaborators. Full dataset and license details are preserved in
[docs/data.md](docs/data.md).
