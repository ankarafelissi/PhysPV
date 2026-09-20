# Architecture

For setup and installation, start with the [README](../README.md).

## Workflow

```text
YAML -> data and features -> training -> saved model
                                      -> prediction -> metrics and health checks
                                                    -> study tables and figures
```

Training fits on TRAIN and selects weights using VAL. Prediction reloads the model
and scalers to evaluate TEST. The study runner repeats this workflow across seeds
and physics settings; reporting reads saved results without training.

## Entry points

Run from the repository root:

```bash
python main.py --model all --smoke
python -m src.study --config config/full.yaml
python -m src.predict --model-dir outputs/models/RUN_ID
python -m src.report --manifest outputs/results/STUDY_ID_manifest.json
```

If all runs finished but reporting failed, retry validated report recovery:

```bash
python -m src.study --recover outputs/results/STUDY_ID_manifest.json
```

Recovery does not resume unfinished training or bypass result checks.

## Code map

| Modules | Responsibility |
| --- | --- |
| `config.py`, `paths.py`, `runtime.py` | Configuration, paths, runtime setup |
| `data.py`, `features.py` | Causal features, windows, splits, scaling |
| `models/` | CNN-LSTM and XGBoost implementations |
| `train.py`, `predict.py` | Separate training and evaluation |
| `constraints.py`, `diagnostics.py` | Post-processing and output health |
| `metrics.py`, `scenarios.py`, `provenance.py` | Scores, strata, traceability |
| `study.py`, `report.py` | Experiment orchestration and aggregation |

## Files to use

- `config/config.yaml`: small execution example; `config/full.yaml`: full experiment.
- `config/plant.yaml`: equipment parameters; `config/smoke.yaml`: shared smoke overrides.
- `data/raw/`: source data; `data/processed/`: regenerated inspection tables.
- `outputs/models/<run_id>/`: model, scalers, frozen configuration, metadata.
- `outputs/results/`: run records and manifests; `<study_id>/` contains summary tables.
- `outputs/figures/`: run and study plots.

Each run also exports validation/test health CSVs. Keep referenced artifacts together;
manifest paths connect reports to their source runs. Large data and generated outputs
are gitignored. Source or dependency changes require retraining before new evaluation.

See [data.md](data.md) for preprocessing and [research.md](research.md) for interpretation.
