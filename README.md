# PhysPV: five-minute physics-guided TPE

Physics defines a compact feature space; TPE jointly chooses feature subset,
previous samples (PRE) and model configuration. Compare Original, Expanded-Pearson,
Expanded-Physics and Intrinsic for XGBoost and CNN_LSTM, with Persistence.

The current protocol follows the five-minute, 60-step (five-hour) experiment in
[Pombo et al., Energy Reports 2022](https://doi.org/10.1016/j.egyr.2022.05.006).
It replaces grid/coarse-to-fine search with TPE and retains the project's two
model families. It is a documented adaptation, not an exact numerical reproduction.

| Space | Optional inputs |
| --- | --- |
| Original | Temperature, humidity, wind speed/direction, GHI, POA |
| Expanded-Pearson | Absolute TRAIN correlation >=0.7; humidity explicitly retained |
| Expanded-Physics | Humidity, POA, Pac, TempCell, HoursOfDay |
| Intrinsic | None; historical target power only |

Historical target power is mandatory. Expanded candidates also include Pdc,
TempModule and MinutesOfDay. PRE ranges from 0 to 120 five-minute samples using
the union of the reference coarse/fine grids. PRE=30 with all pool features is
the first trial. Seed 11, 40 attempts per model/space, ten startup attempts:
320 fits total, fixed before execution. Failures remain recorded and consume budget.
The objective is mean per-horizon daylight VAL RMSE. All eight winners freeze
before TEST; no TEST-driven reselection or budget extension.

CNN_LSTM searches filters, convolution/pooling sizes, LSTM depth/width, dense
layers, batch and learning rate; ceiling 1000 epochs with early stopping. XGBoost
searches depth, learning rate, child weight, regularization and sampling, with
700-tree ceiling and early stopping. GPU hist is enabled for XGBoost; the local
TensorFlow environment currently uses CPU. See [the full protocol](docs/research.md)
for ranges, provenance, unresolved reference details and interpretation limits.

## Run

Use Python 3.9 and `pip install -r requirements.txt`. Put SOLETE_Pombo_5min.h5
in data/raw/. Local interpreter: C:/Users/felix/anaconda3/envs/pv2024/python.exe.

```bash
python -m unittest discover -s tests
python -m src.experiments --smoke --evaluate-test
python main.py                     # optimize; TEST closed
python main.py --optimize --evaluate-test
python -m src.experiments --resume outputs/results/STUDY_ID/manifest.json --evaluate-test
python -m src.evaluation --manifest outputs/results/STUDY_ID/manifest.json
```

`--feature-study` aliases `--optimize`. Standalone `--model` / `--physics` are
diagnostics, not optimized comparison arms. Smoke is an execution check only.

## Data and outputs

Chronological 70/20/10; continuous five-minute histories; common forecast origins
using maximum PRE; TRAIN-only scalers/Pearson. Primary VAL/TEST scores omit
geometric night per target timestamp and normalize by 10 kW AC rating. Source HDF5
measurements are preserved; the available-power target uses explicit 20% Pac
correction. Predictions also retain raw observed targets and raw-target metrics.
See [the data contract](docs/data.md). Corrected targets are partly physics-derived;
results on these targets do not alone prove better measured-power forecasting.

Existing outputs/results/STUDY_ID/ holds the manifest, selector/correlation tables,
tuning_results.csv, frozen_config.yaml, validation_screening_summary.csv,
model_comparison.csv, optimized_configurations.csv, physics_comparison.csv, and
per-attempt logs/history/predictions/metrics. Models and comparison figures live
under outputs/models/ and outputs/figures/. Resume rejects changed settings/source.

The previous 96-fit hourly result remains in [research](docs/research.md#legacy-hourly-experiment-4-october-2026)
as a legacy adaptation. It does not use this protocol. Current results are pending.
Source/license attribution is retained in [data](docs/data.md).

The five-minute study `20261004_163215_553398_physics_tpe` is running with the
fixed 320-fit budget. [Current manifest](outputs/results/20261004_163215_553398_physics_tpe/manifest.json).
It uses 91,951 / 26,265 / 13,103 common TRAIN/VAL/TEST origins. No final TEST
comparison exists until all eight configurations freeze and evaluation completes.
