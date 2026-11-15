# Running the project

## Environment and input

Use Python 3.9 and `python -m pip install -r requirements.txt`. The existing GPU
path uses XGBoost 1.7.6; CNN_LSTM retains its TensorFlow 2.10 interface. A compatible
NVIDIA driver/runtime is required for training; installed TensorFlow alone does
not establish that its GPU path works. No cloud or container deployment is provided.

Obtain the five-minute SOLETE HDF5 and place it in
`data/raw/SOLETE_Pombo_5min.h5`. The dataset is attributed in the README. Local
raw files, model binaries and outputs are excluded from Git.

## Three-stage workflow

```bash
python main.py prepare --config config/config.yaml
python main.py train --resume outputs/results/<new-study-id>/manifest.json --model XGBoost
python main.py evaluate --resume outputs/results/<new-study-id>/manifest.json
```

`prepare` restores the time grid, applies configured QC and target construction,
fits initial TRAIN feature screening and saves the prepared data. It does not train.
`train` binds the selected model and independently searches four feature spaces on
TRAIN/VAL with fixed attempted-trial budgets. Each best model and configuration is
saved. There are no extra benchmark refits. `evaluate` uses these frozen models
on TEST and writes consolidated horizon metrics and predictions. Completed space
predictions are reused rather than evaluated repeatedly.

Outputs are linked by a readable study/run ID. The manifest stores command, seed,
settings, source records, input metadata, status and errors. This bookkeeping avoids
misidentifying results; it is not an independent audit subsystem. Changed settings,
source or environment require a new study. Historical manifests are not relabeled.
Workflow version 8 rejects earlier manifests; restore frozen historical sources in
a separate checkout if historical execution is needed.

Use `--model CNN_LSTM` on the first training command for the retained alternative.
Changing models requires a new study. `prepare --smoke` uses the small diagnostic
configuration; it still requires data, and subsequent training still needs a GPU.
Neither path has been retrained during cleanup.

## EDA and checks

Open `eda/solete_eda.ipynb` with the project Python kernel. It loads the raw grid,
compares core QC outputs, and displays TRAIN distributions and correlations.
No separate EDA scripts or cleaned-data export pipeline remain. The measured-data
view is separate from the physics-corrected forecasting target.

```bash
python main.py --help
python -m unittest discover -s tests
```

## Historical case evidence

The tables/figures in `assets/case_study/` describe the completed four-fold study,
not the default run above. Their source JSON and static CSV values remain available.
Four-fold coordinators, bootstrap inference, independent audit/verification, figure
generation, cloud-cost generation and manuscript tooling are outside the active code.
They are not requirements for running this project.

The source/configuration/documentation layout before this simplification is stored
locally in `outputs/research_v2/archive/20261115_main_pipeline/`, with a manifest
and source diff. Earlier original tool layouts are in adjacent dated archives.
Original experiment manifests/models/predictions remain in place. This archive is
excluded from Git; a checkout alone does not contain every historical experiment
artifact. No fresh-machine reconstruction of all historical conclusions is claimed.
