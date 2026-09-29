# Project rules

## Research scope
Test whether physics-informed feature expansion and model-specific feature selection
improve observed PV-power forecasting across XGBoost and CNN-LSTM. Persistence is
the deterministic floor. Negative and mixed results are valid. Do not add model
architectures or optimize toward a desired conclusion.

The current workflow uses one fixed seed (`11`) selected from TRAIN/VAL diagnostics
only. Never choose the seed, features or hyperparameters from TEST performance.

## Working style
- Keep one clear default experiment path and simple YAML configuration.
- Prefer short, explicit functions over abstractions that hide data flow.
- Preserve chronological splits, train-only scaler fitting and VAL-only selection.
- Run no training experiment unless the user explicitly asks for it.
- Freeze the source layout before starting an experiment. Do not rename, move, add,
  delete, or reorganize code files while an experiment is running or being resumed.
- Unit tests and compilation checks are allowed after source changes.
- Report a single-seed result as one reproducible run, not as evidence of seed
  stability or a publication-level uncertainty estimate.
- Keep English code, comments and documentation. Preserve dataset attribution in
  [docs/data.md](docs/data.md).
