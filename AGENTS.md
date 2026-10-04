# Project rules

## Research scope
PhysPV studies physics-guided search-space design for PV-power forecasting.
The default workflow is raw SOLETE data -> construct Pac/Pdc/TempCell/TempModule/
MinutesOfDay/HoursOfDay -> predeclare physically relevant, nonredundant feature space -> jointly
optimize feature subset, PRE and model hyperparameters with bounded Optuna TPE.
Compare independently optimized Original, Expanded-Pearson, Expanded-Physics and
Intrinsic spaces for XGBoost and CNN_LSTM, with Persistence as a forecast baseline.

Physics defines the search space; TPE adapts model configurations inside it.
Do not tune only Non-PI then freeze parameters for individual PI additions.
Use equal predeclared attempt budgets and common forecast origins across spaces.
Fit Pearson selection and scaling on TRAIN only, select configurations by mean
per-horizon VAL RMSE, freeze every space winner, then evaluate TEST once.
Negative and mixed results are valid. Never optimize toward a desired conclusion
or add model families. A similar result structure to SOLETE does not imply matching
its numeric results or conclusions. Use one default seed; extra seeds require an
explicit request. Do not expand budgets or retune after seeing TEST.
The default reference is Energy Reports 2022 (10.1016/j.egyr.2022.05.006):
5-minute data, H=60 (5 hours), chronological 70/20/10, PRE up to 120 samples.
Use paper Table 1 equipment parameters and explicit target correction for the
available-power task. Preserve raw measurements and report raw-target sensitivity.
Primary VAL/TEST metrics omit geometric night at each target timestamp; keep the
continuous timeline and causal observed histories. Normalize errors by the stated
10 kW AC rating. Document unresolved reference details and timestamp-year mismatch;
never claim exact numerical reproduction. Keep the previous hourly study labelled
as a legacy adaptation, not the current default experiment.

## Working style
- Keep one clear default experiment path and simple YAML configuration.
- Prefer short, explicit functions over abstractions that hide data flow.
- Do not add hash-related code, SHA-256 identities, file digests, or cryptographic provenance checks.
- Run no training experiment unless the user explicitly asks for it.
- Unit tests and compilation checks are allowed after source changes.
- Minimize token use: read targeted files and line ranges, summarize command output,
  avoid repeated searches or explanations, and keep updates concise. Do not skip
  necessary implementation or verification to save tokens.

## Experiment hygiene
- Reuse canonical model, experiment and config names from existing code and YAML.
  Use descriptive names for new variants; avoid arbitrary aliases or duplicate configs.
- Assign each run a unique, readable run ID without hashing. Use that same ID to
  link logs, checkpoints, metrics and the effective config in the existing manifest.
  Record the command, seed, run type, Git commit reference and dirty-worktree status;
  preserve the relevant source diff when dirty so the commit is not the sole record.
- Freeze the effective config and source layout during a run or resume. Validate
  resume compatibility and create a new run ID for changed settings; never overwrite
  another run's outputs or silently associate them with a different config or commit.
- Record started, completed, failed or interrupted status in the existing manifest
  for every run. Keep failed attempts, their command, error reason and log paths;
  record retries as distinct attempts rather than erasing failure history.
- Reuse the existing manifest and output directories for machine-readable records.
  Update existing documentation in place. Do not create per-run, per-failure or
  duplicate Markdown reports, or add a new .md file unless explicitly requested
  or no existing document can reasonably hold the necessary content.
