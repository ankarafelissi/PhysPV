# Project rules

## Research scope
PhysPV studies which physics-informed features improve PV-power forecasting and whether their usefulness differs between XGBoost and CNN-LSTM.

Use bounded Bayesian/TPE optimization to obtain reasonable configurations; tuning is
not a research objective. Freeze parameters before individual and combination feature
experiments. Negative and mixed results are valid. Never optimize toward a desired
conclusion or add model families. Use one default seed; extra seeds are only used for
final selected experiments when explicitly requested.

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
