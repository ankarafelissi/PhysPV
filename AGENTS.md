# Project rules

## Research scope

This project studies physics guided search space design for PV power forecasting.
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
its numeric results or conclusions. Do not expand budgets or retune after seeing TEST.

## Working style

- Keep one clear default experiment path and simple YAML configuration.

- Prefer short, explicit functions over abstractions that hide data flow.

- Do not add hash-related code, SHA-256 identities, file digests, or cryptographic provenance checks.

- Run no training experiment unless the user explicitly asks for it.

- Unit tests and compilation checks are allowed after source changes.

- Minimize token use: read targeted files and line ranges, summarize command output,
  avoid repeated searches or explanations, and keep updates concise. 
  
- Do not skip necessary implementation or verification to save tokens.

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

## Coding Guidelines

Write Python code that is simple, readable, and easy to maintain.

### Principles

- Prefer the simplest solution that meets the requirements.

- Avoid overengineering and premature abstraction.

- Use clear, descriptive variable and function names.

- Keep functions focused on a single responsibility.

- Prefer straightforward control flow over clever shortcuts.

- Avoid unnecessary classes, wrappers, and helper functions.

- Do not introduce design patterns without a concrete need.

- Follow PEP 8 and existing project conventions.

### Code Structure

- Prefer functions over classes when no persistent state is needed.

- Separate concerns only when it improves readability.

- Avoid excessive file splitting and deeply nested abstractions.

- Use established libraries instead of reinventing functionality.

### Comments

- Explain non-obvious reasoning, not obvious syntax.

- Use concise docstrings for important functions.

- Avoid redundant comments and excessive documentation.

### Refactoring

- Preserve existing behavior and public interfaces.

- Make small, targeted changes.

- Remove redundant code without introducing new complexity.

- Run relevant tests after refactoring.
