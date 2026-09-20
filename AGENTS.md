# AGENTS.md — Project Rules for PV Forecasting

Binding contract for anyone (human or agent) changing this repository. It is not a tutorial.
When a rule here conflicts with a request, a convenience, or a "quick experiment", this file
wins. If you believe a rule is wrong, propose the change instead of quietly working around it.

Explanation and protocol detail live in [docs/](docs/): [research.md](docs/research.md)
(evidence protocol), [data.md](docs/data.md) (input contract, features, physics switch),
[architecture.md](docs/architecture.md) (workflow, code map, files). This file states what
must be true.

---

## 1. The research question (frozen)

> **Do physics-informed features consistently improve PV power forecasting, and does that
> improvement transfer across model families (gradient-boosted trees and deep sequence models)?**

- The project is **not** "build a stronger deep learning model". Model accuracy is
  instrumental; a CNN-LSTM that beats XGBoost is a side observation.
- The word **consistently** is load-bearing (§5). A single-seed improvement is not evidence.
- **In scope:** the model ladder (§2), the physics ablation (§6), the scenario analysis (§7),
  and the metrics supporting them.
- **Out of scope:** Transformer models and other new architectures, attention mechanisms,
  hybrid variants, anything not traceable to the question above. Adding a model is not a
  neutral act — it changes the comparison and forces every ablation arm to be rebuilt.

---

## 2. The model ladder

| Rung | Model | Role | Where |
| --- | --- | --- | --- |
| 0 | **Persistence** | Naive floor: repeats the target observed at origin `t` | `part['persistence']` in [src/data.py](src/data.py), scored in [src/metrics.py](src/metrics.py) |
| 1a | **XGBoost** | Tree baseline, physics off | [src/models/xgboost.py](src/models/xgboost.py) |
| 1b | **PI-XGBoost** | Rung 1a + physics features | `data.physics: true` (§6) |
| 2a | **CNN-LSTM** | Deep baseline, physics off | [src/models/cnn_lstm.py](src/models/cnn_lstm.py) |
| 2b | **PI-CNN-LSTM** | Rung 2a + physics features | `data.physics: true` (§6) |

The reported ladder is **Persistence → PI-XGBoost → PI-CNN-LSTM**. Every number names the rung
that produced it; the `run_id` encodes it.

---

## 3. Fair comparison

When models or ablation arms share a table, **only `model` and `data.physics` may differ**.
Identical across arms: `data.path`, `resolution`, `split`, `pre`, `horizon`, `target`, `scaler`,
`correct_power`, `max_rows`, `constraints`, `p_nom_kw`, `plant`, the seed set, and the forecast
origins actually evaluated.

- `data.correct_power` must be `false`; the observed target is never replaced or corrected.
- Enforcement is [src/config.py](src/config.py) `data_contract()` / `comparison_contract()` plus
  the mismatch check in [src/predict.py](src/predict.py). **Extend the contract whenever you add
  a knob that affects comparability** — a knob outside it is a silent hole.
- `python main.py --physics on|off` overrides the YAML switch for both families; the run still
  records the effective value.

---

## 4. Leakage invariants

These hold today. **Do not weaken them**, and treat a failing invariant as a stop-the-line bug.

1. Scalers are fitted on TRAIN only, then reused for VAL/TEST.
2. Every target window lies entirely inside one partition; partitions are disjoint in target
   timestamps.
3. Windows containing any non-finite input, target, or origin-baseline value are discarded.
   **Never interpolate, forward-fill, or back-fill.** `asfreq` inserts gaps as missing rows by
   design.
4. Feature construction is causal: `build_features` may only use information available at or
   before each timestamp.
5. Analysis labels obey the same rule (§7), except the deliberately post-hoc ramp class.
6. Both physics arms share one physical-input eligibility mask, so they are scored on identical
   origins.

Coverage: [tests/test_physics.py](tests/test_physics.py), [tests/test_pipeline.py](tests/test_pipeline.py).
Any change to windows, splits, scaling or features needs a test that fails loudly.

---

## 5. Metrics and the evidence bar

Report **MAE and RMSE in kW** (target `P_Solar[kW]`) and **`nRMSE_cap = 100 × RMSE / p_nom_kw`**,
at **every horizon**, for **every rung** — a horizon average alone hides the degradation curve.

`p_nom_kw` is the DC nameplate capacity from [config/plant.yaml](config/plant.yaml):
`18×2×165 + 6×2×125 = 7440 W` = **7.44 kW**. It is an explicit config value, checked against the
plant, and in the contract. **Never** normalize by the test-set mean or maximum.

An improvement claim requires all three:

1. **Multiple seeds** — at least 5, the *same set* (`study.seeds`) for every arm; report mean ±
   std. The CNN-LSTM is stochastic, XGBoost less so; both get the same treatment.
2. **Dispersion reported** — a ± column in the table. An improvement smaller than the seed spread
   is not an improvement; say so plainly.
3. **A paired test** — arms share forecast origins, so compare **per-origin** errors: circular
   block bootstrap of ΔMAE (≥1000 resamples, block ≥ horizon) or Diebold–Mariano. Paired intervals
   condition on the seed set and are pointwise, with no multiplicity correction.

Below `study.min_test_origins` (500) forecast origins in TEST, the runner adds
`study.fallback_split` and the report refuses a single-split conclusion. Report both splits: a
conclusion that reverses between them is a finding about the split, not about physics.

---

## 6. The physics ablation

The core result. Design it so it cannot be misread.

**Feature set, exactly:** `Pac`, `Pdc`, `TempModule`, `TempCell`, from `pv_power_features()` in
[src/features.py](src/features.py). `TempModule_RP` is unsupported. Everything else — `HoursOfDay`,
`MeanPrevH`, `StdPrevH`, the wind rolling statistics — is calendar or historical, **not physics**,
and is held constant across arms. Calling rolling history "physics-informed" inflates the result.

**The switch:** `data.physics: true | false`. The code — not the config author — injects the
columns, in the fixed order above, so the flattened XGBoost input is deterministic. Listing physics
columns in `data.features` is an error. The invariant is the point: physics columns present **iff**
`physics` is `true`, and `physics` is in the contract, so the ablation diff reduces to one boolean.

**The four runs**, reported as separate rows:

```
XGBoost (false)  vs  PI-XGBoost (true)   → Δ₁
CNN_LSTM (false) vs  PI-CNN-LSTM (true)  → Δ₂
```

Δ = **PI − baseline** in MAE, RMSE and nRMSE, in absolute units and as **% of the baseline**. Never
average Δ₁ and Δ₂ into one "physics helps by X%" number. Whether they differ in *magnitude* is a
result; in *sign*, a headline.

---

## 7. Scenario analysis

A stratification of the **same** test errors from §6: it retrains nothing and re-splits nothing.

| Stratum | Statistic | Split |
| --- | --- | --- |
| Sky condition | Variability of `POA Irr[kW1m2]` over the trailing window `[t-pre, t]` | clear vs cloudy |
| Power level | Mean observed `P_Solar[kW]` over the trailing window | high vs low |
| Ramp events | Realized power change across the horizon, per step | ramp vs steady |

- **Thresholds are fitted on TRAIN** (medians by default) and frozen for VAL/TEST — the scaler
  rule again. Hard-coded thresholds are acceptable only if justified from physics or literature,
  recorded in the config, with the source documented.
- Sky and power labels use trailing (origin-time) data: what an operator would know when the
  forecast is issued. Ramp class is the deliberate exception — it describes what actually happened,
  so it is legitimate for **post-hoc stratification** and becomes leakage the moment it reaches a
  model input.
- Per stratum, per arm, report n, MAE, RMSE, nRMSE and Δ versus the non-physics arm. Strata below
  `study.min_stratum_origins` are reported with their `n` and flagged `underpowered` — never
  quietly dropped. Scenario differences are descriptive; there are no per-scenario significance
  tests.

---

## 8. Provenance and reproducibility

- Every number traces to a `run_id`: `outputs/results/<run_id>.csv`, `<run_id>_metrics.csv` and
  `<run_id>_metadata.json` (model, arm, seed, contract, partitions, input hash, environment,
  scenario thresholds). A result you cannot name the run of is not a result.
- Never hand-edit a metrics CSV, a figure tolerance, or a table cell. Fix the run. The report
  re-verifies file hashes, contracts, partition identity and arm completeness, and rejects
  altered, mismatched or missing arms.
- Report the seed set actually used. Reporting the best seed is fabrication.
- Runs are comparable only when trained on the same input data, source and dependency versions;
  [src/predict.py](src/predict.py) refuses otherwise. Retrain rather than forcing a comparison.
- Smoke runs are execution diagnostics: the report refuses them as evidence, and no performance
  conclusion may be drawn from `SOLETE_short.h5`.
- Runs finished but reporting failed? `python -m src.study --recover outputs/results/STUDY_ID_manifest.json`
  rebuilds the tables without retraining.

---

## 9. Definition of done

Complete when all four artifacts exist and agree. [src/report.py](src/report.py) writes the tables
and conclusion under `outputs/results/<study_id>/` and the figures under `outputs/figures/`:

1. **`main_table.csv`** — rows: the five rungs of §2; columns: MAE, RMSE, `nRMSE_cap` with
   dispersion across seeds (§5), a `physics` on/off indicator, per-horizon rows plus a horizon mean
   (the mean of per-horizon scores, not pooled residuals).
2. **`ablation_table.csv`** — rows: Δ₁ and Δ₂; columns: ΔMAE, ΔRMSE, ΔnRMSE in absolute and %, with
   the paired 95% CI from §5. nRMSE deltas are percentage points; relative deltas divide by the
   baseline mean.
3. **Figures** — observed vs predicted over a predetermined test period with persistence for scale;
   error versus horizon for all rungs; error by scenario stratum.
4. **`conclusion.txt`** — one explicit sentence answering §1, negative results included. "Physics
   features did not consistently improve either model family" is a complete and publishable answer.
   Do not chase a positive result.

---

## 10. Working conventions

- **Config over code.** Hyperparameters, feature lists, `pre`, `horizon`, `split`, `p_nom_kw` and
  `constraints` live in YAML under [config/](config/). No magic numbers in model or metric code.
- **Style.** Match the existing source: English docstrings, a module-level docstring on every file,
  comments only where the reason is not obvious, `ValueError` with a specific message for contract
  violations. Read [src/data.py](src/data.py) and [src/features.py](src/features.py) first.
- **Structure.** One responsibility per module. Model code lives in [src/models/](src/models/) and
  registers in `SUPPORTED_MODELS`; do not add per-model branches to `train.py`.
- **Commands.**

  ```bash
  python main.py --model all --smoke                # fast end-to-end check
  python main.py --config config/full.yaml --model all
  python -m src.study --config config/full.yaml     # predeclared four-arm, multi-seed study
  python -m src.study --recover outputs/results/STUDY_ID_manifest.json
  python -m src.predict --model-dir outputs/models/RUN_ID
  python -m src.report --manifest outputs/results/STUDY_ID_manifest.json
  python -m unittest discover -s tests -v
  python -m compileall -q src main.py tests
  ```

  Run the smoke check and the test suite before claiming a change works. Report failures with their
  output; do not describe an unverified change as working.
- **Data.** Large raw HDF5 files are gitignored; the tracked `SOLETE_short.h5` (24 rows) validates
  execution only. `data/processed/` tables are inspection artifacts — always go through the
  raw-data pipeline.
- **Dependencies.** Pinned in [requirements.txt](requirements.txt) (Python 3.9, TensorFlow/Keras
  2.10, XGBoost 1.7.6). A new dependency needs a justification tied to §1; a clear-sky irradiance
  model, for example, would need one, and the trailing-window definition in §7 exists partly to
  avoid it.

---

## 11. Anti-patterns

Each of these invalidates results. If you catch one, stop and say so.

1. Tuning one arm and not the other, then reporting the comparison.
2. Selecting hyperparameters, thresholds, or seeds on TEST.
3. Changing `split`, `pre`, `horizon`, `target`, `scaler`, `constraints` or `p_nom_kw` between
   compared runs.
4. Interpolating or filling gaps to make a window valid.
5. Fitting any scaler or threshold on VAL or TEST.
6. Reporting the best seed, or a single seed, as the result.
7. Presenting aggregate improvement without dispersion or a paired test.
8. Calling rolling history or calendar features "physics-informed".
9. Adding an architecture, including a Transformer, without revisiting the scope decision in §1.
10. Framing the project as "a better deep learning model for PV forecasting".
11. Producing a figure or table cell that cannot be traced to a `run_id`.
12. Quietly dropping an underpowered stratum, or a negative result, because it is inconvenient.

---

## 12. Current status

Everything in §2–§9 is implemented and covered by [tests/](tests/). A complete study has run on
[config/full.yaml](config/full.yaml) — 5 seeds, all four arms, `pre=24`, `horizon=10`, 1088 test
origins, single `[0.7, 0.2, 0.1]` split — under `outputs/results/`. As of 2026-09-20 its automated
conclusion is **negative**: a consistent MAE improvement from physics features was not established
across both model families. That is the standing result until a new `study_id` supersedes it — cite
the `study_id` and read `conclusion.txt` rather than paraphrasing it, and do not treat a rerun of
the same configuration as independent confirmation. Generated outputs are gitignored, so a fresh
checkout must re-run the study.

---

## 13. Attribution

Attribution for the SOLETE dataset and the adapted physical-feature implementation is recorded in
[docs/data.md](docs/data.md). Keep it current when data or code is reused, and do not remove it.
