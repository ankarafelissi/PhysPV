# PhysPV

**Physics-guided feature-space design for PV power forecasting.**

Does a physically relevant feature space improve forecasting after joint optimization?
PhysPV tests this on SOLETE with **XGBoost**, **CNN_LSTM** and a **Persistence** baseline.

[Full report (PDF)](report/main.pdf) 

## What this project does

- Constructs physical power, temperature and time-of-day features from five-minute data.
- Independently optimizes **Original**, **Expanded-Pearson**, **Expanded-Physics** and **Intrinsic** spaces with bounded Optuna TPE.
- Selects features, history length and model settings on TRAIN/VAL, then freezes winners before TEST.

## Key result

Physics-guided search reduced TEST RMSE versus each model's Original space:

| Model | Original RMSE | Physics RMSE | Reduction |
| --- | ---: | ---: | ---: |
| XGBoost | 1.0151 kW | **0.9536 kW** | **6.06%** |
| CNN_LSTM | 1.4615 kW | **1.2383 kW** | **15.27%** |

CNN_LSTM used a smaller diagnostic training/search budget, so these results do not support a direct model ranking. The report contains the methods, all eight figures, full metric tables and limitations.

## Run

Use Python 3.9 and place `SOLETE_Pombo_5min.h5` in `data/raw/`.
Settings: [experiment](config/config.yaml) · [plant](config/plant.yaml).

```bash
python -m pip install -r requirements.txt
python main.py prepare
python main.py train --resume "outputs/results/<study_id>/manifest.json"
python main.py evaluate --resume "outputs/results/<study_id>/manifest.json"
python main.py report --resume "outputs/results/<study_id>/manifest.json"
```

Use the study ID printed by `prepare` throughout. Tests: `python -m unittest discover -s tests`.

## License

Code and documentation: [MIT](LICENSE), © 2026 Felix Lan.
[SOLETE data](https://doi.org/10.11583/DTU.17040767): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
