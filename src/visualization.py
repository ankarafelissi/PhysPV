"""Plots for the final one-seed model comparison."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {'Persistence': '#777777', 'XGBoost': '#37799b', 'CNN_LSTM': '#b1794d'}


def _save(figure, directory, name):
    figure.tight_layout()
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(directory / f'{name}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(figure)


def create_figures(state, comparison, ablation, screening, predictions, output_dir):
    """Export the four figures used to inspect the frozen TEST comparison."""
    output = Path(output_dir) / state['study_id']
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3), sharey=True)
    persistence = comparison.query('model == "Persistence" and horizon > 0')
    for axis, model in zip(axes, ('XGBoost', 'CNN_LSTM')):
        selected = state['selected'][model]
        for arm, label, style in (('nonpi_reference', 'Non-PI', '--'),
                                  (selected, 'Selected PI', '-')):
            values = comparison.query('model == @model and arm == @arm and horizon > 0')
            axis.plot(values.horizon, values.RMSE, style, color=COLORS[model], label=label)
        axis.plot(persistence.horizon, persistence.RMSE, color=COLORS['Persistence'], label='Persistence')
        axis.set(title=model, xlabel='Forecast horizon')
        axis.legend()
    axes[0].set_ylabel('RMSE (kW)')
    _save(figure, output, 'forecast_horizon_error')

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5), sharex=True)
    seed = state['config']['seed']
    for axis, model in zip(axes, ('XGBoost', 'CNN_LSTM')):
        selected = state['selected'][model]
        observed = None
        for arm, label, style in (('nonpi_reference', 'Non-PI', '--'),
                                  (selected, 'Selected PI', '-')):
            frame = predictions[model, arm, seed].query('horizon == 1').iloc[:48]
            time = frame['target_time']
            observed = frame
            axis.plot(time, frame.predicted_raw, style, color=COLORS[model], label=label)
        axis.plot(observed.target_time, observed.observed, color='#202020', label='Observed')
        axis.plot(observed.target_time, observed.persistence, color=COLORS['Persistence'], label='Persistence')
        axis.set(title=model, ylabel='Power (kW)')
        axis.legend(ncol=4)
    axes[-1].tick_params(axis='x', rotation=25)
    _save(figure, output, 'prediction_curves')

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3), sharey=True)
    for axis, model in zip(axes, ('XGBoost', 'CNN_LSTM')):
        selected = state['selected'][model]
        for arm, label, style in (('nonpi_reference', 'Non-PI', '--'),
                                  (selected, 'Selected PI', '-')):
            errors = np.sort(predictions[model, arm, seed].absolute_error.to_numpy())
            axis.plot(errors, np.linspace(0, 1, len(errors)), style,
                      color=COLORS[model], label=label)
        axis.set(title=model, xlabel='Absolute error (kW)')
        axis.legend()
    axes[0].set_ylabel('Empirical probability')
    _save(figure, output, 'error_distribution')

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True)
    for axis, model in zip(axes, ('XGBoost', 'CNN_LSTM')):
        values = screening.query('model == @model and arm != "nonpi_reference"')
        positions = np.arange(len(values))
        axis.barh(positions, values.delta_vs_reference, color=COLORS[model])
        axis.set(yticks=positions, yticklabels=values.arm, title=model,
                 xlabel='Validation MAE change (kW)')
        axis.axvline(0, color='#555555', linewidth=.7)
    _save(figure, output, 'feature_ablation')
