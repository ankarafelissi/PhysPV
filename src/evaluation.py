"""Create verified result tables from a completed one-seed experiment."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from .metrics import paired_bootstrap


COLORS = {'Persistence': '#777777', 'XGBoost': '#37799b', 'CNN_LSTM': '#b1794d'}


def _save(figure, directory, name):
    figure.tight_layout()
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(directory / f'{name}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(figure)


def create_figures(state, comparison, screening, predictions, output_dir):
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


def classify_result(mae_change_percent, rmse_change_percent, ci_low, ci_high):
    """Predeclared descriptive classes; never used to choose a model."""
    if abs(mae_change_percent) < 1.0:
        return 'small_effect'
    if mae_change_percent * rmse_change_percent < 0:
        return 'metric_tradeoff'
    if ci_low <= 0 <= ci_high:
        return 'uncertain'
    return 'improvement' if mae_change_percent < 0 else 'degradation'


def report(directory, state):
    config = state['config']
    raw_rows = []
    prediction_frames = {}
    persistence = None
    for record in state['final']:
        metadata_path = Path(record['prediction_metadata'])
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metrics_path = metadata_path.with_name(
            metadata.get('metrics_file', metadata['run_id'] + '_metrics.csv'))
        prediction_path = metadata_path.with_name(
            metadata.get('prediction_file', metadata['run_id'] + '.csv'))
        metrics = pd.read_csv(metrics_path)
        if persistence is None:
            persistence = metrics[(metrics['variant'] == 'raw') & (metrics['model'] == 'Persistence')].copy()
            persistence['model'], persistence['arm'], persistence['seed'] = 'Persistence', 'persistence', -1
        metrics = metrics[(metrics['variant'] == 'raw') & (metrics['model'] != 'Persistence')].copy()
        metrics['model'], metrics['arm'], metrics['seed'] = record['model'], record['arm'], record['seed']
        raw_rows.append(metrics)
        prediction_frames[record['model'], record['arm'], record['seed']] = pd.read_csv(prediction_path)
    results = pd.concat(raw_rows + [persistence], ignore_index=True)
    mean_rows = []
    for (model, arm, seed), group in results.groupby(['model', 'arm', 'seed']):
        row = {'model': model, 'arm': arm, 'seed': seed, 'horizon': 0, 'n': int(group['n'].min())}
        for metric in ('MAE', 'RMSE', 'nRMSE_cap', 'R2'):
            row[metric] = group[metric].mean()
        mean_rows.append(row)
    results = pd.concat([results, pd.DataFrame(mean_rows)], ignore_index=True, sort=False)
    comparison = results[['model', 'arm', 'horizon', 'MAE', 'RMSE', 'nRMSE_cap', 'R2', 'n']].copy()
    comparison.sort_values(['model', 'arm', 'horizon']).to_csv(directory / 'model_comparison.csv', index=False)

    ablations = []
    settings = config['study']
    for model, selected in state['selected'].items():
        for horizon in [0] + list(range(1, config['data']['horizon'] + 1)):
            seed = config['seed']
            base = prediction_frames[model, 'nonpi_reference', seed]
            pi = prediction_frames[model, selected, seed]
            if not base[['forecast_origin', 'target_time', 'horizon', 'observed']].equals(
                    pi[['forecast_origin', 'target_time', 'horizon', 'observed']]):
                raise ValueError('Final comparison predictions are not paired on identical targets.')
            if horizon:
                base, pi = base[base.horizon == horizon], pi[pi.horizon == horizon]
            delta = pi['absolute_error'].to_numpy() - base['absolute_error'].to_numpy()
            paired = pd.DataFrame({'origin': base['forecast_origin'], 'delta': delta}).groupby('origin').delta.mean().to_numpy()
            interval = paired_bootstrap(paired, settings['bootstrap_resamples'], settings['bootstrap_seed'],
                                        max(settings['bootstrap_block_length'], config['data']['horizon']))
            ablations.append({'model': model, 'selected_arm': selected, 'horizon': horizon, **interval})
    ablation = pd.DataFrame(ablations)
    ablation.to_csv(directory / 'feature_ablation.csv', index=False)
    screen = pd.DataFrame(state['screening'])
    screen_summary = screen[['model', 'arm', 'validation_mae']].copy()
    references = screen_summary[screen_summary.arm == 'nonpi_reference'].set_index('model')['validation_mae']
    screen_summary['delta_vs_reference'] = [row['validation_mae'] - references[row['model']]
                                            for _, row in screen_summary.iterrows()]
    screen_summary['selected'] = [state['selected'].get(row['model']) == row['arm']
                                  for _, row in screen_summary.iterrows()]
    screen_summary.to_csv(directory / 'validation_screening_summary.csv', index=False)

    from .runtime import project_path
    figure_dir = project_path(config['output_dir']) / 'figures'
    create_figures(state, comparison, screen_summary, prediction_frames, figure_dir)

    signs = []
    conclusions = []
    for model, selected in state['selected'].items():
        base = comparison.query('model == @model and arm == "nonpi_reference" and horizon == 0').iloc[0]
        pi = comparison.query('model == @model and arm == @selected and horizon == 0').iloc[0]
        delta = ablation.query('model == @model and horizon == 0').iloc[0]
        signs.append(np.sign(delta.delta_mae))
        mae_percent = 100 * (pi.MAE / base.MAE - 1) if base.MAE > 0 else np.nan
        rmse_percent = 100 * (pi.RMSE / base.RMSE - 1) if base.RMSE > 0 else np.nan
        conclusions.append({'model': model, 'selected_arm': selected,
                            'mae_change_percent': mae_percent, 'rmse_change_percent': rmse_percent,
                            'ci_low': float(delta.ci_low), 'ci_high': float(delta.ci_high),
                            'classification': classify_result(mae_percent, rmse_percent,
                                                              delta.ci_low, delta.ci_high)})
    consistent = len(set(signs)) == 1
    direction = 'improvement' if consistent and signs[0] < 0 else 'degradation' if consistent else 'mixed'
    # Machine-readable conclusions belong in the existing manifest, not a per-run Markdown report.
    state['summary'] = {'cross_model_mae_trend': direction, 'seed': config['seed'],
                        'smoke_diagnostic_only': bool(config.get('smoke_study')),
                        'models': conclusions, 'small_effect_threshold_percent': 1.0,
                        'interpretation': 'Single-seed exploratory comparison; inspect both MAE and RMSE.'}
    from .runtime import write_json
    write_json(directory / 'manifest.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    state = json.loads(manifest.read_text(encoding='utf-8'))
    if state['status'] not in ('evaluated', 'complete', 'completed'):
        raise ValueError('The experiment has no complete TEST predictions to report.')
    report(manifest.parent, state)


if __name__ == '__main__':
    main()
