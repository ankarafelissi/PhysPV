"""Four-space comparison of independently optimized frozen forecasters."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import SPACE_NAMES
from .metrics import paired_bootstrap
from .runtime import project_path, write_json


def classify_result(mae_change_percent, rmse_change_percent, ci_low, ci_high):
    """Predeclared descriptive classes; never used to choose a model."""
    if abs(mae_change_percent) < 1.0:
        return 'small_effect'
    if mae_change_percent * rmse_change_percent < 0:
        return 'metric_tradeoff'
    if ci_low <= 0 <= ci_high:
        return 'uncertain'
    return 'improvement' if mae_change_percent < 0 else 'degradation'


def screening_summary(state):
    rows = []
    for result in state['screening']:
        record = state['tuning'][result['model'] + '/' + result['arm']]
        trials = record['trials']
        rows.append({'model': result['model'], 'arm': result['arm'], 'pre': result['pre'],
                     'subset': ';'.join(result['subset']), 'n_features': len(result['subset']) + 1,
                     'validation_rmse': result['validation_rmse'], 'validation_mae': result['validation_mae'],
                     'best_trial': record['best_trial'], 'attempts': len(trials),
                     'successful_trials': sum(item['status'] == 'completed' for item in trials),
                     'total_training_seconds': sum(item.get('training_seconds') or 0 for item in trials),
                     'selected': state['selected'][result['model']] == result['arm'],
                     'training_run_id': result['training_run_id']})
    return pd.DataFrame(rows)


def _save(figure, directory, name):
    figure.tight_layout()
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(directory / f'{name}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(figure)


def create_figures(state, comparison, predictions, output):
    output.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(SPACE_NAMES, ('#777777', '#37799b', '#c26633', '#7d6594')))
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for axis, model in zip(axes, ('XGBoost', 'CNN_LSTM')):
        for space in SPACE_NAMES:
            group = comparison.query('model == @model and arm == @space and horizon > 0')
            hours = group.horizon * pd.Timedelta(state['config']['data']['resolution']).total_seconds() / 3600
            axis.plot(hours, group.RMSE, color=colors[space], label=space)
        persistence = comparison.query('model == "Persistence" and horizon > 0')
        hours = persistence.horizon * pd.Timedelta(state['config']['data']['resolution']).total_seconds() / 3600
        axis.plot(hours, persistence.RMSE, ':', color='black', label='Persistence')
        axis.set(title=model, xlabel='Forecast horizon (hours)')
        axis.legend(fontsize=7)
    axes[0].set_ylabel('TEST RMSE (kW)')
    _save(figure, output, 'forecast_horizon_error')
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for axis, model in zip(axes, ('XGBoost', 'CNN_LSTM')):
        for space in SPACE_NAMES:
            trials = state['tuning'][model + '/' + space]['trials']
            values = [item.get('validation_rmse', np.nan) for item in trials]
            axis.plot(np.arange(1, len(trials) + 1), pd.Series(values).cummin(),
                      color=colors[space], label=space)
        axis.set(title=model, xlabel='TPE attempt')
        axis.legend(fontsize=7)
    axes[0].set_ylabel('Best validation RMSE (kW)')
    _save(figure, output, 'optimization_progress')
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for axis, model in zip(axes, ('XGBoost', 'CNN_LSTM')):
        for space in SPACE_NAMES:
            frame = predictions[model, space].query('horizon == 1').iloc[:864]
            axis.plot(pd.to_datetime(frame.target_time), frame.predicted_raw, color=colors[space], label=space)
        axis.plot(pd.to_datetime(frame.target_time), frame.observed, color='black', label='Observed')
        axis.set(title=model, ylabel='Power (kW)')
        axis.legend(ncol=3, fontsize=7)
    axes[-1].tick_params(axis='x', rotation=25)
    _save(figure, output, 'prediction_curves')


def report(directory, state):
    config = state['config']
    rows, predictions = [], {}
    persistence = None
    for record in state['final']:
        metadata_path = Path(record['prediction_metadata'])
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metrics = pd.read_csv(metadata_path.with_name(metadata['metrics_file']))
        frame = pd.read_csv(metadata_path.with_name(metadata['prediction_file']))
        if persistence is None:
            persistence = metrics.query('variant == "raw" and model == "Persistence"').copy()
            persistence['arm'] = 'Persistence'
        metrics = metrics.query('variant == "raw" and model != "Persistence"').copy()
        metrics['model'], metrics['arm'] = record['model'], record['arm']
        rows.append(metrics)
        predictions[record['model'], record['arm']] = frame
    anchor = next(iter(predictions.values()))
    pairing = ['forecast_origin', 'target_time', 'horizon', 'observed']
    for frame in predictions.values():
        if not frame[pairing].equals(anchor[pairing]):
            raise ValueError('All optimized configurations must share identical TEST targets.')
    comparison = pd.concat([*rows, persistence], ignore_index=True)
    means = []
    for (model, arm), group in comparison.groupby(['model', 'arm']):
        means.append({'model': model, 'arm': arm, 'horizon': 0, 'n': int(group['n'].min()),
                      **{metric: group[metric].mean() for metric in ('MAE', 'RMSE', 'nRMSE_cap', 'R2')}})
    comparison = pd.concat([comparison, pd.DataFrame(means)], ignore_index=True)
    columns = ['model', 'arm', 'horizon', 'MAE', 'RMSE', 'nRMSE_cap', 'R2', 'n']
    comparison[columns].sort_values(['model', 'arm', 'horizon']).to_csv(directory / 'model_comparison.csv', index=False)
    screening = screening_summary(state)
    screening.to_csv(directory / 'validation_screening_summary.csv', index=False)
    optimized = comparison.query('horizon == 0 and model != "Persistence"')[columns].merge(
        screening, on=['model', 'arm'], validate='one_to_one')
    optimized.to_csv(directory / 'optimized_configurations.csv', index=False)
    settings = config['study']
    paired = []
    for model in ('XGBoost', 'CNN_LSTM'):
        physics = predictions[model, 'Expanded-Physics']
        for baseline in ('Original', 'Expanded-Pearson', 'Intrinsic'):
            base = predictions[model, baseline]
            for horizon in range(config['data']['horizon'] + 1):
                left, right = (physics.query('evaluated and horizon == @horizon'), base.query('evaluated and horizon == @horizon')) if horizon else (physics.query('evaluated'), base.query('evaluated'))
                delta = left.absolute_error.to_numpy() - right.absolute_error.to_numpy()
                origin_delta = pd.DataFrame({'origin': left.forecast_origin.to_numpy(), 'delta': delta}).groupby('origin').delta.mean().to_numpy()
                interval = paired_bootstrap(origin_delta, settings['bootstrap_resamples'], settings['bootstrap_seed'],
                                           max(settings['bootstrap_block_length'], config['data']['horizon']))
                paired.append({'model': model, 'baseline': baseline, 'horizon': horizon, **interval})
    paired = pd.DataFrame(paired)
    paired.to_csv(directory / 'physics_comparison.csv', index=False)
    create_figures(state, comparison, predictions, project_path(config['output_dir']) / 'figures' / state['study_id'])
    state['summary'] = {'seed': config['seed'], 'smoke_diagnostic_only': bool(config.get('smoke_study')),
                        'validation_selected_space': state['selected'],
                        'test_mean_horizon_metrics': comparison.query('horizon == 0')[columns].to_dict('records'),
                        'physics_vs_baselines': paired.query('horizon == 0').to_dict('records'),
                        'interpretation': 'Single-seed equal-attempt-budget joint optimization; effects combine feature space and model adaptation, not isolated feature causality. TEST never selects configurations.'}
    write_json(directory / 'manifest.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest = project_path(args.manifest)
    state = json.loads(manifest.read_text(encoding='utf-8'))
    if state.get('workflow_version') != 3 or state['status'] not in ('evaluated', 'completed'):
        raise ValueError('A joint-search study with complete TEST predictions is required.')
    report(manifest.parent, state)


if __name__ == '__main__':
    main()
