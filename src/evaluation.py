"""Create verified result tables from a completed one-seed experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import paired_bootstrap


def report(directory, state):
    config = state['config']
    raw_rows = []
    prediction_frames = {}
    persistence = None
    for record in state['final']:
        metadata_path = Path(record['prediction_metadata'])
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metrics_path = metadata_path.with_name(metadata['run_id'] + '_metrics.csv')
        prediction_path = metadata_path.with_name(metadata['run_id'] + '.csv')
        metrics = pd.read_csv(metrics_path)
        if persistence is None:
            persistence = metrics[(metrics['variant'] == 'raw') & (metrics['model'] == 'Persistence')].copy()
            persistence['model'], persistence['arm'], persistence['seed'] = 'Persistence', 'persistence', -1
            persistence['features'], persistence['hyperparameters'] = '[]', '{}'
            persistence['training_seconds'], persistence['prediction_seconds'] = 0.0, 0.0
        metrics = metrics[(metrics['variant'] == 'raw') & (metrics['model'] != 'Persistence')].copy()
        metrics['model'], metrics['arm'], metrics['seed'] = record['model'], record['arm'], record['seed']
        metrics['features'] = json.dumps(record['subset'])
        metrics['hyperparameters'] = json.dumps(config['models'][record['model']], sort_keys=True)
        metrics['training_seconds'] = record['training_seconds']
        metrics['prediction_seconds'] = metadata['prediction_seconds']
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
    results.to_csv(directory / 'results.csv', index=False)
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
    screen.to_csv(directory / 'validation_screening.csv', index=False)
    screen_summary = screen[['model', 'arm', 'validation_mae']].copy()
    references = screen_summary[screen_summary.arm == 'nonpi_reference'].set_index('model')['validation_mae']
    screen_summary['delta_vs_reference'] = [row['validation_mae'] - references[row['model']]
                                            for _, row in screen_summary.iterrows()]
    screen_summary['selected'] = [state['selected'].get(row['model']) == row['arm']
                                  for _, row in screen_summary.iterrows()]
    screen_summary.to_csv(directory / 'validation_screening_summary.csv', index=False)

    from .paths import project_path
    from .visualization import create_figures
    figure_dir = project_path(config['output_dir']) / 'figures'
    create_figures(state, comparison, ablation, screen_summary, prediction_frames, figure_dir)

    lines = [f"# Staged physics-aware study: {state['study_id']}", '',
             'Hyperparameters were fixed before feature screening. Physics candidates were formed only by adding derived features to the same strong non-PI reference.', '',
             f"Fixed seed: {config['seed']}. This run does not estimate seed-to-seed variability.", '',
             '| Model | Selected PI arm | Reference MAE | Selected MAE | Delta MAE | Paired 95% CI |',
             '|---|---|---:|---:|---:|---:|']
    signs = []
    for model, selected in state['selected'].items():
        base = comparison.query('model == @model and arm == "nonpi_reference" and horizon == 0').iloc[0]
        pi = comparison.query('model == @model and arm == @selected and horizon == 0').iloc[0]
        delta = ablation.query('model == @model and horizon == 0').iloc[0]
        signs.append(np.sign(delta.delta_mae))
        lines.append(f'| {model} | {selected} | {base.MAE:.6f} | '
                     f'{pi.MAE:.6f} | {delta.delta_mae:+.6f} | '
                     f'[{delta.ci_low:+.6f}, {delta.ci_high:+.6f}] |')
    consistent = len(set(signs)) == 1
    direction = 'improvement' if consistent and signs[0] < 0 else 'degradation' if consistent else 'mixed'
    lines += ['', f'Cross-model TEST trend: **{direction}**.',
              'A matching sign is reported as an observation, not manufactured as a selection constraint. '
              'Feature selection used validation data only; TEST results did not alter hyperparameters or subsets.', '',
              'Negative and mixed results remain valid. Screening tables are validation diagnostics and are not final TEST evidence.']
    if config.get('smoke_study'):
        lines.insert(2, '**Execution diagnostic only: this smoke run is not research evidence.**')
    (directory / 'research_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    state = json.loads(manifest.read_text(encoding='utf-8'))
    if state['status'] not in ('evaluated', 'complete'):
        raise ValueError('The experiment has no complete TEST predictions to report.')
    report(manifest.parent, state)


if __name__ == '__main__':
    main()
