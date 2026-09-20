"""Assemble auditable study tables from saved predictions; never train models."""
import argparse
import json
import copy
from pathlib import Path

from .runtime import configure_runtime
if __name__ == '__main__':
    configure_runtime()

import numpy as np
import pandas as pd

from .config import arm_name, comparison_contract, data_contract, validate_study
from .metrics import paired_bootstrap, evaluate
from .provenance import file_hash
from .diagnostics import forecast_health, require_healthy_forecasts

METRICS = ('MAE', 'RMSE', 'nRMSE_cap')
ARMS = ('Persistence', 'XGBoost', 'PI-XGBoost', 'CNN_LSTM', 'PI-CNN_LSTM')
# Forecasts are serialized from float32 model output, so stored per-origin errors
# reproduce only to a few float32 ulps of the operands. See read_study().
FLOAT32_ULP_BOUND = 4
EPSILON32 = float(np.finfo(np.float32).eps)
STRATA = {'sky_condition': ('clear', 'cloudy'), 'power_level_class': ('low', 'high'),
          'ramp_class': ('steady', 'ramp')}


def read_study(manifest):
    """Reject incomplete arms, altered outputs, and incompatible experiments."""
    validate_study(manifest['config']['study'])
    # 'runs_complete' means every arm finished but the report was not assembled yet;
    # rebuilding from it is the supported recovery path and validates identically.
    if manifest['status'] not in ('runs_complete', 'complete'):
        raise ValueError('Only a manifest with completed runs can be reported.')
    seeds = manifest['seeds']
    if not manifest['smoke'] and seeds != manifest['config']['study']['seeds']:
        raise ValueError('Reported seeds differ from the predeclared seed set.')
    if len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate study seeds.')
    runs, seen, references = {}, set(), {}
    for entry in manifest['runs']:
        key = (entry['split_id'], entry['seed'], entry['model'], entry['physics'])
        if key in seen:
            raise ValueError('Duplicate study arm.')
        seen.add(key)
        path = Path(entry['metadata_path'])
        meta = json.loads(path.read_text(encoding='utf-8'))
        run_id = entry['run_id']
        if (meta['run_id'], meta['seed'], meta['model'], meta['data_contract']['physics']) != (
                run_id, entry['seed'], entry['model'], entry['physics']):
            raise ValueError('Run identity does not match the study manifest.')
        if not manifest['smoke'] and meta['smoke']:
            raise ValueError('Smoke output cannot be used as scientific evidence.')
        if meta['data_contract'] != data_contract(meta['config']['data']) or meta['seed'] != meta['config']['seed']:
            raise ValueError('Recorded preprocessing contract or seed disagrees with the saved configuration.')
        csv_path, metric_path = path.parent / f'{run_id}.csv', path.parent / f'{run_id}_metrics.csv'
        if file_hash(csv_path) != meta['csv_sha256'] or file_hash(metric_path) != meta['metrics_sha256']:
            raise ValueError('Prediction or metric file changed after evaluation.')
        records = pd.read_csv(csv_path)
        if records.duplicated(['forecast_origin', 'horizon']).any():
            raise ValueError('Duplicate forecast-origin/horizon records.')
        records = records.sort_values(['forecast_origin', 'horizon']).reset_index(drop=True)
        error = records['predicted_raw'] - records['observed']
        # The forecast head runs in float32, and the CSV stores those values with
        # fewer significant digits than a float64 round trip needs. Errors recomputed
        # here therefore differ from the stored ones by a few float32 ulps of the
        # operands. Bound the difference by that precision rather than float64's: a
        # default relative tolerance collapses where the error itself is near zero --
        # which is exactly where forecasts are accurate -- and rejects valid runs.
        operand_scale = np.abs(records['observed']) + np.abs(records['predicted_raw'])
        if not (np.allclose(records['absolute_error'], np.abs(error), rtol=0,
                            atol=FLOAT32_ULP_BOUND * EPSILON32 * (operand_scale + 1.0))
                and np.allclose(records['squared_error'], error**2, rtol=0,
                                atol=FLOAT32_ULP_BOUND * EPSILON32
                                * (operand_scale**2 + error**2 + 1.0))):
            raise ValueError('Stored per-origin errors disagree with raw predictions.')
        if not (records['run_id'] == run_id).all():
            raise ValueError('Prediction provenance does not match the run.')
        identity = {name: meta[name] for name in ('partitions', 'input', 'scenario_thresholds', 'environment')}
        identity['contract'] = comparison_contract(meta['config'])
        expected_split = manifest['splits'][entry['split_id']]
        if meta['config']['data']['split'] != expected_split:
            raise ValueError('Run split differs from the declared study split.')
        expected_config = copy.deepcopy(manifest['config'])
        expected_config['data']['split'] = expected_split
        if comparison_contract(meta['config']) != comparison_contract(expected_config):
            raise ValueError('Run configuration differs from the predeclared study configuration.')
        invariant_columns = ['forecast_origin', 'target_time', 'horizon', 'observed', 'persistence',
                             'sky_condition', 'power_level_class', 'ramp_class']
        if entry['split_id'] in references:
            reference_identity, reference_records = references[entry['split_id']]
            if reference_identity != identity:
                raise ValueError('Incompatible configuration, partitions, data, code or thresholds between arms.')
            try:
                pd.testing.assert_frame_equal(reference_records, records[invariant_columns])
            except AssertionError as error:
                raise ValueError('Forecast origins, observations, or scenario labels differ between arms.') from error
        else:
            references[entry['split_id']] = (identity, records[invariant_columns])
        metrics = pd.read_csv(metric_path)
        if not (metrics['run_id'] == run_id).all():
            raise ValueError('Metric provenance does not match the run.')
        width = meta['data_contract']['horizon']
        values = {name: records.pivot(index='forecast_origin', columns='horizon', values=name)
                  for name in ('observed', 'predicted_raw', 'persistence')}
        if values['observed'].columns.tolist() != list(range(1, width+1)):
            raise ValueError('Incomplete forecast horizons in prediction records.')
        if not manifest['smoke']:
            require_healthy_forecasts(forecast_health(values['observed'].to_numpy(),
                values['predicted_raw'].to_numpy()), 'Test')
        recalculated = evaluate(*(values[name].to_numpy() for name in values),
                                meta['data_contract']['p_nom_kw'], meta['arm'])
        # These scores inherit the same float32 provenance as the stored errors, so they
        # reproduce only within that precision. The relative term covers scores on the
        # scale of the observations; the absolute term covers scores near zero, where a
        # relative bound alone is meaningless.
        metric_tolerance = FLOAT32_ULP_BOUND * EPSILON32 * max(
            float(np.max(np.abs(values['observed'].to_numpy()))), 1.0)
        stored_raw = metrics[metrics['variant'] == 'raw'].set_index(['model', 'horizon'])
        for row in recalculated.to_dict('records'):
            stored_row = stored_raw.loc[(row['model'], row['horizon'])]
            if not np.allclose([stored_row[m] for m in METRICS], [row[m] for m in METRICS],
                               rtol=FLOAT32_ULP_BOUND * EPSILON32, atol=metric_tolerance):
                raise ValueError('Metrics do not reproduce from the saved per-origin predictions.')
        runs[key] = (meta, records, metrics)
    expected = {(split, seed, model, physics) for split in range(len(manifest['splits']))
                for seed in seeds for model in ('XGBoost', 'CNN_LSTM') for physics in (False, True)}
    if seen != expected:
        raise ValueError('Study is incomplete: every split/seed requires all four ablation arms.')
    return runs


def score_values(observed, predicted, p_nom):
    if not len(observed):
        return {metric: np.nan for metric in METRICS}
    error = np.asarray(predicted) - np.asarray(observed)
    rmse = float(np.sqrt(np.mean(error**2)))
    return {'MAE': float(np.mean(np.abs(error))), 'RMSE': rmse, 'nRMSE_cap': 100 * rmse / p_nom}


def summarize(rows, keys):
    table = pd.DataFrame(rows)
    output = []
    for identity, group in table.groupby(keys, sort=False, dropna=False):
        if not isinstance(identity, tuple):
            identity = (identity,)
        row = dict(zip(keys, identity))
        row.update(n_origins=int(group['n_origins'].min()), n_seeds=len(group['seed'].unique()),
                   run_ids=json.dumps(sorted(group['run_id'].unique().tolist())))
        for metric in METRICS:
            row[metric + '_mean'] = group[metric].mean()
            row[metric + '_std'] = group[metric].std(ddof=1)
        output.append(row)
    return pd.DataFrame(output)


def assemble_report(manifest_path):
    """Create tables plus diagnostic figures from immutable per-run CSV files."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    runs = read_study(manifest)
    settings = manifest['config']['study']
    horizon = next(iter(runs.values()))[0]['data_contract']['horizon']
    block = max(horizon, settings['bootstrap_block_length'])
    report_dir = manifest_path.parent / manifest['study_id']
    report_dir.mkdir(exist_ok=True)
    main_rows, scenario_rows = [], []
    for (split, seed, model, physics), (meta, records, metrics) in runs.items():
        labels = [arm_name(model, physics)]
        if model == 'XGBoost' and not physics:
            labels.append('Persistence')
        for label in labels:
            raw = metrics[(metrics['model'] == label) & (metrics['variant'] == 'raw')]
            if raw['horizon'].tolist() != list(range(1, horizon+1)):
                raise ValueError('Incomplete per-horizon metrics.')
            base = dict(split_id=split, seed=seed, arm=label, physics=physics if label != 'Persistence' else False,
                        run_id=meta['run_id'], n_origins=meta['samples'])
            for row in raw.to_dict('records'):
                main_rows.append({**base, 'horizon': str(row['horizon']), **{m: row[m] for m in METRICS}})
            main_rows.append({**base, 'horizon': 'mean', **{m: raw[m].mean() for m in METRICS}})
            prediction = 'persistence' if label == 'Persistence' else 'predicted_raw'
            for h in range(1, horizon+1):
                step = records[records['horizon'] == h]
                for stratum, levels in STRATA.items():
                    for level in levels:
                        subset = step[step[stratum] == level]
                        scenario_rows.append({**base, 'horizon': h, 'stratum': stratum, 'level': level,
                            'n_origins': len(subset), **score_values(subset['observed'], subset[prediction],
                                                                  meta['data_contract']['p_nom_kw'])})
    main = summarize(main_rows, ['split_id', 'arm', 'physics', 'horizon'])
    main.to_csv(report_dir / 'main_table.csv', index=False)
    scenario = summarize(scenario_rows, ['split_id', 'arm', 'physics', 'horizon', 'stratum', 'level'])
    scenario['underpowered'] = scenario['n_origins'] < settings['min_stratum_origins']
    scenario['baseline_run_ids'] = ''
    for metric in METRICS:
        scenario['delta_' + metric] = np.nan
    for idx, row in scenario.iterrows():
        if row['arm'] == 'Persistence':
            continue
        baseline = row['arm'].replace('PI-', '')
        matched = scenario[(scenario['split_id'] == row['split_id']) & (scenario['arm'] == baseline)
                           & (scenario['horizon'] == row['horizon']) & (scenario['stratum'] == row['stratum'])
                           & (scenario['level'] == row['level'])].iloc[0]
        scenario.loc[idx, 'baseline_run_ids'] = matched['run_ids']
        for metric in METRICS:
            scenario.loc[idx, 'delta_' + metric] = row[metric + '_mean'] - matched[metric + '_mean']
    scenario.to_csv(report_dir / 'scenario_table.csv', index=False)
    ablation_rows = []
    for split in range(len(manifest['splits'])):
        for model in ('XGBoost', 'CNN_LSTM'):
            for h in [*range(1, horizon+1), 'mean']:
                deltas, per_seed, source_ids = [], [], []
                for seed in manifest['seeds']:
                    bmeta, base, _ = runs[(split, seed, model, False)]
                    pmeta, physical, _ = runs[(split, seed, model, True)]
                    source_ids.extend([bmeta['run_id'], pmeta['run_id']])
                    delta = physical['absolute_error'].to_numpy() - base['absolute_error'].to_numpy()
                    delta = delta.reshape(-1, horizon)
                    deltas.append(delta.mean(axis=1) if h == 'mean' else delta[:, h-1])
                    values = {}
                    for physics in (False, True):
                        label = arm_name(model, physics)
                        selected = [row for row in main_rows if row['split_id'] == split and row['seed'] == seed
                                    and row['arm'] == label and row['horizon'] == str(h)][0]
                        values[physics] = selected
                    per_seed.append({m: values[True][m] - values[False][m] for m in METRICS})
                ci = paired_bootstrap(np.mean(deltas, axis=0), settings['bootstrap_resamples'],
                                      settings['bootstrap_seed'], block)
                row = {'split_id': split, 'model': model, 'horizon': str(h), **ci,
                       'run_ids': json.dumps(source_ids), 'n_seeds': len(manifest['seeds'])}
                baseline_row = main[(main['split_id'] == split) & (main['arm'] == model)
                                    & (main['horizon'] == str(h))].iloc[0]
                for metric in METRICS:
                    delta_values = np.array([d[metric] for d in per_seed])
                    row['delta_' + metric] = delta_values.mean()
                    row['delta_' + metric + '_std'] = delta_values.std(ddof=1) if len(delta_values) > 1 else np.nan
                    denominator = baseline_row[metric + '_mean']
                    row['delta_' + metric + '_pct'] = 100 * delta_values.mean() / denominator if denominator != 0 else np.nan
                ablation_rows.append(row)
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(report_dir / 'ablation_table.csv', index=False)
    short = main['n_origins'].min() < settings['min_test_origins']
    if short and len(manifest['splits']) < 2 and not manifest['smoke']:
        raise ValueError('Small test partitions require a second split before a research conclusion.')
    if manifest['smoke']:
        conclusion = 'Execution diagnostics only: smoke runs cannot establish whether physics features improve forecasting.'
    elif ((ablation['ci_high'] < 0) & (ablation['delta_MAE'] < -ablation['delta_MAE_std'])
          & (ablation['effective_blocks'] >= settings['min_bootstrap_blocks'])).all():
        conclusion = ('Physics-informed features consistently reduced MAE across both model families in the '
                      'reported seeds, horizons and splits; the paired pointwise intervals exclude zero.')
    else:
        conclusion = ('A consistent MAE improvement from physics-informed features across both model families '
                      'was not established across all reported seeds, horizons and splits.')
    notes = {'conclusion': conclusion, 'seeds': manifest['seeds'], 'smoke': manifest['smoke'],
             'manifest': str(manifest_path.resolve()), 'bootstrap': settings,
             'notes': ['Primary tables use raw predictions; shared constrained results remain in per-run metrics.',
                       'Horizon mean is the arithmetic mean of per-horizon scores, not pooled RMSE.',
                       'nRMSE_cap and its absolute delta are percentages and percentage points respectively.',
                       'Relative delta is 100 * (mean PI - mean baseline) / mean baseline; undefined at zero.',
                       'Bootstrap pairs origins, averages seed differences first, and uses circular temporal blocks.',
                       'Intervals are pointwise; there is no multiplicity correction.',
                       'Sky labels are trailing-variability proxies, not observed meteorological truth.',
                       'Ramp labels are post-hoc only; sparse strata are retained with underpowered=true.']}
    (report_dir / 'report.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')
    (report_dir / 'conclusion.txt').write_text(conclusion + '\n', encoding='utf-8')
    plot_report(main, scenario, runs, manifest, report_dir)
    return report_dir


def plot_report(main, scenario, runs, manifest, report_dir):
    import matplotlib.pyplot as plt
    from .config import output_paths
    paths = output_paths(manifest['config'])
    plot_prefix = paths['figures'] / manifest['study_id']
    split_ids = list(range(len(manifest['splits'])))
    fig, axes = plt.subplots(1, len(split_ids), figsize=(7*len(split_ids), 4), squeeze=False)
    for split, ax in zip(split_ids, axes[0]):
        for arm in ARMS:
            data = main[(main['split_id'] == split) & (main['arm'] == arm) & (main['horizon'] != 'mean')].copy()
            data['horizon'] = data['horizon'].astype(int)
            data = data.sort_values('horizon')
            x, y = data['horizon'].to_numpy(), data['RMSE_mean'].to_numpy()
            spread = data['RMSE_std'].fillna(0).to_numpy()
            ax.plot(x, y, marker='.', label=arm)
            ax.fill_between(x, y-spread, y+spread, alpha=.12)
        ax.set(title=f'Split {split}: raw forecasts', xlabel='Horizon (steps)', ylabel='RMSE (kW)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(plot_prefix) + '_horizon.png', dpi=180)
    plt.close(fig)
    first_seed = manifest['seeds'][0]
    fig, ax = plt.subplots(figsize=(12, 4))
    for model in ('XGBoost', 'CNN_LSTM'):
        for physics in (False, True):
            _, records, _ = runs[(0, first_seed, model, physics)]
            series = records[records['horizon'] == 1].iloc[:manifest['config']['study']['representative_origins']]
            times = pd.to_datetime(series['target_time'])
            ax.plot(times, series['predicted_raw'], label=arm_name(model, physics))
            if model == 'XGBoost' and not physics:
                ax.plot(times, series['observed'], color='black', label='Observed')
                ax.plot(times, series['persistence'], linestyle='--', label='Persistence')
    ax.set(title=f'First test period, split 0, seed {first_seed}, horizon 1', ylabel='Power (kW)')
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(str(plot_prefix) + '_forecast.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(split_ids), 3, figsize=(15, 4*len(split_ids)), squeeze=False)
    last_h = int(scenario['horizon'].max())
    for split in split_ids:
        for column, (stratum, levels) in enumerate(STRATA.items()):
            ax = axes[split, column]
            for i, arm in enumerate(ARMS):
                data = scenario[(scenario['split_id'] == split) & (scenario['arm'] == arm)
                                & (scenario['horizon'] == last_h) & (scenario['stratum'] == stratum)].set_index('level')
                y = data.reindex(levels)['MAE_mean'].to_numpy()
                ax.bar(np.arange(2) + (i-2)*.15, y, width=.15, label=arm)
            ax.set(xticks=[0, 1], xticklabels=list(levels), ylabel='MAE (kW)',
                   title=f'Split {split}, h={last_h}: {stratum}')
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(str(plot_prefix) + '_scenarios.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Rebuild study tables from stored per-origin results')
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    print(assemble_report(args.manifest))


if __name__ == '__main__':
    main()
