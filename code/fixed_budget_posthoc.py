"""Matched post-training diagnostics for the fixed-epoch evidence only.

No selection from test outcomes. The complete grid and operator settings are
committed before threshold computation; all thresholds precede test metrics.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/fixed_budget_calibration_v2'
OUT = BASE / 'diagnostics'
sys.path.insert(0, str(BASE / 'source_snapshot'))
from protocol import digest
from matched_sacp_posthoc import neighbor_aggregate, quantile_higher

ALPHAS = [.025, .05, .075, .10, .125, .15, .175, .20, .25, .30]
METHODS = ['pointwise_global', 'bc_mps', 'sacp_global', 'sacp_tile_diagnostic']


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def csv_save(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(include, labels, tiles):
    correct = include[np.arange(len(labels)), labels]
    sizes = include.sum(1)
    single = sizes == 1
    return dict(coverage=float(correct.mean()),
                min_tile_coverage=float(min(correct[tiles == t].mean() for t in np.unique(tiles))),
                mean_set_size=float(sizes.mean()),
                singleton_fraction=float(single.mean()),
                empty_fraction=float((sizes == 0).mean()),
                singleton_accuracy=float(correct[single].mean()) if single.any() else 0.)


def aggregate(vp, tp, vc, tc, k=1, neighborhood='8'):
    coords = np.concatenate([vc, tc])
    grid = np.full((325, 220, 11), np.nan, dtype=np.float64)
    # Complements preserve the classifier output dtype, as in the prior control.
    grid[coords[:, 0], coords[:, 1]] = 1 - np.concatenate([vp, tp])
    scores = neighbor_aggregate(coords, grid, lam=.5, k=k, neighborhood=neighborhood)
    return scores[:len(vc)], scores[len(vc):]


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'summary.csv').exists():
        raise RuntimeError('Completed diagnostics exist; preserve them.')
    records = [json.loads(s) for s in (BASE / 'records.jsonl').read_text().splitlines()]
    assert len(records) == 9
    sources = []
    for r in records:
        folder = Path(r['checkpoint']).parent
        sources.append({'split_seed': r['split_seed'], 'seed': r['seed'],
                        'arrays': str((folder / 'probabilities.npz').relative_to(ROOT)),
                        'arrays_sha256': digest(folder / 'probabilities.npz'),
                        'checkpoint_sha256': digest(folder / 'final.pt')})
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'alpha_grid': ALPHAS, 'beta': .75, 'methods': METHODS,
                'primary_alpha': .1, 'lambda': .5, 'k': 1, 'neighborhood': '8',
                'sensitivity': [{'k': 1, 'neighborhood': '4'}, {'k': 2, 'neighborhood': '8'}],
                'sources': sources, 'script_sha256': digest(__file__),
                'purpose': 'complete post-training diagnostics; no operating-point selection'}
    save(OUT / 'protocol.json', protocol)
    thresholds, data, sensitivity = [], {}, []
    for r, spec in zip(records, sources):
        a = np.load(ROOT / spec['arrays'])
        vp, tp = a['val_outputs'][0], a['test_outputs'][0]
        yv, tv = a['val_labels'], a['val_tiles']
        sv, st = aggregate(vp, tp, a['val_coords'], a['test_coords'])
        key = (r['split_seed'], r['seed'])
        data[key] = (vp, tp, sv, st, a)
        for alpha in ALPHAS:
            for family, values in [('pointwise', 1-vp[np.arange(len(yv)), yv]),
                                   ('sacp', sv[np.arange(len(yv)), yv])]:
                qs = [quantile_higher(values[tv == t], 1-alpha) for t in np.unique(tv)]
                for kind, q in [('global', quantile_higher(values, 1-alpha)),
                                ('tile', quantile_higher(qs, .75))]:
                    name = {('pointwise', 'global'): 'pointwise_global',
                            ('pointwise', 'tile'): 'bc_mps',
                            ('sacp', 'global'): 'sacp_global',
                            ('sacp', 'tile'): 'sacp_tile_diagnostic'}[family, kind]
                    thresholds.append(dict(split_seed=key[0], seed=key[1], alpha=alpha,
                                           method=name, q=q, tile_quantiles=qs))
        for config in protocol['sensitivity']:
            sx, tx = aggregate(vp, tp, a['val_coords'], a['test_coords'], **config)
            q = quantile_higher(sx[np.arange(len(yv)), yv], .9)
            sensitivity.append((key, config, q, tx))
    save(OUT / 'thresholds_frozen.json',
         {'committed_utc': datetime.now(timezone.utc).isoformat(), 'test_metrics_started': False,
          'protocol_sha256': digest(OUT / 'protocol.json'), 'rows': thresholds,
          'sensitivity': [dict(split_seed=k[0], seed=k[1], **c, q=q) for k,c,q,_ in sensitivity]})
    rows = []
    for x in thresholds:
        key = (x['split_seed'], x['seed'])
        vp, tp, sv, st, a = data[key]
        take = tp >= 1-x['q'] if x['method'] in METHODS[:2] else st <= x['q']
        m = metric(take, a['test_labels'], a['test_tiles'])
        rows.append({k:v for k,v in x.items() if k != 'tile_quantiles'} | m)
        if x['alpha'] == .1 and x['method'] in METHODS[:2]:
            old = next(r for r in records if (r['split_seed'], r['seed']) == key)
            rule = 'global_conformal' if x['method'] == METHODS[0] else 'blocked_conformal'
            expected = old['predictors']['fused'][rule]
            for k,v in m.items():
                target = 'block_coverage_min' if k == 'min_tile_coverage' else k
                assert abs(v-expected[target]) < 1e-12, (key, k, v, expected[target])
    summary = []
    for alpha in ALPHAS:
        for method in METHODS:
            group = [x for x in rows if x['alpha'] == alpha and x['method'] == method]
            assert len(group) == 9
            s = dict(alpha=alpha, method=method, n=9)
            for k in ('coverage', 'min_tile_coverage', 'mean_set_size',
                      'singleton_fraction', 'empty_fraction', 'singleton_accuracy'):
                s[k] = float(np.mean([g[k] for g in group]))
                s[k+'_sample_sd'] = float(np.std([g[k] for g in group], ddof=1))
            summary.append(s)
    sr = []
    for key, config, q, tx in sensitivity:
        a = data[key][-1]
        sr.append(dict(split_seed=key[0], seed=key[1], **config, q=q,
                       **metric(tx <= q, a['test_labels'], a['test_tiles'])))
    ss = []
    for config in protocol['sensitivity']:
        group = [x for x in sr if all(x[k] == v for k,v in config.items())]
        s = dict(**config, n=9)
        for k in ('coverage', 'min_tile_coverage', 'mean_set_size'):
            s[k] = float(np.mean([g[k] for g in group]))
            s[k+'_sample_sd'] = float(np.std([g[k] for g in group], ddof=1))
        ss.append(s)
    csv_save(OUT / 'run_metrics.csv', rows)
    csv_save(OUT / 'summary.csv', summary)
    csv_save(OUT / 'sensitivity_runs.csv', sr)
    csv_save(OUT / 'sensitivity_summary.csv', ss)
    save(OUT / 'provenance.json', {'rows': len(rows), 'new_training_runs': 0,
                                 'thresholds_sha256': digest(OUT / 'thresholds_frozen.json'),
                                 'protocol_sha256': digest(OUT / 'protocol.json')})
    print(json.dumps({'status': 'PASS', 'rows': len(rows), 'sensitivity_rows': len(sr)}))


if __name__ == '__main__':
    main()
