"""Fixed-budget retraining and calibration without label-based checkpoint choice.

All allocated validation tiles are used for calibration (three nonempty tiles
for each MUUFL allocation). The classifier is frozen at epoch 30. Thresholds
are written before final test metrics. This is a new controlled within-scene
evaluation, not a recovery or replay of the historical trainer.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import gc
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fixed_budget_calibration_v2'
SNAPSHOT = OUT / 'source_snapshot'
sys.path.insert(0, str(SNAPSHOT))
import numpy as np
import torch
from protocol import Scene, digest
from train import Classifier
from adaptive_reliability_experiments import cache, seed_everything, noise_targets, gate_weights
from matched_sacp_posthoc import outputs, quantile_higher
from block_conformal_multimodal import adaptive_probs, sets_and_metrics, block_summary

SPLITS = (20261031, 20261054, 20261059)
SEEDS = (0, 1, 2)
EPOCHS, BATCH, NOISE = 30, 256, .40
ALPHA, BETA = .10, .75


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def run(split, seed, device):
    name = f'muufl_split{split}_s{seed}_n0.40_fixed30'
    folder = ROOT / 'runs/fixed_budget_calibration_v2' / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        return json.loads((folder / 'result.json').read_text())
    if (folder / 'history.jsonl').exists():
        raise RuntimeError(f'Preserved incomplete run: {folder}')
    scene = Scene(ROOT / 'data/muufl_spatial.npz', 64, 11, split)
    seed_everything(seed)
    train = cache(scene, 'train', device)
    model = Classifier(scene.h.shape[2], scene.classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    observed, changed = noise_targets(scene, train, seed, NOISE)
    ema = torch.full((train['n'],), float(np.log(scene.classes)), device=device)
    manifest = {'spec': {'split_seed': split, 'seed': seed, 'epochs': EPOCHS,
                        'noise': NOISE, 'batch': BATCH},
                'protocol': scene.manifest,
                'selection': 'final epoch 30; no validation-based checkpoint selection',
                'training_noise_count': int(changed.sum()),
                'trainer_sha256': digest(__file__),
                'source_hashes': {p.name: digest(p) for p in SNAPSHOT.glob('*.py')},
                'torch_version': torch.__version__, 'device': str(device)}
    write_json(folder / 'manifest.json', manifest)
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(train['n'], device=device)
        total = 0.
        diagnostics = []
        for st in range(0, train['n'], BATCH):
            ids = order[st:st+BATCH]
            y = observed[ids]
            pred = model(train['h'][ids], train['d'][ids])
            with torch.no_grad():
                if epoch > 5:
                    w, diag = gate_weights(ema[ids], pred[1].detach(), pred[2].detach(),
                                           y, train['b'][ids], 'full')
                else:
                    w = torch.ones(len(ids), device=device)
                    diag = {'weight_mean': 1., 'risk_mean': 0.}
            parts = []
            for coefficient, logits in zip((1., .25, .25), pred):
                probability = logits.softmax(1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-6)
                parts.append(coefficient * ((1 - probability.pow(.7)) / .7))
            loss = sum((w * part).sum() / w.sum().clamp_min(1e-6) for part in parts)
            with torch.no_grad():
                ce = torch.nn.functional.cross_entropy(pred[0], y, reduction='none')
                ema[ids] = .9 * ema[ids] + .1 * ce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            diagnostics.append(diag)
        history = {'epoch': epoch, 'training_loss': total / train['n'],
                   'gate_weight_mean': float(np.mean([d['weight_mean'] for d in diagnostics]))}
        with (folder / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(history) + '\n')
    torch.save({'student': model.state_dict(), 'epoch': EPOCHS}, folder / 'final.pt')
    np.savez_compressed(folder / 'training_masks.npz', changed=changed, observed=observed.cpu().numpy())
    del train, optimizer, observed, ema
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    val = cache(scene, 'val', device)
    ov = outputs(model, val, device)
    yv = val['y'].cpu().numpy()
    cv = val['coords']
    tv = scene.tile[tuple(cv.T)]
    assert len(np.unique(tv)) == 3, (split, np.unique(tv))
    vp = {'fused': ov[0], 'hsi': ov[1], 'lidar': ov[2], 'adaptive': adaptive_probs(ov)}
    thresholds = {}
    for name_head, probability in vp.items():
        score = 1 - probability[np.arange(len(yv)), yv]
        q_tiles = {int(t): quantile_higher(score[tv == t], 1-ALPHA) for t in np.unique(tv)}
        thresholds[name_head] = {'global': quantile_higher(score, 1-ALPHA),
                                 'blocked': quantile_higher(list(q_tiles.values()), BETA),
                                 'tile_quantiles': q_tiles}
    write_json(folder / 'thresholds_frozen.json', {'committed_utc': datetime.now(timezone.utc).isoformat(),
                                                'alpha': ALPHA, 'beta': BETA,
                                                'test_metrics_started': False, 'thresholds': thresholds})
    # Test labels are first passed to a metric after the above durable commit.
    test = cache(scene, 'test', device)
    ot = outputs(model, test, device)
    yt = test['y'].cpu().numpy()
    ct = test['coords']
    tt = scene.tile[tuple(ct.T)]
    tp = {'fused': ot[0], 'hsi': ot[1], 'lidar': ot[2], 'adaptive': adaptive_probs(ot)}
    predictors = {}
    for name_head in vp:
        qg, qb = thresholds[name_head]['global'], thresholds[name_head]['blocked']
        g = sets_and_metrics(tp[name_head], yt, qg)
        g.update(block_summary(tp[name_head], yt, tt, {int(t): qg for t in np.unique(tt)}))
        b = sets_and_metrics(tp[name_head], yt, qb)
        b.update(block_summary(tp[name_head], yt, tt, {int(t): qb for t in np.unique(tt)}))
        predictors[name_head] = {'global_conformal': g, 'blocked_conformal': b,
                                'thresholds': {'global': qg, 'block_q75': qb}}
    result = {'split_seed': split, 'seed': seed, 'checkpoint': str(folder / 'final.pt'),
              'epoch': EPOCHS, 'alpha': ALPHA, 'beta': BETA,
              'calibration_pixels': len(yv), 'calibration_blocks': len(np.unique(tv)),
              'nonempty_calibration_tile_ids': np.unique(tv).tolist(),
              'test_pixels': len(yt), 'test_blocks': len(np.unique(tt)),
              'training_noise_count': int(changed.sum()), 'train_pixels': manifest['protocol']['class_counts']['train'],
              'selection': manifest['selection'], 'predictors': predictors,
              'threshold_file_sha256': digest(folder / 'thresholds_frozen.json'),
              'checkpoint_sha256': digest(folder / 'final.pt'),
              'trainer_sha256': digest(__file__), 'seconds': time.time()-start}
    np.savez_compressed(folder / 'probabilities.npz', val_outputs=ov, test_outputs=ot,
                        val_labels=yv, test_labels=yt, val_tiles=tv, test_tiles=tt,
                        val_coords=cv, test_coords=ct)
    write_json(folder / 'result.json', result)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = {'created_utc': datetime.now(timezone.utc).isoformat(), 'epochs': EPOCHS,
              'selection': 'fixed final epoch, no validation-based model choice',
              'splits': SPLITS, 'seeds': SEEDS, 'alpha': ALPHA, 'beta': BETA,
              'calibration': 'all allocated validation centers; assert three nonempty tiles',
              'purpose': 'controlled retraining; supersedes exploratory v1 mixed tile counts',
              'training_noise': NOISE, 'data_sha256': digest(ROOT / 'data/muufl_spatial.npz'),
              'source_hashes': {p.name: digest(p) for p in SNAPSHOT.glob('*.py')},
              'trainer_sha256': digest(__file__),
              'caveat': 'Within-scene, retrospectively motivated control; no new-scene theorem.'}
    if not (OUT / 'protocol.json').exists():
        write_json(OUT / 'protocol.json', config)
    else:
        saved = json.loads((OUT / 'protocol.json').read_text())
        assert saved['trainer_sha256'] == config['trainer_sha256']
        assert saved['source_hashes'] == config['source_hashes']
    device = torch.device('cuda')
    records = []
    for split in SPLITS:
        for seed in SEEDS:
            record = run(split, seed, device)
            records.append(record)
            print(json.dumps({'split': split, 'seed': seed, 'calibration_tiles': record['calibration_blocks'],
                              'global': record['predictors']['fused']['global_conformal']['coverage'],
                              'blocked': record['predictors']['fused']['blocked_conformal']['coverage']}), flush=True)
    (OUT / 'records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n', encoding='utf-8')
    summaries = {}
    rows = []
    for head in ('fused', 'adaptive', 'hsi', 'lidar'):
        for rule in ('global_conformal', 'blocked_conformal'):
            values = [r['predictors'][head][rule] for r in records]
            summaries[head + ':' + rule] = {k: {'mean': float(np.mean([v[k] for v in values])),
                                                'population_sd': float(np.std([v[k] for v in values])),
                                                'sample_sd': float(np.std([v[k] for v in values], ddof=1))}
                                            for k in values[0]}
            for i, record in enumerate(records, 1):
                rows.append({'run': i, 'split_seed': record['split_seed'], 'predictor': head,
                             'calibration': rule, **record['predictors'][head][rule]})
    write_json(OUT / 'summary.json', summaries)
    with (OUT / 'flat.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summaries['fused:blocked_conformal'], indent=2), flush=True)


if __name__ == '__main__':
    main()
