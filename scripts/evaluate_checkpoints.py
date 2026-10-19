"""Evaluate released final-epoch weights on the official scene without retraining."""
import argparse
import json
from pathlib import Path
from _common import REPO, digest, save, summarize_records
import numpy as np
import torch
from protocol import Scene
from train import Classifier
from adaptive_reliability_experiments import cache, seed_everything
from matched_sacp_posthoc import outputs, quantile_higher
from block_conformal_multimodal import adaptive_probs, sets_and_metrics, block_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=REPO / 'data/muufl_spatial.npz')
    parser.add_argument('--output-root', type=Path, default=REPO / 'reproduced')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--splits', type=int, nargs='+', default=[20261031, 20261054, 20261059])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error('Prepare the official MUUFL arrays first; see data/README.md.')
    published = [json.loads(s) for s in (REPO / 'artifacts/paper_results/records.jsonl').read_text().splitlines()]
    records, differences = [], []
    device = torch.device(args.device)
    for expected in published:
        split, seed = expected['split_seed'], expected['seed']
        if split not in args.splits or seed not in args.seeds:
            continue
        name = f'muufl_split{split}_s{seed}_n0.40_fixed30'
        folder = args.output_root / 'runs/fixed_budget_calibration_v2' / name
        folder.mkdir(parents=True, exist_ok=False)
        checkpoint = REPO / 'artifacts/checkpoints' / name / 'final.pt'
        assert digest(checkpoint) == expected['checkpoint_sha256']
        seed_everything(seed)
        scene = Scene(args.data, 64, 11, split)
        model = Classifier(64, 11).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        assert state['epoch'] == 30
        model.load_state_dict(state['student'])
        model.eval()
        val = cache(scene, 'val', device)
        ov = outputs(model, val, device)
        yv, cv = val['y'].cpu().numpy(), val['coords']
        tv = scene.tile[tuple(cv.T)]
        assert len(np.unique(tv)) == 3
        vp = {'fused': ov[0], 'hsi': ov[1], 'lidar': ov[2], 'adaptive': adaptive_probs(ov)}
        thresholds = {}
        for head, probability in vp.items():
            score = 1 - probability[np.arange(len(yv)), yv]
            qtiles = [quantile_higher(score[tv == t], .9) for t in np.unique(tv)]
            thresholds[head] = {'global': quantile_higher(score, .9),
                                'blocked': quantile_higher(qtiles, .75), 'tile_quantiles': qtiles}
        save(folder / 'thresholds_frozen.json', {'test_metrics_started': False,
                                               'alpha': .1, 'beta': .75, 'thresholds': thresholds})
        del val
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        test = cache(scene, 'test', device)
        ot = outputs(model, test, device)
        yt, ct = test['y'].cpu().numpy(), test['coords']
        tt = scene.tile[tuple(ct.T)]
        tp = {'fused': ot[0], 'hsi': ot[1], 'lidar': ot[2], 'adaptive': adaptive_probs(ot)}
        predictors = {}
        for head in vp:
            pair = {}
            for kind, label in [('global', 'global_conformal'), ('blocked', 'blocked_conformal')]:
                q = thresholds[head][kind]
                metric = sets_and_metrics(tp[head], yt, q)
                metric.update(block_summary(tp[head], yt, tt, {int(t): q for t in np.unique(tt)}))
                pair[label] = metric
                for key, value in metric.items():
                    error = abs(value - expected['predictors'][head][label][key])
                    if error > 1e-10:
                        differences.append({'split': split, 'seed': seed, 'head': head,
                                            'rule': label, 'metric': key, 'absolute_error': error})
            predictors[head] = pair
        record = {'split_seed': split, 'seed': seed, 'epoch': 30, 'alpha': .1, 'beta': .75,
                  'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                  'calibration_blocks': len(np.unique(tv)), 'test_blocks': len(np.unique(tt)),
                  'calibration_pixels': len(yv), 'test_pixels': len(yt), 'predictors': predictors,
                  'probabilities': str(folder / 'probabilities.npz')}
        np.savez_compressed(folder / 'probabilities.npz', val_outputs=ov, test_outputs=ot,
                            val_labels=yv, test_labels=yt, val_tiles=tv, test_tiles=tt,
                            val_coords=cv, test_coords=ct)
        save(folder / 'result.json', record)
        records.append(record)
        del test, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(json.dumps({'completed': len(records), 'split': split, 'seed': seed}), flush=True)
    if not records:
        parser.error('No released checkpoint matched the requested splits and seeds.')
    base = args.output_root / 'results/fixed_budget_calibration_v2'
    summarize_records(records, base)
    report = {'runs': len(records), 'device': str(device), 'data_sha256': digest(args.data),
              'matches_recorded_metrics_at_1e_minus_10': not differences, 'differences': differences,
              'note': 'Different devices or numerical libraries can change floating-point thresholds.'}
    save(base / 'checkpoint_evaluation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
