"""Train the manuscript's fixed-30-epoch experiment using archived computations."""
import argparse
import json
from pathlib import Path
from _common import REPO, SNAPSHOT, digest, save, summarize_records
import torch
import fixed_budget_calibration as original
from protocol import Scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=REPO / 'data/muufl_spatial.npz')
    parser.add_argument('--output-root', type=Path, default=REPO / 'reproduced')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--splits', type=int, nargs='+', choices=original.SPLITS, default=list(original.SPLITS))
    parser.add_argument('--seeds', type=int, nargs='+', choices=original.SEEDS, default=list(original.SEEDS))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error('Prepare the official MUUFL arrays first; see data/README.md.')
    root = args.output_root.resolve()
    original.ROOT = root
    original.OUT = root / 'results/fixed_budget_calibration_v2'
    original.SNAPSHOT = SNAPSHOT
    original.Scene = lambda _path, block, patch, split: Scene(args.data, block, patch, split)
    config = {'epochs': 30, 'selection': 'fixed final epoch, no validation-based checkpoint choice',
              'splits': args.splits, 'seeds': args.seeds, 'alpha': .1, 'beta': .75,
              'training_noise': .4, 'device': args.device, 'data_sha256': digest(args.data),
              'source_hashes': {p.name: digest(p) for p in sorted(SNAPSHOT.glob('*.py'))},
              'portable_entry_point': 'scripts/run_experiment.py'}
    protocol_file = original.OUT / 'public_run_protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != config:
        raise ValueError('Output directory belongs to a different protocol; choose a new --output-root.')
    save(protocol_file, config)
    device = torch.device(args.device)
    records = []
    for split in args.splits:
        for seed in args.seeds:
            record = original.run(split, seed, device)
            records.append(record)
            print(json.dumps({'completed': len(records), 'split': split, 'seed': seed,
                              'coverage': record['predictors']['fused']['blocked_conformal']['coverage']}), flush=True)
    summary = summarize_records(records, original.OUT)
    print(json.dumps(summary['fused:blocked_conformal'], indent=2))


if __name__ == '__main__':
    main()
