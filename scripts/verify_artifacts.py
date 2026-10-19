"""Verify release checksums and reconstruct the paper summary without dependencies."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-file-manifest', action='store_true', help='For maintainers before manifest generation')
    args = parser.parse_args()
    verified = 0
    if not args.skip_file_manifest:
        files = json.loads((REPO / 'SHA256SUMS.json').read_text())
        for name, expected in files.items():
            path = REPO / name
            assert path.is_file(), f'Missing release file: {name}'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Checksum mismatch: {name}'
            verified += 1
    base = REPO / 'artifacts/paper_results'
    records = [json.loads(s) for s in (base / 'records.jsonl').read_text().splitlines()]
    assert len(records) == 9
    assert {(r['split_seed'], r['seed']) for r in records} == {
        (split, seed) for split in (20261031, 20261054, 20261059) for seed in (0, 1, 2)}
    assert all(r['epoch'] == 30 and r['calibration_blocks'] == 3 for r in records)
    for name, expected in json.loads((base / 'protocol.json').read_text())['source_hashes'].items():
        assert hashlib.sha256((REPO / 'code/original_snapshot' / name).read_bytes()).hexdigest() == expected
    saved_summary = json.loads((base / 'summary.json').read_text())
    for head in ('fused', 'adaptive', 'hsi', 'lidar'):
        for rule in ('global_conformal', 'blocked_conformal'):
            for metric, target in saved_summary[head + ':' + rule].items():
                values = [r['predictors'][head][rule][metric] for r in records]
                for key, actual in [('mean', statistics.mean(values)),
                                    ('population_sd', statistics.pstdev(values)),
                                    ('sample_sd', statistics.stdev(values))]:
                    assert abs(actual - target[key]) < 1e-12, (head, rule, metric, key)
    for r in records:
        name = f"muufl_split{r['split_seed']}_s{r['seed']}_n0.40_fixed30"
        checkpoint = REPO / 'artifacts/checkpoints' / name / 'final.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == r['checkpoint_sha256']
        history = (checkpoint.parent / 'history.jsonl').read_text().splitlines()
        assert [json.loads(s)['epoch'] for s in history] == list(range(1, 31))
    primary = {}
    for rule in ('global_conformal', 'blocked_conformal'):
        primary[rule] = {metric: statistics.mean(r['predictors']['fused'][rule][metric] for r in records)
                         for metric in ('coverage', 'block_coverage_min', 'mean_set_size', 'singleton_fraction')}
    print(json.dumps({'status': 'PASS', 'release_files_verified': verified, 'runs': 9,
                      'source_snapshots_verified': 6, 'checkpoints_verified': 9,
                      'fused_primary': primary}, indent=2))


if __name__ == '__main__':
    main()
