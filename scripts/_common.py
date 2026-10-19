"""Portable helpers shared by the supported reproduction entry points."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / 'code/original_snapshot'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(SNAPSHOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def summarize_records(records, destination):
    import numpy as np
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'records.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    summary, rows = {}, []
    for head in ('fused', 'adaptive', 'hsi', 'lidar'):
        for rule in ('global_conformal', 'blocked_conformal'):
            values = [r['predictors'][head][rule] for r in records]
            summary[head + ':' + rule] = {
                key: {'mean': float(np.mean([v[key] for v in values])),
                      'population_sd': float(np.std([v[key] for v in values])),
                      'sample_sd': float(np.std([v[key] for v in values], ddof=1)) if len(values) > 1 else None}
                for key in values[0]}
            for index, record in enumerate(records, 1):
                rows.append({'run': index, 'split_seed': record['split_seed'],
                             'predictor': head, 'calibration': rule,
                             **record['predictors'][head][rule]})
    save(destination / 'summary.json', summary)
    with (destination / 'flat.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary
