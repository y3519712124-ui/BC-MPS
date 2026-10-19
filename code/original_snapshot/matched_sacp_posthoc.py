"""Matched post-hoc SACP comparison on the nine frozen MUUFL checkpoints.

This script does not train a model.  It reproduces the public SACP defaults
(lambda=0.5, k=1) on the same fused probabilities, spatial allocations,
calibration labels, and test supports used by block_conformal_multimodal.py.
The neighbor score is computed over retained non-training centers in the
8-connected immediate neighborhood.  The primary SACP result uses one global
threshold, as in the published algorithm; an additional tile-threshold variant
is recorded only as a two-factor diagnostic.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import sys
import argparse

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validated_protocol"))
from protocol import Scene  # noqa: E402
from train import Classifier  # noqa: E402
from adaptive_reliability_experiments import cache  # noqa: E402


ALPHA = 0.10
LAMBDA = 0.50
DATA = ROOT / "data" / "muufl_spatial.npz"
RAW = ROOT / "results" / "block_conformal_v2.jsonl"
OUT = ROOT / "results" / "sacp_matched_v1.jsonl"


def outputs(model, bundle, device, batch=256):
    out = []
    with torch.no_grad():
        for st in range(0, bundle["n"], batch):
            z = model(bundle["h"][st : st + batch], bundle["d"][st : st + batch])
            out.append(np.stack([x.softmax(1).cpu().numpy() for x in z]))
    return np.concatenate(out, axis=1)


def quantile_higher(x, level):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        raise ValueError("empty calibration scores")
    rank = min(len(x) - 1, max(0, int(math.ceil((len(x) + 1) * level)) - 1))
    return float(np.sort(x)[rank])


def neighbor_offsets(neighborhood):
    if neighborhood == "4":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighborhood == "8":
        return [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy, dx) != (0, 0)]
    raise ValueError(f"unsupported neighborhood: {neighborhood}")


def neighbor_aggregate(coords, score_grid, lam=LAMBDA, k=1, neighborhood="8"):
    """Apply SACP's score operator for *k* iterations.

    The matched protocol stores scores only at retained validation/test centers;
    NaNs therefore exclude background and training centers from the neighborhood.
    """
    h, w, _ = score_grid.shape
    offsets = neighbor_offsets(neighborhood)
    current = score_grid.copy()
    for _ in range(k):
        updated = current.copy()
        base = current[coords[:, 0], coords[:, 1]]
        total = np.zeros_like(base, dtype=np.float64)
        count = np.zeros((len(coords), 1), dtype=np.float64)
        for dy, dx in offsets:
            yy = coords[:, 0] + dy
            xx = coords[:, 1] + dx
            valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            if not valid.any():
                continue
            vals = np.full_like(base, np.nan, dtype=np.float64)
            vals[valid] = current[yy[valid], xx[valid]]
            present = np.isfinite(vals[:, 0])
            if present.any():
                total[present] += vals[present]
                count[present, 0] += 1
        mean_neighbor = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        updated_values = (1.0 - lam) * base + lam * np.where(count > 0, mean_neighbor, base)
        updated[coords[:, 0], coords[:, 1]] = updated_values
        current = updated
    return current[coords[:, 0], coords[:, 1]]


def set_metrics(scores, labels, q):
    include = scores <= q
    true = include[np.arange(len(labels)), labels]
    size = include.sum(axis=1)
    return {
        "coverage": float(true.mean()),
        "mean_set_size": float(size.mean()),
        "singleton_fraction": float((size == 1).mean()),
        "empty_fraction": float((size == 0).mean()),
        "q": float(q),
    }


def tile_metrics(scores, labels, coords, tile, q):
    values = []
    for tid in sorted(np.unique(tile)):
        ix = tile == tid
        if ix.any():
            values.append(set_metrics(scores[ix], labels[ix], q)["coverage"])
    return {
        "block_coverage_mean": float(np.mean(values)),
        "block_coverage_min": float(np.min(values)),
        "block_coverage_sd": float(np.std(values)),
        "blocks": int(len(values)),
    }


def one_record(spec, device, neighborhood="8", k=1):
    scene = Scene(DATA, 64, 11, spec["split_seed"])
    val = cache(scene, "val", device)
    test = cache(scene, "test", device)
    model = Classifier(scene.h.shape[2], scene.classes).to(device)
    checkpoint = Path(spec["checkpoint"])
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["student"])
    model.eval()
    ov = outputs(model, val, device)
    ot = outputs(model, test, device)
    yv = val["y"].cpu().numpy()
    yt = test["y"].cpu().numpy()
    cv = val["coords"]
    ct = test["coords"]
    all_coords = np.concatenate([cv, ct], axis=0)
    fused = np.concatenate([ov[0], ot[0]], axis=0)
    raw_scores = 1.0 - fused
    # score_grid is indexed by image coordinates and stores all retained
    # non-training centers; its NaNs exclude missing/unlabelled neighbors.
    grid = np.full((*scene.y.shape, scene.classes), np.nan, dtype=np.float64)
    grid[all_coords[:, 0], all_coords[:, 1]] = raw_scores
    # SACP aggregates the retained non-training pool jointly.  Applying the
    # operator once to the concatenated validation/test centers also makes
    # k>1 use the previous iteration's validation scores when updating test
    # scores (and vice versa).
    agg_all = neighbor_aggregate(all_coords, grid, k=k, neighborhood=neighborhood)
    agg_v = agg_all[:len(cv)]
    agg_t = agg_all[len(cv):]
    true_v = agg_v[np.arange(len(yv)), yv]
    q_global = quantile_higher(true_v, 1.0 - ALPHA)
    include_global = agg_t <= q_global
    result = {
        "split_seed": spec["split_seed"],
        "seed": spec["seed"],
        "checkpoint": str(checkpoint),
        "alpha": ALPHA,
        "lambda": LAMBDA,
        "k": k,
        "neighbor_definition": f"{neighborhood}-connected immediate neighbors among retained validation/test centers; training centers excluded",
        "calibration_pixels": int(len(yv)),
        "test_pixels": int(len(yt)),
        "calibration_blocks": int(len(np.unique(scene.tile[tuple(cv.T)]))),
        "predictor": "fused",
        "method_definition": "SACP score aggregation followed by one global higher-order threshold",
        "sacp_global": set_metrics(agg_t, yt, q_global),
        "sacp_global_tile_summary": tile_metrics(agg_t, yt, ct, scene.tile[tuple(ct.T)], q_global),
    }
    # Factor diagnostic: apply the BC-MPS second-stage tile threshold to the
    # SACP-refined scores. This is not the primary SACP algorithm.
    val_tiles = scene.tile[tuple(cv.T)]
    block_q = {int(t): quantile_higher(true_v[val_tiles == t], 1.0 - ALPHA)
               for t in np.unique(val_tiles)}
    q_block = quantile_higher(list(block_q.values()), 0.75)
    result["sacp_score_plus_tile_threshold"] = set_metrics(agg_t, yt, q_block)
    result["sacp_score_plus_tile_threshold"].update(
        tile_metrics(agg_t, yt, ct, scene.tile[tuple(ct.T)], q_block))
    result["thresholds"] = {"global": q_global, "block_q75": q_block,
                             "tile_count": len(block_q)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neighborhood", choices=["4", "8"], default="8")
    parser.add_argument("--k", type=int, choices=[1, 2], default=1)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        specs.append({"split_seed": r["split_seed"], "seed": len(specs) % 3,
                      "checkpoint": r["checkpoint"]})
    results = [one_record(s, device, neighborhood=args.neighborhood, k=args.k) for s in specs]
    args.out.write_text("\n".join(json.dumps(x, separators=(",", ":")) for x in results) + "\n",
                   encoding="utf-8")
    print(json.dumps({"records": len(results), "device": str(device),
                      "neighborhood": args.neighborhood, "k": args.k,
                      "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
