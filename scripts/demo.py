"""Synthetic calibration example; these values are not manuscript experiments."""
from _common import SNAPSHOT
import json
import numpy as np
from matched_sacp_posthoc import quantile_higher
from block_conformal_multimodal import sets_and_metrics

rng = np.random.default_rng(42)
scores = [rng.uniform(.01, limit, size=50) for limit in (.35, .55, .80)]
tile_q = [quantile_higher(x, .9) for x in scores]
pooled_q = quantile_higher(np.concatenate(scores), .9)
blocked_q = quantile_higher(tile_q, .75)
probabilities = rng.dirichlet([2., 1., 1.], size=30)
labels = rng.integers(0, 3, size=30)
print(json.dumps({'example': 'synthetic, not experimental data', 'tile_thresholds': tile_q,
                  'global_threshold': pooled_q, 'blocked_threshold': blocked_q,
                  'blocked_is_maximum_for_three_tiles': blocked_q == max(tile_q),
                  'global': sets_and_metrics(probabilities, labels, pooled_q),
                  'blocked': sets_and_metrics(probabilities, labels, blocked_q)}, indent=2))
