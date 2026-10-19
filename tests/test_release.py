"""Checks for calibration semantics, support separation, and backbone structure."""
from pathlib import Path
import os
import sys
import unittest
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code/original_snapshot'))
import numpy as np
from scipy.ndimage import maximum_filter
import torch
from protocol import split_tiles
from train import Classifier
from matched_sacp_posthoc import quantile_higher
from block_conformal_multimodal import sets_and_metrics


class ScientificSemantics(unittest.TestCase):
    def test_corrected_rank_and_small_sample_cap(self):
        self.assertEqual(quantile_higher(np.arange(10) / 10, .8), .8)
        self.assertEqual(quantile_higher([.1, .3, .2], .9), .3)

    def test_three_tile_outer_quantile(self):
        self.assertEqual(quantile_higher([.1, .7, .3], .75), .7)

    def test_inclusion_boundary_and_empty_sets(self):
        p = np.array([[.5, .5], [.49, .51]])
        metrics = sets_and_metrics(p, np.array([0, 0]), .5)
        self.assertEqual(metrics['coverage'], .5)
        self.assertEqual(metrics['mean_set_size'], 1.5)
        empty = sets_and_metrics(np.array([[.4, .35, .25]]), np.array([0]), .1)
        self.assertEqual(empty['empty_fraction'], 1.)
        self.assertEqual(empty['coverage'], 0.)

    def test_support_separation(self):
        _, _, masks, _ = split_tiles((325, 220), 64, 11, 20261031)
        supports = [maximum_filter(m.astype('uint8'), size=11, mode='constant') > 0
                    for m in masks.values()]
        for i, support in enumerate(supports):
            for other in supports[i + 1:]:
                self.assertFalse(np.any(support & other))

    def test_backbone_dimensions_and_parameter_count(self):
        model = Classifier(64, 11).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 38977)
        with torch.no_grad():
            outputs = model(torch.zeros(2, 64, 11, 11), torch.zeros(2, 1, 11, 11))
        self.assertEqual([tuple(x.shape) for x in outputs], [(2, 11)] * 3)


if __name__ == '__main__':
    unittest.main()
