# Reproduction and evidence guide

## Scope of the release

The primary evidence consists of nine fixed-30-epoch MUUFL runs: allocation seeds 20261031, 20261054, and 20261059 crossed with training seeds 0, 1, and 2. `artifacts/paper_results/records.jsonl` contains run-level records; `summary.json` gives unweighted means and population/sample standard deviations over the nine runs. Training seeds are repeated fits within fixed geographic allocations, not independent geographic samples.

The original six-file source snapshot is kept byte-for-byte in `code/original_snapshot/`. Its hashes are recorded in `artifacts/paper_results/protocol.json`. The supported scripts provide portable paths around these functions. Their introduction does not relabel archived results as fresh training runs.

## Three verification levels

1. `verify_artifacts.py` needs no data or scientific dependencies. It verifies release files, weights, source snapshots, 30-epoch histories, and reconstructs all stored summary metrics. It cannot by itself establish correctness of image preparation or predictions.
2. `evaluate_checkpoints.py` needs the official prepared scene. It loads each released final-epoch state dictionary, recomputes outputs, freezes calibration thresholds, evaluates test predictions, and compares metrics with the recorded runs. This validates the weights-to-results route without retraining. Original imagery, annotations, and prediction arrays generated locally are not pushed to the repository.
3. `run_experiment.py` trains from scratch using the archived backbone, gate-weighted generalized cross entropy, optimizer, noise simulation, seeds, and fixed final-epoch rule. Numerical equivalence across hardware and library versions is assessed from the newly saved records rather than assumed.

The release was locally checked by evaluating all nine released checkpoints on the recorded CUDA environment. The comparison includes all reported scalar calibration metrics for fused, HSI auxiliary, LiDAR auxiliary, and adaptive probabilities. Training from scratch was not repeated as part of the release assembly.

## Input and split construction

Inputs are a 325 x 220 x 64 hyperspectral cube and first-return DEM, with labels 0--10 and background -1. Training-only standardization includes all pixels in assigned training regions. Each nominal 64 x 64 tile is eroded separately by a radius of five for 11 x 11 patches. Input supports must be pairwise disjoint between subsets.

Each allocation assigns 16 of 24 tiles to training, four to calibration, and four to testing. Retained labeled centers are 25,380/6,220/5,471 for allocation 1; 25,987/5,910/5,174 for allocation 2; and 27,772/4,775/4,524 for allocation 3, in train/calibration/test order. Calibration has three nonempty tiles in each allocation. Test has four, three, and three nonempty tiles. See each checkpoint's `manifest.json` for tile IDs, class counts, and normalization parameters.

## Classifier and training

Two convolutional branches use 32 channels, two 3 x 3 convolution--GroupNorm--GELU stages, and center-feature extraction. Fused and auxiliary linear heads are trained jointly. All heads and normalization parameters total 38,977 trainable parameters for 64 bands and 11 classes.

The fixed source uses AdamW with learning rate 0.001, weight decay 0.0001, batch size 256, gradient-norm clipping at 5, and 30 epochs. Heads have loss coefficients 1, 0.25, 0.25 and generalized-cross-entropy exponent 0.7. Detached sample weights are one during epochs 1--5; later weights use loss history, auxiliary-head disagreement, and input complexity. The archive records the exact functions. Simulated label replacements affect training centers only; each selected label changes to a different class. No calibration-label checkpoint selection is used.

## Calibration semantics

The reference-class score is `1 - p[y]`. With `n` scores at desired probability level `u`, the selected one-based rank is `min(n, ceil((n + 1) * u))`. This is an observed, capped upper order statistic, not interpolation. The within-tile level is 0.9. The outer level is beta=0.75, so three tile thresholds select rank three, their maximum.

The implementation's out-of-range cap uses the largest observed score; ordinary split-conformal inference can instead assign infinity. Neither primary rank cap binds for the recorded experiment. The code documents this distinction. Spatial support separation alone does not establish group-level exchangeability; this release evaluates coverage empirically under its stated protocol.

All classes satisfying `p[c] >= 1 - q` are retained, including equality. Empty sets remain empty. Every test tile shares the selected cutoff. Global and tile-threshold calibrators receive identical frozen model probabilities; neither changes the classifier's argmax.

## Reading metrics and controls

- `coverage`: proportion of retained test centers whose reference label is included.
- `block_coverage_min`: minimum coverage across the nonempty test tiles in that run. The paper averages this minimum over runs; it is not the minimum over the entire collection.
- `mean_set_size`: average candidate count including empty sets.
- `singleton_fraction`: fraction with exactly one candidate. `singleton_accuracy` is conditional on those centers.
- `oa_argmax`: argmax accuracy, expressed as a proportion in raw records.

HSI/LiDAR auxiliary controls are heads of the jointly trained network. They do not represent independently trained unimodal baselines. The adaptive probability combination is an additional frozen-output diagnostic. The SACP control applies a matched neighborhood score operator to the same base probabilities and retained nontraining centers; it does not reproduce the original SACP benchmark.

The fixed ten-value alpha grid and sensitivity settings are archived in `artifacts/paper_results/diagnostics/protocol.json`. The wrapper requires all nine generated probability arrays and runs the complete grid. Saved plot data and class support include absent or sparsely represented classes; aggregate coverage does not imply class-conditional coverage.

## Licenses and provenance

Code and model weights are original project artifacts released under MIT. Raw MUUFL imagery/annotations retain upstream ownership and terms. `data/PROVENANCE.json` identifies the source file version. JSON paths that pointed to a local Windows account were made repository-relative when assembling public records; numerical entries and checkpoint bytes were preserved. Historical trainer/source hashes identify archived files, not the new wrapper scripts or reformatted public JSON files.

`SHA256SUMS.json` records the public release-file hashes. No manuscript PDF, cover letter, private account configuration, authentication token, external source checkout, or local raw imagery is included.
