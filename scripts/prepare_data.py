"""Prepare the original MUUFL MAT file; source data remain with their owners."""
import argparse
from pathlib import Path
from _common import REPO, digest, save
import json
import numpy as np
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original Campus 1 labeled MAT file')
    parser.add_argument('--output', type=Path, default=REPO / 'data/muufl_spatial.npz')
    args = parser.parse_args()
    expected = json.loads((REPO / 'data/PROVENANCE.json').read_text())
    raw_hash = digest(args.source)
    if raw_hash != expected['source_sha256']:
        raise ValueError('Source SHA-256 differs from the manuscript data version. See data/PROVENANCE.json.')
    hsi = loadmat(args.source, simplify_cells=True)['hsi']
    spectra = np.asarray(hsi['Data'], dtype=np.float32)
    lidar = np.asarray(hsi['Lidar'][0]['z'][:, :, 0], dtype=np.float32)
    raw = np.asarray(hsi['sceneLabels']['labels'], dtype=np.int64)
    labels = np.where(raw > 0, raw - 1, -1)
    assert spectra.shape == (325, 220, 64)
    assert spectra.shape[:2] == lidar.shape == labels.shape
    assert np.isfinite(spectra).all() and np.isfinite(lidar).all()
    assert np.array_equal(np.unique(labels[labels >= 0]), np.arange(11))
    assert np.bincount(labels[labels >= 0], minlength=11).tolist() == expected['label_counts']
    if args.output.exists():
        with np.load(args.output) as old:
            for name, array in [('hsi', spectra), ('lidar', lidar), ('labels', labels)]:
                if not np.array_equal(old[name], array):
                    raise FileExistsError('Different prepared arrays already exist; choose a new output path.')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, hsi=spectra, lidar=lidar, labels=labels)
    result = {'source_sha256': raw_hash, 'prepared_sha256': digest(args.output),
              'shape': list(spectra.shape), 'array_checks': 'PASS'}
    save(args.output.with_suffix('.provenance.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
