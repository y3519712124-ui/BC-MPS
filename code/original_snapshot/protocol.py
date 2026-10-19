"""Strict spatial development protocol. Evaluation units are spatial tiles.

Uses arrays only from legacy NPZ files; ignores their masks. Tile assignments
are generated independently of labels. No patch support crosses a tile edge.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.ndimage import binary_erosion, maximum_filter
import torch
from torch.utils.data import Dataset


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def split_tiles(shape, block=64, patch=11, seed=20261005):
    if patch % 2 != 1 or block < 2 * patch:
        raise ValueError('Odd patch required; tile size must be >= 2*patch')
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    ncols = (w + block - 1) // block
    tile = (yy // block) * ncols + xx // block
    ids = np.unique(tile)
    if len(ids) < 5:
        raise ValueError('At least five spatial tiles required')
    order = np.random.default_rng(seed).permutation(ids)
    nval = max(1, len(ids)//5)
    groups = {'train': order[2*nval:], 'val': order[:nval], 'test': order[nval:2*nval]}
    radius = patch // 2
    regions, masks = {}, {}
    for name, selected in groups.items():
        region = np.isin(tile, selected)
        # Erode EACH tile separately, including boundaries between same-fold
        # tiles, to give every patch exactly one spatial experimental unit.
        centre = np.zeros(shape, dtype=bool)
        for i in selected:
            centre |= binary_erosion(tile == i, structure=np.ones((patch,patch)), border_value=0)
        regions[name], masks[name] = region, centre
    supports = {k: maximum_filter(v.astype(np.uint8), size=patch, mode='constant') > 0 for k,v in masks.items()}
    names = list(masks)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if np.any(supports[a] & supports[b]):
                raise AssertionError('Patch support leakage')
    return tile, regions, masks, {k: v.tolist() for k,v in groups.items()}


def observed_boundaries(labels):
    """Only adjacent labelled, unequal classes; no wrapping or background edges.

    A sparse annotation cannot define all scene boundaries. This map is for
    evaluation of observed transitions only; it never enters the model.
    """
    b = np.zeros_like(labels, dtype=bool)
    for axis in (0,1):
        a = [slice(None)] * 2; c = [slice(None)] * 2
        a[axis] = slice(None,-1); c[axis] = slice(1,None)
        a,c = tuple(a),tuple(c)
        edge = (labels[a]>=0) & (labels[c]>=0) & (labels[a]!=labels[c])
        b[a] |= edge; b[c] |= edge
    return b


class Scene:
    def __init__(self, path, block=64, patch=11, split_seed=20261005):
        self.path = Path(path)
        with np.load(path) as data:
            self.h = data['hsi'].astype(np.float32)
            self.d = data['lidar'].astype(np.float32)
            self.y = data['labels'].astype(np.int64)
        if self.h.shape[:2]!=self.y.shape or self.d.shape!=self.y.shape:
            raise ValueError('Misaligned modalities')
        if not np.isfinite(self.h).all() or not np.isfinite(self.d).all():
            raise ValueError('Nonfinite input')
        ids = np.unique(self.y[self.y>=0])
        if not np.array_equal(ids, np.arange(len(ids))):
            raise ValueError('Labels must be 0..C-1; background -1')
        self.classes, self.patch = len(ids), patch
        self.tile, self.regions, self.masks, groups = split_tiles(self.y.shape,block,patch,split_seed)
        # Train pixels only, including unlabelled input within train tiles.
        hp = self.h[self.regions['train']]
        dp = self.d[self.regions['train']]
        hm,hs = hp.mean(0),hp.std(0).clip(1e-6)
        dm,ds = dp.mean(),max(dp.std(),1e-6)
        self.h=(self.h-hm)/hs; self.d=(self.d-dm)/ds
        # Input-only edge score used to define the boundary-focused noise
        # pool. It is computed after train-only normalization and never uses
        # labels, validation masks or test masks.
        edge=np.zeros(self.y.shape,dtype=np.float32)
        edge[:,1:]+=np.sqrt(np.mean((self.h[:,1:]-self.h[:,:-1])**2,axis=2))
        edge[1:,:]+=np.sqrt(np.mean((self.h[1:,:]-self.h[:-1,:])**2,axis=2))
        edge[:,1:]+=np.abs(self.d[:,1:]-self.d[:,:-1])
        edge[1:,:]+=np.abs(self.d[1:,:]-self.d[:-1,:])
        self.edge_score=np.tanh(.25*edge)
        self.boundary=observed_boundaries(self.y)
        self.manifest={'dataset':self.path.stem,'data_sha256':digest(path),
            'split_seed':split_seed,'block':block,'patch':patch,'tiles':groups,
            'normalization':'per-band HSI and scalar LiDAR; fit train regions only',
            'patch_support_overlap':0,'class_counts':{},
            'boundary_noise_definition':'top quartile of input-only edge_score within train samples',
            'normalization_parameters':{'hsi_mean':hm.tolist(),'hsi_std':hs.tolist(),
                                         'lidar_mean':float(dm),'lidar_std':float(ds)}}
        for k in self.masks:
            mask=self.masks[k] & (self.y>=0)
            self.manifest['class_counts'][k]=np.bincount(self.y[mask],minlength=self.classes).tolist()
        # Missing training classes make a closed-set comparison uninterpretable.
        if min(self.manifest['class_counts']['train'])==0:
            raise ValueError('A class is absent from training tiles; document this split as infeasible')


class Patches(Dataset):
    def __init__(self, scene, split, noise=0., seed=0, noise_model='uniform'):
        if split!='train' and noise:
            raise ValueError('Corruption restricted to training')
        if not 0<=noise<1:
            raise ValueError('Noise rate must be in [0,1)')
        self.scene,self.split=scene,split
        self.coords=np.argwhere(scene.masks[split] & (scene.y>=0))
        if not len(self.coords): raise ValueError('Empty split')
        self.clean=scene.y[tuple(self.coords.T)].copy()
        self.target=self.clean.copy()
        rng=np.random.default_rng(seed)
        eligible=np.ones(len(self.clean),bool)
        if noise_model=='boundary':
            # The boundary-focused simulator uses only an input-derived
            # boundary-risk pool. This avoids relying on clean labels and
            # remains non-degenerate when annotated transitions are sparse.
            scores=scene.edge_score[tuple(self.coords.T)]
            eligible=scores>=np.quantile(scores,.75)
        elif noise_model!='uniform': raise ValueError(noise_model)
        self.corrupt=(rng.random(len(self.clean))<noise) & eligible
        off=rng.integers(1,scene.classes,len(self.clean))
        self.target[self.corrupt]=(self.clean[self.corrupt]+off[self.corrupt])%scene.classes
        self.eligible=eligible

    def __len__(self): return len(self.coords)

    def __getitem__(self,i):
        y,x=self.coords[i]; r=self.scene.patch//2
        h=self.scene.h[y-r:y+r+1,x-r:x+r+1].transpose(2,0,1).copy()
        d=self.scene.d[y-r:y+r+1,x-r:x+r+1][None].copy()
        # No clean target or injected corruption flag is supplied to training.
        return torch.from_numpy(h),torch.from_numpy(d),torch.tensor(self.target[i]),torch.tensor(i)
