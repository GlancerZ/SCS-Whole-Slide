"""Compact, integer-safe SCS paper preprocessing on the bundled liver sections.

Same Seurat-v3 HVGs, occupied-bin ring order, labels and scan-order background
cap as SCS/src/preprocessing.py, without materializing N x 50 x 2000 counts.
Nucleus extraction uses the workspace's Spateo compatibility implementation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SCS'))
from src import spateo_compat as st


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ring_offsets():
    result = []
    for d in range(1, 11):
        result.extend((-d, y) for y in range(-d, d + 1))
        result.extend((d, y) for y in range(-d, d + 1))
        result.extend((x, -d) for x in range(-d + 1, d))
        result.extend((x, d) for x in range(-d + 1, d))
    return result


def neighbors_for(occupied, shape, n=50):
    """Vectorized over centres, exact upstream ring/within-ring ordering."""
    centres = np.flatnonzero(occupied)
    r, c = np.divmod(centres, shape[1])
    neighbors = np.full((len(centres), n), -1, np.int32)
    neighbors[:, 0] = centres
    length = np.ones(len(centres), np.int32)
    for dr, dc in ring_offsets():
        rr, cc = r + dr, c + dc
        ids = np.flatnonzero((length < n) & (rr >= 0) & (rr < shape[0]) &
                             (cc >= 0) & (cc < shape[1]))
        idx = rr[ids] * shape[1] + cc[ids]
        use = occupied[idx]
        ids, idx = ids[use], idx[use]
        neighbors[ids, length[ids]] = idx
        length[ids] += 1
    return neighbors[length == n]


def prepare(tile, destination):
    started = time.time()
    out = destination / tile
    out.mkdir(parents=True, exist_ok=False)
    data = out / 'data'
    data.mkdir()
    archive = ROOT / 'SCS' / 'data' / f'Mouse_liver_bin_{tile}.tsv.tar.gz'
    image = ROOT / 'SCS' / 'data' / f'tile_{tile}.tiff'
    expected = f'Mouse_liver_bin_{tile}.tsv'
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if len(members) != 1 or members[0].name != expected or not members[0].isfile():
            raise ValueError('unexpected archive members')
        tar.extract(members[0], out)
    tsv = out / expected
    print(f'{tile}: read and segment stain', flush=True)
    a = st.io.read_bgi_agg(str(tsv), str(image), prealigned=True)
    st.cs.mask_nuclei_from_stain(a, otsu_classes=4, otsu_index=1)
    st.cs.find_peaks_from_mask(a, 'stain', 7)
    st.cs.watershed(a, 'stain', 5, out_layer='watershed_labels')
    a.write_h5ad(data / 'spots0:0:0:0.h5ad')
    labels = a.layers['watershed_labels'].astype(np.int32)
    shape = labels.shape
    nuclei = np.unique(labels[labels > 0])
    centers = np.zeros((labels.max() + 1, 2), np.float64)
    centers[nuclei] = ndimage.center_of_mass(np.ones(shape), labels, nuclei)
    if not len(nuclei):
        raise ValueError('No nuclei detected')
    df = pd.read_csv(tsv, sep='\t')
    gene, row, col, count = df.columns
    r = df[row].to_numpy(np.int64)
    c = df[col].to_numpy(np.int64)
    origin = [int(r.min()), int(c.min())]
    r -= origin[0]
    c -= origin[1]
    codes, genes = pd.factorize(df[gene], sort=False)
    values = df[count].to_numpy(np.int64)
    if np.any(values < 0) or np.any(r >= shape[0]) or np.any(c >= shape[1]):
        raise ValueError('invalid coordinates/counts')
    sparse.save_npz(out / 'rna_pixels.npz', sparse.csr_matrix(
        (values, (r * shape[1] + c, codes)), shape=(math.prod(shape), len(genes))))
    bin_shape = tuple(math.ceil(s / 3) for s in shape)
    x = sparse.csr_matrix((values, ((r // 3) * bin_shape[1] + c // 3, codes)),
                          shape=(math.prod(bin_shape), len(genes)))
    print(f'{tile}: HVG selection from {x.shape}, {x.nnz} nonzeros', flush=True)
    hvg = ad.AnnData(x.astype(np.float64))
    sc.pp.highly_variable_genes(hvg, n_top_genes=2000, flavor='seurat_v3', span=1.)
    chosen = np.flatnonzero(hvg.var.highly_variable.to_numpy())
    expr = x[:, chosen].astype(np.float32).tocsr()
    expr.eliminate_zeros()
    sparse.save_npz(out / 'expression.npz', expr)
    (out / 'genes.json').write_text(json.dumps(genes[chosen].tolist()))
    (out / 'all_genes.json').write_text(json.dumps(genes.tolist()))
    ng = neighbors_for(np.diff(expr.indptr) > 0, bin_shape)
    rc = np.stack(np.unravel_index(ng[:, 0], bin_shape), axis=-1) * 3
    nucleus = labels[rc[:, 0], rc[:, 1]]
    foreground = nucleus > 0
    nearest = cKDTree(centers[nuclei]).query(rc)[0]
    background = (~foreground) & (a.layers['stain'][rc[:, 0], rc[:, 1]] <= 10) & (nearest > 30)
    # Upstream background cap is applied during spatial scan, not after it.
    accepted_bg = np.zeros(len(rc), bool)
    nfg = nbg = 0
    for i in range(len(rc)):
        if foreground[i]:
            nfg += 1
        elif background[i] and nbg < nfg:
            accepted_bg[i] = True
            nbg += 1
    eligible = foreground | accepted_bg
    delta = centers[nucleus] - rc
    direction = (np.mod(np.arctan2(delta[:, 0], delta[:, 1]), 2 * np.pi) /
                 (2 * np.pi / 16)).astype(np.int64)
    direction[~foreground] = 0
    # User-requested random centre split, fixed across all candidate methods.
    # A second untouched random test split prevents reporting selected val as test.
    rng = np.random.RandomState(20260905)
    split = np.full(len(rc), -1, np.int8)
    for b in (False, True):
        for d in (range(16) if b else [0]):
            ids = np.flatnonzero(eligible & (foreground == b) & (direction == d))
            rng.shuffle(ids)
            nval = max(1, len(ids) // 10) if len(ids) >= 10 else 0
            split[ids] = 0
            split[ids[:nval]] = 1
            split[ids[nval:2*nval]] = 2
    paper_holdout = (rc[:, 0] > int(shape[0]*.75)) & (rc[:, 1] > int(shape[1]*.75))
    paper_split = np.full(len(rc), -1, np.int8)
    paper_split[eligible & ~paper_holdout] = 0
    paper_split[eligible & paper_holdout] = 1
    np.savez_compressed(out / 'samples.npz', neighbors=ng, coords=rc, foreground=foreground,
                        direction=direction, split=split, paper_split=paper_split,
                        nucleus=nucleus, bin_shape=bin_shape, shape=shape)
    stats = dict(tile=tile, dataset='Seq-scope mouse liver', shape=shape, origin=origin,
                 bin_shape=bin_shape, n_genes=len(genes), hvg_count=len(chosen),
                 rna_records=len(df), total_umi=int(values.sum()), nuclei=len(nuclei),
                 input_centres=len(rc), labeled_foreground=nfg, labeled_background=nbg,
                 random_train=int(sum(split == 0)), random_validation=int(sum(split == 1)),
                 random_test=int(sum(split == 2)), seconds=time.time()-started,
                 archive_sha256=sha(archive), image_sha256=sha(image),
                 preprocessing='SCS rules; workspace Spateo compatibility nuclei; int64 RNA sums',
                 split='stratified random 80/10/10 centres; context overlap allowed',
                 hvg_selection='per-section Seurat-v3, n_top_genes=2000, span=1; transductive',
                 dependencies={k: __import__(k).__version__ for k in ['numpy','scipy','scanpy','anndata']})
    (out / 'prepared.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--tiles', nargs='+', default=['2104','2105','2106','2107'])
    args = p.parse_args()
    for tile in args.tiles:
        prepare(tile, args.root)
