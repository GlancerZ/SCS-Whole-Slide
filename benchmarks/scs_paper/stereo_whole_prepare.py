"""Whole MOSTA adult-brain inputs; run only inside a Slurm allocation.

The source image is indexed in SCS (row=x, column=y) coordinates. Crop BOTH
RNA and image at the measured RNA origin, as upstream Spateo does. Never use
the workspace's ST19-only read_bgi_agg/alignment shim for this dataset.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import anndata as ad
import cv2
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from tifffile import imread

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'SCS'))
from src import spateo_compat as st
from benchmarks.scs_paper.prepare import neighbors_for
from optimizations.scs_streaming.genept import load_genept_embeddings, make_gene_lookup


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_index(root):
    marker = root / 'indexed.json'
    if marker.exists():
        return json.loads(marker.read_text())
    source = root / 'source'
    gem = source / 'Mouse_brain_Adult_GEM_bin1.tsv.gz'
    image = source / 'Mouse_brain_Adult.tif'
    # File sizes from the official server; gzip's CRC is checked by full reads.
    if gem.stat().st_size != 466140570:
        raise ValueError('GEM download incomplete or upstream asset changed')
    stain = imread(image)
    if stain.ndim != 2 or stain.dtype != np.uint8:
        raise ValueError(f'Expected original uint8 2-D stain, got {stain.shape}/{stain.dtype}')
    print(f'IMAGE {stain.shape}, {image.stat().st_size} bytes', flush=True)
    dtype = dict(geneID='category', x=np.int32, y=np.int32, MIDCounts=np.int64)
    minimum = np.array([np.iinfo(np.int32).max] * 2, np.int64)
    maximum = np.zeros(2, np.int64)
    genes, lookup = [], {}
    records = umis = 0
    temporary = root / 'records.building.h5'
    with h5py.File(temporary, 'w') as h:
        for key, dt in [('x', 'i4'), ('y', 'i4'), ('gene', 'i4'), ('count', 'i8')]:
            h.create_dataset(key, (0,), maxshape=(None,), dtype=dt, chunks=(262144,), compression='lzf')
        for frame in pd.read_csv(gem, sep='\t', dtype=dtype, chunksize=1000000):
            categories = frame.geneID.cat.categories.tolist()
            for symbol in categories:
                if symbol not in lookup:
                    lookup[symbol] = len(genes)
                    genes.append(symbol)
            gids = np.array([lookup[g] for g in categories], np.int32)[frame.geneID.cat.codes]
            x, y, count = [frame[k].to_numpy() for k in ['x', 'y', 'MIDCounts']]
            if (count <= 0).any() or (x < 0).any() or (y < 0).any():
                raise ValueError('Invalid source coordinate or non-positive count')
            minimum = np.minimum(minimum, [x.min(), y.min()])
            maximum = np.maximum(maximum, [x.max(), y.max()])
            end = records + len(frame)
            for key, value in [('x', x), ('y', y), ('gene', gids), ('count', count)]:
                h[key].resize((end,))
                h[key][records:end] = value
            records = end
            umis += int(count.sum())
            if records % 10000000 == 0:
                print(f'INDEX {records:,} records / {umis:,} UMIs', flush=True)
        h.attrs['complete'] = True
    shape = maximum - minimum + 1
    if np.any(maximum >= stain.shape):
        raise ValueError('RNA coordinates extend outside the source stain; manual alignment review needed')
    crop = stain[minimum[0]:maximum[0]+1, minimum[1]:maximum[1]+1]
    np.save(root / 'stain_source.npy', crop)
    # A direct equality check against the authors' bundled example anchors axes.
    reference = imread(ROOT / 'SCS/data/Mouse_brain_Adult_sub.tif')
    example = crop[5700:6900, 5700:6900]
    example_equal = example.shape == reference.shape and np.array_equal(example, reference)
    np.save(root / 'source_example.npy', example)
    metadata = dict(dataset='MOSTA STDS0000058 adult mouse brain, original SCS source',
        origin=minimum.tolist(), shape=shape.tolist(), source_image_shape=list(stain.shape),
        records=records, umis=umis, genes=len(genes),
        source_example_exact_match=bool(example_equal),
        gem_sha256=sha(gem), image_sha256=sha(image),
        axes='axis 0 = GEM x; axis 1 = GEM y; image and RNA cropped at the same origin',
        crop='entire RNA bounding box; no density-selected patches')
    write_json(root / 'genes.json', genes)
    temporary.replace(root / 'records.h5')
    write_json(marker, metadata)
    print(json.dumps(metadata), flush=True)
    return metadata


def expression(root, meta):
    if (root / 'expression_indexed.json').exists():
        return
    shape = tuple(math.ceil(s / 3) for s in meta['shape'])
    rows, cols, values = [], [], []
    pixel_umi = np.zeros(meta['shape'], np.int64)
    with h5py.File(root / 'records.h5') as h:
        for start in range(0, meta['records'], 2000000):
            sl = slice(start, start + 2000000)
            x = h['x'][sl].astype(np.int64) - meta['origin'][0]
            y = h['y'][sl].astype(np.int64) - meta['origin'][1]
            count = h['count'][sl]
            np.add.at(pixel_umi, (x, y), count)
            rows.append((x // 3) * shape[1] + y // 3)
            cols.append(h['gene'][sl])
            values.append(count)
    x = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(math.prod(shape), meta['genes']), dtype=np.int64)
    x.eliminate_zeros()
    if int(x.sum()) != meta['umis'] or int(pixel_umi.sum()) != meta['umis']:
        raise AssertionError('Counts not conserved in global 3x3 grid')
    sparse.save_npz(root / 'all_expression.npz', x)
    np.save(root / 'pixel_umi.npy', pixel_umi)
    write_json(root / 'expression_indexed.json', dict(bin_size=3, bin_shape=shape,
        nnz=x.nnz, occupied_bins=int(np.count_nonzero(np.diff(x.indptr))), umis=meta['umis']))


def normalized_to_pixel(theta, shape):
    """Convert torch affine_grid(align_corners=False) to OpenCV inverse mapping."""
    height, width = shape
    norm = np.array([[2/width, 0, -(width-1)/width],
                     [0, 2/height, -(height-1)/height], [0, 0, 1.]])
    return (np.linalg.inv(norm) @ np.vstack((theta, [0, 0, 1])) @ norm)[:2]


def align(root):
    raise RuntimeError('Global alignment draft disabled: use align_patches for paper-aligned comparisons')
    if (root / 'alignment.json').exists():
        return
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(8)
    stain = np.load(root / 'stain_source.npy')
    rna = np.load(root / 'pixel_umi.npy', mmap_mode='r')
    # Same affine objective and Adam defaults as Spateo 1.0.2, with explicit
    # downscale for a whole image. This is not historical Spateo 0.0.0 parity.
    scale = min(1., 2500 / max(stain.shape))
    small = cv2.resize(stain.astype(np.float32), (0, 0), fx=scale, fy=scale)
    blurred = cv2.GaussianBlur(np.asarray(rna, np.float32), (5, 5), 0)
    reference = cv2.resize(blurred, small.shape[::-1])
    del blurred
    source = torch.tensor(small / max(float(small.max()), 1.))[None, None]
    target = torch.tensor(reference / max(float(reference.max()), 1.))[None, None]
    theta = torch.nn.Parameter(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
    optimizer = torch.optim.Adam([theta])
    history = []
    for epoch in range(100):
        grid = F.affine_grid(theta[None], source.shape, align_corners=False)
        pred = F.grid_sample(source, grid, align_corners=False)
        loss = -((target + 1) * target * pred).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss))
    transform = theta.detach().numpy()
    matrix = normalized_to_pixel(transform, stain.shape)
    aligned = cv2.warpAffine(stain, matrix, stain.shape[::-1],
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
    corners = np.array([[0, 0, 1], [stain.shape[1]-1, 0, 1],
                        [0, stain.shape[0]-1, 1], [stain.shape[1]-1, stain.shape[0]-1, 1]])
    shift = np.linalg.norm(corners @ matrix.T - corners[:, :2], axis=1)
    result = dict(theta=transform.tolist(), inverse_pixel_matrix=matrix.tolist(),
        max_corner_displacement=float(shift.max()), loss_before=history[0], loss_last=history[-1],
        downscale=scale, epochs=100, method='Spateo 1.0.2 affine objective, CPU Adam, global downscaled fit',
        interpolation='OpenCV bilinear (quantized interpolation weights), inverse warp',
        not_exact_paper_alignment='Paper applies per-patch alignment; shared whole-slide transform here')
    np.save(root / 'stain_aligned.npy', aligned)
    # A numeric guard is not a substitute for inspecting alignment previews.
    result['numeric_guard_passed'] = bool(history[-1] < history[0] and shift.max() < 120
        and np.all(np.linalg.svd(transform[:, :2], compute_uv=False) > .95)
        and np.all(np.linalg.svd(transform[:, :2], compute_uv=False) < 1.05))
    write_json(root / 'alignment.json', result)
    print(json.dumps(result), flush=True)
    if not result['numeric_guard_passed']:
        raise ValueError('Alignment exceeds conservative guard; do not train before review')


def nuclei(root):
    raise RuntimeError('Global nucleus draft disabled: use align_patches for paper-aligned comparisons')
    if (root / 'nuclei_completed.json').exists():
        return
    if not json.loads((root / 'alignment.json').read_text())['numeric_guard_passed']:
        raise ValueError('Alignment QA failed')
    stain = np.load(root / 'stain_aligned.npy')
    # Lightweight layer carrier avoids copying a huge AnnData expression matrix.
    from types import SimpleNamespace
    a = SimpleNamespace(layers={'stain': stain})
    st.cs.mask_nuclei_from_stain(a, otsu_classes=4, otsu_index=1)
    st.cs.find_peaks_from_mask(a, 'stain', 7)
    st.cs.watershed(a, 'stain', 5, out_layer='watershed_labels')
    labels = a.layers['watershed_labels'].astype(np.int32)
    count = int(labels.max())
    if count == 0:
        raise ValueError('No nuclei on the whole slide')
    np.save(root / 'nuclei.npy', labels)
    write_json(root / 'nuclei_completed.json', dict(nuclei=count,
        method='global stain watershed; no tile-level label seams', parameters=[4, 1, 7, 5]))
    print(f'NUCLEI {count:,}', flush=True)


def original_aligner():
    """Load the two unmodified classes from the bundled upstream source.

    Only trusted local class definitions are executed, avoiding unrelated
    Spateo package imports. Preserve the upstream optimizer and grid_sample.
    """
    import ast
    import torch
    from typing import Optional
    from tqdm import tqdm
    source = ROOT / 'spateo-release-1.0.2/spateo/segmentation/align.py'
    tree = ast.parse(source.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name in ('AlignmentRefiner', 'RigidAlignmentRefiner')]
    if len(selected) != 2:
        raise ValueError('Upstream alignment classes not found')
    scope = dict(np=np, torch=torch, nn=torch.nn, F=torch.nn.functional,
                 Optional=Optional, tqdm=tqdm)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), scope)
    return scope['RigidAlignmentRefiner'], sha(source)


def align_one(job):
    root, tile = job
    out = root / 'patches' / tile['id']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'aligned.json').exists():
        return json.loads((out/'aligned.json').read_text())
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    r, c, height, width = [tile[k] for k in ['row', 'col', 'height', 'width']]
    stain = np.array(np.load(root/'stain_source.npy', mmap_mode='r')[r:r+height, c:c+width])
    rna = np.array(np.load(root/'pixel_umi.npy', mmap_mode='r')[r:r+height, c:c+width])
    result = dict(**tile, umis=int(rna.sum()), nuclei=0, numeric_guard_passed=True)
    if rna.max() == 0 or stain.max() == 0:
        labels = np.zeros(stain.shape, np.int32)
        result['status'] = 'no_RNA_or_no_stain; identity alignment, no pseudo-labelled nuclei'
    else:
        cls, source_sha = original_aligner()
        blurred = cv2.GaussianBlur(rna.astype(float), (5, 5), sigmaX=0, sigmaY=0)
        aligner = cls(blurred, stain)
        aligner.train(100)
        params = aligner.get_params()
        aligned = aligner.transform(stain, params).astype(stain.dtype)
        theta = params['theta']
        matrix = normalized_to_pixel(theta, stain.shape)
        corners = np.array([[0,0,1],[width-1,0,1],[0,height-1,1],[width-1,height-1,1]])
        displacement = np.linalg.norm(corners@matrix.T-corners[:,:2], axis=1)
        result.update(theta=theta.tolist(), source_sha256=source_sha,
            loss_before=aligner.history['loss'][0], loss_last=aligner.history['loss'][-1],
            max_corner_displacement=float(displacement.max()),
            numeric_guard_passed=bool(np.isfinite(theta).all() and displacement.max()<120),
            status='aligned with upstream RigidAlignmentRefiner; no downscale')
        stain = aligned
        if len(np.unique(stain)) < 4:
            labels = np.zeros(stain.shape, np.int32)
            result['status'] += '; fewer than 4 intensity levels, no nucleus segmentation'
        else:
            a = SimpleNamespace(layers={'stain': stain})
            st.cs.mask_nuclei_from_stain(a, otsu_classes=4, otsu_index=1)
            st.cs.find_peaks_from_mask(a, 'stain', 7)
            st.cs.watershed(a, 'stain', 5, out_layer='watershed_labels')
            labels = a.layers['watershed_labels'].astype(np.int32)
            result['nuclei'] = int(labels.max())
    np.save(out/'stain.npy', stain)
    np.save(out/'nuclei.npy', labels)
    write_json(out/'aligned.json', result)
    return result


def align_patches(root, workers=8):
    if (root/'patch_alignment_completed.json').exists():
        return
    import multiprocessing as mp
    meta = json.loads((root/'indexed.json').read_text())
    shape = meta['shape']
    tiles = [dict(id=f'r{r:05d}_c{c:05d}', row=r, col=c,
                  height=min(1200,shape[0]-r), width=min(1200,shape[1]-c))
             for r in range(0,shape[0],1200) for c in range(0,shape[1],1200)]
    with mp.get_context('spawn').Pool(workers) as pool:
        results = list(pool.imap(align_one, [(root,t) for t in tiles]))
    aligned = np.lib.format.open_memmap(root/'stain_aligned.npy', mode='w+', dtype=np.uint8, shape=tuple(shape))
    nuclei = np.lib.format.open_memmap(root/'nuclei.npy', mode='w+', dtype=np.int32, shape=tuple(shape))
    offset = 0
    for tile in results:
        r,c,h,w = [tile[k] for k in ['row','col','height','width']]
        directory = root/'patches'/tile['id']
        labels = np.load(directory/'nuclei.npy')
        aligned[r:r+h,c:c+w] = np.load(directory/'stain.npy')
        nuclei[r:r+h,c:c+w] = np.where(labels>0,labels+offset,0)
        tile['nucleus_id_offset'] = offset
        offset += int(labels.max())
    aligned.flush(); nuclei.flush()
    result = dict(patches=results, nuclei=offset, patch_size=1200,
        method='SCS patch grid, upstream Spateo RigidAlignmentRefiner, k=5, downscale=1, epochs=100',
        dependencies='bundled Spateo 1.0.2 alignment source; not a claim of historical environment identity',
        coordinate_frame='RNA unchanged; stain transformed separately per original 1200px patch',
        boundaries='patch seams retained as in original SCS; no invented cross-patch nucleus merges',
        numeric_guard_passed=all(t['numeric_guard_passed'] for t in results))
    write_json(root/'patch_alignment_completed.json',result)
    if not result['numeric_guard_passed']:
        raise ValueError('Some patches exceed alignment guard; review before training')


def pack(root, genept):
    if (root / 'prepared.json').exists():
        return
    meta = json.loads((root / 'indexed.json').read_text())
    grid = json.loads((root / 'expression_indexed.json').read_text())
    x = sparse.load_npz(root / 'all_expression.npz')
    genes = json.loads((root / 'genes.json').read_text())
    embeddings, _ = load_genept_embeddings(genept)
    table, remap, symbols, coverage = make_gene_lookup(genes, embeddings)
    coo = x.tocoo()
    keep = remap[coo.col] >= 0
    mapped = sparse.csr_matrix((coo.data[keep], (coo.row[keep], remap[coo.col[keep]])),
        shape=(x.shape[0], len(table)), dtype=np.int64)
    mapped.eliminate_zeros()
    coverage.update(total_umi=int(x.sum()), mapped_umi=int(mapped.sum()),
        mapped_umi_fraction=float(mapped.sum() / x.sum()),
        empty_bins_after_mapping=int(np.count_nonzero((np.diff(x.indptr)>0) & (np.diff(mapped.indptr)==0))),
        dictionary_sha256=sha(genept))
    np.save(root / 'gene_embeddings.npy', table)
    write_json(root / 'mapped_genes.json', symbols.tolist())
    for key, value in [('data', mapped.data.astype(np.float32)),
                       ('indices', mapped.indices.astype(np.int32)),
                       ('indptr', mapped.indptr.astype(np.int64))]:
        np.save(root / f'expression_{key}.npy', value)
    # ALL measured genes define the graph, including unmapped genes. This graph
    # remains fixed for the later 2000-gene ablation; missing vectors cannot move spots.
    occupied = np.diff(x.indptr) > 0
    del coo, mapped, x
    # Retain original patch-local neighbourhood boundaries, while sharing ONE
    # model and globally shuffling every patch's centre rows during training.
    alignment = json.loads((root/'patch_alignment_completed.json').read_text())
    if not alignment['numeric_guard_passed']:
        raise ValueError('Alignment guard failed')
    grids = occupied.reshape(grid['bin_shape'])
    pieces, patch_ranges = [], []
    position = 0
    for tile in alignment['patches']:
        r,c,h,w = [tile[k] for k in ['row','col','height','width']]
        gh,gw = math.ceil(h/3),math.ceil(w/3)
        local = neighbors_for(grids[r//3:r//3+gh,c//3:c//3+gw].ravel(), (gh,gw))
        global_nb = (local//gw+r//3)*grid['bin_shape'][1]+local%gw+c//3
        pieces.append(global_nb)
        patch_ranges.append(dict(**tile, sample_start=position, sample_stop=position+len(local)))
        position += len(local)
    nb = np.concatenate(pieces)
    write_json(root/'patch_ranges.json',patch_ranges)
    print(f'GRAPH {len(nb):,} centres, 50 tokens, all-gene occupancy', flush=True)
    np.save(root / 'neighbors.npy', nb)
    coords = np.stack(np.divmod(nb[:, 0], grid['bin_shape'][1]), -1) * 3
    labels = np.load(root / 'nuclei.npy', mmap_mode='r')
    stain = np.load(root / 'stain_aligned.npy', mmap_mode='r')
    ids = np.unique(labels)
    ids = ids[ids > 0]
    centers = np.zeros((int(labels.max())+1, 2))
    centers[ids] = ndimage.center_of_mass(labels > 0, labels, ids)
    nl = labels[coords[:, 0], coords[:, 1]]
    foreground = nl > 0
    distance = np.full(len(nb), np.inf)
    for tile in patch_ranges:
        start,stop=tile['sample_start'],tile['sample_stop']
        patch_ids=np.arange(tile['nucleus_id_offset']+1,tile['nucleus_id_offset']+tile['nuclei']+1)
        if len(patch_ids):
            distance[start:stop]=cKDTree(centers[patch_ids]).query(coords[start:stop],workers=2)[0]
    background = (~foreground) & (stain[coords[:, 0], coords[:, 1]] <= 10) & (distance > 30)
    delta = centers[nl] - coords
    angle = np.mod(np.arctan2(delta[:, 0], delta[:, 1]), 2*np.pi)
    direction = np.floor(angle * 16 / (2*np.pi)).astype(np.uint8)
    direction[~foreground] = 0
    split = np.full(len(nb), -1, np.int8)
    rng = np.random.RandomState(20260906)
    for group in [background] + [foreground & (direction == k) for k in range(16)]:
        rows = rng.permutation(np.flatnonzero(group))
        nval = len(rows) // 10
        split[rows[:nval]] = 1
        split[rows[nval:2*nval]] = 2
        split[rows[2*nval:]] = 0
    for key, value in [('coords', coords.astype(np.int32)), ('foreground', foreground.astype(np.uint8)),
                       ('directions', direction), ('split', split)]:
        np.save(root / f'{key}.npy', value)
    write_json(root / 'prepared.json', dict(complete=True, shape=meta['shape'],
        origin=meta['origin'], bin_shape=grid['bin_shape'], bin_size=3,
        centres=len(nb), train=int(sum(split==0)), validation=int(sum(split==1)), test=int(sum(split==2)),
        foreground=int(sum(foreground)), background=int(sum(background)), genept=coverage,
        neighbors_sha256=sha(root/'neighbors.npy'), split_sha256=sha(root/'split.npy'),
        source_sha256=meta['gem_sha256'],
        sampling='all eligible centres, no density cap; global stratified 80/10/10',
        validation_scope='within-slide interpolation; overlapping contexts, not external generalization',
        graph='all measured genes, 50 tokens within original patch and 10-ring radius; insufficient neighbours omitted',
        omitted_occupied_bins=int(sum(occupied)-len(nb))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT/'runs/SCS_stereo_whole_v1')
    p.add_argument('--stage', choices=['index', 'align', 'pack', 'all'], default='all')
    p.add_argument('--genept', type=Path, default=ROOT/'runs/ST19_shared_6000/genept_assets/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle')
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.stage in ('index', 'all'):
        meta = source_index(args.root)
        expression(args.root, meta)
    if args.stage in ('align', 'all'):
        align_patches(args.root)
    if args.stage in ('pack', 'all'):
        pack(args.root, args.genept)
    print(f'PREPARE {args.stage} finished in {time.time()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
