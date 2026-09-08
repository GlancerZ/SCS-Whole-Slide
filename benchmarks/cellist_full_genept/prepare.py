"""Shared all-gene RNA inputs, fixed image nuclei, and isolated Cellist shards."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import h5py
import numpy as np
import pandas as pd
from scipy import sparse, ndimage
from scipy.io import loadmat
from scipy.spatial import cKDTree
from skimage.draw import polygon
import tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.scs_paper.prepare import neighbors_for, sha, st
from benchmarks.scs_paper.stereo_whole_prepare import write_json
from optimizations.scs_streaming.genept import load_genept_embeddings, make_gene_lookup

BASE = ROOT / 'runs/Cellist_full_genept_v1'
GENEPT = ROOT / 'runs/ST19_shared_6000/genept_assets/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle'


def write_rna(directory, frame, genes, shape, nuclei, stain):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'prepared.json').exists():
        return
    coords = frame[['x', 'y']].drop_duplicates().sort_values(['x', 'y']).to_numpy(np.int32)
    keys = coords[:, 0].astype(np.int64) * shape[1] + coords[:, 1]
    source_keys = frame.x.to_numpy(np.int64) * shape[1] + frame.y.to_numpy(np.int64)
    rows = np.searchsorted(keys, source_keys)
    gids = pd.Index(genes).get_indexer(frame.geneID)
    if np.any(gids < 0) or len(frame) and not np.array_equal(keys[rows], source_keys):
        raise ValueError('Gene/coordinate mismatch in common input')
    x = sparse.csr_matrix((frame.MIDCounts.to_numpy(np.int64), (rows, gids)), shape=(len(coords), len(genes)))
    x.sum_duplicates(); x.eliminate_zeros()
    if int(x.sum()) != int(frame.MIDCounts.sum()):
        raise AssertionError('UMI conservation failure')
    if len(coords) and (np.any(coords < 0) or np.any(coords >= np.array(shape))):
        raise ValueError('Coordinates outside common image')
    np.save(directory / 'coords.npy', coords)
    sparse.save_npz(directory / 'rna.npz', x)
    np.save(directory / 'nuclei.npy', nuclei)
    np.save(directory / 'stain.npy', stain)
    frame[['geneID', 'x', 'y', 'MIDCounts']].to_csv(directory / 'expression.tsv', sep='\t', index=False)
    write_json(directory / 'prepared.json', dict(shape=list(shape), occupied_spots=len(coords),
        source_umis=int(x.sum()), source_records=len(frame), genes=len(genes),
        nuclei_sha256=sha(directory / 'nuclei.npy'), rna_sha256=sha(directory / 'rna.npz')))


def seqfish(fov):
    import anndata as ad
    source = ROOT / 'runs/SCS_paper_benchmark_v1/seqfish_source'
    sys.path.insert(0, str(source / 'vendor'))
    from roifile import ImagejRoi
    out = BASE / 'datasets' / f'seqfish_rep1_fov{fov}'
    if (out / 'prepared.json').exists() and (out / 'molecule_ground_truth.npz').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(source / 'seqFISH_NIH3T3_point_locations.zip')
    tot = loadmat(io.BytesIO(z.read('RNA_locations_run_1.mat')))['tot']
    genes = [str(g[0]) for g in loadmat(io.BytesIO(z.read('all_gene_Names.mat')))['allNames'].ravel()]
    points, columns, manual_ids = [], [], []
    for cell in range(tot.shape[1]):
        for g, values in enumerate(tot[fov, cell]):
            if not values.size:
                continue
            if np.any(values[:, 2] != 1):
                raise ValueError('Unexpected RNA z coordinate')
            points.append(values[:, :2]); columns.append(np.full(len(values), g, np.int32))
            manual_ids.append(np.full(len(values), cell+1, np.int32))
    xy, columns = np.concatenate(points), np.concatenate(columns)
    if np.any(xy < 0) or np.any(xy >= 2048):
        raise ValueError('RNA outside native image')
    factor, side = 5, math.ceil(2048 / 5)
    grid = np.floor(xy / factor).astype(np.int32)
    np.savez_compressed(out / 'molecule_ground_truth.npz', grid_xy=grid,
                        manual_cell=np.concatenate(manual_ids), native_xy=xy)
    x = sparse.csr_matrix((np.ones(len(xy), np.int64), (grid[:, 1] * side + grid[:, 0], columns)),
        shape=(side * side, len(genes)))
    x.sum_duplicates(); x.eliminate_zeros()
    with zipfile.ZipFile(source / 'DAPI_experiment1.zip') as images:
        with tifffile.TiffFile(io.BytesIO(images.read(f'final_background_experiment1/MMStack_Pos{fov}.ome.tif'))) as t:
            if t.series[0].axes != 'CZYX':
                raise ValueError('Unexpected image axes')
            native_dapi = t.series[0].asarray()[3].max(axis=0).astype(np.float32)
    # Block means on an edge-padded image; the final bin is clipped at native export.
    pad = side * factor - 2048
    dapi = np.pad(native_dapi, ((0, pad), (0, pad)), mode='edge').reshape(side, factor, side, factor).mean((1, 3))
    lo, hi = np.percentile(dapi, [1, 99.9])
    stain = np.clip((dapi-lo) / max(hi-lo, 1) * 255, 0, 255).astype(np.uint8)
    umi = np.asarray(x.sum(1)).reshape(side, side)
    a = ad.AnnData(sparse.csr_matrix(umi), layers={'stain': stain})
    st.cs.mask_nuclei_from_stain(a, otsu_classes=4, otsu_index=1)
    st.cs.find_peaks_from_mask(a, 'stain', 7)
    st.cs.watershed(a, 'stain', 5, out_layer='watershed_labels')
    labels = a.layers['watershed_labels'].astype(np.int32)
    ids = np.unique(labels[labels > 0])
    if not len(ids):
        raise ValueError('No shared image nuclei')
    centers = np.zeros((int(labels.max())+1, 2))
    centers[ids] = ndimage.center_of_mass(labels > 0, labels, ids)
    (out / 'data').mkdir(exist_ok=True)
    a.write_h5ad(out / 'data/spots0:0:0:0.h5ad')
    np.save(out / 'nuclei.npy', labels); np.save(out / 'stain_aligned.npy', stain)
    sparse.save_npz(out / 'all_expression.npz', x)
    write_json(out / 'genes.json', genes)
    embedding, _ = load_genept_embeddings(GENEPT)
    table, remap, symbols, coverage = make_gene_lookup(genes, embedding)
    coo = x.tocoo(); keep = remap[coo.col] >= 0
    mapped = sparse.csr_matrix((coo.data[keep], (coo.row[keep], remap[coo.col[keep]])), shape=(x.shape[0], len(table)))
    mapped.sum_duplicates(); mapped.eliminate_zeros()
    coverage.update(total_umi=int(x.sum()), mapped_umi=int(mapped.sum()), mapped_umi_fraction=float(mapped.sum()/x.sum()),
        dictionary_sha256=sha(GENEPT), mapping='case-insensitive exact symbols; not validated orthology',
        missing_genes=[g for g, index in zip(genes, remap) if index < 0])
    np.save(out / 'gene_embeddings.npy', table)
    write_json(out / 'mapped_genes.json', symbols.tolist())
    for key, values in [('data', mapped.data.astype(np.float32)), ('indices', mapped.indices.astype(np.int32)), ('indptr', mapped.indptr.astype(np.int64))]:
        np.save(out / f'expression_{key}.npy', values)
    nb = neighbors_for(np.diff(x.indptr) > 0, (side, side))
    coords = np.stack(np.divmod(nb[:, 0], side), -1).astype(np.int32)
    nl = labels[coords[:, 0], coords[:, 1]]; fg = nl > 0
    bg = (~fg) & (stain[coords[:, 0], coords[:, 1]] <= 10) & (cKDTree(centers[ids]).query(coords)[0] > 30)
    delta = centers[nl] - coords
    direction = np.floor(np.mod(np.arctan2(delta[:, 0], delta[:, 1]), 2*np.pi)*16/(2*np.pi)).astype(np.uint8)
    direction[~fg] = 0
    split = np.full(len(nb), -1, np.int8); rng = np.random.RandomState(20260906)
    for group in [bg] + [fg & (direction == k) for k in range(16)]:
        rows = rng.permutation(np.flatnonzero(group)); n = len(rows)//10
        split[rows[:n]] = 1; split[rows[n:2*n]] = 2; split[rows[2*n:]] = 0
    for key, values in [('neighbors', nb), ('coords', coords), ('directions', direction), ('foreground', fg.astype(np.uint8)), ('split', split)]:
        np.save(out / f'{key}.npy', values)
    patch = dict(id=f'fov{fov}', row=0, col=0, height=side, width=side, sample_start=0, sample_stop=len(nb), nucleus_id_offset=0, nuclei=int(labels.max()))
    write_json(out / 'patch_ranges.json', [patch])
    gt = np.zeros((2048, 2048), np.uint32); overlaps = np.zeros_like(gt, dtype=np.uint8)
    with zipfile.ZipFile(source / 'ROIs_Experiment1_NIH3T3.zip') as rois:
        names = sorted(n for n in rois.namelist() if n.startswith(f'ALL_Roi/RoiSet_Pos{fov}/') and n.endswith('.roi'))
        for ident, name in enumerate(names, 1):
            vertices = ImagejRoi.frombytes(rois.read(name)).coordinates()
            r, c = polygon(vertices[:, 1], vertices[:, 0], shape=gt.shape)
            gt[r, c] = ident; overlaps[r, c] += 1
    ignore = overlaps > 1; gt[ignore] = 0
    np.savez_compressed(out / 'ground_truth.npz', cells=gt, ignore=ignore, native_pixels_per_grid=factor)
    frame = pd.DataFrame({'geneID': np.asarray(genes)[coo.col], 'x': coo.row//side, 'y': coo.row%side, 'MIDCounts': coo.data})
    write_rna(out / 'cellist_inputs' / patch['id'], frame, genes, labels.shape, labels, stain)
    write_json(out / 'comparison_input.json', dict(platform='imaging', resolution_um=0.103*factor,
        native_pixels_per_grid=factor, native_shape=[2048,2048], common_nuclei='shared watershed',
        source=str(source), original_umis=len(xy), manual_cells=len(names), annotation_conditioned_RNA=True))
    write_json(out / 'prepared.json', dict(complete=True, shape=[side, side], origin=[0,0], bin_shape=[side,side], bin_size=1,
        native_pixels_per_grid=factor, radius_for_postprocessing=20, centres=len(nb),
        train=int(sum(split==0)), validation=int(sum(split==1)), test=int(sum(split==2)),
        neighbors_sha256=sha(out/'neighbors.npy'), split_sha256=sha(out/'split.npy'), genept=coverage,
        source_sha256=sha(source/'seqFISH_NIH3T3_point_locations.zip'),
        omitted_occupied_bins=int(np.count_nonzero(np.diff(x.indptr))-len(nb)),
        scope='full FOV supplied RNA; selected-cell-conditioned source; all genes, no HVG cutoff'))
    print(f'PREPARED {out.name}: {len(nb)} centers, {len(table)} mapped genes', flush=True)


def barcoding(name):
    data = BASE / 'datasets' / name
    marker = data / 'cellist_inputs_completed.json'
    if marker.exists(): return
    patches = json.loads((data / 'patch_ranges.json').read_text())
    meta = json.loads((data / 'prepared.json').read_text())
    genes = json.loads((data / 'genes.json').read_text())
    nuclei = np.load(data / 'nuclei.npy', mmap_mode='r')
    stain = np.load(data / 'stain_aligned.npy', mmap_mode='r')
    if name != 'stereo_mouse_brain':
        source = ROOT / 'runs/SCS_paper_benchmark_v1' / name
        rna = sparse.load_npz(source / 'rna_pixels.npz').tocoo()
        r, c = np.divmod(rna.row, meta['shape'][1])
        frame = pd.DataFrame({'geneID': np.asarray(genes)[rna.col], 'x': r, 'y': c, 'MIDCounts': rna.data})
        write_rna(data/'cellist_inputs'/patches[0]['id'], frame, genes, meta['shape'], np.array(nuclei), np.array(stain))
    else:
        # One streaming pass, never repeatedly scan the whole brain per patch.
        shard_marker = data / 'cellist_shards_completed.json'
        if not shard_marker.exists():
            for p in patches:
                d = data / 'cellist_inputs' / p['id']; d.mkdir(parents=True, exist_ok=True)
                (d / 'raw_shard.tsv').write_text('geneID\tx\ty\tMIDCounts\n')
            by_position = {(p['row']//1200, p['col']//1200): p for p in patches}
            with h5py.File(data / 'records.h5') as h:
                for start in range(0, len(h['x']), 1000000):
                    sl = slice(start, start+1000000)
                    r = h['x'][sl].astype(np.int64)-meta['origin'][0]
                    c = h['y'][sl].astype(np.int64)-meta['origin'][1]
                    frame = pd.DataFrame({'geneID': np.asarray(genes)[h['gene'][sl]], 'x': r, 'y': c, 'MIDCounts': h['count'][sl], 'pr': r//1200, 'pc': c//1200})
                    for pos, part in frame.groupby(['pr','pc']):
                        p = by_position[pos]
                        part = part[['geneID','x','y','MIDCounts']].copy()
                        part.x -= p['row']; part.y -= p['col']
                        part.to_csv(data/'cellist_inputs'/p['id']/'raw_shard.tsv', mode='a', sep='\t', header=False, index=False)
                    print(f'BRAIN SHARD {start:,}', flush=True)
            write_json(shard_marker, dict(complete=True))
        total = 0
        for p in patches:
            d = data/'cellist_inputs'/p['id']
            r,c,h,w = [p[k] for k in ('row','col','height','width')]
            if not (d/'prepared.json').exists():
                frame = pd.read_csv(d/'raw_shard.tsv', sep='\t')
                write_rna(d, frame, genes, [h,w], np.array(nuclei[r:r+h,c:c+w]), np.array(stain[r:r+h,c:c+w]))
            total += json.loads((d/'prepared.json').read_text())['source_umis']
        if total != meta['genept']['total_umi']:
            raise AssertionError('Brain shard UMIs do not sum to full source')
    write_json(marker, dict(complete=True, patches=len(patches), common_nuclei_sha256=sha(data/'nuclei.npy')))
    print(f'CELLIST INPUTS {name}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset', required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Compute node required')
    if args.dataset.startswith('seqfish_rep1_fov'):
        seqfish(int(args.dataset[-1]))
    else:
        barcoding(args.dataset)
