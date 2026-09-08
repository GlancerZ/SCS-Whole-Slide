"""Official CPU Cellist on shared nuclei and full RNA inputs, with strict export QA."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import traceback
import subprocess
import threading

import numpy as np
import pandas as pd
from scipy import sparse
from skimage.measure import regionprops_table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from cellist_cpu_stage import load_core, save_json
BASE = ROOT / 'runs/Cellist_full_genept_v1'


def protect_degenerate_hvg(seg, audit):
    """Avoid native loess segfault for <3 nuclei; serialize non-reentrant fitting."""
    original = seg.get_hvg
    lock = threading.Lock()
    def guarded(matrix, genes, cells, n_top):
        with lock:
            if len(cells) < 3:
                positive = np.asarray(matrix.sum(axis=1)).ravel() > 0
                selected = np.asarray(genes)[positive].tolist()
                audit.append(dict(nuclei=len(cells), selected_genes=len(selected),
                    reason='Seurat-v3 variance fit undefined with fewer than 3 nuclei; all nucleus-expressed genes'))
                return selected
            return original(matrix, genes, cells, n_top)
    seg.get_hvg = guarded


def run_patch(task):
    data, patch = task
    source = data/'cellist_inputs'/patch['id']
    out = data/'segmentation_patches'/patch['id']/'cellist_paper'
    out.mkdir(parents=True, exist_ok=True)
    marker = out/'segmentation_completed.json'
    if marker.exists(): return patch['id'], 'completed'
    start = time.monotonic()
    try:
        ws, seg = load_core()
        hvg_fallbacks = []
        protect_degenerate_hvg(seg, hvg_fallbacks)
        from Cellist.IO import gem_to_mat
        meta = json.loads((source/'prepared.json').read_text())
        config = json.loads((data/'comparison_input.json').read_text())
        global_nuclei = np.load(source/'nuclei.npy')
        original_ids = np.unique(np.r_[0, global_nuclei.ravel()])
        nuclei = np.searchsorted(original_ids, global_nuclei).astype(np.uint32)
        coords = np.load(source/'coords.npy')
        nuclear_spots = nuclei[coords[:,0],coords[:,1]]
        sizes = np.bincount(nuclear_spots, minlength=len(original_ids))
        nuc_dir = out/'nuclei'; nuc_dir.mkdir(exist_ok=True)
        sparse.save_npz(nuc_dir/'shared_Watershed_nucleus_matrix.npz', sparse.csc_matrix(nuclei))
        props = regionprops_table(nuclei, properties=['label','area','centroid','equivalent_diameter_area'])
        pd.DataFrame(props).to_csv(nuc_dir/'shared_Watershed_nucleus_property.txt', sep='\t', index=False)
        assignments = np.zeros(len(coords), np.uint32)
        no_eligible = not np.any(sizes[1:] >= 20)
        if len(coords) and not no_eligible:
            frame = pd.read_csv(source/'expression.tsv', sep='\t')
            gem = gem_to_mat(frame, str(nuc_dir/'shared_bin1.h5'), countname='MIDCounts')
            nc = ws.write_segmentation_coord(nuclei, gem[['x','y','x_y']].drop_duplicates('x_y'), str(nuc_dir), 'shared')
            ws.write_segmentation_cell(nc, gem, 'shared', str(nuc_dir))
            # Defaults for barcoding; published imaging tutorial uses radius 10 and no noise rejection.
            imaging = config['platform'] == 'imaging'
            liver = data.name in ('2104','2105','2106','2107')
            parameters = dict(alpha=0.9, sigma=1.0, beta=5.0,
                max_dist=10 if imaging else (25 if liver else 15),
                noise_prop=0.0 if imaging else (0.3 if liver else 0.45),
                neigh_dist=3.0 if liver else 2.5)
            seg.Cellist(platform=config['platform'], resolution=config['resolution_um'], nucleus_seg_method='Watershed',
                props_file=str(nuc_dir/'shared_Watershed_nucleus_property.txt'),
                nucleus_count_h5_file=str(nuc_dir/'shared_Watershed_segmentation_cell_count.h5'),
                nucleus_coord_file=str(nuc_dir/'shared_Watershed_nucleus_coord.txt'),
                all_spot_count_h5_file=str(nuc_dir/'shared_bin1.h5'), spot_expr_file=str(source/'expression.tsv'),
                patch_data_dir=None, num_workers=1, gene_use='HVG', two_step=False, cyto=False,
                max_dist_s1_scale=0.5, noise_prop_s1=0.4,
                out_dir=str(out/'official'), out_prefix='shared', **parameters)
            files = list((out/'official').glob('*/shared_Cellist_segmentation.txt'))
            if len(files) != 1 or not (files[0].parent/'parameters.json').exists():
                raise RuntimeError('Missing official Cellist completion/unique output')
            result = pd.read_csv(files[0], sep='\t').sort_values(['x','y'])
            if result.duplicated(['x','y']).any() or not np.array_equal(result[['x','y']].to_numpy(), coords):
                raise ValueError('Cellist omitted or duplicated source coordinates')
            ids = result.Cellist.fillna(0).to_numpy()
            if np.any(~np.isfinite(ids)) or np.any(ids < 0) or np.any(ids != ids.astype(np.uint32)):
                raise ValueError('Invalid cell IDs')
            assignments = ids.astype(np.uint32)
        labels = np.zeros(tuple(meta['shape']), np.uint32)
        labels[coords[:,0],coords[:,1]] = assignments
        rna = sparse.load_npz(source/'rna.npz')
        per_spot = np.asarray(rna.sum(1, dtype=np.int64)).ravel()
        if int(per_spot.sum()) != meta['source_umis']:
            raise AssertionError('Common RNA input changed')
        np.save(out/'assignments.npy', assignments)
        np.savez_compressed(out/'segmentation.npz', cells=labels, nuclei=global_nuclei)
        save_json(marker, dict(complete=True, method='Cellist 1.1.1', host=os.uname().nodename,
            job=os.environ.get('SLURM_JOB_ID'), cpu_only=True, seconds=time.monotonic()-start,
            cells=int(len(np.unique(assignments[assignments>0]))), observed_spots=len(coords),
            assigned_spots=int(sum(assignments>0)), source_umis=int(per_spot.sum()),
            assigned_umis=int(per_spot[assignments>0].sum()), source_nuclei_sha256=meta['nuclei_sha256'],
            status='no_eligible_nuclei' if no_eligible else 'segmented',
            reason='Cellist requires >=20 observed nuclear spots' if no_eligible else None,
            representation='labels at every observed RNA spot; zero=unassigned',
            shared_nucleus_prior=True, native_method_seed_reproduction=False))
        save_json(out/'hvg_compatibility.json', dict(fallbacks=hvg_fallbacks,
            rule='All nucleus-expressed genes only when fewer than 3 nuclei; other HVG fits unchanged and serialized'))
        (out/'failed.json').unlink(missing_ok=True)
        if not no_eligible:
            save_json(out/'paper_parameters.json',dict(**parameters, source='Zenodo 18638251 Fig2/Fig3 Config',
                platform=config['platform'],resolution_um=config['resolution_um']))
        print(f'CELLIST {data.name}/{patch["id"]} complete', flush=True)
        return patch['id'], 'completed'
    except Exception:
        save_json(out/'failed.json', dict(error=traceback.format_exc(), seconds=time.monotonic()-start))
        print(traceback.format_exc(), flush=True)
        return patch['id'], 'failed'


def isolated_patch(task):
    data, patch = task
    out = data/'segmentation_patches'/patch['id']/'cellist_paper'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'segmentation_completed.json').exists(): return patch['id'], 'completed'
    with (out/'process.log').open('a') as log:
        result = subprocess.run([sys.executable, '-u', '-X', 'faulthandler', __file__,
            '--dataset', data.name, '--patch', patch['id']], stdout=log, stderr=subprocess.STDOUT)
    okay = result.returncode == 0 and (out/'segmentation_completed.json').exists()
    if not okay:
        save_json(out/'process_failed.json', dict(returncode=result.returncode, log=str(out/'process.log')))
    print(f'CELLIST {data.name}/{patch["id"]} {"complete" if okay else "FAILED"}', flush=True)
    return patch['id'], 'completed' if okay else 'failed'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', required=True); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--patch')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
        raise RuntimeError('Cellist benchmark requires a separate CPU-only allocation')
    data = BASE/'datasets'/a.dataset
    patches = json.loads((data/'patch_ranges.json').read_text())
    if a.patch:
        patch = next(p for p in patches if p['id'] == a.patch)
        _, status = run_patch((data, patch))
        sys.exit(0 if status == 'completed' else 1)
    lock = (data/'cellist.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = dict(pool.map(isolated_patch, [(data, patch) for patch in patches]))
    save_json(data/'cellist_status.json', dict(complete=all(s=='completed' for s in results.values()), patches=results))
    if not all(s=='completed' for s in results.values()):
        raise RuntimeError('Some Cellist patches failed; see per-patch failed.json')
    lock.close()


if __name__ == '__main__': main()
