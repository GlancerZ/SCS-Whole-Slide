"""CPU postprocessing and Cellist-style paired evaluation of every requested region."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, ndimage

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.scs_paper.evaluate import segment
from benchmarks.scs_paper.stereo_whole_prepare import write_json, sha
from benchmarks.cellist_full_genept.metrics import aggregate, random_correlations, cross_correlation, paired_purity, transcript_iou

BASE=ROOT/'runs/Cellist_full_genept_v1'
METHOD='genept_all_scale4'
POST_METHOD='genept_paperpost_scale4'
CELLIST='cellist_paper'


def prediction_file(data,patch):
    local=data/'segmentation_patches'/patch['id']/POST_METHOD/'segmentation.npz'
    if local.exists(): return local
    return local


def postprocess(task):
    data,patch=task
    existing=prediction_file(data,patch)
    if existing.exists(): return str(existing)
    if not (data/METHOD/'inference_completed.json').exists():
        raise RuntimeError('Full inference not finished')
    meta=json.loads((data/'prepared.json').read_text())
    source=data/'cellist_inputs'/patch['id']
    tile=data/'segmentation_patches'/patch['id']
    out=tile/POST_METHOD; out.mkdir(parents=True,exist_ok=True)
    (tile/'data').mkdir(exist_ok=True)
    nuclei=np.load(source/'nuclei.npy')
    ids=np.unique(np.r_[0,nuclei.ravel()]); local_nuclei=np.searchsorted(ids,nuclei).astype(np.int32)
    stain=np.load(source/'stain.npy')
    a=ad.AnnData(sparse.csr_matrix(stain.shape,dtype=np.int32),layers={'stain':stain,'watershed_labels':local_nuclei})
    a.write_h5ad(tile/'data/spots0:0:0:0.h5ad')
    radius=40 if data.name.startswith('seqfish') else (15 if data.name=='stereo_mouse_brain' else 25)
    write_json(tile/'prepared.json',dict(bin_size=meta['bin_size'],shape=list(stain.shape),
        radius_for_postprocessing=radius,parameter_source='Cellist author SCS_DIAMETER: brain=15, liver=25, seqFISH=200 native pixels / 5'))
    start,stop=patch['sample_start'],patch['sample_stop']
    if start==stop:
        np.savez_compressed(out/'segmentation.npz',cells=np.zeros(stain.shape,np.uint32),nuclei=nuclei)
        write_json(out/'segmentation_completed.json',dict(complete=True,status='no_supported_centres',covered=False))
        return str(out/'segmentation.npz')
    info=json.loads((data/METHOD/'inference_completed.json').read_text())
    if info['world_size']!=1: raise ValueError('Expected one GPU worker per dataset')
    coords=np.load(data/'coords.npy',mmap_mode='r')[start:stop]-[patch['row'],patch['col']]
    logits=np.load(data/METHOD/'logits_rank0.npy',mmap_mode='r')[start:stop]
    probability=np.load(data/METHOD/'foreground_rank0.npy',mmap_mode='r')[start:stop]
    with (out/'spot_prediction.txt').open('w') as f:
        for xy,dl,b in zip(coords,logits,probability):
            f.write(f'{xy[0]}\t{xy[1]}\t{b:.8g}\t'+':'.join(f'{v:.8g}' for v in dl)+'\n')
    segment(tile,POST_METHOD)
    return str(out/'segmentation.npz')


def common_hvg(x,nuclear,n_top=1500):
    import scanpy as sc
    ids=np.unique(np.r_[0,nuclear]); codes=np.searchsorted(ids,nuclear)
    nuc=aggregate(x,codes)[1:]
    valid=np.asarray(nuc.sum(0)).ravel()>0
    kept=np.flatnonzero(valid)
    if len(kept)<2:
        return np.arange(x.shape[1]), 'No usable nuclear HVG fit; all measured genes, common to both methods'
    if len(kept)<=n_top or nuc.shape[0]<3:
        return kept, f'all nucleus-expressed genes; at most {n_top} or fewer than 3 nuclei'
    data=ad.AnnData(nuc[:,kept].astype(np.float64))
    sc.pp.highly_variable_genes(data,n_top_genes=n_top,flavor='seurat_v3',span=1.)
    return kept[data.var.highly_variable.to_numpy()], 'Seurat-v3 span=1; common image-nucleus counts per evaluation patch'


def metric_summary(frame,column,eligible=None):
    values=frame[column] if eligible is None else frame.loc[eligible,column]
    values=pd.to_numeric(values,errors='coerce')
    values=values[np.isfinite(values)]
    return dict(n=len(values),mean=float(values.mean()) if len(values) else None,
        median=float(values.median()) if len(values) else None)


def evaluate_patch(task):
    data,patch=task
    out=data/'evaluation'/patch['id']; out.mkdir(parents=True,exist_ok=True)
    marker=out/'completed.json'
    if marker.exists(): return json.loads(marker.read_text())
    start=time.monotonic()
    source=data/'cellist_inputs'/patch['id']
    if not (data/'segmentation_patches'/patch['id']/CELLIST/'segmentation_completed.json').exists():
        raise RuntimeError('Cellist patch not complete')
    coords=np.load(source/'coords.npy'); x=sparse.load_npz(source/'rna.npz').astype(np.int64)
    nuclei=np.load(source/'nuclei.npy'); nuclear=nuclei[coords[:,0],coords[:,1]]
    a=np.load(data/'segmentation_patches'/patch['id']/CELLIST/'assignments.npy')
    pred=np.load(prediction_file(data,patch))['cells']
    b=pred[coords[:,0],coords[:,1]]
    if len(a)!=len(coords) or pred.shape!=nuclei.shape: raise ValueError('Coordinate/mask mismatch')
    input_meta=json.loads((source/'prepared.json').read_text())
    if int(x.sum())!=input_meta['source_umis']: raise AssertionError('RNA conservation failure')
    if not len(coords):
        result=dict(patch=patch['id'],complete=True,source_umis=0,source_spots=0,methods={},empty_RNA=True)
        write_json(marker,result); return result
    liver=data.name in ('2104','2105','2106','2107')
    genes, hvg_rule=common_hvg(x,nuclear,1000 if liver else 1500)
    if len(genes)<2: raise ValueError('Fewer than two common evaluation genes; cannot compute correlation')
    np.save(out/'evaluation_hvg_indices.npy',genes)
    expression=x[:,genes].tocsr()
    results={}; frames=[]
    per_spot=np.asarray(x.sum(1)).ravel()
    for method,labels in [('cellist',a),('full_genept_scale4',b)]:
        frame=pd.DataFrame(random_correlations(expression,coords,labels))
        # Library's counts above are evaluation-HVG counts; preserve all-gene totals separately.
        ids=np.unique(np.r_[0,labels]); full=aggregate(x,np.searchsorted(ids,labels))
        frame['n_umis']=np.asarray(full.sum(1)).ravel()[1:]
        frame['n_genes']=np.diff(full.indptr)[1:]
        frame['method']=method; frame['patch']=patch['id']; frames.append(frame)
        results[method]=dict(cells=len(frame),assigned_spots=int(sum(labels>0)),assigned_umis=int(per_spot[labels>0].sum()),
            random_all_cells=metric_summary(frame,'random_correlation'),
            random_nspot_gt100=metric_summary(frame,'random_correlation',frame.n_spots>100),
            directional_nspot_gt100=metric_summary(frame,'directional_correlation',frame.n_spots>100))
    pd.concat(frames,ignore_index=True).to_csv(out/'per_cell.csv.gz',index=False)
    cross={}
    for name,s,t in [('genept_to_cellist',b,a),('cellist_to_genept',a,b)]:
        values=cross_correlation(expression,s,t)
        keys=['target_cell','source_correlation','target_correlation','overlap_umis','source_unique_umis','target_unique_umis','sensitivity_100umi']
        frame=pd.DataFrame({k:values[k] for k in keys})
        frame.to_csv(out/f'{name}.csv.gz',index=False)
        cross[name]={k:v for k,v in values.items() if k not in keys}
        cross[name].update(source=metric_summary(frame,'source_correlation'),target=metric_summary(frame,'target_correlation'),
            difference_source_minus_target=metric_summary(frame.assign(difference=frame.source_correlation-frame.target_correlation),'difference'))
    centers=np.zeros((int(nuclei.max())+1,2)); ids=np.unique(nuclei[nuclei>0])
    if len(ids): centers[ids]=ndimage.center_of_mass(nuclei>0,nuclei,ids)
    config=json.loads((data/'comparison_input.json').read_text())
    purity_rows,purity_meta=paired_purity(expression,coords,nuclear,centers,a,b,config['resolution_um'],
        neigh_dist_um=3.0 if liver else 2.5,half_width_units=20)
    purity=pd.DataFrame(purity_rows,columns=['nucleus','cell_first','cell_second','n_spots_first','n_spots_second','purity_first','purity_second','neighborhood_spots'])
    purity.to_csv(out/'purity.csv.gz',index=False)
    purity_meta.update(cellist=metric_summary(purity,'purity_first'),genept=metric_summary(purity,'purity_second'))
    result=dict(patch=patch['id'],complete=True,source_umis=int(x.sum()),source_spots=len(coords),
        methods=results,cross_correlation=cross,purity=purity_meta,evaluation_genes=len(genes),hvg_rule=hvg_rule,
        source_rna_sha256=input_meta['rna_sha256'],source_nuclei_sha256=input_meta['nuclei_sha256'],seconds=time.monotonic()-start,
        scope='common complete observed RNA support, including unassigned spots',
        caveats=['Expression coherence is not independent segmentation truth.',
            'Evaluation HVGs are shared image-nucleus HVGs per patch, not method-selected genes.',
            'Purity pairs share eligible nuclei; repeated cell pairs are reported.',
            'Cross matching uses observed-spot overlap and source-to-target unions, in both directions.'])
    write_json(marker,result)
    print(f'EVALUATED {data.name}/{patch["id"]}',flush=True)
    return result


def finish(data,patches):
    results=[json.loads((data/'evaluation'/p['id']/'completed.json').read_text()) for p in patches]
    frame=pd.concat([pd.read_csv(data/'evaluation'/p['id']/'per_cell.csv.gz') for p,r in zip(patches,results) if not r.get('empty_RNA')],ignore_index=True)
    frame.to_csv(data/'evaluation/per_cell.csv.gz',index=False)
    methods={}
    total=sum(r['source_umis'] for r in results); spots=sum(r['source_spots'] for r in results)
    for method,f in frame.groupby('method'):
        assigned=sum(r.get('methods',{}).get(method,{}).get('assigned_umis',0) for r in results)
        methods[method]=dict(cells=len(f),assigned_umis=assigned,assigned_umi_fraction=assigned/total,
            median_umis=float(f.n_umis.median()),median_genes=float(f.n_genes.median()),median_spots=float(f.n_spots.median()),
            random=metric_summary(f,'random_correlation',f.n_spots>100),directional=metric_summary(f,'directional_correlation',f.n_spots>100))
    cross={}
    for direction in ['genept_to_cellist','cellist_to_genept']:
        rows=pd.concat([pd.read_csv(data/'evaluation'/p['id']/f'{direction}.csv.gz') for p,r in zip(patches,results) if not r.get('empty_RNA')],ignore_index=True)
        cross[direction]=dict(source=metric_summary(rows,'source_correlation'),target=metric_summary(rows,'target_correlation'),
            paired_difference=metric_summary(rows.assign(delta=rows.source_correlation-rows.target_correlation),'delta'))
    purity=pd.concat([pd.read_csv(data/'evaluation'/p['id']/'purity.csv.gz').assign(patch=p['id']) for p,r in zip(patches,results) if not r.get('empty_RNA')],ignore_index=True)
    purity.to_csv(data/'evaluation/purity.csv.gz',index=False)
    summary=dict(dataset=data.name,complete=True,patches=len(patches),source_umis=total,source_spots=spots,
        methods=methods,cross_correlation=cross,purity=dict(cellist=metric_summary(purity,'purity_first'),genept=metric_summary(purity,'purity_second')),
        physical_resolution_um=json.loads((data/'comparison_input.json').read_text())['resolution_um'],
        limitations=['Same image watershed priors; not an exact Cellist-native Cellpose replication.',
            'Patch boundaries are shared and unmerged; within-slide correlated cells are not independent replicates.',
            'Official GenePT exact-symbol mapping may omit mouse genes; coverage is separately reported.'])
    summary['genept_coverage']=json.loads((data/'prepared.json').read_text())['genept']
    if (data/'molecule_ground_truth.npz').exists():
        ground=np.load(data/'molecule_ground_truth.npz'); grid=ground['grid_xy']
        manual=ground['manual_cell']; p=patches[0]
        cm=np.load(data/'segmentation_patches'/p['id']/CELLIST/'segmentation.npz')['cells']
        gm=np.load(prediction_file(data,p))['cells']
        summary['transcript_iou']={m:transcript_iou(manual,mask[grid[:,1],grid[:,0]]) for m,mask in [('cellist',cm),('full_genept_scale4',gm)]}
    write_json(data/'comparison_completed.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--post-workers',type=int,default=8)
    p.add_argument('--postprocess-only',action='store_true');p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'): raise RuntimeError('Separate CPU-only allocation required')
    data=BASE/'datasets'/args.dataset;patches=json.loads((data/'patch_ranges.json').read_text())
    if not args.evaluate_only:
        with ProcessPoolExecutor(max_workers=args.post_workers) as pool:list(pool.map(postprocess,[(data,p) for p in patches]))
    if not args.postprocess_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(evaluate_patch,[(data,p) for p in patches]))
        finish(data,patches)
