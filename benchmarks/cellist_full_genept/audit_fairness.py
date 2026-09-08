"""Read-only audit of completed benchmark; preserves primary outputs."""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/Cellist_full_genept_v1'

def run():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):raise RuntimeError('CPU allocation required')
    result={}
    for name in ['stereo_mouse_brain','2104','2105','2106','2107','seqfish_rep1_fov0','seqfish_rep1_fov1']:
        d=BASE/'datasets'/name
        meta=json.loads((d/'prepared.json').read_text());summary=json.loads((d/'comparison_completed.json').read_text())
        patches=json.loads((d/'patch_ranges.json').read_text());res=json.loads((d/'comparison_input.json').read_text())['resolution_um']
        cells=pd.read_csv(d/'evaluation/per_cell.csv.gz');purity=pd.read_csv(d/'evaluation/purity.csv.gz')
        v={'genept_bin_size':meta['bin_size'],'native_resolution_um':res,'genept_input_spacing_um':res*meta['bin_size'],
           'cellist_input_spacing_um':res,'methods':{},'cross_100hvgUMI':{},'fallback_patches':[]}
        for method,f in cells.groupby('method'):
            q=f[f.n_spots>100]
            v['methods'][method]={'cells':len(f),'qc_cells':len(q),'qc_retention':len(q)/len(f),
                'qc_median_spots':float(q.n_spots.median()),'qc_median_umis':float(q.n_umis.median())}
        eligible=cells[(cells.n_spots>100)&np.isfinite(cells.random_correlation)].copy()
        eligible['count_bin']=np.floor(np.log2(eligible.n_umis.clip(lower=1))).astype(int)
        groups=eligible.groupby(['count_bin','method']).random_correlation.agg(['mean','count']).unstack('method')
        paired=groups.dropna()
        if len(paired):
            weights=paired['count'].min(axis=1)
            delta=paired['mean']['full_genept_scale4']-paired['mean']['cellist']
            v['RNA_depth_stratified_diagnostic']={'common_log2_UMI_bins':len(paired),
                'weighted_GenePT_minus_Cellist':float(np.average(delta,weights=weights)),
                'definition':'Within log2 total-UMI strata; weights=min(method cell counts); descriptive, not randomized matching'}
        unique=purity.drop_duplicates(['patch','cell_first','cell_second'])
        v['purity_pairs']={'nucleus_rows':len(purity),'unique_cell_pairs':len(unique),
            'unique_pair_mean_cellist':float(unique.purity_first.mean()),'unique_pair_mean_genept':float(unique.purity_second.mean())}
        genes=json.loads((d/'genes.json').read_text());mapped=set(json.loads((d/'mapped_genes.json').read_text()))
        hvg_total=hvg_mapped=0
        for p in patches:
            ep=d/'evaluation'/p['id'];cp=d/'segmentation_patches'/p['id']/'cellist_paper'
            hvg=ep/'evaluation_hvg_indices.npy'
            if hvg.exists():
                indices=np.load(hvg);hvg_total+=len(indices);hvg_mapped+=sum(genes[i].upper() in mapped for i in indices)
            fallback=cp/'hvg_compatibility.json'
            if fallback.exists() and json.loads(fallback.read_text())['fallbacks']:
                mark=json.loads((cp/'segmentation_completed.json').read_text())
                v['fallback_patches'].append({'patch':p['id'],'source_umis':mark['source_umis'],'cells':mark['cells']})
        v['evaluation_HVG_mapping_fraction_gene_patch_entries']=hvg_mapped/hvg_total
        v['fallback_source_UMI_fraction']=sum(p['source_umis'] for p in v['fallback_patches'])/summary['source_umis']
        for direction in ['genept_to_cellist','cellist_to_genept']:
            frames=[]
            for p in patches:
                path=d/'evaluation'/p['id']/f'{direction}.csv.gz'
                if path.exists():frames.append(pd.read_csv(path))
            f=pd.concat(frames,ignore_index=True);q=f[f.sensitivity_100umi.astype(bool)]
            delta=q.source_correlation-q.target_correlation
            if direction=='cellist_to_genept':delta=-delta
            v['cross_100hvgUMI'][direction]={'all_eligible':len(f),'over100_each_part':len(q),
                'mean_GenePT_minus_Cellist':float(delta.mean()) if len(q) else None}
        if name.startswith('seqfish'):
            gt=np.load(d/'molecule_ground_truth.npz');xy=gt['grid_xy'];coords=np.load(d/'coords.npy')
            fg=np.load(d/'foreground.npy');split=np.load(d/'split.npy')
            split_image=np.full(tuple(meta['shape']),-2,np.int8);foreground_image=np.zeros(tuple(meta['shape']),bool)
            split_image[coords[:,0],coords[:,1]]=split;foreground_image[coords[:,0],coords[:,1]]=fg.astype(bool)
            gs=split_image[xy[:,1],xy[:,0]];gf=foreground_image[xy[:,1],xy[:,0]]
            v['manual_RNA_pseudolabel_audit']={'molecules':len(xy),'all_have_manual_cell':bool(np.all(gt['manual_cell']>0)),
                'fraction_given_background_pseudolabel':float(np.mean((gs>=0)&(~gf))),
                'fraction_training_background':float(np.mean((gs==0)&(~gf))),
                'fraction_validation_background':float(np.mean((gs==1)&(~gf))),
                'fraction_test_background':float(np.mean((gs==2)&(~gf))),
                'fraction_nucleus_positive':float(np.mean(gf)),
                'fraction_ignored_pseudolabel':float(np.mean(gs==-1)),
                'background_distance_um':30*res}
        result[name]=v
    out=BASE/'fairness_audit';out.mkdir(exist_ok=True)
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':run()
