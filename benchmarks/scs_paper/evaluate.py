"""Paper-style postprocessing and nucleus-matched RNA consistency evaluation.

RNA consistency is a paired proxy, not a ground-truth segmentation accuracy.
Every compared intersection AND both unique regions must have >=100 UMIs.
Counts accumulate in int64 (upstream evaluator's int8 overwrite bug is not copied).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'SCS'))
sys.path.insert(0,str(ROOT))
from src import postprocessing as post
from benchmarks.scs_paper.prepare import sha


def segment(data, method, radius=20):
    out=data/method
    final=out/'segmentation.npz'
    if final.exists():
        return
    started=time.time()
    if method=='direction_prior_only':
        out.mkdir(exist_ok=False)
        p=np.load(data/'scs_reference'/'all_predictions.npz')
        s=np.load(data/'samples.npz')
        # Same foreground as trained SCS: only the direction network is ablated.
        with (out/'spot_prediction.txt').open('w') as f:
            for (r,c),b in zip(s['coords'],p['foreground_probability']):
                f.write(f'{r}\t{c}\t{b:.8g}\t'+':'.join(['0']*16)+'\n')
    a=ad.read_h5ad(data/'data'/'spots0:0:0:0.h5ad')
    metadata=json.loads((data/'prepared.json').read_text())
    bin_size=int(metadata.get('bin_size',3))
    radius=int(metadata.get('radius_for_postprocessing',radius))
    intensity,dx,dy,*_=post.read_gradient(str(out/'spot_prediction.txt'),a,a.shape,bin_size,radius)
    seg,sinks,*_=post.gvf_tracking(dx,dy,intensity,bin_size)
    merged=post.merge_sinks(seg,sinks,bin_size)
    # Exact expansion of bin anchor labels into non-overlapping 3x3 blocks.
    merged=np.repeat(np.repeat(merged[::bin_size,::bin_size],bin_size,axis=0),bin_size,axis=1)[:a.shape[0],:a.shape[1]]
    merged=post.remove_small_cells(merged).astype(np.uint32)
    np.savez_compressed(final,cells=merged,nuclei=a.layers['watershed_labels'].astype(np.uint32))
    (out/'segmentation_completed.json').write_text(json.dumps(dict(
        seconds=time.time()-started, postprocessing_sha256=sha(ROOT/'SCS/src/postprocessing.py'),
        radius=radius,bin_size=bin_size,foreground_threshold=.1,diffusion_passes=2,
        implementation='workspace SCS postprocessing; equivalent vectorized bin expansion'),indent=2))
    print(f'SEGMENTED {data.name}/{method}',flush=True)


def match_nuclei(nuclei, cells):
    n,c=nuclei.ravel(),cells.ravel()
    keep=(n>0)&(c>0)
    pairs,counts=np.unique(np.stack((n[keep],c[keep]),-1),axis=0,return_counts=True)
    best={}
    for (nu,cell),count in zip(pairs,counts):
        if int(nu) not in best or count>best[int(nu)][1]:
            best[int(nu)]=(int(cell),int(count))
    return {nu:item[0] for nu,item in best.items()}


def label_expression(labels, rna):
    flat=labels.ravel().astype(np.int64)
    occupied=np.flatnonzero(flat>0)
    assignment=sparse.csr_matrix((np.ones(len(occupied),np.int64),(flat[occupied],occupied)),
                                 shape=(int(flat.max())+1,len(flat)))
    return (assignment @ rna).tocsr()


def correlation(a,b):
    a=a-a.mean();b=b-b.mean()
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denom) if denom else 0.


def instance_iou(gt,prediction):
    """One-to-one IoU; missed annotated cells contribute zero to the mean.

    The source only annotates selected whole cells, so outside predictions are
    not all certifiable false positives. Do not report whole-frame precision/AP.
    Predicted mask areas include all pixels, penalizing merged/expanded matches.
    """
    g_ids=np.unique(gt[gt>0]);p_ids=np.unique(prediction[prediction>0])
    if not len(g_ids): raise ValueError('no annotated ground-truth cells')
    ious=np.zeros((len(g_ids),len(p_ids)))
    if len(p_ids):
        gm=np.zeros(int(gt.max())+1,int);gm[g_ids]=np.arange(len(g_ids))
        pm=np.zeros(int(prediction.max())+1,int);pm[p_ids]=np.arange(len(p_ids))
        both=(gt>0)&(prediction>0)
        intersection=np.bincount(gm[gt[both]]*len(p_ids)+pm[prediction[both]],
                                  minlength=len(g_ids)*len(p_ids)).reshape(ious.shape)
        ga=np.bincount(gt.ravel())[g_ids]
        pa=np.bincount(prediction.ravel())[p_ids]
        ious=intersection/(ga[:,None]+pa[None,:]-intersection)
    matched=np.zeros(len(g_ids))
    if len(p_ids):
        rows,cols=linear_sum_assignment(-ious)
        matched[rows]=ious[rows,cols]
    return dict(annotated_cells=len(g_ids),predicted_cells=len(p_ids),mean_iou=float(matched.mean()),
                median_iou=float(np.median(matched)),gt_recall_iou50=float((matched>=.5).mean()),
                per_gt_iou=matched.tolist(),definition='maximum-total-IoU one-to-one assignment; unmatched GT=0',
                precision_ap='not reported: annotation only covers selected whole cells')


def paired_consistency(nuclei, first, second, rna, excluded_genes=None):
    m1,m2=match_nuclei(nuclei,first),match_nuclei(nuclei,second)
    e1,e2=label_expression(first,rna),label_expression(second,rna)
    # Intersections are aggregated once for every unique cell-label pair.
    flat1,flat2=first.ravel(),second.ravel()
    both=(flat1>0)&(flat2>0)
    ids=np.flatnonzero(both)
    pairs,inverse=np.unique(np.stack((flat1[both],flat2[both]),-1),axis=0,return_inverse=True)
    mapper=sparse.csr_matrix((np.ones(len(ids),np.int64),(inverse,ids)),shape=(len(pairs),len(flat1)))
    eint=(mapper @ rna).tocsr()
    pairindex={tuple(map(int,p)):i for i,p in enumerate(pairs)}
    rows=[]
    common=sorted(set(m1)&set(m2))
    for nu in common:
        pair=(m1[nu],m2[nu])
        i=pairindex.get(pair)
        if i is None: continue
        shared=eint[i].toarray().ravel().astype(np.float64)
        unique1=e1[pair[0]].toarray().ravel()-shared
        unique2=e2[pair[1]].toarray().ravel()-shared
        if np.any(unique1<0) or np.any(unique2<0):
            raise AssertionError('intersection cannot exceed its matched cell')
        umi=[int(v.sum()) for v in (shared,unique1,unique2)]
        if min(umi)<100: continue
        row=dict(nucleus=nu,cell_first=pair[0],cell_second=pair[1],umi_intersection=umi[0],
                 umi_unique_first=umi[1],umi_unique_second=umi[2],
                 correlation_first=correlation(shared,unique1),correlation_second=correlation(shared,unique2))
        if excluded_genes is not None:
            keep=~excluded_genes
            row.update(nonhvg_correlation_first=correlation(shared[keep],unique1[keep]),
                       nonhvg_correlation_second=correlation(shared[keep],unique2[keep]))
        rows.append(row)
    summary=dict(nuclei_total=int(len(np.unique(nuclei[nuclei>0]))),nuclei_matched_both=len(common),
                 eligible_nuclei=len(rows),unique_matched_pairs=len(set((r['cell_first'],r['cell_second']) for r in rows)),
                 min_umi_each_region=100)
    for key in ['correlation_first','correlation_second','nonhvg_correlation_first','nonhvg_correlation_second']:
        v=[r[key] for r in rows if key in r]
        summary[key+'_mean']=float(np.mean(v)) if v else None
        summary[key+'_median']=float(np.median(v)) if v else None
    summary['mean_paired_difference_first_minus_second']=float(np.mean([
        r['correlation_first']-r['correlation_second'] for r in rows])) if rows else None
    return summary,rows


def assess(data, methods, output_name='benchmark.json'):
    rna=sparse.load_npz(data/'rna_pixels.npz').astype(np.int64)
    all_genes=json.loads((data/'all_genes.json').read_text())
    hvg=set(json.loads((data/'genes.json').read_text()))
    excluded=np.array([g in hvg for g in all_genes])
    segmentations={m:np.load(data/m/'segmentation.npz')['cells'] for m in methods}
    nuclei=np.load(data/methods[0]/'segmentation.npz')['nuclei']
    summaries={}
    pixel_umi=np.asarray(rna.sum(1)).ravel()
    for method,cells in segmentations.items():
        ids,counts=np.unique(cells[cells>0],return_counts=True)
        summaries[method]=dict(cells=len(ids),median_cell_area_pixels=float(np.median(counts)) if len(counts) else None,
                               assigned_umi_fraction=float(pixel_umi[cells.ravel()>0].sum()/pixel_umi.sum()),
                               nucleus_match_fraction=len(match_nuclei(nuclei,cells))/len(np.unique(nuclei[nuclei>0])))
    pairs=[]
    # Each comparison uses one shared eligible nucleus population for both methods.
    for first,second in [('raw_scale4','scs_reference'),('genept_scale4','scs_reference'),
                         ('genept_scale4','raw_scale4'),('scs_reference','direction_prior_only')]:
        if first not in segmentations or second not in segmentations: continue
        summary,rows=paired_consistency(nuclei,segmentations[first],segmentations[second],rna,excluded)
        pairs.append(dict(first=first,second=second,**summary))
        (data/f'paired_{first}_vs_{second}.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
    result=dict(tile=data.name,quality=summaries,paired_rna_consistency=pairs,
                caveats=['RNA correlation is a proxy, not ground-truth IoU.',
                         'Matched cells may repeat across nuclei; no independent-cell p-values claimed.',
                         'Eligible populations differ between pairwise comparisons.',
                         'Non-HVG sensitivity uses the same all-gene UMI eligibility population.'])
    if (data/'ground_truth.npz').exists():
        ground=np.load(data/'ground_truth.npz')
        factor=int(ground['native_pixels_per_grid'])
        result['ground_truth_iou']={}
        for m,cells in segmentations.items():
            expanded=np.repeat(np.repeat(cells,factor,axis=0),factor,axis=1)
            if 'ignore' in ground: expanded[ground['ignore']]=0
            result['ground_truth_iou'][m]=instance_iou(ground['cells'],expanded)
    (data/output_name).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--tiles',nargs='+',default=['2104','2105','2106','2107'])
    p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    for tile in args.tiles:
        data=args.root/tile
        for method in ['scs_reference','raw_scale4','genept_scale4']:
            while not (data/method/'completed.json').exists():
                if not args.watch: raise RuntimeError(f'{tile}/{method} training incomplete')
                time.sleep(20)
            segment(data,method)
            if method=='scs_reference': segment(data,'direction_prior_only')
        assess(data,['scs_reference','raw_scale4','genept_scale4','direction_prior_only'])
