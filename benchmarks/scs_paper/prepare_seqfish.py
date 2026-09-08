"""Controlled seqFISH+ subset, four native image pixels per RNA grid point.

The paper describes ~0.4um grids but does not publish its FISH conversion code
or exact calibration here. Four pixels is ~0.4um from its ~200um/2048 FOV;
record this approximation, never claim exact replication of its reported IoU.
Manual ROIs are written for evaluation only, never used for pseudo-labels.
"""
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import ndimage,sparse
from scipy.io import loadmat
from scipy.spatial import cKDTree
from skimage.draw import polygon
import tifffile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.scs_paper.prepare import neighbors_for, sha, st


def prepare(fov,root,source,tot,genes,rois,images):
    from roifile import ImagejRoi
    started=time.time()
    out=root/f'seqfish_rep1_fov{fov}'
    out.mkdir(exist_ok=False)
    (out/'data').mkdir()
    native_shape=(2048,2048)
    gt=np.zeros(native_shape,np.uint32)
    overlap_count=np.zeros(native_shape,np.uint16)
    names=sorted(n for n in rois.namelist() if n.startswith(f'ALL_Roi/RoiSet_Pos{fov}/') and n.endswith('.roi'))
    for label,name in enumerate(names,1):
        xy=ImagejRoi.frombytes(rois.read(name)).coordinates()
        r,c=polygon(xy[:,1],xy[:,0],shape=native_shape)
        overlap_count[r,c]+=1
        gt[r,c]=label
    ignore=overlap_count>1
    ambiguous_fraction=float(ignore.sum()/max(1,(overlap_count>0).sum()))
    if ambiguous_fraction>.01: raise ValueError('substantial overlapping manual annotations; manual audit required')
    # RNA locations are x,y,z; first z plane corresponds to z=1 in Matlab.
    points,gcodes=[],[]
    for cell in range(tot.shape[1]):
        for g,v in enumerate(tot[fov,cell]):
            if not v.size: continue
            if np.any(v[:,2] != 1): raise ValueError('unexpected multiple RNA z planes')
            points.append(v[:,:2]);gcodes.append(np.full(len(v),g,np.int32))
    xy=np.concatenate(points);gcode=np.concatenate(gcodes)
    native=np.floor(xy).astype(np.int64)
    if np.any(native<0) or np.any(native>=2048): raise ValueError('out of image RNA coordinate')
    overlap=float((gt[native[:,1],native[:,0]]>0).mean())
    if overlap < .99: raise ValueError('RNA/ROI coordinate alignment failed')
    with tifffile.TiffFile(io.BytesIO(images.read(f'final_background_experiment1/MMStack_Pos{fov}.ome.tif'))) as t:
        s=t.series[0]
        if s.axes!='CZYX' or s.shape!=(4,2,2048,2048): raise ValueError('unexpected microscopy axes')
        # OME channel Name=405 is channel 3, verified in coordinate_audit.json.
        dapi=s.asarray()[3].max(axis=0).astype(np.float32)
    dapi=dapi.reshape(512,4,512,4).mean((1,3))
    lo,hi=np.percentile(dapi,[1,99.9])
    stain=np.clip((dapi-lo)/max(hi-lo,1)*255,0,255).astype(np.uint8)
    grid=np.floor(xy/4).astype(np.int64)
    grid_id=grid[:,1]*512+grid[:,0]
    x=sparse.csr_matrix((np.ones(len(grid),np.int64),(grid_id,gcode)),shape=(512*512,len(genes)))
    umi=np.asarray(x.sum(1)).reshape(512,512)
    a=ad.AnnData(sparse.csr_matrix(umi),layers={'stain':stain})
    st.cs.mask_nuclei_from_stain(a,otsu_classes=4,otsu_index=1)
    st.cs.find_peaks_from_mask(a,'stain',7)
    st.cs.watershed(a,'stain',5,out_layer='watershed_labels')
    labels=a.layers['watershed_labels'].astype(np.int32)
    nuclei=np.unique(labels[labels>0])
    centers=np.zeros((int(labels.max())+1,2))
    centers[nuclei]=ndimage.center_of_mass(np.ones(labels.shape),labels,nuclei)
    if not len(nuclei): raise ValueError('no DAPI nuclei')
    a.write_h5ad(out/'data'/'spots0:0:0:0.h5ad')
    hvg=ad.AnnData(x.astype(np.float64))
    sc.pp.highly_variable_genes(hvg,n_top_genes=2000,flavor='seurat_v3',span=1.)
    chosen=np.flatnonzero(hvg.var.highly_variable)
    expr=x[:,chosen].astype(np.float32).tocsr();expr.eliminate_zeros()
    nb=neighbors_for(np.diff(expr.indptr)>0,(512,512))
    rc=np.stack(np.unravel_index(nb[:,0],(512,512)),-1)
    nucleus=labels[rc[:,0],rc[:,1]]
    fg=nucleus>0
    bg=(~fg)&(stain[rc[:,0],rc[:,1]]<=10)&(cKDTree(centers[nuclei]).query(rc)[0]>30)
    accepted_bg=np.zeros(len(rc),bool)
    nfg=nbg=0
    for i in range(len(rc)):
        if fg[i]: nfg+=1
        elif bg[i] and nbg<nfg: accepted_bg[i]=True;nbg+=1
    delta=centers[nucleus]-rc
    direction=(np.mod(np.arctan2(delta[:,0],delta[:,1]),2*np.pi)/(2*np.pi/16)).astype(np.int64)
    direction[~fg]=0
    rng=np.random.RandomState(20260905)
    split=np.full(len(rc),-1,np.int8)
    for b in (False,True):
        for d in (range(16) if b else [0]):
            ids=np.flatnonzero((fg|accepted_bg)&(fg==b)&(direction==d));rng.shuffle(ids)
            n=len(ids)//10
            split[ids]=0;split[ids[:n]]=1;split[ids[n:2*n]]=2
    paper_split=np.full(len(rc),-1,np.int8)
    held=(rc[:,0]>384)&(rc[:,1]>384)
    paper_split[(fg|accepted_bg)&~held]=0;paper_split[(fg|accepted_bg)&held]=1
    np.savez_compressed(out/'samples.npz',neighbors=nb,coords=rc,foreground=fg,direction=direction,
                        split=split,paper_split=paper_split,nucleus=nucleus,
                        bin_shape=(512,512),shape=(512,512),bin_size=1)
    # Boundary-overlap pixels are void for IoU, not arbitrarily assigned to one ROI.
    gt[ignore]=0
    np.savez_compressed(out/'ground_truth.npz',cells=gt,ignore=ignore,native_pixels_per_grid=4)
    sparse.save_npz(out/'expression.npz',expr)
    sparse.save_npz(out/'rna_pixels.npz',x)
    (out/'genes.json').write_text(json.dumps([genes[i] for i in chosen]))
    (out/'all_genes.json').write_text(json.dumps(genes))
    stats=dict(tile=out.name,dataset='seqFISH+ NIH3T3 experiment 1',fov=fov,shape=[512,512],
               n_genes=len(genes),hvg_count=len(chosen),bin_size=1,native_pixels_per_grid=4,
               total_umi=len(xy),nuclei=len(nuclei),ground_truth_cells=len(names),
               input_centres=len(rc),labeled_foreground=nfg,labeled_background=nbg,
               random_train=int(sum(split==0)),random_validation=int(sum(split==1)),random_test=int(sum(split==2)),
               input_rna_inside_manual_roi_fraction=overlap,seconds=time.time()-started,
               dapi_channel=405,z_reduction='maximum projection over the two DAPI planes',stain_percentiles=[1,99.9],
               ambiguous_annotation_fraction=ambiguous_fraction,
               radius_for_postprocessing=20,coordinate_convention='unshifted x,y, zero-based floor',
               source_sha256={f.name:sha(f) for f in source.glob('*.zip')},
               caveats=['RNA coordinates are already partitioned by manually selected whole cells in source.',
                        'Four native pixels per grid point approximates published 0.4um; calibration not supplied.',
                        'DAPI contrast normalization and bin_size=1 are explicit controlled-protocol choices.',
                        'Manual ROI labels are evaluation-only; used to audit coordinate convention, not tune models.',
                        'Overlapping ROI boundary pixels are ignored identically for all methods in IoU.',
                        'No claim of exact reproduction of the paper\'s reported 0.75 IoU.'])
    (out/'prepared.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats),flush=True)


if __name__=='__main__':
    root=ROOT/'runs/SCS_paper_benchmark_v1'
    source=root/'seqfish_source'
    sys.path.insert(0,str(source/'vendor'))
    z=zipfile.ZipFile(source/'seqFISH_NIH3T3_point_locations.zip')
    tot=loadmat(io.BytesIO(z.read('RNA_locations_run_1.mat')))['tot']
    genes=[str(g[0]) for g in loadmat(io.BytesIO(z.read('all_gene_Names.mat')))['allNames'].ravel()]
    rois=zipfile.ZipFile(source/'ROIs_Experiment1_NIH3T3.zip')
    images=zipfile.ZipFile(source/'DAPI_experiment1.zip')
    for fov in (0,1): prepare(fov,root,source,tot,genes,rois,images)
