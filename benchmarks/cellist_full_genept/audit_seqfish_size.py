"""Audit native-pixel versus five-pixel-grid size cutoff; never overwrites the benchmark."""
import json,os
from pathlib import Path
import numpy as np
import anndata as ad
from benchmarks.scs_paper.evaluate import post
from benchmarks.cellist_full_genept.metrics import transcript_iou
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/Cellist_full_genept_v1'
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):raise RuntimeError('CPU allocation required')
results={}
for fov in (0,1):
    data=BASE/'datasets'/f'seqfish_rep1_fov{fov}';tile=data/'segmentation_patches'/f'fov{fov}'
    prediction=tile/'genept_paperpost_scale4';a=ad.read_h5ad(tile/'data/spots0:0:0:0.h5ad')
    intensity,dx,dy,*_=post.read_gradient(str(prediction/'spot_prediction.txt'),a,a.shape,1,40)
    labels,sinks,*_=post.gvf_tracking(dx,dy,intensity,1)
    labels=post.merge_sinks(labels,sinks,1).astype(np.uint32)
    size=np.bincount(labels.ravel());truth=np.load(data/'molecule_ground_truth.npz');xy=truth['grid_xy']
    methods={}
    for cutoff in (200,8):
        mask=np.where(size[labels]>=cutoff,labels,0)
        if cutoff==200:np.testing.assert_array_equal(mask,np.load(prediction/'segmentation.npz')['cells'])
        ids=mask[xy[:,1],xy[:,0]]
        methods[str(cutoff)]=dict(cells=int(len(np.unique(mask[mask>0]))),assigned_RNA_fraction=float(np.mean(ids>0)),
            transcript_iou=transcript_iou(truth['manual_cell'],ids))
    results[data.name]=methods
    print(json.dumps({data.name:methods}),flush=True)
(BASE/'fairness_audit/seqfish_size_cutoff.json').write_text(json.dumps(results,indent=2)+'\n')
