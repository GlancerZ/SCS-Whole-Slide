"""Measure physical reach of the stored 50-token contexts on fixed sampled centres."""
import json,os
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[2]/'runs/Cellist_full_genept_v1'
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):raise RuntimeError('CPU allocation required')
out={}
for name in ['stereo_mouse_brain','2104','2105','2106','2107','seqfish_rep1_fov0','seqfish_rep1_fov1']:
    d=BASE/'datasets'/name;m=json.loads((d/'prepared.json').read_text());res=json.loads((d/'comparison_input.json').read_text())['resolution_um']
    nb=np.load(d/'neighbors.npy',mmap_mode='r');nuc=np.load(d/'nuclei.npy',mmap_mode='r')
    rows=np.random.RandomState(1234).choice(len(nb),min(4096,len(nb)),replace=False);a=np.asarray(nb[rows]);width=m['bin_shape'][1]
    xy=np.stack([a//width,a%width],-1)*m['bin_size']
    radius=np.sqrt(np.sum((xy-xy[:,:1])**2,axis=2)).max(1)*res
    labels=nuc[xy[:,:,0],xy[:,:,1]];outside=labels[:,0]==0
    out[name]={'sampled_centres':len(rows),'tokens':a.shape[1],'radius_um_p50_p90_p99':np.quantile(radius,[.5,.9,.99]).tolist(),
        'sampled_outside_nucleus':int(outside.sum()),'outside_context_has_any_nucleus_fraction':float(np.mean(np.any(labels[outside]>0,axis=1)))}
(BASE/'fairness_audit/context_reach.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
