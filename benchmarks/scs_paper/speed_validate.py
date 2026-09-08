"""Numerical checks on isolated variants; no claims of segmentation quality."""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.scs_paper.stereo_whole_train import SparseCache,make,per_example_loss
from benchmarks.scs_paper.speed_explore import DenseTrunk,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',choices=['compile_blocks','compile_trunk','normalized_pool','compile_strict'],required=True)
    p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--rows',type=int,default=256)
    a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=SparseCache(ROOT/'runs/SCS_stereo_whole_v1',torch.device('cuda',0))
    snapshot=torch.load(a.out/'snapshot.pt',map_location='cpu',weights_only=False)
    rows=np.random.RandomState(159).choice(np.flatnonzero(cache.split==0),a.rows,replace=False)
    model,optimizers,config=make(cache,3812);del optimizers
    model.load_state_dict(snapshot['model']);model.eval()
    x,pos,d,b=cache.batch(rows)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16'):
        dl,bl=model(x,pos);loss=per_example_loss(dl,bl,d,b).mean()
    loss.backward()
    expected_dl=dl.float().detach().cpu();expected_bl=bl.float().detach().cpu()
    expected_loss=float(loss.detach())
    expected_grads={name:param.grad.detach().float().cpu().clone() for name,param in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    del dl,bl,loss
    if a.variant=='compile_blocks':
        model.blocks=torch.nn.ModuleList([torch.compile(block,fullgraph=True,dynamic=False,
            options={'triton.cudagraphs':False}) for block in model.blocks])
    trunk=DenseTrunk(model)
    if a.variant in ('compile_trunk','compile_strict'):
        options={'triton.cudagraphs':False}
        if a.variant=='compile_strict':
            options.update(emulate_precision_casts=True,emulate_precision_casts_on_saved_tensors=True)
        trunk=torch.compile(trunk,fullgraph=True,dynamic=False,options=options)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16'):
        if a.variant=='normalized_pool':
            ids,raw,offsets,shape=x
            lengths=offsets[1:]-offsets[:-1]
            normalized=raw/torch.repeat_interleave(lengths,lengths).float()
            table=model.expression_projection(model.gene_embeddings)
            pooled=F.embedding_bag(ids,table,offsets,mode='sum',include_last_offset=True,
                per_sample_weights=normalized.to(table.dtype))
            valid=(lengths>0).to(pooled.dtype)[:,None]
            tokens=(pooled+valid*model.spot_bias.to(pooled.dtype)).reshape(*shape,config.width)*config.expression_scale
        else:
            tokens=model.project_expression(x)
        dl,bl=trunk(tokens,pos);loss=per_example_loss(dl,bl,d,b).mean()
    loss.backward()
    actual_dl=dl.float().detach().cpu();actual_bl=bl.float().detach().cpu()
    error=denominator=0.;worst=0.;nonfinite=[]
    for name,param in model.named_parameters():
        name=name.replace('_orig_mod.','')
        actual=param.grad.detach().float().cpu();expected=expected_grads[name]
        error+=float((actual-expected).double().square().sum())
        denominator+=float(expected.double().square().sum())
        worst=max(worst,float((actual-expected).abs().max()))
        if not torch.isfinite(actual).all():nonfinite.append(name)
    result=dict(variant=a.variant,rows=len(rows),dtype=a.precision,mode='eval with gradients; dropout disabled on both',
        expected_loss=expected_loss,actual_loss=float(loss.detach()),
        direction_logit_max_abs_difference=float((actual_dl-expected_dl).abs().max()),
        direction_logit_relative_l2=float(torch.linalg.vector_norm(actual_dl-expected_dl)/torch.linalg.vector_norm(expected_dl)),
        direction_argmax_agreement=float((actual_dl.argmax(1)==expected_dl.argmax(1)).float().mean()),
        foreground_probability_max_abs_difference=float((actual_bl.sigmoid()-expected_bl.sigmoid()).abs().max()),
        gradient_relative_l2=(error/max(denominator,1e-30))**.5,gradient_max_abs_difference=worst,
        nonfinite_gradient_parameters=nonfinite,
        caveat='Compiler dropout RNG differs in train mode; eval agreement is not identical training trajectory or validated segmentation quality')
    suffix='_fp32' if a.precision=='fp32' else ''
    save(a.out/f'validation_{a.variant}{suffix}.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
