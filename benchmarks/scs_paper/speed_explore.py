"""Isolated real-brain throughput experiments; never mutate a training run.

Run with CUDA_VISIBLE_DEVICES set to the verified idle GPU. Every variant uses
the same checkpoint snapshot, sampled training rows, batch size and losses.
Profiler timings and compiler cold-start are kept separate from steady state.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.scs_paper.stereo_whole_train import SparseCache, make, per_example_loss


def save(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


class DenseTrunk(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.position_projection=model.position_projection
        self.blocks=model.blocks
        self.final_norm=model.final_norm
        self.head_mlp=model.head_mlp
        self.direction_head=model.direction_head
        self.foreground_head=model.foreground_head
        self.scale=model.config.coordinate_scale

    def forward(self,tokens,positions):
        x=tokens+self.position_projection(positions.to(tokens.dtype)*self.scale)
        for block in self.blocks:
            x=block(x)
        features=self.head_mlp(self.final_norm(x)[:,0])
        return self.direction_head(features),self.foreground_head(features).squeeze(-1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'runs/SCS_stereo_whole_v1')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',choices=['baseline','fused_adam','compile_blocks','compile_trunk','normalized_pool','compile_muon','compile_combo','compile_strict'],default='baseline')
    p.add_argument('--batch',type=int,default=4096)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--label',default='')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    suffix=f'_{args.label}' if args.label else ''
    result_path=args.out/f'{args.variant}_b{args.batch}{suffix}.json'
    if result_path.exists():
        raise FileExistsError(result_path)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',0)
    torch.cuda.set_device(device)
    tick=time.time()
    cache=SparseCache(args.root,device)
    train=np.flatnonzero(cache.split==0)
    rows=np.random.RandomState(78431).choice(train,(args.steps+6,args.batch),replace=True)
    # Snapshot once: atomic training checkpoints remain untouched. Persist our
    # own copy so later processes start from exactly the same model/optimizer.
    snapshot_path=args.out/'snapshot.pt'
    if not snapshot_path.exists():
        snapshot=torch.load(args.root/'genept_all_scale4/latest.pt',map_location='cpu',weights_only=False)
        torch.save(snapshot,snapshot_path)
    else:
        snapshot=torch.load(snapshot_path,map_location='cpu',weights_only=False)
    model,optimizers,config=make(cache,3812)
    model.load_state_dict(snapshot['model'])
    for optimizer,state in zip(optimizers,snapshot['optimizers']):
        optimizer.load_state_dict(state)
    source=inspect.getsourcefile(torch.optim.Muon)
    save(args.out/'runtime.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),muon_signature=str(inspect.signature(torch.optim.Muon)),
        muon_source=source,model=config.to_dict(),snapshot_epoch=snapshot['epoch'],snapshot_next_step=snapshot['next_step'],
        snapshot_training_config=snapshot['config'],load_seconds=time.time()-tick,
        caveats=['other GPU on the same node is training; CPU/storage are shared',
                 'timing experiment, not a retrained biological benchmark']))
    if args.variant in ('fused_adam','compile_combo'):
        state=optimizers[1].state_dict()
        _,params=model.optimizer_parameter_groups()
        opt=torch.optim.AdamW(params,lr=.0003,betas=(.9,.95),weight_decay=.01,fused=True)
        opt.load_state_dict(state)
        for group in opt.param_groups:
            group['fused']=True;group['foreach']=False
        for state_value in opt.state.values():
            if 'step' in state_value:
                state_value['step']=state_value['step'].to(device)
        optimizers[1]=opt
    normalized=args.variant=='normalized_pool'
    if normalized:
        lengths=cache.expression_indptr[1:]-cache.expression_indptr[:-1]
        # Expression remains raw sparse gene weights, not cached spot embeddings.
        cache.expression_data=cache.expression_data/torch.repeat_interleave(lengths,lengths).float()
    if args.variant=='compile_blocks':
        model.blocks=torch.nn.ModuleList([torch.compile(block,fullgraph=True,dynamic=False,
            options={'triton.cudagraphs':False}) for block in model.blocks])
    trunk=DenseTrunk(model)
    if args.variant in ('compile_trunk','compile_combo','compile_strict'):
        options={'triton.cudagraphs':False}
        if args.variant=='compile_strict':
            options.update(emulate_precision_casts=True,emulate_precision_casts_on_saved_tensors=True)
        trunk=torch.compile(trunk,fullgraph=True,dynamic=False,options=options)
    muon_step=optimizers[0].step
    if args.variant in ('compile_muon','compile_combo'):
        muon_step=torch.compile(muon_step,fullgraph=False,options={'triton.cudagraphs':False})
    phases=['input','pool','trunk','loss','backward','muon','adamw']
    events={phase:[] for phase in phases}
    last_loss=None

    def timed(phase,fn,record):
        if record:
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
        with torch.profiler.record_function(phase):
            output=fn()
        if record:
            end.record();events[phase].append((begin,end))
        return output

    def step(selected,record=False):
        nonlocal last_loss
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        x,pos,d,b=timed('input',lambda:cache.batch(selected),record)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            def pooling():
                if not normalized:
                    return model.project_expression(x)
                indices,values,offsets,shape=x
                table=model.expression_projection(model.gene_embeddings)
                pooled=F.embedding_bag(indices,table,offsets,mode='sum',
                    per_sample_weights=values.to(table.dtype),include_last_offset=True)
                valid=(offsets[1:]>offsets[:-1]).to(pooled.dtype)[:,None]
                return ((pooled+valid*model.spot_bias.to(pooled.dtype)).reshape(*shape,config.width)*config.expression_scale)
            tokens=timed('pool',pooling,record)
            dl,bl=timed('trunk',lambda:trunk(tokens,pos),record)
            loss=timed('loss',lambda:per_example_loss(dl,bl,d,b).mean(),record)
        timed('backward',lambda:loss.backward(),record)
        timed('muon',muon_step,record)
        timed('adamw',lambda:optimizers[1].step(),record)
        last_loss=loss.detach()

    model.train()
    torch.manual_seed(934);torch.cuda.manual_seed_all(934)
    cold=time.perf_counter()
    step(rows[0]);torch.cuda.synchronize()
    cold=time.perf_counter()-cold
    print(f'COLD {args.variant} {cold:.3f}s',flush=True)
    for selected in rows[1:6]:
        step(selected)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for selected in rows[6:]:
        step(selected,record=True)
    torch.cuda.synchronize()
    wall=time.perf_counter()-start
    timing={phase:dict(mean_ms=float(np.mean([a.elapsed_time(b) for a,b in pairs])),
                       median_ms=float(np.median([a.elapsed_time(b) for a,b in pairs])))
            for phase,pairs in events.items()}
    result=dict(variant=args.variant,batch=args.batch,steps=args.steps,cold_first_step_seconds=cold,
        steady_seconds=wall,step_ms=wall/args.steps*1000,samples_per_second=args.steps*args.batch/wall,
        gpu_phase_times=timing,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),last_loss=float(last_loss),
        snapshot_epoch=snapshot['epoch'],snapshot_next_step=snapshot['next_step'],
        note='microbenchmark does not include epoch validation/checkpoint I/O or one-rank DDP bookkeeping')
    save(result_path,result);print(json.dumps(result),flush=True)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,profile_memory=False,with_stack=False) as prof:
            for selected in rows[:3]:
                step(selected)
        # Profiler times are diagnostic, never used as the throughput number.
        (args.out/'profile_cuda.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=45))
        (args.out/'profile_cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=45))
        prof.export_chrome_trace(str(args.out/'profile_trace.json'))


if __name__=='__main__':
    main()
