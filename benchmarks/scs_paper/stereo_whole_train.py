"""Whole-slide GenePT model with GPU-resident sparse inputs.

Use one torchrun worker per dataset on its assigned GPU. Distributed support
is retained, but the two-dataset experiment runs two independent world_size=1
processes, not a two-GPU joint model.
No per-spot 1536-D cache and no per-tile sampling caps. Final partial global
batches are masked, not duplicated as additional training observations.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from optimizations.scs_streaming.torch_model import ModelConfig, SCSClassifier


def save_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


class SparseCache:
    def __init__(self, root, device):
        self.device = device
        self.meta = json.loads((root / 'prepared.json').read_text())
        if not self.meta['complete']:
            raise ValueError('Incomplete whole-slide dataset')
        for name, dtype in [('expression_data', torch.float32), ('expression_indices', torch.int64),
                            ('expression_indptr', torch.int64), ('neighbors', torch.int64),
                            ('directions', torch.int64), ('foreground', torch.float32)]:
            setattr(self, name, torch.as_tensor(np.load(root / f'{name}.npy'), dtype=dtype, device=device))
        self.split = np.load(root / 'split.npy')
        self.table = torch.as_tensor(np.load(root/'gene_embeddings.npy'), device=device)

    def batch(self, rows):
        rows = torch.as_tensor(rows, dtype=torch.int64, device=self.device)
        nb = self.neighbors[rows]
        flat = nb.reshape(-1)
        starts = self.expression_indptr[flat]
        lengths = self.expression_indptr[flat+1] - starts
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        source = torch.arange(int(offsets[-1]), device=self.device)
        source = source + torch.repeat_interleave(starts - offsets[:-1], lengths)
        expression = (self.expression_indices[source], self.expression_data[source], offsets, tuple(nb.shape))
        width = self.meta['bin_shape'][1]
        coords = torch.stack((nb // width, nb % width), -1).float() * self.meta.get('bin_size', 3)
        return expression, coords-coords[:, :1], self.directions[rows], self.foreground[rows]


def make(cache, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = ModelConfig(input_dim=cache.table.shape[1], n_genes=len(cache.table),
        scale=4, expression_scale=math.sqrt(cache.table.shape[1]), coordinate_scale=1/30)
    model = SCSClassifier(config, cache.table).to(cache.device)
    mu, adam = model.optimizer_parameter_groups()
    optimizers = [torch.optim.Muon(mu, lr=.002, momentum=.95, weight_decay=.01, adjust_lr_fn='original'),
                  torch.optim.AdamW(adam, lr=.0003, betas=(.9, .95), weight_decay=.01)]
    return model, optimizers, config


def per_example_loss(dl, bl, d, b):
    return F.cross_entropy(dl.float(), d, reduction='none') * b + F.binary_cross_entropy_with_logits(
        bl.float(), b, reduction='none')


def tune(cache, train, seed, rank):
    chosen, trials = 0, []
    # Probe representative random rows, with both forward/backward and optimizer
    # states allocated. Keep 15% headroom for sparse-density variation and DDP.
    rng = np.random.RandomState(seed)
    for batch in [256, 512, 1024, 2048, 3072, 4096, 5120, 6144]:
        model = optimizers = x = pos = d = b = dl = bl = loss = None
        try:
            model, optimizers, _ = make(cache, seed)
            torch.cuda.reset_peak_memory_stats()
            tick = time.time()
            for _ in range(2):
                x, pos, d, b = cache.batch(rng.choice(train, batch))
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    dl, bl = model(x, pos)
                    loss = per_example_loss(dl, bl, d, b).mean()
                loss.backward()
                for opt in optimizers:
                    opt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            fits = peak < .85 * torch.cuda.get_device_properties(cache.device).total_memory
            trials.append(dict(batch=batch, seconds=time.time()-tick, peak_bytes=peak, headroom_passed=fits))
            if fits:
                chosen = batch
            else:
                break
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(batch=batch, oom=True))
            break
        finally:
            del model, optimizers, x, pos, d, b, dl, bl, loss
            gc.collect()
            torch.cuda.empty_cache()
    value = torch.tensor(chosen, device=cache.device)
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    if int(value) == 0:
        raise RuntimeError('No safe batch fits on every GPU')
    return int(value), dict(rank=rank, trials=trials)


@torch.no_grad()
def evaluate(model, cache, rows, batch, rank, world):
    model.eval()
    # Every example exactly once; direct module forward avoids DDP collectives
    # when validation shards have different numbers of batches.
    counts = torch.zeros(7, device=cache.device, dtype=torch.float64)
    local = rows[rank::world]
    for start in range(0, len(local), batch):
        x, pos, d, b = cache.batch(local[start:start+batch])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            dl, bl = model(x, pos)
        counts += torch.stack((per_example_loss(dl, bl, d, b).sum(),
            b.new_tensor(len(b)), b.sum(), ((dl.argmax(1)==d)*b).sum(),
            ((bl>=0)*b).sum(), (1-b).sum(), ((bl<0)*(1-b)).sum())).double()
    dist.all_reduce(counts)
    loss, n, fg, correct, tp, bg, tn = counts.cpu().tolist()
    return dict(loss=loss/max(n, 1), n=int(n), foreground=int(fg),
        direction_accuracy=correct/max(fg, 1), foreground_recall=tp/max(fg, 1),
        background_specificity=tn/max(bg, 1), foreground_balanced_accuracy=(tp/max(fg, 1)+tn/max(bg, 1))/2)


def checkpoint(path, model, optimizers, epoch, step, best, config, rank, world):
    local_rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
    rngs = [None] * world
    dist.all_gather_object(rngs, local_rng)
    if rank == 0:
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(), optimizers=[o.state_dict() for o in optimizers],
            epoch=epoch, next_step=step, best=best, config=config, rngs=rngs), temporary)
        temporary.replace(path)
    dist.barrier()


@torch.no_grad()
def predict(model, cache, root, out, batch, rank, world):
    marker = out / f'prediction_rank{rank}_completed.json'
    if marker.exists():
        return
    model.eval()
    rows = np.arange(len(cache.split), dtype=np.int64)[rank::world]
    logits = np.lib.format.open_memmap(out/f'logits_rank{rank}.npy', mode='w+', dtype=np.float32, shape=(len(rows), 16))
    binary = np.lib.format.open_memmap(out/f'foreground_rank{rank}.npy', mode='w+', dtype=np.float32, shape=(len(rows),))
    for start in range(0, len(rows), batch):
        selected = rows[start:start+batch]
        x, pos, _, _ = cache.batch(selected)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            dl, bl = model(x, pos)
        logits[start:start+len(selected)] = dl.float().cpu().numpy()
        binary[start:start+len(selected)] = bl.float().sigmoid().cpu().numpy()
    logits.flush()
    binary.flush()
    save_json(marker, dict(rows=len(rows), rank=rank, world_size=world,
        row_rule='arange(number_of_centres)[rank::world_size]'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT/'runs/SCS_stereo_whole_v1')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-per-gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=3812)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint-every', type=int, default=200)
    p.add_argument('--stop-after-steps', type=int, default=0, help='smoke test only; does not mark training complete')
    p.add_argument('--run-name', default='genept_all_scale4')
    args = p.parse_args()
    rank, world, local = [int(os.environ[k]) for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']]
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = SparseCache(args.root, torch.device('cuda', local))
    train, val, test = [np.flatnonzero(cache.split == k) for k in (0, 1, 2)]
    if min(map(len, [train, val, test])) == 0:
        raise ValueError('Empty data split')
    out = args.root / args.run_name
    if rank == 0:
        out.mkdir(exist_ok=args.resume)
    dist.barrier()
    previous = torch.load(out/'latest.pt', map_location='cpu', weights_only=False) if args.resume else None
    if previous:
        batch = previous['config']['batch_per_gpu']
    elif args.batch_per_gpu:
        batch = args.batch_per_gpu
    else:
        batch, trials = tune(cache, train, args.seed, rank)
        save_json(out/f'batch_probe_rank{rank}.json', trials)
    model, optimizers, model_config = make(cache, args.seed)
    config = dict(model=model_config.to_dict(), batch_per_gpu=batch, world_size=world,
        global_batch=batch*world, epochs=args.epochs, seed=args.seed,
        train=len(train), validation=len(val), test=len(test),
        neighbours_sha256=cache.meta['neighbors_sha256'], split_sha256=cache.meta['split_sha256'],
        genept=cache.meta['genept'], amp='bf16', sdpa=True,
        checkpoint_selection='minimum validation joint BCE + masked direction CE; never test correlations',
        optimizer=dict(muon_lr=.002, adamw_lr=.0003, weight_decay=.01),
        architecture_scope='one shared full-slide model; not independent tile models')
    start_epoch, start_step, best = 1, 0, float('inf')
    if previous:
        if previous['config'] != config:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(previous['model'])
        for opt, state in zip(optimizers, previous['optimizers']):
            opt.load_state_dict(state)
        start_epoch, start_step, best = previous['epoch'], previous['next_step'], previous['best']
    if rank == 0:
        save_json(out/'config.json', config)
        print(json.dumps(config), flush=True)
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    torch.cuda.reset_peak_memory_stats()
    if previous:
        torch.set_rng_state(previous['rngs'][rank]['cpu'])
        torch.cuda.set_rng_state(previous['rngs'][rank]['cuda'])
    else:
        torch.manual_seed(args.seed+rank)
        torch.cuda.manual_seed(args.seed+rank)
    global_batch = batch*world
    steps = math.ceil(len(train)/global_batch)
    seen_steps = 0
    for epoch in range(start_epoch, args.epochs+1):
        ddp.train()
        order = np.random.RandomState(args.seed+epoch).permutation(train)
        padded = np.resize(order, steps*global_batch)
        tick = time.time()
        for step in range(start_step if epoch==start_epoch else 0, steps):
            offset = step*global_batch + rank*batch
            rows = padded[offset:offset+batch]
            valid = torch.as_tensor(np.arange(offset, offset+batch)<len(train), device=cache.device)
            n_global = min(global_batch, len(train)-step*global_batch)
            x, pos, d, b = cache.batch(rows)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dl, bl = ddp(x, pos)
                loss = (per_example_loss(dl, bl, d, b)*valid).sum() * (world/n_global)
            finite = torch.isfinite(loss).int()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite):
                raise FloatingPointError('Non-finite loss on a DDP rank')
            loss.backward()
            for opt in optimizers:
                opt.step()
            seen_steps += 1
            if rank == 0 and (step+1)%25==0:
                print(f'epoch={epoch} step={step+1}/{steps} local_loss={float(loss):.5f}', flush=True)
            if (step+1)%args.checkpoint_every==0 or (args.stop_after_steps and seen_steps>=args.stop_after_steps):
                checkpoint(out/'latest.pt', model, optimizers, epoch, step+1, best, config, rank, world)
            if args.stop_after_steps and seen_steps>=args.stop_after_steps:
                if rank==0:
                    save_json(out/'smoke_completed.json', dict(steps=seen_steps, world_size=world, training_complete=False))
                dist.destroy_process_group()
                return
        val_metrics = evaluate(model, cache, val, batch, rank, world)
        improved = val_metrics['loss'] < best
        if improved:
            best = val_metrics['loss']
        checkpoint(out/'latest.pt', model, optimizers, epoch+1, 0, best, config, rank, world)
        if improved:
            checkpoint(out/'best.pt', model, optimizers, epoch+1, 0, best, config, rank, world)
        if rank == 0:
            row = dict(epoch=epoch, validation=val_metrics, seconds=time.time()-tick,
                peak_gpu_bytes=torch.cuda.max_memory_allocated())
            save_json(out/f'epoch_{epoch:03d}.json', row)
            print(json.dumps(row), flush=True)
    selected = torch.load(out/'best.pt', map_location=cache.device, weights_only=False)
    model.load_state_dict(selected['model'])
    test_metrics = evaluate(model, cache, test, batch, rank, world)
    if rank == 0:
        save_json(out/'training_completed.json', dict(epochs=args.epochs, selected_epoch=selected['epoch']-1,
            test=test_metrics, segmentation_complete=False))
    predict(model, cache, args.root, out, batch, rank, world)
    dist.barrier()
    if rank == 0:
        save_json(out/'inference_completed.json', dict(centres=len(cache.split), world_size=world,
            segmentation_complete=False))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
