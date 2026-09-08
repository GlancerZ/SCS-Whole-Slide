"""Controlled, train-only collapse diagnostics; does not overwrite training runs.

Run on an allocated GPU. All ablations use identical 64 training examples,
initialization and no dropout to test memorization, not generalization.
"""
import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .shared_data import save_json
from .shared_train_torch import move_batch
from .torch_data import GenePTBatchDataset
from .torch_model import ModelConfig, SCSClassifier, scs_loss


def select_training_genes(root, directory, out, count=2000):
    """Mean-binned dispersion ranking on 50k training centres only.

    This is a diagnostic selection among GenePT-mapped symbols, not a refit of
    the original Seurat-v3 6000-gene schema and not a validation-based ranking.
    """
    from scipy import sparse
    base = Path(root) / directory
    table = np.load(base / 'train_neighbors.npy', mmap_mode='r')
    ptr = np.load(base / 'expression_indptr.npy', mmap_mode='r')
    values = np.load(base / 'expression_data.npy', mmap_mode='r')
    columns = np.load(base / 'expression_indices.npy', mmap_mode='r')
    symbols = np.load(base / 'gene_symbols.npy', allow_pickle=False)
    samples = np.random.RandomState(721).choice(len(table), min(50000, len(table)), replace=False)
    rows = table[samples, 0].astype(np.int64)
    starts, stops = ptr[rows].astype(np.int64), ptr[rows+1].astype(np.int64)
    lengths = stops-starts
    offsets = np.r_[0, np.cumsum(lengths)]
    source = np.arange(offsets[-1]) + np.repeat(starts-offsets[:-1], lengths)
    matrix = sparse.csr_matrix((values[source].astype(np.float64), columns[source], offsets),
                              shape=(len(rows), len(symbols)))
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    variance = np.asarray(matrix.power(2).mean(axis=0)).ravel()-mean**2
    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    eligible = (detected >= 20) & (variance > 0)
    dispersion = np.log(np.maximum(variance / np.maximum(mean, 1e-12), 1e-12))
    bins = np.searchsorted(np.quantile(np.log1p(mean[eligible]), np.linspace(0,1,21))[1:-1],
                           np.log1p(mean))
    score = np.full(len(symbols), -np.inf)
    for b in range(20):
        mask = eligible & (bins == b)
        if mask.any():
            score[mask] = (dispersion[mask]-dispersion[mask].mean()) / max(dispersion[mask].std(), 1e-8)
    if eligible.sum() < count:
        raise ValueError(f'only {eligible.sum()} genes pass diagnostic selection')
    chosen = np.sort(np.argsort(-score, kind='stable')[:count])
    save_json(out / 'gene_selection.json', dict(
        method='log dispersion z-score in 20 quantile bins of log1p mean; raw counts',
        split='train', selection_seed=721, sampled_centres=len(rows),
        minimum_detected_spots=20, eligible_genes=int(eligible.sum()),
        universe='all GenePT-mapped symbols', selected_gene_ids=chosen.tolist(),
        selected_gene_symbols=[str(x) for x in symbols[chosen]]))
    return chosen


def restrict_batch(batch, selected, raw=False):
    """Keep identical centre spots/neighbours; only filter expression channels."""
    indices, values, offsets, shape = batch[0]
    universe = max(int(indices.max())+1, int(np.max(selected))+1)
    mapping = torch.full((universe,), -1, device=indices.device, dtype=torch.long)
    mapping[torch.as_tensor(selected, device=indices.device)] = torch.arange(len(selected), device=indices.device)
    mapped = mapping[indices]
    keep = mapped >= 0
    prefix = torch.cat((keep.new_zeros(1, dtype=torch.long), keep.long().cumsum(0)))
    new_offsets = prefix[offsets]
    if raw:
        token_ids = torch.arange(len(offsets)-1, device=indices.device).repeat_interleave(offsets.diff())[keep]
        dense = torch.zeros((len(offsets)-1, len(selected)), device=indices.device)
        dense.index_put_((token_ids, mapped[keep]), values[keep], accumulate=True)
        expression = dense.reshape(*shape, len(selected))
    else:
        expression = (indices[keep], values[keep], new_offsets, shape)
    return (expression, *batch[1:]), dict(
        retained_nonzero=int(keep.sum()), original_nonzero=len(indices),
        empty_tokens=int((new_offsets.diff()==0).sum()), total_tokens=len(offsets)-1,
        empty_centres=int((new_offsets.diff().reshape(*shape)[:, 0]==0).sum()))


def stats(x):
    x = x.detach().float()
    return dict(rms=float(x.square().mean().sqrt()), std=float(x.std()),
                minimum=float(x.min()), maximum=float(x.max()),
                finite=bool(x.isfinite().all()))


def measure(model, batch):
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        d, b = model(batch[0], batch[1])
        loss, dl, bl = scs_loss(d, b, batch[2], batch[3])
        positive = batch[3].bool()
        result = dict(loss=float(loss), direction_loss=float(dl), binary_loss=float(bl),
                      direction_accuracy=float((d.argmax(-1)[positive] == batch[2][positive]).float().mean()),
                      sensitivity=float((b[positive] >= 0).float().mean()),
                      specificity=float((b[~positive] < 0).float().mean()),
                      direction_logit_between_sample_std=float(d.float().std(dim=0).mean()),
                      foreground_probability=stats(b.sigmoid()))
        result['predicted_direction_counts'] = torch.bincount(d.argmax(-1), minlength=16).tolist()
    return result


class ScaledInputs(SCSClassifier):
    """Diagnostic-only fixed rescalings; pooling and count information unchanged."""
    expression_scale = 1.0
    coordinate_scale = 1.0

    def project_expression(self, expression):
        return super().project_expression(expression) * self.expression_scale

    def forward(self, expression, relative_positions):
        return super().forward(expression, relative_positions.float() * self.coordinate_scale)


def get_batch(root, directory, size=64):
    dataset = GenePTBatchDataset(root, 'train', size, 1234, dataset_name=directory, residency='mmap')
    rng = np.random.RandomState(418)
    ids = []
    for k in range(16):
        choices = np.flatnonzero((dataset.foreground == 1) & (dataset.directions == k))
        ids.extend(rng.choice(choices, size // 32, replace=False).tolist())
    choices = np.flatnonzero(dataset.foreground == 0)
    ids.extend(rng.choice(choices, size // 2, replace=False).tolist())
    rng.shuffle(ids)
    dataset.order = np.asarray(ids, dtype=np.int64)
    cpu = dataset[0]
    audit = dict(sample_ids=ids, split='train', shape=cpu[0][-1].tolist(),
                 foreground_counts=np.bincount(cpu[3], minlength=2).tolist(),
                 direction_counts=np.bincount(cpu[2][cpu[3] == 1], minlength=16).tolist(),
                 centre_position_is_zero=bool(np.all(cpu[1][:, 0] == 0)),
                 min_count=int(cpu[0][1].min()), max_count=int(cpu[0][1].max()),
                 min_position=int(cpu[1].min()), max_position=int(cpu[1].max()),
                 min_expressed_genes=int(np.diff(cpu[0][2]).min()),
                 max_expressed_genes=int(np.diff(cpu[0][2]).max()))
    return move_batch(cpu, torch.device('cuda'), torch.float32), audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs/ST19_shared_6000')
    p.add_argument('--dataset-name', default='genept_allgenes_linear_random90')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--cases', default='baseline_muon,scaled_muon,scaled_low_muon,baseline_adamw,scaled_adamw')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    batch, audit = get_batch(args.root, args.dataset_name)
    genes = torch.from_numpy(np.load(Path(args.root) / args.dataset_name / 'gene_embeddings.npy')).cuda()
    audit['gene_table'] = stats(genes)
    audit['gene_norm'] = stats(genes.norm(dim=-1))
    save_json(out / 'input_audit.json', audit)
    print(json.dumps({'input_audit': audit}), flush=True)
    selected = (select_training_genes(args.root, args.dataset_name, out)
                if '2000' in args.cases else None)

    # Reproduce the saved failed model without changing it.
    old = torch.load(Path(args.root) / 'torch_genept_allgenes_scale4_bs6016/latest.pt',
                     map_location='cpu', weights_only=False)
    fields = ModelConfig.__dataclass_fields__
    config = ModelConfig(**{k:v for k,v in old['config']['model'].items() if k in fields})
    model = SCSClassifier(config, genes).cuda()
    model.load_state_dict(old['model'])
    del old
    with torch.no_grad():
        ex = model.project_expression(batch[0])
        pos = model.position_projection(batch[1].float())
        audit_checkpoint = dict(expression=stats(ex), position=stats(pos),
                                prediction=measure(model, batch))
    save_json(out / 'checkpoint_audit.json', audit_checkpoint)
    print(json.dumps({'checkpoint': audit_checkpoint}), flush=True)
    del model, ex, pos
    gc.collect()
    torch.cuda.empty_cache()

    cases = {
        'baseline_muon': (1., 1., .02, True),
        'scaled_muon': (math.sqrt(1536), 1/30, .02, True),
        'scaled_low_muon': (math.sqrt(1536), 1/30, .002, True),
        'baseline_adamw': (1., 1., .02, False),
        'scaled_adamw': (math.sqrt(1536), 1/30, .02, False),
        'genept2000_muon': (1., 1., .02, True),
        'raw2000_muon': (1., 1., .02, True),
        'genept2000_scaled_muon': (math.sqrt(1536), 1/30, .02, True),
        'raw2000_low_muon': (1., 1., .002, True),
        'position_only_muon': (1., 1/30, .02, True),
        'expression_only_muon': (math.sqrt(1536), 1., .02, True),
        'baseline_low_muon': (1., 1., .002, True),
    }
    results = {}
    for name in args.cases.split(','):
        expression_scale, coordinate_scale, lr, use_muon = cases[name]
        active_batch, coverage = (restrict_batch(batch, selected, raw=name.startswith('raw'))
                                  if '2000' in name else (batch, None))
        torch.manual_seed(912)
        torch.cuda.manual_seed_all(912)
        is_raw = name.startswith('raw')
        cfg = ModelConfig(input_dim=2000 if is_raw else 1536, n_genes=0 if is_raw else len(genes),
                          scale=4, dropout=0., head_dropout=0.)
        model = ScaledInputs(cfg, None if is_raw else genes).cuda()
        model.expression_scale, model.coordinate_scale = expression_scale, coordinate_scale
        hidden, other = model.optimizer_parameter_groups()
        optimizers = ([torch.optim.Muon(hidden, lr=lr, momentum=.95, weight_decay=.01,
                                        adjust_lr_fn='original'),
                       torch.optim.AdamW(other, lr=3e-4, betas=(.9,.95), weight_decay=.01)]
                      if use_muon else [torch.optim.AdamW(model.parameters(), lr=3e-4,
                                                         betas=(.9,.95), weight_decay=.01)])
        record = dict(case=name, expression_scale=expression_scale, coordinate_scale=coordinate_scale,
                      muon_lr=lr if use_muon else None, dropout=0., coverage=coverage, history=[])
        with torch.no_grad():
            record['initial_expression'] = stats(model.project_expression(active_batch[0]))
            record['initial_position'] = stats(model.position_projection(batch[1].float() * coordinate_scale))
        record['history'].append(dict(step=0, **measure(model, active_batch)))
        started = time.monotonic()
        for step in range(1, args.steps + 1):
            model.train()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                d,b = model(active_batch[0], active_batch[1])
                loss, _, _ = scs_loss(d,b,active_batch[2],active_batch[3])
            if not torch.isfinite(loss):
                record['failure'] = f'nonfinite loss at {step}'
                break
            loss.backward()
            if step in (1, 25, args.steps):
                record[f'gradient_step_{step}'] = {
                    key: stats(value.grad) for key,value in model.named_parameters()
                    if key in ('expression_projection.weight', 'position_projection.weight',
                               'blocks.0.attention.qkv.weight', 'blocks.31.attention.qkv.weight',
                               'direction_head.weight', 'foreground_head.weight') and value.grad is not None}
            for opt in optimizers:
                opt.step()
            if step % 50 == 0 or step == args.steps:
                row = dict(step=step, seconds=time.monotonic()-started, **measure(model, active_batch))
                record['history'].append(row)
                print(json.dumps({'case':name, **row}), flush=True)
                save_json(out / f'{name}.json', record)
        results[name] = record
        save_json(out / f'{name}.json', record)
        del model, optimizers, hidden, other, d, b, loss
        gc.collect()
        torch.cuda.empty_cache()
    save_json(out / 'completed.json', dict(completed=True, cases=list(results)))


if __name__ == '__main__':
    main()
