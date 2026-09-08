"""Reproducible 2000-gene repair pilot with fixed, disjoint centre subsets.

The source neighbourhood graph is retained to isolate expression changes.
Training/validation context can overlap, as in the original random-point split.
This executable writes a new run and never resumes/overwrites the old baseline.
"""
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .shared_data import save_json
from .shared_train_torch import atomic_torch_save, configure_runtime, run_epoch
from .torch_data import GenePTBatchDataset
from .torch_model import ModelConfig, SCSClassifier


def loader(dataset, order, batch_size):
    dataset.order = order
    dataset.samples = len(order)
    dataset.batch_size = batch_size
    dataset.steps = math.ceil(len(order)/batch_size)
    return DataLoader(dataset, batch_size=None, num_workers=2, pin_memory=True,
                      prefetch_factor=2, multiprocessing_context='fork')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs/ST19_shared_6000')
    p.add_argument('--dataset-name', default='genept_allgenes_linear_random90')
    p.add_argument('--selection', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--encoding', choices=['genept', 'raw_counts'], required=True)
    p.add_argument('--train-samples', type=int, default=131072)
    p.add_argument('--validation-samples', type=int, default=32768)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=512)
    args = p.parse_args()
    if min(args.train_samples, args.validation_samples, args.epochs, args.batch_size) <= 0:
        raise ValueError('pilot sizes, epochs and batch size must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    _, device, dtype = configure_runtime(3812, 'bf16')
    selection_bytes = Path(args.selection).read_bytes()
    selection = json.loads(selection_bytes)
    selected = selection['selected_gene_ids']
    if len(selected) != 2000 or selection['split'] != 'train':
        raise ValueError('expected 2000 genes selected using training centres only')
    train = GenePTBatchDataset(args.root, 'train', args.batch_size, 8241,
                              dataset_name=args.dataset_name, residency='memory',
                              selected_gene_ids=selected)
    validation = GenePTBatchDataset(args.root, 'validation', args.batch_size, 8241,
                                   dataset_name=args.dataset_name, residency='mmap',
                                   selected_gene_ids=selected)
    train_ids = train.order[:min(args.train_samples, train.samples)].copy()
    validation_ids = validation.order[:min(args.validation_samples, validation.samples)].copy()
    # Dataset split tables contain global expression IDs, making leakage checks explicit.
    train_centres = np.asarray(train.neighbors[train_ids, 0])
    validation_centres = np.asarray(validation.neighbors[validation_ids, 0])
    if np.intersect1d(train_centres, validation_centres).size:
        raise ValueError('train and validation centre overlap')
    np.save(output / 'train_sample_ids.npy', train_ids)
    np.save(output / 'validation_sample_ids.npy', validation_ids)
    save_json(output / 'gene_selection.json', selection)
    negative, positive = np.bincount(train.foreground[train_ids], minlength=2).tolist()
    if not negative or not positive:
        raise ValueError('pilot must contain both foreground classes')
    class_weights = (len(train_ids)/(2*negative), len(train_ids)/(2*positive))
    train_direction_counts = np.bincount(
        train.directions[train_ids][train.foreground[train_ids] == 1], minlength=16)
    majority_direction = int(train_direction_counts.argmax())
    val_foreground = validation.foreground[validation_ids] == 1
    majority_accuracy = float(np.mean(
        validation.directions[validation_ids][val_foreground] == majority_direction))
    genept = args.encoding == 'genept'
    if genept:
        raw_genes = np.load(Path(args.root) / args.dataset_name / "gene_embeddings.npy")
        genes = torch.from_numpy(raw_genes[selected])
        gene_dim = int(raw_genes.shape[1])
        if gene_dim <= 0:
            raise ValueError("invalid gene embedding dimension")
    else:
        genes = None
        gene_dim = 2000
    config = ModelConfig(input_dim=gene_dim if genept else 2000,
                         n_genes=2000 if genept else 0, scale=4,
                         expression_scale=math.sqrt(gene_dim) if genept else 1.,
                         coordinate_scale=1/30, expression_encoding=args.encoding)
    model = SCSClassifier(config, genes).to(device)
    hidden, other = model.optimizer_parameter_groups()
    muon = torch.optim.Muon(hidden, lr=.002, momentum=.95, weight_decay=.01, adjust_lr_fn='original')
    adamw = torch.optim.AdamW(other, lr=3e-4, betas=(.9,.95), weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    run_config = dict(model=config.to_dict(), train_samples=len(train_ids),
                      validation_samples=len(validation_ids), train_class_counts=[negative, positive],
                      foreground_class_weights=class_weights, selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
                      dataset_name=args.dataset_name, schema_fingerprint=train.schema['fingerprint'],
                      batch_size=args.batch_size, epochs=args.epochs, muon_lr=.002, adamw_lr=.0003,
                      weight_decay=.01, seed=3812, sample_seed=8241,
                      gene_selection=str(Path(args.selection).resolve()), amp='bf16',
                      train_majority_direction_class=majority_direction,
                      validation_majority_direction_baseline=majority_accuracy,
                      validation_note='Disjoint centre spots; overlapping same-slide spatial context is possible',
                      scope='fixed random subset pilot; not a full-slide epoch')
    save_json(output/'config.json', run_config)
    print(json.dumps({'config': run_config}), flush=True)
    started = time.monotonic()
    best_score = -math.inf
    for epoch in range(args.epochs):
        order = np.random.RandomState(9021+epoch).permutation(train_ids)
        metrics = run_epoch(model, loader(train, order, args.batch_size), device, dtype,
                            muon, adamw, scaler, log_every=50,
                            foreground_class_weights=class_weights)
        val = run_epoch(model, loader(validation, validation_ids, args.batch_size), device, dtype,
                        foreground_class_weights=class_weights)
        record = dict(epoch=epoch+1, training=metrics, validation=val,
                      seconds=time.monotonic()-started)
        save_json(output/f'epoch_{epoch+1:05d}.json', record)
        print(json.dumps(record), flush=True)
        # A checkpoint must reflect both tasks; avoid choosing only a direction majority baseline.
        score = val['foreground_accuracy'] + val['binary_balanced_accuracy']
        state = dict(model=model.state_dict(), muon=muon.state_dict(), adamw=adamw.state_dict(),
                     epoch=epoch+1, config=run_config, score=score,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        atomic_torch_save(torch, state, output/'latest.pt')
        if score > best_score:
            best_score = score
            atomic_torch_save(torch, state, output/'best.pt')
        save_json(output/'training_state.json', dict(completed=False, epoch=epoch+1,
                                                    target_epochs=args.epochs, best_score=best_score))
        del state
        gc.collect()
    result = dict(completed=True, epochs=args.epochs, best_score=best_score,
                  elapsed_seconds=time.monotonic()-started, final_validation=val)
    save_json(output/'completed.json', result)
    save_json(output/'training_state.json', result)


if __name__ == '__main__':
    main()
