"""Controlled paper-data benchmark: reference SCS vs raw/GenePT scaled models.

Training centres, validation selection, loss, batch budget and postprocessing
are held fixed. This is NOT a claim of bit-exact reproduction of paper scores:
default batch 512, random 80/10/10 split and BF16 differ from upstream defaults.
Only per-bin counts are cached on GPU; GenePT pooling/projection is online.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from optimizations.scs_original_torch.model import OriginalSCS
from optimizations.scs_original_torch.performance import ForeachReferenceAdamW
from optimizations.scs_streaming.genept import load_genept_embeddings
from optimizations.scs_streaming.torch_model import ModelConfig, SCSClassifier, scs_loss


class CountGenePT(SCSClassifier):
    """Equivalent sum(x_g E_g)/number of expressed mapped genes, then Linear.

    Linearity permits projecting the fixed gene table before pooling. Missing
    genes have zero vectors and are excluded from the divisor, explicitly logged.
    No spot x 1536 intermediate is formed or saved.
    """
    def project_expression(self, counts):
        table = self.expression_projection(self.gene_embeddings)
        valid_genes = self.gene_embeddings.square().sum(-1) > 0
        k = ((counts > 0) & valid_genes).sum(-1, keepdim=True)
        pooled = counts @ table / k.clamp_min(1).to(table.dtype)
        return (pooled + (k > 0) * self.spot_bias) * self.config.expression_scale


def metrics(direction, foreground, target, binary):
    positive = binary.astype(bool)
    predicted = foreground >= .5
    fg_recall = float(predicted[positive].mean()) if positive.any() else None
    bg_recall = float((~predicted[~positive]).mean()) if (~positive).any() else None
    confusion = np.bincount(target[positive] * 16 + direction[positive].argmax(1), minlength=256).reshape(16,16)
    total = confusion.sum()
    recalls = np.divide(confusion.diagonal(), confusion.sum(1), out=np.full(16,np.nan), where=confusion.sum(1)>0)
    bacc = (fg_recall + bg_recall)/2 if fg_recall is not None and bg_recall is not None else None
    return dict(n=len(binary), foreground_n=int(sum(positive)), background_n=int(sum(~positive)),
                direction_accuracy=float(confusion.trace()/total) if total else None,
                direction_macro_recall=float(np.nanmean(recalls)) if total else None,
                foreground_accuracy=float((predicted == positive).mean()),
                foreground_recall=fg_recall, background_specificity=bg_recall,
                foreground_balanced_accuracy=bacc, direction_confusion=confusion.tolist())


class Cache:
    def __init__(self, data):
        sample = np.load(data / 'samples.npz')
        self.sample = {k: sample[k] for k in sample.files}
        self.expression = torch.from_numpy(sparse.load_npz(data / 'expression.npz').toarray()).cuda()
        self.neighbors = torch.as_tensor(self.sample['neighbors'].astype(np.int64), device='cuda')
        bs = self.sample['bin_shape']
        ids = torch.arange(len(self.expression), device='cuda')
        bin_size = int(self.sample.get('bin_size', 3))
        self.coords = torch.stack((ids // int(bs[1]), ids % int(bs[1])), -1).float() * bin_size
        self.direction = torch.as_tensor(self.sample['direction'],device='cuda')
        self.foreground = torch.as_tensor(self.sample['foreground'].astype(np.float32),device='cuda')

    def batch(self, rows):
        rows = torch.as_tensor(rows,device='cuda',dtype=torch.int64)
        nb = self.neighbors[rows]
        pos = self.coords[nb]
        return self.expression[nb], pos-pos[:,:1], self.direction[rows], self.foreground[rows]


def evaluate(model, cache, rows, batch_size, amp, save=None):
    model.eval()
    direction, foreground = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            x,pos,d,b = cache.batch(rows[start:start+batch_size])
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                dl,bl = model(x,pos)
            direction.append(dl.float().cpu().numpy())
            foreground.append(bl.float().sigmoid().cpu().numpy())
    direction = np.concatenate(direction)
    foreground = np.concatenate(foreground)
    if save is not None:
        np.savez_compressed(save, rows=rows, direction_logits=direction, foreground_probability=foreground)
    return metrics(direction, foreground, cache.sample['direction'][rows], cache.sample['foreground'][rows]), direction, foreground


def run(args):
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = args.root / args.tile
    out = data / args.method
    out.mkdir(exist_ok=False)
    cache = Cache(data)
    ngenes = cache.expression.shape[1]
    split = cache.sample['split'] if args.split == 'random' else cache.sample['paper_split']
    train, val = [np.flatnonzero(split == s) for s in (0,1)]
    test = np.flatnonzero(split == 2)
    if min(len(train), len(val)) == 0:
        raise ValueError('empty train/validation split')
    genes = json.loads((data / 'genes.json').read_text())
    coverage = None
    if args.method == 'scs_reference':
        model = OriginalSCS(ngenes).cuda()
        optimizers = [ForeachReferenceAdamW(model.parameters())]
        configuration = {'architecture':'OriginalSCS','width':64,'layers':8,'optimizer':'TFA-compatible AdamW','lr':.001}
    else:
        encoding = 'genept' if args.method == 'genept_scale4' else 'raw_counts'
        table = None
        if encoding == 'genept' or args.method == 'raw_matched_scale4':
            lookup,dim = load_genept_embeddings(args.genept)
            matched = np.array([g.upper() in lookup for g in genes])
            per_gene = cache.expression.sum(0).cpu().numpy()
            coverage = dict(genes_matched=int(matched.sum()), genes_total=ngenes,
                            umi_fraction=float(per_gene[matched].sum()/per_gene.sum()),
                            mapping='case-insensitive exact gene symbols; no inferred orthology')
            if encoding == 'genept':
                table = torch.from_numpy(np.stack([lookup.get(g.upper(),np.zeros(dim,np.float32)) for g in genes]))
            else:
                # Coverage-only ablation: preserve raw gene channels, graph,
                # labels and initialization, but remove exactly the same counts.
                cache.expression[:,torch.as_tensor(~matched,device='cuda')]=0
        config = ModelConfig(input_dim=table.shape[1] if table is not None else ngenes,
                             n_genes=ngenes if table is not None else 0, scale=4,
                             expression_encoding=encoding, coordinate_scale=1/30,
                             expression_scale=math.sqrt(table.shape[1]) if table is not None else 1.)
        model = (CountGenePT(config,table) if table is not None else SCSClassifier(config)).cuda()
        mu, adam = model.optimizer_parameter_groups()
        optimizers = [torch.optim.Muon(mu,lr=.002,momentum=.95,weight_decay=.01,adjust_lr_fn='original'),
                      torch.optim.AdamW(adam,lr=.0003,betas=(.9,.95),weight_decay=.01)]
        configuration = config.to_dict() | {'muon_lr':.002,'adamw_lr':.0003}
    npos = int(cache.sample['foreground'][train].sum())
    majority = int(np.bincount(cache.sample['direction'][train][cache.sample['foreground'][train]],minlength=16).argmax())
    run_config = dict(method=args.method, tile=args.tile, seed=args.seed, epochs=args.epochs,
                      batch_size=args.batch_size, amp=args.amp, split=args.split,
                      train_n=len(train), validation_n=len(val), test_n=len(test),
                      train_foreground=npos, train_background=len(train)-npos,
                      train_majority_direction=majority, model=configuration, genept_coverage=coverage,
                      parameters=sum(p.numel() for p in model.parameters()),
                      preparation_sha256=hashlib.sha256((data/'prepared.json').read_bytes()).hexdigest(),
                      samples_sha256=hashlib.sha256((data/'samples.npz').read_bytes()).hexdigest(),
                      torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
                      checkpoint_selection='maximum validation foreground-only direction accuracy',
                      loss='unweighted BCE + foreground-masked direction CE / whole batch; identical across methods')
    if args.method in ('genept_scale4','raw_matched_scale4'):
        h=hashlib.sha256()
        with args.genept.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        run_config['genept_asset_sha256']=h.hexdigest()
    (out/'config.json').write_text(json.dumps(run_config,indent=2))
    print(json.dumps(run_config),flush=True)
    rng = np.random.RandomState(args.seed)
    best = -1.
    history = []
    for epoch in range(1,args.epochs+1):
        tick = time.time()
        model.train()
        order = rng.permutation(train)
        totals = torch.zeros(3,device='cuda')
        for start in range(0,len(order),args.batch_size):
            rows = order[start:start+args.batch_size]
            x,pos,d,b = cache.batch(rows)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.amp=='bf16'):
                dl,bl = model(x,pos)
                loss,ld,lb = scs_loss(dl.float(),bl.float(),d,b)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            loss.backward()
            for opt in optimizers:
                opt.step()
            totals += torch.stack((loss.detach()*len(rows), ((dl.argmax(1)==d)*b).sum(),b.sum()))
        val_metrics,_,_ = evaluate(model,cache,val,args.batch_size,args.amp=='bf16')
        score = val_metrics['direction_accuracy']
        loss_sum,dir_correct,fg_total = totals.cpu().tolist()
        row = dict(epoch=epoch,train_loss=loss_sum/len(train),train_direction_accuracy=dir_correct/max(fg_total,1),
                   val_direction_accuracy=score,val_foreground_bacc=val_metrics['foreground_balanced_accuracy'],
                   epoch_seconds=time.time()-tick)
        history.append(row)
        with (out/'history.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
        if score > best:
            best = score
            torch.save({'model':model.state_dict(),'config':run_config,'epoch':epoch,'validation':val_metrics},out/'best.pt')
        # Resumable state is saved independently of the selected checkpoint.
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save({'model':model.state_dict(),'optimizers':[o.state_dict() for o in optimizers],
                        'epoch':epoch,'config':run_config,'rng':rng.get_state()},out/'latest.pt')
        print(json.dumps(row),flush=True)
    checkpoint = torch.load(out/'best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    result = dict(selected_epoch=checkpoint['epoch'],validation=checkpoint['validation'],
                  elapsed_seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    if len(test):
        result['test'],_,_ = evaluate(model,cache,test,args.batch_size,args.amp=='bf16',out/'test_predictions.npz')
        positive = cache.sample['foreground'][test]
        result['test_majority_direction_accuracy'] = float((cache.sample['direction'][test][positive]==majority).mean())
    # All RNA-bearing input centres, including unlabelled centres, for segmentation.
    all_rows = np.arange(len(cache.neighbors))
    _,dl,fg = evaluate(model,cache,all_rows,args.batch_size,args.amp=='bf16',out/'all_predictions.npz')
    with (out/'spot_prediction.txt').open('w') as f:
        for (r,c),b,logits in zip(cache.sample['coords'],fg,dl):
            f.write(f'{r}\t{c}\t{b:.8g}\t'+':'.join(f'{z:.8g}' for z in logits)+'\n')
    result['elapsed_seconds']=time.time()-started
    (out/'completed.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE '+json.dumps(result),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--tile',required=True)
    p.add_argument('--method',choices=['scs_reference','raw_scale4','genept_scale4','raw_matched_scale4'],required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--amp',choices=['bf16','none'],default='bf16')
    p.add_argument('--split',choices=['random','paper'],default='random')
    p.add_argument('--seed',type=int,default=3812)
    p.add_argument('--genept',type=Path,default=ROOT/'runs/ST19_shared_6000/genept_assets/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle')
    run(p.parse_args())
