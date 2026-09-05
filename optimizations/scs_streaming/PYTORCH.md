# PyTorch whole-slide SCS with GenePT spots

This path trains one model across all prepared ST19 regions. It preserves the
original SCS 50-spot spatial sequence, while replacing each raw sparse spot
vector with one GenePT-w embedding. The old 6,000-HVG list does not filter the
genes used by this representation and does not define RNA occupancy: the
50-spot graph is rebuilt from all source genes that have a GenePT mapping. The
saved nucleus segmentation is reused to label the resulting center spots.

## Exact spot representation

For one spot with `k` nonzero genes that map to GenePT, the model represents it
as

```text
spot_1536 = sum(raw_count_i * GenePT_1536(gene_i)) / k
spot_token = Linear_1536_to_width(spot_1536)
```

Five mapped nonzero genes are divided by five, ten by ten, and three by three.
Missing GenePT genes are excluded from both numerator and denominator. Counts
from duplicate source identifiers with the same gene symbol are combined, so
the symbol is counted once in `k`. Every 3x3 spot is encoded independently.
The 50 spots are never pooled together before the segmentation Transformer.

The official GenePT `text-embedding-ada-002` vectors have 1,536 dimensions.
There is no PCA. At scale 4 the learned expression layer is
`Linear(1536, 256)`, while relative `(dx, dy)` positions pass through a separate
`Linear(2, 256)` and are added to the corresponding spot token. Scale 1 and 2
use widths 64 and 128 respectively.

The disk dataset stores only raw sparse counts, compact gene indices, neighbor
references, positions, and labels, plus one shared native GenePT gene table. It
does not store a 1,536-dimensional vector per spot. By linearity, each forward
pass first projects the shared gene table with the current learned weight and
then performs sparse weighted pooling. This is exactly equal to constructing
each temporary 1,536-dimensional spot average and applying the same linear
layer, but uses substantially less batch memory.

## Model and training

- PyTorch 2.13 with BF16 autocast on CUDA.
- Explicit `torch.nn.functional.scaled_dot_product_attention`.
- Official `torch.optim.Muon` for Transformer hidden matrices; AdamW for the
  input/position projections, heads, normalization parameters, and biases.
- Coupled width/depth scaling: 1x = 64/8, 2x = 128/16, 4x = 256/32.
- A global, foreground/background-stratified 90%/10% random point split.
- All 6,894,028 training points per full epoch; no per-tile density cap.
- 766,030 fixed validation points; train/validation center spots are disjoint.
- Atomic epoch checkpoints containing the model and both optimizer states.

Random point validation intentionally shares overlapping spatial context with
training points and is therefore a same-slide interpolation metric, not a
strict cross-region generalization estimate.

## Current ST19 systems trial

On one NVIDIA H100 80GB, the 4x model passed a full forward/backward plus
Muon/AdamW step at batch 6,016 and failed at 6,144. A three-epoch run sustained
about 12.5k samples/s with observed GPU utilization of 95--100%. The best
validation direction accuracy was 6.476%, effectively tied with the 6.453%
majority-class baseline. Binary specificity was zero because the head predicted
every validation point as foreground. These checkpoints demonstrate a working,
restartable whole-slide pipeline but should not be treated as a useful trained
segmenter. Preserve full-data sampling and address the minority-background loss
weight and learning-rate schedule before a long run.

## Reproducible commands

Download and checksum the CC-BY-4.0 GenePT asset from its official Zenodo
record on a networked node:

```bash
bash scripts/download_genept_assets.sh \
  runs/ST19_shared_6000/genept_assets
```

Run the probe and full preprocessing inside the existing CPU allocation:

```bash
source .venv-scs/bin/activate
python -B -m optimizations.scs_streaming.genept_dataset probe \
  --root runs/ST19_shared_6000 \
  --gene-embeddings <GenePT_gene_embedding_ada_text.pickle>
python -B -m optimizations.scs_streaming.genept_dataset build \
  --root runs/ST19_shared_6000 \
  --gene-embeddings <GenePT_gene_embedding_ada_text.pickle> \
  --output-name genept_allgenes_linear_random90
```

Then inspect and autotune on the H100 before training:

```bash
source .venv-scs-torch/bin/activate
python -B -m optimizations.scs_streaming.shared_train_torch inspect \
  --root runs/ST19_shared_6000
python -B -m optimizations.scs_streaming.shared_train_torch autotune \
  --root runs/ST19_shared_6000 --scale 4 --amp bf16 \
  --start-batch 4096 --max-batch 8192 --batch-granularity 128 \
  --input-pipeline genept --dataset-name genept_allgenes_linear_random90 \
  --residency mmap
python -B -m optimizations.scs_streaming.shared_train_torch train \
  --root runs/ST19_shared_6000 --scale 4 --amp bf16 \
  --batch-size <selected_batch_size> --epochs 1 --per-class-cap 0 \
  --input-pipeline genept --dataset-name genept_allgenes_linear_random90 \
  --residency memory --run-name torch_genept_scale4
```
