# Whole-slide SCS

Memory-bounded whole-slide spatial transcriptomics training. The current main
path trains **one PyTorch model across the complete slide**, instead of fitting
one model per tile.

## September 2026 benchmark snapshot

The [Cellist / full GenePT benchmark](benchmarks/cellist_full_genept/README.md)
contains the seven-region preparation, training, CPU segmentation, evaluation,
and audit workflows. [Completed results](benchmarks/cellist_full_genept/results/comparison.md)
include the [fairness review](benchmarks/cellist_full_genept/results/fairness_audit/review.md):
these are fixed-configuration comparisons, with unresolved pseudo-label,
resolution, and evaluation confounders, not a definitive ranking of algorithms.

There are two separate embedding paths in this snapshot. The benchmark uses
the official GenePT dictionary and **all mappable genes**. The evolving shared
dataset builder (`genept_dataset.py`) instead requires a complete NCBI-backed
embedding table; it rejects missing source symbols. These paths and their
checkpoints are not interchangeable.

See [upstream patches and setup](upstream_patches/README.md) for the exact SCS
and Cellist versions and local adaptations needed by these scripts. Launchers
retain the original Slurm/environment paths and must be adapted on another
cluster. Raw data, embeddings, model weights, environments, and full run outputs
are excluded; only compact aggregate results accompany the source code.

## PyTorch scaled model

The PyTorch implementation reads the compact CPU-prepared representation and
provides:

- one shared model and optimizer state across every training tile;
- coupled width/depth scaling: 1x = 64 wide / 8 layers, 2x = 128 / 16,
  and 4x = 256 / 32;
- explicit `torch.nn.functional.scaled_dot_product_attention`;
- BF16 or FP16 autocast on CUDA;
- official `torch.optim.Muon` for hidden transformer matrices, with AdamW for
  input projections, output heads, biases, and normalization parameters;
- real-data physical batch-size autotuning with forward, backward, and both
  optimizer steps included;
- independent GenePT-w embeddings for every original 3x3 spot using every raw
  source gene that has a GenePT mapping, not an old 6,000-HVG subset;
- RNA occupancy and each 50-spot neighborhood are rebuilt from those all-gene
  inputs, while the saved nucleus segmentation is reused for unchanged label
  semantics;
- a learned `Linear(1536, width)` projection (256 outputs for the 4x model) and
  a matching learned `Linear(2, width)` relative-position projection; there is
  no PCA and the unchanged 50 spots remain separate Transformer tokens;
- one-time packing of sparse counts and indices into a globally shuffleable
  array. Full 1,536-dimensional spot embeddings are computed equivalently in
  the current batch and are never materialized or saved to disk;
- fixed 90%/10% foreground/background-stratified random point validation in
  every nonempty tile (with stricter whole-tile holdout still optional);
- atomic, resumable model plus optimizer checkpoints at epoch boundaries.

See [`optimizations/scs_streaming/PYTORCH.md`](optimizations/scs_streaming/PYTORCH.md)
for setup and commands. The prepared biological data and run outputs are not
stored in Git.

## ST19 trial configuration

The raw-expression September 2026 H100 baseline used 6,000 genes, 50 spatial
neighbors, BF16, the 4x model (256 wide, 32 layers), and physical batch size
5,760. The GenePT path instead starts from native 1,536-dimensional vectors for
all mappable source genes, so its physical batch limit is measured again rather
than copied from that obsolete baseline.

With the all-gene GenePT support graph, the current ST19 slide has 6,894,028
training points (6,707,055 foreground and 186,973 background) and 766,030 fixed
validation points (745,235 foreground and 20,795 background). A full epoch
visits every training point once; `--per-class-cap` remains available for
explicit trials. The globally shuffleable sparse dataset has a 3.3 GiB logical
size and contains no per-spot dense embedding array.

The first three-epoch H100 systems trial found a physical batch limit of 6,016
(6,144 OOM), sustained about 12.5k training samples/s, and kept observed GPU
utilization at 95--100%. This validates the pipeline, not model quality: the
best validation direction accuracy was 6.476%, versus a 6.453% majority-class
baseline, and the foreground head predicted no background. The next modeling
iteration should keep full-data random sampling but reweight the binary loss
and retune the learning-rate schedule before spending on longer runs.

## Lineage

The data semantics and reference topology derive from
[chenhcs/SCS](https://github.com/chenhcs/SCS), licensed under MIT. The PyTorch
model preserves its two prediction tasks while replacing the training stack.
Gene-level representations come from
[yiqunchen/GenePT](https://github.com/yiqunchen/GenePT); its official embedding
asset is downloaded separately from the
[GenePT Zenodo record](https://doi.org/10.5281/zenodo.10833191) and is not
committed to this repository.
