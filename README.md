# Whole-slide SCS

Memory-bounded whole-slide spatial transcriptomics training. The current main
path trains **one PyTorch model across the complete slide**, instead of fitting
one model per tile.

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
- atomic, resumable model plus optimizer checkpoints at epoch boundaries.

See [`optimizations/scs_streaming/PYTORCH.md`](optimizations/scs_streaming/PYTORCH.md)
for setup and commands. The prepared biological data and run outputs are not
stored in Git.

## ST19 trial configuration

The September 2026 H100 trial uses 6,000 genes, 50 spatial neighbors, BF16,
the 4x model (256 wide, 32 layers, 23,670,289 parameters), and physical batch
size 5,760. On an 80 GB H100, 5,760 completed repeated real-data training
steps while 5,888 failed with CUDA OOM at 128-sample search granularity.

The training split contains 6,391,825 eligible points. The default balanced
regional sampling budget selects 962,007 points per epoch; this cap is explicit
and should not be described as a full pass over all eligible points.

## Lineage

The data semantics and reference topology derive from
[chenhcs/SCS](https://github.com/chenhcs/SCS), licensed under MIT. The PyTorch
model preserves its two prediction tasks while replacing the training stack.
