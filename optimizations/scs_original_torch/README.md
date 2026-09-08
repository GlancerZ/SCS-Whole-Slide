# Original SCS → PyTorch + SDPA

This is an isolated port of the original **per-tile** SCS model, not the separate
`scs_streaming/shared_train_torch.py` scaled/Muon experiment. Existing running
queues, original source, checkpoints and segmentation results are unchanged.

## Preserved behavior

- Input: original `data/x_train_*.npz`, positions, labels, test arrays, and
  `spots*.h5ad`. Uses each tile's existing gene features (currently 2,000), not
  the separately prepared shared 6,000-gene dataset. Never mix tile gene orders.
- 50 neighbors by default, raw expression and relative coordinates.
- 8 pre-norm Transformer blocks, width 64, one 64-dimensional attention head.
- Attention weight dropout 0.1, exact GELU, MLP 128→64, LayerNorm epsilon 1e-6.
- First-token readout; head 1024→256 with dropout 0.5; 16 direction logits and
  binary foreground prediction. Binary logits are internal; output is sigmoid.
- Masked categorical cross entropy and binary cross entropy; direction masking
  uses the direction labels and reduction divides by the full batch size.
- Original spatial validation split and foreground-only validation. As upstream,
  >=100 validation points selects `val_pos_out_accuracy`; otherwise selects
  training `pos_out_accuracy` (not a reliable generalization estimate).
- Default batch 10, 100 epochs, **no early stopping**, reload **best.pt** before
  inference. `latest.pt` also stores optimizer state, but automatic training
  resumption is **not implemented** by this initial port.
- TFA legacy AdamW: lr 0.001, betas 0.9/0.999, epsilon 1e-7, per-step weight
  decay 0.0001, independent of learning rate. This is deliberately not a direct
  substitution of PyTorch AdamW with superficially identical arguments.
- Original prediction text contract: all training points followed by test points,
  columns `x`, `y`, foreground probability, and 16 colon-separated direction
  logits. Original SCS postprocessing can consume this format; this port does
  not automatically run postprocessing or promote output to the active queue.

## SDPA and numerical scope

`model.Attention` calls `torch.nn.functional.scaled_dot_product_attention`
explicitly. Evaluation uses `dropout_p=0.0`; training uses 0.1. PyTorch dispatches
the supported backend: **using SDPA does not promise FlashAttention** on every
device/dtype. See [official SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

Default precision is float32. Optional `--amp bf16` changes precision and must be
validated on GPU before production use. No fp16, forced Flash backend, increased
model size, Muon, or changed optimizer settings are silently enabled.

Dense layers use Glorot-uniform/zero-bias initialization. Framework RNGs,
dropout masks, shuffles and floating point kernels differ, so independently
trained models are not expected to be bit-identical. The checkpoint bridge
tests deterministic evaluation and gradients with the **same weights**.

## Environment and execution

Use the existing `.venv-scs-torch` (PyTorch, NumPy, h5py). Only h5py was added;
the installed PyTorch and NumPy versions were not replaced. Run project code
inside an allocated compute job, not on the login node. Long training must be
launched in compute-node tmux, consistent with this workspace's Slurm policy.

After entering the allocated node:

```bash
source /lustre09/project/6102157/glancerz/codex/TLS/.venv-scs-torch/bin/activate
python -B -m optimizations.scs_original_torch.runner \
  --data-dir runs/ST19_whole_scs/tiles/x11_y12/data \
  --output-dir runs/ST19_original_torch_x11_y12 \
  --epochs 100 --batch-size 10 --inference-batch-size 128 \
  --device cuda --gpu-memory-mib 16384
```

Output must be a fresh directory. Memory cap bounds PyTorch's caching allocator,
not all driver/context memory. Original NPZ arrays are decompressed once to a
private scratch directory and memory mapped; only minibatches are cast to float32.
Test expression is not loaded during training. Scratch must have enough free
space for the decompressed arrays; SLURM_TMPDIR is preferred.

## Checkpoint conversion and validation

The two stages use separate processes/environments, so TensorFlow and PyTorch
need not coexist in one environment. Run both in the same CPU allocation if
using node-local `/tmp` exchange paths.

```bash
# Existing TensorFlow environment, on a compute node:
source /lustre09/project/6102157/glancerz/codex/TLS/.venv-scs/bin/activate
python -B -m optimizations.scs_original_torch.bridge export-tf \
  --checkpoint runs/ST19_whole_scs/tiles/x11_y12/optimized_training/job20305630_1788623771281538687/ckpt/ckpt \
  --data-dir runs/ST19_whole_scs/tiles/x11_y12/data \
  --output /tmp/scs-original-reference.npz

# PyTorch environment, same compute node:
source /lustre09/project/6102157/glancerz/codex/TLS/.venv-scs-torch/bin/activate
python -B -m optimizations.scs_original_torch.bridge check-torch \
  --source /tmp/scs-original-reference.npz \
  --report /tmp/scs-original-equivalence.json \
  --checkpoint-output /tmp/scs-original-converted-best.pt
python -B -m unittest optimizations.scs_original_torch.test_original -v
```

Conversion transfers model weights, **not TensorFlow optimizer state**. Real
validation reads the first 16 training points directly from compressed streams,
not an entire expanded expression tensor. Tests compare direction logits,
foreground probabilities, losses, expression/position input gradients, class
argmax, foreground threshold, parameter count, and a separate multi-step AdamW
reference. They also cover explicit SDPA forward/backward, dropout behavior,
label masking and an end-to-end synthetic train→best checkpoint→prediction run.

To infer using an already converted best checkpoint (no retraining):

```bash
python -B -m optimizations.scs_original_torch.predict \
  --checkpoint optimizations/scs_original_torch/validation/st19_x11_y12_best.pt \
  --data-dir runs/ST19_whole_scs/tiles/x11_y12/data \
  --output-dir runs/ST19_original_torch_x11_y12_prediction \
  --batch-size 128 --device cuda
```

This command is documented but has not been run on the full real tile. The
checkpoint's recorded tile path must match; mismatched tile feature orders are
rejected. Synthetic checkpoints without a recorded data path cannot be used here.

See `validation/` for observed results. CPU numerical tests are not a GPU
throughput benchmark, a full-slide quality assessment, or validation of bf16.

## Opt-in H100 tuning (2026-09-05)

Original defaults remain unchanged. These independent switches were added:

- `--optimizer-mode foreach`: multi-tensor updates preserving the custom TFA
  AdamW equations. Thirty CPU/CUDA reference steps, including missing gradients,
  are tested; the observed CUDA maximum parameter error was zero.
- `--input-mode gpu_cache`: decompress one tile, upload its full training input
  in FP32 chunks, then gather batches on GPU. Validation rows share the cache;
  test expression is still not opened until inference. This deliberately trades
  VRAM for throughput, and is not appropriate when the full training tensor will
  exceed the allocation's memory budget. The streaming path remains the default.
- `--matmul-precision high`: optional reduced internal matmul precision, distinct
  from the default `highest`. BF16 remains a separate `--amp bf16` option.
- `--compile-model`: optional CUDA-only compiled training forward/backward
  (`reduce-overhead`, full graph, fixed-shape specialization). Optimizer updates
  remain outside the compiled model, so the original Python step counter and
  bias correction are not accidentally frozen in a CUDA graph. Validation and
  final inference use the original eager model; checkpoints retain original
  parameter names and work with the existing predictor. Compiled dropout uses
  a different RNG implementation: do not expect same-seed weights to be equal.

Example fast **experimental**, FP32 configuration, inside a compute-node tmux:

```bash
python -B -u -m optimizations.scs_original_torch.runner \
  --data-dir runs/ST19_whole_scs/tiles/x11_y12/data \
  --output-dir runs/ST19_original_torch_fast_new_run \
  --epochs 100 --batch-size 256 --inference-batch-size 256 \
  --optimizer-mode foreach --input-mode gpu_cache \
  --amp none --matmul-precision highest --gpu-memory-mib 70000 --threads 4
```

For a conservative path that keeps the original update schedule, use batch 10.
Adding `--compile-model` can further accelerate that path; set
`TORCHINDUCTOR_COMPILE_THREADS=4` to bound compiler CPU parallelism in the tested
16-CPU allocation. Compilation has a cold-start cost and may specialize again
for a different final batch shape. Numerical and throughput observations live
in `runs/ST19_original_torch_compile_20305630`; see the tuning notes for the
separate full-pipeline validation and its limited epoch count.

Tested compiled path (3 epochs plus all predictions, **not** a 100-epoch run):

```bash
TORCHINDUCTOR_COMPILE_THREADS=4 python -B -u -m optimizations.scs_original_torch.runner \
  --data-dir runs/ST19_whole_scs/tiles/x11_y12/data \
  --output-dir runs/ST19_original_torch_compiled_new_run \
  --epochs 3 --batch-size 10 --inference-batch-size 256 \
  --optimizer-mode foreach --input-mode gpu_cache --compile-model \
  --amp none --matmul-precision highest --gpu-memory-mib 70000 --threads 4
```

Increasing batch size is **not** a numerically equivalent implementation change:
at x11_y12, batch 10 makes 2,138 optimizer updates/epoch, versus 84 at batch 256.
The original per-step weight decay therefore also accumulates differently.
Learning rate, weight decay, architecture, 100-epoch limit, validation split, and
best-checkpoint selection have not been silently retuned. A fast run or an
unchanged model's precision check does not establish comparable segmentation
quality or convergence. No tuning output is promoted to the whole-slide queue.

The isolated sweep is in `runs/ST19_original_torch_tuning_20305630`: 16 configs,
three timed trials each, excluding decompression, validation, checkpoint writing,
and output-to-disk. Profiles are separate short captures; their overhead must
not be used as unprofiled wall time. Large-batch warm throughput can also benefit
from page caching, so use the full-run benchmark for practical completion times.

Full-run checks are in `runs/ST19_original_torch_tuned_fp32_bs256_20305630`, with
separate batch-10 and batch-64 controls. `completed.json` now separates setup,
training (including checkpoint writes), and inference time. Driver wall time
also includes process startup, cleanup and the sampling interval. GPU allocator
peaks exclude CUDA context memory; the benchmark's nvidia-smi process samples
capture that separately. Process RSS is a lifetime high-water mark, not current
resident memory. All tests reuse allocation 20305630 and run serially.

Measured results and caveats: `runs/ST19_original_torch_tuning_notes.md`.
