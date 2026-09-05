# PyTorch whole-slide SCS

This path trains one model across all prepared ST19 training regions. It reads
the compact CPU output in `runs/ST19_shared_6000`; it does not materialize a
whole-slide dense tensor and it never resets weights between tiles.

## What changed

- PyTorch 2.13 is the primary training implementation.
- Attention calls `torch.nn.functional.scaled_dot_product_attention` directly.
- CUDA training uses BF16 autocast by default (FP16 is optional).
- Hidden transformer matrices use official `torch.optim.Muon`; input/position
  projections, heads, normalization parameters, and biases use AdamW.
- `--scale 1`, `2`, or `4` scales both width and depth: 64/8, 128/16,
  or 256/32. Head dimension remains 64 while head count grows with width.
- The batch-size tuner runs real forward, backward, Muon, and AdamW steps on
  real prepared examples and records the largest tested stable physical batch.

## Environment

On the login node (networked), create a modern manylinux-compatible Python
environment and install the H100 build:

```bash
./tools/uv venv .venv-scs-torch --python 3.11
./tools/uv pip install --python .venv-scs-torch/bin/python \
  'torch==2.13.0+cu130' numpy scipy \
  --index https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
```

Run all model code inside a GPU allocation:

```bash
source .venv-scs-torch/bin/activate
python -B -m unittest optimizations.scs_streaming.test_torch -v
python -B -m optimizations.scs_streaming.shared_train_torch inspect \
  --root runs/ST19_shared_6000
python -B -m optimizations.scs_streaming.shared_train_torch autotune \
  --root runs/ST19_shared_6000 --scale 4 --amp bf16 \
  --start-batch 64 --max-batch 4096
python -B -m optimizations.scs_streaming.shared_train_torch train \
  --root runs/ST19_shared_6000 --scale 4 --amp bf16 \
  --batch-size <selected_batch_size> --epochs 1 --run-name torch_scale4_trial
```

The full training run writes configuration, epoch metrics, resumable model and
both optimizer states, and best/latest checkpoints below the selected run
directory. A `--max-steps` run is explicitly marked as a trial and is not
published as a completed epoch.
