#!/usr/bin/env bash
# Bounded engineering verification on real ST19 data, NOT a full segmentation run.
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
source /home/glancerz/labwork/codex/TLS/.venv-scs/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
shared_smoke_root=runs/ST19_shared_6000_smoke
CUDA_VISIBLE_DEVICES= python -B -m optimizations.scs_streaming.shared_prepare init \
  --source runs/ST19_whole_scs --output "$shared_smoke_root" --n-genes 6000 \
  --hvg-bins-per-tile 20000 --tiles x10_y11 x12_y11 x11_y11 --validation-tiles x11_y11
CUDA_VISIBLE_DEVICES= python -B -m optimizations.scs_streaming.shared_prepare prepare --output "$shared_smoke_root"
python -B -m optimizations.scs_streaming.shared_train train --root "$shared_smoke_root" \
  --epochs 1 --per-class-cap 32 --gpu-memory-mib 4096
python -B -m optimizations.scs_streaming.shared_train train --root "$shared_smoke_root" \
  --epochs 2 --per-class-cap 32 --gpu-memory-mib 4096 --resume
python -B -m optimizations.scs_streaming.shared_train predict --root "$shared_smoke_root" \
  --tiles x10_y11 x12_y11 --inference-batch-size 32 --gpu-memory-mib 4096
CUDA_VISIBLE_DEVICES= python -B -m optimizations.scs_streaming.shared_train postprocess \
  --root "$shared_smoke_root" --tiles x10_y11
CUDA_VISIBLE_DEVICES= python -B -m optimizations.scs_streaming.validate_shared_real \
  --root "$shared_smoke_root"
