#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
source .venv-scs-torch/bin/activate
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:?}/scs_speed_compile"
out=runs/SCS_stereo_whole_v1/speed_explore_v1
while tmux -L scs-speed-strict has-session -t speed 2>/dev/null; do sleep 10; done
python -u benchmarks/scs_paper/speed_validate.py --out "$out" --variant compile_trunk --precision fp32 --rows 64 > "$out/validate_compile_trunk_fp32.log" 2>&1
