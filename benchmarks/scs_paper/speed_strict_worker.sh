#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
source .venv-scs-torch/bin/activate
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:?}/scs_speed_compile"
out=runs/SCS_stereo_whole_v1/speed_explore_v1
# Queued behind our own GPU 1 experiments; never overlap benchmark processes.
while tmux -L scs-speed-followup has-session -t speed 2>/dev/null; do sleep 10; done
python -u benchmarks/scs_paper/speed_validate.py --out "$out" --variant compile_strict > "$out/validate_compile_strict.log" 2>&1
python -u benchmarks/scs_paper/speed_explore.py --out "$out" --variant compile_strict --steps 100 > "$out/compile_strict.log" 2>&1
