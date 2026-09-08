#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
source .venv-scs-torch/bin/activate
test "${SLURM_JOB_ID}" = "20305630"
scs_compile_socket=scs-compile-20305630
if tmux -L "$scs_compile_socket" has-session -t compile 2>/dev/null; then
    echo 'Compile benchmark already running' >&2
    exit 1
fi
tmux -L "$scs_compile_socket" new-session -d -s compile \
    'cd /lustre09/project/6102157/glancerz/codex/TLS && source .venv-scs-torch/bin/activate && while tmux -L scs-tuned-validation-20305630 has-session -t validation 2>/dev/null; do sleep 15; done; export CUDA_VISIBLE_DEVICES=GPU-3d51d05e-eef7-c209-2519-7bd67ed19d82 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4; timeout 900 python -B -u -m optimizations.scs_original_torch.tune_compile >> runs/ST19_original_torch_compile_20305630.log 2>&1'
while tmux -L "$scs_compile_socket" has-session -t compile 2>/dev/null; do
    sleep 15
done
tail -8 runs/ST19_original_torch_compile_20305630.log
python -c 'import json; from pathlib import Path; s=json.loads(Path("runs/ST19_original_torch_compile_20305630/status.json").read_text()); assert s["state"]=="completed", s'
