#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
source .venv-scs-torch/bin/activate
test "${SLURM_JOB_ID}" = "20305630"
python -c 'import json; from pathlib import Path; s=json.loads(Path("runs/ST19_original_torch_compile_20305630/status.json").read_text()); assert s["state"]=="completed" and s["numerical_check"]["passed"]; assert s["results"][1]["samples_per_second"] > 1.5*s["results"][0]["samples_per_second"]'
scs_compiled_validation_socket=scs-compiled-validation-20305630
if tmux -L "$scs_compiled_validation_socket" has-session -t validation 2>/dev/null; then
    echo 'Compiled validation already running' >&2
    exit 1
fi
tmux -L "$scs_compiled_validation_socket" new-session -d -s validation \
    'cd /lustre09/project/6102157/glancerz/codex/TLS && source .venv-scs-torch/bin/activate && export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 && exec python -B -u scripts/scs_original_torch_benchmark.py --output runs/ST19_original_torch_compiled_fp32_bs10_20305630_v2 --gpu GPU-3d51d05e-eef7-c209-2519-7bd67ed19d82 --tiles x11_y12 --epochs 3 --batch-size 10 --inference-batch-size 256 --optimizer-mode foreach --input-mode gpu_cache --compile-model >> runs/ST19_original_torch_compiled_validation_20305630.log 2>&1'
while tmux -L "$scs_compiled_validation_socket" has-session -t validation 2>/dev/null; do
    sleep 15
done
tail -1 runs/ST19_original_torch_compiled_validation_20305630.log
python -c 'import json; from pathlib import Path; s=json.loads(Path("runs/ST19_original_torch_compiled_fp32_bs10_20305630_v2/status.json").read_text()); assert s["state"]=="completed", s'
