#!/usr/bin/env bash
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
: "${SLURM_JOB_ID:?Run inside the selected CPU allocation}"
source /home/glancerz/labwork/codex/TLS/.venv-scs/bin/activate
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
if ! tmux -L scs-shared-cpu has-session -t prepare 2>/dev/null; then
    tmux -L scs-shared-cpu new-session -d -s prepare \
        'python -B -u scripts/scs_shared_cpu.py >> runs/ST19_shared_6000.cpu.log 2>&1'
fi
# Keep the Slurm step alive so allocation cleanup cannot kill the tmux worker.
while tmux -L scs-shared-cpu has-session -t prepare 2>/dev/null; do
    sleep 10
done
