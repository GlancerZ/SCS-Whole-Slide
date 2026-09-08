#!/usr/bin/env bash
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
source /home/glancerz/labwork/codex/TLS/.venv-scs/bin/activate
scs_socket="scs-parallel-${SLURM_JOB_ID}"
scs_runtime="runs/ST19_whole_scs/runtime/${SLURM_JOB_ID}"
mkdir -p "$scs_runtime"
if tmux -L "$scs_socket" has-session -t queue 2>/dev/null; then
    echo "Parallel queue already exists for job ${SLURM_JOB_ID}"
    exit 0
fi
tmux -L "$scs_socket" new-session -d -s queue "cd /home/glancerz/labwork/codex/TLS && source /home/glancerz/labwork/codex/TLS/.venv-scs/bin/activate && export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 && python -u scripts/scs_whole_parallel.py >> ${scs_runtime}/queue.log 2>&1"
# Keep this srun step alive: Slurm otherwise cleans up the tmux children.
while tmux -L "$scs_socket" has-session -t queue 2>/dev/null; do
    sleep 15
done
tail -30 "${scs_runtime}/queue.log"
