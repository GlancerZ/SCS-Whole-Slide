#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
test -z "${SLURM_JOB_GPUS:-}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source .venv-scs/bin/activate
while true; do
    python benchmarks/cellist_full_genept/report.py
    if python - <<'PY'
import json,sys
from pathlib import Path
p=Path('runs/Cellist_full_genept_v1/comparison.json')
sys.exit(0 if json.loads(p.read_text())['complete'] else 1)
PY
    then break; fi
    sleep 60
done
python benchmarks/cellist_full_genept/audit_seqfish.py
