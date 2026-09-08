#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
test -z "${SLURM_JOB_GPUS:-}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source .venv-scs/bin/activate
for dataset in 2104 2105 2106 2107 seqfish_rep1_fov0 seqfish_rep1_fov1 stereo_mouse_brain; do
    data="runs/Cellist_full_genept_v1/datasets/${dataset}"
    while ! python - "$data" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
if not (p/'genept_all_scale4/inference_completed.json').exists(): sys.exit(1)
patches=json.loads((p/'patch_ranges.json').read_text())
sys.exit(0 if all((p/'segmentation_patches'/t['id']/'cellist_paper/segmentation_completed.json').exists() for t in patches) else 1)
PY
    do sleep 20; done
    if ! test -f "$data/comparison_completed.json"; then
        python -u benchmarks/cellist_full_genept/evaluate.py --dataset "$dataset" --workers 4 \
          > "runs/Cellist_full_genept_v1/logs/evaluate_${dataset}.log" 2>&1
    fi
done
