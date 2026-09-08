#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
: "${SLURM_JOB_ID:?Requires compute allocation}"
stage=${1:?cpu or seqfish}
case "$stage" in
  cpu|cpu_pilot|evaluation|monitor) test -z "${SLURM_JOB_GPUS:-}" ;;
  seqfish) : ;;
  *) exit 2 ;;
esac
socket="cellist-full-genept-${SLURM_JOB_ID}-${stage}"
if ! tmux -L "$socket" has-session -t "$stage" 2>/dev/null; then
    tmux -L "$socket" new-session -d -s "$stage" \
      "bash benchmarks/cellist_full_genept/${stage}_worker.sh > runs/Cellist_full_genept_v1/logs/${stage}_worker.log 2>&1"
fi
while tmux -L "$socket" has-session -t "$stage" 2>/dev/null; do sleep 10; done
if test "$stage" = evaluation || test "$stage" = monitor; then
    test -f runs/Cellist_full_genept_v1/datasets/stereo_mouse_brain/comparison_completed.json
elif test "$stage" = cpu_pilot; then
    test -f runs/Cellist_full_genept_v1/datasets/2105/segmentation_patches/2105/cellist_paper/segmentation_completed.json
elif test "$stage" = cpu; then
    python - <<'PY'
import json,sys
from pathlib import Path
p=Path('runs/Cellist_full_genept_v1/datasets/stereo_mouse_brain/cellist_status.json')
sys.exit(0 if p.exists() and json.loads(p.read_text())['complete'] else 1)
PY
else
    test -f runs/Cellist_full_genept_v1/datasets/seqfish_rep1_fov1/genept_all_scale4/inference_completed.json
fi
