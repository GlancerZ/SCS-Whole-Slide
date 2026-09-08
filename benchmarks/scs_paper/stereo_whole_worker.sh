#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
source .venv-scs/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -m unittest benchmarks.scs_paper.test_stereo_whole -v
# This bring-up stage intentionally does not auto-start untested training.
# Preserve compute-node tmux and logs while source/alignment QA runs.
python -u benchmarks/scs_paper/stereo_whole_prepare.py --stage index > runs/SCS_stereo_whole_v1/index.log 2>&1
python -u benchmarks/scs_paper/stereo_whole_prepare.py --stage align > runs/SCS_stereo_whole_v1/alignment.log 2>&1 &
brain_pid=$!
for tile in 2104 2105 2106 2107; do
    python -u benchmarks/scs_paper/stereo_whole_liver.py --tile "$tile" > "runs/SCS_stereo_whole_v1/liver_${tile}.log" 2>&1
done
wait "$brain_pid"
python -u benchmarks/scs_paper/stereo_whole_prepare.py --stage pack > runs/SCS_stereo_whole_v1/pack.log 2>&1
