#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
source .venv-scs-torch/bin/activate
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:?}/scs_speed_compile"
out=runs/SCS_stereo_whole_v1/speed_explore_v1
for variant in compile_trunk normalized_pool; do
    python -u benchmarks/scs_paper/speed_validate.py --out "$out" --variant "$variant" > "$out/validate_${variant}.log" 2>&1
done
for variant in baseline normalized_pool compile_trunk compile_muon compile_combo; do
    if ! python -u benchmarks/scs_paper/speed_explore.py --out "$out" --variant "$variant" --steps 100 --label repeat \
        > "$out/repeat_${variant}.log" 2>&1; then
        echo "Follow-up variant failed: $variant; retained diagnostic log" >&2
    fi
done
