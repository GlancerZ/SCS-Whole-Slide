#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
source .venv-scs-torch/bin/activate
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:?}/scs_speed_compile"
out=runs/SCS_stereo_whole_v1/speed_explore_v1
if test ! -f "$out/baseline_b4096.json"; then
    python -u benchmarks/scs_paper/speed_explore.py --out "$out" --variant baseline --profile >> "$out/baseline.log" 2>&1
fi
for variant in fused_adam normalized_pool compile_blocks compile_trunk; do
    if test ! -f "$out/${variant}_b4096.json"; then
        if ! python -u benchmarks/scs_paper/speed_explore.py --out "$out" --variant "$variant" >> "$out/$variant.log" 2>&1; then
            echo "Variant failed: $variant; retained its diagnostic log, continuing" >&2
        fi
    fi
done
