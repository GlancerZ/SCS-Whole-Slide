#!/usr/bin/env bash
# Source this from the shared project directory on a compute node.
cellist_workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$cellist_workspace/.venv-cellist/bin/activate"
export UV_CACHE_DIR="$cellist_workspace/tools/uv-cache"
export XDG_CACHE_HOME="$cellist_workspace/runs/ST19_cellist/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CELLIST_THREADS=${CELLIST_THREADS:-4}
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TORCH_HOME"
