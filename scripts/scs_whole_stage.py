#!/usr/bin/env python3
"""One isolated SCS stage with bounded GPU memory and reproducible seeds."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SCS"))
sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "preprocess", "train", "postprocess"])
    p.add_argument("--tile", required=True)
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    directory = ROOT / "runs/ST19_whole_scs/tiles" / args.tile
    from scs_parallel_resources import lease, BASE
    configured = (BASE / "runtime" / os.environ.get("SLURM_JOB_ID", "none") / "config.json").exists()
    if configured or args.stage == "train":
        with lease(args.stage, directory):
            run_stage(args, directory)
    else:
        # Preserve independent CPU-only preprocessing workflows.
        run_stage(args, directory)


def run_stage(args, directory):
    for name in ["data", "results", "fig"]:
        (directory / name).mkdir(exist_ok=True)
    os.chdir(directory)
    import numpy as np
    np.random.seed(20260905)
    if args.stage == "prepare":
        from scs_whole_prepare import BASE, prepare_tile
        manifest = json.loads((BASE / "manifest.json").read_text())
        prepare_tile(next(t for t in manifest["tiles"] if t["id"] == args.tile))
    elif args.stage == "preprocess":
        from src.preprocessing import preprocess
        preprocess(str(directory / "expression.tsv"), str(directory / "hematoxylin.tif"),
                   True, None, 0, 0, 0, 3, 50, preserve_origin=True)
    elif args.stage == "train":
        from optimizations.scs_streaming.train import train
        output = directory / "optimized_training" / f"job{os.environ['SLURM_JOB_ID']}_{time.time_ns()}"
        result = train(directory / "data", output, epochs=args.epochs, val_ratio=0.0625,
                       gpu_memory_mib=int(os.environ["SCS_GPU_MEMORY_MB"]), input_mode="arrays",
                       inference_batch_size=10)
        assert result["inference_completed"] and result["prediction_rows"] > 0
        source = output / "results/spot_prediction_0:0:0:0.txt"
        target = directory / "results/spot_prediction_0:0:0:0.txt"
        temporary = target.with_suffix(".txt.promoting")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    else:
        from src.postprocessing import postprocess
        postprocess(0, 0, 0, 3, 15, plot=False)


if __name__ == "__main__":
    main()
