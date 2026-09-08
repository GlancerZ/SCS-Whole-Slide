#!/usr/bin/env python3
"""Sequential, isolated real-tile benchmark. Run in compute-node tmux."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def gpu_processes(uuid):
    text = subprocess.check_output(["nvidia-smi", "--id=" + uuid,
        "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    return [line.strip() for line in text.splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--tiles", nargs="+", default=["x11_y12", "x11_y09"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--inference-batch-size", type=int, default=10)
    p.add_argument("--optimizer-mode", choices=["reference", "foreach"], default="reference")
    p.add_argument("--input-mode", choices=["stream", "gpu_cache"], default="stream")
    p.add_argument("--matmul-precision", choices=["highest", "high"], default="highest")
    p.add_argument("--amp", choices=["none", "bf16"], default="none")
    p.add_argument("--compile-model", action="store_true")
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Must run inside a Slurm allocation")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    results = []
    for tile in args.tiles:
        if "/" in tile or not (ROOT / "runs/ST19_whole_scs/tiles" / tile / "preprocess.done.json").exists():
            raise ValueError(f"Tile preprocessing not complete: {tile}")
        occupied = gpu_processes(args.gpu)
        if occupied:
            raise RuntimeError(f"GPU is not exclusive; refusing contaminated benchmark: {occupied}")
        output = root / tile
        command = [sys.executable, "-B", "-u", "-m", "optimizations.scs_original_torch.runner",
                   "--data-dir", str(ROOT / "runs/ST19_whole_scs/tiles" / tile / "data"),
                   "--output-dir", str(output), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--inference-batch-size", str(args.inference_batch_size), "--device", "cuda", "--amp", args.amp,
                   "--optimizer-mode", args.optimizer_mode, "--input-mode", args.input_mode,
                   "--matmul-precision", args.matmul_precision,
                   "--gpu-memory-mib", "70000", "--threads", "4"]
        if args.compile_model:
            command.append("--compile-model")
        write_json(root / "status.json", dict(state="running", tile=tile, command=command,
                   completed_tiles=len(results), total_tiles=len(args.tiles), job=os.environ["SLURM_JOB_ID"]))
        started = time.monotonic()
        peak_process_gpu = 0.0
        with (root / f"{tile}.log").open("x") as log, (root / f"{tile}.samples.csv").open("x", newline="") as sample:
            writer = csv.writer(sample)
            writer.writerow(["elapsed_seconds", "gpu_used_mib", "gpu_utilization_percent", "power_w", "process_gpu_mib"])
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    try:
                        gpu = subprocess.check_output(["nvidia-smi", "--id=" + args.gpu,
                            "--query-gpu=memory.used,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits"], text=True, timeout=10).strip().split(",")
                        rows = gpu_processes(args.gpu)
                        other = [r for r in rows if int(r.split(",")[0]) != process.pid]
                        if other:
                            raise RuntimeError(f"Another compute process appeared during benchmark: {other}")
                        memory = float(rows[0].split(",")[1]) if rows else 0.0
                        peak_process_gpu = max(peak_process_gpu, memory)
                        writer.writerow([round(time.monotonic()-started, 2), *gpu, memory])
                        sample.flush()
                    except subprocess.SubprocessError as error:
                        print(f"Sampling warning: {error}", flush=True)
                    time.sleep(2)
            except BaseException:
                process.terminate()
                process.wait(timeout=30)
                raise
            if process.returncode:
                write_json(root / "status.json", dict(state="failed", tile=tile, returncode=process.returncode))
                raise RuntimeError(f"Benchmark failed for {tile}; see its log")
        result = json.loads((output / "completed.json").read_text())
        with (output / "training_history.csv").open() as f:
            history = list(csv.DictReader(f))
        result.update(tile=tile, wall_seconds=time.monotonic()-started,
                      nvidia_smi_sampled_peak_process_gib=peak_process_gpu/1024,
                      epoch_seconds=[float(r["epoch_seconds"]) for r in history])
        results.append(result)
        write_json(root / "summary.json", dict(results=results,
            scope=f"{args.epochs} epochs and full inference per tile; sequential, exclusive GPU",
            benchmark_config=vars(args)))
        print(json.dumps(result), flush=True)
    write_json(root / "status.json", dict(state="completed", completed_tiles=len(results), results=results))


if __name__ == "__main__":
    main()
