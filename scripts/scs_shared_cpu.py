#!/usr/bin/env python3
"""CPU-only whole-slide preparation, bounded workers, restart markers and counts."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optimizations.scs_streaming.shared_data import load_schema, save_json, interior_mask


def remaining_seconds():
    value = subprocess.check_output(["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%L"], text=True).strip()
    days = 0
    if "-" in value:
        d, value = value.split("-")
        days = int(d)
    parts = list(map(int, value.split(":")))
    if len(parts) == 2:
        parts.insert(0, 0)
    return days*86400 + parts[0]*3600 + parts[1]*60 + parts[2]


def summarize(root, per_class_cap=4096):
    import numpy as np
    schema = load_schema(root)
    manifest = json.loads((root / "manifest.json").read_text())
    result = dict(fingerprint=schema["fingerprint"], tiles=len(manifest["tiles"]),
                  prepared_tiles=0, empty_tiles=0, pending_tiles=[], failures=[],
                  n_genes=len(schema["genes"]), n_neighbors=schema["n_neighbors"],
                  training=dict(regions=len(schema["splits"]["train"]), foreground=0, background=0,
                                available=0, planned_samples_per_epoch=0),
                  validation=dict(regions=len(schema["splits"]["validation"]), foreground=0, background=0,
                                  available=0, planned_samples_per_epoch=0),
                  inference_with_halo=0, inference_core_unique=0,
                  insufficient_neighbor_bins_with_halo=0, per_region_per_class_cap=per_class_cap,
                  partial_scope=schema["partial_scope"],
                  training_started=(root / "model/config.json").exists(), inference_started=False)
    for tile in manifest["tiles"]:
        tid = tile["id"]
        directory = root / "tiles" / tid
        result["inference_started"] |= (directory / "prediction.json").exists()
        marker = directory / "prepared.json"
        if not marker.exists():
            result["pending_tiles"].append(tid)
            status = directory / "cpu_status.json"
            if status.exists() and json.loads(status.read_text())["state"] == "failed":
                result["failures"].append(dict(tile=tid, **json.loads(status.read_text())))
            continue
        meta = json.loads(marker.read_text())
        if meta["fingerprint"] != schema["fingerprint"]:
            raise ValueError(f"Wrong feature schema for {tid}")
        result["prepared_tiles"] += 1
        if meta["state"] == "empty":
            result["empty_tiles"] += 1
            continue
        split = "training" if tid in schema["splits"]["train"] else "validation"
        positive, negative = meta["foreground_samples"], meta["background_samples"]
        result[split]["foreground"] += positive
        result[split]["background"] += negative
        result[split]["available"] += positive + negative
        npos = min(positive, per_class_cap)
        result[split]["planned_samples_per_epoch"] += npos + min(negative, per_class_cap, npos if npos else per_class_cap)
        result["inference_with_halo"] += meta["supported_centers"]
        result["insufficient_neighbor_bins_with_halo"] += meta["insufficient_neighbor_bins"]
        neighbors = np.load(directory / "neighbors.npy", mmap_mode="r")
        centers = neighbors[:, 0]
        ny = meta["grid_shape"][1]
        xy = np.column_stack((centers // ny, centers % ny))*schema["bin_size"]
        result["inference_core_unique"] += int(interior_mask(xy, tile, 0).sum())
    result["complete"] = not result["pending_tiles"]
    result["whole_slide_complete"] = result["complete"] and not schema["partial_scope"]
    result["counts_scope"] = ("whole slide" if result["whole_slide_complete"] else
                             "specified tiles only (partial slide)" if result["complete"] else
                             "prepared tiles only; incomplete")
    save_json(root / "point_counts.json", result)
    return result


def run(source, root, workers=4):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Execute in the CPU Slurm allocation, never on the login node")
    job = subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]], text=True)
    if "gres/gpu" in job:
        raise RuntimeError("User forbids this preparation from using a GPU allocation")
    if workers < 1 or workers > 4:
        raise ValueError("Use 1..4 workers within the 16-CPU/64-GiB allocation")
    deadline = time.time() + remaining_seconds() - 180
    source, root = Path(source).resolve(), Path(root).resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4", NUMBA_NUM_THREADS="4", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ("." + root.name + ".cpu.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch = dict(job_id=os.environ["SLURM_JOB_ID"], host=socket.gethostname(), workers=workers,
                      cpu_only=True, source=str(source), output=str(root), state="initializing",
                      deadline_unix=deadline)
        launch_path = root.parent / (root.name + ".cpu_status.json")
        save_json(launch_path, launch)
        if not (root / "schema.json").exists():
            subprocess.run([sys.executable, "-B", "-m", "optimizations.scs_streaming.shared_prepare", "init",
                            "--source", str(source), "--output", str(root), "--n-genes", "6000"],
                           cwd=ROOT, env=env, check=True, timeout=max(1, deadline-time.time()))
        schema = load_schema(root)
        if schema["partial_scope"] or len(schema["genes"]) != 6000 or Path(schema["source"]) != source:
            raise ValueError("Existing preparation does not match the requested full-slide 6000-gene run")
        save_json(launch_path, dict(launch, state="preparing", fingerprint=schema["fingerprint"]))

        def prepare(tid):
            directory = root / "tiles" / tid
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "prepared.json").exists():
                return tid, "already_prepared"
            if deadline-time.time() < 120:
                return tid, "deferred_walltime"
            save_json(directory / "cpu_status.json", dict(state="running", job_id=launch["job_id"]))
            try:
                with (directory / "cpu_prepare.log").open("a") as log:
                    subprocess.run([sys.executable, "-B", "-m", "optimizations.scs_streaming.shared_prepare", "prepare",
                                    "--output", str(root), "--tiles", tid], cwd=ROOT, env=env, check=True,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=max(1, deadline-time.time()))
                save_json(directory / "cpu_status.json", dict(state="prepared", job_id=launch["job_id"]))
                return tid, "prepared"
            except Exception as error:
                save_json(directory / "cpu_status.json", dict(state="failed", job_id=launch["job_id"], error=str(error)))
                return tid, "failed"

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(prepare, tid) for tid in schema["tile_ids"]]
            for future in as_completed(futures):
                tid, state = future.result()
                done += 1
                print(f"{done}/{len(futures)} {tid}: {state}", flush=True)
                save_json(launch_path, dict(launch, state="preparing", processed_tiles=done, last_tile=tid,
                                          last_state=state, fingerprint=schema["fingerprint"]))
        counts = summarize(root)
        save_json(launch_path, dict(launch, state="complete" if counts["complete"] else "incomplete",
                                  prepared_tiles=counts["prepared_tiles"], failed_tiles=len(counts["failures"])))
        print(json.dumps(counts, indent=2), flush=True)
        return 0 if counts["complete"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=str(ROOT / "runs/ST19_whole_scs"))
    p.add_argument("--root", default=str(ROOT / "runs/ST19_shared_6000"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    if args.summary_only:
        print(json.dumps(summarize(Path(args.root).resolve()), indent=2))
    else:
        sys.exit(run(args.source, args.root, args.workers))
