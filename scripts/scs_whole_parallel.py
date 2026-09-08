#!/usr/bin/env python3
"""Multiple allocations share tile locks; each stage acquires a resource lease."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from scs_parallel_resources import BASE, ROOT, atomic_json, configure, lease, runtime


def now():
    return datetime.now(timezone.utc).isoformat()


def status(directory, state, **extra):
    data = dict(tile=directory.name, state=state, at=now(), job_id=os.environ["SLURM_JOB_ID"],
                host=socket.gethostname(), implementation="memory_optimized_original_scs", **extra)
    atomic_json(directory / "status.json", data)
    print(json.dumps(data), flush=True)


def process(tile, end_time):
    directory = BASE / "tiles" / tile["id"]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "completed.json").exists() or time.time() > end_time - 1200:
        return
    with (directory / ".lock").open("a") as tile_lock:
        try:
            fcntl.flock(tile_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (directory / "completed.json").exists():
            return
        if not tile["records"]:
            atomic_json(directory / "completed.json", dict(tile=tile["id"], state="no_expression", cell_count=0, records=0, at=now()))
            return
        try:
            env = dict(os.environ, OMP_NUM_THREADS="3", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       NUMBA_NUM_THREADS="3", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
            for stage in ("prepare", "preprocess", "train", "postprocess"):
                done = directory / ("prepared.json" if stage == "prepare" else stage + ".done.json")
                if done.exists():
                    continue
                if time.time() > end_time - (1200 if stage == "train" else 300):
                    status(directory, "paused_for_walltime", next_stage=stage)
                    return
                status(directory, "running", stage=stage)
                start = time.time()
                command = [sys.executable, "-u", str(ROOT / "scripts/scs_whole_stage.py"), stage, "--tile", tile["id"]]
                with (directory / (stage + ".log")).open("a") as log:
                    log.write(f"\nSTART {now()} job={env['SLURM_JOB_ID']} optimized_original_scs\n"); log.flush()
                    subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
                if stage != "prepare":
                    atomic_json(done, dict(stage=stage, at=now(), elapsed_seconds=time.time()-start,
                                          job_id=env["SLURM_JOB_ID"], implementation="memory_optimized_original_scs"))
            import numpy as np
            with np.load(directory / "results/labels_0:0:0:0.npz") as f:
                labels = f["cells"]
                assert labels.shape == (tile["width"]+2*tile["halo"], tile["height"]+2*tile["halo"])
                count = len(np.unique(labels[labels > 0]))
            atomic_json(directory / "completed.json", dict(tile=tile["id"], state="segmented", cell_count=count,
                        records=tile["records"], at=now(), job_id=env["SLURM_JOB_ID"]))
            status(directory, "segmented", cells=count)
        except Exception as error:
            status(directory, "failed", error=str(error))
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = configure()
    with (runtime() / ".controller.lock").open("a") as master:
        fcntl.flock(master, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((BASE / "manifest.json").read_text())
        tiles = sorted(manifest["tiles"], key=lambda t: (-t["umis"], t["id"]))
        workers = args.workers or len(config["gpus"]) * 4
        atomic_json(runtime() / "launch.json", dict(config, workers=workers, pid=os.getpid(), at=now()))
        # One attempt per tile per controller; failures remain visibly unresolved.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda t: process(t, config["end_time"]), tiles))
        complete = all((BASE / "tiles" / t["id"] / "completed.json").exists() for t in tiles)
        atomic_json(runtime() / "exit.json", dict(at=now(), all_tiles_complete=complete))
        if complete:
            with (BASE / ".merge.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not (BASE / "merged/summary.json").exists():
                    with lease("merge", BASE):
                        subprocess.run([sys.executable, str(ROOT / "scripts/scs_whole_merge.py")], check=True)


if __name__ == "__main__":
    main()
