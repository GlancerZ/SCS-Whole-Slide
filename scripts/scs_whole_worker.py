#!/usr/bin/env python3
"""Whole-slide queue with per-stage checkpoints and at most two concurrent tiles."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from scs_whole_prepare import BASE, ROOT, prepare_tile, save_json

def now():
    return datetime.now(timezone.utc).isoformat()


def training_reservation(directory):
    import numpy as np
    shapes = []
    for name in ("x_train", "x_test"):
        path = directory / f"data/{name}_0:0:0:0.npz"
        with zipfile.ZipFile(path) as z, z.open(name + ".npy") as f:
            version = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format._read_array_header(f, version)
        shapes.append(shape)
    train, test = shapes
    return max(12, (3*train[0]+2*test[0])*train[1]*train[2]*4 / 2**30 + 6)


class Queue:
    def __init__(self, hours, workers, only=None):
        self.manifest = json.loads((BASE / "manifest.json").read_text())
        self.tiles = sorted(self.manifest["tiles"], key=lambda t: (-t["umis"], t["id"]))
        if only:
            self.tiles = [t for t in self.tiles if t["id"] in only]
        self.lock = threading.Condition()
        self.reserved = 0.0
        self.active = {}
        self.deadline = time.time() + hours*3600
        self.workers = workers

    def status(self, tile, state, **extra):
        directory = BASE / "tiles" / tile["id"]
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / "status.json", dict(tile=tile["id"], state=state,
                  at=now(), job_id=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(), **extra))
        print(f"{now()} {tile['id']} {state} {extra}", flush=True)

    def process(self, tile):
        directory = BASE / "tiles" / tile["id"]
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "completed.json").exists():
            return
        if not tile["records"]:
            save_json(directory / "completed.json", dict(tile=tile["id"], state="no_expression",
                       cell_count=0, at=now(), records=0))
            self.status(tile, "no_expression")
            return
        if time.time() > self.deadline - 180:
            return
        with (directory / ".lock").open("w") as lockfile:
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            try:
                self.status(tile, "preparing")
                prepare_tile(tile)
                for stage in ("preprocess", "train", "postprocess"):
                    done = directory / (stage + ".done.json")
                    if done.exists():
                        continue
                    if time.time() > self.deadline - (900 if stage == "train" else 180):
                        self.status(tile, "paused_for_walltime", next_stage=stage)
                        return
                    memory = training_reservation(directory) if stage == "train" else 24.0
                    if memory > 104:
                        raise RuntimeError(f"Estimated host memory {memory:.1f} GiB exceeds safe single-stage budget")
                    with self.lock:
                        announced_wait = False
                        while self.reserved + memory > 104:
                            if not announced_wait:
                                self.status(tile, "waiting_for_memory", stage=stage, estimated_gib=memory)
                                announced_wait = True
                            self.lock.wait(timeout=30)
                            if time.time() > self.deadline - 900:
                                self.status(tile, "paused_for_walltime", next_stage=stage)
                                return
                        self.reserved += memory
                        self.active[tile["id"]] = stage
                    start = time.time()
                    self.status(tile, "running", stage=stage, estimated_gib=memory)
                    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                               MKL_NUM_THREADS="4", NUMBA_NUM_THREADS="4", SCS_GPU_MEMORY_MB="28000",
                               SCS_STEPS_PER_EXECUTION="100", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
                    if stage != "train":
                        env["CUDA_VISIBLE_DEVICES"] = ""
                    try:
                        with (directory / (stage + ".log")).open("a") as log:
                            log.write(f"\nSTART {now()} job={env.get('SLURM_JOB_ID')}\n"); log.flush()
                            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/scs_whole_stage.py"),
                                            stage, "--tile", tile["id"]], check=True, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                    finally:
                        with self.lock:
                            self.reserved -= memory
                            self.active.pop(tile["id"], None)
                            self.lock.notify_all()
                    save_json(done, dict(stage=stage, at=now(), elapsed_seconds=time.time()-start))
                import numpy as np
                with np.load(directory / "results/labels_0:0:0:0.npz") as f:
                    labels = f["cells"]
                    assert labels.shape == (tile["width"]+2*tile["halo"], tile["height"]+2*tile["halo"])
                    n = len(np.unique(labels[labels>0]))
                save_json(directory / "completed.json", dict(tile=tile["id"], state="segmented",
                          cell_count=n, at=now(), records=tile["records"]))
                self.status(tile, "segmented", cells=n)
            except Exception as error:
                self.status(tile, "failed", error=str(error))
                traceback.print_exc()

    def run(self):
        save_json(BASE / "launch.json", dict(at=now(), job_id=os.environ.get("SLURM_JOB_ID"),
                  host=socket.gethostname(), workers=self.workers, gpu_limit_mib_per_worker=28000,
                  deadline_utc=datetime.fromtimestamp(self.deadline, timezone.utc).isoformat()))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self.process, self.tiles))
        complete = all((BASE / "tiles" / t["id"] / "completed.json").exists() for t in self.manifest["tiles"])
        save_json(BASE / "queue.exit.json", dict(at=now(), all_tiles_complete=complete))
        if complete:
            subprocess.run([sys.executable, str(ROOT / "scripts/scs_whole_merge.py")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=5.75)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    with (BASE / ".queue.lock").open("w") as master:
        fcntl.flock(master, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Queue(args.hours, args.workers, args.only).run()
