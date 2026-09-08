#!/usr/bin/env python3
"""Resumeable CPU Cellist tiles with bounded memory and explicit completion."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from cellist_cpu_stage import ROOT, save_json

SOURCE = ROOT / "runs/ST19_whole_scs"
BASE = ROOT / "runs/ST19_cellist/whole_shared_nuclei_tissue"
SHARED_NUCLEI = ROOT / "runs/ST19_cellist/shared_nuclei_tissue"


def initialize():
    BASE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert (SHARED_NUCLEI / "completed.json").exists()
    manifest.update(method="Cellist 1.1.1", resolution_um=0.5,
                    initial_segmentation="Watershed on registered hematoxylin",
                    method_scope="Cellist on overlapping expression tiles with one whole-slide watershed nucleus atlas",
                    shared_nuclei=str(SHARED_NUCLEI),
                    core_ownership="Each observed spot is exported only by its unique non-overlapping core")
    save_json(BASE / "manifest.json", manifest)
    index = BASE / "spatial_index.h5"
    if not index.exists():
        index.symlink_to(SOURCE / "spatial_index.h5")
    return manifest


def prepare(tile):
    target = BASE / "tiles" / tile["id"]
    target.mkdir(parents=True, exist_ok=True)
    source = SOURCE / "tiles" / tile["id"]
    names = ["expression.tsv", "prepared.json"]
    if all((source / name).exists() for name in names):
        metadata = json.loads((source / "prepared.json").read_text())
        for key in ["origin_x", "origin_y", "width", "height", "halo"]:
            assert metadata[key] == tile[key], (tile["id"], key)
        for name in names:
            if not (target / name).exists():
                (target / name).symlink_to(source / name)
    else:
        import scs_whole_prepare as preparation
        preparation.BASE = BASE
        preparation.prepare_tile(tile)
    if not (target / "shared_image.done.json").exists():
        import numpy as np
        from tifffile import memmap, imwrite
        image = memmap(SHARED_NUCLEI / "hematoxylin.tif", mode="r")
        ox, oy = tile["origin_x"], tile["origin_y"]
        w, h = tile["width"]+2*tile["halo"], tile["height"]+2*tile["halo"]
        region = np.zeros((w, h), dtype=np.uint8)
        x0, y0 = max(ox, 0), max(oy, 0)
        x1, y1 = min(ox+w, image.shape[0]), min(oy+h, image.shape[1])
        region[x0-ox:x1-ox, y0-oy:y1-oy] = image[x0:x1, y0:y1]
        imwrite(target / "hematoxylin.building.tif", region, photometric="minisblack")
        (target / "hematoxylin.building.tif").replace(target / "hematoxylin.tif")
        save_json(target / "shared_image.done.json", dict(source=str(SHARED_NUCLEI / "hematoxylin.tif")))
    return target


def run_tile(tile):
    directory = BASE / "tiles" / tile["id"]
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "completed.json"
    if marker.exists():
        return tile["id"], json.loads(marker.read_text())["state"]
    start = time.monotonic()
    try:
        if tile["records"] == 0:
            save_json(marker, dict(state="no_expression", source_records=0, source_umis=0))
            return tile["id"], "no_expression"
        save_json(directory / "status.json", dict(state="preparing", tile=tile["id"]))
        prepare(tile)
        command = [sys.executable, "-u", str(ROOT / "scripts/cellist_cpu_stage.py")]
        parameters = ["--gem", str(directory / "expression.tsv"), "--image", str(directory / "hematoxylin.tif"),
                      "--output", str(directory), "--workers", "1", "--shared-nuclei", str(SHARED_NUCLEI)]
        with (directory / "run.log").open("a") as log:
            for stage in ["watershed", "seg", "qa"]:
                if marker.exists():
                    break
                if stage != "qa" and (directory / f"{stage}.done.json").exists():
                    continue
                save_json(directory / "status.json", dict(state=stage, tile=tile["id"],
                          elapsed_seconds=time.monotonic()-start))
                subprocess.run(command + [stage] + parameters, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, NUMBA_NUM_THREADS="1", BLIS_NUM_THREADS="1"))
        save_json(directory / "status.json", dict(state="completed", tile=tile["id"],
                  elapsed_seconds=time.monotonic()-start))
        return tile["id"], json.loads(marker.read_text())["state"]
    except Exception:
        save_json(directory / "status.json", dict(state="failed", tile=tile["id"],
                  elapsed_seconds=time.monotonic()-start, error=traceback.format_exc()))
        return tile["id"], "failed"


def queue(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inside the CPU allocation")
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = initialize()
        tiles = manifest["tiles"]
        if args.tiles:
            wanted = set(args.tiles.split(","))
            tiles = [tile for tile in tiles if tile["id"] in wanted]
            assert {tile["id"] for tile in tiles} == wanted
        # Exercise sparse-edge cases early as well as the memory-heavy cores.
        ordered = sorted((tile for tile in tiles if tile["records"]), key=lambda tile: -tile["records"])
        empty = [tile for tile in tiles if not tile["records"]]
        interleaved = []
        while ordered:
            interleaved.append(ordered.pop(0))
            if ordered:
                interleaved.append(ordered.pop())
        tiles = empty + interleaved
        save_json(BASE / "launch.json", dict(job_id=os.environ["SLURM_JOB_ID"], host=os.uname().nodename,
                  workers=args.workers, threads_per_cellist_process=os.environ.get("CELLIST_THREADS"),
                  selected_tiles=[tile["id"] for tile in tiles], started_at=time.time()))
        completed = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_tile, tile) for tile in tiles]
            for future in as_completed(futures):
                tile_id, state = future.result()
                completed[tile_id] = state
                print(f"{len(completed)}/{len(tiles)} {tile_id} {state}", flush=True)
                save_json(BASE / "queue_status.json", dict(state="running", results=completed))
        failures = [tile_id for tile_id, state in completed.items() if state == "failed"]
        save_json(BASE / "queue_status.json", dict(state="failed" if failures else "tiles_completed", results=completed))
        if failures:
            raise RuntimeError(f"{len(failures)} failed tiles; see per-tile status.json")
        if not args.tiles:
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/cellist_merge.py")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tiles")
    queue(parser.parse_args())
