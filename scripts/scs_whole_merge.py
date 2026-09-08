#!/usr/bin/env python3
"""Stitch core regions, reconcile halo labels, and export global SCS coordinates."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import h5py
import numpy as np

from scs_whole_prepare import BASE, save_json


class Union:
    def __init__(self):
        self.parent = {}
        self.tiles = {}

    def find(self, key):
        if key not in self.parent:
            self.parent[key] = key
            self.tiles[key] = {key[0]}
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            key, self.parent[key] = self.parent[key], root
        return root

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return True
        # Never collapse two distinct labels from the same original patch.
        if self.tiles[a] & self.tiles[b]:
            return False
        if b < a:
            a, b = b, a
        self.parent[b] = a
        self.tiles[a] |= self.tiles.pop(b)
        return True


def overlap_matches(a, b, minimum_pixels=30, minimum_overlap=0.5):
    """Mutual best overlap, normalized by the smaller in-overlap label area."""
    valid = (a > 0) & (b > 0)
    if not np.any(valid):
        return []
    pairs, counts = np.unique(np.stack([a[valid], b[valid]], axis=1), axis=0, return_counts=True)
    na = np.bincount(a.ravel()); nb = np.bincount(b.ravel())
    best_a, best_b = {}, {}
    for (x, y), n in zip(pairs, counts):
        if x not in best_a or n > best_a[x][1]:
            best_a[x] = (y, n)
        if y not in best_b or n > best_b[y][1]:
            best_b[y] = (x, n)
    result = []
    for (x, y), n in zip(pairs, counts):
        ratio = n / min(na[x], nb[y])
        if best_a[x][0] == y and best_b[y][0] == x and n >= minimum_pixels and ratio >= minimum_overlap:
            result.append((int(x), int(y), int(n), float(ratio)))
    return result


def load_labels(tile):
    with np.load(BASE / "tiles" / tile["id"] / "results/labels_0:0:0:0.npz") as f:
        return f["cells"]


def core(tile, labels):
    h = tile["halo"]
    return labels[h:h+tile["width"], h:h+tile["height"]]


def merge(partial=False):
    manifest = json.loads((BASE / "manifest.json").read_text())
    pending = []
    segmented = {}
    empty = []
    for tile in manifest["tiles"]:
        marker = BASE / "tiles" / tile["id"] / "completed.json"
        if not marker.exists():
            pending.append(tile["id"])
        elif json.loads(marker.read_text())["state"] == "segmented":
            segmented[tile["id"]] = tile
        else:
            empty.append(tile["id"])
    if pending and not partial:
        raise RuntimeError(f"Refusing a complete-slide export: {len(pending)} unresolved tiles")
    out = BASE / ("merged_partial" if pending else "merged")
    out.mkdir(exist_ok=True)
    union = Union()
    core_ids = {}
    for tile in segmented.values():
        labels = load_labels(tile)
        for ident in np.unique(labels):
            if ident:
                union.find((tile["id"], int(ident)))
        core_ids[tile["id"]] = np.unique(core(tile, labels))
    matches = []
    for tile in segmented.values():
        a = load_labels(tile)
        for dx, dy in [(0, 1), (1, -1), (1, 0), (1, 1)]:
            tid = f"x{tile['ix']+dx:02d}_y{tile['iy']+dy:02d}"
            if tid not in segmented:
                continue
            other = segmented[tid]
            b = load_labels(other)
            x0, y0 = max(tile["origin_x"], other["origin_x"]), max(tile["origin_y"], other["origin_y"])
            x1 = min(tile["origin_x"]+a.shape[0], other["origin_x"]+b.shape[0])
            y1 = min(tile["origin_y"]+a.shape[1], other["origin_y"]+b.shape[1])
            if x1 <= x0 or y1 <= y0:
                continue
            aa = a[x0-tile["origin_x"]:x1-tile["origin_x"], y0-tile["origin_y"]:y1-tile["origin_y"]]
            bb = b[x0-other["origin_x"]:x1-other["origin_x"], y0-other["origin_y"]:y1-other["origin_y"]]
            for ia, ib, pixels, ratio in overlap_matches(aa, bb):
                accepted = union.join((tile["id"], ia), (tid, ib))
                matches.append(dict(tile_a=tile["id"], label_a=ia, tile_b=tid, label_b=ib,
                                    overlap_pixels=pixels, overlap_fraction=ratio, accepted=accepted))
    active_roots = sorted({union.find((tid, int(ident))) for tid, ids in core_ids.items() for ident in ids if ident})
    root_ids = {key: i+1 for i, key in enumerate(active_roots)}
    n = len(root_ids)
    areas = np.zeros(n+1, dtype=np.uint64)
    sum_x = np.zeros(n+1, dtype=np.float64); sum_y = np.zeros(n+1, dtype=np.float64)
    preview = np.zeros(tuple((v+9)//10 for v in manifest["shape"]), dtype=np.uint32)
    with h5py.File(out / "cell_labels.building.h5", "w") as hf, gzip.open(out / "spot2cell.tsv.gz", "wt", compresslevel=3) as mapping:
        ds = hf.create_dataset("cell_labels", shape=tuple(manifest["shape"]), dtype="u4",
                               chunks=(600, 600), compression="gzip", compression_opts=1, fillvalue=0)
        hf.attrs["axis_order"] = "x,y"
        hf.attrs["coordinates"] = "whole-slide GEF bin1; zero-based"
        hf.attrs["complete"] = not pending
        hf.attrs["unresolved_tiles"] = json.dumps(pending)
        hf.attrs["cell_count"] = n
        mapping.write("x\ty\tcell_id\n")
        for tile in segmented.values():
            labels = core(tile, load_labels(tile))
            lookup = np.zeros(int(labels.max())+1, dtype=np.uint32)
            for ident in core_ids[tile["id"]]:
                if ident:
                    lookup[ident] = root_ids[union.find((tile["id"], int(ident)))]
            global_labels = lookup[labels]
            x0, y0 = tile["x"], tile["y"]
            ds[x0:x0+tile["width"], y0:y0+tile["height"]] = global_labels
            preview[x0//10:(x0+tile["width"]+9)//10, y0//10:(y0+tile["height"]+9)//10] = global_labels[::10, ::10]
            xx, yy = np.nonzero(global_labels)
            ids = global_labels[xx, yy]
            areas += np.bincount(ids, minlength=n+1).astype(np.uint64)
            sum_x += np.bincount(ids, weights=xx+x0, minlength=n+1)
            sum_y += np.bincount(ids, weights=yy+y0, minlength=n+1)
            for start in range(0, len(ids), 10000):
                mapping.writelines(f"{x+x0}\t{y+y0}\t{i}\n" for x, y, i in zip(xx[start:start+10000], yy[start:start+10000], ids[start:start+10000]))
            print("Merged", tile["id"], "assigned pixels", len(ids), flush=True)
    (out / "cell_labels.building.h5").replace(out / "cell_labels.h5")
    assert np.all(areas[1:] > 0)
    with (out / "cell_stats.csv").open("w") as f:
        writer = csv.writer(f); writer.writerow(["cell_id", "area_bin1", "centroid_x", "centroid_y"])
        for ident in range(1, n+1):
            writer.writerow([ident, int(areas[ident]), sum_x[ident]/areas[ident], sum_y[ident]/areas[ident]])
    save_json(out / "overlap_matches.json", matches)
    save_json(out / "summary.json", dict(complete=not pending, cell_count=n, assigned_bins=int(areas.sum()),
              segmented_tiles=len(segmented), no_expression_tiles=len(empty), unresolved_tiles=pending,
              total_tiles=len(manifest["tiles"]), shape=manifest["shape"], halo=manifest["halo"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colored = np.where(preview > 0, preview % 19 + 1, 0)
    plt.imsave(out / "cell_masks_overview.png", colored.T, cmap="tab20", vmin=0, vmax=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    merge(parser.parse_args().partial)
