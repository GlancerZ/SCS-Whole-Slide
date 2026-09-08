#!/usr/bin/env python3
"""Regression checks for coordinate ownership and the optimized SCS prior."""
import subprocess
import gzip
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from scs_whole_merge import Union, core, overlap_matches
from scs_whole_prepare import ROOT


def test_overlap():
    a = np.zeros((30, 30), dtype=np.uint32)
    b = np.zeros_like(a)
    a[2:20, 3:18] = 2; b[2:20, 3:18] = 8
    assert overlap_matches(a, b) == [(2, 8, 270, 1.0)]
    b[2:20, 3:18] = 0
    assert overlap_matches(a, b) == []
    u = Union()
    assert u.join(("a", 1), ("b", 2))
    assert u.join(("b", 2), ("c", 3))
    assert not u.join(("c", 3), ("a", 4))
    assert u.find(("a", 1)) == u.find(("c", 3))
    tile = dict(halo=2, width=5, height=3)
    labels = np.arange(9*7).reshape(9, 7)
    assert np.array_equal(core(tile, labels), labels[2:7, 2:5])


def test_prior():
    sys.path.insert(0, str(ROOT / "SCS"))
    from src import postprocessing as current
    original = subprocess.check_output(["git", "-C", str(ROOT/"SCS"), "show", "HEAD:src/postprocessing.py"], text=True)
    original = original.replace("import spateo as st", "")
    reference = types.ModuleType("scs_original_postprocessing")
    exec(compile(original, "upstream_postprocessing.py", "exec"), reference.__dict__)
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[4:10, 7:15] = 1; labels[30:40, 30:40] = 2; labels[40:49, 5:14] = 3
    adata = types.SimpleNamespace(layers={"watershed_labels": labels})
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory(prefix="scs-prior-") as temp:
        path = Path(temp) / "predictions.txt"
        lines = []
        for x, y in rng.integers(0, 20, size=(100, 2))*3:
            logits = ":".join(map(str, rng.normal(size=16)))
            lines.append(f"{x}\t{y}\t0.8\t{logits}\n")
        path.write_text("".join(lines))
        old = reference.read_gradient(str(path), adata, labels.shape, 3, 15)
        new = current.read_gradient(str(path), adata, labels.shape, 3, 15)
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)


def test_complete_stitch():
    import h5py
    import scs_whole_merge as stitch
    original_base = stitch.BASE
    with tempfile.TemporaryDirectory(prefix="scs-stitch-") as tmp:
        stitch.BASE = Path(tmp)
        tiles = []
        try:
            for ix in range(2):
                tile = dict(id=f"x{ix:02d}_y00", ix=ix, iy=0, x=1200*ix, y=0,
                            width=1200, height=1200, halo=60, origin_x=1200*ix-60, origin_y=-60)
                tiles.append(tile)
                directory = Path(tmp) / "tiles" / tile["id"]
                (directory / "results").mkdir(parents=True)
                (directory / "completed.json").write_text(json.dumps(dict(state="segmented")))
                mask = np.zeros((1320, 1320), dtype=np.uint32)
                mask[1196-tile["origin_x"]:1204-tile["origin_x"], 110:120] = ix+1
                np.savez_compressed(directory / "results/labels_0:0:0:0.npz", cells=mask)
            (Path(tmp) / "manifest.json").write_text(json.dumps(dict(tiles=tiles, shape=[2400, 1200], halo=60)))
            stitch.merge()
            with h5py.File(Path(tmp) / "merged/cell_labels.h5") as f:
                assert f.attrs["complete"] and f.attrs["cell_count"] == 1
                mask = f["cell_labels"][:]
                assert np.count_nonzero(mask) == 80
                assert np.all(mask[1196:1204, 50:60] == 1)
            with gzip.open(Path(tmp) / "merged/spot2cell.tsv.gz", "rt") as f:
                lines = f.read().splitlines()[1:]
            assert len(lines) == len(set(lines)) == 80
        finally:
            stitch.BASE = original_base


if __name__ == "__main__":
    test_overlap()
    test_prior()
    test_complete_stitch()
    print("PASS: overlap identity, duplicate-label protection, core coordinates, exact SCS prior equivalence, complete two-tile export")
