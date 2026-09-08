#!/usr/bin/env python3
"""Check all prepared tile dimensions, seed counts and split margins on CPU."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scs_shared_cpu import summarize
from optimizations.scs_streaming.shared_data import load_schema, interior_mask, save_json


def validate(root):
    root = Path(root).resolve()
    schema = load_schema(root)
    assert not schema["partial_scope"] and len(schema["genes"]) == 6000
    assert not set(schema["splits"]["train"]) & set(schema["splits"]["validation"])
    assert set(schema["hvg_sample_counts"]) == set(schema["splits"]["train"])
    manifest = json.loads((root / "manifest.json").read_text())
    active, empty, no_supported, no_labels = 0, 0, [], []
    for tile in manifest["tiles"]:
        directory = root / "tiles" / tile["id"]
        meta = json.loads((directory / "prepared.json").read_text())
        assert meta["fingerprint"] == schema["fingerprint"]
        if meta["state"] == "empty":
            assert tile["records"] == 0
            empty += 1
            continue
        assert tile["records"] > 0
        active += 1
        neighbors = np.load(directory / "neighbors.npy", mmap_mode="r")
        binary = np.load(directory / "binary.npy", mmap_mode="r")
        eligible = np.load(directory / "eligible.npy", mmap_mode="r")
        directions = np.load(directory / "directions.npy", mmap_mode="r")
        n = meta["supported_centers"]
        assert neighbors.shape == (n, schema["n_neighbors"])
        assert binary.shape == eligible.shape == (n,) and directions.shape == (n, 2)
        assert np.isfinite(directions).all()
        assert np.isin(binary, [-1, 0, 1]).all()
        assert int(np.sum(eligible & (binary == 1))) == meta["foreground_samples"]
        assert int(np.sum(eligible & (binary == 0))) == meta["background_samples"]
        if n:
            assert neighbors.min() >= 0 and neighbors.max() < int(np.prod(meta["grid_shape"]))
            centers = neighbors[:, 0]
            assert np.all(np.diff(centers) > 0)
            xy = np.column_stack((centers // meta["grid_shape"][1], centers % meta["grid_shape"][1]))*schema["bin_size"]
            assert np.array_equal(eligible, interior_mask(xy, tile, schema["core_margin"]) & (binary >= 0))
        else:
            no_supported.append(tile["id"])
        if not np.any(eligible):
            no_labels.append(tile["id"])
        with np.load(directory / "expression.npz") as data:
            assert tuple(data["shape"]) == (int(np.prod(meta["grid_shape"])), 6000)
        expected = (tile["width"]+2*tile["halo"], tile["height"]+2*tile["halo"])
        with h5py.File(directory / "data/spots0:0:0:0.h5ad") as data:
            assert data["layers/watershed_labels"].shape == data["layers/stain"].shape == expected
        print(f"Validated {tile['id']}: {n} supported centers", flush=True)
    counts = summarize(root)
    assert counts["whole_slide_complete"] and not counts["failures"]
    result = dict(passed=True, fingerprint=schema["fingerprint"], active_tiles=active, empty_tiles=empty,
                  nonempty_tiles_without_supported_centers=no_supported, tiles_without_training_labels=no_labels,
                  biological_segmentation_quality_evaluated=False)
    save_json(root / "preparation_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    validate(parser.parse_args().root)
