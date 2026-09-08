"""Measure expression sparsity before and after SCS neighborhood aggregation."""

import argparse
import json
from pathlib import Path

import numpy as np

from .shared_data import SharedBatches, TileCache, load_schema


def summarize(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "min": int(values.min()),
        "max": int(values.max()),
    }


def measure(root, samples_per_tile=20, seed=20260905):
    root = Path(root).resolve()
    schema = load_schema(root)
    batches = SharedBatches(
        root,
        "train",
        batch_size=1,
        per_class_cap=0,
        seed=seed,
        split_mode="random",
        validation_fraction=0.1,
    )
    rng = np.random.RandomState(seed)
    cache = TileCache(root, schema, capacity=1)
    center_genes = []
    neighborhood_genes = []
    neighborhood_counts = []
    for tile_id, pools in batches.pools.items():
        rows = np.concatenate(pools)
        selected = rng.choice(rows, min(samples_per_tile, len(rows)), replace=False)
        tile = cache.get(tile_id)
        neighbors = np.asarray(tile.neighbors[selected])
        for row_ids in neighbors:
            center = tile.expression[row_ids[0]]
            neighborhood = tile.expression[row_ids]
            center_genes.append(center.nnz)
            neighborhood_genes.append(len(np.unique(neighborhood.indices)))
            neighborhood_counts.append(int(neighborhood.data.sum()))
    return {
        "sampled_centers": len(center_genes),
        "center_nonzero_genes": summarize(center_genes),
        "fifty_spot_unique_genes": summarize(neighborhood_genes),
        "fifty_spot_total_counts": summarize(neighborhood_counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--samples-per-tile", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(measure(args.root, args.samples_per_tile), indent=2))


if __name__ == "__main__":
    main()
