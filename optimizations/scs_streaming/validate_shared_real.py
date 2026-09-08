"""Read-only verification of the real-data shared-model smoke test."""
import argparse
import json
from pathlib import Path

import numpy as np

from .shared_data import TileStore, load_schema, save_json
from .shared_prepare import halo_records, aggregate


def validate(root):
    import tensorflow as tf
    root = Path(root).resolve()
    schema = load_schema(root)
    assert len(schema["genes"]) == len(schema["gene_indices"]) == 6000
    assert set(schema["hvg_sample_counts"]) == set(schema["splits"]["train"])
    assert not set(schema["splits"]["validation"]) & set(schema["hvg_sample_counts"])
    manifest = json.loads((root / "manifest.json").read_text())
    completed = json.loads((root / "model/completed.json").read_text())
    checkpoint = tf.train.load_checkpoint(completed["latest_checkpoint"])
    optimizer_keys = [k for k in checkpoint.get_variable_to_shape_map()
                      if k.startswith("optimizer/") and "/iter/" in k]
    assert len(optimizer_keys) == 1, optimizer_keys
    iterations = int(checkpoint.get_tensor(optimizer_keys[0]))
    sampling = json.loads((root / "model/sampling.json").read_text())
    assert iterations == sampling["train_steps_per_epoch"] * completed["epochs"]
    predictions = []
    for tid in schema["splits"]["train"]:
        tile = next(t for t in manifest["tiles"] if t["id"] == tid)
        store = TileStore(root, tid, schema)
        records, ng = halo_records(schema["source"], tile)
        shape = (tile["width"]+2*tile["halo"], tile["height"]+2*tile["halo"])
        source = aggregate(records, shape, (tile["origin_x"], tile["origin_y"]), schema["bin_size"], ng)
        expected = source[:, schema["gene_indices"]]
        assert (expected != store.expression).nnz == 0
        marker = json.loads((root / "tiles" / tid / "prediction.json").read_text())
        assert marker["checkpoint"] == completed["best_checkpoint"]
        file = root / "tiles" / tid / "results/spot_prediction_0:0:0:0.txt"
        row = 0
        with file.open() as handle:
            for line in handle:
                x, y, probability, logits = line.split()
                center = int(store.neighbors[row, 0])
                assert (int(x), int(y)) == (center // store.grid_shape[1]*schema["bin_size"],
                                           center % store.grid_shape[1]*schema["bin_size"])
                values = np.array([float(probability)] + list(map(float, logits.split(":"))))
                assert len(values) == 17 and np.all(np.isfinite(values))
                assert 0 <= values[0] <= 1
                row += 1
        assert row == marker["rows"] == len(store.neighbors)
        predictions.append(dict(tile=tid, rows=row, source_counts_exact=True,
                                model_fingerprint=marker["model_fingerprint"]))
    assert len({p["model_fingerprint"] for p in predictions}) == 1
    first = schema["splits"]["train"][0]
    first_tile = next(t for t in manifest["tiles"] if t["id"] == first)
    marker = json.loads((root / "tiles" / first / "completed.json").read_text())
    with np.load(root / "tiles" / first / "results/labels_0:0:0:0.npz") as arrays:
        labels = arrays["cells"]
        assert labels.dtype == np.uint32
        assert labels.shape == (first_tile["width"]+2*first_tile["halo"], first_tile["height"]+2*first_tile["halo"])
        assert len(np.unique(labels[labels > 0])) == marker["cell_count"]
    result = dict(passed=True, n_genes=6000, n_neighbors=schema["n_neighbors"],
                  training_regions=schema["splits"]["train"], validation_regions=schema["splits"]["validation"],
                  completed_epochs=completed["epochs"], optimizer_iterations=iterations,
                  predictions=predictions, postprocessed_tile=first,
                  engineering_test_only=True, segmentation_quality_evaluated=False)
    save_json(root / "engineering_validation.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    validate(parser.parse_args().root)
