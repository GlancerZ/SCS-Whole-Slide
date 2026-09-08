"""Point-count reporting must distinguish partial data and halo duplication."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scs_shared_cpu import summarize
from optimizations.scs_streaming.shared_data import fingerprint, save_json


class CountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scs-cpu-count-test-")
        self.root = Path(self.temp.name)
        schema = dict(partial_scope=True, genes=["g"], n_neighbors=50,
                      splits=dict(train=["a"], validation=["b"]))
        schema["fingerprint"] = fingerprint(schema)
        self.schema = schema
        save_json(self.root / "schema.json", schema)
        save_json(self.root / "manifest.json", dict(tiles=[dict(id=tid, halo=10, width=100, height=100)
                                                         for tid in ("a", "b")]))
        for tid in ("a", "b"):
            directory = self.root / "tiles" / tid
            directory.mkdir(parents=True)
            save_json(directory / "prepared.json", dict(fingerprint=schema["fingerprint"], state="prepared",
                      foreground_samples=20, background_samples=3, supported_centers=4,
                      insufficient_neighbor_bins=0, grid_shape=[120, 120]))
            centers = np.array([0, 10*120+10, 109*120+109, 110*120+110])
            np.save(directory / "neighbors.npy", np.repeat(centers[:, None], 50, axis=1))
        # In this artificial fixture each grid step is one original coordinate.
        schema = {k: v for k, v in self.schema.items() if k != "fingerprint"}
        schema["bin_size"] = 1
        schema["fingerprint"] = fingerprint(schema)
        self.schema = schema
        save_json(self.root / "schema.json", schema)
        import json
        for tid in ("a", "b"):
            path = self.root / "tiles" / tid / "prepared.json"
            meta = json.loads(path.read_text())
            meta["fingerprint"] = schema["fingerprint"]
            save_json(path, meta)

    def tearDown(self):
        self.temp.cleanup()

    def test_partial_scope_and_halo_count(self):
        counts = summarize(self.root, per_class_cap=5)
        self.assertTrue(counts["complete"])
        self.assertFalse(counts["whole_slide_complete"])
        self.assertEqual(counts["counts_scope"], "specified tiles only (partial slide)")
        self.assertEqual(counts["inference_with_halo"], 8)
        self.assertEqual(counts["inference_core_unique"], 4)
        self.assertEqual(counts["training"]["available"], 23)
        self.assertEqual(counts["training"]["planned_samples_per_epoch"], 8)
        self.assertEqual(counts["validation"]["available"], 23)

    def test_existing_training_and_inference_not_reported_as_unstarted(self):
        (self.root / "model").mkdir()
        save_json(self.root / "model/config.json", {})
        save_json(self.root / "tiles/a/prediction.json", {})
        counts = summarize(self.root)
        self.assertTrue(counts["training_started"])
        self.assertTrue(counts["inference_started"])


if __name__ == "__main__":
    unittest.main()
