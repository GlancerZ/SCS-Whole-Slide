"""Shared SCS schema, sparse features, spatial isolation, and sampling checks."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from .shared_data import (fingerprint, save_json, load_schema, neighbor_table, ring_offsets,
                          interior_mask, direction_classes, TileStore, SharedBatches)
from .shared_prepare import aggregate
from .shared_train import ready_model, merge


class SharedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scs-shared-test-")
        self.root = Path(self.temp.name)
        self.schema = dict(bin_size=3, n_neighbors=50, gene_indices=[2, 0, 3], seed=11,
                           splits=dict(train=["a", "b"], validation=["v"]), tile_ids=["a", "b", "v"])
        self.schema["fingerprint"] = fingerprint(self.schema)
        save_json(self.root / "schema.json", self.schema)
        rng = np.random.RandomState(4)
        self.expression = sparse.csr_matrix(rng.randint(0, 700, (30*30, 3)).astype(np.uint16))
        self.neighbors, _ = neighbor_table(self.expression, (30, 30))
        for tid in ("a", "b", "v"):
            directory = self.root / "tiles" / tid
            directory.mkdir(parents=True)
            sparse.save_npz(directory / "expression.npz", self.expression)
            binary = np.arange(len(self.neighbors)) % 3 - 1
            for key, data in dict(neighbors=self.neighbors, binary=binary.astype(np.int8),
                                  directions=np.ones((len(binary), 2), dtype=np.float32),
                                  eligible=np.ones(len(binary), dtype=bool)).items():
                np.save(directory / (key + ".npy"), data)
            save_json(directory / "prepared.json", dict(fingerprint=self.schema["fingerprint"],
                                                       state="prepared", grid_shape=[30, 30]))

    def tearDown(self):
        self.temp.cleanup()

    def test_sparse_batch_is_exactly_expanded_input(self):
        tile = TileStore(self.root, "a", self.schema)
        rows = np.array([120, 0, 800, 319])
        x, p, centers = tile.inputs(rows)
        expected = self.expression.toarray()[self.neighbors[rows]].astype(np.float32)
        np.testing.assert_array_equal(x, expected)
        np.testing.assert_array_equal(p[:, 0], 0)
        xy = np.stack((self.neighbors[rows] // 30, self.neighbors[rows] % 30), axis=-1)*3
        np.testing.assert_array_equal(p, xy - xy[:, :1])
        np.testing.assert_array_equal(centers, xy[:, 0])
        self.assertEqual(x.dtype, np.float32)
        self.assertGreater(x.max(), 255)  # No uint8 overflow/truncation.

    def test_neighbors_equal_upstream_ring_order(self):
        occupied = np.ones((25, 28), dtype=np.int32)
        occupied[::3, ::4] = 0
        expression = sparse.csr_matrix(occupied.reshape(-1, 1))
        actual, total = neighbor_table(expression, occupied.shape, 50)
        expected = []
        for i, j in zip(*np.where(occupied)):
            rows = []
            for dx, dy in ring_offsets():
                x, y = i+dx, j+dy
                if 0 <= x < 25 and 0 <= y < 28 and occupied[x, y]:
                    rows.append(x*28+y)
                if len(rows) == 50:
                    break
            if len(rows) == 50:
                expected.append(rows)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(total, np.count_nonzero(occupied))
        self.assertEqual(len(np.unique(actual[:, 0])), len(actual))

    def test_sparse_and_empty_regions_have_explicit_coverage(self):
        neighbors, total = neighbor_table(sparse.csr_matrix(np.eye(8)), (2, 4), 50)
        self.assertEqual(neighbors.shape, (0, 50))
        self.assertEqual(total, 8)
        neighbors, total = neighbor_table(sparse.csr_matrix((12, 3)), (3, 4), 50)
        self.assertEqual(neighbors.shape, (0, 50))
        self.assertEqual(total, 0)
        with self.assertRaises(ValueError):
            neighbor_table(self.expression, (30, 30), 442)

    def test_core_erosion_is_strict_and_contexts_do_not_overlap(self):
        tile = dict(halo=60, width=1200, height=1200)
        points = np.array([[149, 200], [150, 150], [1169, 1169], [1170, 200], [59, 100]])
        np.testing.assert_array_equal(interior_mask(points, tile, 90), [False, True, True, False, False])
        # Adjacent cores with radius30 contexts retain a gap after erosion.
        training_max_global = 1200-90-1+30
        validation_min_global = 1200+90-30
        self.assertLess(training_max_global, validation_min_global)

    def test_region_balance_split_and_epoch_reproducibility(self):
        training = SharedBatches(self.root, "train", batch_size=7, per_class_cap=19, seed=11)
        validation = SharedBatches(self.root, "validation", batch_size=7, per_class_cap=19, seed=11)
        self.assertEqual(training.samples, 76)
        self.assertEqual(training.steps, 12)
        self.assertEqual(set(training.pools), {"a", "b"})
        self.assertEqual(set(validation.pools), {"v"})
        def plan(batches, epoch):
            return [(tid, rows.tolist()) for tid, rows in batches.plan(epoch)]
        self.assertEqual(plan(training, 3), plan(training, 3))
        self.assertNotEqual(plan(training, 3), plan(training, 4))
        self.assertEqual(plan(validation, 3), plan(validation, 4))
        for tid in training.pools:
            rows = np.concatenate([r for t, r in training.plan(0) if t == tid])
            self.assertEqual(len(np.unique(rows)), len(rows))
            self.assertFalse(np.any(TileStore(self.root, tid, self.schema).binary[rows] == -1))

    def test_fingerprint_prevents_gene_order_mixing(self):
        altered = dict(self.schema, gene_indices=[0, 2, 3])
        save_json(self.root / "schema.json", altered)
        with self.assertRaises(ValueError):
            load_schema(self.root)
        altered["fingerprint"] = "different"
        with self.assertRaises(ValueError):
            TileStore(self.root, "a", altered)

    def test_bin_aggregation_preserves_counts_and_gene_order(self):
        records = np.array([(60, 60, 2, 200), (61, 61, 2, 200), (63, 60, 0, 5)],
                           dtype=[("x", "u2"), ("y", "u2"), ("gene", "u2"), ("count", "u1")])
        matrix = aggregate(records, (6, 6), (60, 60), 3, 4)[:, [2, 0, 3]].toarray()
        self.assertEqual(matrix.sum(), 405)
        np.testing.assert_array_equal(matrix[0], [400, 0, 0])
        np.testing.assert_array_equal(matrix[2], [0, 5, 0])

    def test_direction_classes_and_background_mask(self):
        directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0], [1, 1]])
        labels = direction_classes(directions, [1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(labels[:5].argmax(axis=1), [0, 4, 8, 12, 0])
        self.assertEqual(labels[-1].sum(), 0)

    def test_stale_training_completion_is_rejected(self):
        directory = self.root / "model"
        directory.mkdir()
        completed = dict(epochs=1, fingerprint=self.schema["fingerprint"])
        state = dict(completed=True, target_epochs=1, fingerprint=self.schema["fingerprint"])
        save_json(directory / "completed.json", completed)
        save_json(directory / "training_state.json", state)
        self.assertEqual(ready_model(self.root, self.schema), completed)
        save_json(directory / "training_state.json", dict(state, completed=False, target_epochs=2))
        with self.assertRaises(ValueError):
            ready_model(self.root, self.schema)

    def test_partial_test_cannot_be_merged_as_full_slide(self):
        schema = {k: v for k, v in self.schema.items() if k != "fingerprint"}
        schema["partial_scope"] = True
        schema["fingerprint"] = fingerprint(schema)
        save_json(self.root / "schema.json", schema)
        with self.assertRaisesRegex(ValueError, "Smoke-test scope"):
            merge(self.root)


if __name__ == "__main__":
    unittest.main()
