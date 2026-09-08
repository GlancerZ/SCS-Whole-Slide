"""Equivalence checks for batching, validation boundaries, predictions, and fit."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .data import ArrayStore, TensorTrainingData, batch_inputs, dataset, partition, predict_to_file, relative_positions
from .model_reference import create_transformer_classifier, dir_to_class, masked_categorical_cross_entropy

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scs-stream-test-")
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        rng = np.random.RandomState(13)
        self.values = {}
        for subset, n in [("train", 13), ("test", 5)]:
            expression = rng.randint(0, 220, (n, 50, 12)).astype(np.uint16)
            expression[:, 0, 0] = np.arange(n)
            position = rng.randint(-15, 16, (n, 50, 2)).astype(np.int32)
            position[:, 0] = 0
            position += np.arange(n)[:, None, None] * 100
            self.values["x_" + subset] = expression
            self.values["x_" + subset + "_pos"] = position
        for key, value in self.values.items():
            np.savez_compressed(self.data / f"{key}_0:0:0:0.npz", **{key: value})
        self.store = ArrayStore(self.data, self.root / "cache")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()
        tf.keras.backend.clear_session()

    def test_readonly_original_dtype_and_lazy_loading(self):
        source = self.store.path("x_train")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(self.store.header("x_test")[0], (5, 50, 12))
        self.assertEqual(self.store.opened, [])
        mapped = self.store.array("x_train")
        self.assertIsInstance(mapped, np.memmap)
        self.assertEqual(mapped.dtype, np.uint16)
        self.assertFalse(mapped.flags.writeable)
        np.testing.assert_array_equal(mapped, self.values["x_train"])
        self.assertNotIn("x_test", self.store.opened)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_batch_values_and_relative_coordinates_match_original(self):
        rows = np.array([12, 1, 8, 0])
        x, p = batch_inputs(self.store, "train", rows)
        expected = self.values["x_train_pos"].copy()
        for i in range(len(expected)):
            for j in range(1, expected.shape[1]):
                expected[i, j] -= expected[i, 0]
            expected[i, 0] = 0
        np.testing.assert_array_equal(x, self.values["x_train"][rows].astype(np.float32))
        np.testing.assert_array_equal(p, expected[rows])
        # Signed offsets must not wrap when the source array is unsigned.
        positions = np.array([[[100, 100], [97, 99]]], dtype=np.uint16)
        np.testing.assert_array_equal(relative_positions(positions), [[[0, 0], [-3, -1]]])

    def test_validation_selection_matches_original_strict_boundary(self):
        positions = self.values["x_train_pos"].copy()
        positions[9, 0] = [990, 1100]
        binary = np.ones(13, dtype=np.float32)
        binary[11] = 0
        train, val = partition(positions, binary, (1320, 1320), 0.0625)
        expected_train, expected_val = [], []
        for i in range(len(positions)):
            if positions[i, 0, 0] > 990 and positions[i, 0, 1] > 990:
                if binary[i] == 1:
                    expected_val.append(i)
            else:
                expected_train.append(i)
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(val, expected_val)
        self.assertIn(9, train)
        self.assertNotIn(11, train)
        self.assertNotIn(11, val)

    def test_dataset_visits_every_sample_once_and_keeps_test_unopened(self):
        labels = np.eye(16, dtype=np.float32)[np.arange(13)]
        binary = np.ones(13, dtype=np.float32)
        ds = dataset(self.store, np.arange(13), labels, binary, batch_size=5, shuffle=True, seed=123)
        epoch_orders = []
        for _ in range(2):
            seen = []
            for (x, p), (y, b) in ds.as_numpy_iterator():
                rows = x[:, 0, 0].astype(int)
                seen.extend(rows)
                np.testing.assert_array_equal(y, labels[rows])
                np.testing.assert_array_equal(b, binary[rows])
                np.testing.assert_array_equal(p, relative_positions(self.values["x_train_pos"][rows]))
            self.assertEqual(sorted(seen), list(range(13)))
            epoch_orders.append(seen)
        self.assertNotEqual(epoch_orders[0], epoch_orders[1])
        self.assertNotIn("x_test", self.store.opened)

    def test_prediction_file_matches_same_model_and_preserves_order(self):
        tf.keras.utils.set_random_seed(9)
        model = create_transformer_classifier(16, (50, 12), (50, 2), 50, 64, 1, [128, 64], 8, [1024, 256])
        expected = []
        coords = []
        for subset in ("train", "test"):
            x = self.values["x_" + subset].astype(np.float32)
            p = relative_positions(self.values["x_" + subset + "_pos"])
            # Match the original SCS model.predict path at the same batch size.
            # H100 TF32 kernels may differ when compared to a different batch shape.
            logits, probability = model.predict([x, p], batch_size=4, verbose=0)
            expected.extend(np.column_stack([probability, logits]))
            coords.extend(self.values["x_" + subset + "_pos"][:, 0])
        output = self.root / "predictions.txt"
        self.assertEqual(predict_to_file(model, self.store, output, batch_size=4), 18)
        actual, actual_coords = [], []
        for line in output.read_text().splitlines():
            x, y, probability, logits = line.split()
            actual_coords.append([int(x), int(y)])
            actual.append([float(probability)] + list(map(float, logits.split(":"))))
        np.testing.assert_array_equal(actual_coords, coords)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_tensor_dataset_shares_integer_features_and_preserves_batches(self):
        labels = np.eye(16, dtype=np.float32)[np.arange(13)]
        binary = np.ones(13, dtype=np.float32)
        data = TensorTrainingData(self.store, labels, binary)
        self.assertEqual(data.expression.dtype, tf.uint16)
        self.assertIn("CPU:0", data.expression.device)
        for indices in (np.array([12, 0, 5, 9, 1]), np.array([2, 3])):
            seen = []
            for (x, p), (y, b) in data.dataset(indices, batch_size=3).as_numpy_iterator():
                rows = x[:, 0, 0].astype(int)
                seen.extend(rows)
                np.testing.assert_array_equal(x, self.values["x_train"][rows].astype(np.float32))
                np.testing.assert_array_equal(p, relative_positions(self.values["x_train_pos"][rows]))
                np.testing.assert_array_equal(y, labels[rows])
            np.testing.assert_array_equal(seen, indices)
        self.assertNotIn("x_test", self.store.opened)

    def test_empty_test_array_has_no_extra_rows(self):
        key = "x_test"
        np.savez_compressed(self.data / f"{key}_0:0:0:0.npz", **{key: np.empty((0, 50, 12), dtype=np.uint16)})
        model = create_transformer_classifier(16, (50, 12), (50, 2), 50, 64, 1, [128, 64], 8, [1024, 256])
        self.assertEqual(predict_to_file(model, self.store, self.root / "empty-test.txt", batch_size=10), 13)
        self.assertNotIn("x_test", self.store.opened)

    def test_training_dataset_runs_original_model_losses_and_optimizer(self):
        tf.keras.utils.set_random_seed(9)
        model = create_transformer_classifier(16, (50, 12), (50, 2), 50, 64, 1, [128, 64], 8, [1024, 256])
        labels = dir_to_class(np.tile([[1, 1]], (13, 1)), 16).astype(np.float32)
        binary = np.ones(13, dtype=np.float32)
        model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
                      loss={"pos_out": masked_categorical_cross_entropy,
                            "cat_out": tf.keras.losses.BinaryCrossentropy()},
                      metrics={"pos_out": tf.keras.metrics.CategoricalAccuracy(name="accuracy")},
                      steps_per_execution=100)
        ds = dataset(self.store, np.arange(13), labels, binary, shuffle=True)
        history = model.fit(ds, epochs=1, verbose=0)
        self.assertTrue(np.isfinite(history.history["loss"][0]))
        self.assertEqual(int(model.optimizer.iterations), 2)
        self.assertNotIn("x_test", self.store.opened)


if __name__ == "__main__":
    unittest.main(verbosity=2)
