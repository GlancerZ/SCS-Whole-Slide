"""Small CPU tests: SDPA, loss semantics, labels, and full train/predict path."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .model import Attention, OriginalSCS, ReferenceAdamW, losses
from .runner import configure_device, direction_labels, train
from .predict import predict


class OriginalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_topology_and_sdpa_dropout(self):
        model = OriginalSCS(23, 50)
        x, p = torch.randn(2, 50, 23), torch.zeros(2, 50, 2)
        sdpa = F.scaled_dot_product_attention
        with patch.object(F, "scaled_dot_product_attention", wraps=sdpa) as call:
            model.eval()
            first = model(x, p)
            second = model(x, p)
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertEqual(call.call_count, 16)
            self.assertTrue(all(c.kwargs["dropout_p"] == 0 for c in call.call_args_list))
        with patch.object(F, "scaled_dot_product_attention", wraps=sdpa) as call:
            model.train()
            model(x, p)
            self.assertEqual(call.call_count, 8)
            self.assertTrue(all(c.kwargs["dropout_p"] == 0.1 for c in call.call_args_list))
        self.assertEqual(first[0].shape, (2, 16))
        self.assertEqual(first[1].shape, (2,))

    def test_sdpa_forward_and_backward_match_explicit_attention(self):
        layer = Attention().double().eval()
        x = torch.randn(2, 5, 64, dtype=torch.float64, requires_grad=True)
        actual = layer(x)
        q, k, v = layer.query(x), layer.key(x), layer.value(x)
        expected = layer.output(((q @ k.transpose(-2, -1)) / 8).softmax(-1) @ v)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        a = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
        b = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)

    def test_mask_and_reduction(self):
        direction = torch.randn(3, 16, requires_grad=True)
        labels = torch.eye(16)[[1, 3, 5]]
        labels[1] = 0
        # Deliberately inconsistent binary labels: direction mask must follow
        # direction labels, not the foreground head target.
        total, ld, lb = losses(direction, torch.zeros(3), labels, torch.ones(3))
        expected = F.cross_entropy(direction[[0, 2]], torch.tensor([1, 5]), reduction="sum") / 3
        torch.testing.assert_close(ld, expected)
        total.backward()
        self.assertTrue(torch.equal(direction.grad[1], torch.zeros(16)))
        torch.testing.assert_close(lb, torch.tensor(np.log(2), dtype=torch.float32))

    def test_directions_and_unsigned_positions(self):
        positions = np.full((9, 50, 2), 100, dtype=np.uint16)
        offsets = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])
        centers = np.concatenate([100+offsets, [[-1, -1]]])
        labels = direction_labels(centers, positions)
        np.testing.assert_array_equal(labels[:8].argmax(-1), np.arange(0, 16, 2))
        self.assertEqual(labels[-1].sum(), 0)

    def test_adamw_decay_is_not_multiplied_by_learning_rate(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = ReferenceAdamW([p])
        p.grad = torch.zeros_like(p)
        optimizer.step()
        torch.testing.assert_close(p, torch.tensor([0.9999]))

    def test_foreach_optimizer_matches_reference_with_missing_gradients(self):
        from .tune import optimizer_check
        self.assertTrue(optimizer_check(torch.device("cpu"))["passed"])

    def test_cached_batches_match_streamed_batches(self):
        from .performance import GPUTrainingCache
        from optimizations.scs_streaming.data import relative_positions
        rng = np.random.RandomState(18)
        arrays = {"x_train": rng.randint(0, 4, (9, 50, 7)).astype(np.uint16),
                  "x_train_pos": rng.randint(0, 100, (9, 50, 2)).astype(np.uint16)}
        class Store:
            def array(self, name):
                return arrays[name]
        labels = np.eye(16, dtype=np.float32)[:9]
        binary = np.ones(9, dtype=np.float32)
        cache = GPUTrainingCache(Store(), labels, binary, torch.device("cpu"), chunk_size=4)
        rows = np.array([8, 0, 2, 2])
        expected = (arrays["x_train"][rows].astype(np.float32),
                    relative_positions(arrays["x_train_pos"])[rows], labels[rows], binary[rows])
        for actual, wanted in zip(cache.batch(rows), expected):
            torch.testing.assert_close(actual, torch.from_numpy(wanted))

    def test_compiled_forward_must_not_be_truth_tested(self):
        from .runner import run_epoch
        model = OriginalSCS(7)
        class Proxy:
            def __bool__(self):
                raise TypeError("Compiled modules need not support truth testing")
            def __call__(self, x, p):
                return model(x, p)
        with patch("optimizations.scs_original_torch.runner.batch_inputs", return_value=(
                np.zeros((2, 50, 7), dtype=np.float32), np.zeros((2, 50, 2), dtype=np.int32))), \
             patch.object(torch.compiler, "cudagraph_mark_step_begin") as mark:
            result = run_epoch(model, None, np.arange(2), np.eye(16, dtype=np.float32)[:2],
                np.ones(2, dtype=np.float32), 2, torch.device("cpu"),
                optimizer=ReferenceAdamW(model.parameters()), compiled_forward=Proxy())
            self.assertTrue(np.isfinite(result["loss"]))
            mark.assert_called_once()

    def test_cuda_device_has_explicit_index_for_allocator_cap(self):
        from types import SimpleNamespace
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=80*2**30)), \
             patch.object(torch.cuda, "set_per_process_memory_fraction") as cap:
            device = configure_device("cuda", "none", 16384)
            self.assertEqual(device, torch.device("cuda", 0))
            cap.assert_called_once_with(0.2, torch.device("cuda", 0))

    def test_train_and_inference_restore_best_and_preserve_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            rng = np.random.RandomState(7)
            # >=100 held-out foreground points exercises validation checkpointing.
            pos = np.full((106, 50, 2), 90, dtype=np.uint16)
            pos[:6] = 10
            arrays = dict(x_train=rng.randint(0, 3, (106, 50, 7)).astype(np.uint16),
                          x_train_pos=pos, y_train=pos[:, 0].astype(np.int32)+1,
                          y_binary_train=np.ones(106, dtype=np.int32),
                          x_test=rng.randint(0, 3, (3, 50, 7)).astype(np.uint16),
                          x_test_pos=np.full((3, 50, 2), 30, dtype=np.uint16))
            for key, value in arrays.items():
                np.savez_compressed(data / f"{key}_0:0:0:0.npz", **{key: value})
            with h5py.File(data / "spots0:0:0:0.h5ad", "w") as f:
                f.create_dataset("X", shape=(100, 100), dtype="u1")
            output = root / "output"
            result = train(data, output, epochs=2, batch_size=64, device="cpu", cache_root=tmp)
            self.assertTrue(result["completed"])
            self.assertEqual(result["prediction_rows"], 109)
            best = torch.load(output / "best.pt", weights_only=True)
            self.assertEqual(result["best_epoch"], best["epoch"])
            self.assertEqual(best["config"]["monitor"], "val_pos_out_accuracy")
            prediction = (output / "results/spot_prediction_0:0:0:0.txt").read_text().splitlines()
            self.assertEqual(len(prediction), 109)
            self.assertEqual(prediction[0].split("\t")[:2], ["10", "10"])
            self.assertEqual(prediction[-1].split("\t")[:2], ["30", "30"])
            self.assertEqual(len(prediction[0].split("\t")[3].split(":")), 16)
            # Compare exported values directly to best.pt (not latest.pt).
            model = OriginalSCS(7).eval()
            model.load_state_dict(best["model"])
            with torch.no_grad():
                logits, prob = model(torch.tensor(arrays["x_train"][:1].astype(np.float32)),
                                     torch.zeros(1, 50, 2))
            fields = prediction[0].split("\t")
            np.testing.assert_allclose(np.fromstring(fields[3], sep=":"), logits.numpy()[0], atol=2e-5)
            self.assertAlmostEqual(float(fields[2]), prob.sigmoid().item(), places=5)
            standalone = root / "standalone"
            predicted = predict(output / "best.pt", data, standalone, batch_size=128,
                                device="cpu", cache_root=tmp)
            self.assertEqual(predicted["prediction_rows"], 109)
            self.assertEqual((standalone / "spot_prediction_0:0:0:0.txt").read_text().splitlines(), prediction)
            with self.assertRaises(ValueError):
                predict(output / "best.pt", root / "wrong_tile", root / "bad", device="cpu")
            with self.assertRaises(FileExistsError):
                train(data, output, epochs=1, device="cpu")


if __name__ == "__main__":
    unittest.main()
