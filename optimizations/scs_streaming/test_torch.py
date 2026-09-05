"""Small architecture/loss checks for the PyTorch whole-slide model."""

import unittest
from unittest import mock

import torch

from .shared_train_torch import move_batch
from .torch_data import PlannedBatchDataset
from .torch_model import ModelConfig, SCSClassifier, scs_loss


class TorchModelTests(unittest.TestCase):
    def test_scaled_shapes_and_explicit_sdpa(self):
        config = ModelConfig(n_genes=31, n_neighbors=5, scale=2, base_layers=2)
        model = SCSClassifier(config).eval()
        expression = torch.randn(3, 5, 31)
        positions = torch.randint(-4, 5, (3, 5, 2))
        original = torch.nn.functional.scaled_dot_product_attention
        with mock.patch(
            "torch.nn.functional.scaled_dot_product_attention", wraps=original
        ) as sdpa:
            direction, foreground = model(expression, positions)
        self.assertEqual(direction.shape, (3, 16))
        self.assertEqual(foreground.shape, (3,))
        self.assertEqual(sdpa.call_count, 4)
        self.assertEqual(config.width, 128)
        self.assertEqual(config.layers, 4)
        self.assertEqual(config.head_dim, 64)

    def test_optimizer_groups_are_complete_and_muon_is_hidden_matrices(self):
        model = SCSClassifier(
            ModelConfig(n_genes=17, n_neighbors=5, scale=1, base_layers=1)
        )
        muon, adamw = model.optimizer_parameter_groups()
        self.assertTrue(muon)
        self.assertTrue(adamw)
        self.assertTrue(all(parameter.ndim == 2 for parameter in muon))
        self.assertEqual(
            {id(p) for p in model.parameters()}, {id(p) for p in muon + adamw}
        )

    def test_block_matches_reference_activation_and_dropout_order(self):
        model = SCSClassifier(
            ModelConfig(n_genes=17, n_neighbors=5, scale=1, base_layers=1)
        )
        block = model.blocks[0]
        self.assertFalse(hasattr(block, "attention_dropout"))
        self.assertEqual(
            [type(layer) for layer in block.mlp],
            [
                torch.nn.Linear,
                torch.nn.GELU,
                torch.nn.Dropout,
                torch.nn.Linear,
                torch.nn.GELU,
                torch.nn.Dropout,
            ],
        )

    def test_background_directions_do_not_contribute_to_direction_loss(self):
        direction_logits = torch.tensor([[0.0, 3.0], [2.0, -2.0]])
        foreground_logits = torch.zeros(2)
        foreground = torch.tensor([1.0, 0.0])
        first, _, _ = scs_loss(
            direction_logits, foreground_logits, torch.tensor([1, 0]), foreground
        )
        second, _, _ = scs_loss(
            direction_logits, foreground_logits, torch.tensor([1, 1]), foreground
        )
        self.assertTrue(torch.equal(first, second))

    def test_direct_numpy_batch_converts_one_hot_directions(self):
        import numpy as np

        nested = (
            (np.zeros((2, 5, 7), np.float32), np.zeros((2, 5, 2), np.int32)),
            (np.eye(16, dtype=np.float32)[[3, 9]], np.array([1, 0], np.float32)),
        )
        moved = move_batch(nested, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.equal(moved[2], torch.tensor([3, 9])))

    def test_tile_chunks_are_merged_and_split_without_loss(self):
        source = [("a", torch.arange(3).numpy()), ("b", torch.arange(3, 8).numpy())]
        merged = PlannedBatchDataset._merge(source, batch_size=4)
        self.assertEqual(
            [[len(rows) for _, rows in group] for group in merged], [[3, 1], [4]]
        )
        self.assertEqual(
            [tile for group in merged for tile, _ in group], ["a", "b", "b"]
        )


if __name__ == "__main__":
    unittest.main()
