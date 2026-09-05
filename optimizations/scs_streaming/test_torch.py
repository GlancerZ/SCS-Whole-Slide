"""Small architecture/loss checks for the PyTorch whole-slide model."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch
from scipy import sparse

from .genept import make_gene_lookup, pool_spot_embeddings
from .shared_data import (
    direction_class_indices,
    direction_classes,
    fingerprint,
    random_split_rows,
    save_json,
)
from .shared_train_torch import move_batch
from .torch_data import GenePTBatchDataset, PlannedBatchDataset
from .torch_model import ModelConfig, SCSClassifier, scs_loss


class TorchModelTests(unittest.TestCase):
    def test_scaled_shapes_and_explicit_sdpa(self):
        config = ModelConfig(input_dim=31, n_neighbors=5, scale=2, base_layers=2)
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

    def test_genept_spot_pooling_weights_then_divides_by_nonzero_genes(self):
        expression = sparse.csr_matrix(
            np.array([[2, 0, 3], [0, 5, 0], [0, 0, 0]], dtype=np.uint16)
        )
        lookup = np.array([[1, 2], [8, 9], [3, 4]], dtype=np.float32)
        pooled, nonzero = pool_spot_embeddings(
            expression, lookup, np.array([True, False, True])
        )
        np.testing.assert_allclose(pooled[0], [5.5, 8.0])
        np.testing.assert_array_equal(nonzero, [2, 0, 0])
        np.testing.assert_array_equal(pooled[1:], 0)

    def test_genept_lookup_uses_all_columns_and_merges_duplicate_symbols(self):
        embeddings = {
            "A": np.array([1, 2], dtype=np.float32),
            "B": np.array([3, 4], dtype=np.float32),
        }
        table, source_to_gene, symbols, metadata = make_gene_lookup(
            ["A", "OUTSIDE_OLD_HVG", "a", "B"], embeddings
        )
        np.testing.assert_array_equal(source_to_gene, [0, -1, 0, 1])
        np.testing.assert_array_equal(symbols, ["A", "B"])
        np.testing.assert_array_equal(table, [[1, 2], [3, 4]])
        self.assertEqual(metadata["source_genes"], 4)
        self.assertEqual(metadata["mapped_unique_genes"], 2)

    def test_genept_dataset_keeps_neighbor_spots_as_separate_tokens(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = {"gene_indices": [0, 1], "seed": 17}
            schema["fingerprint"] = fingerprint(schema)
            save_json(root / "schema.json", schema)
            directory = root / "genept"
            directory.mkdir()
            save_json(
                directory / "manifest.json",
                {
                    "complete": True,
                    "schema_fingerprint": schema["fingerprint"],
                    "representation": (
                        "all-source-gene sparse spots with native GenePT and "
                        "learned projection"
                    ),
                    "gene_embedding_dimension": 3,
                    "n_genes": 3,
                    "expression_rows": 4,
                    "expression_nonzero": 6,
                    "n_neighbors": 2,
                    "splits": {"train": {"samples": 3, "counts": {}}},
                },
            )
            np.save(
                directory / "expression_data.npy",
                np.array([2, 3, 5, 1, 4, 7], dtype=np.uint16),
            )
            np.save(
                directory / "expression_indices.npy",
                np.array([0, 1, 2, 0, 2, 1], dtype=np.uint16),
            )
            np.save(
                directory / "expression_indptr.npy",
                np.array([0, 2, 3, 5, 6], dtype=np.uint32),
            )
            np.save(
                directory / "train_neighbors.npy",
                np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32),
            )
            np.save(
                directory / "train_positions.npy",
                np.zeros((3, 2, 2), dtype=np.int8),
            )
            np.save(directory / "train_directions.npy", np.arange(3, dtype=np.uint8))
            np.save(directory / "train_foreground.npy", np.ones(3, dtype=np.uint8))
            dataset = GenePTBatchDataset(
                root, "train", batch_size=2, seed=17, dataset_name="genept"
            )
            sparse_expression, positions, directions, foreground = dataset[0]
            indices, values, offsets, shape = sparse_expression
            np.testing.assert_array_equal(shape, [2, 2])
            self.assertEqual(len(offsets), 2 * 2 + 1)
            self.assertEqual(offsets[-1], len(indices))
            self.assertEqual(len(indices), len(values))
            self.assertEqual(positions.shape, (2, 2, 2))
            self.assertEqual(directions.shape, (2,))
            self.assertEqual(foreground.shape, (2,))

    def test_sparse_projection_exactly_matches_native_pool_then_linear(self):
        gene_embeddings = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        config = ModelConfig(
            input_dim=3, n_genes=3, n_neighbors=2, scale=1, base_layers=1
        )
        model = SCSClassifier(config, gene_embeddings).eval()
        sparse_expression = (
            torch.tensor([0, 2, 1]),
            torch.tensor([2.0, 3.0, 5.0]),
            torch.tensor([0, 2, 3]),
            (1, 2),
        )
        pooled_native = torch.stack(
            ((2 * gene_embeddings[0] + 3 * gene_embeddings[2]) / 2, 5 * gene_embeddings[1])
        ).unsqueeze(0)
        expected = model.expression_projection(pooled_native) + model.spot_bias
        actual = model.project_expression(sparse_expression)
        torch.testing.assert_close(actual, expected)

    def test_position_projection_matches_scaled_token_width(self):
        model = SCSClassifier(ModelConfig(input_dim=1536, scale=4, base_layers=1))
        self.assertEqual(model.expression_projection.in_features, 1536)
        self.assertEqual(model.expression_projection.out_features, 256)
        self.assertEqual(model.position_projection.in_features, 2)
        self.assertEqual(model.position_projection.out_features, 256)

    def test_optimizer_groups_are_complete_and_muon_is_hidden_matrices(self):
        model = SCSClassifier(
            ModelConfig(input_dim=17, n_neighbors=5, scale=1, base_layers=1)
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
            ModelConfig(input_dim=17, n_neighbors=5, scale=1, base_layers=1)
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

    def test_random_point_split_is_fixed_disjoint_and_exhaustive(self):
        rows = torch.arange(101).numpy()
        training = random_split_rows(rows, "train", 0.1, 17, "tile:positive")
        validation = random_split_rows(rows, "validation", 0.1, 17, "tile:positive")
        self.assertEqual(len(validation), 10)
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(set(rows), set(training) | set(validation))
        self.assertTrue(
            (
                validation
                == random_split_rows(rows, "validation", 0.1, 17, "tile:positive")
            ).all()
        )

    def test_direction_indices_match_one_hot_targets(self):
        directions = np.array([[1, 0], [0, 1], [-1, 0], [4, -2]])
        binary = np.array([1, 1, 0, 1])
        indices = direction_class_indices(directions, binary)
        one_hot = direction_classes(directions, binary)
        self.assertTrue(
            np.array_equal(indices[binary == 1], one_hot.argmax(-1)[binary == 1])
        )
        self.assertEqual(indices[2], 0)
        self.assertTrue((one_hot[2] == 0).all())


if __name__ == "__main__":
    unittest.main()
