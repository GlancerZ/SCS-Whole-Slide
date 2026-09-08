"""PyTorch minibatch loading over compact whole-slide SCS data."""

import json
import math
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset

from .shared_data import SharedBatches, TileCache, load_schema


class PlannedBatchDataset(Dataset):
    """One deterministic dense-expression epoch plan for compatibility checks."""

    def __init__(
        self,
        batches,
        epoch=0,
        batch_size=None,
        merge_tiles=True,
        expression_dtype=np.float16,
    ):
        self.batches = batches
        self.expression_dtype = expression_dtype
        source = list(batches.plan(epoch))
        self.plan = (
            self._merge(source, batch_size or batches.batch_size)
            if merge_tiles
            else [[item] for item in source]
        )
        self.cache = None

    @staticmethod
    def _merge(source, batch_size):
        result, current, count = [], [], 0
        for tile_id, source_rows in source:
            offset = 0
            while offset < len(source_rows):
                take = min(batch_size - count, len(source_rows) - offset)
                current.append((tile_id, source_rows[offset : offset + take]))
                count += take
                offset += take
                if count == batch_size:
                    result.append(current)
                    current, count = [], 0
        if current:
            result.append(current)
        return result

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        parts = self.plan[index]
        if self.cache is None:
            self.cache = TileCache(
                self.batches.root, self.batches.schema, capacity=max(2, len(parts))
            )
        arrays = [
            self.cache.get(tile_id).batch(rows, self.expression_dtype)
            for tile_id, rows in parts
        ]
        expression = np.concatenate([part[0][0] for part in arrays], axis=0)
        positions = np.concatenate([part[0][1] for part in arrays], axis=0)
        directions = np.concatenate([part[1][0] for part in arrays], axis=0)
        foreground = np.concatenate([part[1][1] for part in arrays], axis=0)
        return expression, positions, directions.argmax(axis=-1), foreground


class GenePTBatchDataset(Dataset):
    """Globally shuffled sparse spots for temporary GenePT projection on GPU."""

    def __init__(
        self,
        root,
        split,
        batch_size,
        seed,
        epoch=0,
        dataset_name="genept_allgenes_linear_random90",
        residency="memory",
        selected_gene_ids=None,
    ):
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.root = Path(root)
        self.directory = self.root / dataset_name
        self.schema = load_schema(root)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.gene_remap = None
        if selected_gene_ids is not None:
            selected = np.asarray(selected_gene_ids, dtype=np.int64)
            if (selected.ndim != 1 or not len(selected) or len(np.unique(selected)) != len(selected)
                    or selected.min() < 0 or selected.max() >= self.manifest["n_genes"]):
                raise ValueError("selected_gene_ids must be unique valid gene IDs")
            self.gene_remap = np.full(self.manifest["n_genes"], -1, dtype=np.int64)
            self.gene_remap[selected] = np.arange(len(selected))
        if not self.manifest.get("complete"):
            raise ValueError("GenePT dataset is incomplete")
        if self.manifest["schema_fingerprint"] != self.schema["fingerprint"]:
            raise ValueError("GenePT dataset does not match the feature schema")
        expected_representation = (
            "all-source-gene sparse spots with native GenePT and learned projection"
        )
        if self.manifest["representation"] != expected_representation:
            raise ValueError("dataset is not the learned GenePT representation")
        if split not in self.manifest["splits"]:
            raise ValueError(f"unknown GenePT split: {split}")
        if residency not in ("memory", "mmap"):
            raise ValueError("residency must be memory or mmap")
        self.split = split
        self.batch_size = batch_size
        self.samples = self.manifest["splits"][split]["samples"]
        self.counts = self.manifest["splits"][split]["counts"]
        self.steps = math.ceil(self.samples / batch_size)
        mode = None if residency == "memory" else "r"
        self.expression_data = np.load(
            self.directory / "expression_data.npy", mmap_mode=mode
        )
        self.expression_indices = np.load(
            self.directory / "expression_indices.npy", mmap_mode=mode
        )
        self.expression_indptr = np.load(
            self.directory / "expression_indptr.npy", mmap_mode=mode
        )
        self.neighbors = np.load(
            self.directory / f"{split}_neighbors.npy", mmap_mode=mode
        )
        self.positions = np.load(
            self.directory / f"{split}_positions.npy", mmap_mode=mode
        )
        self.directions = np.load(
            self.directory / f"{split}_directions.npy", mmap_mode=mode
        )
        self.foreground = np.load(
            self.directory / f"{split}_foreground.npy", mmap_mode=mode
        )
        if len(self.expression_indptr) != self.manifest["expression_rows"] + 1:
            raise ValueError("sparse spot pointer table has the wrong shape")
        if len(self.expression_data) != self.manifest["expression_nonzero"]:
            raise ValueError("sparse spot value table has the wrong shape")
        if len(self.expression_indices) != len(self.expression_data):
            raise ValueError("sparse spot indices and values have different lengths")
        if (
            self.expression_indptr[0] != 0
            or self.expression_indptr[-1] != len(self.expression_data)
        ):
            raise ValueError("sparse spot pointer bounds are invalid")
        maximum_gene = (
            int(self.expression_indices.max()) if len(self.expression_indices) else -1
        )
        if maximum_gene >= self.manifest["n_genes"]:
            raise ValueError("sparse spot gene index is out of bounds")
        if self.neighbors.shape != (self.samples, self.manifest["n_neighbors"]):
            raise ValueError("GenePT sample table has the wrong shape")
        if self.positions.shape != (*self.neighbors.shape, 2):
            raise ValueError("GenePT position table has the wrong shape")
        if self.directions.shape != (self.samples,) or self.foreground.shape != (
            self.samples,
        ):
            raise ValueError("GenePT label table has the wrong shape")
        rng = np.random.RandomState(seed + (epoch if split == "train" else 0))
        self.order = rng.permutation(self.samples)

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        start = index * self.batch_size
        sample_ids = self.order[start : start + self.batch_size]
        token_rows = np.asarray(self.neighbors[sample_ids]).reshape(-1)
        starts = self.expression_indptr[token_rows].astype(np.int64)
        stops = self.expression_indptr[token_rows + 1].astype(np.int64)
        lengths = stops - starts
        offsets = np.empty(len(token_rows) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        source = np.arange(offsets[-1], dtype=np.int64)
        source += np.repeat(starts - offsets[:-1], lengths)
        genes = np.asarray(self.expression_indices[source])
        values = np.asarray(self.expression_data[source])
        if self.gene_remap is not None:
            mapped = self.gene_remap[genes]
            keep = mapped >= 0
            prefix = np.empty(len(keep)+1, dtype=np.int64)
            prefix[0] = 0
            np.cumsum(keep, out=prefix[1:])
            offsets = prefix[offsets]
            genes, values = mapped[keep], values[keep]
        sparse_expression = (
            genes,
            values,
            offsets,
            np.asarray(
                [len(sample_ids), self.manifest["n_neighbors"]], dtype=np.int64
            ),
        )
        return (
            sparse_expression,
            np.asarray(self.positions[sample_ids]),
            np.asarray(self.directions[sample_ids]),
            np.asarray(self.foreground[sample_ids]),
        )


def input_dimension(root, input_pipeline, dataset_name):
    if input_pipeline == "genept":
        manifest = json.loads((Path(root) / dataset_name / "manifest.json").read_text())
        return int(manifest["gene_embedding_dimension"])
    if input_pipeline == "cpu-dense":
        return len(load_schema(root)["gene_indices"])
    raise ValueError("input_pipeline must be genept or cpu-dense")


def whole_slide_batches(
    root,
    split,
    batch_size,
    per_class_cap,
    seed,
    epoch=0,
    workers=4,
    prefetch=2,
    merge_tiles=True,
    split_mode="random",
    validation_fraction=0.1,
    expression_dtype=np.float16,
    input_pipeline="genept",
    dataset_name="genept_allgenes_linear_random90",
    residency="memory",
):
    if input_pipeline == "genept":
        if per_class_cap:
            raise ValueError("GenePT data is full-split; per-class cap must be zero")
        if split_mode != "random" or validation_fraction != 0.1:
            raise ValueError("GenePT data was built for random 90/10 splitting")
        dataset = GenePTBatchDataset(
            root,
            split,
            batch_size,
            seed,
            epoch,
            dataset_name,
            residency,
        )
        batches = dataset
    elif input_pipeline == "cpu-dense":
        batches = SharedBatches(
            root,
            split,
            batch_size,
            per_class_cap,
            seed,
            split_mode,
            validation_fraction,
        )
        dataset = PlannedBatchDataset(
            batches,
            epoch,
            batch_size,
            merge_tiles,
            expression_dtype=expression_dtype,
        )
    else:
        raise ValueError("input_pipeline must be genept or cpu-dense")
    kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers:
        kwargs.update(
            prefetch_factor=prefetch,
            persistent_workers=False,
            multiprocessing_context="fork",
        )
    return batches, DataLoader(**kwargs)
