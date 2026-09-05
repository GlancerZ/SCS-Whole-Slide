"""Sparse, bounded-memory inputs for ONE SCS model across spatial tiles.

Store each expression bin once, and store neighbor row indices rather than an
N x neighbors x genes tensor. Only the current minibatch becomes float32/dense.
"""
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import sparse


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_schema(root):
    schema = json.loads((Path(root) / "schema.json").read_text())
    content = {k: v for k, v in schema.items() if k != "fingerprint"}
    if fingerprint(content) != schema["fingerprint"]:
        raise ValueError("Shared feature/split schema was modified")
    return schema


def ring_offsets(rings=10):
    # Exactly the upstream square-ring traversal; this is not Euclidean KNN.
    offsets = [(0, 0)]
    for d in range(1, rings + 1):
        offsets.extend((-d, y) for y in range(-d, d + 1))
        offsets.extend((d, y) for y in range(-d, d + 1))
        offsets.extend((x, -d) for x in range(-d + 1, d))
        offsets.extend((x, d) for x in range(-d + 1, d))
    return np.asarray(offsets, dtype=np.int32)


def neighbor_table(expression, grid_shape, n_neighbors=50, rings=10, chunk_size=2048):
    if not 1 <= n_neighbors <= (2 * rings + 1) ** 2:
        raise ValueError("Neighbor count exceeds search-window capacity")
    occupied = np.asarray(expression.getnnz(axis=1)).ravel() > 0
    centers = np.flatnonzero(occupied)
    offsets = ring_offsets(rings)
    tables = []
    for start in range(0, len(centers), chunk_size):
        c = centers[start:start + chunk_size]
        x = c[:, None] // grid_shape[1] + offsets[None, :, 0]
        y = c[:, None] % grid_shape[1] + offsets[None, :, 1]
        valid = (x >= 0) & (x < grid_shape[0]) & (y >= 0) & (y < grid_shape[1])
        rows = x * grid_shape[1] + y
        valid &= occupied[np.clip(rows, 0, len(occupied) - 1)]
        eligible = valid.sum(axis=1) >= n_neighbors
        rows, valid = rows[eligible], valid[eligible]
        # Stable selection preserves the published ring order and center first.
        ranks = np.cumsum(valid, axis=1)
        tables.append(rows[valid & (ranks <= n_neighbors)].reshape(-1, n_neighbors).astype(np.int32))
    return np.concatenate(tables) if tables else np.empty((0, n_neighbors), np.int32), len(centers)


def interior_mask(local_xy, tile, margin):
    """No halo duplicates; erode each core to isolate train/validation contexts."""
    h = tile["halo"]
    return ((local_xy[:, 0] >= h + margin) & (local_xy[:, 0] < h + tile["width"] - margin) &
            (local_xy[:, 1] >= h + margin) & (local_xy[:, 1] < h + tile["height"] - margin))


def direction_classes(directions, binary):
    # Match SCS.dir_to_class, including the zero-vector convention.
    angle = np.mod(np.arctan2(directions[:, 0], directions[:, 1]), 2 * np.pi)
    classes = (angle / (2 * np.pi / 16)).astype(int) % 16
    result = np.eye(16, dtype=np.float32)[classes]
    result[np.asarray(binary) != 1] = 0
    return result


class TileStore:
    def __init__(self, root, tile_id, schema):
        directory = Path(root) / "tiles" / tile_id
        self.meta = json.loads((directory / "prepared.json").read_text())
        if self.meta["fingerprint"] != schema["fingerprint"]:
            raise ValueError(f"Incompatible feature schema: {tile_id}")
        self.expression = sparse.load_npz(directory / "expression.npz").tocsr()
        self.neighbors = np.load(directory / "neighbors.npy", mmap_mode="r")
        self.binary = np.load(directory / "binary.npy", mmap_mode="r")
        self.directions = np.load(directory / "directions.npy", mmap_mode="r")
        self.eligible = np.load(directory / "eligible.npy", mmap_mode="r")
        self.bin_size = schema["bin_size"]
        self.grid_shape = self.meta["grid_shape"]
        if self.expression.shape[1] != len(schema["gene_indices"]) or self.neighbors.shape[1] != schema["n_neighbors"]:
            raise ValueError("Tile dimensions do not match shared model")

    def inputs(self, rows):
        neighbors = np.asarray(self.neighbors[rows])
        expression = self.expression[neighbors.ravel()].toarray().reshape(
            len(neighbors), neighbors.shape[1], self.expression.shape[1]).astype(np.float32)
        absolute = np.stack((neighbors // self.grid_shape[1], neighbors % self.grid_shape[1]), axis=-1) * self.bin_size
        return expression, (absolute - absolute[:, :1]).astype(np.int32), absolute[:, 0]

    def batch(self, rows):
        x, p, _ = self.inputs(rows)
        binary = np.asarray(self.binary[rows], dtype=np.float32)
        return (x, p), (direction_classes(self.directions[rows], binary), binary)


class TileCache:
    def __init__(self, root, schema, capacity=2):
        self.root, self.schema, self.capacity = Path(root), schema, capacity
        self.cache = OrderedDict()

    def get(self, tile_id):
        if tile_id not in self.cache:
            self.cache[tile_id] = TileStore(self.root, tile_id, self.schema)
        self.cache.move_to_end(tile_id)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return self.cache[tile_id]


class SharedBatches:
    """Region-balanced, per-epoch resampling, without retaining all tile arrays.

    Batches are grouped by tile for disk locality; tile and sample order are
    reshuffled each epoch. Every training tile contributes to the SAME optimizer.
    A per-class cap is an explicit sampling budget, not all-data epochs.
    """
    def __init__(self, root, split, batch_size=10, per_class_cap=4096, seed=20260905):
        if batch_size < 1 or per_class_cap < 1:
            raise ValueError("Batch size and regional cap must be positive")
        self.root, self.schema = Path(root), load_schema(root)
        self.split, self.batch_size, self.cap, self.seed = split, batch_size, per_class_cap, seed
        self.pools = {}
        self.counts = {}
        cache = TileCache(root, self.schema, capacity=1)
        for tid in self.schema["splits"][split]:
            meta = json.loads((self.root / "tiles" / tid / "prepared.json").read_text())
            if meta["fingerprint"] != self.schema["fingerprint"]:
                raise ValueError("Tile schema mismatch")
            if meta["state"] == "empty":
                continue
            tile = cache.get(tid)
            positive = np.flatnonzero(tile.eligible & (tile.binary == 1))
            negative = np.flatnonzero(tile.eligible & (tile.binary == 0))
            # Nucleus-free RNA-containing tiles still contribute background and
            # remain inference targets; they are never called empty.
            npos = min(len(positive), self.cap)
            nneg = min(len(negative), self.cap, npos if npos else self.cap)
            if npos + nneg:
                self.pools[tid] = (positive, negative)
                self.counts[tid] = (npos, nneg)
        self.samples = sum(sum(v) for v in self.counts.values())
        self.steps = sum(math.ceil(sum(v) / batch_size) for v in self.counts.values())
        if not self.samples:
            raise ValueError(f"No labeled {split} samples; inspect preparation QA")
        if not sum(v[0] for v in self.counts.values()):
            raise ValueError(f"No foreground {split} samples")

    def plan(self, epoch=0):
        rng = np.random.RandomState(self.seed + (epoch if self.split == "train" else 0))
        tile_ids = list(self.pools)
        if self.split == "train":
            rng.shuffle(tile_ids)
        for tid in tile_ids:
            rows = np.concatenate([rng.choice(pool, n, replace=False)
                                   for pool, n in zip(self.pools[tid], self.counts[tid])]).astype(np.int64)
            rng.shuffle(rows)
            for start in range(0, len(rows), self.batch_size):
                yield tid, rows[start:start + self.batch_size]

    def dataset(self, epoch=0):
        import tensorflow as tf
        n, g = self.schema["n_neighbors"], len(self.schema["gene_indices"])

        def generate():
            cache = TileCache(self.root, self.schema, capacity=1)
            for tid, rows in self.plan(epoch):
                yield cache.get(tid).batch(rows)

        signature = ((tf.TensorSpec((None, n, g), tf.float32), tf.TensorSpec((None, n, 2), tf.int32)),
                     (tf.TensorSpec((None, 16), tf.float32), tf.TensorSpec((None,), tf.float32)))
        ds = tf.data.Dataset.from_generator(generate, output_signature=signature)
        ds = ds.apply(tf.data.experimental.assert_cardinality(self.steps))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        options.threading.max_intra_op_parallelism = 1
        return ds.with_options(options).prefetch(2)
