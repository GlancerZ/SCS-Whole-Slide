"""Read compressed SCS arrays without materializing full expression tensors in RAM."""
import contextlib
import math
import mmap
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np


class ArrayStore:
    """Extract original-dtype NPY streams to temporary disk, then memory-map them.

    The caller owns a private cache directory. Original NPZ files are read-only.
    No expression array is cast to float32 until a batch is requested.
    """

    def __init__(self, data_dir, cache_dir, suffix="0:0:0:0"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.suffix = suffix
        self.arrays = {}
        self.opened = []

    def path(self, key):
        return self.data_dir / f"{key}_{self.suffix}.npz"

    def header(self, key):
        with zipfile.ZipFile(self.path(key)) as archive, archive.open(key + ".npy") as stream:
            version = np.lib.format.read_magic(stream)
            return np.lib.format._read_array_header(stream, version)

    def array(self, key):
        if key not in self.arrays:
            path = self.cache_dir / (key + ".npy")
            # Fresh store directories prevent reuse of stale arrays from another tile.
            if path.exists():
                raise FileExistsError(f"Cache is not private/empty: {path}")
            temporary = path.with_suffix(".npy.partial")
            try:
                with zipfile.ZipFile(self.path(key)) as archive, archive.open(key + ".npy") as source, temporary.open("xb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                temporary.replace(path)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    temporary.unlink()
                raise
            self.arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
            self.opened.append(key)
        return self.arrays[key]

    def close(self):
        # Call only after consumers and TensorFlow datasets have finished.
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()


def relative_positions(absolute):
    positions = np.asarray(absolute, dtype=np.int32)
    return positions - positions[:, :1, :]


def batch_inputs(store, subset, indices):
    expression = store.array("x_" + subset)[indices].astype(np.float32, copy=False)
    positions = relative_positions(store.array("x_" + subset + "_pos")[indices])
    return expression, positions


def partition(absolute_positions, binary, shape, val_ratio):
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must lie in [0, 1)")
    origins = np.asarray(absolute_positions[:, 0, :])
    held_out = ((origins[:, 0] > int(shape[0] * (1 - np.sqrt(val_ratio)))) &
                (origins[:, 1] > int(shape[1] * (1 - np.sqrt(val_ratio)))))
    return np.flatnonzero(~held_out), np.flatnonzero(held_out & (binary == 1))


def index_batches(indices, batch_size, shuffle=False, rng=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    order = np.array(indices, dtype=np.int64, copy=True)
    if shuffle:
        if rng is None:
            raise ValueError("A seeded generator is required for shuffled batches")
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def dataset(store, indices, labels, binary, batch_size=10, shuffle=False, seed=20260905):
    # Import TensorFlow only for execution, keeping metadata utilities lightweight.
    import tensorflow as tf
    shape, _, _ = store.header("x_train")
    rng = np.random.RandomState(seed)

    def generate():
        for rows in index_batches(indices, batch_size, shuffle, rng):
            expression, positions = batch_inputs(store, "train", rows)
            yield (expression, positions), (labels[rows], binary[rows])

    spec = ((tf.TensorSpec((None,) + shape[1:], tf.float32),
             tf.TensorSpec((None, shape[1], 2), tf.int32)),
            (tf.TensorSpec((None, 16), tf.float32), tf.TensorSpec((None,), tf.float32)))
    ds = tf.data.Dataset.from_generator(generate, output_signature=spec)
    ds = ds.apply(tf.data.experimental.assert_cardinality(math.ceil(len(indices) / batch_size)))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = True
    return ds.with_options(options).prefetch(2)


class TensorTrainingData:
    """One original-dtype CPU tensor shared by training and validation datasets.

    Unlike the Python generator, gathers and float32 conversion run as TensorFlow
    graph operations. This avoids a Python callback for every training batch.
    """

    def __init__(self, store, labels, binary):
        import tensorflow as tf
        expression = store.array("x_train")
        with tf.device("/CPU:0"):
            self.expression = tf.convert_to_tensor(expression)
            self.positions = tf.convert_to_tensor(relative_positions(store.array("x_train_pos")))
            self.labels = tf.convert_to_tensor(labels, dtype=tf.float32)
            self.binary = tf.convert_to_tensor(binary, dtype=tf.float32)
        if hasattr(expression._mmap, "madvise"):
            expression._mmap.madvise(mmap.MADV_DONTNEED)

    def dataset(self, indices, batch_size=10, shuffle=False, seed=20260905):
        import tensorflow as tf
        with tf.device("/CPU:0"):
            ds = tf.data.Dataset.from_tensor_slices(np.asarray(indices, dtype=np.int32))
            if shuffle:
                ds = ds.shuffle(len(indices), seed=seed, reshuffle_each_iteration=True)
            ds = ds.batch(batch_size, drop_remainder=False)

        def gather(rows):
            with tf.device("/CPU:0"):
                return ((tf.cast(tf.gather(self.expression, rows), tf.float32),
                         tf.gather(self.positions, rows)),
                        (tf.gather(self.labels, rows), tf.gather(self.binary, rows)))

        ds = ds.map(gather, num_parallel_calls=2, deterministic=True)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        options.threading.max_intra_op_parallelism = 1
        options.experimental_deterministic = True
        return ds.with_options(options).prefetch(2)


def predict_to_file(model, store, output_path, batch_size=128):
    """Preserve SCS row order: all training spots, then all test spots."""
    import tensorflow as tf
    signature = [tf.TensorSpec((None,) + tuple(model.input_shape[0][1:]), tf.float32),
                 tf.TensorSpec((None,) + tuple(model.input_shape[1][1:]), tf.int32)]

    @tf.function(input_signature=signature)
    def infer(expression, positions):
        return model([expression, positions], training=False)

    destination = Path(output_path)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    nrows = 0
    with temporary.open("w", buffering=1024 * 1024) as handle:
        for subset in ("train", "test"):
            shape, _, _ = store.header("x_" + subset)
            if not shape[0]:
                continue
            absolute = store.array("x_" + subset + "_pos")
            for start in range(0, shape[0], batch_size):
                rows = slice(start, min(start + batch_size, shape[0]))
                expression, positions = batch_inputs(store, subset, rows)
                logits, foreground = infer(expression, positions)
                logits, foreground = logits.numpy(), foreground.numpy()
                if not (np.all(np.isfinite(logits)) and np.all(np.isfinite(foreground))):
                    raise ValueError("Nonfinite model predictions")
                coordinates = np.asarray(absolute[rows, 0, :])
                handle.writelines(f"{int(x)}\t{int(y)}\t{float(prob[0])}\t" +
                                  ":".join(map(str, logit)) + "\n"
                                  for (x, y), prob, logit in zip(coordinates, foreground, logits))
                nrows += len(coordinates)
    temporary.replace(destination)
    return nrows
