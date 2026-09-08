#!/usr/bin/env python3
"""Memory-bounded SCS trainer; all output is isolated from the original run."""
import argparse
import gc
import json
import mmap
import os
import resource
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

from .data import ArrayStore, TensorTrainingData, dataset, partition, predict_to_file


def memory_record(tf):
    gpu = tf.config.experimental.get_memory_info("GPU:0") if tf.config.list_physical_devices("GPU") else {}
    return dict(process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                gpu_current_mib=gpu.get("current", 0) / 2**20,
                gpu_peak_mib=gpu.get("peak", 0) / 2**20)


def train(data_dir, output_dir, epochs=100, val_ratio=0.0625, suffix="0:0:0:0",
          cache_root=None, gpu_memory_mib=16384, inference_batch_size=128, seed=20260905,
          inference=True, input_mode="arrays"):
    import tensorflow as tf
    import tensorflow_addons as tfa
    from .model_reference import create_transformer_classifier, dir_to_class, masked_categorical_cross_entropy

    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if data_dir.parent == output_dir or data_dir == output_dir:
        raise ValueError("Use a separate output directory; existing runs must remain untouched")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "training_history.csv").exists():
        raise FileExistsError("Output already contains training history; use a fresh output directory")
    tf.keras.utils.set_random_seed(seed)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if len(gpus) != 1:
            raise ValueError("Expected a single allocated GPU")
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_mib)])
    for name in ("results", "ckpt"):
        (output_dir / name).mkdir(exist_ok=True)
    with h5py.File(data_dir / f"spots{suffix}.h5ad") as f:
        x = f["X"]
        shape = tuple(x.attrs["shape"]) if isinstance(x, h5py.Group) else x.shape
    started = time.time()
    scratch_root = cache_root or os.environ.get("SLURM_TMPDIR") or None
    with tempfile.TemporaryDirectory(prefix="scs-stream-", dir=scratch_root) as cache_dir:
        store = ArrayStore(data_dir, cache_dir, suffix)
        try:
            # Only training arrays are opened before fit. Test expression stays compressed.
            expression = store.array("x_train")
            positions = store.array("x_train_pos")
            centers = np.array(store.array("y_train"), copy=True)
            binary = np.asarray(store.array("y_binary_train"), dtype=np.float32)
            foreground = centers[:, 0] != -1
            centers[foreground] -= positions[foreground, 0, :]
            centers[~foreground] = -9999
            labels = dir_to_class(centers, 16).astype(np.float32)
            train_indices, validation_indices = partition(positions, binary, shape, val_ratio)
            if not len(train_indices):
                raise ValueError("No training samples outside the validation region")
            inputs = (expression.shape[1], expression.shape[2])
            model = create_transformer_classifier(16, inputs, (inputs[0], 2), inputs[0], 64, 1,
                                                  [128, 64], 8, [1024, 256])
            model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
                          loss={"pos_out": masked_categorical_cross_entropy,
                                "cat_out": tf.keras.losses.BinaryCrossentropy(from_logits=False)},
                          metrics={"pos_out": tf.keras.metrics.CategoricalAccuracy(name="accuracy")},
                          steps_per_execution=100)
            training_data = None
            if input_mode == "arrays":
                def selected_arrays(indices):
                    return ([expression[indices].astype(np.float32),
                             np.asarray(positions[indices], dtype=np.int32) - np.asarray(positions[indices, :1], dtype=np.int32)],
                            [labels[indices], binary[indices]])
                training = selected_arrays(train_indices)
                validation = selected_arrays(validation_indices) if len(validation_indices) else None
                if hasattr(expression._mmap, "madvise"):
                    expression._mmap.madvise(mmap.MADV_DONTNEED)
            elif input_mode == "tensor":
                training_data = TensorTrainingData(store, labels, binary)
                training = training_data.dataset(train_indices, shuffle=True, seed=seed)
                validation = training_data.dataset(validation_indices) if len(validation_indices) else None
            elif input_mode == "stream":
                training = dataset(store, train_indices, labels, binary, shuffle=True, seed=seed)
                validation = dataset(store, validation_indices, labels, binary) if len(validation_indices) else None
            else:
                raise ValueError("input_mode must be arrays, tensor, or stream")
            checkpoint = str(output_dir / "ckpt/ckpt")
            monitor = "val_pos_out_accuracy" if len(validation_indices) >= 100 else "pos_out_accuracy"

            class MemoryLog(tf.keras.callbacks.Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    self.epoch_started = time.time()

                def on_epoch_end(self, epoch, logs=None):
                    epoch_seconds = time.time()-self.epoch_started
                    if logs is not None:
                        logs["epoch_seconds"] = epoch_seconds
                    record = dict(epoch=epoch+1, epoch_seconds=epoch_seconds,
                                  elapsed_seconds=time.time()-started, **memory_record(tf))
                    (output_dir / "training_memory.json").write_text(json.dumps(record, indent=2) + "\n")

            config = dict(epochs=epochs, batch_size=10, inference_batch_size=inference_batch_size,
                          seed=seed, val_ratio=val_ratio, training_samples=len(train_indices),
                          validation_foreground_samples=len(validation_indices),
                          training_array_shape=list(expression.shape), monitor=monitor,
                          inference_enabled=inference,
                          data_dir=str(data_dir), gpu_memory_limit_mib=gpu_memory_mib,
                          training_input_mode=input_mode,
                          array_loading={"arrays": "selected training/validation float32 arrays only; no full float32 source copy",
                                         "tensor": "single original-dtype CPU tensor; graph gathers cast batches",
                                         "stream": "original-dtype mmap; Python generator casts batches"}[input_mode],
                          test_loading="opened after training; inference converts one batch at a time",
                          shuffle={"arrays": "original Keras NumPy adapter shuffle",
                                   "tensor": "seeded tf.data permutation",
                                   "stream": "seeded NumPy per-epoch permutation"}[input_mode])
            (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            fit_args = dict(epochs=epochs, validation_data=validation, verbose=2,
                            callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor=monitor,
                                       mode="max", save_best_only=True, save_weights_only=True), MemoryLog(),
                                       tf.keras.callbacks.CSVLogger(str(output_dir / "training_history.csv"))])
            if input_mode == "arrays":
                model.fit(x=training[0], y=training[1], batch_size=10, shuffle=True, **fit_args)
            else:
                model.fit(training, **fit_args)
            assert "x_test" not in store.opened, "Inference data loaded during training"
            model.load_weights(checkpoint)
            del training, validation, training_data
            gc.collect()
            rows = predict_to_file(model, store, output_dir / f"results/spot_prediction_{suffix}.txt",
                                   batch_size=inference_batch_size) if inference else 0
            result = dict(completed=True, inference_completed=inference, prediction_rows=rows,
                          elapsed_seconds=time.time()-started,
                          **memory_record(tf))
            (output_dir / "completed.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        finally:
            store.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--cache-root")
    p.add_argument("--gpu-memory-mib", type=int, default=16384)
    p.add_argument("--inference-batch-size", type=int, default=128)
    p.add_argument("--train-only", action="store_true", help="Benchmark training without inference")
    p.add_argument("--input-mode", choices=["arrays", "tensor", "stream"], default="arrays")
    args = p.parse_args()
    print(json.dumps(train(args.data_dir, args.output_dir, epochs=args.epochs, cache_root=args.cache_root,
                          gpu_memory_mib=args.gpu_memory_mib, inference_batch_size=args.inference_batch_size,
                          inference=not args.train_only, input_mode=args.input_mode), indent=2))
