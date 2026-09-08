"""One resumable SCS checkpoint shared by all tiles; sparse minibatch loading."""
import argparse
import contextlib
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .shared_data import SharedBatches, TileStore, load_schema, save_json, fingerprint

ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def run_lock(root):
    with (Path(root) / ".shared-model.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def ready_model(root, schema):
    directory = Path(root) / "model"
    completed = json.loads((directory / "completed.json").read_text())
    state = json.loads((directory / "training_state.json").read_text())
    if not state["completed"] or completed["epochs"] != state["target_epochs"]:
        raise ValueError("Training target has not finished; refusing a stale completed model")
    if any(value["fingerprint"] != schema["fingerprint"] for value in (completed, state)):
        raise ValueError("Model and inference gene schemas differ")
    return completed


def setup_model(schema, gpu_memory_mib):
    import tensorflow as tf
    import tensorflow_addons as tfa
    from .model_reference import create_transformer_classifier, masked_categorical_cross_entropy
    tf.keras.utils.set_random_seed(schema["seed"])
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if len(gpus) != 1:
            raise ValueError("Expected one allocated GPU")
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(
            memory_limit=gpu_memory_mib)])
    n, g = schema["n_neighbors"], len(schema["gene_indices"])
    model = create_transformer_classifier(16, (n, g), (n, 2), n, 64, 1, [128, 64], 8, [1024, 256])

    class ForegroundAccuracy(tf.keras.metrics.CategoricalAccuracy):
        def update_state(self, y_true, y_pred, sample_weight=None):
            weight = tf.reduce_sum(y_true, axis=-1)
            if sample_weight is not None:
                weight *= sample_weight
            return super().update_state(y_true, y_pred, sample_weight=weight)

    model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
                  loss={"pos_out": masked_categorical_cross_entropy,
                        "cat_out": tf.keras.losses.BinaryCrossentropy()},
                  metrics={"pos_out": ForegroundAccuracy(name="foreground_accuracy"),
                           "cat_out": [tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                                       tf.keras.metrics.AUC(name="auc")]},
                  steps_per_execution=100)
    return tf, model


def train(root, epochs=100, batch_size=10, per_class_cap=4096, gpu_memory_mib=16384, resume=False):
    from .train import memory_record
    root = Path(root).resolve()
    schema = load_schema(root)
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    directory = root / "model"
    directory.mkdir(exist_ok=True)
    with run_lock(root):
        config = dict(fingerprint=schema["fingerprint"], batch_size=batch_size, per_class_cap=per_class_cap,
                      seed=schema["seed"], sampling="resample per tile/class each epoch; fixed validation",
                      model="SCS 8-layer reference; raw counts; one optimizer across training tiles",
                      monitor="val_pos_out_foreground_accuracy")
        config_path = directory / "config.json"
        if config_path.exists():
            if not resume:
                raise FileExistsError("Model output exists; explicitly use --resume")
            if json.loads(config_path.read_text()) != config:
                raise ValueError("Resume configuration does not match existing shared model")
        elif resume:
            raise FileNotFoundError("No shared training configuration to resume")
        training = SharedBatches(root, "train", batch_size, per_class_cap, schema["seed"])
        validation = SharedBatches(root, "validation", batch_size, per_class_cap, schema["seed"])
        # Explicitly require preparation of every tile, including validation and
        # empty markers, before training can be called a whole-slide run.
        for tid in schema["tile_ids"]:
            meta = json.loads((root / "tiles" / tid / "prepared.json").read_text())
            if meta["fingerprint"] != schema["fingerprint"]:
                raise ValueError("Incomplete/incompatible tile preparation")
        tf, model = setup_model(schema, gpu_memory_mib)
        save_json(config_path, config)
        save_json(directory / "sampling.json", dict(train=training.counts, validation=validation.counts,
                  train_samples_per_epoch=training.samples, validation_samples=validation.samples,
                  train_steps_per_epoch=training.steps, validation_steps=validation.steps))
        epoch_state = tf.Variable(0, dtype=tf.int64, trainable=False)
        best_state = tf.Variable(-np.inf, dtype=tf.float32, trainable=False)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer, epoch=epoch_state, best=best_state)
        latest = tf.train.CheckpointManager(checkpoint, str(directory / "latest"), max_to_keep=2)
        best = tf.train.CheckpointManager(checkpoint, str(directory / "best"), max_to_keep=1)
        if resume:
            if not latest.latest_checkpoint:
                if (directory / "completed.json").exists() or list((directory / "epochs").glob("*.json")):
                    raise FileNotFoundError("Previously committed training checkpoint is missing")
                print("No epoch was committed; restarting epoch 1 with the same configuration", flush=True)
            else:
                checkpoint.restore(latest.latest_checkpoint).assert_existing_objects_matched()
        if epochs < int(epoch_state.numpy()):
            raise ValueError("Requested epoch target precedes the saved checkpoint")
        save_json(directory / "training_state.json", dict(completed=False, target_epochs=epochs,
                  resumed_from_epoch=int(epoch_state.numpy()), fingerprint=schema["fingerprint"]))
        (directory / "epochs").mkdir(exist_ok=True)
        start = time.time()
        for epoch in range(int(epoch_state.numpy()), epochs):
            epoch_start = time.time()
            # Deterministic epoch-specific data sampling; optimizer is retained.
            # Dropout RNG is re-seeded by epoch for restartable epoch boundaries.
            tf.random.set_seed(schema["seed"] + epoch)
            history = model.fit(training.dataset(epoch), validation_data=validation.dataset(),
                                initial_epoch=epoch, epochs=epoch+1, verbose=2)
            metrics = {key: float(value[-1]) for key, value in history.history.items()}
            if not all(np.isfinite(v) for v in metrics.values()):
                raise ValueError("Nonfinite training/validation metric; checkpoint not committed")
            score = metrics[config["monitor"]]
            epoch_state.assign(epoch+1)
            if score > float(best_state.numpy()):
                best_state.assign(score)
                best.save(checkpoint_number=epoch+1)
            latest.save(checkpoint_number=epoch+1)
            record = dict(epoch=epoch+1, metrics=metrics, epoch_seconds=time.time()-epoch_start,
                          fingerprint=schema["fingerprint"], **memory_record(tf))
            save_json(directory / "epochs" / f"{epoch+1:05d}.json", record)
            save_json(directory / "training_memory.json", record)
            print(json.dumps(record), flush=True)
        result = dict(completed=True, epochs=int(epoch_state.numpy()), best_validation_accuracy=float(best_state.numpy()),
                      fingerprint=schema["fingerprint"], partial_scope=schema["partial_scope"],
                      latest_checkpoint=latest.latest_checkpoint, best_checkpoint=best.latest_checkpoint,
                      elapsed_seconds=time.time()-start, **memory_record(tf))
        save_json(directory / "completed.json", result)
        save_json(directory / "training_state.json", dict(completed=True, target_epochs=epochs,
                  fingerprint=schema["fingerprint"]))
        return result


def predict(root, tile_ids=None, batch_size=32, gpu_memory_mib=16384):
    """Same trained weights for every tile, including tiles with no nucleus seeds."""
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    root = Path(root).resolve()
    schema = load_schema(root)
    with run_lock(root):
        completed = ready_model(root, schema)
        tf, model = setup_model(schema, gpu_memory_mib)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer,
                                        epoch=tf.Variable(0, dtype=tf.int64), best=tf.Variable(0.0))
        checkpoint.restore(completed["best_checkpoint"]).expect_partial().assert_existing_objects_matched()
        provenance = fingerprint(dict(schema=schema["fingerprint"], checkpoint=completed["best_checkpoint"]))
        n, g = schema["n_neighbors"], len(schema["gene_indices"])

        @tf.function(input_signature=[tf.TensorSpec((None, n, g), tf.float32),
                                      tf.TensorSpec((None, n, 2), tf.int32)])
        def infer(x, p):
            return model([x, p], training=False)

        for tid in tile_ids or schema["tile_ids"]:
            if tid not in schema["tile_ids"]:
                raise ValueError("Inference tile is outside the shared schema")
            directory = root / "tiles" / tid
            meta = json.loads((directory / "prepared.json").read_text())
            if meta["fingerprint"] != schema["fingerprint"]:
                raise ValueError("Stale inference input")
            if meta["state"] == "empty":
                continue
            marker = directory / "prediction.json"
            if marker.exists():
                if json.loads(marker.read_text())["model_fingerprint"] != provenance:
                    raise ValueError("Tile predictions belong to different weights; use a new run")
                continue
            tile = TileStore(root, tid, schema)
            (directory / "results").mkdir(exist_ok=True)
            destination = directory / "results/spot_prediction_0:0:0:0.txt"
            temporary = destination.with_suffix(".partial")
            with temporary.open("w", buffering=1024*1024) as handle:
                for start in range(0, len(tile.neighbors), batch_size):
                    x, p, coordinates = tile.inputs(slice(start, start+batch_size))
                    logits, foreground = infer(x, p)
                    logits, foreground = logits.numpy(), foreground.numpy()
                    if not (np.all(np.isfinite(logits)) and np.all(np.isfinite(foreground))):
                        raise ValueError("Nonfinite inference output")
                    handle.writelines(f"{int(x)}\t{int(y)}\t{float(prob[0])}\t" +
                                      ":".join(map(str, logit)) + "\n"
                                      for (x, y), prob, logit in zip(coordinates, foreground, logits))
            temporary.replace(destination)
            save_json(marker, dict(rows=len(tile.neighbors), fingerprint=schema["fingerprint"],
                      model_fingerprint=provenance, checkpoint=completed["best_checkpoint"]))
            print(f"Predicted {tid}: {len(tile.neighbors)} centers, shared {completed['best_checkpoint']}", flush=True)


def postprocess(root, tile_ids=None):
    root = Path(root).resolve()
    schema = load_schema(root)
    completed = ready_model(root, schema)
    provenance = fingerprint(dict(schema=schema["fingerprint"], checkpoint=completed["best_checkpoint"]))
    sys.path.insert(0, str(ROOT / "SCS"))
    from src.postprocessing import postprocess as reference_postprocess
    for tid in tile_ids or schema["tile_ids"]:
        directory = root / "tiles" / tid
        meta = json.loads((directory / "prepared.json").read_text())
        if meta["state"] == "empty":
            continue
        prediction = json.loads((directory / "prediction.json").read_text())
        if prediction["model_fingerprint"] != provenance:
            raise ValueError("Prediction uses different shared weights")
        marker = directory / "completed.json"
        if marker.exists():
            if json.loads(marker.read_text()).get("model_fingerprint") != provenance:
                raise ValueError("Existing segmentation uses different weights")
            continue
        if not meta["supported_centers"]:
            raise ValueError(f"{tid}: RNA present but no supported neighborhood; not silently marking empty")
        previous = Path.cwd()
        try:
            os.chdir(directory)
            reference_postprocess(0, 0, 0, schema["bin_size"], 15, plot=False)
        finally:
            os.chdir(previous)
        with np.load(directory / "results/labels_0:0:0:0.npz") as arrays:
            cells = len(np.unique(arrays["cells"][arrays["cells"] > 0]))
        save_json(marker, dict(state="segmented", cell_count=cells, fingerprint=schema["fingerprint"],
                  model_fingerprint=provenance))


def merge(root):
    root = Path(root).resolve()
    schema = load_schema(root)
    if schema["partial_scope"]:
        raise ValueError("Smoke-test scope cannot be exported as a complete slide")
    completed = ready_model(root, schema)
    provenance = fingerprint(dict(schema=schema["fingerprint"], checkpoint=completed["best_checkpoint"]))
    for tid in schema["tile_ids"]:
        marker = json.loads((root / "tiles" / tid / "completed.json").read_text())
        if marker["fingerprint"] != schema["fingerprint"]:
            raise ValueError("Cannot merge incompatible feature schemas")
        if tid in schema["empty_tiles"]:
            if marker["state"] != "empty":
                raise ValueError("Unexpected state for an empty core")
        elif marker["state"] != "segmented" or marker["model_fingerprint"] != provenance:
            raise ValueError("Cannot merge different models or unresolved RNA-containing tiles")
    sys.path.insert(0, str(ROOT / "scripts"))
    import scs_whole_merge as reference
    reference.BASE = root
    reference.merge(partial=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["train", "predict", "postprocess", "merge"])
    p.add_argument("--root", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--per-class-cap", type=int, default=4096)
    p.add_argument("--inference-batch-size", type=int, default=32)
    p.add_argument("--gpu-memory-mib", type=int, default=16384)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tiles", nargs="+")
    args = p.parse_args()
    if args.action == "train":
        print(json.dumps(train(args.root, args.epochs, args.batch_size, args.per_class_cap,
                               args.gpu_memory_mib, args.resume), indent=2))
    elif args.action == "predict":
        predict(args.root, args.tiles, args.inference_batch_size, args.gpu_memory_mib)
    elif args.action == "postprocess":
        with run_lock(args.root):
            postprocess(args.root, args.tiles)
    else:
        with run_lock(args.root):
            merge(args.root)


if __name__ == "__main__":
    main()
