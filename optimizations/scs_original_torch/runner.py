"""Train/predict on the original per-tile SCS NPZ files, with bounded RAM."""
import argparse
import contextlib
import csv
import json
import os
import resource
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np
import torch

from optimizations.scs_streaming.data import batch_inputs, index_batches, partition
from .data import ArrayStore
from .model import OriginalSCS, ReferenceAdamW, losses


def direction_labels(centers, positions):
    # Equivalent to upstream atan(x/y) and quadrant corrections; cast before
    # subtraction so unsigned source coordinates cannot wrap around.
    centers = np.asarray(centers, dtype=np.float64)
    foreground = centers[:, 0] != -1
    delta = centers - np.asarray(positions[:, 0], dtype=np.float64)
    angle = np.mod(np.arctan2(delta[:, 0], delta[:, 1]), 2 * np.pi)
    classes = (angle / (2 * np.pi / 16)).astype(np.int64) % 16
    labels = np.eye(16, dtype=np.float32)[classes]
    labels[~foreground] = 0
    return labels


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".pt.partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def memory_record(device):
    """Process RSS high-water mark plus PyTorch allocator statistics (GiB)."""
    result = dict(process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    if device.type == "cuda":
        result.update(gpu_allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                      gpu_reserved_gib=torch.cuda.memory_reserved(device) / 2**30,
                      gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                      gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
    return result


def amp_context(device, amp):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp == "bf16" else contextlib.nullcontext())


def run_epoch(model, store, indices, labels, binary, batch_size, device,
              amp="none", optimizer=None, seed=20260905, gpu_cache=None, compiled_forward=None):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    count = 0
    with torch.set_grad_enabled(training):
        for rows in index_batches(indices, batch_size, training, np.random.RandomState(seed)):
            if compiled_forward is not None:
                torch.compiler.cudagraph_mark_step_begin()
            if gpu_cache is None:
                expression, positions = batch_inputs(store, "train", rows)
                x, p, y, b = [torch.as_tensor(a, device=device) for a in
                              (expression, positions, labels[rows], binary[rows])]
            else:
                x, p, y, b = gpu_cache.batch(rows)
            with amp_context(device, amp):
                forward = model if compiled_forward is None else compiled_forward
                direction, foreground = forward(x, p)
                total, pos_loss, cat_loss = losses(direction, foreground, y, b)
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
            # Deliberately match original CategoricalAccuracy: background
            # all-zero labels have argmax 0 and are included in training accuracy.
            correct = (direction.argmax(-1) == y.argmax(-1)).sum()
            totals[:3] += torch.stack([total.detach(), pos_loss.detach(), cat_loss.detach()]).double() * len(rows)
            totals[3] += correct
            count += len(rows)
    if not count:
        return {}
    values = (totals / count).cpu().tolist()
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite epoch metrics; checkpoint not committed")
    return dict(zip(("loss", "pos_out_loss", "cat_out_loss", "pos_out_accuracy"), values))


@torch.inference_mode()
def predict_to_file(model, store, path, batch_size, device, amp="none"):
    model.eval()
    temporary = path.with_suffix(path.suffix + ".partial")
    count = 0
    with temporary.open("x", buffering=1024 * 1024) as handle:
        for subset in ("train", "test"):
            n = store.header("x_" + subset)[0][0]
            if not n:
                continue
            absolute = store.array("x_" + subset + "_pos")
            for start in range(0, n, batch_size):
                rows = slice(start, min(start + batch_size, n))
                x, p = batch_inputs(store, subset, rows)
                with amp_context(device, amp):
                    logits, foreground = model(torch.as_tensor(x, device=device),
                                                torch.as_tensor(p, device=device))
                logits = logits.float().cpu().numpy()
                foreground = foreground.float().sigmoid().cpu().numpy()
                if not (np.isfinite(logits).all() and np.isfinite(foreground).all()):
                    raise ValueError("Nonfinite predictions")
                coords = absolute[rows, 0]
                handle.writelines(f"{int(x)}\t{int(y)}\t{float(prob)}\t" +
                                  ":".join(map(str, direction)) + "\n"
                                  for (x, y), prob, direction in zip(coords, foreground, logits))
                count += len(coords)
    temporary.replace(path)
    return count


def configure_device(device, amp, gpu_memory_mib):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    if amp not in ("none", "bf16") or (amp != "none" and device.type != "cuda"):
        raise ValueError("bf16 is an explicit CUDA-only option; default is float32")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu only for tests")
        if amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support bf16")
        total = torch.cuda.get_device_properties(device).total_memory
        if not 0 < gpu_memory_mib * 2**20 <= total:
            raise ValueError("Invalid GPU allocator memory limit")
        torch.cuda.set_per_process_memory_fraction(gpu_memory_mib * 2**20 / total, device)
    return device


def train(data_dir, output_dir, epochs=100, batch_size=10, inference_batch_size=128,
          val_ratio=0.0625, suffix="0:0:0:0", device="cuda", amp="none",
          cache_root=None, seed=20260905, inference=True, gpu_memory_mib=16384,
          optimizer_mode="reference", input_mode="stream", matmul_precision="highest",
          compile_model=False):
    if epochs < 1 or min(batch_size, inference_batch_size) < 1:
        raise ValueError("Epochs and batch sizes must be positive")
    device = configure_device(device, amp, gpu_memory_mib)
    if optimizer_mode not in ("reference", "foreach") or input_mode not in ("stream", "gpu_cache"):
        raise ValueError("Invalid performance mode")
    if input_mode == "gpu_cache" and device.type != "cuda":
        raise ValueError("gpu_cache requires CUDA")
    if compile_model and device.type != "cuda":
        raise ValueError("The experimental compiled path requires CUDA")
    if matmul_precision not in ("highest", "high"):
        raise ValueError("Invalid matmul precision")
    torch.set_float32_matmul_precision(matmul_precision)
    torch.manual_seed(seed)
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir == data_dir or output_dir in data_dir.parents:
        raise ValueError("Output must not overwrite the input/run directory")
    # Exclusive directory creation prevents two trainers overwriting one another.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "results").mkdir()
    with h5py.File(data_dir / f"spots{suffix}.h5ad") as f:
        shape = tuple(f["X"].attrs["shape"]) if isinstance(f["X"], h5py.Group) else f["X"].shape
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="scs-original-torch-",
                                     dir=cache_root or os.environ.get("SLURM_TMPDIR")) as cache:
        store = ArrayStore(data_dir, cache, suffix)
        try:
            dimensions = store.header("x_train")[0]
            positions = store.array("x_train_pos")
            labels = direction_labels(store.array("y_train"), positions)
            binary = np.asarray(store.array("y_binary_train"), dtype=np.float32)
            if len(labels) != dimensions[0] or binary.shape != (dimensions[0],):
                raise ValueError("Mismatched source rows")
            train_rows, val_rows = partition(positions, binary, shape, val_ratio)
            if not len(train_rows):
                raise ValueError("No training points")
            model = OriginalSCS(dimensions[2], dimensions[1]).to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            from .performance import ForeachReferenceAdamW, GPUTrainingCache
            optimizer_class = ReferenceAdamW if optimizer_mode == "reference" else ForeachReferenceAdamW
            optimizer = optimizer_class(model.parameters())
            training_forward = (torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)
                                if compile_model else None)
            gpu_cache = GPUTrainingCache(store, labels, binary, device) if input_mode == "gpu_cache" else None
            monitor = "val_pos_out_accuracy" if len(val_rows) >= 100 else "pos_out_accuracy"
            config = dict(implementation="original_scs_sdpa_v1", data_dir=str(data_dir),
                          suffix=suffix, epochs=epochs, batch_size=batch_size,
                          inference_batch_size=inference_batch_size, seed=seed, amp=amp,
                          n_genes=dimensions[2], n_neighbors=dimensions[1], layers=8,
                          width=64, heads=1, monitor=monitor, val_ratio=val_ratio,
                          training_samples=len(train_rows), validation_foreground_samples=len(val_rows),
                          optimizer="TFA-compatible dense AdamW", lr=0.001,
                          weight_decay_per_step=0.0001, betas=[0.9, 0.999], epsilon=1e-7,
                          torch_version=str(torch.__version__), early_stopping=False,
                          gpu_memory_mib=gpu_memory_mib, optimizer_mode=optimizer_mode,
                          input_mode=input_mode, matmul_precision=matmul_precision,
                          compile_model=compile_model)
            save_json(output_dir / "config.json", config)
            training_started = time.monotonic()
            setup_seconds = training_started - started
            best = -float("inf")
            best_epoch = None
            with (output_dir / "training_history.csv").open("x", newline="") as history:
                writer = None
                for epoch in range(epochs):
                    tick = time.monotonic()
                    metrics = run_epoch(model, store, train_rows, labels, binary, batch_size,
                                        device, amp, optimizer, seed + epoch, gpu_cache, training_forward)
                    metrics.update({"val_" + k: v for k, v in run_epoch(
                        model, store, val_rows, labels, binary, batch_size, device, amp,
                        gpu_cache=gpu_cache).items()})
                    record = dict(epoch=epoch, **metrics, epoch_seconds=time.monotonic() - tick,
                                  **memory_record(device))
                    payload = dict(config=config, epoch=epoch + 1, model=model.state_dict(),
                                   optimizer=optimizer.state_dict(), metrics=metrics)
                    if metrics[monitor] > best:
                        best, best_epoch = metrics[monitor], epoch + 1
                        save_checkpoint(output_dir / "best.pt", payload)
                    save_checkpoint(output_dir / "latest.pt", payload)
                    if writer is None:
                        writer = csv.DictWriter(history, fieldnames=list(record))
                        writer.writeheader()
                    writer.writerow(record)
                    history.flush()
                    save_json(output_dir / "training_state.json", dict(completed=False,
                              epoch=epoch + 1, target_epochs=epochs, best_epoch=best_epoch,
                              best_score=best, **metrics))
                    save_json(output_dir / "training_memory.json", dict(phase="training", **record))
                    print(json.dumps(record), flush=True)
            assert "x_test" not in store.opened
            training_seconds = time.monotonic() - training_started
            del gpu_cache, optimizer, payload, training_forward
            model.zero_grad(set_to_none=True)
            checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            inference_started = time.monotonic()
            count = predict_to_file(model, store, output_dir / "results" / f"spot_prediction_{suffix}.txt",
                                    inference_batch_size, device, amp) if inference else 0
            result = dict(completed=True, epochs=epochs, best_epoch=best_epoch, best_score=best,
                          inference_completed=inference, prediction_rows=count,
                          setup_seconds=setup_seconds, training_seconds=training_seconds,
                          inference_seconds=time.monotonic() - inference_started,
                          elapsed_seconds=time.monotonic() - started, **memory_record(device))
            save_json(output_dir / "completed.json", result)
            save_json(output_dir / "training_state.json", result)
            return result
        finally:
            store.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True, help="Fresh, separate output directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--inference-batch-size", type=int, default=128)
    p.add_argument("--suffix", default="0:0:0:0")
    p.add_argument("--val-ratio", type=float, default=0.0625)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--amp", choices=["none", "bf16"], default="none")
    p.add_argument("--cache-root")
    p.add_argument("--gpu-memory-mib", type=int, default=16384)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--optimizer-mode", choices=["reference", "foreach"], default="reference")
    p.add_argument("--input-mode", choices=["stream", "gpu_cache"], default="stream")
    p.add_argument("--matmul-precision", choices=["highest", "high"], default="highest")
    p.add_argument("--compile-model", action="store_true", help="Experimental compiled training forward/backward")
    args = vars(p.parse_args())
    torch.set_num_threads(args.pop("threads"))
    args["inference"] = not args.pop("train_only")
    print(json.dumps(train(**args), indent=2))


if __name__ == "__main__":
    main()
