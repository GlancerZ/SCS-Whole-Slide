"""Train one width-scaled PyTorch SCS model across an entire prepared slide."""

import argparse
import contextlib
import fcntl
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from .shared_data import SharedBatches, load_schema, save_json
from .torch_data import PlannedBatchDataset, whole_slide_batches
from .torch_model import ModelConfig, SCSClassifier, scs_loss


@contextlib.contextmanager
def run_lock(root, run_name):
    with (Path(root) / f".{run_name}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def require_torch():
    import torch

    version = tuple(
        int(part) for part in torch.__version__.split("+")[0].split(".")[:2]
    )
    if version < (2, 13):
        raise RuntimeError("PyTorch >= 2.13 is required for official torch.optim.Muon")
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        raise RuntimeError("This PyTorch build does not provide SDPA")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("This PyTorch build does not provide torch.optim.Muon")
    return torch


def configure_runtime(seed, amp):
    torch = require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for whole-slide training")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if amp == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bfloat16")
        dtype = torch.bfloat16
    elif amp == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("amp must be bf16 or fp16")
    return torch, torch.device("cuda", 0), dtype


def validate_preparation(root, schema):
    root = Path(root)
    for tile_id in schema["tile_ids"]:
        marker = json.loads((root / "tiles" / tile_id / "prepared.json").read_text())
        if marker["fingerprint"] != schema["fingerprint"]:
            raise ValueError(f"incomplete/incompatible tile preparation: {tile_id}")


def make_model_and_optimizers(schema, scale, muon_lr, adamw_lr, weight_decay, device):
    torch = require_torch()
    config = ModelConfig(
        n_genes=len(schema["gene_indices"]),
        n_neighbors=schema["n_neighbors"],
        scale=scale,
    )
    model = SCSClassifier(config).to(device)
    muon_parameters, adamw_parameters = model.optimizer_parameter_groups()
    muon = torch.optim.Muon(
        muon_parameters,
        lr=muon_lr,
        momentum=0.95,
        weight_decay=weight_decay,
        adjust_lr_fn="original",
    )
    adamw = torch.optim.AdamW(
        adamw_parameters,
        lr=adamw_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    return config, model, muon, adamw


def move_batch(batch, device, amp_dtype):
    if len(batch) == 2 and len(batch[0]) == 2 and len(batch[1]) == 2:
        batch = (*batch[0], *batch[1])
    expression, positions, directions, foreground = batch
    # Direct autotuning reads the compact NumPy representation without a
    # DataLoader, whereas training receives tensors from the collator.
    torch = require_torch()
    batch = tuple(
        torch.as_tensor(value)
        for value in (expression, positions, directions, foreground)
    )
    expression, positions, directions, foreground = batch
    if directions.ndim == 2:
        directions = directions.argmax(dim=-1)
    return (
        expression.to(device=device, dtype=amp_dtype, non_blocking=True),
        positions.to(device=device, non_blocking=True),
        directions.to(device=device, non_blocking=True),
        foreground.to(device=device, non_blocking=True),
    )


def optimizer_step(torch, loss, muon, adamw, scaler):
    muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(muon)
    scaler.step(adamw)
    scaler.update()


def run_epoch(
    model,
    loader,
    device,
    amp_dtype,
    muon=None,
    adamw=None,
    scaler=None,
    max_steps=None,
    log_every=25,
):
    torch = require_torch()
    training = muon is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "direction_loss": 0.0,
        "foreground_loss": 0.0,
        "examples": 0,
        "foreground_examples": 0,
        "direction_correct": 0,
        "binary_correct": 0,
        "steps": 0,
    }
    started = time.perf_counter()
    grad_context = contextlib.nullcontext() if training else torch.no_grad()
    with grad_context:
        for step, cpu_batch in enumerate(loader, 1):
            expression, positions, directions, foreground = move_batch(
                cpu_batch, device, amp_dtype
            )
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                direction_logits, foreground_logits = model(expression, positions)
                loss, direction_loss, foreground_loss = scs_loss(
                    direction_logits, foreground_logits, directions, foreground
                )
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss at step {step}")
            if training:
                optimizer_step(torch, loss, muon, adamw, scaler)
            batch_size = len(foreground)
            positives = foreground > 0.5
            totals["loss"] += float(loss.detach()) * batch_size
            totals["direction_loss"] += float(direction_loss.detach()) * batch_size
            totals["foreground_loss"] += float(foreground_loss.detach()) * batch_size
            totals["examples"] += batch_size
            totals["foreground_examples"] += int(positives.sum())
            totals["direction_correct"] += int(
                ((direction_logits.argmax(-1) == directions) & positives).sum()
            )
            totals["binary_correct"] += int(
                ((foreground_logits >= 0) == positives).sum()
            )
            totals["steps"] = step
            if training and log_every and step % log_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "step": step,
                            "examples": totals["examples"],
                            "examples_per_second": totals["examples"] / elapsed,
                            "loss": totals["loss"] / totals["examples"],
                        }
                    ),
                    flush=True,
                )
            if max_steps and step >= max_steps:
                break
    elapsed = time.perf_counter() - started
    examples = totals["examples"]
    foreground_examples = totals["foreground_examples"]
    return {
        "loss": totals["loss"] / examples,
        "direction_loss": totals["direction_loss"] / examples,
        "foreground_loss": totals["foreground_loss"] / examples,
        "foreground_accuracy": (
            totals["direction_correct"] / foreground_examples
            if foreground_examples
            else math.nan
        ),
        "binary_accuracy": totals["binary_correct"] / examples,
        "examples": examples,
        "foreground_examples": foreground_examples,
        "steps": totals["steps"],
        "seconds": elapsed,
        "examples_per_second": examples / elapsed,
    }


def atomic_torch_save(torch, value, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(destination)


def checkpoint_state(model, muon, adamw, epoch, best, config):
    torch = require_torch()
    return {
        "model": model.state_dict(),
        "muon": muon.state_dict(),
        "adamw": adamw.state_dict(),
        "epoch": epoch,
        "best_validation_accuracy": best,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def restore_checkpoint(torch, checkpoint, model, muon, adamw, expected_config):
    if checkpoint["config"] != expected_config:
        raise ValueError("resume configuration does not match checkpoint")
    model.load_state_dict(checkpoint["model"])
    muon.load_state_dict(checkpoint["muon"])
    adamw.load_state_dict(checkpoint["adamw"])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    return int(checkpoint["epoch"]), float(checkpoint["best_validation_accuracy"])


def train(
    root,
    epochs=1,
    batch_size=256,
    per_class_cap=4096,
    scale=4,
    amp="bf16",
    workers=4,
    prefetch=2,
    muon_lr=0.02,
    adamw_lr=3e-4,
    weight_decay=0.01,
    run_name=None,
    resume=False,
    max_steps=None,
):
    root = Path(root).resolve()
    schema = load_schema(root)
    validate_preparation(root, schema)
    run_name = run_name or f"torch_scale{scale}"
    output = root / run_name
    output.mkdir(exist_ok=True)
    with run_lock(root, run_name):
        torch, device, amp_dtype = configure_runtime(schema["seed"], amp)
        model_config, model, muon, adamw = make_model_and_optimizers(
            schema, scale, muon_lr, adamw_lr, weight_decay, device
        )
        config = {
            "schema_fingerprint": schema["fingerprint"],
            "model": model_config.to_dict(),
            "framework": f"pytorch {torch.__version__}",
            "optimizer": "Muon + auxiliary AdamW",
            "muon_lr": muon_lr,
            "adamw_lr": adamw_lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "per_class_cap": per_class_cap,
            "amp": amp,
            "attention": "torch.nn.functional.scaled_dot_product_attention",
            "workers": workers,
            "prefetch": prefetch,
        }
        config_path = output / "config.json"
        latest_path = output / "latest.pt"
        best_path = output / "best.pt"
        start_epoch, best_accuracy = 0, -math.inf
        if resume:
            if not latest_path.exists():
                raise FileNotFoundError("resume requested but latest.pt is missing")
            stored = torch.load(latest_path, map_location=device, weights_only=False)
            start_epoch, best_accuracy = restore_checkpoint(
                torch, stored, model, muon, adamw, config
            )
        elif config_path.exists() or latest_path.exists():
            raise FileExistsError("run output exists; use --resume or a new --run-name")
        save_json(config_path, config)
        save_json(
            output / "training_state.json",
            {
                "completed": False,
                "epoch": start_epoch,
                "target_epochs": epochs,
                "schema_fingerprint": schema["fingerprint"],
            },
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
        started = time.time()
        for epoch in range(start_epoch, epochs):
            _, train_loader = whole_slide_batches(
                root,
                "train",
                batch_size,
                per_class_cap,
                schema["seed"],
                epoch,
                workers,
                prefetch,
            )
            training = run_epoch(
                model,
                train_loader,
                device,
                amp_dtype,
                muon,
                adamw,
                scaler,
                max_steps=max_steps,
            )
            if max_steps:
                result = {
                    "completed": False,
                    "trial": True,
                    "epoch": epoch + 1,
                    "training": training,
                    "parameter_count": model.parameter_count,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                }
                save_json(output / "trial.json", result)
                return result
            _, validation_loader = whole_slide_batches(
                root,
                "validation",
                batch_size,
                per_class_cap,
                schema["seed"],
                0,
                workers,
                prefetch,
            )
            validation = run_epoch(model, validation_loader, device, amp_dtype)
            record = {
                "epoch": epoch + 1,
                "training": training,
                "validation": validation,
                "parameter_count": model.parameter_count,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "schema_fingerprint": schema["fingerprint"],
            }
            save_json(output / f"epoch_{epoch + 1:05d}.json", record)
            state = checkpoint_state(
                model, muon, adamw, epoch + 1, best_accuracy, config
            )
            if validation["foreground_accuracy"] > best_accuracy:
                best_accuracy = validation["foreground_accuracy"]
                state["best_validation_accuracy"] = best_accuracy
                atomic_torch_save(torch, state, best_path)
            atomic_torch_save(torch, state, latest_path)
            save_json(
                output / "training_state.json",
                {
                    "completed": False,
                    "epoch": epoch + 1,
                    "target_epochs": epochs,
                    "best_validation_accuracy": best_accuracy,
                    "schema_fingerprint": schema["fingerprint"],
                },
            )
            print(json.dumps(record), flush=True)
        completed = {
            "completed": True,
            "epochs": epochs,
            "best_validation_accuracy": best_accuracy,
            "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path),
            "elapsed_seconds": time.time() - started,
            "parameter_count": model.parameter_count,
            "schema_fingerprint": schema["fingerprint"],
        }
        save_json(output / "completed.json", completed)
        save_json(
            output / "training_state.json",
            {
                "completed": True,
                "epoch": epochs,
                "target_epochs": epochs,
                "best_validation_accuracy": best_accuracy,
                "schema_fingerprint": schema["fingerprint"],
            },
        )
        return completed


def real_merged_batch(batches, candidate):
    dataset = PlannedBatchDataset(
        batches, epoch=0, batch_size=candidate, merge_tiles=True
    )
    batch = dataset[0]
    tile_ids = [tile_id for tile_id, _ in dataset.plan[0]]
    if len(batch[-1]) != candidate:
        raise ValueError(f"could only construct a real batch of {len(batch[-1])}")
    return batch, tile_ids


def try_batch(model, muon, adamw, cpu_batch, device, amp_dtype, repeats):
    torch = require_torch()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = move_batch(cpu_batch, device, amp_dtype)
    started = time.perf_counter()
    for _ in range(repeats):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(batch[0], batch[1])
            loss, _, _ = scs_loss(outputs[0], outputs[1], batch[2], batch[3])
        optimizer_step(torch, loss, muon, adamw, scaler)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return {
        "loss": float(loss.detach()),
        "seconds": seconds,
        "examples_per_second": len(batch[3]) * repeats / seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def autotune(
    root,
    scale=4,
    amp="bf16",
    per_class_cap=4096,
    start_batch=64,
    max_batch=4096,
    repeats=2,
    muon_lr=0.02,
    adamw_lr=3e-4,
    weight_decay=0.01,
    output=None,
    granularity=128,
):
    root = Path(root).resolve()
    schema = load_schema(root)
    validate_preparation(root, schema)
    torch, device, amp_dtype = configure_runtime(schema["seed"], amp)
    model_config, model, muon, adamw = make_model_and_optimizers(
        schema, scale, muon_lr, adamw_lr, weight_decay, device
    )
    source_batches = SharedBatches(
        root, "train", max_batch, per_class_cap, schema["seed"]
    )
    passed, failed, trials = 0, None, []

    def attempt(candidate):
        nonlocal passed, failed
        try:
            cpu_batch, tile_ids = real_merged_batch(source_batches, candidate)
            metrics = try_batch(
                model, muon, adamw, cpu_batch, device, amp_dtype, repeats
            )
            trial = dict(
                batch_size=candidate, tile_ids=tile_ids, passed=True, **metrics
            )
            passed = max(passed, candidate)
        except torch.OutOfMemoryError as error:
            failed = candidate if failed is None else min(failed, candidate)
            trial = {
                "batch_size": candidate,
                "passed": False,
                "error": type(error).__name__,
            }
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(json.dumps(trial), flush=True)
        trials.append(trial)
        return trial["passed"]

    candidate = start_batch
    while candidate <= max_batch:
        if attempt(candidate):
            candidate *= 2
        else:
            break
    if not passed:
        raise RuntimeError("no candidate batch size completed")
    # Refine the first pass/fail interval. Keeping the result aligned makes the
    # selected value practical for tensor cores and reproducible in the CLI.
    if failed is not None:
        while failed - passed > granularity:
            midpoint = ((passed + failed) // (2 * granularity)) * granularity
            if midpoint <= passed:
                break
            attempt(midpoint)
    result = {
        "scale": scale,
        "amp": amp,
        "selected_batch_size": passed,
        "first_failed_batch_size": failed,
        "search_ceiling": max_batch,
        "granularity": granularity,
        "parameter_count": model.parameter_count,
        "model": model_config.to_dict(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "trials": trials,
    }
    destination = Path(output) if output else root / f"autotune_scale{scale}.json"
    save_json(destination, result)
    return result


def inspect(root, scales):
    schema = load_schema(root)
    torch = require_torch()
    result = {}
    for scale in scales:
        model = SCSClassifier(
            ModelConfig(
                n_genes=len(schema["gene_indices"]),
                n_neighbors=schema["n_neighbors"],
                scale=scale,
            )
        )
        result[str(scale)] = {
            "parameters": model.parameter_count,
            "config": model.config.to_dict(),
        }
    result["torch"] = torch.__version__
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "autotune", "train"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--scale", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--amp", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--start-batch", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-granularity", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-class-cap", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adamw-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--run-name")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.action == "inspect":
        print(json.dumps(inspect(args.root, (1, 2, 4)), indent=2))
    elif args.action == "autotune":
        print(
            json.dumps(
                autotune(
                    args.root,
                    args.scale,
                    args.amp,
                    args.per_class_cap,
                    args.start_batch,
                    args.max_batch,
                    args.repeats,
                    args.muon_lr,
                    args.adamw_lr,
                    args.weight_decay,
                    args.output,
                    args.batch_granularity,
                ),
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                train(
                    args.root,
                    args.epochs,
                    args.batch_size,
                    args.per_class_cap,
                    args.scale,
                    args.amp,
                    args.workers,
                    args.prefetch,
                    args.muon_lr,
                    args.adamw_lr,
                    args.weight_decay,
                    args.run_name,
                    args.resume,
                    args.max_steps,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
