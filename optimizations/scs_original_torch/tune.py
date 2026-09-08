"""Real-data, exclusive-GPU throughput sweep and short profiler captures."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import h5py
import numpy as np
import torch

from optimizations.scs_streaming.data import batch_inputs, partition
from .data import ArrayStore
from .model import OriginalSCS, ReferenceAdamW, losses
from .performance import ForeachReferenceAdamW, GPUTrainingCache
from .runner import amp_context, configure_device, direction_labels, memory_record, save_json


def optimizer_check(device):
    torch.manual_seed(91)
    a = [torch.nn.Parameter(torch.randn(shape, device=device)) for shape in ((64, 64), (64,), (7,))]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    original, batched = ReferenceAdamW(a), ForeachReferenceAdamW(b)
    for step in range(30):
        for i, (left, right) in enumerate(zip(a, b)):
            gradient = None if i == 2 and step % 3 else torch.randn_like(left)
            left.grad = gradient
            right.grad = None if gradient is None else gradient.clone()
        original.step()
        batched.step()
    errors = [float((left-right).detach().abs().max()) for left, right in zip(a, b)]
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)
    return dict(passed=True, max_parameter_error=max(errors), steps=30)


def one_case(store, train_rows, labels, binary, cache, config, device, output, profile=False,
             compile_model=False):
    batch_size, precision, optimizer_mode, input_mode = config
    name = f"bs{batch_size}_{precision}_{optimizer_mode}_{input_mode}"
    if compile_model:
        name += "_compiled"
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")
    amp = "bf16" if precision == "bf16" else "none"
    order = np.random.RandomState(20260905).permutation(train_rows)
    steps = max(32, math.ceil(4096 / batch_size))
    planned_rows = np.resize(order, (steps+5)*batch_size)
    plan = [planned_rows[i*batch_size:(i+1)*batch_size] for i in range(steps+5)]
    timings = []
    warmup_times = []
    for repetition in range(1 if profile else 3):
        torch.manual_seed(20260905)
        model = OriginalSCS(store.header("x_train")[0][2]).to(device)
        optimizer = (ReferenceAdamW if optimizer_mode == "reference" else ForeachReferenceAdamW)(model.parameters())
        if compile_model:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)

        def step(rows):
            if compile_model:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.profiler.record_function("scs/data"):
                if input_mode == "gpu_cache":
                    x, p, y, b = cache.batch(rows)
                else:
                    expression, positions = batch_inputs(store, "train", rows)
                    x, p, y, b = [torch.as_tensor(a, device=device) for a in
                                  (expression, positions, labels[rows], binary[rows])]
            optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function("scs/forward"), amp_context(device, amp):
                direction, foreground = model(x, p)
                loss = losses(direction, foreground, y, b)[0]
            with torch.profiler.record_function("scs/backward"):
                loss.backward()
            with torch.profiler.record_function("scs/optimizer"):
                optimizer.step()
            return loss

        warmup_started = time.perf_counter()
        for rows in plan[:5]:
            step(rows)
        torch.cuda.synchronize()
        warmup_times.append(time.perf_counter() - warmup_started)
        torch.cuda.reset_peak_memory_stats()
        if profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                for rows in plan[5:10]:
                    step(rows)
                    profiler.step()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(output / f"{name}.trace.json"))
            events = profiler.key_averages()
            report = [dict(key=e.key, calls=e.count, cpu_total_us=e.cpu_time_total,
                           self_cpu_us=e.self_cpu_time_total,
                           device_total_us=getattr(e, "device_time_total", 0)) for e in events]
            save_json(output / f"{name}.profile.json", dict(scopes=[r for r in report if r["key"].startswith("scs/")],
                attention=[r for r in report if "scaled_dot_product" in r["key"] or "flash" in r["key"]],
                top_cpu=sorted(report, key=lambda r:r["self_cpu_us"], reverse=True)[:20],
                top_device=sorted(report, key=lambda r:r["device_total_us"], reverse=True)[:20]))
        else:
            started = time.perf_counter()
            for rows in plan[5:]:
                last_loss = step(rows)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not torch.isfinite(last_loss):
                raise ValueError("Nonfinite training loss")
            timings.append(elapsed)
        memory = memory_record(device)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    if profile:
        return
    result = dict(name=name, batch_size=batch_size, precision=precision, optimizer_mode=optimizer_mode,
                  input_mode=input_mode, steps_per_trial=steps, samples_per_trial=steps*batch_size,
                  compile_model=compile_model, warmup_seconds=warmup_times,
                  trial_seconds=timings, median_seconds=statistics.median(timings),
                  samples_per_second=steps*batch_size/statistics.median(timings), **memory)
    save_json(output / f"{name}.json", result)
    print(json.dumps(result), flush=True)
    return result


def prediction_precision_check(store, labels, binary, validation, output, device):
    path = Path(__file__).parent / "validation/st19_x11_y12_best.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = OriginalSCS(2000).to(device).eval()
    model.load_state_dict(saved["model"])
    rows = validation[:512]
    x, p = batch_inputs(store, "train", rows)
    x, p = torch.as_tensor(x, device=device), torch.as_tensor(p, device=device)
    outputs = {}
    with torch.no_grad():
        for precision in ("fp32", "tf32", "bf16"):
            torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")
            with amp_context(device, "bf16" if precision == "bf16" else "none"):
                d, b = model(x, p)
            outputs[precision] = (d.float().cpu().numpy(), b.float().sigmoid().cpu().numpy())
    reference_d, reference_b = outputs["fp32"]
    report = {}
    for precision, (d, b) in outputs.items():
        report[precision] = dict(direction_max_abs_error=float(np.max(np.abs(d-reference_d))),
            foreground_max_abs_error=float(np.max(np.abs(b-reference_b))),
            direction_agreement=float(np.mean(d.argmax(-1)==reference_d.argmax(-1))),
            foreground_threshold_01_agreement=float(np.mean((b>=0.1)==(reference_b>=0.1))),
            validation_direction_accuracy=float(np.mean(d.argmax(-1)==labels[rows].argmax(-1))))
    save_json(output / "precision_check.json", dict(samples=len(rows), results=report,
        caveat="Single tile/weights, not a biological quality or convergence guarantee"))
    print(json.dumps(report), flush=True)


def main(output):
    if os.environ.get("SLURM_JOB_ID") != "20305630":
        raise RuntimeError("This tuning run is scoped to the freed GPU allocation 20305630")
    occupants = subprocess.check_output(["nvidia-smi", "--id="+os.environ["CUDA_VISIBLE_DEVICES"],
        "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if occupants:
        raise RuntimeError("Tuning requires an idle exclusive GPU: " + occupants)
    torch.set_num_threads(4)
    device = configure_device("cuda", "none", 70000)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "optimizer_equivalence.json", optimizer_check(device))
    data_dir = Path("runs/ST19_whole_scs/tiles/x11_y12/data").resolve()
    with h5py.File(data_dir / "spots0:0:0:0.h5ad") as f:
        shape = tuple(f["X"].attrs["shape"]) if isinstance(f["X"], h5py.Group) else f["X"].shape
    with tempfile.TemporaryDirectory(prefix="scs-tuning-", dir=os.environ.get("SLURM_TMPDIR")) as temporary:
        store = ArrayStore(data_dir, temporary)
        try:
            positions = store.array("x_train_pos")
            labels = direction_labels(store.array("y_train"), positions)
            binary = np.asarray(store.array("y_binary_train"), dtype=np.float32)
            training, validation = partition(positions, binary, shape, 0.0625)
            # Decompression is intentionally outside warm-throughput timing.
            store.array("x_train")
            configs = [(10, "fp32", "reference", "stream"), (10, "fp32", "foreach", "stream")]
            configs += [(bs, "fp32", "foreach", "stream") for bs in (32,64,128,256,512)]
            configs += [(256, "tf32", "foreach", "stream")]
            configs += [(bs, "bf16", "foreach", "stream") for bs in (64,128,256,512)]
            configs += [(10, "fp32", "foreach", "gpu_cache"), (256, "fp32", "foreach", "gpu_cache"),
                        (256, "tf32", "foreach", "gpu_cache"), (512, "bf16", "foreach", "gpu_cache")]
            results, cache = [], None
            for i, config in enumerate(configs):
                save_json(output / "status.json", dict(state="running", case=i+1, total=len(configs), config=config))
                if config[-1] == "gpu_cache" and cache is None:
                    cache = GPUTrainingCache(store, labels, binary, device)
                results.append(one_case(store, training, labels, binary, cache, config, device, output))
                save_json(output / "summary.json", dict(results=results, training_points=len(training),
                    scope="Warm throughput, same tile, 3 repeats; excludes startup/validation/checkpoint/I/O-to-disk"))
            fastest = max(results, key=lambda r:r["samples_per_second"])
            for config in (configs[0], (fastest["batch_size"], fastest["precision"], fastest["optimizer_mode"], fastest["input_mode"])):
                one_case(store, training, labels, binary, cache, config, device, output, profile=True)
            prediction_precision_check(store, labels, binary, validation, output, device)
            save_json(output / "status.json", dict(state="completed", fastest=fastest, completed_cases=len(results)))
        finally:
            store.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    main(**vars(p.parse_args()))
