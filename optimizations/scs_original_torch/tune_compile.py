"""Bounded follow-up: compiled forward/backward at unchanged batch 10, FP32."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np
import torch

from optimizations.scs_streaming.data import partition
from .data import ArrayStore
from .model import OriginalSCS
from .performance import GPUTrainingCache
from .runner import configure_device, direction_labels, save_json
from .tune import one_case


def check_numerics(cache, device):
    """Same real weights and inputs, deterministic evaluation and input gradient."""
    saved = torch.load(Path(__file__).parent / "validation/st19_x11_y12_best.pt",
                       map_location="cpu", weights_only=True)
    model = OriginalSCS(2000).to(device).eval()
    model.load_state_dict(saved["model"])
    model.requires_grad_(False)
    x = cache.expression[:16].detach().clone().requires_grad_(True)
    p = cache.positions[:16]
    reference = model(x, p)
    gradient = torch.autograd.grad(sum(y.square().sum() for y in reference), x)[0]
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)
    actual = compiled(x, p)
    actual_gradient = torch.autograd.grad(sum(y.square().sum() for y in actual), x)[0]
    for a, b in zip(reference, actual):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(gradient, actual_gradient, atol=2e-5, rtol=2e-4)
    return dict(passed=True, samples=16,
        direction_max_error=float((reference[0]-actual[0]).detach().abs().max()),
        binary_logit_max_error=float((reference[1]-actual[1]).detach().abs().max()),
        input_gradient_max_error=float((gradient-actual_gradient).abs().max()),
        caveat="Compiler RNG differs during training; same-seed training is not bit-identical")


def main():
    if os.environ.get("SLURM_JOB_ID") != "20305630":
        raise RuntimeError("Scoped to tuning allocation 20305630")
    occupied = subprocess.check_output(["nvidia-smi", "--id="+os.environ["CUDA_VISIBLE_DEVICES"],
        "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if occupied:
        raise RuntimeError("GPU occupied: " + occupied)
    output = Path("runs/ST19_original_torch_compile_20305630")
    output.mkdir(exist_ok=False)
    save_json(output / "status.json", dict(state="loading"))
    torch.set_num_threads(4)
    device = configure_device("cuda", "none", 70000)
    data = Path("runs/ST19_whole_scs/tiles/x11_y12/data")
    with h5py.File(data / "spots0:0:0:0.h5ad") as f:
        shape = tuple(f["X"].attrs["shape"]) if isinstance(f["X"], h5py.Group) else f["X"].shape
    with tempfile.TemporaryDirectory(prefix="scs-compile-", dir=os.environ.get("SLURM_TMPDIR")) as tmp:
        store = ArrayStore(data, tmp)
        try:
            positions = store.array("x_train_pos")
            labels = direction_labels(store.array("y_train"), positions)
            binary = np.asarray(store.array("y_binary_train"), dtype=np.float32)
            training, validation = partition(positions, binary, shape, 0.0625)
            cache = GPUTrainingCache(store, labels, binary, device)
            results = []
            for compiled in (False, True):
                save_json(output / "status.json", dict(state="running", compiled=compiled))
                results.append(one_case(store, training, labels, binary, cache,
                    (10, "fp32", "foreach", "gpu_cache"), device, output, compile_model=compiled))
            save_json(output / "status.json", dict(state="checking_numerics", results=results))
            numeric = check_numerics(cache, device)
            save_json(output / "numerical_check.json", numeric)
            save_json(output / "status.json", dict(state="completed", results=results,
                numerical_check=numeric,
                caveat="Throughput and same-weight check; end-to-end training validation still needed"))
        except BaseException as error:
            save_json(output / "status.json", dict(state="failed", error=repr(error)))
            raise
        finally:
            store.close()


if __name__ == "__main__":
    main()
