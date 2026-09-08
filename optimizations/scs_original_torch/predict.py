"""Standalone inference with a trained or converted original-size checkpoint."""
import argparse
import os
from pathlib import Path
import tempfile

import torch

from .data import ArrayStore
from .model import OriginalSCS
from .runner import configure_device, predict_to_file, save_json


def predict(checkpoint, data_dir, output_dir, batch_size=128, device="cuda", amp="none",
            gpu_memory_mib=16384, cache_root=None):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    device = configure_device(device, amp, gpu_memory_mib)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    directory = Path(data_dir).resolve()
    if not config.get("data_dir") or directory != Path(config["data_dir"]).resolve():
        raise ValueError("Use the checkpoint's own tile data; gene order is tile-specific")
    model = OriginalSCS(config["n_genes"], config["n_neighbors"])
    model.load_state_dict(saved["model"], strict=True)
    model.to(device)
    output = Path(output_dir).resolve()
    if output == directory or output in directory.parents:
        raise ValueError("Use a separate output directory")
    output.mkdir(parents=True, exist_ok=False)
    suffix = config.get("suffix", "0:0:0:0")
    with tempfile.TemporaryDirectory(prefix="scs-torch-predict-",
                                     dir=cache_root or os.environ.get("SLURM_TMPDIR")) as cache:
        store = ArrayStore(directory, cache, suffix)
        try:
            for subset in ("train", "test"):
                if store.header("x_" + subset)[0][1:] != (config["n_neighbors"], config["n_genes"]):
                    raise ValueError("Checkpoint/input dimension mismatch")
            count = predict_to_file(model, store, output / f"spot_prediction_{suffix}.txt",
                                    batch_size, device, amp)
        finally:
            store.close()
    result = dict(inference_completed=True, prediction_rows=count, checkpoint=str(Path(checkpoint).resolve()),
                  data_dir=str(directory), amp=amp, batch_size=batch_size)
    save_json(output / "completed.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--amp", choices=["none", "bf16"], default="none")
    parser.add_argument("--gpu-memory-mib", type=int, default=16384)
    parser.add_argument("--cache-root")
    parser.add_argument("--threads", type=int, default=4)
    args = vars(parser.parse_args())
    torch.set_num_threads(args.pop("threads"))
    print(predict(**args))
