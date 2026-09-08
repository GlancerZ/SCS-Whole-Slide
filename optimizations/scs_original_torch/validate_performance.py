"""Audit completed tuning runs; read-only source/checkpoint/prediction inspection."""
import argparse
import csv
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np
import torch

from .runner import save_json
from .bridge import prefix_array
from .model import OriginalSCS
from optimizations.scs_streaming.data import relative_positions


def audit(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    completed = json.loads((candidate / "completed.json").read_text())
    with (candidate / "training_history.csv").open() as f:
        history = list(csv.DictReader(f))
    best = torch.load(candidate / "best.pt", map_location="cpu", weights_only=True)
    assert completed["completed"] and completed["inference_completed"]
    assert len(history) == completed["epochs"]
    monitor = best["config"]["monitor"]
    best_row = max(history, key=lambda r: float(r[monitor]))
    assert best["epoch"] == int(best_row["epoch"]) + 1 == completed["best_epoch"]
    assert completed["best_score"] == float(best_row[monitor])
    assert not any(candidate.rglob("*.partial"))
    weight_errors = {}
    for checkpoint in ("best.pt", "latest.pt"):
        left = torch.load(baseline / checkpoint, map_location="cpu", weights_only=True)["model"]
        right = torch.load(candidate / checkpoint, map_location="cpu", weights_only=True)["model"]
        assert left.keys() == right.keys()
        weight_errors[checkpoint] = max(float((left[k]-right[k]).abs().max()) for k in left)
    count = agreements = foreground_agreements = 0
    max_logit = max_foreground = 0.0
    name = "results/spot_prediction_0:0:0:0.txt"
    first_directions, first_probabilities = [], []
    with (baseline / name).open() as a, (candidate / name).open() as b:
        for left, right in zip_longest(a, b):
            assert left is not None and right is not None
            l, r = left.rstrip().split("\t"), right.rstrip().split("\t")
            assert len(l) == len(r) == 4 and l[:2] == r[:2]
            ld, rd = np.fromstring(l[3], sep=":"), np.fromstring(r[3], sep=":")
            lp, rp = float(l[2]), float(r[2])
            assert ld.shape == rd.shape == (16,) and np.isfinite(rd).all()
            assert np.isfinite(rp) and 0 <= rp <= 1
            if count < 16:
                first_directions.append(rd)
                first_probabilities.append(rp)
            agreements += int(ld.argmax() == rd.argmax())
            foreground_agreements += int((lp >= 0.1) == (rp >= 0.1))
            max_logit = max(max_logit, float(np.max(np.abs(ld-rd))))
            max_foreground = max(max_foreground, abs(lp-rp))
            count += 1
    assert count == completed["prediction_rows"]
    config = best["config"]
    if config["amp"] != "none" or config.get("matmul_precision", "highest") != "highest":
        raise ValueError("This CPU best-prediction audit currently targets full-FP32 inference")
    data = Path(config["data_dir"])
    read = lambda key: prefix_array(data / f"{key}_{config['suffix']}.npz", key, len(first_directions))
    x = torch.from_numpy(read("x_train").astype(np.float32))
    p = torch.from_numpy(relative_positions(read("x_train_pos")))
    model = OriginalSCS(config["n_genes"], config["n_neighbors"]).eval()
    model.load_state_dict(best["model"])
    with torch.no_grad():
        d, b = model(x, p)
    np.testing.assert_allclose(first_directions, d.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(first_probabilities, b.sigmoid().numpy(), atol=1e-5, rtol=1e-4)
    return dict(passed=True, baseline=str(baseline), candidate=str(candidate),
        best_checkpoint_verified=True, epochs=len(history), prediction_rows=count,
        best_prediction_samples_verified=len(first_directions),
        best_prediction_max_logit_error=float(np.max(np.abs(np.asarray(first_directions)-d.numpy()))),
        coordinate_order_matches=True, finite_outputs=True, max_weight_error=weight_errors,
        direction_agreement=agreements/count, foreground_threshold_01_agreement=foreground_agreements/count,
        max_logit_error=max_logit, max_foreground_error=max_foreground,
        caveat="Numerical agreement is diagnostic, not a segmentation-quality guarantee")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = audit(args.baseline, args.candidate)
    save_json(Path(args.output), report)
    print(json.dumps(report, indent=2))
