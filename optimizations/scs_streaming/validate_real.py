"""Compare inference paths using an existing real ST19 checkpoint, read-only."""
import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import ArrayStore, batch_inputs
from .model_reference import create_transformer_classifier


def validate(data_dir, checkpoint, report):
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
    rng = np.random.RandomState(1107)
    with tempfile.TemporaryDirectory(prefix="scs-real-check-", dir=os.environ.get("SLURM_TMPDIR")) as temp:
        store = ArrayStore(data_dir, temp)
        try:
            shape, _, _ = store.header("x_train")
            model = create_transformer_classifier(16, shape[1:], (shape[1], 2), shape[1], 64, 1,
                                                  [128, 64], 8, [1024, 256])
            status = model.load_weights(checkpoint)
            status.assert_existing_objects_matched()
            status.expect_partial()
            xs, ps = [], []
            for subset in ("train", "test"):
                n = store.header("x_" + subset)[0][0]
                rows = rng.choice(n, min(32, n), replace=False)
                x, p = batch_inputs(store, subset, rows)
                xs.append(x); ps.append(p)
            expression, positions = np.concatenate(xs), np.concatenate(ps)
            original = model.predict([expression, positions], batch_size=10, verbose=0)

            @tf.function
            def streamed(x, p):
                return model([x, p], training=False)

            optimized = [x.numpy() for x in streamed(expression, positions)]
            result = dict(checkpoint=str(checkpoint), sample_count=len(expression),
                          device="GPU" if tf.config.list_physical_devices("GPU") else "CPU",
                          original_batch_size=10, compared_batch_size=len(expression),
                          max_logit_abs_difference=float(np.max(np.abs(original[0] - optimized[0]))),
                          max_probability_abs_difference=float(np.max(np.abs(original[1] - optimized[1]))),
                          direction_class_agreement=float(np.mean(np.argmax(original[0], axis=1) == np.argmax(optimized[0], axis=1))),
                          foreground_threshold_agreement=float(np.mean((original[1] > 0.1) == (optimized[1] > 0.1))))
            np.testing.assert_allclose(original[0], optimized[0], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(original[1], optimized[1], atol=2e-5, rtol=2e-5)
            assert result["direction_class_agreement"] == 1
            assert result["foreground_threshold_agreement"] == 1
            result["passed"] = True
            Path(report).write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            store.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    validate(a.data_dir, a.checkpoint, a.report)
