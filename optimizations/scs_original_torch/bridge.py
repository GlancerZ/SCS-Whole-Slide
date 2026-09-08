"""Two-process TF checkpoint bridge: export in TF env, check in Torch env.

No process needs both frameworks installed. The exchange is numeric NPZ, not
pickle. Export also records reference predictions, input gradients and AdamW.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
from .data import read_header


def prefix_array(path, key, count):
    """Read only the first N contiguous samples from a compressed NPY stream."""
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        shape, fortran, dtype = read_header(f)
        if fortran or dtype.hasobject:
            raise ValueError("Expected numeric C-contiguous SCS arrays")
        n = min(count, shape[0])
        size = n * int(np.prod(shape[1:])) * dtype.itemsize
        return np.frombuffer(f.read(size), dtype=dtype).reshape(n, *shape[1:]).copy()


def export_tf(output, checkpoint=None, data_dir=None, count=16, suffix="0:0:0:0"):
    import tensorflow as tf
    import tensorflow_addons as tfa
    from optimizations.scs_streaming.model_reference import (
        create_transformer_classifier, dir_to_class, masked_categorical_cross_entropy)

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(20260905)
    if checkpoint and not data_dir:
        raise ValueError("Real checkpoint conversion requires its matching tile data")
    if data_dir:
        directory = Path(data_dir)
        read = lambda key: prefix_array(directory / f"{key}_{suffix}.npz", key, count)
        expression = read("x_train").astype(np.float32)
        absolute = read("x_train_pos").astype(np.int32)
        positions = (absolute - absolute[:, :1]).astype(np.float32)
        centers = read("y_train").astype(np.float64)
        valid = centers[:, 0] != -1
        centers[valid] -= absolute[valid, 0]
        centers[~valid] = -9999
        labels = dir_to_class(centers, 16).astype(np.float32)
        binary = read("y_binary_train").astype(np.float32)
    else:
        rng = np.random.RandomState(20260905)
        expression = rng.normal(size=(count, 50, 23)).astype(np.float32)
        positions = rng.randint(-30, 31, (count, 50, 2)).astype(np.float32)
        positions -= positions[:, :1].copy()
        binary = (np.arange(count) % 3 != 0).astype(np.float32)
        labels = np.eye(16, dtype=np.float32)[np.arange(count) % 16] * binary[:, None]
    neighbors, genes = expression.shape[1:]
    model = create_transformer_classifier(16, (neighbors, genes), (neighbors, 2),
                                          neighbors, 64, 1, [128, 64], 8, [1024, 256])
    if checkpoint:
        model.load_weights(checkpoint).expect_partial()
    x, p = tf.Variable(expression), tf.Variable(positions)
    with tf.GradientTape() as tape:
        direction, foreground = model([x, p], training=False)
        ld = masked_categorical_cross_entropy(labels, direction)
        lb = tf.keras.losses.BinaryCrossentropy()(binary[:, None], foreground)
        total = ld + lb
    gx, gp = tape.gradient(total, [x, p])
    payload = dict(expression=expression, positions=positions, labels=labels, binary=binary,
                   direction=direction.numpy(), foreground=foreground.numpy().reshape(-1),
                   losses=np.array([total.numpy(), ld.numpy(), lb.numpy()]),
                   expression_gradient=gx.numpy(), position_gradient=gp.numpy(),
                   metadata=np.array(json.dumps(dict(n_genes=genes, n_neighbors=neighbors,
                       source_checkpoint=str(checkpoint), data_dir=str(Path(data_dir).resolve()) if data_dir else None,
                       parameter_count=model.count_params(), dropout_disabled_for_equivalence=True))))
    def dense(prefix, weights):
        kernel, bias = weights
        payload["weight." + prefix + ".weight"] = kernel.reshape(-1, len(bias)).T
        payload["weight." + prefix + ".bias"] = bias.reshape(-1)
    def norm(prefix, layer):
        gamma, beta = layer.get_weights()
        payload["weight." + prefix + ".weight"] = gamma
        payload["weight." + prefix + ".bias"] = beta
    encoder = next(l for l in model.layers if l.__class__.__name__ == "PatchEncoder")
    dense("expression_projection", encoder.projection.get_weights())
    dense("position_projection", encoder.position_embedding.get_weights())
    norms = [l for l in model.layers if isinstance(l, tf.keras.layers.LayerNormalization)]
    attentions = [l for l in model.layers if isinstance(l, tf.keras.layers.MultiHeadAttention)]
    denses = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    assert len(norms) == 17 and len(attentions) == 8 and len(denses) == 20
    for i, attention in enumerate(attentions):
        prefix = f"blocks.{i}"
        norm(prefix + ".norm1", norms[2*i])
        norm(prefix + ".norm2", norms[2*i+1])
        weights = attention.get_weights()
        for j, name in enumerate(("query", "key", "value", "output")):
            kernel, bias = weights[2*j:2*j+2]
            dense(prefix + ".attention." + name, [kernel.reshape(64, 64), bias.reshape(64)])
        dense(prefix + ".mlp.0", denses[2*i].get_weights())
        dense(prefix + ".mlp.3", denses[2*i+1].get_weights())
    norm("final_norm", norms[-1])
    dense("head.0", denses[16].get_weights())
    dense("head.3", denses[17].get_weights())
    dense("direction_head", model.get_layer("pos_out").get_weights())
    dense("foreground_head", model.get_layer("cat_out").get_weights())
    initial = np.array([1., -2., 0.5], np.float32)
    gradients = np.array([[0.1, -0.2, 0], [1e-6, 0.3, -0.4], [-0.7, 0.8, 0.2]], np.float32)
    variable = tf.Variable(initial)
    optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
    steps = []
    for gradient in gradients:
        optimizer.apply_gradients([(tf.constant(gradient), variable)])
        steps.append(variable.numpy().copy())
    payload.update(optimizer_initial=initial, optimizer_gradients=gradients, optimizer_steps=np.array(steps))
    with Path(output).open("xb") as f:
        np.savez_compressed(f, **payload)
    print(json.dumps(dict(exported=str(output), examples=len(expression), parameters=model.count_params())))


def check_torch(source, report, checkpoint_output=None):
    import torch
    from .model import OriginalSCS, ReferenceAdamW, losses
    from .runner import save_json

    torch.set_num_threads(4)
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        model = OriginalSCS(metadata["n_genes"], metadata["n_neighbors"]).eval()
        weights = {k[len("weight."):]: torch.from_numpy(data[k].copy())
                   for k in data.files if k.startswith("weight.")}
        model.load_state_dict(weights, strict=True)
        assert sum(p.numel() for p in model.parameters()) == metadata["parameter_count"]
        x = torch.tensor(data["expression"], requires_grad=True)
        p = torch.tensor(data["positions"], requires_grad=True)
        direction, binary = model(x, p)
        ls = losses(direction, binary, torch.tensor(data["labels"]), torch.tensor(data["binary"]))
        ls[0].backward()
        actual = dict(direction=direction.detach().numpy(), foreground=binary.detach().sigmoid().numpy(),
                      losses=np.array([v.detach().item() for v in ls]),
                      expression_gradient=x.grad.numpy(), position_gradient=p.grad.numpy())
        variable = torch.nn.Parameter(torch.tensor(data["optimizer_initial"]))
        optimizer = ReferenceAdamW([variable])
        steps = []
        for gradient in data["optimizer_gradients"]:
            variable.grad = torch.tensor(gradient)
            optimizer.step()
            steps.append(variable.detach().numpy().copy())
        actual["optimizer_steps"] = np.array(steps)
        checks = {}
        for key, value in actual.items():
            expected = data[key]
            checks[key] = dict(max_abs_error=float(np.max(np.abs(value-expected))),
                               passed=bool(np.allclose(value, expected, atol=2e-5, rtol=2e-4)))
        checks["direction_argmax"] = dict(passed=bool(np.array_equal(
            actual["direction"].argmax(-1), data["direction"].argmax(-1))))
        checks["foreground_threshold"] = dict(passed=bool(np.array_equal(
            actual["foreground"] >= 0.5, data["foreground"] >= 0.5)))
        result = dict(passed=all(c["passed"] for c in checks.values()), checks=checks,
                      source=str(Path(source).resolve()), metadata=metadata,
                      scope="CPU FP32 forward/loss/input-gradients and dense AdamW; not stochastic training equivalence")
        save_json(Path(report), result)
        if not result["passed"]:
            raise AssertionError(json.dumps(result, indent=2))
        if checkpoint_output:
            with Path(checkpoint_output).open("xb") as f:
                torch.save(dict(model=model.state_dict(), config=metadata,
                                source="converted TensorFlow weights; optimizer state not converted"), f)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    export = sub.add_parser("export-tf")
    export.add_argument("--output", required=True)
    export.add_argument("--checkpoint")
    export.add_argument("--data-dir")
    export.add_argument("--count", type=int, default=16)
    check = sub.add_parser("check-torch")
    check.add_argument("--source", required=True)
    check.add_argument("--report", required=True)
    check.add_argument("--checkpoint-output")
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    (export_tf if mode == "export-tf" else check_torch)(**args)
