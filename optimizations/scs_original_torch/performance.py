"""Opt-in performance paths; original architecture and optimizer equations stay fixed."""
from collections import defaultdict

import numpy as np
import torch

from optimizations.scs_streaming.data import relative_positions
from .model import ReferenceAdamW


class ForeachReferenceAdamW(ReferenceAdamW):
    """Batch independent parameter updates into CUDA multi-tensor operations."""
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            batches = defaultdict(list)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise ValueError("Dense gradients required")
                state = self.state[parameter]
                if not state:
                    state.update(step=0, m=torch.zeros_like(parameter), v=torch.zeros_like(parameter))
                state["step"] += 1
                batches[(parameter.device, parameter.dtype, state["step"])].append(parameter)
            beta1, beta2 = group["betas"]
            for (_, _, step), parameters in batches.items():
                grads = [p.grad for p in parameters]
                first = [self.state[p]["m"] for p in parameters]
                second = [self.state[p]["v"] for p in parameters]
                torch._foreach_mul_(parameters, 1 - group["weight_decay"])
                torch._foreach_lerp_(first, grads, 1 - beta1)
                torch._foreach_mul_(second, beta2)
                torch._foreach_addcmul_(second, grads, grads, value=1 - beta2)
                denominator = torch._foreach_sqrt(second)
                torch._foreach_add_(denominator, group["eps"])
                rate = group["lr"] * (1-beta2**step)**0.5 / (1-beta1**step)
                torch._foreach_addcdiv_(parameters, first, denominator, value=-rate)
        return loss


class GPUTrainingCache:
    """One FP32 training input copy on GPU; avoids per-batch host gathers/copies.

    Includes validation rows but never test data. Fits the existing 2,000-gene
    ST19 tiles in an 80-GB allocation. Larger tiles may not fit: allocation limits
    remain enforced and the caller must explicitly select this memory tradeoff.
    """
    def __init__(self, store, labels, binary, device, chunk_size=256):
        expression = store.array("x_train")
        self.expression = torch.empty(expression.shape, dtype=torch.float32, device=device)
        for start in range(0, len(expression), chunk_size):
            rows = slice(start, min(start+chunk_size, len(expression)))
            self.expression[rows].copy_(torch.from_numpy(expression[rows].astype(np.float32)))
        self.positions = torch.as_tensor(relative_positions(store.array("x_train_pos")), device=device)
        self.labels = torch.as_tensor(labels, device=device)
        self.binary = torch.as_tensor(binary, device=device)
        self.device = device

    def batch(self, rows):
        indices = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        return tuple(value.index_select(0, indices) for value in
                     (self.expression, self.positions, self.labels, self.binary))
