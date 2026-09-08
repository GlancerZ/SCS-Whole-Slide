"""Original SCS topology, with PyTorch SDPA replacing Keras attention.

Architecture derived from SCS/src/transformer.py (MIT; see SCS/LICENSE).
No width/depth scaling, Muon, positional encoding changes, or early stopping.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.output = nn.Linear(64, 64)

    def forward(self, x):
        q, k, v = (layer(x).unsqueeze(1) for layer in (self.query, self.key, self.value))
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.1 if self.training else 0.0, is_causal=False)
        return self.output(x.squeeze(1))


def mlp(widths, dropout):
    layers = []
    for left, right in zip(widths, widths[1:]):
        layers.extend([nn.Linear(left, right), nn.GELU(approximate="none"), nn.Dropout(dropout)])
    return nn.Sequential(*layers)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64, eps=1e-6)
        self.attention = Attention()
        self.norm2 = nn.LayerNorm(64, eps=1e-6)
        self.mlp = mlp([64, 128, 64], 0.1)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class OriginalSCS(nn.Module):
    def __init__(self, n_genes, n_neighbors=50):
        super().__init__()
        if n_genes < 1 or n_neighbors < 1:
            raise ValueError("Input dimensions must be positive")
        self.n_genes, self.n_neighbors = n_genes, n_neighbors
        self.expression_projection = nn.Linear(n_genes, 64)
        self.position_projection = nn.Linear(2, 64)
        self.blocks = nn.ModuleList([Block() for _ in range(8)])
        self.final_norm = nn.LayerNorm(64, eps=1e-6)
        self.head = mlp([64, 1024, 256], 0.5)
        self.direction_head = nn.Linear(256, 16)
        self.foreground_head = nn.Linear(256, 1)
        # Keras Dense defaults: Glorot uniform, zero biases. Random streams
        # differ across frameworks; transferred weights are tested separately.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, expression, relative_positions):
        if expression.ndim != 3 or expression.shape[1:] != (self.n_neighbors, self.n_genes):
            raise ValueError("Expression must have shape [batch, neighbors, genes]")
        if relative_positions.shape != (*expression.shape[:2], 2):
            raise ValueError("Positions must have shape [batch, neighbors, 2]")
        x = self.expression_projection(expression) + self.position_projection(relative_positions.float())
        for block in self.blocks:
            x = block(x)
        features = self.head(self.final_norm(x)[:, 0])
        # Binary logits internally for numerically stable BCE; sigmoid on export.
        return self.direction_head(features), self.foreground_head(features).squeeze(-1)


def losses(direction_logits, foreground_logits, direction_onehot, foreground):
    """Match Keras masked CCE and BCE, including division by total batch size.

    Mask from direction labels (not binary labels), as in upstream SCS.
    """
    labels = direction_onehot.float()
    per_row = -(labels * F.log_softmax(direction_logits.float(), dim=-1)).sum(-1)
    direction = (per_row * labels.sum(-1)).mean()
    binary = F.binary_cross_entropy_with_logits(foreground_logits.float(), foreground.float())
    return direction + binary, direction, binary


class ReferenceAdamW(torch.optim.Optimizer):
    """Dense TFA legacy AdamW semantics, NOT torch.optim.AdamW defaults.

    TFA decays p by wd*p, independently of lr. Its epsilon is applied to
    uncorrected sqrt(v). Preserve both differences and beta2=0.999.
    """
    def __init__(self, params, lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-7):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("Original SCS uses dense gradients only")
                state = self.state[p]
                if not state:
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["step"] += 1
                t = state["step"]
                p.mul_(1 - group["weight_decay"])
                state["m"].lerp_(p.grad, 1 - beta1)
                state["v"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                rate = group["lr"] * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(state["m"], state["v"].sqrt().add_(group["eps"]), value=-rate)
        return loss
