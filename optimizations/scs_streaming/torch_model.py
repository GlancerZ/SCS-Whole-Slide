"""PyTorch SCS transformer with explicit scaled-dot-product attention."""

from dataclasses import asdict, dataclass
import math

from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """A width-scaled version of the published SCS transformer.

    ``scale`` multiplies both hidden width and transformer depth. Attention
    heads grow with width so the 64-dimensional head size stays fixed. This
    makes 1x directly comparable to the reference topology and makes 2x/4x
    explicitly wider *and* deeper without changing the data contract.
    """

    input_dim: int
    n_genes: int = 0
    n_neighbors: int = 50
    n_classes: int = 16
    scale: int = 4
    base_width: int = 64
    base_layers: int = 8
    dropout: float = 0.1
    head_dropout: float = 0.5
    expression_scale: float = 1.0
    coordinate_scale: float = 1.0
    expression_encoding: str = "genept"

    def __post_init__(self):
        if self.scale not in (1, 2, 4):
            raise ValueError("scale must be one of 1, 2, or 4")
        if min(self.input_dim, self.n_neighbors, self.n_classes, self.base_layers) < 1:
            raise ValueError("model dimensions must be positive")
        if self.n_genes < 0:
            raise ValueError("gene count cannot be negative")
        if self.expression_encoding not in ("genept", "raw_counts"):
            raise ValueError("expression_encoding must be genept or raw_counts")
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.expression_scale, self.coordinate_scale)):
            raise ValueError("input scales must be finite and positive")

    @property
    def width(self):
        return self.base_width * self.scale

    @property
    def heads(self):
        return self.scale

    @property
    def head_dim(self):
        return self.base_width

    @property
    def layers(self):
        return self.base_layers * self.scale

    def to_dict(self):
        result = asdict(self)
        # Keep legacy default checkpoints/config comparisons compatible.
        for key, default in (("expression_scale", 1.), ("coordinate_scale", 1.),
                             ("expression_encoding", "genept")):
            if result[key] == default:
                del result[key]
        result.update(
            width=self.width,
            heads=self.heads,
            head_dim=self.head_dim,
            layers=self.layers,
        )
        return result


class SDPAAttention(nn.Module):
    """Multi-head self-attention routed through PyTorch's SDPA dispatcher."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        if width % heads:
            raise ValueError("attention width must be divisible by head count")
        self.heads = heads
        self.head_dim = width // heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, tokens, width))


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attention = SDPAAttention(width, heads, dropout)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SCSClassifier(nn.Module):
    """One whole-slide model with direction and foreground prediction heads."""

    def __init__(self, config, gene_embeddings=None):
        super().__init__()
        self.config = config
        width = config.width
        self.expression_projection = nn.Linear(config.input_dim, width, bias=False)
        self.spot_bias = nn.Parameter(self.expression_projection.weight.new_zeros(width))
        if gene_embeddings is not None:
            if tuple(gene_embeddings.shape) != (config.n_genes, config.input_dim):
                raise ValueError("GenePT table does not match model configuration")
            self.register_buffer(
                "gene_embeddings", gene_embeddings.float(), persistent=False
            )
        else:
            self.register_buffer("gene_embeddings", None, persistent=False)
        self.position_projection = nn.Linear(2, width)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(width, config.heads, config.dropout)
                for _ in range(config.layers)
            ]
        )
        self.final_norm = nn.LayerNorm(width, eps=1e-6)
        self.head_mlp = nn.Sequential(
            nn.Linear(width, 16 * width),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(16 * width, 4 * width),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
        )
        self.direction_head = nn.Linear(4 * width, config.n_classes)
        self.foreground_head = nn.Linear(4 * width, 1)

    def project_expression(self, expression):
        if isinstance(expression, (tuple, list)):
            raw_counts = self.config.expression_encoding == "raw_counts"
            if self.gene_embeddings is None and not raw_counts:
                raise ValueError("sparse spots require a GenePT gene table")
            if len(expression) != 4:
                raise ValueError("sparse spots need indices, values, offsets, shape")
            indices, values, offsets, shape = expression
            batch, neighbors = map(int, shape)
            lengths = offsets[1:] - offsets[:-1]
            if raw_counts:
                normalized_values = values
                projected_genes = self.expression_projection.weight.T.to(dtype=values.dtype)
            else:
                divisors = lengths.repeat_interleave(lengths).to(dtype=values.dtype)
                normalized_values = values / divisors
                projected_genes = self.expression_projection(self.gene_embeddings)
            pooled = F.embedding_bag(
                indices,
                projected_genes,
                offsets,
                mode="sum",
                per_sample_weights=normalized_values.to(projected_genes.dtype),
                include_last_offset=True,
            )
            valid = (lengths > 0).to(dtype=pooled.dtype).unsqueeze(-1)
            bias_mask = 1 if raw_counts else valid
            pooled = pooled + bias_mask * self.spot_bias.to(dtype=pooled.dtype)
            return pooled.reshape(batch, neighbors, self.config.width) * self.config.expression_scale
        if expression.ndim != 3 or expression.shape[-1] != self.config.input_dim:
            raise ValueError("dense spot embeddings have the wrong shape")
        return (self.expression_projection(expression) + self.spot_bias) * self.config.expression_scale

    def forward(self, expression, relative_positions):
        if relative_positions.ndim != 3:
            raise ValueError("positions must be [batch, neighbors, 2]")
        x = self.project_expression(expression)
        x = x + self.position_projection(
            relative_positions.to(dtype=x.dtype) * self.config.coordinate_scale
        )
        for block in self.blocks:
            x = block(x)
        features = self.head_mlp(self.final_norm(x)[:, 0])
        return self.direction_head(features), self.foreground_head(features).squeeze(-1)

    def optimizer_parameter_groups(self):
        """Split parameters according to Muon's intended hidden-matrix use.

        Transformer hidden matrices use Muon. Input/position projections,
        normalization and biases, and both classifier heads use AdamW.
        """
        muon, adamw = [], []
        for name, parameter in self.named_parameters():
            if name.startswith("blocks.") and parameter.ndim == 2:
                muon.append(parameter)
            else:
                adamw.append(parameter)
        if {id(p) for p in muon} & {id(p) for p in adamw}:
            raise RuntimeError("optimizer parameter groups overlap")
        if len(muon) + len(adamw) != sum(1 for _ in self.parameters()):
            raise RuntimeError("optimizer parameter groups are incomplete")
        return muon, adamw

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def scs_loss(
    direction_logits, foreground_logits, direction_targets, foreground_targets,
    foreground_class_weights=None,
):
    """Match the two reference losses, including background-masked direction loss."""
    foreground_targets = foreground_targets.float()
    direction_per_example = F.cross_entropy(
        direction_logits, direction_targets.long(), reduction="none"
    )
    direction = (direction_per_example * foreground_targets).mean()
    binary_per_example = F.binary_cross_entropy_with_logits(
        foreground_logits, foreground_targets, reduction="none"
    )
    if foreground_class_weights is not None:
        negative_weight, positive_weight = foreground_class_weights
        binary_per_example = binary_per_example * (
            (1-foreground_targets)*negative_weight + foreground_targets*positive_weight
        )
    foreground = binary_per_example.mean()
    return direction + foreground, direction, foreground
