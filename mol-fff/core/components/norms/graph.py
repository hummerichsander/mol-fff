from typing import Literal

from torch import nn
from torch_geometric import nn as gnn


class BatchNorm(gnn.BatchNorm):
    def __init__(self, in_channels: int, *args, **kwargs):
        super().__init__(in_channels, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        return super().forward(x)


GraphNormType = Literal["instance", "layer", "batch", "graph"]


def get_graph_norm(
    norm: GraphNormType | None,
    feature_dim: int,
    *args,
    **kwargs,
) -> nn.Module | None:
    """Get graph normalization layer.

    :param norm: Normalization type.
    :param feature_dim: Number of input channels.
    :return: Normalization layer."""

    match norm:
        case None:
            return None
        case "instance":
            return gnn.norm.InstanceNorm(feature_dim, *args, **kwargs)
        case "layer":
            return gnn.norm.LayerNorm(feature_dim, *args, **kwargs)
        case "batch":
            return BatchNorm(feature_dim, *args, **kwargs)
        case "graph":
            return gnn.norm.GraphNorm(feature_dim, *args, **kwargs)
        case other:
            raise ValueError(f"Unrecognized normalization value: '{other}'")
