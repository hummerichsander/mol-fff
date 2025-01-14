from typing import Literal

from torch import nn

VectorNormType = Literal["batch", "layer"]


def get_vector_norm(norm: VectorNormType | None, feature_dim: int, *args, **kwargs) -> nn.Module | None:
    """Configure vector normalization layer.

    :param norm: Normalization type.
    :param feature_dim: Number of input features.
    :return: Normalization layer."""

    match norm:
        case None:
            return None
        case "batch":
            return nn.BatchNorm1d(feature_dim, *args, **kwargs)
        case "layer":
            return nn.LayerNorm(feature_dim, *args, **kwargs)
        case other:
            raise NotImplementedError(f"Unrecognized norm value: '{other}'")
