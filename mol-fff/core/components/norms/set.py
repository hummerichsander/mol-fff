from typing import Optional, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...utils.masking import apply_masks


class SetNorm(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-5):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.0)
        torch.nn.init.constant_(self.biases, 0.0)

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        num_elements = lengths if lengths is not None else x.shape[1]

        x = apply_masks(x, lengths)
        means = x.sum(dim=[1, 2]) / (num_elements * self.feature_dim)

        x_centered = apply_masks(x - means[:, None, None], lengths)
        vars = x_centered.pow(2).sum(dim=[1, 2]) / (num_elements * self.feature_dim)

        x = x_centered / (vars[:, None, None].sqrt() + self.eps)

        x = F.linear(x, torch.diag_embed(self.weights), self.biases)
        x = apply_masks(x, lengths)
        return x


class LayerNorm(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-5):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.0)
        torch.nn.init.constant_(self.biases, 0.0)

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        x = apply_masks(x, lengths)
        means = x.sum(dim=-1) / self.feature_dim

        x_centered = apply_masks(x - means[:, :, None], lengths)
        vars = x_centered.pow(2).sum(dim=-1) / self.feature_dim

        x = x_centered / (vars[:, :, None].sqrt() + self.eps)

        x = F.linear(x, torch.diag_embed(self.weights), self.biases)
        x = apply_masks(x, lengths)
        return x


SetNormType = Literal["layer", "set"]


def get_set_norm(
    norm: SetNormType | None, feature_dim: int, *args, **kwargs
) -> nn.Module | None:
    """Get set normalization layer.

    :param norm: Normalization type.
    :param feature_dim: Number of input features.
    :return: Normalization layer."""

    match norm:
        case None:
            return None
        case "layer":
            return LayerNorm(feature_dim, *args, **kwargs)
        case "set":
            return SetNorm(feature_dim, *args, **kwargs)
        case other:
            raise ValueError(f"Unrecognized norm value: '{other}'")
