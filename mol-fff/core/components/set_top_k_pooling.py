import torch
from torch import nn, Tensor

from ..utils.masking import apply_masks


class GlobalTopKPool1d(nn.Module):
    """Top-k pooling layer for set structured input inputs."""

    def __init__(self, k: int, largest: bool = True, dim: int = 1):
        super().__init__()
        self.register_buffer("k", torch.tensor(k))
        self.register_buffer("largest", torch.tensor(largest))
        self.register_buffer("dim", torch.tensor(dim))

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the TopKPooling layer. It pools the top k elements of each set.

        :param x: Input tensor of shape (batch_size, set_size, feature_dim)
        :param lengths: Lengths of the sets of shape (batch_size,)
        :return: Output tensor of shape (batch_size, k, feature_dim) and the updated lengths.
        """

        x = apply_masks(x, lengths, fill_value=float("-inf"))
        lengths = self._compute_topk_lengths(lengths)

        x = x.topk(
            self.k.item(),
            dim=self.dim.item(),
            largest=self.largest.item(),
            sorted=False,
        ).values

        x = apply_masks(x, lengths)
        return x, lengths

    def _compute_topk_lengths(self, lengths: Tensor) -> Tensor:
        lengths = torch.clamp(lengths, max=self.k.item())
        return lengths

    def extra_repr(self) -> str:
        return (
            f"k={self.k.item()}, largest={self.largest.item()}, dim={self.dim.item()}"
        )
