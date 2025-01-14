from typing import List, Optional, Tuple

import torch
from torch import Tensor


def pad_sets(
    batch: List[Tensor], max_set_size: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """Pads a batch of sets to the maximum set size.
    :param batch: A batch of sets.
    :param max_set_size: The maximum set size. If None, the maximum set size is
        determined from the batch, defaults to None.

    :return: A tuple containing the padded batch and the lengths of the sets."""
    device = batch[0].device
    set_sizes = torch.tensor([len(x) for x in batch], device=device)

    if max_set_size is None:
        max_set_size = max(set_sizes)

    padding_sizes = max_set_size - set_sizes

    padded_batch = torch.stack(
        [
            torch.nn.functional.pad(
                set_data, (0, 0, 0, pad_size), mode="constant", value=0.0
            )
            for set_data, pad_size in zip(batch, padding_sizes)
        ]
    )

    return padded_batch, set_sizes


def unpad_sets(padded_batch: Tensor, lengths: Tensor) -> List[Tensor]:
    return [set[:length, ...] for set, length in zip(padded_batch, lengths)]


def apply_masks(
    input: Tensor, lengths: Optional[Tensor] = None, fill_value: float = 0.0
) -> Tensor:
    """Applies a binary mask to a tensor.
    :param input: The input tensor of shape (batch, set size, feature dim).
    :param mask: The mask tensor (batch, set size).
    :param fill_value: The value to fill the masked elements with, defaults to 0.0.
    :return: The masked tensor."""
    if lengths is None:
        return input
    batch, set_size, _ = input.shape
    masks = torch.arange(set_size, device=input.device).expand(
        batch, set_size
    ) < lengths.unsqueeze(1)
    return input.masked_fill(~masks.unsqueeze(-1), fill_value)


def masked_mean(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    """
    Computes the mean of a tensor along a given dimension, ignoring masked values.
    Args:
        x: A tensor of shape (batchsize, number of samples, feature dimension).
        mask: A tensor of shape (batchsize, number of samples, 1) with 0s and 1s.
        dim: The dimension to compute the mean along.
    Returns:
        A tensor of shape (batchsize,) representing the mean of x along the given dimension.
    """
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(-1)
    return torch.sum(x * mask, dim=dim) / torch.sum(mask, dim=dim)


def masked_std(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    """
    Computes the standard deviation of a tensor along a given dimension, ignoring masked values.
    Args:
        x: A tensor of shape (batchsize, number of samples, feature dimension).
        mask: A tensor of shape (batchsize, number of samples, 1) with 0s and 1s.
        dim: The dimension to compute the standard deviation along.
    Returns:
        A tensor of shape (batchsize,) representing the standard deviation of x along the given
            dimension.
    """
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(-1)
    mean = masked_mean(x, mask, dim=dim)
    return torch.sqrt(masked_mean((x - mean.unsqueeze(dim=dim)) ** 2, mask, dim=dim))
