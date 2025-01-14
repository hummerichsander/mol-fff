from typing import List, Optional

import torch
from torch import Tensor


def pad_sets_to_max_size(
    batch: List[Tensor], max_set_size: Optional[int] = None
) -> tuple[Tensor, Tensor]:
    """Pads a batch of sets to the maximum set size.
    :param batch: A batch of sets.
    :param max_set_size: The maximum set size. If None, the maximum set size is
        determined from the batch.

    :return: A tuple containing the padded batch and the padding masks."""
    set_sizes = [len(s) for s in batch]

    if max_set_size is None:
        max_set_size = max(set_sizes)

    padding_sizes = [max_set_size - size for size in set_sizes]

    padded_data = []
    masks = []

    for i, (set_data, pad_size) in enumerate(zip(batch, padding_sizes)):
        if pad_size > 0:
            pad = torch.zeros(
                pad_size, set_data.size(1), device=set_data.device, dtype=set_data.dtype
            )
            padded_data.append(torch.cat([set_data, pad], dim=0))
            mask = torch.ones((max_set_size,), device=set_data.device)
            mask[-pad_size:] = 0
            masks.append(mask)
        else:
            padded_data.append(set_data)
            masks.append(torch.ones((max_set_size,), device=set_data.device))

    return torch.stack(padded_data, dim=0), torch.stack(masks, dim=0)


def length_encoding(
    lengths: Tensor,
    length_encoding_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    lengths = lengths.float().unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, length_encoding_dim, 2, device=device, dtype=dtype).float()
        * (
            -torch.log(torch.tensor(10000.0, device=device, dtype=dtype))
            / length_encoding_dim
        )
    )

    length_encoding = torch.zeros(
        (lengths.size(0), length_encoding_dim), device=device, dtype=dtype
    )

    length_encoding[:, 0::2] = torch.sin(lengths * div_term)
    if length_encoding_dim % 2 == 0:
        length_encoding[:, 1::2] = torch.cos(lengths * div_term)
    else:
        length_encoding[:, 1::2] = torch.cos(lengths * div_term[:-1])

    return length_encoding
