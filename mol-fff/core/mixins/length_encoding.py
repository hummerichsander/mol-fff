from typing import Optional

import torch
from torch import Tensor

from ..utils.sets import length_encoding


class LengthEncodingMixin:
    """Mixin that provides length encoding functionality."""

    class hparams:
        length_encoding_dim: int | None = None

    def _apply_length_encoding(self, x: Tensor, lengths: Tensor, c: Optional[Tensor] = None) -> Optional[Tensor]:
        if self.hparams.length_encoding_dim is None:
            return c

        le = length_encoding(
            lengths, self.hparams.length_encoding_dim, x.device, x.dtype
        )
        le = le.unsqueeze(1).repeat(1, x.shape[1], 1)
        if c is not None:
            return torch.cat([c, le], dim=-1)
        else:
            return le
