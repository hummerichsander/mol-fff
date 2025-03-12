from typing import Optional

from FrEIA.modules import AllInOneBlock

import torch
from torch import Tensor


class SetAllInOneBlock(AllInOneBlock):
    def forward(
        self, x: Tensor, lengths: Tensor, c: Optional[Tensor] = None, rev: bool = False
    ) -> Tensor:
        """See base class docstring"""
        original_shape = x.shape
        x = x.view(-1, x.size(-1))

        if tuple(x.shape[1:]) != self.dims_in[0]:
            raise RuntimeError(
                f"Expected input of shape {self.dims_in[0]}, got {tuple(x.shape[1:])}."
            )
        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if rev:
            x, global_scaling_jac = self._permute(x, rev=True)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x, rev=False),)

        x1, x2 = torch.split(x, self.splits, dim=-1)

        if not rev:
            x1_set = x1.view(original_shape[0], original_shape[1], -1)
            a1 = self.subnet(x1_set, lengths, c=c)
            a1 = a1.view(-1, a1.size(-1))
            x2, _ = self._affine(x2, a1)
        else:
            x1_set = x1.view(original_shape[0], original_shape[1], -1)
            a1 = self.subnet(x1_set, lengths, c=c)
            a1 = a1.view(-1, a1.size(-1))
            x2, _ = self._affine(x2, a1, rev=True)

        x_out = torch.cat((x1, x2), 1)

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        elif self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        x_out = x_out.view(original_shape)
        return x_out
