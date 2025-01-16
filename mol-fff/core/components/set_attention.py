import math
from typing import Optional, Literal

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .norms.set import get_set_norm
from ..utils.masking import apply_masks


def set_softmax(
    pre_activations: Tensor,
    lengths: Tensor,
    eps: float = 1e-15,
    remove_self_loops: bool = False,
) -> Tensor:
    """Computed the softmax along the second dimension of `pre_activations`.

    :param pre_activations: Pre softmax attention activations of shape
        (batch_size, num_elements, num_elements).
    :param lengths: The lengths of the sets of shape (batch_size,)
    :param eps: numerical stability constant.
    :param remove_self_loops: whether to remove self-loops from the attention matrix.
    :return: softmax of pre-activations"""

    exp_ = pre_activations.exp()
    if remove_self_loops:
        exp = exp_ * (1 - torch.eye(exp_.size(1), device=exp_.device)).unsqueeze(0)
    else:
        exp = exp_
    exp = apply_masks(exp, lengths)
    exp = apply_masks(exp.transpose(1, 2), lengths).transpose(1, 2)

    return exp / (exp.sum(dim=2, keepdim=True) + 1e-15)


class PMA(nn.Module):
    """Taken from https://github.com/rajesh-lab/deep_permutation_invariant.git"""

    def __init__(self, dim: int, heads: int, seeds: int):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, heads, norm=None)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Pools the input set to a set of size `seeds`.

        :param x: The input set of shape (batch_size, num_elements, dim)
        :param lengths: The lengths of the sets of shape (batch_size,)
        :return: The pooled set of shape (batch_size, seeds, dim), and the lengths of the pooled sets.
        """

        lengths_out = torch.ones_like(lengths) * self.S.size(1)
        lengths_out = torch.min(lengths_out, lengths)
        return (
            self.mab(self.S.repeat(x.size(0), 1, 1), x, lengths=lengths, mask=["K"]),
            lengths_out,
        )


class MAB(nn.Module):
    """Taken and adapted from https://github.com/rajesh-lab/deep_permutation_invariant.git"""

    def __init__(
        self,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        heads: int,
        condition_dim: int = 0,
        norm: Literal["layer", "set"] = "layer",
        remove_self_loops: bool = False,
        condition_mode: Literal["element", "set"] = "element",
        bias: bool = False,
    ):
        super(MAB, self).__init__()

        self.dim_V = dim_V
        self.heads = heads
        self.norm = norm

        if condition_dim >= 0:
            if condition_mode == "element":
                self.fc_q = nn.Linear(dim_Q + condition_dim, dim_V, bias=bias)
                self.fc_k = nn.Linear(dim_K + condition_dim, dim_V, bias=bias)
                self.fc_v = nn.Linear(dim_K + condition_dim, dim_V, bias=bias)
            elif condition_mode == "set":
                self.fc_q = nn.Linear(dim_Q, dim_V, bias=bias)
                self.fc_k = nn.Linear(dim_K, dim_V, bias=bias)
                self.fc_v = nn.Linear(dim_K, dim_V, bias=bias)
                self.fc_c = nn.Linear(condition_dim, dim_V, bias=bias)
            else:
                raise ValueError(f"Unknown condition_mode: {condition_mode}")
        else:
            self.fc_q = nn.Linear(dim_Q, dim_V, bias=bias)
            self.fc_k = nn.Linear(dim_K, dim_V, bias=bias)
            self.fc_v = nn.Linear(dim_K, dim_V, bias=bias)

        self.norm0 = get_set_norm(norm, dim_V)
        self.norm1 = get_set_norm(norm, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V, bias=bias)

        self.condition_mode = condition_mode
        self.remove_self_loops = remove_self_loops

        self._reset_parameters()

    def forward(
        self, Q, K, lengths: Tensor, mask: list[str] = [], c: Optional[Tensor] = None
    ) -> Tensor:
        """Computes the multi-head attention between queries and keys and transforms the values.

        :param Q: The queries of shape (batch_size, num_queries, dim_Q)
        :param K: The keys of shape (batch_size, num_keys, dim_K)
        :param lengths: The lengths of the sets of shape (batch_size,)
        :param mask: A list of strings specifying which attention matrices to mask.
        :param c: The condition of shape (batch_size, condition_dim) or (batch_size, num_conditions, condition_dim)
        :return: The transformed values of shape (batch_size, num_queries, dim_V)"""

        Q, K, lengths_c = self.apply_condition(Q, K, lengths, c)

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        if "Q" in mask:
            Q = apply_masks(Q, lengths_c)
        if "K" in mask:
            K = apply_masks(K, lengths_c)
            V = apply_masks(V, lengths_c)

        dim_split = self.dim_V // self.heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        lengths_ = torch.cat([lengths_c] * self.heads, 0)

        A = set_softmax(
            Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V),
            lengths_,
            remove_self_loops=self.remove_self_loops,
        )

        O = Q_ + A.bmm(V_)

        O = self.remove_condition(O, c)

        O = torch.cat(O.split(Q.size(0), 0), 2)
        O = O if getattr(self, "norm0", None) is None else self.norm0(O, lengths)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "norm1", None) is None else self.norm1(O, lengths)

        O = apply_masks(O, lengths)
        return O

    def apply_condition(
        self, Q: Tensor, K: Tensor, lengths: Tensor, c: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Applies the condition to the queries, keys and lengths. If the condition mode is set to "element", it
        concatenates the condition to the queries and keys along the feature dimension; if the condition mode is set
        it concatenates the condition to the keys along the set element dimension.

        :param Q: The queries of shape (batch_size, num_queries, dim_Q)
        :param K: The keys of shape (batch_size, num_keys, dim_K)
        :param lengths: The lengths of the sets of shape (batch_size,)
        :param c: The condition of shape (batch_size, condition_dim) or (batch_size, num_conditions, condition_dim)
        :return: The queries, keys and lengths with the condition applied and the lengths after conditioning.
        """

        if c is not None:
            if self.condition_mode == "element":
                Q = torch.cat([Q, c], dim=-1)
                K = torch.cat([K, c], dim=-1)
            elif self.condition_mode == "set":
                c = c[:, 0, :].unsqueeze(1)
                c = self.fc_c(c)
                K = torch.cat([c, K], dim=1)
                Q = torch.cat([c, Q], dim=1)
                lengths = lengths + 1
            else:
                raise ValueError(f"Unknown condition_mode: {self.condition_mode}")
        return Q, K, lengths

    def remove_condition(self, O: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Removes the conditioning from the output.

        :param O: The output of shape (batch_size, num_queries, dim_V)
        :param c: The condition of shape (batch_size, condition_dim) or (batch_size, num_conditions, condition_dim)
        :return: The output without the conditioning."""

        if c is not None:
            if self.condition_mode == "set":
                O = O[:, 1:, :]
        return O

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.kaiming_normal_(self.fc_o.weight, mode="fan_in", nonlinearity="relu")
        if hasattr(self, "fc_c"):
            nn.init.xavier_uniform_(self.fc_c.weight)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class SAB(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *args, **kwargs):
        super(SAB, self).__init__()

        self.mab = MAB(input_dim, input_dim, output_dim, *args, **kwargs)

    def forward(self, x, lengths: Tensor, c: Optional[Tensor] = None) -> Tensor:
        out = self.mab(x, x, lengths, mask=["Q", "K"], c=c)
        return out

    def reset_parameters(self) -> None:
        self.mab.reset_parameters()


class ISAB(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, seeds: int, *args, **kwargs):
        super(ISAB, self).__init__()
        self.seeds = nn.Parameter(torch.Tensor(1, input_dim, output_dim))
        nn.init.xavier_uniform_(self.seeds)

        self.mab0 = MAB(input_dim, input_dim, output_dim, *args, **kwargs)
        self.mab1 = MAB(input_dim, input_dim, output_dim, *args, **kwargs)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        H = self.mab0(
            self.seeds.expand(x.size(0), -1, -1), x, lengths=lengths, mask=["K"]
        )
        return self.mab1(x, H, lengths=lengths, mask=["Q"])
