from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch

from ..utils.sets import pad_sets_to_max_size
from ..utils.typing import Transform


class Squeeze(nn.Module):
    """Squeeze class. Implements a squeeze module."""

    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim:
            return x.squeeze(dim=self.dim)
        return x.squeeze()


class Concatenate(nn.Module):
    """Concatenate class. Implements a concatenate module."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, *args):
        return torch.cat([*args], dim=self.dim)


class MultiIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return args


class UnbatchAndPad(nn.Module):
    """Module that first unbatches a batch of sets (batch * set size x set dimension) to a
    batch of individual sets (batch x set size x set dimension) according to a batch vector
    (batch * set size x 1) and then pads the sets to the maximum set size in the batch or a
    specified maximum set size.
    :param max_set_size: The maximum set size. If None, the maximum set size is determined
    """

    def __init__(self, max_set_size: Optional[int] = None):
        super().__init__()
        self.max_set_size = max_set_size

    def forward(self, src: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Unbatches and pads a batch of sets.
        :param src: A batch of sets.
        :param batch: The batch vector.

        :return: A tuple containing the padded batch and the padding sizes."""
        unbatched = unbatch(src, batch=batch)
        padded, masks = pad_sets_to_max_size(unbatched, self.max_set_size)
        return padded, masks


class RemoveLowerTriangularAdjacency(nn.Module):
    """Removes the lower triangular part of the adjacency matrix. This is useful for
    undirected graphs, where it is sufficient to model only one direction of the edge.
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch: Batch) -> Batch:
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        mask = edge_index[0] < edge_index[1]
        batch.edge_index = edge_index[:, mask]
        batch.edge_attr = edge_attr[mask]
        return batch


class AttributeIdentity(nn.Module):
    """Implements a module that returns the identity of an attribute of the input arguments.
    :param index: index of the argument which should be returned."""

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index
        self.identity = nn.Identity()

    def forward(self, *args) -> Tensor:
        return self.identity(args[self.index])


class SkipConnection(nn.Module):
    def __init__(
        self,
        inner: Transform,
        id_init: bool = False,
        dim_change: Optional[tuple[int, int]] = None,
    ):
        super().__init__()

        if dim_change:
            self.skip_dim_change = nn.Linear(*dim_change, bias=False)

        self.inner = inner

        if id_init:
            self.scale = torch.nn.Parameter(torch.zeros(1))
        else:
            self.scale = None

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        out = self.inner(x, *args, **kwargs)
        if self.scale is not None:
            out = out * self.scale

        if hasattr(self, "skip_dim_change"):
            x = self.skip_dim_change(x)

        return x[..., : out.shape[-1]] + out


class NonLearnableWeight(nn.Module):
    def __init__(self, weight: Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.weight)
