from typing import Optional, Literal

from math import sqrt

import torch
from torch import Tensor, nn

from abc import ABC

from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices, MatrixPower, SPDAffineMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
import geomstats.backend as gs

from fff.utils.utils import fix_device


class Symmetric(SymmetricMatrices):
    """Symmetric matrices manifold. This overwrite of the `SymmetricMatrices` class
    enables the use of vmap for batched operations."""

    def matrix_representation(self, x: Tensor) -> Tensor:
        """Transforms a batch of symmetric matrices in basis representation to a batch of symmetric matrices.
        Overwrites the corresponding method from SymmetricMatrices to enable batched operations via vmap.

        :param x: batch of symmetric matrices in basis representation
        :return: batch of symmetric matrices in basis representation"""

        indices = torch.triu_indices(self.n, self.n)

        if x.dim() == 2:
            batch_size = x.shape[0]
            A = torch.zeros(
                (batch_size, self.n, self.n), device=x.device, dtype=x.dtype
            )
            A = A.index_put(
                (
                    torch.arange(batch_size).unsqueeze(-1),
                    indices[0].expand(batch_size, -1),
                    indices[1].expand(batch_size, -1),
                ),
                x,
            )
        else:
            A = torch.zeros((self.n, self.n), device=x.device, dtype=x.dtype)
            A = A.index_put((indices[0], indices[1]), x)

        factor = torch.ones_like(A) * 2 - torch.eye(
            self.n, device=x.device, dtype=x.dtype
        ).expand_as(A)
        A = (A + A.transpose(-2, -1)) * factor / 2

        return A


class AffineMetric(SPDAffineMetric):
    """Affine metric on the SPD manifold. This overwrite of the `SPDAffineMetric` class
    from geomstats handles spd matrices in their basis representation."""

    def exp(self, tangent_vec: Tensor, base_point: Tensor) -> Tensor:
        tangent_vec = self._space.from_basis(tangent_vec)
        base_point = self._space.from_basis(base_point)
        exp_point = fix_device(super().exp)(tangent_vec, base_point)
        return self._space.to_basis(exp_point)

    def log(self, point: Tensor, base_point: Tensor) -> Tensor:
        point = self._space.from_basis(point)
        base_point = self._space.from_basis(base_point)
        log_point = fix_device(super().log)(point, base_point)
        return self._space.to_basis(log_point)


class SPDVectorValuedMetric(SPDAffineMetric):
    def vvd(self, a: Tensor, b: Tensor) -> Tensor:
        """Vector-valued distance between two points.

        :param a: first point
        :param b: second point
        :return: distance between two points"""

        vvd_fn = SPDVectorValuedDistance(self._space)
        return vvd_fn(a, b)  # vvd_fn(a, b)

    def squared_dist(self, a: Tensor, b: Tensor) -> Tensor:
        vvd = self.vvd(a, b)
        return torch.sum(vvd**2, axis=-1, keepdims=True)

    @property
    def vvd_dim(self) -> int:
        """Dimension of the vector-valued distance between two points."""
        return self._space.n


class SPD(SPDMatrices):
    """Symmetric Positive Definite (SPD) manifold. This overwrite of the `SPDMatrices` class
    from geomstats handles spd matrices in their basis representation. Furthermore, it enables
    the use of vmap for batched operations."""

    def __init__(
        self,
        n: int,
        equip: bool = True,
        metric_type: Literal["vector_valued", "combined"] = "vector_valued",
    ):
        super().__init__(n=n, equip=equip)
        self.n = n
        self.embedding_space = Symmetric(n)
        self.metric_type = metric_type

    def to_basis(self, A: Tensor) -> Tensor:
        if A.shape[-1] == self.dim:
            return A
        return gs.triu_to_vec(A)

    def from_basis(self, x: Tensor) -> Tensor:
        if x.shape[-2:] == (self.n, self.n):
            return x
        return self.embedding_space.matrix_representation(x)

    def to_tangent_basis(self, A: Tensor) -> Tensor:
        A = fix_device(self.default_metric()(self).log)(
            A, torch.eye(self.n, device=A.device, dtype=A.dtype)
        )
        return gs.triu_to_vec(A)

    def from_tangent_basis(self, x: Tensor) -> Tensor:
        A_prime = self.embedding_space.matrix_representation(x)
        A = fix_device(self.default_metric()(self).exp)(
            A_prime, torch.eye(self.n, device=A_prime.device, dtype=A_prime.dtype)
        )
        return A

    def cholesky_composition(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        indices = torch.triu_indices(self.n, self.n, 1)

        L = torch.zeros((batch_size, self.n, self.n), device=x.device, dtype=x.dtype)
        L[:, indices[0], indices[1]] = x[..., : -self.n]
        L += torch.eye(self.n, device=x.device)[None, ...]

        D = torch.diag_embed(torch.relu(x[..., -self.n :]) + 1e-6)

        A = L @ D @ L.transpose(-2, -1)

        return self.to_basis(A)

    def projection(self, x: Tensor) -> Tensor:
        A = self.from_basis(x)
        A_projected = super().projection(A)
        return self.to_basis(A_projected)

    def to_tangent(self, A: Tensor, base_point: Tensor) -> Tensor:
        A_tangent = super().to_tangent(A, base_point)
        x_tangent = self.to_basis(A_tangent)
        return x_tangent

    @property
    def origin(self) -> Tensor:
        return self.to_basis(torch.eye(self.n))

    def get_metric(self):
        match self.metric_type:
            case "vector_valued":
                return SPDVectorValuedMetric(self)
            case "combined":
                return SPDCombinedMetric(self)
            case _:
                raise ValueError(f"Unknown metric type: {self.metric_type}")

    @staticmethod
    def n_from_d(d: int) -> int:
        n = 1 / 2 * (sqrt(8 * d + 1) - 1)
        if int(n) != n:
            raise ValueError(
                f"Invalid dimension: {d}. Please provide a dimension d=n(n+1)/2 for some integer n."
            )
        return int(n)


class SPDVectorValuedDistance(nn.Module):
    """Implements the vector valued distance between two SPD matrices."""

    def __init__(self, manifold: SPD):
        self.manifold = manifold
        super(SPDVectorValuedDistance, self).__init__()

    def forward(self, x: Tensor, y: Tensor, keepdim=True) -> Tensor:
        A = self.manifold.from_basis(x)
        B = self.manifold.from_basis(y)
        A_inv_sqrt = fix_device(MatrixPower(-0.5))(A)

        M = Matrices.mul(A_inv_sqrt, B, A_inv_sqrt)
        M = Matrices.to_symmetric(M)

        eigvals = torch.linalg.eigvalsh(M)
        log_eigvals = torch.log(eigvals)

        return log_eigvals
