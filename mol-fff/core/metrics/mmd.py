from typing import Optional, List, Callable, Literal

import torch
from torch import Tensor, nn


class MMD(nn.Module):
    """Maximum Mean Discrepancy (MMD) metric.
    :param kernel: The kernel to use for the MMD. Either 'multiscale' or 'rbf'.
    :param reduction: The reduction to apply to the MMD. Either 'mean', 'sum' or 'none'.
    :param bandwidth_range: The range of bandwidths to use for the kernel.
    """

    def __init__(
        self,
        kernel: Literal["rbf", "multiscale"] = "rbf",
        reduction: Literal["mean", "sum"] | None = "mean",
        bandwidth_range: Optional[List[float]] = None,
    ):
        super().__init__()
        self.kernel = self._get_kernel(kernel)
        self.reduction = self._get_reduction(reduction)
        self.bandwidth_range = bandwidth_range

        if self.bandwidth_range is None:
            if kernel == "multiscale":
                self.bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            elif kernel == "rbf":
                self.bandwidth_range = [10, 15, 20, 50]

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        mask: Optional[Tensor] = None,
        padding_size: Optional[int] = None,
    ) -> Tensor:
        """Computes the MMD between two sets of samples.

        :param x: The first set of samples.
        :param y: The second set of samples.
        :param mask: The mask to apply to the samples.
        :param padding_size: The padding size to apply to the samples.
        :return: The MMD between the two sets of samples."""

        if mask is not None:
            x = x[mask]
            y = y[mask]
        if padding_size is not None:
            x = x[:-padding_size, ...]
            y = y[:-padding_size, ...]
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.t() + rx - 2.0 * xx
        dyy = ry.t() + ry - 2.0 * yy
        dxy = rx.t() + ry - 2.0 * zz

        XX, YY, XY = (
            torch.zeros(xx.shape).to(x.device),
            torch.zeros(xx.shape).to(x.device),
            torch.zeros(xx.shape).to(x.device),
        )

        for a in self.bandwidth_range:
            XX += self.kernel(dxx, a)
            YY += self.kernel(dyy, a)
            XY += self.kernel(dxy, a)

        mmd = XX + YY - 2.0 * XY
        return self.reduction(mmd)

    @staticmethod
    def _rbf(x: Tensor, bandwidth):
        return torch.exp(-0.5 * x / bandwidth)

    @staticmethod
    def _multiscale(x: Tensor, bandwidth):
        return bandwidth**2 * (bandwidth**2 + x) ** -1

    def _get_kernel(self, kernel: str) -> Callable:
        match kernel:
            case "multiscale":
                return self._multiscale
            case "rbf":
                return self._rbf
            case other:
                raise ValueError(f"kernel must be in ['multiscale', 'rbf'], got {kernel}")

    @staticmethod
    def _get_reduction(reduction: Literal["mean", "sum"] | None) -> Callable:
        match reduction:
            case "mean":
                return torch.mean
            case "sum":
                return torch.sum
            case None:
                return lambda x: x
            case other:
                raise ValueError(f"reduction must be in ['mean', 'sum', 'none'], got {other}")
