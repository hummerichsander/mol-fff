from typing import Optional

from torch import Tensor, nn
from torch_geometric.data import Batch


class Precision(nn.Module):
    """Top-k precision metric."""

    def __init__(self, k_list: Optional[list[int]] = None):
        super().__init__()
        if k_list is None:
            k_list = [1]
        self.k_list = k_list

    def forward(self, yhat: Tensor, y: Tensor) -> dict[str, float]:
        results = {}

        if y.shape[-1] > 1:
            y = y.argmax(dim=-1)

        for k in self.k_list:
            assert (
                k <= yhat.shape[-1]
            ), f"k={k} is greater than the number of predictions {yhat.shape[1]}"
            top_k = yhat.topk(k, dim=-1).indices
            correct = top_k.eq(y.unsqueeze(1).expand_as(top_k)).sum().item()
            total = y.shape[0]
            results[f"top-{k}-precision"] = correct / total

        return results


class GeometricPrecision(nn.Module):
    """Top-1 precision metric for geometric data."""

    def __init__(self, attr: str = "y"):
        super().__init__()
        self.attr = attr

    def forward(self, batch: Batch, batch_hat: Batch) -> dict[str, float]:
        yhat = getattr(batch_hat, self.attr)
        y = getattr(batch, self.attr)

        if len(y.shape) > 1:
            y = y.argmax(dim=-1)

        top_1 = yhat.topk(1, dim=-1).indices
        correct = top_1.eq(y.unsqueeze(1).expand_as(top_1)).sum().item()
        total = y.shape[0]

        return correct / total
