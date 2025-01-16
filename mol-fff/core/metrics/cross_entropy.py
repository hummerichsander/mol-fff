import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

from torch_geometric.data import Batch


class CrossEntropy(torch.nn.Module):
    def __init__(self, attr: str = "y", **kwargs):
        super().__init__()
        accelerator = kwargs.pop("accelerator", None)
        if "weight" in kwargs:
            kwargs["weight"] = torch.tensor(kwargs["weight"]).to(
                device="cuda" if accelerator == "gpu" else "cpu"
            )
        self.attr = attr
        self.loss_fn = CrossEntropyLoss(**kwargs)

    def forward(self, batch: Batch, batch_hat: Batch) -> Tensor:
        """Compute the cross entropy loss between the predicted and target labels.

        :param batch: The batch of data.
        :param batch_hat: The batch of predictions.
        :return: The cross entropy loss."""

        prediction = getattr(batch_hat, self.attr)
        target = getattr(batch, self.attr)
        target = target.to(dtype=prediction.dtype)
        return self.loss_fn(prediction, target)
