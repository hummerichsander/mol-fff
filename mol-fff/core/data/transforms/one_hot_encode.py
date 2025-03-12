import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class OneHotEncode(BaseTransform):
    """One hot encode the dataset with given one hot encoding constants.
    :param attrs: The attributes to one hot encode.
    :param num_classes: The number of classes to one hot encode with.
    :param shift: The shift to apply to the attributes before one hot encoding."""

    def __init__(self, attrs: list[str], num_classes: list[int], shift=0):
        super().__init__()
        self.attrs = attrs
        self.num_classes = num_classes
        self.shift = shift

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            for attr, num_classes in zip(self.attrs, self.num_classes):
                assert torch.equal(store[attr], store[attr].to(dtype=torch.int)), (
                    "Attributes must be integers"
                )
                store[attr] = F.one_hot(
                    store[attr] + self.shift, num_classes=num_classes
                ).to(dtype=torch.float)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(attrs={self.attrs}, num_classes={self.num_classes})"
