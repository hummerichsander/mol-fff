import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class NormalizeWith(BaseTransform):
    """Normalize the dataset with given normalization constants.
    :param attrs: The attributes to normalize.
    :param mean: The mean to normalize with.
    :param std: The standard deviation to normalize with."""

    def __init__(self, attrs: list[str], mean: Tensor | list, std: Tensor | list):
        super().__init__()
        self.attrs = attrs
        if isinstance(mean, list):
            mean = torch.tensor(mean)
        if isinstance(std, list):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            for attr in self.attrs:
                store[attr] = (store[attr] - self.mean) / self.std
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(attrs={self.attrs}, mean={self.mean}, std={self.std})"
