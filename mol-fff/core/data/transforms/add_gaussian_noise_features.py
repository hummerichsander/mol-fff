import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class AddGaussianNoiseFeatures(BaseTransform):
    """Replaces node features with Gaussian noise.
    :param feature_dim: The dimension of the node features.
    :param std: The standard deviation of the gaussian noise."""

    def __init__(self, feature_dim: int = 5, std: float = 1.0):
        self.feature_dim = feature_dim
        self.std = std
        super().__init__()

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            mean = torch.zeros(store.num_nodes, self.feature_dim)
            std = torch.ones(store.num_nodes, self.feature_dim) * self.std
            store.x = torch.normal(mean, std)
        return data
