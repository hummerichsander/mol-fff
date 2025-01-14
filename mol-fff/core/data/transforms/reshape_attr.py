from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class ReshapeAttribute(BaseTransform):
    """Reshape the attribute of the dataset.
    :param attr: The attribute to reshape.
    :param shape: The shape to reshape the attribute to."""

    def __init__(self, attr: str, shape: tuple[int]):
        super().__init__()
        self.attr = attr
        self.shape = shape

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            store[self.attr] = store[self.attr].reshape(self.shape)
        return data
