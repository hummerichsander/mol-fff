from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class SwapAttributes(BaseTransform):
    """Swap two attributes of the dataset.
    :param attr1: The first attribute to swap.
    :param attr2: The second attribute to swap."""

    def __init__(self, attr1: str, attr2: str):
        super().__init__()
        self.attr1 = attr1
        self.attr2 = attr2

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            store[self.attr1], store[self.attr2] = store[self.attr2], store[self.attr1]
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(attr1={self.attr1}, attr2={self.attr2})"
