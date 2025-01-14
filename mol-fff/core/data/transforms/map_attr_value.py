from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class MapAttrValue(BaseTransform):
    """Map some values of an attribute to another value."""

    def __init__(self, attr: str, value_map: dict):
        self.attr = attr
        self.value_map = value_map
        super().__init__()

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            for k, v in self.value_map.items():
                store[self.attr][store[self.attr] == k] = v
        return data
