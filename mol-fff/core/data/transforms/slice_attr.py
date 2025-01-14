from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class SliceAttribute(BaseTransform):
    """Slices the given attribute of the dataset in the last dimension."""

    def __init__(self, attr: str, start: int, end: int, squeeze: bool = False):
        self.attr = attr
        self.start = start
        self.end = end
        self.squeeze = squeeze
        super().__init__()

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            store[self.attr] = store[self.attr][..., self.start : self.end]
            if self.squeeze:
                store[self.attr] = store[self.attr].squeeze(-1)
        return data
