import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class ChangeDataType(BaseTransform):
    """Changes the data type of the attribute to the specified data type.
    :param attr: The attribute to change the data type of.
    :param dtype: The data type to change the attribute to."""

    def __init__(self, attr: str, dtype: str):
        self.attr = attr
        self.dtype = eval(f"torch.{dtype}")

        assert isinstance(self.dtype, torch.dtype), "dtype must be a valid torch dtype."

        super().__init__()

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            store[self.attr] = store[self.attr].to(dtype=self.dtype)
        return data
