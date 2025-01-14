from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9

from .geometric_data import GeometricData


class QM9Data(GeometricData):
    def __init__(self, data):
        super().__init__(data)

    def get_dataset(self) -> Dataset:
        """Returns the QM9 dataset from pytorch_geometric."""
        print(self.hparams)
        return QM9(
            root=self.hparams.root,
            pre_transform=self.hparams.pre_transform,
            pre_filter=self.hparams.pre_filter,
            transform=self.hparams.transform,
        )
