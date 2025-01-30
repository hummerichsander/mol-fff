from typing import Union, List, Tuple

import torch
import pandas as pd

from torch_geometric.data import InMemoryDataset, download_url, Data, Dataset
from torch_geometric.utils import from_smiles

from .geometric_data import GeometricData


class UnimersDataset(InMemoryDataset):
    """PytorchGeometric unimers dataset with martini beads."""

    download_link: str = "https://heibox.uni-heidelberg.de/f/fdbaaf5ce8a540ba884f/?dl=1"

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ["unimers_cleaned.csv"]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ["data.pt"]

    def download(self):
        download_url(self.download_link, self.raw_dir, filename=self.raw_file_names[0])

    def process(self):
        data_list: list[Data] = []

        raw_data = pd.read_csv(self.raw_paths[0])

        beads = list(set(raw_data["martini_bead"]))
        class_mapping = {bead: i for i, bead in enumerate(beads)}

        for i in range(len(raw_data)):
            smiles = raw_data["smiles"][i]
            bead = raw_data["martini_bead"][i]
            data = from_smiles(smiles, kekulize=True)
            data.x = data.x[..., 0]
            data.edge_attr = data.edge_attr[..., 0]
            data.bead = torch.tensor([class_mapping[bead]])
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


class UnimersData(GeometricData):
    def __init__(self, hparams):
        super().__init__(hparams)

    def get_dataset(self) -> Dataset:
        """Returns the Unimers dataset."""
        return UnimersDataset(
            root=self.hparams.root,
            pre_transform=self.hparams.pre_transform,
            pre_filter=self.hparams.pre_filter,
            transform=self.hparams.transform,
        )
