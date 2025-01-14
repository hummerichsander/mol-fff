from typing import Literal

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class AddQEDScore(BaseTransform):
    """Add the Quantitative Estimation of Drug-likeness (QED) score to the dataset."""

    def __init__(self, profile: Literal["qm9", "zinc"] = "qm9"):
        super().__init__()
        self.profile = profile

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            mol = Chem.MolFromSmiles(store.smiles)
            qed = Descriptors.qed(mol)
            store["qed"] = torch.tensor(qed, dtype=store.x.dtype)
        return data
