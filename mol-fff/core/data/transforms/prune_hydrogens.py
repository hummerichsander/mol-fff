import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class PruneHydrogens(BaseTransform):
    """Remove hydrogen atoms from the dataset.
    :param hydrogen_index: The index of the hydrogen atom in the one-hot
        encoded node features."""

    def __init__(self, hydrogen_index: int = 0):
        super().__init__()
        self.hydrogen_index = hydrogen_index

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            num_nodes = store["x"].shape[0]

            hydrogen_mask = store["x"][:, self.hydrogen_index] == 1
            hydrogen_indices = torch.where(store["x"][:, 0] == 1)[0]
            hydrogen_edge_mask = torch.isin(store["edge_index"], hydrogen_indices).any(dim=0)

            store["x"] = store["x"][~hydrogen_mask]
            if "z" in store:
                store["z"] = store["z"][~hydrogen_mask]
            if "pos" in store:
                store["pos"] = store["pos"][~hydrogen_mask]
            if "batch" in store:
                store["batch"] = store["batch"][~hydrogen_mask]

            store["edge_attr"] = store["edge_attr"][~hydrogen_edge_mask]

            store["edge_index"] = store["edge_index"][:, ~hydrogen_edge_mask]
            old_to_new = torch.zeros(num_nodes, dtype=torch.long)
            old_to_new[~hydrogen_mask] = torch.arange(store["x"].shape[0])
            store["edge_index"] = old_to_new[store["edge_index"]]

        return data
