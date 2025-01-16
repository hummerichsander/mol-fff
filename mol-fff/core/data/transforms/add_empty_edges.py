import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class AddEmptyEdges(BaseTransform):
    """Adds edges to the dataset that are not present in the original dataset and encodes
    them with an additional bond encoding."""

    def __init__(self):
        super().__init__()

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            # Add empty edges
            old_edge_index = store.edge_index
            new_edge_index = self._dense_edge_index(store.num_nodes)

            # encode new edges with additional bond encoding
            old_edge_attr = store.edge_attr
            new_edge_attr = self._get_empty_edges(
                new_edge_index.shape[1], old_edge_attr.shape[1]
            )
            indices = self._get_edge_indices(old_edge_index, new_edge_index)
            old_edge_attr = torch.cat(
                (old_edge_attr, torch.zeros(old_edge_attr.shape[0], 1)), dim=1
            )
            new_edge_attr[indices] = old_edge_attr

            store["edge_index"] = new_edge_index
            store["edge_attr"] = new_edge_attr
        return data

    def _dense_edge_index(self, n_nodes: int) -> Tensor:
        """Returns a dense edge index tensor for a fully connected graph with n_nodes nodes."""
        return torch.tensor(
            [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j]
        ).t()

    def _get_edge_indices(
        self, old_edge_index: Tensor, new_edge_index: Tensor
    ) -> Tensor:
        indices = torch.tensor([])
        for edge_index in old_edge_index.T:
            index = torch.where((new_edge_index.T == edge_index[None, :]).all(dim=1))
            assert len(index) == 1, "Duplicate edge"
            index = index[0]
            indices = torch.cat([indices, index])
        return indices.long()

    def _get_empty_edges(self, num_edges: int, num_edge_features: int) -> Tensor:
        new_num_edge_features = num_edge_features + 1
        empty_edges = torch.zeros(num_edges, new_num_edge_features)
        empty_edges[:, -1] = torch.ones(num_edges)
        return empty_edges
