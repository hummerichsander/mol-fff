from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected


class ToUndirected(BaseTransform):
    """Transforms the dataset to an undirected graph by adding the missing edges."""

    def __init__(self):
        super().__init__()

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            store["edge_index"], store["edge_attr"] = to_undirected(
                store["edge_index"], store["edge_attr"]
            )

        return data
