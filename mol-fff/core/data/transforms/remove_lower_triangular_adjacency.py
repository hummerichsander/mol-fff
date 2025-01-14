from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class RemoveLowerTriangularAdjacency(BaseTransform):
    """Removes the lower triangular part of the adjacency matrix. This is useful for
    undirected graphs, where it is sufficient to model only one direction of the edge.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            edge_index = store.edge_index
            edge_attr = store.edge_attr
            mask = edge_index[0] < edge_index[1]
            store.edge_index = edge_index[:, mask]
            store.edge_attr = edge_attr[mask]
        return data
