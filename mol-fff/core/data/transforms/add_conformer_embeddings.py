from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class AddConformerEmbeddings(BaseTransform):
    """Adds conformer embeddings of each node to the node-features."""

    def __init__(self):
        super().__init__()

    def __call__(self, data: Data | HeteroData):
        for store in data.stores:
            batch = store.
