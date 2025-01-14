from collections import namedtuple
from typing import Callable, List

from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader import DataLoader as GeometricDataLoader

DatasetTrainableLoader = (
    GeometricDataLoader
    | DataLoader
    | List[GeometricDataLoader]
    | List[DataLoader]
    | List[None]
)
Graph = GeometricData | GeometricBatch
Transform = Callable | nn.Module

# Named tuple outputs
AutoencoderOutput = namedtuple(
    "AutoencoderOutput", ["x_code", "edge_attr_code", "x_hat", "edge_attr_hat"]
)
EncoderOutput = namedtuple("EncoderOutput", ["x_code", "edge_attr_code"])
EncoderOutputWithIntermediates = namedtuple(
    "EncoderOutputWithIntermediates", ["x_code", "edge_attr_code", "intermediates"]
)
DecoderOutput = namedtuple("DecoderOutput", ["x_hat", "edge_attr_hat"])
NLLOutput = namedtuple("NLLOutput", ["z", "x1", "nll"])
MMDOutput = namedtuple("MMDOutput", ["z", "x1", "mmd"])
LatentMMDOutput = namedtuple("LatentMMDOutput", ["z", "mmd"])
