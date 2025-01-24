import concurrent.futures
from typing import Optional

import networkx as nx
import torch
from networkx import Graph as NetworkXGraph
from torch import Tensor
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData
from torch_geometric.utils import from_networkx, to_dense_adj


def networkx_to_geometric(graph: NetworkXGraph) -> GeometricData:
    data = from_networkx(graph)
    if not data.edge_attr:
        data.edge_attr = torch.ones(data.edge_index.shape[1], 1)
    return data


def geometric_to_networkx(data: GeometricData) -> NetworkXGraph:
    graph = NetworkXGraph()
    graph.add_nodes_from(data.x.squeeze().tolist())
    graph.add_edges_from(data.edge_index.t().tolist())
    return graph


def geometric_to_adjacency(
    graph: GeometricData | GeometricBatch | list[GeometricData],
    max_workers: Optional[int] = None,
) -> list[Tensor]:
    """Transforms a PyG graph or batch of graphs into a list of adjacency matrices
    :param graph: The graph instance or batch of graphs.
    :param max_workers: The maximum number of workers to use for parallel processing.
    :return: The adjacency matrix or a list of adjacency matrices."""

    def get_adjacency(data: GeometricData) -> Tensor:
        edge_attr = data.edge_attr.argmax(dim=1)
        edge_index = data.edge_index[:, edge_attr == 0]
        adj = torch.zeros((data.num_nodes, data.num_nodes))
        adj[edge_index[0], edge_index[1]] = 1
        return adj

    if isinstance(graph, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            return list(executor.map(get_adjacency, graph))

    elif hasattr(graph, "batch") and graph.batch is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            return list(executor.map(get_adjacency, graph.to_data_list()))

    else:
        return [get_adjacency(graph)]


def adj_from_binary_edge_attr(
    x: Tensor, edge_index: Tensor, edge_attr: Tensor
) -> Tensor:
    """Computes the adjacency matrix from the edge index and edge attribute tensors.
    :param x: node features
    :param edge_index: edge index tensor
    :param edge_attr: edge attribute tensor (binary one-hot encoded edge attributes)
    :return the adjacency matrix of the graph."""
    edge_attr = edge_attr.argmax(dim=1)
    edge_index = edge_index[:, edge_attr == 0]
    adj = torch.zeros((x.shape[0], x.shape[0]))
    adj[edge_index[0], edge_index[1]] = 1
    return adj


def dense_edge_index(
    n_nodes: int, device: torch.device, self_loops: bool = False
) -> Tensor:
    """Generated a dense edge index tensor for a fully connected graph with n nodes
    :param n_nodes: number of nodes in the graph
    :param device: device to place the tensor on
    :param self_loops: whether to include self loops in the graph
    :return: the generated edge index"""
    return torch.tensor(
        [
            [i, j]
            for i in range(n_nodes)
            for j in range(n_nodes)
            if (i != j) or self_loops
        ],
        device=device,
    ).t()


def generate_batch_tensor(num_nodes: list[int], device: torch.device) -> Tensor:
    """Generates a batch tensor with a number of nodes.
    :param num_nodes: The number of nodes in each graph in the batch.
    :param device: The device to place the tensor on.
    :return: A batch tensor with the specified number of nodes."""
    batch = torch.zeros(sum(num_nodes), dtype=torch.long, device=device)
    idx = 0
    for i, n in enumerate(num_nodes):
        batch[idx : idx + n] = i
        idx = idx + n
    return batch


def compute_number_of_connected_components(batch: GeometricBatch) -> Tensor:
    """Computes the number of connected components in each graph in the batch.

    :param batch: The batch of graphs.
    :return: The number of connected components in each graph in the batch."""

    empty_edge_index = batch.edge_attr.shape[-1] - 1
    prob_edge = (batch.edge_attr.argmax(dim=-1) != empty_edge_index).to(torch.float)

    adj = to_dense_adj(batch.edge_index, batch.batch, edge_attr=prob_edge)

    n_components = [
        nx.number_connected_components(nx.Graph(a.cpu().numpy())) for a in adj
    ]

    return torch.tensor(n_components, dtype=torch.float)


def is_equal_graph(data0: GeometricData, data1: GeometricData) -> bool:
    """Determines if the graphs specified by `x` and `edge_attr` are the same.

    :param data0: Graph instance
    :param data1: Graph instance
    :return: A boolean indicating if the graphs are equal."""

    x0_c = data0.x.argmax(dim=-1)
    edge_attr0_c = data0.edge_attr.argmax(dim=-1)
    x1_c = data1.x.argmax(dim=-1)
    edge_attr1_c = data1.edge_attr.argmax(dim=-1)
    return torch.equal(x0_c, x1_c) and torch.equal(edge_attr0_c, edge_attr1_c)
