import concurrent.futures
from typing import List

import networkx as nx
import torch
from torch import nn, Tensor
from torch_geometric.data import Batch as GeometricBatch

from .mmd import MMD
from ..utils.graph_infrastructure import geometric_to_adjacency

"""Implementations of MMD losses over node statistics as explained in
https://arxiv.org/pdf/2106.01098.pdf"""


class DegreeMMD(nn.Module):
    def __init__(self, *args, normalize: bool = True, **kwargs):
        """Compute the MMD between the degree distributions of two sets of graphs.
        :param normalize: If True, the degree distributions are normalized to sum to 1.
            This is useful when the graphs have different numbers of nodes to obtain a
            size-invariant descriptor. (see: https://arxiv.org/pdf/2106.01098.pdf)
        :param args: Arguments to pass to the MMD.
        :param kwargs: Keyword arguments to pass to the MMD."""
        super().__init__()
        self.normalize = normalize
        self.mmd = MMD(*args, **kwargs)

    def forward(self, batch_1: GeometricBatch, batch_2: GeometricBatch) -> Tensor:
        """computes the MMD between the degree distributions of two sets of graphs.
        :param batch_1: First batch of graphs.
        :param adj_2: Second batch of graphs.
        :return: The MMD between the degree distributions of the two sets of graphs."""
        adj_1 = geometric_to_adjacency(batch_1)
        adj_2 = geometric_to_adjacency(batch_2)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            degree_distributions_1 = list(executor.map(self.get_degree_distribution, adj_1))
            degree_distributions_2 = list(executor.map(self.get_degree_distribution, adj_2))

        max_degree = max([len(h) for h in degree_distributions_1] + [len(h) for h in degree_distributions_2])
        degree_distributions_1 = self._pad_to_max_size(degree_distributions_1, max_degree)
        degree_distributions_2 = self._pad_to_max_size(degree_distributions_2, max_degree)

        return self.mmd(degree_distributions_1, degree_distributions_2)

    def get_degree_distribution(self, adj: Tensor) -> Tensor:
        G = nx.from_numpy_array(adj.numpy())
        degree_distribution = torch.tensor(nx.degree_histogram(G))
        if self.normalize:
            degree_distribution = degree_distribution / degree_distribution.sum()
        return degree_distribution

    def _pad_to_max_size(self, hist: List[Tensor], max_size) -> Tensor:
        return torch.stack([torch.cat([h, torch.zeros(max_size - len(h), dtype=torch.float32)]) for h in hist])


class ClusteringMMD(nn.Module):
    def __init__(self, *args, bins: int = 100, normalize: bool = False, **kwargs):
        """Compute the MMD between the clustering coefficient distributions of two sets of graphs.
        :param bins: The number of bins to use for the histogram.
        :param normalize: If True, the histograms are evaluated as density histograms.
        :param args: Arguments to pass to the MMD.
        :param kwargs: Keyword arguments to pass to the MMD."""
        super().__init__()
        self.bins = bins
        self.normalize = normalize
        self.mmd = MMD(*args, **kwargs)

    def forward(self, batch_1: GeometricBatch, batch_2: GeometricBatch) -> Tensor:
        """Compute the MMD between the clustering coefficient distributions of two sets of graphs.
        :param batch_1: First batch of graphs.
        :param adj_2: Second batch of graphs.
        :return: The MMD between the clustering coefficient distributions of the two sets of graphs.
        """
        adj_1 = geometric_to_adjacency(batch_1)
        adj_2 = geometric_to_adjacency(batch_2)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            clustering_coefficients_1 = torch.stack(
                list(executor.map(self.get_cluster_coefficient_distribution, adj_1))
            )
            clustering_coefficients_2 = torch.stack(
                list(executor.map(self.get_cluster_coefficient_distribution, adj_2))
            )

        return self.mmd(clustering_coefficients_1, clustering_coefficients_2)

    def get_cluster_coefficient_distribution(self, adj: Tensor) -> Tensor:
        G = nx.from_numpy_array(adj.numpy())
        cluster_coefficients = torch.tensor(list(nx.clustering(G).values()), dtype=torch.float32)
        return torch.histogram(cluster_coefficients, bins=self.bins, range=(0, 1), density=self.normalize).hist


class LaplacianSpectrumMMD(nn.Module):
    def __init__(self, *args, bins: int = 100, normalize: bool = False, **kwargs):
        """Compute the MMD between the laplacian spectrum distributions of two sets of graphs.
        :param bins: The number of bins to use for the histogram.
        :param normalize: If True, the histograms are evaluated as density histograms.
        :param args: Arguments to pass to the MMD.
        :param kwargs: Keyword arguments to pass to the MMD."""
        super().__init__()
        self.bins = bins
        self.normalize = normalize
        self.mmd = MMD(*args, **kwargs)

    def forward(self, batch_1: GeometricBatch, batch_2: GeometricBatch) -> Tensor:
        """Compute the MMD between the laplacian spectrum distributions of two sets of graphs.
        :param batch_1: First batch of graphs.
        :param adj_2: Second batch of graphs.
        :return: The MMD between the laplacian spectrum distributions of the two sets of graphs.
        """
        adj_1 = geometric_to_adjacency(batch_1)
        adj_2 = geometric_to_adjacency(batch_2)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            laplacian_spectrum_1 = torch.stack(list(executor.map(self.get_laplacian_spectrum, adj_1)))
            laplacian_spectrum_2 = torch.stack(list(executor.map(self.get_laplacian_spectrum, adj_2)))

        return self.mmd(laplacian_spectrum_1, laplacian_spectrum_2)

    def get_laplacian_spectrum(self, adj: Tensor) -> Tensor:
        G = nx.from_numpy_array(adj.numpy())
        laplacian_spectrum = torch.tensor(nx.linalg.normalized_laplacian_spectrum(G))
        return torch.histogram(laplacian_spectrum, bins=self.bins, range=(0, 2), density=self.normalize).hist
