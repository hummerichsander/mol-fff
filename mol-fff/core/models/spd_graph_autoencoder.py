from typing import List

from torch import nn, Tensor
from torch_geometric import nn as gnn

from .graph_autoencoder import GraphAutoencoderHparams, GraphAutoencoder
from ..components.helper import Concatenate
from ..components.mlp import MLP
from ..components.norms.vector import VectorNormType
from ..metrics import GeometricPrecision, CrossEntropy
from ..geometry.spd import SPD


class SPDGraphAutoencoder(GraphAutoencoder):
    class Decoder(gnn.MessagePassing):
        """Decoder."""

        def __init__(
            self,
            in_channels: int,
            node_out_channels: int,
            edge_out_channels: int,
            normalization: VectorNormType = None,
            mlp_node_widths: List[int] = [],
            mlp_edge_widths: List[int] = [],
        ):
            super().__init__()

            self.manifold = SPD(SPD.n_from_d(in_channels), metric_type="vector_valued")
            self.metric = self.manifold.get_metric()
            self.concat = Concatenate(dim=-1)
            self.structure_embedding = MLP(
                input_dim=in_channels,
                output_dim=in_channels,
                intermediate_dims=mlp_node_widths,
                norm=normalization,
            )
            self.edge_classifier = MLP(
                input_dim=2 * in_channels + self.metric.vvd_dim,
                output_dim=edge_out_channels,
                intermediate_dims=mlp_edge_widths,
                norm=normalization,
            )
            self.node_classifier = MLP(
                input_dim=in_channels,
                output_dim=node_out_channels,
                intermediate_dims=mlp_node_widths,
            )

        def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            batch: Tensor,
        ) -> tuple[Tensor, Tensor]:
            edges = self.edge_updater(edge_index, x=x)
            nodes = self.propagate(edge_index, x=x)
            return nodes, edges

        def edge_update(self, edge_index: Tensor, x_i: Tensor, x_j: Tensor) -> Tensor:
            x_i = self.manifold.cholesky_composition(self.structure_embedding(x_i))
            x_j = self.manifold.cholesky_composition(self.structure_embedding(x_j))
            edge_update_in = self.concat(x_i, x_j, self.metric.vvd(x_i, x_j))
            return self.edge_classifier(edge_update_in)

        def message(self, x_j: Tensor) -> Tensor:
            return x_j

        def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
            return self.node_classifier(aggr_out)
