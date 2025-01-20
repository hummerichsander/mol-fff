from typing import Tuple, List, Literal

import torch
from hydrantic.model import ModelHparams, Model
from torch import nn, Tensor
from torch_geometric import nn as gnn

from ..components.helper import Concatenate
from ..components.mlp import MLP
from ..components.norms.graph import GraphNormType, get_graph_norm
from ..components.norms.vector import VectorNormType, get_vector_norm
from ..metrics import GeometricPrecision, CrossEntropy, MMD
from ..utils.graphs import compute_number_of_connected_components, is_equal_graph
from ..utils.utils import import_from_string


class GraphAutoencoderHParams(ModelHparams):
    node_feature_dim: int
    node_feature_embedding_dim: int
    edge_feature_dim: int
    edge_feature_embedding_dim: int

    random_node_feature_dim: int = 0

    encoder_depth: int
    encoder_mlp_widths: list[int]
    encoder_aggr: str = "core.components.aggregations.VPA"
    encoder_node_normalization: GraphNormType = "graph"
    encoder_edge_normalization: VectorNormType = "layer"
    encoder_output_normalization: VectorNormType = None

    node_feature_decoder_mlp_widths: list[int]
    node_feature_decoder_normalization: VectorNormType = "layer"

    structure_decoder_mlp_node_widths: list[int]
    structure_decoder_mlp_edge_widths: list[int]
    structure_decoder_normalization: VectorNormType = "layer"

    noise: float = 0.0
    node_cross_entropy_beta: float = 1.0
    node_cross_entropy_kwargs: dict = {}
    edge_cross_entropy_beta: float = 10.0
    edge_cross_entropy_kwargs: dict = {}
    mmd_beta: float | None = None

    profile: Literal["qm9", "zinc", "unimers"] = "qm9"


class GraphAutoencoder(Model):
    hparams_schema = GraphAutoencoderHParams

    class EncoderLayer(gnn.MessagePassing):
        """Encoder layer for the split autoencoder."""

        def __init__(
                self,
                node_in_channels: int,
                node_out_channels: int,
                edge_in_channels: int,
                edge_out_channels: int,
                aggr: str | gnn.Aggregation = "core.components.aggregations.VPA",
                node_normalization: GraphNormType = "graph",
                edge_normalization: VectorNormType = "batch",
                mlp_widths: List[int] = [],
        ):
            if aggr.startswith("core.components.aggregations"):
                aggr = import_from_string(aggr)()

            super().__init__(aggr=aggr)

            self.node_normalization = get_graph_norm(
                node_normalization, node_out_channels
            )
            self.edge_normalization = get_vector_norm(
                edge_normalization, edge_out_channels
            )
            self.concat = Concatenate()
            self.mlp_edge_update = MLP(
                input_dim=(edge_in_channels + 2 * node_in_channels),
                output_dim=edge_out_channels,
                intermediate_dims=mlp_widths,
            )
            self.mlp_message = MLP(
                input_dim=(node_in_channels + edge_out_channels),
                output_dim=node_out_channels,
                intermediate_dims=mlp_widths,
            )
            self.mlp_update = MLP(
                input_dim=2 * node_in_channels,
                output_dim=node_out_channels,
                intermediate_dims=mlp_widths,
            )

        def forward(
                self,
                x: Tensor,
                edge_index: Tensor,
                edge_attr: Tensor,
                batch: Tensor,
        ) -> Tuple[Tensor, Tensor]:
            edge_attr = self.edge_updater(edge_index, edge_attr=edge_attr, x=x)
            if self.edge_normalization is not None:
                edge_attr = self.edge_normalization(edge_attr)

            x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            if self.node_normalization is not None:
                x = self.node_normalization(x, batch=batch)

            return x, edge_attr

        def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
            return self.mlp_update(self.concat(x, aggr_out))

        def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
            return self.mlp_message(self.concat(x_j, edge_attr))

        def edge_update(
                self, edge_index: Tensor, x_i: Tensor, x_j: Tensor, edge_attr: Tensor
        ) -> Tensor:
            return self.mlp_edge_update(self.concat(x_i, x_j, edge_attr))

    class Decoder(gnn.MessagePassing):
        """Decoder for the split autoencoder."""

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

            self.concat = Concatenate(dim=-1)
            self.structure_embedding = MLP(
                input_dim=in_channels,
                output_dim=in_channels,
                intermediate_dims=mlp_node_widths,
                norm=normalization,
            )
            self.edge_classifier = MLP(
                input_dim=2 * in_channels + 1,
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
            x_i = self.structure_embedding(x_i)
            x_j = self.structure_embedding(x_j)
            edge_update_in = self.concat(
                x_i, x_j, (x_i - x_j).pow(2).sum(-1, keepdim=True)
            )
            return self.edge_classifier(edge_update_in)

        def message(self, x_j: Tensor) -> Tensor:
            return x_j

        def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
            return self.node_classifier(aggr_out)

    def __init__(self, hparams: GraphAutoencoderHParams):
        super().__init__(hparams)

        # nn modules
        self.node_embedding_layer = nn.Linear(
            self.hparams.node_feature_dim + self.hparams.random_node_feature_dim,
            self.hparams.node_feature_embedding_dim,
        )
        self.edge_embedding_layer = nn.Linear(
            self.hparams.edge_feature_dim, self.hparams.edge_feature_embedding_dim
        )
        self.encoder_layers = nn.ModuleList(
            [
                self.EncoderLayer(
                    node_in_channels=self.hparams.node_feature_embedding_dim,
                    node_out_channels=self.hparams.node_feature_embedding_dim,
                    edge_in_channels=self.hparams.edge_feature_embedding_dim,
                    edge_out_channels=self.hparams.edge_feature_embedding_dim,
                    aggr=self.hparams.encoder_aggr,
                    node_normalization=self.hparams.encoder_node_normalization,
                    edge_normalization=self.hparams.encoder_edge_normalization,
                    mlp_widths=self.hparams.encoder_mlp_widths,
                )
                for _ in range(self.hparams.encoder_depth)
            ]
        )
        if (norm := self.hparams.encoder_output_normalization) is not None:
            self.encoder_output_norm = get_vector_norm(norm, self.hparams.node_feature_embedding_dim)

        self.decoder = self.Decoder(
            in_channels=self.hparams.node_feature_embedding_dim,
            node_out_channels=self.hparams.node_feature_dim,
            edge_out_channels=self.hparams.edge_feature_dim,
            normalization=self.hparams.structure_decoder_normalization,
            mlp_node_widths=self.hparams.structure_decoder_mlp_node_widths,
            mlp_edge_widths=self.hparams.structure_decoder_mlp_edge_widths,
        )

        # losses and metrics
        self.node_cross_entropy = CrossEntropy(
            attr="x", **self.hparams.node_cross_entropy_kwargs
        )
        self.edge_cross_entropy = CrossEntropy(
            attr="edge_attr", **self.hparams.edge_cross_entropy_kwargs
        )
        self.node_precision = GeometricPrecision(attr="x")
        self.edge_precision = GeometricPrecision(attr="edge_attr")

    def encode(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Tensor | None = None,
    ) -> Tensor:
        """Encode the input graph as a set of node embeddings.

        :param x: Node features.
        :param edge_index: Edge indices.
        :param edge_attr: Edge features.
        :param batch: Batch indices.
        :return: A tuple of node embeddings and edge embeddings."""

        if self.hparams.random_node_feature_dim > 0:
            x = torch.cat(
                (
                    x,
                    torch.randn(
                        *x.shape[:-1],
                        self.hparams.random_node_feature_dim,
                        dtype=x.dtype,
                        device=x.device,
                    ),
                ),
                dim=-1,
            )
        x = self.node_embedding_layer(x)
        edge_attr = self.edge_embedding_layer(edge_attr)
        for layer in self.encoder_layers:
            x, edge_attr = layer(x, edge_index, edge_attr, batch)
        if hasattr(self, "encoder_output_norm"):
            x = self.encoder_output_norm(x)
        return x

    def decode(
            self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Decode the input graph from node embeddings.

        :param x: Node embeddings.
        :param edge_index: Edge indices.
        :param batch: Batch indices.
        :return: A tuple of node features and edge features."""

        x, edge_attr = self.decoder(x, edge_index, batch)
        return x, edge_attr

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Tensor | None = None,
    ):
        x = self.encode(x, edge_index, edge_attr, batch)
        x, edge_attr = self.decode(x, edge_index, batch)
        return x, edge_attr

    def compute_metrics(self, batch, batch_idx) -> dict:
        metrics: dict[str, Tensor] = {"loss": torch.tensor(0.0, device=self.device)}

        x_code = self.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        if (noise := self.hparams.noise) and self.training:
            x_code += torch.randn_like(x_code) * noise

        x1, edge_attr1 = self.decode(x_code, batch.edge_index, batch.batch)

        batch1 = batch.clone()
        batch1.x = x1
        batch1.edge_attr = edge_attr1

        metrics["node_cross_entropy"] = self.node_cross_entropy(batch, batch1)
        if self.hparams.node_cross_entropy_beta > 0:
            metrics["loss"] += (
                    self.hparams.node_cross_entropy_beta * metrics["node_cross_entropy"]
            )

        metrics["edge_cross_entropy"] = self.edge_cross_entropy(batch, batch1)
        if self.hparams.edge_cross_entropy_beta > 0:
            metrics["loss"] += (
                    self.hparams.edge_cross_entropy_beta * metrics["edge_cross_entropy"]
            )

        if not self.training:
            metrics["node_precision"] = self.node_precision(batch, batch1)
            metrics["edge_precision"] = self.edge_precision(batch, batch1)
            metrics["num_components_reconstruction"] = (
                compute_number_of_connected_components(batch1).mean()
            )

            fully_reconstructed: int = 0
            for i in range(len(batch)):
                fully_reconstructed += int(is_equal_graph(batch[i], batch1[i]))
            metrics["molecule_precision"] = fully_reconstructed / len(batch)

        if self.hparams.mmd_beta:
            metrics["mmd"] = MMD()(x_code, torch.randn_like(x_code))
            metrics["loss"] += self.hparams.mmd_beta * metrics["mmd"]

        return metrics
