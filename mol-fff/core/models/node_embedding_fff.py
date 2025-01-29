from typing import Literal, Any, Callable

from rdkit.Chem.Draw import MolsToGridImage

import torch
from torch import nn, Tensor

from torch_geometric.data import Data, Batch
from torch_geometric.utils import unbatch

from fff.utils.utils import sum_except_batch

from hydrantic.model import ModelHparams, Model

from .graph_autoencoder import GraphAutoencoder
from ..mixins.set_fff import SetFreeFormFlowMixin
from ..mixins.length_encoding import LengthEncodingMixin
from ..components.set_attention import SAB
from ..components.set_rff import RFF
from ..components.set_all_in_one_coupling import SetAllInOneBlock
from ..utils.masking import pad_sets, unpad_sets
from ..utils.graphs import dense_edge_index
from ..utils.molecules import get_molecule_from_data
from ..utils.sets import length_encoding
from ..metrics.molecules import Validity, Components, Uniqueness


class NodeEmbeddingFFFHparams(ModelHparams):
    dim: int
    latent_dim: int
    hidden_dim: int
    rnf_dim: int = 0
    mlp_widths: list[int]
    heads: int

    graph_autoencoder_ckpt: str

    latent_distribution: Literal["normal"] = "normal"
    num_blocks: int

    beta: float


class NodeEmbeddingFFF(Model, SetFreeFormFlowMixin, LengthEncodingMixin):
    hparams_schema = NodeEmbeddingFFFHparams
    graph_autoencoder: GraphAutoencoder

    class HiddenFeatureInteraction(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            heads: int,
            mlp_widths: list[int],
        ):
            super().__init__()
            self.state_update = RFF(
                hidden_dim + input_dim,
                hidden_dim,
                mlp_widths,
                activation="torch.nn.SiLU",
                norm="set",
            )
            self.interaction = SAB(
                hidden_dim,
                hidden_dim,
                heads,
                remove_self_loops=False,
                bias=False,
                norm="set",
            )
            self.observation = RFF(
                input_dim + hidden_dim,
                output_dim,
                mlp_widths,
                activation="torch.nn.SiLU",
                norm="set",
            )

            self._reset_parameters()

        def forward(
            self, x: Tensor, s: Tensor, lengths: Tensor
        ) -> tuple[Tensor, Tensor]:
            s = self.state_update(torch.concat((s, x), dim=-1), lengths)
            x = self.observation(
                torch.concat((x, self.interaction(s, lengths)), dim=-1), lengths
            )
            return x, s

        def _reset_parameters(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def __init__(self, hparams):
        super().__init__(hparams)
        self.encoder_layers = nn.ModuleList(
            [
                self.HiddenFeatureInteraction(
                    self.hparams.dim,
                    self.hparams.dim,
                    self.hparams.hidden_dim,
                    heads=self.hparams.heads,
                    mlp_widths=self.hparams.mlp_widths,
                )
                for _ in range(self.hparams.num_blocks)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                self.HiddenFeatureInteraction(
                    self.hparams.dim,
                    self.hparams.dim,
                    self.hparams.hidden_dim,
                    heads=self.hparams.heads,
                    mlp_widths=self.hparams.mlp_widths,
                )
                for _ in range(self.hparams.num_blocks)
            ]
        )
        self.graph_autoencoder = self._configure_graph_autoencoder()

    def compute_metrics(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        metrics: dict[str, Tensor] = {}

        with torch.no_grad():
            h, lengths = self.embed(batch)

        c = None

        loss = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)

        if self.training:
            h.requires_grad_()
            z, h1, nll = self.nll_surrogate(h, lengths, c=c)
            metrics["nll"] = sum_except_batch(nll).mean()
            mse = sum_except_batch((h - h1).pow(2))
            metrics["mse"] = mse.mean()
            loss += nll + self.hparams.beta * mse

        else:
            z, mmd = self.latent_mmd(h, lengths, c=c)
            h1 = self.decode(z, lengths, c=c)
            mse = sum_except_batch((h - h1).pow(2))
            metrics["mse"] = mse.mean()
            metrics["mmd"] = mmd
            loss += mmd

            if batch_idx == 0:
                z_gen = self.sample_z(z.shape[:-1], z.device, z.dtype)
                h_gen = self.decode(z_gen, lengths, c=c)
                batch_gen = self.reconstruct(h_gen, lengths)
                mols_gen = [
                    get_molecule_from_data(
                        batch_gen[i].x,
                        batch_gen[i].edge_index,
                        batch_gen[i].edge_attr,
                        profile=self.graph_autoencoder.hparams.profile,
                    )
                    for i in range(batch.num_graphs)
                ]
                metrics["validity"] = torch.tensor(Validity()(mols_gen))
                metrics["components"] = torch.tensor(Components()(mols_gen))
                metrics["uniqueness"] = torch.tensor(Uniqueness()(mols_gen))

                try:
                    self.logger.log_image(
                        "mols_gen",
                        [
                            MolsToGridImage(
                                mols_gen[: min(len(mols_gen), 64)], molsPerRow=8
                            )
                        ],
                    )
                except Exception as e:
                    print(e)
        metrics["loss"] = loss.mean()

        return metrics

    def embed(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Embeds a batch of graphs as a set of node embeddings.

        :param batch: A batch of graphs.
        :return: A batch of node embeddings and the corresponding lengths."""

        h_graph = self.graph_autoencoder.encode(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        h_list = unbatch(h_graph, batch.batch)
        h, lengths = pad_sets(h_list)
        return h, lengths

    def reconstruct(self, h: Tensor, lengths) -> Batch:
        """Reconstructs the graph structures from a set of node embeddings.

        :param h: A batch of node embeddings.
        :param lengths: A batch of lengths.
        :return: A batch of graph structures."""

        batch_code = self.set_to_graph(h, lengths)
        x1, edge_attr1 = self.graph_autoencoder.decode(
            batch_code.x, batch_code.edge_index, batch_code.batch
        )
        batch1 = batch_code.clone()
        batch1.x = x1
        batch1.edge_attr = edge_attr1
        return batch1

    def encode(self, h: Tensor, lengths: Tensor, c: Tensor | None = None) -> Tensor:
        """Encodes a batch a node embeddings to its latent representation.

        :param h: A batch of node embeddings.
        :param lengths: A batch of lengths.
        :param c: Conditions
        :return: A batch of latent node embeddings."""

        z = h
        s = (
            length_encoding(lengths, self.hparams.hidden_dim - self.hparams.rnf_dim, z.device, z.dtype)
            .unsqueeze(1)
            .repeat(1, z.shape[1], 1)
        )
        s = s.concat((s, torch.randn((*z.shape[:-1], self.rnf_dim))), dim=-1)
        for layer in self.encoder_layers:
            z, s = layer(z, s, lengths)
        return z

    def decode(self, z: Tensor, lengths: Tensor, c: Tensor | None = None) -> Tensor:
        """Decodes a batch of latent representations to a batch of node embeddings.

        :param z: A batch of node embeddings.
        :param lengths: A batch of lengths.
        :param c: Conditions
        :return: A batch of node embeddings"""

        h = z
        s = (
            length_encoding(lengths, self.hparams.hidden_dim - self.hparams.rnf_dim, z.device, z.dtype)
            .unsqueeze(1)
            .repeat(1, z.shape[1], 1)
        )
        s = s.concat((s, torch.randn((*z.shape[:-1], self.rnf_dim))), dim=-1)
        for layer in self.decoder_layers:
            h, s = layer(h, s, lengths)
        return h

    def _subnet_constructor(self, channels_in: int, channels_out: int) -> nn.Module:
        return self.SelfAttentionNet(
            input_dim=channels_in,
            output_dim=channels_out,
            attention_dim=self.hparams.attention_dim,
            condition_dim=self.hparams.condition_dim,
            mab_heads=self.hparams.mab_heads,
            mab_norm=self.hparams.mab_norm,
            mab_remove_self_loops=self.hparams.mab_remove_self_loops,
            mab_bias=self.hparams.mab_bias,
            rff_intermediate_dims=self.hparams.rff_intermediate_dims,
            rff_activation=self.hparams.rff_activation,
            rff_norm=self.hparams.rff_norm,
            rff_dropout=self.hparams.rff_dropout,
        )

    def _configure_blocks(
        self, subnet_constructor: Callable[[int, int], nn.Module]
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for _ in range(self.hparams.num_blocks):
            blocks.append(
                SetAllInOneBlock(
                    dims_in=[(self.hparams.dim,)],
                    subnet_constructor=subnet_constructor,
                    affine_clamping=2.0,
                    global_affine_init=1.0,
                    global_affine_type="SOFTPLUS",
                    permute_soft=True,
                    learned_householder_permutation=0,
                    reverse_permutation=False,
                )
            )

        return blocks

    def _configure_graph_autoencoder(self) -> GraphAutoencoder:
        graph_autoencoder = GraphAutoencoder.load_from_checkpoint(
            self.hparams.graph_autoencoder_ckpt, map_location=self.device
        )
        graph_autoencoder = graph_autoencoder.eval()
        return graph_autoencoder

    @property
    def num_bond_classes(self) -> int:
        """Number of bond classes."""
        num_edge_classes: dict[str, int] = {
            "qm9": 5,
            "pcqm4m": 5,
            "zinc": 4,
            "unimers": 4,
        }
        return num_edge_classes[self.graph_autoencoder.hparams.profile]

    @property
    def num_atom_classes(self) -> int:
        """Number of atom classes."""
        num_atom_classes: dict[str, int] = {
            "qm9": 9,
            "pcqm4m": 36,
            "zinc": 28,
            "unimers": 4,
        }
        return num_atom_classes[self.graph_autoencoder.hparams.profile]

    def graph_to_set(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Transforms a batch of graphs to a batch of sets.

        :param batch: batch of graphs to transform
        :return: batch of sets"""

        x_list = unbatch(batch.x, batch.batch)
        x_list, lengths = pad_sets(x_list)
        return x_list, lengths

    def set_to_graph(self, x: Tensor, lengths: Tensor) -> Batch:
        """Transforms a batch of sets to a batch of fully connected graphs with empty edge attributes.

        :param x: batch of sets to transform
        :param lengths: batch of sets lengths
        :return: batch of sets"""

        x_list = unpad_sets(x, lengths)
        geom_data_list = [
            Data(
                x=x_list[i],
                edge_index=dense_edge_index(lengths[i].item(), x.device),
                edge_attr=torch.empty(
                    (lengths[i] ** 2 - lengths[i], self.num_bond_classes),
                    device=x.device,
                ),
            )
            for i in range(len(x_list))
        ]
        return Batch.from_data_list(geom_data_list)
