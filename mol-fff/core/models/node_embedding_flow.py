from typing import Literal, Any, Callable

import torch
from torch import nn, Tensor

from torch_geometric.data import Data, Batch
from torch_geometric.utils import unbatch

from fff.utils.utils import sum_except_batch

from hydrantic.model import ModelHparams, Model

from .graph_autoencoder import GraphAutoencoder
from ..mixins.set_fff import SetFreeFormFlowMixin
from ..mixins.length_encoding import LengthEncodingMixin
from ..components.norms.set import SetNormType
from ..components.set_attention import SAB, MAB
from ..components.set_rff import RFF
from ..components.set_all_in_one_coupling import SetAllInOneBlock
from ..utils.masking import apply_masks, pad_sets, unpad_sets
from ..utils.graphs import dense_edge_index
from ..utils.molecules import get_molecule_from_data
from ..utils.sets import length_encoding
from ..metrics.molecules import Validity, Components, Uniqueness


class NodeEmbeddingFlowHparams(ModelHparams):
    dim: int
    latent_dim: int
    attention_dim: int
    condition_dims: list[int]
    length_encoding_dim: int

    graph_autoencoder_ckpt: str

    latent_distribution: Literal["normal"] = "normal"
    num_blocks: int

    mab_heads: int
    mab_norm: SetNormType = "set"
    mab_remove_self_loops: bool
    mab_bias: bool = False
    rff_intermediate_dims: list[int]
    rff_activation: str = "torch.nn.ReLU"
    rff_norm: SetNormType = "set"
    rff_dropout: float = 0.0


class NodeEmbeddingFlow(Model, SetFreeFormFlowMixin, LengthEncodingMixin):
    hparams_schema = NodeEmbeddingFlowHparams
    graph_autoencoder: GraphAutoencoder

    class SelfAttentionNet(nn.Module):
        def __init__(
                self,
                input_dim: int,
                output_dim: int,
                attention_dim: int,
                mab_heads: int,
                mab_norm: SetNormType = "set",
                mab_remove_self_loops: bool = True,
                mab_bias: bool = False,
                rff_intermediate_dims: list[int] = [],
                rff_activation: str = "torch.nn.ReLU",
                rff_norm: SetNormType = "set",
                rff_dropout: float = 0.0,
        ):
            super().__init__()

            self.sab = SAB(
                input_dim=input_dim,
                output_dim=attention_dim,
                heads=mab_heads,
                norm=mab_norm,
                remove_self_loops=mab_remove_self_loops,
                bias=mab_bias,
            )

            self.rff = RFF(
                input_dim=attention_dim,
                output_dim=output_dim,
                intermediate_dims=rff_intermediate_dims,
                activation=rff_activation,
                norm=rff_norm,
                dropout=rff_dropout,
            )

        def forward(self, x: Tensor, lengths: Tensor, c: list[Tensor] | None = None) -> Tensor:
            out = self.sab(x, lengths)
            out = self.rff(out, lengths)
            out = apply_masks(out, lengths)
            return out

    class ConditionAttentionNet(nn.Module):
        def __init__(
                self,
                input_dim: int,
                output_dim: int,
                attention_dim: int,
                condition_dims: list[int],
                mab_heads: int,
                mab_norm: SetNormType = "set",
                rff_intermediate_dims: list[int] = [],
                rff_activation: str = "torch.nn.ReLU",
                rff_norm: SetNormType = "set",
                rff_dropout: float = 0.0,
        ):
            super().__init__()

            self.c_embeddings = nn.ModuleList(
                [
                    nn.Linear(c_dim, max(condition_dims), bias=False)
                    for c_dim in condition_dims
                ]
            )

            self.mab = MAB(
                dim_Q=input_dim,
                dim_K=max(condition_dims),
                dim_V=attention_dim,
                heads=mab_heads,
                norm=mab_norm,
                remove_self_loops=False,
                bias=False,
            )

            self.rff = RFF(
                input_dim=attention_dim,
                output_dim=output_dim,
                intermediate_dims=rff_intermediate_dims,
                activation=rff_activation,
                norm=rff_norm,
                dropout=rff_dropout,
            )

        def forward(self, x: Tensor, lengths: Tensor, c: list[Tensor]) -> Tensor:
            c = torch.cat([self.c_embeddings[i](c[i]).unsqueeze(1) for i in range(len(c))], dim=1)
            out = self.mab(x, c, lengths)
            out = self.rff(out, lengths)
            out = apply_masks(out, lengths)
            return out

    def __init__(self, hparams):
        super().__init__(hparams)
        self.blocks = self._configure_blocks(self._subnet_constructor)
        if self.hparams.condition_dims is not None:
            self.c_blocks = self._configure_blocks(self._c_subnet_constructor)
        self.graph_autoencoder = self._configure_graph_autoencoder()

    def compute_metrics(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        metrics: dict[str, Tensor] = {}

        with torch.no_grad():
            h, lengths = self.embed(batch)

        # conditioning
        c = None
        if (led := self.hparams.length_encoding_dim) > 0:
            le = length_encoding(lengths, led, h.device, h.dtype)
            c = [le]

        loss = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)

        if self.training:
            h.requires_grad_()
            z, _, nll = self.nll_surrogate(h, lengths, c=c)
            metrics["nll"] = sum_except_batch(nll).mean()
            loss += nll

        else:
            z, mmd = self.latent_mmd(h, lengths, c=c)
            metrics["mmd"] = mmd
            loss += mmd

            z_gen = self.sample_z(z.shape[:-1], z.device, z.dtype)
            h_gen = self.decode(z_gen, lengths, c=c)
            batch_gen = self.reconstruct(h_gen, lengths)
            mols_gen = [
                get_molecule_from_data(
                    batch_gen[i].x,
                    batch_gen[i].edge_index,
                    batch_gen[i].edge_attr,
                    profile=self.graph_autoencoder.hparams.profile
                )
                for i in range(batch.num_graphs)
            ]
            metrics["validity"] = torch.tensor(Validity()(mols_gen))
            metrics["components"] = torch.tensor(Components()(mols_gen))
            metrics["uniqueness"] = torch.tensor(Uniqueness()(mols_gen))

        metrics["loss"] = loss.mean()

        return metrics

    def embed(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Embeds a batch of graphs as a set of node embeddings.

        :param batch: A batch of graphs.
        :return: A batch of node embeddings and the corresponding lengths."""

        h_graph = self.graph_autoencoder.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        h_list = unbatch(h_graph, batch.batch)
        h, lengths = pad_sets(h_list)
        return h, lengths

    def reconstruct(self, h: Tensor, lengths) -> Batch:
        """Reconstructs the graph structures from a set of node embeddings.

        :param h: A batch of node embeddings.
        :param lengths: A batch of lengths.
        :return: A batch of graph structures."""

        batch_code = self.set_to_graph(h, lengths)
        x1, edge_attr1 = self.graph_autoencoder.decode(batch_code.x, batch_code.edge_index, batch_code.batch)
        batch1 = batch_code.clone()
        batch1.x = x1
        batch1.edge_attr = edge_attr1
        return batch1

    def encode(self, h: Tensor, lengths: Tensor, c: list[Tensor] | None = None) -> Tensor:
        """Encodes a batch a node embeddings to its latent representation.

        :param h: A batch of node embeddings.
        :param lengths: A batch of lengths.
        :param c: Conditions
        :return: A batch of latent node embeddings."""

        z = h
        if hasattr(self, "c_blocks"):
            for c_block, block in zip(self.c_blocks, self.blocks):
                z = c_block(z, lengths, c)
                z = block(z, lengths)
        else:
            for block in self.blocks:
                z = block(z, lengths, rev=False)
        return z

    def decode(self, z: Tensor, lengths: Tensor, c: list[Tensor] | None = None) -> Tensor:
        """Decodes a batch of latent representations to a batch of node embeddings.

        :param z: A batch of node embeddings.
        :param lengths: A batch of lengths.
        :param c: Conditions
        :return: A batch of node embeddings"""

        h = z
        if hasattr(self, "c_blocks"):
            for c_block, block in zip(reversed(self.c_blocks), reversed(self.blocks)):
                h = block(z, lengths, rev=True)
                h = c_block(h, lengths, c, rev=True)
        else:
            for block in reversed(self.blocks):
                h = block(h, lengths, rev=True)
        return h

    def _subnet_constructor(self, channels_in: int, channels_out: int) -> nn.Module:
        return self.SelfAttentionNet(
            input_dim=channels_in,
            output_dim=channels_out,
            attention_dim=self.hparams.attention_dim,
            mab_heads=self.hparams.mab_heads,
            mab_norm=self.hparams.mab_norm,
            mab_remove_self_loops=self.hparams.mab_remove_self_loops,
            mab_bias=self.hparams.mab_bias,
            rff_intermediate_dims=self.hparams.rff_intermediate_dims,
            rff_activation=self.hparams.rff_activation,
            rff_norm=self.hparams.rff_norm,
            rff_dropout=self.hparams.rff_dropout,
        )

    def _c_subnet_constructor(self, channels_in: int, channels_out: int) -> nn.Module:
        return self.ConditionAttentionNet(
            input_dim=channels_in,
            output_dim=channels_out,
            attention_dim=self.hparams.attention_dim,
            condition_dims=self.hparams.condition_dims,
            mab_heads=1,
            mab_norm=self.hparams.mab_norm,
            rff_intermediate_dims=self.hparams.rff_intermediate_dims,
            rff_activation=self.hparams.rff_activation,
            rff_norm=self.hparams.rff_norm,
            rff_dropout=self.hparams.rff_dropout,
        )

    def _configure_blocks(self, subnet_constructor: Callable[[int, int], nn.Module]) -> nn.ModuleList:
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
                edge_attr=torch.empty((lengths[i] ** 2 - lengths[i], self.num_bond_classes), device=x.device),
            ) for i in range(len(x_list))
        ]
        return Batch.from_data_list(geom_data_list)
