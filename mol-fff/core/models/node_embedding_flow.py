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
    condition_dim: int
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
    rff_norm: SetNormType | None = "set"
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
            condition_dim: int,
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
                condition_dim=condition_dim,
                heads=mab_heads,
                norm=mab_norm,
                remove_self_loops=mab_remove_self_loops,
                bias=mab_bias,
                condition_mode="element",
            )

            self.rff = RFF(
                input_dim=attention_dim,
                output_dim=output_dim,
                intermediate_dims=rff_intermediate_dims,
                condition_dim=condition_dim,
                activation=rff_activation,
                norm=rff_norm,
                dropout=rff_dropout,
            )

        def forward(
            self, x: Tensor, lengths: Tensor, c: Tensor | None = None
        ) -> Tensor:
            out = self.sab(x, lengths, c=c)
            out = self.rff(out, lengths, c=c)
            out = apply_masks(out, lengths)
            return out

    def __init__(self, hparams):
        super().__init__(hparams)
        self.blocks = self._configure_blocks(self._subnet_constructor)
        self.graph_autoencoder = self._configure_graph_autoencoder()

    def compute_metrics(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        metrics: dict[str, Tensor] = {}

        with torch.no_grad():
            h, lengths = self.embed(batch)

        # conditioning
        c = None
        c = self._apply_length_encoding(h, lengths, c)

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
        for block in self.blocks:
            z = block(z, lengths, c=c, rev=False)
        return z

    def decode(self, z: Tensor, lengths: Tensor, c: Tensor | None = None) -> Tensor:
        """Decodes a batch of latent representations to a batch of node embeddings.

        :param z: A batch of node embeddings.
        :param lengths: A batch of lengths.
        :param c: Conditions
        :return: A batch of node embeddings"""

        h = z
        for block in reversed(self.blocks):
            h = block(h, lengths, c=c, rev=True)
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
