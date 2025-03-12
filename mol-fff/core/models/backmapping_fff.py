from typing import Literal, Any, Callable

from itertools import chain

from rdkit.Chem.Draw import MolsToGridImage

import torch
from torch import nn, Tensor, Size
from torch.distributions import Distribution

from torch_geometric.data import Data, Batch
from torch_geometric.utils import unbatch

from hydrantic.model import ModelHparams, Model

from fff.loss import volume_change_surrogate
from fff.utils.func import compute_jacobian, compute_volume_change
from fff.utils.utils import sum_except_batch

from core.models.graph_autoencoder import GraphAutoencoder
from core.models.cross_modality_ae import CrossModalityAE
from core.components.set_attention import SAB
from core.components.set_rff import RFF
from core.components.global_attention import GlobalLinearAttention
from core.utils.masking import pad_sets, unpad_sets, apply_masks
from core.utils.graphs import dense_edge_index
from core.utils.molecules import get_molecule_from_data
from core.utils.sets import length_encoding
from core.utils.utils import import_from_string
from core.metrics.molecules import Validity, Components, Uniqueness
from core.metrics.mmd import MMD


class BackmappingFFFHparams(ModelHparams):
    dim: int
    latent_dim: int
    hidden_dim: int
    num_bead_classes: int
    max_bead_size: int
    mlp_widths: list[int]
    heads: int

    graph_autoencoder_ckpt: str

    latent_distribution: Literal["normal"] = "normal"
    num_blocks: int

    beta: float


class BackmappingFFF(Model):
    hparams_schema = BackmappingFFFHparams
    graph_autoencoder: CrossModalityAE

    class PermutationEquivariantLayer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            heads: int,
            mlp_widths: list[int],
        ):
            super().__init__()
            self.state_update = RFF(2 * hidden_dim, hidden_dim, mlp_widths, activation="torch.nn.ELU")
            self.interaction_sab = GlobalLinearAttention(dim=input_dim + hidden_dim, heads=heads, dim_head=hidden_dim)
            self.post_interaction_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.observation = RFF(hidden_dim, output_dim, mlp_widths, activation="torch.nn.ELU")

        def forward(self, x: Tensor, s: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
            sx = torch.concat((s, x), dim=-1)
            m, _ = self.interaction_sab(sx, sx, lengths)
            m = self.post_interaction_linear(m)
            x = x + self.observation(m, lengths)
            s = self.state_update(torch.concat((s, m), dim=-1), lengths)
            return x, s

    class ConditionalNormalDistribution(nn.Module, Distribution):
        def __init__(self, dim: int, condition_dim: int):
            super().__init__()
            self.mu_embedding = nn.Linear(condition_dim, dim)
            # self.sigma_embedding = nn.Linear(condition_dim, dim)
            self.dim = dim
            self.condition_dim = condition_dim

        def sample(self, sample_shape: Size, condition: Tensor):
            mu = self.mu_embedding(condition)
            sigma = torch.tensor([0.4], device=mu.device, dtype=mu.dtype)
            eps = torch.randn(sample_shape + (self.dim,), dtype=condition.dtype, device=condition.device)
            return mu + sigma * eps

        def log_prob(self, x: Tensor, condition: Tensor):
            mu = self.mu_embedding(condition)
            sigma = torch.tensor([0.4], device=mu.device, dtype=mu.dtype)
            log_p = (
                -(self.dim / 2) * torch.log(2 * torch.tensor(torch.pi))
                - (self.dim / 2) * torch.log(sigma**2)
                - (1 / (2 * sigma**2)) * (x - mu) ** 2
            )
            return log_p.sum(dim=-1)

    def __init__(self, hparams):
        super().__init__(hparams)
        self.encoder_layers = nn.ModuleList(
            [
                self.PermutationEquivariantLayer(
                    input_dim=self.hparams.dim,
                    output_dim=self.hparams.dim,
                    hidden_dim=self.hparams.hidden_dim,
                    heads=self.hparams.heads,
                    mlp_widths=self.hparams.mlp_widths,
                )
                for _ in range(self.hparams.num_blocks)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                self.PermutationEquivariantLayer(
                    input_dim=self.hparams.dim,
                    output_dim=self.hparams.dim,
                    hidden_dim=self.hparams.hidden_dim,
                    heads=self.hparams.heads,
                    mlp_widths=self.hparams.mlp_widths,
                )
                for _ in range(self.hparams.num_blocks)
            ]
        )
        self.bead_embedding = nn.Linear(self.hparams.num_bead_classes, self.hparams.hidden_dim)
        self.size_embedding = nn.Linear(self.hparams.max_bead_size, self.hparams.hidden_dim)
        self.graph_autoencoder = self._configure_graph_autoencoder()
        self.latent_distribution = self.ConditionalNormalDistribution(
            self.hparams.latent_dim, self.hparams.num_bead_classes
        )

    def compute_metrics(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        metrics: dict[str, Tensor] = {}

        with torch.no_grad():
            h, lengths = self.embed(batch)

        loss = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)

        c = batch.bead.unsqueeze(1).repeat(1, h.size(1), 1)

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
                z_gen = self.latent_distribution.sample(z.shape[:-1], c)
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
                        [MolsToGridImage(mols_gen[: min(len(mols_gen), 64)], molsPerRow=8)],
                    )
                except Exception as e:
                    print(e)
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

    def encode(self, h: Tensor, lengths: Tensor, c: Tensor | None = None) -> Tensor:
        """Encodes a batch a node embeddings to its latent representation.

        :param h: A batch of node embeddings.
        :param lengths: A batch of lengths.
        :param c: Conditions
        :return: A batch of latent node embeddings."""

        z = h
        s = self.bead_embedding(c) + self._size_embedding(lengths, z)
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
        s = self.bead_embedding(c) + self._size_embedding(lengths, h)
        for layer in self.decoder_layers:
            h, s = layer(h, s, lengths)
        return h

    def _size_embedding(self, lengths: Tensor, reference: Tensor) -> Tensor:
        one_hot = nn.functional.one_hot(lengths, num_classes=self.hparams.max_bead_size).to(dtype=reference.dtype)
        return self.size_embedding(one_hot).unsqueeze(1).repeat(1, reference.size(1), 1)

    def _configure_graph_autoencoder(self) -> CrossModalityAE:
        graph_autoencoder = CrossModalityAE.load_from_checkpoint(
            self.hparams.graph_autoencoder_ckpt, map_location=self.device
        )
        graph_autoencoder = graph_autoencoder.eval()
        graph_autoencoder = graph_autoencoder.requires_grad_(False)
        graph_autoencoder.freeze()
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

    def _get_encode_prefilled(self, lengths: Tensor, c: Tensor | None = None) -> Callable[[Tensor], Tensor]:
        return lambda x: self.encode(x, lengths, c)

    def _get_decode_prefilled(self, lengths: Tensor, c: Tensor | None = None) -> Callable[[Tensor], Tensor]:
        return lambda z: self.decode(z, lengths, c)

    def nll_surrogate(self, x: Tensor, lengths: Tensor, c: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Computes the negative log likelihood using the volume change surrogate.

        :param x: The input tensor.
        :param lengths: The lengths of the sets.
        :param c: The condition tensor.
        :return: The latent code, the reconstruction and the negative log-likelihood surrogate.
        """

        encode_prefilled = self._get_encode_prefilled(lengths, c)
        decode_prefilled = self._get_decode_prefilled(lengths, c)

        surrogate_output = volume_change_surrogate(x, encode_prefilled, decode_prefilled)

        z = surrogate_output.z
        x1 = surrogate_output.x1

        nll = -self.assemble_log_prob(z, surrogate_output.surrogate, lengths, c)

        return z, x1, nll

    def nll_exact(self, x: Tensor, lengths: Tensor, c: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Computes the exact negative log likelihood of the decoder.

        :param x: The input tensor.
        :param lengths: The lengths of the sets.
        :param c: The condition tensor.
        :return: The latent code, the reconstruction and the exact negative log-likelihood.
        """

        encode_prefilled = self._get_encode_prefilled(lengths, c)

        vmap_args = (lengths, c) if c is not None else (lengths,)

        z = encode_prefilled(x)

        x1, jac = compute_jacobian(
            z,
            self.decode,
            *vmap_args,
            chunk_size=self.hparams.chunk_size,  # type: ignore
            grad_type="backward",
        )
        vol_change = compute_volume_change(
            jac.view(jac.shape[0], jac.shape[1] * jac.shape[2], jac.shape[3] * jac.shape[4])
        )

        # use the negative volume change of the decoder to assemble the log prob
        nll = -self.assemble_log_prob(z, -vol_change, lengths, c)

        return z, x1, nll

    def assemble_log_prob(self, z: Tensor, vol_change: Tensor, lengths: Tensor, c: Tensor | None = None) -> Tensor:
        """Assembles the log probability given the volume change and the lengths.

        :param z: The latent tensor.
        :param vol_change: The volume change tensor.
        :param lengths: The lengths of the sets.
        :return: The log probability tensor."""

        original_shape = z.shape
        # z = z.view(-1, self.hparams.latent_dim)
        log_prob = self.latent_distribution.log_prob(z, c)

        log_prob = log_prob.view(*original_shape[:-1], -1)  # reshape to original
        log_prob = apply_masks(log_prob, lengths)  # mask out the padded elements
        log_prob = sum_except_batch(log_prob)  # sum over elements
        log_prob += vol_change  # add the set wise volume change

        return log_prob

    def latent_mmd(self, x: Tensor, lengths: Tensor, c: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Computes the latent MMD of the model, which can be used to monitor the learned latent distribution.

        :param x: The input tensor.
        :param lengths: The lengths of the sets.
        :param c: The condition tensor.
        :return: The latent code tensor and the mmd value."""

        z = self.encode(x, lengths, c)
        z_sampled = self.latent_distribution.sample(z.shape[:-1], c)
        mmd = MMD(
            kernel="multiscale",
            bandwidth_range=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0],
        )(z.reshape(z.shape[0], -1), z_sampled.view(z_sampled.shape[0], -1))
        return z, mmd
