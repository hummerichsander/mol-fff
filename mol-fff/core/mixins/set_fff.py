from typing import Callable

from torch import Tensor

from fff.loss import volume_change_surrogate
from fff.utils.func import compute_jacobian, compute_volume_change
from fff.utils.utils import sum_except_batch

from ..utils.masking import apply_masks
from ..metrics.mmd import MMD

from .latent_distribution import LatentDistributionMixin


class SetFreeFormFlowMixin(LatentDistributionMixin):
    """Implements the free form loss for set structured data."""

    class hparams:
        latent_dim: int
        chunk_size: Tensor | None = None

    def encode(
            self, x: Tensor, lengths: Tensor, *args, c: Tensor | None = None, **kwargs
    ) -> Tensor: ...

    def decode(
            self, z: Tensor, lengths: Tensor, *args, c: Tensor | None = None, **kwargs
    ) -> Tensor: ...

    def _get_encode_prefilled(
            self, lengths: Tensor, c: Tensor | None = None
    ) -> Callable[[Tensor], Tensor]:
        return lambda x: self.encode(x, lengths, c)

    def _get_decode_prefilled(
            self, lengths: Tensor, c: Tensor | None = None
    ) -> Callable[[Tensor], Tensor]:
        return lambda z: self.decode(z, lengths, c)

    def nll_surrogate(
            self, x: Tensor, lengths: Tensor, c: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Computes the negative log likelihood using the volume change surrogate.

        :param x: The input tensor.
        :param lengths: The lengths of the sets.
        :param c: The condition tensor.
        :return: The latent code, the reconstruction and the negative log-likelihood surrogate."""

        encode_prefilled = self._get_encode_prefilled(lengths, c)
        decode_prefilled = self._get_decode_prefilled(lengths, c)

        surrogate_output = volume_change_surrogate(
            x, encode_prefilled, decode_prefilled
        )

        z = surrogate_output.z
        x1 = surrogate_output.x1

        nll = -self.assemble_log_prob(z, surrogate_output.surrogate, lengths)

        return z, x1, nll

    def nll_exact(
            self, x: Tensor, lengths: Tensor, c: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Computes the exact negative log likelihood of the decoder.

        :param x: The input tensor.
        :param lengths: The lengths of the sets.
        :param c: The condition tensor.
        :return: The latent code, the reconstruction and the exact negative log-likelihood."""

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
            jac.view(
                jac.shape[0], jac.shape[1] * jac.shape[2], jac.shape[3] * jac.shape[4]
            )
        )

        # use the negative volume change of the decoder to assemble the log prob
        nll = -self.assemble_log_prob(z, -vol_change, lengths)

        return z, x1, nll

    def assemble_log_prob(
            self, z: Tensor, vol_change: Tensor, lengths: Tensor
    ) -> Tensor:
        """Assembles the log probability given the volume change and the lengths.

        :param z: The latent tensor.
        :param vol_change: The volume change tensor.
        :param lengths: The lengths of the sets.
        :return: The log probability tensor."""

        latent_distribution = self._get_latent_distribution(z.device, z.dtype)

        original_shape = z.shape
        z = z.view(-1, self.hparams.latent_dim)
        log_prob = latent_distribution.log_prob(z)

        log_prob = log_prob.view(*original_shape[:-1], -1)  # reshape to original
        log_prob = apply_masks(log_prob, lengths)  # mask out the padded elements
        log_prob = sum_except_batch(log_prob)  # sum over elements
        log_prob += vol_change  # add the set wise volume change

        return log_prob

    def latent_mmd(
            self, x: Tensor, lengths: Tensor, c: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Computes the latent MMD of the model, which can be used to monitor the learned latent distribution.

        :param x: The input tensor.
        :param lengths: The lengths of the sets.
        :param c: The condition tensor.
        :return: The latent code tensor and the mmd value."""

        z = self.encode(x, lengths, c)
        z_sampled = self.sample_z(z.shape[:-1], z.device, z.dtype)
        mmd = MMD(kernel="multiscale", bandwidth_range=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])(
            z.reshape(z.shape[0], -1), z_sampled.view(z_sampled.shape[0], -1)
        )
        return z, mmd
