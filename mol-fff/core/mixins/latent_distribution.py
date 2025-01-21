from typing import Literal

import torch
from torch import Tensor, Size
from torch.distributions import Distribution, Normal, Independent


class LatentDistributionMixin:
    """Implements some core functionality to sample from a latent distribution."""

    class hparams:
        latent_dim: int
        latent_distribution: Literal["normal"]

    def sample_z(self, shape: Size, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        """Samples from the latent distribution.

        :param shape: Shape of the samples.
        :param device: Device to create the samples on.
        :param dtype: Dtype of the samples.
        :return: Samples of shape (n_samples, dim)"""

        latent_distribution = self._get_latent_distribution(device, dtype)
        return latent_distribution.sample(shape)

    def _get_latent_distribution(
            self, device: torch.device, dtype: torch.dtype
    ) -> Distribution:
        """Returns the latent distribution specified in the hparams on the given device.

        :param device: Device to create the distribution on.
        :return: The latent distribution."""

        match self.hparams.latent_distribution:
            case "normal":
                return Independent(
                    Normal(
                        torch.zeros(
                            self.hparams.latent_dim, device=device, dtype=dtype
                        ),
                        torch.ones(self.hparams.latent_dim, device=device, dtype=dtype),
                    ),
                    1,
                )

        raise ValueError(f"Unknown latent distribution: {self.hparams.latent_distribution}")
