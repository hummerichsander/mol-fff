from typing import Optional, Literal

import torch
from torch import nn, Tensor

from .norms import get_set_norm
from ..utils.masking import apply_masks
from ..utils.utils import import_from_string


class RFF(nn.Module):
    """Row-wise feed forward network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        intermediate_dims: list[int] = [],
        condition_dim: int = 0,
        activation: str = "torch.nn.ReLU",
        norm: Literal["layer", "set"] | None = None,
        dropout: float = 0.0,
    ):
        super(RFF, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.intermediate_dims = intermediate_dims
        self.condition_dim = condition_dim
        self.activation = activation
        self.norm = norm
        self.dropout = dropout
        self.activation = activation
        self.dropout = dropout

        self.network = self._configure_network()
        self._reset_parameters()

    def forward(self, x: Tensor, lengths: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the RFF.

        :param x: Input tensor of shape (batch_size, set_size, input_dim)
        :param lengths: Lengths of the sets of shape (batch_size,)
        :param c: Condition tensor of shape (batch_size, set_size, condition_dim)
        :return: Output tensor of shape (batch_size, set_size, output_dim)"""

        for i in range(len(self.network["linear_layers"])):
            if c is not None:
                x = torch.concat([x, c], dim=-1)
            x = self.network["linear_layers"][i](x)
            if i < len(self.network["linear_layers"]) - 1:
                if len(self.network["normalization_layers"]) > 0:
                    x = self.network["normalization_layers"][i](x, lengths=lengths)
                x = self.network["activation"](x)
                x = self.network["dropout"](x)
        x = apply_masks(x, lengths)
        return x

    def _configure_network(self):
        network = nn.ModuleDict()
        network["linear_layers"] = nn.ModuleList()
        network["normalization_layers"] = nn.ModuleList()
        network["dropout"] = nn.Dropout(self.dropout)
        network["activation"] = import_from_string(self.activation)()

        dims = [self.input_dim] + self.intermediate_dims + [self.output_dim]

        for i in range(len(dims) - 2):
            network["linear_layers"].append(nn.Linear(dims[i] + self.condition_dim, dims[i + 1]))
            if norm := get_set_norm(self.norm, dims[i + 1]):
                network["normalization_layers"].append(norm)
        network["linear_layers"].append(nn.Linear(dims[-2] + self.condition_dim, dims[-1]))

        return network

    def _reset_parameters(self):
        for layer in self.network["linear_layers"]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
