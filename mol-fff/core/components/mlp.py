from typing import Optional, Literal

import torch
from torch import nn, Tensor

from .norms.vector import get_vector_norm
from ..utils.utils import import_from_string


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            intermediate_dims: list[int] = [],
            condition_dim: int = 0,
            condition_mode: Literal["concat_input", "concat_every"] | None = None,
            activation: str = "torch.nn.ReLU",
            activation_kwargs: dict = {},
            norm: Literal["batch", "layer"] | None = None,
            norm_kwargs: dict = {},
            dropout: float = 0.0,
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.intermediate_dims = intermediate_dims
        self.condition_dim = condition_dim
        self.condition_mode = condition_mode
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.dropout = dropout

        self.layers = self._configure_network()

    def forward(self, x: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the MLP.

        :param x: Input tensor of shape (batch_size, input_dim)
        :param c: Condition tensor of shape (batch_size, condition_dim)
        :return: Output tensor of shape (batch_size, output_dim)"""

        match self.condition_mode:
            case None:
                return nn.Sequential(*self.layers)(x)
            case "concat_input":
                x = torch.cat([x, c], dim=-1)
                return nn.Sequential(*self.layers)(x)
            case "concat_every":
                for layer in self.layers:
                    if isinstance(layer, nn.Linear) or isinstance(layer, nn.LazyLinear):
                        x = torch.cat([x, c], dim=-1)
                    x = layer(x)
                return x

    def _configure_network(self):
        widths = [self.input_dim] + self.intermediate_dims + [self.output_dim]
        activation = self._configure_activation()

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(widths[:-1], widths[1:])):
            if self.condition_mode == "concat_every":
                in_dim += self.condition_dim

            if i != 0 and (norm := get_vector_norm(self.norm, in_dim)) is not None:
                layers.append(norm)

            if i != 0 and (dropout := self._configure_dropout()) is not None:
                layers.append(dropout)

            layers.append(self._configure_linear(in_dim, out_dim))

            if i != len(widths) - 2:
                layers.append(activation)

        return nn.Sequential(*layers)

    def _configure_linear(self, in_features: int, out_features: int) -> nn.Module:
        match in_features:
            case "lazy":
                return nn.LazyLinear(out_features)
            case int() as in_features:
                return nn.Linear(in_features, out_features)
            case other:
                raise NotImplementedError(f"Unrecognized input_dim value: '{other}'")

    def _configure_activation(self) -> nn.Module:
        return import_from_string(self.activation)(**self.activation_kwargs)

    def _configure_dropout(self) -> nn.Module | None:
        if self.dropout > 0:
            return nn.Dropout(self.dropout)
