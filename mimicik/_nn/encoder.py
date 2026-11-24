# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Chaoyi Pan
# Extracted state encoder from Much Ado About Noising (CleanDiffuser lineage).
# See THIRD_PARTY_NOTICES.md. The state_dict hierarchy is preserved.
"""State-only encoder; no image or simulator dependencies."""

import torch
from torch import nn


def at_least_ndim(value, dimensions):
    if not isinstance(value, torch.Tensor):
        return value
    return value.reshape(value.shape + (1,) * max(0, dimensions - value.ndim))


def get_mask(
    mask: torch.Tensor,
    mask_shape: tuple,
    dropout: float,
    train: bool,
    device: torch.device,
):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1.0 if mask is None else mask
    return mask


class BaseEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        raise NotImplementedError


class Mlp(nn.Module):
    """**Multilayer perceptron.** A simple pytorch MLP module.

    Args:
        in_dim: int,
            The dimension of the input tensor.
        hidden_dims: List[int],
            A list of integers, each element is the dimension of the hidden layer.
        out_dim: int,
            The dimension of the output tensor.
        activation: nn.Module,
            The activation function used in the hidden layers.
        out_activation: nn.Module,
            The activation function used in the output layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: nn.Module | None = None,
        out_activation: nn.Module | None = None,
    ):
        super().__init__()
        activation = nn.ReLU() if activation is None else activation
        out_activation = nn.Identity() if out_activation is None else out_activation
        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
                    activation,
                )
                for i in range(len(hidden_dims))
            ],
            nn.Linear(hidden_dims[-1], out_dim),
            out_activation,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


class MLPEncoder(BaseEncoder):
    """A simple MLP encoder.

    Use a simple MLP to project the input condition to the desired dimension.

    Args:
        in_dim: int,
            The input dimension of the condition
        out_dim: int,
            The output dimension of the condition
        hidden_dims: List[int],
            The hidden dimensions of the MLP
        act: nn.Module,
            The activation function of the MLP
        dropout: float,
            The label dropout rate

    Examples:
        >>> encoder = MLPEncoder(in_dim=5, out_dim=10)
        >>> condition = torch.randn(2, 5)
        >>> encoder(condition).shape
        torch.Size([2, 10])
        >>> condition = torch.randn(2, 20, 5)
        >>> encoder(condition).shape
        torch.Size([2, 20, 10])
    """

    def __init__(
        self,
        obs_dim: int,
        emb_dim: int,
        To: int,
        hidden_dims: list[int],
        act=None,
        dropout: float = 0.25,
    ):
        super().__init__()
        act = nn.LeakyReLU() if act is None else act
        self.dropout = dropout
        self.To = To
        self.emb_dim = emb_dim
        hidden_dims = (
            [
                hidden_dims,
            ]
            if isinstance(hidden_dims, int)
            else hidden_dims
        )
        self.mlp = Mlp(obs_dim * To, hidden_dims, emb_dim * To, act)

    def forward(self, obs: torch.Tensor | dict, mask: torch.Tensor = None):
        # Handle dict input by concatenating all tensors
        if isinstance(obs, dict):
            # Sort keys for consistent ordering and concatenate all values
            keys = sorted(obs.keys())
            obs_list = [obs[k] for k in keys]
        else:
            obs_list = [obs]
        # Flatten each tensor to (batch, -1) and concatenate
        flattened = [t.reshape(t.shape[0], -1) for t in obs_list]
        obs = torch.cat(flattened, dim=-1)

        mask = at_least_ndim(
            get_mask(
                mask,
                (obs.shape[0],),
                self.dropout,
                self.training,
                obs.device,
            ),
            obs.dim(),
        )
        emb_features = self.mlp(obs) * mask
        return emb_features.reshape(obs.shape[0], self.To, self.emb_dim)
