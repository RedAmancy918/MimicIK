"""Parameter-free time encoding, implemented from its mathematical definition.

For frequency index k, the phase is t * max_positions ** (-k / denominator).
The output concatenates cosines and sines. This preserves the trained model's
frequency convention without vendoring unrelated embedding implementations.
"""

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        if dim < 4 or dim % 2 or max_positions <= 0:
            raise ValueError("The positional embedding needs an even dim >= 4")
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        index = torch.arange(self.dim // 2, dtype=torch.float32, device=time.device)
        exponent = index / (self.dim // 2 - int(self.endpoint))
        frequencies = torch.pow(1.0 / self.max_positions, exponent).to(time.dtype)
        phase = time[:, None] * frequencies[None, :]
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


SUPPORTED_TIMESTEP_EMBEDDING = {"positional": PositionalEmbedding}
