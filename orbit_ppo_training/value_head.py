from __future__ import annotations

import torch
from torch import nn


class ValueHead(nn.Module):
    """Scalar value head on the encoded global token."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, global_context: torch.Tensor) -> torch.Tensor:
        return self.net(global_context).squeeze(-1)

