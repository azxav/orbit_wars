from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DecisionRecord:
    planet_features: torch.Tensor
    global_features: torch.Tensor
    target_state_features: torch.Tensor
    pair_features: torch.Tensor
    source_slot: int
    target_action: int
    amount_action: int
    target_mask: torch.Tensor
    amount_mask: torch.Tensor
    logprob: float
    entropy: float
    value: float
    reward: float = 0.0
    done: bool = False
    step: int = 0
    decoded_moves: list[list[Any]] = field(default_factory=list)
    skipped_invalid_action_count: int = 0


def collate_decisions(records: list[DecisionRecord], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot collate empty trajectory")
    return {
        "planet_features": torch.stack([r.planet_features for r in records]).to(device),
        "global_features": torch.stack([r.global_features for r in records]).to(device),
        "target_state_features": torch.stack([r.target_state_features for r in records]).to(device),
        "pair_features": torch.stack([r.pair_features for r in records]).to(device),
        "source_slot": torch.as_tensor([r.source_slot for r in records], dtype=torch.long, device=device),
        "target_label": torch.as_tensor([r.target_action for r in records], dtype=torch.long, device=device),
        "amount_label": torch.as_tensor([r.amount_action for r in records], dtype=torch.long, device=device),
        "target_mask": torch.stack([r.target_mask for r in records]).to(device).bool(),
        "amount_mask": torch.stack([r.amount_mask for r in records]).to(device).bool(),
        "old_logprob": torch.as_tensor([r.logprob for r in records], dtype=torch.float32, device=device),
        "old_value": torch.as_tensor([r.value for r in records], dtype=torch.float32, device=device),
        "reward": torch.as_tensor([r.reward for r in records], dtype=torch.float32, device=device),
        "done": torch.as_tensor([r.done for r in records], dtype=torch.float32, device=device),
    }
