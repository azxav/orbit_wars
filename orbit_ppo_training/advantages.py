from __future__ import annotations

import torch


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_adv = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
    next_value = torch.tensor(0.0, dtype=values.dtype, device=values.device)
    for t in reversed(range(rewards.numel())):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + float(gamma) * next_value * nonterminal - values[t]
        last_adv = delta + float(gamma) * float(gae_lambda) * nonterminal * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() < 2:
        return advantages
    return (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

