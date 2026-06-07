from __future__ import annotations

import torch

from .bc_reference import FrozenBCReference


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    var_y = torch.var(y_true)
    if float(var_y.detach().cpu()) < 1e-8:
        return torch.tensor(0.0, device=y_true.device)
    return 1.0 - torch.var(y_true - y_pred) / var_y


def ppo_loss(
    policy,
    batch: dict[str, torch.Tensor],
    *,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
    bc_ref: FrozenBCReference | None,
    kl_to_bc_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ev = policy.evaluate_actions(batch)
    logprob = ev["logprob"]
    ratio = torch.exp(logprob - batch["old_logprob"])
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = 0.5 * (returns - ev["value"]).pow(2).mean()
    entropy = ev["entropy"].mean()
    kl_bc = bc_ref.kl_penalty(ev, batch) if bc_ref is not None and kl_to_bc_coef > 0 else torch.tensor(0.0, device=logprob.device)
    loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy + float(kl_to_bc_coef) * kl_bc
    approx_kl = (batch["old_logprob"] - logprob).mean()
    clip_frac = ((ratio - 1.0).abs() > clip_range).float().mean()
    metrics = {
        "total_loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "kl_to_bc": float(kl_bc.detach().cpu()),
        "clip_frac": float(clip_frac.detach().cpu()),
        "value_explained_variance": float(explained_variance(ev["value"].detach(), returns.detach()).detach().cpu()),
    }
    return loss, metrics

