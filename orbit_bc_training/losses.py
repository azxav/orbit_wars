from __future__ import annotations

import torch
import torch.nn.functional as F

from orbit_training_prep.schema import NOOP_TARGET_SLOT


NEG_INF = -1.0e9


def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask.to(torch.bool), NEG_INF)


def masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return apply_mask(logits, mask).argmax(dim=-1)


def _mean_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def bc_loss_and_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    target_logits = apply_mask(outputs["target_logits"], batch["target_mask"])
    amount_logits = apply_mask(outputs["amount_logits"], batch["amount_mask"])
    target_labels = batch["target_label"]
    amount_labels = batch["amount_label"]
    target_loss = _mean_ce(target_logits, target_labels)
    amount_rows = target_labels != NOOP_TARGET_SLOT
    if amount_rows.any():
        amount_loss = _mean_ce(amount_logits[amount_rows], amount_labels[amount_rows])
    else:
        amount_loss = amount_logits.sum() * 0.0
    loss = target_loss + amount_loss
    target_pred = target_logits.argmax(dim=1)
    amount_pred = amount_logits.argmax(dim=1)
    is_noop = batch["is_noop"].to(torch.bool)
    non_noop = ~is_noop
    top3 = target_logits.topk(k=min(3, target_logits.shape[1]), dim=1).indices
    noop_pred = target_pred == target_logits.shape[1] - 1
    metrics = {
        "total_loss": float(loss.detach().cpu()),
        "target_loss": float(target_loss.detach().cpu()),
        "amount_loss": float(amount_loss.detach().cpu()),
        "target_accuracy": float((target_pred == target_labels).float().mean().detach().cpu()),
        "target_non_noop_accuracy": float((target_pred[non_noop] == target_labels[non_noop]).float().mean().detach().cpu()) if non_noop.any() else 0.0,
        "target_top3_accuracy": float((top3 == target_labels.unsqueeze(1)).any(dim=1).float().mean().detach().cpu()),
        "amount_accuracy": float((amount_pred == amount_labels).float().mean().detach().cpu()),
        "launch_vs_noop_accuracy": float((noop_pred == is_noop).float().mean().detach().cpu()),
        "noop_rate_predicted": float(noop_pred.float().mean().detach().cpu()),
        "noop_rate_true": float(is_noop.float().mean().detach().cpu()),
    }
    return loss, metrics
