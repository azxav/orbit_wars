from __future__ import annotations

import torch
import torch.nn.functional as F

from orbit_bc_training.losses import apply_mask

from .policy import PPOPolicy


class FrozenBCReference:
    def __init__(self, checkpoint: str, *, device: str | torch.device = "cpu"):
        self.policy = PPOPolicy.from_bc_checkpoint(checkpoint, device=device)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def logits(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.policy.forward(batch)

    def kl_penalty(self, current_eval: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            ref = self.logits(batch)
            ref_target = apply_mask(ref["target_logits"], batch["target_mask"])
            ref_amount = apply_mask(ref["amount_logits"], batch["amount_mask"])
        cur_target = current_eval["target_logits"]
        cur_amount = current_eval["amount_logits"]
        target_kl = F.kl_div(F.log_softmax(cur_target, dim=-1), F.softmax(ref_target, dim=-1), reduction="none").sum(dim=-1)
        amount_kl = F.kl_div(F.log_softmax(cur_amount, dim=-1), F.softmax(ref_amount, dim=-1), reduction="none").sum(dim=-1)
        is_noop = batch["target_label"] == self.policy.config.noop_target_slot
        return (target_kl + torch.where(is_noop, torch.zeros_like(amount_kl), amount_kl)).mean()
