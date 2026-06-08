from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from orbit_bc_eval.bc_agent_runtime import (
    _amount_count_key,
    _decode_none_reason,
    _increment_count,
    _target_count_key,
    build_source_batch,
    target_mask_for_source,
    validate_env_move,
)
from orbit_bc_training.checkpoints import load_checkpoint
from orbit_bc_training.losses import NEG_INF, apply_mask, masked_argmax
from orbit_bc_training.decode_policy import decode_bc_prediction
from orbit_training_prep.geometry_bridge import make_geometry
from orbit_training_prep.schema import AMOUNT_BIN_NONE, NOOP_TARGET_SLOT, P_MAX, owned_source_slots

from .trajectory import DecisionRecord
from .value_head import ValueHead


@dataclass
class PolicyTurn:
    moves: list[list[Any]]
    records: list[DecisionRecord]
    illegal_action_count: int
    skipped_invalid_action_count: int
    predicted_launches: int
    no_op_source_decisions: int
    entropy: float
    value: float
    skip_reasons: dict[str, int] = field(default_factory=dict)
    opening_prediction_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"target": {}, "amount": {}, "target_amount": {}}
    )


class PPOPolicy(nn.Module):
    def __init__(self, bc_model: nn.Module):
        super().__init__()
        self.bc = bc_model
        self.config = bc_model.config
        self.value_head = ValueHead(int(self.config.hidden_size))

    @classmethod
    def from_bc_checkpoint(cls, checkpoint: str, *, device: str | torch.device = "cpu") -> "PPOPolicy":
        bc_model, _ = load_checkpoint(checkpoint, device=str(device))
        model = cls(bc_model).to(device)
        return model

    def _encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        planets = batch["planet_features"]
        global_features = batch["global_features"]
        pmax = planets.shape[1]
        planet_tokens = self.bc.planet_encoder(planets)
        global_token = self.bc.global_encoder(global_features).unsqueeze(1)
        encoded = self.bc.encoder(torch.cat([global_token, planet_tokens], dim=1))
        planet_ctx = encoded[:, 1 : pmax + 1]
        target_state_features = batch.get("target_state_features")
        if getattr(self.bc, "target_state_encoder", None) is not None and target_state_features is not None and target_state_features.shape[-1] > 0:
            planet_ctx = planet_ctx + self.bc.target_state_encoder(target_state_features.to(planets.dtype))
        return encoded[:, 0], planet_ctx, planets

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        global_ctx, _, _ = self._encode(batch)
        out = self.bc(batch)
        return {"target_logits": out["target_logits"], "amount_logits": out["amount_logits"], "value": self.value_head(global_ctx)}

    def evaluate_actions(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.forward(batch)
        target_logits = apply_mask(out["target_logits"], batch["target_mask"])
        amount_logits = apply_mask(out["amount_logits"], batch["amount_mask"])
        target_dist = torch.distributions.Categorical(logits=target_logits)
        amount_dist = torch.distributions.Categorical(logits=amount_logits)
        target_action = batch["target_label"]
        amount_action = batch["amount_label"]
        is_noop = target_action == NOOP_TARGET_SLOT
        logprob = target_dist.log_prob(target_action)
        logprob = logprob + torch.where(is_noop, torch.zeros_like(logprob), amount_dist.log_prob(amount_action))
        entropy = target_dist.entropy() + torch.where(is_noop, torch.zeros_like(logprob), amount_dist.entropy())
        return {"logprob": logprob, "entropy": entropy, "value": out["value"], "target_logits": target_logits, "amount_logits": amount_logits}

    @torch.no_grad()
    def act_observation(
        self,
        obs: dict[str, Any],
        player_id: int,
        *,
        deterministic: bool,
        device: str | torch.device = "cpu",
        geometry=None,
    ) -> PolicyTurn:
        self.eval()
        geometry = geometry or make_geometry(device="cpu")
        moves: list[list[Any]] = []
        records: list[DecisionRecord] = []
        illegal = 0
        skipped = 0
        predicted_launches = 0
        noops = 0
        skip_reasons: dict[str, int] = {}
        opening_prediction_counts: dict[str, dict[str, int]] = {"target": {}, "amount": {}, "target_amount": {}}
        entropies: list[float] = []
        values: list[float] = []
        for source_slot in owned_source_slots(obs, player_id):
            batch = build_source_batch(obs, player_id, source_slot, device=str(device))
            target_mask = target_mask_for_source(obs, source_slot).to(device)
            amount_mask = torch.ones(int(self.config.amount_bins), dtype=torch.bool, device=device)
            batch["target_mask"] = target_mask.unsqueeze(0)
            batch["amount_mask"] = amount_mask.unsqueeze(0)
            out = self.forward(batch)
            target_logits = apply_mask(out["target_logits"], batch["target_mask"])[0]
            target_dist = torch.distributions.Categorical(logits=target_logits)
            target = int(torch.argmax(target_logits).item()) if deterministic else int(target_dist.sample().item())
            amount = AMOUNT_BIN_NONE
            amount_logprob = torch.tensor(0.0, device=device)
            amount_entropy = torch.tensor(0.0, device=device)
            if target == NOOP_TARGET_SLOT:
                noops += 1
            else:
                predicted_launches += 1
                amount_mask = amount_mask.clone()
                amount_mask[AMOUNT_BIN_NONE] = False
                batch["amount_mask"] = amount_mask.unsqueeze(0)
                amount_batch = dict(batch)
                amount_batch["target_label"] = torch.as_tensor([target], dtype=torch.long, device=device)
                amount_logits = apply_mask(self.forward(amount_batch)["amount_logits"], batch["amount_mask"])[0]
                amount_dist = torch.distributions.Categorical(logits=amount_logits)
                amount = int(torch.argmax(amount_logits).item()) if deterministic else int(amount_dist.sample().item())
                amount_logprob = amount_dist.log_prob(torch.as_tensor(amount, device=device))
                amount_entropy = amount_dist.entropy()
            step = int(obs.get("step", 0) or 0)
            if 0 <= step < 100:
                target_key = _target_count_key(target)
                amount_key = _amount_count_key(amount)
                _increment_count(opening_prediction_counts["target"], target_key)
                _increment_count(opening_prediction_counts["amount"], amount_key)
                _increment_count(opening_prediction_counts["target_amount"], f"{target_key}|{amount_key}")
            target_logprob = target_dist.log_prob(torch.as_tensor(target, device=device))
            entropy = target_dist.entropy() + amount_entropy
            source = obs.get("planets", [])[source_slot]
            move = decode_bc_prediction(
                obs,
                player_id,
                int(source[0]),
                F.one_hot(torch.as_tensor(target), num_classes=P_MAX + 1).float() * 1.0e6,
                F.one_hot(torch.as_tensor(amount), num_classes=int(self.config.amount_bins)).float() * 1.0e6,
                geometry,
            )
            decoded_moves: list[list[Any]] = []
            if move is not None:
                validation = validate_env_move(obs, player_id, move)
                if validation.ok:
                    decoded_moves = [[int(move[0]), float(move[1]), int(move[2])]]
                    moves.extend(decoded_moves)
                else:
                    illegal += 1
                    skipped += 1
                    _increment_count(skip_reasons, validation.reason)
            elif target != NOOP_TARGET_SLOT:
                skipped += 1
                _increment_count(skip_reasons, _decode_none_reason(obs, player_id, int(source_slot), target, amount))
            records.append(
                DecisionRecord(
                    planet_features=batch["planet_features"][0].detach().cpu(),
                    global_features=batch["global_features"][0].detach().cpu(),
                    target_state_features=batch["target_state_features"][0].detach().cpu(),
                    pair_features=batch["pair_features"][0].detach().cpu(),
                    source_slot=int(source_slot),
                    target_action=int(target),
                    amount_action=int(amount),
                    target_mask=target_mask.detach().cpu(),
                    amount_mask=amount_mask.detach().cpu(),
                    logprob=float((target_logprob + amount_logprob).detach().cpu()),
                    entropy=float(entropy.detach().cpu()),
                    value=float(out["value"][0].detach().cpu()),
                    step=step,
                    decoded_moves=decoded_moves,
                    skipped_invalid_action_count=1 if target != NOOP_TARGET_SLOT and not decoded_moves else 0,
                )
            )
            entropies.append(float(entropy.detach().cpu()))
            values.append(float(out["value"][0].detach().cpu()))
        return PolicyTurn(
            moves=moves,
            records=records,
            illegal_action_count=illegal,
            skipped_invalid_action_count=skipped,
            predicted_launches=predicted_launches,
            no_op_source_decisions=noops,
            entropy=sum(entropies) / max(1, len(entropies)),
            value=sum(values) / max(1, len(values)),
            skip_reasons=skip_reasons,
            opening_prediction_counts=opening_prediction_counts,
        )
