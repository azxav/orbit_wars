from __future__ import annotations

from typing import Optional

import torch

from orbit_training_prep.schema import NOOP_TARGET_SLOT, capture_needed_ships, decode_amount_bin

from .losses import masked_argmax


def _slot_for_planet_id(obs: dict, planet_id: int) -> int | None:
    for i, p in enumerate(obs.get("planets", [])):
        if len(p) >= 7 and int(p[0]) == int(planet_id):
            return i
    return None


def _target_mask(obs: dict, source_slot: int) -> torch.Tensor:
    mask = torch.zeros(NOOP_TARGET_SLOT + 1, dtype=torch.bool)
    for i, p in enumerate(obs.get("planets", [])[:NOOP_TARGET_SLOT]):
        if len(p) >= 7 and int(p[0]) >= 0 and i != source_slot:
            mask[i] = True
    mask[NOOP_TARGET_SLOT] = True
    return mask


def decode_bc_prediction(
    obs,
    player_id,
    source_planet_id,
    target_logits,
    amount_logits,
    geometry,
    *,
    target_mask: torch.Tensor | None = None,
    amount_mask: torch.Tensor | None = None,
) -> Optional[list]:
    source_slot = _slot_for_planet_id(obs, int(source_planet_id))
    if source_slot is None:
        return None
    source = obs.get("planets", [])[source_slot]
    if len(source) < 7 or int(source[1]) != int(player_id):
        return None
    mask = _target_mask(obs, source_slot) if target_mask is None else torch.as_tensor(target_mask, dtype=torch.bool)
    mask = mask.to(torch.as_tensor(target_logits).device)
    target = int(masked_argmax(torch.as_tensor(target_logits).float().unsqueeze(0), mask.unsqueeze(0))[0].item())
    if target == NOOP_TARGET_SLOT:
        return None
    amount_tensor = torch.as_tensor(amount_logits).float()
    if amount_mask is not None:
        amount_mask_t = torch.as_tensor(amount_mask, dtype=torch.bool, device=amount_tensor.device)
        if not bool(amount_mask_t.any().item()):
            return None
        amount_bin = int(masked_argmax(amount_tensor.unsqueeze(0), amount_mask_t.unsqueeze(0))[0].item())
    else:
        amount_bin = int(amount_tensor.argmax().item())
    target_planet = obs.get("planets", [])[target] if target < len(obs.get("planets", [])) else None
    ships = decode_amount_bin(amount_bin, float(source[5]), capture_needed_ships(source, target_planet, int(player_id)))
    if ships <= 0:
        return None
    moves = geometry.to_env_moves(
        obs=obs,
        source_slots=torch.tensor([source_slot], dtype=torch.long),
        target_slots=torch.tensor([target], dtype=torch.long),
        ships=torch.tensor([ships], dtype=torch.float32),
        player_id=int(player_id),
        valid=torch.tensor([True]),
    )
    return moves[0] if moves else None
