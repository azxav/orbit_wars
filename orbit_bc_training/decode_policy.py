from __future__ import annotations

from typing import Optional

import torch

from orbit_lite.constants import COMET_SPAWN_STEPS
from orbit_training_prep.exact_target_sim import ExactTargetSimulator
from orbit_training_prep.schema import NOOP_TARGET_SLOT, capture_needed_ships, decode_amount_bin

from .losses import apply_mask, masked_argmax


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


def _next_unobserved_comet_spawn_delta(obs: dict) -> int | None:
    if obs.get("comet_planet_ids") or obs.get("comets"):
        return None
    try:
        step = int(obs.get("step", 0) or 0)
    except Exception:
        step = 0
    for spawn_step in COMET_SPAWN_STEPS:
        delta = int(spawn_step) - step
        if delta > 0:
            return delta
    return None


def _eta_for_decision(
    obs: dict,
    player_id: int,
    geometry,
    *,
    source_slot: int,
    target: int,
    ships: int,
) -> float | None:
    if not all(hasattr(geometry, name) for name in ("obs_to_tensors", "build_or_update_movement", "aim_source_to_target")):
        return None
    obs_for_movement = dict(obs)
    obs_for_movement["fleets"] = []
    obs_tensors = geometry.obs_to_tensors(obs_for_movement, player_id=int(player_id))
    movement = geometry.build_or_update_movement(obs_tensors)
    source_t = torch.tensor([int(source_slot)], dtype=torch.long, device=movement.device)
    target_t = torch.tensor([int(target)], dtype=torch.long, device=movement.device)
    ships_t = torch.tensor([max(1, int(ships))], dtype=movement.dtype, device=movement.device)
    active_t = torch.ones(1, dtype=torch.bool, device=movement.device)
    aim = geometry.aim_source_to_target(
        source_slots=source_t,
        target_slots=target_t,
        fleet_sizes=ships_t,
        movement=movement,
        active=active_t,
    )
    if not bool(aim["viable"][0].detach().cpu().item()):
        return None
    return float(aim["eta"][0].detach().cpu().item())


def _crosses_unobserved_comet_spawn(obs: dict, eta: float | None) -> bool:
    if eta is None:
        return False
    delta = _next_unobserved_comet_spawn_delta(obs)
    if delta is None:
        return False
    return float(eta) >= float(delta)


def _geometry_horizon(geometry) -> int:
    cfg = getattr(geometry, "config", None)
    try:
        return int(getattr(cfg, "movement_horizon"))
    except Exception:
        return 160


def _move_hits_intended_target(obs: dict, player_id: int, move: list, target_planet_id: int | None, *, horizon: int) -> bool:
    if target_planet_id is None:
        return False
    hit = ExactTargetSimulator(horizon=int(horizon), device="cpu").first_hit_for_launch(
        obs,
        int(player_id),
        {
            "source_planet_id": int(move[0]),
            "raw_angle": float(move[1]),
            "ships": int(move[2]),
        },
    )
    return hit.get("hit_type") in {"planet", "comet"} and int(hit.get("hit_id", -1)) == int(target_planet_id)


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
    noop_threshold: float = 0.85,
) -> Optional[list]:
    source_slot = _slot_for_planet_id(obs, int(source_planet_id))
    if source_slot is None:
        return None
    source = obs.get("planets", [])[source_slot]
    if len(source) < 7 or int(source[1]) != int(player_id):
        return None
    mask = _target_mask(obs, source_slot) if target_mask is None else torch.as_tensor(target_mask, dtype=torch.bool)
    mask = mask.to(torch.as_tensor(target_logits).device)
    # Launch-bias decode. Pure argmax over the 65-way head collapses to NOOP at
    # rollout (NOOP is the single fattest slot; under covariate shift target
    # logits go mushy and NOOP wins). Only NOOP when its probability genuinely
    # dominates (>= noop_threshold); otherwise commit to the best launch slot.
    masked = apply_mask(torch.as_tensor(target_logits).float().unsqueeze(0), mask.unsqueeze(0))
    probs = masked.softmax(dim=-1)[0]
    launch_probs = probs.clone()
    launch_probs[NOOP_TARGET_SLOT] = -1.0
    best_launch = int(launch_probs.argmax().item())
    noop_prob = float(probs[NOOP_TARGET_SLOT].item())
    if launch_probs[best_launch] <= 0.0 or noop_prob >= float(noop_threshold):
        return None
    target = best_launch
    amount_tensor = torch.as_tensor(amount_logits).float()
    if amount_mask is not None:
        amount_mask_t = torch.as_tensor(amount_mask, dtype=torch.bool, device=amount_tensor.device)
        if not bool(amount_mask_t.any().item()):
            return None
        amount_bin = int(masked_argmax(amount_tensor.unsqueeze(0), amount_mask_t.unsqueeze(0))[0].item())
    else:
        amount_bin = int(amount_tensor.argmax().item())
    target_planet = obs.get("planets", [])[target] if target < len(obs.get("planets", [])) else None
    target_planet_id = int(target_planet[0]) if target_planet is not None and len(target_planet) >= 1 else None
    ships = decode_amount_bin(amount_bin, float(source[5]), capture_needed_ships(source, target_planet, int(player_id)))
    if ships <= 0:
        return None
    eta = _eta_for_decision(obs, int(player_id), geometry, source_slot=source_slot, target=target, ships=ships)
    if _crosses_unobserved_comet_spawn(obs, eta):
        return None
    moves = geometry.to_env_moves(
        obs=obs,
        source_slots=torch.tensor([source_slot], dtype=torch.long),
        target_slots=torch.tensor([target], dtype=torch.long),
        ships=torch.tensor([ships], dtype=torch.float32),
        player_id=int(player_id),
        valid=torch.tensor([True]),
    )
    if not moves:
        return None
    move = moves[0]
    if not _move_hits_intended_target(obs, int(player_id), move, target_planet_id, horizon=_geometry_horizon(geometry)):
        return None
    return move
