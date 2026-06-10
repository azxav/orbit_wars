from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .geometry_bridge import make_geometry
from .schema import (
    AMOUNT_BIN_NONE,
    AMOUNT_BIN_NAMES,
    NOOP_TARGET_SLOT,
    P_MAX,
    alive_target_slots,
    capture_needed_ships,
    decode_amount_bin,
    owned_source_slots,
    safe_float,
)


def _fleet_owner(fleet: Any) -> int | None:
    if isinstance(fleet, dict):
        for key in ("owner", "player", "player_id"):
            if key in fleet:
                return int(fleet[key])
        return None
    if isinstance(fleet, (list, tuple)) and len(fleet) >= 2:
        return int(fleet[1])
    return None


def _fleet_target_id(fleet: Any) -> int | None:
    if isinstance(fleet, dict):
        for key in ("target_planet_id", "target", "to_planet_id", "destination"):
            if key in fleet and fleet[key] is not None:
                return int(fleet[key])
        return None
    if isinstance(fleet, (list, tuple)):
        for idx in (3, 2):
            if len(fleet) > idx:
                try:
                    return int(fleet[idx])
                except Exception:
                    pass
    return None


def _fleet_ships(fleet: Any) -> float:
    if isinstance(fleet, dict):
        for key in ("ships", "num_ships", "ship_count"):
            if key in fleet:
                return max(0.0, safe_float(fleet[key]))
        return 0.0
    if isinstance(fleet, (list, tuple)):
        for idx in (5, 4, 3):
            if len(fleet) > idx:
                value = safe_float(fleet[idx], -1.0)
                if value >= 0.0:
                    return value
    return 0.0


def _fleet_eta(fleet: Any) -> float:
    if isinstance(fleet, dict):
        for key in ("eta", "remaining_turns", "turns_remaining", "remaining"):
            if key in fleet:
                return safe_float(fleet[key], math.inf)
        return math.inf
    if isinstance(fleet, (list, tuple)):
        for idx in (6, 7, 8):
            if len(fleet) > idx:
                value = safe_float(fleet[idx], math.inf)
                if math.isfinite(value):
                    return value
    return math.inf


def projected_capture_needed_ships(
    obs: dict[str, Any],
    source: Any,
    target: Any,
    player_id: int,
    *,
    horizon: int = 20,
) -> int:
    """Ships required after already-visible incoming fleets up to horizon are applied.

    This is deliberately conservative: owned targets stay reinforcement-viable with one ship,
    while neutral/enemy targets require enough ships to beat projected hostile garrison.
    """
    if target is None or not isinstance(target, (list, tuple)) or len(target) < 7:
        return capture_needed_ships(source, target, int(player_id))
    owner = int(target[1])
    if owner == int(player_id):
        return 1
    target_id = int(target[0])
    base_ships = max(0.0, safe_float(target[5]))
    friendly = 0.0
    hostile = 0.0
    for fleet in obs.get("fleets", []) or []:
        if _fleet_target_id(fleet) != target_id:
            continue
        eta = _fleet_eta(fleet)
        if not math.isfinite(eta) or eta > int(horizon):
            continue
        ships = _fleet_ships(fleet)
        if _fleet_owner(fleet) == int(player_id):
            friendly += ships
        else:
            hostile += ships
    projected = max(0.0, base_ships + hostile - friendly)
    return max(1, int(math.floor(projected)) + 1)


def compute_viability_masks(
    obs: dict[str, Any],
    player_id: int,
    *,
    horizon: int = 160,
    device: str = "cpu",
    geometry: Any = None,
    require_capture_viability: bool = True,
    capture_horizon: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic source-target and source-target-amount viability masks.

    A positive amount is valid only if the exact geometry reaches the intended target.
    For non-owned targets, it must also send enough ships to beat projected garrison.
    """
    amount_bins = len(AMOUNT_BIN_NAMES)
    target_mask = np.zeros((P_MAX, P_MAX + 1), dtype=bool)
    amount_mask = np.zeros((P_MAX, P_MAX + 1, amount_bins), dtype=bool)
    target_mask[:, NOOP_TARGET_SLOT] = True
    amount_mask[:, NOOP_TARGET_SLOT, AMOUNT_BIN_NONE] = True

    source_slots = owned_source_slots(obs, int(player_id))
    if not source_slots:
        return target_mask, amount_mask

    geometry = geometry or make_geometry(horizon=int(horizon), device=str(device))
    obs_tensors = geometry.obs_to_tensors(obs, player_id=int(player_id))
    movement = geometry.build_or_update_movement(obs_tensors)

    sources: list[int] = []
    targets: list[int] = []
    bins: list[int] = []
    ships_values: list[int] = []
    planets = obs.get("planets", [])
    for source_slot in source_slots:
        if source_slot >= len(planets) or len(planets[source_slot]) < 7:
            continue
        source = planets[source_slot]
        available = safe_float(source[5])
        for target_slot in alive_target_slots(obs, include_noop=False, exclude_slot=source_slot):
            if target_slot >= len(planets) or len(planets[target_slot]) < 7:
                continue
            target = planets[target_slot]
            current_needed = capture_needed_ships(source, target, int(player_id))
            projected_needed = projected_capture_needed_ships(
                obs,
                source,
                target,
                int(player_id),
                horizon=int(capture_horizon),
            )
            target_is_owned = int(target[1]) == int(player_id)
            for amount_bin in range(1, amount_bins):
                ships = decode_amount_bin(amount_bin, available, current_needed)
                if ships <= 0:
                    continue
                if require_capture_viability and not target_is_owned and int(ships) < int(projected_needed):
                    continue
                sources.append(int(source_slot))
                targets.append(int(target_slot))
                bins.append(int(amount_bin))
                ships_values.append(int(ships))

    if not sources:
        return target_mask, amount_mask

    source_t = torch.as_tensor(sources, dtype=torch.long, device=movement.device)
    target_t = torch.as_tensor(targets, dtype=torch.long, device=movement.device)
    ships_t = torch.as_tensor(ships_values, dtype=movement.dtype, device=movement.device)
    active_t = torch.ones_like(source_t, dtype=torch.bool, device=movement.device)
    aim = geometry.aim_source_to_target(
        source_slots=source_t,
        target_slots=target_t,
        fleet_sizes=ships_t,
        movement=movement,
        active=active_t,
    )
    viable = aim["viable"].detach().cpu().numpy().astype(bool)
    for source_slot, target_slot, amount_bin, is_viable in zip(sources, targets, bins, viable, strict=True):
        if not is_viable:
            continue
        amount_mask[source_slot, target_slot, amount_bin] = True
        target_mask[source_slot, target_slot] = True
    return target_mask, amount_mask


def target_mask_for_source_from_viability(
    obs: dict[str, Any],
    player_id: int,
    source_slot: int,
    *,
    horizon: int = 160,
    device: str = "cpu",
    geometry: Any = None,
) -> torch.Tensor:
    target_mask, _ = compute_viability_masks(obs, int(player_id), horizon=int(horizon), device=str(device), geometry=geometry)
    return torch.as_tensor(target_mask[int(source_slot)], dtype=torch.bool)
