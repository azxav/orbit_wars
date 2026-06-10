from __future__ import annotations

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


def compute_viability_masks(
    obs: dict[str, Any],
    player_id: int,
    *,
    horizon: int = 160,
    device: str = "cpu",
    geometry: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic source-target and source-target-amount viability masks."""
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
            capture_needed = capture_needed_ships(source, target, int(player_id))
            for amount_bin in range(1, amount_bins):
                ships = decode_amount_bin(amount_bin, available, capture_needed)
                if ships <= 0:
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
