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

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
BIG = 1_000_000.0


def _fleet_owner(fleet: Any) -> int | None:
    if isinstance(fleet, dict):
        for key in ("owner", "player", "player_id"):
            if key in fleet:
                return int(fleet[key])
        return None
    if isinstance(fleet, (list, tuple)) and len(fleet) >= 2:
        return int(fleet[1])
    return None


def _is_native_fleet_observation(fleet: Any) -> bool:
    if not isinstance(fleet, (list, tuple)) or len(fleet) != 6:
        return False
    x = safe_float(fleet[2], math.inf)
    y = safe_float(fleet[3], math.inf)
    angle = safe_float(fleet[4], math.inf)
    ships = safe_float(fleet[5], -1.0)
    return math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE and math.isfinite(angle) and ships > 0.0


def _fleet_target_id(fleet: Any) -> int | None:
    if isinstance(fleet, dict):
        for key in ("target_planet_id", "target", "to_planet_id", "destination"):
            if key in fleet and fleet[key] is not None:
                return int(fleet[key])
        return None
    if _is_native_fleet_observation(fleet):
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
    if _is_native_fleet_observation(fleet):
        return max(0.0, safe_float(fleet[5]))
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
    if _is_native_fleet_observation(fleet):
        return math.inf
    if isinstance(fleet, (list, tuple)):
        for idx in (6, 7, 8):
            if len(fleet) > idx:
                value = safe_float(fleet[idx], math.inf)
                if math.isfinite(value):
                    return value
    return math.inf


def _fleet_speed(ships: float) -> float:
    s = max(1.0, float(ships))
    return 1.0 + 5.0 * (math.log(s) / math.log(1000.0)) ** 1.5


def _point_segment_hits_sun(old_x: torch.Tensor, old_y: torch.Tensor, new_x: torch.Tensor, new_y: torch.Tensor) -> torch.Tensor:
    vx = new_x - old_x
    vy = new_y - old_y
    wx = CENTER - old_x
    wy = CENTER - old_y
    vv = (vx * vx + vy * vy).clamp(min=1e-12)
    t = ((wx * vx + wy * vy) / vv).clamp(0.0, 1.0)
    cx = old_x + t * vx
    cy = old_y + t * vy
    return (cx - CENTER) ** 2 + (cy - CENTER) ** 2 < SUN_RADIUS * SUN_RADIUS


def _swept_circle_hit(
    ax: torch.Tensor,
    ay: torch.Tensor,
    bx: torch.Tensor,
    by: torch.Tensor,
    p0x: torch.Tensor,
    p0y: torch.Tensor,
    p1x: torch.Tensor,
    p1y: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = bx - ax - (p1x - p0x)
    dvy = by - ay - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - radius * radius
    near_static = a < 1e-12
    disc = b * b - 4.0 * a * c
    safe_a = torch.where(near_static, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    return torch.where(near_static, c <= 0.0, (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0))


def _native_fleet_target_and_eta(fleet: Any, movement: Any, *, horizon: int) -> tuple[int | None, float]:
    if not _is_native_fleet_observation(fleet) or movement is None:
        return None, math.inf
    h = max(1, int(horizon))
    h = min(h, int(movement.x.shape[0]) - 1)
    if h <= 0:
        return None, math.inf
    device = movement.device
    dtype = movement.dtype
    x0 = torch.tensor(float(safe_float(fleet[2])), dtype=dtype, device=device)
    y0 = torch.tensor(float(safe_float(fleet[3])), dtype=dtype, device=device)
    angle = float(safe_float(fleet[4]))
    speed = _fleet_speed(safe_float(fleet[5]))
    k = torch.arange(h + 1, dtype=dtype, device=device)
    fx = x0 + math.cos(angle) * speed * k
    fy = y0 + math.sin(angle) * speed * k
    px = movement.x[: h + 1]
    py = movement.y[: h + 1]
    radii = movement.radii
    alive0 = movement.alive_at(0)
    step_axis = torch.arange(1, h + 1, dtype=dtype, device=device)
    hit = _swept_circle_hit(
        fx[:-1].view(h, 1),
        fy[:-1].view(h, 1),
        fx[1:].view(h, 1),
        fy[1:].view(h, 1),
        px[:-1],
        py[:-1],
        px[1:],
        py[1:],
        radii.view(1, -1),
    )
    hit = hit & alive0.view(1, -1)
    hit_step_by_slot = torch.where(hit, step_axis.view(-1, 1), torch.full_like(hit, BIG, dtype=dtype)).amin(0)
    first_planet_step = float(hit_step_by_slot.amin().item())
    first_planet_slot = int(torch.argmin(hit_step_by_slot).item())
    nfx = fx[1:]
    nfy = fy[1:]
    ofx = fx[:-1]
    ofy = fy[:-1]
    bounds = (nfx < 0.0) | (nfx > BOARD_SIZE) | (nfy < 0.0) | (nfy > BOARD_SIZE)
    sun = _point_segment_hits_sun(ofx, ofy, nfx, nfy)
    sun_step = float(torch.where(sun, step_axis, torch.full_like(sun, BIG, dtype=dtype)).amin().item())
    bounds_step = float(torch.where(bounds, step_axis, torch.full_like(bounds, BIG, dtype=dtype)).amin().item())
    first_env_step = min(sun_step, bounds_step)
    if first_planet_step <= first_env_step and first_planet_step < BIG:
        return int(movement.planet_ids.long().tolist()[first_planet_slot]), float(first_planet_step)
    return None, math.inf


def projected_capture_needed_ships(
    obs: dict[str, Any],
    source: Any,
    target: Any,
    player_id: int,
    *,
    horizon: int = 20,
    movement: Any = None,
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
        fleet_target_id = _fleet_target_id(fleet)
        eta = _fleet_eta(fleet)
        if (fleet_target_id is None or not math.isfinite(eta)) and movement is not None:
            derived_target_id, derived_eta = _native_fleet_target_and_eta(fleet, movement, horizon=int(horizon))
            if fleet_target_id is None:
                fleet_target_id = derived_target_id
            if not math.isfinite(eta):
                eta = derived_eta
        if fleet_target_id != target_id:
            continue
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
    obs_for_movement = dict(obs)
    obs_for_movement["fleets"] = []
    obs_tensors = geometry.obs_to_tensors(obs_for_movement, player_id=int(player_id))
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
                movement=movement,
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
