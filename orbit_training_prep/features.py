from __future__ import annotations

import math
from typing import Any

from .schema import NOOP_TARGET_SLOT, relative_owner, safe_float, wrap_angle

CENTER = 50.0
BOARD = 100.0
ROTATION_RADIUS_LIMIT = 50.0

PLANET_FEATURE_NAMES = [
    "alive",
    "rel_owner_neutral",
    "rel_owner_own",
    "rel_owner_enemy",
    "x_centered",
    "y_centered",
    "radius_norm",
    "ships_log_norm",
    "production_norm",
    "is_comet",
    "is_orbiting",
    "distance_center_norm",
    "step_norm",
]

PAIR_FEATURE_NAMES = [
    "is_noop",
    "source_ships_log_norm",
    "target_ships_log_norm",
    "target_prod_norm",
    "target_rel_owner_neutral",
    "target_rel_owner_own",
    "target_rel_owner_enemy",
    "dx_norm",
    "dy_norm",
    "distance_norm",
    "direct_angle_sin",
    "direct_angle_cos",
    "capture_needed_norm",
    "ship_margin_norm",
    "amount_fraction",
    "step_norm",
]


def is_orbiting_planet(p: list[Any], initial_by_id: dict[int, list[Any]] | None = None) -> bool:
    if len(p) < 7:
        return False
    pid = int(p[0])
    base = initial_by_id.get(pid, p) if initial_by_id else p
    dx = safe_float(base[2]) - CENTER
    dy = safe_float(base[3]) - CENTER
    r = safe_float(base[4])
    orbital_radius = math.sqrt(dx * dx + dy * dy)
    return orbital_radius + r < ROTATION_RADIUS_LIMIT and orbital_radius > 0.5


def planet_features(obs: dict[str, Any], player_id: int, slot: int) -> list[float]:
    planets = obs.get("planets", [])
    initial_by_id = {int(p[0]): p for p in obs.get("initial_planets", []) if len(p) >= 7}
    comet_ids = set(int(x) for x in obs.get("comet_planet_ids", []) if int(x) >= 0)
    step_norm = safe_float(obs.get("step"), 0.0) / max(safe_float(obs.get("episode_steps"), 500.0), 1.0)
    if slot < 0 or slot >= len(planets) or len(planets[slot]) < 7:
        return [0.0] * len(PLANET_FEATURE_NAMES)
    p = planets[slot]
    owner = int(p[1])
    rel = relative_owner(owner, player_id)
    x = safe_float(p[2])
    y = safe_float(p[3])
    dx = x - CENTER
    dy = y - CENTER
    dist_center = math.sqrt(dx * dx + dy * dy)
    ships = max(0.0, safe_float(p[5]))
    return [
        1.0,
        1.0 if rel == 0 else 0.0,
        1.0 if rel == 1 else 0.0,
        1.0 if rel == -1 else 0.0,
        dx / BOARD,
        dy / BOARD,
        safe_float(p[4]) / 5.0,
        math.log1p(ships) / math.log1p(1000.0),
        safe_float(p[6]) / 5.0,
        1.0 if int(p[0]) in comet_ids else 0.0,
        1.0 if is_orbiting_planet(p, initial_by_id) else 0.0,
        dist_center / (math.sqrt(2.0) * BOARD),
        step_norm,
    ]


def all_planet_features(obs: dict[str, Any], player_id: int, max_planets: int = 64) -> list[list[float]]:
    return [planet_features(obs, player_id, i) for i in range(max_planets)]


def pair_features(obs: dict[str, Any], player_id: int, source_slot: int, target_slot: int, amount_ships: int = 0) -> list[float]:
    planets = obs.get("planets", [])
    step_norm = safe_float(obs.get("step"), 0.0) / max(safe_float(obs.get("episode_steps"), 500.0), 1.0)
    if source_slot < 0 or source_slot >= len(planets) or len(planets[source_slot]) < 7:
        return [0.0] * len(PAIR_FEATURE_NAMES)
    source = planets[source_slot]
    source_ships = max(0.0, safe_float(source[5]))
    amount_fraction = float(amount_ships) / max(source_ships, 1.0) if amount_ships > 0 else 0.0
    if target_slot == NOOP_TARGET_SLOT or target_slot < 0 or target_slot >= len(planets) or len(planets[target_slot]) < 7:
        return [
            1.0,
            math.log1p(source_ships) / math.log1p(1000.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            amount_fraction,
            step_norm,
        ]
    target = planets[target_slot]
    sx = safe_float(source[2])
    sy = safe_float(source[3])
    tx = safe_float(target[2])
    ty = safe_float(target[3])
    dx = tx - sx
    dy = ty - sy
    dist = math.sqrt(dx * dx + dy * dy)
    direct_angle = math.atan2(dy, dx)
    target_owner = int(target[1])
    rel = relative_owner(target_owner, player_id)
    target_ships = max(0.0, safe_float(target[5]))
    capture_needed = 1.0 if rel == 1 else target_ships + 1.0
    ship_margin = float(amount_ships) - capture_needed
    return [
        0.0,
        math.log1p(source_ships) / math.log1p(1000.0),
        math.log1p(target_ships) / math.log1p(1000.0),
        safe_float(target[6]) / 5.0,
        1.0 if rel == 0 else 0.0,
        1.0 if rel == 1 else 0.0,
        1.0 if rel == -1 else 0.0,
        dx / 100.0,
        dy / 100.0,
        dist / (math.sqrt(2.0) * 100.0),
        math.sin(direct_angle),
        math.cos(direct_angle),
        capture_needed / 1000.0,
        ship_margin / 1000.0,
        amount_fraction,
        step_norm,
    ]
