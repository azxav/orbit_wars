from __future__ import annotations

import math
from typing import Any

from .schema import relative_owner, safe_float

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
