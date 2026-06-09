from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .schema import NOOP_TARGET_SLOT, P_MAX, relative_owner, safe_float

CENTER = 50.0
BOARD = 100.0
ROTATION_RADIUS_LIMIT = 50.0
SHIP_LOG_DENOM = math.log1p(1000.0)

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
    "owner_ship_share",
    "owner_prod_share",
    "projected_garrison_20",
    "under_threat_20",
]

GLOBAL_FEATURE_NAMES = [
    "step_norm",
    "remaining_steps_norm",
    "is_2p",
    "is_4p",
    "my_ship_share",
    "my_prod_share",
    "my_planet_share",
    "leader_ship_gap_norm",
    "leader_prod_gap_norm",
    "weakest_enemy_ship_gap_norm",
]

TARGET_STATE_FEATURE_NAMES = [
    "nearest_own_eta_to_target",
    "nearest_enemy_eta_to_target",
    "enemy_before_own_flag",
    "hostile_arrivals_before_10",
    "projected_owner_20",
    "projected_garrison_20",
    "target_contested_flag",
    "target_easy_neutral_flag",
    "target_high_prod_flag",
]

PAIR_FEATURE_NAMES = [
    "capture_needed",
    "capture_ratio",
    "surplus_after_capture",
    "roi_prod_per_ship",
    "is_neutral",
    "is_enemy",
    "is_own",
    "cheap_neutral",
    "high_prod_target",
    "distance",
    "angle_sin",
    "angle_cos",
    "safe_sendable_ships",
    "post_send_frac_capture",
    "overkill_ratio_capture",
    "enemy_before_us",
    "our_arrival_margin",
    "enemy_can_capture_before_us",
    "local_ship_advantage_20",
    "projected_garrison_at_arrival",
    "projected_owner_at_arrival",
    "is_noop_candidate",
]


@dataclass(frozen=True)
class FeatureState:
    planet_features: np.ndarray
    global_features: np.ndarray
    target_state_features: np.ndarray


def ships_from_log_norm(x: float) -> float:
    return math.expm1(float(x) * SHIP_LOG_DENOM)


def _step_norm(obs: dict[str, Any]) -> float:
    return safe_float(obs.get("step"), 0.0) / max(safe_float(obs.get("episode_steps"), 500.0), 1.0)


def _num_players(obs: dict[str, Any]) -> int:
    owners = {int(p[1]) for p in obs.get("planets", []) if len(p) >= 7 and int(p[1]) >= 0}
    player_count = int(obs.get("num_players", obs.get("players", 0)) or 0)
    return max(player_count, len(owners), 2)


def _owner_totals(obs: dict[str, Any]) -> dict[int, dict[str, float]]:
    totals: dict[int, dict[str, float]] = {}
    for p in obs.get("planets", [])[:P_MAX]:
        if len(p) < 7:
            continue
        owner = int(p[1])
        if owner < 0:
            continue
        row = totals.setdefault(owner, {"ships": 0.0, "prod": 0.0, "planets": 0.0})
        row["ships"] += max(0.0, safe_float(p[5]))
        row["prod"] += max(0.0, safe_float(p[6]))
        row["planets"] += 1.0
    return totals


def _fleet_owner(fleet: Any) -> int | None:
    if isinstance(fleet, dict):
        for key in ("owner", "player", "player_id"):
            if key in fleet:
                return int(fleet[key])
        return None
    if isinstance(fleet, (list, tuple)) and len(fleet) >= 2:
        return int(fleet[1])
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


def _fleet_eta(fleet: Any) -> float:
    if isinstance(fleet, dict):
        for key in ("eta", "remaining_turns", "turns_remaining", "remaining"):
            if key in fleet:
                return safe_float(fleet[key], math.inf)
        return math.inf
    if isinstance(fleet, (list, tuple)):
        for idx in (6, 7, 8):
            if len(fleet) > idx:
                v = safe_float(fleet[idx], math.inf)
                if math.isfinite(v):
                    return v
    return math.inf


def defaultdict_floats() -> dict[str, float]:
    return {
        "enemy_5": 0.0,
        "enemy_10": 0.0,
        "enemy_20": 0.0,
        "friendly_5": 0.0,
        "friendly_10": 0.0,
        "friendly_20": 0.0,
    }


def _incoming_by_slot(obs: dict[str, Any], player_id: int, max_planets: int) -> dict[int, dict[str, float]]:
    id_to_slot = {int(p[0]): i for i, p in enumerate(obs.get("planets", [])[:max_planets]) if len(p) >= 7}
    out: dict[int, dict[str, float]] = {i: defaultdict_floats() for i in range(max_planets)}
    for fleet in obs.get("fleets", []) or []:
        target_id = _fleet_target_id(fleet)
        if target_id not in id_to_slot:
            continue
        slot = id_to_slot[target_id]
        owner = _fleet_owner(fleet)
        eta = _fleet_eta(fleet)
        ships = _fleet_ships(fleet)
        if not math.isfinite(eta):
            continue
        prefix = "friendly" if owner == int(player_id) else "enemy"
        for horizon in (5, 10, 20):
            if eta <= horizon:
                out[slot][f"{prefix}_{horizon}"] += ships
    return out


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


def planet_features(obs: dict[str, Any], player_id: int, slot: int, max_planets: int = P_MAX) -> list[float]:
    planets = obs.get("planets", [])[:max_planets]
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
    prod = max(0.0, safe_float(p[6]))
    totals = _owner_totals(obs)
    owner_total = totals.get(owner, {"ships": 0.0, "prod": 0.0}) if owner >= 0 else {"ships": 0.0, "prod": 0.0}
    incoming = _incoming_by_slot(obs, player_id, max_planets).get(slot, defaultdict_floats())
    projected_20 = ships + incoming["friendly_20"] - incoming["enemy_20"] if rel == 1 else ships + incoming["enemy_20"] - incoming["friendly_20"]
    initial_by_id = {int(x[0]): x for x in obs.get("initial_planets", []) if len(x) >= 7}
    comet_ids = set(int(x) for x in obs.get("comet_planet_ids", []) if int(x) >= 0)
    row = [
        1.0,
        1.0 if rel == 0 else 0.0,
        1.0 if rel == 1 else 0.0,
        1.0 if rel == -1 else 0.0,
        dx / BOARD,
        dy / BOARD,
        safe_float(p[4]) / 5.0,
        math.log1p(ships) / SHIP_LOG_DENOM,
        prod / 5.0,
        1.0 if int(p[0]) in comet_ids else 0.0,
        1.0 if is_orbiting_planet(p, initial_by_id) else 0.0,
        dist_center / (math.sqrt(2.0) * BOARD),
        ships / max(1.0, owner_total["ships"]),
        prod / max(1.0, owner_total["prod"]),
        projected_20 / 100.0,
        1.0 if incoming["enemy_20"] > ships + incoming["friendly_20"] else 0.0,
    ]
    return [float(x) for x in np.nan_to_num(np.asarray(row, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)]


def all_planet_features(obs: dict[str, Any], player_id: int, max_planets: int = P_MAX) -> np.ndarray:
    return np.asarray([planet_features(obs, player_id, i, max_planets) for i in range(max_planets)], dtype=np.float32)


def global_features(obs: dict[str, Any], player_id: int, max_planets: int = P_MAX) -> np.ndarray:
    totals = _owner_totals(obs)
    players = _num_players(obs)
    my = totals.get(int(player_id), {"ships": 0.0, "prod": 0.0, "planets": 0.0})
    total_ships = sum(v["ships"] for v in totals.values())
    total_prod = sum(v["prod"] for v in totals.values())
    total_owned_planets = sum(v["planets"] for v in totals.values())
    leader_ships = max((v["ships"] for v in totals.values()), default=0.0)
    leader_prod = max((v["prod"] for v in totals.values()), default=0.0)
    enemy_ships = [v["ships"] for owner, v in totals.items() if owner != int(player_id)]
    weakest_enemy = min(enemy_ships, default=0.0)
    step = _step_norm(obs)
    arr = np.asarray(
        [
            step,
            max(0.0, 1.0 - step),
            1.0 if players <= 2 else 0.0,
            1.0 if players >= 4 else 0.0,
            my["ships"] / max(1.0, total_ships),
            my["prod"] / max(1.0, total_prod),
            my["planets"] / max(1.0, total_owned_planets),
            (my["ships"] - leader_ships) / max(1.0, total_ships),
            (my["prod"] - leader_prod) / max(1.0, total_prod),
            (my["ships"] - weakest_enemy) / max(1.0, total_ships),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def target_state_features(obs: dict[str, Any], player_id: int, max_planets: int = P_MAX) -> np.ndarray:
    planets = obs.get("planets", [])[:max_planets]
    incoming = _incoming_by_slot(obs, player_id, max_planets)
    out = np.zeros((max_planets, len(TARGET_STATE_FEATURE_NAMES)), dtype=np.float32)
    own_slots = [i for i, p in enumerate(planets) if len(p) >= 7 and int(p[1]) == int(player_id)]
    enemy_slots = [i for i, p in enumerate(planets) if len(p) >= 7 and int(p[1]) >= 0 and int(p[1]) != int(player_id)]
    for target_slot in range(max_planets):
        if target_slot >= len(planets) or len(planets[target_slot]) < 7:
            continue
        target = planets[target_slot]
        tx, ty = safe_float(target[2]), safe_float(target[3])

        def nearest(slots: list[int]) -> float:
            best_eta = math.inf
            for slot in slots:
                p = planets[slot]
                dist = math.hypot(safe_float(p[2]) - tx, safe_float(p[3]) - ty)
                best_eta = min(best_eta, dist / 10.0)
            return 0.0 if not math.isfinite(best_eta) else min(1.0, best_eta / 50.0)

        own_eta = nearest(own_slots)
        enemy_eta = nearest(enemy_slots)
        inc = incoming.get(target_slot, defaultdict_floats())
        owner = relative_owner(int(target[1]), player_id)
        projected_20 = safe_float(target[5]) + inc["friendly_20"] - inc["enemy_20"]
        out[target_slot] = np.asarray(
            [
                own_eta,
                enemy_eta,
                1.0 if enemy_eta < own_eta else 0.0,
                inc["enemy_10"] / 100.0,
                float(owner if projected_20 > 0 else -owner),
                projected_20 / 100.0,
                1.0 if inc["friendly_10"] > 0 and inc["enemy_10"] > 0 else 0.0,
                1.0 if int(target[1]) < 0 and safe_float(target[5]) <= 5.0 else 0.0,
                1.0 if safe_float(target[6]) >= 3.0 else 0.0,
            ],
            dtype=np.float32,
        )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_state(obs: dict[str, Any], player_id: int, max_planets: int = P_MAX) -> FeatureState:
    return FeatureState(
        planet_features=all_planet_features(obs, player_id, max_planets),
        global_features=global_features(obs, player_id, max_planets),
        target_state_features=target_state_features(obs, player_id, max_planets),
    )


def pair_features_from_dense(
    planet_features: np.ndarray,
    target_state_features: np.ndarray,
    source_slot: int,
    *,
    max_planets: int = P_MAX,
) -> np.ndarray:
    out = np.zeros((max_planets + 1, len(PAIR_FEATURE_NAMES)), dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(PLANET_FEATURE_NAMES)}
    target_name_to_idx = {name: i for i, name in enumerate(TARGET_STATE_FEATURE_NAMES)}
    if not (0 <= int(source_slot) < max_planets):
        out[NOOP_TARGET_SLOT, -1] = 1.0
        return out
    source = planet_features[int(source_slot)]
    sx = float(source[name_to_idx["x_centered"]])
    sy = float(source[name_to_idx["y_centered"]])
    source_ships = max(0.0, ships_from_log_norm(float(source[name_to_idx["ships_log_norm"]])))
    source_prod = max(0.0, float(source[name_to_idx["production_norm"]]) * 5.0)
    source_under_threat = float(source[name_to_idx["under_threat_20"]])
    for target_slot in range(max_planets):
        target = planet_features[target_slot]
        if float(target[name_to_idx["alive"]]) <= 0.0:
            continue
        tx = float(target[name_to_idx["x_centered"]])
        ty = float(target[name_to_idx["y_centered"]])
        dx = tx - sx
        dy = ty - sy
        distance = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) if distance > 0 else 0.0
        target_ships = max(0.0, ships_from_log_norm(float(target[name_to_idx["ships_log_norm"]])))
        target_prod = max(0.0, float(target[name_to_idx["production_norm"]]) * 5.0)
        is_own = float(target[name_to_idx["rel_owner_own"]])
        is_enemy = float(target[name_to_idx["rel_owner_enemy"]])
        is_neutral = float(target[name_to_idx["rel_owner_neutral"]])
        capture_needed = 1.0 if is_own > 0.5 else target_ships + 1.0
        safe_reserve = 2.0 + source_prod + 10.0 * source_under_threat
        safe_sendable = max(0.0, source_ships - safe_reserve)
        ts = target_state_features[target_slot]
        nearest_enemy = float(ts[target_name_to_idx["nearest_enemy_eta_to_target"]])
        our_arrival = min(1.0, (distance * 10.0) / 50.0)
        projected_20 = float(ts[target_name_to_idx["projected_garrison_20"]])
        row = [
            capture_needed / 100.0,
            capture_needed / max(1.0, source_ships),
            (source_ships - capture_needed) / 100.0,
            target_prod / max(1.0, capture_needed),
            is_neutral,
            is_enemy,
            is_own,
            1.0 if is_neutral > 0.5 and capture_needed <= 5.0 else 0.0,
            1.0 if target_prod >= 3.0 else 0.0,
            distance,
            math.sin(angle),
            math.cos(angle),
            safe_sendable / 100.0,
            (source_ships - capture_needed) / max(1.0, source_ships),
            source_ships / max(1.0, capture_needed),
            1.0 if nearest_enemy < our_arrival else 0.0,
            nearest_enemy - our_arrival,
            1.0 if nearest_enemy < our_arrival and is_enemy > 0.5 else 0.0,
            (source_ships / 100.0) - projected_20,
            projected_20,
            float(ts[target_name_to_idx["projected_owner_20"]]),
            0.0,
        ]
        out[target_slot] = np.asarray(row, dtype=np.float32)
    out[NOOP_TARGET_SLOT, -1] = 1.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
