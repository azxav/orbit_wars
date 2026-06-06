from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

P_MAX = 64
F_MAX = 256
NOOP_TARGET_SLOT = P_MAX
NOOP_TARGET_ID = -999
MAX_STEP_DEFAULT = 500

AMOUNT_BIN_NAMES = [
    "none",
    "one_ship",
    "capture_plus_one",
    "quarter",
    "half",
    "three_quarter",
    "all",
]

AMOUNT_BIN_NONE = 0
AMOUNT_BIN_ONE = 1
AMOUNT_BIN_CAPTURE = 2
AMOUNT_BIN_QUARTER = 3
AMOUNT_BIN_HALF = 4
AMOUNT_BIN_THREE_QUARTER = 5
AMOUNT_BIN_ALL = 6


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_abs_diff(a: float, b: float) -> float:
    return abs(wrap_angle(a - b))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def normalize_replay_obs(obs: dict[str, Any], *, player_id: int, step_index: int, episode_steps: int = MAX_STEP_DEFAULT) -> dict[str, Any]:
    """Make Kaggle replay observation complete from the perspective of one player."""
    out = dict(obs or {})
    out["player"] = int(player_id)
    out["step"] = int(out.get("step", step_index))
    out["episode_steps"] = int(out.get("episode_steps", episode_steps))
    out.setdefault("planets", [])
    out.setdefault("initial_planets", out.get("planets", []))
    out.setdefault("fleets", [])
    out.setdefault("comets", [])
    out.setdefault("comet_planet_ids", [])
    out.setdefault("angular_velocity", 0.0)
    out.setdefault("next_fleet_id", 0)
    out.setdefault("remainingOverageTime", 0.0)
    return out


def build_planet_slot_maps(obs: dict[str, Any]) -> tuple[dict[int, int], dict[int, list[Any]]]:
    id_to_slot: dict[int, int] = {}
    id_to_planet: dict[int, list[Any]] = {}
    for slot, p in enumerate(obs.get("planets", [])[:P_MAX]):
        if len(p) < 7:
            continue
        pid = int(p[0])
        id_to_slot[pid] = slot
        id_to_planet[pid] = p
    return id_to_slot, id_to_planet


def owned_source_slots(obs: dict[str, Any], player_id: int) -> list[int]:
    out: list[int] = []
    for slot, p in enumerate(obs.get("planets", [])[:P_MAX]):
        if len(p) >= 7 and int(p[1]) == int(player_id) and safe_float(p[5]) >= 1:
            out.append(slot)
    return out


def alive_target_slots(obs: dict[str, Any], *, include_noop: bool = True, exclude_slot: int | None = None) -> list[int]:
    out: list[int] = []
    for slot, p in enumerate(obs.get("planets", [])[:P_MAX]):
        if len(p) >= 7 and int(p[0]) >= 0 and slot != exclude_slot:
            out.append(slot)
    if include_noop:
        out.append(NOOP_TARGET_SLOT)
    return out


def capture_needed_ships(source_planet: list[Any] | None, target_planet: list[Any] | None, player_id: int) -> int:
    if target_planet is None or len(target_planet) < 7:
        return 1
    owner = int(target_planet[1])
    ships = max(0, int(round(safe_float(target_planet[5]))))
    if owner == int(player_id):
        return 1
    return ships + 1


def amount_candidates(available: float, capture_needed: int) -> dict[int, int]:
    avail = max(0, int(math.floor(safe_float(available))))
    cap = max(1, min(avail, int(capture_needed))) if avail > 0 else 0
    candidates = {
        AMOUNT_BIN_NONE: 0,
        AMOUNT_BIN_ONE: min(1, avail),
        AMOUNT_BIN_CAPTURE: cap,
        AMOUNT_BIN_QUARTER: max(1, int(round(0.25 * avail))) if avail > 0 else 0,
        AMOUNT_BIN_HALF: max(1, int(round(0.50 * avail))) if avail > 0 else 0,
        AMOUNT_BIN_THREE_QUARTER: max(1, int(round(0.75 * avail))) if avail > 0 else 0,
        AMOUNT_BIN_ALL: avail,
    }
    return candidates


def encode_amount_bin(num_ships: float, available: float, capture_needed: int) -> int:
    ships = int(round(safe_float(num_ships)))
    if ships <= 0:
        return AMOUNT_BIN_NONE
    candidates = amount_candidates(available, capture_needed)
    best_bin = AMOUNT_BIN_ONE
    best_err = float("inf")
    for b, v in candidates.items():
        if b == AMOUNT_BIN_NONE:
            continue
        err = abs(float(ships) - float(v))
        # Prefer tactical exact capture over generic fractions when close.
        if err < best_err or (err == best_err and b == AMOUNT_BIN_CAPTURE):
            best_bin = b
            best_err = err
    return int(best_bin)


def decode_amount_bin(amount_bin: int, available: float, capture_needed: int) -> int:
    candidates = amount_candidates(available, capture_needed)
    return int(candidates.get(int(amount_bin), 0))


def relative_owner(owner: int, player_id: int) -> int:
    if owner < 0:
        return 0
    if owner == int(player_id):
        return 1
    return -1


@dataclass(frozen=True)
class ActionSpaceSpec:
    """Single contract shared by dataset, BC, and RL action decoder."""
    policy_granularity: str = "per_owned_source_per_turn"
    target_slots: int = P_MAX + 1
    noop_target_slot: int = NOOP_TARGET_SLOT
    amount_bins: tuple[str, ...] = tuple(AMOUNT_BIN_NAMES)
    angle_policy: str = "geometry_solver_not_model_output"
    multi_launch_v1: str = "largest_launch_primary_with_multi_launch_flag"

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_granularity": self.policy_granularity,
            "target_slots": self.target_slots,
            "noop_target_slot": self.noop_target_slot,
            "amount_bins": list(self.amount_bins),
            "angle_policy": self.angle_policy,
            "multi_launch_v1": self.multi_launch_v1,
        }
