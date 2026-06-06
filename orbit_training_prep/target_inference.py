from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .exact_target_sim import ExactTargetSimulator
from .schema import (
    NOOP_TARGET_ID,
    NOOP_TARGET_SLOT,
    alive_target_slots,
    angle_abs_diff,
    build_planet_slot_maps,
    capture_needed_ships,
    encode_amount_bin,
    safe_float,
)

CENTER = 50.0
SUN_RADIUS = 10.0
ROT_RADIUS_LIMIT = 50.0
LAUNCH_SURFACE_OFFSET = 0.1
TARGET_HIT_SURFACE_OFFSET = 0.0


def fleet_speed_formula(ships: int | float, max_speed: float = 6.0) -> float:
    s = max(1.0, float(ships))
    return 1.0 + (max_speed - 1.0) * (math.log(s) / math.log(1000.0)) ** 1.5


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx = bx - ax
    vy = by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    cx = ax + t * vx
    cy = ay + t * vy
    return math.hypot(px - cx, py - cy)


@dataclass(frozen=True)
class InferredMove:
    valid_source: bool
    source_slot: int
    source_planet_id: int
    raw_angle: float
    ships: int
    available_ships: int
    contact_target_slot: int
    contact_target_id: int
    contact_eta: float
    inferred_target_slot: int
    inferred_target_id: int
    target_inference_method: str
    angle_error: float
    geometry_viable: bool
    amount_bin: int
    capture_needed: int
    amount_fraction: float
    invalid_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class FastProjector:
    """Pure-Python mirror of the extracted geometry skeleton's future position logic."""

    def __init__(self, obs: dict[str, Any], horizon: int):
        self.obs = obs
        self.horizon = int(horizon)
        self.planets = obs.get("planets", [])
        self.initial_by_id = {int(p[0]): p for p in obs.get("initial_planets", []) if len(p) >= 7}
        self.step = int(obs.get("step", 0) or 0)
        self.angular_velocity = safe_float(obs.get("angular_velocity"), 0.0)
        self.comet_path_by_id: dict[int, tuple[list[Any], int]] = {}
        for group in obs.get("comets", []) or []:
            ids = group.get("planet_ids", []) or []
            paths = group.get("paths", []) or []
            path_index = int(group.get("path_index", -1))
            for i, pid in enumerate(ids):
                if i < len(paths):
                    self.comet_path_by_id[int(pid)] = (paths[i], path_index)

    def position_at(self, slot: int, k: float | int) -> tuple[float, float, bool]:
        if slot < 0 or slot >= len(self.planets) or len(self.planets[slot]) < 7:
            return 0.0, 0.0, False
        p = self.planets[slot]
        pid = int(p[0])
        kk = int(max(0, min(self.horizon, round(float(k)))))
        if pid in self.comet_path_by_id:
            path, idx0 = self.comet_path_by_id[pid]
            idx = idx0 + kk
            if idx0 >= 0 and 0 <= idx < len(path):
                return safe_float(path[idx][0]), safe_float(path[idx][1]), True
            return safe_float(p[2]), safe_float(p[3]), False
        base = self.initial_by_id.get(pid, p)
        dx0 = safe_float(base[2]) - CENTER
        dy0 = safe_float(base[3]) - CENTER
        r_orbit = math.hypot(dx0, dy0)
        planet_radius = safe_float(base[4])
        if r_orbit + planet_radius < ROT_RADIUS_LIMIT and r_orbit > 0.5:
            a0 = math.atan2(dy0, dx0)
            a = a0 + self.angular_velocity * float(self.step + kk)
            return CENTER + r_orbit * math.cos(a), CENTER + r_orbit * math.sin(a), True
        return safe_float(p[2]), safe_float(p[3]), True


class TargetInferer:
    """Map raw Kaggle moves to model labels using exact first-hit simulation."""

    def __init__(self, *, horizon: int = 96, device: str = "cpu"):
        self.horizon = int(horizon)
        self.device = device
        self.exact_sim = ExactTargetSimulator(horizon=horizon, device=device)

    def infer_moves(self, obs: dict[str, Any], player_id: int, moves: list[tuple[int, float, int]]) -> list[InferredMove]:
        if not moves:
            return []
        id_to_slot, id_to_planet = build_planet_slot_maps(obs)
        projector = FastProjector(obs, self.horizon)
        out: list[InferredMove] = []
        for from_planet_id, raw_angle, ships in moves:
            source_planet = id_to_planet.get(int(from_planet_id))
            source_slot = id_to_slot.get(int(from_planet_id), -1)
            available = int(math.floor(safe_float(source_planet[5]))) if source_planet is not None else 0
            if source_planet is None or source_slot < 0:
                out.append(self._invalid(-1, int(from_planet_id), raw_angle, ships, 0, "source_not_found")); continue
            if int(source_planet[1]) != int(player_id):
                out.append(self._invalid(source_slot, int(from_planet_id), raw_angle, ships, available, "source_not_owned")); continue
            if int(ships) <= 0 or int(ships) > available:
                out.append(self._invalid(source_slot, int(from_planet_id), raw_angle, ships, available, "bad_ship_count")); continue
            hit = self.exact_sim.first_hit_for_launch(
                obs,
                player_id,
                {
                    "source_planet_id": int(from_planet_id),
                    "source_slot": int(source_slot),
                    "raw_angle": float(raw_angle),
                    "ships": int(ships),
                },
            )
            fallback_slot, fallback_id, fallback_err, fallback_eta, _fallback_viable = self._angular_nearest(
                obs,
                projector,
                int(source_slot),
                source_planet,
                float(raw_angle),
                int(ships),
            )
            if hit["hit_type"] in {"planet", "comet"}:
                contact_target_slot = int(hit["hit_slot"])
                contact_target_id = int(hit["hit_id"])
                contact_eta = float(hit["eta"]) if hit["eta"] is not None else math.inf
                target_slot = contact_target_slot
                target_id = contact_target_id
                err = angle_abs_diff(float(raw_angle), self._angle_to_slot(obs, projector, source_slot, target_slot))
                eta = contact_eta
                viable = True
                method = "first_contact"
            else:
                contact_target_slot = NOOP_TARGET_SLOT
                contact_target_id = NOOP_TARGET_ID
                contact_eta = float(hit["eta"]) if hit["eta"] is not None else math.inf
                target_slot = fallback_slot
                target_id = fallback_id
                err = fallback_err
                eta = fallback_eta
                viable = False
                method = "angular_nearest"
            target_planet = obs["planets"][target_slot] if 0 <= target_slot < len(obs.get("planets", [])) else None
            cap_need = capture_needed_ships(source_planet, target_planet, player_id)
            amount_bin = encode_amount_bin(int(ships), available, cap_need)
            out.append(InferredMove(
                True,
                int(source_slot),
                int(from_planet_id),
                float(raw_angle),
                int(ships),
                int(available),
                int(contact_target_slot),
                int(contact_target_id),
                float(contact_eta),
                int(target_slot),
                int(target_id),
                method,
                float(err),
                bool(viable),
                int(amount_bin),
                int(cap_need),
                float(ships) / float(max(available, 1)),
                "",
            ))
        return out

    def infer_move(self, obs: dict[str, Any], player_id: int, move: tuple[int, float, int]) -> InferredMove:
        return self.infer_moves(obs, player_id, [move])[0]

    def _angular_nearest(
        self,
        obs: dict[str, Any],
        projector: FastProjector,
        source_slot: int,
        source_planet: list[Any],
        raw_angle: float,
        ships: int,
    ) -> tuple[int, int, float, float, bool]:
        sx, sy, _ = projector.position_at(source_slot, 0)
        sr = safe_float(source_planet[4])
        speed = fleet_speed_formula(ships)
        best: tuple[float, float, bool, float, int] | None = None
        for target_slot in alive_target_slots(obs, include_noop=False, exclude_slot=source_slot):
            if target_slot < 0 or target_slot >= len(obs.get("planets", [])):
                continue
            target = obs["planets"][target_slot]
            tr = safe_float(target[4])
            t0x, t0y, alive = projector.position_at(target_slot, 0)
            if not alive:
                continue
            gap = sr + tr + LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET
            eta = max(0.0, min(float(self.horizon), (math.hypot(t0x - sx, t0y - sy) - gap) / max(speed, 1e-6)))
            tx, ty = t0x, t0y
            alive_future = alive
            for _ in range(4):
                tx, ty, alive_future = projector.position_at(target_slot, eta)
                eta = max(0.0, min(float(self.horizon), (math.hypot(tx - sx, ty - sy) - gap) / max(speed, 1e-6)))
            tx, ty, alive_future = projector.position_at(target_slot, eta)
            if not alive_future:
                continue
            pred_angle = math.atan2(ty - sy, tx - sx)
            err = angle_abs_diff(float(raw_angle), pred_angle)
            launch_x = sx + math.cos(pred_angle) * (sr + LAUNCH_SURFACE_OFFSET)
            launch_y = sy + math.sin(pred_angle) * (sr + LAUNCH_SURFACE_OFFSET)
            sun_dist = point_segment_distance(CENTER, CENTER, launch_x, launch_y, tx, ty)
            sun_clear = sun_dist >= SUN_RADIUS
            score = err + (0.75 if not sun_clear else 0.0)
            if best is None or score < best[0]:
                best = (score, err, bool(sun_clear), eta, int(target_slot))
        if best is None:
            return NOOP_TARGET_SLOT, NOOP_TARGET_ID, math.inf, math.inf, False
        _, err, viable, eta, target_slot = best
        return int(target_slot), int(obs["planets"][target_slot][0]), float(err), float(eta), bool(viable)

    def _angle_to_slot(self, obs: dict[str, Any], projector: FastProjector, source_slot: int, target_slot: int) -> float:
        sx, sy, _ = projector.position_at(source_slot, 0)
        tx, ty, _ = projector.position_at(target_slot, 0)
        return math.atan2(ty - sy, tx - sx)

    def _invalid(self, source_slot: int, source_planet_id: int, raw_angle: float, ships: int, available: int, reason: str) -> InferredMove:
        return InferredMove(False, int(source_slot), int(source_planet_id), float(raw_angle), int(ships), int(available), NOOP_TARGET_SLOT, NOOP_TARGET_ID, math.inf, NOOP_TARGET_SLOT, NOOP_TARGET_ID, "invalid_" + reason, math.inf, False, 0, 1, 0.0, reason)
