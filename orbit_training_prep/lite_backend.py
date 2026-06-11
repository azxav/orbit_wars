from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import PAIR_FEATURE_NAMES, PLANET_FEATURE_NAMES, FeatureState, ships_from_log_norm
from .schema import (
    AMOUNT_BIN_NAMES,
    AMOUNT_BIN_NONE,
    NOOP_TARGET_ID,
    NOOP_TARGET_SLOT,
    P_MAX,
    angle_abs_diff,
    build_planet_slot_maps,
    capture_needed_ships,
    decode_amount_bin,
    encode_amount_bin,
    owned_source_slots,
    relative_owner,
    safe_float,
)
from .target_inference import FastProjector, InferredMove, fleet_speed_formula

LITE_TARGET_INFERENCE_MODE = "lite-arrival"
LITE_MASK_MODE = "lite-permissive-label-corrected"
LITE_PAIR_ETA_MODE = "lite-movement-cache"
LAUNCH_SURFACE_OFFSET = 0.1
TARGET_HIT_SURFACE_OFFSET = 0.0


@dataclass(frozen=True)
class LiteMovementCache:
    x: np.ndarray  # [H+1, P]
    y: np.ndarray
    alive_by_step: np.ndarray
    radii: np.ndarray
    owner: np.ndarray
    ships: np.ndarray
    prod: np.ndarray


@dataclass(frozen=True)
class LiteDistanceCache:
    cross_dist: np.ndarray  # [H+1, P_src, P_tgt], dist(source@0, target@k)
    alive_by_step: np.ndarray
    K: int


@dataclass(frozen=True)
class LiteGarrisonStatus:
    owner: np.ndarray  # [P, H+1]
    ships: np.ndarray
    enemy_before: np.ndarray  # [P, H+1]
    friendly_before: np.ndarray  # [P, H+1]


@dataclass(frozen=True)
class LiteStateContext:
    obs: dict[str, Any]
    player_id: int
    horizon: int
    movement: LiteMovementCache
    distance_cache: LiteDistanceCache
    garrison_status: LiteGarrisonStatus


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
        if len(fleet) == 6:
            return None
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
        if len(fleet) == 6:
            return math.inf
        for idx in (6, 7, 8):
            if len(fleet) > idx:
                value = safe_float(fleet[idx], math.inf)
                if math.isfinite(value):
                    return value
    return math.inf


def _fleet_ships(fleet: Any) -> float:
    if isinstance(fleet, dict):
        for key in ("ships", "num_ships", "ship_count"):
            if key in fleet:
                return max(0.0, safe_float(fleet[key]))
        return 0.0
    if isinstance(fleet, (list, tuple)):
        if len(fleet) == 6:
            return max(0.0, safe_float(fleet[5]))
        for idx in (6, 5, 4, 3):
            if len(fleet) > idx:
                value = safe_float(fleet[idx], -1.0)
                if value >= 0.0:
                    return value
    return 0.0


def build_lite_context(obs: dict[str, Any], player_id: int, *, horizon: int = 160, device: str = "cpu") -> LiteStateContext:
    del device
    h = max(1, int(horizon))
    planets = obs.get("planets", [])[:P_MAX]
    projector = FastProjector(obs, h)
    x = np.zeros((h + 1, P_MAX), dtype=np.float32)
    y = np.zeros((h + 1, P_MAX), dtype=np.float32)
    alive = np.zeros((h + 1, P_MAX), dtype=bool)
    radii = np.zeros((P_MAX,), dtype=np.float32)
    owner = np.full((P_MAX,), -1, dtype=np.int16)
    ships = np.zeros((P_MAX,), dtype=np.float32)
    prod = np.zeros((P_MAX,), dtype=np.float32)
    for slot in range(P_MAX):
        if slot < len(planets) and len(planets[slot]) >= 7:
            radii[slot] = safe_float(planets[slot][4])
            owner[slot] = int(planets[slot][1])
            ships[slot] = safe_float(planets[slot][5])
            prod[slot] = safe_float(planets[slot][6])
        for k in range(h + 1):
            px, py, ok = projector.position_at(slot, k)
            x[k, slot] = px
            y[k, slot] = py
            alive[k, slot] = bool(ok and slot < len(planets) and len(planets[slot]) >= 7 and int(planets[slot][0]) >= 0)
    dx = x[0].reshape(1, P_MAX, 1) - x.reshape(h + 1, 1, P_MAX)
    dy = y[0].reshape(1, P_MAX, 1) - y.reshape(h + 1, 1, P_MAX)
    cross_dist = np.sqrt(np.maximum(dx * dx + dy * dy, 0.0)).astype(np.float32)
    movement = LiteMovementCache(x=x, y=y, alive_by_step=alive, radii=radii, owner=owner, ships=ships, prod=prod)
    distance_cache = LiteDistanceCache(cross_dist=cross_dist, alive_by_step=alive, K=h)
    garrison = _build_lite_garrison(obs, int(player_id), movement, h)
    return LiteStateContext(obs=obs, player_id=int(player_id), horizon=h, movement=movement, distance_cache=distance_cache, garrison_status=garrison)


def _build_lite_garrison(obs: dict[str, Any], player_id: int, movement: LiteMovementCache, horizon: int) -> LiteGarrisonStatus:
    owner = np.repeat(movement.owner.reshape(P_MAX, 1), horizon + 1, axis=1).astype(np.int16)
    ships = np.zeros((P_MAX, horizon + 1), dtype=np.float32)
    enemy_before = np.zeros((P_MAX, horizon + 1), dtype=np.float32)
    friendly_before = np.zeros((P_MAX, horizon + 1), dtype=np.float32)
    for k in range(horizon + 1):
        grows = movement.owner >= 0
        ships[:, k] = movement.ships + np.where(grows, movement.prod * float(k), 0.0)
    id_to_slot = {int(p[0]): i for i, p in enumerate(obs.get("planets", [])[:P_MAX]) if len(p) >= 7}
    for fleet in obs.get("fleets", []) or []:
        tid = _fleet_target_id(fleet)
        eta = _fleet_eta(fleet)
        if tid not in id_to_slot or not math.isfinite(eta):
            continue
        slot = id_to_slot[int(tid)]
        k = int(max(0, min(horizon, math.ceil(float(eta)))))
        amount = _fleet_ships(fleet)
        if _fleet_owner(fleet) == int(player_id):
            friendly_before[slot, k:] += amount
        else:
            enemy_before[slot, k:] += amount
    return LiteGarrisonStatus(owner=owner, ships=ships, enemy_before=enemy_before, friendly_before=friendly_before)


def _surface_eta_from_cache(ctx: LiteStateContext, source_slot: int, target_slot: int, ships: float) -> float:
    if int(source_slot) == int(target_slot):
        return 0.0
    if source_slot < 0 or target_slot < 0 or source_slot >= P_MAX or target_slot >= P_MAX:
        return math.inf
    speed = max(1e-6, fleet_speed_formula(float(ships)))
    sr = float(ctx.movement.radii[int(source_slot)])
    tr = float(ctx.movement.radii[int(target_slot)])
    for k in range(1, int(ctx.distance_cache.K) + 1):
        if not bool(ctx.distance_cache.alive_by_step[k, int(target_slot)]):
            continue
        surface = max(0.0, float(ctx.distance_cache.cross_dist[k, int(source_slot), int(target_slot)]) - sr - tr - LAUNCH_SURFACE_OFFSET - TARGET_HIT_SURFACE_OFFSET)
        if surface / speed <= float(k):
            return float(k)
    k = int(ctx.distance_cache.K)
    surface = max(0.0, float(ctx.distance_cache.cross_dist[k, int(source_slot), int(target_slot)]) - sr - tr)
    return min(float(k), surface / speed)


def lite_candidate_etas(ctx: LiteStateContext, source_slot: int, ships_by_target: list[float]) -> list[float]:
    return [_surface_eta_from_cache(ctx, int(source_slot), int(target_slot), float(ships)) for target_slot, ships in enumerate(ships_by_target[:P_MAX])]


def compute_lite_viability_masks(ctx: LiteStateContext, *, labels_by_source: dict[int, tuple[int, int]] | None = None) -> tuple[np.ndarray, np.ndarray]:
    amount_bins = len(AMOUNT_BIN_NAMES)
    target_mask = np.zeros((P_MAX, P_MAX + 1), dtype=bool)
    amount_mask = np.zeros((P_MAX, P_MAX + 1, amount_bins), dtype=bool)
    target_mask[:, NOOP_TARGET_SLOT] = True
    amount_mask[:, NOOP_TARGET_SLOT, AMOUNT_BIN_NONE] = True
    planets = ctx.obs.get("planets", [])[:P_MAX]
    alive_targets = [i for i, p in enumerate(planets) if len(p) >= 7 and int(p[0]) >= 0]
    for source_slot in owned_source_slots(ctx.obs, int(ctx.player_id)):
        if source_slot >= len(planets) or len(planets[source_slot]) < 7:
            continue
        source = planets[source_slot]
        available = safe_float(source[5])
        for target_slot in alive_targets:
            if int(target_slot) == int(source_slot):
                continue
            target = planets[target_slot]
            target_mask[source_slot, target_slot] = True
            needed = capture_needed_ships(source, target, int(ctx.player_id))
            for amount_bin in range(1, amount_bins):
                ships = decode_amount_bin(amount_bin, available, needed)
                amount_mask[source_slot, target_slot, amount_bin] = 0 < int(ships) <= int(math.floor(available))
    for source_slot, (target_label, amount_label) in (labels_by_source or {}).items():
        if 0 <= int(source_slot) < P_MAX and 0 <= int(target_label) <= NOOP_TARGET_SLOT:
            target_mask[int(source_slot), int(target_label)] = True
            if 0 <= int(amount_label) < amount_bins:
                amount_mask[int(source_slot), int(target_label), int(amount_label)] = True
    return target_mask, amount_mask


def pair_features_lite(
    ctx: LiteStateContext,
    feature_state: FeatureState,
    source_slot: int,
    *,
    target_viability_mask: np.ndarray | None = None,
    amount_viability_mask: np.ndarray | None = None,
) -> np.ndarray:
    fs = feature_state
    planets = ctx.obs.get("planets", [])[:P_MAX]
    out = np.zeros((P_MAX + 1, len(PAIR_FEATURE_NAMES)), dtype=np.float32)
    if not (0 <= int(source_slot) < len(planets)) or len(planets[int(source_slot)]) < 7:
        return out
    ni = {name: i for i, name in enumerate(PLANET_FEATURE_NAMES)}
    src = fs.planet_features[int(source_slot)]
    sx = float(src[ni["x_centered"]])
    sy = float(src[ni["y_centered"]])
    ss = max(0.0, ships_from_log_norm(float(src[ni["ships_log_norm"]])))
    sp = max(0.0, float(src[ni["production_norm"]]) * 5.0)
    threat = float(src[ni["under_threat_20"]])
    abc = max(1, int(np.asarray(amount_viability_mask).shape[-1]) - 1) if amount_viability_mask is not None else 0
    eta_ships_by_target = [1.0] * P_MAX
    for tslot in range(P_MAX):
        if tslot < len(planets) and len(planets[tslot]) >= 7 and float(fs.planet_features[tslot][ni["alive"]]) > 0.0:
            tgt = fs.planet_features[tslot]
            ts = max(0.0, ships_from_log_norm(float(tgt[ni["ships_log_norm"]])))
            own = float(tgt[ni["rel_owner_own"]])
            need = 1.0 if own > 0.5 else ts + 1.0
            eta_ships_by_target[tslot] = max(1.0, min(max(1.0, ss), max(1.0, need)))
    etas = lite_candidate_etas(ctx, int(source_slot), eta_ships_by_target)
    for tslot in range(P_MAX):
        if tslot >= len(planets) or len(planets[tslot]) < 7:
            continue
        tgt = fs.planet_features[tslot]
        if float(tgt[ni["alive"]]) <= 0.0:
            continue
        dx = float(tgt[ni["x_centered"]]) - sx
        dy = float(tgt[ni["y_centered"]]) - sy
        dist = math.hypot(dx, dy)
        ang = math.atan2(dy, dx) if dist > 0 else 0.0
        ts = max(0.0, ships_from_log_norm(float(tgt[ni["ships_log_norm"]])))
        tp = max(0.0, float(tgt[ni["production_norm"]]) * 5.0)
        own = float(tgt[ni["rel_owner_own"]])
        need = 1.0 if own > 0.5 else ts + 1.0
        safe = max(0.0, ss - (2.0 + sp + 10.0 * threat))
        gaf = float(np.asarray(amount_viability_mask)[tslot, 1:].sum()) / float(abc) if amount_viability_mask is not None and abc > 0 else 0.0
        eta = float(etas[tslot]) if math.isfinite(float(etas[tslot])) else float(ctx.horizon)
        k = int(max(0, min(ctx.horizon, round(eta))))
        proj_owner = int(ctx.garrison_status.owner[tslot, k])
        proj_ships = float(ctx.garrison_status.ships[tslot, k])
        enemy_before = float(ctx.garrison_status.enemy_before[tslot, k])
        friendly_before = float(ctx.garrison_status.friendly_before[tslot, k])
        rel_proj = relative_owner(proj_owner, int(ctx.player_id))
        hostile_need = 0.0 if rel_proj == 1 else proj_ships + 1.0
        capture_margin = (ss - hostile_need) / 100.0
        values = {
            "capture_ratio": need / max(1.0, ss),
            "surplus_after_capture": (ss - need) / 100.0,
            "roi_prod_per_ship": tp / max(1.0, need),
            "distance": dist,
            "angle_sin": math.sin(ang),
            "angle_cos": math.cos(ang),
            "geom_viable_amount_frac": gaf,
            "safe_sendable_ships": safe / 100.0,
            "post_send_frac_capture": (ss - need) / max(1.0, ss),
            "our_eta_norm": min(1.0, eta / float(max(1, ctx.horizon))),
            "enemy_ships_before_our_arrival": enemy_before / 100.0,
            "friendly_ships_before_our_arrival": friendly_before / 100.0,
            "projected_garrison_at_our_arrival": proj_ships / 100.0,
            "projected_owner_at_our_arrival": float(rel_proj),
            "target_capture_margin_at_arrival": capture_margin,
        }
        out[tslot] = np.asarray([values[name] for name in PAIR_FEATURE_NAMES], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class LiteTargetInferer:
    def __init__(self, *, horizon: int = 160, max_angle_error: float = math.pi):
        self.horizon = int(horizon)
        self.max_angle_error = float(max_angle_error)

    def infer_moves(self, ctx: LiteStateContext, moves: list[tuple[int, float, int]]) -> list[InferredMove]:
        id_to_slot, id_to_planet = build_planet_slot_maps(ctx.obs)
        out: list[InferredMove] = []
        for from_planet_id, raw_angle, ships in moves:
            source_planet = id_to_planet.get(int(from_planet_id))
            source_slot = id_to_slot.get(int(from_planet_id), -1)
            available = int(math.floor(safe_float(source_planet[5]))) if source_planet is not None else 0
            if source_planet is None or source_slot < 0:
                out.append(self._invalid(-1, int(from_planet_id), raw_angle, ships, 0, "source_not_found"))
                continue
            if int(source_planet[1]) != int(ctx.player_id):
                out.append(self._invalid(source_slot, int(from_planet_id), raw_angle, ships, available, "source_not_owned"))
                continue
            if int(ships) <= 0 or int(ships) > available:
                out.append(self._invalid(source_slot, int(from_planet_id), raw_angle, ships, available, "bad_ship_count"))
                continue
            target_slot, err, eta = self._best_target(ctx, int(source_slot), float(raw_angle), int(ships))
            if target_slot == NOOP_TARGET_SLOT:
                out.append(self._invalid(source_slot, int(from_planet_id), raw_angle, ships, available, "no_target"))
                continue
            target_planet = ctx.obs["planets"][target_slot]
            cap_need = capture_needed_ships(source_planet, target_planet, int(ctx.player_id))
            amount_bin = encode_amount_bin(int(ships), available, cap_need)
            out.append(InferredMove(
                True,
                int(source_slot),
                int(from_planet_id),
                float(raw_angle),
                int(ships),
                int(available),
                int(target_slot),
                int(target_planet[0]),
                float(eta),
                int(target_slot),
                int(target_planet[0]),
                LITE_TARGET_INFERENCE_MODE,
                float(err),
                True,
                int(amount_bin),
                int(cap_need),
                float(ships) / float(max(available, 1)),
                "",
            ))
        return out

    def _best_target(self, ctx: LiteStateContext, source_slot: int, raw_angle: float, ships: int) -> tuple[int, float, float]:
        if not (0 <= int(source_slot) < P_MAX):
            return NOOP_TARGET_SLOT, math.inf, math.inf
        sx = float(ctx.movement.x[0, source_slot])
        sy = float(ctx.movement.y[0, source_slot])
        best: tuple[float, int, float, float] | None = None
        for target_slot in range(P_MAX):
            if target_slot == source_slot or not bool(ctx.movement.alive_by_step[0, target_slot]):
                continue
            eta = _surface_eta_from_cache(ctx, source_slot, target_slot, float(ships))
            if not math.isfinite(eta):
                continue
            k = int(max(0, min(ctx.horizon, round(eta))))
            tx = float(ctx.movement.x[k, target_slot])
            ty = float(ctx.movement.y[k, target_slot])
            pred_angle = math.atan2(ty - sy, tx - sx)
            err = angle_abs_diff(float(raw_angle), pred_angle)
            if err > self.max_angle_error:
                continue
            score = err * 10.0 + eta / max(1.0, float(ctx.horizon))
            if best is None or score < best[0]:
                best = (score, int(target_slot), float(err), float(eta))
        if best is None:
            return NOOP_TARGET_SLOT, math.inf, math.inf
        _, target_slot, err, eta = best
        return target_slot, err, eta

    def _invalid(self, source_slot: int, source_planet_id: int, raw_angle: float, ships: int, available: int, reason: str) -> InferredMove:
        return InferredMove(False, int(source_slot), int(source_planet_id), float(raw_angle), int(ships), int(available), NOOP_TARGET_SLOT, NOOP_TARGET_ID, math.inf, NOOP_TARGET_SLOT, NOOP_TARGET_ID, "invalid_" + reason, math.inf, False, 0, 1, 0.0, reason)
