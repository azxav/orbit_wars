from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .schema import wrap_angle, safe_float

CENTER = 50.0
BOARD = 100.0


@dataclass(frozen=True)
class CanonicalObservation:
    obs: dict[str, Any]
    original_player_id: int
    canonical_player_id: int
    num_players: int
    rotation_radians: float
    id_to_canonical_slot: dict[int, int]
    canonical_slot_to_original_slot: dict[int, int]


def infer_num_players(obs: dict[str, Any], player_id: int) -> int:
    raw = obs.get("num_players", obs.get("players", 0))
    try:
        value = int(raw or 0)
    except Exception:
        value = 0
    owners = [int(p[1]) for p in obs.get("planets", []) if isinstance(p, (list, tuple)) and len(p) >= 2 and int(p[1]) >= 0]
    if owners:
        value = max(value, max(owners) + 1)
    return max(value, int(player_id) + 1, 2)


def player_rotation_radians(player_id: int, num_players: int) -> float:
    n = max(2, int(num_players))
    return wrap_angle(-2.0 * math.pi * float(int(player_id)) / float(n))


def rotate_point(x: float, y: float, rotation_radians: float) -> tuple[float, float]:
    dx = safe_float(x) - CENTER
    dy = safe_float(y) - CENTER
    c = math.cos(float(rotation_radians))
    s = math.sin(float(rotation_radians))
    return CENTER + c * dx - s * dy, CENTER + s * dx + c * dy


def canonical_owner(owner: int, player_id: int, num_players: int) -> int:
    owner_i = int(owner)
    if owner_i < 0:
        return owner_i
    return int((owner_i - int(player_id)) % max(2, int(num_players)))


def _transform_planet(p: Any, *, player_id: int, num_players: int, rotation: float) -> list[Any]:
    out = list(p)
    if len(out) >= 2:
        out[1] = canonical_owner(int(out[1]), int(player_id), int(num_players))
    if len(out) >= 4:
        out[2], out[3] = rotate_point(safe_float(out[2]), safe_float(out[3]), rotation)
    return out


def _planet_sort_key(item: tuple[int, list[Any]]) -> tuple[float, float, float, int]:
    old_slot, p = item
    if len(p) < 4:
        return (float("inf"), float("inf"), float("inf"), old_slot)
    return (round(safe_float(p[2]), 6), round(safe_float(p[3]), 6), round(safe_float(p[4]) if len(p) > 4 else 0.0, 6), int(p[0]) if len(p) > 0 else old_slot)


def _transform_planets_with_mapping(planets: list[Any], *, player_id: int, num_players: int, rotation: float) -> tuple[list[list[Any]], dict[int, int], dict[int, int]]:
    transformed: list[tuple[int, list[Any]]] = []
    for old_slot, p in enumerate(planets):
        if isinstance(p, (list, tuple)):
            transformed.append((old_slot, _transform_planet(p, player_id=player_id, num_players=num_players, rotation=rotation)))
    transformed.sort(key=_planet_sort_key)
    out = [p for _, p in transformed]
    id_to_slot: dict[int, int] = {}
    slot_to_old: dict[int, int] = {}
    for new_slot, (old_slot, p) in enumerate(transformed):
        slot_to_old[new_slot] = old_slot
        if len(p) >= 1:
            try:
                id_to_slot[int(p[0])] = new_slot
            except Exception:
                pass
    return out, id_to_slot, slot_to_old


def _transform_planets_sorted(planets: Any, *, player_id: int, num_players: int, rotation: float) -> list[list[Any]]:
    if not isinstance(planets, list):
        return []
    transformed = [(i, _transform_planet(p, player_id=player_id, num_players=num_players, rotation=rotation)) for i, p in enumerate(planets) if isinstance(p, (list, tuple))]
    transformed.sort(key=_planet_sort_key)
    return [p for _, p in transformed]


def _transform_planets_in_id_order(planets: Any, ordered_planets: list[list[Any]], *, player_id: int, num_players: int, rotation: float) -> list[list[Any]]:
    if not isinstance(planets, list):
        planets = []
    by_id: dict[int, list[Any]] = {}
    for p in planets:
        if not isinstance(p, (list, tuple)) or len(p) < 1:
            continue
        transformed = _transform_planet(p, player_id=player_id, num_players=num_players, rotation=rotation)
        try:
            by_id[int(transformed[0])] = transformed
        except Exception:
            continue
    out: list[list[Any]] = []
    for current in ordered_planets:
        if len(current) < 1:
            continue
        try:
            pid = int(current[0])
        except Exception:
            continue
        out.append(list(by_id.get(pid, current)))
    return out


def _transform_path_point(point: Any, rotation: float) -> Any:
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        out = list(point)
        out[0], out[1] = rotate_point(safe_float(out[0]), safe_float(out[1]), rotation)
        return out
    return point


def _transform_comets(comets: Any, rotation: float) -> Any:
    if not isinstance(comets, list):
        return comets
    out = []
    for group in comets:
        if not isinstance(group, dict):
            out.append(group)
            continue
        g = dict(group)
        paths = g.get("paths")
        if isinstance(paths, list):
            g["paths"] = [[_transform_path_point(pt, rotation) for pt in path] if isinstance(path, list) else path for path in paths]
        out.append(g)
    return out


def _transform_fleet(fleet: Any, *, player_id: int, num_players: int, rotation: float) -> Any:
    if isinstance(fleet, dict):
        out = dict(fleet)
        for key in ("owner", "player", "player_id"):
            if key in out and out[key] is not None:
                out[key] = canonical_owner(int(out[key]), player_id, num_players)
        for xkey, ykey in (("x", "y"), ("pos_x", "pos_y"), ("start_x", "start_y")):
            if xkey in out and ykey in out:
                out[xkey], out[ykey] = rotate_point(safe_float(out[xkey]), safe_float(out[ykey]), rotation)
        for akey in ("angle", "heading", "direction"):
            if akey in out and out[akey] is not None:
                out[akey] = wrap_angle(safe_float(out[akey]) + rotation)
        return out
    if isinstance(fleet, (list, tuple)):
        out = list(fleet)
        if len(out) >= 2:
            try:
                out[1] = canonical_owner(int(out[1]), player_id, num_players)
            except Exception:
                pass
        # Native format used in this project: [id, owner, x, y, angle, ships].
        if len(out) == 6:
            out[2], out[3] = rotate_point(safe_float(out[2]), safe_float(out[3]), rotation)
            out[4] = wrap_angle(safe_float(out[4]) + rotation)
        return out
    return fleet


def canonicalize_observation(obs: dict[str, Any], player_id: int | None = None) -> CanonicalObservation:
    raw = copy.deepcopy(dict(obs or {}))
    original_planets = list(raw.get("planets", []) or [])
    original_initial_planets = list(raw.get("initial_planets", original_planets) or [])
    original_player_id = int(raw.get("player", 0) if player_id is None else player_id)
    num_players = infer_num_players(raw, original_player_id)
    rotation = player_rotation_radians(original_player_id, num_players)
    planets, id_to_slot, slot_to_old = _transform_planets_with_mapping(
        original_planets,
        player_id=original_player_id,
        num_players=num_players,
        rotation=rotation,
    )
    raw["planets"] = planets
    raw["initial_planets"] = _transform_planets_in_id_order(
        original_initial_planets,
        planets,
        player_id=original_player_id,
        num_players=num_players,
        rotation=rotation,
    )
    raw["fleets"] = [_transform_fleet(f, player_id=original_player_id, num_players=num_players, rotation=rotation) for f in (raw.get("fleets", []) or [])]
    raw["comets"] = _transform_comets(raw.get("comets", []), rotation)
    raw["player"] = 0
    raw["players"] = num_players
    raw["num_players"] = num_players
    raw["perspective_canonicalized"] = True
    raw["canonicalized_player_id"] = original_player_id
    raw["canonical_rotation_radians"] = float(rotation)
    return CanonicalObservation(
        obs=raw,
        original_player_id=original_player_id,
        canonical_player_id=0,
        num_players=num_players,
        rotation_radians=float(rotation),
        id_to_canonical_slot=id_to_slot,
        canonical_slot_to_original_slot=slot_to_old,
    )


def canonicalize_action(action: Any, transform: CanonicalObservation) -> Iterator[tuple[int, float, int]]:
    if not isinstance(action, list):
        return
    for move in action:
        if not isinstance(move, (list, tuple)) or len(move) < 3:
            continue
        try:
            yield (
                int(round(float(move[0]))),
                wrap_angle(float(move[1]) + float(transform.rotation_radians)),
                int(round(float(move[2]))),
            )
        except Exception:
            continue


def canonicalize_launches(moves: Iterable[tuple[int, float, int]], transform: CanonicalObservation) -> list[tuple[int, float, int]]:
    return [(int(pid), wrap_angle(float(angle) + float(transform.rotation_radians)), int(ships)) for pid, angle, ships in moves]


def uncanonicalize_move(move: list[Any] | tuple[Any, ...] | None, transform: CanonicalObservation) -> list[Any] | None:
    if move is None:
        return None
    if not isinstance(move, (list, tuple)) or len(move) < 3:
        return None
    return [int(move[0]), wrap_angle(float(move[1]) - float(transform.rotation_radians)), int(move[2])]
