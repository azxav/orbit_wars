from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path
from typing import Any, Callable

import torch

from orbit_training_prep.geometry_bridge import make_geometry
from orbit_training_prep.schema import P_MAX, capture_needed_ships, owned_source_slots, safe_float


def _player_id(obs: dict[str, Any]) -> int:
    return int(obs.get("player", 0) or 0)


def _valid_targets(obs: dict[str, Any], source_slot: int) -> list[int]:
    out = []
    for slot, p in enumerate(obs.get("planets", [])[:P_MAX]):
        if len(p) >= 7 and int(p[0]) >= 0 and slot != source_slot:
            out.append(slot)
    return out


def _to_move(obs: dict[str, Any], player_id: int, source_slot: int, target_slot: int, ships: int, geometry=None) -> list[list[Any]]:
    if ships <= 0:
        return []
    geom = geometry or make_geometry(device="cpu")
    return geom.to_env_moves(
        obs=obs,
        source_slots=torch.as_tensor([source_slot], dtype=torch.long),
        target_slots=torch.as_tensor([target_slot], dtype=torch.long),
        ships=torch.as_tensor([ships], dtype=torch.float32),
        player_id=player_id,
        valid=torch.as_tensor([True], dtype=torch.bool),
    )


def passive_agent(obs, config):
    return []


def random_valid_agent(obs, config, *, geometry=None, rng: random.Random | None = None):
    obs = dict(obs or {})
    player_id = _player_id(obs)
    sources = owned_source_slots(obs, player_id)
    if not sources:
        return []
    local_rng = rng or random
    source_slot = local_rng.choice(sources)
    targets = _valid_targets(obs, source_slot)
    if not targets:
        return []
    target_slot = local_rng.choice(targets)
    source = obs["planets"][source_slot]
    ships = max(1, int(math.floor(safe_float(source[5]) * local_rng.uniform(0.10, 0.35))))
    ships = min(ships, int(safe_float(source[5])))
    return _to_move(obs, player_id, source_slot, target_slot, ships, geometry)


def simple_expand_agent(obs, config, *, geometry=None):
    obs = dict(obs or {})
    player_id = _player_id(obs)
    best: tuple[float, int, int, int] | None = None
    for source_slot in owned_source_slots(obs, player_id):
        source = obs["planets"][source_slot]
        available = int(safe_float(source[5]))
        if available <= 1:
            continue
        sx = safe_float(source[2])
        sy = safe_float(source[3])
        for target_slot, target in enumerate(obs.get("planets", [])[:P_MAX]):
            if len(target) < 7 or int(target[1]) != -1 or target_slot == source_slot:
                continue
            needed = capture_needed_ships(source, target, player_id)
            if needed <= 0 or needed > available:
                continue
            dx = safe_float(target[2]) - sx
            dy = safe_float(target[3]) - sy
            dist = math.sqrt(dx * dx + dy * dy)
            candidate = (dist, source_slot, target_slot, int(needed))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return []
    _, source_slot, target_slot, ships = best
    return _to_move(obs, player_id, source_slot, target_slot, ships, geometry)


def load_heuristic_agent(path: str | Path) -> Callable:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("orbit_bc_eval_loaded_heuristic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load heuristic module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("agent", "heuristic_agent", "my_agent"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(f"Heuristic module {path} does not define agent(obs, config)")


def make_opponent(name: str, *, heuristic_path: str | Path | None = None) -> Callable:
    key = name.lower()
    if key == "passive":
        return passive_agent
    if key == "random":
        return random_valid_agent
    if key == "simple_expand":
        return simple_expand_agent
    if key == "heuristic_path":
        if heuristic_path is None:
            raise RuntimeError("--heuristic_path is required when --opponent heuristic_path")
        return load_heuristic_agent(heuristic_path)
    raise RuntimeError(f"Unsupported opponent {name!r}")
