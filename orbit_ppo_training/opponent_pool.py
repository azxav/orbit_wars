from __future__ import annotations

from pathlib import Path
from typing import Callable

from orbit_bc_eval.base_agents import make_opponent


def make_ppo_opponent(name: str, *, heuristic_path: str | Path = "orbit_wars_base.py") -> Callable:
    key = str(name).lower()
    if key == "orbit_wars_base":
        key = "heuristic_path"
    if key in {"random", "passive", "simple_expand", "heuristic_path"}:
        return make_opponent(key, heuristic_path=heuristic_path)
    if key == "self_play":
        raise NotImplementedError("self_play opponent pool is intentionally deferred")
    raise RuntimeError(f"Unsupported PPO opponent {name!r}")

