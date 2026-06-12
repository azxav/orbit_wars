from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from .state import EnvState, state_from_observation


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if hasattr(value, "keys"):
        return {k: _plain(value[k]) for k in value.keys()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _state_value(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _empty_agent(_obs, _config):
    return []


def generate_official_initial_states(
    *,
    players: int,
    seeds: Iterable[int],
    episode_steps: int = 500,
    ship_speed: float = 6.0,
) -> tuple[EnvState, dict[str, Any]]:
    try:
        from kaggle_environments import make
    except ModuleNotFoundError as exc:
        raise RuntimeError("kaggle_environments is required to build an official state bank") from exc

    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("at least one seed is required")

    states: list[EnvState] = []
    for seed in seed_list:
        env = make("orbit_wars", configuration={"episodeSteps": int(episode_steps), "seed": seed}, debug=False)
        env.run([_empty_agent for _ in range(int(players))])
        initial_frame = env.steps[0]
        obs = _plain(_state_value(initial_frame[0], "observation", {}))
        states.append(
            state_from_observation(
                obs,
                num_players=int(players),
                episode_steps=int(episode_steps),
                ship_speed=float(ship_speed),
            )
        )

    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    metadata = {
        "players": int(players),
        "episode_steps": int(episode_steps),
        "ship_speed": float(ship_speed),
        "source": "kaggle_official",
        "seed_start": int(seed_list[0]),
        "seed_count": int(len(seed_list)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return batched, metadata


def save_state_bank(path: str | Path, states: EnvState, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        f"state/{field.name}": np.asarray(getattr(states, field.name))
        for field in fields(EnvState)
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)


def load_state_bank(path: str | Path) -> tuple[EnvState, dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as data:
        values = {
            field.name: jnp.asarray(data[f"state/{field.name}"])
            for field in fields(EnvState)
        }
        metadata = json.loads(str(data["metadata_json"].item()))
    return EnvState(**values), metadata


def apply_runtime_config(states: EnvState, *, players: int, episode_steps: int, ship_speed: float = 6.0) -> EnvState:
    return EnvState(
        **{
            **states.__dict__,
            "num_players": jnp.full_like(states.num_players, int(players)),
            "episode_steps": jnp.full_like(states.episode_steps, int(episode_steps)),
            "ship_speed": jnp.full_like(states.ship_speed, float(ship_speed)),
        }
    )


def sample_state_bank(states: EnvState, key: jax.Array, *, mode: str, cycle_index: jax.Array) -> tuple[EnvState, jax.Array]:
    bank_size = states.step.shape[0]
    if mode == "random":
        idx = jax.random.randint(key, (), 0, bank_size)
        next_cycle_index = cycle_index
    elif mode == "cycle":
        idx = jnp.mod(cycle_index, bank_size)
        next_cycle_index = cycle_index + jnp.array(1, dtype=cycle_index.dtype)
    else:
        raise ValueError(f"unsupported state bank mode: {mode!r}")
    return jax.tree_util.tree_map(lambda x: x[idx], states), next_cycle_index


def has_imported_comet_paths(states: EnvState) -> bool:
    return bool(np.any(np.asarray(states.planet_is_comet)) and np.any(np.asarray(states.comet_path_len) > 0))
