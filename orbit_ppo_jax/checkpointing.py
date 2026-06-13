from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _flatten(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            out.update(_flatten(value, child))
    elif isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            child = f"{prefix}/{idx}" if prefix else str(idx)
            out.update(_flatten(value, child))
    elif tree is not None:
        out[prefix] = np.asarray(tree)
    return out


def _insert(root: dict[str, Any], path: str, value: np.ndarray) -> None:
    parts = path.split("/")
    cur: Any = root
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = jnp.asarray(value)


def _dicts_to_lists(obj: Any) -> Any:
    if isinstance(obj, dict):
        converted = {k: _dicts_to_lists(v) for k, v in obj.items()}
        if converted and all(str(k).isdigit() for k in converted):
            return [converted[str(i)] for i in range(len(converted))]
        return converted
    return obj


def save_jax_checkpoint(path: str | Path, params: dict[str, Any], config: dict[str, Any], metrics: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "params.npz", **_flatten(params))
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    with open(path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics or {}, f, indent=2, sort_keys=True)


def load_jax_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = Path(path)
    params_file = path / "params.npz" if path.is_dir() else path
    root: dict[str, Any] = {}
    with np.load(params_file, allow_pickle=False) as data:
        for key in data.files:
            _insert(root, key, data[key])
    params = _dicts_to_lists(root)
    config_file = params_file.with_name("config.json")
    metrics_file = params_file.with_name("metrics.json")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return params, config, metrics


def _tree_arrays(tree: Any) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]


def _restore_tree(data: Any, prefix: str, template: Any) -> Any:
    leaves, treedef = jax.tree_util.tree_flatten(template)
    restored = []
    for idx, _leaf in enumerate(leaves):
        key = f"{prefix}/{idx:06d}"
        if key not in data:
            raise RuntimeError(f"training checkpoint is missing {key}")
        restored.append(jnp.asarray(data[key]))
    return jax.tree_util.tree_unflatten(treedef, restored)


def save_jax_training_state(
    path: str | Path,
    *,
    opt_state: Any,
    rng_key: Any,
    env_states: Any,
    state_bank_cycle_index: Any,
    update_index: int,
    env_steps: int,
    best_score: float,
    players: int,
    envs: int,
    episode_steps: int,
    enable_comets: bool,
    initial_state_bank: str | None,
    state_bank_mode: str,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "rng_key": np.asarray(rng_key),
        "state_bank_cycle_index": np.asarray(state_bank_cycle_index),
        "update_index": np.asarray(int(update_index), dtype=np.int64),
        "env_steps": np.asarray(int(env_steps), dtype=np.int64),
        "best_score": np.asarray(float(best_score), dtype=np.float64),
        "players": np.asarray(int(players), dtype=np.int64),
        "envs": np.asarray(int(envs), dtype=np.int64),
        "episode_steps": np.asarray(int(episode_steps), dtype=np.int64),
        "enable_comets": np.asarray(bool(enable_comets), dtype=np.bool_),
        "initial_state_bank": np.asarray("" if initial_state_bank is None else str(initial_state_bank)),
        "state_bank_mode": np.asarray(str(state_bank_mode)),
    }
    for idx, leaf in enumerate(_tree_arrays(opt_state)):
        arrays[f"opt_state/{idx:06d}"] = leaf
    for idx, leaf in enumerate(_tree_arrays(env_states)):
        arrays[f"env_states/{idx:06d}"] = leaf
    np.savez_compressed(path / "trainer_state.npz", **arrays)


def load_jax_training_state(
    path: str | Path,
    *,
    opt_state_template: Any,
    env_states_template: Any | None = None,
) -> dict[str, Any]:
    path = Path(path)
    state_file = path / "trainer_state.npz"
    if not state_file.exists():
        raise FileNotFoundError(state_file)
    with np.load(state_file, allow_pickle=False) as data:
        loaded: dict[str, Any] = {
            "rng_key": jnp.asarray(data["rng_key"]),
            "state_bank_cycle_index": jnp.asarray(data["state_bank_cycle_index"]),
            "update_index": int(np.asarray(data["update_index"])),
            "env_steps": int(np.asarray(data["env_steps"])),
            "best_score": float(np.asarray(data["best_score"])),
            "players": int(np.asarray(data["players"])),
            "envs": int(np.asarray(data["envs"])),
            "episode_steps": int(np.asarray(data["episode_steps"])),
            "enable_comets": bool(np.asarray(data["enable_comets"])),
            "initial_state_bank": str(np.asarray(data["initial_state_bank"])),
            "state_bank_mode": str(np.asarray(data["state_bank_mode"])),
            "opt_state": _restore_tree(data, "opt_state", opt_state_template),
        }
        if env_states_template is not None:
            loaded["env_states"] = _restore_tree(data, "env_states", env_states_template)
        if loaded["initial_state_bank"] == "":
            loaded["initial_state_bank"] = None
        return loaded


def assert_finite_tree(tree: dict[str, Any]) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if not all(bool(jnp.all(jnp.isfinite(x))) for x in leaves):
        raise RuntimeError("checkpoint parameter tree contains non-finite values")
