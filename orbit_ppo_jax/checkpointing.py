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


def assert_finite_tree(tree: dict[str, Any]) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if not all(bool(jnp.all(jnp.isfinite(x))) for x in leaves):
        raise RuntimeError("checkpoint parameter tree contains non-finite values")
