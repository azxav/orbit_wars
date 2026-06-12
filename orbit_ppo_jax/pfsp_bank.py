from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from .checkpointing import load_jax_checkpoint
from .pfsp import PFSPManifest

REQUIRED_BC_CONFIG_KEYS = (
    "planet_feature_dim",
    "global_feature_dim",
    "target_state_feature_dim",
    "pair_feature_dim",
    "max_planets",
    "target_classes",
    "amount_bins",
    "noop_target_slot",
    "hidden_size",
    "num_layers",
    "num_heads",
)


@dataclass
class PFSPBank:
    bc_params: dict[str, Any]
    active_mask: jnp.ndarray
    entry_ids: list[str]
    entry_kinds: list[str]
    entry_paths: list[str | None]


def _stack_leaf(*xs):
    if xs[0] is None:
        return None
    return jnp.stack(xs, axis=0)


def tree_stack(params_by_slot: list[dict[str, Any]]) -> dict[str, Any]:
    return jax.tree_util.tree_map(_stack_leaf, *params_by_slot, is_leaf=lambda x: x is None)


def tree_take(tree: dict[str, Any], slot: jnp.ndarray) -> dict[str, Any]:
    return jax.tree_util.tree_map(lambda x: None if x is None else x[slot], tree, is_leaf=lambda x: x is None)


def assert_bc_config_compatible(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    for key in REQUIRED_BC_CONFIG_KEYS:
        if expected.get(key) != actual.get(key):
            raise RuntimeError(f"incompatible BC config for PFSP bank: {key} expected {expected.get(key)!r}, got {actual.get(key)!r}")


def _checkpoint_bc_params(path: str | Path, bc_config: dict[str, Any]) -> dict[str, Any]:
    params, config, _metrics = load_jax_checkpoint(path)
    checkpoint_bc_config = config.get("bc_model_config", config.get("model_config", {}))
    assert_bc_config_compatible(bc_config, checkpoint_bc_config)
    if "bc" in params:
        return params["bc"]
    return params


def build_pfsp_bank(manifest: PFSPManifest, initial_bc_params: dict[str, Any], bc_config: dict[str, Any]) -> PFSPBank:
    max_slots = int(manifest.max_policy_slots)
    params_by_slot = [initial_bc_params for _ in range(max_slots)]
    active = [False for _ in range(max_slots)]
    entry_ids = ["" for _ in range(max_slots)]
    entry_kinds = ["" for _ in range(max_slots)]
    entry_paths: list[str | None] = [None for _ in range(max_slots)]

    for entry in manifest.entries:
        if not entry.active or entry.kind != "frozen_policy" or entry.slot is None:
            continue
        slot = int(entry.slot)
        if slot < 0 or slot >= max_slots:
            raise RuntimeError(f"PFSP entry {entry.id!r} slot {slot} is outside max slots {max_slots}")
        if slot == 0:
            params_by_slot[slot] = initial_bc_params
        elif entry.path is None:
            raise RuntimeError(f"PFSP frozen policy entry {entry.id!r} has no checkpoint path")
        else:
            params_by_slot[slot] = _checkpoint_bc_params(entry.path, bc_config)
        active[slot] = True
        entry_ids[slot] = entry.id
        entry_kinds[slot] = entry.kind
        entry_paths[slot] = entry.path

    return PFSPBank(
        bc_params=tree_stack(params_by_slot),
        active_mask=jnp.asarray(active, dtype=jnp.bool_),
        entry_ids=entry_ids,
        entry_kinds=entry_kinds,
        entry_paths=entry_paths,
    )
