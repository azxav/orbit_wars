from __future__ import annotations

import jax.numpy as jnp

from .config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS


def empty_actions() -> jnp.ndarray:
    return jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
