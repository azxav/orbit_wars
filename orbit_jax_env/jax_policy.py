"""Pure-JAX policies for the vmapped rollout (P1 scaffolding).

`greedy_policy_apply` is a `policy_apply` compatible with `rollout.jax_rollout`:
it takes batched observations and returns `{"actions": [B, MAX_PLAYERS,
MAX_ACTIONS_PER_PLAYER, 3]}`. It implements a nearest-target capture heuristic
in pure JAX (no Python loops, fully jit/vmap-able), so it both:
  * proves the end-to-end GPU self-play loop (env + policy on device), and
  * serves as a fast scripted opponent in the self-play pool.

The learned Flax policy will plug into the same interface; the aim helper here
(`aim_angle`) is the JAX angle solver both will share. Lead-targeting for
orbiting planets is a TODO (start with direct atan2; add the intercept
fixed-point loop next — same math as OrbitLord's `solve_intercept`).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS, P_MAX


def aim_angle(sx, sy, tx, ty):
    """Direct angle from source to target. TODO(lead): intercept loop."""
    return jnp.arctan2(ty - sy, tx - sx)


def greedy_actions(planets, num_players):
    """planets: [P,9] = id,owner,x,y,r,ships,prod,alive,is_comet.
    Returns [MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3] = (from_planet, angle, ships).
    For each player, every owned planet sends floor(ships/2) to its nearest
    non-owned alive planet (>=2 ships required)."""
    pid = planets[:, 0]
    owner = planets[:, 1].astype(jnp.int32)
    x = planets[:, 2]
    y = planets[:, 3]
    ships = planets[:, 5]
    alive = planets[:, 7] > 0.5

    # pairwise distances [src, tgt]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx * dx + dy * dy)

    players = jnp.arange(MAX_PLAYERS)

    def per_player(p):
        non_owned = alive & (owner != p)                       # [tgt]
        big = jnp.where(non_owned[None, :], dist, jnp.inf)      # [src,tgt]
        tgt = jnp.argmin(big, axis=1)                          # [src] nearest non-owned
        has_target = jnp.isfinite(jnp.min(big, axis=1))        # [src]
        tx = x[tgt]
        ty = y[tgt]
        ang = aim_angle(x, y, tx, ty)                          # [src]
        send = jnp.floor(ships / 2.0)
        owned_src = (owner == p) & alive & (send >= 2.0) & has_target & (p < num_players)
        rows = jnp.stack([
            jnp.where(owned_src, pid, 0.0),
            jnp.where(owned_src, ang, 0.0),
            jnp.where(owned_src, send, 0.0),
        ], axis=-1)                                            # [P,3]
        # pad/truncate P -> MAX_ACTIONS_PER_PLAYER
        return rows[:MAX_ACTIONS_PER_PLAYER]

    return jax.vmap(per_player)(players)                       # [MAX_PLAYERS, A, 3]


def greedy_policy_apply(params, obs):
    """policy_apply for jax_rollout. obs batched: planets [B,P,9], global [B,6]."""
    planets = obs["planets"]
    glob = obs["global"]
    num_players = glob[:, 3].astype(jnp.int32)                 # global[3] = num_players
    actions = jax.vmap(greedy_actions)(planets, num_players)
    B = planets.shape[0]
    return {
        "actions": actions,
        "logprobs": jnp.zeros((B,), dtype=jnp.float32),
        "values": jnp.zeros((B,), dtype=jnp.float32),
    }
