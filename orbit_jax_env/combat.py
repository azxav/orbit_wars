from __future__ import annotations

import jax.numpy as jnp

from .config import MAX_PLAYERS, P_MAX


def resolve_combat(
    planet_owner: jnp.ndarray,
    planet_ships: jnp.ndarray,
    planet_alive: jnp.ndarray,
    hit_planet: jnp.ndarray,
    hit_owner: jnp.ndarray,
    hit_ships: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    owner_out = planet_owner
    ships_out = planet_ships
    pids = jnp.arange(P_MAX, dtype=jnp.int32)

    for pid in range(P_MAX):
        hits = hit_planet == pid
        ships_by_owner = jnp.zeros((MAX_PLAYERS,), dtype=jnp.float32)
        for owner in range(MAX_PLAYERS):
            ships_by_owner = ships_by_owner.at[owner].set(
                jnp.sum(jnp.where(hits & (hit_owner == owner), hit_ships, 0.0))
            )
        top_owner = jnp.argmax(ships_by_owner).astype(jnp.int32)
        top_ships = ships_by_owner[top_owner]
        masked_second = ships_by_owner.at[top_owner].set(-1.0)
        second_ships = jnp.max(masked_second)
        tied = (top_ships > 0.0) & jnp.any((pids[:MAX_PLAYERS] != top_owner) & (ships_by_owner == top_ships))
        survivor_ships = jnp.where(tied, 0.0, top_ships - jnp.maximum(second_ships, 0.0))
        has_survivor = planet_alive[pid] & (survivor_ships > 0.0)
        same_owner = owner_out[pid] == top_owner
        new_ships_same = ships_out[pid] + survivor_ships
        diff = ships_out[pid] - survivor_ships
        attacker_wins = diff < 0.0
        new_owner = jnp.where(attacker_wins, top_owner, owner_out[pid])
        new_ships_diff = jnp.abs(diff)
        owner_out = owner_out.at[pid].set(jnp.where(has_survivor & (~same_owner), new_owner, owner_out[pid]))
        ships_out = ships_out.at[pid].set(
            jnp.where(has_survivor, jnp.where(same_owner, new_ships_same, new_ships_diff), ships_out[pid])
        )
    return owner_out, ships_out
