from __future__ import annotations

import jax.numpy as jnp

from .config import MAX_PLAYERS


def scores_and_ranks(planet_owner, planet_ships, planet_alive, fleet_owner, fleet_ships, fleet_alive):
    scores = jnp.zeros((MAX_PLAYERS,), dtype=jnp.float32)
    for player in range(MAX_PLAYERS):
        p_score = jnp.sum(jnp.where(planet_alive & (planet_owner == player), planet_ships, 0.0))
        f_score = jnp.sum(jnp.where(fleet_alive & (fleet_owner == player), fleet_ships, 0.0))
        scores = scores.at[player].set(p_score + f_score)
    order = jnp.argsort(-scores)
    ranks = jnp.zeros((MAX_PLAYERS,), dtype=jnp.int32)
    ranks = ranks.at[order].set(jnp.arange(1, MAX_PLAYERS + 1, dtype=jnp.int32))
    return scores, ranks


def terminal_rewards(scores, ranks, num_players):
    two_player = jnp.array([1.0, -1.0, 0.0, 0.0], dtype=jnp.float32)
    four_player_by_rank = jnp.array([1.0, 0.3, -0.3, -1.0], dtype=jnp.float32)
    top = jnp.max(jnp.where(jnp.arange(MAX_PLAYERS) < num_players, scores, -1.0))
    winners = (scores == top) & (top > 0.0) & (jnp.arange(MAX_PLAYERS) < num_players)
    rewards_2p = jnp.where(winners, two_player[0], two_player[1])
    rewards_2p = jnp.where(jnp.sum(winners) != 1, 0.0, rewards_2p)
    rewards_4p = four_player_by_rank[jnp.clip(ranks - 1, 0, 3)]
    rewards = jnp.where(num_players == 2, rewards_2p, rewards_4p)
    return jnp.where(jnp.arange(MAX_PLAYERS) < num_players, rewards, 0.0)
