from __future__ import annotations

import jax.numpy as jnp

from .config import MAX_PLAYERS
from .state import EnvState


def build_observation(state: EnvState) -> dict[str, jnp.ndarray]:
    planets = jnp.stack(
        [
            state.planet_id.astype(jnp.float32),
            state.planet_owner.astype(jnp.float32),
            state.planet_x,
            state.planet_y,
            state.planet_radius,
            state.planet_ships,
            state.planet_production,
            state.planet_alive.astype(jnp.float32),
            state.planet_is_comet.astype(jnp.float32),
        ],
        axis=-1,
    )
    fleets = jnp.stack(
        [
            jnp.arange(state.fleet_owner.shape[0], dtype=jnp.float32),
            state.fleet_owner.astype(jnp.float32),
            state.fleet_x,
            state.fleet_y,
            state.fleet_angle,
            state.fleet_source.astype(jnp.float32),
            state.fleet_ships,
            state.fleet_alive.astype(jnp.float32),
        ],
        axis=-1,
    )
    players = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)[:, None]
    valid_sources = state.planet_alive[None, :] & (state.planet_owner[None, :] == players) & (state.planet_ships[None, :] >= 1.0)
    valid_sources = valid_sources & (players < state.num_players)
    valid_targets = state.planet_alive[None, :] & (players < state.num_players)
    global_tensor = jnp.array(
        [
            state.step.astype(jnp.float32),
            state.episode_steps.astype(jnp.float32),
            state.angular_velocity,
            state.num_players.astype(jnp.float32),
            jnp.sum(state.planet_alive).astype(jnp.float32),
            jnp.sum(state.fleet_alive).astype(jnp.float32),
        ],
        dtype=jnp.float32,
    )
    return {
        "planets": planets,
        "fleets": fleets,
        "global": global_tensor,
        "valid_source_mask": valid_sources,
        "valid_target_mask": valid_targets,
    }
