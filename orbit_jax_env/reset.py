from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import COMET_PATH_MAX, COMET_SPAWN_COUNT, EnvConfig
from .config import P_MAX
from .state import EnvState, empty_state


def _build_comet_schedule(key: jax.Array, enable: bool):
    spawn_idx = jnp.arange(COMET_SPAWN_COUNT, dtype=jnp.float32)
    t = jnp.linspace(0.0, 1.0, COMET_PATH_MAX, dtype=jnp.float32)
    phase = jax.random.uniform(key, (COMET_SPAWN_COUNT,), minval=-0.25, maxval=0.25)
    y_base = 18.0 + 5.0 * jnp.sin((t[None, :] + phase[:, None]) * jnp.pi)
    x_base = 99.0 - 3.2 * jnp.arange(COMET_PATH_MAX, dtype=jnp.float32)[None, :] - 2.0 * spawn_idx[:, None]
    x_base = jnp.clip(x_base, 1.0, 99.0)
    y_base = jnp.clip(y_base, 1.0, 99.0)
    xs = jnp.stack([x_base, 100.0 - y_base, y_base, 100.0 - x_base], axis=1)
    ys = jnp.stack([y_base, x_base, 100.0 - x_base, 100.0 - y_base], axis=1)
    lens = jnp.full((COMET_SPAWN_COUNT,), COMET_PATH_MAX, dtype=jnp.int32)
    ships = 3.0 + (jnp.arange(COMET_SPAWN_COUNT, dtype=jnp.float32) % 5.0)
    spawned = jnp.full((COMET_SPAWN_COUNT,), not bool(enable), dtype=jnp.bool_)
    return xs, ys, lens, ships, spawned


def reset(key: jax.Array, config: EnvConfig | None = None) -> EnvState:
    cfg = config or EnvConfig()
    k_angle, k_group, k_comets = jax.random.split(key, 3)
    angular_velocity = jax.random.uniform(k_angle, (), minval=0.025, maxval=0.05)
    home_group = jax.random.randint(k_group, (), 0, 4)
    base_rows = jnp.array(
        [
            [0, -1, 25.0, 25.0, 2.6, 18.0, 2.0],
            [1, -1, 75.0, 25.0, 2.6, 18.0, 2.0],
            [2, -1, 25.0, 75.0, 2.6, 18.0, 2.0],
            [3, -1, 75.0, 75.0, 2.6, 18.0, 2.0],
            [4, -1, 35.0, 50.0, 2.1, 12.0, 1.0],
            [5, -1, 65.0, 50.0, 2.1, 12.0, 1.0],
            [6, -1, 50.0, 35.0, 2.1, 12.0, 1.0],
            [7, -1, 50.0, 65.0, 2.1, 12.0, 1.0],
            [8, -1, 20.0, 50.0, 2.4, 20.0, 3.0],
            [9, -1, 80.0, 50.0, 2.4, 20.0, 3.0],
            [10, -1, 50.0, 20.0, 2.4, 20.0, 3.0],
            [11, -1, 50.0, 80.0, 2.4, 20.0, 3.0],
            [12, -1, 15.0, 15.0, 2.0, 8.0, 1.0],
            [13, -1, 85.0, 15.0, 2.0, 8.0, 1.0],
            [14, -1, 15.0, 85.0, 2.0, 8.0, 1.0],
            [15, -1, 85.0, 85.0, 2.0, 8.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    owners_2p = jnp.array([0, -1, -1, 1], dtype=jnp.float32)
    owners_4p = jnp.array([0, 1, 2, 3], dtype=jnp.float32)
    owners = jnp.where(int(cfg.num_players) == 2, owners_2p, owners_4p)
    row_idx = jnp.arange(base_rows.shape[0])
    group_start = home_group * 4
    in_home = (row_idx >= group_start) & (row_idx < group_start + 4)
    owner_by_row = owners[jnp.clip(row_idx - group_start, 0, 3)]
    owner_col = jnp.where(in_home, owner_by_row, base_rows[:, 1])
    ships_col = jnp.where(in_home & (owner_by_row >= 0.0), 10.0, base_rows[:, 5])
    rows = base_rows.at[:, 1].set(owner_col).at[:, 5].set(ships_col)

    state = empty_state(num_players=int(cfg.num_players), episode_steps=int(cfg.episode_steps), ship_speed=float(cfg.ship_speed))
    n = rows.shape[0]
    idx = jnp.arange(n)
    comet_spawn_x, comet_spawn_y, comet_spawn_len, comet_spawn_ships, comet_spawned = _build_comet_schedule(k_comets, bool(cfg.enable_comets))
    return EnvState(
        **{
            **state.__dict__,
            "planet_id": state.planet_id.at[idx].set(rows[:, 0].astype(jnp.int32)),
            "planet_owner": state.planet_owner.at[idx].set(rows[:, 1].astype(jnp.int32)),
            "planet_x": state.planet_x.at[idx].set(rows[:, 2]),
            "planet_y": state.planet_y.at[idx].set(rows[:, 3]),
            "planet_initial_x": state.planet_initial_x.at[idx].set(rows[:, 2]),
            "planet_initial_y": state.planet_initial_y.at[idx].set(rows[:, 3]),
            "planet_radius": state.planet_radius.at[idx].set(rows[:, 4]),
            "planet_ships": state.planet_ships.at[idx].set(rows[:, 5]),
            "planet_production": state.planet_production.at[idx].set(rows[:, 6]),
            "planet_alive": state.planet_alive.at[idx].set(True),
            "angular_velocity": angular_velocity.astype(jnp.float32),
            "comet_spawn_x": comet_spawn_x,
            "comet_spawn_y": comet_spawn_y,
            "comet_spawn_len": comet_spawn_len,
            "comet_spawn_ships": comet_spawn_ships,
            "comet_spawned": comet_spawned,
        }
    )
