from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .config import COMET_PATH_MAX, COMET_SPAWN_COUNT, F_MAX, MAX_PLAYERS, P_MAX


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EnvState:
    planet_id: jax.Array
    planet_owner: jax.Array
    planet_x: jax.Array
    planet_y: jax.Array
    planet_initial_x: jax.Array
    planet_initial_y: jax.Array
    planet_radius: jax.Array
    planet_ships: jax.Array
    planet_production: jax.Array
    planet_alive: jax.Array
    planet_is_comet: jax.Array
    comet_path_x: jax.Array
    comet_path_y: jax.Array
    comet_path_len: jax.Array
    comet_path_index: jax.Array
    comet_spawn_x: jax.Array
    comet_spawn_y: jax.Array
    comet_spawn_len: jax.Array
    comet_spawn_ships: jax.Array
    comet_spawned: jax.Array
    fleet_owner: jax.Array
    fleet_x: jax.Array
    fleet_y: jax.Array
    fleet_angle: jax.Array
    fleet_source: jax.Array
    fleet_ships: jax.Array
    fleet_alive: jax.Array
    next_fleet_id: jax.Array
    step: jax.Array
    angular_velocity: jax.Array
    num_players: jax.Array
    episode_steps: jax.Array
    ship_speed: jax.Array

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return (
            (
                self.planet_id,
                self.planet_owner,
                self.planet_x,
                self.planet_y,
                self.planet_initial_x,
                self.planet_initial_y,
                self.planet_radius,
                self.planet_ships,
                self.planet_production,
                self.planet_alive,
                self.planet_is_comet,
                self.comet_path_x,
                self.comet_path_y,
                self.comet_path_len,
                self.comet_path_index,
                self.comet_spawn_x,
                self.comet_spawn_y,
                self.comet_spawn_len,
                self.comet_spawn_ships,
                self.comet_spawned,
                self.fleet_owner,
                self.fleet_x,
                self.fleet_y,
                self.fleet_angle,
                self.fleet_source,
                self.fleet_ships,
                self.fleet_alive,
                self.next_fleet_id,
                self.step,
                self.angular_velocity,
                self.num_players,
                self.episode_steps,
                self.ship_speed,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "EnvState":
        return cls(*children)


def empty_state(*, num_players: int = 2, episode_steps: int = 500, ship_speed: float = 6.0) -> EnvState:
    return EnvState(
        planet_id=jnp.full((P_MAX,), -1, dtype=jnp.int32),
        planet_owner=jnp.full((P_MAX,), -1, dtype=jnp.int32),
        planet_x=jnp.zeros((P_MAX,), dtype=jnp.float32),
        planet_y=jnp.zeros((P_MAX,), dtype=jnp.float32),
        planet_initial_x=jnp.zeros((P_MAX,), dtype=jnp.float32),
        planet_initial_y=jnp.zeros((P_MAX,), dtype=jnp.float32),
        planet_radius=jnp.zeros((P_MAX,), dtype=jnp.float32),
        planet_ships=jnp.zeros((P_MAX,), dtype=jnp.float32),
        planet_production=jnp.zeros((P_MAX,), dtype=jnp.float32),
        planet_alive=jnp.zeros((P_MAX,), dtype=jnp.bool_),
        planet_is_comet=jnp.zeros((P_MAX,), dtype=jnp.bool_),
        comet_path_x=jnp.zeros((P_MAX, COMET_PATH_MAX), dtype=jnp.float32),
        comet_path_y=jnp.zeros((P_MAX, COMET_PATH_MAX), dtype=jnp.float32),
        comet_path_len=jnp.zeros((P_MAX,), dtype=jnp.int32),
        comet_path_index=jnp.full((P_MAX,), -1, dtype=jnp.int32),
        comet_spawn_x=jnp.zeros((COMET_SPAWN_COUNT, 4, COMET_PATH_MAX), dtype=jnp.float32),
        comet_spawn_y=jnp.zeros((COMET_SPAWN_COUNT, 4, COMET_PATH_MAX), dtype=jnp.float32),
        comet_spawn_len=jnp.zeros((COMET_SPAWN_COUNT,), dtype=jnp.int32),
        comet_spawn_ships=jnp.zeros((COMET_SPAWN_COUNT,), dtype=jnp.float32),
        comet_spawned=jnp.ones((COMET_SPAWN_COUNT,), dtype=jnp.bool_),
        fleet_owner=jnp.full((F_MAX,), -1, dtype=jnp.int32),
        fleet_x=jnp.zeros((F_MAX,), dtype=jnp.float32),
        fleet_y=jnp.zeros((F_MAX,), dtype=jnp.float32),
        fleet_angle=jnp.zeros((F_MAX,), dtype=jnp.float32),
        fleet_source=jnp.full((F_MAX,), -1, dtype=jnp.int32),
        fleet_ships=jnp.zeros((F_MAX,), dtype=jnp.float32),
        fleet_alive=jnp.zeros((F_MAX,), dtype=jnp.bool_),
        next_fleet_id=jnp.array(0, dtype=jnp.int32),
        step=jnp.array(0, dtype=jnp.int32),
        angular_velocity=jnp.array(0.0, dtype=jnp.float32),
        num_players=jnp.array(num_players, dtype=jnp.int32),
        episode_steps=jnp.array(episode_steps, dtype=jnp.int32),
        ship_speed=jnp.array(ship_speed, dtype=jnp.float32),
    )


def manual_state(
    *,
    planet_rows: list[list[float]],
    num_players: int = 2,
    angular_velocity: float = 0.0,
    episode_steps: int = 500,
    ship_speed: float = 6.0,
) -> EnvState:
    state = empty_state(num_players=num_players, episode_steps=episode_steps, ship_speed=ship_speed)
    rows = jnp.asarray(planet_rows, dtype=jnp.float32)
    n = rows.shape[0]
    idx = jnp.arange(n)
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
            "angular_velocity": jnp.array(angular_velocity, dtype=jnp.float32),
        }
    )


def state_from_observation(
    obs: dict[str, Any],
    *,
    num_players: int = 2,
    episode_steps: int = 500,
    ship_speed: float = 6.0,
) -> EnvState:
    state = empty_state(num_players=num_players, episode_steps=episode_steps, ship_speed=ship_speed)
    planets = jnp.asarray(obs.get("planets", []), dtype=jnp.float32).reshape((-1, 7))
    initial_planets = jnp.asarray(obs.get("initial_planets", obs.get("planets", [])), dtype=jnp.float32).reshape((-1, 7))
    fleets = jnp.asarray(obs.get("fleets", []), dtype=jnp.float32).reshape((-1, 7))
    n_planets = min(int(planets.shape[0]), P_MAX)
    n_initial = min(int(initial_planets.shape[0]), P_MAX)
    n_fleets = min(int(fleets.shape[0]), F_MAX)
    pidx = jnp.arange(n_planets)
    iidx = jnp.arange(n_initial)
    fidx = jnp.arange(n_fleets)
    comet_ids = {int(x) for x in obs.get("comet_planet_ids", [])}
    comet_mask = jnp.asarray([int(planets[i, 0]) in comet_ids for i in range(n_planets)], dtype=jnp.bool_) if n_planets else jnp.zeros((0,), dtype=jnp.bool_)
    comet_path_x = state.comet_path_x
    comet_path_y = state.comet_path_y
    comet_path_len = state.comet_path_len
    comet_path_index = state.comet_path_index
    slot_by_pid = {int(planets[i, 0]): i for i in range(n_planets)}
    for group in obs.get("comets", []) or []:
        group_index = int(group.get("path_index", -1))
        for pid, path in zip(group.get("planet_ids", []), group.get("paths", [])):
            slot = slot_by_pid.get(int(pid))
            if slot is None:
                continue
            clipped = list(path)[:COMET_PATH_MAX]
            if not clipped:
                continue
            path_arr = jnp.asarray(clipped, dtype=jnp.float32).reshape((-1, 2))
            path_idx = jnp.arange(path_arr.shape[0])
            comet_path_x = comet_path_x.at[slot, path_idx].set(path_arr[:, 0])
            comet_path_y = comet_path_y.at[slot, path_idx].set(path_arr[:, 1])
            comet_path_len = comet_path_len.at[slot].set(path_arr.shape[0])
            comet_path_index = comet_path_index.at[slot].set(group_index)
    state = EnvState(
        **{
            **state.__dict__,
            "planet_id": state.planet_id.at[pidx].set(planets[:n_planets, 0].astype(jnp.int32)),
            "planet_owner": state.planet_owner.at[pidx].set(planets[:n_planets, 1].astype(jnp.int32)),
            "planet_x": state.planet_x.at[pidx].set(planets[:n_planets, 2]),
            "planet_y": state.planet_y.at[pidx].set(planets[:n_planets, 3]),
            "planet_initial_x": state.planet_initial_x.at[iidx].set(initial_planets[:n_initial, 2]),
            "planet_initial_y": state.planet_initial_y.at[iidx].set(initial_planets[:n_initial, 3]),
            "planet_radius": state.planet_radius.at[pidx].set(planets[:n_planets, 4]),
            "planet_ships": state.planet_ships.at[pidx].set(planets[:n_planets, 5]),
            "planet_production": state.planet_production.at[pidx].set(planets[:n_planets, 6]),
            "planet_alive": state.planet_alive.at[pidx].set(True),
            "planet_is_comet": state.planet_is_comet.at[pidx].set(comet_mask),
            "comet_path_x": comet_path_x,
            "comet_path_y": comet_path_y,
            "comet_path_len": comet_path_len,
            "comet_path_index": comet_path_index,
            "step": jnp.array(int(obs.get("step", 0) or 0), dtype=jnp.int32),
            "angular_velocity": jnp.array(float(obs.get("angular_velocity", 0.0) or 0.0), dtype=jnp.float32),
            "next_fleet_id": jnp.array(int(obs.get("next_fleet_id", n_fleets) or n_fleets), dtype=jnp.int32),
        }
    )
    return EnvState(
        **{
            **state.__dict__,
            "fleet_owner": state.fleet_owner.at[fidx].set(fleets[:n_fleets, 1].astype(jnp.int32)),
            "fleet_x": state.fleet_x.at[fidx].set(fleets[:n_fleets, 2]),
            "fleet_y": state.fleet_y.at[fidx].set(fleets[:n_fleets, 3]),
            "fleet_angle": state.fleet_angle.at[fidx].set(fleets[:n_fleets, 4]),
            "fleet_source": state.fleet_source.at[fidx].set(fleets[:n_fleets, 5].astype(jnp.int32)),
            "fleet_ships": state.fleet_ships.at[fidx].set(fleets[:n_fleets, 6]),
            "fleet_alive": state.fleet_alive.at[fidx].set(True),
        }
    )
