from __future__ import annotations

import jax.numpy as jnp

from .collision import crosses_sun, out_of_bounds, swept_pair_hit
from .combat import resolve_combat
from .config import COMET_SPAWN_STEPS, F_MAX, LAUNCH_CLEARANCE, MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS, P_MAX
from .movement import fleet_speed, rotated_planet_positions
from .observation import build_observation
from .rewards import scores_and_ranks, terminal_rewards
from .state import EnvState


def _remove_expired_comets(state: EnvState) -> EnvState:
    expired = state.planet_is_comet & state.planet_alive & (state.comet_path_index >= state.comet_path_len) & (state.comet_path_len > 0)
    return EnvState(
        **{
            **state.__dict__,
            "planet_alive": state.planet_alive & (~expired),
            "planet_owner": jnp.where(expired, -1, state.planet_owner),
            "planet_ships": jnp.where(expired, 0.0, state.planet_ships),
        }
    )


def _spawn_comets(state: EnvState) -> EnvState:
    spawn_steps = jnp.asarray(COMET_SPAWN_STEPS, dtype=jnp.int32)
    spawn_match = spawn_steps == (state.step + 1)
    spawn_idx = jnp.argmax(spawn_match).astype(jnp.int32)
    should_spawn = jnp.any(spawn_match) & (~state.comet_spawned[spawn_idx])
    free_planets = ~state.planet_alive
    free_rank = jnp.cumsum(free_planets.astype(jnp.int32)) - 1
    spawn_slot_mask = free_planets & (free_rank >= 0) & (free_rank < 4)
    spawn_order_by_slot = jnp.clip(free_rank, 0, 3)
    slot_axis = jnp.arange(P_MAX, dtype=jnp.int32)
    spawn_len = state.comet_spawn_len[spawn_idx]
    spawn_ships = state.comet_spawn_ships[spawn_idx]
    path_x = state.comet_spawn_x[spawn_idx, spawn_order_by_slot]
    path_y = state.comet_spawn_y[spawn_idx, spawn_order_by_slot]
    active_spawn = should_spawn & spawn_slot_mask
    new_planet_id = jnp.max(jnp.where(state.planet_alive, state.planet_id, -1)) + 1 + spawn_order_by_slot
    return EnvState(
        **{
            **state.__dict__,
            "planet_id": jnp.where(active_spawn, new_planet_id, state.planet_id),
            "planet_owner": jnp.where(active_spawn, -1, state.planet_owner),
            "planet_x": jnp.where(active_spawn, -99.0, state.planet_x),
            "planet_y": jnp.where(active_spawn, -99.0, state.planet_y),
            "planet_initial_x": jnp.where(active_spawn, -99.0, state.planet_initial_x),
            "planet_initial_y": jnp.where(active_spawn, -99.0, state.planet_initial_y),
            "planet_radius": jnp.where(active_spawn, 1.0, state.planet_radius),
            "planet_ships": jnp.where(active_spawn, spawn_ships, state.planet_ships),
            "planet_production": jnp.where(active_spawn, 1.0, state.planet_production),
            "planet_alive": state.planet_alive | active_spawn,
            "planet_is_comet": state.planet_is_comet | active_spawn,
            "comet_path_x": jnp.where(active_spawn[:, None], path_x, state.comet_path_x),
            "comet_path_y": jnp.where(active_spawn[:, None], path_y, state.comet_path_y),
            "comet_path_len": jnp.where(active_spawn, spawn_len, state.comet_path_len),
            "comet_path_index": jnp.where(active_spawn, -1, state.comet_path_index),
            "comet_spawned": state.comet_spawned.at[spawn_idx].set(state.comet_spawned[spawn_idx] | should_spawn),
        }
    )


def _apply_comet_paths(state: EnvState, next_px: jnp.ndarray, next_py: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    next_index = state.comet_path_index + 1
    comet_active = state.planet_is_comet & state.planet_alive & (state.comet_path_len > 0)
    comet_has_next = comet_active & (next_index < state.comet_path_len)
    safe_index = jnp.clip(next_index, 0, jnp.maximum(state.comet_path_len - 1, 0))
    slot = jnp.arange(P_MAX)
    comet_x = state.comet_path_x[slot, safe_index]
    comet_y = state.comet_path_y[slot, safe_index]
    out_x = jnp.where(comet_has_next, comet_x, next_px)
    out_y = jnp.where(comet_has_next, comet_y, next_py)
    out_index = jnp.where(comet_active, next_index, state.comet_path_index)
    expired_after_tick = comet_active & (~comet_has_next)
    return out_x, out_y, out_index, expired_after_tick


def _launch_all(state: EnvState, actions: jnp.ndarray) -> tuple[EnvState, dict[str, jnp.ndarray]]:
    flat = actions.reshape((MAX_PLAYERS * MAX_ACTIONS_PER_PLAYER, 3))
    player_ids = jnp.repeat(jnp.arange(MAX_PLAYERS, dtype=jnp.int32), MAX_ACTIONS_PER_PLAYER)
    source_ids = flat[:, 0].astype(jnp.int32)
    angles = flat[:, 1]
    ships_raw = flat[:, 2]
    ships = jnp.floor(ships_raw)
    submitted = (source_ids != 0) | (angles != 0.0) | (ships_raw != 0.0)
    source_match = source_ids[:, None] == state.planet_id[None, :]
    source_match = source_match & state.planet_alive[None, :]
    source_slots = jnp.argmax(source_match, axis=1).astype(jnp.int32)
    valid_source = jnp.any(source_match, axis=1)
    active_player = player_ids < state.num_players
    source_owner = state.planet_owner[source_slots]
    owned = source_owner == player_ids
    positive = ships > 0.0
    total_by_source = jnp.zeros((P_MAX,), dtype=jnp.float32).at[source_slots].add(jnp.where(valid_source & owned & positive, ships, 0.0))
    affordable = total_by_source[source_slots] <= state.planet_ships[source_slots]
    invalid_source_id = submitted & (~valid_source)
    invalid_inactive_player_id = submitted & valid_source & (~active_player)
    invalid_source_not_owned = submitted & valid_source & active_player & (~owned)
    invalid_non_positive_ship_amount = submitted & valid_source & active_player & owned & (~positive)
    invalid_unaffordable_source_total = submitted & valid_source & active_player & owned & positive & (~affordable)
    valid_before_capacity = submitted & valid_source & active_player & owned & positive & affordable
    free_count = jnp.sum((~state.fleet_alive).astype(jnp.int32))
    capacity_ok = jnp.cumsum(valid_before_capacity.astype(jnp.int32)) <= free_count
    valid = valid_before_capacity & capacity_ok
    invalid_no_free_fleet_slot = valid_before_capacity & (~capacity_ok)
    submitted_count = jnp.sum(submitted.astype(jnp.int32))
    valid_count = jnp.sum(valid.astype(jnp.int32))
    invalid_count = submitted_count - valid_count
    launch_info = {
        "submitted_action_count": submitted_count,
        "valid_action_count": valid_count,
        "invalid_action_count": invalid_count,
        "invalid_action_rate": jnp.where(submitted_count > 0, invalid_count.astype(jnp.float32) / submitted_count.astype(jnp.float32), 0.0),
        "invalid_source_id_count": jnp.sum(invalid_source_id.astype(jnp.int32)),
        "invalid_inactive_player_id_count": jnp.sum(invalid_inactive_player_id.astype(jnp.int32)),
        "invalid_source_not_owned_count": jnp.sum(invalid_source_not_owned.astype(jnp.int32)),
        "invalid_non_positive_ship_amount_count": jnp.sum(invalid_non_positive_ship_amount.astype(jnp.int32)),
        "invalid_unaffordable_source_total_count": jnp.sum(invalid_unaffordable_source_total.astype(jnp.int32)),
        "invalid_no_free_fleet_slot_count": jnp.sum(invalid_no_free_fleet_slot.astype(jnp.int32)),
    }
    launch_order = jnp.cumsum(valid.astype(jnp.int32)) - 1
    launch_ships_by_planet = jnp.zeros((P_MAX,), dtype=jnp.float32).at[source_slots].add(jnp.where(valid, ships, 0.0))
    planet_ships = state.planet_ships - launch_ships_by_planet

    free_rank = jnp.cumsum((~state.fleet_alive).astype(jnp.int32)) - 1
    new_for_slot = (~state.fleet_alive) & (free_rank < valid_count)
    slot_order = free_rank
    action_for_slot = jnp.argmax((launch_order[:, None] == slot_order[None, :]) & valid[:, None], axis=0)
    slot_source = source_slots[action_for_slot]
    slot_angle = angles[action_for_slot]
    slot_ships = ships[action_for_slot]
    start_x = state.planet_x[slot_source] + jnp.cos(slot_angle) * (state.planet_radius[slot_source] + LAUNCH_CLEARANCE)
    start_y = state.planet_y[slot_source] + jnp.sin(slot_angle) * (state.planet_radius[slot_source] + LAUNCH_CLEARANCE)
    next_state = EnvState(
        **{
            **state.__dict__,
            "planet_ships": planet_ships,
            "fleet_owner": jnp.where(new_for_slot, player_ids[action_for_slot], state.fleet_owner),
            "fleet_x": jnp.where(new_for_slot, start_x, state.fleet_x),
            "fleet_y": jnp.where(new_for_slot, start_y, state.fleet_y),
            "fleet_angle": jnp.where(new_for_slot, slot_angle, state.fleet_angle),
            "fleet_source": jnp.where(new_for_slot, state.planet_id[slot_source], state.fleet_source),
            "fleet_ships": jnp.where(new_for_slot, slot_ships, state.fleet_ships),
            "fleet_alive": state.fleet_alive | new_for_slot,
            "next_fleet_id": state.next_fleet_id + valid_count,
        }
    )
    return next_state, launch_info


def _move_fleets(state: EnvState, next_px: jnp.ndarray, next_py: jnp.ndarray):
    old_x = state.fleet_x
    old_y = state.fleet_y
    speed = fleet_speed(state.fleet_ships, state.ship_speed)
    new_x = old_x + jnp.cos(state.fleet_angle) * speed
    new_y = old_y + jnp.sin(state.fleet_angle) * speed
    hit_matrix = jnp.zeros((F_MAX, P_MAX), dtype=jnp.bool_)
    for pid in range(P_MAX):
        hit = swept_pair_hit(old_x, old_y, new_x, new_y, state.planet_x[pid], state.planet_y[pid], next_px[pid], next_py[pid], state.planet_radius[pid])
        hit_matrix = hit_matrix.at[:, pid].set(state.fleet_alive & state.planet_alive[pid] & hit)
    hit_any = jnp.any(hit_matrix, axis=1)
    hit_planet = jnp.argmax(hit_matrix, axis=1).astype(jnp.int32)
    bounds = state.fleet_alive & out_of_bounds(new_x, new_y)
    sun = state.fleet_alive & (~hit_any) & (~bounds) & crosses_sun(old_x, old_y, new_x, new_y)
    remove = hit_any | bounds | sun
    return new_x, new_y, hit_any, hit_planet, remove


def _terminal(state: EnvState):
    scores, ranks = scores_and_ranks(
        state.planet_owner,
        state.planet_ships,
        state.planet_alive,
        state.fleet_owner,
        state.fleet_ships,
        state.fleet_alive,
    )
    players = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
    planet_live = jnp.array([jnp.any(state.planet_alive & (state.planet_owner == p)) for p in range(MAX_PLAYERS)])
    fleet_live = jnp.array([jnp.any(state.fleet_alive & (state.fleet_owner == p)) for p in range(MAX_PLAYERS)])
    alive_players = (planet_live | fleet_live) & (players < state.num_players)
    done = (state.step >= state.episode_steps - 2) | (jnp.sum(alive_players) <= 1)
    rewards = jnp.where(done, terminal_rewards(scores, ranks, state.num_players), jnp.zeros((MAX_PLAYERS,), dtype=jnp.float32))
    return rewards, done, {"scores": scores, "ranks": ranks}


def step(state: EnvState, actions: jnp.ndarray):
    state = _remove_expired_comets(state)
    state = _spawn_comets(state)
    state, launch_info = _launch_all(state, actions)
    planet_ships = state.planet_ships + jnp.where(state.planet_alive & (state.planet_owner != -1), state.planet_production, 0.0)
    state = EnvState(**{**state.__dict__, "planet_ships": planet_ships})
    next_px, next_py = rotated_planet_positions(
        state.planet_initial_x,
        state.planet_initial_y,
        state.planet_x,
        state.planet_y,
        state.planet_radius,
        state.planet_alive,
        state.planet_is_comet,
        state.angular_velocity,
        state.step,
    )
    next_px, next_py, comet_path_index, expired_comets = _apply_comet_paths(state, next_px, next_py)
    next_step = state.step + 1
    fleet_x, fleet_y, hit_any, hit_planet, remove = _move_fleets(state, next_px, next_py)
    hit_owner = jnp.where(hit_any, state.fleet_owner, -1)
    hit_ships = jnp.where(hit_any, state.fleet_ships, 0.0)
    fleet_alive = state.fleet_alive & (~remove)
    planet_owner, planet_ships = resolve_combat(state.planet_owner, state.planet_ships, state.planet_alive, hit_planet, hit_owner, hit_ships)
    next_state = EnvState(
        **{
            **state.__dict__,
            "planet_owner": planet_owner,
            "planet_ships": planet_ships,
            "planet_x": next_px,
            "planet_y": next_py,
            "planet_alive": state.planet_alive & (~expired_comets),
            "comet_path_index": comet_path_index,
            "fleet_x": fleet_x,
            "fleet_y": fleet_y,
            "fleet_alive": fleet_alive,
            "step": next_step,
        }
    )
    rewards, done, info = _terminal(next_state)
    info = {**info, **launch_info}
    return next_state, build_observation(next_state), rewards, done, info
