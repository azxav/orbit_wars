from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def test_reset_returns_fixed_shape_observation_and_masks() -> None:
    from orbit_jax_env import EnvConfig, reset, step
    from orbit_jax_env.config import F_MAX, MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS, P_MAX

    state = reset(jax.random.PRNGKey(0), EnvConfig(num_players=2, enable_comets=False))
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    next_state, obs, rewards, done, info = step(state, actions)

    assert state.planet_owner.shape == (P_MAX,)
    assert state.fleet_owner.shape == (F_MAX,)
    assert obs["planets"].shape[0] == P_MAX
    assert obs["fleets"].shape[0] == F_MAX
    assert obs["valid_source_mask"].shape == (MAX_PLAYERS, P_MAX)
    assert obs["valid_target_mask"].shape == (MAX_PLAYERS, P_MAX)
    assert rewards.shape == (MAX_PLAYERS,)
    assert info["scores"].shape == (MAX_PLAYERS,)
    assert bool(done) is False
    assert int(next_state.step) == 1


def test_launch_production_and_movement_from_manual_state() -> None:
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.state import manual_state
    from orbit_jax_env.step import step

    state = manual_state(
        planet_rows=[
            [10, 0, 20.0, 50.0, 2.0, 10.0, 3.0],
            [11, -1, 80.0, 50.0, 2.0, 5.0, 1.0],
            [12, 1, 80.0, 80.0, 2.0, 10.0, 0.0],
        ],
        num_players=2,
        angular_velocity=0.0,
    )
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    actions = actions.at[0, 0].set(jnp.array([10.0, 0.0, 4.0], dtype=jnp.float32))

    next_state, _obs, rewards, done, _info = step(state, actions)

    alive_fleets = np.asarray(next_state.fleet_alive)
    fleet_idx = int(np.flatnonzero(alive_fleets)[0])
    assert int(next_state.planet_ships[0]) == 9  # launch 4, then produce 3
    assert int(next_state.fleet_owner[fleet_idx]) == 0
    assert int(next_state.fleet_ships[fleet_idx]) == 4
    assert float(next_state.fleet_x[fleet_idx]) > 23.0
    assert np.allclose(np.asarray(rewards), 0.0)
    assert bool(done) is False


def test_launch_info_ignores_padding_noop_rows() -> None:
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.state import manual_state
    from orbit_jax_env.step import step

    state = manual_state(
        planet_rows=[[10, 0, 20.0, 50.0, 2.0, 10.0, 3.0]],
        num_players=2,
        angular_velocity=0.0,
    )
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)

    _next_state, _obs, _rewards, _done, info = step(state, actions)

    assert int(info["submitted_action_count"]) == 0
    assert int(info["valid_action_count"]) == 0
    assert int(info["invalid_action_count"]) == 0
    assert float(info["invalid_action_rate"]) == 0.0


def test_launch_info_counts_valid_launch() -> None:
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.state import manual_state
    from orbit_jax_env.step import step

    state = manual_state(
        planet_rows=[[10, 0, 20.0, 50.0, 2.0, 10.0, 3.0]],
        num_players=2,
        angular_velocity=0.0,
    )
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    actions = actions.at[0, 0].set(jnp.array([10.0, 0.0, 4.0], dtype=jnp.float32))

    _next_state, _obs, _rewards, _done, info = step(state, actions)

    assert int(info["submitted_action_count"]) == 1
    assert int(info["valid_action_count"]) == 1
    assert int(info["invalid_action_count"]) == 0


def test_launch_info_counts_staged_exclusive_invalid_reasons() -> None:
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.state import manual_state
    from orbit_jax_env.step import step

    state = manual_state(
        planet_rows=[
            [10, 0, 20.0, 50.0, 2.0, 10.0, 3.0],
            [12, 1, 80.0, 50.0, 2.0, 10.0, 0.0],
        ],
        num_players=2,
        angular_velocity=0.0,
    )
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    actions = actions.at[0, 0].set(jnp.array([999.0, 0.0, 1.0], dtype=jnp.float32))
    actions = actions.at[0, 1].set(jnp.array([12.0, 0.0, 1.0], dtype=jnp.float32))
    actions = actions.at[0, 2].set(jnp.array([10.0, 1.0, -1.0], dtype=jnp.float32))
    actions = actions.at[2, 0].set(jnp.array([10.0, 0.0, 1.0], dtype=jnp.float32))

    _next_state, _obs, _rewards, _done, info = step(state, actions)

    assert int(info["submitted_action_count"]) == 4
    assert int(info["valid_action_count"]) == 0
    assert int(info["invalid_action_count"]) == 4
    assert int(info["invalid_source_id_count"]) == 1
    assert int(info["invalid_inactive_player_id_count"]) == 1
    assert int(info["invalid_source_not_owned_count"]) == 1
    assert int(info["invalid_non_positive_ship_amount_count"]) == 1
    assert int(info["invalid_unaffordable_source_total_count"]) == 0
    assert int(info["invalid_no_free_fleet_slot_count"]) == 0


def test_launch_info_counts_unaffordable_multi_launch() -> None:
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.state import manual_state
    from orbit_jax_env.step import step

    state = manual_state(
        planet_rows=[[10, 0, 20.0, 50.0, 2.0, 10.0, 3.0]],
        num_players=2,
        angular_velocity=0.0,
    )
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    actions = actions.at[0, 0].set(jnp.array([10.0, 0.0, 8.0], dtype=jnp.float32))
    actions = actions.at[0, 1].set(jnp.array([10.0, 1.0, 8.0], dtype=jnp.float32))

    _next_state, _obs, _rewards, _done, info = step(state, actions)

    assert int(info["submitted_action_count"]) == 2
    assert int(info["valid_action_count"]) == 0
    assert int(info["invalid_action_count"]) == 2
    assert int(info["invalid_unaffordable_source_total_count"]) == 2


def test_launch_info_counts_no_free_fleet_slot() -> None:
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.state import EnvState, manual_state
    from orbit_jax_env.step import step

    state = manual_state(
        planet_rows=[[10, 0, 20.0, 50.0, 2.0, 10.0, 3.0]],
        num_players=2,
        angular_velocity=0.0,
    )
    state = EnvState(**{**state.__dict__, "fleet_alive": jnp.ones_like(state.fleet_alive)})
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    actions = actions.at[0, 0].set(jnp.array([10.0, 0.0, 1.0], dtype=jnp.float32))

    _next_state, _obs, _rewards, _done, info = step(state, actions)

    assert int(info["submitted_action_count"]) == 1
    assert int(info["valid_action_count"]) == 0
    assert int(info["invalid_action_count"]) == 1
    assert int(info["invalid_no_free_fleet_slot_count"]) == 1


def test_planet_collision_captures_neutral_planet() -> None:
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.state import manual_state
    from orbit_jax_env.step import step

    state = manual_state(
        planet_rows=[
            [1, 0, 45.0, 50.0, 2.0, 20.0, 0.0],
            [2, -1, 49.5, 50.0, 2.0, 3.0, 0.0],
        ],
        num_players=2,
        angular_velocity=0.0,
    )
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    actions = actions.at[0, 0].set(jnp.array([1.0, 0.0, 5.0], dtype=jnp.float32))

    next_state, _obs, _rewards, _done, _info = step(state, actions)

    assert bool(next_state.fleet_alive.any()) is False
    assert int(next_state.planet_owner[1]) == 0
    assert int(next_state.planet_ships[1]) == 2


def test_sun_and_bounds_remove_fleets() -> None:
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.state import manual_state
    from orbit_jax_env.step import step

    sun_state = manual_state(
        planet_rows=[[1, 0, 38.9, 50.0, 1.0, 10.0, 0.0]],
        num_players=2,
        angular_velocity=0.0,
    )
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    actions = actions.at[0, 0].set(jnp.array([1.0, 0.0, 10.0], dtype=jnp.float32))
    after_sun, *_ = step(sun_state, actions)
    assert bool(after_sun.fleet_alive.any()) is False

    bounds_state = manual_state(
        planet_rows=[[1, 0, 97.0, 80.0, 1.0, 10.0, 0.0]],
        num_players=2,
        angular_velocity=0.0,
    )
    after_bounds, *_ = step(bounds_state, actions)
    assert bool(after_bounds.fleet_alive.any()) is False


def test_jit_vmap_and_scan_step_work() -> None:
    from orbit_jax_env import EnvConfig, reset, step
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS

    cfg = EnvConfig(num_players=4, enable_comets=False)
    keys = jax.random.split(jax.random.PRNGKey(42), 4)
    states = jax.vmap(lambda k: reset(k, cfg))(keys)
    actions = jnp.zeros((4, MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    next_states, obs, rewards, dones, _info = jax.jit(jax.vmap(step))(states, actions)

    assert next_states.planet_owner.shape[0] == 4
    assert obs["planets"].shape[0] == 4
    assert rewards.shape == (4, MAX_PLAYERS)
    assert dones.shape == (4,)

    def body(carry, _):
        zero_actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
        new_carry, _obs, reward, done, _info = step(carry, zero_actions)
        return new_carry, (reward, done)

    scanned_state, (scan_rewards, scan_dones) = jax.lax.scan(body, reset(jax.random.PRNGKey(7), cfg), None, length=8)
    assert int(scanned_state.step) == 8
    assert scan_rewards.shape == (8, MAX_PLAYERS)
    assert scan_dones.shape == (8,)


def test_state_from_observation_imports_kaggle_rows() -> None:
    from orbit_jax_env.state import state_from_observation

    obs = {
        "step": 12,
        "angular_velocity": 0.03,
        "next_fleet_id": 9,
        "planets": [[101, 0, 20.0, 30.0, 2.0, 11.0, 3.0], [102, 1, 70.0, 30.0, 2.0, 9.0, 1.0]],
        "initial_planets": [[101, 0, 21.0, 31.0, 2.0, 11.0, 3.0], [102, 1, 71.0, 31.0, 2.0, 9.0, 1.0]],
        "fleets": [[8, 0, 25.0, 30.0, 0.1, 101, 4]],
    }
    state = state_from_observation(obs, num_players=2)

    assert int(state.step) == 12
    assert int(state.planet_id[0]) == 101
    assert float(state.planet_initial_x[0]) == 21.0
    assert int(state.fleet_owner[0]) == 0
    assert bool(state.fleet_alive[0]) is True


def test_jax_rollout_adapter_runs_scan() -> None:
    from orbit_jax_env import EnvConfig
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.rollout import jax_rollout

    def policy_apply(_params, obs):
        batch = obs["global"].shape[0]
        return {
            "actions": jnp.zeros((batch, MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32),
            "logprobs": jnp.zeros((batch,), dtype=jnp.float32),
            "values": jnp.zeros((batch,), dtype=jnp.float32),
        }

    final_states, traj = jax_rollout(policy_apply, None, jax.random.split(jax.random.PRNGKey(3), 2), EnvConfig(num_players=2), steps=3)

    assert int(final_states.step[0]) == 3
    assert traj["rewards"].shape == (3, 2, MAX_PLAYERS)
    assert traj["dones"].shape == (3, 2)
    assert traj["obs"]["global"].shape[0] == 3
    assert traj["next_obs"]["global"].shape[0] == 3
    assert np.allclose(np.asarray(traj["obs"]["global"][0, :, 0]), 0.0)
    assert np.allclose(np.asarray(traj["next_obs"]["global"][0, :, 0]), 1.0)


def test_jax_rollout_adapter_uses_policy_raw_actions() -> None:
    from orbit_jax_env import EnvConfig
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.rollout import jax_rollout

    def policy_apply(_params, obs):
        batch = obs["global"].shape[0]
        source_slot = jnp.argmax(obs["valid_source_mask"][:, 0, :], axis=1)
        source_id = obs["planets"][jnp.arange(batch), source_slot, 0]
        actions = jnp.zeros((batch, MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
        actions = actions.at[:, 0, 0, 0].set(source_id)
        actions = actions.at[:, 0, 0, 1].set(0.0)
        actions = actions.at[:, 0, 0, 2].set(1.0)
        return {
            "actions": actions,
            "logprobs": jnp.ones((batch,), dtype=jnp.float32) * -0.5,
            "values": jnp.ones((batch,), dtype=jnp.float32) * 0.25,
        }

    final_states, traj = jax.jit(lambda keys: jax_rollout(policy_apply, None, keys, EnvConfig(num_players=2), steps=1))(
        jax.random.split(jax.random.PRNGKey(13), 2)
    )

    assert int(jnp.sum(final_states.fleet_alive)) == 2
    assert np.allclose(np.asarray(traj["logprobs"][0]), -0.5)
    assert np.allclose(np.asarray(traj["values"][0]), 0.25)
