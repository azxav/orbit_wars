from __future__ import annotations

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


def test_simple_heuristic_actions_shape_dtype_and_jit_valid_launches() -> None:
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.simple_heuristic_jax import simple_heuristic_actions
    from orbit_jax_env.state import manual_state
    from orbit_jax_env.step import step

    state = manual_state(
        planet_rows=[
            [10, 0, 20.0, 50.0, 2.0, 80.0, 3.0],
            [11, -1, 35.0, 70.0, 2.0, 5.0, 1.0],
            [12, 1, 80.0, 80.0, 2.0, 30.0, 2.0],
        ],
        num_players=2,
        angular_velocity=0.0,
    )

    actions = jax.jit(simple_heuristic_actions)(state)
    next_state, _obs, _rewards, _done, info = step(state, actions)

    assert actions.shape == (MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3)
    assert actions.dtype == jnp.float32
    assert int(info["invalid_action_count"]) == 0
    assert int(jnp.sum(next_state.fleet_alive)) > 0


def test_simple_heuristic_actions_vmap_on_reset_states() -> None:
    from orbit_jax_env import EnvConfig, reset
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.simple_heuristic_jax import simple_heuristic_actions

    states = jax.vmap(lambda k: reset(k, EnvConfig(num_players=4, enable_comets=False)))(
        jax.random.split(jax.random.PRNGKey(7), 3)
    )

    actions = jax.jit(jax.vmap(simple_heuristic_actions))(states)

    assert actions.shape == (3, MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3)
    assert np.all(np.isfinite(np.asarray(actions)))


def test_simple_heuristic_jax_matches_reference_opening_target_direction() -> None:
    import simple_heuristic

    from orbit_jax_env.simple_heuristic_jax import simple_heuristic_actions
    from orbit_jax_env.state import state_from_observation

    obs = {
        "player": 0,
        "step": 1,
        "angular_velocity": 0.0,
        "planets": [
            [10, 0, 20.0, 50.0, 2.0, 80.0, 3.0],
            [11, -1, 35.0, 70.0, 2.0, 5.0, 1.0],
            [12, 1, 80.0, 80.0, 2.0, 30.0, 2.0],
        ],
        "initial_planets": [
            [10, 0, 20.0, 50.0, 2.0, 40.0, 3.0],
            [11, -1, 70.0, 50.0, 2.0, 5.0, 1.0],
            [12, 1, 80.0, 80.0, 2.0, 30.0, 2.0],
        ],
        "fleets": [],
        "comet_planet_ids": [],
    }
    reference = simple_heuristic.agent(obs)
    state = state_from_observation(obs, num_players=2, episode_steps=500)
    rows = np.asarray(simple_heuristic_actions(state)[0])
    emitted = rows[rows[:, 2] > 0.0]

    assert reference
    assert emitted.shape[0] > 0
    assert int(emitted[0, 0]) == int(reference[0][0])
    assert math.isclose(float(emitted[0, 1]), float(reference[0][1]), abs_tol=0.20)


def test_default_jax_ppo_opponent_is_simple_heuristic() -> None:
    from orbit_ppo_jax.train import build_arg_parser, config_from_args

    args = build_arg_parser().parse_args(["--bc_checkpoint", "bc.pt", "--out_dir", "out"])
    config = config_from_args(args)

    assert config.opponent == "simple_heuristic_jax"


def test_explicit_jax_proxy_opponent_still_parses() -> None:
    from orbit_ppo_jax.train import build_arg_parser, config_from_args

    args = build_arg_parser().parse_args(
        ["--bc_checkpoint", "bc.pt", "--out_dir", "out", "--opponent", "jax_proxy"]
    )
    config = config_from_args(args)

    assert config.opponent == "jax_proxy"
