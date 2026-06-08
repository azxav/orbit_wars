from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def test_state_from_official_comet_observation_steps_comet_paths() -> None:
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
    from orbit_jax_env.parity.compare_official import _official_rollout
    from orbit_jax_env.parity.scripted_actions import no_actions
    from orbit_jax_env.state import state_from_observation
    from orbit_jax_env.step import step

    official = _official_rollout(seed=123, players=2, steps=51, scripted=no_actions)
    obs_50 = official[50]
    obs_51 = official[51]
    assert obs_50["comet_planet_ids"]
    assert obs_50["comets"][0]["path_index"] == 0
    assert obs_51["comets"][0]["path_index"] == 1

    state = state_from_observation(obs_50, num_players=2, episode_steps=60)
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    next_state, *_ = step(state, actions)

    expected_by_id = {int(p[0]): p for p in obs_51["planets"] if int(p[0]) in set(obs_51["comet_planet_ids"])}
    for pid, expected in expected_by_id.items():
        slot = int(np.flatnonzero(np.asarray(next_state.planet_id) == pid)[0])
        assert bool(next_state.planet_alive[slot]) is True
        assert bool(next_state.planet_is_comet[slot]) is True
        assert np.allclose(
            [float(next_state.planet_x[slot]), float(next_state.planet_y[slot])],
            [float(expected[2]), float(expected[3])],
            atol=1.0e-4,
        )


def test_imported_comet_movement_parity_case_passes() -> None:
    from orbit_jax_env.parity.compare_official import run_parity_case

    result = run_parity_case("case_010_imported_comet_movement")

    assert result["implemented"] is True
    assert result["passed"] is True
    assert result["owner_mismatches"] == 0
    assert result["ship_mismatches"] == 0


def test_reset_enable_comets_spawns_jax_native_comets_under_jit_scan() -> None:
    import jax

    from orbit_jax_env import EnvConfig, reset, step
    from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS

    cfg = EnvConfig(num_players=2, enable_comets=True)
    state = reset(jax.random.PRNGKey(0), cfg)
    actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)

    def body(carry, _):
        next_state, *_ = step(carry, actions)
        return next_state, None

    state_49, _ = jax.jit(lambda s: jax.lax.scan(body, s, None, length=49))(state)
    assert int(jnp.sum(state_49.planet_is_comet & state_49.planet_alive)) == 0

    state_50, _ = jax.jit(lambda s: jax.lax.scan(body, s, None, length=50))(state)
    comet_mask = np.asarray(state_50.planet_is_comet & state_50.planet_alive)
    assert int(comet_mask.sum()) == 4
    assert set(np.asarray(state_50.planet_id)[comet_mask].astype(int)) == {16, 17, 18, 19}
    assert np.all(np.asarray(state_50.planet_radius)[comet_mask] == 1.0)
    assert np.all(np.asarray(state_50.comet_path_index)[comet_mask] == 0)
