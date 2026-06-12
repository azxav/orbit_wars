from __future__ import annotations

from pathlib import Path
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _batched_states():
    from orbit_jax_env.state import manual_state

    state0 = manual_state(
        planet_rows=[[10, 0, 20.0, 50.0, 2.0, 10.0, 3.0]],
        num_players=2,
        episode_steps=500,
    )
    state1 = manual_state(
        planet_rows=[[20, 0, 30.0, 50.0, 2.0, 12.0, 3.0]],
        num_players=2,
        episode_steps=500,
    )
    return jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), state0, state1)


def test_state_bank_save_load_roundtrip_preserves_arrays_and_metadata(tmp_path: Path) -> None:
    from orbit_jax_env.official_state_dataset import load_state_bank, save_state_bank

    states = _batched_states()
    path = tmp_path / "bank.npz"
    metadata = {
        "players": 2,
        "episode_steps": 500,
        "ship_speed": 6.0,
        "source": "kaggle_official",
        "seed_start": 0,
        "seed_count": 2,
    }

    save_state_bank(path, states, metadata)
    loaded, loaded_metadata = load_state_bank(path)

    assert loaded_metadata["players"] == 2
    assert loaded_metadata["episode_steps"] == 500
    assert loaded_metadata["source"] == "kaggle_official"
    assert loaded.planet_id.shape[0] == 2
    np.testing.assert_array_equal(np.asarray(loaded.planet_id[:, 0]), np.asarray([10, 20], dtype=np.int32))


def test_state_bank_sampling_is_jit_safe_random_and_cycle() -> None:
    from orbit_jax_env.official_state_dataset import sample_state_bank

    states = _batched_states()

    random_state, next_counter = jax.jit(lambda key: sample_state_bank(states, key, mode="random", cycle_index=jnp.array(0)))(jax.random.PRNGKey(0))
    assert random_state.planet_id.shape == states.planet_id.shape[1:]
    assert int(next_counter) == 0

    cycle_fn = jax.jit(lambda idx: sample_state_bank(states, jax.random.PRNGKey(0), mode="cycle", cycle_index=idx))
    cycle0, counter1 = cycle_fn(jnp.array(0, dtype=jnp.int32))
    cycle1, counter2 = cycle_fn(counter1)

    assert int(cycle0.planet_id[0]) == 10
    assert int(cycle1.planet_id[0]) == 20
    assert int(counter2) == 2


def test_generate_official_initial_states_uses_kaggle_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    from orbit_jax_env.official_state_dataset import generate_official_initial_states

    class FakeEnv:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.steps = [
                [
                    types.SimpleNamespace(
                        observation={
                            "step": 0,
                            "planets": [[100 + seed, 0, 20.0, 50.0, 2.0, 10.0, 3.0]],
                            "initial_planets": [[100 + seed, 0, 20.0, 50.0, 2.0, 10.0, 3.0]],
                            "fleets": [],
                        }
                    )
                ]
            ]

        def run(self, agents):
            return self.steps

    def fake_make(_name, configuration, debug=False):
        return FakeEnv(int(configuration["seed"]))

    monkeypatch.setitem(sys.modules, "kaggle_environments", types.SimpleNamespace(make=fake_make))

    states, metadata = generate_official_initial_states(players=2, seeds=range(2), episode_steps=500)

    assert metadata["players"] == 2
    assert metadata["seed_start"] == 0
    assert metadata["seed_count"] == 2
    assert states.planet_id.shape[0] == 2
    np.testing.assert_array_equal(np.asarray(states.planet_id[:, 0]), np.asarray([100, 101], dtype=np.int32))
