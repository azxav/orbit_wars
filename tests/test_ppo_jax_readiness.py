from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _tiny_bc_checkpoint(path: Path) -> Path:
    from orbit_bc_training.checkpoints import save_checkpoint
    from orbit_bc_training.config import BCModelConfig
    from orbit_bc_training.model import EntityBCPolicy

    torch.manual_seed(123)
    cfg = BCModelConfig(
        planet_feature_dim=16,
        global_feature_dim=10,
        target_state_feature_dim=9,
        pair_feature_dim=15,
        max_planets=64,
        target_classes=65,
        amount_bins=7,
        noop_target_slot=64,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        mlp_size=32,
        dropout=0.0,
    )
    model = EntityBCPolicy(cfg)
    save_checkpoint(path, model, None, 1, {"valid_total_loss": 1.0}, cfg)
    return path / "checkpoint.pt"


def _torch_batch(seed: int = 7) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    return {
        "planet_features": torch.as_tensor(rng.normal(size=(3, 64, 16)).astype(np.float32)),
        "global_features": torch.as_tensor(rng.normal(size=(3, 10)).astype(np.float32)),
        "target_state_features": torch.as_tensor(rng.normal(size=(3, 64, 9)).astype(np.float32)),
        "pair_features": torch.as_tensor(rng.normal(size=(3, 65, 15)).astype(np.float32)),
        "source_slot": torch.as_tensor([0, 7, 12], dtype=torch.long),
        "target_label": torch.as_tensor([3, 64, 21], dtype=torch.long),
    }


def test_imported_bc_policy_matches_torch_logits(tmp_path: Path) -> None:
    from orbit_bc_training.checkpoints import load_checkpoint
    from orbit_ppo_jax.bc_policy import bc_forward, load_bc_jax_params

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    torch_model, _ = load_checkpoint(ckpt, device="cpu")
    batch = _torch_batch()

    with torch.no_grad():
        expected = torch_model(batch)

    params, config = load_bc_jax_params(ckpt)
    actual = bc_forward(
        params,
        {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in batch.items()},
        config,
    )

    np.testing.assert_allclose(np.asarray(actual["target_logits"]), expected["target_logits"].numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(actual["amount_logits"]), expected["amount_logits"].numpy(), rtol=1e-4, atol=1e-4)


def test_jax_feature_contract_matches_dense_python_features() -> None:
    from orbit_jax_env.state import manual_state
    from orbit_ppo_jax.features import build_bc_features_for_seat
    from orbit_training_prep.features import build_feature_state, pair_features_from_dense

    state = manual_state(
        planet_rows=[
            [10, 0, 20.0, 50.0, 2.0, 20.0, 3.0],
            [11, -1, 70.0, 50.0, 2.0, 5.0, 1.0],
            [12, 1, 80.0, 75.0, 2.0, 12.0, 2.0],
        ],
        num_players=2,
        angular_velocity=0.0,
    )
    obs = {
        "planets": [
            [10, 0, 20.0, 50.0, 2.0, 20.0, 3.0],
            [11, -1, 70.0, 50.0, 2.0, 5.0, 1.0],
            [12, 1, 80.0, 75.0, 2.0, 12.0, 2.0],
        ],
        "initial_planets": [
            [10, 0, 20.0, 50.0, 2.0, 20.0, 3.0],
            [11, -1, 70.0, 50.0, 2.0, 5.0, 1.0],
            [12, 1, 80.0, 75.0, 2.0, 12.0, 2.0],
        ],
        "fleets": [],
        "step": 0,
        "episode_steps": 500,
        "num_players": 2,
        "comet_planet_ids": [],
    }

    actual = build_bc_features_for_seat(state, 0)
    expected_fs = build_feature_state(obs, 0)
    expected_pair = pair_features_from_dense(
        expected_fs.planet_features,
        expected_fs.target_state_features,
        0,
        target_viability_mask=np.asarray(actual.target_mask[0]),
        amount_viability_mask=np.asarray(actual.amount_mask[0]),
    )

    np.testing.assert_allclose(np.asarray(actual.planet_features), expected_fs.planet_features, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(actual.global_features), expected_fs.global_features, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(actual.target_state_features), expected_fs.target_state_features, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(actual.pair_features[0]), expected_pair, rtol=1e-5, atol=1e-5)


def test_compact_features_select_top_ship_active_owned_sources() -> None:
    from orbit_jax_env.state import manual_state
    from orbit_ppo_jax.features import build_bc_features_for_seat

    rows = []
    for i in range(40):
        rows.append([100 + i, 0, float(i % 8) * 10.0, float(i // 8) * 10.0, 2.0, float(i + 1), 1.0])
    rows.extend(
        [
            [300, 1, 90.0, 10.0, 2.0, 200.0, 1.0],
            [301, -1, 90.0, 20.0, 2.0, 201.0, 1.0],
            [302, 0, 90.0, 30.0, 2.0, 0.0, 1.0],
        ]
    )
    state = manual_state(planet_rows=rows, num_players=2)

    features = build_bc_features_for_seat(state, 0, source_cap=32)

    expected = np.arange(8, 40, dtype=np.int32)[::-1]
    np.testing.assert_array_equal(np.asarray(features.source_slots), expected)
    assert np.asarray(features.source_mask).tolist() == [True] * 32
    assert int(features.active_source_count) == 40
    assert int(features.selected_source_count) == 32
    assert features.pair_features.shape == (32, 65, 15)
    assert features.target_mask.shape == (32, 65)
    assert features.amount_mask.shape == (32, 65, 7)


def test_compact_features_pad_below_cap_and_filter_inactive_sources() -> None:
    from orbit_jax_env.state import manual_state
    from orbit_ppo_jax.features import build_bc_features_for_seat

    state = manual_state(
        planet_rows=[
            [10, 0, 20.0, 50.0, 2.0, 12.0, 3.0],
            [11, 0, 30.0, 50.0, 2.0, 5.0, 1.0],
            [12, 1, 40.0, 50.0, 2.0, 100.0, 1.0],
            [13, -1, 50.0, 50.0, 2.0, 100.0, 1.0],
            [14, 0, 60.0, 50.0, 2.0, 0.0, 1.0],
        ],
        num_players=2,
    )

    features = build_bc_features_for_seat(state, 0, source_cap=5)

    np.testing.assert_array_equal(np.asarray(features.source_slots[:2]), np.asarray([0, 1], dtype=np.int32))
    assert np.asarray(features.source_mask).tolist() == [True, True, False, False, False]
    assert int(features.active_source_count) == 2
    assert int(features.selected_source_count) == 2
    assert not np.asarray(features.target_mask[2:]).any()
    assert not np.asarray(features.amount_mask[2:]).any()


def test_amount_decode_contract_matches_schema_bins() -> None:
    from orbit_ppo_jax.actions import decode_amount_bin_jax
    from orbit_training_prep.schema import decode_amount_bin

    available = 20.0
    capture_needed = 6.0
    actual = np.asarray(decode_amount_bin_jax(jnp.arange(7), available, capture_needed))
    expected = np.asarray([decode_amount_bin(i, available, capture_needed) for i in range(7)])

    np.testing.assert_array_equal(actual, expected)


def test_jax_checkpoint_roundtrip(tmp_path: Path) -> None:
    from orbit_ppo_jax.bc_policy import init_value_head, load_bc_jax_params
    from orbit_ppo_jax.checkpointing import load_jax_checkpoint, save_jax_checkpoint

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    bc_params, config = load_bc_jax_params(ckpt)
    params = {"bc": bc_params, "value": init_value_head(jax.random.PRNGKey(0), int(config["hidden_size"]))}
    save_jax_checkpoint(tmp_path / "jax" / "latest", params, {"bc_checkpoint": str(ckpt), "players": 2}, {"loss": 1.25})

    loaded_params, loaded_config, loaded_metrics = load_jax_checkpoint(tmp_path / "jax" / "latest")

    assert loaded_config["bc_checkpoint"] == str(ckpt)
    assert loaded_metrics["loss"] == 1.25
    np.testing.assert_allclose(np.asarray(loaded_params["bc"]["planet_encoder"]["weight"]), np.asarray(params["bc"]["planet_encoder"]["weight"]))


def test_tiny_train_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    from orbit_ppo_jax.train import main

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    out_dir = tmp_path / "ppo"
    main(
        [
            "--bc_checkpoint",
            str(ckpt),
            "--out_dir",
            str(out_dir),
            "--players",
            "2",
            "--envs",
            "1",
            "--steps",
            "1",
            "--episode_steps",
            "500",
            "--updates",
            "1",
            "--eval_games",
            "0",
        ]
    )

    assert (out_dir / "config.json").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "latest" / "params.npz").exists()
    config = json.loads((out_dir / "config.json").read_text())
    assert "steps" not in config
    assert config["rollout_steps"] == 1
    assert config["episode_steps"] == 500
    metrics = json.loads((out_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert metrics["update"] == 1
    assert metrics["env_steps"] == 1
    assert metrics["update_env_steps"] == 1
    assert metrics["rollout_steps"] == 1
    assert metrics["episode_steps"] == 500
    assert metrics["reset_source"] == "jax_reset"
    assert metrics["steps_per_second"] > 0.0


def test_steps_alias_normalizes_to_rollout_steps() -> None:
    from orbit_ppo_jax.train import build_arg_parser, config_from_args

    args = build_arg_parser().parse_args(
        [
            "--bc_checkpoint",
            "bc.pt",
            "--out_dir",
            "out",
            "--steps",
            "7",
        ]
    )
    config = config_from_args(args)

    assert not hasattr(config, "steps")
    assert config.rollout_steps == 7
    assert config.episode_steps == 500
    assert config.source_cap == 32


def test_source_cap_arg_is_accepted() -> None:
    from orbit_ppo_jax.train import build_arg_parser, config_from_args

    args = build_arg_parser().parse_args(
        [
            "--bc_checkpoint",
            "bc.pt",
            "--out_dir",
            "out",
            "--source_cap",
            "2",
        ]
    )
    config = config_from_args(args)

    assert config.source_cap == 2


def test_action_rows_scatter_compact_sources_and_ignore_padding() -> None:
    from orbit_jax_env.state import manual_state
    from orbit_ppo_jax.actions import action_rows_from_source_choices

    state = manual_state(
        planet_rows=[
            [10, 0, 10.0, 10.0, 2.0, 20.0, 1.0],
            [11, 0, 20.0, 10.0, 2.0, 20.0, 1.0],
            [12, 1, 30.0, 10.0, 2.0, 5.0, 1.0],
        ],
        num_players=2,
    )

    rows = action_rows_from_source_choices(
        state,
        0,
        jnp.asarray([1, 0, 0], dtype=jnp.int32),
        jnp.asarray([2, 64, 2], dtype=jnp.int32),
        jnp.asarray([2, 0, 2], dtype=jnp.int32),
        jnp.asarray([True, True, False]),
    )

    rows_np = np.asarray(rows)
    assert rows_np[0].tolist() == [0.0, 0.0, 0.0]
    assert rows_np[1, 0] == 11.0
    assert rows_np[1, 2] > 0.0
    assert rows_np[2].tolist() == [0.0, 0.0, 0.0]
    assert not rows_np[3:].any()

    padded_duplicate_rows = action_rows_from_source_choices(
        state,
        0,
        jnp.asarray([0, 0, 0], dtype=jnp.int32),
        jnp.asarray([2, 2, 2], dtype=jnp.int32),
        jnp.asarray([2, 2, 2], dtype=jnp.int32),
        jnp.asarray([True, False, False]),
    )
    padded_duplicate_np = np.asarray(padded_duplicate_rows)
    assert padded_duplicate_np[0, 0] == 10.0
    assert padded_duplicate_np[0, 2] > 0.0


def test_tiny_train_with_small_source_cap_records_compact_metrics(tmp_path: Path) -> None:
    from orbit_ppo_jax.train import main

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    out_dir = tmp_path / "ppo_source_cap"
    main(
        [
            "--bc_checkpoint",
            str(ckpt),
            "--out_dir",
            str(out_dir),
            "--players",
            "2",
            "--envs",
            "1",
            "--steps",
            "1",
            "--episode_steps",
            "500",
            "--updates",
            "1",
            "--eval_games",
            "0",
            "--source_cap",
            "2",
        ]
    )

    config = json.loads((out_dir / "config.json").read_text())
    metrics = json.loads((out_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert config["source_cap"] == 2
    assert metrics["source_cap"] == 2.0
    assert "selected_decisions" in metrics
    assert "dropped_decisions" in metrics
    assert metrics["selected_decisions"] <= metrics["decisions"]
    assert (out_dir / "latest" / "params.npz").exists()



def test_explicit_rollout_steps_wins_over_legacy_steps() -> None:
    from orbit_ppo_jax.train import build_arg_parser, config_from_args

    args = build_arg_parser().parse_args(
        [
            "--bc_checkpoint",
            "bc.pt",
            "--out_dir",
            "out",
            "--steps",
            "7",
            "--rollout_steps",
            "11",
            "--episode_steps",
            "123",
        ]
    )
    config = config_from_args(args)

    assert config.rollout_steps == 11
    assert config.episode_steps == 123


def test_compute_gae_uses_last_value_for_truncated_rollout() -> None:
    from orbit_ppo_jax.train import _compute_gae

    rewards = jnp.zeros((2, 1), dtype=jnp.float32)
    values = jnp.zeros((2, 1), dtype=jnp.float32)
    dones = jnp.zeros((2, 1), dtype=jnp.float32)
    last_values = jnp.asarray([10.0], dtype=jnp.float32)

    advantages, returns = _compute_gae(rewards, values, dones, last_values, gamma=1.0, lam=1.0)

    np.testing.assert_allclose(np.asarray(advantages[:, 0]), np.asarray([10.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(returns[:, 0]), np.asarray([10.0, 10.0], dtype=np.float32))


def test_compute_gae_masks_terminal_bootstrap() -> None:
    from orbit_ppo_jax.train import _compute_gae

    rewards = jnp.zeros((2, 1), dtype=jnp.float32)
    values = jnp.zeros((2, 1), dtype=jnp.float32)
    dones = jnp.asarray([[0.0], [1.0]], dtype=jnp.float32)
    last_values = jnp.asarray([10.0], dtype=jnp.float32)

    advantages, returns = _compute_gae(rewards, values, dones, last_values, gamma=1.0, lam=1.0)

    np.testing.assert_allclose(np.asarray(advantages[:, 0]), np.asarray([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(returns[:, 0]), np.asarray([0.0, 0.0], dtype=np.float32))


def test_compute_gae_mixed_done_envs_bootstraps_only_unfinished_envs() -> None:
    from orbit_ppo_jax.train import _compute_gae

    rewards = jnp.zeros((2, 2), dtype=jnp.float32)
    values = jnp.zeros((2, 2), dtype=jnp.float32)
    dones = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    last_values = jnp.asarray([10.0, 10.0], dtype=jnp.float32)

    advantages, returns = _compute_gae(rewards, values, dones, last_values, gamma=1.0, lam=1.0)

    np.testing.assert_allclose(np.asarray(advantages[:, 0]), np.asarray([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(advantages[:, 1]), np.asarray([10.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(returns), np.asarray(advantages))


def test_persistent_train_state_advances_across_updates(tmp_path: Path) -> None:
    from orbit_ppo_jax.train import main

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    out_dir = tmp_path / "ppo_persistent"
    main(
        [
            "--bc_checkpoint",
            str(ckpt),
            "--out_dir",
            str(out_dir),
            "--players",
            "2",
            "--envs",
            "1",
            "--rollout_steps",
            "1",
            "--episode_steps",
            "500",
            "--updates",
            "2",
            "--eval_games",
            "0",
        ]
    )

    rows = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["mean_episode_step"] == 1.0
    assert rows[1]["mean_episode_step"] == 2.0
    assert rows[1]["reset_count"] == 0.0


def test_tiny_train_can_reset_from_official_state_bank(tmp_path: Path) -> None:
    from orbit_jax_env.official_state_dataset import save_state_bank
    from orbit_jax_env.state import manual_state
    from orbit_ppo_jax.train import main

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    bank_path = tmp_path / "bank.npz"
    state0 = manual_state(
        planet_rows=[[10, 0, 20.0, 50.0, 2.0, 10.0, 3.0], [11, -1, 70.0, 50.0, 2.0, 5.0, 1.0]],
        num_players=2,
        episode_steps=500,
    )
    state1 = manual_state(
        planet_rows=[[20, 0, 30.0, 50.0, 2.0, 10.0, 3.0], [21, -1, 80.0, 50.0, 2.0, 5.0, 1.0]],
        num_players=2,
        episode_steps=500,
    )
    states = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), state0, state1)
    save_state_bank(
        bank_path,
        states,
        {
            "players": 2,
            "episode_steps": 500,
            "ship_speed": 6.0,
            "source": "kaggle_official",
            "seed_start": 0,
            "seed_count": 2,
        },
    )
    out_dir = tmp_path / "ppo_bank"

    main(
        [
            "--bc_checkpoint",
            str(ckpt),
            "--out_dir",
            str(out_dir),
            "--players",
            "2",
            "--envs",
            "1",
            "--rollout_steps",
            "1",
            "--episode_steps",
            "500",
            "--updates",
            "1",
            "--eval_games",
            "0",
            "--initial_state_bank",
            str(bank_path),
            "--state_bank_mode",
            "cycle",
        ]
    )

    config = json.loads((out_dir / "config.json").read_text())
    metrics = json.loads((out_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert config["reset_source"] == "official_state_bank"
    assert metrics["reset_source"] == "official_state_bank"


def test_eval_vs_heuristic_runs_with_mocked_kaggle_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from orbit_ppo_jax.bc_policy import init_value_head, load_bc_jax_params
    from orbit_ppo_jax.checkpointing import save_jax_checkpoint
    from orbit_ppo_jax.eval_vs_heuristic import evaluate

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    bc_params, bc_config = load_bc_jax_params(ckpt)
    params = {"bc": bc_params, "value": init_value_head(jax.random.PRNGKey(0), int(bc_config["hidden_size"]))}
    save_jax_checkpoint(
        tmp_path / "jax" / "latest",
        params,
        {"bc_checkpoint": str(ckpt), "bc_model_config": bc_config, "players": 2, "steps": 20},
        {},
    )

    class FakeEnv:
        def __init__(self) -> None:
            self.steps = [[types.SimpleNamespace(reward=1.0, status="DONE"), types.SimpleNamespace(reward=-1.0, status="DONE")]]

        def run(self, agents):
            obs = {
                "player": 0,
                "planets": [[10, 0, 20.0, 50.0, 2.0, 20.0, 3.0], [11, -1, 70.0, 50.0, 2.0, 5.0, 1.0]],
                "initial_planets": [[10, 0, 20.0, 50.0, 2.0, 20.0, 3.0], [11, -1, 70.0, 50.0, 2.0, 5.0, 1.0]],
                "fleets": [],
                "step": 0,
                "episode_steps": 20,
                "num_players": 2,
            }
            moves = agents[0](obs, types.SimpleNamespace(players=2, episodeSteps=20))
            assert isinstance(moves, list)
            return self.steps

    fake_kaggle = types.SimpleNamespace(make=lambda *args, **kwargs: FakeEnv())
    monkeypatch.setitem(sys.modules, "kaggle_environments", fake_kaggle)
    monkeypatch.setattr("orbit_ppo_jax.eval_vs_heuristic.make_opponent", lambda *args, **kwargs: (lambda obs, config: []))

    made_configs = []

    def fake_make(_name, configuration, debug=False):
        made_configs.append(configuration)
        return FakeEnv()

    fake_kaggle.make = fake_make

    summary = evaluate(tmp_path / "jax" / "latest", "orbit_wars_base.py", games=1, players=2, out_dir=tmp_path / "eval")

    assert summary["average_final_reward"] == 1.0
    assert summary["episode_steps"] == 500
    assert made_configs[0]["episodeSteps"] == 500
    assert (tmp_path / "eval" / "summary.json").exists()


def test_eval_vs_heuristic_episode_steps_override_with_mocked_kaggle_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from orbit_ppo_jax.bc_policy import init_value_head, load_bc_jax_params
    from orbit_ppo_jax.checkpointing import save_jax_checkpoint
    from orbit_ppo_jax.eval_vs_heuristic import evaluate

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    bc_params, bc_config = load_bc_jax_params(ckpt)
    params = {"bc": bc_params, "value": init_value_head(jax.random.PRNGKey(0), int(bc_config["hidden_size"]))}
    save_jax_checkpoint(
        tmp_path / "jax" / "latest",
        params,
        {"bc_checkpoint": str(ckpt), "bc_model_config": bc_config, "players": 2, "episode_steps": 300},
        {},
    )

    made_configs = []

    class FakeEnv:
        def __init__(self) -> None:
            self.steps = [[types.SimpleNamespace(reward=1.0, status="DONE"), types.SimpleNamespace(reward=-1.0, status="DONE")]]

        def run(self, _agents):
            return self.steps

    fake_kaggle = types.SimpleNamespace(make=lambda _name, configuration, debug=False: (made_configs.append(configuration) or FakeEnv()))
    monkeypatch.setitem(sys.modules, "kaggle_environments", fake_kaggle)
    monkeypatch.setattr("orbit_ppo_jax.eval_vs_heuristic.make_opponent", lambda *args, **kwargs: (lambda obs, config: []))

    summary = evaluate(tmp_path / "jax" / "latest", "orbit_wars_base.py", games=1, players=2, out_dir=tmp_path / "eval", episode_steps=222)

    assert summary["episode_steps"] == 222
    assert made_configs[0]["episodeSteps"] == 222
