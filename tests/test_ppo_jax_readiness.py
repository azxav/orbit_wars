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
            "--updates",
            "1",
            "--eval_games",
            "0",
        ]
    )

    assert (out_dir / "config.json").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "latest" / "params.npz").exists()
    assert json.loads((out_dir / "metrics.jsonl").read_text().splitlines()[0])["update"] == 1


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

    summary = evaluate(tmp_path / "jax" / "latest", "orbit_wars_base.py", games=1, players=2, out_dir=tmp_path / "eval")

    assert summary["average_final_reward"] == 1.0
    assert (tmp_path / "eval" / "summary.json").exists()
