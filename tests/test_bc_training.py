from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from orbit_training_prep.schema import NOOP_TARGET_SLOT


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_dense_dataset(path: Path, *, rows: int = 16) -> None:
    from orbit_training_prep.features import (
        GLOBAL_FEATURE_NAMES,
        PAIR_FEATURE_NAMES,
        PLANET_FEATURE_NAMES,
        TARGET_STATE_FEATURE_NAMES,
    )

    pmax = 64
    rng = np.random.default_rng(7)
    planet_features = rng.normal(size=(rows, pmax, len(PLANET_FEATURE_NAMES))).astype(np.float32)
    planet_features[:, :, 0] = 1.0
    planet_features[:, :, 7] = np.clip(planet_features[:, :, 7], 0.0, 1.0)
    global_features = rng.normal(size=(rows, len(GLOBAL_FEATURE_NAMES))).astype(np.float32)
    target_state_features = rng.normal(size=(rows, pmax, len(TARGET_STATE_FEATURE_NAMES))).astype(np.float32)
    target_labels = np.full((rows, pmax), NOOP_TARGET_SLOT, dtype=np.int64)
    amount_labels = np.zeros((rows, pmax), dtype=np.int64)
    source_mask = np.zeros((rows, pmax), dtype=np.float32)
    source_rows = []
    state_rows = []
    for i in range(rows):
        source = i % 4
        target = (source + 1) % 4 if i % 3 else NOOP_TARGET_SLOT
        amount = 0 if target == NOOP_TARGET_SLOT else 3
        source_mask[i, source] = 1.0
        target_labels[i, source] = target
        amount_labels[i, source] = amount
        obs_uid = f"e{i // 4}:{i}:p0"
        state_rows.append({"obs_uid": obs_uid, "episode_id": f"e{i // 4}", "step_index": i, "player_id": 0})
        source_rows.append(
            {
                "source_turn_uid": f"{obs_uid}:s{source}",
                "obs_uid": obs_uid,
                "episode_id": f"e{i // 4}",
                "step_index": i,
                "player_id": 0,
                "source_slot": source,
                "target_slot_label": target,
                "amount_bin_label": amount,
                "sample_weight": 0.2 if target == NOOP_TARGET_SLOT else 1.0,
                "drop_for_v1_bc": False,
            }
        )
    np.savez_compressed(
        path / "dense_bc_arrays.npz",
        planet_features=planet_features,
        global_features=global_features,
        target_state_features=target_state_features,
        target_labels=target_labels,
        amount_labels=amount_labels,
        source_mask=source_mask,
        planet_feature_names=np.asarray(PLANET_FEATURE_NAMES),
        global_feature_names=np.asarray(GLOBAL_FEATURE_NAMES),
        target_state_feature_names=np.asarray(TARGET_STATE_FEATURE_NAMES),
        pair_feature_names=np.asarray(PAIR_FEATURE_NAMES),
    )
    _write_jsonl(path / "state_rows.jsonl", state_rows)
    _write_jsonl(path / "source_turn_rows.jsonl", source_rows)


def test_feature_contract_excludes_future_labels() -> None:
    from orbit_training_prep.features import GLOBAL_FEATURE_NAMES, build_feature_state

    obs = {
        "player": 0,
        "step": 25,
        "episode_steps": 500,
        "planets": [
            [101, 0, 10.0, 10.0, 1.0, 20.0, 1.0],
            [202, -1, 20.0, 10.0, 1.0, 4.0, 1.0],
            [303, 1, 50.0, 50.0, 1.0, 30.0, 1.0],
        ],
        "initial_planets": [],
        "fleets": [],
    }

    forbidden = {"final_reward", "winner", "winner_action", "target_label", "expert_action"}
    assert forbidden.isdisjoint(set(GLOBAL_FEATURE_NAMES))
    state = build_feature_state(obs, player_id=0, max_planets=64)
    assert state.planet_features.shape == (64, 16)
    assert state.global_features.shape == (10,)
    assert state.target_state_features.shape == (64, 9)
    assert np.isfinite(state.planet_features).all()
    assert np.isfinite(state.global_features).all()
    assert np.isfinite(state.target_state_features).all()


def test_dataset_loads_required_fields_and_labels(tmp_path: Path) -> None:
    from orbit_bc_training.dataset import OrbitBCDataset

    _make_dense_dataset(tmp_path)
    ds = OrbitBCDataset(tmp_path)
    sample = ds[0]

    assert sample["planet_features"].shape == (64, 16)
    assert sample["global_features"].shape == (10,)
    assert sample["target_state_features"].shape == (64, 9)
    assert sample["pair_features"].shape == (65, 22)
    assert sample["target_mask"].shape == (65,)
    assert sample["target_mask"][NOOP_TARGET_SLOT]
    assert 0 <= int(sample["target_label"]) <= NOOP_TARGET_SLOT
    assert int(ds.noop_target_slot) == NOOP_TARGET_SLOT


def test_dataset_filters_v1_invalid_source_rows(tmp_path: Path) -> None:
    from orbit_bc_training.dataset import OrbitBCDataset

    _make_dense_dataset(tmp_path, rows=4)
    rows = [
        {
            "source_turn_uid": "keep",
            "obs_uid": "e0:0:p0",
            "episode_id": "e0",
            "step_index": 0,
            "player_id": 0,
            "source_slot": 0,
            "target_slot_label": 1,
            "amount_bin_label": 3,
            "drop_for_v1_bc": False,
            "geometry_viable": True,
            "ambiguous_multi_launch": False,
            "target_inference_method": "first_contact",
        },
        {
            "source_turn_uid": "drop_flag",
            "obs_uid": "e0:1:p0",
            "episode_id": "e0",
            "step_index": 1,
            "player_id": 0,
            "source_slot": 1,
            "target_slot_label": 2,
            "amount_bin_label": 3,
            "drop_for_v1_bc": True,
        },
        {
            "source_turn_uid": "drop_geometry",
            "obs_uid": "e0:2:p0",
            "episode_id": "e0",
            "step_index": 2,
            "player_id": 0,
            "source_slot": 2,
            "target_slot_label": 3,
            "amount_bin_label": 3,
            "geometry_viable": False,
        },
        {
            "source_turn_uid": "drop_angular",
            "obs_uid": "e0:3:p0",
            "episode_id": "e0",
            "step_index": 3,
            "player_id": 0,
            "source_slot": 3,
            "target_slot_label": 4,
            "amount_bin_label": 3,
            "target_inference_method": "angular_nearest",
        },
    ]
    _write_jsonl(tmp_path / "source_turn_rows.jsonl", rows)

    ds = OrbitBCDataset(tmp_path)

    assert len(ds) == 1
    assert ds.rows[0]["source_turn_uid"] == "keep"


def test_dataset_uses_dense_geometry_viability_masks(tmp_path: Path) -> None:
    from orbit_bc_training.dataset import OrbitBCDataset

    _make_dense_dataset(tmp_path, rows=1)
    dense = dict(np.load(tmp_path / "dense_bc_arrays.npz"))
    target_mask = np.zeros((1, 64, 65), dtype=bool)
    amount_mask = np.zeros((1, 64, 65, 7), dtype=bool)
    target_mask[:, :, NOOP_TARGET_SLOT] = True
    amount_mask[:, :, NOOP_TARGET_SLOT, 0] = True
    target_mask[0, 0, 2] = True
    amount_mask[0, 0, 2, 4] = True
    dense["target_viability_mask"] = target_mask
    dense["amount_viability_mask"] = amount_mask
    dense["target_labels"][0, 0] = 2
    dense["amount_labels"][0, 0] = 4
    np.savez_compressed(tmp_path / "dense_bc_arrays.npz", **dense)
    _write_jsonl(
        tmp_path / "source_turn_rows.jsonl",
        [
            {
                "source_turn_uid": "masked",
                "obs_uid": "e0:0:p0",
                "episode_id": "e0",
                "step_index": 0,
                "player_id": 0,
                "source_slot": 0,
                "target_slot_label": 2,
                "amount_bin_label": 4,
                "drop_for_v1_bc": False,
                "geometry_viable": True,
                "target_inference_method": "first_contact",
            }
        ],
    )

    sample = OrbitBCDataset(tmp_path)[0]

    assert sample["target_mask"][2]
    assert not sample["target_mask"][1]
    assert sample["target_mask"][NOOP_TARGET_SLOT]
    assert sample["amount_mask"].tolist() == [False, False, False, False, True, False, False]


def test_dataset_rejects_label_outside_dense_viability_mask(tmp_path: Path) -> None:
    import pytest

    from orbit_bc_training.dataset import OrbitBCDataset

    _make_dense_dataset(tmp_path, rows=1)
    dense = dict(np.load(tmp_path / "dense_bc_arrays.npz"))
    target_mask = np.zeros((1, 64, 65), dtype=bool)
    amount_mask = np.zeros((1, 64, 65, 7), dtype=bool)
    target_mask[:, :, NOOP_TARGET_SLOT] = True
    amount_mask[:, :, NOOP_TARGET_SLOT, 0] = True
    dense["target_viability_mask"] = target_mask
    dense["amount_viability_mask"] = amount_mask
    dense["target_labels"][0, 0] = 2
    dense["amount_labels"][0, 0] = 4
    np.savez_compressed(tmp_path / "dense_bc_arrays.npz", **dense)
    _write_jsonl(
        tmp_path / "source_turn_rows.jsonl",
        [
            {
                "source_turn_uid": "bad",
                "obs_uid": "e0:0:p0",
                "episode_id": "e0",
                "step_index": 0,
                "player_id": 0,
                "source_slot": 0,
                "target_slot_label": 2,
                "amount_bin_label": 4,
                "drop_for_v1_bc": False,
                "geometry_viable": True,
                "target_inference_method": "first_contact",
            }
        ],
    )

    ds = OrbitBCDataset(tmp_path)
    with pytest.raises(RuntimeError, match="target label .* is not geometry-viable"):
        _ = ds[0]


def test_dataset_rejects_old_contract_dense_arrays(tmp_path: Path) -> None:
    import pytest

    from orbit_bc_training.dataset import OrbitBCDataset

    _make_dense_dataset(tmp_path)
    dense = dict(np.load(tmp_path / "dense_bc_arrays.npz"))
    dense["planet_features" + "_" + "v2"] = dense.pop("planet_features")
    dense["global_features" + "_" + "v2"] = dense.pop("global_features")
    dense["target_state_features" + "_" + "v2"] = dense.pop("target_state_features")
    dense["feature" + "_version"] = np.asarray("old")
    np.savez_compressed(tmp_path / "dense_bc_arrays.npz", **dense)

    with pytest.raises(RuntimeError, match="Old feature-versioned dataset detected"):
        OrbitBCDataset(tmp_path)


def test_dataset_resolves_dense_arrays_from_combined_split_layout(tmp_path: Path) -> None:
    from orbit_bc_training.dataset import OrbitBCDataset

    combined = tmp_path / "combined"
    train = combined / "splits" / "train"
    train.mkdir(parents=True)
    _make_dense_dataset(combined)
    (train / "source_turn_rows.jsonl").write_text((combined / "source_turn_rows.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    ds = OrbitBCDataset(train)

    assert ds[0]["planet_features"].shape[-1] == ds.planet_feature_dim


def test_eval_rejects_old_dense_arrays(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from orbit_bc_training import eval_bc_policy
    from orbit_bc_training.config import BCModelConfig

    class FakeModel:
        config = BCModelConfig(
            planet_feature_dim=16,
            global_feature_dim=10,
            target_state_feature_dim=9,
            pair_feature_dim=22,
        )

        def __call__(self, batch):
            raise AssertionError("eval should reject the old dataset before model inference")

    _make_dense_dataset(tmp_path)
    dense = dict(np.load(tmp_path / "dense_bc_arrays.npz"))
    dense["feature" + "_version"] = np.asarray("old")
    np.savez_compressed(tmp_path / "dense_bc_arrays.npz", **dense)
    monkeypatch.setattr(eval_bc_policy, "load_checkpoint", lambda checkpoint, device="cpu": (FakeModel(), {}))

    with pytest.raises(RuntimeError, match="Old feature-versioned dataset detected"):
        eval_bc_policy.evaluate("fake.pt", tmp_path, device="cpu")


def test_dataset_loads_pair_features_and_leakage_free_globals(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    from orbit_bc_training.dataset import OrbitBCDataset, collate_bc_samples
    from orbit_training_prep.features import GLOBAL_FEATURE_NAMES, PAIR_FEATURE_NAMES

    _make_dense_dataset(tmp_path)
    ds = OrbitBCDataset(tmp_path)
    sample = ds[1]
    assert "feature" + "_version" not in sample
    assert sample["pair_features"].shape == (65, len(PAIR_FEATURE_NAMES))
    assert sample["target_state_features"].shape[0] == 64
    assert np.isfinite(sample["pair_features"]).all()
    assert np.allclose(sample["pair_features"][NOOP_TARGET_SLOT, :-1], 0.0)
    assert sample["pair_features"][NOOP_TARGET_SLOT, -1] == 1.0
    assert sample["global_features"].shape == (len(GLOBAL_FEATURE_NAMES),)

    batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=collate_bc_samples)))
    assert tuple(batch["pair_features"].shape) == (4, 65, len(PAIR_FEATURE_NAMES))


def test_masked_target_argmax_ignores_invalid_logits() -> None:
    from orbit_bc_training.losses import masked_argmax

    logits = torch.tensor([[1.0, 100.0, 2.0]])
    mask = torch.tensor([[True, False, True]])
    pred = masked_argmax(logits, mask)
    assert pred.tolist() == [2]


def test_loss_backward_has_finite_gradients(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    from orbit_bc_training.config import BCModelConfig
    from orbit_bc_training.dataset import OrbitBCDataset, collate_bc_samples
    from orbit_bc_training.losses import bc_loss_and_metrics
    from orbit_bc_training.model import EntityBCPolicy

    _make_dense_dataset(tmp_path)
    batch = next(iter(DataLoader(OrbitBCDataset(tmp_path), batch_size=8, collate_fn=collate_bc_samples)))
    model = EntityBCPolicy(
        BCModelConfig(
            planet_feature_dim=batch["planet_features"].shape[-1],
            global_feature_dim=batch["global_features"].shape[-1],
            target_state_feature_dim=batch["target_state_features"].shape[-1],
            pair_feature_dim=batch["pair_features"].shape[-1],
            hidden_size=32,
            num_layers=1,
            num_heads=4,
        )
    )
    out = model(batch)
    loss, metrics = bc_loss_and_metrics(out, batch)
    loss.backward()

    assert math.isfinite(float(loss.detach()))
    assert math.isfinite(metrics["target_loss"])
    assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())


def test_model_forward_uses_pair_features(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    from orbit_bc_training.config import BCModelConfig
    from orbit_bc_training.dataset import OrbitBCDataset, collate_bc_samples
    from orbit_bc_training.model import EntityBCPolicy

    _make_dense_dataset(tmp_path)
    batch = next(iter(DataLoader(OrbitBCDataset(tmp_path), batch_size=4, collate_fn=collate_bc_samples)))
    model = EntityBCPolicy(
        BCModelConfig(
            planet_feature_dim=batch["planet_features"].shape[-1],
            global_feature_dim=batch["global_features"].shape[-1],
            target_state_feature_dim=batch["target_state_features"].shape[-1],
            pair_feature_dim=batch["pair_features"].shape[-1],
            hidden_size=32,
            num_layers=1,
            num_heads=4,
        )
    )
    out = model(batch)
    assert tuple(out["target_logits"].shape) == (4, 65)
    assert tuple(out["amount_logits"].shape) == (4, 7)
    assert torch.isfinite(out["target_logits"]).all()
    assert torch.isfinite(out["amount_logits"]).all()


def test_tiny_batch_can_overfit(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    from orbit_bc_training.config import BCModelConfig
    from orbit_bc_training.dataset import OrbitBCDataset, collate_bc_samples
    from orbit_bc_training.losses import bc_loss_and_metrics
    from orbit_bc_training.model import EntityBCPolicy

    _make_dense_dataset(tmp_path, rows=32)
    loader = DataLoader(OrbitBCDataset(tmp_path), batch_size=32, shuffle=True, collate_fn=collate_bc_samples)
    batch = next(iter(loader))
    model = EntityBCPolicy(
        BCModelConfig(
            planet_feature_dim=batch["planet_features"].shape[-1],
            global_feature_dim=batch["global_features"].shape[-1],
            target_state_feature_dim=batch["target_state_features"].shape[-1],
            pair_feature_dim=batch["pair_features"].shape[-1],
            hidden_size=64,
            num_layers=1,
            num_heads=4,
        )
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    last = None
    for _ in range(35):
        opt.zero_grad()
        loss, _ = bc_loss_and_metrics(model(batch), batch)
        if first is None:
            first = float(loss.detach())
        loss.backward()
        opt.step()
        last = float(loss.detach())
    assert last is not None and first is not None
    assert last < first * 0.35


def test_decode_noop_and_launch_uses_geometry_angle() -> None:
    from orbit_bc_training.decode_policy import decode_bc_prediction

    class FakeGeometry:
        def to_env_moves(self, **kwargs):
            return [[101, 1.2345, 7]]

    obs = {"planets": [[101, 0, 0, 0, 1, 20, 1], [202, 1, 10, 0, 1, 5, 1]]}
    noop_logits = torch.full((65,), -10.0)
    noop_logits[NOOP_TARGET_SLOT] = 10.0
    amount_logits = torch.zeros(7)
    assert decode_bc_prediction(obs, 0, 101, noop_logits, amount_logits, FakeGeometry()) is None

    launch_logits = torch.full((65,), -10.0)
    launch_logits[1] = 5.0
    amount_logits = torch.full((7,), -10.0)
    amount_logits[4] = 5.0
    assert decode_bc_prediction(obs, 0, 101, launch_logits, amount_logits, FakeGeometry()) == [101, 1.2345, 7]


def test_decode_prediction_respects_precomputed_target_and_amount_masks() -> None:
    from orbit_bc_training.decode_policy import decode_bc_prediction

    class FakeGeometry:
        def to_env_moves(self, **kwargs):
            return [[101, 0.0, int(kwargs["ships"][0].item())]]

    obs = {"planets": [[101, 0, 0, 0, 1, 20, 1], [202, 1, 10, 0, 1, 5, 1], [303, 1, 20, 0, 1, 5, 1]]}
    target_logits = torch.full((65,), -10.0)
    target_logits[1] = 100.0
    target_logits[2] = 10.0
    amount_logits = torch.full((7,), -10.0)
    amount_logits[1] = 100.0
    amount_logits[4] = 10.0
    target_mask = torch.zeros(65, dtype=torch.bool)
    target_mask[2] = True
    amount_mask = torch.zeros(7, dtype=torch.bool)
    amount_mask[4] = True

    move = decode_bc_prediction(
        obs,
        0,
        101,
        target_logits,
        amount_logits,
        FakeGeometry(),
        target_mask=target_mask,
        amount_mask=amount_mask,
    )

    assert move == [101, 0.0, 10]


def test_resolve_device_auto_prefers_cuda_when_available(monkeypatch) -> None:
    import torch

    from orbit_bc_training.config import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_resolve_device_cuda_fails_loudly_when_cpu_torch(monkeypatch) -> None:
    import pytest
    import torch

    from orbit_bc_training.config import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_checkpoint_selection_without_eval_dir_is_debug_fallback(tmp_path: Path) -> None:
    from orbit_bc_training.train_bc_policy import checkpoint_selection_metrics

    metrics = checkpoint_selection_metrics(
        valid_metrics={"target_non_noop_accuracy": 0.99},
        selection_eval_dir=None,
    )

    assert metrics["best_selection_mode"] == "validation_debug_fallback"
    assert metrics["true_best"] is False
    assert "gameplay_score" not in metrics


def test_checkpoint_selection_uses_gameplay_eval_dir(tmp_path: Path) -> None:
    from orbit_bc_training.train_bc_policy import checkpoint_selection_metrics

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "games.jsonl").write_text(
        json.dumps(
            {
                "game_id": "g1",
                "players": 2,
                "reward": 1.0,
                "rank": 1,
                "owned_planets_auc": 4.0,
                "total_ships_auc": 40.0,
                "decode_success_rate": 0.75,
                "invalid_decode_rate": 0.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = checkpoint_selection_metrics(
        valid_metrics={"target_non_noop_accuracy": 0.01},
        selection_eval_dir=str(eval_dir),
    )

    assert metrics["best_selection_mode"] == "gameplay_eval"
    assert metrics["true_best"] is True
    assert metrics["gameplay_score"] > 0.0
