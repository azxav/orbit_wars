from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from orbit_training_prep.schema import NOOP_TARGET_SLOT


def _make_dense_dataset(path: Path, *, rows: int = 16) -> None:
    """Historical helper name; now writes source_turn_memmap_v1."""
    from orbit_training_prep.features import (
        GLOBAL_FEATURE_NAMES,
        PAIR_FEATURE_NAMES,
        PLANET_FEATURE_NAMES,
        TARGET_STATE_FEATURE_NAMES,
    )
    from orbit_training_prep.source_turn_store import SourceTurnDatasetWriter

    pmax = 64
    rng = np.random.default_rng(7)
    writer = SourceTurnDatasetWriter(path)
    for i in range(rows):
        planet_features = rng.normal(size=(pmax, len(PLANET_FEATURE_NAMES))).astype(np.float32)
        planet_features[:, 0] = 1.0
        planet_features[:, 7] = np.clip(planet_features[:, 7], 0.0, 1.0)
        global_features = rng.normal(size=(len(GLOBAL_FEATURE_NAMES),)).astype(np.float32)
        target_state_features = rng.normal(size=(pmax, len(TARGET_STATE_FEATURE_NAMES))).astype(np.float32)
        state_idx = writer.append_state(
            planet_features=planet_features,
            global_features=global_features,
            target_state_features=target_state_features,
        )
        source = i % 4
        target = (source + 1) % 4 if i % 3 else NOOP_TARGET_SLOT
        amount = 0 if target == NOOP_TARGET_SLOT else 3
        pair_features = rng.normal(size=(pmax + 1, len(PAIR_FEATURE_NAMES))).astype(np.float32)
        pair_features[NOOP_TARGET_SLOT] = 0.0
        target_mask = np.zeros((pmax + 1,), dtype=bool)
        amount_mask = np.zeros((7,), dtype=bool)
        target_mask[NOOP_TARGET_SLOT] = True
        if target == NOOP_TARGET_SLOT:
            amount_mask[0] = True
        else:
            target_mask[target] = True
            amount_mask[amount] = True
        writer.append_sample(
            state_index=state_idx,
            source_slot=source,
            target_label=target,
            amount_label=amount,
            sample_weight=np.float32(0.1 if target == NOOP_TARGET_SLOT else 0.55),
            step=i,
            pair_features=pair_features,
            target_mask=target_mask,
            amount_mask=amount_mask,
        )
    writer.finalize(extra_metadata={"stats": {"states": rows, "source_turn_rows": rows}})


def test_dataset_loads_required_fields_and_labels(tmp_path: Path) -> None:
    from orbit_training_prep.features import PAIR_FEATURE_NAMES
    from orbit_bc_training.dataset import OrbitBCDataset

    _make_dense_dataset(tmp_path)
    ds = OrbitBCDataset(tmp_path)
    sample = ds[0]

    assert sample["planet_features"].shape == (64, 16)
    assert sample["global_features"].shape == (10,)
    assert sample["target_state_features"].shape == (64, 9)
    assert sample["pair_features"].shape == (65, len(PAIR_FEATURE_NAMES))
    assert sample["pair_features"].dtype == np.float32
    assert sample["target_mask"].shape == (65,)
    assert sample["target_mask"][NOOP_TARGET_SLOT]
    assert 0 <= int(sample["target_label"]) <= NOOP_TARGET_SLOT
    assert int(ds.noop_target_slot) == NOOP_TARGET_SLOT


def test_dataset_uses_sample_level_viability_masks(tmp_path: Path) -> None:
    from orbit_bc_training.dataset import OrbitBCDataset

    _make_dense_dataset(tmp_path, rows=1)
    target_mask = np.load(tmp_path / "samples" / "target_mask.npy")
    amount_mask = np.load(tmp_path / "samples" / "amount_mask.npy")
    target_label = np.load(tmp_path / "samples" / "target_label.npy")
    amount_label = np.load(tmp_path / "samples" / "amount_label.npy")
    target_mask[0] = False
    target_mask[0, 2] = True
    amount_mask[0] = False
    amount_mask[0, 4] = True
    target_label[0] = 2
    amount_label[0] = 4
    np.save(tmp_path / "samples" / "target_mask.npy", target_mask, allow_pickle=False)
    np.save(tmp_path / "samples" / "amount_mask.npy", amount_mask, allow_pickle=False)
    np.save(tmp_path / "samples" / "target_label.npy", target_label, allow_pickle=False)
    np.save(tmp_path / "samples" / "amount_label.npy", amount_label, allow_pickle=False)

    sample = OrbitBCDataset(tmp_path)[0]

    assert sample["target_mask"][2]
    assert not sample["target_mask"][1]
    assert sample["amount_mask"].tolist() == [False, False, False, False, True, False, False]


def test_dataset_rejects_label_outside_sample_viability_mask(tmp_path: Path) -> None:
    import pytest

    from orbit_bc_training.dataset import OrbitBCDataset

    _make_dense_dataset(tmp_path, rows=1)
    target_label = np.load(tmp_path / "samples" / "target_label.npy")
    amount_label = np.load(tmp_path / "samples" / "amount_label.npy")
    target_label[0] = 2
    amount_label[0] = 4
    np.save(tmp_path / "samples" / "target_label.npy", target_label, allow_pickle=False)
    np.save(tmp_path / "samples" / "amount_label.npy", amount_label, allow_pickle=False)

    ds = OrbitBCDataset(tmp_path)
    with pytest.raises(RuntimeError, match="target label .* outside"):
        _ = ds[0]


def test_dataset_rejects_old_dense_arrays(tmp_path: Path) -> None:
    import pytest

    from orbit_bc_training.dataset import OrbitBCDataset

    np.savez_compressed(tmp_path / "dense_bc_arrays.npz", planet_features=np.zeros((1, 64, 16), dtype=np.float32))

    with pytest.raises(RuntimeError, match="Old dense_bc_arrays.npz dataset detected"):
        OrbitBCDataset(tmp_path)


def test_dataset_loads_pair_features_and_leakage_free_globals(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    from orbit_bc_training.dataset import OrbitBCDataset, collate_bc_samples
    from orbit_training_prep.features import GLOBAL_FEATURE_NAMES, PAIR_FEATURE_NAMES

    _make_dense_dataset(tmp_path)
    ds = OrbitBCDataset(tmp_path)
    sample = ds[1]
    assert sample["pair_features"].shape == (65, len(PAIR_FEATURE_NAMES))
    assert sample["target_state_features"].shape[0] == 64
    assert np.isfinite(sample["pair_features"]).all()
    assert np.allclose(sample["pair_features"][NOOP_TARGET_SLOT], 0.0)
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


def test_amount_loss_ignores_noop_rows() -> None:
    from orbit_bc_training.losses import bc_loss_and_metrics

    target_logits = torch.zeros((2, NOOP_TARGET_SLOT + 1))
    target_logits[0, NOOP_TARGET_SLOT] = 10.0
    target_logits[1, 1] = 10.0
    outputs = {
        "target_logits": target_logits,
        "amount_logits": torch.tensor([[100.0, -100.0], [-100.0, 100.0]]),
    }
    batch = {
        "target_mask": torch.ones((2, NOOP_TARGET_SLOT + 1), dtype=torch.bool),
        "amount_mask": torch.ones((2, 2), dtype=torch.bool),
        "target_label": torch.tensor([NOOP_TARGET_SLOT, 1]),
        "amount_label": torch.tensor([1, 1]),
        "sample_weight": torch.tensor([10.0, 1.0]),
        "is_noop": torch.tensor([True, False]),
    }

    _, metrics = bc_loss_and_metrics(outputs, batch)

    assert metrics["amount_loss"] < 1.0e-6


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


def test_tiny_batch_loss_decreases(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    from orbit_bc_training.config import BCModelConfig
    from orbit_bc_training.dataset import OrbitBCDataset, collate_bc_samples
    from orbit_bc_training.losses import bc_loss_and_metrics
    from orbit_bc_training.model import EntityBCPolicy

    torch.set_num_threads(1)
    _make_dense_dataset(tmp_path, rows=4)
    batch = next(iter(DataLoader(OrbitBCDataset(tmp_path), batch_size=4, shuffle=False, collate_fn=collate_bc_samples)))
    model = EntityBCPolicy(
        BCModelConfig(
            planet_feature_dim=batch["planet_features"].shape[-1],
            global_feature_dim=batch["global_features"].shape[-1],
            target_state_feature_dim=batch["target_state_features"].shape[-1],
            pair_feature_dim=batch["pair_features"].shape[-1],
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            mlp_size=16,
        )
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    opt.zero_grad()
    first, _ = bc_loss_and_metrics(model(batch), batch)
    first.backward()
    opt.step()
    with torch.no_grad():
        last, _ = bc_loss_and_metrics(model(batch), batch)
    assert float(last.detach()) < float(first.detach())


def test_model_init_does_not_emit_nested_tensor_warning() -> None:
    import warnings

    from orbit_bc_training.config import BCModelConfig
    from orbit_bc_training.model import EntityBCPolicy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        EntityBCPolicy(
            BCModelConfig(
                planet_feature_dim=16,
                global_feature_dim=10,
                target_state_feature_dim=9,
                pair_feature_dim=15,
                hidden_size=8,
                num_layers=1,
                num_heads=2,
                mlp_size=16,
            )
        )

    assert not any("enable_nested_tensor" in str(w.message) for w in caught)


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
