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
    pmax = 64
    rng = np.random.default_rng(7)
    planet_features = rng.normal(size=(rows, pmax, 13)).astype(np.float32)
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
        target_labels=target_labels,
        amount_labels=amount_labels,
        source_mask=source_mask,
    )
    _write_jsonl(path / "state_rows.jsonl", state_rows)
    _write_jsonl(path / "source_turn_rows.jsonl", source_rows)


def test_dataset_loads_required_fields_and_labels(tmp_path: Path) -> None:
    from orbit_bc_training.dataset import OrbitBCDataset

    _make_dense_dataset(tmp_path)
    ds = OrbitBCDataset(tmp_path)
    sample = ds[0]

    assert sample["planet_features"].shape == (64, 13)
    assert sample["global_features"].shape[0] > 0
    assert sample["target_mask"].shape == (65,)
    assert sample["target_mask"][NOOP_TARGET_SLOT]
    assert 0 <= int(sample["target_label"]) <= NOOP_TARGET_SLOT
    assert int(ds.noop_target_slot) == NOOP_TARGET_SLOT


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
    model = EntityBCPolicy(BCModelConfig(planet_feature_dim=13, global_feature_dim=batch["global_features"].shape[1], hidden_size=32, num_layers=1, num_heads=4))
    out = model(batch)
    loss, metrics = bc_loss_and_metrics(out, batch)
    loss.backward()

    assert math.isfinite(float(loss.detach()))
    assert math.isfinite(metrics["target_loss"])
    assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())


def test_tiny_batch_can_overfit(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    from orbit_bc_training.config import BCModelConfig
    from orbit_bc_training.dataset import OrbitBCDataset, collate_bc_samples
    from orbit_bc_training.losses import bc_loss_and_metrics
    from orbit_bc_training.model import EntityBCPolicy

    _make_dense_dataset(tmp_path, rows=32)
    loader = DataLoader(OrbitBCDataset(tmp_path), batch_size=32, shuffle=True, collate_fn=collate_bc_samples)
    batch = next(iter(loader))
    model = EntityBCPolicy(BCModelConfig(planet_feature_dim=13, global_feature_dim=batch["global_features"].shape[1], hidden_size=64, num_layers=1, num_heads=4))
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
