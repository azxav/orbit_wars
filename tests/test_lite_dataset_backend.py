from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from orbit_training_prep import dataset_builder
from orbit_training_prep.dataset_builder import DatasetBuilder
from orbit_training_prep.schema import NOOP_TARGET_SLOT
from orbit_training_prep.source_turn_store import SourceTurnDatasetReader


def _write_two_planet_replay(path: Path) -> None:
    replay = {
        "info": {"EpisodeId": "lite-backend"},
        "configuration": {"episodeSteps": 3},
        "rewards": [1.0],
        "steps": [
            [
                {
                    "observation": {
                        "step": 0,
                        "player": 0,
                        "episode_steps": 3,
                        "players": 2,
                        "planets": [
                            [1, 0, 10.0, 10.0, 1.0, 30.0, 1.0],
                            [2, 1, 30.0, 10.0, 1.0, 5.0, 1.0],
                        ],
                        "initial_planets": [
                            [1, 0, 10.0, 10.0, 1.0, 30.0, 1.0],
                            [2, 1, 30.0, 10.0, 1.0, 5.0, 1.0],
                        ],
                        "fleets": [],
                        "comets": [],
                    },
                    "status": "ACTIVE",
                    "reward": 0,
                }
            ],
            [
                {
                    "observation": {},
                    "action": [[1, 0.0, 10]],
                    "status": "ACTIVE",
                    "reward": 0,
                }
            ],
            [{"observation": {}, "action": [], "status": "DONE", "reward": 1}],
        ],
    }
    path.write_text(json.dumps(replay), encoding="utf-8")


def test_lite_backend_builds_without_exact_target_or_viability_simulation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_path = tmp_path / "replay.json"
    _write_two_planet_replay(replay_path)

    def fail_target_inferer(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("lite backend must not construct the exact TargetInferer")

    def fail_exact_masks(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("lite backend must not call exact compute_viability_masks")

    monkeypatch.setattr(dataset_builder, "TargetInferer", fail_target_inferer)
    monkeypatch.setattr(dataset_builder, "compute_viability_masks", fail_exact_masks)

    out_dir = tmp_path / "dataset"
    metadata = DatasetBuilder(horizon=12, device="cpu", backend="lite").build_from_replay(replay_path, out_dir)
    reader = SourceTurnDatasetReader(out_dir)

    assert metadata["backend"] == "lite"
    assert metadata["target_inference_mode"] == "lite-arrival"
    assert metadata["mask_mode"] == "lite-permissive-label-corrected"
    assert metadata["pair_eta_mode"] == "lite-movement-cache"
    assert int(reader.samples["target_label"][0]) == 1
    assert int(reader.samples["amount_label"][0]) > 0
    assert bool(reader.samples["target_mask"][0, 1])
    assert bool(reader.samples["amount_mask"][0, int(reader.samples["amount_label"][0])])
    assert reader.samples["pair_features"].shape == (1, 65, 15)
    assert np.isfinite(reader.samples["pair_features"][0]).all()


def test_exact_backend_still_uses_existing_contract(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    _write_two_planet_replay(replay_path)

    metadata = DatasetBuilder(horizon=12, device="cpu", backend="exact").build_from_replay(replay_path, tmp_path / "dataset")

    assert metadata["backend"] == "exact"
    assert metadata["target_inference_mode"] == "batched_exact_first_hit_with_angular_fallback"


def test_lite_backend_permissive_masks_are_label_corrected() -> None:
    from orbit_training_prep.lite_backend import build_lite_context, compute_lite_viability_masks

    obs = {
        "step": 0,
        "player": 0,
        "episode_steps": 3,
        "players": 2,
        "planets": [
            [1, 0, 10.0, 10.0, 1.0, 30.0, 1.0],
            [2, 1, 30.0, 10.0, 1.0, 5.0, 1.0],
        ],
        "initial_planets": [
            [1, 0, 10.0, 10.0, 1.0, 30.0, 1.0],
            [2, 1, 30.0, 10.0, 1.0, 5.0, 1.0],
        ],
        "fleets": [],
        "comets": [],
    }
    ctx = build_lite_context(obs, 0, horizon=12)
    target_mask, amount_mask = compute_lite_viability_masks(
        ctx,
        labels_by_source={0: (NOOP_TARGET_SLOT, 0), 5: (1, 6)},
    )

    assert target_mask[0, NOOP_TARGET_SLOT]
    assert amount_mask[0, NOOP_TARGET_SLOT, 0]
    assert target_mask[5, 1]
    assert amount_mask[5, 1, 6]
