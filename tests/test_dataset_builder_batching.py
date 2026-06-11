from __future__ import annotations

import json
import math
from pathlib import Path

from orbit_training_prep.dataset_builder import DatasetBuilder, read_jsonl
from orbit_training_prep.source_turn_store import SourceTurnDatasetReader
from orbit_training_prep.exact_target_sim import ExactTargetSimulator, resolve_geometry_device


def static_obs(planets: list[list[float]]) -> dict:
    return {
        "step": 0,
        "player": 0,
        "planets": planets,
        "initial_planets": [list(p) for p in planets],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }


def replay_with_actions(path: Path, obs: dict, action: list[list[float]]) -> None:
    replay = {
        "info": {"EpisodeId": "batched-target-test"},
        "configuration": {"episodeSteps": 2},
        "rewards": [1.0],
        "steps": [
            [{"observation": obs, "status": "ACTIVE", "reward": 0}],
            [{"observation": {}, "action": action, "status": "DONE", "reward": 1}],
        ],
    }
    path.write_text(json.dumps(replay), encoding="utf-8")


def test_exact_target_simulator_batches_launches_with_single_observation() -> None:
    obs = static_obs(
        [
            [1, 0, 10.0, 50.0, 1.0, 80.0, 1.0],
            [2, -1, 30.0, 52.0, 3.0, 5.0, 1.0],
            [3, -1, 80.0, 50.0, 2.0, 5.0, 1.0],
        ]
    )
    moves = [
        {"source_planet_id": 1, "source_slot": 0, "raw_angle": 0.0, "ships": 10},
        {"source_planet_id": 1, "source_slot": 0, "raw_angle": math.pi, "ships": 10},
    ]
    sim = ExactTargetSimulator(horizon=80, device="cpu")

    batched = sim.first_hits_for_launches(obs, 0, moves)
    scalar = [sim.first_hit_for_launch(obs, 0, move) for move in moves]

    assert batched == scalar
    assert [hit["hit_type"] for hit in batched] == ["planet", "bounds"]


def test_exact_target_simulator_normalizes_device_case() -> None:
    assert resolve_geometry_device("CPU") == "cpu"


def test_dataset_builder_records_device_and_batched_inference_mode(tmp_path: Path) -> None:
    obs = static_obs(
        [
            [1, 0, 10.0, 50.0, 1.0, 80.0, 1.0],
            [2, -1, 30.0, 52.0, 3.0, 5.0, 1.0],
            [3, -1, 80.0, 50.0, 2.0, 5.0, 1.0],
        ]
    )
    replay_path = tmp_path / "replay.json"
    out_dir = tmp_path / "dataset"
    replay_with_actions(replay_path, obs, [[1, 0.0, 10], [1, math.pi, 10]])

    metadata = DatasetBuilder(horizon=80, device="cpu", write_debug_jsonl=True).build_from_replay(replay_path, out_dir)
    launch_rows = read_jsonl(out_dir / "debug" / "launch_rows.jsonl")

    assert metadata["geometry_device"] == "cpu"
    assert metadata["target_inference_mode"] == "batched_exact_first_hit_with_angular_fallback"
    assert metadata["stats"]["raw_launch_batches"] == 1
    assert metadata["stats"]["max_raw_launch_batch_size"] == 2
    assert [row["target_inference_method"] for row in launch_rows] == ["first_contact", "angular_nearest"]
    assert "pair_rank_rows" not in metadata["files"]
    assert "pair_feature_names" in metadata
    assert not (out_dir / "pair_rank_rows.jsonl").exists()


def test_dataset_builder_source_rows_are_bc_ready_and_dense_dropped_positive_is_noop(tmp_path: Path) -> None:
    obs = static_obs(
        [
            [1, 0, 10.0, 50.0, 1.0, 80.0, 1.0],
            [2, -1, 30.0, 52.0, 3.0, 5.0, 1.0],
            [3, -1, 80.0, 50.0, 2.0, 5.0, 1.0],
        ]
    )
    replay_path = tmp_path / "replay.json"
    out_dir = tmp_path / "dataset"
    replay_with_actions(replay_path, obs, [[1, 0.0, 10], [1, math.pi, 10]])

    metadata = DatasetBuilder(horizon=80, device="cpu", write_debug_jsonl=True).build_from_replay(replay_path, out_dir)
    source_rows = read_jsonl(out_dir / "debug" / "source_turn_rows.jsonl")
    reader = SourceTurnDatasetReader(out_dir)

    assert source_rows == []
    assert metadata["stats"]["raw_source_turns"] == 1
    assert metadata["stats"]["raw_positive_source_turns"] == 1
    assert metadata["stats"]["train_source_turns"] == 0
    assert metadata["stats"]["train_positive_source_turns"] == 0
    assert metadata["stats"]["dropped_source_turns"] == 1
    assert metadata["stats"]["dropped_ambiguous_sources"] == 1
    assert int(reader.samples["target_label"].shape[0]) == 0
    assert int(reader.samples["amount_label"].shape[0]) == 0


def test_dataset_builder_fresh_output_validates_without_repair(tmp_path: Path) -> None:
    from orbit_training_prep.validate_dataset import validate_dataset

    obs = static_obs(
        [
            [1, 0, 10.0, 50.0, 1.0, 80.0, 1.0],
            [2, -1, 30.0, 52.0, 3.0, 5.0, 1.0],
        ]
    )
    replay_path = tmp_path / "replay.json"
    out_dir = tmp_path / "dataset"
    replay_with_actions(replay_path, obs, [[1, 0.0, 10]])

    DatasetBuilder(horizon=80, device="cpu").build_from_replay(replay_path, out_dir)
    report = validate_dataset(out_dir)

    assert report["counts"]["bc_invalid_source_rows"] == 0
    assert report["counts"]["drop_for_v1_bc_rows"] == 0
    assert report["counts"]["source_turn_rows"] == 1


def test_dataset_builder_emits_geometry_viability_masks(tmp_path: Path) -> None:
    obs = static_obs(
        [
            [1, 0, 10.0, 50.0, 1.0, 80.0, 1.0],
            [2, -1, 30.0, 52.0, 3.0, 5.0, 1.0],
            [3, -1, 80.0, 50.0, 2.0, 5.0, 1.0],
        ]
    )
    replay_path = tmp_path / "replay.json"
    out_dir = tmp_path / "dataset"
    replay_with_actions(replay_path, obs, [[1, 0.0, 10]])

    DatasetBuilder(horizon=80, device="cpu").build_from_replay(replay_path, out_dir)
    reader = SourceTurnDatasetReader(out_dir)

    assert reader.samples["target_mask"].shape == (1, 65)
    assert reader.samples["amount_mask"].shape == (1, 7)
    assert bool(reader.samples["target_mask"][0, 1])
    assert not bool(reader.samples["target_mask"][0, 2])
    assert bool(reader.samples["target_mask"][0, 64])
    assert reader.samples["amount_mask"][0].tolist() == [False, False, True, True, True, True, True]


def test_viability_mask_is_amount_conditioned() -> None:
    from orbit_training_prep.viability import compute_viability_masks

    obs = static_obs(
        [
            [1, 0, 10.0, 10.0, 1.0, 80.0, 1.0],
            [2, -1, 30.0, 10.0, 1.0, 5.0, 1.0],
        ]
    )

    target_mask, amount_mask = compute_viability_masks(obs, 0, horizon=8, device="cpu")

    assert bool(target_mask[0, 1])
    assert not bool(amount_mask[0, 1, 1])
    assert bool(amount_mask[0, 1, 6])
