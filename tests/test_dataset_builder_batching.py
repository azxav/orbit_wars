from __future__ import annotations

import json
import math
from pathlib import Path

from orbit_training_prep.dataset_builder import DatasetBuilder, read_jsonl
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

    metadata = DatasetBuilder(horizon=80, device="cpu").build_from_replay(replay_path, out_dir)
    launch_rows = read_jsonl(out_dir / "launch_rows.jsonl")

    assert metadata["geometry_device"] == "cpu"
    assert metadata["target_inference_mode"] == "batched_exact_first_hit_with_angular_fallback"
    assert metadata["stats"]["raw_launch_batches"] == 1
    assert metadata["stats"]["max_raw_launch_batch_size"] == 2
    assert [row["target_inference_method"] for row in launch_rows] == ["first_contact", "angular_nearest"]
