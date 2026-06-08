from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbit_training_prep.dataset_builder import DatasetBuilder


def _write_minimal_replay(path: Path, episode_id: str) -> None:
    replay = {
        "info": {"EpisodeId": episode_id},
        "configuration": {"episodeSteps": 2},
        "rewards": [1.0],
        "steps": [
            [
                {
                    "observation": {
                        "step": 0,
                        "player": 0,
                        "planets": [[1, 0, 10.0, 10.0, 1.0, 5.0, 1.0]],
                        "initial_planets": [[1, 0, 10.0, 10.0, 1.0, 5.0, 1.0]],
                        "fleets": [],
                        "comets": [],
                    },
                    "status": "ACTIVE",
                    "reward": 0,
                }
            ],
            [{"observation": {}, "action": [], "status": "DONE", "reward": 1}],
        ],
    }
    path.write_text(json.dumps(replay), encoding="utf-8")


def test_dataset_builder_parallel_workers_matches_serial_output(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    _write_minimal_replay(replay_dir / "a.json", "episode-a")
    _write_minimal_replay(replay_dir / "b.json", "episode-b")

    serial_out = tmp_path / "serial"
    parallel_out = tmp_path / "parallel"

    serial_meta = DatasetBuilder(horizon=8, device="cpu", workers=1).build_from_replay(replay_dir, serial_out)
    parallel_meta = DatasetBuilder(horizon=8, device="cpu", workers=2).build_from_replay(replay_dir, parallel_out)

    assert parallel_meta["replay_paths"] == serial_meta["replay_paths"]
    assert parallel_meta["stats"] == serial_meta["stats"]
    assert (parallel_out / "launch_rows.jsonl").read_text(encoding="utf-8") == (serial_out / "launch_rows.jsonl").read_text(encoding="utf-8")
    assert (parallel_out / "source_turn_rows.jsonl").read_text(encoding="utf-8") == (serial_out / "source_turn_rows.jsonl").read_text(encoding="utf-8")
    assert (parallel_out / "state_rows.jsonl").read_text(encoding="utf-8") == (serial_out / "state_rows.jsonl").read_text(encoding="utf-8")

    with np.load(serial_out / "dense_bc_arrays.npz") as serial_npz, np.load(parallel_out / "dense_bc_arrays.npz") as parallel_npz:
        for key in (
            "planet_features",
            "planet_features_v2",
            "global_features_v2",
            "target_state_features_v2",
            "target_labels",
            "amount_labels",
            "source_mask",
        ):
            assert np.array_equal(parallel_npz[key], serial_npz[key])
