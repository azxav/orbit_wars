from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbit_training_prep import dataset_builder
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


def test_dataset_builder_streams_rows_instead_of_buffering_until_finish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_path = tmp_path / "replay.json"
    _write_minimal_replay(replay_path, "streaming-episode")

    def fail_if_bulk_writer_is_used(path: Path, rows: list[dict]) -> None:
        raise AssertionError(f"bulk write_jsonl called for {path} with {len(rows)} buffered rows")

    monkeypatch.setattr(dataset_builder, "write_jsonl", fail_if_bulk_writer_is_used)

    metadata = DatasetBuilder(horizon=8, device="cpu").build_from_replay(replay_path, tmp_path / "dataset")

    assert metadata["stats"]["states"] == 1
    assert metadata["stats"]["source_turn_rows"] == 1
    assert (tmp_path / "dataset" / "source_turn_rows.jsonl").read_text(encoding="utf-8").strip()
