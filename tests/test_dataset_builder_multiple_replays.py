from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orbit_training_prep.dataset_builder import DatasetBuilder


def write_minimal_replay(path: Path, episode_id: str) -> None:
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


class DatasetBuilderMultipleReplaysTest(unittest.TestCase):
    def test_builds_one_dataset_from_replay_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_dir = root / "replays"
            replay_dir.mkdir()
            write_minimal_replay(replay_dir / "b.json", "episode-b")
            write_minimal_replay(replay_dir / "a.json", "episode-a")
            (replay_dir / "replay_metadata.json").write_text("{}", encoding="utf-8")

            out_dir = root / "dataset"
            metadata = DatasetBuilder(horizon=8).build_from_replay(replay_dir, out_dir)

            self.assertEqual(
                metadata["replay_paths"],
                [str(replay_dir / "a.json"), str(replay_dir / "b.json")],
            )
            self.assertEqual(metadata["stats"]["states"], 2)
            self.assertEqual(metadata["stats"]["source_turn_rows"], 2)
            self.assertTrue((out_dir / "source_turn_rows.jsonl").exists())

    def test_limits_replay_directory_to_requested_file_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_dir = root / "replays"
            replay_dir.mkdir()
            write_minimal_replay(replay_dir / "b.json", "episode-b")
            write_minimal_replay(replay_dir / "a.json", "episode-a")

            out_dir = root / "dataset"
            metadata = DatasetBuilder(horizon=8, max_replay_files=1).build_from_replay(replay_dir, out_dir)

            self.assertEqual(metadata["replay_paths"], [str(replay_dir / "a.json")])
            self.assertEqual(metadata["stats"]["states"], 1)
            self.assertEqual(metadata["stats"]["source_turn_rows"], 1)


if __name__ == "__main__":
    unittest.main()
