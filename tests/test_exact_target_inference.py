from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from orbit_training_prep.dataset_builder import DatasetBuilder, read_jsonl
from orbit_training_prep.target_inference import TargetInferer


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


def replay_with_action(path: Path, obs: dict, action: list[list[float]]) -> None:
    replay = {
        "info": {"EpisodeId": "exact-target-test"},
        "configuration": {"episodeSteps": 2},
        "rewards": [1.0],
        "steps": [
            [{"observation": obs, "status": "ACTIVE", "reward": 0}],
            [{"observation": {}, "action": action, "status": "DONE", "reward": 1}],
        ],
    }
    path.write_text(json.dumps(replay), encoding="utf-8")


class ExactTargetInferenceTest(unittest.TestCase):
    def test_raw_launch_label_uses_first_planet_hit_before_angle_nearest_target(self) -> None:
        obs = static_obs(
            [
                [1, 0, 10.0, 50.0, 1.0, 50.0, 1.0],
                [2, -1, 30.0, 52.0, 3.0, 5.0, 1.0],
                [3, -1, 80.0, 50.0, 2.0, 5.0, 1.0],
            ]
        )

        move = TargetInferer(horizon=80).infer_move(obs, 0, (1, 0.0, 10))

        self.assertEqual(move.inferred_target_id, 2)
        self.assertEqual(move.contact_target_id, 2)
        self.assertEqual(move.target_inference_method, "first_contact")
        self.assertTrue(move.geometry_viable)

    def test_raw_launch_falls_back_to_angle_nearest_when_first_contact_is_bounds(self) -> None:
        obs = static_obs(
            [
                [1, 0, 10.0, 50.0, 1.0, 50.0, 1.0],
                [2, -1, 80.0, 50.0, 2.0, 5.0, 1.0],
            ]
        )

        move = TargetInferer(horizon=80).infer_move(obs, 0, (1, math.pi, 10))

        self.assertEqual(move.inferred_target_id, 2)
        self.assertEqual(move.contact_target_id, -999)
        self.assertEqual(move.target_inference_method, "angular_nearest")
        self.assertFalse(move.geometry_viable)

    def test_dataset_builder_emits_exact_first_hit_label(self) -> None:
        obs = static_obs(
            [
                [1, 0, 10.0, 50.0, 1.0, 50.0, 1.0],
                [2, -1, 30.0, 52.0, 3.0, 5.0, 1.0],
                [3, -1, 80.0, 50.0, 2.0, 5.0, 1.0],
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_path = root / "replay.json"
            out_dir = root / "dataset"
            replay_with_action(replay_path, obs, [[1, 0.0, 10]])

            metadata = DatasetBuilder(horizon=80).build_from_replay(replay_path, out_dir)
            launch_row = read_jsonl(out_dir / "launch_rows.jsonl")[0]
            source_row = read_jsonl(out_dir / "source_turn_rows.jsonl")[0]

        self.assertEqual(metadata["target_inference_mode"], "exact_first_hit_with_angular_fallback")
        self.assertEqual(launch_row["inferred_target_id"], 2)
        self.assertEqual(launch_row["target_inference_method"], "first_contact")
        self.assertEqual(source_row["target_planet_id_label"], 2)
        self.assertEqual(source_row["target_inference_method"], "first_contact")


if __name__ == "__main__":
    unittest.main()
