from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orbit_training_prep.materialize_splits import materialize_splits, source_turn_sample_weight
from orbit_training_prep.schema import NOOP_TARGET_SLOT
from orbit_training_prep.split_episodes import make_episode_splits


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class SplitMaterializeTest(unittest.TestCase):
    def test_make_episode_splits_is_deterministic_and_keeps_episode_ids_whole(self) -> None:
        splits = make_episode_splits(["episode_1", "episode_2", "episode_3", "episode_4"], valid_frac=0.25, seed=42)

        self.assertEqual(set(splits), {"train", "valid"})
        self.assertEqual(len(splits["valid"]), 1)
        self.assertEqual(set(splits["train"]) | set(splits["valid"]), {"episode_1", "episode_2", "episode_3", "episode_4"})
        self.assertEqual(set(splits["train"]) & set(splits["valid"]), set())
        self.assertEqual(splits, make_episode_splits(["episode_4", "episode_3", "episode_2", "episode_1"], valid_frac=0.25, seed=42))

    def test_source_turn_sample_weight_matches_contract(self) -> None:
        self.assertEqual(source_turn_sample_weight({"target_slot_label": NOOP_TARGET_SLOT, "winner_action": False, "step_index": 12}), 0.2)
        self.assertEqual(source_turn_sample_weight({"target_slot_label": 3, "winner_action": False, "step_index": 12}), 1.0)
        self.assertEqual(source_turn_sample_weight({"target_slot_label": 3, "winner_action": True, "step_index": 12}), 1.25)
        self.assertEqual(source_turn_sample_weight({"target_slot_label": 3, "winner_action": True, "step_index": 431}), 0.625)

    def test_materialize_splits_filters_by_episode_and_adds_sample_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            out = root / "combined"
            splits_path = root / "splits.json"
            splits_path.write_text(json.dumps({"train": ["episode_a"], "valid": ["episode_b"]}), encoding="utf-8")
            write_jsonl(
                dataset / "source_turn_rows.jsonl",
                [
                    {"source_turn_uid": "a", "episode_id": "episode_a", "target_slot_label": 1, "winner_action": True, "step_index": 5},
                    {"source_turn_uid": "b", "episode_id": "episode_b", "target_slot_label": NOOP_TARGET_SLOT, "winner_action": False, "step_index": 5},
                ],
            )
            write_jsonl(
                dataset / "pair_rank_rows.jsonl",
                [
                    {"pair_uid": "a0", "source_turn_uid": "a", "episode_id": "episode_a", "label": 1, "features": [1.0]},
                    {"pair_uid": "a1", "source_turn_uid": "a", "episode_id": "episode_a", "label": 0, "features": [0.0]},
                    {"pair_uid": "b0", "episode_id": "episode_b", "label": 1, "features": [0.0]},
                ],
            )

            summary = materialize_splits(dataset_root=dataset, splits_path=splits_path, out_dir=out)

            self.assertEqual(summary["train"]["source_turn_rows"], 1)
            train_source = (out / "train" / "source_turn_rows.jsonl").read_text(encoding="utf-8").splitlines()
            valid_source = (out / "valid" / "source_turn_rows.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(train_source[0])["episode_id"], "episode_a")
            self.assertEqual(json.loads(train_source[0])["sample_weight"], 1.25)
            self.assertEqual(json.loads(valid_source[0])["sample_weight"], 0.2)
            self.assertEqual(summary["train"]["pair_rank_rows"], 2)
            self.assertEqual(json.loads((out / "valid" / "pair_rank_rows.jsonl").read_text(encoding="utf-8").strip())["episode_id"], "episode_b")


if __name__ == "__main__":
    unittest.main()
