from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from orbit_training_prep.features import PAIR_FEATURE_NAMES
from orbit_training_prep.materialize_splits import materialize_splits, source_turn_sample_weight
from orbit_training_prep.schema import NOOP_TARGET_SLOT
from orbit_training_prep.source_turn_store import SourceTurnDatasetReader, SourceTurnDatasetWriter
from orbit_training_prep.split_episodes import discover_episode_ids, make_episode_splits


def _write_source_turn_dataset(path: Path) -> None:
    writer = SourceTurnDatasetWriter(path)
    for episode_id, source_slot, target_label in (
        ("episode_a", 0, 1),
        ("episode_b", 1, NOOP_TARGET_SLOT),
    ):
        state_idx = writer.append_state(
            planet_features=np.zeros((64, 16), dtype=np.float32),
            global_features=np.zeros((10,), dtype=np.float32),
            target_state_features=np.zeros((64, 9), dtype=np.float32),
            episode_id=episode_id,
        )
        target_mask = np.zeros((65,), dtype=bool)
        amount_mask = np.zeros((7,), dtype=bool)
        target_mask[target_label] = True
        amount_label = 0 if target_label == NOOP_TARGET_SLOT else 3
        amount_mask[amount_label] = True
        writer.append_sample(
            state_index=state_idx,
            source_slot=source_slot,
            target_label=target_label,
            amount_label=amount_label,
            sample_weight=source_turn_sample_weight(
                {"target_slot_label": target_label, "winner_action": episode_id == "episode_a", "step_index": 5}
            ),
            step=5,
            pair_features=np.full((65, len(PAIR_FEATURE_NAMES)), source_slot, dtype=np.float32),
            target_mask=target_mask,
            amount_mask=amount_mask,
        )
    writer.finalize()


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

    def test_discover_episode_ids_reads_source_turn_memmap_state_episode_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_source_turn_dataset(dataset)

            self.assertEqual(discover_episode_ids(dataset), ["episode_a", "episode_b"])

    def test_materialize_splits_filters_source_turn_memmap_by_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            out = root / "combined"
            splits_path = root / "splits.json"
            splits_path.write_text(json.dumps({"train": ["episode_a"], "valid": ["episode_b"]}), encoding="utf-8")
            _write_source_turn_dataset(dataset)

            summary = materialize_splits(dataset_root=dataset, splits_path=splits_path, out_dir=out)

            self.assertEqual(summary["train"]["source_turn_samples"], 1)
            self.assertEqual(summary["valid"]["source_turn_samples"], 1)
            train_reader = SourceTurnDatasetReader(out / "train")
            valid_reader = SourceTurnDatasetReader(out / "valid")
            self.assertEqual([str(x) for x in train_reader.states["episode_id"].tolist()], ["episode_a"])
            self.assertEqual([str(x) for x in valid_reader.states["episode_id"].tolist()], ["episode_b"])
            self.assertEqual(int(train_reader.samples["target_label"][0]), 1)
            self.assertEqual(int(valid_reader.samples["target_label"][0]), NOOP_TARGET_SLOT)
            self.assertEqual(float(train_reader.samples["sample_weight"][0]), np.float32(1.25))
            self.assertEqual(float(valid_reader.samples["sample_weight"][0]), np.float32(0.2))
            self.assertFalse((out / "train" / "source_turn_rows.jsonl").exists())
            self.assertFalse((out / "valid" / "source_turn_rows.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
