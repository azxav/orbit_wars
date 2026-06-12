from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from orbit_training_prep.dataset_builder import DatasetBuilder, _balance_source_turn_dataset, _select_balanced_replay_paths
from orbit_training_prep.features import PAIR_FEATURE_NAMES
from orbit_training_prep.schema import AMOUNT_BIN_ALL, AMOUNT_BIN_CAPTURE, NOOP_TARGET_SLOT
from orbit_training_prep.source_turn_store import SourceTurnDatasetWriter


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


def write_player_count_replay(path: Path, episode_id: str, players: int) -> None:
    replay = {
        "info": {"EpisodeId": episode_id},
        "configuration": {"episodeSteps": 2},
        "rewards": [0.0 for _ in range(players)],
        "steps": [
            [{"observation": {}, "status": "ACTIVE", "reward": 0} for _ in range(players)],
            [{"observation": {}, "action": [], "status": "DONE", "reward": 0} for _ in range(players)],
        ],
    }
    path.write_text(json.dumps(replay), encoding="utf-8")


def write_balance_source_turn_dataset(path: Path, *, noop_rows: int, amount_bins: list[int]) -> dict:
    writer = SourceTurnDatasetWriter(path)
    target_mask = np.zeros((65,), dtype=bool)
    target_mask[NOOP_TARGET_SLOT] = True
    target_mask[1] = True
    noop_amount_mask = np.zeros((7,), dtype=bool)
    noop_amount_mask[0] = True
    op_amount_mask = np.ones((7,), dtype=bool)
    for i in range(noop_rows + len(amount_bins)):
        state_idx = writer.append_state(
            planet_features=np.zeros((64, 16), dtype=np.float32),
            global_features=np.zeros((10,), dtype=np.float32),
            target_state_features=np.zeros((64, 9), dtype=np.float32),
            episode_id=f"episode_{i}",
        )
        is_noop = i < noop_rows
        writer.append_sample(
            state_index=state_idx,
            source_slot=0,
            target_label=NOOP_TARGET_SLOT if is_noop else 1,
            amount_label=0 if is_noop else amount_bins[i - noop_rows],
            sample_weight=1.0,
            step=i,
            pair_features=np.zeros((65, len(PAIR_FEATURE_NAMES)), dtype=np.float32),
            target_mask=target_mask,
            amount_mask=noop_amount_mask if is_noop else op_amount_mask,
        )
    return writer.finalize(
        extra_metadata={
            "stats": {
                "raw_source_turns": noop_rows + len(amount_bins),
                "train_source_turns": noop_rows + len(amount_bins),
                "raw_positive_source_turns": len(amount_bins),
                "train_positive_source_turns": len(amount_bins),
                "noop_source_turns": noop_rows,
                "source_turn_rows": noop_rows + len(amount_bins),
                "positive_source_turns": len(amount_bins),
            }
        }
    )


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
            self.assertTrue((out_dir / "samples" / "pair_features.npy").exists())
            self.assertFalse((out_dir / "dense_bc_arrays.npz").exists())
            self.assertEqual(metadata["dataset_format"], "source_turn_memmap_v1")

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

    def test_balanced_replay_selection_targets_even_two_and_four_player_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths: list[Path] = []
            for i in range(4):
                path = root / f"two_{i}.json"
                write_player_count_replay(path, f"two-{i}", 2)
                paths.append(path)
            for i in range(2):
                path = root / f"four_{i}.json"
                write_player_count_replay(path, f"four-{i}", 4)
                paths.append(path)

            selected, report = _select_balanced_replay_paths(sorted(paths), max_files=4, balance_seed=7)
            selected_again, report_again = _select_balanced_replay_paths(sorted(paths), max_files=4, balance_seed=7)

            self.assertEqual(selected, selected_again)
            self.assertEqual(report, report_again)
            self.assertEqual(len(selected), 4)
            self.assertEqual(report["selected_player_counts"], {"2": 2, "4": 2})
            self.assertFalse(report["infeasible"])

    def test_balanced_replay_selection_records_infeasible_unequal_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths: list[Path] = []
            for i in range(4):
                path = root / f"two_{i}.json"
                write_player_count_replay(path, f"two-{i}", 2)
                paths.append(path)
            path = root / "four_0.json"
            write_player_count_replay(path, "four-0", 4)
            paths.append(path)

            selected, report = _select_balanced_replay_paths(sorted(paths), max_files=4, balance_seed=7)

            self.assertEqual(len(selected), 2)
            self.assertEqual(report["selected_player_counts"], {"2": 1, "4": 1})
            self.assertTrue(report["infeasible"])

    def test_source_turn_balancing_reaches_noop_op_ratio_when_feasible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = write_balance_source_turn_dataset(
                root,
                noop_rows=30,
                amount_bins=[AMOUNT_BIN_ALL for _ in range(15)],
            )

            balanced = _balance_source_turn_dataset(
                root,
                metadata,
                balance_seed=11,
                noop_ratio=0.4,
                op_ratio=0.6,
            )
            target_labels = np.load(root / "samples" / "target_label.npy", allow_pickle=False)

            self.assertEqual(int(target_labels.shape[0]), 25)
            self.assertEqual(int(np.sum(target_labels == NOOP_TARGET_SLOT)), 10)
            self.assertEqual(int(np.sum(target_labels != NOOP_TARGET_SLOT)), 15)
            self.assertEqual(balanced["stats"]["source_turn_rows"], 25)
            self.assertEqual(balanced["stats"]["noop_source_turns"], 10)
            self.assertEqual(balanced["stats"]["positive_source_turns"], 15)

    def test_source_turn_balancing_softens_dominant_amount_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = write_balance_source_turn_dataset(
                root,
                noop_rows=10,
                amount_bins=[AMOUNT_BIN_ALL for _ in range(16)] + [AMOUNT_BIN_CAPTURE for _ in range(4)],
            )

            balanced = _balance_source_turn_dataset(
                root,
                metadata,
                balance_seed=11,
                noop_ratio=0.4,
                op_ratio=0.6,
            )
            amount_labels = np.load(root / "samples" / "amount_label.npy", allow_pickle=False)
            target_labels = np.load(root / "samples" / "target_label.npy", allow_pickle=False)
            op_amounts = amount_labels[target_labels != NOOP_TARGET_SLOT]

            self.assertLess(int(np.sum(op_amounts == AMOUNT_BIN_ALL)), 16)
            self.assertEqual(int(np.sum(op_amounts == AMOUNT_BIN_CAPTURE)), 4)
            self.assertEqual(balanced["amount_bin_counts"]["all"], int(np.sum(op_amounts == AMOUNT_BIN_ALL)))
            self.assertEqual(balanced["amount_bin_counts"]["capture_plus_one"], 4)

    def test_source_turn_balancing_keeps_single_class_and_marks_infeasible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = write_balance_source_turn_dataset(root, noop_rows=5, amount_bins=[])

            balanced = _balance_source_turn_dataset(
                root,
                metadata,
                balance_seed=11,
                noop_ratio=0.4,
                op_ratio=0.6,
            )

            self.assertEqual(balanced["sample_count"], 5)
            self.assertEqual(balanced["proportion_correction"]["sample_balance"]["selected"]["noop"], 5)
            self.assertTrue(balanced["proportion_correction"]["sample_balance"]["infeasible"])

    def test_skips_invalid_json_replay_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_dir = root / "replays"
            replay_dir.mkdir()
            write_minimal_replay(replay_dir / "good.json", "episode-good")
            (replay_dir / "bad.json").write_text("", encoding="utf-8")

            out_dir = root / "dataset"
            metadata = DatasetBuilder(horizon=8).build_from_replay(replay_dir, out_dir)

            self.assertEqual(metadata["replay_paths"], [str(replay_dir / "good.json")])
            self.assertEqual(metadata["input_replay_files"]["requested"], 2)
            self.assertEqual(metadata["input_replay_files"]["used"], 1)
            self.assertEqual(metadata["input_replay_files"]["skipped_invalid_json"], 1)
            self.assertEqual(metadata["stats"]["states"], 1)
            self.assertEqual(metadata["stats"]["source_turn_rows"], 1)


if __name__ == "__main__":
    unittest.main()
