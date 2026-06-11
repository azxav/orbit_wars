from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from orbit_training_prep.features import PAIR_FEATURE_NAMES
from orbit_training_prep.schema import NOOP_TARGET_SLOT


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_minimal_dense(path: Path) -> None:
    target_viability_mask = np.zeros((1, 64, 65), dtype=bool)
    amount_viability_mask = np.zeros((1, 64, 65, 7), dtype=bool)
    target_viability_mask[:, :, NOOP_TARGET_SLOT] = True
    amount_viability_mask[:, :, NOOP_TARGET_SLOT, 0] = True
    target_viability_mask[0, 0, 1] = True
    amount_viability_mask[0, 0, 1, 3] = True
    np.savez_compressed(
        path / "dense_bc_arrays.npz",
        planet_features=np.zeros((1, 64, 16), dtype=np.float32),
        global_features=np.zeros((1, 10), dtype=np.float32),
        target_state_features=np.zeros((1, 64, 9), dtype=np.float32),
        pair_features=np.zeros((1, 64, 65, len(PAIR_FEATURE_NAMES)), dtype=np.float32),
        target_labels=np.full((1, 64), NOOP_TARGET_SLOT, dtype=np.int64),
        amount_labels=np.zeros((1, 64), dtype=np.int64),
        source_mask=np.zeros((1, 64), dtype=np.float32),
        target_viability_mask=target_viability_mask,
        amount_viability_mask=amount_viability_mask,
    )
    _write_jsonl(path / "state_rows.jsonl", [{"obs_uid": "e0:0:p0"}])


def test_validate_dataset_fails_for_invalid_bc_rows(tmp_path: Path) -> None:
    from orbit_training_prep.validate_dataset import validate_dataset

    _write_minimal_dense(tmp_path)
    _write_jsonl(
        tmp_path / "source_turn_rows.jsonl",
        [
            {
                "obs_uid": "e0:0:p0",
                "source_slot": 0,
                "target_slot_label": 1,
                "amount_bin_label": 3,
                "drop_for_v1_bc": True,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="BC-invalid source rows"):
        validate_dataset(tmp_path)


def test_validate_dataset_fails_when_labels_are_not_in_viability_masks(tmp_path: Path) -> None:
    from orbit_training_prep.validate_dataset import validate_dataset

    _write_minimal_dense(tmp_path)
    _write_jsonl(
        tmp_path / "source_turn_rows.jsonl",
        [
            {
                "obs_uid": "e0:0:p0",
                "source_slot": 0,
                "target_slot_label": 2,
                "amount_bin_label": 3,
                "drop_for_v1_bc": False,
                "geometry_viable": True,
                "target_inference_method": "first_contact",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="target labels outside target_viability_mask"):
        validate_dataset(tmp_path)
