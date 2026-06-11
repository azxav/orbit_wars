from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orbit_training_prep.features import PAIR_FEATURE_NAMES
from orbit_training_prep.schema import NOOP_TARGET_SLOT
from orbit_training_prep.source_turn_store import SourceTurnDatasetWriter


def _write_minimal_source_turn_dataset(path: Path) -> None:
    writer = SourceTurnDatasetWriter(path)
    state_idx = writer.append_state(
        planet_features=np.zeros((64, 16), dtype=np.float32),
        global_features=np.zeros((10,), dtype=np.float32),
        target_state_features=np.zeros((64, 9), dtype=np.float32),
    )
    target_mask = np.zeros((65,), dtype=bool)
    amount_mask = np.zeros((7,), dtype=bool)
    target_mask[NOOP_TARGET_SLOT] = True
    target_mask[1] = True
    amount_mask[3] = True
    writer.append_sample(
        state_index=state_idx,
        source_slot=0,
        target_label=1,
        amount_label=3,
        sample_weight=1.0,
        step=0,
        pair_features=np.zeros((65, len(PAIR_FEATURE_NAMES)), dtype=np.float32),
        target_mask=target_mask,
        amount_mask=amount_mask,
    )
    writer.finalize(extra_metadata={"stats": {"states": 1, "source_turn_rows": 1}})


def test_validate_dataset_accepts_source_turn_memmap_dataset(tmp_path: Path) -> None:
    from orbit_training_prep.validate_dataset import validate_dataset

    _write_minimal_source_turn_dataset(tmp_path)

    report = validate_dataset(tmp_path)

    assert report["decision_checks"]["dataset_format"] == "source_turn_memmap_v1"
    assert report["decision_checks"]["pair_feature_dim"] == len(PAIR_FEATURE_NAMES)
    assert report["dense_array_shapes"] == {}
    assert report["source_turn_array_shapes"]["pair_features"] == [1, 65, len(PAIR_FEATURE_NAMES)]


def test_validate_dataset_fails_when_labels_are_not_in_sample_masks(tmp_path: Path) -> None:
    from orbit_training_prep.validate_dataset import validate_dataset

    _write_minimal_source_turn_dataset(tmp_path)
    target_label = np.load(tmp_path / "samples" / "target_label.npy")
    target_label[0] = 2
    np.save(tmp_path / "samples" / "target_label.npy", target_label, allow_pickle=False)

    with pytest.raises(RuntimeError, match="target labels outside target_mask"):
        validate_dataset(tmp_path)


def test_validate_dataset_rejects_old_dense_dataset(tmp_path: Path) -> None:
    from orbit_training_prep.validate_dataset import validate_dataset

    np.savez_compressed(tmp_path / "dense_bc_arrays.npz", pair_features=np.zeros((1, 64, 65, 30), dtype=np.float32))

    with pytest.raises(RuntimeError, match="Old dense_bc_arrays.npz dataset detected"):
        validate_dataset(tmp_path)
