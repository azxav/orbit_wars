from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orbit_training_prep.features import PAIR_FEATURE_NAMES
from orbit_training_prep.schema import NOOP_TARGET_SLOT
from orbit_training_prep.source_turn_store import SourceTurnDatasetWriter
from orbit_training_prep.training_io import load_dense_bc_arrays, load_source_turn_rows


def _write_dataset(path: Path) -> None:
    writer = SourceTurnDatasetWriter(path)
    state_idx = writer.append_state(
        planet_features=np.zeros((64, 16), dtype=np.float32),
        global_features=np.zeros((10,), dtype=np.float32),
        target_state_features=np.zeros((64, 9), dtype=np.float32),
        episode_id="episode_a",
    )
    target_mask = np.zeros((65,), dtype=bool)
    target_mask[NOOP_TARGET_SLOT] = True
    amount_mask = np.zeros((7,), dtype=bool)
    amount_mask[0] = True
    writer.append_sample(
        state_index=state_idx,
        source_slot=3,
        target_label=NOOP_TARGET_SLOT,
        amount_label=0,
        sample_weight=0.2,
        step=12,
        pair_features=np.zeros((65, len(PAIR_FEATURE_NAMES)), dtype=np.float32),
        target_mask=target_mask,
        amount_mask=amount_mask,
    )
    writer.finalize()


def test_load_source_turn_rows_reads_compact_memmap_samples(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    rows = load_source_turn_rows(tmp_path)

    assert rows == [
        {
            "source_turn_uid": "episode_a:0:s3",
            "episode_id": "episode_a",
            "state_index": 0,
            "source_slot": 3,
            "target_slot_label": NOOP_TARGET_SLOT,
            "amount_bin_label": 0,
            "sample_weight": np.float32(0.2),
            "step_index": 12,
            "winner_action": False,
        }
    ]


def test_load_dense_bc_arrays_rejects_removed_format(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="dense_bc_arrays.npz has been removed"):
        load_dense_bc_arrays(tmp_path)
