from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbit_training_prep.features import PAIR_FEATURE_NAMES
from orbit_training_prep.schema import NOOP_TARGET_SLOT
from orbit_training_prep.source_turn_store import SourceTurnDatasetReader, SourceTurnDatasetWriter


def test_source_turn_store_writes_compact_memmap_contract(tmp_path: Path) -> None:
    writer = SourceTurnDatasetWriter(tmp_path)
    state_idx = writer.append_state(
        planet_features=np.zeros((64, 16), dtype=np.float32),
        global_features=np.zeros((10,), dtype=np.float32),
        target_state_features=np.zeros((64, 9), dtype=np.float32),
        episode_id="episode_a",
    )
    writer.append_sample(
        state_index=state_idx,
        source_slot=2,
        target_label=NOOP_TARGET_SLOT,
        amount_label=0,
        sample_weight=0.25,
        step=12,
        pair_features=np.ones((65, len(PAIR_FEATURE_NAMES)), dtype=np.float32),
        target_mask=np.eye(1, 65, NOOP_TARGET_SLOT, dtype=bool)[0],
        amount_mask=np.eye(1, 7, 0, dtype=bool)[0],
    )
    metadata = writer.finalize(extra_metadata={"test": True})

    assert metadata["dataset_format"] == "source_turn_memmap_v1"
    assert metadata["state_count"] == 1
    assert metadata["sample_count"] == 1

    reader = SourceTurnDatasetReader(tmp_path)
    assert reader.states["planet_features"].shape == (1, 64, 16)
    assert reader.states["episode_id"].shape == (1,)
    assert str(reader.states["episode_id"][0]) == "episode_a"
    assert reader.samples["pair_features"].shape == (1, 65, len(PAIR_FEATURE_NAMES))
    assert reader.samples["pair_features"].dtype == np.float16
    assert reader.samples["source_slot"].dtype == np.uint8
    assert reader.samples["target_label"].dtype == np.uint8
    assert reader.samples["amount_label"].dtype == np.uint8
    assert reader.samples["state_index"].dtype == np.uint32
    assert reader.samples["step"].dtype == np.uint16
    assert reader.samples["target_mask"].dtype == bool
    assert reader.samples["amount_mask"].dtype == bool

    metadata_on_disk = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata_on_disk["pair_feature_names"] == list(PAIR_FEATURE_NAMES)


def test_source_turn_store_flushes_chunks_before_finalize(tmp_path: Path) -> None:
    writer = SourceTurnDatasetWriter(tmp_path, chunk_size=1)
    for i in range(2):
        state_idx = writer.append_state(
            planet_features=np.zeros((64, 16), dtype=np.float32),
            global_features=np.zeros((10,), dtype=np.float32),
            target_state_features=np.zeros((64, 9), dtype=np.float32),
            episode_id=f"episode_{i}",
        )
        writer.append_sample(
            state_index=state_idx,
            source_slot=i,
            target_label=NOOP_TARGET_SLOT,
            amount_label=0,
            sample_weight=0.2,
            step=i,
            pair_features=np.zeros((65, len(PAIR_FEATURE_NAMES)), dtype=np.float32),
            target_mask=np.eye(1, 65, NOOP_TARGET_SLOT, dtype=bool)[0],
            amount_mask=np.eye(1, 7, 0, dtype=bool)[0],
        )

    writer.finalize()

    reader = SourceTurnDatasetReader(tmp_path)
    assert reader.metadata["state_count"] == 2
    assert reader.metadata["sample_count"] == 2
    assert [str(x) for x in reader.states["episode_id"].tolist()] == ["episode_0", "episode_1"]
    assert not (tmp_path / ".source_turn_chunks").exists()
