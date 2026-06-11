from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from orbit_training_prep.features import PAIR_FEATURE_NAMES
from orbit_training_prep.schema import NOOP_TARGET_SLOT
from orbit_training_prep.source_turn_store import DATASET_FORMAT, SourceTurnDatasetReader

NOT_BC_READY_MESSAGE = "Dataset is not BC-ready. Rebuild with the updated source-turn dataset_builder."
OLD_DATASET_MESSAGE = "Old dense_bc_arrays.npz dataset detected. Rebuild dataset with source_turn_memmap_v1."


class OrbitBCDataset(Dataset):
    """One sample per owned source planet using compact source-turn memmap features."""

    noop_target_slot = NOOP_TARGET_SLOT

    def __init__(self, dataset_dir: str | Path, *, allow_filter_invalid_rows: bool = False):
        del allow_filter_invalid_rows
        self.dataset_dir = Path(dataset_dir)
        if (self.dataset_dir / "dense_bc_arrays.npz").exists():
            raise RuntimeError(OLD_DATASET_MESSAGE)
        self.reader = SourceTurnDatasetReader(self.dataset_dir)
        if self.reader.metadata.get("dataset_format") != DATASET_FORMAT:
            raise RuntimeError(NOT_BC_READY_MESSAGE)
        self.states = self.reader.states
        self.samples = self.reader.samples
        self.planet_feature_dim = int(self.states["planet_features"].shape[-1])
        self.global_feature_dim = int(self.states["global_features"].shape[-1])
        self.target_state_feature_dim = int(self.states["target_state_features"].shape[-1])
        self.pair_feature_dim = len(PAIR_FEATURE_NAMES)
        if int(self.samples["pair_features"].shape[-1]) != self.pair_feature_dim:
            raise RuntimeError("pair_features shape does not match current compact feature contract")

    def __len__(self) -> int:
        return int(self.samples["state_index"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        state_idx = int(self.samples["state_index"][idx])
        source_slot = int(self.samples["source_slot"][idx])
        target_label = int(self.samples["target_label"][idx])
        amount_label = int(self.samples["amount_label"][idx])
        is_noop = target_label == NOOP_TARGET_SLOT
        if is_noop:
            amount_label = 0
        target_mask = np.asarray(self.samples["target_mask"][idx], dtype=bool).copy()
        amount_mask = np.asarray(self.samples["amount_mask"][idx], dtype=bool).copy()
        if not bool(target_mask[target_label]):
            raise RuntimeError(
                f"BC sample {idx!r} target label {target_label} is outside the geometry/capture viability mask for source {source_slot}."
            )
        if not bool(amount_mask[amount_label]):
            raise RuntimeError(
                f"BC sample {idx!r} amount label {amount_label} is outside the geometry/capture viability mask for source {source_slot}, target {target_label}."
            )
        return {
            "planet_features": np.asarray(self.states["planet_features"][state_idx], dtype=np.float32),
            "fleet_features": np.zeros((0, 0), dtype=np.float32),
            "global_features": np.asarray(self.states["global_features"][state_idx], dtype=np.float32),
            "target_state_features": np.asarray(self.states["target_state_features"][state_idx], dtype=np.float32),
            "pair_features": np.asarray(self.samples["pair_features"][idx], dtype=np.float32),
            "source_slot": np.int64(source_slot),
            "target_label": np.int64(target_label),
            "amount_label": np.int64(amount_label),
            "target_mask": target_mask,
            "amount_mask": amount_mask,
            "sample_weight": np.float32(self.samples["sample_weight"][idx]),
            "is_noop": bool(is_noop),
            "episode_id": str(self.states.get("episode_id", np.asarray([""], dtype="<U1"))[state_idx]),
            "step": np.int64(self.samples["step"][idx]),
        }


def collate_bc_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = {
        "planet_features": torch.float32,
        "global_features": torch.float32,
        "target_state_features": torch.float32,
        "pair_features": torch.float32,
        "source_slot": torch.long,
        "target_label": torch.long,
        "amount_label": torch.long,
        "target_mask": torch.bool,
        "amount_mask": torch.bool,
        "sample_weight": torch.float32,
        "is_noop": torch.bool,
        "step": torch.long,
    }
    batch: dict[str, Any] = {}
    for key, dtype in tensor_keys.items():
        batch[key] = torch.as_tensor(np.stack([s[key] for s in samples]), dtype=dtype)
    batch["episode_id"] = [s["episode_id"] for s in samples]
    return batch
