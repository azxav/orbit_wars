from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .features import GLOBAL_FEATURE_NAMES, PAIR_FEATURE_NAMES, PLANET_FEATURE_NAMES, TARGET_STATE_FEATURE_NAMES
from .schema import AMOUNT_BIN_NAMES, NOOP_TARGET_SLOT, P_MAX, ActionSpaceSpec

DATASET_FORMAT = "source_turn_memmap_v1"

STATE_SPECS = {
    "planet_features": (np.float32, (P_MAX, len(PLANET_FEATURE_NAMES))),
    "global_features": (np.float32, (len(GLOBAL_FEATURE_NAMES),)),
    "target_state_features": (np.float32, (P_MAX, len(TARGET_STATE_FEATURE_NAMES))),
    "episode_id": (np.dtype("<U128"), ()),
}

SAMPLE_SPECS = {
    "state_index": (np.uint32, ()),
    "source_slot": (np.uint8, ()),
    "target_label": (np.uint8, ()),
    "amount_label": (np.uint8, ()),
    "sample_weight": (np.float32, ()),
    "step": (np.uint16, ()),
    "pair_features": (np.float16, (P_MAX + 1, len(PAIR_FEATURE_NAMES))),
    "target_mask": (bool, (P_MAX + 1,)),
    "amount_mask": (bool, (len(AMOUNT_BIN_NAMES),)),
}


class SourceTurnDatasetWriter:
    def __init__(self, root: str | Path, *, chunk_size: int = 4096):
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be at least 1")
        self.root = Path(root)
        self.states_dir = self.root / "states"
        self.samples_dir = self.root / "samples"
        self.chunks_dir = self.root / ".source_turn_chunks"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = int(chunk_size)
        self._states: dict[str, list[np.ndarray | np.generic | str]] = {k: [] for k in STATE_SPECS}
        self._samples: dict[str, list[np.ndarray | np.generic]] = {k: [] for k in SAMPLE_SPECS}
        self._flushed_state_count = 0
        self._flushed_sample_count = 0
        self._state_chunk_index = 0
        self._sample_chunk_index = 0

    @property
    def state_count(self) -> int:
        return int(self._flushed_state_count + len(self._states["planet_features"]))

    @property
    def sample_count(self) -> int:
        return int(self._flushed_sample_count + len(self._samples["state_index"]))

    def append_state(
        self,
        *,
        planet_features: np.ndarray,
        global_features: np.ndarray,
        target_state_features: np.ndarray,
        episode_id: str = "",
    ) -> int:
        state_index = self.state_count
        values = {
            "planet_features": np.asarray(planet_features, dtype=np.float32),
            "global_features": np.asarray(global_features, dtype=np.float32),
            "target_state_features": np.asarray(target_state_features, dtype=np.float32),
            "episode_id": str(episode_id),
        }
        for key, (_, shape) in STATE_SPECS.items():
            arr = np.asarray(values[key])
            if shape and arr.shape != shape:
                raise ValueError(f"{key} shape {arr.shape} does not match expected {shape}")
            self._states[key].append(values[key])
        if len(self._states["planet_features"]) >= self.chunk_size:
            self._flush_states()
        return state_index

    def append_sample(
        self,
        *,
        state_index: int,
        source_slot: int,
        target_label: int,
        amount_label: int,
        sample_weight: float,
        step: int,
        pair_features: np.ndarray,
        target_mask: np.ndarray,
        amount_mask: np.ndarray,
    ) -> int:
        sample_index = self.sample_count
        values = {
            "state_index": np.uint32(state_index),
            "source_slot": np.uint8(source_slot),
            "target_label": np.uint8(target_label),
            "amount_label": np.uint8(amount_label),
            "sample_weight": np.float32(sample_weight),
            "step": np.uint16(max(0, min(int(step), np.iinfo(np.uint16).max))),
            "pair_features": np.asarray(pair_features, dtype=np.float16),
            "target_mask": np.asarray(target_mask, dtype=bool),
            "amount_mask": np.asarray(amount_mask, dtype=bool),
        }
        for key, (_, shape) in SAMPLE_SPECS.items():
            arr = np.asarray(values[key])
            if shape and arr.shape != shape:
                raise ValueError(f"{key} shape {arr.shape} does not match expected {shape}")
            self._samples[key].append(values[key])
        if len(self._samples["state_index"]) >= self.chunk_size:
            self._flush_samples()
        return sample_index

    def _write_chunk(
        self,
        *,
        group_name: str,
        group: dict[str, list[Any]],
        specs: dict[str, tuple[Any, tuple[int, ...]]],
        chunk_index: int,
    ) -> int:
        row_count = len(next(iter(group.values())))
        if row_count == 0:
            return 0
        out_dir = self.chunks_dir / group_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, (dtype, _) in specs.items():
            np.save(out_dir / f"{key}_{chunk_index:06d}.npy", np.asarray(group[key], dtype=dtype), allow_pickle=False)
            group[key].clear()
        return row_count

    def _flush_states(self) -> None:
        written = self._write_chunk(
            group_name="states",
            group=self._states,
            specs=STATE_SPECS,
            chunk_index=self._state_chunk_index,
        )
        if written:
            self._flushed_state_count += written
            self._state_chunk_index += 1

    def _flush_samples(self) -> None:
        written = self._write_chunk(
            group_name="samples",
            group=self._samples,
            specs=SAMPLE_SPECS,
            chunk_index=self._sample_chunk_index,
        )
        if written:
            self._flushed_sample_count += written
            self._sample_chunk_index += 1

    def _merge_group(
        self,
        *,
        group_name: str,
        specs: dict[str, tuple[Any, tuple[int, ...]]],
        out_dir: Path,
        total_count: int,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir = self.chunks_dir / group_name
        for key, (dtype, tail_shape) in specs.items():
            out_path = out_dir / f"{key}.npy"
            shape = (int(total_count), *tail_shape)
            arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=shape)
            offset = 0
            for chunk_path in sorted(chunk_dir.glob(f"{key}_*.npy")) if chunk_dir.exists() else []:
                chunk = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
                end = offset + int(chunk.shape[0])
                arr[offset:end] = chunk
                offset = end
            if offset != int(total_count):
                raise RuntimeError(f"{group_name}/{key} chunk row mismatch: expected {total_count}, wrote {offset}")
            arr.flush()

    def finalize(self, *, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self._flush_states()
        self._flush_samples()
        state_count = int(self._flushed_state_count)
        sample_count = int(self._flushed_sample_count)
        self._merge_group(group_name="states", specs=STATE_SPECS, out_dir=self.states_dir, total_count=state_count)
        self._merge_group(group_name="samples", specs=SAMPLE_SPECS, out_dir=self.samples_dir, total_count=sample_count)
        shutil.rmtree(self.chunks_dir, ignore_errors=True)
        metadata: dict[str, Any] = {
            "dataset_format": DATASET_FORMAT,
            "state_count": state_count,
            "sample_count": sample_count,
            "action_space": ActionSpaceSpec().as_dict(),
            "noop_target_slot": int(NOOP_TARGET_SLOT),
            "planet_feature_names": list(PLANET_FEATURE_NAMES),
            "global_feature_names": list(GLOBAL_FEATURE_NAMES),
            "target_state_feature_names": list(TARGET_STATE_FEATURE_NAMES),
            "pair_feature_names": list(PAIR_FEATURE_NAMES),
            "pair_feature_dtype": "float16",
            "files": {
                "states": {k: f"states/{k}.npy" for k in STATE_SPECS},
                "samples": {k: f"samples/{k}.npy" for k in SAMPLE_SPECS},
            },
        }
        if extra_metadata:
            metadata.update(extra_metadata)
            metadata["dataset_format"] = DATASET_FORMAT
            metadata["state_count"] = state_count
            metadata["sample_count"] = sample_count
            metadata["planet_feature_names"] = list(PLANET_FEATURE_NAMES)
            metadata["global_feature_names"] = list(GLOBAL_FEATURE_NAMES)
            metadata["target_state_feature_names"] = list(TARGET_STATE_FEATURE_NAMES)
            metadata["pair_feature_names"] = list(PAIR_FEATURE_NAMES)
            metadata["files"] = {
                "states": {k: f"states/{k}.npy" for k in STATE_SPECS},
                "samples": {k: f"samples/{k}.npy" for k in SAMPLE_SPECS},
            }
        with open(self.root / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        return metadata


class SourceTurnDatasetReader:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            raise RuntimeError(f"Missing metadata.json for source-turn dataset: {self.root}")
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.metadata.get("dataset_format") != DATASET_FORMAT:
            raise RuntimeError("Dataset is not source_turn_memmap_v1. Rebuild dataset with the compact source-turn memmap builder.")
        stored_pair_names = [str(x) for x in self.metadata.get("pair_feature_names", [])]
        if stored_pair_names != list(PAIR_FEATURE_NAMES):
            raise RuntimeError("pair_feature_names do not match current compact feature contract")
        files = self.metadata.get("files", {})
        self.states = {
            key: np.load(self.root / files.get("states", {}).get(key, f"states/{key}.npy"), mmap_mode="r", allow_pickle=False)
            for key in STATE_SPECS
        }
        self.samples = {
            key: np.load(self.root / files.get("samples", {}).get(key, f"samples/{key}.npy"), mmap_mode="r", allow_pickle=False)
            for key in SAMPLE_SPECS
        }
