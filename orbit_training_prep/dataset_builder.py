from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .features import (
    GLOBAL_FEATURE_NAMES_V2,
    PAIR_FEATURE_NAMES_V2,
    PLANET_FEATURE_NAMES,
    PLANET_FEATURE_NAMES_V2,
    TARGET_STATE_FEATURE_NAMES_V2,
    all_planet_features,
    build_feature_state_v2,
)
from .replay_io import iter_actual_launches, iter_player_steps, load_replay
from .schema import (
    AMOUNT_BIN_NAMES,
    NOOP_TARGET_ID,
    NOOP_TARGET_SLOT,
    P_MAX,
    ActionSpaceSpec,
    build_planet_slot_maps,
    owned_source_slots,
    safe_float,
)
from .target_inference import TargetInferer


def resolve_replay_paths(replay_input: str | Path, max_files: int | None = None) -> list[Path]:
    if max_files is not None and int(max_files) < 1:
        raise ValueError("max_files must be at least 1")
    replay_input = Path(replay_input)
    if replay_input.is_dir():
        paths = sorted(p for p in replay_input.glob("*-replay.json") if p.is_file())
        if not paths:
            paths = sorted(
                p
                for p in replay_input.glob("*.json")
                if p.is_file() and p.name.lower() != "replay_metadata.json"
            )
        if not paths:
            raise ValueError(f"No replay JSON files found in directory: {replay_input}")
        if max_files is not None:
            paths = paths[: int(max_files)]
        return paths
    return [replay_input]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = open(path, "w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self._file.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class DatasetBuilder:
    def __init__(
        self,
        *,
        horizon: int = 160,
        device: str = "auto",
        batch_size: int | None = 256,
        include_loser_actions: bool = True,
        max_replay_files: int | None = None,
    ):
        self.horizon = int(horizon)
        if batch_size is not None and int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = None if batch_size is None else int(batch_size)
        self.include_loser_actions = bool(include_loser_actions)
        self.max_replay_files = None if max_replay_files is None else int(max_replay_files)
        self.inferer = TargetInferer(horizon=horizon, device=device, batch_size=batch_size)
        self.device = self.inferer.device

    def _count_dense_states(self, replay_paths: list[Path]) -> int:
        count = 0
        for replay_path in replay_paths:
            replay = load_replay(replay_path)
            for sample in iter_player_steps(replay):
                final_reward = float(sample["final_reward"])
                if not self.include_loser_actions and final_reward <= 0:
                    continue
                action = list(iter_actual_launches(sample["action"]))
                owned_slots = owned_source_slots(sample["obs"], int(sample["player_id"]))
                if not owned_slots and not action:
                    continue
                count += 1
        return count

    def build_from_replay(self, replay_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
        replay_input = Path(replay_path)
        replay_paths = resolve_replay_paths(replay_input, self.max_replay_files)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dense_state_count = self._count_dense_states(replay_paths)
        dense_tmp_dir = out_dir / ".dense_bc_arrays_tmp"
        if dense_tmp_dir.exists():
            shutil.rmtree(dense_tmp_dir)
        dense_tmp_dir.mkdir(parents=True, exist_ok=True)
        dense_planet_features = None
        dense_planet_features_v2 = None
        dense_global_features_v2 = None
        dense_target_state_features_v2 = None
        dense_source_labels = None
        dense_amount_labels = None
        dense_source_mask = None
        if dense_state_count:
            dense_planet_features = np.lib.format.open_memmap(
                dense_tmp_dir / "planet_features.npy",
                mode="w+",
                dtype=np.float32,
                shape=(dense_state_count, P_MAX, len(PLANET_FEATURE_NAMES)),
            )
            dense_planet_features_v2 = np.lib.format.open_memmap(
                dense_tmp_dir / "planet_features_v2.npy",
                mode="w+",
                dtype=np.float32,
                shape=(dense_state_count, P_MAX, len(PLANET_FEATURE_NAMES_V2)),
            )
            dense_global_features_v2 = np.lib.format.open_memmap(
                dense_tmp_dir / "global_features_v2.npy",
                mode="w+",
                dtype=np.float32,
                shape=(dense_state_count, len(GLOBAL_FEATURE_NAMES_V2)),
            )
            dense_target_state_features_v2 = np.lib.format.open_memmap(
                dense_tmp_dir / "target_state_features_v2.npy",
                mode="w+",
                dtype=np.float32,
                shape=(dense_state_count, P_MAX, len(TARGET_STATE_FEATURE_NAMES_V2)),
            )
            dense_source_labels = np.lib.format.open_memmap(
                dense_tmp_dir / "target_labels.npy",
                mode="w+",
                dtype=np.int64,
                shape=(dense_state_count, P_MAX),
            )
            dense_amount_labels = np.lib.format.open_memmap(
                dense_tmp_dir / "amount_labels.npy",
                mode="w+",
                dtype=np.int64,
                shape=(dense_state_count, P_MAX),
            )
            dense_source_mask = np.lib.format.open_memmap(
                dense_tmp_dir / "source_mask.npy",
                mode="w+",
                dtype=np.float32,
                shape=(dense_state_count, P_MAX),
            )
        dense_index = 0
        stats = Counter()
        inference_methods = Counter()
        amount_bins = Counter()

        try:
            with (
                JsonlWriter(out_dir / "launch_rows.jsonl") as launch_rows,
                JsonlWriter(out_dir / "source_turn_rows.jsonl") as source_turn_rows,
                JsonlWriter(out_dir / "state_rows.jsonl") as state_rows,
            ):
                for replay_path in replay_paths:
                    replay = load_replay(replay_path)
                    for sample in iter_player_steps(replay):
                        obs = sample["obs"]
                        player_id = int(sample["player_id"])
                        final_reward = float(sample["final_reward"])
                        if not self.include_loser_actions and final_reward <= 0:
                            continue
                        action = list(iter_actual_launches(sample["action"]))
                        owned_slots = owned_source_slots(obs, player_id)
                        if not owned_slots and not action:
                            continue
                        obs_uid = f"{sample['episode_id']}:{sample['step_index']}:p{player_id}"
                        pfeat = all_planet_features(obs, player_id, P_MAX)
                        feature_state_v2 = build_feature_state_v2(obs, player_id, P_MAX)
                        source_labels = [NOOP_TARGET_SLOT] * P_MAX
                        amount_labels = [0] * P_MAX
                        source_mask = [1.0 if s in owned_slots else 0.0 for s in range(P_MAX)]
                        state_rows.write({
                            "obs_uid": obs_uid,
                            "episode_id": sample["episode_id"],
                            "step_index": sample["step_index"],
                            "player_id": player_id,
                            "final_reward": final_reward,
                            "num_owned_sources": len(owned_slots),
                            "num_raw_launches": len(action),
                        })
                        stats["states"] += 1
                        stats["owned_source_slots"] += len(owned_slots)
                        stats["raw_launches"] += len(action)
                        if action:
                            stats["raw_launch_batches"] += 1
                            stats["max_raw_launch_batch_size"] = max(int(stats["max_raw_launch_batch_size"]), len(action))

                        inferred_by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
                        inferred_moves = self.inferer.infer_moves(obs, player_id, action) if action else []
                        for launch_index, inf in enumerate(inferred_moves):
                            row = {
                                "obs_uid": obs_uid,
                                "launch_uid": f"{obs_uid}:l{launch_index}",
                                "episode_id": sample["episode_id"],
                                "step_index": sample["step_index"],
                                "obs_step": sample["obs_step"],
                                "player_id": player_id,
                                "final_reward": final_reward,
                                "winner_action": final_reward > 0,
                                "status": sample["status"],
                                **inf.as_dict(),
                            }
                            launch_rows.write(row)
                            stats["valid_launches"] += int(bool(row["valid_source"]))
                            stats["invalid_launches"] += int(not bool(row["valid_source"]))
                            inference_methods[str(row["target_inference_method"])] += 1
                            amount_bins[int(row["amount_bin"])] += 1
                            if row["valid_source"]:
                                inferred_by_source[int(row["source_slot"])].append(row)

                        # Per-owned-source policy rows. This is the primary BC/RL-compatible dataset.
                        for source_slot in owned_slots:
                            launches = inferred_by_source.get(source_slot, [])
                            launches_sorted = sorted(launches, key=lambda r: (-int(r.get("ships", 0)), int(r.get("launch_uid", "0").split("l")[-1])))
                            primary = launches_sorted[0] if launches_sorted else None
                            if primary is None:
                                target_slot = NOOP_TARGET_SLOT
                                target_id = NOOP_TARGET_ID
                                amount_bin = 0
                                ships = 0
                                amount_fraction = 0.0
                                method = "noop"
                                angle_error = 0.0
                                geometry_viable = True
                                contact_target_slot = NOOP_TARGET_SLOT
                                contact_eta = math.inf
                                capture_needed = 0
                            else:
                                target_slot = int(primary["inferred_target_slot"])
                                target_id = int(primary["inferred_target_id"])
                                amount_bin = int(primary["amount_bin"])
                                ships = int(primary["ships"])
                                amount_fraction = float(primary["amount_fraction"])
                                method = str(primary["target_inference_method"])
                                angle_error = float(primary["angle_error"])
                                geometry_viable = bool(primary["geometry_viable"])
                                contact_target_slot = int(primary["contact_target_slot"])
                                contact_eta = float(primary["contact_eta"])
                                capture_needed = int(primary["capture_needed"])
                            multi_launch_count = len(launches_sorted)
                            ambiguous = multi_launch_count > 1
                            row = {
                                "source_turn_uid": f"{obs_uid}:s{source_slot}",
                                "obs_uid": obs_uid,
                                "episode_id": sample["episode_id"],
                                "step_index": sample["step_index"],
                                "obs_step": sample["obs_step"],
                                "player_id": player_id,
                                "final_reward": final_reward,
                                "winner_action": final_reward > 0,
                                "source_slot": int(source_slot),
                                "source_planet_id": int(obs["planets"][source_slot][0]),
                                "target_slot_label": int(target_slot),
                                "target_planet_id_label": int(target_id),
                                "amount_bin_label": int(amount_bin),
                                "amount_bin_name": AMOUNT_BIN_NAMES[int(amount_bin)],
                                "ships_label": int(ships),
                                "amount_fraction": float(amount_fraction),
                                "capture_needed": int(capture_needed),
                                "target_inference_method": method,
                                "contact_target_slot": int(contact_target_slot),
                                "contact_eta": float(contact_eta) if math.isfinite(float(contact_eta)) else None,
                                "angle_error": float(angle_error),
                                "geometry_viable": bool(geometry_viable),
                                "multi_launch_count": int(multi_launch_count),
                                "ambiguous_multi_launch": bool(ambiguous),
                                "train_weight": 1.0 if final_reward > 0 else 0.35,
                                "drop_for_v1_bc": bool(ambiguous or not geometry_viable),
                            }
                            source_turn_rows.write(row)
                            source_labels[source_slot] = int(target_slot)
                            amount_labels[source_slot] = int(amount_bin)
                            stats["source_turn_rows"] += 1
                            stats["noop_source_turns"] += int(primary is None)
                            stats["positive_source_turns"] += int(primary is not None)
                            stats["ambiguous_multi_launch_sources"] += int(ambiguous)

                        if dense_planet_features is not None:
                            dense_planet_features[dense_index] = np.asarray(pfeat, dtype=np.float32)
                            dense_planet_features_v2[dense_index] = feature_state_v2.planet_features
                            dense_global_features_v2[dense_index] = feature_state_v2.global_features
                            dense_target_state_features_v2[dense_index] = feature_state_v2.target_state_features
                            dense_source_labels[dense_index] = np.asarray(source_labels, dtype=np.int64)
                            dense_amount_labels[dense_index] = np.asarray(amount_labels, dtype=np.int64)
                            dense_source_mask[dense_index] = np.asarray(source_mask, dtype=np.float32)
                            dense_index += 1
            if dense_index != dense_state_count:
                raise RuntimeError(f"Dense row count changed while building dataset: expected {dense_state_count}, wrote {dense_index}")

            if dense_planet_features is not None:
                dense_planet_features.flush()
                dense_planet_features_v2.flush()
                dense_global_features_v2.flush()
                dense_target_state_features_v2.flush()
                dense_source_labels.flush()
                dense_amount_labels.flush()
                dense_source_mask.flush()
                np.savez(
                    out_dir / "dense_bc_arrays.npz",
                    planet_features=np.load(dense_tmp_dir / "planet_features.npy", mmap_mode="r"),
                    planet_features_v2=np.load(dense_tmp_dir / "planet_features_v2.npy", mmap_mode="r"),
                    global_features_v2=np.load(dense_tmp_dir / "global_features_v2.npy", mmap_mode="r"),
                    target_state_features_v2=np.load(dense_tmp_dir / "target_state_features_v2.npy", mmap_mode="r"),
                    target_labels=np.load(dense_tmp_dir / "target_labels.npy", mmap_mode="r"),
                    amount_labels=np.load(dense_tmp_dir / "amount_labels.npy", mmap_mode="r"),
                    source_mask=np.load(dense_tmp_dir / "source_mask.npy", mmap_mode="r"),
                    planet_feature_names=np.asarray(PLANET_FEATURE_NAMES),
                    planet_feature_names_v2=np.asarray(PLANET_FEATURE_NAMES_V2),
                    global_feature_names_v2=np.asarray(GLOBAL_FEATURE_NAMES_V2),
                    target_state_feature_names_v2=np.asarray(TARGET_STATE_FEATURE_NAMES_V2),
                    pair_feature_names_v2=np.asarray(PAIR_FEATURE_NAMES_V2),
                    feature_version=np.asarray("v2"),
                )
        finally:
            shutil.rmtree(dense_tmp_dir, ignore_errors=True)
        metadata = {
            "replay_path": str(replay_input),
            "replay_paths": [str(p) for p in replay_paths],
            "action_space": ActionSpaceSpec().as_dict(),
            "geometry_horizon": self.horizon,
            "device": self.device,
            "geometry_device": self.device,
            "inference_batch_size": self.batch_size,
            "target_inference_mode": "batched_exact_first_hit_with_angular_fallback",
            "replay_action_alignment": "action at replay index t is paired with observation at replay index t-1",
            "include_loser_actions": self.include_loser_actions,
            "max_replay_files": self.max_replay_files,
            "files": {
                "launch_rows": "launch_rows.jsonl",
                "source_turn_rows": "source_turn_rows.jsonl",
                "state_rows": "state_rows.jsonl",
                "dense_bc_arrays": "dense_bc_arrays.npz",
            },
            "planet_feature_names": PLANET_FEATURE_NAMES,
            "feature_version": "v2",
            "planet_feature_names_v2": PLANET_FEATURE_NAMES_V2,
            "global_feature_names_v2": GLOBAL_FEATURE_NAMES_V2,
            "target_state_feature_names_v2": TARGET_STATE_FEATURE_NAMES_V2,
            "pair_feature_names_v2": PAIR_FEATURE_NAMES_V2,
            "stats": dict(stats),
            "target_inference_methods": dict(inference_methods),
            "amount_bin_counts": {AMOUNT_BIN_NAMES[int(k)]: int(v) for k, v in amount_bins.items() if int(k) < len(AMOUNT_BIN_NAMES)},
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        return metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, help="Replay JSON file or directory containing replay JSON files.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon", type=int, default=160)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256, help="Maximum number of valid launches to exact-simulate per GPU/CPU batch.")
    ap.add_argument("--winner-only", action="store_true")
    ap.add_argument("--max-files", type=int, default=None, help="Maximum number of replay files to include from a replay directory.")
    args = ap.parse_args()
    builder = DatasetBuilder(
        horizon=args.horizon,
        device=args.device,
        batch_size=args.batch_size,
        include_loser_actions=not args.winner_only,
        max_replay_files=args.max_files,
    )
    meta = builder.build_from_replay(args.replay, args.out_dir)
    print(json.dumps({"out_dir": args.out_dir, "stats": meta["stats"], "target_inference_methods": meta["target_inference_methods"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
