from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .features import (
    GLOBAL_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    PLANET_FEATURE_NAMES,
    TARGET_STATE_FEATURE_NAMES,
    _targeted_fleets,
    build_feature_state,
    pair_features_from_obs,
)
from .geometry_bridge import make_geometry
from .canonical import canonicalize_launches, canonicalize_observation
from .replay_io import iter_actual_launches, iter_player_steps, load_replay
from .schema import (
    AMOUNT_BIN_NAMES,
    NOOP_TARGET_ID,
    NOOP_TARGET_SLOT,
    P_MAX,
    ActionSpaceSpec,
    owned_source_slots,
)
from .source_turn_store import DATASET_FORMAT, SAMPLE_SPECS, STATE_SPECS, SourceTurnDatasetReader, SourceTurnDatasetWriter
from .lite_backend import (
    LITE_MASK_MODE,
    LITE_PAIR_ETA_MODE,
    LITE_TARGET_INFERENCE_MODE,
    LiteTargetInferer,
    build_lite_context,
    compute_lite_viability_masks,
    pair_features_lite,
)
from .target_inference import TargetInferer
from .viability import compute_viability_masks


def resolve_replay_paths(replay_input: str | Path, max_files: int | None = None) -> list[Path]:
    if max_files is not None and int(max_files) < 1:
        raise ValueError("max_files must be at least 1")
    replay_input = Path(replay_input)
    if replay_input.is_dir():
        paths = sorted(p for p in replay_input.glob("*-replay.json") if p.is_file())
        if not paths:
            paths = sorted(
                p for p in replay_input.glob("*.json") if p.is_file() and p.name.lower() != "replay_metadata.json"
            )
        if not paths:
            raise ValueError(f"No replay JSON files found in directory: {replay_input}")
        return paths[: int(max_files)] if max_files is not None else paths
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


def _stable_hash_int(seed: int, *parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(seed)).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8", errors="replace"))
    return int.from_bytes(h.digest(), "big", signed=False)


def _replay_player_count(replay: dict[str, Any]) -> int | None:
    rewards = replay.get("rewards")
    if isinstance(rewards, list) and rewards:
        return int(len(rewards))
    steps = replay.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[0], list) and steps[0]:
        return int(len(steps[0]))
    return None


def _profile_replay_player_count(path: Path) -> int | None:
    try:
        return _replay_player_count(load_replay(path))
    except Exception:
        return None


def _str_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def _select_balanced_replay_paths(
    replay_paths: list[Path],
    *,
    max_files: int | None,
    balance_seed: int,
) -> tuple[list[Path], dict[str, Any]]:
    paths = list(replay_paths)
    limit = len(paths) if max_files is None else min(int(max_files), len(paths))
    if limit <= 0 or not paths:
        return [], {
            "requested": len(paths),
            "limit": int(limit),
            "observed_player_counts": {},
            "selected_player_counts": {},
            "infeasible": False,
            "shortfall": 0,
        }

    profiles: list[tuple[int, Path, int | None]] = [
        (i, path, _profile_replay_player_count(path)) for i, path in enumerate(paths)
    ]
    observed = Counter(player_count if player_count is not None else "unknown" for _, _, player_count in profiles)
    by_players: dict[int, list[tuple[int, Path]]] = {2: [], 4: []}
    other: list[tuple[int, Path]] = []
    for original_index, path, player_count in profiles:
        if player_count in by_players:
            by_players[int(player_count)].append((original_index, path))
        else:
            other.append((original_index, path))

    def ranked(items: list[tuple[int, Path]]) -> list[tuple[int, Path]]:
        return sorted(items, key=lambda x: (_stable_hash_int(balance_seed, x[1]), x[0]))

    selected_pairs: list[tuple[int, Path]] = []
    infeasible = False
    if by_players[2] and by_players[4]:
        best: tuple[float, int, int, int] | None = None
        for n2 in range(1, min(len(by_players[2]), limit) + 1):
            for n4 in range(1, min(len(by_players[4]), limit - n2) + 1):
                total = n2 + n4
                ratio_error = abs((n2 / float(total)) - 0.5)
                candidate = (ratio_error, -total, n2, n4)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            infeasible = True
            ranked_known = ranked(by_players[2] + by_players[4])
            selected_pairs.extend(ranked_known[:limit])
        else:
            _, neg_total, n2, n4 = best
            selected_pairs.extend(ranked(by_players[2])[:n2])
            selected_pairs.extend(ranked(by_players[4])[:n4])
            infeasible = int(-neg_total) < min(limit, len(by_players[2]) + len(by_players[4]))
    elif by_players[2] or by_players[4]:
        infeasible = True
        selected_pairs.extend(ranked(by_players[2] + by_players[4])[:limit])
    else:
        selected_pairs.extend(other[:limit])

    selected_indexes = {idx for idx, _ in selected_pairs}
    remaining = [item for item in ranked(other) if item[0] not in selected_indexes]
    if len(selected_pairs) < limit and not (by_players[2] and by_players[4]):
        selected_pairs.extend(remaining[: limit - len(selected_pairs)])

    selected_pairs = sorted(selected_pairs, key=lambda x: x[0])
    selected = [path for _, path in selected_pairs]
    selected_profile = Counter()
    for _, path in selected_pairs:
        player_count = next((pc for i, p, pc in profiles if p == path), None)
        selected_profile[player_count if player_count is not None else "unknown"] += 1
    shortfall = int(limit - len(selected))
    report = {
        "requested": len(paths),
        "limit": int(limit),
        "observed_player_counts": _str_counter(observed),
        "selected_player_counts": _str_counter(selected_profile),
        "selected": len(selected),
        "infeasible": bool(infeasible or shortfall > 0),
        "shortfall": shortfall,
        "target_player_mix": {"2": 0.5, "4": 0.5},
    }
    return selected, report


class JsonlWriter:
    def __init__(self, path: Path, enabled: bool = True):
        self.enabled = bool(enabled)
        self.path = path
        self._file = None
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        if self._file is not None:
            self._file.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _source_row_train_contract(row: dict[str, Any], target_viability_mask: np.ndarray, amount_viability_mask: np.ndarray) -> tuple[bool, dict[str, bool]]:
    target_label = int(row.get("target_slot_label", NOOP_TARGET_SLOT))
    amount_label = int(row.get("amount_bin_label", 0))
    is_noop = target_label == NOOP_TARGET_SLOT
    if is_noop:
        return True, {"ambiguous": False, "geometry_invalid": False, "mask_invalid": False, "angular_fallback": False}
    source_slot = int(row.get("source_slot", -1))
    ambiguous = bool(row.get("ambiguous_multi_launch", False))
    geometry_invalid = not bool(row.get("geometry_viable", False))
    angular_fallback = str(row.get("target_inference_method", "") or "") == "angular_nearest"
    mask_invalid = True
    if (
        0 <= source_slot < int(target_viability_mask.shape[0])
        and 0 <= target_label < int(target_viability_mask.shape[1])
        and 0 <= amount_label < int(amount_viability_mask.shape[2])
    ):
        target_ok = bool(target_viability_mask[source_slot, target_label])
        amount_ok = bool(amount_viability_mask[source_slot, target_label, amount_label])
        mask_invalid = not (target_ok and amount_ok)
    reasons = {
        "ambiguous": ambiguous,
        "geometry_invalid": geometry_invalid,
        "mask_invalid": mask_invalid,
        "angular_fallback": angular_fallback,
    }
    return not any(reasons.values()), reasons


TRAIN_STATS_KEYS = (
    "raw_source_turns",
    "train_source_turns",
    "raw_positive_source_turns",
    "train_positive_source_turns",
    "dropped_source_turns",
    "dropped_ambiguous_sources",
    "dropped_geometry_invalid_sources",
    "dropped_mask_invalid_sources",
    "dropped_angular_fallback_sources",
    "noop_source_turns",
)


def _stats_with_required_train_keys(stats: Counter[str]) -> dict[str, int]:
    out = {str(k): int(v) for k, v in stats.items()}
    for key in TRAIN_STATS_KEYS:
        out.setdefault(key, 0)
    out.setdefault("source_turn_rows", out["train_source_turns"])
    out.setdefault("positive_source_turns", out["train_positive_source_turns"])
    return out



def _build_single_replay_shard(
    replay_path: str,
    out_dir: str,
    *,
    horizon: int,
    device: str,
    batch_size: int | None,
    include_loser_actions: bool,
    write_debug_jsonl: bool,
    backend: str,
    canonicalize_perspective: bool,
) -> dict[str, Any]:
    builder = DatasetBuilder(
        horizon=horizon,
        device=device,
        batch_size=batch_size,
        include_loser_actions=include_loser_actions,
        max_replay_files=1,
        workers=1,
        write_debug_jsonl=write_debug_jsonl,
        backend=backend,
        canonicalize_perspective=canonicalize_perspective,
        balance_proportions=False,
    )
    return builder.build_from_replay(replay_path, out_dir)


def _copy_group_from_shards(
    *,
    group_name: str,
    specs: dict[str, tuple[Any, tuple[int, ...]]],
    shard_dirs: list[Path],
    out_dir: Path,
    total_count: int,
    state_offsets: list[int] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, (dtype, tail_shape) in specs.items():
        out_path = out_dir / f"{key}.npy"
        merged = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(int(total_count), *tail_shape))
        offset = 0
        for shard_i, shard_dir in enumerate(shard_dirs):
            shard_path = shard_dir / group_name / f"{key}.npy"
            if not shard_path.exists():
                continue
            arr = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            rows = int(arr.shape[0])
            if rows <= 0:
                continue
            end = offset + rows
            if group_name == "samples" and key == "state_index" and state_offsets is not None:
                merged[offset:end] = np.asarray(arr, dtype=np.uint32) + np.uint32(state_offsets[shard_i])
            else:
                merged[offset:end] = arr
            offset = end
        if offset != int(total_count):
            raise RuntimeError(f"Merged {group_name}/{key} row mismatch: expected {total_count}, wrote {offset}")
        merged.flush()


def _concat_debug_jsonl(out_path: Path, input_paths: list[Path]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for input_path in input_paths:
            if input_path.exists():
                with open(input_path, "r", encoding="utf-8") as in_f:
                    shutil.copyfileobj(in_f, out_f)

def _sample_weight(row: dict[str, Any]) -> float:
    target = int(row.get("target_slot_label", NOOP_TARGET_SLOT))
    is_noop = target == NOOP_TARGET_SLOT
    weight = float(row.get("train_weight", 1.0))
    if is_noop:
        weight *= 0.2
    if bool(row.get("winner_action", False)) or float(row.get("final_reward", 0.0) or 0.0) > 0.0:
        weight *= 1.1
    step = int(row.get("step_index", row.get("step", row.get("obs_step", 0))) or 0)
    if not is_noop and step <= 100:
        weight *= 1.1
    if step > 430:
        weight *= 0.8
    return float(weight)


def _sample_hash_order(arrays: dict[str, np.ndarray], seed: int, indexes: np.ndarray) -> list[int]:
    return sorted(
        (int(i) for i in indexes.tolist()),
        key=lambda i: _stable_hash_int(
            seed,
            i,
            int(arrays["state_index"][i]),
            int(arrays["source_slot"][i]),
            int(arrays["target_label"][i]),
            int(arrays["amount_label"][i]),
            int(arrays["step"][i]),
        ),
    )


def _allocate_amount_quotas(amount_counts: Counter[int], total: int) -> dict[int, int]:
    if total <= 0 or not amount_counts:
        return {}
    bins = sorted(int(b) for b, count in amount_counts.items() if int(b) != 0 and int(count) > 0)
    if not bins:
        return {}
    weights = {b: math.sqrt(float(amount_counts[b])) for b in bins}
    weight_sum = sum(weights.values())
    raw = {b: (float(total) * weights[b] / weight_sum) for b in bins}
    quotas = {b: min(int(amount_counts[b]), int(math.floor(raw[b]))) for b in bins}
    assigned = sum(quotas.values())

    if total >= len(bins):
        for b in bins:
            if assigned >= total:
                break
            if quotas[b] == 0:
                quotas[b] = 1
                assigned += 1

    while assigned < total:
        candidates = [b for b in bins if quotas[b] < int(amount_counts[b])]
        if not candidates:
            break
        b = max(candidates, key=lambda x: (raw[x] - quotas[x], int(amount_counts[x]) - quotas[x], -x))
        quotas[b] += 1
        assigned += 1
    return {b: int(q) for b, q in quotas.items() if int(q) > 0}


def _select_balanced_sample_indexes(
    arrays: dict[str, np.ndarray],
    *,
    balance_seed: int,
    noop_ratio: float,
    op_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_label = np.asarray(arrays["target_label"])
    amount_label = np.asarray(arrays["amount_label"])
    all_count = int(target_label.shape[0])
    noop_indexes = np.flatnonzero(target_label == NOOP_TARGET_SLOT)
    op_indexes = np.flatnonzero(target_label != NOOP_TARGET_SLOT)
    noop_count = int(noop_indexes.shape[0])
    op_count = int(op_indexes.shape[0])
    infeasible = False

    if all_count == 0:
        selected = np.asarray([], dtype=np.int64)
        selected_noop = 0
        selected_op = 0
    elif noop_count == 0 or op_count == 0:
        infeasible = True
        selected = np.arange(all_count, dtype=np.int64)
        selected_noop = noop_count
        selected_op = op_count
    else:
        needed_noop_for_all_ops = int(math.floor(op_count * float(noop_ratio) / max(float(op_ratio), 1e-12)))
        if needed_noop_for_all_ops <= noop_count:
            selected_noop = max(1, needed_noop_for_all_ops)
            selected_op = op_count
        else:
            selected_noop = noop_count
            selected_op = min(op_count, int(math.floor(noop_count * float(op_ratio) / max(float(noop_ratio), 1e-12))))
            selected_op = max(1, selected_op)
        infeasible = selected_noop < noop_count or selected_op < op_count

        selected_noops = _sample_hash_order(arrays, balance_seed, noop_indexes)[:selected_noop]
        op_amount_counts = Counter(int(amount_label[i]) for i in op_indexes.tolist() if int(amount_label[i]) != 0)
        quotas = _allocate_amount_quotas(op_amount_counts, selected_op)
        selected_ops: list[int] = []
        if quotas:
            for amount_bin, quota in quotas.items():
                bin_indexes = op_indexes[amount_label[op_indexes] == int(amount_bin)]
                selected_ops.extend(_sample_hash_order(arrays, balance_seed, bin_indexes)[: int(quota)])
        if len(selected_ops) < selected_op:
            already = set(selected_ops)
            remaining = np.asarray([int(i) for i in op_indexes.tolist() if int(i) not in already], dtype=np.int64)
            selected_ops.extend(_sample_hash_order(arrays, balance_seed, remaining)[: selected_op - len(selected_ops)])
        selected = np.asarray(sorted(selected_noops + selected_ops), dtype=np.int64)

    final_amounts = Counter(int(amount_label[i]) for i in selected.tolist())
    report = {
        "requested": {
            "noop_ratio": float(noop_ratio),
            "op_ratio": float(op_ratio),
        },
        "observed": {
            "total": all_count,
            "noop": noop_count,
            "op": op_count,
            "amount_bins": {
                AMOUNT_BIN_NAMES[int(k)]: int(v)
                for k, v in sorted(Counter(int(a) for a in amount_label.tolist()).items())
                if 0 <= int(k) < len(AMOUNT_BIN_NAMES)
            },
        },
        "selected": {
            "total": int(selected.shape[0]),
            "noop": int(selected_noop),
            "op": int(selected_op),
            "amount_bins": {
                AMOUNT_BIN_NAMES[int(k)]: int(v)
                for k, v in sorted(final_amounts.items())
                if 0 <= int(k) < len(AMOUNT_BIN_NAMES)
            },
        },
        "infeasible": bool(infeasible),
        "dropped": int(all_count - int(selected.shape[0])),
    }
    return selected, report


def _load_sample_arrays(out_dir: Path, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    files = metadata.get("files", {}).get("samples", {}) if isinstance(metadata.get("files"), dict) else {}
    arrays: dict[str, np.ndarray] = {}
    for key in ("state_index", "source_slot", "target_label", "amount_label", "step"):
        path = out_dir / str(files.get(key, f"samples/{key}.npy"))
        arrays[key] = np.load(path, allow_pickle=False)
    return arrays


def _rewrite_sample_arrays(out_dir: Path, metadata: dict[str, Any], selected: np.ndarray) -> None:
    files = metadata.get("files", {}).get("samples", {}) if isinstance(metadata.get("files"), dict) else {}
    for key, (dtype, _) in SAMPLE_SPECS.items():
        path = out_dir / str(files.get(key, f"samples/{key}.npy"))
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        filtered = np.asarray(arr[selected], dtype=dtype)
        del arr
        tmp_path = path.with_name(f"{path.name}.tmp.npy")
        np.save(tmp_path, filtered, allow_pickle=False)
        tmp_path.replace(path)


def _balance_source_turn_dataset(
    out_dir: str | Path,
    metadata: dict[str, Any],
    *,
    balance_seed: int,
    noop_ratio: float,
    op_ratio: float,
    replay_selection_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    arrays = _load_sample_arrays(out_dir, metadata)
    selected, sample_report = _select_balanced_sample_indexes(
        arrays,
        balance_seed=balance_seed,
        noop_ratio=noop_ratio,
        op_ratio=op_ratio,
    )
    _rewrite_sample_arrays(out_dir, metadata, selected)

    final_count = int(selected.shape[0])
    final_target = np.asarray(arrays["target_label"])[selected]
    final_amount = np.asarray(arrays["amount_label"])[selected]
    final_noop = int(np.sum(final_target == NOOP_TARGET_SLOT))
    final_op = int(final_count - final_noop)
    final_amount_counts = {
        AMOUNT_BIN_NAMES[int(k)]: int(v)
        for k, v in sorted(Counter(int(a) for a in final_amount.tolist()).items())
        if 0 <= int(k) < len(AMOUNT_BIN_NAMES)
    }

    updated = dict(metadata)
    old_stats = dict(metadata.get("stats", {}))
    stats = dict(old_stats)
    stats["train_source_turns"] = final_count
    stats["train_positive_source_turns"] = final_op
    stats["source_turn_rows"] = final_count
    stats["positive_source_turns"] = final_op
    stats["noop_source_turns"] = final_noop
    stats["balance_dropped_source_turns"] = int(sample_report["dropped"])
    updated["stats"] = _stats_with_required_train_keys(Counter(stats))
    updated["sample_count"] = final_count
    updated["amount_bin_counts"] = final_amount_counts

    correction = dict(metadata.get("proportion_correction", {}))
    correction.update(
        {
            "enabled": True,
            "balance_seed": int(balance_seed),
            "unbalanced_stats": old_stats,
            "sample_balance": sample_report,
        }
    )
    if replay_selection_report is not None:
        correction["replay_selection"] = replay_selection_report
    updated["proportion_correction"] = correction
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, sort_keys=True)
    return updated


class DatasetBuilder:
    def __init__(
        self,
        *,
        horizon: int = 160,
        device: str = "auto",
        batch_size: int | None = 256,
        include_loser_actions: bool = True,
        max_replay_files: int | None = None,
        workers: int = 1,
        write_debug_jsonl: bool = False,
        backend: str = "exact",
        canonicalize_perspective: bool = True,
        balance_proportions: bool = True,
        balance_seed: int = 42,
        noop_ratio: float = 0.40,
        op_ratio: float = 0.60,
    ):
        self.horizon = int(horizon)
        if batch_size is not None and int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        if int(workers) < 1:
            raise ValueError("workers must be at least 1")
        self.batch_size = None if batch_size is None else int(batch_size)
        self.include_loser_actions = bool(include_loser_actions)
        self.max_replay_files = None if max_replay_files is None else int(max_replay_files)
        self.workers = int(workers)
        self.write_debug_jsonl = bool(write_debug_jsonl)
        self.canonicalize_perspective = bool(canonicalize_perspective)
        self.balance_proportions = bool(balance_proportions)
        self.balance_seed = int(balance_seed)
        self.noop_ratio = float(noop_ratio)
        self.op_ratio = float(op_ratio)
        if self.noop_ratio < 0.0 or self.op_ratio < 0.0 or (self.noop_ratio + self.op_ratio) <= 0.0:
            raise ValueError("noop_ratio and op_ratio must be non-negative with a positive sum")
        ratio_total = self.noop_ratio + self.op_ratio
        self.noop_ratio = float(self.noop_ratio / ratio_total)
        self.op_ratio = float(self.op_ratio / ratio_total)
        backend_value = str(backend).lower().strip()
        if backend_value not in {"exact", "lite"}:
            raise ValueError("backend must be one of: exact, lite")
        self.backend = backend_value
        if self.backend == "exact":
            self.inferer = TargetInferer(horizon=horizon, device=device, batch_size=batch_size)
            self.lite_inferer = None
            self.device = self.inferer.device
        else:
            self.inferer = None
            self.lite_inferer = LiteTargetInferer(horizon=horizon)
            self.device = "cpu" if str(device).lower() == "auto" else str(device)

    def _load_replay_or_none(self, replay_path: Path, *, invalid_json_paths: set[str], unreadable_paths: set[str]) -> dict[str, Any] | None:
        try:
            return load_replay(replay_path)
        except json.JSONDecodeError as exc:
            path_str = str(replay_path)
            if path_str not in invalid_json_paths:
                invalid_json_paths.add(path_str)
                print(json.dumps({"warning": "Skipping replay with invalid JSON", "replay_path": path_str, "error": str(exc)}, sort_keys=True), file=sys.stderr)
            return None
        except OSError as exc:
            path_str = str(replay_path)
            if path_str not in unreadable_paths:
                unreadable_paths.add(path_str)
                print(json.dumps({"warning": "Skipping unreadable replay file", "replay_path": path_str, "error": str(exc)}, sort_keys=True), file=sys.stderr)
            return None

    def _build_from_replay_parallel(self, replay_input: Path, replay_paths: list[Path], out_dir: Path) -> dict[str, Any]:
        shard_root = out_dir / ".dataset_builder_worker_shards"
        if shard_root.exists():
            shutil.rmtree(shard_root)
        shard_root.mkdir(parents=True, exist_ok=True)
        worker_count = min(int(self.workers), len(replay_paths))
        shard_dirs = [shard_root / f"shard_{i:06d}" for i in range(len(replay_paths))]
        shard_metas: list[dict[str, Any]] = [{} for _ in replay_paths]
        try:
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as pool:
                futures = [
                    pool.submit(
                        _build_single_replay_shard,
                        str(replay_path_obj),
                        str(shard_dirs[i]),
                        horizon=self.horizon,
                        device=self.device,
                        batch_size=self.batch_size,
                        include_loser_actions=self.include_loser_actions,
                        write_debug_jsonl=self.write_debug_jsonl,
                        backend=self.backend,
                        canonicalize_perspective=self.canonicalize_perspective,
                    )
                    for i, replay_path_obj in enumerate(replay_paths)
                ]
                for i, future in enumerate(futures):
                    try:
                        shard_metas[i] = future.result()
                    except Exception as exc:  # pragma: no cover - exercised in integration builds
                        raise RuntimeError(f"Worker failed while processing replay: {replay_paths[i]}") from exc

            state_counts = [int(meta.get("state_count", 0) or 0) for meta in shard_metas]
            sample_counts = [int(meta.get("sample_count", 0) or 0) for meta in shard_metas]
            state_offsets: list[int] = []
            running_states = 0
            for count in state_counts:
                state_offsets.append(running_states)
                running_states += int(count)
            total_states = int(sum(state_counts))
            total_samples = int(sum(sample_counts))
            used_shard_dirs = [d for d, meta in zip(shard_dirs, shard_metas, strict=True) if int(meta.get("state_count", 0) or 0) > 0]
            used_state_offsets = [state_offsets[i] for i, meta in enumerate(shard_metas) if int(meta.get("state_count", 0) or 0) > 0]
            _copy_group_from_shards(
                group_name="states",
                specs=STATE_SPECS,
                shard_dirs=used_shard_dirs,
                out_dir=out_dir / "states",
                total_count=total_states,
            )
            _copy_group_from_shards(
                group_name="samples",
                specs=SAMPLE_SPECS,
                shard_dirs=used_shard_dirs,
                out_dir=out_dir / "samples",
                total_count=total_samples,
                state_offsets=used_state_offsets,
            )
            if self.write_debug_jsonl:
                _concat_debug_jsonl(out_dir / "debug" / "launch_rows.jsonl", [d / "debug" / "launch_rows.jsonl" for d in used_shard_dirs])
                _concat_debug_jsonl(out_dir / "debug" / "source_turn_rows.jsonl", [d / "debug" / "source_turn_rows.jsonl" for d in used_shard_dirs])
                _concat_debug_jsonl(out_dir / "debug" / "state_rows.jsonl", [d / "debug" / "state_rows.jsonl" for d in used_shard_dirs])

            stats = Counter()
            inference_methods = Counter()
            amount_bin_counts = Counter()
            invalid_json_paths: set[str] = set()
            unreadable_paths: set[str] = set()
            valid_replay_paths: list[str] = []
            for meta in shard_metas:
                if not meta:
                    continue
                for replay_path_str in meta.get("replay_paths", []):
                    if replay_path_str:
                        valid_replay_paths.append(str(replay_path_str))
                for key, value in meta.get("stats", {}).items():
                    if key == "max_raw_launch_batch_size":
                        stats[key] = max(int(stats.get(key, 0) or 0), int(value or 0))
                    else:
                        stats[key] += int(value or 0)
                for method, value in meta.get("target_inference_methods", {}).items():
                    inference_methods[str(method)] += int(value or 0)
                for name, value in meta.get("amount_bin_counts", {}).items():
                    amount_bin_counts[str(name)] += int(value or 0)
                input_meta = meta.get("input_replay_files", {})
                invalid_json_paths.update(str(p) for p in input_meta.get("skipped_invalid_json_paths", []) if p)
                unreadable_paths.update(str(p) for p in input_meta.get("skipped_unreadable_paths", []) if p)

            metadata = {
                "dataset_format": DATASET_FORMAT,
                "state_count": total_states,
                "sample_count": total_samples,
                "replay_path": str(replay_input),
                "replay_paths": valid_replay_paths,
                "replay_paths_requested": [str(p) for p in replay_paths],
                "action_space": ActionSpaceSpec().as_dict(),
                "geometry_horizon": self.horizon,
                "device": self.device,
                "geometry_device": self.device,
                "inference_batch_size": self.batch_size,
                "backend": self.backend,
                "target_inference_mode": LITE_TARGET_INFERENCE_MODE if self.backend == "lite" else "batched_exact_first_hit_with_angular_fallback",
                "mask_mode": LITE_MASK_MODE if self.backend == "lite" else "exact-geometry",
                "pair_eta_mode": LITE_PAIR_ETA_MODE if self.backend == "lite" else "exact-geometry",
                "replay_action_alignment": "action at replay index t is paired with observation at replay index t-1",
                "include_loser_actions": self.include_loser_actions,
                "max_replay_files": self.max_replay_files,
                "write_debug_jsonl": self.write_debug_jsonl,
                "perspective_canonicalization": {"enabled": self.canonicalize_perspective, "frame": "p0", "rotation": "-2*pi*player_id/num_players"},
                "parallel": {
                    "enabled": True,
                    "worker_count": worker_count,
                    "shards": len(shard_dirs),
                },
                "planet_feature_names": PLANET_FEATURE_NAMES,
                "global_feature_names": GLOBAL_FEATURE_NAMES,
                "target_state_feature_names": TARGET_STATE_FEATURE_NAMES,
                "pair_feature_names": PAIR_FEATURE_NAMES,
                "pair_feature_dtype": "float16",
                "files": {
                    "states": {k: f"states/{k}.npy" for k in STATE_SPECS},
                    "samples": {k: f"samples/{k}.npy" for k in SAMPLE_SPECS},
                },
                "stats": _stats_with_required_train_keys(stats),
                "input_replay_files": {
                    "requested": len(replay_paths),
                    "used": len(valid_replay_paths),
                    "skipped_invalid_json": len(invalid_json_paths),
                    "skipped_unreadable": len(unreadable_paths),
                    "skipped_invalid_json_paths": sorted(invalid_json_paths),
                    "skipped_unreadable_paths": sorted(unreadable_paths),
                },
                "target_inference_methods": dict(inference_methods),
                "amount_bin_counts": dict(amount_bin_counts),
            }
            with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
            return metadata
        finally:
            shutil.rmtree(shard_root, ignore_errors=True)

    def build_from_replay(self, replay_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
        replay_input = Path(replay_path)
        if self.balance_proportions:
            requested_replay_paths = resolve_replay_paths(replay_input, None)
            replay_paths, replay_selection_report = _select_balanced_replay_paths(
                requested_replay_paths,
                max_files=self.max_replay_files,
                balance_seed=self.balance_seed,
            )
        else:
            replay_paths = resolve_replay_paths(replay_input, self.max_replay_files)
            replay_selection_report = {
                "requested": len(replay_paths),
                "limit": len(replay_paths),
                "observed_player_counts": {},
                "selected_player_counts": {},
                "selected": len(replay_paths),
                "infeasible": False,
                "shortfall": 0,
                "target_player_mix": {"2": 0.5, "4": 0.5},
            }
        out_dir = Path(out_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.workers > 1 and len(replay_paths) > 1:
            if self.backend == "exact" and str(self.device).lower().startswith("cuda"):
                print(
                    json.dumps(
                        {
                            "warning": "workers>1 is disabled for CUDA exact target inference to avoid spawning multiple GPU contexts; falling back to serial build",
                            "workers": int(self.workers),
                            "device": str(self.device),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
            else:
                metadata = self._build_from_replay_parallel(replay_input, replay_paths, out_dir)
                if self.balance_proportions:
                    metadata = _balance_source_turn_dataset(
                        out_dir,
                        metadata,
                        balance_seed=self.balance_seed,
                        noop_ratio=self.noop_ratio,
                        op_ratio=self.op_ratio,
                        replay_selection_report=replay_selection_report,
                    )
                else:
                    metadata["proportion_correction"] = {"enabled": False, "replay_selection": replay_selection_report}
                    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, sort_keys=True)
                return metadata
        writer = SourceTurnDatasetWriter(out_dir)
        invalid_json_paths: set[str] = set()
        unreadable_paths: set[str] = set()
        valid_replay_paths: list[Path] = []
        stats: Counter[str] = Counter()
        inference_methods: Counter[str] = Counter()
        amount_bins: Counter[int] = Counter()
        debug_dir = out_dir / "debug"

        with (
            JsonlWriter(debug_dir / "launch_rows.jsonl", enabled=self.write_debug_jsonl) as launch_rows,
            JsonlWriter(debug_dir / "source_turn_rows.jsonl", enabled=self.write_debug_jsonl) as source_turn_rows,
            JsonlWriter(debug_dir / "state_rows.jsonl", enabled=self.write_debug_jsonl) as state_rows,
        ):
            for replay_path_obj in replay_paths:
                replay = self._load_replay_or_none(
                    replay_path_obj,
                    invalid_json_paths=invalid_json_paths,
                    unreadable_paths=unreadable_paths,
                )
                if replay is None:
                    continue
                used_replay = False
                for sample in iter_player_steps(replay):
                    raw_obs = sample["obs"]
                    original_player_id = int(sample["player_id"])
                    player_id = original_player_id
                    final_reward = float(sample["final_reward"])
                    if not self.include_loser_actions and final_reward <= 0:
                        continue
                    raw_action = list(iter_actual_launches(sample["action"]))
                    if self.canonicalize_perspective:
                        transform = canonicalize_observation(raw_obs, original_player_id)
                        obs = transform.obs
                        action = canonicalize_launches(raw_action, transform)
                        player_id = 0
                    else:
                        obs = raw_obs
                        action = raw_action
                    owned_slots = owned_source_slots(obs, player_id)
                    if not owned_slots and not action:
                        continue
                    stats["canonicalized_player_steps"] += int(self.canonicalize_perspective and original_player_id != 0)
                    used_replay = True
                    obs_uid = f"{sample['episode_id']}:{sample['step_index']}:p{original_player_id}:canon{player_id}"
                    feature_state = build_feature_state(obs, player_id, P_MAX)
                    lite_ctx = None
                    pair_geometry = None
                    pair_movement = None
                    pair_incoming = None
                    if self.backend == "lite":
                        lite_ctx = build_lite_context(obs, player_id, horizon=self.horizon, device="cpu")
                        target_viability_mask, amount_viability_mask = compute_lite_viability_masks(lite_ctx)
                    else:
                        target_viability_mask, amount_viability_mask = compute_viability_masks(
                            obs,
                            player_id,
                            horizon=self.horizon,
                            device=self.device,
                        )
                        pair_geometry = make_geometry(horizon=200, device="cpu")
                        try:
                            obs_no_fleets = dict(obs)
                            obs_no_fleets["fleets"] = []
                            pair_movement = pair_geometry.build_or_update_movement(pair_geometry.obs_to_tensors(obs_no_fleets, player_id=player_id))
                        except Exception:
                            pair_movement = None
                        pair_incoming = _targeted_fleets(obs, player_id, P_MAX)
                    state_index = writer.append_state(
                        planet_features=feature_state.planet_features,
                        global_features=feature_state.global_features,
                        target_state_features=feature_state.target_state_features,
                        episode_id=str(sample["episode_id"]),
                    )
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
                    if action and self.backend == "lite":
                        assert lite_ctx is not None and self.lite_inferer is not None
                        inferred_moves = self.lite_inferer.infer_moves(lite_ctx, action)
                    elif action:
                        assert self.inferer is not None
                        inferred_moves = self.inferer.infer_moves(obs, player_id, action)
                    else:
                        inferred_moves = []
                    for launch_index, inf in enumerate(inferred_moves):
                        row = {
                            "obs_uid": obs_uid,
                            "launch_uid": f"{obs_uid}:l{launch_index}",
                            "episode_id": sample["episode_id"],
                            "step_index": sample["step_index"],
                            "obs_step": sample["obs_step"],
                            "player_id": player_id,
                            "original_player_id": original_player_id,
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

                    for source_slot in owned_slots:
                        launches = inferred_by_source.get(source_slot, [])
                        launches_sorted = sorted(launches, key=lambda r: (-int(r.get("ships", 0)), int(str(r.get("launch_uid", "0")).split("l")[-1])))
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
                            "original_player_id": original_player_id,
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
                            "drop_for_v1_bc": False,
                        }
                        train_valid, drop_reasons = _source_row_train_contract(row, target_viability_mask, amount_viability_mask)
                        is_positive = primary is not None
                        stats["raw_source_turns"] += 1
                        stats["raw_positive_source_turns"] += int(is_positive)
                        stats["noop_source_turns"] += int(not is_positive)
                        stats["ambiguous_multi_launch_sources"] += int(ambiguous)
                        stats["dropped_source_turns"] += int(not train_valid)
                        stats["dropped_ambiguous_sources"] += int(drop_reasons["ambiguous"])
                        stats["dropped_geometry_invalid_sources"] += int(drop_reasons["geometry_invalid"])
                        stats["dropped_mask_invalid_sources"] += int(drop_reasons["mask_invalid"])
                        stats["dropped_angular_fallback_sources"] += int(drop_reasons["angular_fallback"])
                        if not train_valid:
                            continue
                        if self.backend == "lite":
                            assert lite_ctx is not None
                            pair_features = pair_features_lite(
                                lite_ctx,
                                feature_state,
                                int(source_slot),
                                target_viability_mask=target_viability_mask[int(source_slot)],
                                amount_viability_mask=amount_viability_mask[int(source_slot)],
                            )
                        else:
                            pair_features = pair_features_from_obs(
                                obs,
                                player_id,
                                int(source_slot),
                                max_planets=P_MAX,
                                target_viability_mask=target_viability_mask[int(source_slot)],
                                amount_viability_mask=amount_viability_mask[int(source_slot)],
                                feature_state=feature_state,
                                geometry=pair_geometry,
                                movement=pair_movement,
                                incoming_by_target=pair_incoming,
                            )
                        target_label = int(target_slot)
                        amount_label = 0 if target_label == NOOP_TARGET_SLOT else int(amount_bin)
                        writer.append_sample(
                            state_index=state_index,
                            source_slot=int(source_slot),
                            target_label=target_label,
                            amount_label=amount_label,
                            sample_weight=_sample_weight(row),
                            step=int(sample["step_index"]),
                            pair_features=pair_features,
                            target_mask=target_viability_mask[int(source_slot)],
                            amount_mask=amount_viability_mask[int(source_slot), target_label],
                        )
                        source_turn_rows.write(row)
                        stats["train_source_turns"] += 1
                        stats["train_positive_source_turns"] += int(is_positive)
                        stats["source_turn_rows"] += 1
                        stats["positive_source_turns"] += int(is_positive)
                if used_replay:
                    valid_replay_paths.append(replay_path_obj)

        metadata = {
            "dataset_format": DATASET_FORMAT,
            "replay_path": str(replay_input),
            "replay_paths": [str(p) for p in valid_replay_paths],
            "replay_paths_requested": [str(p) for p in replay_paths],
            "action_space": ActionSpaceSpec().as_dict(),
            "geometry_horizon": self.horizon,
            "device": self.device,
            "geometry_device": self.device,
            "inference_batch_size": self.batch_size,
            "backend": self.backend,
            "target_inference_mode": LITE_TARGET_INFERENCE_MODE if self.backend == "lite" else "batched_exact_first_hit_with_angular_fallback",
            "mask_mode": LITE_MASK_MODE if self.backend == "lite" else "exact-geometry",
            "pair_eta_mode": LITE_PAIR_ETA_MODE if self.backend == "lite" else "exact-geometry",
            "replay_action_alignment": "action at replay index t is paired with observation at replay index t-1",
            "include_loser_actions": self.include_loser_actions,
            "max_replay_files": self.max_replay_files,
            "write_debug_jsonl": self.write_debug_jsonl,
            "perspective_canonicalization": {"enabled": self.canonicalize_perspective, "frame": "p0", "rotation": "-2*pi*player_id/num_players"},
            "proportion_correction": {
                "enabled": bool(self.balance_proportions),
                "balance_seed": int(self.balance_seed),
                "requested": {"noop_ratio": float(self.noop_ratio), "op_ratio": float(self.op_ratio)},
                "replay_selection": replay_selection_report,
            },
            "parallel": {"enabled": False, "worker_count": 1, "shards": 1},
            "planet_feature_names": PLANET_FEATURE_NAMES,
            "global_feature_names": GLOBAL_FEATURE_NAMES,
            "target_state_feature_names": TARGET_STATE_FEATURE_NAMES,
            "pair_feature_names": PAIR_FEATURE_NAMES,
            "stats": _stats_with_required_train_keys(stats),
            "input_replay_files": {
                "requested": len(replay_paths),
                "used": len(valid_replay_paths),
                "skipped_invalid_json": len(invalid_json_paths),
                "skipped_unreadable": len(unreadable_paths),
                "skipped_invalid_json_paths": sorted(invalid_json_paths),
                "skipped_unreadable_paths": sorted(unreadable_paths),
            },
            "target_inference_methods": dict(inference_methods),
            "amount_bin_counts": {AMOUNT_BIN_NAMES[int(k)]: int(v) for k, v in amount_bins.items() if int(k) < len(AMOUNT_BIN_NAMES)},
        }
        finalized = writer.finalize(extra_metadata=metadata)
        if self.balance_proportions:
            return _balance_source_turn_dataset(
                out_dir,
                finalized,
                balance_seed=self.balance_seed,
                noop_ratio=self.noop_ratio,
                op_ratio=self.op_ratio,
                replay_selection_report=replay_selection_report,
            )
        return finalized


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, help="Replay JSON file or directory containing replay JSON files.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon", type=int, default=160)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256, help="Maximum number of valid launches to exact-simulate per GPU/CPU batch.")
    ap.add_argument("--winner-only", action="store_true")
    ap.add_argument("--max-files", type=int, default=None, help="Maximum number of replay files to include from a replay directory.")
    ap.add_argument("--workers", type=int, default=6, help="Number of replay files to process in parallel. CUDA exact builds fall back to serial to avoid multiple GPU contexts.")
    ap.add_argument("--write-debug-jsonl", action="store_true", help="Write debug JSONL files under debug/.")
    ap.add_argument("--backend", choices=("lite", "exact"), default="lite", help="Dataset build backend. lite uses orbit_lite movement-cache heuristics; exact keeps the old simulator path.")
    ap.add_argument("--no-canonicalize-perspective", action="store_true", help="Disable player-perspective canonicalization. Not recommended for BC/PPO because it can reintroduce map/seat bias.")
    ap.add_argument("--no-balance-proportions", action="store_true", help="Disable default replay and source-turn proportion correction.")
    ap.add_argument("--balance-seed", type=int, default=42, help="Seed for deterministic replay/sample downsampling.")
    ap.add_argument("--noop-ratio", type=float, default=0.6, help="Requested final noop source-turn ratio before normalization.")
    ap.add_argument("--op-ratio", type=float, default=0.4, help="Requested final non-noop source-turn ratio before normalization.")
    args = ap.parse_args()
    builder = DatasetBuilder(
        horizon=args.horizon,
        device=args.device,
        batch_size=args.batch_size,
        include_loser_actions=not args.winner_only,
        max_replay_files=args.max_files,
        workers=args.workers,
        write_debug_jsonl=args.write_debug_jsonl,
        backend=args.backend,
        canonicalize_perspective=not args.no_canonicalize_perspective,
        balance_proportions=not args.no_balance_proportions,
        balance_seed=args.balance_seed,
        noop_ratio=args.noop_ratio,
        op_ratio=args.op_ratio,
    )
    meta = builder.build_from_replay(args.replay, args.out_dir)
    print(json.dumps({"out_dir": args.out_dir, "stats": meta["stats"], "target_inference_methods": meta["target_inference_methods"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
