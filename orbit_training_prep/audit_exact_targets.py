from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .dataset_builder import resolve_replay_paths
from .exact_target_sim import first_hit_for_launch
from .replay_io import iter_player_steps, load_replay
from .schema import NOOP_TARGET_ID


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def obs_key(episode_id: Any, step_index: Any, player_id: Any) -> tuple[str, int, int]:
    return (str(episode_id), int(step_index), int(player_id))


def build_obs_index(replay_dir: str | Path, wanted: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not wanted:
        return by_key
    remaining = set(wanted)
    for replay_path in resolve_replay_paths(replay_dir):
        replay = load_replay(replay_path)
        episode_id = str(replay.get("info", {}).get("EpisodeId", replay.get("id", "unknown")))
        if not any(k[0] == episode_id for k in remaining):
            continue
        for sample in iter_player_steps(replay):
            key = obs_key(sample["episode_id"], sample["step_index"], sample["player_id"])
            if key in remaining:
                by_key[key] = sample["obs"]
                remaining.remove(key)
                if not remaining:
                    return by_key
    return by_key


def sampled_rows(rows: list[dict[str, Any]], sample_size: int, *, seed: int, focus_source_id: int, focus_target_id: int) -> list[dict[str, Any]]:
    valid_rows = [r for r in rows if r.get("valid_source")]
    focus = [
        r
        for r in valid_rows
        if int(r.get("source_planet_id", -1)) == int(focus_source_id)
        and int(r.get("inferred_target_id", NOOP_TARGET_ID)) == int(focus_target_id)
    ]
    pool = [r for r in valid_rows if r not in focus]
    rng = random.Random(seed)
    rng.shuffle(pool)
    budget = max(0, int(sample_size) - len(focus))
    return focus + pool[:budget]


def summarize(audited: list[dict[str, Any]]) -> dict[str, Any]:
    denom = max(1, len(audited))
    matches = [r for r in audited if r["target_matches"]]
    by_actual = Counter(str(r["actual_first_hit_type"]) for r in audited)
    mismatch_examples = [r for r in audited if not r["target_matches"]][:25]
    return {
        "audited_launches": len(audited),
        "target_matches": len(matches),
        "target_mismatches": len(audited) - len(matches),
        "target_match_rate": len(matches) / denom,
        "actual_first_hit_type_counts": dict(by_actual),
        "mismatch_examples": mismatch_examples,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    launch_rows = read_jsonl(dataset_dir / "launch_rows.jsonl")
    rows = sampled_rows(
        launch_rows,
        args.sample_size,
        seed=args.seed,
        focus_source_id=args.focus_source_id,
        focus_target_id=args.focus_target_id,
    )
    wanted = {obs_key(r["episode_id"], r["step_index"], r["player_id"]) for r in rows}
    obs_by_key = build_obs_index(args.replay_dir, wanted)

    audited: list[dict[str, Any]] = []
    missing_obs = 0
    for row in rows:
        key = obs_key(row["episode_id"], row["step_index"], row["player_id"])
        obs = obs_by_key.get(key)
        if obs is None:
            missing_obs += 1
            continue
        hit = first_hit_for_launch(obs, int(row["player_id"]), row, horizon=args.horizon)
        inferred_id = int(row.get("inferred_target_id", NOOP_TARGET_ID))
        target_matches = hit["hit_type"] in {"planet", "comet"} and hit["hit_id"] == inferred_id
        audited.append(
            {
                "launch_uid": row.get("launch_uid"),
                "episode_id": row.get("episode_id"),
                "step_index": row.get("step_index"),
                "player_id": row.get("player_id"),
                "source_planet_id": int(row.get("source_planet_id", -1)),
                "raw_angle": float(row.get("raw_angle", 0.0)),
                "ships": int(row.get("ships", 0)),
                "inferred_target_id": inferred_id,
                "inferred_target_slot": int(row.get("inferred_target_slot", -1)),
                "actual_first_hit_type": hit["hit_type"],
                "actual_first_hit_id": hit["hit_id"],
                "actual_first_hit_slot": hit["hit_slot"],
                "actual_first_hit_eta": hit["eta"],
                "target_matches": bool(target_matches),
                "reason": hit["reason"],
            }
        )

    focus_rows = [
        r
        for r in audited
        if r["source_planet_id"] == int(args.focus_source_id)
        and r["inferred_target_id"] == int(args.focus_target_id)
    ]
    report = {
        "config": {
            "dataset_dir": str(dataset_dir),
            "replay_dir": str(args.replay_dir),
            "sample_size": int(args.sample_size),
            "seed": int(args.seed),
            "horizon": int(args.horizon),
            "focus_source_id": int(args.focus_source_id),
            "focus_target_id": int(args.focus_target_id),
        },
        "counts": {
            "launch_rows_total": len(launch_rows),
            "sampled_rows_requested": int(args.sample_size),
            "sampled_rows_selected": len(rows),
            "missing_replay_observations": missing_obs,
        },
        "summary": summarize(audited),
        "focus_label_summary": summarize(focus_rows),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--replay_dir", required=True)
    ap.add_argument("--sample_size", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--focus_source_id", type=int, default=3)
    ap.add_argument("--focus_target_id", type=int, default=9)
    args = ap.parse_args()
    report = audit(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps({"out": str(out), "summary": report["summary"], "focus_label_summary": report["focus_label_summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
