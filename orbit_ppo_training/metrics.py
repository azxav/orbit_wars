from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
    return sum(vals) / max(1, len(vals))


def sum_count_maps(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        for name, value in (row.get(key, {}) or {}).items():
            out[str(name)] = out.get(str(name), 0) + int(value or 0)
    return dict(sorted(out.items()))


def summarize_rollout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "winrate": mean(rows, "win"),
        "average_rank": mean(rows, "rank"),
        "average_final_reward": mean(rows, "reward"),
        "average_owned_planets": mean(rows, "avg_owned_planets"),
        "average_total_ships": mean(rows, "avg_total_ships"),
        "average_final_ships": mean(rows, "final_ship_count"),
        "average_launches_per_game": mean(rows, "launches"),
        "predicted_launch_rate": mean(rows, "predicted_launch_rate"),
        "early_launch_rate": mean(rows, "early_launch_rate"),
        "illegal_action_count": sum(int(r.get("illegal_actions", 0) or 0) for r in rows),
        "timeout_count": sum(int(r.get("timeout_count", 0) or 0) for r in rows),
        "skipped_decoded_actions": sum(int(r.get("skipped_invalid_decoded_actions", 0) or 0) for r in rows),
        "skip_reason_counts": sum_count_maps(rows, "skip_reason_counts"),
        "opening_prediction_target_counts": sum_count_maps(rows, "opening_prediction_target_counts"),
        "opening_prediction_amount_counts": sum_count_maps(rows, "opening_prediction_amount_counts"),
        "opening_prediction_target_amount_counts": sum_count_maps(rows, "opening_prediction_target_amount_counts"),
        "average_entropy": mean(rows, "average_entropy"),
    }


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
