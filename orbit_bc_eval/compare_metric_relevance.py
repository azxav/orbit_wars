from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from .gameplay_score import load_games_jsonl


_ID_KEYS = {"game_id", "players", "seed", "bc_seat", "opponent"}
_NON_METRIC_KEYS = _ID_KEYS | {"win", "skip_reason_counts", "opening_prediction_target_counts", "opening_prediction_amount_counts", "opening_prediction_target_amount_counts"}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if all(k in row for k in ("players", "seed", "bc_seat", "opponent")):
        return (row.get("players"), row.get("seed"), row.get("bc_seat"), row.get("opponent"))
    return (row.get("game_id"),)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = sum((x - mx) ** 2 for x in rx) ** 0.5
    den_y = sum((y - my) ** 2 for y in ry) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def _metric_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in _NON_METRIC_KEYS or key in {"reward", "rank"}:
                continue
            if _safe_float(value) is not None:
                names.add(key)
    return sorted(names)


def compare(paths: list[str | Path]) -> list[dict[str, Any]]:
    runs = [load_games_jsonl(path) for path in paths]
    if len(runs) < 2 or not runs[0]:
        return []
    baseline = {_pair_key(row): row for row in runs[0]}
    candidates = []
    for run in runs[1:]:
        lookup = {_pair_key(row): row for row in run}
        for key, old in baseline.items():
            new = lookup.get(key)
            if new is not None:
                candidates.append((old, new))
    if not candidates:
        return []

    reward_delta: list[float | None] = []
    rank_delta: list[float | None] = []
    for old, new in candidates:
        old_reward = _safe_float(old.get("reward"))
        new_reward = _safe_float(new.get("reward"))
        old_rank = _safe_float(old.get("rank"))
        new_rank = _safe_float(new.get("rank"))
        reward_delta.append(None if old_reward is None or new_reward is None else new_reward - old_reward)
        rank_delta.append(None if old_rank is None or new_rank is None else new_rank - old_rank)
    metric_names = _metric_names([row for pair in candidates for row in pair])
    output = []
    for metric in metric_names:
        deltas: list[float] = []
        rewards: list[float] = []
        ranks: list[float] = []
        for idx, (old, new) in enumerate(candidates):
            old_value = _safe_float(old.get(metric))
            new_value = _safe_float(new.get(metric))
            if old_value is None or new_value is None or reward_delta[idx] is None or rank_delta[idx] is None:
                continue
            deltas.append(new_value - old_value)
            rewards.append(float(reward_delta[idx]))
            ranks.append(float(rank_delta[idx]))
        corr_reward = _spearman(deltas, rewards)
        corr_rank = _spearman(deltas, ranks)
        if corr_reward >= 0.0 and corr_rank <= 0.0:
            direction = "higher"
            keep_drop = "keep"
        elif corr_reward <= 0.0 and corr_rank >= 0.0:
            direction = "lower"
            keep_drop = "keep"
        else:
            direction = "mixed"
            keep_drop = "drop"
        output.append(
            {
                "metric": metric,
                "spearman_corr_with_reward": round(corr_reward, 6),
                "spearman_corr_with_rank": round(corr_rank, 6),
                "direction": direction,
                "keep_drop": keep_drop,
            }
        )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare paired BC eval metric deltas against reward and rank movement.")
    parser.add_argument("runs", nargs="+", help="Two or more games.jsonl files or directories containing games.jsonl.")
    args = parser.parse_args(argv)
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["metric", "spearman_corr_with_reward", "spearman_corr_with_rank", "direction", "keep_drop"],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in compare(args.runs):
        writer.writerow(row)


if __name__ == "__main__":
    main()
