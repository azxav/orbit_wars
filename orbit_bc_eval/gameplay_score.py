from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rank_score(rank: Any, players: Any) -> float:
    player_count = max(1, int(_safe_float(players, 1.0)))
    if player_count <= 1:
        return 1.0
    value = min(max(1.0, _safe_float(rank, float(player_count))), float(player_count))
    return (float(player_count) - value) / float(player_count - 1)


def _bounded_score(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if value > 0.0 else 0.0
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))


def score_game_rows(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    rewards = [_safe_float(r.get("reward")) for r in rows]
    owned = [_safe_float(r.get("owned_planets_auc", r.get("avg_owned_planets"))) for r in rows]
    ships = [_safe_float(r.get("total_ships_auc", r.get("avg_total_ships"))) for r in rows]
    reward_lo, reward_hi = min(rewards), max(rewards)
    owned_lo, owned_hi = min(owned), max(owned)
    ships_lo, ships_hi = min(ships), max(ships)
    scores = []
    for row in rows:
        row_rank_score = rank_score(row.get("rank"), row.get("players", 2))
        reward_score = _bounded_score(_safe_float(row.get("reward")), reward_lo, reward_hi)
        owned_score = _bounded_score(_safe_float(row.get("owned_planets_auc", row.get("avg_owned_planets"))), owned_lo, owned_hi)
        ships_score = _bounded_score(_safe_float(row.get("total_ships_auc", row.get("avg_total_ships"))), ships_lo, ships_hi)
        decode_success_rate = min(1.0, max(0.0, _safe_float(row.get("decode_success_rate"))))
        invalid_decode_rate = min(1.0, max(0.0, _safe_float(row.get("invalid_decode_rate"))))
        scores.append(
            0.35 * row_rank_score
            + 0.25 * reward_score
            + 0.15 * owned_score
            + 0.10 * ships_score
            + 0.10 * decode_success_rate
            - 0.05 * invalid_decode_rate
        )
    return sum(scores) / len(scores)


def load_games_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        p = p / "games.jsonl"
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_eval_dir(path: str | Path | None) -> float | None:
    if not path:
        return None
    base = Path(path)
    if not base.exists():
        return None
    rows_2p = load_games_jsonl(base / "2p")
    rows_4p = load_games_jsonl(base / "4p")
    if rows_2p or rows_4p:
        score_2p = score_game_rows(rows_2p)
        score_4p = score_game_rows(rows_4p)
        if rows_2p and rows_4p:
            return 0.55 * score_2p + 0.45 * score_4p
        return score_2p if rows_2p else score_4p
    rows = load_games_jsonl(base)
    if not rows:
        return None
    return score_game_rows(rows)
