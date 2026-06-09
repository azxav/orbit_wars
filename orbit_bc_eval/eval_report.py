from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .gameplay_score import score_game_rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(r.get(key, 0.0) or 0.0) for r in rows]
    return sum(values) / len(values) if values else 0.0


def _sum_count_maps(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        for name, value in (row.get(key, {}) or {}).items():
            out[str(name)] = out.get(str(name), 0) + int(value or 0)
    return dict(sorted(out.items()))


def write_eval_report(
    game_rows: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    opponent: str,
    players: int,
    bc_seats: list[int],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    notes_out = list(notes or [])
    total_launch_decisions = sum(float(r.get("launches", 0) or 0) + float(r.get("no_op_source_decisions", 0) or 0) for r in game_rows)
    early_launches = sum(float(r.get("launches_0_100", 0) or 0) for r in game_rows)
    total_launches = sum(int(r.get("launches", 0) or 0) for r in game_rows)
    total_predicted_launches = sum(int(r.get("predicted_launches", 0) or 0) for r in game_rows)
    total_no_ops = sum(int(r.get("no_op_source_decisions", 0) or 0) for r in game_rows)
    total_returned_moves = sum(int(r.get("actual_returned_move_count", 0) or 0) for r in game_rows)
    if game_rows and _mean(game_rows, "launches") <= 0:
        notes_out.append("BC returned no launches on average; check no-op bias and checkpoint path.")
    if sum(int(r.get("illegal_actions", 0) or 0) for r in game_rows):
        notes_out.append("Illegal actions were observed; inspect games.jsonl and debug traces.")
    summary = {
        "num_games": len(game_rows),
        "opponent": opponent,
        "players": int(players),
        "bc_seat_positions_tested": sorted(set(int(s) for s in bc_seats)),
        "winrate": sum(1 for r in game_rows if bool(r.get("win"))) / len(game_rows) if game_rows else 0.0,
        "average_rank": _mean(game_rows, "rank"),
        "average_launches_per_game": _mean(game_rows, "launches"),
        "early_launch_rate": early_launches / max(1.0, total_launch_decisions),
        "total_launch_decisions": float(total_launch_decisions),
        "total_launches": int(total_launches),
        "early_launches_0_100": int(early_launches),
        "total_predicted_launches": int(total_predicted_launches),
        "total_no_op_source_decisions": int(total_no_ops),
        "total_actual_returned_move_count": int(total_returned_moves),
        "timeout_count": int(sum(int(r.get("timeout_count", 0) or 0) for r in game_rows)),
        "illegal_action_count": int(sum(int(r.get("illegal_actions", 0) or 0) for r in game_rows)),
        "average_final_reward": _mean(game_rows, "reward"),
        "average_owned_planets": _mean(game_rows, "avg_owned_planets"),
        "average_final_owned_planets": _mean(game_rows, "final_owned_planets"),
        "average_owned_planets_auc": _mean(game_rows, "owned_planets_auc"),
        "average_owned_planets_auc_0_100": _mean(game_rows, "owned_planets_auc_0_100"),
        "average_planet_control_delta": _mean(game_rows, "planet_control_delta"),
        "average_first_capture_step": _mean(game_rows, "first_capture_step"),
        "average_capture_efficiency": _mean(game_rows, "capture_efficiency"),
        "average_total_ships": _mean(game_rows, "avg_total_ships"),
        "average_total_ships_auc": _mean(game_rows, "total_ships_auc"),
        "average_total_ships_auc_0_100": _mean(game_rows, "total_ships_auc_0_100"),
        "average_ship_delta_final_vs_initial": _mean(game_rows, "ship_delta_final_vs_initial"),
        "average_final_ship_count": _mean(game_rows, "final_ship_count"),
        "average_planets_captured": _mean(game_rows, "planets_captured"),
        "average_predicted_launch_rate": _mean(game_rows, "predicted_launch_rate"),
        "average_decode_success_rate": _mean(game_rows, "decode_success_rate"),
        "average_invalid_decode_rate": _mean(game_rows, "invalid_decode_rate"),
        "average_actual_launch_rate": _mean(game_rows, "actual_launch_rate"),
        "average_noop_decision_rate": _mean(game_rows, "noop_decision_rate"),
        "average_actual_returned_move_count": _mean(game_rows, "actual_returned_move_count"),
        "gameplay_score": score_game_rows(game_rows),
        "total_skipped_invalid_decoded_actions": int(sum(int(r.get("skipped_invalid_decoded_actions", 0) or 0) for r in game_rows)),
        "skip_reason_counts": _sum_count_maps(game_rows, "skip_reason_counts"),
        "opening_prediction_target_counts": _sum_count_maps(game_rows, "opening_prediction_target_counts"),
        "opening_prediction_amount_counts": _sum_count_maps(game_rows, "opening_prediction_amount_counts"),
        "opening_prediction_target_amount_counts": _sum_count_maps(game_rows, "opening_prediction_target_amount_counts"),
        "notes_warnings": notes_out,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(out / "games.jsonl", "w", encoding="utf-8") as f:
        for row in game_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    fieldnames = sorted({k for row in game_rows for k in row.keys()}) or ["game_id"]
    with open(out / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in game_rows:
            writer.writerow(row)
    return summary
