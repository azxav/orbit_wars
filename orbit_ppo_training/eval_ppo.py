from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .checkpointing import load_ppo_checkpoint
from .rollout_worker import collect_rollouts


LOGGER = logging.getLogger(__name__)


def save_game_replay(env, replay_dir, game_idx, save_html=True):
    replay_dir = Path(replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)

    json_path = replay_dir / f"game_{game_idx:03d}.json"
    html_path = replay_dir / f"game_{game_idx:03d}.html"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(env.toJSON(), f)

    html_saved = False
    if save_html:
        try:
            html = env.render(mode="html")
        except Exception as exc:
            LOGGER.warning("Failed to render HTML replay for game %s: %s", game_idx, exc)
        else:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            html_saved = True

    return {"json": str(json_path), "html": str(html_path) if html_saved else None}


def _write_eval_outputs(out: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    for name in ("eval_summary.json", "summary.json"):
        with open(out / name, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    for name in ("eval_rows.jsonl", "games.jsonl"):
        with open(out / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate(
    checkpoint: str,
    *,
    opponent: str,
    players: int,
    num_games: int,
    out_dir: str,
    seed: int = 1000,
    device: str = "cpu",
    save_replays: int = 0,
    save_html_replays: bool = False,
    replay_dir: str | None = None,
) -> dict:
    policy, config, _ = load_ppo_checkpoint(checkpoint, device=device)
    config.opponent = opponent
    config.players = int(players)
    config.device = device
    config.seed = int(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected_replays = max(0, int(save_replays))
    replay_out = Path(replay_dir) if replay_dir is not None else out / "replays"
    replay_paths: list[dict[str, str | None]] = []

    def replay_callback(env, game_idx: int, row: dict[str, Any]) -> None:
        if game_idx >= selected_replays:
            return
        paths = save_game_replay(env, replay_out, game_idx, save_html=bool(save_html_replays))
        row["replay_paths"] = paths
        replay_paths.append(paths)

    rollout = collect_rollouts(
        policy,
        config,
        games=int(num_games),
        deterministic=True,
        seed_start=int(seed),
        replay_callback=replay_callback if selected_replays > 0 else None,
    )
    summary = dict(rollout.summary)
    if selected_replays > 0:
        summary["replay_dir"] = str(replay_out)
        summary["replay_paths"] = replay_paths
    _write_eval_outputs(out, summary, rollout.rows)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate a PPO Orbit Wars checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="orbit_wars_base", choices=["random", "passive", "simple_expand", "orbit_wars_base", "heuristic_path"])
    ap.add_argument("--players", type=int, default=4, choices=[2, 4])
    ap.add_argument("--num_games", type=int, default=50)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save_replays", type=int, default=0, help="Save replay JSON for the first N evaluated games.")
    ap.add_argument("--save_html_replays", action="store_true", help="Also save interactive HTML replays for selected games.")
    ap.add_argument("--replay_dir", default=None, help="Replay output directory (default: <out_dir>/replays).")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    summary = evaluate(
        args.checkpoint,
        opponent=args.opponent,
        players=args.players,
        num_games=args.num_games,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
        save_replays=args.save_replays,
        save_html_replays=args.save_html_replays,
        replay_dir=args.replay_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
