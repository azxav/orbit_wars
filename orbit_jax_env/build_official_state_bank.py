from __future__ import annotations

import argparse
import json
from pathlib import Path

from .official_state_dataset import generate_official_initial_states, save_state_bank


def _parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start_s, stop_s = value.split(":", 1)
        return list(range(int(start_s), int(stop_s)))
    return [int(part) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a compact Orbit Wars official initial-state bank.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--players", type=int, default=4, choices=[2, 4])
    parser.add_argument("--seeds", required=True, help="Seed range like 0:1024 or comma list like 0,1,2.")
    parser.add_argument("--episode_steps", type=int, default=500)
    parser.add_argument("--ship_speed", type=float, default=6.0)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    states, metadata = generate_official_initial_states(
        players=int(args.players),
        seeds=_parse_seeds(args.seeds),
        episode_steps=int(args.episode_steps),
        ship_speed=float(args.ship_speed),
    )
    save_state_bank(Path(args.out), states, metadata)
    summary = {**metadata, "out": str(args.out), "state_count": int(states.step.shape[0])}
    if argv is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
