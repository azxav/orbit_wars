from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from . import bc_agent_runtime
from .base_agents import make_opponent
from .bc_agent_runtime import validate_env_move
from .config import DEFAULT_ACT_TIMEOUT, DEFAULT_ENVIRONMENT, DEFAULT_EPISODE_STEPS, DEFAULT_GEOMETRY_HORIZON
from .eval_report import write_eval_report
from .rollout_metrics import RolloutMetrics


class RecordingAgent:
    def __init__(self, fn: Callable, *, player_id: int | None = None, is_bc: bool = False):
        self.fn = fn
        self.player_id = player_id
        self.is_bc = is_bc
        self.records: list[dict[str, Any]] = []

    def __call__(self, obs, config):
        actions = self.fn(obs, config)
        obs_dict = dict(obs or {})
        player_id = int(obs_dict.get("player", self.player_id if self.player_id is not None else 0) or 0)
        illegal = 0
        if self.is_bc:
            illegal = sum(1 for move in actions if not validate_env_move(obs_dict, player_id, move).ok)
        self.records.append(
            {
                "step": int(obs_dict.get("step", len(self.records)) or 0),
                "player_id": player_id,
                "obs": obs_dict,
                "obs_summary": summarize_obs(obs_dict, player_id),
                "actions": actions,
                "illegal_actions": illegal,
                "debug": bc_agent_runtime.get_last_debug() if self.is_bc else {},
            }
        )
        return actions


def summarize_obs(obs: dict[str, Any], player_id: int) -> dict[str, Any]:
    owned = 0
    total_ships = 0.0
    neutral = 0
    enemy = 0
    for p in obs.get("planets", []):
        if len(p) < 7:
            continue
        owner = int(p[1])
        if owner == int(player_id):
            owned += 1
            total_ships += max(0.0, float(p[5]))
        elif owner < 0:
            neutral += 1
        else:
            enemy += 1
    return {
        "step": int(obs.get("step", 0) or 0),
        "owned_planets": owned,
        "owned_ships": total_ships,
        "neutral_planets": neutral,
        "enemy_planets": enemy,
        "fleets": len(obs.get("fleets", []) or []),
    }


def _state_value(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _import_make():
    try:
        from kaggle_environments import make
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "kaggle_environments is not installed. Install it in this environment before running local matches: "
            "python -m pip install kaggle-environments"
        ) from exc
    return make


def _final_state(env) -> tuple[list[float], list[str], dict[str, Any] | None]:
    rewards: list[float] = []
    statuses: list[str] = []
    final_obs = None
    try:
        final_step = env.steps[-1]
        for state in final_step:
            rewards.append(float(_state_value(state, "reward", 0.0) or 0.0))
            statuses.append(str(_state_value(state, "status", "")))
        if final_step:
            final_obs = _state_value(final_step[0], "observation", None)
    except Exception:
        pass
    return rewards, statuses, final_obs


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "episodeSteps": int(args.episode_steps),
        "actTimeout": float(args.act_timeout),
    }


def _bc_seat_for(game_index: int, players: int) -> int:
    return int(game_index % players)


def run_games(args: argparse.Namespace) -> dict[str, Any]:
    make = _import_make()
    opponent_fn = make_opponent(args.opponent, heuristic_path=args.heuristic_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if args.debug_game:
        debug_dir.mkdir(parents=True, exist_ok=True)
    bc_agent_runtime.configure_bc_agent(
        checkpoint=args.bc_checkpoint,
        device=args.device,
        geometry_horizon=args.geometry_horizon,
        debug=args.debug_game,
    )
    rows: list[dict[str, Any]] = []
    seats: list[int] = []
    notes: list[str] = []
    for game_index in range(int(args.num_games)):
        seed = int(args.seed_start) + game_index
        bc_seat = _bc_seat_for(game_index, int(args.players))
        seats.append(bc_seat)
        bc_agent_runtime.reset_runtime_state()
        bc_wrapper = RecordingAgent(bc_agent_runtime.agent, player_id=bc_seat, is_bc=True)
        agents: list[Any] = []
        for seat in range(int(args.players)):
            agents.append(bc_wrapper if seat == bc_seat else RecordingAgent(opponent_fn, player_id=seat))
        env = make(args.environment, configuration=_configuration(args), debug=bool(args.debug_game))
        try:
            if hasattr(env, "seed"):
                env.seed(seed)
        except Exception:
            notes.append(f"Environment did not accept seed {seed}.")
        env.run(agents)
        rewards, statuses, final_obs = _final_state(env)
        metrics = RolloutMetrics(game_id=f"game_{game_index:05d}", bc_player_id=bc_seat, players=args.players, opponent=args.opponent)
        for record in bc_wrapper.records:
            metrics.record_observation(dict(record.get("obs", {}) or {}))
            metrics.record_step(
                step=int(record["step"]),
                actions=list(record.get("actions", []) or []),
                illegal_actions=int(record.get("illegal_actions", 0) or 0),
                runtime_debug=dict(record.get("debug", {}) or {}),
            )
        row = metrics.finalize(rewards=rewards, statuses=statuses, final_obs=final_obs)
        row["seed"] = seed
        rows.append(row)
        if args.debug_game:
            debug_payload = {
                "game_id": row["game_id"],
                "seed": seed,
                "bc_seat": bc_seat,
                "bc_records": bc_wrapper.records,
            }
            try:
                debug_payload["replay"] = env.render(mode="json")
            except Exception as exc:
                debug_payload["replay_error"] = f"{type(exc).__name__}: {exc}"
            with open(debug_dir / f"{row['game_id']}.json", "w", encoding="utf-8") as f:
                json.dump(debug_payload, f, indent=2, sort_keys=True)
    return write_eval_report(rows, out_dir=out_dir, opponent=args.opponent, players=args.players, bc_seats=seats, notes=notes)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run local Orbit Wars matches for a trained BC policy.")
    ap.add_argument("--bc_checkpoint", required=True)
    ap.add_argument("--opponent", choices=["random", "passive", "simple_expand", "heuristic_path"], default="simple_expand")
    ap.add_argument("--num_games", type=int, default=20)
    ap.add_argument("--players", type=int, choices=[2, 4], default=4)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    ap.add_argument("--episode_steps", type=int, default=DEFAULT_EPISODE_STEPS)
    ap.add_argument("--act_timeout", type=float, default=DEFAULT_ACT_TIMEOUT)
    ap.add_argument("--geometry_horizon", type=int, default=DEFAULT_GEOMETRY_HORIZON)
    ap.add_argument("--heuristic_path")
    ap.add_argument("--debug_game", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_games(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
