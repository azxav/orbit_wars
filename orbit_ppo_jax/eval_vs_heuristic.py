from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from orbit_bc_eval.base_agents import make_opponent
from orbit_jax_env.state import state_from_observation
from orbit_training_prep.schema import P_MAX

from .actions import action_rows_from_choices
from .bc_policy import bc_forward
from .checkpointing import load_jax_checkpoint
from .features import NOOP_TARGET_SLOT, build_bc_features_for_seat
from .train import _safe_amount_logits, _safe_target_logits, _source_batch


def _final_state(env) -> tuple[list[float], list[str]]:
    final = env.steps[-1]
    rewards = [float(step.reward or 0.0) for step in final]
    statuses = [str(step.status) for step in final]
    return rewards, statuses


class JaxCheckpointAgent:
    def __init__(self, checkpoint: str | Path):
        params, config, _metrics = load_jax_checkpoint(checkpoint)
        self.params = params
        self.config = config
        self.bc_config = config.get("bc_model_config") or {}

    def __call__(self, obs: dict[str, Any], config: Any) -> list[list[Any]]:
        player_id = int(obs.get("player", 0) or 0)
        players = int(getattr(config, "players", getattr(config, "num_players", self.config.get("players", 2))) or self.config.get("players", 2))
        episode_steps = int(getattr(config, "episodeSteps", self.config.get("steps", 500)) or self.config.get("steps", 500))
        state = state_from_observation(obs, num_players=players, episode_steps=episode_steps)
        features = build_bc_features_for_seat(state, player_id)
        out = bc_forward(self.params["bc"], _source_batch(features), self.bc_config)
        target_logits = _safe_target_logits(out["target_logits"], features.target_mask)
        target_idx = jnp.argmax(target_logits, axis=-1).astype(jnp.int32)
        amount_out = bc_forward(self.params["bc"], _source_batch(features, target_idx), self.bc_config)
        chosen_amount_mask = features.amount_mask[jnp.arange(P_MAX), jnp.clip(target_idx, 0, P_MAX)]
        amount_logits = _safe_amount_logits(amount_out["amount_logits"], chosen_amount_mask)
        amount_idx = jnp.argmax(amount_logits, axis=-1).astype(jnp.int32)
        rows = np.asarray(action_rows_from_choices(state, player_id, target_idx, amount_idx))
        return [[int(row[0]), float(row[1]), int(row[2])] for row in rows if int(row[2]) > 0]


def evaluate(
    checkpoint: str | Path,
    heuristic_path: str | Path,
    *,
    games: int = 4,
    players: int = 4,
    out_dir: str | Path = "ppo_eval_runs/jax_eval",
    seed: int = 42,
) -> dict[str, Any]:
    try:
        from kaggle_environments import make
    except ImportError as exc:
        raise RuntimeError("kaggle_environments is required for heuristic evaluation") from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rewards: list[float] = []
    rows: list[dict[str, Any]] = []
    for game_idx in range(int(games)):
        learner = JaxCheckpointAgent(checkpoint)
        agents = [learner]
        for _ in range(1, int(players)):
            agents.append(make_opponent("heuristic_path", heuristic_path=heuristic_path))
        env = make("orbit_wars", configuration={"episodeSteps": 500, "seed": int(seed) + game_idx}, debug=False)
        env.run(agents)
        final_rewards, statuses = _final_state(env)
        reward = float(final_rewards[0])
        rewards.append(reward)
        row = {"game": game_idx, "reward": reward, "status": statuses[0], "players": int(players)}
        rows.append(row)
        with open(out / "games.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "checkpoint": str(checkpoint),
        "heuristic_path": str(heuristic_path),
        "games": int(games),
        "players": int(players),
        "average_final_reward": float(np.mean(rewards)) if rewards else 0.0,
        "min_final_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_final_reward": float(np.max(rewards)) if rewards else 0.0,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate a JAX PPO checkpoint against orbit_wars_base.py.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--heuristic_path", default="orbit_wars_base.py")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--players", type=int, default=4, choices=[2, 4])
    ap.add_argument("--out_dir", default="ppo_eval_runs/jax_eval")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate(args.checkpoint, args.heuristic_path, games=args.games, players=args.players, out_dir=args.out_dir, seed=args.seed)
    if argv is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return None
    return summary


if __name__ == "__main__":
    main()
