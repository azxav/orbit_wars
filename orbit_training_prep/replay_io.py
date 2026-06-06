from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .schema import MAX_STEP_DEFAULT, normalize_replay_obs


def load_replay(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def final_rewards(replay: dict[str, Any]) -> list[float]:
    rewards = replay.get("rewards")
    if isinstance(rewards, list):
        return [float(x) if x is not None else 0.0 for x in rewards]
    steps = replay.get("steps", [])
    if not steps:
        return []
    return [float(s.get("reward", 0.0) or 0.0) for s in steps[-1]]


def iter_player_steps(replay: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one normalized player-perspective training step.

    Kaggle replay rows store `action` beside the observation after that action has already
    affected the board. Therefore, for supervised learning, action at replay index `t` is
    paired with the observation from replay index `t-1` for the same player.
    """
    steps = replay.get("steps", [])
    cfg = replay.get("configuration", {}) or {}
    episode_steps = int(cfg.get("episodeSteps", MAX_STEP_DEFAULT) or MAX_STEP_DEFAULT)
    rewards = final_rewards(replay)
    episode_id = replay.get("info", {}).get("EpisodeId", replay.get("id", "unknown"))
    team_names = replay.get("info", {}).get("TeamNames", []) or []
    if len(steps) < 2:
        return
    for replay_index in range(1, len(steps)):
        prev_step = steps[replay_index - 1]
        cur_step = steps[replay_index]
        if not isinstance(prev_step, list) or not isinstance(cur_step, list):
            continue
        for player_id, cur_row in enumerate(cur_step):
            if player_id >= len(prev_step):
                continue
            prev_row = prev_step[player_id]
            if not isinstance(cur_row, dict) or not isinstance(prev_row, dict):
                continue
            obs_raw = prev_row.get("observation") or {}
            obs = normalize_replay_obs(
                obs_raw,
                player_id=player_id,
                step_index=replay_index - 1,
                episode_steps=episode_steps,
            )
            yield {
                "episode_id": str(episode_id),
                "step_index": int(replay_index - 1),
                "action_replay_index": int(replay_index),
                "obs_step": int(obs.get("step", replay_index - 1)),
                "player_id": int(player_id),
                "player_name": str(team_names[player_id]) if player_id < len(team_names) else "",
                "status": str(prev_row.get("status", cur_row.get("status", ""))),
                "reward_at_step": float(prev_row.get("reward", 0.0) or 0.0),
                "final_reward": float(rewards[player_id]) if player_id < len(rewards) else 0.0,
                "action": cur_row.get("action") or [],
                "obs": obs,
            }


def iter_actual_launches(action: Any) -> Iterator[tuple[int, float, int]]:
    if not isinstance(action, list):
        return
    for move in action:
        if not isinstance(move, (list, tuple)) or len(move) < 3:
            continue
        try:
            from_planet_id = int(round(float(move[0])))
            angle = float(move[1])
            ships = int(round(float(move[2])))
        except Exception:
            continue
        if ships <= 0:
            continue
        yield (from_planet_id, angle, ships)
