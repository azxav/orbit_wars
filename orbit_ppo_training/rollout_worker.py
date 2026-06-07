from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from orbit_bc_eval.bc_agent_runtime import validate_env_move
from orbit_bc_eval.rollout_metrics import RolloutMetrics
from orbit_training_prep.geometry_bridge import make_geometry

from .metrics import summarize_rollout
from .opponent_pool import make_ppo_opponent
from .rewards import reward_from_rewards
from .trajectory import DecisionRecord


def _import_make():
    try:
        from kaggle_environments import make
    except ModuleNotFoundError as exc:
        raise RuntimeError("kaggle_environments is required for PPO rollouts") from exc
    return make


def _state_value(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


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


@dataclass
class RolloutResult:
    trajectories: list[list[DecisionRecord]]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _early_launch_rate(row: dict[str, Any]) -> float:
    return float(row.get("launches_0_100", 0) or 0) / 100.0


class PPORecordingAgent:
    def __init__(self, policy, *, player_id: int, deterministic: bool, device: str, geometry):
        self.policy = policy
        self.player_id = int(player_id)
        self.deterministic = bool(deterministic)
        self.device = device
        self.geometry = geometry
        self.records: list[DecisionRecord] = []
        self.step_debug: list[dict[str, Any]] = []

    def __call__(self, obs, config):
        obs = dict(obs or {})
        player_id = int(obs.get("player", self.player_id) or self.player_id)
        turn = self.policy.act_observation(obs, player_id, deterministic=self.deterministic, device=self.device, geometry=self.geometry)
        illegal = sum(1 for move in turn.moves if not validate_env_move(obs, player_id, move).ok)
        self.records.extend(turn.records)
        self.step_debug.append(
            {
                "step": int(obs.get("step", 0) or 0),
                "obs": obs,
                "actions": turn.moves,
                "illegal_actions": illegal + turn.illegal_action_count,
                "runtime_debug": {
                    "skipped_invalid_decoded_actions": turn.skipped_invalid_action_count,
                    "no_op_source_decisions": turn.no_op_source_decisions,
                    "predicted_launches": turn.predicted_launches,
                    "skip_reasons": dict(turn.skip_reasons),
                    "opening_prediction_counts": {
                        "target": dict(turn.opening_prediction_counts.get("target", {})),
                        "amount": dict(turn.opening_prediction_counts.get("amount", {})),
                        "target_amount": dict(turn.opening_prediction_counts.get("target_amount", {})),
                    },
                    "returned_moves": len(turn.moves),
                    "timeout": False,
                    "error": None,
                },
                "entropy": turn.entropy,
            }
        )
        return turn.moves


def _make_config(max_episode_steps: int, act_timeout: float) -> dict[str, Any]:
    return {"episodeSteps": int(max_episode_steps), "actTimeout": float(act_timeout)}


def _bc_seat_for(game_index: int, players: int) -> int:
    return int(game_index % players)


def collect_rollouts(
    policy,
    config,
    *,
    games: int | None = None,
    deterministic: bool = False,
    seed_start: int | None = None,
    replay_callback: Callable[[Any, int, dict[str, Any]], None] | None = None,
) -> RolloutResult:
    make = _import_make()
    game_count = int(games if games is not None else config.rollout_games_per_update)
    geometry = make_geometry(horizon=int(config.geometry_horizon), device="cpu")
    rows: list[dict[str, Any]] = []
    trajectories: list[list[DecisionRecord]] = []
    for game_index in range(game_count):
        seed = int(config.seed if seed_start is None else seed_start) + game_index
        seat = _bc_seat_for(game_index, int(config.players))
        ppo_agent = PPORecordingAgent(policy, player_id=seat, deterministic=deterministic, device=str(config.device), geometry=geometry)
        agents: list[Callable] = []
        for player in range(int(config.players)):
            if player == seat:
                agents.append(ppo_agent)
            else:
                agents.append(make_ppo_opponent(config.opponent, heuristic_path=config.heuristic_path))
        env = make(config.environment, configuration=_make_config(config.max_episode_steps, config.act_timeout), debug=False)
        try:
            if hasattr(env, "seed"):
                env.seed(seed)
        except Exception:
            pass
        env.run(agents)
        rewards, statuses, final_obs = _final_state(env)
        terminal_reward, rank, win = reward_from_rewards(rewards, seat, int(config.players))
        if ppo_agent.records:
            for r in ppo_agent.records:
                r.reward = 0.0
                r.done = False
            ppo_agent.records[-1].reward = float(terminal_reward)
            ppo_agent.records[-1].done = True
            trajectories.append(ppo_agent.records)
        metrics = RolloutMetrics(game_id=f"game_{game_index:05d}", bc_player_id=seat, players=int(config.players), opponent=str(config.opponent))
        entropy_values: list[float] = []
        for step in ppo_agent.step_debug:
            metrics.record_observation(dict(step["obs"]))
            metrics.record_step(
                step=int(step["step"]),
                actions=list(step["actions"]),
                illegal_actions=int(step["illegal_actions"]),
                runtime_debug=dict(step["runtime_debug"]),
            )
            entropy_values.append(float(step["entropy"]))
        row = metrics.finalize(rewards=rewards, statuses=statuses, final_obs=final_obs)
        row.update(
            {
                "seed": seed,
                "reward": float(terminal_reward),
                "rank": int(rank),
                "win": bool(win),
                "average_entropy": sum(entropy_values) / max(1, len(entropy_values)),
                "early_launch_rate": _early_launch_rate(row),
            }
        )
        if replay_callback is not None:
            replay_callback(env, game_index, row)
        rows.append(row)
    return RolloutResult(trajectories=trajectories, rows=rows, summary=summarize_rollout(rows))
