from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class PPOConfig:
    bc_checkpoint: str
    out_dir: str
    players: int = 4
    opponent: str = "orbit_wars_base"
    num_envs: int = 8
    rollout_games_per_update: int = 32
    max_episode_steps: int = 500
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ppo_epochs: int = 2
    minibatch_size: int = 512
    lr: float = 2e-5
    clip_range: float = 0.10
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_to_bc_coef: float = 0.02
    bc_aux_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    eval_interval_updates: int = 5
    save_interval_updates: int = 5
    seed: int = 42
    device: str = "cpu"
    environment: str = "orbit_wars"
    act_timeout: float = 1.0
    geometry_horizon: int = 160
    updates: int = 50
    eval_games: int = 10
    heuristic_path: str = "orbit_wars_base.py"
    reward_shaping: bool = False

    def __post_init__(self) -> None:
        self.players = int(self.players)
        if self.players not in (2, 4):
            raise ValueError("players must be 2 or 4")
        if self.opponent == "orbit_wars_base":
            self.opponent = "heuristic_path"


def save_config(config: PPOConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)


def load_config(path: str | Path) -> PPOConfig:
    with open(path, "r", encoding="utf-8") as f:
        return PPOConfig(**json.load(f))
