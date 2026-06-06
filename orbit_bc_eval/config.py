from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENVIRONMENT = "orbit_wars"
DEFAULT_EPISODE_STEPS = 500
DEFAULT_ACT_TIMEOUT = 1.0
DEFAULT_GEOMETRY_HORIZON = 160
DEFAULT_DEVICE = "cpu"
STEP_BUCKETS: tuple[tuple[int, int], ...] = ((0, 100), (100, 250), (250, 430), (430, 500))


@dataclass(frozen=True)
class EvalConfig:
    bc_checkpoint: Path
    opponent: str = "simple_expand"
    num_games: int = 20
    players: int = 4
    seed_start: int = 0
    out_dir: Path = Path("bc_eval_runs/default")
    device: str = DEFAULT_DEVICE
    environment: str = DEFAULT_ENVIRONMENT
    episode_steps: int = DEFAULT_EPISODE_STEPS
    act_timeout: float = DEFAULT_ACT_TIMEOUT
    geometry_horizon: int = DEFAULT_GEOMETRY_HORIZON
    heuristic_path: Path | None = None
    debug_game: bool = False
