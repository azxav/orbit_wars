from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BC_CHECKPOINT = PROJECT_ROOT / "checkpoint.pt"
DEFAULT_HEURISTIC_PATH = PROJECT_ROOT / "orbit_wars_base.py"
DEFAULT_ENVIRONMENT = "orbit_wars"
DEFAULT_EPISODE_STEPS = 500
DEFAULT_ACT_TIMEOUT = 1.0
DEFAULT_GEOMETRY_HORIZON = 160
DEFAULT_DEVICE = "cpu"
STEP_BUCKETS: tuple[tuple[int, int], ...] = ((0, 100), (100, 250), (250, 430), (430, 500))


@dataclass(frozen=True)
class EvalConfig:
    bc_checkpoint: Path = DEFAULT_BC_CHECKPOINT
    opponent: str = "heuristic_path"
    num_games: int = 20
    players: int | str = "both"
    seed_start: int = 0
    out_dir: Path = Path("bc_eval_runs/checkpoint_vs_orbit_wars_base")
    device: str = DEFAULT_DEVICE
    environment: str = DEFAULT_ENVIRONMENT
    episode_steps: int = DEFAULT_EPISODE_STEPS
    act_timeout: float = DEFAULT_ACT_TIMEOUT
    geometry_horizon: int = DEFAULT_GEOMETRY_HORIZON
    heuristic_path: Path | None = DEFAULT_HEURISTIC_PATH
    debug_game: bool = False
