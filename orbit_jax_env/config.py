from __future__ import annotations

from dataclasses import dataclass

P_MAX = 64
F_MAX = 1024
MAX_PLAYERS = 4
MAX_ACTIONS_PER_PLAYER = 64
COMET_PATH_MAX = 40
COMET_SPAWN_COUNT = 5
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
LAUNCH_CLEARANCE = 0.1
LOG_1000 = 6.907755278982137


@dataclass(frozen=True)
class EnvConfig:
    num_players: int = 2
    episode_steps: int = 500
    ship_speed: float = 6.0
    enable_comets: bool = False
