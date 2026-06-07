from __future__ import annotations


FOUR_PLAYER_RANK_REWARD = {1: 1.0, 2: 0.3, 3: -0.3, 4: -1.0}


def reward_from_rewards(rewards: list[float], player_id: int, players: int) -> tuple[float, int, bool]:
    if not rewards or player_id >= len(rewards):
        return 0.0, players, False
    own = float(rewards[player_id])
    rank = 1 + sum(float(r) > own for r in rewards)
    if int(players) == 2:
        wins = sum(float(r) == own for r in rewards)
        if wins > 1:
            return 0.0, rank, False
        return (1.0 if rank == 1 else -1.0), rank, rank == 1
    return FOUR_PLAYER_RANK_REWARD.get(rank, -1.0), rank, rank == 1

