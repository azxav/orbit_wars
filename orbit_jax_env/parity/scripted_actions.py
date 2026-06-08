from __future__ import annotations

import math
import random


def no_actions(_step: int, players: int):
    return [[] for _ in range(players)]


def simple_right_launch(source_id: int, ships: int):
    def policy(_step: int, players: int):
        actions = [[] for _ in range(players)]
        if _step == 0:
            actions[0] = [[int(source_id), 0.0, int(ships)]]
        return actions

    return policy


def sun_launch(source_id: int, ships: int):
    return simple_right_launch(source_id, ships)


def upward_launch(source_id: int, ships: int):
    def policy(_step: int, players: int):
        actions = [[] for _ in range(players)]
        if _step == 0:
            actions[0] = [[int(source_id), math.pi / 2.0, int(ships)]]
        return actions

    return policy


def nearest_neutral_capture(step: int, players: int, obs=None):
    actions = [[] for _ in range(players)]
    if step != 0 or obs is None:
        return actions
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    sources = [p for p in planets if int(p[1]) == 0 and int(p[5]) > 0]
    targets = [p for p in planets if int(p[1]) == -1]
    if not sources or not targets:
        return actions
    _, source, target = min(
        (
            ((float(t[2]) - float(s[2])) ** 2 + (float(t[3]) - float(s[3])) ** 2, s, t)
            for s in sources
            for t in targets
        ),
        key=lambda row: row[0],
    )
    angle = math.atan2(float(target[3]) - float(source[3]), float(target[2]) - float(source[2]))
    ships = min(int(source[5]), int(target[5]) + 5)
    actions[0] = [[int(source[0]), float(angle), int(ships)]]
    return actions


def _first_owned_source(obs, owner: int = 0):
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    sources = [p for p in planets if int(p[1]) == int(owner) and int(p[5]) > 0]
    if not sources:
        return None
    return min(sources, key=lambda p: int(p[0]))


def launch_toward_sun(step: int, players: int, obs=None):
    actions = [[] for _ in range(players)]
    if step != 0 or obs is None:
        return actions
    source = _first_owned_source(obs, 0)
    if source is None:
        return actions
    angle = math.atan2(50.0 - float(source[3]), 50.0 - float(source[2]))
    actions[0] = [[int(source[0]), float(angle), int(source[5])]]
    return actions


def launch_outward_to_bounds(step: int, players: int, obs=None):
    actions = [[] for _ in range(players)]
    if step != 0 or obs is None:
        return actions
    source = _first_owned_source(obs, 0)
    if source is None:
        return actions
    angle = math.atan2(float(source[3]) - 50.0, float(source[2]) - 50.0)
    actions[0] = [[int(source[0]), float(angle), int(source[5])]]
    return actions


def opposing_neutral_attack(step: int, players: int, obs=None):
    actions = [[] for _ in range(players)]
    if step != 0 or obs is None or players < 2:
        return actions
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    p0_sources = [p for p in planets if int(p[1]) == 0 and int(p[5]) > 0]
    p1_sources = [p for p in planets if int(p[1]) == 1 and int(p[5]) > 0]
    targets = [p for p in planets if int(p[1]) == -1]
    if not p0_sources or not p1_sources or not targets:
        return actions
    _, source0, source1, target = min(
        (
            (
                ((float(t[2]) - float(s0[2])) ** 2 + (float(t[3]) - float(s0[3])) ** 2)
                + ((float(t[2]) - float(s1[2])) ** 2 + (float(t[3]) - float(s1[3])) ** 2),
                s0,
                s1,
                t,
            )
            for s0 in p0_sources
            for s1 in p1_sources
            for t in targets
        ),
        key=lambda row: row[0],
    )
    for player, source in ((0, source0), (1, source1)):
        angle = math.atan2(float(target[3]) - float(source[3]), float(target[2]) - float(source[2]))
        actions[player] = [[int(source[0]), float(angle), max(1, int(source[5]))]]
    return actions


def moving_planet_intercept(step: int, players: int, obs=None):
    actions = [[] for _ in range(players)]
    if step != 0 or obs is None:
        return actions
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    sources = [p for p in planets if int(p[1]) == 0 and int(p[5]) > 0]
    targets = []
    for p in planets:
        if int(p[1]) == 0:
            continue
        dx = float(p[2]) - 50.0
        dy = float(p[3]) - 50.0
        orbital_r = math.sqrt(dx * dx + dy * dy)
        if orbital_r + float(p[4]) < 50.0:
            targets.append(p)
    if not sources or not targets:
        return nearest_neutral_capture(step, players, obs)
    _, source, target = min(
        (
            ((float(t[2]) - float(s[2])) ** 2 + (float(t[3]) - float(s[3])) ** 2, s, t)
            for s in sources
            for t in targets
        ),
        key=lambda row: row[0],
    )
    angle = math.atan2(float(target[3]) - float(source[3]), float(target[2]) - float(source[2]))
    ships = min(int(source[5]), max(1, int(target[5]) + 3))
    actions[0] = [[int(source[0]), float(angle), int(ships)]]
    return actions


def random_scripted_actions(step: int, players: int, obs=None):
    actions = [[] for _ in range(players)]
    if obs is None or step >= 50:
        return actions
    rng = random.Random(f"orbit-jax-parity-{players}-{step}")
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    live_targets = [p for p in planets if int(p[1]) >= -1]
    for player in range(players):
        if step % (player + 3) != 0:
            continue
        sources = [p for p in planets if int(p[1]) == player and int(p[5]) > 2]
        if not sources or not live_targets:
            continue
        source = sources[rng.randrange(len(sources))]
        candidates = [p for p in live_targets if int(p[0]) != int(source[0])]
        if not candidates:
            continue
        target = candidates[rng.randrange(len(candidates))]
        angle = math.atan2(float(target[3]) - float(source[3]), float(target[2]) - float(source[2]))
        ships = max(1, min(int(source[5]), 1 + rng.randrange(max(1, min(8, int(source[5]))))))
        actions[player] = [[int(source[0]), float(angle), int(ships)]]
    return actions
