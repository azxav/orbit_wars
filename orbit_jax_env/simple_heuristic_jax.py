from __future__ import annotations

import jax
import jax.numpy as jnp

from .collision import crosses_sun, out_of_bounds, swept_pair_hit
from .config import (
    CENTER,
    LAUNCH_CLEARANCE,
    MAX_ACTIONS_PER_PLAYER,
    MAX_PLAYERS,
    P_MAX,
    ROTATION_RADIUS_LIMIT,
)
from .movement import fleet_speed
from .state import EnvState

H = 80
D = 30
NEG = -1.0e9


def _dist(ax, ay, bx, by):
    return jnp.sqrt(jnp.maximum((ax - bx) ** 2 + (ay - by) ** 2, 0.0))


def _future_positions(state: EnvState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dt = jnp.arange(H + 1, dtype=jnp.int32)
    dx = state.planet_initial_x - CENTER
    dy = state.planet_initial_y - CENTER
    orbit_radius = jnp.sqrt(jnp.maximum(dx * dx + dy * dy, 0.0))
    orbiting = state.planet_alive & (~state.planet_is_comet) & ((orbit_radius + state.planet_radius) < ROTATION_RADIUS_LIMIT)
    angle0 = jnp.arctan2(dy, dx)
    future_step = state.step + jnp.maximum(dt - 1, 0)
    angle = angle0[:, None] + state.angular_velocity * future_step[None, :].astype(jnp.float32)
    rot_x = CENTER + orbit_radius[:, None] * jnp.cos(angle)
    rot_y = CENTER + orbit_radius[:, None] * jnp.sin(angle)

    path_idx = state.comet_path_index[:, None] + dt[None, :]
    comet_valid = state.planet_is_comet[:, None] & state.planet_alive[:, None] & (state.comet_path_len[:, None] > 0)
    comet_valid = comet_valid & (path_idx >= 0) & (path_idx < state.comet_path_len[:, None])
    safe_idx = jnp.clip(path_idx, 0, jnp.maximum(state.comet_path_len[:, None] - 1, 0))
    comet_x = jnp.take_along_axis(state.comet_path_x, safe_idx, axis=1)
    comet_y = jnp.take_along_axis(state.comet_path_y, safe_idx, axis=1)

    static_x = jnp.broadcast_to(state.planet_x[:, None], (P_MAX, H + 1))
    static_y = jnp.broadcast_to(state.planet_y[:, None], (P_MAX, H + 1))
    px = jnp.where(orbiting[:, None], rot_x, static_x)
    py = jnp.where(orbiting[:, None], rot_y, static_y)
    px = jnp.where(comet_valid, comet_x, px)
    py = jnp.where(comet_valid, comet_y, py)
    px = px.at[:, 0].set(state.planet_x)
    py = py.at[:, 0].set(state.planet_y)
    valid = state.planet_alive[:, None] & jnp.where(state.planet_is_comet[:, None], comet_valid | (dt[None, :] == 0), True)
    alive_until = jnp.sum(valid.astype(jnp.int32), axis=1) - 1
    return px, py, valid, jnp.maximum(alive_until, 0)


def _trace_one(state: EnvState, px: jnp.ndarray, py: jnp.ndarray, valid_pos: jnp.ndarray, x, y, angle, ships):
    spd = fleet_speed(jnp.maximum(ships, 1.0), state.ship_speed)
    vx = jnp.cos(angle) * spd
    vy = jnp.sin(angle) * spd

    def body(carry, dt):
        cx, cy, hit_pid, hit_dt, done = carry
        nx = cx + vx
        ny = cy + vy
        prev = dt - 1
        hit_vec = state.planet_alive & valid_pos[:, dt] & valid_pos[:, prev]
        hit_vec = hit_vec & swept_pair_hit(cx, cy, nx, ny, px[:, prev], py[:, prev], px[:, dt], py[:, dt], state.planet_radius)
        any_hit = jnp.any(hit_vec)
        pid = jnp.argmax(hit_vec).astype(jnp.int32)
        lost = out_of_bounds(nx, ny) | ((~any_hit) & crosses_sun(cx, cy, nx, ny))
        new_done = done | any_hit | lost
        return (
            jnp.where(done, cx, nx),
            jnp.where(done, cy, ny),
            jnp.where((~done) & any_hit, pid, hit_pid),
            jnp.where((~done) & any_hit, dt, hit_dt),
            new_done,
        ), None

    (fx, fy, hit_pid, hit_dt, done), _ = jax.lax.scan(
        body,
        (x, y, jnp.array(-1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32), jnp.array(False)),
        jnp.arange(1, H + 1, dtype=jnp.int32),
    )
    del fx, fy
    return hit_pid, hit_dt, hit_pid >= 0


def _existing_arrivals(state: EnvState, px: jnp.ndarray, py: jnp.ndarray, valid_pos: jnp.ndarray) -> jnp.ndarray:
    fleet_slots = jnp.arange(state.fleet_alive.shape[0], dtype=jnp.int32)

    def one(i):
        pid, dt, hit = _trace_one(state, px, py, valid_pos, state.fleet_x[i], state.fleet_y[i], state.fleet_angle[i], state.fleet_ships[i])
        active = state.fleet_alive[i] & hit
        return pid, dt, state.fleet_owner[i], jnp.where(active, state.fleet_ships[i], 0.0), active

    hit_pid, hit_dt, owner, ships, active = jax.vmap(one)(fleet_slots)
    arrivals = jnp.zeros((P_MAX, H + 1, MAX_PLAYERS), dtype=jnp.float32)
    safe_owner = jnp.clip(owner, 0, MAX_PLAYERS - 1)
    return arrivals.at[hit_pid, hit_dt, safe_owner].add(jnp.where(active, ships, 0.0))


def _timeline(state: EnvState, arrivals: jnp.ndarray, alive_until: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    def per_planet(pid):
        owner0 = state.planet_owner[pid]
        ships0 = state.planet_ships[pid]

        def body(carry, dt):
            owner, ships = carry
            alive = dt <= alive_until[pid]
            ships = jnp.where(alive & (owner != -1), ships + state.planet_production[pid], ships)
            by_player = arrivals[pid, dt]
            has_arrivals = jnp.any(by_player > 0.0)
            top_owner = jnp.argmax(by_player).astype(jnp.int32)
            top = by_player[top_owner]
            second = jnp.max(jnp.where(jnp.arange(MAX_PLAYERS) == top_owner, -1.0, by_player))
            surplus = top - jnp.maximum(second, 0.0)
            tied = top <= jnp.maximum(second, 0.0)
            attack_owner = jnp.where(tied, -1, top_owner)

            same = attack_owner == owner
            new_owner = owner
            new_ships = ships
            new_ships = jnp.where(has_arrivals & same, ships + surplus, new_ships)
            reduced = ships - surplus
            captured = reduced < 0.0
            new_owner = jnp.where(has_arrivals & (~same) & captured, attack_owner, new_owner)
            new_ships = jnp.where(has_arrivals & (~same), jnp.abs(reduced), new_ships)
            new_owner = jnp.where(alive, new_owner, owner)
            new_ships = jnp.where(alive, new_ships, ships)
            return (new_owner, new_ships), (new_owner, new_ships)

        (_owner, _ships), (owners_tail, ships_tail) = jax.lax.scan(
            body,
            (owner0, ships0),
            jnp.arange(1, H + 1, dtype=jnp.int32),
        )
        return jnp.concatenate([owner0[None], owners_tail]), jnp.concatenate([ships0[None], ships_tail])

    return jax.vmap(per_planet)(jnp.arange(P_MAX, dtype=jnp.int32))


def _intercept(state: EnvState, px: jnp.ndarray, py: jnp.ndarray, valid_pos: jnp.ndarray, src, tgt, ships):
    spd = fleet_speed(jnp.maximum(ships, 1.0), state.ship_speed)
    sx = state.planet_x[src]
    sy = state.planet_y[src]

    def body(t, _):
        idx = jnp.clip(jnp.ceil(t).astype(jnp.int32), 0, H)
        tx = px[tgt, idx]
        ty = py[tgt, idx]
        return _dist(sx, sy, tx, ty) / spd, None

    t, _ = jax.lax.scan(body, _dist(sx, sy, state.planet_x[tgt], state.planet_y[tgt]) / spd, None, length=5)
    dt = jnp.clip(jnp.ceil(t).astype(jnp.int32), 1, H)
    ok = valid_pos[tgt, dt]
    ax = px[tgt, dt]
    ay = py[tgt, dt]
    return ax, ay, dt, ok


def _launch_valid(state: EnvState, px: jnp.ndarray, py: jnp.ndarray, valid_pos: jnp.ndarray, src, tgt, aim_x, aim_y, ships, seat):
    angle = jnp.arctan2(aim_y - state.planet_y[src], aim_x - state.planet_x[src])
    sx = state.planet_x[src] + jnp.cos(angle) * (state.planet_radius[src] + LAUNCH_CLEARANCE)
    sy = state.planet_y[src] + jnp.sin(angle) * (state.planet_radius[src] + LAUNCH_CLEARANCE)
    hit_pid, _hit_dt, hit = _trace_one(state, px, py, valid_pos, sx, sy, angle, ships)
    owned = state.planet_alive[src] & (state.planet_owner[src] == seat)
    affordable = ships > 0.0
    return owned & affordable & hit & (hit_pid == tgt), angle


def _safe_keep_for_seat(state: EnvState, owners_t: jnp.ndarray, seat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    own = state.planet_alive & (state.planet_owner == seat)
    doomed = own & jnp.any(owners_t[:, : D + 1] != seat, axis=1)
    near_threat = doomed | jnp.any(owners_t[:, : D + 1] != state.planet_owner[:, None], axis=1)
    reserve = jnp.where(near_threat, jnp.ceil(state.planet_ships * 0.45), 0.0)
    avail = jnp.where(own, jnp.maximum(0.0, jnp.floor(state.planet_ships - reserve)), 0.0)
    avail = jnp.where(doomed, 0.0, avail)
    return avail, doomed


def _seat_actions(state: EnvState, seat: jnp.ndarray) -> jnp.ndarray:
    px, py, valid_pos, alive_until = _future_positions(state)
    arrivals = _existing_arrivals(state, px, py, valid_pos)
    owners_t, ships_t = _timeline(state, arrivals, alive_until)
    avail, doomed = _safe_keep_for_seat(state, owners_t, seat)

    planet_idx = jnp.arange(P_MAX, dtype=jnp.int32)
    alive = state.planet_alive
    own = alive & (state.planet_owner == seat)
    enemy = alive & (state.planet_owner != seat) & (state.planet_owner != -1)
    active_player = seat < state.num_players
    owners_present = jnp.zeros((MAX_PLAYERS,), dtype=jnp.int32)
    owners_present = owners_present.at[jnp.clip(state.planet_owner, 0, MAX_PLAYERS - 1)].max(
        (alive & (state.planet_owner >= 0)).astype(jnp.int32)
    )
    owners_present = owners_present.astype(jnp.bool_)
    four_p = (state.num_players >= 4) | (seat >= 2) | jnp.any(owners_present & (jnp.arange(MAX_PLAYERS) >= 2))
    turns_left = jnp.maximum(1, state.episode_steps - state.step)

    strength = jnp.zeros((MAX_PLAYERS,), dtype=jnp.float32)
    strength = strength.at[jnp.clip(state.planet_owner, 0, MAX_PLAYERS - 1)].add(jnp.where(enemy, state.planet_ships, 0.0))
    strength = strength.at[jnp.clip(state.fleet_owner, 0, MAX_PLAYERS - 1)].add(
        jnp.where(state.fleet_alive & (state.fleet_owner != seat) & (state.fleet_owner >= 0), state.fleet_ships, 0.0)
    )
    valid_enemy_strength = (jnp.arange(MAX_PLAYERS) < state.num_players) & (jnp.arange(MAX_PLAYERS) != seat) & (strength > 0.0)
    weakest = jnp.argmin(jnp.where(valid_enemy_strength, strength, jnp.inf)).astype(jnp.int32)
    my_strength = jnp.sum(jnp.where(own, state.planet_ships, 0.0)) + jnp.sum(
        jnp.where(state.fleet_alive & (state.fleet_owner == seat), state.fleet_ships, 0.0)
    )

    dx = state.planet_x[:, None] - state.planet_x[None, :]
    dy = state.planet_y[:, None] - state.planet_y[None, :]
    dist = jnp.sqrt(jnp.maximum(dx * dx + dy * dy, 0.0))
    nearest_enemy_d = jnp.min(jnp.where(enemy[None, :], dist, jnp.inf), axis=1)
    established = four_p | (jnp.sum(own.astype(jnp.int32)) >= 6) | (state.step >= 50)

    def score_target(src, tgt):
        guess = jnp.maximum(state.planet_ships[tgt] + 3.0, 10.0)
        ax, ay, dt, ok = _intercept(state, px, py, valid_pos, src, tgt, guess)
        owner_pred = owners_t[tgt, dt]
        ships_pred = ships_t[tgt, dt]
        not_taken = owner_pred != seat
        buffer = jnp.where(owner_pred == -1, 1.0, 3.0)
        need = jnp.floor(ships_pred + 1.0 + buffer)
        ax, ay, dt, ok2 = _intercept(state, px, py, valid_pos, src, tgt, need)
        owner_pred = owners_t[tgt, dt]
        ships_pred = ships_t[tgt, dt]
        buffer = jnp.where(owner_pred == -1, 1.0, 3.0)
        need = jnp.floor(ships_pred + 1.0 + buffer)
        prod_value = state.planet_production[tgt] * jnp.minimum(turns_left - dt, 250)
        comet_value = state.planet_production[tgt] * jnp.maximum(0, alive_until[tgt] - dt)
        prod_value = jnp.where(state.planet_is_comet[tgt], comet_value, prod_value)
        value = prod_value + jnp.where(owner_pred != -1, ships_pred, 0.0)
        value = jnp.where(owner_pred == weakest, value * 1.3, value)
        value = jnp.where((owner_pred != -1) & (owner_pred != seat), value * 1.6, value)
        value = jnp.where((owner_pred == -1) & (jnp.sum(own.astype(jnp.int32)) < 6), value * 1.3, value)
        strong_enemy = (owner_pred >= 0) & (owner_pred < MAX_PLAYERS) & (strength[jnp.clip(owner_pred, 0, MAX_PLAYERS - 1)] > 1.2 * my_strength)
        value = jnp.where(four_p & strong_enemy, value * 0.45, value)
        nearby_enemy = enemy & (dist[:, tgt] < 25.0)
        support = jnp.sum(jnp.where(nearby_enemy, state.planet_ships, 0.0))
        score = value / (need + 2.0 * dt.astype(jnp.float32) + 1.0)
        score = score / (1.0 + 0.004 * jnp.minimum(support, 400.0))
        timing_ok = (dt <= turns_left) & (prod_value > 0.0)
        compact_ok = (~four_p) | (state.step >= 60) | (dt <= 22)
        source_ok = own[src] & (avail[src] >= need) & (src != tgt)
        target_ok = alive[tgt] & ((state.planet_owner[tgt] != seat) | doomed[tgt])
        return jnp.where(active_player & ok & ok2 & not_taken & timing_ok & compact_ok & source_ok & target_ok, score, NEG), need, ax, ay, dt

    def source_action(src):
        # Evacuation beats ordinary scoring for doomed planets and expiring comets.
        expiring_comet = own[src] & state.planet_is_comet[src] & (alive_until[src] <= 3)
        evac = (doomed[src] | expiring_comet) & (state.planet_ships[src] > 0.0)
        safe_own = own & (~doomed) & (planet_idx != src)
        safe_score = jnp.where(safe_own, -dist[src], NEG)
        safe_tgt = jnp.argmax(safe_score).astype(jnp.int32)

        # Defense: if a friendly planet is falling soon, nearby helpers reinforce it.
        first_fall = jnp.argmax(owners_t != seat, axis=1).astype(jnp.int32)
        falls = own & doomed & (first_fall > 1)
        defense_score = jnp.where(falls, -dist[src], NEG)
        defense_tgt = jnp.argmax(defense_score).astype(jnp.int32)
        defense = own[src] & (~doomed[src]) & (avail[src] > 0.0) & (jnp.max(defense_score) > NEG / 2.0)

        scores, needs, aim_x, aim_y, dts = jax.vmap(lambda tgt: score_target(src, tgt))(planet_idx)
        best_tgt = jnp.argmax(scores).astype(jnp.int32)
        best_score = scores[best_tgt]

        # Relay idle rear garrisons if no capture is worth taking.
        closer_own = own & (~doomed) & (planet_idx != src) & (nearest_enemy_d < nearest_enemy_d[src] - 4.0)
        relay_score = jnp.where(closer_own, -dist[src], NEG)
        relay_tgt = jnp.argmax(relay_score).astype(jnp.int32)
        relay = established & own[src] & (~doomed[src]) & (avail[src] >= jnp.where(four_p, 8.0, 20.0)) & (best_score <= 0.0)
        relay = relay & (jnp.max(relay_score) > NEG / 2.0)

        tgt = jnp.where(evac, safe_tgt, jnp.where(defense, defense_tgt, jnp.where(relay, relay_tgt, best_tgt)))
        send = jnp.where(evac, state.planet_ships[src], jnp.where(relay, avail[src], needs[best_tgt]))
        send = jnp.where(defense, jnp.minimum(avail[src], ships_t[defense_tgt, first_fall[defense_tgt]] + 2.0), send)
        ax = jnp.where(evac | defense | relay, state.planet_x[tgt], aim_x[best_tgt])
        ay = jnp.where(evac | defense | relay, state.planet_y[tgt], aim_y[best_tgt])
        valid_choice = evac | defense | relay | (best_score > NEG / 2.0)
        ax2, ay2, _dt2, intercept_ok = _intercept(state, px, py, valid_pos, src, tgt, send)
        ax = jnp.where(evac | defense | relay, ax2, ax)
        ay = jnp.where(evac | defense | relay, ay2, ay)
        path_ok, angle = _launch_valid(state, px, py, valid_pos, src, tgt, ax, ay, send, seat)
        valid = valid_choice & intercept_ok & path_ok & (send > 0.0)
        return jnp.array(
            [
                jnp.where(valid, state.planet_id[src].astype(jnp.float32), 0.0),
                jnp.where(valid, angle, 0.0),
                jnp.where(valid, jnp.floor(send), 0.0),
            ],
            dtype=jnp.float32,
        )

    rows = jax.vmap(source_action)(planet_idx)
    rows = jnp.where(active_player, rows, jnp.zeros_like(rows))
    if P_MAX >= MAX_ACTIONS_PER_PLAYER:
        return rows[:MAX_ACTIONS_PER_PLAYER]
    pad = jnp.zeros((MAX_ACTIONS_PER_PLAYER - P_MAX, 3), dtype=jnp.float32)
    return jnp.concatenate([rows, pad], axis=0)


def simple_heuristic_actions(state: EnvState) -> jnp.ndarray:
    """Pure-JAX training opponent inspired by ``simple_heuristic.agent``.

    The function returns fixed-shape raw Orbit Wars actions for every player so
    the PPO trainer can overwrite the learner seat and keep rollout inside
    ``jit``/``vmap``.
    """

    return jax.vmap(lambda seat: _seat_actions(state, seat))(jnp.arange(MAX_PLAYERS, dtype=jnp.int32))
