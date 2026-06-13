from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from orbit_jax_env.config import P_MAX

CENTER = 50.0
BOARD = 100.0
ROTATION_RADIUS_LIMIT = 50.0
SHIP_LOG_DENOM = math.log1p(1000.0)
SQRT2 = math.sqrt(2.0)
MAX_ETA_FEATURE_HORIZON = 200.0
NOOP_TARGET_SLOT = P_MAX


@dataclass(frozen=True)
class JaxBCFeatures:
    planet_features: jnp.ndarray
    global_features: jnp.ndarray
    target_state_features: jnp.ndarray
    pair_features: jnp.ndarray
    target_mask: jnp.ndarray
    amount_mask: jnp.ndarray
    source_slots: jnp.ndarray
    source_mask: jnp.ndarray
    active_source_count: jnp.ndarray
    selected_source_count: jnp.ndarray


def _ships_from_log_norm(x):
    return jnp.expm1(x * SHIP_LOG_DENOM)


def _owner_totals(owner, alive, ships, prod):
    players = jnp.arange(8, dtype=jnp.int32)
    valid = alive & (owner >= 0)
    owner_ships = jnp.array([jnp.sum(jnp.where(valid & (owner == p), ships, 0.0)) for p in range(8)])
    owner_prod = jnp.array([jnp.sum(jnp.where(valid & (owner == p), prod, 0.0)) for p in range(8)])
    owner_planets = jnp.array([jnp.sum(jnp.where(valid & (owner == p), 1.0, 0.0)) for p in range(8)])
    return players, owner_ships, owner_prod, owner_planets


def planet_features_from_state(state, player_id: int | jnp.ndarray) -> jnp.ndarray:
    seat = jnp.asarray(player_id, dtype=jnp.int32)
    owner = state.planet_owner
    alive = state.planet_alive
    ships = jnp.maximum(0.0, state.planet_ships)
    prod = jnp.maximum(0.0, state.planet_production)
    x = state.planet_x
    y = state.planet_y
    dx = x - CENTER
    dy = y - CENTER
    players, owner_ships, owner_prod, _owner_planets = _owner_totals(owner, alive, ships, prod)
    del players
    owner_idx = jnp.clip(owner, 0, 7)
    owner_ship_total = jnp.where(owner >= 0, owner_ships[owner_idx], 0.0)
    owner_prod_total = jnp.where(owner >= 0, owner_prod[owner_idx], 0.0)
    mine = owner == seat
    neutral = owner < 0
    enemy = (~mine) & (~neutral)
    init_dx = state.planet_initial_x - CENTER
    init_dy = state.planet_initial_y - CENTER
    orbit_radius = jnp.hypot(init_dx, init_dy)
    is_orbiting = (orbit_radius + state.planet_radius < ROTATION_RADIUS_LIMIT) & (orbit_radius > 0.5)
    projected = ships
    cols = jnp.stack(
        [
            alive.astype(jnp.float32),
            neutral.astype(jnp.float32),
            mine.astype(jnp.float32),
            enemy.astype(jnp.float32),
            dx / BOARD,
            dy / BOARD,
            state.planet_radius / 5.0,
            jnp.log1p(ships) / SHIP_LOG_DENOM,
            prod / 5.0,
            state.planet_is_comet.astype(jnp.float32),
            is_orbiting.astype(jnp.float32),
            jnp.hypot(dx, dy) / (SQRT2 * BOARD),
            ships / jnp.maximum(1.0, owner_ship_total),
            prod / jnp.maximum(1.0, owner_prod_total),
            projected / 100.0,
            jnp.zeros((P_MAX,), dtype=jnp.float32),
        ],
        axis=-1,
    )
    return jnp.nan_to_num(jnp.where(alive[:, None], cols, 0.0))


def global_features_from_state(state, player_id: int | jnp.ndarray) -> jnp.ndarray:
    seat = jnp.asarray(player_id, dtype=jnp.int32)
    owner = state.planet_owner
    alive = state.planet_alive
    ships = jnp.maximum(0.0, state.planet_ships)
    prod = jnp.maximum(0.0, state.planet_production)
    _players, owner_ships, owner_prod, owner_planets = _owner_totals(owner, alive, ships, prod)
    present = (owner_ships + owner_prod) > 0
    my_ships = owner_ships[seat]
    my_prod = owner_prod[seat]
    my_planets = owner_planets[seat]
    total_ships = jnp.sum(owner_ships)
    total_prod = jnp.sum(owner_prod)
    total_planets = jnp.sum(owner_planets)
    leader_ships = jnp.max(jnp.where(present, owner_ships, -jnp.inf))
    leader_prod = jnp.max(jnp.where(present, owner_prod, -jnp.inf))
    enemy_present = present.at[seat].set(False)
    weakest_enemy = jnp.min(jnp.where(enemy_present, owner_ships, jnp.inf))
    leader_ships = jnp.where(jnp.isfinite(leader_ships), leader_ships, 0.0)
    leader_prod = jnp.where(jnp.isfinite(leader_prod), leader_prod, 0.0)
    weakest_enemy = jnp.where(jnp.isfinite(weakest_enemy), weakest_enemy, 0.0)
    step_norm = state.step.astype(jnp.float32) / jnp.maximum(state.episode_steps.astype(jnp.float32), 1.0)
    return jnp.nan_to_num(
        jnp.array(
            [
                step_norm,
                jnp.maximum(0.0, 1.0 - step_norm),
                jnp.where(state.num_players <= 2, 1.0, 0.0),
                jnp.where(state.num_players >= 4, 1.0, 0.0),
                my_ships / jnp.maximum(1.0, total_ships),
                my_prod / jnp.maximum(1.0, total_prod),
                my_planets / jnp.maximum(1.0, total_planets),
                (my_ships - leader_ships) / jnp.maximum(1.0, total_ships),
                (my_prod - leader_prod) / jnp.maximum(1.0, total_prod),
                (my_ships - weakest_enemy) / jnp.maximum(1.0, total_ships),
            ],
            dtype=jnp.float32,
        )
    )


def target_state_features_from_state(state, player_id: int | jnp.ndarray) -> jnp.ndarray:
    seat = jnp.asarray(player_id, dtype=jnp.int32)
    owner = state.planet_owner
    alive = state.planet_alive
    x = state.planet_x
    y = state.planet_y
    ships = jnp.maximum(0.0, state.planet_ships)
    prod = jnp.maximum(0.0, state.planet_production)
    mine = alive & (owner == seat)
    enemy = alive & (owner >= 0) & (owner != seat)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.hypot(dx, dy) / 10.0
    nearest_own = jnp.min(jnp.where(mine[None, :], dist, jnp.inf), axis=1)
    nearest_enemy = jnp.min(jnp.where(enemy[None, :], dist, jnp.inf), axis=1)
    nearest_own = jnp.where(jnp.isfinite(nearest_own), jnp.minimum(1.0, nearest_own / 50.0), 0.0)
    nearest_enemy = jnp.where(jnp.isfinite(nearest_enemy), jnp.minimum(1.0, nearest_enemy / 50.0), 0.0)
    rel_owner = jnp.where(owner < 0, 0, jnp.where(owner == seat, 1, -1))
    projected_owner = jnp.where(ships > 0, rel_owner, -rel_owner).astype(jnp.float32)
    out = jnp.stack(
        [
            nearest_own,
            nearest_enemy,
            (nearest_enemy < nearest_own).astype(jnp.float32),
            jnp.zeros((P_MAX,), dtype=jnp.float32),
            projected_owner,
            ships / 100.0,
            jnp.zeros((P_MAX,), dtype=jnp.float32),
            ((owner < 0) & (ships <= 5.0)).astype(jnp.float32),
            (prod >= 3.0).astype(jnp.float32),
        ],
        axis=-1,
    )
    return jnp.nan_to_num(jnp.where(alive[:, None], out, 0.0))


def build_masks(state, player_id: int | jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    seat = jnp.asarray(player_id, dtype=jnp.int32)
    source_axis = jnp.arange(P_MAX)
    target_axis = jnp.arange(P_MAX)
    source_alive = state.planet_alive & (state.planet_owner == seat) & (state.planet_ships >= 1.0)
    target_alive = state.planet_alive
    target = jnp.broadcast_to(target_alive[None, :], (P_MAX, P_MAX)) & (source_axis[:, None] != target_axis[None, :])
    target = target & source_alive[:, None]
    target_mask = jnp.concatenate([target, source_alive[:, None]], axis=1)
    amount_mask = jnp.broadcast_to(jnp.arange(7)[None, None, :] >= 0, (P_MAX, P_MAX + 1, 7))
    amount_mask = amount_mask & target_mask[:, :, None]
    amount_mask = amount_mask.at[:, :P_MAX, 0].set(False)
    amount_mask = amount_mask.at[:, NOOP_TARGET_SLOT, :].set(False)
    amount_mask = amount_mask.at[:, NOOP_TARGET_SLOT, 0].set(source_alive)
    return target_mask, amount_mask


def pair_features_for_source(planet_features: jnp.ndarray, target_state_features: jnp.ndarray, source_slot: int, amount_mask: jnp.ndarray) -> jnp.ndarray:
    src = planet_features[source_slot]
    sx, sy = src[4], src[5]
    source_ships = jnp.maximum(0.0, _ships_from_log_norm(src[7]))
    source_prod = jnp.maximum(0.0, src[8] * 5.0)
    threat = src[15]
    target = planet_features
    dx = target[:, 4] - sx
    dy = target[:, 5] - sy
    dist = jnp.hypot(dx, dy)
    ang = jnp.arctan2(dy, dx)
    target_ships = jnp.maximum(0.0, _ships_from_log_norm(target[:, 7]))
    target_prod = jnp.maximum(0.0, target[:, 8] * 5.0)
    target_own = target[:, 2]
    need = jnp.where(target_own > 0.5, 1.0, target_ships + 1.0)
    safe = jnp.maximum(0.0, source_ships - (2.0 + source_prod + 10.0 * threat))
    projected_garrison = target_state_features[:, 5]
    projected_owner = target_state_features[:, 4]
    hostile_need = jnp.where(projected_owner.astype(jnp.int32) == 1, 0.0, jnp.maximum(0.0, projected_garrison * 100.0) + 1.0)
    capture_margin = (source_ships - hostile_need) / 100.0
    amount_bins = jnp.maximum(1.0, float(amount_mask.shape[-1] - 1))
    geom_viable = jnp.sum(amount_mask[:P_MAX, 1:].astype(jnp.float32), axis=-1) / amount_bins
    rows = jnp.stack(
        [
            need / jnp.maximum(1.0, source_ships),
            (source_ships - need) / 100.0,
            target_prod / jnp.maximum(1.0, need),
            dist,
            jnp.sin(ang),
            jnp.cos(ang),
            geom_viable,
            jnp.full((P_MAX,), safe / 100.0, dtype=jnp.float32),
            (source_ships - need) / jnp.maximum(1.0, source_ships),
            jnp.minimum(1.0, dist * 10.0 / MAX_ETA_FEATURE_HORIZON),
            jnp.zeros((P_MAX,), dtype=jnp.float32),
            jnp.zeros((P_MAX,), dtype=jnp.float32),
            projected_garrison,
            projected_owner,
            capture_margin,
        ],
        axis=-1,
    )
    rows = jnp.where(target[:, 0:1] > 0.0, rows, 0.0)
    noop = jnp.zeros((1, rows.shape[-1]), dtype=jnp.float32)
    return jnp.nan_to_num(jnp.concatenate([rows, noop], axis=0))


def _pair_features_for_sources(
    planet_features: jnp.ndarray,
    target_state_features: jnp.ndarray,
    source_slots: jnp.ndarray,
    amount_mask: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(
        lambda source_slot, source_amount_mask: pair_features_for_source(
            planet_features,
            target_state_features,
            source_slot,
            source_amount_mask,
        )
    )(source_slots, amount_mask)


def _full_features_and_masks(state, player_id: int | jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    seat = jnp.asarray(player_id, dtype=jnp.int32)
    pf = planet_features_from_state(state, seat)
    gf = global_features_from_state(state, seat)
    tsf = target_state_features_from_state(state, seat)
    target_mask, amount_mask = build_masks(state, seat)
    return pf, gf, tsf, target_mask, amount_mask


def _select_source_slots(state, active_source: jnp.ndarray, source_cap: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cap = min(int(source_cap), P_MAX)
    active_count = jnp.sum(active_source.astype(jnp.int32))
    scores = jnp.where(active_source, state.planet_ships, -jnp.inf)
    _top_scores, top_slots = jax.lax.top_k(scores, cap)
    source_mask = jnp.arange(cap, dtype=jnp.int32) < active_count
    safe_slot = jnp.argmax(active_source.astype(jnp.int32)).astype(jnp.int32)
    source_slots = jnp.where(source_mask, top_slots.astype(jnp.int32), safe_slot)
    selected_count = jnp.minimum(active_count, jnp.asarray(cap, dtype=jnp.int32))
    return source_slots, source_mask, active_count, selected_count


def build_bc_features_for_seat(state, player_id: int | jnp.ndarray, source_cap: int | None = None) -> JaxBCFeatures:
    seat = jnp.asarray(player_id, dtype=jnp.int32)
    pf, gf, tsf, target_mask, amount_mask = _full_features_and_masks(state, seat)
    if source_cap is None:
        source_slots = jnp.arange(P_MAX, dtype=jnp.int32)
        source_mask = target_mask[:, NOOP_TARGET_SLOT]
        active_count = jnp.sum(source_mask.astype(jnp.int32))
        selected_count = active_count
    else:
        source_slots, source_mask, active_count, selected_count = _select_source_slots(
            state,
            target_mask[:, NOOP_TARGET_SLOT],
            int(source_cap),
        )
        target_mask = target_mask[source_slots] & source_mask[:, None]
        amount_mask = amount_mask[source_slots] & source_mask[:, None, None]
    pair = _pair_features_for_sources(pf, tsf, source_slots, amount_mask)
    return JaxBCFeatures(pf, gf, tsf, pair, target_mask, amount_mask, source_slots, source_mask, active_count, selected_count)
