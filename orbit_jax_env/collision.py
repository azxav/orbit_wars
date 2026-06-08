from __future__ import annotations

import jax.numpy as jnp

from .config import BOARD_SIZE, CENTER, SUN_RADIUS


def point_to_segment_distance(px, py, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    l2 = vx * vx + vy * vy
    raw_t = ((px - ax) * vx + (py - ay) * vy) / jnp.where(l2 <= 0.0, 1.0, l2)
    t = jnp.clip(raw_t, 0.0, 1.0)
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    dx = px - proj_x
    dy = py - proj_y
    return jnp.sqrt(jnp.maximum(dx * dx + dy * dy, 0.0))


def swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, radius):
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - radius * radius
    disc = b * b - 4.0 * a * c
    near_static = a < 1.0e-12
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    denom = jnp.where(near_static, 1.0, 2.0 * a)
    t1 = (-b - sq) / denom
    t2 = (-b + sq) / denom
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(near_static, c <= 0.0, moving_hit)


def out_of_bounds(x, y):
    return (x < 0.0) | (x > BOARD_SIZE) | (y < 0.0) | (y > BOARD_SIZE)


def crosses_sun(ax, ay, bx, by):
    return point_to_segment_distance(CENTER, CENTER, ax, ay, bx, by) < SUN_RADIUS
