"""
Orbit Wars agent — "OrbitLord v2"

Core idea: simulate the future. Every turn we:
  1. Precompute every planet/comet position for the next H ticks
     (orbiting planets rotate, comets follow their published paths).
  2. Simulate every in-flight fleet (ours and the enemy's) to find which
     planet it will hit and when.
  3. Build a per-planet timeline of (owner, garrison) for the next H ticks,
     applying production and the engine's exact combat rules at each
     predicted arrival.
  4. Decide moves against the *predicted* board, not the current one:
       - never double-send to a target an in-flight fleet already takes,
       - snipe planets the enemy is about to weaken/capture,
       - coordinate multi-source captures of big targets,
       - reinforce own planets predicted to fall,
       - evacuate doomed planets and expiring comets,
       - funnel idle rear garrisons to the frontline.
  5. Validate each launch by simulating its actual straight-line path so we
     never crash into the sun or an unintended planet on the way.

"""

import math
import time

BOARD = 100.0
CENTER = 50.0
SUN_R = 10.0
ROT_LIMIT = 50.0
MAX_SPEED = 6.0
EPISODE_STEPS = 500
H = 80              # prediction horizon (ticks)
D = 30              # defense-reserve horizon: only hold back ships for
                    # attacks landing this soon; later threats are re-handled
                    # next turn as they approach (avoids permanent lockdown)
TIME_BUDGET = 0.75  # seconds per turn before we bail with what we have


def _get(obs, key, default):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def fleet_speed(ships):
    if ships <= 0:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def seg_point_dist(px, py, ax, ay, bx, by):
    l2 = (ax - bx) ** 2 + (ay - by) ** 2
    if l2 == 0.0:
        return dist(px, py, ax, ay)
    t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2
    t = max(0.0, min(1.0, t))
    return dist(px, py, ax + t * (bx - ax), ay + t * (by - ay))


def swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """Engine-identical: fleet A->B vs planet P0->P1 within r during tick."""
    d0x, d0y = ax - p0x, ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    return (-b + sq) / (2.0 * a) >= 0.0 and (-b - sq) / (2.0 * a) <= 1.0


class Sim:
    """Board predictor: planet positions, fleet hits, garrison timelines."""

    def __init__(self, obs):
        self.t0 = time.time()
        self.step = int(_get(obs, "step", 0))
        self.me = int(_get(obs, "player", 0))
        self.ang_vel = float(_get(obs, "angular_velocity", 0.0))
        self.planets = [list(p) for p in _get(obs, "planets", [])]
        self.fleets = [list(f) for f in _get(obs, "fleets", [])]
        self.comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
        init = _get(obs, "initial_planets", []) or []
        init_by_id = {p[0]: p for p in init}

        # --- position table: pid -> [ (x,y) for dt in 0..H ] ---
        # Verified vs engine: at obs.step=s a rotating planet sits at
        # angle a0 + ang_vel*(s-1); dt ticks later at a0 + ang_vel*(s-1+dt).
        self.pos = {}
        self.alive_until = {}  # pid -> last dt the planet exists (comets expire)
        comet_path = {}
        for group in _get(obs, "comets", []) or []:
            idx = group["path_index"] if isinstance(group, dict) else group.path_index
            paths = group["paths"] if isinstance(group, dict) else group.paths
            pids = group["planet_ids"] if isinstance(group, dict) else group.planet_ids
            for i, pid in enumerate(pids):
                comet_path[pid] = (paths[i], idx)

        base = self.step - 1
        for p in self.planets:
            pid = p[0]
            if pid in comet_path:
                path, idx = comet_path[pid]
                tab = []
                last = -1
                for dt in range(H + 1):
                    j = idx + dt
                    if 0 <= j < len(path):
                        tab.append((path[j][0], path[j][1]))
                        last = dt
                    else:
                        tab.append(None)
                self.pos[pid] = tab
                self.alive_until[pid] = last
                continue
            ip = init_by_id.get(pid)
            rotating = False
            if ip is not None:
                dx, dy = ip[2] - CENTER, ip[3] - CENTER
                orad = math.hypot(dx, dy)
                if orad + p[4] < ROT_LIMIT:
                    rotating = True
                    a0 = math.atan2(dy, dx)
            if rotating:
                tab = []
                for dt in range(H + 1):
                    ang = a0 + self.ang_vel * (base + dt)
                    tab.append((CENTER + orad * math.cos(ang),
                                CENTER + orad * math.sin(ang)))
                self.pos[pid] = tab
            else:
                self.pos[pid] = [(p[2], p[3])] * (H + 1)
            self.alive_until[pid] = H

        self.by_id = {p[0]: p for p in self.planets}
        self.radius = {p[0]: p[4] for p in self.planets}
        self.prod = {p[0]: p[6] for p in self.planets}

        # --- predict every in-flight fleet's hit ---
        # arrivals[pid] = list of (dt, owner, ships)
        self.arrivals = {p[0]: [] for p in self.planets}
        for f in self.fleets:
            hit = self._trace(f[2], f[3], f[4], f[6])
            if hit is not None:
                pid, dt = hit
                self.arrivals[pid].append((dt, f[1], f[6]))

        self._timeline_cache = {}

    # ---- fleet path tracing -------------------------------------------------
    def _trace(self, x, y, angle, ships, max_dt=H, ignore_until=0):
        """Simulate a fleet; return (pid, dt) of first planet hit or None.
        ignore_until: skip collision checks for dt <= this (launch clearance)."""
        spd = fleet_speed(ships)
        vx, vy = math.cos(angle) * spd, math.sin(angle) * spd
        pl = self.planets
        pos = self.pos
        radius = self.radius
        for dt in range(1, max_dt + 1):
            nx, ny = x + vx, y + vy
            if dt > ignore_until:
                # planets first (engine order), coarse filter then exact sweep
                reach = spd + 7.0
                for p in pl:
                    pid = p[0]
                    tab = pos[pid]
                    p1 = tab[dt] if dt < len(tab) else None
                    p0 = tab[dt - 1] if dt - 1 < len(tab) else None
                    if p1 is None or p0 is None:
                        continue
                    if abs(p1[0] - x) > reach or abs(p1[1] - y) > reach:
                        continue
                    if swept_pair_hit(x, y, nx, ny, p0[0], p0[1], p1[0], p1[1],
                                      radius[pid]):
                        return (pid, dt)
            if not (0.0 <= nx <= BOARD and 0.0 <= ny <= BOARD):
                return None
            if seg_point_dist(CENTER, CENTER, x, y, nx, ny) < SUN_R:
                return None
            x, y = nx, ny
        return None

    # ---- garrison timeline --------------------------------------------------
    def timeline(self, pid, extra=()):
        """[(owner, ships) for dt in 0..H] applying production + arrivals.
        extra: additional (dt, owner, ships) events (our planned launches)."""
        key = (pid, tuple(sorted(extra)))
        cached = self._timeline_cache.get(key)
        if cached is not None:
            return cached
        p = self.by_id[pid]
        owner, ships = p[1], p[5]
        prod = p[6]
        events = {}
        for (dt, o, s) in self.arrivals[pid]:
            events.setdefault(dt, []).append((o, s))
        for (dt, o, s) in extra:
            events.setdefault(dt, []).append((o, s))
        out = [(owner, ships)]
        for dt in range(1, H + 1):
            if dt > self.alive_until.get(pid, H):
                out.append((owner, ships))
                continue
            if owner != -1:
                ships += prod
            evs = events.get(dt)
            if evs:
                per = {}
                for (o, s) in evs:
                    per[o] = per.get(o, 0) + s
                ranked = sorted(per.items(), key=lambda kv: kv[1], reverse=True)
                top_o, top_s = ranked[0]
                if len(ranked) > 1:
                    second = ranked[1][1]
                    surv = top_s - second
                    if surv <= 0:
                        surv = 0
                        top_o = -1
                else:
                    surv = top_s
                if surv > 0:
                    if owner == top_o:
                        ships += surv
                    else:
                        ships -= surv
                        if ships < 0:
                            owner = top_o
                            ships = -ships
            out.append((owner, ships))
        self._timeline_cache[key] = out
        return out

    def state_at(self, pid, dt):
        tl = self.timeline(pid)
        dt = max(0, min(H, dt))
        return tl[dt]

    def commit_arrival(self, pid, dt, owner, ships):
        self.arrivals[pid].append((dt, owner, ships))
        self._timeline_cache = {k: v for k, v in self._timeline_cache.items()
                                if k[0] != pid}

    # ---- helpers ------------------------------------------------------------
    def intercept(self, sx, sy, pid, ship_guess):
        """(aim_x, aim_y, arrival_dt) leading target pid, or None."""
        spd = fleet_speed(max(1, ship_guess))
        tab = self.pos[pid]
        cur = tab[0]
        if cur is None:
            return None
        tx, ty = cur
        t = dist(sx, sy, tx, ty) / spd
        for _ in range(5):
            j = min(H, int(math.ceil(t)))
            fut = tab[j] if j < len(tab) else None
            if fut is None:
                return None
            tx, ty = fut
            t = dist(sx, sy, tx, ty) / spd
        dt = int(math.ceil(t))
        if dt > self.alive_until.get(pid, H):
            return None
        return tx, ty, max(1, dt)


def safe_keep(sim, pid):
    """Max ships this planet can send away and still hold through horizon.
    Returns (available, doomed): doomed=True if it falls even keeping all."""
    p = sim.by_id[pid]
    tl = sim.timeline(pid)
    me = sim.me
    doomed = any(o != me for (o, s) in tl)
    if doomed:
        return 0, True
    near_events = [(dt, o, s) for (dt, o, s) in sim.arrivals[pid] if dt <= D]
    if not near_events:
        return p[5], False
    # binary search the largest sacrifice that keeps ownership through D
    lo, hi = 0, p[5]
    base_ships = p[5]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        # simulate timeline with reduced garrison
        owner, ships = p[1], base_ships - mid
        prod = p[6]
        events = {}
        for (dt, o, s) in near_events:
            events.setdefault(dt, []).append((o, s))
        ok = True
        for dt in range(1, D + 1):
            if owner != -1:
                ships += prod
            evs = events.get(dt)
            if evs:
                per = {}
                for (o, s) in evs:
                    per[o] = per.get(o, 0) + s
                ranked = sorted(per.items(), key=lambda kv: kv[1], reverse=True)
                top_o, top_s = ranked[0]
                surv = top_s - (ranked[1][1] if len(ranked) > 1 else 0)
                if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                    surv = 0
                if surv > 0:
                    if owner == top_o:
                        ships += surv
                    else:
                        ships -= surv
                        if ships < 0:
                            ok = False
                            break
        if ok:
            lo = mid
        else:
            hi = mid - 1
    return lo, False


def agent(obs):
    try:
        return _agent(obs)
    except Exception:
        return []  # never forfeit on an unexpected edge case


def _agent(obs):
    sim = Sim(obs)
    me = sim.me
    step = sim.step
    turns_left = max(1, EPISODE_STEPS - step)

    my_planets = [p for p in sim.planets if p[1] == me]
    if not my_planets:
        return []

    enemy_planets = [p for p in sim.planets if p[1] not in (me, -1)]

    # 2p rush vs 4p compact-then-explode posture
    owners = {p[1] for p in sim.planets if p[1] != -1}
    owners.update(f[1] for f in sim.fleets)
    four_p = me >= 2 or any(o >= 2 for o in owners)

    # ---- availability & doomed detection ----
    avail = {}
    doomed = {}
    for p in my_planets:
        a, d = safe_keep(sim, p[0])
        avail[p[0]] = a
        doomed[p[0]] = d

    moves = []
    spent = {p[0]: 0 for p in my_planets}

    def launch(src, aim_x, aim_y, ships, target_pid, arrive_dt):
        """Validate path & commit a move. Returns True if launched."""
        ships = int(ships)
        if ships <= 0 or spent[src[0]] + ships > src[5]:
            return False
        ang = math.atan2(aim_y - src[3], aim_x - src[2])
        sx = src[2] + math.cos(ang) * (src[4] + 0.1)
        sy = src[3] + math.sin(ang) * (src[4] + 0.1)
        hit = sim._trace(sx, sy, ang, ships)
        if hit is None or hit[0] != target_pid:
            return False
        moves.append([src[0], ang, ships])
        spent[src[0]] += ships
        avail[src[0]] -= ships
        sim.commit_arrival(target_pid, hit[1], me, ships)
        return True

    # ---- 1. defense: reinforce own planets predicted to fall ----
    for p in my_planets:
        if not doomed[p[0]]:
            continue
        tl = sim.timeline(p[0])
        fall_dt = next((dt for dt in range(H + 1) if tl[dt][0] != me), None)
        if fall_dt is None or fall_dt <= 1:
            continue
        # deficit: enemy surplus at fall + a margin
        deficit = tl[fall_dt][1] + 2
        helpers = sorted(
            (q for q in my_planets if q[0] != p[0] and avail[q[0]] > 0
             and not doomed[q[0]]),
            key=lambda q: dist(q[2], q[3], p[2], p[3]))
        for q in helpers[:4]:
            if deficit <= 0:
                break
            send = min(avail[q[0]], deficit)
            sol = sim.intercept(q[2], q[3], p[0], send)
            if sol is None:
                continue
            ax, ay, tt = sol
            if tt >= fall_dt:  # must arrive before it falls
                continue
            if launch(q, ax, ay, send, p[0], tt):
                deficit -= send
                # re-evaluate the planet with reinforcements committed
        # refresh doomed status after committed reinforcements
        a, d = safe_keep(sim, p[0])
        avail[p[0]] = a if not d else 0
        doomed[p[0]] = d

    # ---- 2. evacuation: doomed planets & expiring comets dump ships ----
    evac_sources = []
    for p in my_planets:
        pid = p[0]
        rem = sim.alive_until.get(pid, H)
        if pid in sim.comet_ids and rem <= 3:
            evac_sources.append(p)
        elif doomed[pid]:
            tl = sim.timeline(pid)
            fall_dt = next((dt for dt in range(H + 1) if tl[dt][0] != me), H)
            if fall_dt <= 6:
                evac_sources.append(p)
    for p in evac_sources:
        avail[p[0]] = max(avail[p[0]], p[5] - spent[p[0]])

    # ---- 3. offense: capture targets on the predicted board ----
    # candidate targets: anything not ours now, or ours-now-but-falls,
    # judged at predicted arrival state.
    cands = []
    deadline = sim.t0 + TIME_BUDGET

    # per-enemy strength (ships on planets + fleets) — pick on the weak
    strength = {}
    for p in sim.planets:
        if p[1] not in (-1, me):
            strength[p[1]] = strength.get(p[1], 0) + p[5]
    for f in sim.fleets:
        if f[1] not in (-1, me):
            strength[f[1]] = strength.get(f[1], 0) + f[6]
    weakest = min(strength, key=strength.get) if strength else None
    my_strength = sum(p[5] for p in my_planets) + sum(
        f[6] for f in sim.fleets if f[1] == me)

    def enemy_support(tid, tx, ty):
        """Enemy garrison mass near a target (can reinforce/retake it)."""
        s = 0
        for q in sim.planets:
            if q[1] in (-1, me) or q[0] == tid:
                continue
            if dist(q[2], q[3], tx, ty) < 25.0:
                s += q[5]
        return s
    for t in sim.planets:
        tid = t[0]
        if t[1] == me and not doomed.get(tid, False):
            continue
        is_comet = tid in sim.comet_ids
        rem_alive = sim.alive_until.get(tid, H)
        if is_comet and rem_alive < 6:
            continue  # not worth catching a leaving comet
        for s in my_planets:
            if avail[s[0]] <= 0:
                continue
            guess = max(t[5] + 3, 10)
            sol = sim.intercept(s[2], s[3], tid, guess)
            if sol is None:
                continue
            ax, ay, tt = sol
            if tt > turns_left:
                continue
            owner_pred, ships_pred = sim.state_at(tid, tt)
            if owner_pred == me:
                continue  # already taken by an in-flight friendly fleet
            if not four_p and owner_pred not in (-1, me):
                # defender watches our fleet for tt turns; nearby friendly
                # garrisons reinforce the wall before we land — skip attacks
                # that reactive support makes unwinnable
                react = 0
                tgt_p = sim.by_id[tid]
                for q in sim.planets:
                    if q[1] != owner_pred or q[0] == tid:
                        continue
                    if dist(q[2], q[3], tgt_p[2], tgt_p[3]) / 3.0 < tt:
                        react += q[5]
                if 0.60 * react > ships_pred + 40:
                    continue
            if not four_p and owner_pred == -1:
                # don't buy neutrals the enemy can steal right back: after
                # we land with buffer-thin garrison, a nearby enemy stack
                # arrives before production rebuilds the wall
                tgt_p = sim.by_id[tid]
                steal = 0
                for q in sim.planets:
                    if q[1] in (-1, me) or q[0] == tid:
                        continue
                    d_q = dist(q[2], q[3], tgt_p[2], tgt_p[3])
                    lag = d_q / 3.0  # ticks after our landing they arrive
                    if lag < 10:
                        my_wall = 2 + t[6] * lag
                        if 0.45 * q[5] > my_wall + 15:
                            steal = 1
                            break
                if steal:
                    continue
            buffer = 1 if owner_pred == -1 else 3
            need = ships_pred + 1 + buffer
            # refine with true fleet size (speed changes arrival)
            sol2 = sim.intercept(s[2], s[3], tid, need)
            if sol2 is None:
                continue
            ax, ay, tt = sol2
            owner_pred, ships_pred = sim.state_at(tid, tt)
            if owner_pred == me:
                continue
            buffer = 1 if owner_pred == -1 else 3
            need = int(ships_pred + 1 + buffer)
            # early game: skip timing-fragile snipes (ownership shifts within
            # a tick of arrival) — a 1-tick error feeds the old garrison and
            # burns scarce opening capital. Later, capital is cheap; risk it.
            if step < 40:
                o_a = sim.state_at(tid, max(0, tt - 1))[0]
                o_b = sim.state_at(tid, min(H, tt + 1))[0]
                if o_a != owner_pred or o_b != owner_pred:
                    continue
            # 4p opening: stay compact — only grab nearby neutrals
            if four_p and step < 60 and tt > 22:
                continue
            prod_value = t[6] * min(turns_left - tt, 250)
            if is_comet:
                prod_value = t[6] * max(0, rem_alive - tt)
            if prod_value <= 0:
                continue
            value = prod_value + (ships_pred if owner_pred != -1 else 0)
            if owner_pred not in (-1, me):
                value *= 1.6  # denial: enemy loses the stream too
                if owner_pred == weakest and len(strength) > 1:
                    value *= 1.3  # finish off the weakest player first
                elif four_p and strength.get(owner_pred, 0) > 1.2 * my_strength:
                    value *= 0.45  # don't pick fights with stronger players
            elif len(my_planets) < 6:
                value *= 1.3  # young empire: farm neutrals, avoid wars
            score = value / (need + 2.0 * tt + 1.0)
            # contested targets near enemy mass invite retake ping-pong
            sup = enemy_support(tid, sim.by_id[tid][2], sim.by_id[tid][3])
            score /= (1.0 + 0.004 * min(sup, 400))
            cands.append((score, s[0], tid, need, tt))

    cands.sort(key=lambda c: -c[0])

    claimed = set()
    src_by_id = {p[0]: p for p in my_planets}
    for score, sid, tid, need, tt in cands:
        if time.time() > deadline:
            break
        if tid in claimed:
            continue
        src = src_by_id[sid]
        if avail[sid] >= need:
            sol = sim.intercept(src[2], src[3], tid, need)
            if sol is None:
                continue
            ax, ay, tt2 = sol
            owner_pred, ships_pred = sim.state_at(tid, tt2)
            if owner_pred == me:
                claimed.add(tid)
                continue
            need2 = int(ships_pred + 1 + (1 if owner_pred == -1 else 3))
            if need2 > avail[sid]:
                continue
            if owner_pred not in (-1, me):
                # hammer doctrine: overwhelming force; surplus lands as the
                # new frontier garrison and survives the counterattack
                need2 = min(avail[sid], max(int(need2 * 1.8), need2 + 25))
            if launch(src, ax, ay, need2, tid, tt2):
                claimed.add(tid)
            continue
        # ---- multi-source coordinated capture ----
        helpers = sorted(
            (q for q in my_planets if avail[q[0]] > 0),
            key=lambda q: dist(q[2], q[3], sim.by_id[tid][2], sim.by_id[tid][3]))
        plan = []
        total = 0
        latest = 0
        for q in helpers[:4]:
            give = avail[q[0]]
            if give <= 0:
                continue
            sol = sim.intercept(q[2], q[3], tid, give)
            if sol is None:
                continue
            ax, ay, qt = sol
            plan.append((q, give, ax, ay, qt))
            total += give
            latest = max(latest, qt)
            owner_pred, ships_pred = sim.state_at(tid, latest)
            need_now = ships_pred + 1 + (2 if owner_pred == -1 else 4)
            if owner_pred != me and total >= need_now:
                break
        owner_pred, ships_pred = sim.state_at(tid, latest)
        if owner_pred == me:
            claimed.add(tid)
            continue
        need_now = int(ships_pred + 1 + (2 if owner_pred == -1 else 4))
        if total < need_now or not plan:
            continue
        # trim the last contributor to exactly what's needed
        sendable = []
        acc = 0
        for (q, give, ax, ay, qt) in plan:
            take = min(give, need_now - acc)
            if take <= 0:
                break
            sendable.append((q, take, ax, ay, qt))
            acc += take
        if acc < need_now:
            continue
        # all-or-nothing: dry-run every component's path first so we never
        # send an under-strength wave that just feeds the garrison
        valid = True
        for (q, take, ax, ay, qt) in sendable:
            ang = math.atan2(ay - q[3], ax - q[2])
            sx = q[2] + math.cos(ang) * (q[4] + 0.1)
            sy = q[3] + math.sin(ang) * (q[4] + 0.1)
            hit = sim._trace(sx, sy, ang, take)
            if hit is None or hit[0] != tid:
                valid = False
                break
        if not valid:
            continue
        for (q, take, ax, ay, qt) in sendable:
            launch(q, ax, ay, take, tid, qt)
        claimed.add(tid)

    # ---- 4. relay idle garrisons toward the frontline ----
    # Idle ships are dead capital: chain-forward them to the next own planet
    # closer to the enemy, so production from the whole empire keeps feeding
    # the border where captures and defense actually happen.
    # 2p: don't relay while still land-grabbing — streaming ships to the
    # front early drains the capital rear planets need to buy neutrals.
    # 4p: compact mode keeps the planet count low by design, so the gate
    # would never open — and concentrated mass IS the 4p survival plan.
    established = four_p or len(my_planets) >= 6 or step >= 50
    if enemy_planets and established and time.time() < deadline:
        def nearest_enemy_d(p):
            return min(dist(p[2], p[3], q[2], q[3]) for q in enemy_planets)
        front = min(my_planets, key=nearest_enemy_d)
        for p in my_planets:
            pid = p[0]
            relay_min = 8 if four_p else 20
            if pid == front[0] or doomed.get(pid) or avail[pid] < relay_min:
                continue
            my_d = nearest_enemy_d(p)
            # receiver: nearest own planet meaningfully closer to the enemy
            recv = None
            best = 1e9
            for q in my_planets:
                if q[0] == pid or doomed.get(q[0]):
                    continue
                if nearest_enemy_d(q) > my_d - 4.0:
                    continue
                dq = dist(p[2], p[3], q[2], q[3])
                if dq < best:
                    best = dq
                    recv = q
            if recv is None:
                continue
            send = avail[pid]
            sol = sim.intercept(p[2], p[3], recv[0], send)
            if sol is None:
                continue
            ax, ay, tt = sol
            launch(p, ax, ay, send, recv[0], tt)

    # ---- 5. evacuate comets about to leave / planets about to fall ----
    for p in evac_sources:
        pid = p[0]
        left = p[5] - spent[pid]
        if left <= 0:
            continue
        # nearest own safe planet
        safes = [q for q in my_planets
                 if q[0] != pid and not doomed.get(q[0])]
        best = None
        for q in sorted(safes, key=lambda q: dist(q[2], q[3], p[2], p[3]))[:3]:
            sol = sim.intercept(p[2], p[3], q[0], left)
            if sol is None:
                continue
            ax, ay, tt = sol
            if launch(p, ax, ay, left, q[0], tt):
                best = q
                break

    return moves
