"""JAX-native PPO self-play trainer for Orbit Wars (P1/P2 deliverable).

End-to-end on GPU: vmapped env + Flax entity policy controlling all seats
(shared params = self-play). Memory-smart: the rollout stores the board
observation once per step (planets [P,9] + global [6]) plus sampled action
indices; per-seat features are recomputed in the PPO update from that same
board via a light feature-shim, so the rollout and the update use *identical*
inputs (the importance ratio is exact) and trajectory memory stays small.

Run (from WSL):
  cd /mnt/c/.../orbit_wars
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 /opt/jaxenv/bin/python -m orbit_ppo_jax.train \
      --envs 16 --steps 500 --updates 3
"""
from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax

from orbit_jax_env.config import EnvConfig, MAX_PLAYERS
from orbit_jax_env.reset import reset
from orbit_jax_env.step import step
from orbit_jax_env.observation import build_observation
from orbit_jax_env.features_jax import planet_features_jax, global_features_jax
from orbit_ppo_jax.policy import Encoder, AmountHead, target_mask, AMOUNT_FRACS

NEG = -1e9


def shim_from_obs(planets, glob):
    """Build a feature-sufficient state shim from a board observation.
    planets [P,9] = id,owner,x,y,r,ships,prod,alive,is_comet ; glob [6]."""
    return SimpleNamespace(
        planet_id=planets[:, 0],
        planet_owner=planets[:, 1].astype(jnp.int32),
        planet_x=planets[:, 2], planet_y=planets[:, 3],
        planet_radius=planets[:, 4], planet_ships=planets[:, 5],
        planet_production=planets[:, 6],
        planet_alive=(planets[:, 7] > 0.5),
        planet_is_comet=(planets[:, 8] > 0.5),
        num_players=glob[3].astype(jnp.int32),
        step=glob[0], episode_steps=glob[1],
    )


def seat_feats(planets, glob, seat):
    st = shim_from_obs(planets, glob)
    return planet_features_jax(st, seat), global_features_jax(st, seat)


def _encode(params, enc, amt, planets, glob, seat):
    pf, gf = seat_feats(planets, glob, seat)
    tl, h, v = enc.apply({"params": params["enc"]}, pf, gf)   # [P,P+1],[P,H],scalar
    return tl, h, v


def _amount_logits(params, amt, h, tgt_clamped):
    h_tgt = h[tgt_clamped]                                     # [P,H] chosen-target ctx
    return amt.apply({"params": params["amt"]}, h, h_tgt)      # [P,N_BINS]


def seat_eval(params, enc, amt, planets, glob, seat, ti, ai):
    """logprob + value + entropy of stored action (ti,ai) for one seat/board."""
    tl, h, v = _encode(params, enc, amt, planets, glob, seat)
    owner = planets[:, 1].astype(jnp.int32)
    alive = planets[:, 7] > 0.5
    ships = planets[:, 5]
    nump = glob[3].astype(jnp.int32)
    P = owner.shape[0]
    tmask = target_mask(owner, alive, seat)
    tlm = jnp.where(tmask, tl, NEG)
    tlp_all = jax.nn.log_softmax(tlm)
    tlp = tlp_all[jnp.arange(P), ti]
    is_noop = ti == P
    tc = jnp.clip(ti, 0, P - 1)
    am = _amount_logits(params, amt, h, tc)
    alp = jax.nn.log_softmax(am)[jnp.arange(P), ai]
    owned = (owner == seat) & alive & (ships >= 2.0) & (seat < nump)
    lp = jnp.sum(jnp.where(owned, tlp + jnp.where(is_noop, 0.0, alp), 0.0))
    tent = -jnp.sum(jnp.where(tmask, jax.nn.softmax(tlm) * tlp_all, 0.0), axis=-1)
    ent = jnp.sum(jnp.where(owned, tent, 0.0))
    return lp, v, ent


def seat_act(params, enc, amt, planets, glob, seat, key):
    """Sample an action for one seat. Returns rows[P,3], lp, value, ti[P], ai[P]."""
    tl, h, v = _encode(params, enc, amt, planets, glob, seat)
    owner = planets[:, 1].astype(jnp.int32)
    alive = planets[:, 7] > 0.5
    ships = planets[:, 5]
    x, y, pids = planets[:, 2], planets[:, 3], planets[:, 0]
    nump = glob[3].astype(jnp.int32)
    P = owner.shape[0]
    tmask = target_mask(owner, alive, seat)
    tlm = jnp.where(tmask, tl, NEG)
    kt, ka = jax.random.split(key)
    ti = jax.random.categorical(kt, tlm, axis=-1)             # [P]
    is_noop = ti == P
    tc = jnp.clip(ti, 0, P - 1)
    am = _amount_logits(params, amt, h, tc)                   # [P,N_BINS]
    ai = jax.random.categorical(ka, am, axis=-1)             # [P]
    frac = AMOUNT_FRACS[ai]
    send = jnp.floor(ships * frac)
    owned = (owner == seat) & alive & (ships >= 2.0) & (seat < nump)
    act = owned & (~is_noop) & (send >= 1.0)
    tx, ty = x[tc], y[tc]
    ang = jnp.arctan2(ty - y, tx - x)
    rows = jnp.stack([jnp.where(act, pids, 0.0), jnp.where(act, ang, 0.0), jnp.where(act, send, 0.0)], axis=-1)
    tlp = jax.nn.log_softmax(tlm)[jnp.arange(P), ti]
    alp = jax.nn.log_softmax(am)[jnp.arange(P), ai]
    lp = jnp.sum(jnp.where(owned, tlp + jnp.where(is_noop, 0.0, alp), 0.0))
    return rows, lp, v, ti, ai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    args = ap.parse_args()

    cfg = EnvConfig(num_players=args.players, episode_steps=args.steps)
    enc = Encoder()
    amt = AmountHead()
    key = jax.random.PRNGKey(0)
    s0 = reset(jax.random.PRNGKey(1), cfg)
    o0 = build_observation(s0)
    pf0, gf0 = seat_feats(o0["planets"], o0["global"], 0)
    ke, ka = jax.random.split(key)
    enc_p = enc.init(ke, pf0, gf0)["params"]
    _, h0, _ = enc.apply({"params": enc_p}, pf0, gf0)
    amt_p = amt.init(ka, h0, h0)["params"]
    params = {"enc": enc_p, "amt": amt_p}
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.lr))
    opt_state = opt.init(params)
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"device={jax.default_backend()} policy_params={nparams} envs={args.envs} steps={args.steps} players={args.players}")

    seats = jnp.arange(MAX_PLAYERS)

    def rollout(params, key):
        keys = jax.random.split(key, args.envs)
        states = jax.vmap(lambda k: reset(k, cfg))(keys)

        def body(states, k):
            obs = jax.vmap(build_observation)(states)            # planets [B,P,9], global [B,6]
            aks = jax.random.split(k, args.envs)

            def env_act(planets, glob, kk):
                sks = jax.random.split(kk, MAX_PLAYERS)
                rows, lp, v, ti, ai = jax.vmap(
                    lambda se, kkk: seat_act(params, enc, amt, planets, glob, se, kkk))(seats, sks)
                return rows, lp, v, ti, ai

            rows, lp, v, ti, ai = jax.vmap(env_act)(obs["planets"], obs["global"], aks)
            ns, nobs, rew, done, info = jax.vmap(step)(states, rows)
            store = dict(planets=obs["planets"], glob=obs["global"], ti=ti, ai=ai,
                         lp=lp, v=v, rew=rew, done=done)
            return ns, store

        _, traj = jax.lax.scan(body, states, jax.random.split(key, args.steps))
        return traj

    def compute_gae(rew, val, done):
        T = rew.shape[0]
        done = done[..., None]

        def sc(carry, t):
            adv, nv = carry
            d = 1.0 - done[t]
            delta = rew[t] + args.gamma * nv * d - val[t]
            adv = delta + args.gamma * args.lam * d * adv
            return (adv, val[t]), adv

        _, advs = jax.lax.scan(sc, (jnp.zeros_like(val[0]), jnp.zeros_like(val[0])), jnp.arange(T)[::-1])
        advs = advs[::-1]
        return advs, advs + val

    @jax.jit
    def update(params, opt_state, key):
        traj = rollout(params, key)
        advs, returns = compute_gae(traj["rew"], traj["v"], traj["done"])
        advn = (advs - advs.mean()) / (advs.std() + 1e-8)

        def loss_fn(params):
            # recompute logprob/value/entropy for every (T,B,seat)
            def per_env(planets, glob, ti, ai):
                return jax.vmap(lambda se, t, a: seat_eval(params, enc, amt, planets, glob, se, t, a))(seats, ti, ai)
            # vmap over T and B
            new_lp, new_v, ent = jax.vmap(jax.vmap(per_env))(
                traj["planets"], traj["glob"], traj["ti"], traj["ai"])
            ratio = jnp.exp(new_lp - traj["lp"])
            unclipped = ratio * advn
            clipped = jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * advn
            pg = -jnp.mean(jnp.minimum(unclipped, clipped))
            vloss = jnp.mean((returns - new_v) ** 2)
            entropy = jnp.mean(ent)
            loss = pg + args.vf * vloss - args.ent * entropy
            return loss, (pg, vloss, entropy, jnp.mean(jnp.abs(ratio - 1)))

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        pg, vloss, entropy, clipfrac = aux
        frac_pos = jnp.mean((traj["rew"][-1] > 0).astype(jnp.float32))
        return params2, opt_state2, dict(loss=loss, pg=pg, vloss=vloss, ent=entropy,
                                         clipdev=clipfrac, ret=jnp.mean(returns), frac_pos=frac_pos)

    for u in range(args.updates):
        key, k = jax.random.split(key)
        t0 = time.time()
        params, opt_state, m = update(params, opt_state, k)
        jax.block_until_ready(params)
        dt = time.time() - t0
        print(f"update {u}: {dt:.1f}s loss={float(m['loss']):.4f} pg={float(m['pg']):.4f} "
              f"vloss={float(m['vloss']):.3f} ent={float(m['ent']):.3f} ret={float(m['ret']):.3f} "
              f"frac_pos={float(m['frac_pos']):.3f}")
    print("done.")


if __name__ == "__main__":
    main()
