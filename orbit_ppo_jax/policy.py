"""JAX-native entity-transformer policy for Orbit Wars PPO self-play.

Mirrors the BC model (transformer over [global ⊕ planet] tokens, factored
target+amount head) but JAX/Flax-native so it runs inside the vmapped GPU
rollout. Action factorization per owned source planet i:
    target_j ~ Categorical(target_logits[i, :P+1])   # +1 = no-op
    amount_b ~ Categorical(amount_logits[i, :])      # given chosen target j*

Two modules so the amount head is evaluated only for the *chosen* target per
source ([P,N_BINS]) instead of all P×P pairs (the latter OOMs a 6GB card).
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

P_MAX = 64
AMOUNT_FRACS = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=jnp.float32)
N_BINS = AMOUNT_FRACS.shape[0]
NEG = -1e9


class Encoder(nn.Module):
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    mlp: int = 256

    @nn.compact
    def __call__(self, planet_feats, global_feats):
        H = self.hidden
        pt = nn.Dense(H)(planet_feats)                  # [P,H]
        gt = nn.Dense(H)(global_feats)[None, :]         # [1,H]
        x = jnp.concatenate([gt, pt], axis=0)           # [P+1,H]
        for _ in range(self.layers):
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadDotProductAttention(num_heads=self.heads)(y, y)
            x = x + y
            y = nn.LayerNorm()(x)
            y = nn.Dense(self.mlp)(y)
            y = nn.gelu(y)
            y = nn.Dense(H)(y)
            x = x + y
        g = x[0]
        h = x[1:]                                       # [P,H]
        q = nn.Dense(H)(h)
        scores = q @ h.T / jnp.sqrt(H)                  # [P,P]
        noop = nn.Dense(1)(h)                           # [P,1]
        target_logits = jnp.concatenate([scores, noop], axis=-1)   # [P,P+1]
        value = nn.Dense(1)(nn.gelu(nn.Dense(H)(g)))[0]
        return target_logits, h, value


class AmountHead(nn.Module):
    hidden: int = 128
    mlp: int = 256

    @nn.compact
    def __call__(self, h_src, h_tgt):
        # h_src,h_tgt [P,H] (h_tgt already gathered for the chosen target)
        z = jnp.concatenate([h_src, h_tgt], axis=-1)    # [P,2H]
        z = nn.gelu(nn.Dense(self.mlp)(z))
        return nn.Dense(N_BINS)(z)                       # [P,N_BINS]


def target_mask(owner, alive, seat):
    """[P,P+1] legal targets per source: alive & not-self & not-own; noop always."""
    P = owner.shape[0]
    non_own = alive & (owner != seat)
    tgt = jnp.broadcast_to(non_own[None, :], (P, P)) & (~jnp.eye(P, dtype=bool))
    return jnp.concatenate([tgt, jnp.ones((P, 1), dtype=bool)], axis=-1)
