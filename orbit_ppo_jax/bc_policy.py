from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch


ArrayTree = dict[str, Any]


def _compute_dtype(config: dict[str, Any]) -> jnp.dtype:
    name = str(config.get("compute_dtype", "float32")).lower()
    if name in {"float32", "fp32", "none"}:
        return jnp.float32
    if name in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if name in {"float16", "fp16"}:
        return jnp.float16
    raise RuntimeError(f"unsupported JAX BC compute dtype: {name}")


def _cast_floating_tree(tree: Any, dtype: jnp.dtype) -> Any:
    def cast_leaf(x):
        if x is None:
            return None
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(cast_leaf, tree, is_leaf=lambda x: x is None)


def _cast_floating_batch(batch: dict[str, jax.Array], dtype: jnp.dtype) -> dict[str, jax.Array]:
    return {
        key: value.astype(dtype) if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating) else value
        for key, value in batch.items()
    }


def _to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.asarray(t.detach().cpu().numpy(), dtype=jnp.float32)


def _linear_params(state: dict[str, torch.Tensor], prefix: str) -> dict[str, jax.Array]:
    return {"weight": _to_jax(state[f"{prefix}.weight"]), "bias": _to_jax(state[f"{prefix}.bias"])}


def _seq2_params(state: dict[str, torch.Tensor], prefix: str) -> dict[str, dict[str, jax.Array]]:
    return {"linear0": _linear_params(state, f"{prefix}.0"), "linear2": _linear_params(state, f"{prefix}.2")}


def _layer_params(state: dict[str, torch.Tensor], prefix: str) -> dict[str, Any]:
    return {
        "self_attn": {
            "in_proj_weight": _to_jax(state[f"{prefix}.self_attn.in_proj_weight"]),
            "in_proj_bias": _to_jax(state[f"{prefix}.self_attn.in_proj_bias"]),
            "out_proj": _linear_params(state, f"{prefix}.self_attn.out_proj"),
        },
        "linear1": _linear_params(state, f"{prefix}.linear1"),
        "linear2": _linear_params(state, f"{prefix}.linear2"),
        "norm1": {"weight": _to_jax(state[f"{prefix}.norm1.weight"]), "bias": _to_jax(state[f"{prefix}.norm1.bias"])},
        "norm2": {"weight": _to_jax(state[f"{prefix}.norm2.weight"]), "bias": _to_jax(state[f"{prefix}.norm2.bias"])},
    }


def load_bc_jax_params(checkpoint: str | Path) -> tuple[ArrayTree, dict[str, Any]]:
    """Load a PyTorch BC checkpoint into a functional JAX parameter tree."""
    ckpt = torch.load(Path(checkpoint), map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "model_config" not in ckpt:
        raise RuntimeError(f"Unsupported BC checkpoint format: {checkpoint}")
    state: dict[str, torch.Tensor] = ckpt["model_state"]
    cfg = dict(ckpt["model_config"])
    layers = [_layer_params(state, f"encoder.layers.{i}") for i in range(int(cfg["num_layers"]))]
    params: ArrayTree = {
        "planet_encoder": _linear_params(state, "planet_encoder"),
        "global_encoder": _linear_params(state, "global_encoder"),
        "layers": layers,
        "target_state_encoder": _linear_params(state, "target_state_encoder")
        if int(cfg.get("target_state_feature_dim", 0)) > 0 and "target_state_encoder.weight" in state
        else None,
        "target_pair": _seq2_params(state, "target_pair"),
        "noop_head": _seq2_params(state, "noop_head"),
        "noop_target_context": _to_jax(state["noop_target_context"]),
        "amount_head": _seq2_params(state, "amount_head"),
    }
    return params, cfg


def _linear(x: jax.Array, p: dict[str, jax.Array]) -> jax.Array:
    return jnp.matmul(x, jnp.swapaxes(p["weight"], -1, -2)) + p["bias"]


def _layer_norm(x: jax.Array, p: dict[str, jax.Array], eps: float = 1.0e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * p["weight"] + p["bias"]


def _mlp2(x: jax.Array, p: dict[str, dict[str, jax.Array]]) -> jax.Array:
    return _linear(jax.nn.gelu(_linear(x, p["linear0"]), approximate=False), p["linear2"])


def _self_attention(x: jax.Array, p: dict[str, Any], num_heads: int) -> jax.Array:
    bsz, tokens, hidden = x.shape
    head_dim = hidden // int(num_heads)
    qkv = _linear(x, {"weight": p["in_proj_weight"], "bias": p["in_proj_bias"]})
    q, k, v = jnp.split(qkv, 3, axis=-1)

    def split_heads(a: jax.Array) -> jax.Array:
        return jnp.transpose(a.reshape((bsz, tokens, int(num_heads), head_dim)), (0, 2, 1, 3))

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
    weights = jax.nn.softmax(jnp.matmul(qh, jnp.swapaxes(kh, -1, -2)) / jnp.sqrt(float(head_dim)), axis=-1)
    ctx = jnp.matmul(weights, vh)
    merged = jnp.transpose(ctx, (0, 2, 1, 3)).reshape((bsz, tokens, hidden))
    return _linear(merged, p["out_proj"])


def _transformer_layer(x: jax.Array, p: dict[str, Any], num_heads: int) -> jax.Array:
    y = _layer_norm(x, p["norm1"])
    x = x + _self_attention(y, p["self_attn"], num_heads)
    y = _layer_norm(x, p["norm2"])
    x = x + _linear(jax.nn.gelu(_linear(y, p["linear1"]), approximate=False), p["linear2"])
    return x


def encode_context(params: ArrayTree, batch: dict[str, jax.Array], config: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
    planets = batch["planet_features"]
    global_features = batch["global_features"]
    pmax = planets.shape[1]
    planet_tokens = _linear(planets, params["planet_encoder"])
    global_token = _linear(global_features, params["global_encoder"])[:, None, :]
    encoded = jnp.concatenate([global_token, planet_tokens], axis=1)
    for layer in params["layers"]:
        encoded = _transformer_layer(encoded, layer, int(config["num_heads"]))
    global_ctx = encoded[:, 0]
    planet_ctx = encoded[:, 1 : pmax + 1]
    target_state = batch.get("target_state_features")
    if params.get("target_state_encoder") is not None and target_state is not None:
        planet_ctx = planet_ctx + _linear(target_state, params["target_state_encoder"])
    return global_ctx, planet_ctx


def bc_forward(params: ArrayTree, batch: dict[str, jax.Array], config: dict[str, Any]) -> dict[str, jax.Array]:
    compute_dtype = _compute_dtype(config)
    if compute_dtype != jnp.float32:
        params = _cast_floating_tree(params, compute_dtype)
        batch = _cast_floating_batch(batch, compute_dtype)
    global_ctx, planet_ctx = encode_context(params, batch, config)
    bsz, pmax, hidden = planet_ctx.shape
    source_idx = jnp.clip(batch["source_slot"].astype(jnp.int32), 0, pmax - 1)
    batch_idx = jnp.arange(bsz)
    source_ctx = planet_ctx[batch_idx, source_idx]
    source_expanded = jnp.broadcast_to(source_ctx[:, None, :], (bsz, pmax, hidden))
    global_expanded = jnp.broadcast_to(global_ctx[:, None, :], (bsz, pmax, hidden))
    pair_features = batch.get("pair_features")
    pair_dim = int(config.get("pair_feature_dim", 0))
    if pair_features is None or pair_features.shape[-1] != pair_dim:
        pair_features = jnp.zeros((bsz, pmax + 1, pair_dim), dtype=planet_ctx.dtype)
    target_pair_features = pair_features[:, :pmax]
    target_logits = _mlp2(
        jnp.concatenate([source_expanded, planet_ctx, target_pair_features, global_expanded], axis=-1),
        params["target_pair"],
    ).squeeze(-1)
    noop_logit = _mlp2(source_ctx, params["noop_head"])
    target_logits = jnp.concatenate([target_logits, noop_logit], axis=1)

    teacher_target = batch.get("target_label")
    if teacher_target is None:
        teacher_target = jnp.argmax(target_logits, axis=1).astype(jnp.int32)
    teacher_target = teacher_target.astype(jnp.int32)
    target_idx = jnp.clip(teacher_target, 0, pmax - 1)
    target_ctx = planet_ctx[batch_idx, target_idx]
    selected_pair = pair_features[batch_idx, jnp.clip(teacher_target, 0, pmax)]
    noop_ctx = jnp.broadcast_to(params["noop_target_context"][None, :], target_ctx.shape)
    target_ctx = jnp.where((teacher_target == int(config["noop_target_slot"]))[:, None], noop_ctx, target_ctx)
    amount_logits = _mlp2(jnp.concatenate([source_ctx, target_ctx, selected_pair, global_ctx], axis=1), params["amount_head"])
    return {"target_logits": target_logits, "amount_logits": amount_logits, "global_ctx": global_ctx}


def init_value_head(key: jax.Array, hidden_size: int) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(key)
    scale = 1.0 / jnp.sqrt(float(hidden_size))
    return {
        "weight": jax.random.normal(k1, (int(hidden_size),), dtype=jnp.float32) * scale,
        "bias": jnp.asarray(0.0, dtype=jnp.float32),
    }


def value_apply(params: dict[str, jax.Array], global_ctx: jax.Array) -> jax.Array:
    return jnp.matmul(global_ctx, params["weight"]) + params["bias"]


def tree_to_numpy(tree: ArrayTree) -> ArrayTree:
    return jax.tree_util.tree_map(lambda x: np.asarray(x) if x is not None else None, tree)
