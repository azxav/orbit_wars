from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path
from typing import Any

import torch

from .advantages import compute_gae, normalize_advantages
from .bc_reference import FrozenBCReference
from .checkpointing import save_ppo_checkpoint
from .config import PPOConfig, save_config
from .metrics import append_jsonl
from .policy import PPOPolicy
from .ppo_loss import ppo_loss
from .rollout_worker import collect_rollouts
from .trajectory import DecisionRecord, collate_decisions


def _seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _flatten_with_advantages(trajectories: list[list[DecisionRecord]], config: PPOConfig, device: torch.device) -> tuple[list[DecisionRecord], torch.Tensor, torch.Tensor]:
    records: list[DecisionRecord] = []
    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for traj in trajectories:
        if not traj:
            continue
        rewards = torch.as_tensor([r.reward for r in traj], dtype=torch.float32, device=device)
        values = torch.as_tensor([r.value for r in traj], dtype=torch.float32, device=device)
        dones = torch.as_tensor([r.done for r in traj], dtype=torch.float32, device=device)
        adv, ret = compute_gae(rewards, values, dones, config.gamma, config.gae_lambda)
        records.extend(traj)
        advantages.append(adv)
        returns.append(ret)
    if not records:
        raise RuntimeError("rollout produced no controlled decisions")
    adv_all = normalize_advantages(torch.cat(advantages))
    ret_all = torch.cat(returns)
    return records, adv_all, ret_all


def run_ppo_update(policy: PPOPolicy, optimizer: torch.optim.Optimizer, bc_ref: FrozenBCReference, records: list[DecisionRecord], advantages: torch.Tensor, returns: torch.Tensor, config: PPOConfig) -> dict[str, float]:
    device = torch.device(config.device)
    batch = collate_decisions(records, device=device)
    n = len(records)
    metrics_accum: list[dict[str, float]] = []
    early_stop = False
    for _epoch in range(int(config.ppo_epochs)):
        order = torch.randperm(n, device=device)
        for start in range(0, n, int(config.minibatch_size)):
            idx = order[start : start + int(config.minibatch_size)]
            mb = {k: v[idx] for k, v in batch.items()}
            loss, metrics = ppo_loss(
                policy,
                mb,
                advantages=advantages[idx],
                returns=returns[idx],
                clip_range=config.clip_range,
                value_coef=config.value_coef,
                entropy_coef=config.entropy_coef,
                bc_ref=bc_ref,
                kl_to_bc_coef=config.kl_to_bc_coef,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite PPO loss: {metrics}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.max_grad_norm))
            optimizer.step()
            metrics["grad_norm"] = float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            metrics_accum.append(metrics)
            if metrics["approx_kl"] > float(config.target_kl) * 2.0:
                early_stop = True
                break
        if early_stop:
            break
    keys = sorted({k for m in metrics_accum for k in m})
    out = {k: sum(float(m.get(k, 0.0)) for m in metrics_accum) / max(1, len(metrics_accum)) for k in keys}
    out["minibatches"] = float(len(metrics_accum))
    out["early_stop_kl"] = float(early_stop)
    return out


def train(config: PPOConfig) -> dict[str, Any]:
    _seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.json")
    device = torch.device(config.device)
    policy = PPOPolicy.from_bc_checkpoint(config.bc_checkpoint, device=device)
    bc_ref = FrozenBCReference(config.bc_checkpoint, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.lr))
    best_score = -1.0e9
    last_eval: dict[str, Any] = {}
    collapse_clip_streak = 0
    for update in range(1, int(config.updates) + 1):
        save_ppo_checkpoint(out_dir / "rollback" / f"update_{update:05d}", policy, optimizer, config, update - 1, {"rollback_before_update": update})
        policy.eval()
        rollout = collect_rollouts(policy, config, games=config.rollout_games_per_update, deterministic=False, seed_start=config.seed + update * 10000)
        records, advantages, returns = _flatten_with_advantages(rollout.trajectories, config, device)
        policy.train()
        loss_metrics = run_ppo_update(policy, optimizer, bc_ref, records, advantages, returns, config)
        update_metrics: dict[str, Any] = {
            "update": update,
            "decisions": len(records),
            **rollout.summary,
            **loss_metrics,
        }
        collapse_clip_streak = collapse_clip_streak + 1 if update_metrics.get("clip_frac", 0.0) > 0.35 else 0
        update_metrics["collapse_clip_streak"] = collapse_clip_streak
        if update % int(config.eval_interval_updates) == 0 or update == 1:
            policy.eval()
            eval_rollout = collect_rollouts(policy, config, games=int(config.eval_games), deterministic=True, seed_start=config.seed + 500000 + update * 1000)
            last_eval = eval_rollout.summary
            update_metrics.update({f"eval_{k}": v for k, v in last_eval.items()})
            with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
                json.dump(last_eval, f, indent=2, sort_keys=True)
            score = float(last_eval.get("average_final_reward", 0.0))
            if score > best_score:
                best_score = score
                save_ppo_checkpoint(out_dir / "best", policy, optimizer, config, update, update_metrics)
        append_jsonl(out_dir / "metrics.jsonl", update_metrics)
        save_ppo_checkpoint(out_dir / "latest", policy, optimizer, config, update, update_metrics)
        if update % int(config.save_interval_updates) == 0:
            save_ppo_checkpoint(out_dir / "checkpoints" / f"update_{update:05d}", policy, optimizer, config, update, update_metrics)
        if update_metrics.get("illegal_action_count", 0) > 0 or update_metrics.get("timeout_count", 0) > 0 or collapse_clip_streak >= 3:
            update_metrics["stopped_for_collapse_guard"] = True
            append_jsonl(out_dir / "metrics.jsonl", update_metrics)
            break
    return {"out_dir": str(out_dir), "best_score": best_score, "last_eval": last_eval}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train PPO from an Orbit Wars BC checkpoint.")
    ap.add_argument("--bc_checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--players", type=int, default=4, choices=[2, 4])
    ap.add_argument("--opponent", default="orbit_wars_base", choices=["random", "passive", "simple_expand", "orbit_wars_base", "heuristic_path", "self_play"])
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--rollout_games_per_update", type=int, default=32)
    ap.add_argument("--updates", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip_range", type=float, default=0.10)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--kl_to_bc_coef", type=float, default=0.02)
    ap.add_argument("--target_kl", type=float, default=0.03)
    ap.add_argument("--ppo_epochs", type=int, default=2)
    ap.add_argument("--minibatch_size", type=int, default=512)
    ap.add_argument("--eval_interval_updates", type=int, default=5)
    ap.add_argument("--save_interval_updates", type=int, default=5)
    ap.add_argument("--eval_games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--heuristic_path", default="orbit_wars_base.py")
    ap.add_argument("--max_episode_steps", type=int, default=500)
    return ap


def config_from_args(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(**vars(args))


def main() -> None:
    config = config_from_args(build_arg_parser().parse_args())
    print(json.dumps(train(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

