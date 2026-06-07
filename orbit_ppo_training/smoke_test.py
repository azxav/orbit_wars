from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .bc_reference import FrozenBCReference
from .checkpointing import load_ppo_checkpoint, save_ppo_checkpoint
from .config import PPOConfig, save_config
from .eval_ppo import evaluate
from .policy import PPOPolicy
from .rollout_worker import collect_rollouts
from .train_ppo import _flatten_with_advantages, run_ppo_update


def smoke_test(bc_checkpoint: str, out_dir: str, *, device: str = "cpu") -> dict:
    out = Path(out_dir)
    config = PPOConfig(
        bc_checkpoint=bc_checkpoint,
        out_dir=str(out),
        players=4,
        opponent="simple_expand",
        rollout_games_per_update=1,
        updates=1,
        eval_games=2,
        minibatch_size=128,
        device=device,
        seed=123,
    )
    out.mkdir(parents=True, exist_ok=True)
    save_config(config, out / "config.json")
    policy = PPOPolicy.from_bc_checkpoint(bc_checkpoint, device=device)
    bc_ref = FrozenBCReference(bc_checkpoint, device=device)
    rollout = collect_rollouts(policy, config, games=1, deterministic=False, seed_start=config.seed)
    if not rollout.trajectories or not rollout.trajectories[0]:
        raise RuntimeError("smoke rollout produced no trajectory decisions")
    if rollout.summary["illegal_action_count"] != 0:
        raise RuntimeError(f"illegal actions during smoke rollout: {rollout.summary}")
    records, advantages, returns = _flatten_with_advantages(rollout.trajectories, config, torch.device(device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    metrics = run_ppo_update(policy, optimizer, bc_ref, records, advantages, returns, config)
    if not all(torch.isfinite(torch.tensor(float(v))).item() for v in metrics.values()):
        raise RuntimeError(f"non-finite smoke metrics: {metrics}")
    save_ppo_checkpoint(out / "latest", policy, optimizer, config, 1, metrics)
    loaded_policy, _, _ = load_ppo_checkpoint(out / "latest", device=device)
    del loaded_policy
    eval_summary = evaluate(str(out / "latest"), opponent="simple_expand", players=4, num_games=2, out_dir=str(out / "eval"), seed=999, device=device)
    summary = {"rollout": rollout.summary, "loss": metrics, "eval": eval_summary, "checkpoint": str(out / "latest" / "checkpoint.pt")}
    with open(out / "smoke_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the PPO smoke test.")
    ap.add_argument("--bc_checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(smoke_test(args.bc_checkpoint, args.out_dir, device=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

