from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from .config import PPOConfig, save_config


def save_ppo_checkpoint(path: str | Path, policy, optimizer, config: PPOConfig, update: int, metrics: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "update": int(update),
            "metrics": metrics or {},
            "ppo_config": asdict(config),
            "bc_model_config": asdict(policy.config),
        },
        path / "checkpoint.pt",
    )
    save_config(config, path / "config.json")
    with open(path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics or {}, f, indent=2, sort_keys=True)


def load_ppo_checkpoint(path: str | Path, *, device: str | torch.device = "cpu"):
    from .config import PPOConfig
    from .policy import PPOPolicy

    path = Path(path)
    checkpoint_file = path / "checkpoint.pt" if path.is_dir() else path
    ckpt = torch.load(checkpoint_file, map_location=device)
    cfg = PPOConfig(**ckpt["ppo_config"])
    policy = PPOPolicy.from_bc_checkpoint(cfg.bc_checkpoint, device=device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()
    return policy, cfg, ckpt

