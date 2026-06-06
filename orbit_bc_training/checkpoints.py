from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .config import BCModelConfig, load_json
from .model import EntityBCPolicy


def save_checkpoint(path: str | Path, model: EntityBCPolicy, optimizer, epoch: int, metrics: dict, model_config: BCModelConfig) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "metrics": metrics,
            "model_config": asdict(model_config),
        },
        path / "checkpoint.pt",
    )


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[EntityBCPolicy, dict]:
    path = Path(path)
    ckpt = torch.load(path / "checkpoint.pt", map_location=device)
    cfg = BCModelConfig(**ckpt["model_config"])
    model = EntityBCPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt

