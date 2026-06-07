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
    """Load a BC checkpoint from either a run directory or a .pt file.

    Training saves ``<out_dir>/checkpoint.pt`` with an embedded ``model_config``.
    Ad-hoc eval artifacts may instead be provided as ``checkpoint.pt`` next to
    ``model_config.json``.  Support both layouts so runtime simulation can point
    directly at the provided model file.
    """
    path = Path(path)
    checkpoint_file = path / "checkpoint.pt" if path.is_dir() else path
    ckpt = torch.load(checkpoint_file, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_file}")

    model_config = ckpt.get("model_config")
    if model_config is None:
        config_file = checkpoint_file.with_name("model_config.json")
        if not config_file.exists():
            raise RuntimeError(f"Checkpoint {checkpoint_file} does not contain model_config and {config_file} is missing")
        model_config = load_json(config_file)

    state = ckpt.get("model_state", ckpt)
    cfg = BCModelConfig(**model_config)
    model = EntityBCPolicy(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt

