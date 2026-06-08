from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from orbit_training_prep.schema import AMOUNT_BIN_NAMES, NOOP_TARGET_SLOT, P_MAX


@dataclass
class BCModelConfig:
    planet_feature_dim: int
    global_feature_dim: int
    target_state_feature_dim: int = 0
    pair_feature_dim: int = 0
    feature_version: str = "v1"
    fleet_feature_dim: int = 0
    max_planets: int = P_MAX
    target_classes: int = P_MAX + 1
    amount_bins: int = len(AMOUNT_BIN_NAMES)
    noop_target_slot: int = NOOP_TARGET_SLOT
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mlp_size: int = 256
    dropout: float = 0.0


@dataclass
class BCTrainConfig:
    train_dir: str
    valid_dir: str
    out_dir: str
    batch_size: int = 512
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mlp_size: int = 256
    dropout: float = 0.0


def save_json_dataclass(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(obj), f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(device: str | None = "auto") -> torch.device:
    requested = (device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment is using a CPU-only "
            f"PyTorch build (torch {torch.__version__}, torch.version.cuda={torch.version.cuda}). "
            "Install a CUDA-enabled PyTorch wheel in this environment or run with --device cpu."
        )
    return torch.device(requested)
