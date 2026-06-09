from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoints import save_checkpoint
from .config import BCModelConfig, BCTrainConfig, resolve_device, save_json_dataclass
from .dataset import OrbitBCDataset, collate_bc_samples
from .losses import bc_loss_and_metrics
from .model import EntityBCPolicy
from orbit_bc_eval.gameplay_score import score_eval_dir


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


def run_epoch(model, loader, device, optimizer=None, grad_clip: float = 1.0) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    metrics: list[dict[str, float]] = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            loss, m = bc_loss_and_metrics(model(batch), batch)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        metrics.append(m)
    return _mean_metrics(metrics)


def checkpoint_selection_metrics(*, valid_metrics: dict[str, float], selection_eval_dir: str | None) -> dict:
    gameplay_score = score_eval_dir(selection_eval_dir)
    if gameplay_score is None:
        return {
            "best_selection_mode": "validation_debug_fallback",
            "true_best": False,
        }
    return {
        "best_selection_mode": "gameplay_eval",
        "true_best": True,
        "gameplay_score": float(gameplay_score),
    }


def train(config: BCTrainConfig) -> dict[str, float]:
    _seed(config.seed)
    device = resolve_device(config.device)
    print(json.dumps({"device": str(device), "cuda_available": torch.cuda.is_available(), "torch": torch.__version__}, sort_keys=True))
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = OrbitBCDataset(config.train_dir, feature_version=config.feature_version)
    valid_ds = OrbitBCDataset(config.valid_dir, feature_version=config.feature_version)
    sample = train_ds[0]
    model_cfg = BCModelConfig(
        planet_feature_dim=int(sample["planet_features"].shape[-1]),
        global_feature_dim=int(sample["global_features"].shape[-1]),
        target_state_feature_dim=int(sample.get("target_state_features", np.zeros((0, 0))).shape[-1]),
        pair_feature_dim=int(sample.get("pair_features", np.zeros((0, 0))).shape[-1]),
        feature_version=str(sample.get("feature_version", "v1")),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_size=config.mlp_size,
        dropout=config.dropout,
    )
    save_json_dataclass(out_dir / "config.json", config)
    save_json_dataclass(out_dir / "model_config.json", model_cfg)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate_bc_samples)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_bc_samples)
    model = EntityBCPolicy(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    best_score = float("-inf")
    latest_metrics: dict[str, float] = {}
    with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as metrics_file:
        for epoch in range(1, config.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer, config.grad_clip)
            valid_metrics = run_epoch(model, valid_loader, device)
            scheduler.step()
            latest_metrics = {f"train_{k}": v for k, v in train_metrics.items()} | {f"valid_{k}": v for k, v in valid_metrics.items()}
            latest_metrics["epoch"] = epoch
            latest_metrics["lr"] = float(scheduler.get_last_lr()[0])
            selection_metrics = checkpoint_selection_metrics(valid_metrics=valid_metrics, selection_eval_dir=config.selection_eval_dir)
            latest_metrics.update(selection_metrics)
            metrics_file.write(json.dumps(latest_metrics, sort_keys=True) + "\n")
            metrics_file.flush()
            save_checkpoint(out_dir / "latest", model, optimizer, epoch, latest_metrics, model_cfg)
            score = float(selection_metrics["gameplay_score"]) if selection_metrics.get("true_best") else float("-inf")
            if selection_metrics.get("true_best") and score >= best_score:
                best_score = score
                save_checkpoint(out_dir / "best", model, optimizer, epoch, latest_metrics, model_cfg)
            elif not selection_metrics.get("true_best"):
                save_checkpoint(out_dir / "best", model, optimizer, epoch, latest_metrics, model_cfg)
            print(json.dumps(latest_metrics, sort_keys=True))
    return latest_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--valid_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--selection_eval_dir", default=None)
    ap.add_argument("--feature_version", choices=["auto", "v1", "v2"], default="v2")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, or cuda:N. auto prefers CUDA when available.")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--mlp_size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()
    train(BCTrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
