from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint
from .config import resolve_device
from .dataset import OrbitBCDataset, collate_bc_samples
from .losses import apply_mask, bc_loss_and_metrics


STEP_BUCKETS = [(0, 100), (100, 250), (250, 430), (430, 501)]


def _safe_div(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def evaluate(checkpoint: str | Path, valid_dir: str | Path, device: str = "auto") -> dict:
    resolved_device = resolve_device(device)
    model, _ = load_checkpoint(checkpoint, device=str(resolved_device))
    ds = OrbitBCDataset(valid_dir)
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_bc_samples)
    all_rows: list[dict] = []
    metric_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(resolved_device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch)
            _, metrics = bc_loss_and_metrics(outputs, batch)
            metric_rows.append(metrics)
            target_logits = apply_mask(outputs["target_logits"], batch["target_mask"])
            pred = target_logits.argmax(dim=1)
            top3 = target_logits.topk(k=3, dim=1).indices
            amount_pred = apply_mask(outputs["amount_logits"], batch["amount_mask"]).argmax(dim=1)
            for i in range(pred.shape[0]):
                all_rows.append(
                    {
                        "pred": int(pred[i].cpu()),
                        "true": int(batch["target_label"][i].cpu()),
                        "amount_pred": int(amount_pred[i].cpu()),
                        "amount_true": int(batch["amount_label"][i].cpu()),
                        "is_noop": bool(batch["is_noop"][i].cpu()),
                        "step": int(batch["step"][i].cpu()),
                        "top3": [int(x) for x in top3[i].cpu().tolist()],
                    }
                )
    noop_slot = model.config.noop_target_slot
    n = len(all_rows)
    non_noop = [r for r in all_rows if not r["is_noop"]]
    noop_true = [r for r in all_rows if r["is_noop"]]
    noop_pred = [r for r in all_rows if r["pred"] == noop_slot]
    launch_true = non_noop
    launch_pred = [r for r in all_rows if r["pred"] != noop_slot]
    report = {
        "num_samples": n,
        "target_accuracy": _safe_div(sum(r["pred"] == r["true"] for r in all_rows), n),
        "target_non_noop_accuracy": _safe_div(sum(r["pred"] == r["true"] for r in non_noop), len(non_noop)),
        "amount_accuracy": _safe_div(sum(r["amount_pred"] == r["amount_true"] for r in all_rows), n),
        "noop_precision": _safe_div(sum(r["is_noop"] for r in noop_pred), len(noop_pred)),
        "noop_recall": _safe_div(sum(r["pred"] == noop_slot for r in noop_true), len(noop_true)),
        "launch_recall": _safe_div(sum(r["pred"] != noop_slot for r in launch_true), len(launch_true)),
        "top1_target_accuracy": _safe_div(sum(r["pred"] == r["true"] for r in all_rows), n),
        "top3_target_accuracy": _safe_div(sum(r["true"] in r["top3"] for r in all_rows), n),
        "confusion": {
            "true_noop_pred_noop": sum(r["is_noop"] and r["pred"] == noop_slot for r in all_rows),
            "true_noop_pred_launch": sum(r["is_noop"] and r["pred"] != noop_slot for r in all_rows),
            "true_launch_pred_noop": sum((not r["is_noop"]) and r["pred"] == noop_slot for r in all_rows),
            "true_launch_pred_launch": sum((not r["is_noop"]) and r["pred"] != noop_slot for r in all_rows),
        },
        "batch_metrics_mean": {k: float(np.mean([m[k] for m in metric_rows])) for k in metric_rows[0]} if metric_rows else {},
        "step_buckets": {},
    }
    for lo, hi in STEP_BUCKETS:
        rows = [r for r in all_rows if lo <= r["step"] < hi]
        report["step_buckets"][f"{lo}-{hi if hi < 501 else 500}"] = {
            "samples": len(rows),
            "target_accuracy": _safe_div(sum(r["pred"] == r["true"] for r in rows), len(rows)),
            "launch_recall": _safe_div(sum(r["pred"] != noop_slot for r in rows if not r["is_noop"]), sum(not r["is_noop"] for r in rows)),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--valid_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", help="Eval device: auto, cpu, cuda, or cuda:N. auto prefers CUDA when available.")
    args = ap.parse_args()
    report = evaluate(args.checkpoint, args.valid_dir, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
