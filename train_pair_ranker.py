from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rows_to_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("No pair-ranker rows loaded")
    x = np.asarray([r["features"] for r in rows], dtype=np.float32)
    y = np.asarray([r["label"] for r in rows], dtype=np.float32)
    w = np.asarray([r.get("sample_weight", r.get("train_weight", 1.0)) for r in rows], dtype=np.float32)
    return x, y, w


def standardize(train_x: np.ndarray, valid_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (valid_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(y_score, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def top1_target_accuracy(rows: list[dict[str, Any]], scores: np.ndarray) -> float:
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        groups[str(row["group_uid"])].append((float(score), int(row["label"])))
    if not groups:
        return float("nan")
    correct = 0
    for candidates in groups.values():
        best_score, best_label = max(candidates, key=lambda item: item[0])
        correct += int(best_label == 1)
    return float(correct / len(groups))


def random_top1_baseline(rows: list[dict[str, Any]]) -> float:
    sizes: dict[str, int] = defaultdict(int)
    for row in rows:
        sizes[str(row["group_uid"])] += 1
    if not sizes:
        return float("nan")
    return float(np.mean([1.0 / max(1, size) for size in sizes.values()]))


class PairRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
) -> PairRanker:
    torch.manual_seed(int(seed))
    model = PairRanker(train_x.shape[1], hidden_dim=hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    x_t = torch.as_tensor(train_x, dtype=torch.float32)
    y_t = torch.as_tensor(train_y, dtype=torch.float32)
    w_t = torch.as_tensor(train_w, dtype=torch.float32)
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = model(x_t)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_t, weight=w_t)
        loss.backward()
        opt.step()
    return model


def predict_scores(model: PairRanker, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


def evaluate(rows: list[dict[str, Any]], y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auc": binary_auc(y, scores),
        "top1_target_accuracy": top1_target_accuracy(rows, scores),
        "random_top1_baseline": random_top1_baseline(rows),
        "positive_rate": float(np.mean(y)),
    }


def default_pair_paths(dataset_root: str | Path) -> tuple[Path, Path]:
    root = Path(dataset_root)
    return root / "train" / "pair_rank_rows.jsonl", root / "valid" / "pair_rank_rows.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="./orbit_dataset_work/combined")
    ap.add_argument("--train", default=None, help="Train pair_rank_rows.jsonl path. Defaults to dataset_root/train/pair_rank_rows.jsonl.")
    ap.add_argument("--valid", default=None, help="Valid pair_rank_rows.jsonl path. Defaults to dataset_root/valid/pair_rank_rows.jsonl.")
    ap.add_argument("--hidden_dim", type=int, default=32, help="Use 0 for logistic regression.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_auc", type=float, default=0.80)
    ap.add_argument("--min_top1_lift", type=float, default=1.25)
    args = ap.parse_args()

    default_train, default_valid = default_pair_paths(args.dataset_root)
    train_path = Path(args.train) if args.train else default_train
    valid_path = Path(args.valid) if args.valid else default_valid
    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path)
    train_x, train_y, train_w = rows_to_arrays(train_rows)
    valid_x, valid_y, _ = rows_to_arrays(valid_rows)
    train_x, valid_x, _, _ = standardize(train_x, valid_x)

    model = train_model(
        train_x,
        train_y,
        train_w,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    scores = predict_scores(model, valid_x)
    metrics = evaluate(valid_rows, valid_y, scores)
    top1_threshold = metrics["random_top1_baseline"] * float(args.min_top1_lift)
    passed = bool(metrics["auc"] > args.min_auc and metrics["top1_target_accuracy"] > top1_threshold)
    report = {
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "metrics": metrics,
        "pass_condition": {
            "auc_gt": args.min_auc,
            "top1_target_accuracy_gt": top1_threshold,
            "passed": passed,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        print("Pair-ranker sanity check failed; do not train the transformer until labels are fixed.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
