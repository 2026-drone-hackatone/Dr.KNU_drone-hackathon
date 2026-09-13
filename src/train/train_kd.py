from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from src.train.train_detector import train
from src.utils.project import load_yaml, resolve_path


def select_teacher_checkpoint(run_dir: str, metric: str) -> tuple[Path, int, float]:
    """Select the saved teacher epoch with the highest validation metric."""
    run = resolve_path(run_dir)
    results_path = run / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Teacher results not found: {results_path}. Train the teacher model first."
        )

    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or metric not in rows[0]:
        columns = list(rows[0]) if rows else []
        raise ValueError(f"Metric {metric!r} is unavailable in {results_path}; columns={columns}")

    valid_rows = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid_rows.append((value, row))
    if not valid_rows:
        raise ValueError(f"No finite values for {metric!r} in {results_path}")

    value, row = max(valid_rows, key=lambda item: item[0])
    epoch = int(float(row["epoch"]))
    checkpoint = run / "weights" / f"epoch{epoch}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Selected teacher checkpoint is missing: {checkpoint}. "
            "Teacher configs must use save_period: 1."
        )
    return checkpoint, epoch, value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a YOLO11 nano student with feature-based knowledge distillation."
    )
    parser.add_argument("--config", required=True, help="KD experiment YAML")
    parser.add_argument("--teacher", help="Explicit teacher checkpoint; overrides automatic metric selection")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--name", help="Override run id/name")
    parser.add_argument("--dis", type=float, help="Override distillation loss weight")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.teacher:
        teacher = resolve_path(args.teacher)
        if not teacher.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher}")
        print(f"teacher: {teacher} (explicit)")
    else:
        if not config.get("teacher_run"):
            raise ValueError("KD config requires teacher_run or an explicit --teacher checkpoint")
        metric = config.get("teacher_metric", "metrics/mAP50(B)")
        teacher, epoch, value = select_teacher_checkpoint(config["teacher_run"], metric)
        print(f"teacher: {teacher} ({metric}={value:.6f}, epoch={epoch})")

    overrides = {
        "distill_model": str(teacher),
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "id": args.name,
        "dis": args.dis,
    }
    output = train(args.config, overrides)
    print(f"run: {output}")


if __name__ == "__main__":
    main()
