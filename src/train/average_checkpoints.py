"""Average the EMA weights of several Ultralytics epoch checkpoints.

Produces a checkpoint that loads with `YOLO(path)` exactly like best.pt
(same architecture, same stock modules), so it satisfies the eval.py
constraints. Only floating-point tensors are averaged; integer buffers
(BatchNorm num_batches_tracked) are copied from the last checkpoint.

Example:
    python -m src.train.average_checkpoints \
        --run runs/detection/U2_yolo11s_up2_from_T0_company_1280x720 \
        --epochs 9-20 --output runs/detection/U2_.../weights_avg/avg_ep9-20.pt
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from src.utils.project import resolve_path


def parse_epochs(spec: str) -> list[int]:
    epochs: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            epochs.extend(range(int(start), int(end) + 1))
        elif part:
            epochs.append(int(part))
    if not epochs:
        raise ValueError(f"No epochs parsed from {spec!r}")
    return sorted(set(epochs))


def epoch_checkpoint(run_dir: Path, epoch: int) -> Path:
    # Ultralytics names save_period files epoch{N}.pt with N = 0-based index,
    # while results.csv / best_map50.json use 1-based epochs.
    path = run_dir / "weights" / f"epoch{epoch - 1}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found: {path}")
    return path


def unwrap_student(module):
    """Return the bare student for KD epoch checkpoints (DistillationModel wrapper).

    Ultralytics strips the wrapper only from the final best.pt/last.pt; the
    per-epoch files keep student + projector + (None) teacher, and the averaged
    checkpoint must be a stock DetectionModel to satisfy the eval.py constraint.
    """
    if module is not None and hasattr(module, "student_model"):
        if hasattr(module, "_remove_feature_hooks"):
            module._remove_feature_hooks()
        module = module.student_model
    return module


def average(run: str, epochs: list[int], output: str) -> dict:
    run_dir = resolve_path(run)
    paths = [epoch_checkpoint(run_dir, epoch) for epoch in epochs]
    base = torch.load(paths[-1], map_location="cpu", weights_only=False)
    base_module = unwrap_student(base.get("ema") or base["model"])
    if base_module is None:
        raise ValueError(f"{paths[-1]} has neither ema nor model")
    averaged = copy.deepcopy(base_module).float()
    target = averaged.state_dict()
    sums: dict[str, torch.Tensor] = {}
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        module = unwrap_student(checkpoint.get("ema") or checkpoint["model"])
        state = module.float().state_dict()
        if set(state) != set(target):
            raise ValueError(f"{path} has a different parameter set than {paths[-1]}")
        for key, value in state.items():
            if not torch.is_floating_point(value):
                continue
            if tuple(value.shape) != tuple(target[key].shape):
                raise ValueError(f"Shape mismatch for {key} in {path}")
            sums[key] = sums.get(key, torch.zeros_like(value, dtype=torch.float64)) + value.double()
    for key, total in sums.items():
        target[key] = (total / len(paths)).to(target[key].dtype)
    averaged.load_state_dict(target, strict=True)
    averaged = averaged.half()

    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": averaged,
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": base.get("train_args"),
        "train_metrics": None,
        "train_results": None,
        "date": base.get("date"),
        "version": base.get("version"),
        "license": base.get("license"),
        "docs": base.get("docs"),
        "averaged_from": [str(path.relative_to(run_dir)) for path in paths],
    }
    output_path = resolve_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    torch.save(checkpoint, output_path)
    report = {
        "run": str(run_dir),
        "epochs": epochs,
        "checkpoints": checkpoint["averaged_from"],
        "averaged_tensors": len(sums),
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Average EMA weights of epoch checkpoints into one loadable .pt")
    parser.add_argument("--run", required=True, help="Run directory containing weights/epoch{N}.pt")
    parser.add_argument("--epochs", required=True, help="1-based epochs as in results.csv, e.g. '9-20' or '1,2,3'")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(average(args.run, parse_epochs(args.epochs), args.output), indent=2))


if __name__ == "__main__":
    main()
