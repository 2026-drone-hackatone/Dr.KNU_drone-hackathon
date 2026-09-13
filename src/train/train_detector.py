from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from src.utils.project import REPO_ROOT, load_yaml, resolve_path


def environment() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "not-a-git-worktree"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "git_commit": commit,
        "command": shlex.join(sys.argv),
    }


def create_model(config: dict[str, Any]) -> YOLO:
    if config.get("model"):
        model = YOLO(str(resolve_path(config["model"])))
        if config.get("pretrained", True):
            model.load(config["weights"])
        return model
    if config.get("pretrained", True):
        return YOLO(config["weights"])
    return YOLO(f"{config['architecture']}.yaml")


def train(config_path: str, overrides: dict[str, Any]) -> Path:
    config = load_yaml(config_path)
    config.update({key: value for key, value in overrides.items() if value is not None})
    data_path = resolve_path(config["data"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}. Run scripts/run_preprocessing.sh first.")
    if config.get("distill_model"):
        teacher_path = resolve_path(config["distill_model"])
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {teacher_path}. Train the teacher before starting KD."
            )
        config["distill_model"] = str(teacher_path)
    project = REPO_ROOT / "runs" / config["family"]
    expected = project / config["id"]
    if expected.exists():
        raise FileExistsError(f"Run exists: {expected}. Choose --name or resume from its last.pt.")

    model = create_model(config)
    expected_strides = [4, 8, 16, 32] if config["architecture"].endswith("_p2") else [8, 16, 32]
    actual_strides = [int(value) for value in model.model.stride.tolist()]
    if actual_strides != expected_strides:
        raise RuntimeError(f"Unexpected detect strides: {actual_strides}, expected {expected_strides}")

    metadata = environment()
    metadata["detect_strides"] = actual_strides
    metadata["parameters"] = sum(parameter.numel() for parameter in model.model.parameters())
    input_shape = config.get("profile_shape")
    if input_shape is None:
        input_shape = [480, 832] if config.get("rect") and int(config["imgsz"]) == 832 else int(config["imgsz"])
    metadata["gflops"] = get_flops(model.model, imgsz=input_shape)
    metadata["input_shape"] = input_shape

    def write_run_metadata(trainer) -> None:
        save_dir = Path(trainer.save_dir)
        (save_dir / "experiment_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (save_dir / "environment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (save_dir / "command.txt").write_text(metadata["command"] + "\n", encoding="utf-8")

    model.add_callback("on_train_start", write_run_metadata)
    reserved = {
        "id",
        "family",
        "architecture",
        "weights",
        "model",
        "data",
        "profile_shape",
        "teacher_run",
        "teacher_metric",
    }
    train_args = {key: value for key, value in config.items() if key not in reserved}
    train_args.update(
        data=str(data_path), project=str(project), name=config["id"], exist_ok=False, plots=True, save=True
    )
    model.train(**train_args)
    return expected


def resume(checkpoint: str, device: str | None) -> None:
    path = resolve_path(checkpoint)
    if path.name != "last.pt" or not path.exists():
        raise FileNotFoundError(f"Exact resume requires an existing last.pt: {path}")
    kwargs = {"resume": True}
    if device is not None:
        kwargs["device"] = device
    YOLO(str(path)).train(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one reproducible YOLOv8n, YOLOv8n-P2, YOLO11n, YOLO11n-P2, "
            "or teacher experiment."
        )
    )
    parser.add_argument("--config", help="Experiment YAML")
    parser.add_argument("--resume", help="Path to last.pt for exact continuation")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--name", help="Override run id/name")
    args = parser.parse_args()
    if bool(args.config) == bool(args.resume):
        parser.error("provide exactly one of --config or --resume")
    if args.resume:
        resume(args.resume, args.device)
        return
    overrides = {
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "id": args.name,
    }
    output = train(args.config, overrides)
    print(f"run: {output}")


if __name__ == "__main__":
    main()
