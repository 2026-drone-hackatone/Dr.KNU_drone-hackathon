from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, init_seeds

from src.utils.project import REPO_ROOT, load_yaml, resolve_path


MAP50_METRIC = "metrics/mAP50(B)"


def add_map50_checkpoint_callback(model: YOLO) -> None:
    """Keep best.pt aligned with the highest validation mAP@0.5."""
    state: dict[str, Any] = {"initialized": False, "value": float("-inf"), "epoch": None}

    def save_map50_best(trainer) -> None:
        save_dir = Path(trainer.save_dir)
        weights_dir = Path(trainer.wdir)
        record_path = save_dir / "best_map50.json"
        map50_checkpoint = weights_dir / "best_map50.pt"

        if not state["initialized"]:
            if record_path.exists():
                previous = json.loads(record_path.read_text(encoding="utf-8"))
                state["value"] = float(previous["value"])
                state["epoch"] = int(previous["epoch"])
            state["initialized"] = True

        value = float(trainer.metrics[MAP50_METRIC])
        if value > state["value"]:
            shutil.copy2(trainer.last, map50_checkpoint)
            state["value"] = value
            state["epoch"] = int(trainer.epoch) + 1

        if not map50_checkpoint.exists():
            raise FileNotFoundError(f"mAP50 checkpoint was not created: {map50_checkpoint}")
        shutil.copy2(map50_checkpoint, trainer.best)
        record_path.write_text(
            json.dumps(
                {
                    "selection_metric": MAP50_METRIC,
                    "value": state["value"],
                    "epoch": state["epoch"],
                    "checkpoint": str(map50_checkpoint.relative_to(save_dir)),
                    "best": str(Path(trainer.best).relative_to(save_dir)),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    model.add_callback("on_model_save", save_map50_best)


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


def load_with_layer_offset(model: YOLO, weights: str, offset: int) -> dict[str, int]:
    """Load a checkpoint whose layer indices are shifted by `offset`.

    Used by the *_up2 architectures, where layer 0 is a parameter-free
    nn.Upsample and every stock layer moves from model.{i} to model.{i+offset}.
    Only tensors whose remapped name AND shape match are copied."""
    checkpoint = torch.load(resolve_path(weights), map_location="cpu", weights_only=False)
    source = checkpoint_state_dict(checkpoint)
    target = model.model.state_dict()
    remapped = {}
    for key, value in source.items():
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit():
            parts[1] = str(int(parts[1]) + offset)
        remapped[".".join(parts)] = value
    compatible = {k: v for k, v in remapped.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    model.model.load_state_dict(compatible, strict=False)
    # Model.train() only reuses the in-memory model when `self.ckpt` is truthy;
    # otherwise the trainer rebuilds a random model from the yaml and the
    # offset-loaded weights are silently lost. Keep `pretrained` a bool so the
    # trainer copies from this module instead of re-reading the file by name.
    model.ckpt = {"layer_offset": offset, "source_weights": str(resolve_path(weights))}
    return {"transferred_tensors": len(compatible), "source_tensors": len(source), "target_tensors": len(target),
            "layer_offset": offset}


def load_with_layer_map(model: YOLO, weights: str, layer_map: dict) -> dict[str, Any]:
    """Copy explicitly mapped backbone/neck layers; leave unmapped layers fresh.

    Unlike name-only loading, this preserves scale semantics when P2 insertion
    shifts the bottom-up P4/P5 neck. Detect is intentionally not mapped.
    """
    source_model = YOLO(str(resolve_path(weights))).model
    mapping = {int(src): int(dst) for src, dst in layer_map.items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("weights_layer_map must have unique destinations")
    source = source_model.state_dict()
    target = model.model.state_dict()
    compatible = {}
    for src, dst in mapping.items():
        source_layer, target_layer = source_model.model[src], model.model.model[dst]
        if type(source_layer) is not type(target_layer) or source_layer.__class__.__name__ == "Detect":
            raise ValueError(f"Invalid backbone/neck mapping: {src} -> {dst}")
        prefix = f"model.{src}."
        for key, value in source.items():
            if key.startswith(prefix):
                dest = f"model.{dst}." + key[len(prefix):]
                if dest not in target or target[dest].shape != value.shape:
                    raise ValueError(f"Incompatible mapped tensor: {key} -> {dest}")
                compatible[dest] = value
    if not compatible:
        raise ValueError("weights_layer_map transferred no tensors")
    model.model.load_state_dict(compatible, strict=False)
    # Preserve this module through YOLO.train()'s trainer reconstruction.
    model.ckpt = {"layer_map": mapping, "source_weights": str(resolve_path(weights))}
    return {"transferred_tensors": len(compatible), "source_tensors": len(source),
            "target_tensors": len(target), "layer_map": mapping,
            "unmapped_target_layers": sorted(set(range(len(model.model.model))) - set(mapping.values()))}


def create_model(config: dict[str, Any]) -> YOLO:
    if config.get("model"):
        model = YOLO(str(resolve_path(config["model"])))
        if model.model.yaml.get("spd_layers"):
            from src.train.spd_model import SPDDetectionModel

            model.model = SPDDetectionModel(model.model.yaml, verbose=False)
        if config.get("pretrained", True):
            offset = int(config.get("weights_layer_offset", 0) or 0)
            if config.get("weights_layer_map") is not None:
                if offset:
                    raise ValueError("Use either weights_layer_map or weights_layer_offset")
                model._p2_transfer_report = load_with_layer_map(model, config["weights"], config["weights_layer_map"])
            elif offset:
                model._p2_transfer_report = load_with_layer_offset(model, config["weights"], offset)
            else:
                model._p2_transfer_report = pretrained_transfer_report(model, config["weights"])
                model.load(config["weights"])
        return model
    if config.get("pretrained", True):
        return YOLO(config["weights"])
    return YOLO(f"{config['architecture']}.yaml")


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Ultralytics run checkpoints store the EMA under "ema" and set "model" to None;
    official release weights store the model under "model"."""
    if not isinstance(checkpoint, dict):
        return {}
    for key in ("ema", "model"):
        module = checkpoint.get(key)
        if module is not None:
            return module.float().state_dict()
    return {}


def pretrained_transfer_report(model: YOLO, weights: str) -> dict[str, int]:
    """Count checkpoint tensors that can transfer by name and shape."""
    checkpoint = torch.load(resolve_path(weights), map_location="cpu", weights_only=False)
    source = checkpoint_state_dict(checkpoint)
    target = model.model.state_dict()
    compatible = sum(key in target and tuple(value.shape) == tuple(target[key].shape) for key, value in source.items())
    return {"transferred_tensors": compatible, "source_tensors": len(source), "target_tensors": len(target)}


def install_nwd_criterion(trainer, nwd_weight: float, nwd_constant: float) -> None:
    """Install the experiment loss after Ultralytics has built the train model."""
    from src.losses.nwd_loss import HybridNWDDetectionLoss

    model = getattr(trainer.model, "module", trainer.model)
    model.criterion = HybridNWDDetectionLoss(
        model, nwd_weight=nwd_weight, nwd_constant=nwd_constant
    )


def install_frozen_batchnorm(trainer) -> None:
    """Keep every BatchNorm layer in eval mode so running stats stay at the
    pretrained values. Use when the batch is too small (<=2) for stable BN
    statistics, e.g. fine-tuning at 1920 input. Affine weights still train."""
    import torch.nn as nn

    model = getattr(trainer.model, "module", trainer.model)
    frozen = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            frozen += 1
    if frozen == 0:
        raise RuntimeError("freeze_bn requested but no BatchNorm layers were found")


def install_epoch_stop(model: YOLO, stop_after_epoch: int) -> None:
    """End training gracefully after `stop_after_epoch` epochs (1-based, as in
    results.csv) while keeping the LR schedule of the configured `epochs`.

    Used by the final train+val (Z) runs: their validation split is in-sample,
    so early stopping / best-epoch selection are meaningless and the horizon is
    fixed from the provider-split experiment instead. Setting trainer.stop in
    on_fit_epoch_end takes the same exit path as EarlyStopping, so last.pt /
    best.pt are stripped and final_eval runs as usual."""

    def stop_at_epoch(trainer) -> None:
        if trainer.epoch + 1 >= stop_after_epoch:
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", stop_at_epoch)


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

    # Seed before constructing new heads/SPD, not only inside the trainer.
    init_seeds(int(config.get("seed", 0)), deterministic=bool(config.get("deterministic", False)))
    model = create_model(config)
    architecture = config["architecture"]
    if architecture.endswith("_p2_p4_up2"):
        expected_strides = [2, 4, 8]
    elif architecture.endswith("_p2_p5_up2"):
        expected_strides = [2, 4, 8, 16]
    elif architecture.endswith("_p2_p4") or architecture.endswith("_up2"):
        expected_strides = [4, 8, 16]  # _up2: stock [8,16,32] halved by the internal x2 upsample
    elif architecture.endswith("_up3"):
        expected_strides = [8 / 3, 16 / 3, 32 / 3]  # _up3: stock strides divided by 3 (non-integer)
    elif architecture.endswith("_p2"):
        expected_strides = [4, 8, 16, 32]
    else:
        expected_strides = [8, 16, 32]
    actual_strides = [round(float(value), 4) for value in model.model.stride.tolist()]
    if actual_strides != [round(float(value), 4) for value in expected_strides]:
        raise RuntimeError(f"Unexpected detect strides: {actual_strides}, expected {expected_strides}")

    metadata = environment()
    metadata["detect_strides"] = actual_strides
    metadata["parameters"] = sum(parameter.numel() for parameter in model.model.parameters())
    input_shape = config.get("profile_shape")
    if input_shape is None:
        input_shape = [480, 832] if config.get("rect") and int(config["imgsz"]) == 832 else int(config["imgsz"])
    metadata["gflops"] = get_flops(model.model, imgsz=input_shape)
    metadata["input_shape"] = input_shape
    if hasattr(model, "_p2_transfer_report"):
        metadata["pretrained_transfer"] = model._p2_transfer_report
    if config.get("use_nwd"):
        metadata["loss"] = {
            "name": "hybrid_default_plus_nwd",
            "nwd_weight": float(config.get("nwd_weight", 0.2)),
            "nwd_constant": float(config.get("nwd_constant", 12.8)),
        }

    def write_run_metadata(trainer) -> None:
        save_dir = Path(trainer.save_dir)
        trained_model = getattr(trainer.model, "module", trainer.model)
        deployment_model = getattr(trained_model, "student_model", trained_model)
        metadata["detect_strides"] = [round(float(value), 4) for value in deployment_model.stride.tolist()]
        metadata["parameters"] = sum(parameter.numel() for parameter in deployment_model.parameters())
        metadata["gflops"] = get_flops(deployment_model, imgsz=input_shape)
        if deployment_model is not trained_model:
            metadata["distillation_training_parameters"] = sum(
                parameter.numel() for parameter in trained_model.parameters()
            )
        raw_names = getattr(deployment_model, "names", {})
        name_items = raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names)
        metadata["class_names"] = {int(key): str(value) for key, value in name_items}
        (save_dir / "experiment_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (save_dir / "environment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (save_dir / "command.txt").write_text(metadata["command"] + "\n", encoding="utf-8")

    model.add_callback("on_train_start", write_run_metadata)
    if config.get("use_nwd"):
        model.add_callback(
            "on_pretrain_routine_end",
            lambda trainer: install_nwd_criterion(
                trainer,
                float(config.get("nwd_weight", 0.2)),
                float(config.get("nwd_constant", 12.8)),
            ),
        )
    if config.get("freeze_bn"):
        metadata["freeze_bn"] = True
        # the trainer calls model.train() AFTER on_train_epoch_start, so the
        # only hook that reliably runs after it is on_train_batch_start
        model.add_callback("on_train_batch_start", install_frozen_batchnorm)
    stop_after_epoch = config.get("stop_after_epoch")
    if stop_after_epoch is not None:
        stop_after_epoch = int(stop_after_epoch)
        if not 1 <= stop_after_epoch <= int(config["epochs"]):
            raise ValueError(f"stop_after_epoch={stop_after_epoch} must be within 1..epochs={config['epochs']}")
        metadata["stop_after_epoch"] = stop_after_epoch
        install_epoch_stop(model, stop_after_epoch)
    add_map50_checkpoint_callback(model)
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
        "kd_alignment",
        "use_nwd",
        "nwd_weight",
        "nwd_constant",
        "loss_config",
        "freeze_bn",
        "weights_layer_offset",
        "weights_layer_map",
        "stop_after_epoch",
    }
    train_args = {key: value for key, value in config.items() if key not in reserved}
    train_args.update(
        data=str(data_path), project=str(project), name=config["id"], exist_ok=False, plots=True, save=True
    )
    if model.model.yaml.get("spd_layers"):
        from src.train.spd_model import SPDDetectionTrainer

        train_args["trainer"] = SPDDetectionTrainer
    model.train(**train_args)
    return expected


def resume(checkpoint: str, device: str | None) -> None:
    path = resolve_path(checkpoint)
    if path.name != "last.pt" or not path.exists():
        raise FileNotFoundError(f"Exact resume requires an existing last.pt: {path}")
    kwargs = {"resume": True}
    if device is not None:
        kwargs["device"] = device
    model = YOLO(str(path))
    if model.model.yaml.get("spd_layers"):
        from src.train.spd_model import SPDDetectionTrainer

        kwargs["trainer"] = SPDDetectionTrainer
    add_map50_checkpoint_callback(model)
    model.train(**kwargs)


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
