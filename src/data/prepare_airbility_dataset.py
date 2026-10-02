from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.data.prepare_datasets import reset_output
from src.utils.project import REPO_ROOT, dump_json, load_yaml, resolve_path, write_jsonl


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def convert_label_text(text: str, class_count: int) -> tuple[str, list[int]]:
    """Convert Airbility six-column detection rows to standard five-column YOLO rows."""
    output: list[str] = []
    track_ids: list[int] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 6:
            raise ValueError(f"line {line_no}: expected 6 fields, got {len(fields)}")
        try:
            class_id = int(fields[0])
            xc, yc, width, height = map(float, fields[1:5])
            track_id = int(fields[5])
        except ValueError as error:
            raise ValueError(f"line {line_no}: invalid numeric value") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"line {line_no}: class_id={class_id} outside 0..{class_count - 1}")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"line {line_no}: invalid normalized box")
        if (
            xc - width / 2 < -1e-6
            or yc - height / 2 < -1e-6
            or xc + width / 2 > 1 + 1e-6
            or yc + height / 2 > 1 + 1e-6
        ):
            raise ValueError(f"line {line_no}: box edges outside image")
        if track_id < 1:
            raise ValueError(f"line {line_no}: track_id={track_id} must be one-based")
        output.append(" ".join(fields[:5]))
        track_ids.append(track_id)
    return ("\n".join(output) + ("\n" if output else ""), track_ids)


# Where the provider's val sequences land in the processed dataset.
#   val           -> provider split preserved (train = provider train, val = provider val)
#   train_and_val -> final "train on everything" build: provider val sequences are
#                    linked into train AND kept in val, so the training-time val
#                    metric is in-sample and must not be used for model selection.
PROVIDER_VAL_DESTINATIONS: dict[str, tuple[str, ...]] = {
    "val": ("val",),
    "train_and_val": ("train", "val"),
}


def destination_splits(
    source_split: str,
    provider_val_destination: str = "val",
) -> tuple[str, ...]:
    if source_split == "val":
        if provider_val_destination not in PROVIDER_VAL_DESTINATIONS:
            raise ValueError(
                f"provider_val_destination must be one of {sorted(PROVIDER_VAL_DESTINATIONS)}, "
                f"got {provider_val_destination!r}"
            )
        return PROVIDER_VAL_DESTINATIONS[provider_val_destination]
    if source_split != "train":
        raise ValueError(f"Unsupported source split: {source_split}")
    return ("train",)


def link_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    if mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
        return
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    raise ValueError(f"Unsupported link_mode: {mode}")


def write_dataset_yaml(output: Path, names: dict[int, str]) -> None:
    # Omitting `path` makes Ultralytics resolve split paths from this YAML's directory.
    content = {
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (output / "data.yaml").write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def size_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def percentile(fraction: float) -> float:
        return round(ordered[round((len(ordered) - 1) * fraction)], 3)

    return {
        "min": percentile(0),
        "q25": percentile(0.25),
        "median": percentile(0.5),
        "q75": percentile(0.75),
        "q95": percentile(0.95),
        "max": percentile(1),
    }


def build(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    source = resolve_path(config["source_root"])
    output = resolve_path(config["output_root"])
    names = {int(key): str(value) for key, value in config["names"].items()}
    source_frame_policy = config.get("source_frame_policy", "available_images_only")
    if source_frame_policy != "available_images_only":
        raise ValueError(f"Unsupported source_frame_policy: {source_frame_policy}")
    if sorted(names) != list(range(len(names))):
        raise ValueError(f"Class IDs must be contiguous from zero: {names}")
    excluded = set(config.get("excluded_sequences", []))
    provider_val_destination = config.get("provider_val_destination", "val")
    if provider_val_destination not in PROVIDER_VAL_DESTINATIONS:
        raise ValueError(
            f"provider_val_destination must be one of {sorted(PROVIDER_VAL_DESTINATIONS)}, "
            f"got {provider_val_destination!r}"
        )

    reset_output(output, force)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    totals: dict[str, Counter] = {split: Counter() for split in SPLITS}
    box_widths: dict[str, list[float]] = {split: [] for split in SPLITS}
    box_heights: dict[str, list[float]] = {split: [] for split in SPLITS}
    metadata: list[dict[str, Any]] = []
    source_sequence_counts = Counter()
    for source_split in ("train", "val"):
        image_root = source / "images" / source_split
        label_root = source / "labels" / source_split
        sequences = sorted(path for path in image_root.iterdir() if path.is_dir())
        for sequence_dir in sequences:
            sequence = sequence_dir.name
            if sequence in excluded:
                continue
            target_splits = destination_splits(source_split, provider_val_destination)
            for target_split in target_splits:
                source_sequence_counts[(source_split, target_split)] += 1
            images = sorted(path for path in sequence_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
            for image_path in images:
                relative = image_path.relative_to(sequence_dir).with_suffix("")
                label_path = label_root / sequence / relative.with_suffix(".txt")
                if not label_path.exists():
                    raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
                converted, track_ids = convert_label_text(label_path.read_text(encoding="utf-8"), len(names))
                rows = [line for line in converted.splitlines() if line.strip()]
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                if (width, height) != (1280, 720):
                    raise ValueError(f"Unexpected image size {width}x{height}: {image_path}")

                for target_split in target_splits:
                    image_out = output / "images" / target_split / sequence / image_path.name
                    label_out = output / "labels" / target_split / sequence / f"{image_path.stem}.txt"
                    link_image(image_path, image_out, config["link_mode"])
                    label_out.parent.mkdir(parents=True, exist_ok=True)
                    label_out.write_text(converted, encoding="utf-8")

                    class_ids = [int(line.split()[0]) for line in rows]
                    totals[target_split]["images"] += 1
                    totals[target_split]["boxes"] += len(rows)
                    totals[target_split]["negative_frames"] += int(not rows)
                    for line, class_id in zip(rows, class_ids):
                        totals[target_split][f"class_{class_id}_boxes"] += 1
                        _, _, _, normalized_width, normalized_height = line.split()
                        pixel_width = float(normalized_width) * width
                        pixel_height = float(normalized_height) * height
                        box_widths[target_split].append(pixel_width)
                        box_heights[target_split].append(pixel_height)
                        max_side = max(pixel_width, pixel_height)
                        size_bin = "tiny" if max_side <= 16 else "small" if max_side <= 32 else "large"
                        totals[target_split][f"{size_bin}_boxes"] += 1
                    metadata.append(
                        {
                            "sample_id": f"{sequence}__{image_path.stem}",
                            "dataset": "airbility_uav_detection",
                            "source_split": source_split,
                            "split": target_split,
                            "sequence_id": sequence,
                            "frame_id": int(image_path.stem),
                            "original_path": display_path(image_path),
                            "original_label_path": display_path(label_path),
                            "image_path": display_path(image_out),
                            "label_path": display_path(label_out),
                            "width": width,
                            "height": height,
                            "num_boxes": len(rows),
                            "class_ids": class_ids,
                            "track_ids": track_ids,
                        }
                    )

    for source_split in ("train", "val"):
        label_root = source / "labels" / source_split
        image_root = source / "images" / source_split
        image_keys = {
            path.relative_to(image_root).with_suffix("").as_posix()
            for path in image_root.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES and path.parts[-2] not in excluded
        }
        label_keys = {path.relative_to(label_root).with_suffix("").as_posix() for path in label_root.rglob("*.txt")}
        label_keys = {key for key in label_keys if key.split("/", 1)[0] not in excluded}
        if image_keys != label_keys:
            raise ValueError(
                f"Source image/label mismatch in {source_split}: "
                f"missing_labels={sorted(image_keys - label_keys)[:5]}, missing_images={sorted(label_keys - image_keys)[:5]}"
            )

    write_dataset_yaml(output, names)
    write_jsonl(output / "metadata.jsonl", metadata)
    summary = {
        "name": config["name"],
        "source_root": display_path(source),
        "output_root": display_path(output),
        "seed": int(config["seed"]),
        "source_frame_policy": source_frame_policy,
        "names": names,
        "provider_val_destination": provider_val_destination,
        "excluded_sequences": sorted(excluded),
        "source_sequence_counts": {
            f"{source_split}_to_{target_split}": count
            for (source_split, target_split), count in sorted(source_sequence_counts.items())
        },
        "dataset": {
            split: {
                **dict(totals[split]),
                "negative_ratio": round(
                    totals[split]["negative_frames"] / max(1, totals[split]["images"]), 6
                ),
                "bbox_width_px": size_summary(box_widths[split]),
                "bbox_height_px": size_summary(box_heights[split]),
            }
            for split in SPLITS
        },
    }
    dump_json(output / "build_summary.json", summary)
    dump_json(config["report"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Airbility 6-column labels for 3-class YOLO detection.")
    parser.add_argument("--config", default="configs/data/airbility_uav_detection.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args.config, args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
