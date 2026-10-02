from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml
from PIL import Image

from src.data.common import (
    Box,
    IMAGE_EXTENSIONS,
    SPLIT_NAMES,
    annotation_path,
    assign_group_splits,
    clip_box,
    parse_counted_annotations,
    parse_mot_annotations,
    probe_video,
    scale_bin_from_boxes,
    video_files,
    yolo_line,
)
from src.utils.project import REPO_ROOT, dump_json, load_yaml, resolve_path, write_jsonl

DEFAULT_CONFIGS = [
    "configs/data/drone_detection.yaml",
    "configs/data/drone_bird.yaml",
    "configs/data/purdue_uav.yaml",
]


def reset_output(output: Path, force: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"Output exists: {output}. Use --force to rebuild it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def write_data_yaml(output: Path, path: Path | None = None) -> None:
    content = {
        "path": str((path or output).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "drone"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def parse_yolo_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    boxes = []
    for line_no, line in enumerate(label_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_no}: expected five fields")
        class_id = int(fields[0])
        xc, yc, bw, bh = map(float, fields[1:])
        if class_id != 0:
            raise ValueError(f"{label_path}:{line_no}: unexpected class {class_id}")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"{label_path}:{line_no}: invalid normalized box")
        pixel_w, pixel_h = bw * width, bh * height
        boxes.append(Box(0, None, xc * width - pixel_w / 2, yc * height - pixel_h / 2, pixel_w, pixel_h))
    return boxes


def prepare_existing_yolo(config: dict[str, Any], force: bool) -> dict:
    root = resolve_path(config["root"])
    output = resolve_path(config["output_root"])
    reset_output(output, force)
    rows = []
    summary: dict[str, Any] = {"name": config["name"], "splits": {}}
    for split in config["existing_splits"]:
        images = sorted(
            path for path in (root / "images" / split).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        count_boxes = count_empty = 0
        for image_path in images:
            label_path = root / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label: {label_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            boxes = parse_yolo_boxes(label_path, width, height)
            count_boxes += len(boxes)
            count_empty += not boxes
            rows.append(
                {
                    "sample_id": f"{config['name']}__{split}__{image_path.stem}",
                    "dataset": config["name"],
                    "sequence_id": None,
                    "frame_id": image_path.stem,
                    "split": split,
                    "image_path": str(image_path.relative_to(REPO_ROOT)),
                    "label_path": str(label_path.relative_to(REPO_ROOT)),
                    "original_path": str(image_path.relative_to(REPO_ROOT)),
                    "width": width,
                    "height": height,
                    "num_boxes": len(boxes),
                    "scale_bin": scale_bin_from_boxes(boxes),
                    "track_ids": [],
                }
            )
        summary["splits"][split] = {
            "images": len(images), "boxes": count_boxes, "negative_frames": count_empty
        }
    write_jsonl(output / "metadata.jsonl", rows)
    # Point to the existing immutable YOLO tree without duplicating 54k images.
    write_data_yaml(output, root)
    dump_json(output / "conversion_summary.json", summary)
    return summary


def keep_frame(frame_zero: int, boxes: list[Box], sampling: dict[str, Any]) -> tuple[bool, str]:
    if not boxes:
        stride, reason = int(sampling["negative_stride"]), "negative_stride"
    elif min(max(box.w, box.h) for box in boxes) <= float(sampling["tiny_max_side"]):
        stride, reason = int(sampling["tiny_stride"]), "tiny_stride"
    else:
        stride, reason = int(sampling["positive_stride"]), "positive_stride"
    return frame_zero % stride == 0, reason


def prepare_video_dataset(config: dict[str, Any], force: bool) -> dict:
    output = resolve_path(config["output_root"])
    reset_output(output, force)
    for split in SPLIT_NAMES:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    videos = video_files(resolve_path(config["video_root"]))
    annotation_root = resolve_path(config["annotation_root"])
    suffix = config.get("annotation_suffix", "")
    missing = [
        stem for stem in videos
        if not annotation_path(annotation_root, stem, config["kind"], suffix).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing annotations for: {missing}")
    infos = {stem: probe_video(path) for stem, path in videos.items()}
    assignment, grouped = assign_group_splits(
        {stem: info.frames for stem, info in infos.items()},
        list(config["split_ratio"]),
        int(config["seed"]),
        group_related=config["kind"] == "counted_video",
    )
    split_payload = {
        "seed": config["seed"],
        "ratio": config["split_ratio"],
        "unit": "source_sequence_or_related_recording_group",
        **grouped,
    }
    dump_json(config["split_file"], split_payload)

    metadata = []
    stats = Counter()
    per_split: dict[str, Counter] = {name: Counter() for name in SPLIT_NAMES}
    for stem, video_path in sorted(videos.items()):
        info = infos[stem]
        split = assignment[stem]
        ann_path = annotation_path(annotation_root, stem, config["kind"], suffix)
        annotations = (
            parse_counted_annotations(ann_path)
            if config["kind"] == "counted_video"
            else parse_mot_annotations(ann_path)
        )
        capture = cv2.VideoCapture(str(video_path))
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_zero = decoded
            annotation_frame = frame_zero + int(config["frame_index_base"])
            decoded += 1
            raw_boxes = annotations.get(annotation_frame, [])
            accepted: list[Box] = []
            clipped_count = 0
            for box in raw_boxes:
                if box.class_name != "drone":
                    stats["removed_non_drone_boxes"] += 1
                    continue
                if box.confidence < float(config.get("min_annotation_confidence", 0.0)):
                    stats["removed_low_confidence_boxes"] += 1
                    continue
                clipped, was_clipped = clip_box(box, info.width, info.height)
                if clipped is None:
                    stats["removed_outside_boxes"] += 1
                    continue
                clipped_count += int(was_clipped)
                accepted.append(clipped)
            stats["clipped_boxes"] += clipped_count
            keep, sampling_reason = keep_frame(frame_zero, accepted, config["sampling"])
            if not keep:
                continue
            sample_id = f"{config['name']}__{stem}__{frame_zero:06d}"
            image_path = output / "images" / split / f"{sample_id}.jpg"
            label_path = output / "labels" / split / f"{sample_id}.txt"
            if not cv2.imwrite(
                str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(config.get("jpeg_quality", 95))]
            ):
                raise OSError(f"Failed to write {image_path}")
            label_path.write_text("\n".join(yolo_line(box, info.width, info.height) for box in accepted) + ("\n" if accepted else ""), encoding="utf-8")
            per_split[split]["images"] += 1
            per_split[split]["boxes"] += len(accepted)
            per_split[split]["negative_frames"] += not accepted
            metadata.append(
                {
                    "sample_id": sample_id,
                    "dataset": config["name"],
                    "sequence_id": stem,
                    "frame_id": annotation_frame,
                    "split": split,
                    "image_path": str(image_path.relative_to(REPO_ROOT)),
                    "label_path": str(label_path.relative_to(REPO_ROOT)),
                    "original_path": str(video_path.relative_to(REPO_ROOT)),
                    "width": info.width,
                    "height": info.height,
                    "num_boxes": len(accepted),
                    "scale_bin": scale_bin_from_boxes(accepted),
                    "track_ids": sorted({box.track_id for box in accepted if box.track_id is not None}),
                    "interpolated_boxes": sum(box.confidence < 1.0 for box in accepted),
                    "sampling_reason": sampling_reason,
                    "clipped_boxes": clipped_count,
                }
            )
        capture.release()
        if decoded != info.frames:
            stats["video_metadata_decode_mismatch"] += 1
        invalid_annotation_boxes = sum(
            len(boxes)
            for frame_id, boxes in annotations.items()
            if frame_id < int(config["frame_index_base"])
            or frame_id >= info.frames + int(config["frame_index_base"])
        )
        stats["annotation_boxes_past_video"] += invalid_annotation_boxes
    write_jsonl(output / "metadata.jsonl", metadata)
    write_data_yaml(output)
    summary = {
        "name": config["name"],
        "frame_index_base": config["frame_index_base"],
        "sampling": config["sampling"],
        "splits": {name: dict(values) for name, values in per_split.items()},
        "conversion_adjustments": dict(stats),
        "split_file": str(resolve_path(config["split_file"]).relative_to(REPO_ROOT)),
    }
    dump_json(output / "conversion_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert/register Phase-1 datasets as YOLO detection datasets.")
    parser.add_argument("--config", action="append", help="Dataset YAML; repeat for multiple datasets")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the configured output directories")
    args = parser.parse_args()
    for path in args.config or DEFAULT_CONFIGS:
        config = load_yaml(path)
        print(f"Preparing {config['name']}...")
        summary = (
            prepare_existing_yolo(config, args.force)
            if config["kind"] == "yolo_images"
            else prepare_video_dataset(config, args.force)
        )
        print(json.dumps(summary["splits"], ensure_ascii=False))


if __name__ == "__main__":
    main()
