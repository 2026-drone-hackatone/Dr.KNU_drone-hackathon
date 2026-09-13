from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".mpg", ".mpeg"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frames: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class Box:
    frame: int
    track_id: int | None
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    class_name: str = "drone"


def video_files(root: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    info = VideoInfo(
        path=path,
        frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    if min(info.frames, info.width, info.height) <= 0:
        raise ValueError(f"Invalid video metadata: {path}: {info}")
    return info


def parse_counted_annotations(path: Path) -> dict[int, list[Box]]:
    """Parse `frame count (x y w h class)*`; frame ids remain unchanged."""
    output: dict[int, list[Box]] = defaultdict(list)
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) < 2:
            raise ValueError(f"{path}:{line_no}: expected frame and object count")
        frame, count = int(fields[0]), int(fields[1])
        if len(fields) != 2 + count * 5:
            raise ValueError(
                f"{path}:{line_no}: count={count}, fields={len(fields)} (expected {2 + count * 5})"
            )
        for index in range(count):
            start = 2 + index * 5
            x, y, width, height = map(float, fields[start : start + 4])
            output[frame].append(
                Box(frame, None, x, y, width, height, class_name=fields[start + 4].lower())
            )
        output.setdefault(frame, [])
    return dict(output)


def parse_mot_annotations(path: Path, suffix: str = "_refined") -> dict[int, list[Box]]:
    """Parse MOT rows: frame,id,left,top,width,height,confidence,x,y,z."""
    output: dict[int, list[Box]] = defaultdict(list)
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split(",")
        if len(fields) != 10:
            raise ValueError(f"{path}:{line_no}: expected 10 MOT fields, got {len(fields)}")
        frame, track_id = int(fields[0]), int(fields[1])
        x, y, width, height, confidence = map(float, fields[2:7])
        output[frame].append(Box(frame, track_id, x, y, width, height, confidence))
    return dict(output)


def annotation_path(annotation_root: Path, stem: str, kind: str, suffix: str = "") -> Path:
    name = f"{stem}{suffix}.txt" if kind == "mot_video" else f"{stem}.txt"
    return annotation_root / name


def clip_box(box: Box, width: int, height: int) -> tuple[Box | None, bool]:
    x1 = max(0.0, min(float(width), box.x))
    y1 = max(0.0, min(float(height), box.y))
    x2 = max(0.0, min(float(width), box.x + box.w))
    y2 = max(0.0, min(float(height), box.y + box.h))
    clipped = (x1, y1, x2, y2) != (box.x, box.y, box.x + box.w, box.y + box.h)
    if x2 <= x1 or y2 <= y1:
        return None, clipped
    return (
        Box(box.frame, box.track_id, x1, y1, x2 - x1, y2 - y1, box.confidence, box.class_name),
        clipped,
    )


def yolo_line(box: Box, width: int, height: int) -> str:
    return (
        f"0 {(box.x + box.w / 2) / width:.8f} {(box.y + box.h / 2) / height:.8f} "
        f"{box.w / width:.8f} {box.h / height:.8f}"
    )


def scale_bin_from_boxes(boxes: Iterable[Box]) -> str:
    values = [max(box.w, box.h) for box in boxes]
    if not values:
        return "negative"
    side = min(values)
    if side < 8:
        return "ultra_tiny"
    if side < 16:
        return "tiny"
    if side < 32:
        return "small"
    if side < 96:
        return "medium"
    return "large"


def related_group(stem: str) -> str:
    """Keep clips cut from the same likely source recording in one split."""
    if re.match(r"^\d{4}_\d{2}_\d{2}_", stem):
        return "_".join(stem.split("_")[:4])
    match = re.match(r"^(GOPR\d{4})_", stem, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    if re.match(r"^gopro_\d+$", stem, re.IGNORECASE):
        return "gopro_series"
    if re.match(r"^\d{2}_\d{2}_\d{2}_to_", stem):
        return "time_cut_series"
    return stem.lower()


def assign_group_splits(
    sequence_weights: dict[str, int], ratios: list[float], seed: int, group_related: bool
) -> tuple[dict[str, str], dict[str, list[str]]]:
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("split_ratio must contain three positive values")
    ratio_sum = sum(ratios)
    normalized = [value / ratio_sum for value in ratios]
    grouped: dict[str, list[str]] = defaultdict(list)
    for sequence in sequence_weights:
        grouped[related_group(sequence) if group_related else sequence].append(sequence)
    group_weights = {key: sum(sequence_weights[name] for name in names) for key, names in grouped.items()}

    rng = random.Random(seed)
    groups = list(group_weights)
    rng.shuffle(groups)
    groups.sort(key=lambda key: group_weights[key], reverse=True)
    total = sum(group_weights.values())
    targets = {name: total * ratio for name, ratio in zip(SPLIT_NAMES, normalized)}
    current = {name: 0 for name in SPLIT_NAMES}
    group_assignment: dict[str, str] = {}
    for group in groups:
        # Largest relative deficit receives the next group.
        split = max(SPLIT_NAMES, key=lambda name: (targets[name] - current[name]) / targets[name])
        group_assignment[group] = split
        current[split] += group_weights[group]

    sequence_assignment = {
        sequence: group_assignment[group]
        for group, sequences in grouped.items()
        for sequence in sequences
    }
    grouped_result = {
        split: sorted(sequence for sequence, assigned in sequence_assignment.items() if assigned == split)
        for split in SPLIT_NAMES
    }
    return sequence_assignment, grouped_result


def stable_key(seed: int, *values: object) -> str:
    text = "|".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
