from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

from src.data.common import (
    IMAGE_EXTENSIONS,
    annotation_path,
    clip_box,
    parse_counted_annotations,
    parse_mot_annotations,
    probe_video,
    video_files,
)
from src.utils.project import REPO_ROOT, dump_json, load_yaml, resolve_path

DEFAULT_CONFIGS = [
    "configs/data/drone_detection.yaml",
    "configs/data/drone_bird.yaml",
    "configs/data/purdue_uav.yaml",
]
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {f"p{p}": None for p in PERCENTILES}
    ordered = sorted(values)
    return {
        f"p{p}": round(ordered[round((len(ordered) - 1) * p / 100)], 6) for p in PERCENTILES
    }


def bbox_row(dataset: str, sequence: str, frame: int | str, width: int, height: int, box) -> dict:
    scale_640 = min(640 / width, 640 / height)
    scale_16x9 = min(832 / width, 480 / height)
    return {
        "dataset": dataset,
        "sequence_id": sequence,
        "frame_id": frame,
        "class_id": 0,
        "bbox_w_original": box.w,
        "bbox_h_original": box.h,
        "bbox_area_original": box.w * box.h,
        "bbox_area_ratio": box.w * box.h / (width * height),
        "bbox_w_640": box.w * scale_640,
        "bbox_h_640": box.h * scale_640,
        "bbox_w_832x480": box.w * scale_16x9,
        "bbox_h_832x480": box.h * scale_16x9,
    }


def inspect_yolo(config: dict[str, Any]) -> tuple[dict, list[dict]]:
    root = resolve_path(config["root"])
    rows: list[dict] = []
    split_summary = {}
    dimensions: Counter[str] = Counter()
    missing_labels: list[str] = []
    malformed: list[str] = []
    class_ids: Counter[int] = Counter()
    for split in config["existing_splits"]:
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        empty = boxes = 0
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(str(label_path.relative_to(REPO_ROOT)))
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            dimensions[f"{width}x{height}"] += 1
            lines = [line for line in label_path.read_text().splitlines() if line.strip()]
            if not lines:
                empty += 1
            for line_no, line in enumerate(lines, 1):
                fields = line.split()
                try:
                    if len(fields) != 5:
                        raise ValueError("expected 5 fields")
                    class_id = int(fields[0])
                    xc, yc, bw, bh = map(float, fields[1:])
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        raise ValueError("normalized coordinate out of range")
                    class_ids[class_id] += 1
                    box = type("PixelBox", (), {"w": bw * width, "h": bh * height})
                    rows.append(bbox_row(config["name"], split, image_path.stem, width, height, box))
                    boxes += 1
                except Exception as error:
                    malformed.append(f"{label_path.relative_to(REPO_ROOT)}:{line_no}: {error}")
        split_summary[split] = {"images": len(images), "boxes": boxes, "empty_frames": empty}
    return {
        "name": config["name"],
        "kind": config["kind"],
        "root": str(root.relative_to(REPO_ROOT)),
        "splits": split_summary,
        "num_images": sum(value["images"] for value in split_summary.values()),
        "num_videos": 0,
        "num_sequences": 0,
        "num_annotations": len(rows),
        "image_resolution_distribution": dict(dimensions),
        "class_distribution": {str(k): v for k, v in sorted(class_ids.items())},
        "missing_labels": missing_labels,
        "malformed_annotations": malformed,
    }, rows


def inspect_video(config: dict[str, Any]) -> tuple[dict, list[dict]]:
    video_root = resolve_path(config["video_root"])
    annotation_root = resolve_path(config["annotation_root"])
    videos = video_files(video_root)
    suffix = config.get("annotation_suffix", "")
    expected_annotations = {
        path.stem.removesuffix(suffix): path for path in annotation_root.glob("*.txt")
    }
    rows: list[dict] = []
    resolutions: Counter[str] = Counter()
    fps_distribution: Counter[str] = Counter()
    class_names: Counter[str] = Counter()
    confidence_distribution: Counter[str] = Counter()
    track_ids: set[tuple[str, int]] = set()
    issues: list[dict] = []
    empty_frames = total_frames = 0
    for stem, video_path in videos.items():
        info = probe_video(video_path)
        total_frames += info.frames
        resolutions[f"{info.width}x{info.height}"] += 1
        fps_distribution[f"{info.fps:.3f}"] += 1
        ann_path = annotation_path(annotation_root, stem, config["kind"], suffix)
        if not ann_path.exists():
            issues.append({"sequence": stem, "error": "missing_annotation"})
            continue
        annotations = (
            parse_counted_annotations(ann_path)
            if config["kind"] == "counted_video"
            else parse_mot_annotations(ann_path)
        )
        valid_frames = set(range(info.frames)) if config["frame_index_base"] == 0 else set(range(1, info.frames + 1))
        empty_frames += sum(1 for frame in valid_frames if not annotations.get(frame))
        for frame, boxes in annotations.items():
            if frame not in valid_frames:
                issues.append({"sequence": stem, "frame": frame, "error": "annotation_frame_out_of_video", "boxes": len(boxes)})
                continue
            for box in boxes:
                class_names[box.class_name] += 1
                confidence_distribution[f"{box.confidence:g}"] += 1
                if box.track_id is not None:
                    track_ids.add((stem, box.track_id))
                clipped_box, clipped = clip_box(box, info.width, info.height)
                if clipped_box is None:
                    issues.append({"sequence": stem, "frame": frame, "error": "box_outside_frame"})
                    continue
                if clipped:
                    issues.append({"sequence": stem, "frame": frame, "error": "box_requires_clipping"})
                rows.append(bbox_row(config["name"], stem, frame, info.width, info.height, clipped_box))
        observed = set(annotations)
        if observed and (min(observed) < min(valid_frames) or max(observed) > max(valid_frames)):
            issues.append({
                "sequence": stem,
                "error": "annotation_range_mismatch",
                "video_frames": info.frames,
                "annotation_min": min(observed),
                "annotation_max": max(observed),
            })
    return {
        "name": config["name"],
        "kind": config["kind"],
        "num_images": 0,
        "num_videos": len(videos),
        "num_sequences": len(videos),
        "num_frames": total_frames,
        "num_annotations": len(rows),
        "empty_frames": empty_frames,
        "video_resolution_distribution": dict(resolutions),
        "fps_distribution": dict(fps_distribution),
        "class_distribution": dict(class_names),
        "annotation_confidence_distribution": dict(confidence_distribution),
        "num_tracks": len(track_ids),
        "missing_videos": sorted(set(expected_annotations) - set(videos)),
        "missing_annotations": sorted(set(videos) - set(expected_annotations)),
        "issues": issues,
    }, rows


def add_bbox_summary(report: dict, rows: list[dict]) -> None:
    for key in ("bbox_w_original", "bbox_h_original", "bbox_area_original", "bbox_area_ratio", "bbox_w_640", "bbox_h_640", "bbox_w_832x480", "bbox_h_832x480"):
        report.setdefault("bbox_statistics", {})[key] = percentiles([float(row[key]) for row in rows])
    bins = Counter()
    for row in rows:
        side = max(float(row["bbox_w_original"]), float(row["bbox_h_original"]))
        name = "ultra_tiny" if side < 8 else "tiny" if side < 16 else "small" if side < 32 else "medium" if side < 96 else "large"
        bins[name] += 1
    report["scale_distribution_original"] = dict(bins)


def markdown_report(report: dict) -> str:
    lines = [f"# {report['name']} inspection", "", f"- Format: `{report['kind']}`"]
    for key in ("num_images", "num_videos", "num_sequences", "num_frames", "num_annotations", "empty_frames", "num_tracks"):
        if key in report:
            lines.append(f"- {key}: {report[key]:,}")
    lines.extend(["", "## Classes", "", f"`{json.dumps(report.get('class_distribution', {}), ensure_ascii=False)}`", "", "## Resolutions", "", f"`{json.dumps(report.get('image_resolution_distribution', report.get('video_resolution_distribution', {})), ensure_ascii=False)}`", "", "## Bbox percentiles", "", "| Value | P5 | P50 | P95 |", "|---|---:|---:|---:|"])
    for key, value in report.get("bbox_statistics", {}).items():
        lines.append(f"| {key} | {value['p5']} | {value['p50']} | {value['p95']} |")
    issue_count = len(report.get("issues", [])) + len(report.get("malformed_annotations", []))
    lines.extend(["", "## Validation", "", f"- Recorded issues: {issue_count:,}", "- Full issue details are stored in the JSON report.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect all raw Phase-1 datasets without modifying them.")
    parser.add_argument("--config", action="append", help="Dataset YAML; repeat for multiple datasets")
    parser.add_argument("--report-dir", default="data/reports")
    args = parser.parse_args()
    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    reports = []
    for path in args.config or DEFAULT_CONFIGS:
        config = load_yaml(path)
        report, rows = inspect_yolo(config) if config["kind"] == "yolo_images" else inspect_video(config)
        add_bbox_summary(report, rows)
        reports.append(report)
        all_rows.extend(rows)
        dump_json(report_dir / f"{config['name']}_inspection.json", report)
        (report_dir / f"{config['name']}_inspection.md").write_text(markdown_report(report), encoding="utf-8")
        print(f"{config['name']}: {report.get('num_annotations', 0):,} valid boxes")
    csv_path = report_dir / "bbox_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]) if all_rows else [])
        if all_rows:
            writer.writeheader()
            writer.writerows(all_rows)
    dump_json(report_dir / "inspection_summary.json", reports)
    print(f"reports: {report_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
