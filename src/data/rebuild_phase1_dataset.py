from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from src.data.build_combined_dataset import content_hash, link_or_copy
from src.data.common import (
    SPLIT_NAMES,
    annotation_path,
    parse_counted_annotations,
    parse_mot_annotations,
    probe_video,
    related_group,
    stable_key,
    video_files,
)
from src.data.prepare_datasets import reset_output, write_data_yaml
from src.utils.project import REPO_ROOT, dump_json, load_yaml, read_jsonl, resolve_path, write_jsonl


PROXY_BINS = ("negative", "subpixel", "company_like", "small", "large")


def parse_effective_boxes(row: dict[str, Any], canvas: tuple[int, int]) -> list[tuple[float, float]]:
    width, height = int(row["width"]), int(row["height"])
    target_w, target_h = canvas
    scale = min(target_w / width, target_h / height)
    boxes: list[tuple[float, float]] = []
    for raw in resolve_path(row["label_path"]).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5 or int(fields[0]) != 0:
            raise ValueError(f"Invalid YOLO row in {row['label_path']}: {raw!r}")
        _, _, bw, bh = map(float, fields[1:])
        boxes.append((bw * width * scale, bh * height * scale))
    return boxes


def proxy_bin(boxes: list[tuple[float, float]], median: tuple[float, float]) -> str:
    if not boxes:
        return "negative"
    if all(min(w, h) < 1.5 for w, h in boxes):
        return "subpixel"
    median_w, median_h = median
    if any(0.45 * median_w <= w <= 2.0 * median_w and 0.45 * median_h <= h <= 2.5 * median_h for w, h in boxes):
        return "company_like"
    if any(max(w, h) < 32 for w, h in boxes):
        return "small"
    return "large"


def is_uniform_suspicious_image(row: dict[str, Any], byte_limit: int, std_limit: float) -> bool:
    path = resolve_path(row["image_path"])
    if path.stat().st_size > byte_limit:
        return False
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image is None or float(image.std()) < std_limit


def annotate_and_filter(
    rows: list[dict[str, Any]], config: dict[str, Any], *, filter_quality: bool
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    company = config["company_proxy"]
    training = config["training_canvas"]
    company_canvas = (int(company["width"]), int(company["height"]))
    training_canvas = (int(training["width"]), int(training["height"]))
    median = (float(company["median_bbox_width"]), float(company["median_bbox_height"]))
    quality = config["quality"]
    accepted, excluded = [], []
    for original in rows:
        row = dict(original)
        company_boxes = parse_effective_boxes(row, company_canvas)
        training_boxes = parse_effective_boxes(row, training_canvas)
        row["company_proxy_bin"] = proxy_bin(company_boxes, median)
        row["effective_boxes_1280x720"] = len(company_boxes)
        reason = None
        if filter_quality and quality["exclude_clipped_annotation_frames"] and row.get("clipped_boxes", 0):
            reason = "clipped_annotation"
        elif filter_quality and training_boxes and all(
            min(width, height) < float(quality["min_effective_box_side"])
            for width, height in training_boxes
        ):
            reason = "all_boxes_below_effective_pixel_limit"
        elif (
            filter_quality
            and row["dataset"] == "purdue_uav"
            and quality["exclude_all_interpolated_tiny_purdue"]
            and row.get("num_boxes", 0) > 0
            and row.get("interpolated_boxes", 0) == row.get("num_boxes", 0)
            and training_boxes
            and max(max(box) for box in training_boxes) < float(quality["interpolated_tiny_max_side"])
        ):
            reason = "all_interpolated_and_tiny"
        elif filter_quality and is_uniform_suspicious_image(
            row, int(quality["suspicious_jpeg_bytes"]), float(quality["uniform_gray_std"])
        ):
            reason = "uniform_or_undecodable_image"
        if reason:
            excluded.append({"sample_id": row["sample_id"], "dataset": row["dataset"], "reason": reason})
        else:
            accepted.append(row)
    return accepted, excluded


def distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(row["company_proxy_bin"] for row in rows)
    total = max(1, sum(counts.values()))
    return {name: counts[name] / total for name in PROXY_BINS}


def choose_validation_groups(
    rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], fraction: float, seed: int, source: str
) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[related_group(str(row["sequence_id"])) if source == "drone_bird" else str(row["sequence_id"])].append(row)
    groups = sorted(grouped)
    if len(groups) < 2:
        raise ValueError(f"Need at least two development groups for {source}")
    target_dist = distribution(test_rows)
    target_count = len(rows) * fraction
    stats = {
        group: (len(items), Counter(item["company_proxy_bin"] for item in items))
        for group, items in grouped.items()
    }

    def score(chosen: set[str]) -> float:
        count = sum(stats[group][0] for group in chosen)
        bins = sum((stats[group][1] for group in chosen), Counter())
        size_error = abs(count - target_count) / max(target_count, 1)
        dist_error = sum(abs(bins[name] / max(count, 1) - target_dist[name]) for name in PROXY_BINS)
        return 2.5 * size_error + dist_error

    rng = random.Random(stable_key(seed, source, "val_groups"))
    best: set[str] = set()
    best_score = math.inf
    for _ in range(40000):
        probability = rng.uniform(max(0.05, fraction - 0.08), min(0.45, fraction + 0.12))
        chosen = {group for group in groups if rng.random() < probability}
        if not chosen or len(chosen) == len(groups):
            continue
        value = score(chosen)
        if value < best_score:
            best, best_score = chosen, value
    return best


def stratified_take(rows: list[dict[str, Any]], count: int, seed: int, key: str) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return sorted(rows, key=lambda row: row["sample_id"])
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["company_proxy_bin"]].append(row)
    for name, bucket in buckets.items():
        bucket.sort(key=lambda row: stable_key(seed, key, name, row["sample_id"]))
    allocation = {name: int(count * len(bucket) / len(rows)) for name, bucket in buckets.items()}
    while sum(allocation.values()) < count:
        name = max(buckets, key=lambda item: len(buckets[item]) / (allocation[item] + 1))
        allocation[name] += 1
    return [row for name in sorted(buckets) for row in buckets[name][: allocation[name]]]


def cap_sequences(rows: list[dict[str, Any]], cap: int, seed: int, split: str) -> list[dict[str, Any]]:
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_sequence = []
    for row in rows:
        if row.get("sequence_id"):
            by_sequence[f"{row['dataset']}::{row['sequence_id']}"] .append(row)
        else:
            no_sequence.append(row)
    output = list(no_sequence)
    for sequence, items in sorted(by_sequence.items()):
        output.extend(stratified_take(items, cap, seed, f"{split}:{sequence}"))
    return output


def source_balanced_take(
    by_source: dict[str, list[dict[str, Any]]], ratios: dict[str, float], seed: int, split: str
) -> list[dict[str, Any]]:
    effective_total = min(len(by_source[name]) / ratios[name] for name in ratios)
    targets = {name: min(len(by_source[name]), int(effective_total * ratios[name])) for name in ratios}
    return [
        row
        for name in sorted(by_source)
        for row in stratified_take(by_source[name], targets[name], seed, f"{split}:{name}")
    ]


def add_annotation_negatives(
    config: dict[str, Any], selected: list[dict[str, Any]], assignments: dict[str, dict[str, list[str]]], force: bool
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extract extra empty-label frames from development sequences only."""
    target_ratio = float(config.get("negative_target_ratio", 0.0))
    cache = resolve_path(config["negative_cache_root"])
    reset_output(cache, force)
    for split in ("train", "val"):
        (cache / "images" / split).mkdir(parents=True, exist_ok=True)
        (cache / "labels" / split).mkdir(parents=True, exist_ok=True)
    existing_ids = {
        row["sample_id"]
        for source_config in config["sources"].values()
        for row in read_jsonl(source_config["metadata"])
    }
    candidates: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    stride = int(config["additional_negative_stride"])
    offset = int(config["additional_negative_offset"])
    per_sequence_cap = int(config["max_additional_negatives_per_sequence"])
    for source, source_config in config["sources"].items():
        if "raw_config" not in source_config:
            continue
        raw_config = load_yaml(source_config["raw_config"])
        videos = video_files(resolve_path(raw_config["video_root"]))
        annotation_root = resolve_path(raw_config["annotation_root"])
        split_by_sequence = {
            sequence: split
            for split in ("train", "val")
            for sequence in assignments[source][split]
        }
        for sequence, split in sorted(split_by_sequence.items()):
            video_path = videos[sequence]
            info = probe_video(video_path)
            path = annotation_path(
                annotation_root, sequence, raw_config["kind"], raw_config.get("annotation_suffix", "")
            )
            annotations = (
                parse_counted_annotations(path)
                if raw_config["kind"] == "counted_video"
                else parse_mot_annotations(path)
            )
            sequence_candidates = []
            for frame_zero in range(offset, info.frames, stride):
                frame_id = frame_zero + int(raw_config["frame_index_base"])
                if annotations.get(frame_id):
                    continue
                sample_id = f"{source}__{sequence}__{frame_zero:06d}"
                if sample_id in existing_ids:
                    continue
                sequence_candidates.append({
                    "sample_id": sample_id, "dataset": source, "sequence_id": sequence,
                    "frame_id": frame_id, "split": split, "original_path": str(video_path.relative_to(REPO_ROOT)),
                    "width": info.width, "height": info.height, "num_boxes": 0,
                    "scale_bin": "negative", "company_proxy_bin": "negative", "track_ids": [],
                    "sampling_reason": "additional_annotation_negative",
                })
            sequence_candidates.sort(key=lambda row: stable_key(config["seed"], "extra_negative", row["sample_id"]))
            candidates[split].extend(sequence_candidates[:per_sequence_cap])

    requested: dict[str, int] = {}
    chosen: list[dict[str, Any]] = []
    for split in ("train", "val"):
        rows = [row for row in selected if row["split"] == split]
        negatives = sum(row["num_boxes"] == 0 for row in rows)
        needed = max(0, math.ceil((target_ratio * len(rows) - negatives) / max(1e-9, 1 - target_ratio)))
        requested[split] = needed
        candidates[split].sort(key=lambda row: stable_key(config["seed"], split, row["dataset"], row["sample_id"]))
        # Alternate sources to retain both bird/sky and moving-camera/terrain negatives.
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates[split]:
            by_source[row["dataset"]].append(row)
        names = sorted(by_source)
        while len(chosen) < sum(requested.values()) and any(by_source[name] for name in names):
            progressed = False
            for name in names:
                if by_source[name] and sum(row["split"] == split for row in chosen) < needed:
                    chosen.append(by_source[name].pop())
                    progressed = True
            if not progressed or sum(row["split"] == split for row in chosen) >= needed:
                break

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chosen:
        by_video[row["original_path"]].append(row)
    output_rows = []
    uniform_removed = 0
    quality = config["quality"]
    for relative_video, rows in sorted(by_video.items()):
        wanted = {int(row["frame_id"]) - (1 if row["dataset"] == "purdue_uav" else 0): row for row in rows}
        capture = cv2.VideoCapture(str(resolve_path(relative_video)))
        frame_zero = 0
        while wanted:
            ok, image = capture.read()
            if not ok:
                break
            row = wanted.pop(frame_zero, None)
            frame_zero += 1
            if row is None:
                continue
            if float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std()) < float(quality["uniform_gray_std"]):
                uniform_removed += 1
                continue
            image_out = cache / "images" / row["split"] / f"{row['sample_id']}.jpg"
            label_out = cache / "labels" / row["split"] / f"{row['sample_id']}.txt"
            if not cv2.imwrite(str(image_out), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Failed to write {image_out}")
            label_out.write_text("", encoding="utf-8")
            row["image_path"] = str(image_out.relative_to(REPO_ROOT))
            row["label_path"] = str(label_out.relative_to(REPO_ROOT))
            output_rows.append(row)
        capture.release()
        if wanted:
            raise ValueError(f"Could not decode selected frames from {relative_video}: {sorted(wanted)[:5]}")
    write_jsonl(cache / "metadata.jsonl", output_rows)
    return output_rows, {
        "requested_train": requested["train"], "requested_val": requested["val"],
        "written_train": sum(row["split"] == "train" for row in output_rows),
        "written_val": sum(row["split"] == "val" for row in output_rows),
        "uniform_removed": uniform_removed,
    }


def deduplicate_with_frozen_test(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    priority = {"test": 0, "val": 1, "train": 2}
    kept, removed, seen = [], [], {}
    for row in sorted(rows, key=lambda item: (priority[item["split"]], stable_key(seed, item["sample_id"]))):
        digest = content_hash(resolve_path(row["image_path"]))
        if digest in seen:
            removed.append({
                "sample_id": row["sample_id"], "split": row["split"],
                "duplicate_of": seen[digest]["sample_id"], "kept_split": seen[digest]["split"],
            })
            continue
        seen[digest] = row
        kept.append(row)
    return kept, removed


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        selected = [row for row in rows if row["split"] == split]
        result[split] = {
            "images": len(selected),
            "boxes": sum(int(row["num_boxes"]) for row in selected),
            "negative_frames": sum(int(row["num_boxes"]) == 0 for row in selected),
            "negative_ratio": sum(int(row["num_boxes"]) == 0 for row in selected) / max(1, len(selected)),
            "sources": dict(Counter(row["dataset"] for row in selected)),
            "proxy_bins_1280x720": dict(Counter(row["company_proxy_bin"] for row in selected)),
            "sequences": len({(row["dataset"], row["sequence_id"]) for row in selected if row.get("sequence_id")}),
        }
    return result


def build(config_path: str, force: bool) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config["seed"])
    output = resolve_path(config["output_root"])
    reset_output(output, force)
    for split in SPLIT_NAMES:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    fixed_rows_raw = [row for row in read_jsonl(config["fixed_test_metadata"]) if row["split"] == "test"]
    fixed_test, _ = annotate_and_filter(fixed_rows_raw, config, filter_quality=False)
    test_ids = {row["sample_id"] for row in fixed_test}
    ratios = {name: float(value["ratio"]) for name, value in config["sources"].items()}
    ratio_sum = sum(ratios.values())
    ratios = {name: value / ratio_sum for name, value in ratios.items()}
    source_rows: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, str]] = []
    assignments: dict[str, dict[str, list[str]]] = {}
    train_by_source: dict[str, list[dict[str, Any]]] = {}
    val_by_source: dict[str, list[dict[str, Any]]] = {}

    for source, source_config in config["sources"].items():
        raw = [row for row in read_jsonl(source_config["metadata"]) if row["split"] != "test" and row["sample_id"] not in test_ids]
        blocked_sequences = set(config.get("exclude_sequences", {}).get(source, []))
        blocked = [row for row in raw if row.get("sequence_id") in blocked_sequences]
        exclusions.extend(
            {"sample_id": row["sample_id"], "dataset": source, "reason": "sequence_overlaps_fixed_test"}
            for row in blocked
        )
        raw = [row for row in raw if row.get("sequence_id") not in blocked_sequences]
        clean, removed = annotate_and_filter(raw, config, filter_quality=True)
        source_rows[source] = clean
        exclusions.extend(removed)
        if source_config.get("preserve_official_train_val"):
            train_rows = [row for row in clean if row["split"] == "train"]
            val_rows = [row for row in clean if row["split"] == "val"]
            assignments[source] = {"train": ["official_train"], "val": ["official_val"]}
        else:
            source_test = [row for row in fixed_test if row["dataset"] == source]
            val_groups = choose_validation_groups(
                clean, source_test, float(config["val_fraction_of_development"]), seed, source
            )
            def group_name(row: dict[str, Any]) -> str:
                value = str(row["sequence_id"])
                return related_group(value) if source == "drone_bird" else value
            val_rows = [row for row in clean if group_name(row) in val_groups]
            train_rows = [row for row in clean if group_name(row) not in val_groups]
            assignments[source] = {
                "train": sorted({str(row["sequence_id"]) for row in train_rows}),
                "val": sorted({str(row["sequence_id"]) for row in val_rows}),
            }
        for row in train_rows:
            row["split"] = "train"
        for row in val_rows:
            row["split"] = "val"
        train_by_source[source] = train_rows
        val_by_source[source] = val_rows

    train_by_source = {
        name: cap_sequences(rows, int(config["max_train_frames_per_sequence"]), seed, "train")
        for name, rows in train_by_source.items()
    }
    val_by_source = {
        name: cap_sequences(rows, int(config["max_val_frames_per_sequence"]), seed, "val")
        for name, rows in val_by_source.items()
    }
    selected = source_balanced_take(train_by_source, ratios, seed, "train")
    selected += source_balanced_take(val_by_source, ratios, seed, "val")
    extra_negatives, negative_summary = add_annotation_negatives(config, selected, assignments, force)
    selected += extra_negatives
    selected += fixed_test
    if config.get("deduplicate_images", True):
        selected, duplicates = deduplicate_with_frozen_test(selected, seed)
    else:
        duplicates = []

    output_rows = []
    for row in sorted(selected, key=lambda item: (item["split"], item["sample_id"])):
        source_image = resolve_path(row["image_path"]).resolve()
        source_label = resolve_path(row["label_path"]).resolve()
        image_out = output / "images" / row["split"] / f"{row['sample_id']}{source_image.suffix.lower()}"
        label_out = output / "labels" / row["split"] / f"{row['sample_id']}.txt"
        link_or_copy(source_image, image_out, config["link_mode"])
        link_or_copy(source_label, label_out, config["link_mode"])
        updated = dict(row)
        updated["image_path"] = str(image_out.relative_to(REPO_ROOT))
        updated["label_path"] = str(label_out.relative_to(REPO_ROOT))
        output_rows.append(updated)

    write_jsonl(output / "metadata.jsonl", output_rows)
    write_data_yaml(output)
    summary = {
        "name": config["name"], "seed": seed, "fixed_test": True,
        "requested_source_ratios": ratios,
        "company_proxy": config["company_proxy"],
        "training_canvas": config["training_canvas"],
        "quality_rules": config["quality"],
        "sequence_assignments": assignments,
        "excluded_count": len(exclusions),
        "excluded_by_reason": dict(Counter(item["reason"] for item in exclusions)),
        "excluded_samples": exclusions,
        "duplicate_count": len(duplicates), "duplicates": duplicates,
        "additional_negatives": negative_summary,
        "dataset": summarize(output_rows),
    }
    dump_json(config["split_file"], summary)
    dump_json(output / "build_summary.json", summary)
    dump_json(config["quality_report"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the Phase-1 train/val set while freezing the original test set.")
    parser.add_argument("--config", default="configs/data/external_uav_phase1_v2.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args.config, args.force)
    print(json.dumps({
        "name": summary["name"],
        "excluded_count": summary["excluded_count"],
        "excluded_by_reason": summary["excluded_by_reason"],
        "additional_negatives": summary["additional_negatives"],
        "dataset": summary["dataset"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
