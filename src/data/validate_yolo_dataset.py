from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import yaml
from PIL import Image

from src.utils.project import REPO_ROOT, dump_json, read_jsonl, resolve_path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def file_hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(dataset_root: Path, hash_leakage: bool, visualize: int, report_path: Path) -> dict:
    metadata_path = dataset_root / "metadata.jsonl"
    metadata = read_jsonl(metadata_path) if metadata_path.exists() else []
    metadata_by_image = {Path(row["image_path"]).name: row for row in metadata}
    issues = []
    sequence_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    original_splits: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    totals = defaultdict(int)
    visualization_candidates = []
    for split in ("train", "val", "test"):
        image_dir, label_dir = dataset_root / "images" / split, dataset_root / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in images:
            totals[f"{split}_images"] += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                issues.append({"type": "missing_label", "path": str(image_path)})
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as error:
                issues.append({"type": "corrupt_image", "path": str(image_path), "error": str(error)})
                continue
            lines = [line for line in label_path.read_text().splitlines() if line.strip()]
            totals[f"{split}_boxes"] += len(lines)
            totals[f"{split}_negative_frames"] += not lines
            valid_boxes = []
            for line_no, line in enumerate(lines, 1):
                try:
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError("expected five fields")
                    class_id = int(fields[0]); values = list(map(float, fields[1:]))
                    xc, yc, bw, bh = values
                    if class_id != 0:
                        raise ValueError(f"class_id={class_id}")
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        raise ValueError(f"coordinates={values}")
                    if xc - bw / 2 < -1e-6 or yc - bh / 2 < -1e-6 or xc + bw / 2 > 1 + 1e-6 or yc + bh / 2 > 1 + 1e-6:
                        raise ValueError(f"box edges outside image={values}")
                    valid_boxes.append(values)
                except Exception as error:
                    issues.append({"type": "invalid_label", "path": str(label_path), "line": line_no, "error": str(error)})
            row = metadata_by_image.get(image_path.name)
            if row:
                if row.get("sequence_id"):
                    sequence_splits[(row["dataset"], row["sequence_id"])].add(split)
                original_splits[row["original_path"]].add(split)
            if hash_leakage:
                hashes[file_hash(image_path)].append((split, str(image_path.relative_to(REPO_ROOT))))
            if valid_boxes:
                visualization_candidates.append((image_path, valid_boxes, width, height))
    for key, splits in sequence_splits.items():
        if len(splits) > 1:
            issues.append({"type": "sequence_leakage", "dataset": key[0], "sequence": key[1], "splits": sorted(splits)})
    for path, splits in original_splits.items():
        if len(splits) > 1:
            issues.append({"type": "original_source_leakage", "path": path, "splits": sorted(splits)})
    if hash_leakage:
        for digest, occurrences in hashes.items():
            splits = {split for split, _ in occurrences}
            if len(splits) > 1:
                issues.append({"type": "image_hash_leakage", "hash": digest, "splits": sorted(splits), "paths": [path for _, path in occurrences]})

    if visualize:
        destination = report_path.parent / "visualizations"
        destination.mkdir(parents=True, exist_ok=True)
        rng = random.Random(42)
        for image_path, boxes, width, height in rng.sample(visualization_candidates, min(visualize, len(visualization_candidates))):
            image = cv2.imread(str(image_path))
            for xc, yc, bw, bh in boxes:
                x1, y1 = int((xc - bw / 2) * width), int((yc - bh / 2) * height)
                x2, y2 = int((xc + bw / 2) * width), int((yc + bh / 2) * height)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), max(1, round(width / 640)))
            cv2.imwrite(str(destination / image_path.name), image)
    report = {"dataset_root": str(dataset_root.relative_to(REPO_ROOT)), "totals": dict(totals), "hash_leakage_checked": hash_leakage, "issue_count": len(issues), "issues": issues}
    dump_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO labels, image integrity, and split leakage.")
    parser.add_argument("--data", default="data/processed/external_uav_combined/data.yaml")
    parser.add_argument("--hash-leakage", action="store_true", help="Hash all images to detect cross-split duplicates")
    parser.add_argument("--visualize", type=int, default=0)
    parser.add_argument("--report", default="data/reports/external_uav_combined_validation.json")
    args = parser.parse_args()
    data_yaml = resolve_path(args.data)
    config = yaml.safe_load(data_yaml.read_text())
    root = Path(config["path"])
    root = root if root.is_absolute() else (data_yaml.parent / root).resolve()
    report = validate(root, args.hash_leakage, args.visualize, resolve_path(args.report))
    print(json.dumps({"totals": report["totals"], "issue_count": report["issue_count"]}, indent=2))
    if report["issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
