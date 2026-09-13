from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.data.common import SPLIT_NAMES, stable_key
from src.data.prepare_datasets import reset_output, write_data_yaml
from src.utils.project import REPO_ROOT, dump_json, load_yaml, read_jsonl, resolve_path, write_jsonl


def select_samples(rows: list[dict], count: int, seed: int, source: str, split: str) -> list[dict]:
    """Deterministic stratified selection that retains rare scale buckets."""
    if count >= len(rows):
        return sorted(rows, key=lambda row: row["sample_id"])
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get("scale_bin", "unknown")].append(row)
    for name, bucket in buckets.items():
        bucket.sort(key=lambda row: stable_key(seed, source, split, name, row["sample_id"]))
    allocation = {
        name: min(len(bucket), int(count * len(bucket) / len(rows))) for name, bucket in buckets.items()
    }
    # Keep at least one example of every represented bin when possible.
    if count >= len(buckets):
        for name, bucket in buckets.items():
            allocation[name] = max(1, allocation[name])
    while sum(allocation.values()) < count:
        candidate = max(
            (name for name in buckets if allocation[name] < len(buckets[name])),
            key=lambda name: len(buckets[name]) / (allocation[name] + 1),
        )
        allocation[candidate] += 1
    while sum(allocation.values()) > count:
        candidate = max(
            (name for name in buckets if allocation[name] > 1),
            key=lambda name: allocation[name],
        )
        allocation[candidate] -= 1
    return [row for name, bucket in sorted(buckets.items()) for row in bucket[: allocation[name]]]


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported link_mode: {mode}")


def content_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep one exact image copy, preferring test then val then train."""
    priority = {"test": 0, "val": 1, "train": 2}
    ordered = sorted(
        rows,
        key=lambda row: (priority[row["split"]], stable_key(42, row["sample_id"])),
    )
    kept, removed = [], []
    first_by_hash: dict[str, dict] = {}
    for row in ordered:
        digest = content_hash(resolve_path(row["image_path"]))
        if digest in first_by_hash:
            removed.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "duplicate_of": first_by_hash[digest]["sample_id"],
                    "kept_split": first_by_hash[digest]["split"],
                    "hash": digest,
                }
            )
            continue
        first_by_hash[digest] = row
        kept.append(row)
    return kept, removed


def build(config_path: str, force: bool) -> dict:
    config = load_yaml(config_path)
    output = resolve_path(config["output_root"])
    reset_output(output, force)
    for split in SPLIT_NAMES:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    ratios = {name: float(source["ratio"]) for name, source in config["sources"].items()}
    ratio_total = sum(ratios.values())
    ratios = {name: ratio / ratio_total for name, ratio in ratios.items()}
    all_source_rows = {
        name: read_jsonl(source["metadata"]) for name, source in config["sources"].items()
    }
    selected_rows = []
    selection_summary = {}
    for split in SPLIT_NAMES:
        available = {
            name: [row for row in rows if row["split"] == split]
            for name, rows in all_source_rows.items()
        }
        missing = [name for name, rows in available.items() if not rows]
        if missing:
            raise ValueError(f"No {split} samples for sources: {missing}")
        effective_total = min(len(available[name]) / ratios[name] for name in available)
        target = {name: min(len(rows), int(effective_total * ratios[name])) for name, rows in available.items()}
        selection_summary[split] = {"available": {k: len(v) for k, v in available.items()}, "selected": target}
        for name, rows in available.items():
            selected_rows.extend(select_samples(rows, target[name], int(config["seed"]), name, split))

    duplicate_rows = []
    if config.get("deduplicate_images", True):
        selected_rows, duplicate_rows = deduplicate(selected_rows)
    output_metadata = []
    for row in sorted(selected_rows, key=lambda item: (item["split"], item["sample_id"])):
        source_image = resolve_path(row["image_path"])
        source_label = resolve_path(row["label_path"])
        image_destination = output / "images" / row["split"] / f"{row['sample_id']}{source_image.suffix.lower()}"
        label_destination = output / "labels" / row["split"] / f"{row['sample_id']}.txt"
        link_or_copy(source_image, image_destination, config["link_mode"])
        link_or_copy(source_label, label_destination, config["link_mode"])
        updated = dict(row)
        updated["image_path"] = str(image_destination.relative_to(REPO_ROOT))
        updated["label_path"] = str(label_destination.relative_to(REPO_ROOT))
        output_metadata.append(updated)
    write_jsonl(output / "metadata.jsonl", output_metadata)
    write_data_yaml(output)
    payload = {
        "name": config["name"],
        "seed": config["seed"],
        "requested_ratios": ratios,
        "selection": selection_summary,
        "total_selected": len(output_metadata),
        "link_mode": config["link_mode"],
        "duplicate_images_removed": len(duplicate_rows),
        "duplicates": duplicate_rows,
        "scale_distribution": dict(Counter(row["scale_bin"] for row in output_metadata)),
        "actual_by_split_and_source": {
            split: dict(Counter(row["dataset"] for row in output_metadata if row["split"] == split))
            for split in SPLIT_NAMES
        },
    }
    dump_json(config["split_file"], payload)
    dump_json(output / "build_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a source-balanced combined YOLO dataset.")
    parser.add_argument("--config", default="configs/data/external_uav_combined.yaml")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the output directory")
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.force), indent=2))


if __name__ == "__main__":
    main()
