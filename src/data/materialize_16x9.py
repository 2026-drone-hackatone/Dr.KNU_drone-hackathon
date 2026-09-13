from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.data.prepare_datasets import reset_output, write_data_yaml
from src.utils.project import REPO_ROOT, dump_json, read_jsonl, resolve_path, write_jsonl


def transform_label(line: str, source_w: int, source_h: int, scale: float, left: int, top: int, width: int, height: int) -> str:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"Invalid YOLO label: {line!r}")
    class_id = int(fields[0])
    xc, yc, bw, bh = map(float, fields[1:])
    xc = (xc * source_w * scale + left) / width
    yc = (yc * source_h * scale + top) / height
    bw = bw * source_w * scale / width
    bh = bh * source_h * scale / height
    return f"{class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"


def materialize(source: Path, output: Path, width: int, height: int, force: bool, limit: int | None) -> dict:
    reset_output(output, force)
    metadata = read_jsonl(source / "metadata.jsonl")
    if limit is not None:
        metadata = metadata[:limit]
    output_rows = []
    for row in metadata:
        image_path, label_path = resolve_path(row["image_path"]), resolve_path(row["label_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot decode {image_path}")
        source_h, source_w = image.shape[:2]
        scale = min(width / source_w, height / source_h)
        resized_w, resized_h = round(source_w * scale), round(source_h * scale)
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        left, top = (width - resized_w) // 2, (height - resized_h) // 2
        canvas = np.full((height, width, 3), 114, dtype=np.uint8)
        canvas[top : top + resized_h, left : left + resized_w] = resized
        image_out = output / "images" / row["split"] / f"{row['sample_id']}.jpg"
        label_out = output / "labels" / row["split"] / f"{row['sample_id']}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Failed to write {image_out}")
        lines = [line for line in label_path.read_text().splitlines() if line.strip()]
        converted = [transform_label(line, source_w, source_h, scale, left, top, width, height) for line in lines]
        label_out.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
        updated = dict(row)
        updated.update(
            image_path=str(image_out.relative_to(REPO_ROOT)),
            label_path=str(label_out.relative_to(REPO_ROOT)),
            width=width,
            height=height,
            letterbox={"scale": scale, "left": left, "top": top, "source_width": source_w, "source_height": source_h},
        )
        output_rows.append(updated)
    write_jsonl(output / "metadata.jsonl", output_rows)
    write_data_yaml(output)
    summary = {"source": str(source.relative_to(REPO_ROOT)), "output": str(output.relative_to(REPO_ROOT)), "width": width, "height": height, "images": len(output_rows)}
    dump_json(output / "build_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a fixed 16:9 letterbox view for rectangular YOLO training.")
    parser.add_argument("--source", default="data/processed/external_uav_combined")
    parser.add_argument("--output", default="data/processed/external_uav_combined_832x480")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--limit", type=int, help="Only convert the first N samples (smoke tests)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    summary = materialize(resolve_path(args.source), resolve_path(args.output), args.width, args.height, args.force, args.limit)
    print(summary)


if __name__ == "__main__":
    main()
