from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO

from src.utils.project import REPO_ROOT, dump_json, resolve_path


DATASETS = {
    "combined": "data/processed/external_uav_combined/data.yaml",
    "drone_detection": "data/processed/drone_detection/data.yaml",
    "drone_bird": "data/processed/drone_bird/data.yaml",
    "purdue_uav": "data/processed/purdue_uav/data.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained detector on combined and source test sets.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", default="runs/evaluation")
    args = parser.parse_args()
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(resolve_path(args.weights)))
    rows = []
    for name, data in DATASETS.items():
        path = resolve_path(data)
        if not path.exists():
            raise FileNotFoundError(path)
        metrics = model.val(
            data=str(path), split="test", imgsz=args.imgsz, device=args.device, batch=args.batch,
            project=str(output), name=name, exist_ok=False, plots=True,
        )
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        rows.append(
            {
                "dataset": name,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }
        )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    dump_json(output / "metrics.json", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
