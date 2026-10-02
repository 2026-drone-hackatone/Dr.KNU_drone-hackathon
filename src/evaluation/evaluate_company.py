from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.utils.project import dump_json, resolve_path


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model_path = resolve_path(args.model)
    data_path = resolve_path(args.data)
    output_path = resolve_path(args.output)
    run_dir = output_path.parent
    project_dir = run_dir.parent
    project_dir.mkdir(parents=True, exist_ok=True)
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Evaluation run exists: {run_dir}. Choose another --output or use --exist-ok.")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        rect=True,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        augment=args.augment,
        single_cls=args.single_cls,
        plots=args.plots,
        save_json=args.save_json,
        project=str(project_dir),
        name=run_dir.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )

    raw_names = model.names
    names = raw_names if isinstance(raw_names, dict) else dict(enumerate(raw_names))
    class_ids = [int(value) for value in metrics.box.ap_class_index.tolist()]
    report = {
        "model": str(model_path),
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "rect": True,
        "batch": args.batch,
        "augment": args.augment,
        "single_cls": args.single_cls,
        "conf": args.conf,
        "iou": args.iou,
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "classes": {
            str(class_id): {
                "name": str(names[class_id]),
                "precision": float(metrics.box.p[index]),
                "recall": float(metrics.box.r[index]),
                "map50": float(metrics.box.ap50[index]),
                "map50_95": float(metrics.box.ap[index]),
            }
            for index, class_id in enumerate(class_ids)
        },
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "save_dir": str(metrics.save_dir),
    }
    metrics_path = Path(metrics.save_dir) / output_path.name
    dump_json(metrics_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one detector on the company split and persist metrics.")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--data", default="data/processed/airbility_uav_detection_yolo/data.yaml"
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="Class-agnostic scoring: all GT and predictions collapse to one class (measures detection without classification)",
    )
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
