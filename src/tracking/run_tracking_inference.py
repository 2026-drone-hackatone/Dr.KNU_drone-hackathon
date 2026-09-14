from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src.utils.project import dump_json, resolve_path


DATASETS = {
    "drone_bird": {
        "videos": "data/drone_bird/videos",
        "splits": "data/splits/drone_bird/splits.json",
    },
    "purdue_uav": {
        "videos": "data/purdue_uav/videos",
        "splits": "data/splits/purdue_uav/splits.json",
    },
}
VIDEO_SUFFIXES = {".avi", ".mov", ".mp4", ".mpeg", ".mpg", ".mkv"}
CSV_FIELDS = [
    "processed_frame",
    "source_frame",
    "track_id",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
]


def load_sequences(dataset: str, split: str) -> list[str]:
    dataset_config = DATASETS[dataset]
    videos_dir = resolve_path(dataset_config["videos"])
    if split == "all":
        return sorted(
            path.stem for path in videos_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )

    split_path = resolve_path(dataset_config["splits"])
    with split_path.open(encoding="utf-8") as handle:
        split_data = json.load(handle)
    sequences = split_data.get(split)
    if not isinstance(sequences, list) or not all(isinstance(item, str) for item in sequences):
        raise ValueError(f"Invalid '{split}' sequence list: {split_path}")
    return sequences


def resolve_video(dataset: str, sequence: str) -> Path:
    videos_dir = resolve_path(DATASETS[dataset]["videos"])
    matches = sorted(
        path
        for path in videos_dir.iterdir()
        if path.is_file() and path.stem == sequence and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one video for {dataset}/{sequence}, found {len(matches)} in {videos_dir}"
        )
    return matches[0]


def video_metadata(video_path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0:
        fps = 30.0
    return fps, frame_count


def write_track_rows(writer: csv.DictWriter, result: Any, processed_frame: int, source_frame: int) -> int:
    boxes = result.boxes
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return 0

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    track_ids = boxes.id.detach().cpu().numpy().astype(int)
    for coordinates, confidence, class_id, track_id in zip(xyxy, confidences, classes, track_ids):
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        writer.writerow(
            {
                "processed_frame": processed_frame,
                "source_frame": source_frame,
                "track_id": int(track_id),
                "class_id": int(class_id),
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        )
    return len(track_ids)


def run_video(
    model: YOLO,
    dataset: str,
    split: str,
    video_path: Path,
    output_root: Path,
    tracker_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = output_root / dataset / split / video_path.stem
    video_output = output_dir / "annotated.mp4"
    tracks_output = output_dir / "tracks.csv"
    summary_output = output_dir / "summary.json"

    if output_dir.exists() and not args.overwrite:
        print(f"[SKIP] output exists: {output_dir} (use --overwrite to replace files)")
        return {"dataset": dataset, "split": split, "sequence": video_path.stem, "status": "skipped"}

    output_dir.mkdir(parents=True, exist_ok=True)
    source_fps, source_frames = video_metadata(video_path)
    output_fps = source_fps / args.vid_stride
    video_writer: cv2.VideoWriter | None = None
    processed_frames = 0
    track_rows = 0
    unique_track_ids: set[int] = set()
    started = time.perf_counter()

    stream = model.track(
        source=str(video_path),
        tracker=str(tracker_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        quantize=16 if args.half else None,
        vid_stride=args.vid_stride,
        persist=False,
        stream=True,
        save=False,
        verbose=args.verbose,
    )

    try:
        with tracks_output.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            for processed_frames, result in enumerate(stream, start=1):
                source_frame = 1 + (processed_frames - 1) * args.vid_stride
                track_rows += write_track_rows(csv_writer, result, processed_frames, source_frame)
                if result.boxes is not None and result.boxes.id is not None:
                    unique_track_ids.update(result.boxes.id.int().cpu().tolist())

                annotated = result.plot()
                if args.save_video:
                    if video_writer is None:
                        height, width = annotated.shape[:2]
                        video_writer = cv2.VideoWriter(
                            str(video_output),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps,
                            (width, height),
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError(f"Cannot create output video: {video_output}")
                    video_writer.write(annotated)

                if args.max_frames is not None and processed_frames >= args.max_frames:
                    break
    finally:
        if video_writer is not None:
            video_writer.release()

    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete",
        "dataset": dataset,
        "split": split,
        "sequence": video_path.stem,
        "source": str(video_path),
        "source_fps": source_fps,
        "source_frames": source_frames,
        "processed_frames": processed_frames,
        "vid_stride": args.vid_stride,
        "track_rows": track_rows,
        "unique_track_ids": sorted(unique_track_ids),
        "unique_track_count": len(unique_track_ids),
        "elapsed_seconds": elapsed,
        "processing_fps": processed_frames / elapsed if elapsed else 0.0,
        "weights": str(args.weights_path),
        "tracker": str(tracker_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "half": args.half,
        "video_output": str(video_output) if args.save_video else None,
        "tracks_output": str(tracks_output),
    }
    dump_json(summary_output, summary)
    print(
        f"[DONE] {dataset}/{split}/{video_path.name}: frames={processed_frames}, "
        f"tracks={len(unique_track_ids)}, fps={summary['processing_fps']:.2f}, output={output_dir}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO tracking on sequence-safe validation or test videos and save explicit outputs."
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], required=True)
    parser.add_argument("--split", choices=["val", "test", "all"], required=True)
    parser.add_argument(
        "--weights",
        default="runs/retrain_yolo11/R9_yolo11n_1280x720_v2/weights/epoch23.pt",
    )
    parser.add_argument("--tracker", default="configs/tracker/botsort_gmc_drone.yaml")
    parser.add_argument("--output-root", default="runs/tracking/inference")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.vid_stride < 1:
        parser.error("--vid-stride must be at least 1")
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max-videos must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    args.weights_path = resolve_path(args.weights)
    tracker_path = resolve_path(args.tracker)
    output_root = resolve_path(args.output_root)
    if not args.weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights_path}")
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker config not found: {tracker_path}")

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    jobs: list[tuple[str, Path]] = []
    for dataset in datasets:
        sequences = load_sequences(dataset, args.split)
        if args.max_videos is not None:
            sequences = sequences[: args.max_videos]
        jobs.extend((dataset, resolve_video(dataset, sequence)) for sequence in sequences)

    print(
        f"jobs={len(jobs)} split={args.split} weights={args.weights_path} "
        f"tracker={tracker_path} output={output_root}"
    )
    model = YOLO(str(args.weights_path))
    summaries = []
    failures = []
    for dataset, video_path in jobs:
        try:
            summaries.append(
                run_video(model, dataset, args.split, video_path, output_root, tracker_path, args)
            )
        except Exception as error:  # Continue the batch and report every failed sequence.
            failures.append({"dataset": dataset, "sequence": video_path.stem, "error": repr(error)})
            print(f"[FAILED] {dataset}/{video_path.name}: {error!r}")

    batch_summary = {
        "status": "complete" if not failures else "completed_with_failures",
        "jobs": len(jobs),
        "completed": sum(item.get("status") == "complete" for item in summaries),
        "skipped": sum(item.get("status") == "skipped" for item in summaries),
        "failed": len(failures),
        "failures": failures,
    }
    dump_json(output_root / f"batch_{args.dataset}_{args.split}.json", batch_summary)
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
