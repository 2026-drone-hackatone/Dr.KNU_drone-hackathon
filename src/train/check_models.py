from __future__ import annotations

import argparse
import json

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_gradients, get_num_params

from src.utils.project import dump_json, resolve_path


def describe(model: YOLO) -> dict:
    core = model.model
    layers = sum(1 for module in core.modules() if not list(module.children()))
    parameters = get_num_params(core)
    trainable = get_num_gradients(core)
    return {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "detect_strides": [int(value) for value in core.stride.tolist()],
        "layers": layers,
        "gflops_640x640": get_flops(core, imgsz=640),
        "gflops_832x480": get_flops(core, imgsz=[480, 832]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build both Phase-1 architectures and verify detection strides.")
    parser.add_argument("--baseline", default="configs/model/yolov8n.yaml")
    parser.add_argument("--p2", default="configs/model/yolov8n_p2.yaml")
    parser.add_argument("--output", default="data/reports/model_preflight.json")
    args = parser.parse_args()
    baseline = YOLO(str(resolve_path(args.baseline)))
    p2 = YOLO(str(resolve_path(args.p2)))
    result = {"yolov8n": describe(baseline), "yolov8n_p2": describe(p2)}
    if result["yolov8n"]["detect_strides"] != [8, 16, 32]:
        raise RuntimeError(result)
    if result["yolov8n_p2"]["detect_strides"] != [4, 8, 16, 32]:
        raise RuntimeError(result)
    dump_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
