#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.data.prepare_airbility_dataset \
  --config configs/data/airbility_uav_detection.yaml \
  "$@"

python -m src.data.validate_yolo_dataset \
  --data data/processed/airbility_uav_detection_yolo/data.yaml \
  --hash-leakage \
  --report data/reports/airbility_uav_detection_validation.json
