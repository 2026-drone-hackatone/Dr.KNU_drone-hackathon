#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data.validate_yolo_dataset \
  --data data/processed/external_uav_combined/data.yaml \
  --hash-leakage \
  --visualize 100
python -m src.data.validate_yolo_dataset \
  --data data/processed/external_uav_combined_832x480/data.yaml \
  --report data/reports/external_uav_combined_832x480_validation.json

