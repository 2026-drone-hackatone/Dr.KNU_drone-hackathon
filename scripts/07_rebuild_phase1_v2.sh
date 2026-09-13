#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.data.rebuild_phase1_dataset \
  --config configs/data/external_uav_phase1_v2.yaml "$@"
python -m src.data.materialize_16x9 \
  --source data/processed/external_uav_phase1_v2 \
  --output data/processed/external_uav_phase1_v2_832x480 \
  --width 832 --height 480 "$@"
python -m src.data.validate_yolo_dataset \
  --data data/processed/external_uav_phase1_v2/data.yaml \
  --hash-leakage \
  --report data/reports/external_uav_phase1_v2_validation.json
python -m src.data.validate_yolo_dataset \
  --data data/processed/external_uav_phase1_v2_832x480/data.yaml \
  --hash-leakage \
  --report data/reports/external_uav_phase1_v2_832x480_validation.json

