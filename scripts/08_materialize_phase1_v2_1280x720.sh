#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.data.materialize_16x9 \
  --source data/processed/external_uav_phase1_v2 \
  --output data/processed/external_uav_phase1_v2_1280x720 \
  --width 1280 \
  --height 720 \
  "$@"

python -m src.data.validate_yolo_dataset \
  --data data/processed/external_uav_phase1_v2_1280x720/data.yaml \
  --hash-leakage \
  --report data/reports/external_uav_phase1_v2_1280x720_validation.json
