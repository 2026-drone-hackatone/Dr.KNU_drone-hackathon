# Phase-1 데이터 전처리 및 학습 결과 종합

작성일: 2026-09-10

## 1. 목표와 범위

Phase-1의 목표는 세 외부 데이터셋을 이용해 generic 1-class `drone` detector를
학습하고, 향후 기업 데이터가 제공됐을 때 3-class fine-tuning에 사용할 초기 weight와
재현 가능한 파이프라인을 확보하는 것이다.

최종 기업 데이터의 알려진 조건은 다음과 같다.

- 1280x720 영상
- 약 70 sequences, 27,000 frames, 40,000 bounding boxes
- median target 약 9x4 px
- terrain, cloud overlap, distractor와 moving-camera 포함
- 3개 drone class와 track ID 제공

현재 기업 데이터 파일은 없으므로 외부 데이터의 class를 임의로 3개 class에 매핑하지
않았다. 기업 해상도와 bbox 크기는 외부 데이터의 sample 선택 및 입력 해상도 판단에
사용했다.

## 2. 원본 데이터 구조

| 데이터셋 | 구조 | Annotation | 역할 |
|---|---|---|---|
| Drone Detection | 정적 이미지와 YOLO label | class 0, normalized bbox | 비교적 크고 명확한 drone appearance |
| Drone-vs-Bird | 77개 영상과 frame별 annotation | 0-based counted format | 원거리 drone, bird/sky hard-negative |
| Purdue UAV | 50개 영상과 refined MOT annotation | 1-based, track ID와 confidence | UAV-to-UAV, moving-camera, 작은 객체 |

초기 전처리에서는 영상별 sequence split, adaptive frame sampling, bbox clipping,
YOLO 변환, exact duplicate 제거를 적용했다. 초기 combined 데이터는
Drone Detection / Drone-vs-Bird / Purdue를 20 / 40 / 40 비율로 구성했다.

## 3. 초기 네 모델 결과와 문제점

초기 데이터로 YOLOv8n과 YOLOv8n-P2를 640 및 832x480에서 학습했다.

| 모델 | 입력 | 최고 val mAP50 | 최고 epoch | 고정 test AP50 재집계 |
|---|---:|---:|---:|---:|
| B0 YOLOv8n | 640x640 | 0.6392 | 34 | 약 0.4485 |
| B1 YOLOv8n | 832x480 | 0.6565 | 26 | 약 0.4486 |
| P0 YOLOv8n-P2 | 640x640 | 0.6585 | 32 | 약 0.4025 |
| P1 YOLOv8n-P2 | 832x480 | 0.6703 | 26 | 약 0.4213 |

초기 test 값은 저장된 `predictions.json`을 IoU 0.5 기준으로 재집계한 진단값이다.
모든 모델은 약 25~35 epoch 이후 val 지표가 정체됐고, P2는 val을 높였지만 test
일반화는 개선하지 못했다.

분석 결과 주요 병목은 다음과 같았다.

- Drone Detection AP50은 약 0.71 이상이지만 Purdue는 약 0.15~0.20이었다.
- Drone-vs-Bird test bbox 중앙값은 약 14x11 px로 기존 val의 약 24x18 px보다
  작았다.
- 기존 val은 test보다 쉬워 조기 종료와 모델 선택에 낙관적인 신호를 제공했다.
- 긴 영상이 frame 수와 box 수를 과도하게 지배할 수 있었다.
- negative frame은 약 4%였으며 무작위 배경 비중이 컸다.
- 832x480에서도 Purdue bbox 중앙값은 약 8~10 px 수준이었다.

## 4. Phase-1 v2 데이터 재구성

기존 test sample ID 5,557개는 그대로 고정하고 train/val만 다시 구성했다.

### 4.1 구성 기준

- source 비율을 15 / 42.5 / 42.5로 변경했다.
- Drone Detection의 공식 train/val 구조는 유지했다.
- 영상 데이터는 기존 train+val sequence만 다시 배치했다.
- 1280x720에서의 bbox 분포가 고정 test와 가까워지도록 val sequence를 선택했다.
- sequence별 positive frame은 최대 500장으로 제한했다.
- 빈 annotation이 확인된 bird/sky 및 Purdue terrain frame을 sequence당 최대 80장
  추가했다.
- blur와 low contrast는 실제 기업 환경에도 존재하므로 해당 조건만으로 제거하지
  않았다.

### 4.2 제외 기준

| 사유 | 제외 frame |
|---|---:|
| 화면 경계에서 clip된 annotation | 239 |
| 고정 test와 겹치는 개발 sequence | 167 |
| 균일하거나 디코딩 불가능한 이미지 | 16 |
| 832x480에서 모든 bbox 짧은 변이 1.5 px 미만 | 8 |
| 모든 bbox가 보간 annotation이고 작은 Purdue frame | 8 |
| 합계 | 438 |

`parrot_clear_birds_med_range`에서 고정 test의 `off_focus_parrot_birds`와 동일한
non-black frame 69개가 발견됐다. 인접 프레임 누수를 방지하기 위해 개발 후보
sequence 전체 167장을 제외했다. 최종 split 간 exact duplicate는 0건이다.

### 4.3 최종 데이터 구성

| Split | Images | Boxes | Negative | Negative 비율 | Source 구성 |
|---|---:|---:|---:|---:|---|
| train | 26,177 | 28,149 | 2,618 | 10.00% | DD 3,714 / Bird 11,232 / Purdue 11,231 |
| val | 5,020 | 6,201 | 460 | 9.16% | DD 709 / Bird 2,082 / Purdue 2,229 |
| test | 5,557 | 9,153 | 181 | 3.26% | DD 1,111 / Bird 2,223 / Purdue 2,223 |

1280x720 기준 bbox 중앙값은 train 16.25x12.50, val 18.75x15.00, test
16.25x13.75 px이다. 기업 크기 범위로 분류된 train frame은 8,985장이다.

다음 두 YOLO 데이터셋을 생성했다.

- 원본 비율: `data/processed/external_uav_phase1_v2/data.yaml`
- 고정 832x480: `data/processed/external_uav_phase1_v2_832x480/data.yaml`

두 버전 모두 image/label pair, YOLO 좌표, class ID, 이미지 손상, sequence와 원본
경로 누수, split 간 image hash를 전수 검사했으며 issue는 0건이다. 832x480 버전
36,754장의 실제 크기도 모두 832x480으로 확인했다.

재생성 명령은 다음과 같다.

```bash
bash scripts/07_rebuild_phase1_v2.sh --force
```

## 5. R1 재학습 설정

R1은 새 split의 공정한 평가를 위해 기존 B1 checkpoint가 아닌 COCO pretrained
`yolov8n.pt`에서 시작했다.

| 항목 | 값 |
|---|---|
| 모델 | YOLOv8n, stride 8/16/32 |
| 입력 | 고정 832x480, `rect=true` |
| 최대 epoch / patience | 80 / 20 |
| optimizer | AdamW |
| learning rate | `lr0=0.001`, `lrf=0.01`, cosine decay |
| augmentation | mosaic/mixup/copy-paste/erasing off |
| 기하 변환 | scale 0.2, translate 0.05, horizontal flip 0.5 |
| seed | 42, deterministic |
| GPU | NVIDIA GeForce RTX 5060 Ti 16 GB |
| 모델 크기 | 3,157,200 parameters, 약 8.63 GFLOPs |

학습 config는 `configs/train/R1_yolov8n_832x480_v2.yaml`이고 결과는
`runs/retrain/R1_yolov8n_832x480_v2/`에 저장됐다.

## 6. R1 학습 및 평가 결과

학습은 34 epoch에서 patience 20으로 종료됐으며 약 4,038초가 걸렸다.

| 항목 | Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| 최고 mAP50 row | 16 | 0.7422 | 0.5385 | 0.5480 | 0.2151 |
| 저장된 `best.pt` | 14 | 0.7342 | 0.5430 | 0.5438 | 0.2243 |
| 마지막 row | 34 | 0.7362 | 0.5043 | 0.5171 | 0.2145 |

Ultralytics 8.4.138은 fitness를 mAP50-95 기준으로 계산하므로 `best.pt`는 mAP50이
가장 높았던 epoch 16이 아니라 mAP50-95가 가장 높았던 epoch 14 checkpoint이다.
R1에서는 epoch별 checkpoint를 저장하지 않아 epoch 16 weight는 남아 있지 않다.

완료 후 저장된 R1 `best.pt`와 `last.pt`는 optimizer가 제거된 inference checkpoint다.
따라서 R1 완료 run을 optimizer 상태까지 정확히 resume할 수는 없고, weight 기반 새
fine-tuning만 가능하다.

### 6.1 고정 test 결과

`best.pt`를 고정 832x480 test 5,557장에서 평가했다.

| Precision | Recall | mAP50 | mAP50-95 | 전처리 | 추론 | 후처리 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.732 | 0.578 | 0.582 | 0.194 | 0.4 ms | 1.6 ms | 0.6 ms |

속도는 RTX 5060 Ti에서 측정한 per-image 값이며 Jetson Nano 성능으로 해석하지 않는다.

source별 AP50 진단값은 다음과 같다.

| Source | AP50 | 해석 |
|---|---:|---|
| Drone Detection | 0.7382 | Phase-1 목표 이상 |
| Drone-vs-Bird | 0.7387 | v2에서 크게 개선 |
| Purdue UAV | 0.2782 | 전체 성능의 주 병목 |
| Combined | 0.5826 | 목표 0.70까지 약 0.117 부족 |

초기 B1의 동일 test 재집계 AP50 약 0.4486과 비교하면 R1은 약 0.134 상승했다.
새 val 점수는 기존 val과 split 난이도가 달라 직접 비교하지 않는다. 고정 test 개선은
데이터 재구성과 학습 조건 변경이 일반화에 도움이 됐다는 근거다.

평가 결과와 prediction은
`runs/detect/runs/evaluation/R1_yolov8n_832x480_v2_test/`에 저장했다.

## 7. 해상도 검토와 다음 단계

기업 median bbox 9x4 px를 각 입력에 letterbox했을 때 예상 크기는 다음과 같다.

| 입력 | scale | 변환 bbox | canvas pixels |
|---|---:|---:|---:|
| 640x640 | 0.50 | 4.50x2.00 | 409,600 |
| 832x480 | 0.65 | 5.85x2.60 | 399,360 |
| 960x544 | 0.75 | 6.75x3.00 | 522,240 |
| 1280x736 | 1.00 | 9.00x4.00 | 942,080 |

640 square는 832x480과 계산 픽셀이 비슷하지만 실제 16:9 객체가 더 작고 padding이
크다. 현재 주력 입력으로는 832x480 이상이 적절하다. 960x544는 832x480보다 객체의
가로와 세로를 약 15% 키우며 canvas 연산량은 약 31% 증가한다.

R1의 두 영상 source 중 Purdue만 크게 낮기 때문에 960x544는 다음 controlled
experiment로 진행할 가치가 있다. 다만 해상도만으로 combined AP50을 0.582에서
0.70까지 높인다고 가정하지 않는다.

R2는 R1과 동일하게 COCO pretrained weight에서 시작해 해상도만 비교한다. 목표가
mAP50이므로 `save_period: 1`로 모든 epoch checkpoint를 저장하도록 설정했다.

현재 960x544 데이터와 R2 학습 결과는 아직 생성되지 않았다. 다음 명령으로 진행한다.

```bash
python -m src.data.materialize_16x9 \
  --source data/processed/external_uav_phase1_v2 \
  --output data/processed/external_uav_phase1_v2_960x544 \
  --width 960 --height 544

python -m src.data.validate_yolo_dataset \
  --data data/processed/external_uav_phase1_v2_960x544/data.yaml \
  --hash-leakage \
  --report data/reports/external_uav_phase1_v2_960x544_validation.json

python -m src.train.train_detector \
  --config configs/train/R2_yolov8n_960x544_v2.yaml
```

R2는 test를 반복 확인하지 않고 새 val에서 먼저 판단한다. R1 최고치 대비 val mAP50이
최소 0.02 상승하고 recall 및 mAP50-95가 유지되면 고정 test 평가로 진행한다. 개선이
없으면 1280 입력으로 바로 확대하기보다 Purdue annotation/sequence 오류 분석,
Purdue-aware sampling 또는 P2 960x544를 다음 후보로 검토한다.

## 8. 주요 파일

- 데이터 전처리 설정: `configs/data/external_uav_phase1_v2.yaml`
- 데이터 재구성 코드: `src/data/rebuild_phase1_dataset.py`
- 전처리 상세 보고서: `docs/phase1_v2_preprocessing.md`
- 재학습 계획: `docs/phase1_retraining_plan.md`
- 학습 결과표: `docs/experiment_log.md`
- R1 결과 CSV: `runs/retrain/R1_yolov8n_832x480_v2/results.csv`
- R1 best weight: `runs/retrain/R1_yolov8n_832x480_v2/weights/best.pt`
- R1 고정 test predictions:
  `runs/detect/runs/evaluation/R1_yolov8n_832x480_v2_test/predictions.json`

