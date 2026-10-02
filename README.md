# Dr.KNU Drone Hackathon — UAV Detection (A0 / N1 / P1 / T0 / U1Z)

기업 제공 UAV 데이터셋(3클래스: `quad_civil`, `fixed_wing`, `target_uav`, 1280x720)에 대한 YOLO11 계열 탐지 모델 다섯 개의
학습 설정, 학습 코드, 결과 기록을 정리한 저장소다. 가중치(`*.pt`)와 데이터는 커밋하지 않는다.

## 1. 모델 요약

| ID | 구조 | 초기값 | 학습 데이터 | Detect stride | Params | FLOPs(B) @736x1280 | 학습 로그 mAP50 (epoch) | 독립 평가 mAP50 |
|---|---|---|---|---|---:|---:|---:|---:|
| A0 | YOLO11n stock | COCO `yolo11n.pt` | company train | 8/16/32 | 2,590,425 | 15.14 | 0.6180 (11) | 0.6190 |
| N1 | YOLO11n P2–P5 4-head, COCO backbone/neck 명시적 layer mapping | COCO `yolo11n.pt` | company train | 4/8/16/32 | 2,667,084 | 24.23 | 0.6241 (31) | – |
| P1 | YOLO11n P2–P4 (P5 head 제거) | COCO `yolo11n.pt` | company train | 4/8/16 | 1,939,145 | 22.89 | 0.6477 (9) | 0.6491 |
| T0 | YOLO11s stock | COCO `yolo11s.pt` | company train | 8/16/32 | 9,428,953 | 50.21 | 0.6637 (19) | 0.6629* |
| U1Z | YOLO11n + 내부 x2 bilinear upsample | A0 best.pt (layer offset 1) | company train **+ val** | 4/8/16 | 2,590,425 | 64.70 | 0.7652 (29, in-sample) | – |

- Params / FLOPs는 `best.pt`를 Ultralytics `get_flops`(thop)로 직접 측정한 값이다. 736x1280은 `imgsz=1280, rect=True`에서 1280x720 입력이 stride 32 배수로 패딩된 실제 추론 크기다. 640 기준 A0는 6.50 GFLOPs로 Ultralytics 공식 YOLO11n 표와 일치한다.
- 학습 로그 mAP50은 각 run의 `best_map50.json`(EMA, 학습 중 val). 독립 평가는 `src/evaluation/evaluate_company.py`(imgsz 1280, rect, conf 0.001, iou 0.7, max_det 300, TTA off) JSON이며 A0, P1만 보존돼 있다. T0의 *0.6629는 다른 머신에서 기록된 값으로 JSON이 없다.
- **U1Z는 기업 val을 학습에 포함한 최종 제출용 재학습이다.** 학습 중 val 점수(0.7652)는 in-sample이므로 다른 행과 비교하지 않는다. 같은 구조를 provider split으로 학습한 U1의 독립 평가는 0.6957이다(`docs/experiment_summary_2026-09-21.md`). 제출 체크포인트는 epoch 18–22 평균(`weights_avg/avg_ep18-22.pt`)이며 생성 기록은 `experiments/U1Z_.../weights_avg/avg_ep18-22.json`에 있다.

클래스별 독립 평가 AP50:

| ID | quad_civil | fixed_wing | target_uav | precision | recall |
|---|---:|---:|---:|---:|---:|
| A0 | 0.4253 | 0.9312 | 0.5006 | 0.8228 | 0.6028 |
| P1 | 0.4670 | 0.9377 | 0.5424 | 0.8161 | 0.6177 |

## 2. 저장소 구성

```text
configs/
  train/        다섯 실험의 학습 설정 (id, 구조, 초기값, 데이터, 하이퍼파라미터)
  model/        yolo11n_mapped_p2.yaml (N1), yolo11n_p2_p4.yaml (P1), yolo11n_up2.yaml (U1Z)
  data/         기업 데이터 전처리 설정 (provider split / train+val 합산)
src/
  data/         prepare_airbility_dataset.py (6열 라벨 → 5열 YOLO, split 보존), validate_yolo_dataset.py
  train/        train_detector.py (재현 가능한 학습, mAP50 기준 best.pt), average_checkpoints.py (epoch 평균)
  evaluation/   evaluate_company.py (동일 조건 독립 평가 + JSON)
  tests/        test_evaluate_company.py
scripts/        09_prepare_airbility_dataset.sh
experiments/<RUN_ID>/
  train/        best_map50.json, results.csv, environment.json, experiment_config.yaml, args.yaml, results.png
  eval_company_val_1280/   독립 평가 metrics.json, PR curve, confusion matrix (A0, P1)
  weights_avg/  U1Z epoch 평균 기록
docs/           실험 전체 요약(2026-09-21), 전체 실험 결과 정리(2026-09-30), Z 재학습 설계
```

`runs/`, `data/processed/`, `*.pt`는 `.gitignore`로 제외된다. 학습 산출물의 요약만 `experiments/`에 복사했다.

## 3. 환경

```text
Python 3.10.21 / PyTorch 2.14.0+cu130 / Ultralytics 8.4.138 / RTX 5060 Ti 16 GB
```

```bash
pip install -r requirements.txt
```

## 4. 데이터 준비

기업 원본을 `data/airbility_uav_detection_dataset/{images,labels}/{train,val}/<sequence>/`에 둔다 (라벨은 6열 `cls cx cy w h track_id`).

```bash
# provider split 보존 (A0, N1, P1, T0 용)
bash scripts/09_prepare_airbility_dataset.sh
#   -> data/processed/airbility_uav_detection_yolo/data.yaml + data/reports/*.json

# train + val 합산 (U1Z 용, val 폴더는 in-sample 모니터링용 복사본)
python -m src.data.prepare_airbility_dataset --config configs/data/airbility_uav_detection_trainval.yaml
```

## 5. 학습

run 디렉터리 `runs/detection/<id>`가 이미 있으면 trainer가 중단한다. 재실행은 `--name <NEW_ID>`, 이어서 학습은 `--resume runs/detection/<id>/weights/last.pt`.

```bash
python -m src.train.train_detector --config configs/train/A0_yolo11n_coco_airbility_provider_split_1280x720.yaml
python -m src.train.train_detector --config configs/train/N1_yolo11n_mapped_p2_company.yaml
python -m src.train.train_detector --config configs/train/P1_yolo11n_p2_p4_company_1280x720.yaml
python -m src.train.train_detector --config configs/train/T0_yolo11s_coco_airbility_provider_split_1280x720.yaml

# U1Z는 A0 best.pt를 초기값으로 쓰므로 A0 학습 후 실행 (stop_after_epoch: 32 로 horizon 고정)
python -m src.train.train_detector --config configs/train/U1Z_yolo11n_up2_company_trainval_1280x720.yaml
python -m src.train.average_checkpoints \
  --run runs/detection/U1Z_yolo11n_up2_company_trainval_1280x720 --epochs 18-22 \
  --output runs/detection/U1Z_yolo11n_up2_company_trainval_1280x720/weights_avg/avg_ep18-22.pt
```

`train_detector.py`는 validation mAP50 최고 checkpoint를 `weights/best_map50.pt`에 저장하고 `best.pt`로 복사한다. 각 run에는 `experiment_config.yaml`, `environment.json`(패키지 버전, GPU, params, GFLOPs, stride), `command.txt`가 함께 기록된다.

## 6. 독립 평가

```bash
python -m src.evaluation.evaluate_company \
  --model runs/detection/<RUN_ID>/weights/best.pt \
  --data data/processed/airbility_uav_detection_yolo/data.yaml \
  --split val --imgsz 1280 --batch 4 --device 0 --workers 8 \
  --conf 0.001 --iou 0.7 --max-det 300 --plots --save-json \
  --output runs/evaluation/<RUN_ID>_company_val_1280/metrics.json
```

## 7. 가중치

`best.pt`는 저장소에 포함하지 않는다. 공유 위치(Google Drive 등)의 링크를 아래에 기록한다.

| ID | checkpoint | SHA256 | 링크 |
|---|---|---|---|
| U1Z | `weights_avg/avg_ep18-22.pt` | `ed2107d48f3842974db8c41a2ad02cc1c9d8b6c62575a31a1e79af0af0c3f0b5` | (추가 예정) |
| U1Z | `weights/best.pt` | `718cef051d8338036cdda3ee3caa78bcfa7cef43e0bccb5900fc55bfdb92cc2a` | (추가 예정) |
