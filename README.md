# Dr.KNU Drone Hackathon — UAV Detection (A0 / N1 / P1 / T0 / U1Z)

기업 제공 UAV 데이터셋(3클래스: `quad_civil`, `fixed_wing`, `target_uav`, 1280x720)에 대한 YOLO11 계열 탐지 모델 다섯 개의
학습 설정, 학습 코드, 결과 기록을 정리한 저장소다. 

## 1. 모델 요약

| ID  | 설정 파일       | 구조                                                           | 초기값                      | 학습 데이터   | Detect stride |    Params | FLOPs(B) @736x1280 | mAP50  |
| --- | --------------- | -------------------------------------------------------------- | --------------------------- | ------------- | ------------- | --------: | -----------------: | ----------------------: | ---------------: |
| A0  | `A0_baseline` | YOLO11n stock                                                  | COCO`yolo11n.pt`          | company train | 8/16/32       | 2,590,425 |              15.14 |            0.6167 |
| N1  | `N1_model1`   | YOLO11n P2–P5 4-head, COCO backbone/neck 명시적 layer mapping | COCO`yolo11n.pt`          | company train | 4/8/16/32     | 2,667,084 |              24.23 |            0.6179 |
| P1  | `P1_model2`   | YOLO11n P2–P4 (P5 head 제거)                                  | COCO`yolo11n.pt`          | company train | 4/8/16        | 1,939,145 |              22.89 |             0.6411|
| T0  | `T0_model3`   | YOLO11s stock                                                  | COCO`yolo11s.pt`          | company train | 8/16/32       | 9,428,953 |              50.21 |            0.6602 |
| U1Z | `U1Z_ours`    | YOLO11n + 내부 x2 bilinear upsample                            | A0 best.pt (layer offset 1) | company train | 4/8/16        | 2,590,425 |              64.70 |            0.7122 | 

- 다섯 모델 모두 같은 provider split(기업 train 55 시퀀스 학습, 기업 val 14 시퀀스로 선택·평가)을 사용하므로 직접 비교할 수 있다.
- Params / FLOPs는 `best.pt`를 Ultralytics `get_flops`(thop)로 직접 측정한 값이다. 736x1280은 `imgsz=1280, rect=True`에서 1280x720 입력이 stride 32 배수로 패딩된 실제 추론 크기다. 640 기준 A0는 6.50 GFLOPs로 Ultralytics 공식 YOLO11n 표와 일치한다.
- 학습 로그 mAP50은 각 run의 `best_map50.json`(EMA, 학습 중 val). 독립 평가는 `src/evaluation/evaluate_company.py`(imgsz 1280, rect, conf 0.001, iou 0.7, max_det 300, TTA off)로 `best.pt`를 다시 평가한 값이다. T0의 *0.6629는 다른 머신에서 기록된 값이다.

## 2. 저장소 구성

```text
configs/
  train/        A0_baseline / N1_model1 / P1_model2 / T0_model3 / U1Z_ours .yaml
                (id = 파일 이름 = runs/detection/<id> run 디렉터리 이름)
  model/        yolo11n_mapped_p2.yaml (N1), yolo11n_p2_p4.yaml (P1), yolo11n_up2.yaml (U1Z)
  data/         airbility_uav_detection.yaml (기업 데이터 전처리, provider split 보존)
src/
  data/         prepare_airbility_dataset.py (6열 라벨 → 5열 YOLO, split 보존), validate_yolo_dataset.py
  train/        train_detector.py (재현 가능한 학습, mAP50 기준 best.pt), average_checkpoints.py (epoch 평균)
  evaluation/   evaluate_company.py (동일 조건 독립 평가 + JSON)
  tests/        test_evaluate_company.py
scripts/        09_prepare_airbility_dataset.sh
experiments/<ID>/train/
  best_map50.json          학습 중 val mAP50 최고값과 epoch
  environment.json         패키지 버전, GPU, params, GFLOPs, detect stride
  experiment_config.yaml   실행 당시 학습 설정
  args.yaml                Ultralytics 최종 인자
```

`runs/`, `data/processed/`, `*.pt`는 `.gitignore`로 제외된다. `experiments/`는 실제 학습 run의 기록 파일을 그대로 복사한 것이라
그 안의 `id`/`name`은 실행 당시 로컬 run 이름(A0는 `A4_...`, U1Z는 `U1_...`)으로 남아 있다. 이 저장소의 설정으로 새로 학습하면 `runs/detection/<ID>`에 같은 형식의 파일이 생성된다.

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
# provider split 보존 (다섯 실험 공통: train = 기업 train, val = 기업 val)
bash scripts/01_prepare_airbility_dataset.sh
#   -> data/processed/airbility_uav_detection_yolo/data.yaml + data/reports/*.json
```

## 5. 학습

run 디렉터리 `runs/detection/<ID>`가 이미 있으면 trainer가 중단한다. 재실행은 `--name <NEW_ID>`, 이어서 학습은 `--resume runs/detection/<ID>/weights/last.pt`.

```bash
python -m src.train.train_detector --config configs/train/A0_baseline.yaml
python -m src.train.train_detector --config configs/train/N1_model1.yaml
python -m src.train.train_detector --config configs/train/P1_model2.yaml
python -m src.train.train_detector --config configs/train/T0_model3.yaml

# U1Z_ours는 runs/detection/A0_baseline/weights/best.pt 를 초기값으로 쓰므로 A0 학습 후 실행
# (40 epoch cosine, patience 15, val mAP50 기준 best.pt)
python -m src.train.train_detector --config configs/train/U1Z_ours.yaml

# (선택) epoch 평균 체크포인트. U1Z에서는 best.pt보다 낮았으므로 참고용
python -m src.train.average_checkpoints \
  --run runs/detection/U1Z_ours --epochs 18-22 \
  --output runs/detection/U1Z_ours/weights_avg/avg_ep18-22.pt
```

`train_detector.py`는 validation mAP50 최고 checkpoint를 `weights/best_map50.pt`에 저장하고 `best.pt`로 복사한다. 각 run에는 `experiment_config.yaml`, `environment.json`, `command.txt`가 함께 기록된다.

### 6. 평가

```bash
python -m src.evaluation.evaluate_company \
  --model runs/detection/<ID>/weights/best.pt \
  --data data/processed/airbility_uav_detection_yolo/data.yaml \
  --split val --imgsz 1280 --batch 4 --device 0 --workers 8 \
  --conf 0.001 --iou 0.7 --max-det 300 --plots --save-json \
  --output runs/evaluation/<ID>_company_val_1280/metrics.json
```
