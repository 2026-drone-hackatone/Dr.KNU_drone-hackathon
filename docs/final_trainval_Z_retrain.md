# 최종 제출용 Z 재학습 (기업 train + val 전체 학습)

작성: 2026-09-22. 대상: U1, U2, U2avg 세 모델의 최종 제출 버전(U1Z, U2Z, U2avgZ).
기존 run(U1, U2, U2avg 등)은 건드리지 않는다.

## 1. 왜 다시 학습하는가

지금까지의 모든 run은 기업 제공 split을 그대로 써서 `images/train`(55 시퀀스, 19,655장)으로 학습하고
`images/val`(14 시퀀스, 5,625장)로 checkpoint 선택과 독립 평가를 했다.
대회 당일에는 별도의 test 셋이 주어지므로, 최종 제출 모델은 **train + val 전체(69 시퀀스, 25,280장)**로
학습한다. 대신 checkpoint 선택에 쓸 깨끗한 validation이 없어지므로, **어느 epoch을 쓸지는
기존 split 실험 결과로 미리 고정**하고 재학습에서는 그 epoch을 그대로 쓴다.

## 2. 데이터셋 (이미 생성됨)

- config: `configs/data/airbility_uav_detection_trainval.yaml` (`provider_val_destination: train_and_val`)
- 출력: `data/processed/airbility_uav_detection_trainval_yolo/` (hardlink, 추가 디스크 거의 없음)
- train = 기업 train 55 + 기업 val 14 = 69 시퀀스, 25,280장 / val = 기업 val 14 시퀀스, 5,625장(train과 중복)
- 따라서 학습 중 찍히는 val mAP, `best.pt`, `best_map50.json`은 **in-sample 값**이다. 모델 선택에 쓰지 않는다.
- 재생성이 필요하면:

```bash
cd /home/user/workspace/drone_comp
/home/user/miniconda3/envs/drone/bin/python -m src.data.prepare_airbility_dataset \
    --config configs/data/airbility_uav_detection_trainval.yaml --force
```

## 3. 학습 recipe와 checkpoint 정책

`src/train/train_detector.py`에 `stop_after_epoch`(1-based, results.csv 기준)을 추가했다.
설정한 `epochs`의 cosine LR 스케줄은 그대로 두고 지정 epoch이 끝나면 EarlyStopping과 같은 경로로
정상 종료한다(last.pt/best.pt strip, final_eval 수행). 320px 소형 run으로 동작 확인함.

| 항목 | U1Z | U2Z / U2avgZ |
|---|---|---|
| config | `configs/train/U1Z_yolo11n_up2_company_trainval_1280x720.yaml` | `configs/train/U2Z_yolo11s_up2_from_T0_company_trainval_1280x720.yaml` |
| 원본 대비 변경 | data, `patience: 0`, `stop_after_epoch: 32` | data, `patience: 0`, `stop_after_epoch: 4` |
| LR 스케줄 | 원본과 동일 (cos, epochs 40) | 원본과 동일 (cos, epochs 20) |
| 초기 가중치 | A0 best.pt (기존 그대로) | T0 best.pt (기존 그대로) |
| 원본 split 근거 | best ep19 0.6957, avg18-20 0.6901, avg18-22 0.6903, avg19-32 0.6845 | best ep2 0.7062, avg1-3(U2avg) 0.7132, 이후 평균 모두 0.69 미만 |
| 최종 checkpoint | `weights_avg/avg_ep18-22.pt` (+ 단일 ep19 `ep19.pt` 보조) | U2avgZ = `weights_avg/avg_ep1-3.pt`, U2Z = `weights_avg/ep2.pt` |
| 예상 시간 | 32 epoch × 약 21분 ≈ 11시간 | 4 epoch × 약 46분 ≈ 3시간 |

주의: 학습 데이터가 29% 늘어 epoch당 step 수도 29% 늘어난다. "같은 epoch"은 LR 위치가 같다는 뜻이고
누적 update 수는 더 많다. 이것은 train+val 재학습의 표준 방식이며 별도 보정은 하지 않는다.

U1Z에서 단일 epoch(ep19)이 아니라 평균(18-22)을 기본으로 두는 이유: split 실험에서 ep19는 이웃
epoch(0.677~0.687)보다 눈에 띄게 높은 단발 피크였고, val 없이 재학습하면 같은 위치에 피크가 재현된다는
보장이 없다. 평균은 단일 epoch보다 재현성이 높고 손실은 -0.0054 수준이며, 18-20과 18-22는 사실상 같아(0.6901 vs 0.6903) 더 넓은 창을 택했다. `stop_after_epoch`를 22로 낮추면 약 3.5시간을 아끼지만 이후 epoch 평균으로 바꿀 여지가 없어져 원본과 같은 32를 유지했다.

## 4. 실행 명령 (모두 포그라운드)

### 4.1 U1Z 학습 (약 11시간)

```bash
cd /home/user/workspace/drone_comp
/home/user/miniconda3/envs/drone/bin/python -m src.train.train_detector \
    --config configs/train/U1Z_yolo11n_up2_company_trainval_1280x720.yaml \
    --device 0 --workers 8
```

OOM이면 `--batch 2`로 재시작하고 config의 `freeze_bn`을 `true`로 바꾼다(원본 U1 주석과 동일).

### 4.2 U1Z 최종 checkpoint 생성

```bash
cd /home/user/workspace/drone_comp
/home/user/miniconda3/envs/drone/bin/python -m src.train.average_checkpoints \
    --run runs/detection/U1Z_yolo11n_up2_company_trainval_1280x720 \
    --epochs 18-22 \
    --output runs/detection/U1Z_yolo11n_up2_company_trainval_1280x720/weights_avg/avg_ep18-22.pt

/home/user/miniconda3/envs/drone/bin/python -m src.train.average_checkpoints \
    --run runs/detection/U1Z_yolo11n_up2_company_trainval_1280x720 \
    --epochs 19 \
    --output runs/detection/U1Z_yolo11n_up2_company_trainval_1280x720/weights_avg/ep19.pt
```

### 4.3 U2Z 학습 (약 3시간, epoch 4 후 자동 종료)

```bash
cd /home/user/workspace/drone_comp
/home/user/miniconda3/envs/drone/bin/python -m src.train.train_detector \
    --config configs/train/U2Z_yolo11s_up2_from_T0_company_trainval_1280x720.yaml \
    --device 0 --workers 8
```

### 4.4 U2avgZ / U2Z 최종 checkpoint 생성

```bash
cd /home/user/workspace/drone_comp
/home/user/miniconda3/envs/drone/bin/python -m src.train.average_checkpoints \
    --run runs/detection/U2Z_yolo11s_up2_from_T0_company_trainval_1280x720 \
    --epochs 1-3 \
    --output runs/detection/U2Z_yolo11s_up2_from_T0_company_trainval_1280x720/weights_avg/avg_ep1-3.pt

/home/user/miniconda3/envs/drone/bin/python -m src.train.average_checkpoints \
    --run runs/detection/U2Z_yolo11s_up2_from_T0_company_trainval_1280x720 \
    --epochs 2 \
    --output runs/detection/U2Z_yolo11s_up2_from_T0_company_trainval_1280x720/weights_avg/ep2.pt
```

`--epochs 2`처럼 epoch 하나만 주면 EMA 가중치를 optimizer 없이 깨끗한 stock DetectionModel 형식으로
저장하므로 eval.py 제약(사용자 모듈 없이 unpickle)을 그대로 만족한다.

### 4.5 sanity check (in-sample, 회귀 확인용)

train에 포함된 val이므로 값 자체는 성능 지표가 아니다. 로딩 가능 여부와 기존 모델보다 **낮지 않은지**만 본다.

```bash
cd /home/user/workspace/drone_comp
for M in U1Z_yolo11n_up2_company_trainval_1280x720/weights_avg/avg_ep18-22 \
         U2Z_yolo11s_up2_from_T0_company_trainval_1280x720/weights_avg/avg_ep1-3 \
         U2Z_yolo11s_up2_from_T0_company_trainval_1280x720/weights_avg/ep2; do
  NAME=$(echo "$M" | cut -d_ -f1)_$(basename "$M")
  /home/user/miniconda3/envs/drone/bin/python -m src.evaluation.evaluate_company \
      --model runs/detection/$M.pt \
      --data data/processed/airbility_uav_detection_yolo/data.yaml --split val \
      --imgsz 1280 --batch 4 --device 0 \
      --output runs/evaluation/${NAME}_company_val_insample_1280/metrics.json
done
```

## 5. 제출 파일 매핑

| 제출 이름 | 파일 | `scripts/airbility_workflow.py` 키 |
|---|---|---|
| U1Z | `runs/detection/U1Z_yolo11n_up2_company_trainval_1280x720/weights_avg/avg_ep18-22.pt` | `U1Z` |
| U2Z | `runs/detection/U2Z_yolo11s_up2_from_T0_company_trainval_1280x720/weights_avg/ep2.pt` | `U2Z` |
| U2avgZ | `runs/detection/U2Z_yolo11s_up2_from_T0_company_trainval_1280x720/weights_avg/avg_ep1-3.pt` | `U2avgZ` |

T0, A0 초기 가중치는 기업 train만으로 학습된 기존 checkpoint를 그대로 쓴다(재학습 범위 밖).
Z 모델의 진짜 성능은 대회 test GT가 나올 때만 확인할 수 있으며, 기존 split 수치(U2avg 0.7132, U1 0.6957)를
하한 기대치로 본다.
