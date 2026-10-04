# Dr.KNU Drone Hackathon — UAV Detection (A0 / N1 / P1 / T0 / U1Z)

기업 제공 UAV 데이터셋(3클래스: `quad_civil`, `fixed_wing`, `target_uav`, 1280x720)에 대한 YOLO11 계열 탐지 모델 다섯 개의
학습 설정, 데이터 전처리·학습·평가 코드, 결과 기록을 정리한 저장소다.
원본 데이터와 모델 가중치는 포함하지 않으며, 아래 명령은 저장소 루트에서 실행한다.

## 1. 모델 요약

| ID | 설정 파일 | 구조 | 초기값 | 학습 데이터 | Detect stride | Params | FLOPs(B), 기록 입력 기준 | 학습 val 최고 mAP50 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| A0 | `A0_baseline` | YOLO11n stock | COCO `yolo11n.pt` | company train | 8/16/32 | 2,590,425 | 15.14 | 0.6167 |
| N1 | `N1_model1` | YOLO11n P2–P5 4-head, COCO backbone/neck 명시적 layer mapping | COCO `yolo11n.pt` | company train | 4/8/16/32 | 2,667,084 | 24.23 | 0.6179 |
| P1 | `P1_model2` | YOLO11n P2–P4 (P5 head 제거) | COCO `yolo11n.pt` | company train | 4/8/16 | 1,939,145 | 22.89 | 0.6411 |
| T0 | `T0_model3` | YOLO11s stock | COCO `yolo11s.pt` | company train | 8/16/32 | 9,428,953 | 50.21 | 0.6602 |
| U1Z | `U1Z_ours` | YOLO11n + 내부 x2 bilinear upsample | A0 best.pt (layer offset 1) | company train | 4/8/16 | 2,590,425 | 63.18 | 0.7122 |

- mAP50은 `experiments/<ID>/train/best_map50.json`의 학습 중 validation 최고값이다. 선택 epoch는 A0 11, N1 31, P1 9, T0 19, U1Z 13이다. 별도 테스트셋 성능이나 독립 재평가 결과를 뜻하지 않는다.
- Params / FLOPs / Detect stride는 각 `environment.json`의 기록이다. FLOPs 프로파일 입력은 A0·N1·P1·T0가 **736×1280**, U1Z가 **720×1280**이므로 동일 입력 크기로 측정된 표는 아니다. `profile_shape`는 연산량 측정용이며 실제 학습·평가 입력 크기를 지정하지 않는다.
- 현재 설정은 provider train으로 학습하고 provider val로 모델을 선택·평가한다. 배치 크기(T0·U1Z 4, 나머지 8), 최대 epoch(U1Z 40, 나머지 60), patience(N1 20, 나머지 15), 초기값 등이 달라 구조만 바꾼 통제 실험으로 해석하면 안 된다.
- P1은 backbone의 P5 단계를 유지하고 Detect 출력을 P2/P3/P4로 구성한다. U1Z는 입력을 내부에서 x2 확대해 입력 좌표 기준 stride가 4/8/16이 된다. N1은 명시적 layer mapping, P1은 이름·shape가 맞는 가중치 로드, U1Z는 A0 가중치의 layer index +1 이동을 사용한다.

## 2. 저장소 구성

```text
configs/
  train/        A0_baseline / N1_model1 / P1_model2 / T0_model3 / U1Z_ours .yaml
                (id = 파일 이름 = runs/detection/<id> run 디렉터리 이름)
  model/        yolo11n_mapped_p2.yaml (N1), yolo11n_p2_p4.yaml (P1), yolo11n_up2.yaml (U1Z)
  data/         airbility_uav_detection.yaml (기업 데이터 전처리, provider split 보존)
src/
  utils/        project.py (저장소 기준 경로·YAML·JSON 유틸리티)
  data/         prepare_airbility_dataset.py (6열 라벨 → 5열 YOLO, split 보존), validate_yolo_dataset.py
  train/        train_detector.py (재현 가능한 학습, mAP50 기준 best.pt), average_checkpoints.py (epoch 평균)
  evaluation/   evaluate_company.py (동일 조건 독립 평가 + JSON)
  tests/        test_evaluate_company.py
scripts/        01_prepare_airbility_dataset.sh
experiments/<ID>/train/
  best_map50.json          학습 중 val mAP50 최고값과 epoch
  environment.json         패키지 버전, GPU, params, GFLOPs, detect stride
  experiment_config.yaml   실행 당시 학습 설정
  args.yaml                Ultralytics 최종 인자
```

`runs/`, `data/processed/`, `*.pt`는 `.gitignore`로 제외된다. `experiments/`는 실제 학습 run의 기록 파일을 그대로 복사한 것이라
그 안의 `id`/`name`은 실행 당시 로컬 run 이름(A0는 `A4_...`, U1Z는 `U1_...`)으로 남아 있다. 이 저장소의 설정으로 새로 학습하면 `runs/detection/<ID>`에 같은 형식의 파일이 생성된다.

## 3. 환경

아래는 저장된 실험의 환경이다. `requirements.txt`는 Ultralytics를 고정하지만 PyTorch/CUDA와 나머지 패키지의 정확한 버전까지 고정하지는 않는다. GPU 학습에는 사용할 장치에 맞는 PyTorch/CUDA 환경이 필요하다.

```text
Python 3.10.21 / PyTorch 2.14.0+cu130 / Ultralytics 8.4.138 / RTX 5060 Ti 16 GB
```

```bash
python -m pip install -r requirements.txt
```

## 4. 데이터 준비

기업 원본을 `data/airbility_uav_detection_dataset/{images,labels}/{train,val}/<sequence>/`에 둔다 (라벨은 6열 `cls cx cy w h track_id`).

```bash
# provider split 보존 (다섯 실험 공통: train = 기업 train, val = 기업 val)
bash scripts/01_prepare_airbility_dataset.sh
#   -> data/processed/airbility_uav_detection_yolo/data.yaml + data/reports/*.json
```

전처리는 이미지 크기(1280×720), 라벨과 클래스 범위를 확인하고 `track_id`를 제거한 5열 라벨을 생성한다. 추적 ID는 `metadata.jsonl`에 보존한다. 이미지는 기본적으로 hardlink하고, 실패하면 복사한다. 파생 이미지의 직접 수정은 원본에도 영향을 줄 수 있으므로 별도 복사가 필요한 경우 데이터 설정의 `link_mode: copy`를 사용한다.

스크립트는 전처리 후 split 간 이미지 해시 중복도 검사한다. 결과 폴더에 파일이 이미 있으면 중단하며, `--force`는 설정된 출력 폴더를 삭제하고 다시 만든다. 원본 데이터 폴더는 `.gitignore`에 별도 제외 규칙이 없으므로 커밋 대상에 추가하지 않는다.

## 5. 학습

run 디렉터리 `runs/detection/<ID>`가 이미 있으면 launcher가 중단한다. 새 run은 `--name <NEW_ID>`로 지정한다.
N1·P1을 바로 실행하려면 COCO `yolo11n.pt`가 저장소 루트에 있어야 한다(먼저 A0를 실행하면 Ultralytics가 다운로드한다).

```bash
python -m src.train.train_detector --config configs/train/A0_baseline.yaml
python -m src.train.train_detector --config configs/train/N1_model1.yaml
python -m src.train.train_detector --config configs/train/P1_model2.yaml
python -m src.train.train_detector --config configs/train/T0_model3.yaml

# U1Z_ours는 runs/detection/A0_baseline/weights/best.pt 를 초기값으로 쓰므로 A0 학습 후 실행
# (40 epoch cosine, patience 15, val mAP50 기준 best.pt)
python -m src.train.train_detector --config configs/train/U1Z_ours.yaml
```

`train_detector.py`는 validation mAP50 최고 checkpoint를 `weights/best_map50.pt`에 저장하고 `best.pt`로 복사한다. 각 run에는 `experiment_config.yaml`, `environment.json`, `command.txt`가 함께 기록된다.

중단된 학습은 optimizer 상태가 남아 있는 `last.pt`로 이어서 실행한다. 정상 종료 후 optimizer가 제거된 체크포인트는 정확한 resume에 사용할 수 없다.

```bash
python -m src.train.train_detector --resume runs/detection/A0_baseline/weights/last.pt
```

## 6. 평가

```bash
RUN_ID=U1Z_ours  # 평가할 설정 ID로 변경
python -m src.evaluation.evaluate_company \
  --model "runs/detection/${RUN_ID}/weights/best.pt" \
  --data data/processed/airbility_uav_detection_yolo/data.yaml \
  --split val --imgsz 1280 --batch 4 --device 0 --workers 8 \
  --conf 0.001 --iou 0.7 --max-det 300 --plots --save-json \
  --output "runs/evaluation/${RUN_ID}_company_val_1280/metrics.json"
```

평가는 `rect=True`, TTA off(기본값), 3클래스 구분 조건으로 수행하며 전체·클래스별 precision, recall, mAP50, mAP50-95와 처리 시간 등을 JSON에 기록한다. 같은 출력 디렉터리가 있으면 중단하므로 새 경로를 지정하거나 의도적으로 재사용할 때 `--exist-ok`를 추가한다. 현재 저장소에는 독립 평가 `metrics.json`과 `best.pt`가 포함되어 있지 않다.

## 7. 간단한 코드 확인과 범위

```bash
python -m unittest discover -s src/tests -v
```

현재 테스트는 평가 함수 import 및 호출 가능 여부만 확인한다. 데이터 전처리 전체, GPU 학습, 탐지 정확도를 검증하는 테스트는 아니다. 제공된 다섯 설정에 해당하는 탐지 파이프라인이 문서의 대상이며, 추적기·실시간 카메라·Jetson 배포 코드는 포함하지 않는다. 학습 코드에 남은 SPD/NWD 선택 분기는 필요한 모듈이 이 저장소에 없으므로 제공 설정 밖에서 해당 옵션을 켜는 방식은 지원하지 않는다.
