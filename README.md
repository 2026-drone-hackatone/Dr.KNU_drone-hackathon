
# Dr.KNU Drone Hackathon — Real-Time UAV Detection & Tracking on Edge AI 🏆

> **2026 Drone AI Hackathon — Grand Prize (대상)**
> 기업이 제시한 **소형 드론 탐지 성능(mAP@0.5 ≥ 0.70)** 및 **실시간 추론 속도(≥ 15 FPS)** 목표를 달성하고,
> **Teacher–Student Knowledge Distillation, Multi-Object Tracking, TensorRT 최적화, NVIDIA Jetson Orin Nano Super 실시간 배포, Sim-to-Real 검증, 위험도 분류**까지 구현한 프로젝트입니다.

---

## 1. Project Overview

본 프로젝트의 목표는 단순한 UAV 객체 탐지를 넘어, **실제 환경에서 동작 가능한 실시간 Edge AI 기반 드론 탐지·추적 시스템**을 구축하는 것이었습니다.

기업이 제공한 UAV 데이터셋을 기반으로 YOLO11 계열 모델의 구조와 학습 전략을 비교하였으며, 소형 객체 탐지 성능을 개선하기 위해 P2 feature 활용, P5 head 제거, 내부 upsampling 등 다양한 구조를 검증했습니다. 또한 Teacher–Student Knowledge Distillation을 활용해 경량 모델의 성능을 보완하고, Tracking과 위험도 분류를 결합해 프레임 단위 탐지를 실제 상황 판단이 가능한 시스템으로 확장했습니다.

최종적으로 모델을 **TensorRT 기반으로 최적화하여 NVIDIA Jetson Orin Nano Super에 배포**하고, 카메라 모듈을 연결해 실제 비행 중인 드론을 실시간으로 탐지·추적하는 시연을 수행했습니다.

---

## 2. Challenge Tasks

대회에서는 기업이 제시한 두 가지 핵심 과제를 수행했습니다.

### Task 1 — Detection Performance & Real-Time Inference

- 기업 제공 UAV 데이터셋 기반 3-class detection
  - `quad_civil`
  - `fixed_wing`
  - `target_uav`
- 입력 해상도: `1280 × 720`
- 목표 탐지 성능: **mAP@0.5 ≥ 0.70**
- 목표 추론 속도: **≥ 15 FPS**
- 경량 모델 기반의 정확도–연산량 trade-off 최적화

### Task 2 — Sim-to-Real UAV Tracking

- 실제 드론 시연장 환경을 모사한 simulation data 구성
- simulation 환경에서 학습한 모델을 실제 시연장에 적용
- 실제 비행 UAV에 대한 detection 및 tracking 검증
- synthetic/simulation 환경과 실제 환경 간 **Sim-to-Real domain gap 완화**

---

## 3. Key Achievements

- 🏆 **2026 Drone AI Hackathon Grand Prize (대상)**
- 기업 목표 **mAP@0.5 ≥ 0.70 달성**
- 기업 목표 **Jetson 기준 실시간 추론 속도 ≥ 15 FPS 달성**
- Baseline YOLO11n 대비 최종 U1Z 모델의 mAP50을 **0.6167 → 0.7122 (+9.55%p)**로 개선
- Teacher–Student **Knowledge Distillation** 기반 경량 모델 성능 개선
- **Multi-Object Tracking**을 통한 드론 ID 유지 및 이동 추적
- Tracking 정보를 활용한 **위험도 분류/판단 모듈** 구현
- **TensorRT 기반 inference optimization** 수행
- **NVIDIA Jetson Orin Nano Super + Camera** 기반 실시간 UAV detection/tracking 시연
- 실제 FPS / Latency를 측정하여 Edge 환경에서의 실시간성 검증
- 실제 시연장 환경을 모사한 simulation data 기반 **Sim-to-Real 검증**
- 코드, 학습 설정, 평가 환경을 GitHub에 정리하여 **재현성(Reproducibility)** 확보

---

## 4. End-to-End System Pipeline

```text
Simulation / Company UAV Dataset
              │
              ▼
      Dataset Preprocessing
              │
              ▼
     YOLO11 Baseline Training
              │
              ├───────────────┐
              ▼               ▼
  Small-Object Architecture   Teacher Model
  Optimization                │
  - P2 feature                ▼
  - P5 removal          Knowledge Distillation
  - Internal x2 upsample      │
              └───────┬───────┘
                      ▼
               Student Detector
                      │
                      ▼
            Multi-Object Tracking
                      │
                      ▼
              Threat Assessment
                      │
                      ▼
                ONNX / TensorRT
                      │
                      ▼
        NVIDIA Jetson Orin Nano Super
                      │
                      ▼
                Camera Input
                      │
                      ▼
       Real-Time UAV Detection & Tracking
```

---

## 5. Technical Highlights

### 5.1 Small-Object Detection Optimization

기업 데이터에서 다수의 UAV가 매우 작은 픽셀 영역으로 관측되는 문제를 고려해, 기본 YOLO11n 외에 소형 객체에 적합한 구조를 비교했습니다.

- **A0** — YOLO11n baseline
- **N1** — P2–P5 4-head 구조
- **P1** — P2–P4 구조, P5 detection head 제거
- **T0** — YOLO11s
- **U1Z** — YOLO11n + internal ×2 bilinear upsampling

| ID            | 구조                                      |              Params | FLOPs(B) @736×1280 |            mAP50 |
| ------------- | ----------------------------------------- | ------------------: | ------------------: | ---------------: |
| A0            | YOLO11n stock                             |           2,590,425 |               15.14 |           0.6167 |
| N1            | YOLO11n P2–P5 4-head                     |           2,667,084 |               24.23 |           0.6179 |
| P1            | YOLO11n P2–P4 (P5 제거)                  |           1,939,145 |               22.89 |           0.6411 |
| T0            | YOLO11s stock                             |           9,428,953 |               50.21 |           0.6602 |
| **U1Z** | **YOLO11n + internal ×2 upsample** | **2,590,425** |     **64.70** | **0.7122** |

최종 U1Z 모델은 baseline 대비 **+0.0955 mAP50 (+9.55%p)**를 기록하며 기업 목표 성능을 충족했습니다.

---

### 5.2 Teacher–Student Knowledge Distillation

실시간 Edge 환경에서는 단순히 큰 모델을 사용하는 것이 어렵기 때문에, 고성능 Teacher 모델의 정보를 경량 Student 모델에 전달하는 **Knowledge Distillation** 전략을 적용했습니다.

이 과정에서 단순 정확도 최대화보다 다음의 trade-off를 함께 고려했습니다.

- Detection Accuracy
- Model Complexity
- Inference Latency
- Real-Time FPS
- Edge Device Deployability

이를 통해 **고성능 모델의 표현력을 최대한 유지하면서 Jetson에서 실시간 추론 가능한 모델**을 구축하는 것을 목표로 했습니다.

---

### 5.3 Multi-Object Tracking & Threat Assessment

프레임 단위 detection 결과를 실제 드론 감시 상황에 활용할 수 있도록 Multi-Object Tracking을 결합했습니다.

```text
Detection
   ↓
Track ID Assignment
   ↓
Trajectory Estimation
   ↓
Temporal UAV State
   ↓
Threat Assessment
```

이를 통해 드론의 ID를 프레임 간 유지하고, 객체의 이동과 접근 정보를 활용해 **위험도를 분류/판단하는 시스템**으로 확장했습니다.

---

### 5.4 Sim-to-Real Validation

대회의 두 번째 핵심 과제는 실제 드론 시연장에 대응할 수 있는 모델을 만드는 것이었습니다.

실제 시연장과 유사한 배경·환경을 simulation data로 구성하고 이를 학습에 활용한 뒤, 실제 비행 UAV에 대한 detection/tracking 성능을 검증했습니다.

```text
Simulation Environment
        ↓
Synthetic UAV Data
        ↓
Model Training
        ↓
Domain Gap
        ↓
Real Demonstration Site
        ↓
Actual UAV Detection / Tracking
```

이를 통해 단순 validation-set 성능이 아니라 **학습 환경과 실제 배포 환경의 차이를 고려한 perception system**을 개발했습니다.

---

### 5.5 TensorRT & Jetson Edge Deployment

최종 모델은 NVIDIA Jetson Orin Nano Super에서 실시간으로 동작할 수 있도록 TensorRT 기반 최적화를 수행했습니다.

```text
PyTorch Model
     ↓
Model Export
     ↓
TensorRT Optimization
     ↓
Jetson Orin Nano Super
     ↓
Camera Stream
     ↓
Real-Time Detection / Tracking
```

대회 현장에서는 Jetson Orin Nano Super에 카메라 모듈을 연결해 **실제로 비행 중인 드론을 실시간으로 탐지하는 시연**을 수행했습니다.

평가 시 정확도뿐 아니라 다음 시스템 지표를 함께 확인했습니다.

- FPS
- End-to-End Latency
- Model Size / Complexity
- Edge Device Runtime
- Real-Time Camera Inference Stability

---

## 6. Why This Project Matters

이 프로젝트는 단순한 모델 학습 실험이 아니라 **CV 모델을 실제 Edge 시스템에 배포하고 현장에서 검증한 End-to-End 프로젝트**입니다.

### Research

- Small Object Detection
- Knowledge Distillation
- Sim-to-Real
- Domain Gap
- Detection / Tracking

### Engineering

- PyTorch / Ultralytics
- Model Architecture Modification
- TensorRT
- NVIDIA Jetson
- Camera Integration
- Real-Time Inference
- FPS / Latency Profiling

### System-Level AI

- Detection → Tracking → Threat Assessment
- Accuracy–Latency Trade-off
- Hardware-aware Optimization
- Real-World Validation

### Reproducibility

- 학습 config 관리
- 독립 평가 코드
- 환경 정보 기록
- 실험별 결과 저장
- GitHub 기반 코드/실험 공개

---

## 7. Repository Structure

```text
configs/
  train/        A0_baseline / N1_model1 / P1_model2 / T0_model3 / U1Z_ours .yaml
                (id = 파일 이름 = runs/detection/<id> run 디렉터리 이름)
  model/        yolo11n_mapped_p2.yaml (N1), yolo11n_p2_p4.yaml (P1), yolo11n_up2.yaml (U1Z)
  data/         airbility_uav_detection.yaml

src/
  data/         prepare_airbility_dataset.py, validate_yolo_dataset.py
  train/        train_detector.py, average_checkpoints.py
  evaluation/   evaluate_company.py
  tests/        test_evaluate_company.py

scripts/        데이터 준비 및 실행 스크립트

experiments/<ID>/train/
  best_map50.json
  environment.json
  experiment_config.yaml
  args.yaml
```

`runs/`, `data/processed/`, `*.pt`는 `.gitignore`로 제외됩니다. `experiments/`에는 실제 학습 run의 핵심 설정과 결과를 저장하여 동일 조건의 실험을 재현할 수 있도록 구성했습니다.

---

## 8. Environment

```text
Python 3.10.21
PyTorch 2.14.0+cu130
Ultralytics 8.4.138
Training GPU: RTX 5060 Ti 16 GB
Edge Device: NVIDIA Jetson Orin Nano Super
```

```bash
pip install -r requirements.txt
```

---

## 9. Dataset Preparation

기업 원본 데이터는 다음 구조를 사용합니다.

```text
data/airbility_uav_detection_dataset/
├── images/
│   ├── train/<sequence>/
│   └── val/<sequence>/
└── labels/
    ├── train/<sequence>/
    └── val/<sequence>/
```

원본 라벨 형식:

```text
cls cx cy w h track_id
```

YOLO detection 학습용 데이터 준비:

```bash
bash scripts/01_prepare_airbility_dataset.sh
```

Provider가 제공한 train/val split을 그대로 유지합니다.

---

## 10. Training

```bash
python -m src.train.train_detector --config configs/train/A0_baseline.yaml
python -m src.train.train_detector --config configs/train/N1_model1.yaml
python -m src.train.train_detector --config configs/train/P1_model2.yaml
python -m src.train.train_detector --config configs/train/T0_model3.yaml

# U1Z는 A0 best checkpoint를 초기값으로 사용
python -m src.train.train_detector --config configs/train/U1Z_ours.yaml
```

`train_detector.py`는 validation mAP50 최고 checkpoint를 `weights/best_map50.pt`에 저장하고 `best.pt`로 복사합니다.

각 run에는 아래 정보를 함께 기록합니다.

- `experiment_config.yaml`
- `environment.json`
- `command.txt`
- `args.yaml`
- `best_map50.json`

---

## 11. Evaluation

```bash
python -m src.evaluation.evaluate_company \
  --model runs/detection/<ID>/weights/best.pt \
  --data data/processed/airbility_uav_detection_yolo/data.yaml \
  --split val --imgsz 1280 --batch 4 --device 0 --workers 8 \
  --conf 0.001 --iou 0.7 --max-det 300 --plots --save-json \
  --output runs/evaluation/<ID>_company_val_1280/metrics.json
```

모든 모델은 동일한 validation split과 평가 조건에서 비교하여 구조 변경에 따른 성능 차이를 확인합니다.

---

## 12. Portfolio Summary

**Real-Time Small UAV Detection, Tracking & Threat Assessment on Edge AI****2026 Drone AI Hackathon — Grand Prize**

- 기업 제공 3-class UAV 데이터 기반 소형 객체 탐지 모델 개발
- YOLO11n baseline `mAP50 0.6167` → 최종 모델 `0.7122`로 개선
- 기업 목표 `mAP@0.5 ≥ 0.70` 및 Jetson 실시간 추론 `≥15 FPS` 달성
- Teacher–Student Knowledge Distillation 기반 경량화 및 성능 보완
- Multi-Object Tracking 및 위험도 분류 파이프라인 구현
- simulation data를 활용한 Sim-to-Real domain gap 대응
- TensorRT 기반 NVIDIA Jetson Orin Nano Super 최적화 및 실시간 카메라 시연
- FPS / Latency 기반 실제 Edge AI 성능 분석
- 학습/평가 코드 및 실험 설정 공개를 통한 재현성 확보

**Keywords:** `Computer Vision` `Object Detection` `Small Object Detection` `Knowledge Distillation` `Multi-Object Tracking` `Sim-to-Real` `TensorRT` `Jetson Orin Nano Super` `Edge AI` `Real-Time Inference`
