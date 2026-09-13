# 소형 드론 실시간 탐지·추적 대회 프로젝트 전체 컨텍스트
## Codex 전달용 통합 프로젝트 명세

> 목적: 이 문서는 대회의 공식 과제 내용, 기업 연구원에게 추가로 전달받은 실제 목표, 현재 보유 데이터셋, 모델/학습 전략, 단계별 개발 범위, 최종 평가 기준을 한 문서에 통합한다.  
> Codex는 이 문서를 **프로젝트의 최상위 컨텍스트 문서**로 취급한다.

---

# 1. 대회 과제 개요

## 과제명

**소형 드론 실시간 탐지 및 위협 분류 AI 모델 개발**

대회의 기본 목적은 공개 영상 데이터를 이용하여 소형 드론을 실시간으로 탐지하고, 필요하면 추적 및 위협 수준 판단까지 수행하는 AI 시스템을 구현하는 것이다.

최종적으로는 단순한 데스크톱 데모보다 **Jetson 계열 엣지 디바이스에 실제 탑재 가능한 실시간 비전 시스템**을 지향한다.

---

# 2. 과제 배경 및 필요성

공식 과제 설명의 주요 배경은 다음과 같다.

- 최근 우크라이나 전쟁을 계기로 저피탐 소형 드론의 위협이 현실적인 문제로 부상
- 국내에서도 북한 무인기 영공 침투, 공항 및 산업시설 주변 불법 드론 사례가 존재
- 현재 AI 기반 anti-drone / counter-UAS 플랫폼이 개발되고 있으나 가격과 규모가 큰 경우가 많음
- 소형 드론을 효과적으로 탐지하고 추적할 수 있는 경량 AI 시스템에 대한 수요가 존재
- 본 과제는 학생 팀이 실제 산업 수요와 직접 연결된 문제를 해결하는 실전형 프로젝트라는 성격을 가짐

---

# 3. 공식 문제 정의

공개 영상 데이터를 활용하여:

1. 소형 드론을 실시간으로 탐지
2. 필요하면 추적
3. 위험 등급 또는 위협 수준을 자동으로 분류

하는 AI 모델 또는 시스템을 구현한다.

단순히 이미지 한 장에서 drone bounding box를 찾는 것보다 **실시간 영상에서의 안정적인 탐지 및 지속적인 객체 인식**이 중요하다.

---

# 4. 공식 필수 요구사항

과제 문서에서 제시된 필수 요구사항은 다음과 같다.

## 4.1 공개 데이터 기반 학습 및 검증

예시:

- HuggingFace Drone Detection Dataset
- Macilou DroneDetection
- 기타 공개 드론 탐지 데이터셋

데이터는 연구/대회 사용이 가능한 라이선스인지 확인해야 한다.

---

## 4.2 탐지 정확도 목표

공식 권장 목표:

```text
mAP@0.5 >= 0.70
```

이 수치는 단순 Accuracy가 아니라 object detection의 `mAP@IoU 0.5`를 의미한다.

---

## 4.3 추론 속도 목표

공식 권장 목표:

```text
>= 15 FPS
```

기본 문서에서는 NVIDIA GPU 환경을 기준으로 제시되어 있으며, 최종 목표에서는 Jetson Nano 등 엣지 환경에서 실제 15 FPS 이상을 확보하는 것을 지향한다.

---

## 4.4 탐지 결과 시각화

최종 출력 영상에 최소 다음 정보를 표시할 수 있어야 한다.

```text
Bounding Box
Class
Confidence
```

tracking을 포함하는 최종 시스템에서는 추가로:

```text
Track ID
FPS
Threat level 또는 상태
```

등을 표시할 수 있다.

---

## 4.5 재현 가능성 확보

다음 내용을 문서화해야 한다.

```text
Source code
Training process
Hyperparameters
Configuration
Environment
Random seed
Model weights
Evaluation results
```

즉, 결과만 제출하는 것이 아니라 다른 사람이 동일한 환경에서 실험을 재현할 수 있어야 한다.

---

# 5. 공식 선택 구현 요소 / 가점 요소

다음 항목은 필수는 아니지만 대회 경쟁력을 높이는 요소다.

## 5.1 엣지 디바이스 탑재

예:

```text
NVIDIA Jetson Nano
NVIDIA Jetson Orin Nano
기타 edge GPU
```

최종 프로젝트에서는 **Jetson Nano 탑재를 주요 목표**로 설정한다.

가능한 최적화 후보:

```text
ONNX
TensorRT
FP16
Layer fusion
Input resolution optimization
```

---

## 5.2 다중 드론 동시 추적

다음 기능을 추가할 수 있다.

```text
Multi-Object Tracking
Track ID 유지
동일 객체 재식별 최소화
ID Switch 감소
```

---

## 5.3 위협 등급 자동 분류

예:

```text
정찰형
공격형
오인 객체
```

또는 상황 기반:

```text
LOW
MEDIUM
HIGH
```

초기 버전에서는 rule 기반 위협 평가도 허용 가능한 방향으로 본다.

---

## 5.4 입력 확장

예:

```text
실시간 카메라 입력
스트리밍 입력
저장 영상
```

---

## 5.5 다양한 환경 검증

예:

```text
밝은 환경
야간
역광
흐림
구름
terrain
low contrast
motion blur
```

---

# 6. 활용 가능한 기술 / 장비

공식적으로 활용 가능한 기술 예:

```text
Python
PyTorch
TensorFlow
YOLO 계열
OpenCV
ROS2
```

하드웨어/환경:

```text
NVIDIA Jetson Nano
NVIDIA Jetson Orin Nano
기타 NVIDIA Jetson 계열
대학 보유 GPU
Google Colab
Kaggle Notebook
클라우드 GPU
```

현재 프로젝트의 중심 stack은 다음과 같이 설정한다.

```text
Python
PyTorch
Ultralytics YOLOv8
OpenCV
ONNX
TensorRT
Jetson Nano
```

---

# 7. 기대 산출물

공식 과제의 기대 산출물은 다음과 같다.

```text
소프트웨어
알고리즘
```

프로젝트 최종 산출물은 구체적으로 다음을 목표로 한다.

```text
1. 학습 데이터 전처리 pipeline
2. YOLO drone detector
3. tracking pipeline
4. 실시간 inference pipeline
5. Jetson Nano용 optimized model
6. 실험 결과 및 benchmark
7. 재현 가능한 source code
8. 모델 weight
9. demo video / live demo
10. 기술 보고서 또는 발표 자료용 결과
```

---

# 8. 평가 기준

과제에서 제시된 평가 기준:

```text
실현가능성
산업적용성
기술완성도
창의성
```

따라서 단순히 높은 mAP를 얻는 것만으로는 충분하지 않다.

프로젝트는 다음 세 가지를 동시에 보여주는 것이 중요하다.

```text
정확한 탐지
실시간 동작
실제 비행/요격 상황과 연결되는 시스템 설계
```

---

# 9. 기업 연구원에게 추가로 전달받은 실제 최종 목표

공식 과제보다 더 구체적인 운영 시나리오가 기업 연구원을 통해 전달되었다.

최종 상황은 다음과 같이 가정한다.

```text
대형 요격/추격 비행체
       │
       └── RGB Camera 장착
                │
                ▼
       다른 드론을 탐색
                │
                ▼
      멀리 있는 target 포착
                │
                ▼
       target에 점차 접근
                │
                ▼
        지속 탐지 + tracking
                │
                ▼
        동일 Track ID 유지
```

즉 고정된 CCTV에서 드론을 찾는 문제가 아니다.

**카메라 자체가 움직이는 airborne-to-airborne detection/tracking 문제**에 가깝다.

---

# 10. 기업 측에서 강조한 핵심 동작

최종 모델에서 중요한 것은 다음이다.

## 10.1 먼 거리에서의 최초 포착

드론은 처음부터 크게 보이지 않는다.

예:

```text
Frame t0     .
Frame t1     •
Frame t2     ●
Frame t3    [●]
Frame t4   [Drone]
```

가능한 한 target이 작은 단계에서부터 detection을 시작하는 것이 중요하다.

---

## 10.2 접근 과정에서 지속 탐지

드론을 한 번 찾았더라도 이후 frame에서 반복적으로 놓치면 tracking이 무너질 수 있다.

따라서 다음도 중요하다.

```text
Frame 100: detected
Frame 101: detected
Frame 102: detected
Frame 103: miss
Frame 104: miss
Frame 105: detected
```

이러한 temporal gap을 최소화해야 한다.

---

## 10.3 동일 ID 유지

탐지된 드론에 동일한 ID를 지속적으로 부여해야 한다.

좋은 예:

```text
Frame 100 -> ID 3
Frame 101 -> ID 3
Frame 102 -> ID 3
...
Frame 400 -> ID 3
```

좋지 않은 예:

```text
Frame 100 -> ID 3
Frame 180 -> miss
Frame 185 -> ID 8
```

따라서 최종적으로 detection뿐 아니라 tracking 품질도 중요하다.

---

# 11. 기업 최종 평가 데이터셋 정보

현재 실제 파일은 전달받지 않은 상태이다.

대회 당일 또는 최종 평가 단계에서 기업 측 dataset을 받을 것으로 예상한다.

알려진 dataset 특성:

```text
Sequences          : 70
Frames             : 약 27,000
Bounding boxes     : 약 40,000
Resolution         : 1280 x 720
Classes            : 3
Target             : 서로 다른 형태의 드론 3종
Median target size : 약 9 x 4 pixels
Target area ratio  : 약 0.0035%
Background         : terrain, cloud overlap, distractor
Temporal sequence  : 있음
Track ID           : 제공
```

---

# 12. 기업 평가 데이터의 가장 중요한 특징: 극소형 객체

median target size:

```text
9 x 4 px
```

즉 box 면적은:

```text
36 pixels
```

정도밖에 되지 않는다.

1280x720에서 일반적인 YOLO `640x640` letterbox를 적용하면 실제 영상 scale은 약 0.5가 된다.

따라서 median target은 대략:

```text
4.5 x 2 px
```

수준으로 축소된다.

이것은 일반적인 object detector에서도 매우 어려운 영역이다.

따라서 프로젝트의 핵심 technical problem은:

> **Tiny / extremely-small UAV detection**

이다.

---

# 13. 최종 Hard Constraints

최종 모델 선택 시 다음 두 조건은 반드시 만족해야 한다.

```text
mAP@0.5 >= 0.70
```

```text
Jetson Nano end-to-end FPS >= 15
```

최종 model은 정확도가 가장 높은 모델이 아니라 **두 hard constraint를 만족하면서 원거리 최초 탐지와 tracking 안정성이 가장 좋은 모델**을 선택한다.

---

# 14. 최종 모델 selection priority

Hard constraints를 만족한 모델 중 다음 순으로 판단한다.

```text
1. Stable acquisition이 빠른가
2. Tiny/Small target Recall이 높은가
3. 동일 Track ID를 오래 유지하는가
4. ID Switch가 적은가
5. mAP@0.5가 높은가
6. Jetson에서 충분한 FPS margin이 있는가
```

15 FPS를 간신히 만족하는 것보다 deployment에서 안정성을 위해 가능한 경우:

```text
18~20 FPS 이상
```

의 detector/tracking pipeline을 목표로 한다.

---

# 15. 현재 보유 중인 외부 데이터셋

기업 데이터가 아직 없으므로 현재는 아래 세 dataset을 사용한다.

```text
1. Drone Detection Dataset
2. Drone-vs-Bird Challenge Dataset
3. Purdue UAV Dataset
```

---

# 16. Drone Detection Dataset의 역할

사용자가 보유한 Drone Detection Dataset은 상대적으로 **크게 보이는 드론**이 많은 dataset이다.

주요 역할:

```text
clear drone appearance
close-range drone
large target
drone silhouette
generic drone feature learning
```

주의:

- 전체 학습을 이 dataset 중심으로 구성하면 large drone에 편향될 수 있음
- tiny target 성능을 대표하는 dataset으로 사용하지 않음
- combined dataset에서 source balancing 필요

---

# 17. Drone-vs-Bird Challenge Dataset의 역할

주요 역할:

```text
far-range drone
small drone
bird distractor
hard negative
sky/cloud background
false-positive suppression
```

현재 Phase-1은 generic 1-class UAV detector이므로 기본적으로:

```text
class 0 = drone
bird = background / hard negative
```

로 사용한다.

단, 실제 annotation 구조를 확인한 후 처리한다.

annotation을 추측하지 않는다.

---

# 18. Purdue UAV Dataset의 역할

현재 외부 데이터 중 최종 시나리오와 가장 관련성이 높은 dataset으로 본다.

주요 역할:

```text
moving-camera
UAV-to-UAV view
video sequence
relative motion
small target
tracking-friendly data
```

Purdue 영상은 detection 학습 시 frame으로 변환할 수 있으나:

**frame-level random split은 금지한다.**

반드시 sequence/video 단위로 train/val/test를 분리한다.

---

# 19. 현재 External Dataset 학습 task

기업 데이터는 최종적으로 3-class지만 외부 데이터의 drone taxonomy가 기업의 세 종류와 동일하지 않다.

따라서 현재 단계에서 임의로:

```text
external drone A -> company class 1
external drone B -> company class 2
```

처럼 매핑하지 않는다.

Phase-1 task:

```yaml
nc: 1

names:
  0: drone
```

즉 현재 모델은:

> **Generic UAV Detector**

로 학습한다.

향후 기업 데이터 제공 후 output head를 3-class로 변경하여 fine-tuning한다.

---

# 20. 전체 학습 전략

현재 계획한 전체 training roadmap은 세 단계다.

```text
Stage 1
External UAV datasets
generic 1-class training
        │
        ▼
Generic UAV pretrained weights

Stage 2
Synthetic dataset
3 drone types
tiny target
far-to-near sequence
domain randomization
        │
        ▼
Synthetic-adapted weights

Stage 3
Company simulation dataset
3-class fine-tuning
        │
        ▼
Final detector
```

현재는 **Stage 1까지만 구현 및 학습한다.**

Synthetic 및 기업 simulation fine-tuning은 이후 별도 단계에서 설계한다.

---

# 21. 현재 Phase-1의 목표

현재 당장 수행할 범위:

```text
외부 데이터셋 확인
        ↓
annotation 구조 분석
        ↓
YOLO 형식 변환
        ↓
sequence-safe split
        ↓
bbox/scale 통계 분석
        ↓
combined generic UAV dataset
        ↓
YOLOv8n baseline training
        ↓
YOLOv8n-P2 training
        ↓
mAP / Recall / Tiny Recall 평가
```

현재 Phase-1의 최종 산출물:

```text
generic UAV pretrained weights
```

기업 dataset이 주어지는 즉시 fine-tuning에 사용할 수 있어야 한다.

---

# 22. Baseline architecture: YOLOv8n

Jetson Nano 배포를 고려하여 기본 detector는 YOLOv8n으로 설정한다.

이유:

```text
약 3M parameters
경량 architecture
TensorRT 변환 가능
실시간 inference 가능성이 높음
```

기존 실험에서 이미 YOLOv8n으로 다음 성능을 얻은 경험이 있다.

```text
Precision   : 0.9170
Recall      : 0.8351
F1          : 0.8742
mAP@0.5     : 0.8950
mAP@0.5:0.95: 0.4507
```

이 결과는 기존 dataset 기준 결과이며 새로운 combined external dataset baseline으로 그대로 사용하지 않는다.

새 pipeline에서 baseline을 다시 학습한다.

---

# 23. P2 Detection Head 전략

최종 기업 데이터의 median target이 `9x4 px`이므로 P2 모델은 핵심 비교 모델이다.

기본 YOLOv8:

```text
P3 / stride 8
P4 / stride 16
P5 / stride 32
```

P2 variant:

```text
P2 / stride 4
P3 / stride 8
P4 / stride 16
P5 / stride 32
```

P2의 목적:

```text
tiny UAV feature representation 향상
small target Recall 향상
더 이른 거리에서 detection 시작
localization 향상 가능성
```

단점:

```text
연산량 증가
memory 증가
Jetson FPS 감소 가능성
```

따라서 P2를 최종 모델로 미리 확정하지 않고 baseline과 비교한다.

---

# 24. 입력 해상도 전략

단순 `640x640`만 사용하지 않는다.

기업 영상은:

```text
1280 x 720
16:9
```

이다.

640x640 letterbox는 영상 자체를 약 640x360으로 축소하므로 tiny target이 지나치게 작아질 수 있다.

핵심 비교 후보:

```text
640 x 640
832 x 480
```

pixel 수:

```text
640 x 640 = 409,600
832 x 480 = 399,360
```

전체 pixel 수는 비슷하지만 832x480은 16:9 source에서 target을 상대적으로 크게 유지할 가능성이 있다.

---

# 25. Phase-1 핵심 모델 실험

최소 네 모델을 비교한다.

| ID | Architecture | Input |
|---|---|---|
| B0 | YOLOv8n | 640x640 |
| B1 | YOLOv8n | 832x480 |
| P0 | YOLOv8n-P2 | 640x640 |
| P1 | YOLOv8n-P2 | 832x480 |

필요 시 추후:

```text
YOLOv8n-P2 @ 736x416
```

등을 Jetson speed optimization 후보로 고려할 수 있다.

---

# 26. Phase-1 초기화

Baseline:

```text
COCO yolov8n.pt
        ↓
external combined UAV dataset
```

P2:

```text
YOLOv8n-P2 architecture
        +
compatible yolov8n.pt weights transfer
        ↓
external combined UAV dataset
```

P2에서 pretrained weight transfer 결과를 반드시 log에 남긴다.

---

# 27. 데이터 split 원칙

가장 중요한 원칙 중 하나다.

Video dataset에 대해 다음은 금지한다.

```text
frame 001 -> train
frame 002 -> val
frame 003 -> train
```

동일 video의 인접 frame은 매우 유사하므로 데이터 leakage가 발생한다.

반드시:

```text
sequence A -> train
sequence B -> train
sequence C -> val
sequence D -> test
```

형태를 사용한다.

권장 기본 split:

```text
Train 70%
Val   15%
Test  15%
```

공식 split이 존재하는 dataset은 공식 split을 우선한다.

---

# 28. Video sampling

Purdue와 같은 연속 video의 모든 frame을 무조건 사용하는 것은 피한다.

이유:

```text
frame t
frame t+1
frame t+2
```

가 거의 동일할 수 있기 때문이다.

sampling 옵션을 config화한다.

예:

```text
every N frames
1 FPS
3 FPS
5 FPS
```

단, tiny target 구간이 sampling 과정에서 지나치게 제거되지 않는지 확인한다.

향후 bbox-size-aware sampling도 고려할 수 있다.

---

# 29. Combined Dataset 전략

세 dataset은 단순 concat하지 않는다.

초기 역할 기반 sampling target 예:

```text
Purdue UAV       : 40%
Drone-vs-Bird    : 40%
Drone Detection  : 20%
```

이 값은 고정 정답이 아니다.

먼저 실제 dataset을 분석한 뒤 다음 통계를 보고 조정한다.

```text
image/frame count
drone box count
negative frame count
median bbox width
median bbox height
tiny target ratio
sequence count
```

source metadata는 항상 유지한다.

---

# 30. Negative Sample 전략

negative image도 적극적으로 사용한다.

예:

```text
bird only
empty sky
cloud
terrain
aircraft
distractor
```

YOLO에서 drone이 없는 negative image는:

```text
image 존재
label file empty
```

형태로 사용할 수 있다.

특히 bird hard-negative는 향후 false alarm 감소에 중요하다.

---

# 31. Tiny Object 분석

기업 target이 극소형이므로 external dataset에도 기업 scale과 가까운 target이 얼마나 존재하는지 반드시 분석한다.

각 GT box에 대해:

```text
bbox_width
bbox_height
bbox_area
bbox_area_ratio
```

를 계산한다.

또한 resize 후 예상 크기:

```text
bbox size @ 640x640
bbox size @ 832x480
```

도 계산한다.

---

# 32. Phase-1 내부 scale 기준

COCO 기준은 현재 task에 너무 크므로 내부 분석용 scale을 별도로 둔다.

초기 기준:

```text
ultra_tiny:
max(w, h) < 8 px

tiny:
8 <= max(w, h) < 16 px

small:
16 <= max(w, h) < 32 px

medium:
32 <= max(w, h) < 96 px

large:
max(w, h) >= 96 px
```

실제 dataset 분포를 본 뒤 수정 가능해야 한다.

---

# 33. Phase-1 Detection 평가 지표

공식 지표 외에도 tiny target 특성을 반영한다.

필수:

```text
Precision
Recall
F1
mAP@0.5
mAP@0.5:0.95
```

추가:

```text
Ultra-tiny Recall
Tiny Recall
Small Recall
Medium Recall
Large Recall
```

가능하면:

```text
Scale별 AP50
```

도 계산한다.

---

# 34. Dataset별 평가

Combined test 결과 하나만 보지 않는다.

각 source dataset에 대해 따로 평가한다.

예:

| Dataset | mAP50 | Recall | Tiny Recall |
|---|---:|---:|---:|
| Drone Detection | | | |
| Drone-vs-Bird | | | |
| Purdue | | | |
| Combined | | | |

이를 통해 특정 dataset에만 성능이 좋은 모델인지 확인한다.

---

# 35. False Positive / False Negative 분석

자동으로 오류 사례를 저장한다.

특히:

```text
bird false positive
cloud false positive
terrain false positive
tiny-drone false negative
```

를 구분해서 확인할 수 있으면 좋다.

추천 추가 지표:

```text
False positives / 1000 negative frames
```

---

# 36. 최종 Tracking 전략

Tracking은 Phase-1 구현 범위에서는 제외하지만 최종 프로젝트의 핵심이다.

첫 baseline:

```text
YOLO
  ↓
ByteTrack
```

ByteTrack을 우선 고려하는 이유:

- detection score가 낮은 box도 association에 활용 가능
- 멀리 있는 low-confidence drone의 track 유지에 유리할 가능성
- 상대적으로 lightweight

---

# 37. Moving Camera 대응

최종 카메라가 비행체에 장착되므로 background 전체가 움직일 수 있다.

따라서 tracker 비교 후보:

```text
ByteTrack
vs
BoT-SORT + Global Motion Compensation
```

tiny target에서는 appearance Re-ID feature의 정보량이 적을 가능성이 있으므로 무거운 Re-ID network보다 camera motion compensation을 우선 검토한다.

---

# 38. 최종 Tracking 평가 지표

기업 dataset에 `track_id`가 제공되므로 최종 단계에서는 다음을 평가한다.

```text
IDF1
HOTA
ID Switch
Track fragmentation
Longest lost interval
Track continuity
```

특히 기업 요구와 직접 연결되는 것은:

```text
언제 track이 시작되었는가
한 번 시작된 ID가 얼마나 오래 유지되는가
몇 번 ID가 바뀌는가
```

이다.

---

# 39. Stable Acquisition 개념

최종 프로젝트에서는 단순 first detection보다 **stable acquisition**을 중요하게 평가하는 것이 좋다.

예시 정의:

> GT target이 등장한 뒤, 일정 window 내에서 반복적으로 검출되고 tracker가 안정적인 동일 ID를 생성한 최초 frame.

예:

```text
5 frame window에서 최소 3 frame 이상 detection
+
동일 track ID 생성
```

이를:

```text
F_visible = GT에서 처음 보인 frame
F_acq     = stable tracking이 시작된 frame
```

이라고 하면:

```text
Acquisition Delay = F_acq - F_visible
```

낮을수록 좋다.

---

# 40. Acquisition Target Size

거리 GT가 없는 경우 최초 안정 tracking 시점의 bbox size를 원거리 탐지 성능의 proxy로 사용할 수 있다.

예:

```text
YOLOv8n
acquisition bbox = 20 x 10

YOLOv8n-P2
acquisition bbox = 10 x 5
```

P2 모델이 더 작은 target 상태에서 track을 시작했다면 의미 있는 개선으로 볼 수 있다.

---

# 41. Temporal Class Smoothing

최종 기업 데이터는 3 drone classes이다.

9x4 px 수준에서는 frame 하나만으로 drone type classification이 흔들릴 가능성이 높다.

같은 track에 대해 class probability를 시간적으로 누적하는 방법을 사용할 수 있다.

예:

```text
Frame 1 -> A
Frame 2 -> A
Frame 3 -> B
Frame 4 -> A
Frame 5 -> A
```

track-level 결과는 A로 안정화할 수 있다.

예시:

```text
EMA class probability
majority voting
track-level cumulative confidence
```

추가 CNN 없이 구현 가능하므로 edge 환경에 적합하다.

---

# 42. Synthetic Dataset 전략 — 향후 Phase-2

현재 구현하지 않지만 프로젝트 핵심 전략으로 유지한다.

최종 기업 evaluation이 simulation 기반이기 때문에 synthetic data를 적극적으로 활용할 가치가 있다.

Synthetic dataset은 단일 이미지 생성보다 **sequence generation**을 지향한다.

목표 통계:

```text
Resolution: 1280x720
Classes: 3 drone types
Tiny target 중심
Median target size 근처를 충분히 포함
Far -> Near sequence
Persistent track_id
Moving camera
Terrain / cloud / sky
Distractors
```

---

# 43. Synthetic Sequence에서 구현할 상황

향후 다음 trajectory를 생성하는 것을 고려한다.

```text
head-on approach
diagonal approach
crossing
curved trajectory
hover-like motion
rapid angular motion
temporary disappearance
reappearance
```

카메라 motion:

```text
yaw
pitch
roll
translation
vibration
camera shake
```

appearance/randomization:

```text
lighting
contrast
motion blur
Gaussian blur
haze
noise
JPEG artifacts
cloud overlap
terrain overlap
background diversity
```

---

# 44. Synthetic Data의 핵심 목적

Synthetic data의 목적은 예쁜 이미지를 만드는 것이 아니라:

```text
1. 기업 데이터와 유사한 tiny target scale 확보
2. 3 drone shape를 충분한 orientation으로 생성
3. far-to-near approach sequence 증가
4. simulation evaluation domain gap 감소
5. tracking용 temporal annotation 확보
```

이다.

---

# 45. 최종 전체 Training Roadmap

향후 전체 흐름:

```text
COCO pretrained YOLOv8n
          │
          ▼
[Phase 1]
External generic UAV training
1-class
          │
          ▼
generic_uav_best.pt
          │
          ▼
[Phase 2]
Synthetic sequence training
3-class
          │
          ▼
synthetic_adapted.pt
          │
          ▼
[Phase 3]
Competition-day company dataset
3-class fine-tuning
          │
          ▼
YOLOv8n / YOLOv8n-P2 candidates
          │
          ▼
Detection evaluation
          │
          ▼
Tracking optimization
          │
          ▼
ONNX
          │
          ▼
TensorRT FP16
          │
          ▼
Jetson Nano
          │
          ▼
>= 15 FPS
```

---

# 46. 대회 당일 예상 Workflow

기업 dataset을 전달받으면 빠르게 다음 순서를 실행할 수 있어야 한다.

```text
1. Dataset directory inspection
2. Annotation parser 확인
3. Sequence / class / bbox statistics 생성
4. YOLO 3-class format 변환
5. Official split 확인
6. Generic external weight에서 fine-tuning
7. Baseline YOLOv8n 학습
8. P2 model fine-tuning
9. mAP50 / Recall 비교
10. Tiny target metric 확인
11. Tracker 연결
12. Stable acquisition / ID switch 평가
13. ONNX export
14. TensorRT build
15. Jetson Nano benchmark
16. 최종 model 선정
```

따라서 Phase-1 프로젝트는 나중에 dataset path/config만 교체하여 위 workflow를 빠르게 실행할 수 있도록 설계해야 한다.

---

# 47. Jetson Nano Deployment 전략

현재 단계에서는 실제 deployment를 필수 구현하지 않지만 최종 architecture는 이를 고려한다.

추천 flow:

```text
best.pt
   ↓
ONNX
   ↓
Jetson Nano
   ↓
TensorRT FP16 engine
   ↓
camera / video
   ↓
detector
   ↓
tracker
   ↓
overlay
```

TensorRT engine은 가능한 한 target Jetson 환경에서 생성한다.

최종 FPS는 detector-only가 아니라:

```text
capture
preprocess
inference
NMS
tracking
visualization
```

을 포함한 end-to-end 기준으로 측정한다.

---

# 48. Knowledge Distillation — 선택적 향후 경쟁력 요소

최종 모델 성능 또는 Jetson 속도에서 필요할 경우 teacher-student KD를 적용할 수 있다.

예:

```text
YOLOv8s / YOLOv8m teacher
           ↓
Knowledge Distillation
           ↓
YOLOv8n student
           ↓
Jetson deployment
```

목적:

```text
student model size/FLOPs 유지
+
Recall / tiny-target performance 개선
```

단, 현재 Phase-1의 필수 범위는 아니다.

---

# 49. Hard Negative Mining — 선택적 향후 개선

학습된 detector로 negative video를 inference하여 false positive를 수집한다.

예:

```text
bird
aircraft
cloud
terrain edge
street object
```

false-positive frame을 다시 training data에 추가하여 재학습할 수 있다.

이후:

```text
False alarms / 1000 frames
```

지표로 효과를 측정한다.

---

# 50. 최종 Competition Story

프로젝트의 기술적 메시지는 단순:

> YOLOv8n을 fine-tuning해서 Jetson에 올렸다.

가 되어서는 안 된다.

최종적으로는 다음과 같은 문제 해결 구조를 지향한다.

> 제한된 연산 자원의 edge platform에서 극소형 UAV를 가능한 한 이른 거리에서 포착하기 위해 input geometry 및 P2 feature pyramid를 비교하고, UAV 특화 external/synthetic pretraining을 적용한다. 이후 temporal tracking을 통해 저신뢰도 초기 detection을 안정적인 track으로 연결하며, moving-camera 환경에서도 동일 ID를 유지하도록 한다. 최종 모델은 TensorRT로 최적화하여 Jetson Nano에서 15 FPS 이상의 실시간 동작을 목표로 한다.

---

# 51. 현재 Codex 구현 범위

## 지금 구현할 것

현재는 **Phase-1만 진행한다.**

```text
External dataset inspection
External annotation conversion
Generic 1-class YOLO dataset 생성
Sequence-safe train/val/test split
Dataset statistics
BBox scale statistics
YOLOv8n baseline
YOLOv8n-P2
640x640
832x480
Detection evaluation
Dataset별 evaluation
Tiny/Small Recall
False positive analysis
Experiment report
```

---

## 지금 구현하지 않을 것

아래 내용은 아직 구현하지 않는다.

```text
Synthetic dataset generator
Company simulation dataset conversion
Company 3-class training
ByteTrack
BoT-SORT
Track metric
Stable acquisition metric
Threat classifier
ONNX/TensorRT deployment
Jetson benchmark
Knowledge Distillation
```

단, 나중에 이 기능들을 추가하기 어렵지 않도록 module 구조는 확장 가능하게 설계한다.

---

# 52. Phase-1 프로젝트 구조 권장안

```text
drone_interceptor/
│
├── PROJECT_CONTEXT.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── data/
│   │   ├── drone_detection.yaml
│   │   ├── drone_bird.yaml
│   │   ├── purdue_uav.yaml
│   │   └── external_uav_combined.yaml
│   │
│   ├── model/
│   │   ├── yolov8n.yaml
│   │   └── yolov8n_p2.yaml
│   │
│   └── train/
│       ├── baseline_640.yaml
│       ├── baseline_832x480.yaml
│       ├── p2_640.yaml
│       └── p2_832x480.yaml
│
├── data/
│   ├── raw/
│   │   ├── drone_detection/
│   │   ├── drone_bird/
│   │   └── purdue_uav/
│   │
│   ├── processed/
│   ├── splits/
│   └── reports/
│
├── src/
│   ├── data/
│   │   ├── inspect_dataset.py
│   │   ├── inspect_annotations.py
│   │   ├── bbox_statistics.py
│   │   ├── convert_drone_detection.py
│   │   ├── convert_drone_bird.py
│   │   ├── convert_purdue.py
│   │   ├── split_by_sequence.py
│   │   ├── sample_video_frames.py
│   │   ├── build_combined_dataset.py
│   │   └── validate_yolo_dataset.py
│   │
│   ├── train/
│   │   ├── train_yolov8n.py
│   │   ├── train_yolov8n_p2.py
│   │   └── common.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_detector.py
│   │   ├── evaluate_by_scale.py
│   │   ├── evaluate_by_dataset.py
│   │   └── compare_models.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── paths.py
│
├── scripts/
│   ├── 00_inspect_all_datasets.sh
│   ├── 01_convert_all_datasets.sh
│   ├── 02_build_splits.sh
│   ├── 03_build_combined_dataset.sh
│   ├── 04_validate_dataset.sh
│   ├── 05_train_baseline.sh
│   ├── 06_train_p2.sh
│   ├── 07_evaluate_all.sh
│   └── run_phase1.sh
│
├── runs/
│   ├── baseline/
│   ├── p2/
│   └── evaluation/
│
└── docs/
    ├── dataset_report.md
    ├── experiment_log.md
    └── phase1_results.md
```

---

# 53. Codex 개발 원칙

## 반드시 지킬 것

```text
raw dataset 수정 금지
annotation format 추측 금지
config-driven pipeline
sequence-level split
random seed 고정
dataset source metadata 유지
모든 실험 config 저장
metrics JSON/CSV 저장
error case visualization 저장
train/val/test leakage 검사
모든 script에 --help 제공
repository root 상대경로 지원
```

---

## 하지 말아야 할 것

```text
video frame random split
외부 dataset class를 기업 3-class로 임의 매핑
raw annotation overwrite
test set으로 hyperparameter tuning
dataset별 다른 split을 실험마다 사용
실험 결과 자동 overwrite
annotation 확인 없이 converter 작성
```

---

# 54. Phase-1 결과 테이블

최종적으로 아래 표를 생성한다.

| Model | Input | Params | GFLOPs | Precision | Recall | mAP50 | mAP50-95 | Tiny Recall | Small Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 YOLOv8n | 640x640 | | | | | | | | |
| B1 YOLOv8n | 832x480 | | | | | | | | |
| P0 YOLOv8n-P2 | 640x640 | | | | | | | | |
| P1 YOLOv8n-P2 | 832x480 | | | | | | | | |

또한 dataset별 결과를 추가한다.

```text
Drone Detection
Drone-vs-Bird
Purdue UAV
Combined
```

---

# 55. Phase-1 완료 조건

## Data

- [ ] 세 external dataset 구조 분석
- [ ] annotation format 문서화
- [ ] YOLO 1-class 변환 완료
- [ ] sequence leakage 없는 split 생성
- [ ] bbox size statistics 생성
- [ ] negative frame statistics 생성
- [ ] combined external dataset 생성
- [ ] bounding-box visualization 검증

## Model

- [ ] YOLOv8n 640 학습
- [ ] YOLOv8n 832x480 학습
- [ ] YOLOv8n-P2 640 학습
- [ ] YOLOv8n-P2 832x480 학습

## Evaluation

- [ ] Precision
- [ ] Recall
- [ ] F1
- [ ] mAP@0.5
- [ ] mAP@0.5:0.95
- [ ] Tiny Recall
- [ ] Small Recall
- [ ] dataset별 평가
- [ ] negative frame false-positive 분석
- [ ] Params / GFLOPs
- [ ] desktop latency / FPS
- [ ] 최종 model comparison table

---

# 56. Codex가 현재 최우선으로 해야 할 작업

바로 학습부터 시작하지 않는다.

첫 순서는 다음이다.

```text
1. 세 raw dataset directory 구조 출력
2. annotation format 식별
3. image/video/annotation 개수 확인
4. class 정의 확인
5. bbox statistics 계산
6. sequence/video ID 존재 여부 확인
7. dataset별 conversion plan 보고
```

실제 파일 구조를 확인하기 전에는 annotation parser를 추측해서 작성하지 않는다.

이 inspection 결과를 먼저 사용자에게 보여주고, 문제가 없으면 conversion 및 training pipeline 구현을 진행한다.

---

# 57. 전체 프로젝트 핵심 요약

이 프로젝트는 단순한 일반 drone detector가 아니다.

최종 문제는:

```text
Moving airborne camera
        ↓
extremely small UAV
        ↓
early acquisition
        ↓
continuous detection
        ↓
persistent Track ID
        ↓
3 drone classes
        ↓
Jetson Nano
        ↓
mAP50 >= 0.70
FPS >= 15
```

이다.

따라서 프로젝트의 기술적 중심은:

```text
External UAV pretraining
Tiny-object-aware data analysis
YOLOv8n
YOLOv8n-P2
16:9 input optimization
향후 synthetic sequence training
향후 temporal tracking
향후 TensorRT edge deployment
```

이다.

현재 Phase-1에서는 그중 **external dataset 기반 generic UAV detector를 완성하고, YOLOv8n과 YOLOv8n-P2의 tiny-drone detection 성능을 비교하는 것**에 집중한다.
