# 7. 전체 실험 결과 정리와 디바이스 탑재 후보 (2026-09-21)

기준: 기업 원본 val 5,625장 / 7,183 box. "학습 로그"는 각 run의 `best_map50.json`(EMA, 학습 중 val), "독립"은 `evaluate_company`(imgsz 1280, rect, conf 0.001, iou 0.7, max_det 300) JSON. GFLOPs는 720x1280 기준 `environment.json`.
목표: 공식 gate mAP50 ≥ 0.70, 개인 목표 0.75(배포 고려 시 도달 어려움 인정). 배포: Jetson Orin Nano Super, end-to-end ≥ 15 FPS.

## 1. 전체 run 표 (mAP50 내림차순)

| ID | 구조 | 초기값 | 학습 데이터 | batch | lr0 | epochs | params | GFLOPs | stride | 로그 best (@ep) | 독립 평가 | 판정 |
|---|---|---|---|---:|---:|---|---:|---:|---|---:|---:|---|
| U2 | yolo11s + 내부 x2 | T0 best | company | 2 | 2.5e-4 | 20/20 | 9.43M | 204.1 | 4/8/16 | 0.7047 (2) | **0.7132** (ep1–3 평균), 0.7062 (best) | 정확도 최고. 204 GFLOPs라 배포 불가 |
| U1 | yolo11n + 내부 x2 | A0 best | company | 4 | 5e-4 | 32/40 | 2.59M | 63.2 | 4/8/16 | 0.6973 (19) | **0.6957** | 배포 가능 범위 내 최고 |
| T0 | yolo11s | yolo11s.pt | company | 4 | 5e-4 | 32/60 | 9.43M | 50.2 | 8/16/32 | 0.6637 (19) | 0.6629 (Mac 기록) | stock s, U2 초기값 |
| P1 | yolo11n P2/P3/P4 (P5 제거) | yolo11n.pt | company | 8 | 5e-4 | 24/60 | 1.94M | 22.9 | 4/8/16 | 0.6477 (9) | 0.6491 | 경량 후보 |
| A2 st2 | yolo11n | A2 external | company generic 1-class | 8 | 2e-4 | 30/30 | 2.59M | 15.1 | 8/16/32 | 0.6355 (24) | – | 1-class, 직접 비교 불가 |
| PA2 st2 | yolo11n P2/P4 | PA2 st1 | company generic 1-class | 8 | 2e-4 | 30/30 | 1.94M | 22.9 | 4/8/16 | 0.6392 (19) | – | 1-class, 직접 비교 불가 |
| D1 | yolo11n P2/P4, 짧은 스케줄 | yolo11n.pt | company | 8 | 5e-4 | 25/25 | 1.94M | 22.9 | 4/8/16 | 0.6254 (9) | 0.6239 | P1보다 낮음 |
| A2b st3 | yolo11n | A2 st2 | company 3-class | 8 | 5e-4 | 44/60 | 2.59M | 15.1 | 8/16/32 | 0.6229 (28) | 0.6236 (docs/3 기록) | external curriculum 최고, A0 +0.005 |
| P3 | P1 + tiny crop fine-tune | P1 best | tiny crop A3 | 8 | 1e-4 | 16/30 | 1.94M | 22.9 | 4/8/16 | 0.6199 (1) | – | 순손실 |
| A0 | yolo11n | yolo11n.pt | company | 8 | 5e-4 | 41/60 | 2.59M | 15.1 | 8/16/32 | 0.6180 (11) | – | company-only 기준선 |
| A3 st3 | A2b + tiny crop 4,000장 | A2 st2 | tiny crop A3 | 8 | 5e-4 | 39/60 | 2.59M | 15.1 | 8/16/32 | 0.6139 (24) | – | crop 순손실 |
| T2 | yolo11m | yolo11m.pt | company | 2 | 5e-4 | 16/60 | 20.1M | 157.5 | 8/16/32 | 0.6104 (15) | – | 큰 모델 무효 |
| D2 | D1 + 프레임 oversampling | yolo11n.pt | oversample D2 | 8 | 5e-4 | 25/25 | 1.94M | 22.9 | 4/8/16 | 0.6050 (7) | – | 순손실 |
| A2 st3 | yolo11n | A2 st2 | company 3-class | 8 | 2e-4 | 39/60 | 2.59M | 15.1 | 8/16/32 | 0.5990 (14) | – | lr 낮아 적응 부족 |
| PA2 st3 | yolo11n P2/P4 | PA2 st2 | company 3-class | 8 | 5e-4 | 21/60 | 1.94M | 22.9 | 4/8/16 | 0.5989 (21) | – | P1보다 낮음 |
| T1 | yolo11m P2/P3/P4 | yolo11m.pt | company | 2 | 5e-4 | 33/60 | 16.1M | 195.5 | 4/8/16 | 0.5923 (10) | 0.5945 | KD teacher 부적합 |
| P2 | P1 + NWD loss | yolo11n.pt | company | 8 | 5e-4 | 15/60 | 1.94M | 22.9 | 4/8/16 | 0.5845 (12) | – | NWD 무효 |
| A1 | yolo11n P2 (4-head) | yolo11n.pt | company | 8 | 5e-4 | 35/60 | 2.67M | 24.2 | 4/8/16/32 | 0.5809 (15) | – | P2 4-head 무효 |
| A2 / PA2 st1 | external 1-class pretrain | yolo11n.pt | external dvb70+purdue30 | 8 | 5e-4 | 34/60, 17/60 | – | – | – | 0.361 / 0.204 | – | 외부 데이터 자체 점수, 비교 대상 아님 |
| **U3** | yolo11n + 내부 x3 | U1 best | company | 2 | 2.5e-4 | **진행 중** 1/20 | 2.59M | ≈142 | 8/3, 16/3, 32/3 | ep1 0.6284 | – | 13:35 시작, 약 41분/epoch |

U2 부속 평가: class-agnostic 0.7366, ep9–20 평균 0.6874, ep14–20 평균 0.6823, last 0.6797. U1 부속: class-agnostic 0.7229, ep19–32 평균 0.6845, ep25–32 평균 0.6737.

## 2. 무엇이 효과가 있었고 없었나

효과 있음
- **내부 x2 upsample**: nano A0 0.618 → U1 0.696 (+0.078), s T0 0.664 → U2 0.706~0.713 (+0.04~0.05). 크기별 recall에서 8–16 px가 0.85~0.87로 해결됐고 <8 px는 0.17 → 0.25~0.29.
- 모델 크기 n → s (1280, x1): +0.046 (A0 → T0).
- P5 제거 P2/P3/P4 nano (P1): +0.03 (A0 → P1).
- Stage 3 lr 2e-4 → 5e-4: +0.024 (A2 → A2b).
- 피크 주변 3 epoch 평균: +0.007 (U2).

효과 없음 또는 역효과
- 4-head P2 추가(A1), NWD loss(P2), tiny crop 추가(A3, P3), 프레임 oversampling(D2), 짧은 스케줄(D1), 외부 데이터 curriculum(A2/PA2, A0 대비 +0.005 이하), m 급 모델(T1, T2), U2의 후반 학습(ep3 이후 순손실).
- 모든 run이 이른 epoch에 피크 후 하락. 후반 epoch 평균도 피크 미만이므로 노이즈가 아닌 과적합(55개 train 시퀀스).

남은 병목
- <8 px 객체(val GT의 29.5%, quad_civil의 52%, target_uav의 41%)의 탐지 실패. class-agnostic 매칭에서도 recall 0.29로 같아 분류 문제가 아니다. 분류 혼동으로 잃는 mAP50은 약 0.03.
- U3(x3)는 이 구간을 겨냥한 마지막 정확도 상한 실험이며, 142 GFLOPs라 배포 후보는 아니다.

## 3. 디바이스 우선 탑재 후보

TensorRT FP16, batch 1, 1280x736 입력, detector 단독과 ByteTrack 포함 end-to-end를 각각 측정한다. FPS 예측은 실측이 아니라 GFLOPs 비례 추정(Orin Nano Super 유효 3~6 TFLOPS 가정).

| 순위 | 체크포인트 | 독립 mAP50 | GFLOPs | 예상 detector FPS | 탑재 이유 |
|---|---|---:|---:|---|---|
| 1 | `runs/detection/U1_yolo11n_up2_company_1280x720/weights/best.pt` | 0.696 | 63.2 | 48~95 (e2e 25~45) | 배포 가능 범위에서 정확도 최고. gate 0.70에 0.004 부족. 2560x1472 activation의 대역폭 비용이 FLOPs 추정보다 클 수 있어 실측이 가장 필요한 모델 |
| 2 | `runs/detection/T0_yolo11s_coco_airbility_provider_split_1280x720/weights/best.pt` | 0.663 | 50.2 | 60~120 (e2e 30~55) | stock 구조라 변환 위험 최소. U1과 비슷한 FLOPs이므로 "큰 모델 x1" 대 "작은 모델 x2" 중 어느 쪽이 Jetson에서 싼지 판별 |
| 3 | `runs/detection/P1_yolo11n_p2_p4_company_1280x720/weights/best.pt` | 0.649 | 22.9 | 130~260 (e2e 60~100) | U1이 15 FPS를 못 넘을 때의 fallback. P5 제거로 U1의 1/3 연산 |

기준선 참고: A2b(15.1 GFLOPs, 0.624)는 파이프라인의 FPS 상한 확인용. U2/U3는 정확도 상한 및 KD teacher 용도로만 취급.

측정 후 판단
- U1 e2e ≥ 18 FPS: U1 채택. U3가 U1보다 +0.02 이상이면 U3 → U1 KD 검토.
- U1 e2e 15 FPS 미만: T0 또는 P1로 내려가되, 부족한 mAP는 train+val 합산 재학습과 피크 평균으로 보충.

## 4. 새로 추가된 코드
- `src/train/average_checkpoints.py` (epoch 평균 가중치, stock 모듈만 포함)
- `src/evaluation/evaluate_company.py --single-cls`
- `src/evaluation/evaluate_detection_by_scale.py` chunk 처리, `--class-agnostic`
- `src/train/train_detector.py` `_up3` 비정수 stride 검사
- `configs/model/yolo11n_up3.yaml`, `configs/train/U3_*.yaml`, `configs/train/U2b_*.yaml`(미실행, 배포 불가로 보류)
