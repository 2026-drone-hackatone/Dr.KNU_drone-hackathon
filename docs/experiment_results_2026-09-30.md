# 학습 방법론 및 성능 정리 — 2026-09-30

요청하신 `config/train`에 해당하는 실제 폴더는 [`configs/train`](configs/train)이다. 현재 체크아웃의 설정, `runs/detection/*/results.csv`, 실행 당시 `experiment_config.yaml`·`args.yaml`, 저장된 평가 JSON을 대조했다. 이번 작업에서는 새 학습이나 추론 평가를 실행하지 않았다.

## 1. 핵심 요약

- **가장 큰 개선은 내부 2배 업샘플링(U1/U2)**이었다. 학습 로그 mAP50 기준 A0 **0.61802 → U1 0.69726**, T0 **0.66373 → U2 0.70469**. 초기값과 학습 조건도 달라 순수 구조 효과만을 분리한 결과는 아니다.
- **저장된 기업 3클래스 별도 평가 중 최고는 U2의 epoch 1–3 가중치 평균 0.71321**이다. U1은 0.69569, 경량 P2–P4 모델 P1은 0.64905다.
- P2 채널 확장에서는 **N2(48채널)**가 N1(32채널)/N3(64채널)보다 높았고, NWD·SPD·crop·oversampling·3배 업샘플링은 각 대조군보다 개선되지 않았다.
- 외부 데이터 전이는 A2b에서 기준선 대비 소폭 개선됐지만 PA2/N4에서는 개선되지 않았다. 최신 **S1은 외부 사전학습 기록만 있어 기업 3클래스 최종 성능이 아직 없다.**
- **KD3는 U1을 개선하지 못했다.** U1Z 및 U1Z_KD는 val까지 학습에 포함한 별도 실험이므로 일반 검증 성능과 구분해야 한다.

## 2. 수치 해석 기준

주 비교 대상은 기업 제공 val의 `quad_civil / fixed_wing / target_uav` 3클래스 검출이다. mAP는 0–1 범위이며 0.70은 70%다. 대부분 COCO 초기화, AdamW, cosine LR, `imgsz=1280`, `rect=true`, mosaic/mixup 비활성화 및 약한 위치·크기·색상 증강을 사용했다. 개별 변형은 아래 설명과 원본 설정을 따른다.

- **학습 로그 최고:** CSV에서 mAP50가 가장 높은 epoch를 선택했다. mAP50–95도 **그 epoch의 값**이다. 별도 평가나 test 결과가 아니다.
- **별도 평가:** 저장된 checkpoint를 기업 val로 평가한 JSON이다. 학습 데이터와 분리된 val이지만 모델 선택에 사용했으므로 최종 독립 test 성능을 뜻하지 않는다.
- **기록/설정 epoch:** 현재 CSV 행 수 / 실행 당시 설정 epoch. 조기 종료·수동 중단·진행 중 여부를 이 숫자만으로 단정하지 않는다.
- 외부 val, 기업 1클래스, train+val 합산 학습 점수는 기업 3클래스 검증 순위에 섞지 않았다. 설정만 있고 로그가 없는 경우는 **학습 결과 미확인**으로 표기했다.

## 3. 기업 3클래스 학습 결과

실험명을 누르면 해당 원본 CSV를 확인할 수 있다.

| 실험 | 방법론 | 기록/설정 epoch | 로그 최고 mAP50 | 같은 epoch mAP50–95 | 최고 epoch |
|---|---|---:|---:|---:|---:|
| [A0](runs/detection/A0_yolo11n_coco_airbility_provider_split_1280x720/results.csv) | COCO 사전학습 YOLO11n → 기업 3클래스 직접 미세조정(기준선) | 41/60 | 0.61802 | 0.38534 | 11 |
| [T0](runs/detection/T0_yolo11s_coco_airbility_provider_split_1280x720/results.csv) | 기본 YOLO11s로 모델 용량 확대(n → s) | 32/60 | 0.66373 | 0.42889 | 19 |
| [T2](runs/detection/T2_yolo11m_coco_airbility_provider_split_1280x720/results.csv) | 기본 YOLO11m으로 모델 용량 확대 | 16/60 | 0.61038 | 0.38670 | 15 |
| [A1](runs/detection/A1_yolo11n_p2_coco_airbility_provider_split_1280x720/results.csv) | P2 고해상도 검출 head 추가, P2–P5 4개 출력 | 35/60 | 0.58087 | 0.35102 | 15 |
| [P1](runs/detection/P1_yolo11n_p2_p4_company_1280x720/results.csv) | P5 제거, P2/P3/P4 고해상도·경량 검출 구조 | 24/60 | 0.64765 | 0.38516 | 9 |
| [T1](runs/detection/T1_yolo11m_p2_p4_company_1280x720/results.csv) | YOLO11m P2–P4 구조로 교사 후보 학습 | 33/60 | 0.59229 | 0.36098 | 10 |
| [N1](runs/detection/N1_yolo11n_mapped_p2_company/results.csv) | P2–P5 구조에 COCO backbone/neck 가중치 명시적 layer mapping | 40/60 | 0.62409 | 0.39267 | 31 |
| [N2](runs/detection/N2_yolo11n_p2_48_company/results.csv) | N1의 P2 출력 채널 32 → 48 확장 | 34/60 | 0.63505 | 0.39189 | 22 |
| [N3](runs/detection/N3_yolo11n_p2_64_company/results.csv) | P2 출력 채널을 64로 확장 | 37/60 | 0.61668 | 0.37964 | 30 |
| [T3](runs/detection/T3_yolo11m_mapped_p2_company/results.csv) | YOLO11m P2–P5 + 명시적 COCO layer mapping | 26/60 | 0.64043 | 0.40334 | 22 |
| [N5](runs/detection/N5_yolo11n_spd_p2_48_company/results.csv) | N2의 두 번째 downsampling을 PixelUnshuffle + Conv(SPD)로 교체 | 40/60 | 0.58588 | 0.36742 | 20 |
| [P2](runs/detection/P2_yolo11n_p2_p4_nwd_company_1280x720/results.csv) | P1에 NWD 박스 손실 추가(weight=0.2, constant=12.8) | 15/60 | 0.58446 | 0.35770 | 12 |
| [D1](runs/detection/D1_yolo11n_p2_p4_short_sched_company_1280x720/results.csv) | P1 구조에 25 epoch 짧은 학습률 스케줄 적용 | 25/25 | 0.62536 | 0.37140 | 9 |
| [D2](runs/detection/D2_yolo11n_p2_p4_short_sched_tiny_oversample_1280x720/results.csv) | D1에 초소형 객체 포함 프레임 oversampling 추가 | 25/25 | 0.60498 | 0.35482 | 7 |
| [A2 stage3](runs/detection/A2_stage3_yolo11n_airbility_generic_to_3class_1280x720/results.csv) | 외부 drone 1클래스 → 기업 1클래스 → 기업 3클래스 순차 전이 | 39/60 | 0.59895 | 0.35632 | 14 |
| [A2b stage3](runs/detection/A2b_stage3_yolo11n_airbility_generic_to_3class_lr5e4_1280x720/results.csv) | A2 최종 단계 lr0를 2e-4 → 5e-4로 상향 | 44/60 | 0.62288 | 0.38165 | 28 |
| [A3 stage3](runs/detection/A3_stage3_yolo11n_airbility_quad_target_tiny_crop_lr5e4_1280x720/results.csv) | A2 최종 단계에 quad/target 초소형 객체 crop 데이터 추가 | 39/60 | 0.61388 | 0.37907 | 24 |
| [P3](runs/detection/P3_yolo11n_p2_p4_from_p1_tiny_crop_company_1280x720/results.csv) | P1 best에서 tiny crop 데이터로 추가 미세조정 | 16/30 | 0.61990 | 0.36995 | 1 |
| [PA2 stage3](runs/detection/PA2_stage3_yolo11n_1280x720/results.csv) | P2–P4 구조에 외부 1클래스 → 기업 1클래스 → 기업 3클래스 전이 | 21/60 | 0.59892 | 0.35930 | 21 |
| [N4 stage2](runs/detection/N4_stage2_yolo11n_p2_48_company/results.csv) | P2 48채널 모델의 외부 1클래스 사전학습 → 기업 3클래스 미세조정 | 48/60 | 0.62952 | 0.38044 | 24 |
| [U1](runs/detection/U1_yolo11n_up2_company_1280x720/results.csv) | A0 가중치 + 모델 내부 bilinear 2배 업샘플링 | 32/40 | 0.69726 | 0.45820 | 19 |
| [U2](runs/detection/U2_yolo11s_up2_from_T0_company_1280x720/results.csv) | T0 가중치 + 내부 2배 업샘플링, BN 통계 고정 | 20/20 | 0.70469 | 0.43862 | 2 |
| [U2b](runs/detection/U2b_yolo11s_up2_from_T0_bn_train_b3_company_1280x720/results.csv) | U2 계열에서 BN 학습 허용, batch=3 및 lr0=5e-4로 변경 | 21/30 | 0.68188 | 0.43508 | 13 |
| [U1c](runs/detection/U1c_yolo11n_up2_low_lr_company_1280x720/results.csv) | U1 best에서 낮은 학습률 5e-5로 추가 미세조정 | 6/8 | 0.68633 | 0.45471 | 6 |
| [U2c](runs/detection/U2c_yolo11s_up2_from_avg_low_lr_company_1280x720/results.csv) | U2 ep1–3 평균 가중치에서 lr0=2.5e-5 추가 미세조정 | 4/6 | 0.69984 | 0.44789 | 1 |
| [U3](runs/detection/U3_yolo11n_up3_from_U1_company_1280x720/results.csv) | U1 가중치에서 내부 업샘플링 배율 2 → 3 확대 | 13/20 | 0.64995 | 0.41579 | 2 |
| [KD3](runs/detection/KD3_yolo11n_up2_from_U1_teacher_U2avg_company_1280x720/results.csv) | U2 ep1–3 평균 교사 → U1 초기화 학생으로 지식 증류 | 12/12 | 0.67716 | 0.44242 | 7 |

A2 stage3(lr0=2e-4), T2, P3는 현재 `configs/train`에 대응 파일이 없지만 실제 학습 로그와 실행 설정이 남아 있어 포함했다. A2b는 파일명이 `A2_stage3_...lr5e4...yaml`이지만 내부 `id`는 `A2b_...`이다. 위 결과는 대부분 단일 seed 실험이므로 방법론 전체의 일반적인 우열로 확대 해석하지 않는다.

## 4. 저장된 별도 평가 성능

동일 기업 3클래스 val, `imgsz=1280`, `rect=true`, conf=0.001, NMS IoU=0.7 기준이다. batch 등 실행 조건은 JSON에 기록되어 있다. `best.pt`와 `best_map50.pt`, 평균 checkpoint를 구분했다. 과거 JSON의 절대 경로에는 이전 루트 `/home/user/workspace/drone_comp`가 남아 있으므로 아래 현재 상대 링크를 근거로 삼는다.

| 실험 / checkpoint | mAP50 | mAP50–95 | 근거 JSON |
|---|---:|---:|---|
| U2 / avg_ep1-3.pt | 0.71321 | 0.45394 | [결과](runs/evaluation/U2_avg_ep1-3_company_val_1280/metrics.json) |
| U2 / best.pt | 0.70623 | 0.43782 | [결과](runs/evaluation/U2_yolo11s_up2_company_val_1280/metrics.json) |
| U2c / best.pt | 0.70301 | 0.44779 | [결과](runs/evaluation/U2c_best_company_val_1280/metrics.json) |
| U1 / best.pt | 0.69569 | 0.45712 | [결과](runs/evaluation/U1_yolo11n_up2_company_val_1280/metrics.json) |
| KD3 / best.pt | 0.67601 | 0.44129 | [결과](runs/evaluation/KD3_best_company_val_1280/metrics.json) |
| KD3 / avg_ep9-12.pt | 0.67672 | 0.44168 | [결과](runs/evaluation/KD3_avg_ep9-12_company_val_1280/metrics.json) |
| P1 / best.pt | 0.64905 | 0.38336 | [결과](runs/evaluation/P1_yolo11n_p2_p4_company_1280x720_val/metrics.json) |
| N2 / best_map50.pt | 0.63477 | 0.39148 | [결과](runs/evaluation/N4_diagnosis/n2_company_3class/metrics.json) |
| N4 stage2 / best_map50.pt | 0.62790 | 0.37852 | [결과](runs/evaluation/N4_diagnosis/n4_company_3class/metrics.json) |
| D1 / best.pt | 0.62389 | 0.37050 | [결과](runs/evaluation/D1_yolo11n_p2_p4_short_sched_company_1280x720_val/metrics.json) |
| T1 / best.pt | 0.59451 | 0.36103 | [결과](runs/evaluation/T1_yolo11m_p2_p4_company_1280x720_val/metrics.json) |

**가중치 평균**은 여러 epoch의 파라미터를 평균해 단일 모델로 평가하는 방법이다. U2는 best 0.70623 → ep1–3 평균 0.71321로 개선됐다. 반면 U1의 [ep18–22 평균](runs/evaluation/U1_avg_ep18-22_company_val_1280/metrics.json)은 0.69028로 best 0.69569보다 낮았다. 항상 효과가 있는 것은 아니다.

N4의 외부 사전학습 효과는 동일 평가에서 N2 0.63477 → N4 0.62790으로 개선되지 않았다. KD3 best도 학생 초기값 U1 0.69569 → 0.67601로 낮아졌다. 이 표에 없는 실험은 위 CSV 점수만 제시하며, 오래된 문서의 점수를 새로 평가한 값처럼 보충하지 않았다.

## 5. 외부 데이터·1클래스 중간 단계

아래는 다른 평가 대상의 점수다. A2/PA2의 stage1은 외부 drone 데이터, stage2는 기업 드론 3종을 1종으로 합친 데이터다. N4 stage1과 S1 stage1도 외부 1클래스 사전학습이다.

| 실험 | 방법론 | 기록/설정 epoch | 로그 최고 mAP50 | 같은 epoch mAP50–95 | 최고 epoch |
|---|---|---:|---:|---:|---:|
| [A2 stage2](runs/detection/A2_stage2_yolo11n_external_to_airbility_generic_1280x720/results.csv) | 외부 drone 1클래스 → 기업 1클래스 → 기업 3클래스 순차 전이 | 30/30 | 0.63554 | 0.36783 | 24 |
| [A2](runs/detection/A2_yolo11n_coco_external_generic_dvb70_purdue30_1280x720/results.csv) | 외부 drone 1클래스 → 기업 1클래스 → 기업 3클래스 순차 전이 | 34/60 | 0.36141 | 0.13451 | 19 |
| [N4 stage1](runs/detection/N4_stage1_yolo11n_p2_48_external/results.csv) | P2 48채널 모델의 외부 1클래스 사전학습 → 기업 3클래스 미세조정 | 33/60 | 0.65634 | 0.24093 | 8 |
| [PA2 stage1](runs/detection/PA2_stage1_yolo11n_1280x720/results.csv) | P2–P4 구조에 외부 1클래스 → 기업 1클래스 → 기업 3클래스 전이 | 17/60 | 0.20352 | 0.06685 | 2 |
| [PA2 stage2](runs/detection/PA2_stage2_yolo11n_1280x720/results.csv) | P2–P4 구조에 외부 1클래스 → 기업 1클래스 → 기업 3클래스 전이 | 30/30 | 0.63916 | 0.36626 | 19 |
| [S1 stage1](runs/detection/S1_stage1_yolo11s_external/results.csv) | 기본 YOLO11s의 외부 → 외부30%+기업70% → 기업 3단계 학습 | 50/60 | 0.68051 | 0.25417 | 38 |

A2/PA2는 외부 → 기업 1클래스 → 기업 3클래스로 순차 적응했다. N4는 외부 → 기업 3클래스로 바로 전이했다. S1은 YOLO11s를 유지하면서 중간 단계에 **외부 30% + 기업 70%**를 혼합하는 설정이며, 현재 stage2/3 학습 로그는 확인되지 않았다. 외부 stage1의 높은 점수가 기업 성능 향상을 보장하지 않는다.

## 6. train+val 합산 학습 — 일반 검증과 분리

기업 val을 학습에 포함한 데이터셋을 사용했다. 아래 점수는 학습에 포함된 val에 대한 **in-sample 점수**이며 3·4절과 직접 순위를 비교할 수 없다.

| 실험 | 방법론 | 기록/설정 epoch | 로그 최고 mAP50 | 같은 epoch mAP50–95 | 최고 epoch |
|---|---|---:|---:|---:|---:|
| [U1Z_KD](runs/detection/U1Z_KD_yolo11n_company_trainval_1280x720/results.csv) | U1Z 평균 교사 → 내부 업샘플링 없는 기본 YOLO11n 학생으로 정렬 증류 | 30/30 | 0.67100 | 0.45358 | 25 |
| [U1Z](runs/detection/U1Z_yolo11n_up2_company_trainval_1280x720/results.csv) | U1 방식으로 기업 train+val 합산 재학습 | 32/40 | 0.76524 | 0.54687 | 29 |

U1Z는 설정에 `stop_after_epoch=32`가 있다. U1Z_KD는 U1Z `avg_ep18-22.pt`를 교사로 사용하며, 학생과 교사의 feature 크기를 정렬해 내부 업샘플링 없는 YOLO11n으로 증류했다. `U1Z_KD_...train_internal_val...yaml`과 `U1Z_KD_...trainval...yaml`은 현재 동일 run ID와 trainval 데이터 경로를 가리켜 **같은 실험으로 한 번만 집계**했다. 파일명만으로 별도 내부 검증 실험이라고 판단하지 않았다.

## 7. 설정은 있으나 학습 결과 미확인

`runs/detection/<id>/results.csv` 기준이며, 실행하지 않았다는 확정은 아니다. 성능을 추정해 채우지 않았다.

| 설정 / 실험 | 방법론 | 성능 |
|---|---|---|
| [A4 stage3](configs/train/A4_stage3_yolo11n_high_lr_tiny_safe_1280x720.yaml) | A2 최종 단계 lr0=1e-3 및 tiny-safe 증강 | 미확인 |
| [E1](configs/train/E1_yolo11s_ft1920_from_T0_company_1920x1080.yaml) | T0에서 입력 1920 해상도 미세조정 | 미확인 |
| [KD1](configs/train/KD1_yolo11n_p2_p4_from_p1_teacher_m_company_1280x720.yaml) | T1 교사 → P1 초기화 P2–P4 학생 증류 | 미확인 |
| [KD2](configs/train/KD2_yolo11n_p2_p4_from_coco_teacher_m_company_1280x720.yaml) | T1 교사 → COCO 초기화 P2–P4 학생 증류 | 미확인 |
| [KD4](configs/train/KD4_yolo11n_mapped_p2_company.yaml) | T3 교사 → N1 P2–P5 학생 증류 | 미확인 |
| [S1 stage2](configs/train/S1_stage2_yolo11s_mixed30_70.yaml) | stage2: 외부30%+기업70% 혼합 / stage3: 기업 3클래스 적응 | 미확인 |
| [S1 stage3](configs/train/S1_stage3_yolo11s_company.yaml) | stage2: 외부30%+기업70% 혼합 / stage3: 기업 3클래스 적응 | 미확인 |
| [U1d](configs/train/U1d_yolo11n_up2_low_lr_seed43_company_1280x720.yaml) | U1 저학습률 추가 미세조정의 seed 43 변형 | 미확인 |
| [U2Z](configs/train/U2Z_yolo11s_up2_from_T0_company_trainval_1280x720.yaml) | U2 방식의 train+val 합산 재학습 | 미확인 |
| [Y1Z](configs/train/Y1Z_yolo26n_up2_company_trainval_1280x720.yaml) | YOLO26n + 내부 2배 업샘플링, train+val 합산 | 미확인 |
| [Y1](configs/train/Y1_yolo26n_up2_company_1280x720.yaml) | YOLO26n + 내부 2배 업샘플링, 기업 train | 미확인 |

## 8. 현재 결과의 의미

정확도 우선 후보는 **U2 epoch 1–3 평균**, nano 계열에서는 **U1**이다. P2 경로만 조정한 실험에서는 P1이 N1/N2/N3보다 높은 점수를 보였고, 외부 전이·증류·손실 변경은 현재 레시피에서 일관된 개선을 만들지 못했다. 최신 N5도 로그 최고 0.58588로 N2 0.63505보다 낮았다. 다만 N5는 모델 생성 전 seed 처리 변경도 있어 엄밀한 구조 비교에는 동일 초기화 절차의 대조군이 필요하다.

이 문서는 검출 정확도 요약이다. 내부 업샘플링은 연산량을 늘리므로 정확도 순위를 배포 속도 순위로 해석하지 않는다. 저장된 평가 시간이나 과거 FPS 추정을 새 Jetson 실측치로 포함하지 않았다.
