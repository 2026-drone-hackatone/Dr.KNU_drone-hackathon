# 소형 드론 탐지·추적

2026 항공·드론 산업수요 기반 해커톤 과제 5.

원거리에서 수 픽셀 크기로 보이는 드론을 탐지하고, 카메라가 접근하는 동안 같은
`track_id`를 유지하는 엣지 추론 시스템을 만듭니다.

팀 Dr.KNU — 서준호, 김나은, 최경진, 하효정, 이상욱

## 목표

기업 평가셋 기준으로 두 가지를 동시에 만족해야 합니다.

- 정확도: mAP@0.5 가 0.70 이상
- 속도: Jetson Orin Nano Super에서 end-to-end 15 FPS 이상

정확도만 높은 모델도, 빠르기만 한 모델도 통과하지 못합니다.

대상 환경은 다음과 같습니다.

| | |
|---|---|
| 연산 | Jetson Orin Nano Super Dev-kit, NVMe 256GB (NVENC/NVDEC 없음) |
| 카메라 | See3CAM_24CUG, 렌즈 교체 불가 |
| 평가 데이터 | 1280×720, 3 class, bbox 중앙값 9×4 px, track_id 제공 |

## 폴더 구조

```
configs/            실험 설정. 여기 있는 파일만으로 같은 학습이 재현되어야 합니다.
  data/             데이터셋 정의 yaml (클래스, 경로, split)
  model/            모델 구조 yaml. P2 head 같은 구조 변경이 여기 들어갑니다.
  train/            학습 하이퍼파라미터. 실험 하나당 파일 하나.

data/
  raw/              원본 데이터. 각자 받아서 넣습니다. git에 올라가지 않습니다.
    drone_detection/
    drone_bird/
    purdue_uav/
  processed/        스크립트가 만들어내는 YOLO 포맷 데이터셋. git에 올라가지 않습니다.
  splits/           어느 시퀀스가 train/val/test로 갔는지 목록. git에 올라갑니다.
  reports/          데이터셋 검증 결과.

src/
  data/             원본을 YOLO 포맷으로 변환, split 생성, 검증
  train/            학습 진입점
  evaluation/       평가. 데이터셋별, 스케일 구간별 분석 포함
  utils/            seed 고정, 로깅, 경로 처리

scripts/            src/ 의 스크립트를 순서대로 묶어둔 실행 진입점.
                    직접 실행할 일은 보통 여기서 시작합니다.
runs/               학습 산출물. git에 올라가지 않습니다.
colab/              Colab 학습 노트북
docs/               문서와 실험 기록
  experiments/      실험별 results.csv와 지표 요약
```

빈 폴더에 들어 있는 `.gitkeep` 은 내용이 없는 파일입니다. git은 폴더를 추적하지 않아서
빈 폴더는 clone할 때 사라지는데, 파일을 하나 넣어두면 폴더가 유지됩니다.
지우지 마세요.

`configs/` 와 `data/splits/` 두 개가 이 레포의 핵심입니다. 어떤 설정으로 어떤 데이터를
썼는지가 여기 남아 있어야 나중에 실험 결과를 비교할 수 있습니다.

## 레포에 올리지 않는 것

| 항목 | 어디에 두나 |
|---|---|
| 원본 데이터셋 (13.5GB) | 각자 로컬 `data/raw/` |
| 전처리 데이터셋 | 각자 로컬 `data/processed/` |
| 모델 가중치 `.pt` | Google Drive |

## 레포 폴더별 설명



## 원본 데이터 준비

`data/raw/` 아래에 세 가지를 아래 폴더명 그대로 놓습니다. 이름이 다르면 스크립트가 못 찾습니다.

| 폴더 | 출처 | 용량 |
|---|---|---|
| `drone_detection/` | HuggingFace `pathikg/drone-detection-dataset` | 4.4GB |
| `drone_bird/` | GitHub `wosdetc/challenge` 의 annotation + 영상 77개 | 7.1GB |
| `purdue_uav/` | Purdue UAV_Dataset, 50 sequence / 70,250 frame | 2.0GB |

이미 다른 곳에 받아뒀다면 복사하지 말고 링크를 거는 편이 낫습니다.

```bash
# Windows (관리자 PowerShell)
New-Item -ItemType SymbolicLink -Path data\raw -Target D:\drone_hackatone\ws\data\raw

# Linux, macOS
ln -s /path/to/raw data/raw
```

## 재현 절차

```bash
# 환경
pip install -r requirements.txt

# 원본을 YOLO 포맷으로 변환하고 시퀀스 단위로 split
bash scripts/01_convert_all_datasets.sh
bash scripts/02_build_splits.sh
bash scripts/03_build_combined_dataset.sh

# 검증
bash scripts/04_validate_dataset.sh
```

검증 결과가 아래와 다르면 split이 틀어진 것이므로 학습을 시작하면 안 됩니다.

| Split | Images | Boxes | Negative | Source 구성 |
|---|---:|---:|---:|---|
| train | 26,177 | 28,149 | 10.00% | DD 3,714 / Bird 11,232 / Purdue 11,231 |
| val | 5,020 | 6,201 | 9.16% | DD 709 / Bird 2,082 / Purdue 2,229 |
| test | 5,557 | 9,153 | 3.26% | DD 1,111 / Bird 2,223 / Purdue 2,223 |

test 5,557장은 고정입니다. R1부터 모든 실험이 이 동일한 test로 평가되기 때문에 한 줄에
놓고 비교할 수 있습니다. 바꾸면 그동안 쌓은 실험 기록이 전부 무의미해집니다.

## 학습

로컬 GPU 또는 Colab에서 진행합니다.

```bash
bash scripts/05_train_baseline.sh    # YOLO11n
bash scripts/06_train_p2.sh          # YOLO11n + P2 head
```

Colab을 쓸 경우 `colab/train_colab.ipynb` 를 사용하세요. 세션이 끊겨도 Drive에 저장된
체크포인트에서 자동으로 이어집니다.

한 가지 주의할 점이 있습니다. Ultralytics가 저장하는 `best.pt` 는 mAP50이 아니라
`0.1 × mAP50 + 0.9 × mAP50-95` 라는 fitness 값을 기준으로 고릅니다. 그래서 mAP50이
가장 높았던 에폭과 다를 수 있습니다. mAP50으로 판정하는 실험이라면 노트북이 따로
저장하는 `best_map50.pt` 를 쓰세요.

## 실험 규칙

1. 실험 하나에 config 하나. `configs/train/` 에 저장하고 커밋합니다.
2. 이름은 `R{번호}_{모델}_{해상도}` 형식으로 (예: `R9_yolo11n_1280x736`).
3. 한 번에 한 가지만 바꿉니다. 해상도와 구조를 같이 바꾸면 어느 쪽 덕분인지 알 수 없습니다.
4. 결과는 `docs/experiments/` 에 results.csv와 지표 요약을 남깁니다.
5. `seed=42`, `deterministic=True` 고정.

현재까지의 베이스라인은 다음과 같습니다.

| Run | 모델 | 입력 | test AP50 | 비고 |
|---|---|---|---:|---|
| R1 | YOLOv8n | 832×480 | 0.582 | 새 split으로 최초 재학습 |
| R6 | YOLO11n | | 0.2971* | 베이스라인 후보 |
| R8 | YOLO11n-P2 | | 0.3008* | 베이스라인 후보 |


## 문서

| 문서 | 내용 |
|---|---|
| `PROJECT_CONTEXT.md` | 과제 정의, 제약, 전체 설계 방향 |
| `docs/phase1_data_training_summary.md` | 데이터셋 구성 과정과 R1 학습 결과 |
| `docs/experiment_log.md` | 실험 기록 |

## Git 사용법

브랜치 없이 `main` 하나로 갑니다. 브랜치가 필요한 상황은 여러 명이 같은 파일을 동시에
고칠 때인데, 우리는 역할별로 건드리는 파일이 달라서 그럴 일이 거의 없습니다.

지킬 것은 두 가지입니다.

1. 작업 시작 전에 `git pull` 하기. 이것만 지켜도 대부분의 문제가 안 생깁니다.
2. 커밋은 작게 자주. 하루치를 몰아서 하지 않기.

평소에는 이 네 줄만 씁니다.

```bash
git pull                     # 작업 시작 전, 남의 작업 받아오기

# 작업

git add .
git commit -m "무엇을 했는지"
git push
```

### 막혔을 때

**push가 거부당한 경우.** 내가 작업하는 동안 다른 사람이 먼저 올린 것입니다.
`git pull` 로 받아온 다음 다시 `git push` 하면 됩니다.

**pull 하다 conflict가 난 경우.** 같은 파일의 같은 줄을 둘이 고친 것입니다.
혼자 해결하려 하지 말고 팀 채팅에 파일 이름을 올리고 상의하세요. 잘못 해결하면
상대방 작업이 소리 없이 사라집니다.

**뭔가 잘못 건드린 경우.** 아직 커밋하지 않았다면 되돌릴 수 있습니다.

```bash
git checkout -- <파일명>     # 마지막 커밋 상태로 되돌리기
```

커밋까지 했다면 혼자 되돌리지 말고 팀에 알리세요. git은 기록이 남아 있어서 거의 다
복구되지만, 잘못 건드리면 그때부터 어려워집니다.

### 브랜치를 쓰는 경우

폴더 구조를 크게 바꾸거나, 잘 돌아가는 학습 코드를 실험적으로 갈아엎을 때만 씁니다.

```bash
git checkout -b 작업이름
# 작업하고 커밋
git push -u origin 작업이름   # 그다음 GitHub에서 Pull Request
```
