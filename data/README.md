# data 디렉토리 안내

이 디렉토리에는 바둑 포지션을 전처리한 `.npz` 파일들이 들어 있습니다.  
모든 `.npz` 파일은 공통적으로 다음 키를 가집니다.

- `states`: float32, shape `(N, 3, 19, 19)`
  - 채널 0: 현재 플레이어의 돌 (0/1)
  - 채널 1: 상대 플레이어의 돌 (0/1)
  - 채널 2: 보조 정보 (예: 턴/플레이어 등, 프로젝트 코드와 동일한 방식)
- `actions`: int64, shape `(N,)`
  - 각 수는 `0 ~ 360` 사이의 인덱스로 인코딩됨  
    (row * 19 + col 형태, `src` 코드와 동일)

raw data는 명시된 출처에서 한 번에 다운받아서 unzip할 수 있습니다. 
---

## 디렉토리 구조

```text
data/
  processed/
    base_policy_100k.npz
    shin_policy_50k.npz

1. processed/base_policy_100k.npz

설명

여러 기사/대국을 섞어서 만든 일반 포지션 데이터셋

기본 정책망(Base policy)을 학습할 때 사용

내용

states: (N, 3, 19, 19)

actions: (N,)

사용처

src/train_bc.py

python src/train_bc.py \
  --data data/processed/base_policy_100k.npz \
  --save models/policy_bc_100k.pt


src/train_style_classifier.py 에서 Base 샘플로 사용
(--base_data data/processed/base_policy_100k.npz)

src/train_style_rl.py 에서 RL 학습용 컨텍스트(state)로 사용
(--data data/processed/base_policy_100k.npz)

2. processed/shin_policy_50k.npz

설명

신진서 9단 대국에서 샘플링한 포지션/착점 데이터

스타일 분류기에서 “Shin” 클래스로 사용

내용

states: (N, 3, 19, 19)

actions: (N,)

사용처

src/train_style_classifier.py

python src/train_style_classifier.py \
  --shin_data data/processed/shin_policy_50k.npz \
  --base_data data/processed/base_policy_100k.npz \
  --out models/style_classifier.pt


model_test.ipynb 등에서 스타일 비교 / 시각화용으로 사용
