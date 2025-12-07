# RL Go Style Project

> 바둑 정책망을 신진서 9단 스타일로 파인튜닝하는  
> Contectual Bandit 기반 강화학습 프로젝트

---

## 1. 프로젝트 개요

- 프로 바둑 기보에서 **Base policy** 를 BC(behavior cloning)으로 학습
- Shin Jinseo vs 기타 기보로 **스타일 분류기** 학습
- 스타일 분류기의 `P(Shin style | s, a)`를 **보상**으로 사용
- Base policy를 초기값으로:
  - 스타일 보상 (policy gradient)
  - Base policy와의 KL penalty
  - 전문가 수 imitation(BC loss)
  를 함께 최적화하여  
  **기력을 크게 떨어뜨리지 않으면서 Shin 스타일에 가까운 policy** 를 얻는 것이 목표

---

## 2. 디렉토리 구조

```text
RL-go-style/
  data/
    processed/
      base_policy_100k.npz
      shin_policy_50k.npz
    README.md        # 데이터 포맷/역할 설명
  models/
    README.md        # 체크포인트 설명
    policy_bc_100k.pt   # 학습된 모델들
    policy_shin_bc_16k.pt
    policy_style_rl_e20_l01.pt
    style_classifier.pt
               
  src/
    __init__.py
    case_study_policies.py
    compare_policies.py
    dataset.py
    debug_list_sgf.py
    models.py
    sgf_parser.py
    sgf_shin_moves.py
    shin_utils.py
    train_bc.py
    train_style_classifier.py
    train_style_rl.py
  model_test.ipynb   # 정책 비교 / 시각화 노트북
  requirements.txt
  README.md

## 3. 환경 설정

```bash
git clone https://github.com/sangho24/RL-go-style.git
cd RL-go-style

# (선택) conda 환경
conda create -n rl-go-style python=3.10
conda activate rl-go-style

# 필수 패키지 설치
pip install -r requirements.txt
```

Google Colab에서도 `requirements.txt` 기준으로 설치하면 동일하게 실행 가능합니다.

---

## 4. 실행 순서 & 재현 커맨드

### 4.1 Base policy (BC) 학습  *(이미 ckpt 있으면 생략 가능)*

```bash
python src/train_bc.py \
  --data data/processed/base_policy_100k.npz \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --device cuda \
  --save models/policy_bc_100k.pt
```

---

### 4.2 스타일 분류기 학습

```bash
python src/train_style_classifier.py \
  --shin_data data/processed/shin_policy_50k.npz \
  --base_data data/processed/base_policy_100k.npz \
  --out models/style_classifier.pt \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --device cuda
```

---

### 4.3 스타일 RL 파인튜닝 (컨텍스추얼 밴딧 + BC + KL)

```bash
python src/train_style_rl.py \
  --data data/processed/base_policy_100k.npz \
  --base_model models/policy_bc_100k.pt \
  --style_model models/style_classifier.pt \
  --out models/policy_style_rl_e20_l01.pt \
  --epochs 20 \
  --batch_size 256 \
  --lr 1e-4 \
  --device cuda \
  --lambda_kl 0.1 \
  --lambda_bc 1.0
```

---

## 5. 평가 및 시각화 (노트북)

`model_test.ipynb`를 열어서:

- `policy_bc_100k.pt` (Base),  
  `policy_style_rl_e20_l01.pt` (Style RL) 등을 로드하고
- 특정 포지션 인덱스에 대해 `visualize_cases([idx1, idx2, ...])` 호출
- Shin 실제 수, Base policy, Style RL policy 의 top-k 착점을 같은 보드 위에 시각화

---

## 6. 추가 참고

- 데이터 포맷/역할: `data/README.md`  
- 체크포인트 설명 및 생성 예시 커맨드: `models/README.md`

