# models 디렉토리 안내

강화학습/BC 실험에서 사용되는 모델 체크포인트(`.pt`) 파일을 저장하는 폴더입니다.

---

## 포함/생성되는 주요 파일

### `policy_bc_100k.pt`
- **역할:** Base policy (프로 기보 imitation)
- **입력:** `(B, 3, 19, 19)` 바둑판 상태
- **출력:** `(B, 361)` logits (19×19 모든 교차점에 대한 분포)
- **생성 예시:**
  ```bash
  python src/train_bc.py \
    --data data/processed/base_policy_100k.npz \
    --save models/policy_bc_100k.pt

---

### `style_classifier.pt`

- **역할:** (state, action)이 **신진서 스타일인지** 판별하는 이진 분류기
- **출력:** `sigmoid(logit) = P(Shin style | s, a)`
- **생성 예시:**

    python src/train_style_classifier.py \
      --shin_data data/processed/shin_policy_50k.npz \
      --base_data data/processed/base_policy_100k.npz \
      --out models/style_classifier.pt \
      --epochs 10 \
      --batch_size 256 \
      --lr 1e-3 \
      --device cuda

---

### `policy_style_rl_e20_l01.pt` (또는 `policy_style_rl.pt`)

- **역할:** Base policy를 초기값으로,  
  스타일 보상 + BC + KL을 이용해 파인튜닝한 RL policy
- **특징:**
  - 스타일 분류기에서 나오는 `P(Shin style | s, a)`를 보상으로 사용
  - Base policy와의 KL, 전문가 수 imitation(BC)을 동시에 고려
- **생성 예시:**

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

---

필요에 따라 다른 실험 세팅에서 나온 `.pt` 파일도 이 폴더에 저장해 사용할 수 있습니다.  
어떤 세팅으로 학습된 모델인지 파일명을 통해 구분해 주세요.
