# RL Go Style Project

> 바둑 정책망을 신진서 9단 스타일로 파인튜닝하는  
> 컨텍스추얼 밴딧 기반 강화학습 프로젝트

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
