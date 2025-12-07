import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import GoPositionDataset
from models import PolicyNet
from train_style_classifier import StyleClassifier  # 같은 폴더에 있다고 가정

BOARD_SIZE = 19
INF = 1e9  # 불법수를 마스킹할 때 쓸 큰 음수


# -------------------------------------------------------
# 1. 합법수(빈 칸) 마스크
# -------------------------------------------------------
def compute_legal_mask(states: torch.Tensor) -> torch.Tensor:
    """
    합법수(후보) masking 함수.
    states: (B, C, 19, 19)
    가정:
      - states[:, 0] = 현재 플레이어 돌
      - states[:, 1] = 상대 돌
    반환:
      - legal_mask: (B, 361) bool, True = 비어있는 자리
    """
    occupied = (states[:, 0] > 0.5) | (states[:, 1] > 0.5)  # (B,19,19)
    legal = ~occupied
    return legal.view(states.size(0), -1)  # (B,361)


# -------------------------------------------------------
# 2. 스타일 보상 계산 (컨텍스추얼 밴딧 reward)
# -------------------------------------------------------
def compute_style_prob(style_model: nn.Module,
                       states: torch.Tensor,
                       actions: torch.Tensor) -> torch.Tensor:
    """
    스타일 분류기를 이용해 "Shin 스타일일 확률" p를 계산.

    - 경우 1: style_model(states, actions) -> (B, 2)  (2-class logits)
        => softmax 후 class 1(Shin)의 확률 사용
    - 경우 2: style_model(states, actions) -> (B,) 또는 (B,1) (scalar logit)
        => sigmoid 후 Shin 확률로 간주

    반환:
      probs: (B,)  = p(Shin | s,a)  \in (0,1)
    """
    with torch.no_grad():
        logits = style_model(states, actions)

        if logits.dim() == 2 and logits.size(-1) == 2:
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Shin 클래스 확률
        else:
            logits_flat = logits.view(-1)                # (B,)
            probs = torch.sigmoid(logits_flat)           # p(Shin)

    return probs   # (B,)


# -------------------------------------------------------
# 3. 스타일 RL + BC + KL 학습 루프
# -------------------------------------------------------
def train_style_rl(
    data_npz: str,
    base_model_path: str,
    style_model_path: str,
    out_path: str,
    batch_size: int = 256,
    epochs: int = 3,
    lr: float = 1e-4,
    device_str: str = "cuda",
    lambda_kl: float = 0.1,
    lambda_bc: float = 1.0,
    normalize_rewards: bool = True,
):

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # ----- Dataset (state, expert_action) -----
    ds = GoPositionDataset(data_npz)
    n_total = len(ds)
    print(f"[GoPositionDataset] Loaded {n_total} samples from {data_npz}")
    print(f"[Data] using {n_total} states from {data_npz}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # ----- Base policy (frozen) -----
    base_policy = PolicyNet(in_channels=3, channels=64, num_blocks=6, board_size=BOARD_SIZE)
    base_state = torch.load(base_model_path, map_location=device)
    base_policy.load_state_dict(base_state)
    base_policy.to(device)
    base_policy.eval()
    for p in base_policy.parameters():
        p.requires_grad_(False)

    # ----- Trainable policy (initialized from base) -----
    policy = PolicyNet(in_channels=3, channels=64, num_blocks=6, board_size=BOARD_SIZE)
    policy.load_state_dict(base_state)   # base에서 초기화
    policy.to(device)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # ----- Style classifier (fixed) -----
    style_model = StyleClassifier()
    style_state = torch.load(style_model_path, map_location="cpu")
    style_model.load_state_dict(style_state)
    style_model.to(device)
    style_model.eval()
    for p in style_model.parameters():
        p.requires_grad_(False)

    # ----- RL + BC 학습 루프 -----
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_pg_loss = 0.0
        total_bc_loss = 0.0
        total_reward_raw = 0.0
        total_kl = 0.0
        n_batches = 0

        for states, expert_actions in loader:
            states = states.to(device)                # (B,3,19,19)
            expert_actions = expert_actions.to(device)  # (B,)

            # 1) 합법수 마스크
            legal_mask = compute_legal_mask(states)  # (B,361) bool

            # 2) policy / base logits
            logits = policy(states)          # (B,361)
            with torch.no_grad():
                base_logits = base_policy(states)  # (B,361)

            # 3) 불법수 마스킹
            logits = logits.masked_fill(~legal_mask, -INF)
            base_logits = base_logits.masked_fill(~legal_mask, -INF)

            # 4) 확률 분포
            probs = F.softmax(logits, dim=-1)       # (B,361)
            base_probs = F.softmax(base_logits, dim=-1)

            # 5) 행동 샘플링 (컨텍스추얼 밴딧)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()                 # (B,)
            log_probs = dist.log_prob(actions)      # (B,)

            # 6) 스타일 보상 (Shin일 확률)
            reward_raw = compute_style_prob(style_model, states, actions)  # (B,)
            if normalize_rewards:
                rewards = (reward_raw - reward_raw.mean()) / (reward_raw.std() + 1e-8)
            else:
                rewards = reward_raw

            # 7) KL(π_rl || π_base)
            log_probs_rl = torch.log(probs + 1e-12)
            log_probs_base = torch.log(base_probs + 1e-12)
            kl_per_state = torch.sum(
                probs * (log_probs_rl - log_probs_base), dim=-1
            )  # (B,)
            kl = kl_per_state.mean()

            # 8) Policy gradient loss (스타일 RL)
            pg_loss = -(rewards * log_probs).mean()

            # 9) Behavior cloning loss (프로 imitation)
            #    -> "기력"을 유지시키는 역할
            bc_loss = F.cross_entropy(logits, expert_actions)

            # 10) 최종 loss: PG + KL + BC
            loss = pg_loss + lambda_kl * kl + lambda_bc * bc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pg_loss += pg_loss.item()
            total_bc_loss += bc_loss.item()
            total_reward_raw += reward_raw.mean().item()
            total_kl += kl.item()
            n_batches += 1

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={total_loss/n_batches:.4f}, "
            f"pg_loss={total_pg_loss/n_batches:.4f}, "
            f"bc_loss={total_bc_loss/n_batches:.4f}, "
            f"avg_reward(p_shin)={total_reward_raw/n_batches:.4f}, "
            f"avg_KL={total_kl/n_batches:.4f}"
        )

    torch.save(policy.state_dict(), out_path)
    print(f"[INFO] RL-updated policy saved to {out_path}")


# -------------------------------------------------------
# 4. CLI entrypoint
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/base_policy_100k.npz")
    parser.add_argument("--base_model", type=str, default="models/policy_bc_100k.pt")
    parser.add_argument("--style_model", type=str, default="models/style_classifier.pt")
    parser.add_argument("--out", type=str, default="models/policy_style_rl.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lambda_kl", type=float, default=0.1)
    parser.add_argument("--lambda_bc", type=float, default=1.0)
    parser.add_argument("--no_norm_reward", action="store_true",
                        help="스타일 보상 정규화 끄기")
    args = parser.parse_args()

    train_style_rl(
        data_npz=args.data,
        base_model_path=args.base_model,
        style_model_path=args.style_model,
        out_path=args.out,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device_str=args.device,
        lambda_kl=args.lambda_kl,
        lambda_bc=args.lambda_bc,
        normalize_rewards=not args.no_norm_reward,
    )


if __name__ == "__main__":
    main()
