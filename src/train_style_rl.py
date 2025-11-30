import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GoPositionDataset
from models import PolicyNet
from train_style_classifier import StyleClassifier  # 같은 폴더에 있다고 가정


def kl_divergence(pi_logits, base_logits):
    """
    D_KL( pi || base ) for each sample.
    pi_logits, base_logits: (B, 361)
    return: (B,)
    """
    pi_log_probs = torch.log_softmax(pi_logits, dim=-1)        # (B,361)
    base_log_probs = torch.log_softmax(base_logits, dim=-1)    # (B,361)
    pi_probs = torch.softmax(pi_logits, dim=-1)

    kl = (pi_probs * (pi_log_probs - base_log_probs)).sum(dim=-1)  # (B,)
    return kl


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
):

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # ----- Dataset (contexts only) -----
    ds = GoPositionDataset(data_npz)
    n_total = len(ds)
    print(f"[Data] using {n_total} states from {data_npz}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # ----- Base policy (frozen) -----
    base_policy = PolicyNet(in_channels=3, channels=64, num_blocks=6, board_size=19)
    base_state = torch.load(base_model_path, map_location=device)
    base_policy.load_state_dict(base_state)
    base_policy.to(device)
    base_policy.eval()
    for p in base_policy.parameters():
        p.requires_grad_(False)

    # ----- Trainable policy (initialized from base) -----
    policy = PolicyNet(in_channels=3, channels=64, num_blocks=6, board_size=19)
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

    for epoch in range(1, epochs + 1):
        policy.train()
        total_loss = 0.0
        total_reward = 0.0
        total_kl = 0.0
        total_samples = 0

        for x, y in loader:
            # x: (B,3,19,19), y: (B,)  (y는 여기선 안 씀)
            x = x.to(device)

            # 정책에서 action 샘플링
            logits_pi = policy(x)                        # (B,361)
            pi_probs = torch.softmax(logits_pi, dim=-1)
            dist = torch.distributions.Categorical(pi_probs)
            actions = dist.sample()                      # (B,)
            logpi_a = dist.log_prob(actions)            # (B,)

            with torch.no_grad():
                # 스타일 점수 = P(Shin | s, a)
                style_logits = style_model(x, actions)
                style_score = torch.sigmoid(style_logits)   # (B,)

                # KL(pi || base)
                base_logits = base_policy(x)
                kl = kl_divergence(logits_pi, base_logits)  # (B,)

            # Advantage = reward - baseline(배치 평균)
            reward = style_score
            baseline = reward.mean()
            advantage = reward - baseline

            # Policy gradient loss (REINFORCE)
            policy_loss = -(advantage * logpi_a).mean()

            # KL penalty
            kl_loss = kl.mean()
            loss = policy_loss + lambda_kl * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_reward += reward.mean().item() * bsz
            total_kl += kl_loss.item() * bsz
            total_samples += bsz

        avg_loss = total_loss / total_samples
        avg_reward = total_reward / total_samples
        avg_kl = total_kl / total_samples

        print(f"[Epoch {epoch:02d}] "
              f"loss={avg_loss:.4f}, "
              f"avg_reward={avg_reward:.4f}, "
              f"avg_KL={avg_kl:.4f}")

    torch.save(policy.state_dict(), out_path)
    print(f"[INFO] RL-updated policy saved to {out_path}")


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
    )


if __name__ == "__main__":
    main()
