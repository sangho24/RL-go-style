import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import GoPositionDataset
from models import PolicyNet


def topk_from_logits(logits: torch.Tensor, k: int = 5):
    """
    logits: (1, 361) 기준
    return: indices (k,), probs (k,)
    """
    probs = torch.softmax(logits, dim=-1)  # (1, 361)
    topk_probs, topk_idx = torch.topk(probs, k=k, dim=-1)
    return topk_idx[0].cpu().numpy(), topk_probs[0].cpu().numpy()


def action_to_rc(action: int, board_size: int = 19):
    r = action // board_size
    c = action % board_size
    return int(r), int(c)


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """
    D_KL( softmax(p) || softmax(q) )
    p_logits, q_logits: (1, 361)
    """
    with torch.no_grad():
        p = torch.softmax(p_logits, dim=-1)             # (1, 361)
        log_p = torch.log(p + 1e-8)
        log_q = torch.log_softmax(q_logits, dim=-1)
        kl = (p * (log_p - log_q)).sum(dim=-1)          # (1,)
        return float(kl.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/base_policy_100k.npz",
                        help="npz dataset used for sampling positions")
    parser.add_argument("--base_model", type=str, default="models/policy_bc_100k.pt")
    parser.add_argument("--shin_model", type=str, default="models/policy_shin_bc_16k.pt")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--use_val_split", action="store_true",
                        help="sample from validation split instead of whole dataset")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # ----- Dataset & loader -----
    full_ds = GoPositionDataset(args.data)
    n_total = len(full_ds)
    print(f"[Data] total samples in {args.data}: {n_total}")

    if args.use_val_split:
        n_val = max(1, int(math.floor(n_total * 0.1)))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        eval_ds = val_ds
        print(f"[Data] using val split: val={len(val_ds)}")
    else:
        eval_ds = full_ds

    # 무작위 샘플 인덱스 선택
    num_samples = min(args.num_samples, len(eval_ds))
    indices = torch.randperm(len(eval_ds))[:num_samples].tolist()

    # ----- Models -----
    base_model = PolicyNet(in_channels=3, channels=64, num_blocks=6, board_size=19)
    shin_model = PolicyNet(in_channels=3, channels=64, num_blocks=6, board_size=19)

    base_state = torch.load(args.base_model, map_location=device)
    shin_state = torch.load(args.shin_model, map_location=device)

    base_model.load_state_dict(base_state)
    shin_model.load_state_dict(shin_state)

    base_model.to(device).eval()
    shin_model.to(device).eval()

    # ----- Compare -----
    diff_top1 = 0
    total_kl = 0.0

    print("========== Sample-wise Policy Comparison ==========")
    for idx in indices:
        x, y = eval_ds[idx]
        x = x.unsqueeze(0).to(device)  # (1, C, 19, 19)
        y = int(y)

        with torch.no_grad():
            logits_base = base_model(x)   # (1, 361)
            logits_shin = shin_model(x)   # (1, 361)

        # top-1
        base_top1 = int(torch.argmax(logits_base, dim=-1).item())
        shin_top1 = int(torch.argmax(logits_shin, dim=-1).item())
        if base_top1 != shin_top1:
            diff_top1 += 1

        # KL divergence
        kl = kl_divergence(logits_shin, logits_base)
        total_kl += kl

        # top-k 리스트 출력
        k = args.topk
        base_idx, base_p = topk_from_logits(logits_base, k=k)
        shin_idx, shin_p = topk_from_logits(logits_shin, k=k)

        y_r, y_c = action_to_rc(y)
        base_r, base_c = action_to_rc(base_top1)
        shin_r, shin_c = action_to_rc(shin_top1)

        print("--------------------------------------------------")
        print(f"[Sample idx {idx}]")
        print(f"  True move (label) : a={y} -> (r={y_r}, c={y_c})")
        print(f"  Base top-1        : a={base_top1} -> (r={base_r}, c={base_c})")
        print(f"  Shin top-1        : a={shin_top1} -> (r={shin_r}, c={shin_c})")
        print(f"  KL( Shin || Base ): {kl:.4f}")
        print(f"  Base top-{k}:")
        for i in range(k):
            r, c = action_to_rc(base_idx[i])
            print(f"    rank {i+1}: a={base_idx[i]:3d} (r={r:2d}, c={c:2d}), p={base_p[i]:.4f}")
        print(f"  Shin top-{k}:")
        for i in range(k):
            r, c = action_to_rc(shin_idx[i])
            print(f"    rank {i+1}: a={shin_idx[i]:3d} (r={r:2d}, c={c:2d}), p={shin_p[i]:.4f}")

    frac_diff = diff_top1 / num_samples
    avg_kl = total_kl / num_samples

    print("==================================================")
    print(f"[Summary] num_samples         : {num_samples}")
    print(f"[Summary] top-1 diff fraction: {frac_diff:.3f}")
    print(f"[Summary] mean KL(Shin||Base): {avg_kl:.4f}")
    print("==================================================")
    

if __name__ == "__main__":
    main()
