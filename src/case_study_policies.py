import argparse
import numpy as np
import torch
import torch.nn.functional as F

from models import PolicyNet  # 네가 이미 쓰고 있는 PolicyNet 그대로 import


def action_to_rc(a, board_size=19):
    r = int(a // board_size)
    c = int(a % board_size)
    return r, c


def print_case(
    idx,
    state_np,
    true_a,
    base_topk,
    shin_topk,
    rl_topk,
    board_size=19,
):
    print("-" * 50)
    print(f"[Case #{idx}]")
    r_t, c_t = action_to_rc(true_a, board_size)
    print(f"  True move (Shin): a={true_a} -> (r={r_t}, c={c_t})")

    def _print_model(name, topk):
        idxs, probs = topk
        print(f"  {name} top-3:")
        for rank, (a, p) in enumerate(zip(idxs, probs), start=1):
            r, c = action_to_rc(a, board_size)
            print(f"    rank {rank}: a={a:3d} -> (r={r:2d}, c={c:2d}), p={p:.4f}")

    _print_model("Base", base_topk)
    _print_model("Shin-BC", shin_topk)
    _print_model("RL", rl_topk)


def run_case_study(
    shin_npz: str,
    base_model_path: str,
    shin_model_path: str,
    rl_model_path: str,
    num_samples: int = 5,
    device_str: str = "cuda",
    seed: int = 42,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 1) Shin 데이터 불러오기
    data = np.load(shin_npz)
    if "states" in data and "actions" in data:
        states = data["states"].astype(np.float32)  # (N,3,19,19)
        actions = data["actions"].astype(np.int64)
    else:
        raise ValueError(f"{shin_npz} 안에 'states', 'actions' 키가 없어요. keys={list(data.keys())}")

    N = states.shape[0]
    print(f"[Info] Loaded Shin dataset from {shin_npz}: N={N}")

    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(N, size=min(num_samples, N), replace=False)
    print(f"[Info] Sampling indices: {sample_indices}")

    # 2) 모델 로드
    def load_policy(path):
        model = PolicyNet(in_channels=3, channels=64, num_blocks=6, board_size=19)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    print(f"[Info] Loading base policy from {base_model_path}")
    base_policy = load_policy(base_model_path)

    print(f"[Info] Loading Shin-BC policy from {shin_model_path}")
    shin_policy = load_policy(shin_model_path)

    print(f"[Info] Loading RL policy from {rl_model_path}")
    rl_policy = load_policy(rl_model_path)

    # 3) 각 케이스에 대해 top-3 비교
    for count, idx in enumerate(sample_indices, start=1):
        state_np = states[idx]       # (3,19,19)
        true_a = int(actions[idx])

        state = torch.from_numpy(state_np[None, ...]).to(device)  # (1,3,19,19)

        with torch.no_grad():
            logits_base = base_policy(state)  # (1,361)
            logits_shin = shin_policy(state)
            logits_rl = rl_policy(state)

            probs_base = F.softmax(logits_base, dim=-1)[0]  # (361,)
            probs_shin = F.softmax(logits_shin, dim=-1)[0]
            probs_rl = F.softmax(logits_rl, dim=-1)[0]

            # top-3
            base_p, base_idx = torch.topk(probs_base, k=3, dim=-1)
            shin_p, shin_idx = torch.topk(probs_shin, k=3, dim=-1)
            rl_p, rl_idx = torch.topk(probs_rl, k=3, dim=-1)

        base_topk = (base_idx.cpu().numpy(), base_p.cpu().numpy())
        shin_topk = (shin_idx.cpu().numpy(), shin_p.cpu().numpy())
        rl_topk = (rl_idx.cpu().numpy(), rl_p.cpu().numpy())

        print_case(count, state_np, true_a, base_topk, shin_topk, rl_topk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shin_data", type=str, default="data/processed/shin_policy_50k.npz")
    parser.add_argument("--base_model", type=str, default="models/policy_bc_100k.pt")
    parser.add_argument("--shin_model", type=str, default="models/policy_shin_bc_16k.pt")
    parser.add_argument("--rl_model", type=str, default="models/policy_style_rl.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_case_study(
        shin_npz=args.shin_data,
        base_model_path=args.base_model,
        shin_model_path=args.shin_model,
        rl_model_path=args.rl_model,
        num_samples=args.num_samples,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
