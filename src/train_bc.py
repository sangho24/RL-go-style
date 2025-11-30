# src/train_bc.py
import argparse
import math
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from dataset import GoPositionDataset
from models import PolicyNet


def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """
    logits: (B, num_classes)
    targets: (B,)
    """
    with torch.no_grad():
        _, pred = torch.topk(logits, k=k, dim=1)  # (B, k)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred)).any(dim=1).float()
        return correct.mean().item()


def create_dataloaders(
    npz_path: str,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    dataset = GoPositionDataset(npz_path)
    n_total = len(dataset)
    n_val = max(1, int(math.floor(n_total * val_ratio)))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"[Data] train: {n_train}, val: {n_val}")
    return train_loader, val_loader


def train_bc(
    npz_path: str,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: str = "models/policy_bc.pt",
):
    # 디바이스 설정
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"
    device = torch.device(device)
    print(f"[Info] Using device: {device}")

    # 데이터 로더
    train_loader, val_loader = create_dataloaders(npz_path, batch_size=batch_size)

    # 모델 / 손실 / 옵티마이저
    model = PolicyNet(in_channels=3, channels=64, num_blocks=6, board_size=19)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ----- Train -----
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_train_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)             # (B, 361)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            acc = accuracy_topk(logits, y, k=1)

            running_loss += loss.item()
            running_acc += acc
            n_train_batches += 1

        train_loss = running_loss / n_train_batches
        train_acc = running_acc / n_train_batches

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_acc1 = 0.0
        val_acc3 = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()
                val_acc1 += accuracy_topk(logits, y, k=1)
                val_acc3 += accuracy_topk(logits, y, k=3)
                n_val_batches += 1

        val_loss /= n_val_batches
        val_acc1 /= n_val_batches
        val_acc3 /= n_val_batches

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f}, train_acc@1={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc@1={val_acc1:.4f}, val_acc@3={val_acc3:.4f}"
        )

        # best 모델 저장
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            save_path = save_path
            # 디렉토리 없으면 생성
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[Info] New best model saved to {save_path} (val_acc@1={best_val_acc:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save", type=str, default="models/policy_bc.pt")
    args = parser.parse_args()

    train_bc(
        npz_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save,
    )
