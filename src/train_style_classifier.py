import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def load_npz_policy(path: str):
    """
    npz에서 (X, y) 로딩.
    - shin_policy_50k.npz: states, actions
    - base_policy_100k.npz: X, y  또는 states, actions
    둘 다 대응하도록 한다.
    """
    data = np.load(path)
    keys = set(data.keys())
    # 1) states / actions 형태
    if "states" in keys and "actions" in keys:
        X = data["states"].astype(np.float32)
        y = data["actions"].astype(np.int64)
    # 2) X / y 형태
    elif "X" in keys and "y" in keys:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
    else:
        raise ValueError(
            f"{path} npz 파일에서 인식 가능한 키를 찾지 못했어요. "
            f"keys={list(keys)}"
        )
    return X, y


class StyleDataset(Dataset):
    """
    Shin / Base 데이터 둘 다 받아서:
      - X: (N, 3, 19, 19)
      - y: (N,) 0~360
      - label: 1 (Shin), 0 (Base)
    """
    def __init__(self, shin_npz: str, base_npz: str, max_base_ratio: float = 1.0):
        # Shin 데이터
        X_shin, y_shin = load_npz_policy(shin_npz)
        # Base 데이터
        X_base, y_base = load_npz_policy(base_npz)

        n_shin = X_shin.shape[0]
        n_base = X_base.shape[0]

        # Base를 Shin 대비 max_base_ratio 배까지만 사용 (imbalance 완화)
        max_base = int(n_shin * max_base_ratio)
        if n_base > max_base:
            idx = np.random.RandomState(42).choice(n_base, size=max_base, replace=False)
            X_base = X_base[idx]
            y_base = y_base[idx]
        n_base = X_base.shape[0]

        self.X = np.concatenate([X_shin, X_base], axis=0)
        self.actions = np.concatenate([y_shin, y_base], axis=0)
        self.labels = np.concatenate(
            [np.ones(n_shin, dtype=np.int64), np.zeros(n_base, dtype=np.int64)],
            axis=0,
        )

        # 셔플
        perm = np.random.RandomState(42).permutation(self.X.shape[0])
        self.X = self.X[perm]
        self.actions = self.actions[perm]
        self.labels = self.labels[perm]

        print(f"[StyleDataset] shin={n_shin}, base={n_base}, total={len(self.X)}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]          # (3, 19, 19)
        a = self.actions[idx]    # scalar
        label = self.labels[idx] # 0/1
        return x, a, label


class StyleClassifier(nn.Module):
    """
    입력: board (B,3,19,19), actions (B,)
    - action 위치를 plane으로 만들어서 채널 concat → (B,4,19,19)
    출력: logit (B,)  (sigmoid(logit) = P(Shin style))
    """
    def __init__(self, in_channels=4, hidden_channels=64, board_size=19):
        super().__init__()
        self.board_size = board_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * board_size * board_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, board, actions):
        """
        board: (B, 3, 19, 19)
        actions: (B,) int
        """
        B = board.size(0)
        device = board.device

        # action plane
        action_plane = torch.zeros((B, 1, self.board_size, self.board_size), device=device)
        rows = actions // self.board_size
        cols = actions % self.board_size
        action_plane[torch.arange(B, device=device), 0, rows, cols] = 1.0

        x = torch.cat([board, action_plane], dim=1)  # (B,4,19,19)
        x = self.conv(x)
        logit = self.head(x).squeeze(-1)  # (B,)
        return logit


def train_style_classifier(
    shin_npz: str,
    base_npz: str,
    out_path: str,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    device_str: str = "cuda",
    max_base_ratio: float = 1.0,
):

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    dataset = StyleDataset(shin_npz, base_npz, max_base_ratio=max_base_ratio)

    n_total = len(dataset)
    n_val = max(1, int(math.floor(n_total * 0.1)))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = StyleClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, a, y in train_loader:
            x = x.to(device)
            a = a.to(device)
            y = y.float().to(device)

            logits = model(x, a)         # (B,)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += (preds == y.long()).sum().item()
            total_samples += x.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for x, a, y in val_loader:
                x = x.to(device)
                a = a.to(device)
                y = y.float().to(device)

                logits = model(x, a)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += (preds == y.long()).sum().item()
                val_samples += x.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"[INFO] Style classifier saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shin_data", type=str, default="data/processed/shin_policy_50k.npz")
    parser.add_argument("--base_data", type=str, default="data/processed/base_policy_100k.npz")
    parser.add_argument("--out", type=str, default="models/style_classifier.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_base_ratio", type=float, default=1.0)
    args = parser.parse_args()

    train_style_classifier(
        shin_npz=args.shin_data,
        base_npz=args.base_data,
        out_path=args.out,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device_str=args.device,
        max_base_ratio=args.max_base_ratio,
    )


if __name__ == "__main__":
    main()
