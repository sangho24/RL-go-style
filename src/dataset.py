# src/dataset.py
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class GoPositionDataset(Dataset):
    """
    npz 파일(states, actions)을 로드해서 (state, action) 샘플을 제공.
    - states: (N, C, H, W) float32
    - actions: (N,) int64 (0~360)
    """

    def __init__(self, npz_path: str):
        super().__init__()
        data = np.load(npz_path)
        self.states = data["states"]       # (N, C, H, W)
        self.actions = data["actions"]     # (N,)
        assert self.states.shape[0] == self.actions.shape[0]
        print(f"[GoPositionDataset] Loaded {self.states.shape[0]} samples from {npz_path}")

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.states[idx])          # float32
        y = torch.tensor(self.actions[idx], dtype=torch.long)
        return x, y
