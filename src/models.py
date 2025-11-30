# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out, inplace=True)
        return out


class PolicyNet(nn.Module):
    """
    간단한 ResNet 스타일 정책 네트워크.
    입력: (B, C, 19, 19)
    출력: 각 위치(0~360)의 logit (B, 361)
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 64,
        num_blocks: int = 6,
        board_size: int = 19,
    ):
        super().__init__()
        self.board_size = board_size
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels))
        self.blocks = nn.Sequential(*blocks)

        # policy head: 1x1 conv → flatten → 361차원
        self.conv_policy = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(2)
        # 2채널 * 19 * 19 = 722
        self.fc_policy = nn.Linear(2 * board_size * board_size, board_size * board_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = F.relu(out, inplace=True)

        out = self.blocks(out)

        # policy head
        out = self.conv_policy(out)
        out = self.bn_policy(out)
        out = F.relu(out, inplace=True)
        out = out.view(out.size(0), -1)      # (B, 2 * 19 * 19)
        logits = self.fc_policy(out)         # (B, 361)
        return logits
