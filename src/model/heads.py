import torch
from torch import nn


class MotionHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)
