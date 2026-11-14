"""Neural Cellular Automaton model.

State tensor layout: ``(B, C, H, W)`` with channels 0-2 = RGB (premultiplied by
alpha), channel 3 = alpha (cell "maturity"), channels 4.. = hidden state.

One CA step (``forward``):
  1. perceive  — fixed (or learned) 3x3 filters applied per channel: identity,
                 Sobel x, Sobel y [, Laplacian]  -> C*K local features
  2. update    — per-cell MLP (1x1 convs): C*K -> hidden -> C, last layer
                 zero-initialised so the untrained rule does nothing
  3. stochastic update — each cell applies its update with probability
                 ``fire_rate`` (no global clock)
  4. alive masking — a cell with no "mature" neighbour (alpha > 0.1 in its 3x3
                 neighbourhood, before or after the update) is reset to zero
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IDENTITY = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
SOBEL_Y = SOBEL_X.t().contiguous()
LAPLACIAN = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0

PERCEPTION_KERNELS: dict[str, list[torch.Tensor]] = {
    "sobel": [IDENTITY, SOBEL_X, SOBEL_Y],
    "sobel_lap": [IDENTITY, SOBEL_X, SOBEL_Y, LAPLACIAN],
    # "learned": same initialisation as "sobel", but the kernels are trainable
    "learned": [IDENTITY, SOBEL_X, SOBEL_Y],
}


class CAModel(nn.Module):
    def __init__(
        self,
        channel_n: int = 16,
        hidden_n: int = 128,
        fire_rate: float = 0.5,
        perception: str = "sobel",
        alive_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        if perception not in PERCEPTION_KERNELS:
            raise ValueError(f"unknown perception '{perception}', expected one of {list(PERCEPTION_KERNELS)}")
        self.channel_n = channel_n
        self.hidden_n = hidden_n
        self.fire_rate = fire_rate
        self.perception = perception
        self.alive_threshold = alive_threshold

        kernels = torch.stack(PERCEPTION_KERNELS[perception])  # (K, 3, 3)
        self.kernel_n = kernels.shape[0]
        # depthwise weight (C*K, 1, 3, 3): output channels [c*K, (c+1)*K) read input channel c
        weight = kernels.repeat(channel_n, 1, 1).unsqueeze(1).clone()
        if perception == "learned":
            self.filters = nn.Parameter(weight)
        else:
            self.register_buffer("filters", weight)

        self.rule = nn.Sequential(
            nn.Conv2d(channel_n * self.kernel_n, hidden_n, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_n, channel_n, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.rule[2].weight)  # "do nothing" initial behaviour

    # ---- components -----------------------------------------------------
    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.filters, padding=1, groups=self.channel_n)

    def alive_mask(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x[:, 3:4], kernel_size=3, stride=1, padding=1) > self.alive_threshold

    # ---- one CA step ------------------------------------------------------
    def forward(self, x: torch.Tensor, fire_rate: float | None = None, step_size: float = 1.0) -> torch.Tensor:
        fire_rate = self.fire_rate if fire_rate is None else fire_rate
        pre_life = self.alive_mask(x)
        dx = self.rule(self.perceive(x)) * step_size
        update = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fire_rate).to(x.dtype)
        x = x + dx * update
        post_life = self.alive_mask(x)
        return x * (pre_life & post_life).to(x.dtype)

    def run(self, x: torch.Tensor, steps: int, **kwargs) -> torch.Tensor:
        for _ in range(steps):
            x = self(x, **kwargs)
        return x

    def describe(self) -> dict:
        return {
            "channel_n": self.channel_n,
            "hidden_n": self.hidden_n,
            "fire_rate": self.fire_rate,
            "perception": self.perception,
            "alive_threshold": self.alive_threshold,
            "parameters": sum(p.numel() for p in self.parameters()),
        }
