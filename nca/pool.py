"""Sample pool (Distill, experiment 2 "What persists, exists").

Training batches are drawn from a pool of previous end states instead of always
starting from the seed, so the model also learns to keep a grown pattern stable.
The pool lives on the CPU; batches are moved to the training device.
"""
from __future__ import annotations

import torch


class SamplePool:
    def __init__(self, seed: torch.Tensor, size: int = 1024) -> None:
        self.x = seed.detach().cpu().unsqueeze(0).repeat(size, 1, 1, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def sample(self, n: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self), (n,), generator=generator)
        return idx, self.x[idx].clone()

    def commit(self, idx: torch.Tensor, x: torch.Tensor) -> None:
        self.x[idx] = x.detach().cpu()
