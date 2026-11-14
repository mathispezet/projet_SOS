"""nca — Neural Cellular Automata for growth and self-repair.

PyTorch reimplementation of "Growing Neural Cellular Automata" (Mordvintsev,
Randazzo, Niklasson, Levin; Distill 2020) with the ablations and robustness
metrics of the Self-Organizing Systems course project (UPB, 2026-2027).
"""

__version__ = "0.1.0"

from .model import CAModel  # noqa: F401
from .pool import SamplePool  # noqa: F401
