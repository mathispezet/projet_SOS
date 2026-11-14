"""Shared helpers: seeding, devices, damage masks, image and plot output."""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ---- damage -----------------------------------------------------------------
def make_circle_masks(
    n: int, h: int, w: int, r_min: float = 0.1, r_max: float = 0.4, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """(n, h, w) float masks, 1 inside a random circle. Coordinates span [-1, 1], centre in
    [-0.5, 0.5], radius in [r_min, r_max] (Distill's ``make_circle_masks``)."""
    x = torch.linspace(-1.0, 1.0, w, device=device)[None, None, :]
    y = torch.linspace(-1.0, 1.0, h, device=device)[None, :, None]
    center = torch.rand(2, n, 1, 1, device=device) - 0.5
    r = torch.rand(n, 1, 1, device=device) * (r_max - r_min) + r_min
    x = (x - center[0]) / r
    y = (y - center[1]) / r
    return (x * x + y * y < 1.0).to(torch.float32)


def circle_mask_at(h: int, w: int, cx: float, cy: float, r: float, device: torch.device | str = "cpu") -> torch.Tensor:
    """(1, h, w) mask of a circle with explicit centre (cx, cy) in [-1, 1] coordinates and radius r."""
    x = torch.linspace(-1.0, 1.0, w, device=device)[None, None, :]
    y = torch.linspace(-1.0, 1.0, h, device=device)[None, :, None]
    return (((x - cx) / r) ** 2 + ((y - cy) / r) ** 2 < 1.0).to(torch.float32)


def erase_half(x: torch.Tensor, side: str = "right") -> torch.Tensor:
    """Zero one half of every state in the batch (columns for left/right, rows for top/bottom)."""
    x = x.clone()
    h, w = x.shape[-2:]
    if side == "right":
        x[..., :, w // 2:] = 0.0
    elif side == "left":
        x[..., :, : w // 2] = 0.0
    elif side == "bottom":
        x[..., h // 2:, :] = 0.0
    elif side == "top":
        x[..., : h // 2, :] = 0.0
    else:
        raise ValueError(side)
    return x


def cell_dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    """Zero each cell (all channels) independently with probability p."""
    keep = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) >= p).to(x.dtype)
    return x * keep


# ---- images and plots -------------------------------------------------------
def tensor_to_uint8(rgb: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [0, 1] -> (H, W, 3) uint8."""
    arr = rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def frames_to_strip(frames: list[np.ndarray], scale: int = 3, gap: int = 4, labels: list[str] | None = None) -> Image.Image:
    """Concatenate (H, W, 3) uint8 frames horizontally, upscaled with nearest neighbour."""
    h, w = frames[0].shape[:2]
    cell_w, cell_h = w * scale, h * scale
    label_h = 14 if labels else 0
    strip = Image.new("RGB", (len(frames) * (cell_w + gap) - gap, cell_h + label_h), "white")
    for i, f in enumerate(frames):
        img = Image.fromarray(f).resize((cell_w, cell_h), Image.NEAREST)
        strip.paste(img, (i * (cell_w + gap), label_h))
    if labels:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(strip)
        for i, text in enumerate(labels):
            draw.text((i * (cell_w + gap) + 2, 1), text, fill=(60, 60, 60))
    return strip


def plot_curves(
    path: str | Path,
    curves: dict[str, np.ndarray],
    xlabel: str = "CA step",
    ylabel: str = "MSE (RGBA)",
    title: str = "",
    logy: bool = True,
    vlines: list[int] | None = None,
    bands: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    for name, ys in curves.items():
        xs = np.arange(len(ys))
        ax.plot(xs, ys, label=name, linewidth=1.5)
        if bands and name in bands:
            lo, hi = bands[name]
            ax.fill_between(xs, lo, hi, alpha=0.2)
    if logy:
        ax.set_yscale("log")
    for v in vlines or []:
        ax.axvline(v, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(curves) > 1:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
