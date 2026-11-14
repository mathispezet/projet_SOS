"""Target images, seeds and colour conversions.

Targets follow the Distill setup: an RGBA emoji (Noto Emoji, 128 px) resized to
``size`` x ``size``, RGB premultiplied by alpha, zero-padded by ``pad`` on every
side (40 px + 16 px padding -> 72 x 72 grid).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EMOJI_CODES = {
    "lizard": "1f98e",
    "mushroom": "1f344",
    "smiley": "1f600",
    "butterfly": "1f98b",
    "fish": "1f420",
    "star": "2b50",
}
# The png/ folder is gone from the noto-emoji main branch; use a tagged release, then Google's CDN.
NOTO_URLS = (
    "https://raw.githubusercontent.com/googlefonts/noto-emoji/v2.042/png/128/emoji_u{code}.png",
    "https://fonts.gstatic.com/s/e/notoemoji/latest/{code}/128.png",
)
SYNTHETIC = ("disc",)


def fetch_emoji(name: str, cache_dir: str | Path = "assets/targets") -> Path:
    """Download a Noto emoji PNG once and cache it locally."""
    code = EMOJI_CODES[name]
    path = Path(cache_dir) / f"{name}_u{code}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for url in NOTO_URLS:
            try:
                with urllib.request.urlopen(url.format(code=code), timeout=30) as resp:
                    path.write_bytes(resp.read())
                break
            except Exception as exc:  # try the next mirror
                last_error = exc
        else:
            raise RuntimeError(f"could not download emoji '{name}' ({last_error}); use --target disc offline")
    return path


def synthetic_disc(size: int = 40) -> np.ndarray:
    """Offline fallback target: a green disc with a dark ring and a light dot (RGBA in [0, 1])."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / (size / 2.0)
    img = np.zeros((size, size, 4), dtype=np.float32)
    inside = r <= 0.95
    ring = (r > 0.7) & inside
    dot = np.sqrt((xx - c - size * 0.15) ** 2 + (yy - c + size * 0.15) ** 2) <= size * 0.08
    img[inside] = (0.18, 0.52, 0.35, 1.0)
    img[ring] = (0.08, 0.33, 0.18, 1.0)
    img[dot] = (0.60, 0.90, 0.70, 1.0)
    return img


def load_rgba(path: str | Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def prepare_target(name: str, size: int = 40, pad: int = 16, cache_dir: str | Path = "assets/targets") -> np.ndarray:
    """Return the padded, premultiplied RGBA target as a float32 array (H, W, 4)."""
    if name in SYNTHETIC:
        rgba = synthetic_disc(size)
    elif name in EMOJI_CODES:
        rgba = load_rgba(fetch_emoji(name, cache_dir), size)
    elif Path(name).exists():
        rgba = load_rgba(name, size)
    else:
        raise ValueError(f"unknown target '{name}': use one of {list(EMOJI_CODES) + list(SYNTHETIC)} or a PNG path")
    rgba = rgba.copy()
    rgba[..., :3] *= rgba[..., 3:]  # premultiply
    return np.pad(rgba, ((pad, pad), (pad, pad), (0, 0)))


def target_to_tensor(target: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    """(H, W, 4) numpy -> (1, 4, H, W) tensor."""
    return torch.from_numpy(np.ascontiguousarray(target)).permute(2, 0, 1).unsqueeze(0).to(device)


def make_seed(h: int, w: int, channel_n: int = 16) -> torch.Tensor:
    """Single seed cell in the centre: alpha and hidden channels set to 1, RGB to 0. Shape (C, H, W)."""
    seed = torch.zeros(channel_n, h, w)
    seed[3:, h // 2, w // 2] = 1.0
    return seed


def to_rgba(x: torch.Tensor) -> torch.Tensor:
    return x[:, :4]


def to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Composite premultiplied RGBA over a white background -> (B, 3, H, W) in [0, 1]."""
    rgb, a = x[:, :3], x[:, 3:4].clamp(0.0, 1.0)
    return (1.0 - a + rgb).clamp(0.0, 1.0)


def per_sample_mse(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error on the RGBA channels, one value per sample. ``target`` is (1, 4, H, W)."""
    return ((to_rgba(x) - target) ** 2).mean(dim=(1, 2, 3))
