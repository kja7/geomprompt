"""Depth corruption operators for broken-depth robustness training/evaluation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

CORRUPTION_NAMES = (
    "quantize",
    "hole",
    "dropout",
    "noise",
    "blur",
    "banding",
    "scale_shift",
)


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    severity: float


def clamp_severity(severity: float) -> float:
    return float(min(max(float(severity), 0.0), 1.0))


def sample_corruption(
    rng: np.random.Generator,
    names: Sequence[str],
    severity_min: float,
    severity_max: float,
) -> CorruptionSpec:
    if not names:
        raise ValueError("No corruption names available for sampling")
    name = str(names[int(rng.integers(0, len(names)))])
    lo = clamp_severity(severity_min)
    hi = clamp_severity(severity_max)
    if hi < lo:
        lo, hi = (hi, lo)
    severity = float(rng.uniform(lo, hi)) if hi > lo else lo
    return CorruptionSpec(name=name, severity=severity)


def _box_blur(depth: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return depth
    pad = k // 2
    src = depth.astype(np.float32)
    padded = np.pad(src, ((pad, pad), (pad, pad)), mode="edge")
    integral = (
        np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    )
    out = (
        integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    ) / float(k * k)
    return out.astype(np.float32)


def apply_depth_corruption(
    depth_u8: np.ndarray, spec: CorruptionSpec, rng: np.random.Generator
) -> np.ndarray:
    """Apply one corruption to a single-channel uint8 depth image."""
    if depth_u8.ndim != 2:
        raise ValueError(f"Expected depth_u8 with shape (H, W), got {depth_u8.shape}")
    name = spec.name
    s = clamp_severity(spec.severity)
    depth = depth_u8.astype(np.float32)
    if name == "quantize":
        bins = int(round(64 - 60 * s))
        bins = max(2, bins)
        q = np.round(depth / 255.0 * (bins - 1)) / float(bins - 1)
        out = q * 255.0
    elif name == "hole":
        out = depth.copy()
        h, w = out.shape
        num_holes = max(1, int(round(2 + 18 * s)))
        for _ in range(num_holes):
            hole_h = int(
                rng.integers(max(4, int(0.02 * h)), max(5, int((0.06 + 0.22 * s) * h)))
            )
            hole_w = int(
                rng.integers(max(4, int(0.02 * w)), max(5, int((0.06 + 0.22 * s) * w)))
            )
            y0 = int(rng.integers(0, max(1, h - hole_h + 1)))
            x0 = int(rng.integers(0, max(1, w - hole_w + 1)))
            out[y0 : y0 + hole_h, x0 : x0 + hole_w] = 0.0
    elif name == "dropout":
        p = 0.01 + 0.39 * s
        mask = rng.random(depth.shape) < p
        out = depth.copy()
        out[mask] = 0.0
    elif name == "noise":
        sigma = 2.0 + 34.0 * s
        out = depth + rng.normal(0.0, sigma, size=depth.shape).astype(np.float32)
    elif name == "blur":
        kernel = int(round(1 + 8 * s))
        out = _box_blur(depth, kernel)
    elif name == "banding":
        out = depth.copy()
        h, _ = out.shape
        n_bands = max(2, int(round(4 + 20 * s)))
        band_h = max(1, h // n_bands)
        for bi in range(n_bands):
            y0 = bi * band_h
            y1 = h if bi == n_bands - 1 else (bi + 1) * band_h
            offset = float(rng.uniform(-25.0 * s, 25.0 * s))
            scale = float(rng.uniform(1.0 - 0.2 * s, 1.0 + 0.2 * s))
            out[y0:y1] = out[y0:y1] * scale + offset
    elif name == "scale_shift":
        scale = float(rng.uniform(1.0 - 0.35 * s, 1.0 + 0.35 * s))
        shift = float(rng.uniform(-50.0 * s, 50.0 * s))
        out = depth * scale + shift
    else:
        raise ValueError(f"Unsupported corruption: {name}")
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def u16_depth_to_u8(depth_u16: np.ndarray) -> np.ndarray:
    if depth_u16.ndim != 2:
        raise ValueError(f"Expected uint16 depth (H, W), got {depth_u16.shape}")
    depth = depth_u16.astype(np.float32) / 256.0
    return np.clip(depth, 0.0, 255.0).astype(np.uint8)
