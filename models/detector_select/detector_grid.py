"""Detector-grid sampling helpers for alpha-continuous CT operators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _direction_values(direction: Any) -> tuple[float, float]:
    if torch.is_tensor(direction):
        values = direction.detach().to(dtype=torch.float64, device="cpu").view(-1).tolist()
    else:
        values = np.asarray(direction, dtype=np.float64).reshape(-1).tolist()
    if len(values) != 2:
        raise ValueError(f"direction must have two entries, got {len(values)}.")
    a1, a2 = float(values[0]), float(values[1])
    if not (math.isfinite(a1) and math.isfinite(a2)):
        raise ValueError(f"direction must be finite, got {direction!r}.")
    if abs(a1) <= 1.0e-12 and abs(a2) <= 1.0e-12:
        raise ValueError("direction must be non-zero.")
    return a1, a2


def support_bounds_b1b1_from_direction(direction: Any) -> tuple[float, float]:
    """Return projection support bounds for ``\\phi=B1(x)B1(y)``."""
    a1, a2 = _direction_values(direction)
    values = (0.0, a1, a2, a1 + a2)
    return float(min(values)), float(max(values))


def detector_grid_interval(
    *,
    sorted_proj: Any,
    direction: Any,
    margin_ratio: float = 0.0,
) -> tuple[float, float]:
    """Return the detector-grid support interval after optional symmetric shrink."""
    proj_np = np.asarray(
        sorted_proj.detach().to(dtype=torch.float64, device="cpu").view(-1).numpy()
        if torch.is_tensor(sorted_proj)
        else sorted_proj,
        dtype=np.float64,
    ).reshape(-1)
    if proj_np.size <= 0:
        raise ValueError("sorted_proj must not be empty.")
    if not np.all(np.isfinite(proj_np)):
        raise ValueError("sorted_proj must contain only finite values.")
    margin = float(margin_ratio)
    if not math.isfinite(margin) or margin < 0.0 or margin >= 0.5:
        raise ValueError(f"margin_ratio must be finite and in [0, 0.5), got {margin_ratio!r}.")

    support_lo, support_hi = support_bounds_b1b1_from_direction(direction)
    t_min = float(proj_np[0]) + float(support_lo)
    t_max = float(proj_np[-1]) + float(support_hi)
    width = float(t_max - t_min)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"detector support interval must have positive width, got {width!r}.")
    if margin > 0.0:
        shrink = margin * width
        t_min += shrink
        t_max -= shrink
        if t_max <= t_min:
            raise ValueError("margin_ratio shrinks detector interval to non-positive width.")
    return float(t_min), float(t_max)


def make_support_detector_grid_sampling_points(
    *,
    sorted_proj: Any,
    direction: Any,
    num_detector_samples: int,
    detector_phase: float = 0.5,
    margin_ratio: float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Create detector-bin center samples over the full projected support.

    If either ``sorted_proj`` or ``direction`` is a torch tensor, the returned
    points are a ``torch.float64`` tensor on the same device as ``sorted_proj``
    when possible. Otherwise a NumPy ``float64`` array is returned.
    """
    n_det = int(num_detector_samples)
    if n_det <= 0:
        raise ValueError(f"num_detector_samples must be positive, got {num_detector_samples!r}.")
    phase = float(detector_phase)
    if not math.isfinite(phase) or phase <= 0.0 or phase >= 1.0:
        raise ValueError(f"detector_phase must be finite and in (0, 1), got {detector_phase!r}.")

    t_min, t_max = detector_grid_interval(
        sorted_proj=sorted_proj,
        direction=direction,
        margin_ratio=float(margin_ratio),
    )
    delta = float(t_max - t_min) / float(n_det)

    if torch.is_tensor(sorted_proj) or torch.is_tensor(direction):
        device = sorted_proj.device if torch.is_tensor(sorted_proj) else direction.device
        indices = torch.arange(n_det, dtype=torch.float64, device=device)
        return torch.tensor(t_min, dtype=torch.float64, device=device) + (indices + phase) * delta

    indices_np = np.arange(n_det, dtype=np.float64)
    return np.asarray(t_min + (indices_np + phase) * delta, dtype=np.float64)
