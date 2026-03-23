import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IMAGE_SIZE


def generate_shepp_logan_phantom(
    image_size: int = IMAGE_SIZE,
    modified: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a 2D Shepp-Logan phantom on [-1,1]^2."""
    size = int(image_size)
    if size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    if modified:
        ellipses = [
            (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
            (-0.8, 0.6624, 0.8740, 0.0, -0.0184, 0.0),
            (-0.2, 0.1100, 0.3100, 0.22, 0.0, -18.0),
            (-0.2, 0.1600, 0.4100, -0.22, 0.0, 18.0),
            (0.1, 0.2100, 0.2500, 0.0, 0.35, 0.0),
            (0.1, 0.0460, 0.0460, 0.0, 0.10, 0.0),
            (0.1, 0.0460, 0.0460, 0.0, -0.10, 0.0),
            (0.1, 0.0460, 0.0230, -0.08, -0.605, 0.0),
            (0.1, 0.0230, 0.0230, 0.0, -0.606, 0.0),
            (0.1, 0.0230, 0.0460, 0.06, -0.605, 0.0),
        ]
    else:
        ellipses = [
            (2.0, 0.69, 0.92, 0.0, 0.0, 0.0),
            (-0.98, 0.6624, 0.8740, 0.0, -0.0184, 0.0),
            (-0.02, 0.1100, 0.3100, 0.22, 0.0, -18.0),
            (-0.02, 0.1600, 0.4100, -0.22, 0.0, 18.0),
            (0.01, 0.2100, 0.2500, 0.0, 0.35, 0.0),
            (0.01, 0.0460, 0.0460, 0.0, 0.10, 0.0),
            (0.02, 0.0460, 0.0460, 0.0, -0.10, 0.0),
            (0.01, 0.0460, 0.0230, -0.08, -0.605, 0.0),
            (0.01, 0.0230, 0.0230, 0.0, -0.606, 0.0),
            (0.01, 0.0230, 0.0460, 0.06, -0.605, 0.0),
        ]

    y = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phantom = torch.zeros((size, size), device=device, dtype=dtype)

    for intensity, axis_x, axis_y, center_x, center_y, angle_deg in ellipses:
        angle = math.radians(float(angle_deg))
        c = math.cos(angle)
        s = math.sin(angle)
        x_shift = xx - float(center_x)
        y_shift = yy - float(center_y)
        x_rot = x_shift * c + y_shift * s
        y_rot = -x_shift * s + y_shift * c
        inside = (x_rot / float(axis_x)) ** 2 + (y_rot / float(axis_y)) ** 2 <= 1.0
        phantom = phantom + float(intensity) * inside.to(dtype)

    return phantom.clamp_min(0.0)


def generate_random_ellipse_phantom(
    image_size: int = IMAGE_SIZE,
    n_ellipses: Optional[int] = None,
) -> torch.Tensor:
    """Generate a random ellipse phantom on [-1,1]^2 using NumPy RNG."""
    size = int(image_size)
    if size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    if n_ellipses is None:
        n_ellipses = int(np.random.poisson(100))

    y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    phantom = np.zeros((size, size), dtype=np.float32)

    for _ in range(int(n_ellipses)):
        intensity = float((np.random.rand() - 0.3) * np.random.exponential(0.3))
        axis_x = max(float(np.random.exponential(0.2)), 0.01)
        axis_y = max(float(np.random.exponential(0.2)), 0.01)
        center_x = float(np.random.rand() - 0.5)
        center_y = float(np.random.rand() - 0.5)
        theta = float(np.random.rand() * 2.0 * np.pi)

        x_shift = x - center_x
        y_shift = y - center_y
        x_rot = x_shift * np.cos(theta) + y_shift * np.sin(theta)
        y_rot = -x_shift * np.sin(theta) + y_shift * np.cos(theta)
        inside = (x_rot / axis_x) ** 2 + (y_rot / axis_y) ** 2 <= 1.0
        phantom[inside] += intensity

    return torch.from_numpy(phantom)


class DifferentiableImageGenerator(nn.Module):
    """
    If coefficient grid matches image size, coefficients are pixels; otherwise bilinear resize.
    """

    def __init__(self, image_size: int = IMAGE_SIZE, coeff_grid: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)):
        super().__init__()
        self.H = image_size
        self.W = image_size
        if isinstance(coeff_grid, int):
            self.grid_H = self.grid_W = coeff_grid
        else:
            self.grid_H, self.grid_W = coeff_grid
        self.is_identity = (self.H == self.grid_H) and (self.W == self.grid_W)

    def forward(self, coeff_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeff_batch: (B,1,H,W), (B,H,W) or (B, H*W)
        Returns:
            images: (B,1,H,W)
        """
        if coeff_batch.dim() == 2:
            b = coeff_batch.shape[0]
            coeff_batch = coeff_batch.view(b, 1, self.grid_H, self.grid_W)
        elif coeff_batch.dim() == 3:
            coeff_batch = coeff_batch.unsqueeze(1)

        if self.is_identity:
            return coeff_batch

        return F.interpolate(
            coeff_batch, size=(self.H, self.W), mode="bilinear", align_corners=True
        )


__all__ = [
    "generate_random_ellipse_phantom",
    "generate_shepp_logan_phantom",
    "DifferentiableImageGenerator",
]
