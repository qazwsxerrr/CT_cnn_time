"""Shared B-spline utilities for phi(x, y) = B2(x) * B1(y)."""

from __future__ import annotations

import numpy as np
import torch


def integral_b2_torch(u: torch.Tensor) -> torch.Tensor:
    """Integral of cardinal B2 from 0 to u (piecewise polynomial)."""
    u = u.to(torch.float64)
    out = torch.zeros_like(u)

    m1 = (u > 0.0) & (u <= 1.0)
    out = torch.where(m1, 0.5 * u * u, out)

    m2 = (u > 1.0) & (u <= 2.0)
    out = torch.where(m2, 2.0 * u - 0.5 * u * u - 1.0, out)

    out = torch.where(u > 2.0, torch.ones_like(out), out)
    return out


def integral_b1_torch(u: torch.Tensor) -> torch.Tensor:
    """Integral of cardinal B1 from 0 to u."""
    u = u.to(torch.float64)
    return torch.clamp(u, min=0.0, max=1.0)


def b1_torch(u: torch.Tensor) -> torch.Tensor:
    """Cardinal B1: characteristic function of (0, 1]."""
    u = u.to(torch.float64)
    return ((u > 0.0) & (u <= 1.0)).to(torch.float64)


def b2_torch(u: torch.Tensor) -> torch.Tensor:
    """Cardinal B2: triangle on (0,2] with peak at 1."""
    u = u.to(torch.float64)
    out = torch.zeros_like(u)

    m1 = (u > 0.0) & (u <= 1.0)
    out = torch.where(m1, u, out)

    m2 = (u > 1.0) & (u <= 2.0)
    out = torch.where(m2, 2.0 - u, out)
    return out


def radon_phi_b2b1(s: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Time-domain kernel h(s) = R_alpha phi(s) for phi(x,y)=B2(x)*B1(y).

    Analytic line-integral for a (unit) normal direction alpha=(a1,a2).

    When |a1| and |a2| are non-zero:
      h(s) = (F(u_hi) - F(u_lo)) / |a2|,
    where u_lo=min(s/a1, (s-a2)/a1) and u_hi=max(s/a1, (s-a2)/a1), and F is the
    antiderivative of B2.

    Degenerate axis-aligned cases:
      - a2 == 0: h(s) = B2(s/a1) / |a1|
      - a1 == 0: h(s) = B1(s/a2) / |a2|
    """
    s = s.to(torch.float64)
    alpha = alpha.to(torch.float64).view(-1)
    if int(alpha.numel()) != 2:
        raise ValueError(f"alpha must have shape (2,), got {tuple(alpha.shape)}")

    a1 = float(alpha[0].item())
    a2 = float(alpha[1].item())
    eps = 1e-12
    if abs(a1) <= eps and abs(a2) <= eps:
        raise ValueError("alpha must be non-zero.")

    if abs(a2) <= eps:
        u = s / a1
        return b2_torch(u) / abs(a1)

    if abs(a1) <= eps:
        u = s / a2
        return b1_torch(u) / abs(a2)

    u0 = s / a1
    u1 = (s - a2) / a1
    u_lo = torch.minimum(u0, u1)
    u_hi = torch.maximum(u0, u1)
    return (integral_b2_torch(u_hi) - integral_b2_torch(u_lo)) / abs(a2)


def radon_phi_b1b1(s: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Time-domain kernel h(s) = R_alpha phi(s) for phi(x,y)=B1(x)*B1(y).

    Here B1 is the box function on (0, 1]. The Radon transform reduces to the
    length of the intersection between the line alpha·x = s and the unit square.
    """
    s = s.to(torch.float64)
    alpha = alpha.to(torch.float64).view(-1)
    if int(alpha.numel()) != 2:
        raise ValueError(f"alpha must have shape (2,), got {tuple(alpha.shape)}")

    a1 = float(alpha[0].item())
    a2 = float(alpha[1].item())
    eps = 1e-12
    if abs(a1) <= eps and abs(a2) <= eps:
        raise ValueError("alpha must be non-zero.")

    if abs(a2) <= eps:
        u = s / a1
        return b1_torch(u) / abs(a1)

    if abs(a1) <= eps:
        u = s / a2
        return b1_torch(u) / abs(a2)

    u0 = s / a1
    u1 = (s - a2) / a1
    u_lo = torch.minimum(u0, u1)
    u_hi = torch.maximum(u0, u1)
    return (integral_b1_torch(u_hi) - integral_b1_torch(u_lo)) / abs(a2)


def phi_support_bounds_b2b1(alpha: torch.Tensor) -> tuple[float, float]:
    """
    Return (L1, L2) such that supp(R_alpha phi) is contained in [L1, L2] for phi=B2(x)B1(y).

    For supp(phi)=[0,2]x[0,1], the projection bounds along alpha are:
      L1 = min(0, alpha2, 2*alpha1, 2*alpha1 + alpha2)
      L2 = max(0, alpha2, 2*alpha1, 2*alpha1 + alpha2)
    """
    a1 = float(alpha[0].item())
    a2 = float(alpha[1].item())
    vals = [0.0, a2, 2.0 * a1, 2.0 * a1 + a2]
    return float(min(vals)), float(max(vals))


def phi_support_bounds_b1b1(alpha: torch.Tensor) -> tuple[float, float]:
    """
    Return (L1, L2) such that supp(R_alpha phi) is contained in [L1, L2] for phi=B1(x)B1(y).

    For supp(phi)=[0,1]x[0,1], the projection bounds along alpha are:
      L1 = min(0, alpha2, alpha1, alpha1 + alpha2)
      L2 = max(0, alpha2, alpha1, alpha1 + alpha2)
    """
    a1 = float(alpha[0].item())
    a2 = float(alpha[1].item())
    vals = [0.0, a2, a1, a1 + a2]
    return float(min(vals)), float(max(vals))


def b1_numpy(t: np.ndarray) -> np.ndarray:
    """Cardinal B1 in NumPy form."""
    return ((t > 0.0) & (t <= 1.0)).astype(np.float32)


def b2_numpy(t: np.ndarray) -> np.ndarray:
    """Cardinal B2 in NumPy form."""
    out = np.zeros_like(t, dtype=np.float32)
    m1 = (t > 0.0) & (t <= 1.0)
    out[m1] = t[m1].astype(np.float32)
    m2 = (t > 1.0) & (t <= 2.0)
    out[m2] = (2.0 - t[m2]).astype(np.float32)
    return out


def synthesize_f_from_coeff_b2b1(
    coeff: np.ndarray,
    image_size: int,
    out_size: int = 256,
) -> np.ndarray:
    """
    Synthesize f(x,y) = sum_k c_k * phi((x,y)-k), with phi(x,y)=B2(x)*B1(y).

    This helper is intended for visualization only.
    """
    coeff = np.asarray(coeff, dtype=np.float32)
    if coeff.ndim != 2:
        raise ValueError(f"Expected coeff as 2D array, got shape {coeff.shape}")

    h, w = coeff.shape
    x = np.linspace(0.0, float(image_size - 1), int(out_size), dtype=np.float32)
    y = np.linspace(0.0, float(image_size - 1), int(out_size), dtype=np.float32)
    kx = np.arange(h, dtype=np.float32)
    ky = np.arange(w, dtype=np.float32)

    bx = b2_numpy(x[:, None] - kx[None, :])
    by = b1_numpy(y[:, None] - ky[None, :])
    f = (by @ coeff.T) @ bx.T
    return f.astype(np.float32)


def build_b2b1_synthesis_matrices(
    coeff_size: int,
    out_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build separable sampling matrices for phi(x,y)=B2(x)B1(y).

    Returns:
        bx: (out_size, coeff_size), basis values along x for B2
        by: (out_size, coeff_size), basis values along y for B1
    """
    coeff_size = int(coeff_size)
    out_size = int(out_size)
    if coeff_size <= 0 or out_size <= 0:
        raise ValueError(
            f"coeff_size and out_size must be positive, got coeff_size={coeff_size}, out_size={out_size}"
        )

    x = np.linspace(0.0, float(coeff_size - 1), out_size, dtype=np.float32)
    y = np.linspace(0.0, float(coeff_size - 1), out_size, dtype=np.float32)
    k = np.arange(coeff_size, dtype=np.float32)
    bx = b2_numpy(x[:, None] - k[None, :])
    by = b1_numpy(y[:, None] - k[None, :])
    return bx.astype(np.float32), by.astype(np.float32)


def fit_image_to_coeff_b2b1(
    image: np.ndarray,
    coeff_size: int,
) -> np.ndarray:
    """
    Least-squares fit of an image sampled on a regular grid to B2*B1 coefficients.

    The input image is interpreted as samples on [0, coeff_size-1]^2.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected image as 2D array, got shape {image.shape}")

    out_h, out_w = image.shape
    if out_h != out_w:
        raise ValueError(f"Expected square image, got shape {image.shape}")

    bx, by = build_b2b1_synthesis_matrices(coeff_size=int(coeff_size), out_size=int(out_h))
    design = np.empty((out_h * out_w, int(coeff_size) * int(coeff_size)), dtype=np.float32)
    col = 0
    for kx in range(int(coeff_size)):
        for ky in range(int(coeff_size)):
            design[:, col] = np.outer(by[:, ky], bx[:, kx]).reshape(-1)
            col += 1

    coeff_vec, _, _, _ = np.linalg.lstsq(design, image.reshape(-1), rcond=None)
    return coeff_vec.reshape(int(coeff_size), int(coeff_size)).astype(np.float32)


__all__ = [
    "integral_b2_torch",
    "integral_b1_torch",
    "b1_torch",
    "b2_torch",
    "radon_phi_b2b1",
    "radon_phi_b1b1",
    "phi_support_bounds_b2b1",
    "phi_support_bounds_b1b1",
    "b1_numpy",
    "b2_numpy",
    "build_b2b1_synthesis_matrices",
    "fit_image_to_coeff_b2b1",
    "synthesize_f_from_coeff_b2b1",
]
