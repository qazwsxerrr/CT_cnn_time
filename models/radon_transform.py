"""Alpha-continuous time-domain Radon operator and data generator.

For each continuous angle alpha in [0, pi), this module sorts lattice
projections

    s_k(alpha) = k1*cos(alpha) + k2*sin(alpha)

and samples at

    t_i = s_(i) + tau.

The single-angle matrix is

    A_alpha_tau[i,j] = R_alpha phi(s_(i) + tau - s_(j)).

Some comparison scripts may provide explicit per-angle sampling points t_i
instead of the shifted-lattice rule; the matrix rows are then assembled from
R_alpha phi(t_i - s_(j)) with the same sparse backend.

Multiple angles are stacked vertically and solved with Tikhonov / Morozov.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

from config import DATA_CONFIG, IMAGE_SIZE, TIME_DOMAIN_CONFIG, device
from image_generator import (
    DifferentiableImageGenerator,
    generate_random_ellipse_phantom,
    generate_shepp_logan_phantom,
)
try:
    from initialization_methods import (
        INIT_METHOD_CHOICES,
        MEASUREMENT_NORMALIZED_REGULARIZED_INIT_METHODS,
        normalize_init_method,
    )
except ImportError:  # pragma: no cover - supports package-style imports.
    from models.initialization_methods import (
        INIT_METHOD_CHOICES,
        MEASUREMENT_NORMALIZED_REGULARIZED_INIT_METHODS,
        normalize_init_method,
    )
from b_spline.b2b1_spline import (
    phi_support_bounds_b1b1,
    radon_phi_b1b1,
)

try:
    from detector_select.detector_grid import make_support_detector_grid_sampling_points
except ImportError:  # pragma: no cover - supports package-style imports.
    from models.detector_select.detector_grid import make_support_detector_grid_sampling_points


def _morozov_settings(max_iter: int, lambda_min: float, lambda_max: float) -> dict[str, float]:
    lam_min = max(float(lambda_min), float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12)))
    lam_max = min(float(lambda_max), float(DATA_CONFIG.get("morozov_lambda_max", 1.0e12)))
    if lam_max <= lam_min:
        lam_max = max(lam_min * 10.0, lam_min + 1.0)
    return {
        "max_iter": int(max_iter),
        "lambda_min": float(lam_min),
        "lambda_max": float(lam_max),
        "newton_tol": float(DATA_CONFIG.get("morozov_newton_tol", 1.0e-10)),
        "initial_lambda": float(DATA_CONFIG.get("morozov_initial_lambda", 1.0)),
    }


def _morozov_newton_scalar(
    residual2_fn,
    derivative_fn,
    target2: float,
    lambda_min: float,
    lambda_max: float,
    initial_lambda: float,
    max_iter: int,
    tol: float,
    min_residual2: float,
    max_residual2: float,
) -> float:
    eps = 1.0e-18
    if target2 <= (float(min_residual2) + eps):
        return float(lambda_min)
    if target2 >= (float(max_residual2) - eps):
        return float(lambda_max)

    lo = float(lambda_min)
    hi = max(float(initial_lambda), lo * 10.0, 1.0)
    hi = min(hi, float(lambda_max))

    phi_lo = float(residual2_fn(lo) - target2)
    if phi_lo >= 0.0:
        return float(lo)

    phi_hi = float(residual2_fn(hi) - target2)
    for _ in range(64):
        if phi_hi >= 0.0 or hi >= float(lambda_max):
            break
        lo = hi
        phi_lo = phi_hi
        hi = min(hi * 10.0, float(lambda_max))
        phi_hi = float(residual2_fn(hi) - target2)
    if phi_hi < 0.0:
        return float(hi)

    scale = max(1.0, float(target2))
    for _ in range(int(max_iter)):
        lam = math.sqrt(lo * hi) if lo > 0.0 else 0.5 * (lo + hi)
        value = float(residual2_fn(lam) - target2)
        if abs(value) <= float(tol) * scale:
            return float(lam)
        if value < 0.0:
            lo = lam
        else:
            hi = lam

    return float(math.sqrt(lo * hi) if lo > 0.0 else 0.5 * (lo + hi))


@torch.no_grad()
def _build_implicit_normal_matrix(operator, chunk_size: int = 64) -> torch.Tensor:
    n = int(operator.N)
    op_device = getattr(getattr(operator, "sampling_points", None), "device", device)
    eye = torch.eye(n, device=op_device, dtype=torch.float32)
    rows = []
    for start in range(0, n, int(chunk_size)):
        basis = eye[start : start + int(chunk_size)].view(-1, 1, operator.height, operator.width)
        rows.append(operator.apply_normal(basis).view(-1, n).detach().to(device="cpu", dtype=torch.float32))
    gram = torch.cat(rows, dim=0)
    return 0.5 * (gram + gram.t())


def _morozov_cache_path(cache_dir: str, fingerprint: dict[str, object]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    prefix = str(fingerprint.get("class_name", "operator")).lower()
    return os.path.join(cache_dir, f"{prefix}_gram_eigh_{digest}.pt")


def _ensure_implicit_gram_spectrum(operator, fingerprint: dict[str, object], chunk_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(operator, "_morozov_gram_eigvals", None) is not None and getattr(operator, "_morozov_gram_eigvecs", None) is not None:
        return operator._morozov_gram_eigvals, operator._morozov_gram_eigvecs

    cache_dir = str(getattr(operator, "_gram_cache_dir_override", None) or DATA_CONFIG.get("alpha_gram_cache_dir", "")).strip()
    if not cache_dir:
        raise ValueError("DATA_CONFIG['alpha_gram_cache_dir'] must be a non-empty path.")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _morozov_cache_path(cache_dir, fingerprint)
    setattr(operator, "_morozov_cache_path", cache_path)

    cache_hit = False
    build_seconds = None
    cached = None
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("fingerprint") == fingerprint:
            eigvals = cached["eigvals"].to(dtype=torch.float32, device="cpu")
            eigvecs = cached["eigvecs"].to(dtype=torch.float32, device="cpu")
            cache_hit = True
        else:
            cached = None

    if cached is None:
        started = time.perf_counter()
        print(f"[Morozov] building exact Gram spectrum cache: {cache_path}")
        gram = _build_implicit_normal_matrix(operator, chunk_size=chunk_size)
        eigvals, eigvecs = torch.linalg.eigh(gram)
        eigvals = eigvals.clamp_min_(0.0).to(dtype=torch.float32, device="cpu")
        eigvecs = eigvecs.to(dtype=torch.float32, device="cpu")
        torch.save({"fingerprint": fingerprint, "eigvals": eigvals, "eigvecs": eigvecs}, cache_path)
        build_seconds = float(time.perf_counter() - started)
        print(f"[Morozov] cached exact Gram spectrum in {build_seconds:.2f}s")
    else:
        print(f"[Morozov] loaded exact Gram spectrum cache: {cache_path}")

    operator._morozov_gram_eigvals = eigvals
    operator._morozov_gram_eigvecs = eigvecs
    operator.last_morozov_cache_hit = bool(cache_hit)
    operator.last_morozov_cache_build_seconds = build_seconds
    return eigvals, eigvecs


def _solve_tikhonov_from_gram_spectrum(
    rhs: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    lambda_reg: float | torch.Tensor,
    *,
    rhs_proj: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch = int(rhs.shape[0])
    rhs_cpu = rhs.detach().to(dtype=torch.float32, device="cpu")
    eigvals_cpu = eigvals.detach().to(dtype=torch.float32, device="cpu")
    eigvecs_cpu = eigvecs.detach().to(dtype=torch.float32, device="cpu")
    if rhs_proj is None:
        rhs_proj_cpu = rhs_cpu @ eigvecs_cpu
    else:
        rhs_proj_cpu = rhs_proj.detach().to(dtype=torch.float32, device="cpu")
    if torch.is_tensor(lambda_reg):
        lam_cpu = lambda_reg.detach().to(dtype=torch.float32, device="cpu").view(-1)
        if int(lam_cpu.numel()) == 1 and batch > 1:
            lam_cpu = lam_cpu.expand(batch)
        elif int(lam_cpu.numel()) != batch:
            raise ValueError(f"lambda_reg has {int(lam_cpu.numel())} entries, expected 1 or batch={batch}.")
    else:
        lam_cpu = torch.full((batch,), float(lambda_reg), dtype=torch.float32, device="cpu")
    denom = eigvals_cpu.view(1, -1) + lam_cpu.view(-1, 1)
    coeff = (rhs_proj_cpu / denom) @ eigvecs_cpu.t()
    return coeff.to(dtype=torch.float32, device=rhs.device)


def _choose_lambda_morozov_from_gram_spectrum(
    b: torch.Tensor,
    rhs: torch.Tensor,
    noise_norm: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    tau: float,
    settings: dict[str, float],
    *,
    rhs_proj: Optional[torch.Tensor] = None,
    b_norm2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if b.dim() == 1:
        b = b.unsqueeze(0)
    batch = int(b.shape[0])
    noise_norm = noise_norm.view(-1)
    if int(noise_norm.numel()) == 1 and batch > 1:
        noise_norm = noise_norm.expand(batch)

    b_cpu = b.detach().to(dtype=torch.float32, device="cpu")
    rhs_cpu = rhs.detach().to(dtype=torch.float32, device="cpu")
    noise_cpu = noise_norm.detach().to(dtype=torch.float64, device="cpu")
    eigvals_cpu = eigvals.detach().to(dtype=torch.float64, device="cpu").numpy()
    eigvecs_cpu = eigvecs.detach().to(dtype=torch.float32, device="cpu")
    if rhs_proj is None:
        rhs_proj_all = (rhs_cpu @ eigvecs_cpu).to(dtype=torch.float64)
    else:
        rhs_proj_all = rhs_proj.detach().to(dtype=torch.float32, device="cpu").to(dtype=torch.float64)

    lam_list = []
    for idx in range(batch):
        rhs_proj2 = rhs_proj_all[idx].square().numpy()
        if b_norm2 is None:
            sample_b_norm2 = float(torch.dot(b_cpu[idx], b_cpu[idx]).item())
        else:
            sample_b_norm2 = float(b_norm2.detach().to(dtype=torch.float32, device="cpu").view(-1)[idx].item())
        target2 = float(float(tau) * float(noise_cpu[idx].item())) ** 2

        def residual2_fn(lam: float) -> float:
            denom = eigvals_cpu + lam
            # Morozov's discrepancy principle is posed in measurement space:
            #     ||A x_lambda - b|| ~= tau * ||noise||.
            #
            # The stacked alpha operator is generally tall (M = K*N > N), so
            # b can contain a nonzero component orthogonal to Range(A).  A
            # coefficient-space/range-only expression such as
            # ``sum((lambda / (sigma^2 + lambda))^2 * rhs_proj^2 / sigma^2)``
            # drops that orthogonal residual and overestimates the lambda
            # needed to reach the target discrepancy.  Use the equivalent full
            # measurement-space identity instead:
            #     ||b||^2 - <rhs, x_lambda> - lambda ||x_lambda||^2.
            x_rhs = float(np.sum(rhs_proj2 / denom))
            x_norm2 = float(np.sum(rhs_proj2 / (denom * denom)))
            return max(0.0, sample_b_norm2 - x_rhs - (lam * x_norm2))

        def derivative_fn(lam: float) -> float:
            return float(2.0 * lam * np.sum(rhs_proj2 / ((eigvals_cpu + lam) ** 3)))

        lam_list.append(
            _morozov_newton_scalar(
                residual2_fn=residual2_fn,
                derivative_fn=derivative_fn,
                target2=target2,
                lambda_min=float(settings["lambda_min"]),
                lambda_max=float(settings["lambda_max"]),
                initial_lambda=float(settings["initial_lambda"]),
                max_iter=int(settings["max_iter"]),
                tol=float(settings["newton_tol"]),
                min_residual2=0.0,
                max_residual2=sample_b_norm2,
            )
        )
    return torch.tensor(lam_list, dtype=b.dtype, device=b.device)


def _soft_threshold(x: torch.Tensor, threshold: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(threshold):
        threshold = x.new_tensor(float(threshold))
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold.to(dtype=x.dtype, device=x.device), min=0.0)


def _project_l2_ball(x: torch.Tensor, radius: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(radius):
        radius = x.new_tensor(float(radius))
    radius = radius.to(dtype=x.dtype, device=x.device).view(-1).clamp_min(0.0)
    flat = x.reshape(x.shape[0], -1)
    norm = torch.norm(flat, dim=1).clamp_min(1.0e-12)
    scale = torch.minimum(torch.ones_like(norm), radius / norm)
    return (flat * scale.view(-1, 1)).view_as(x)


def _project_l1_ball(x: torch.Tensor, radius: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(radius):
        radius = x.new_tensor(float(radius))
    radius = radius.to(dtype=x.dtype, device=x.device).view(-1).clamp_min(0.0)
    flat = x.reshape(x.shape[0], -1)
    abs_flat = torch.abs(flat)
    l1_norm = torch.sum(abs_flat, dim=1)
    inside = l1_norm <= radius
    if bool(torch.all(inside)):
        return x
    sorted_abs, _ = torch.sort(abs_flat, dim=1, descending=True)
    cssv = torch.cumsum(sorted_abs, dim=1)
    arange = torch.arange(1, flat.shape[1] + 1, dtype=x.dtype, device=x.device).view(1, -1)
    cond = sorted_abs * arange > (cssv - radius.view(-1, 1))
    rho = torch.sum(cond.to(dtype=torch.int64), dim=1).clamp_min(1)
    theta = (cssv.gather(1, (rho - 1).view(-1, 1)).squeeze(1) - radius) / rho.to(dtype=x.dtype)
    projected = torch.sign(flat) * torch.clamp(abs_flat - theta.view(-1, 1), min=0.0)
    projected = torch.where(inside.view(-1, 1), flat, projected)
    projected = torch.where((radius <= 0.0).view(-1, 1), torch.zeros_like(projected), projected)
    return projected.view_as(x)


def _sparse_blocks_apply_batched(
    rows: torch.Tensor,
    cols: torch.Tensor,
    values: torch.Tensor,
    nnz: torch.Tensor,
    x: torch.Tensor,
    *,
    num_rows: Optional[int] = None,
) -> torch.Tensor:
    """Apply per-angle sparse projection-order matrices stored as padded COO buffers."""
    if x.dim() != 3:
        raise ValueError(f"x must have shape (B,K,N), got {tuple(x.shape)}")
    batch, num_angles, n_cols = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    out_rows = n_cols if num_rows is None else int(num_rows)
    if out_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {out_rows!r}.")
    y = torch.zeros((batch, num_angles, out_rows), dtype=x.dtype, device=x.device)
    for angle_idx in range(num_angles):
        count = int(nnz[angle_idx].item())
        if count <= 0:
            continue
        r = rows[angle_idx, :count].to(device=x.device)
        c = cols[angle_idx, :count].to(device=x.device)
        v = values[angle_idx, :count].to(dtype=x.dtype, device=x.device)
        contrib = x[:, angle_idx, :].index_select(1, c) * v.unsqueeze(0)
        y[:, angle_idx, :].index_add_(1, r, contrib)
    return y


def _sparse_blocks_adjoint_apply_batched(
    rows: torch.Tensor,
    cols: torch.Tensor,
    values: torch.Tensor,
    nnz: torch.Tensor,
    x: torch.Tensor,
    *,
    num_cols: Optional[int] = None,
) -> torch.Tensor:
    """Apply adjoints of per-angle sparse projection-order matrices stored as padded COO buffers."""
    if x.dim() != 3:
        raise ValueError(f"x must have shape (B,K,M_per_angle), got {tuple(x.shape)}")
    batch, num_angles, n_rows = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    out_cols = n_rows if num_cols is None else int(num_cols)
    if out_cols <= 0:
        raise ValueError(f"num_cols must be positive, got {out_cols!r}.")
    y = torch.zeros((batch, num_angles, out_cols), dtype=x.dtype, device=x.device)
    for angle_idx in range(num_angles):
        count = int(nnz[angle_idx].item())
        if count <= 0:
            continue
        r = rows[angle_idx, :count].to(device=x.device)
        c = cols[angle_idx, :count].to(device=x.device)
        v = values[angle_idx, :count].to(dtype=x.dtype, device=x.device)
        contrib = x[:, angle_idx, :].index_select(1, r) * v.unsqueeze(0)
        y[:, angle_idx, :].index_add_(1, c, contrib)
    return y


def _lex_lattice_indices(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lexicographical lattice order: k1 major, k2 minor."""
    k1 = torch.arange(int(height), dtype=torch.int64).repeat_interleave(int(width))
    k2 = torch.arange(int(width), dtype=torch.int64).repeat(int(height))
    return k1, k2


def _alpha_to_unit_direction(alpha: float) -> torch.Tensor:
    alpha = float(alpha) % math.pi
    return torch.tensor([math.cos(alpha), math.sin(alpha)], dtype=torch.float64)


def _alpha_projection_order(alpha: float, height: int, width: int, *, injective_tol: float = 1.0e-12) -> dict[str, torch.Tensor]:
    """Return projection-order metadata for continuous alpha sampling."""
    k1, k2 = _lex_lattice_indices(int(height), int(width))
    direction = _alpha_to_unit_direction(float(alpha))
    proj = k1.to(torch.float64) * direction[0] + k2.to(torch.float64) * direction[1]
    sort_idx = torch.argsort(proj, stable=True)
    sorted_proj = proj.index_select(0, sort_idx)
    gaps = torch.diff(sorted_proj)
    min_gap = float(gaps.min().item()) if int(gaps.numel()) > 0 else float("inf")
    if min_gap <= float(injective_tol):
        raise ValueError(
            f"alpha={float(alpha):.16f} is numerically non-injective on "
            f"{int(height)}x{int(width)}: min_gap={min_gap:.3e} <= {float(injective_tol):.3e}"
        )
    lex_to_order = torch.empty(int(height) * int(width), dtype=torch.int64)
    lex_to_order[sort_idx] = torch.arange(int(height) * int(width), dtype=torch.int64)
    return {
        "alpha": torch.tensor(float(alpha) % math.pi, dtype=torch.float64),
        "direction": direction,
        "proj_lex": proj,
        "sort_idx": sort_idx.to(torch.int64),
        "sorted_proj": sorted_proj,
        "lex_to_order": lex_to_order,
        "order_to_lex": sort_idx.to(torch.int64),
        "min_gap": torch.tensor(min_gap, dtype=torch.float64),
    }


def _build_sparse_b1b1_block_from_sampling_points(
    *,
    sorted_proj: torch.Tensor,
    direction: torch.Tensor,
    sampling_points: torch.Tensor,
    value_tol: float = 1.0e-15,
) -> dict[str, torch.Tensor]:
    """Build a sparse projection-order block for arbitrary time samples.

    ``sorted_proj`` always has length ``N = H*W`` and indexes coefficient
    columns. ``sampling_points`` may have a smaller length ``R``; in that case
    this constructs a rectangular ``R x N`` projection block.
    """
    sorted_proj = sorted_proj.detach().to(dtype=torch.float64, device="cpu")
    direction = direction.detach().to(dtype=torch.float64, device="cpu")
    sampling_points = sampling_points.detach().to(dtype=torch.float64, device="cpu").view(-1)
    proj_np = sorted_proj.numpy()
    sample_np = sampling_points.numpy()
    n_cols = int(proj_np.shape[0])
    n_rows = int(sample_np.shape[0])
    if n_cols <= 0:
        raise ValueError("sorted_proj must contain at least one coefficient projection.")
    if n_rows <= 0:
        raise ValueError("sampling_points must contain at least one detector sample.")
    support_lo, support_hi = phi_support_bounds_b1b1(direction)
    support_lo = float(support_lo)
    support_hi = float(support_hi)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    lower_width = 0
    upper_width = 0

    for row_idx in range(n_rows):
        t_i = float(sample_np[row_idx])
        left = t_i - support_hi
        right = t_i - support_lo
        col0 = int(np.searchsorted(proj_np, left, side="left"))
        col1 = int(np.searchsorted(proj_np, right, side="right"))
        if col1 <= col0:
            continue
        cols = np.arange(col0, col1, dtype=np.int64)
        diffs = torch.from_numpy(t_i - proj_np[col0:col1]).to(dtype=torch.float64)
        vals = radon_phi_b1b1(diffs, direction).detach().to(dtype=torch.float64, device="cpu").numpy()
        mask = np.abs(vals) > float(value_tol)
        if not np.any(mask):
            continue
        rows = np.full((int(np.count_nonzero(mask)),), row_idx, dtype=np.int64)
        cols = cols[mask]
        vals = vals[mask].astype(np.float64, copy=False)
        row_parts.append(rows)
        col_parts.append(cols)
        val_parts.append(vals)
        lower_width = max(lower_width, int(np.max(rows - cols, initial=0)) + 1)
        upper_width = max(upper_width, int(np.max(cols - rows, initial=0)) + 1)

    if row_parts:
        rows_np = np.concatenate(row_parts).astype(np.int64, copy=False)
        cols_np = np.concatenate(col_parts).astype(np.int64, copy=False)
        vals_np = np.concatenate(val_parts).astype(np.float64, copy=False)
    else:
        rows_np = np.empty((0,), dtype=np.int64)
        cols_np = np.empty((0,), dtype=np.int64)
        vals_np = np.empty((0,), dtype=np.float64)

    return {
        "sparse_rows": torch.from_numpy(rows_np),
        "sparse_cols": torch.from_numpy(cols_np),
        "sparse_values": torch.from_numpy(vals_np),
        "sparse_nnz": torch.tensor(int(vals_np.shape[0]), dtype=torch.int64),
        "num_rows": torch.tensor(int(n_rows), dtype=torch.int64),
        "num_cols": torch.tensor(int(n_cols), dtype=torch.int64),
        "lower_width": torch.tensor(int(lower_width), dtype=torch.int64),
        "upper_width": torch.tensor(int(upper_width), dtype=torch.int64),
        "diag0": torch.tensor(0.0, dtype=torch.float64),
        "support_lo": torch.tensor(float(support_lo), dtype=torch.float64),
        "support_hi": torch.tensor(float(support_hi), dtype=torch.float64),
    }


def uniform_sis_bin_ranges(n: int, num_bins: int) -> list[tuple[int, int]]:
    """Partition ``N`` sorted SIS row indices into non-empty detector bins."""
    n = int(n)
    num_bins = int(num_bins)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}.")
    if num_bins <= 0 or num_bins > n:
        raise ValueError(f"num_bins must be in [1,{n}], got {num_bins!r}.")
    ranges: list[tuple[int, int]] = []
    for bin_idx in range(num_bins):
        start = int(math.floor(float(bin_idx) * float(n) / float(num_bins)))
        end = int(math.floor(float(bin_idx + 1) * float(n) / float(num_bins)))
        if end <= start:
            raise RuntimeError(f"Failed to build a non-empty SIS bin {bin_idx}: [{start},{end}).")
        ranges.append((start, end))
    return ranges


def uniform_sis_row_indices(n: int, num_detector_samples: int) -> torch.Tensor:
    """Uniformly subsample sorted SIS row indices from the full shifted lattice."""
    n = int(n)
    num_detector_samples = int(num_detector_samples)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}.")
    if num_detector_samples <= 0 or num_detector_samples > n:
        raise ValueError(
            f"num_detector_samples must be in [1,{n}], got {num_detector_samples!r}."
        )
    if num_detector_samples == n:
        return torch.arange(n, dtype=torch.int64)
    indices = np.rint(np.linspace(0.0, float(n - 1), num_detector_samples)).astype(np.int64)
    for idx in range(1, int(indices.size)):
        if int(indices[idx]) <= int(indices[idx - 1]):
            indices[idx] = int(indices[idx - 1]) + 1
    overflow = int(indices[-1]) - (n - 1)
    if overflow > 0:
        indices -= overflow
        for idx in range(int(indices.size) - 2, -1, -1):
            if int(indices[idx]) >= int(indices[idx + 1]):
                indices[idx] = int(indices[idx + 1]) - 1
    if int(indices[0]) < 0 or int(indices[-1]) >= n or int(np.unique(indices).size) != num_detector_samples:
        raise RuntimeError(
            f"Failed to build {num_detector_samples} unique uniform SIS row indices from N={n}."
        )
    return torch.as_tensor(indices, dtype=torch.int64)


def _uniform_indices_in_half_open_range(start: int, end: int, count: int) -> torch.Tensor:
    """Return ``count`` rounded-linspace row indices in ``[start, end)``."""
    start = int(start)
    end = int(end)
    count = int(count)
    if count <= 0:
        return torch.empty((0,), dtype=torch.int64)
    if end <= start:
        raise ValueError(f"Invalid half-open range [{start}, {end}).")
    width = int(end - start)
    if count > width:
        raise ValueError(f"Cannot choose {count} unique indices from range width {width}.")
    if count == 1:
        return torch.tensor([int((start + end - 1) // 2)], dtype=torch.int64)
    values = np.rint(np.linspace(float(start), float(end - 1), count)).astype(np.int64)
    for idx in range(1, int(values.size)):
        if int(values[idx]) <= int(values[idx - 1]):
            values[idx] = int(values[idx - 1]) + 1
    overflow = int(values[-1]) - (end - 1)
    if overflow > 0:
        values -= overflow
        for idx in range(int(values.size) - 2, -1, -1):
            if int(values[idx]) >= int(values[idx + 1]):
                values[idx] = int(values[idx + 1]) - 1
    if int(values[0]) < start or int(values[-1]) >= end or int(np.unique(values).size) != count:
        raise RuntimeError(f"Failed to build {count} unique indices in [{start}, {end}).")
    return torch.as_tensor(values, dtype=torch.int64)


def edge_weighted_sis_region_counts(
    num_detector_samples: int,
    *,
    edge_weight: int = 3,
    middle_weight: int = 1,
) -> tuple[int, int, int]:
    """Return front/middle/back detector counts for an edge-weighted subset."""
    total = int(num_detector_samples)
    edge = int(edge_weight)
    middle = int(middle_weight)
    if total <= 0:
        raise ValueError(f"num_detector_samples must be positive, got {num_detector_samples!r}.")
    if edge <= 0 or middle <= 0:
        raise ValueError("edge_weight and middle_weight must be positive.")
    edge_count = int(math.floor(total * edge / float(2 * edge + middle)))
    middle_count = total - 2 * edge_count
    if edge_count <= 0 or middle_count <= 0:
        raise ValueError(f"Invalid region counts {(edge_count, middle_count, edge_count)!r}.")
    return int(edge_count), int(middle_count), int(edge_count)


def edge_weighted_sis_region_bounds(
    n: int,
    *,
    boundary_fraction: float = 0.2,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return front/middle/back half-open row ranges for a full SIS detector."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}.")
    fraction = float(boundary_fraction)
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 0.5:
        raise ValueError(f"boundary_fraction must be in (0, 0.5), got {boundary_fraction!r}.")
    left_end = int(round(n * fraction))
    right_start = n - left_end
    if left_end <= 0 or right_start <= left_end or right_start >= n:
        raise ValueError(
            f"Invalid region bounds for n={n}, boundary_fraction={fraction}: "
            f"left_end={left_end}, right_start={right_start}."
        )
    return (0, int(left_end)), (int(left_end), int(right_start)), (int(right_start), int(n))


def edge_weighted_sis_row_indices(
    *,
    n: int,
    num_detector_samples: int,
    boundary_fraction: float = 0.2,
    edge_weight: int = 3,
    middle_weight: int = 1,
) -> torch.Tensor:
    """Build edge-weighted shifted-lattice row indices.

    The default is the empirically tested 20% boundary split with a symmetric
    front:middle:back weight ratio of 3:1:3.  For 256 detector samples this
    gives counts 109:38:109.
    """
    n = int(n)
    total = int(num_detector_samples)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}.")
    if total <= 0 or total > n:
        raise ValueError(f"num_detector_samples must be in [1,{n}], got {total!r}.")
    front_count, middle_count, back_count = edge_weighted_sis_region_counts(
        total,
        edge_weight=int(edge_weight),
        middle_weight=int(middle_weight),
    )
    front_range, middle_range, back_range = edge_weighted_sis_region_bounds(
        n,
        boundary_fraction=float(boundary_fraction),
    )

    left = _uniform_indices_in_half_open_range(front_range[0], front_range[1], front_count)
    middle = _uniform_indices_in_half_open_range(middle_range[0], middle_range[1], middle_count)
    right = _uniform_indices_in_half_open_range(back_range[0], back_range[1], back_count)
    indices = torch.cat((left, middle, right), dim=0).to(dtype=torch.int64)
    if int(indices.numel()) != total:
        raise RuntimeError(f"Edge-weighted row construction returned {int(indices.numel())} rows, expected {total}.")
    if bool(torch.any(indices[1:] <= indices[:-1])):
        raise RuntimeError("Edge-weighted row construction did not produce strictly increasing indices.")
    return indices


def edge_weighted_sis_sampling_summary(
    *,
    n: int,
    num_angles: int,
    num_detector_samples: int,
    boundary_fraction: float = 0.2,
    edge_weight: int = 3,
    middle_weight: int = 1,
    include_indices: bool = False,
) -> dict[str, object]:
    """Summarize the main edge-weighted subset detector design.

    The current sparse-detector main flow uses a 20%/60%/20% split of the full
    shifted-lattice detector rows and allocates samples with a 3:1:3
    front:middle:back weight ratio.
    """
    n = int(n)
    num_angles = int(num_angles)
    total = int(num_detector_samples)
    front_count, middle_count, back_count = edge_weighted_sis_region_counts(
        total,
        edge_weight=int(edge_weight),
        middle_weight=int(middle_weight),
    )
    front_range, middle_range, back_range = edge_weighted_sis_region_bounds(
        n,
        boundary_fraction=float(boundary_fraction),
    )
    indices = edge_weighted_sis_row_indices(
        n=n,
        num_detector_samples=total,
        boundary_fraction=float(boundary_fraction),
        edge_weight=int(edge_weight),
        middle_weight=int(middle_weight),
    )
    summary: dict[str, object] = {
        "mode": "shifted_lattice_edge_weighted_subset",
        "selection_rule": (
            "split full sorted SIS rows into first 20%, middle 60%, last 20%; "
            "allocate rows as 3:1:3 by default"
        ),
        "N": int(n),
        "num_angles": int(num_angles),
        "num_detector_samples": int(total),
        "M_per_angle": int(total),
        "M": int(num_angles * total),
        "edge_boundary_fraction": float(boundary_fraction),
        "boundary_fraction": float(boundary_fraction),
        "edge_weight": int(edge_weight),
        "middle_weight": int(middle_weight),
        "front_middle_back_ratio": [int(edge_weight), int(middle_weight), int(edge_weight)],
        "region_bounds_half_open": {
            "front": [int(front_range[0]), int(front_range[1])],
            "middle": [int(middle_range[0]), int(middle_range[1])],
            "back": [int(back_range[0]), int(back_range[1])],
        },
        "region_counts_front_middle_back": [int(front_count), int(middle_count), int(back_count)],
        "first_indices": [int(v) for v in indices[: min(10, int(indices.numel()))].tolist()],
        "last_indices": [int(v) for v in indices[-min(10, int(indices.numel())) :].tolist()],
        "formula": (
            "t_i = sorted(k1*cos(alpha)+k2*sin(alpha))[r_i] + tau_star, "
            "r_i from 20%/60%/20% edge-weighted rows"
        ),
    }
    if include_indices:
        selected_rows = [int(v) for v in indices.tolist()]
        summary["selected_row_indices_per_angle"] = [list(selected_rows) for _ in range(int(num_angles))]
    return summary


def _coalesce_sparse_b1b1_block(
    *,
    rows: torch.Tensor,
    cols: torch.Tensor,
    values: torch.Tensor,
    num_rows: int,
    num_cols: int,
    value_tol: float,
    support_lo: torch.Tensor,
    support_hi: torch.Tensor,
    diag0: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Sum duplicate COO entries after detector-bin row aggregation."""
    num_rows = int(num_rows)
    num_cols = int(num_cols)
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError(f"num_rows and num_cols must be positive, got {num_rows!r}, {num_cols!r}.")
    rows = rows.detach().to(dtype=torch.int64, device="cpu").view(-1)
    cols = cols.detach().to(dtype=torch.int64, device="cpu").view(-1)
    values = values.detach().to(dtype=torch.float64, device="cpu").view(-1)
    if int(rows.numel()) != int(cols.numel()) or int(rows.numel()) != int(values.numel()):
        raise ValueError("rows, cols, and values must have the same number of entries.")
    if int(rows.numel()) == 0:
        out_rows = torch.empty((0,), dtype=torch.int64)
        out_cols = torch.empty((0,), dtype=torch.int64)
        out_values = torch.empty((0,), dtype=torch.float64)
    else:
        linear = rows * int(num_cols) + cols
        unique, inverse = torch.unique(linear, sorted=True, return_inverse=True)
        out_values = torch.zeros(int(unique.numel()), dtype=torch.float64)
        out_values.scatter_add_(0, inverse, values)
        out_rows = torch.div(unique, int(num_cols), rounding_mode="floor").to(dtype=torch.int64)
        out_cols = torch.remainder(unique, int(num_cols)).to(dtype=torch.int64)
        if float(value_tol) > 0.0:
            keep = torch.abs(out_values) > float(value_tol)
            out_rows = out_rows.index_select(0, torch.nonzero(keep, as_tuple=False).view(-1))
            out_cols = out_cols.index_select(0, torch.nonzero(keep, as_tuple=False).view(-1))
            out_values = out_values.index_select(0, torch.nonzero(keep, as_tuple=False).view(-1))

    if int(out_rows.numel()) > 0:
        lower_width = int(torch.clamp((out_rows - out_cols).max(), min=0).item()) + 1
        upper_width = int(torch.clamp((out_cols - out_rows).max(), min=0).item()) + 1
    else:
        lower_width = 0
        upper_width = 0

    return {
        "sparse_rows": out_rows,
        "sparse_cols": out_cols,
        "sparse_values": out_values,
        "sparse_nnz": torch.tensor(int(out_values.numel()), dtype=torch.int64),
        "num_rows": torch.tensor(int(num_rows), dtype=torch.int64),
        "num_cols": torch.tensor(int(num_cols), dtype=torch.int64),
        "lower_width": torch.tensor(int(lower_width), dtype=torch.int64),
        "upper_width": torch.tensor(int(upper_width), dtype=torch.int64),
        "diag0": torch.tensor(float(diag0), dtype=torch.float64),
        "support_lo": support_lo.detach().to(dtype=torch.float64, device="cpu"),
        "support_hi": support_hi.detach().to(dtype=torch.float64, device="cpu"),
    }


def _build_sparse_b1b1_block_from_binned_shifted_lattice(
    *,
    sorted_proj: torch.Tensor,
    direction: torch.Tensor,
    tau: float,
    num_detector_bins: int,
    value_tol: float = 1.0e-15,
) -> dict[str, torch.Tensor]:
    """Build detector-bin averages of the full shifted-lattice SIS rows."""
    sorted_proj = sorted_proj.detach().to(dtype=torch.float64, device="cpu")
    direction = direction.detach().to(dtype=torch.float64, device="cpu")
    n = int(sorted_proj.numel())
    num_detector_bins = int(num_detector_bins)
    ranges = uniform_sis_bin_ranges(n, num_detector_bins)
    fine_block = _build_sparse_b1b1_block_from_continuous_proj(
        sorted_proj=sorted_proj,
        direction=direction,
        tau=float(tau),
        value_tol=float(value_tol),
    )
    fine_to_bin = torch.empty(n, dtype=torch.int64)
    fine_weights = torch.empty(n, dtype=torch.float64)
    for bin_idx, (start, end) in enumerate(ranges):
        fine_to_bin[start:end] = int(bin_idx)
        fine_weights[start:end] = 1.0 / float(end - start)

    fine_rows = fine_block["sparse_rows"].to(dtype=torch.int64, device="cpu")
    binned_rows = fine_to_bin.index_select(0, fine_rows)
    binned_cols = fine_block["sparse_cols"].to(dtype=torch.int64, device="cpu")
    binned_values = fine_block["sparse_values"].to(dtype=torch.float64, device="cpu") * fine_weights.index_select(0, fine_rows)
    diag0 = float(fine_block["diag0"].item()) if num_detector_bins == n else 0.0
    return _coalesce_sparse_b1b1_block(
        rows=binned_rows,
        cols=binned_cols,
        values=binned_values,
        num_rows=int(num_detector_bins),
        num_cols=int(n),
        value_tol=float(value_tol),
        support_lo=fine_block["support_lo"],
        support_hi=fine_block["support_hi"],
        diag0=diag0,
    )


def _build_sparse_b1b1_block_from_continuous_proj(
    *,
    sorted_proj: torch.Tensor,
    direction: torch.Tensor,
    tau: float,
    value_tol: float = 1.0e-15,
) -> dict[str, torch.Tensor]:
    """Build the full sparse projection-order block for shifted-lattice sampling."""
    sorted_proj = sorted_proj.detach().to(dtype=torch.float64, device="cpu")
    direction = direction.detach().to(dtype=torch.float64, device="cpu")
    tau = float(tau)
    block = _build_sparse_b1b1_block_from_sampling_points(
        sorted_proj=sorted_proj,
        direction=direction,
        sampling_points=sorted_proj + float(tau),
        value_tol=float(value_tol),
    )
    diag0 = float(radon_phi_b1b1(torch.tensor([tau], dtype=torch.float64), direction)[0].item())
    block["diag0"] = torch.tensor(float(diag0), dtype=torch.float64)
    return block


def _sampling_points_digest(sampling_points: torch.Tensor) -> str:
    values = sampling_points.detach().to(dtype=torch.float64, device="cpu").contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _integer_tensor_digest(values: torch.Tensor) -> str:
    array = values.detach().to(dtype=torch.int64, device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


class AlphaContinuousB1B1Operator2D(torch.nn.Module):
    """Continuous-alpha B1*B1 operator with one sparse block per angle."""

    def __init__(
        self,
        alpha_values,
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        tau_offsets=None,
        sampling_points_per_angle=None,
        selected_row_indices_per_angle=None,
        t0: float = 0.5,
        injective_tol: float = 1.0e-12,
        sampling_mode: str = "shifted_lattice",
        num_detector_samples: int | None = None,
        detector_phase: float = 0.5,
        detector_margin_ratio: float = 0.0,
        subset_selection: str = "uniform",
        edge_boundary_fraction: float = 0.2,
        edge_weight: int = 3,
        middle_weight: int = 1,
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.N = int(self.height * self.width)
        self.alpha_values = [float(v) % math.pi for v in list(alpha_values or [])]
        if not self.alpha_values:
            raise ValueError("alpha_values must be a non-empty list of angles in radians.")
        self.num_angles = int(len(self.alpha_values))
        self.t0 = float(t0)
        requested_sampling_mode = str(sampling_mode or "shifted_lattice").strip().lower().replace("-", "_")
        if requested_sampling_mode in {"", "auto"}:
            requested_sampling_mode = "shifted_lattice"
        if requested_sampling_mode == "shifted_lattice_edge_weighted_subset":
            requested_sampling_mode = "shifted_lattice_subset"
            subset_selection = "edge_weighted"
        if requested_sampling_mode not in {"shifted_lattice", "shifted_lattice_subset", "shifted_lattice_binned", "ct_detector_grid", "custom_points"}:
            raise ValueError(
                f"Unknown sampling_mode={sampling_mode!r}; expected 'shifted_lattice', "
                "'shifted_lattice_subset', 'shifted_lattice_edge_weighted_subset', "
                "'shifted_lattice_binned', 'ct_detector_grid', or 'custom_points'."
            )
        self.detector_phase = float(detector_phase)
        self.detector_margin_ratio = float(detector_margin_ratio)
        self.subset_selection = str(subset_selection or "uniform").strip().lower().replace("-", "_")
        if self.subset_selection in {"edge", "edge20", "edge_weighted_20", "edge_weighted_20pct", "edge_weighted_3_1_3"}:
            self.subset_selection = "edge_weighted"
        if self.subset_selection not in {"uniform", "edge_weighted"}:
            raise ValueError("subset_selection must be 'uniform' or 'edge_weighted'.")
        self.edge_boundary_fraction = float(edge_boundary_fraction)
        self.edge_weight = int(edge_weight)
        self.middle_weight = int(middle_weight)
        default_detector_samples = 256 if requested_sampling_mode in {"shifted_lattice_subset", "shifted_lattice_binned", "ct_detector_grid"} else self.N
        self.num_detector_samples = int(default_detector_samples if num_detector_samples is None else num_detector_samples)
        if self.num_detector_samples <= 0:
            raise ValueError(f"num_detector_samples must be positive, got {self.num_detector_samples!r}.")
        self.formula_mode = "alpha_continuous"
        self.uses_sparse_blocks = True
        self._gram_cache_dir_override = str(
            DATA_CONFIG.get(
                "alpha_gram_cache_dir",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "alpha_gram_cache"),
            )
        )
        if tau_offsets is not None and sampling_points_per_angle is not None:
            raise ValueError("Specify either tau_offsets or sampling_points_per_angle, not both.")
        if sampling_points_per_angle is not None:
            requested_sampling_mode = "custom_points"
        if requested_sampling_mode in {"shifted_lattice_binned", "ct_detector_grid"} and selected_row_indices_per_angle is not None:
            raise ValueError("selected_row_indices_per_angle is only supported for shifted_lattice/custom_points modes.")
        if selected_row_indices_per_angle is not None and requested_sampling_mode == "shifted_lattice_subset":
            self.subset_selection = "manual"
        if tau_offsets is None:
            tau_list = None
        else:
            tau_list = [float(v) for v in list(tau_offsets)]
            if len(tau_list) != self.num_angles:
                raise ValueError(f"tau_offsets length={len(tau_list)} but num_angles={self.num_angles}.")
        self.tau_offsets = tau_list

        selected_row_indices_list_input = None
        selected_row_count: int | None = None
        if selected_row_indices_per_angle is not None:
            selected_row_indices_list_input = []
            for angle_idx, values in enumerate(list(selected_row_indices_per_angle)):
                tensor = torch.as_tensor(values, dtype=torch.int64).view(-1)
                current_count = int(tensor.numel())
                if current_count <= 0:
                    raise ValueError(f"selected_row_indices_per_angle[{angle_idx}] must not be empty.")
                if bool(torch.any(tensor < 0)) or bool(torch.any(tensor >= int(self.N))):
                    raise ValueError(
                        f"selected_row_indices_per_angle[{angle_idx}] contains indices outside [0,{int(self.N)})."
                    )
                if selected_row_count is None:
                    selected_row_count = current_count
                elif current_count != int(selected_row_count):
                    raise ValueError(
                        f"selected_row_indices_per_angle[{angle_idx}] length={int(tensor.numel())} "
                        f"but expected common per-angle length={int(selected_row_count)}."
                    )
                selected_row_indices_list_input.append(tensor)
            if len(selected_row_indices_list_input) != self.num_angles:
                raise ValueError(
                    f"selected_row_indices_per_angle length={len(selected_row_indices_list_input)} "
                    f"but num_angles={self.num_angles}."
                )
        elif requested_sampling_mode == "shifted_lattice_subset":
            if int(self.num_detector_samples) > int(self.N):
                raise ValueError(
                    f"num_detector_samples must be in [1,{int(self.N)}] for shifted_lattice_subset, "
                    f"got {int(self.num_detector_samples)}."
                )
            if self.subset_selection == "edge_weighted":
                row_indices = edge_weighted_sis_row_indices(
                    n=int(self.N),
                    num_detector_samples=int(self.num_detector_samples),
                    boundary_fraction=float(self.edge_boundary_fraction),
                    edge_weight=int(self.edge_weight),
                    middle_weight=int(self.middle_weight),
                )
            else:
                row_indices = uniform_sis_row_indices(int(self.N), int(self.num_detector_samples))
            selected_row_indices_list_input = [row_indices.clone() for _ in range(int(self.num_angles))]
            selected_row_count = int(row_indices.numel())

        if sampling_points_per_angle is None:
            sampling_points_list_input = None
            if requested_sampling_mode == "ct_detector_grid":
                self.sampling_mode = "ct_detector_grid"
            elif requested_sampling_mode == "shifted_lattice_binned":
                self.sampling_mode = "shifted_lattice_binned"
            elif requested_sampling_mode == "shifted_lattice_subset":
                self.sampling_mode = "shifted_lattice_subset"
            else:
                self.sampling_mode = "shifted_lattice" if selected_row_indices_list_input is None else "shifted_lattice_subset"
        else:
            sampling_points_list_input = []
            sample_count: int | None = None
            for angle_idx, values in enumerate(list(sampling_points_per_angle)):
                tensor = torch.as_tensor(values, dtype=torch.float64).view(-1)
                current_count = int(tensor.numel())
                if current_count <= 0:
                    raise ValueError(f"sampling_points_per_angle[{angle_idx}] must not be empty.")
                if sample_count is None:
                    sample_count = current_count
                elif current_count != int(sample_count):
                    raise ValueError(
                        f"sampling_points_per_angle[{angle_idx}] length={int(tensor.numel())} "
                        f"but expected common per-angle length={int(sample_count)}."
                    )
                sampling_points_list_input.append(tensor)
            if len(sampling_points_list_input) != self.num_angles:
                raise ValueError(
                    f"sampling_points_per_angle length={len(sampling_points_list_input)} "
                        f"but num_angles={self.num_angles}."
                )
            self.sampling_mode = "ct_detector_grid" if str(sampling_mode or "").strip().lower().replace("-", "_") == "ct_detector_grid" else "custom_points"
            if selected_row_indices_list_input is not None and int(selected_row_count or 0) != int(sampling_points_list_input[0].numel()):
                raise ValueError(
                    f"selected_row_indices_per_angle common length={int(selected_row_count or 0)} but "
                    f"sampling_points_per_angle common length={int(sampling_points_list_input[0].numel())}."
                )
            if self.sampling_mode == "ct_detector_grid" and int(sampling_points_list_input[0].numel()) != int(self.num_detector_samples):
                raise ValueError(
                    f"ct_detector_grid sampling_points_per_angle length={int(sampling_points_list_input[0].numel())} "
                    f"but num_detector_samples={int(self.num_detector_samples)}."
                )
        if sampling_points_list_input is not None:
            self.M_per_angle = int(sampling_points_list_input[0].numel())
        elif self.sampling_mode in {"shifted_lattice_binned", "ct_detector_grid"}:
            self.M_per_angle = int(self.num_detector_samples)
        elif sampling_points_list_input is None:
            self.M_per_angle = int(self.N if selected_row_indices_list_input is None else selected_row_count)
        self.M = int(self.num_angles * self.M_per_angle)

        with torch.no_grad():
            directions = []
            sorted_proj_list = []
            sampling_points_list = []
            lex_to_order_list = []
            order_to_lex_list = []
            min_gap_list = []
            support_lo_list = []
            support_hi_list = []
            blocks = []
            effective_tau = []
            selected_row_indices_list = []

            for angle_idx, alpha in enumerate(self.alpha_values):
                info = _alpha_projection_order(alpha, self.height, self.width, injective_tol=float(injective_tol))
                direction = info["direction"]
                support_lo, support_hi = phi_support_bounds_b1b1(direction)
                if self.sampling_mode == "ct_detector_grid" and sampling_points_list_input is None:
                    tau = float("nan")
                    selected_rows = torch.full((int(self.M_per_angle),), -1, dtype=torch.int64)
                    sampling_points = make_support_detector_grid_sampling_points(
                        sorted_proj=info["sorted_proj"],
                        direction=direction,
                        num_detector_samples=int(self.M_per_angle),
                        detector_phase=float(self.detector_phase),
                        margin_ratio=float(self.detector_margin_ratio),
                    )
                    block = _build_sparse_b1b1_block_from_sampling_points(
                        sorted_proj=info["sorted_proj"],
                        direction=direction,
                        sampling_points=sampling_points,
                    )
                elif self.sampling_mode == "shifted_lattice_binned" and sampling_points_list_input is None:
                    tau = float(support_lo) + float(t0) * (float(support_hi) - float(support_lo)) if tau_list is None else float(tau_list[angle_idx])
                    fine_sampling_points = info["sorted_proj"] + float(tau)
                    bin_ranges = uniform_sis_bin_ranges(int(self.N), int(self.M_per_angle))
                    representative_rows = []
                    binned_sampling_points = []
                    for start, end in bin_ranges:
                        representative_rows.append(int((start + end - 1) // 2))
                        binned_sampling_points.append(torch.mean(fine_sampling_points[start:end]))
                    selected_rows = torch.as_tensor(representative_rows, dtype=torch.int64)
                    sampling_points = torch.stack(binned_sampling_points).to(dtype=torch.float64)
                    block = _build_sparse_b1b1_block_from_binned_shifted_lattice(
                        sorted_proj=info["sorted_proj"],
                        direction=direction,
                        tau=tau,
                        num_detector_bins=int(self.M_per_angle),
                    )
                elif sampling_points_list_input is None:
                    tau = float(support_lo) + float(t0) * (float(support_hi) - float(support_lo)) if tau_list is None else float(tau_list[angle_idx])
                    if selected_row_indices_list_input is None:
                        selected_rows = torch.arange(int(self.N), dtype=torch.int64)
                        sampling_points = info["sorted_proj"] + float(tau)
                        block = _build_sparse_b1b1_block_from_continuous_proj(
                            sorted_proj=info["sorted_proj"],
                            direction=direction,
                            tau=tau,
                        )
                    else:
                        selected_rows = selected_row_indices_list_input[angle_idx]
                        sampling_points = info["sorted_proj"].index_select(0, selected_rows) + float(tau)
                        block = _build_sparse_b1b1_block_from_sampling_points(
                            sorted_proj=info["sorted_proj"],
                            direction=direction,
                            sampling_points=sampling_points,
                        )
                else:
                    tau = float("nan")
                    sampling_points = sampling_points_list_input[angle_idx]
                    if selected_row_indices_list_input is None:
                        selected_rows = torch.full((int(self.M_per_angle),), -1, dtype=torch.int64)
                    else:
                        selected_rows = selected_row_indices_list_input[angle_idx]
                    block = _build_sparse_b1b1_block_from_sampling_points(
                        sorted_proj=info["sorted_proj"],
                        direction=direction,
                        sampling_points=sampling_points,
                    )
                directions.append(direction)
                sorted_proj_list.append(info["sorted_proj"])
                sampling_points_list.append(sampling_points)
                lex_to_order_list.append(info["lex_to_order"])
                order_to_lex_list.append(info["order_to_lex"])
                min_gap_list.append(info["min_gap"])
                support_lo_list.append(torch.tensor(float(support_lo), dtype=torch.float64))
                support_hi_list.append(torch.tensor(float(support_hi), dtype=torch.float64))
                blocks.append(block)
                effective_tau.append(float(tau) if math.isfinite(float(tau)) else 0.0)
                selected_row_indices_list.append(selected_rows.to(dtype=torch.int64))

            sparse_nnz = torch.stack([blk["sparse_nnz"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            max_nnz = int(sparse_nnz.max().item()) if int(sparse_nnz.numel()) > 0 else 0
            sparse_rows = torch.zeros((self.num_angles, max_nnz), dtype=torch.int64, device=device)
            sparse_cols = torch.zeros((self.num_angles, max_nnz), dtype=torch.int64, device=device)
            sparse_values = torch.zeros((self.num_angles, max_nnz), dtype=torch.float32, device=device)
            for idx, block in enumerate(blocks):
                count = int(block["sparse_nnz"].item())
                if count > 0:
                    sparse_rows[idx, :count] = block["sparse_rows"].to(dtype=torch.int64, device=device)
                    sparse_cols[idx, :count] = block["sparse_cols"].to(dtype=torch.int64, device=device)
                    sparse_values[idx, :count] = block["sparse_values"].to(dtype=torch.float32, device=device)

            r_vectors = torch.zeros((self.num_angles, self.N), dtype=torch.float32, device=device)
            for idx, block in enumerate(blocks):
                r_vectors[idx, 0] = block["diag0"].to(dtype=torch.float32, device=device)

            self.register_buffer("r_vectors", r_vectors)
            self.register_buffer("sparse_rows", sparse_rows)
            self.register_buffer("sparse_cols", sparse_cols)
            self.register_buffer("sparse_values", sparse_values)
            self.register_buffer("sparse_nnz", sparse_nnz)
            self.register_buffer("directions", torch.stack(directions, dim=0).to(dtype=torch.float32, device=device))
            self.register_buffer("alphas", torch.stack(directions, dim=0).to(dtype=torch.float32, device=device))
            self.register_buffer("alpha_values_tensor", torch.tensor(self.alpha_values, dtype=torch.float32, device=device))
            self.register_buffer("tau_offsets_tensor", torch.tensor(effective_tau, dtype=torch.float32, device=device))
            self.register_buffer("sorted_proj_per_angle", torch.stack(sorted_proj_list, dim=0).to(dtype=torch.float32, device=device))
            self.register_buffer("sampling_points_per_angle", torch.stack(sampling_points_list, dim=0).to(dtype=torch.float32, device=device))
            self.register_buffer("sampling_points", self.sampling_points_per_angle.reshape(-1))
            self.register_buffer("selected_row_indices_per_angle", torch.stack(selected_row_indices_list, dim=0).to(dtype=torch.int64, device=device))
            self.register_buffer("lex_to_order_indices", torch.stack(lex_to_order_list, dim=0).to(dtype=torch.int64, device=device))
            self.register_buffer("order_to_lex_indices", torch.stack(order_to_lex_list, dim=0).to(dtype=torch.int64, device=device))
            self.register_buffer("min_projected_gaps", torch.stack(min_gap_list, dim=0).to(dtype=torch.float32, device=device))
            self.register_buffer("support_lo_per_angle", torch.stack(support_lo_list, dim=0).to(dtype=torch.float32, device=device))
            self.register_buffer("support_hi_per_angle", torch.stack(support_hi_list, dim=0).to(dtype=torch.float32, device=device))
            self.register_buffer("lower_widths", torch.stack([blk["lower_width"] for blk in blocks]).to(dtype=torch.int64, device=device))
            self.register_buffer("upper_widths", torch.stack([blk["upper_width"] for blk in blocks]).to(dtype=torch.int64, device=device))

        self._morozov_gram_eigvals: Optional[torch.Tensor] = None
        self._morozov_gram_eigvecs: Optional[torch.Tensor] = None
        self.last_morozov_cache_hit: Optional[bool] = None
        self.last_morozov_cache_build_seconds: Optional[float] = None
        self._last_gram_context_signature: Optional[tuple[object, ...]] = None
        self._last_gram_context: Optional[dict[str, torch.Tensor]] = None
        self._pdhg_lipschitz2_cache: dict[int, float] = {}
        self.last_pdhg_stats: Optional[dict[str, object]] = None

    def _morozov_cache_fingerprint(self) -> dict[str, object]:
        return {
            "class_name": self.__class__.__name__,
            "height": int(self.height),
            "width": int(self.width),
            "num_angles": int(self.num_angles),
            "alpha_values": [round(float(v), 15) for v in self.alpha_values],
            "M_per_angle": int(self.M_per_angle),
            "M": int(self.M),
            "sampling_mode": str(self.sampling_mode),
            "subset_selection": str(getattr(self, "subset_selection", "uniform")),
            "edge_weighted_subset": (
                {
                    "boundary_fraction": round(float(self.edge_boundary_fraction), 15),
                    "edge_weight": int(self.edge_weight),
                    "middle_weight": int(self.middle_weight),
                }
                if str(getattr(self, "subset_selection", "")) == "edge_weighted"
                else None
            ),
            "tau_offsets": (
                [round(float(v), 15) for v in self.tau_offsets_tensor.detach().cpu().tolist()]
                if str(self.sampling_mode) in {"shifted_lattice", "shifted_lattice_subset", "shifted_lattice_binned"}
                else None
            ),
            "sis_binning": (
                {
                    "domain": "sorted_sis_index",
                    "weight": "normalized_mean",
                    "num_bins": int(self.M_per_angle),
                }
                if str(self.sampling_mode) == "shifted_lattice_binned"
                else None
            ),
            "sampling_points_sha256": (
                None
                if str(self.sampling_mode) == "shifted_lattice"
                else _sampling_points_digest(self.sampling_points_per_angle)
            ),
            "selected_row_indices_sha256": (
                None
                if bool(torch.all(self.selected_row_indices_per_angle < 0))
                else _integer_tensor_digest(self.selected_row_indices_per_angle)
            ),
            "sparse_nnz_per_angle": [int(v.item()) for v in self.sparse_nnz],
            "formula_mode": "alpha_continuous",
            "basis": "b1b1",
            "implementation_version": "alpha_continuous_rect_sparse_v6_edge_weighted_subset",
        }

    def _gram_context_signature(self, b: torch.Tensor) -> tuple[object, ...]:
        return (int(b.data_ptr()), tuple(int(v) for v in b.shape), str(b.device), str(b.dtype), int(getattr(b, "_version", 0)))

    @torch.no_grad()
    def _prepare_gram_context(self, b: torch.Tensor) -> dict[str, torch.Tensor]:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        signature = self._gram_context_signature(b)
        if self._last_gram_context_signature == signature and self._last_gram_context is not None:
            return self._last_gram_context
        rhs = self.adjoint(b).view(b.shape[0], self.N)
        eigvals, eigvecs = _ensure_implicit_gram_spectrum(self, self._morozov_cache_fingerprint())
        rhs_cpu = rhs.detach().to(dtype=torch.float32, device="cpu")
        eigvecs_cpu = eigvecs.detach().to(dtype=torch.float32, device="cpu")
        rhs_proj = rhs_cpu @ eigvecs_cpu
        b_norm2 = torch.sum(b.detach().to(dtype=torch.float32, device="cpu").square(), dim=1)
        context = {"b": b, "rhs": rhs, "rhs_proj": rhs_proj, "b_norm2": b_norm2, "eigvals": eigvals, "eigvecs": eigvecs}
        self._last_gram_context_signature = signature
        self._last_gram_context = context
        return context

    def split_measurements(self, g: torch.Tensor) -> torch.Tensor:
        if g.dim() == 3 and g.shape[1] == 1:
            g = g.squeeze(1)
        if g.dim() != 2:
            raise ValueError(f"Expected g with shape (B,M), got {tuple(g.shape)}")
        if int(g.shape[1]) != int(self.M):
            raise ValueError(f"Expected measurement length M={self.M}, got {g.shape[1]}")
        return g.view(g.shape[0], self.num_angles, self.M_per_angle)

    def forward_per_angle(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(1)
        if coeff_matrix.dim() != 4:
            raise ValueError(f"coeff_matrix must have shape (B,1,H,W), got {tuple(coeff_matrix.shape)}")
        coeff_matrix = coeff_matrix.to(dtype=torch.float32, device=self.sampling_points.device)
        batch = int(coeff_matrix.shape[0])
        coeff_flat = coeff_matrix.reshape(batch, self.N)
        gather_index = self.order_to_lex_indices.view(1, self.num_angles, self.N).expand(batch, -1, -1)
        ordered = coeff_flat.unsqueeze(1).expand(-1, self.num_angles, -1).gather(2, gather_index)
        return _sparse_blocks_apply_batched(
            self.sparse_rows,
            self.sparse_cols,
            self.sparse_values,
            self.sparse_nnz,
            ordered,
            num_rows=int(self.M_per_angle),
        )

    def forward(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.forward_per_angle(coeff_matrix).reshape(coeff_matrix.shape[0], self.M)

    def adjoint_per_angle(self, residual_per_angle: torch.Tensor) -> torch.Tensor:
        if residual_per_angle.dim() == 4 and residual_per_angle.shape[2] == 1:
            residual_per_angle = residual_per_angle.squeeze(2)
        if residual_per_angle.dim() != 3:
            raise ValueError(f"Expected residual_per_angle with shape (B,K,M_per_angle), got {tuple(residual_per_angle.shape)}")
        if int(residual_per_angle.shape[1]) != int(self.num_angles) or int(residual_per_angle.shape[2]) != int(self.M_per_angle):
            raise ValueError(
                f"Expected residual_per_angle shape (B,{self.num_angles},{self.M_per_angle}), "
                f"got {tuple(residual_per_angle.shape)}"
            )
        residual_per_angle = residual_per_angle.to(dtype=torch.float32, device=self.sampling_points.device)
        batch = int(residual_per_angle.shape[0])
        grad_ordered = _sparse_blocks_adjoint_apply_batched(
            self.sparse_rows,
            self.sparse_cols,
            self.sparse_values,
            self.sparse_nnz,
            residual_per_angle,
            num_cols=int(self.N),
        )
        gather_index = self.lex_to_order_indices.view(1, self.num_angles, self.N).expand(batch, -1, -1)
        grad_lex = grad_ordered.gather(2, gather_index)
        return grad_lex.view(batch, self.num_angles, 1, self.height, self.width)

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        return self.adjoint_per_angle(self.split_measurements(residual)).sum(dim=1)

    def apply_normal(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(coeff_matrix))

    def _normalize_angle_indices(self, angle_indices) -> list[int]:
        if angle_indices is None:
            indices = []
        elif torch.is_tensor(angle_indices):
            indices = [int(idx) for idx in angle_indices.detach().cpu().view(-1).tolist()]
        else:
            indices = [int(idx) for idx in list(angle_indices)]
        if not indices:
            raise ValueError("angle_indices must not be empty.")
        if len(set(indices)) != len(indices):
            raise ValueError(f"angle_indices contains duplicates: {indices!r}.")
        invalid = [idx for idx in indices if idx < 0 or idx >= int(self.num_angles)]
        if invalid:
            raise ValueError(
                f"angle_indices contains out-of-range indices {invalid!r} "
                f"for num_angles={int(self.num_angles)}."
            )
        return indices

    def apply_normal_selected_angles(self, coeff_matrix: torch.Tensor, angle_indices) -> torch.Tensor:
        """Apply the stacked normal matrix for a selected angle subset.

        This computes ``sum_{k in S} A_k^T A_k coeff_matrix`` without using
        the unselected angle blocks.  It is the normal operator for the
        selected stacked matrix ``A_S``.
        """
        indices = self._normalize_angle_indices(angle_indices)
        measurement_pa = self.forward_per_angle(coeff_matrix)
        mask = torch.zeros(
            int(self.num_angles),
            dtype=measurement_pa.dtype,
            device=measurement_pa.device,
        )
        mask[torch.as_tensor(indices, dtype=torch.long, device=measurement_pa.device)] = 1.0
        measurement_pa = measurement_pa * mask.view(1, int(self.num_angles), 1)
        return self.adjoint_per_angle(measurement_pa).sum(dim=1)

    def tv_gradient(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        """Forward finite-difference gradient used by anisotropic TV.

        The boundary condition is Neumann-like: the last column/row forward
        differences are fixed to zero, so the returned tensor has shape
        ``(B,2,H,W)`` without wrapping around the image.
        """
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(1)
        if coeff_matrix.dim() != 4:
            raise ValueError(f"coeff_matrix must have shape (B,1,H,W), got {tuple(coeff_matrix.shape)}")
        x = coeff_matrix.to(dtype=torch.float32, device=self.sampling_points.device)
        if int(x.shape[1]) != 1 or int(x.shape[2]) != int(self.height) or int(x.shape[3]) != int(self.width):
            raise ValueError(f"Expected coeff_matrix shape (B,1,{self.height},{self.width}), got {tuple(x.shape)}")
        grad = torch.zeros((int(x.shape[0]), 2, self.height, self.width), dtype=x.dtype, device=x.device)
        grad[:, 0, :, :-1] = x[:, 0, :, 1:] - x[:, 0, :, :-1]
        grad[:, 1, :-1, :] = x[:, 0, 1:, :] - x[:, 0, :-1, :]
        return grad

    def tv_divergence_adjoint(self, gradient: torch.Tensor) -> torch.Tensor:
        """Adjoint of :meth:`tv_gradient` under the Euclidean inner product."""
        if gradient.dim() != 4:
            raise ValueError(f"gradient must have shape (B,2,H,W), got {tuple(gradient.shape)}")
        if int(gradient.shape[1]) != 2 or int(gradient.shape[2]) != int(self.height) or int(gradient.shape[3]) != int(self.width):
            raise ValueError(f"Expected gradient shape (B,2,{self.height},{self.width}), got {tuple(gradient.shape)}")
        p = gradient.to(dtype=torch.float32, device=self.sampling_points.device)
        out = torch.zeros((int(p.shape[0]), 1, self.height, self.width), dtype=p.dtype, device=p.device)
        px = p[:, 0]
        py = p[:, 1]
        out[:, 0, :, :-1] -= px[:, :, :-1]
        out[:, 0, :, 1:] += px[:, :, :-1]
        out[:, 0, :-1, :] -= py[:, :-1, :]
        out[:, 0, 1:, :] += py[:, :-1, :]
        return out

    def apply_tv_normal(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        """Apply ``D^T D`` for the finite-difference TV split operator ``D``."""
        return self.tv_divergence_adjoint(self.tv_gradient(coeff_matrix))

    def anisotropic_tv_norm(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        """Return per-sample anisotropic TV, ``sum(|D_x x| + |D_y x|)``."""
        grad = self.tv_gradient(coeff_matrix)
        return torch.sum(torch.abs(grad).reshape(grad.shape[0], -1), dim=1)

    @torch.no_grad()
    def _estimate_pdhg_lipschitz2(self, n_power: int = 8) -> float:
        """Estimate ``||K||^2`` for ``K x = (A x, D x)`` used by PDHG-TV."""
        n_power = max(1, int(n_power))
        cached = self._pdhg_lipschitz2_cache.get(n_power)
        if cached is not None:
            return float(cached)

        x = torch.arange(
            1,
            int(self.N) + 1,
            dtype=torch.float32,
            device=self.sampling_points.device,
        ).view(1, 1, self.height, self.width)
        x = x - torch.mean(x)
        x_norm = torch.norm(x.reshape(1, -1), dim=1).view(1, 1, 1, 1)
        if float(x_norm.item()) <= 1.0e-12:
            x = torch.ones_like(x)
            x_norm = torch.norm(x.reshape(1, -1), dim=1).view(1, 1, 1, 1)
        x = x / x_norm.clamp_min(1.0e-12)

        for _ in range(n_power):
            y = self.apply_normal(x) + self.apply_tv_normal(x)
            y_norm = torch.norm(y.reshape(1, -1), dim=1).view(1, 1, 1, 1).clamp_min(1.0e-12)
            x = y / y_norm

        y = self.apply_normal(x) + self.apply_tv_normal(x)
        num = torch.sum(x * y)
        den = torch.sum(x * x).clamp_min(1.0e-12)
        lipschitz2 = max(float((num / den).detach().cpu().item()), 1.0e-6)
        self._pdhg_lipschitz2_cache[n_power] = lipschitz2
        return lipschitz2

    @torch.no_grad()
    def solve_l2_tv_pdhg(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        *,
        max_iter: int = 10,
        tau: Optional[float] = None,
        sigma: Optional[float] = None,
        theta: float = 1.0,
        nonnegative: bool = False,
        x0: Optional[torch.Tensor] = None,
        power_iters: int = 8,
    ) -> torch.Tensor:
        """Run few-step PDHG for ``0.5 * ||A x - b||_2^2 + lambda * TV(x)``.

        This is intended as a fast TV-informed neural-network initializer, not
        as a fully converged classical TV baseline.  The TV term matches the
        existing ADMM implementation: anisotropic forward-difference TV.
        """
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        if b.dim() != 2:
            raise ValueError(f"Expected b with shape (B,M), got {tuple(b.shape)}")
        batch = int(b.shape[0])
        lam = self._normalize_lambda_reg(lambda_reg, batch_size=batch, target_device=b.device)
        lam_view = lam.view(batch, 1, 1, 1)

        if x0 is None:
            x = torch.zeros((batch, 1, self.height, self.width), dtype=torch.float32, device=b.device)
        else:
            if x0.dim() == 3:
                x0 = x0.unsqueeze(1)
            x = x0.to(dtype=torch.float32, device=b.device).clone()
            if x.shape != (batch, 1, self.height, self.width):
                raise ValueError(
                    f"Expected x0 shape {(batch, 1, self.height, self.width)}, got {tuple(x.shape)}"
                )

        x_bar = x.clone()
        p = torch.zeros_like(b)
        q = torch.zeros((batch, 2, self.height, self.width), dtype=torch.float32, device=b.device)

        lipschitz2 = None
        if tau is None or sigma is None:
            lipschitz2 = max(self._estimate_pdhg_lipschitz2(n_power=power_iters), 1.0e-6)
            # Power iteration may slightly underestimate the true norm.  Use a
            # conservative safety margin because initialization prefers
            # stability over aggressive convergence.
            step = 0.8 / math.sqrt(1.05 * lipschitz2)
            tau = step if tau is None else float(tau)
            sigma = step if sigma is None else float(sigma)

        tau = float(tau)
        sigma = float(sigma)
        theta = float(theta)
        if tau <= 0.0 or sigma <= 0.0:
            raise ValueError(f"PDHG step sizes must be positive, got tau={tau!r}, sigma={sigma!r}.")

        actual_iter = max(0, int(max_iter))
        for _ in range(actual_iter):
            p = (p + sigma * (self.forward(x_bar) - b)) / (1.0 + sigma)

            q_candidate = q + sigma * self.tv_gradient(x_bar)
            q = torch.maximum(torch.minimum(q_candidate, lam_view), -lam_view)

            x_new = x - tau * (self.adjoint(p) + self.tv_divergence_adjoint(q))
            if bool(nonnegative):
                x_new = torch.clamp(x_new, min=0.0)

            x_bar = x_new + theta * (x_new - x)
            x = x_new

        residual = self.forward(x) - b
        self.last_pdhg_stats = {
            "method": "l2_tv_pdhg",
            "iterations": int(actual_iter),
            "lambda_reg": [float(v) for v in lam.detach().cpu().view(-1).tolist()],
            "tau": float(tau),
            "sigma": float(sigma),
            "theta": float(theta),
            "nonnegative": bool(nonnegative),
            "power_iters": int(power_iters),
            "lipschitz2": None if lipschitz2 is None else float(lipschitz2),
            "measurement_l2": [float(v) for v in torch.norm(residual.detach(), dim=-1).cpu().view(-1).tolist()],
            "coeff_tv": [float(v) for v in self.anisotropic_tv_norm(x.detach()).detach().cpu().view(-1).tolist()],
        }
        self.last_split_admm_stats = None
        return x

    def apply_normal_per_angle(self, coeff_per_angle: torch.Tensor) -> torch.Tensor:
        """Apply each single-angle normal matrix independently.

        Args:
            coeff_per_angle: Tensor with shape ``(B,K,1,H,W)`` or ``(B,K,H,W)``.

        Returns:
            Tensor with shape ``(B,K,1,H,W)`` containing
            ``A_k^T A_k coeff_per_angle[:, k]`` for every angle ``k``.
        """
        if coeff_per_angle.dim() == 4:
            coeff_per_angle = coeff_per_angle.unsqueeze(2)
        if coeff_per_angle.dim() != 5:
            raise ValueError(f"coeff_per_angle must have shape (B,K,1,H,W), got {tuple(coeff_per_angle.shape)}")
        if int(coeff_per_angle.shape[1]) != int(self.num_angles):
            raise ValueError(f"Expected K={self.num_angles} angle channels, got {int(coeff_per_angle.shape[1])}.")
        if int(coeff_per_angle.shape[2]) != 1 or int(coeff_per_angle.shape[3]) != int(self.height) or int(coeff_per_angle.shape[4]) != int(self.width):
            raise ValueError(
                f"Expected coeff_per_angle shape (B,{self.num_angles},1,{self.height},{self.width}), "
                f"got {tuple(coeff_per_angle.shape)}."
            )
        coeff_per_angle = coeff_per_angle.to(dtype=torch.float32, device=self.sampling_points.device)
        batch = int(coeff_per_angle.shape[0])
        coeff_flat = coeff_per_angle.reshape(batch, self.num_angles, self.N)
        order_to_lex = self.order_to_lex_indices.view(1, self.num_angles, self.N).expand(batch, -1, -1)
        ordered = coeff_flat.gather(2, order_to_lex)
        measurement_pa = _sparse_blocks_apply_batched(
            self.sparse_rows,
            self.sparse_cols,
            self.sparse_values,
            self.sparse_nnz,
            ordered,
            num_rows=int(self.M_per_angle),
        )
        grad_ordered = _sparse_blocks_adjoint_apply_batched(
            self.sparse_rows,
            self.sparse_cols,
            self.sparse_values,
            self.sparse_nnz,
            measurement_pa,
            num_cols=int(self.N),
        )
        lex_to_order = self.lex_to_order_indices.view(1, self.num_angles, self.N).expand(batch, -1, -1)
        grad_lex = grad_ordered.gather(2, lex_to_order)
        return grad_lex.view(batch, self.num_angles, 1, self.height, self.width)

    def solve_shifted_angle_normal_cg(self, rhs_per_angle: torch.Tensor, damping: float = 1.0e-2, cg_iters: int = 8) -> torch.Tensor:
        """Solve ``(A_k^T A_k + damping I)x_k = rhs_k`` for each angle independently."""
        if rhs_per_angle.dim() == 4:
            rhs_per_angle = rhs_per_angle.unsqueeze(2)
        if rhs_per_angle.dim() != 5:
            raise ValueError(f"rhs_per_angle must have shape (B,K,1,H,W), got {tuple(rhs_per_angle.shape)}")
        rhs_per_angle = rhs_per_angle.to(dtype=torch.float32, device=self.sampling_points.device)
        x = torch.zeros_like(rhs_per_angle)
        mu = float(damping)

        def normal_plus_mu(z: torch.Tensor) -> torch.Tensor:
            return self.apply_normal_per_angle(z) + mu * z

        r = rhs_per_angle - normal_plus_mu(x)
        p = r.clone()
        rs_old = torch.sum(r.reshape(r.shape[0], r.shape[1], -1).square(), dim=2, keepdim=True)
        eps = rhs_per_angle.new_tensor(1.0e-12)
        for _ in range(max(int(cg_iters), 0)):
            Ap = normal_plus_mu(p)
            denom = torch.sum(
                p.reshape(p.shape[0], p.shape[1], -1) * Ap.reshape(Ap.shape[0], Ap.shape[1], -1),
                dim=2,
                keepdim=True,
            ).clamp_min(eps)
            alpha = rs_old / denom
            alpha_view = alpha.view(alpha.shape[0], alpha.shape[1], 1, 1, 1)
            x = x + alpha_view * p
            r = r - alpha_view * Ap
            rs_new = torch.sum(r.reshape(r.shape[0], r.shape[1], -1).square(), dim=2, keepdim=True)
            cg_ratio = rs_new / rs_old.clamp_min(eps)
            p = r + cg_ratio.view(cg_ratio.shape[0], cg_ratio.shape[1], 1, 1, 1) * p
            rs_old = rs_new
        return x

    def solve_shifted_normal_cg(self, rhs: torch.Tensor, damping: float = 1.0e-2, cg_iters: int = 8) -> torch.Tensor:
        if rhs.dim() == 3:
            rhs = rhs.unsqueeze(1)
        if rhs.dim() != 4:
            raise ValueError(f"rhs must have shape (B,1,H,W), got {tuple(rhs.shape)}")
        rhs = rhs.to(dtype=torch.float32, device=self.sampling_points.device)
        x = torch.zeros_like(rhs)
        mu = float(damping)

        def normal_plus_mu(z: torch.Tensor) -> torch.Tensor:
            return self.apply_normal(z) + mu * z

        r = rhs - normal_plus_mu(x)
        p = r.clone()
        rs_old = torch.sum(r.reshape(r.shape[0], -1).square(), dim=1, keepdim=True)
        eps = rhs.new_tensor(1.0e-12)
        for _ in range(max(int(cg_iters), 0)):
            Ap = normal_plus_mu(p)
            denom = torch.sum(p.reshape(p.shape[0], -1) * Ap.reshape(Ap.shape[0], -1), dim=1, keepdim=True).clamp_min(eps)
            alpha = rs_old / denom
            alpha_view = alpha.view(-1, 1, 1, 1)
            x = x + alpha_view * p
            r = r - alpha_view * Ap
            rs_new = torch.sum(r.reshape(r.shape[0], -1).square(), dim=1, keepdim=True)
            cg_ratio = rs_new / rs_old.clamp_min(eps)
            p = r + cg_ratio.view(-1, 1, 1, 1) * p
            rs_old = rs_new
        return x

    def solve_shifted_selected_normal_cg(
        self,
        rhs: torch.Tensor,
        angle_indices,
        damping: float = 1.0e-2,
        cg_iters: int = 8,
    ) -> torch.Tensor:
        """Solve ``(A_S^T A_S + damping I)x = rhs`` for selected stacked angles."""
        indices = self._normalize_angle_indices(angle_indices)
        if rhs.dim() == 3:
            rhs = rhs.unsqueeze(1)
        if rhs.dim() != 4:
            raise ValueError(f"rhs must have shape (B,1,H,W), got {tuple(rhs.shape)}")
        rhs = rhs.to(dtype=torch.float32, device=self.sampling_points.device)
        x = torch.zeros_like(rhs)
        mu = float(damping)

        def normal_plus_mu(z: torch.Tensor) -> torch.Tensor:
            return self.apply_normal_selected_angles(z, indices) + mu * z

        r = rhs - normal_plus_mu(x)
        p = r.clone()
        rs_old = torch.sum(r.reshape(r.shape[0], -1).square(), dim=1, keepdim=True)
        eps = rhs.new_tensor(1.0e-12)
        for _ in range(max(int(cg_iters), 0)):
            Ap = normal_plus_mu(p)
            denom = torch.sum(p.reshape(p.shape[0], -1) * Ap.reshape(Ap.shape[0], -1), dim=1, keepdim=True).clamp_min(eps)
            alpha = rs_old / denom
            alpha_view = alpha.view(-1, 1, 1, 1)
            x = x + alpha_view * p
            r = r - alpha_view * Ap
            rs_new = torch.sum(r.reshape(r.shape[0], -1).square(), dim=1, keepdim=True)
            cg_ratio = rs_new / rs_old.clamp_min(eps)
            p = r + cg_ratio.view(-1, 1, 1, 1) * p
            rs_old = rs_new
        return x

    def residual_inverse_correction(
        self,
        coeff: torch.Tensor,
        g_observed: torch.Tensor,
        damping: float = 1.0e-2,
        cg_iters: int = 8,
        detach: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        if detach:
            coeff = coeff.detach()
            g_observed = g_observed.detach()
        pred = self(coeff)
        residual = g_observed.to(dtype=pred.dtype, device=pred.device) - pred
        rhs = self.adjoint(residual)
        correction = self.solve_shifted_normal_cg(rhs, damping=damping, cg_iters=cg_iters)
        if normalize:
            flat = correction.reshape(correction.shape[0], -1)
            norm = torch.norm(flat, dim=1, keepdim=True).clamp_min(1.0e-6)
            correction = correction / norm.view(-1, 1, 1, 1)
        return correction

    def residual_inverse_correction_selected_angles(
        self,
        coeff: torch.Tensor,
        g_observed: torch.Tensor,
        angle_indices,
        damping: float = 1.0e-2,
        cg_iters: int = 8,
        detach: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Return one selected-stacked inverse-residual correction image.

        This approximates

            ``(A_S^T A_S + damping I)^-1 A_S^T (g_S - A_S c)``

        where ``S`` is the provided subset of original alpha-angle indices.
        The returned shape is ``(B,1,H,W)``.
        """
        indices = self._normalize_angle_indices(angle_indices)
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        if detach:
            coeff = coeff.detach()
            g_observed = g_observed.detach()
        pred_pa = self.forward_per_angle(coeff)
        observed_pa = self.split_measurements(g_observed).to(dtype=pred_pa.dtype, device=pred_pa.device)
        residual_pa = observed_pa - pred_pa
        mask = torch.zeros(
            int(self.num_angles),
            dtype=residual_pa.dtype,
            device=residual_pa.device,
        )
        mask[torch.as_tensor(indices, dtype=torch.long, device=residual_pa.device)] = 1.0
        residual_pa = residual_pa * mask.view(1, int(self.num_angles), 1)
        rhs = self.adjoint_per_angle(residual_pa).sum(dim=1)
        correction = self.solve_shifted_selected_normal_cg(
            rhs,
            indices,
            damping=damping,
            cg_iters=cg_iters,
        )
        if normalize:
            flat = correction.reshape(correction.shape[0], -1)
            norm = torch.norm(flat, dim=1, keepdim=True).clamp_min(1.0e-6)
            correction = correction / norm.view(-1, 1, 1, 1)
        return correction

    def residual_inverse_correction_per_angle(
        self,
        coeff: torch.Tensor,
        g_observed: torch.Tensor,
        damping: float = 1.0e-2,
        cg_iters: int = 8,
        detach: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Return one inverse-residual correction image per alpha angle.

        For each angle ``k`` this approximates

            ``(A_k^T A_k + damping I)^-1 A_k^T (g_k - A_k c)``.

        The returned shape is ``(B,K,1,H,W)`` so learned models can use the
        corrections as per-angle feature channels instead of a single stacked
        correction.
        """
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        if detach:
            coeff = coeff.detach()
            g_observed = g_observed.detach()
        pred_pa = self.forward_per_angle(coeff)
        observed_pa = self.split_measurements(g_observed).to(dtype=pred_pa.dtype, device=pred_pa.device)
        residual_pa = observed_pa - pred_pa
        rhs_pa = self.adjoint_per_angle(residual_pa)
        correction_pa = self.solve_shifted_angle_normal_cg(rhs_pa, damping=damping, cg_iters=cg_iters)
        if normalize:
            flat = correction_pa.reshape(correction_pa.shape[0], correction_pa.shape[1], -1)
            norm = torch.norm(flat, dim=2, keepdim=True).clamp_min(1.0e-6)
            correction_pa = correction_pa / norm.view(correction_pa.shape[0], correction_pa.shape[1], 1, 1, 1)
        return correction_pa

    def _normalize_lambda_reg(self, lambda_reg: float | torch.Tensor, batch_size: int, target_device: torch.device) -> torch.Tensor:
        if torch.is_tensor(lambda_reg):
            lam = lambda_reg.detach().to(dtype=torch.float32, device=target_device).view(-1)
            if int(lam.numel()) == 1 and int(batch_size) > 1:
                lam = lam.expand(int(batch_size))
            elif int(lam.numel()) != int(batch_size):
                raise ValueError(f"lambda_reg has {int(lam.numel())} entries, expected 1 or batch={int(batch_size)}.")
        else:
            lam = torch.full((int(batch_size),), float(lambda_reg), dtype=torch.float32, device=target_device)
        return lam.clamp_min(0.0)

    def _admm_options(
        self,
        *,
        max_iter: Optional[int] = None,
        cg_iters: Optional[int] = None,
        cg_tol: Optional[float] = None,
        rho_data: Optional[float] = None,
        rho_reg: Optional[float] = None,
    ) -> dict[str, float | int | str | bool]:
        return {
            "max_iter": max(0, int(DATA_CONFIG.get("l1_init_admm_iters", 80) if max_iter is None else max_iter)),
            "cg_iters": max(1, int(DATA_CONFIG.get("l1_init_admm_cg_iters", 30) if cg_iters is None else cg_iters)),
            "cg_tol": float(DATA_CONFIG.get("l1_init_admm_cg_tol", 1.0e-4) if cg_tol is None else cg_tol),
            "rho_data": max(float(DATA_CONFIG.get("l1_init_admm_rho_data", 1.0) if rho_data is None else rho_data), 1.0e-12),
            "rho_reg": max(float(DATA_CONFIG.get("l1_init_admm_rho_reg", 1.0) if rho_reg is None else rho_reg), 1.0e-12),
            "stop_mode": str(DATA_CONFIG.get("admm_stop_mode", "fixed")).strip().lower(),
            "min_iter": max(0, int(DATA_CONFIG.get("admm_min_iters", 10))),
            "abs_tol": float(DATA_CONFIG.get("admm_abs_tol", 1.0e-4)),
            "rel_tol": float(DATA_CONFIG.get("admm_rel_tol", 1.0e-3)),
            "check_interval": max(1, int(DATA_CONFIG.get("admm_check_interval", 1))),
        }

    @torch.no_grad()
    def _solve_weighted_normal_cg(
        self,
        rhs: torch.Tensor,
        *,
        normal_weight: float,
        ridge_weight: float,
        tv_weight: float = 0.0,
        max_iter: int,
        tol: float,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rhs.dim() == 3:
            rhs = rhs.unsqueeze(1)
        if rhs.dim() != 4:
            raise ValueError(f"rhs must have shape (B,1,H,W), got {tuple(rhs.shape)}")
        rhs = rhs.to(dtype=torch.float32, device=self.sampling_points.device)
        x = torch.zeros_like(rhs) if x0 is None else x0.to(dtype=torch.float32, device=rhs.device).clone()
        normal_weight = float(normal_weight)
        ridge_weight = float(ridge_weight)
        tv_weight = float(tv_weight)

        def matvec(z: torch.Tensor) -> torch.Tensor:
            out = normal_weight * self.apply_normal(z) + ridge_weight * z
            if tv_weight != 0.0:
                out = out + tv_weight * self.apply_tv_normal(z)
            return out

        r = rhs - matvec(x)
        p = r.clone()
        rr = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)
        eps = rhs.new_tensor(1.0e-12)
        for _ in range(int(max_iter)):
            Ap = matvec(p)
            denom = torch.sum(p * Ap, dim=(1, 2, 3), keepdim=True).clamp_min(eps)
            alpha = rr / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)
            if torch.sqrt(rr_new.max()).item() < float(tol):
                break
            p = r + (rr_new / rr.clamp_min(eps)) * p
            rr = rr_new
        return x

    def _batch_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.norm(x.reshape(x.shape[0], -1), dim=1)

    def _batch_l2_stack_norm(self, *xs: torch.Tensor) -> torch.Tensor:
        vals = None
        for x in xs:
            n2 = self._batch_l2_norm(x).square()
            vals = n2 if vals is None else vals + n2
        if vals is None:
            raise ValueError("_batch_l2_stack_norm requires at least one tensor.")
        return torch.sqrt(vals.clamp_min(0.0))

    def _admm_abs_rel_tol(
        self,
        *,
        batch_size: int,
        dim: int,
        abs_tol: float,
        rel_tol: float,
        reference_terms: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        ref = torch.zeros((int(batch_size),), dtype=torch.float32, device=device)
        for term in reference_terms:
            ref = torch.maximum(ref, self._batch_l2_norm(term))
        return (float(dim) ** 0.5) * float(abs_tol) + float(rel_tol) * ref

    def _admm_should_stop(
        self,
        *,
        opts: dict,
        iteration: int,
        primal_norm: torch.Tensor,
        dual_norm: torch.Tensor,
        eps_pri: torch.Tensor,
        eps_dual: torch.Tensor,
    ) -> tuple[bool, str]:
        if str(opts.get("stop_mode", "fixed")).strip().lower() == "fixed":
            return False, "fixed_iteration_mode"
        current_iter = int(iteration) + 1
        if current_iter < int(opts.get("min_iter", 0)):
            return False, "below_min_iter"
        if current_iter % int(opts.get("check_interval", 1)) != 0:
            return False, "skip_check_interval"

        ok_primal = torch.all(primal_norm <= eps_pri)
        ok_dual = torch.all(dual_norm <= eps_dual)
        ok = bool(ok_primal and ok_dual)
        if ok:
            return True, "primal_dual_residual_satisfied"
        if not bool(ok_primal):
            return False, "primal_residual_not_satisfied"
        if not bool(ok_dual):
            return False, "dual_residual_not_satisfied"
        return False, "not_satisfied"

    @staticmethod
    def _admm_stats_values(x: Optional[torch.Tensor]) -> list[float]:
        if x is None:
            return []
        return [float(v) for v in x.detach().cpu().view(-1).tolist()]

    def _record_admm_stats(
        self,
        *,
        method: str,
        iterations: int,
        coeff: torch.Tensor,
        b: torch.Tensor,
        lambda_reg: torch.Tensor,
        stop_reason: str = "max_iter_reached",
        primal_norm: Optional[torch.Tensor] = None,
        dual_norm: Optional[torch.Tensor] = None,
        eps_pri: Optional[torch.Tensor] = None,
        eps_dual: Optional[torch.Tensor] = None,
    ) -> None:
        residual = self(coeff) - b
        self.last_split_admm_stats = {
            "method": str(method),
            "iterations": int(iterations),
            "stop_reason": str(stop_reason),
            "lambda_reg": [float(v) for v in lambda_reg.detach().cpu().view(-1).tolist()],
            "measurement_l2": [float(v) for v in torch.norm(residual.detach(), dim=-1).cpu().view(-1).tolist()],
            "measurement_l1": [float(v) for v in torch.sum(torch.abs(residual.detach()), dim=-1).cpu().view(-1).tolist()],
            "coeff_l1": [float(v) for v in torch.sum(torch.abs(coeff.detach()).reshape(coeff.shape[0], -1), dim=1).cpu().view(-1).tolist()],
            "coeff_tv": [float(v) for v in self.anisotropic_tv_norm(coeff.detach()).detach().cpu().view(-1).tolist()],
            "primal_norm": self._admm_stats_values(primal_norm),
            "dual_norm": self._admm_stats_values(dual_norm),
            "eps_pri": self._admm_stats_values(eps_pri),
            "eps_dual": self._admm_stats_values(eps_dual),
        }

    @torch.no_grad()
    def l2_l1_zero_threshold(self, b: torch.Tensor) -> torch.Tensor:
        """Return smallest lambda for which x=0 solves 0.5||Ax-b||_2^2 + lambda||x||_1."""
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        adj = self.adjoint(b).reshape(b.shape[0], -1)
        return torch.amax(torch.abs(adj), dim=1).clamp_min(0.0)

    @torch.no_grad()
    def l1_l1_zero_threshold(self, b: torch.Tensor) -> torch.Tensor:
        """Return a zero-solution lambda threshold for ||Ax-b||_1 + lambda||x||_1."""
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        adj = self.adjoint(torch.sign(b)).reshape(b.shape[0], -1)
        return torch.amax(torch.abs(adj), dim=1).clamp_min(0.0)

    def _record_constrained_admm_stats(
        self,
        *,
        method: str,
        iterations: int,
        coeff: torch.Tensor,
        b: torch.Tensor,
        noise_radius: torch.Tensor,
        residual_norm: str,
        stop_reason: str = "max_iter_reached",
    ) -> None:
        residual = self(coeff) - b
        self.last_split_admm_stats = {
            "method": str(method),
            "iterations": int(iterations),
            "stop_reason": str(stop_reason),
            "constraint_radius": [float(v) for v in noise_radius.detach().cpu().view(-1).tolist()],
            "residual_norm": str(residual_norm),
            "measurement_l2": [float(v) for v in torch.norm(residual.detach(), dim=-1).cpu().view(-1).tolist()],
            "measurement_l1": [float(v) for v in torch.sum(torch.abs(residual.detach()), dim=-1).cpu().view(-1).tolist()],
            "coeff_l1": [float(v) for v in torch.sum(torch.abs(coeff.detach()).reshape(coeff.shape[0], -1), dim=1).cpu().view(-1).tolist()],
            "coeff_tv": [float(v) for v in self.anisotropic_tv_norm(coeff.detach()).detach().cpu().view(-1).tolist()],
        }

    @torch.no_grad()
    def solve_l2_l1_admm(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        *,
        max_iter: Optional[int] = None,
        cg_iters: Optional[int] = None,
        cg_tol: Optional[float] = None,
        rho_reg: Optional[float] = None,
    ) -> torch.Tensor:
        """Solve ``0.5 * ||A x - b||_2^2 + lambda * ||x||_1`` by ADMM."""
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        opts = self._admm_options(max_iter=max_iter, cg_iters=cg_iters, cg_tol=cg_tol, rho_reg=rho_reg)
        batch = int(b.shape[0])
        lam = self._normalize_lambda_reg(lambda_reg, batch_size=batch, target_device=b.device)
        zero_threshold = self.l2_l1_zero_threshold(b)
        zero_active = lam >= zero_threshold * (1.0 - 1.0e-6)
        if bool(torch.all(zero_active)):
            x_zero = torch.zeros((batch, 1, self.height, self.width), dtype=torch.float32, device=b.device)
            self._record_admm_stats(method="l2_l1_admm", iterations=0, coeff=x_zero, b=b, lambda_reg=lam, stop_reason="zero_solution_threshold")
            return x_zero
        rho = float(opts["rho_reg"])
        rhs_data = self.adjoint(b)
        x = torch.zeros((batch, 1, self.height, self.width), dtype=torch.float32, device=b.device)
        z = torch.zeros_like(x)
        u = torch.zeros_like(x)
        threshold = (lam / rho).view(batch, 1, 1, 1)
        stop_reason = "max_iter_reached"
        actual_iter = 0
        last_primal_norm = None
        last_dual_norm = None
        last_eps_pri = None
        last_eps_dual = None
        for it in range(int(opts["max_iter"])):
            z_prev = z.clone()
            rhs = rhs_data + rho * (z - u)
            x = self._solve_weighted_normal_cg(
                rhs,
                normal_weight=1.0,
                ridge_weight=rho,
                max_iter=int(opts["cg_iters"]),
                tol=float(opts["cg_tol"]),
                x0=x,
            )
            z = _soft_threshold(x + u, threshold)
            u = u + x - z
            primal = x - z
            dual = rho * (z - z_prev)
            primal_norm = self._batch_l2_norm(primal)
            dual_norm = self._batch_l2_norm(dual)
            eps_pri = self._admm_abs_rel_tol(
                batch_size=batch,
                dim=int(self.height * self.width),
                abs_tol=float(opts["abs_tol"]),
                rel_tol=float(opts["rel_tol"]),
                reference_terms=[x, z],
                device=b.device,
            )
            eps_dual = self._admm_abs_rel_tol(
                batch_size=batch,
                dim=int(self.height * self.width),
                abs_tol=float(opts["abs_tol"]),
                rel_tol=float(opts["rel_tol"]),
                reference_terms=[rho * u],
                device=b.device,
            )
            last_primal_norm = primal_norm
            last_dual_norm = dual_norm
            last_eps_pri = eps_pri
            last_eps_dual = eps_dual
            actual_iter = it + 1
            should_stop, stop_candidate = self._admm_should_stop(
                opts=opts,
                iteration=it,
                primal_norm=primal_norm,
                dual_norm=dual_norm,
                eps_pri=eps_pri,
                eps_dual=eps_dual,
            )
            if should_stop:
                stop_reason = stop_candidate
                break
        if bool(torch.any(zero_active)):
            x = x.clone()
            x[zero_active.view(-1, 1, 1, 1).expand_as(x)] = 0.0
        self._record_admm_stats(
            method="l2_l1_admm",
            iterations=int(actual_iter),
            coeff=x,
            b=b,
            lambda_reg=lam,
            stop_reason=stop_reason,
            primal_norm=last_primal_norm,
            dual_norm=last_dual_norm,
            eps_pri=last_eps_pri,
            eps_dual=last_eps_dual,
        )
        return x

    @torch.no_grad()
    def solve_l2_tv_admm(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        *,
        max_iter: Optional[int] = None,
        cg_iters: Optional[int] = None,
        cg_tol: Optional[float] = None,
        rho_reg: Optional[float] = None,
    ) -> torch.Tensor:
        """Solve ``0.5 * ||A x - b||_2^2 + lambda * TV(x)`` by split ADMM."""
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        opts = self._admm_options(max_iter=max_iter, cg_iters=cg_iters, cg_tol=cg_tol, rho_reg=rho_reg)
        batch = int(b.shape[0])
        lam = self._normalize_lambda_reg(lambda_reg, batch_size=batch, target_device=b.device)
        rho = float(opts["rho_reg"])
        rhs_data = self.adjoint(b)
        x = torch.zeros((batch, 1, self.height, self.width), dtype=torch.float32, device=b.device)
        z = torch.zeros((batch, 2, self.height, self.width), dtype=torch.float32, device=b.device)
        u = torch.zeros_like(z)
        threshold = (lam / rho).view(batch, 1, 1, 1)
        stop_reason = "max_iter_reached"
        actual_iter = 0
        last_primal_norm = None
        last_dual_norm = None
        last_eps_pri = None
        last_eps_dual = None
        for it in range(int(opts["max_iter"])):
            z_prev = z.clone()
            rhs = rhs_data + rho * self.tv_divergence_adjoint(z - u)
            x = self._solve_weighted_normal_cg(
                rhs,
                normal_weight=1.0,
                ridge_weight=0.0,
                tv_weight=rho,
                max_iter=int(opts["cg_iters"]),
                tol=float(opts["cg_tol"]),
                x0=x,
            )
            grad = self.tv_gradient(x)
            z = _soft_threshold(grad + u, threshold)
            u = u + grad - z
            primal = grad - z
            dual = rho * self.tv_divergence_adjoint(z - z_prev)
            primal_norm = self._batch_l2_norm(primal)
            dual_norm = self._batch_l2_norm(dual)
            eps_pri = self._admm_abs_rel_tol(
                batch_size=batch,
                dim=int(2 * self.height * self.width),
                abs_tol=float(opts["abs_tol"]),
                rel_tol=float(opts["rel_tol"]),
                reference_terms=[grad, z],
                device=b.device,
            )
            eps_dual = self._admm_abs_rel_tol(
                batch_size=batch,
                dim=int(self.height * self.width),
                abs_tol=float(opts["abs_tol"]),
                rel_tol=float(opts["rel_tol"]),
                reference_terms=[rho * self.tv_divergence_adjoint(u)],
                device=b.device,
            )
            last_primal_norm = primal_norm
            last_dual_norm = dual_norm
            last_eps_pri = eps_pri
            last_eps_dual = eps_dual
            actual_iter = it + 1
            should_stop, stop_candidate = self._admm_should_stop(
                opts=opts,
                iteration=it,
                primal_norm=primal_norm,
                dual_norm=dual_norm,
                eps_pri=eps_pri,
                eps_dual=eps_dual,
            )
            if should_stop:
                stop_reason = stop_candidate
                break
        self._record_admm_stats(
            method="l2_tv_admm",
            iterations=int(actual_iter),
            coeff=x,
            b=b,
            lambda_reg=lam,
            stop_reason=stop_reason,
            primal_norm=last_primal_norm,
            dual_norm=last_dual_norm,
            eps_pri=last_eps_pri,
            eps_dual=last_eps_dual,
        )
        return x

    @torch.no_grad()
    def solve_l1_l1_admm(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        *,
        max_iter: Optional[int] = None,
        cg_iters: Optional[int] = None,
        cg_tol: Optional[float] = None,
        rho_data: Optional[float] = None,
        rho_reg: Optional[float] = None,
    ) -> torch.Tensor:
        """Solve ``||A x - b||_1 + lambda * ||x||_1`` by split ADMM."""
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        opts = self._admm_options(max_iter=max_iter, cg_iters=cg_iters, cg_tol=cg_tol, rho_data=rho_data, rho_reg=rho_reg)
        batch = int(b.shape[0])
        lam = self._normalize_lambda_reg(lambda_reg, batch_size=batch, target_device=b.device)
        zero_threshold = self.l1_l1_zero_threshold(b)
        zero_active = lam >= zero_threshold * (1.0 - 1.0e-6)
        if bool(torch.all(zero_active)):
            x_zero = torch.zeros((batch, 1, self.height, self.width), dtype=torch.float32, device=b.device)
            self._record_admm_stats(method="l1_l1_admm", iterations=0, coeff=x_zero, b=b, lambda_reg=lam, stop_reason="zero_solution_threshold")
            return x_zero
        rho_d = float(opts["rho_data"])
        rho_r = float(opts["rho_reg"])
        x = torch.zeros((batch, 1, self.height, self.width), dtype=torch.float32, device=b.device)
        r = torch.zeros_like(b)
        u_data = torch.zeros_like(b)
        z = torch.zeros_like(x)
        u_reg = torch.zeros_like(x)
        threshold_data = 1.0 / rho_d
        threshold_reg = (lam / rho_r).view(batch, 1, 1, 1)
        stop_reason = "max_iter_reached"
        actual_iter = 0
        last_primal_norm = None
        last_dual_norm = None
        last_eps_pri = None
        last_eps_dual = None
        for it in range(int(opts["max_iter"])):
            r_prev = r.clone()
            z_prev = z.clone()
            rhs = rho_d * self.adjoint(b + r - u_data) + rho_r * (z - u_reg)
            x = self._solve_weighted_normal_cg(
                rhs,
                normal_weight=rho_d,
                ridge_weight=rho_r,
                max_iter=int(opts["cg_iters"]),
                tol=float(opts["cg_tol"]),
                x0=x,
            )
            residual = self(x) - b
            r = _soft_threshold(residual + u_data, threshold_data)
            z = _soft_threshold(x + u_reg, threshold_reg)
            u_data = u_data + residual - r
            u_reg = u_reg + x - z
            primal_data = residual - r
            primal_reg = x - z
            dual = rho_d * self.adjoint(r - r_prev) + rho_r * (z - z_prev)
            primal_norm = self._batch_l2_stack_norm(primal_data, primal_reg)
            dual_norm = self._batch_l2_norm(dual)
            eps_pri = self._admm_abs_rel_tol(
                batch_size=batch,
                dim=int(b.shape[-1]) + int(self.height * self.width),
                abs_tol=float(opts["abs_tol"]),
                rel_tol=float(opts["rel_tol"]),
                reference_terms=[residual, r, x, z],
                device=b.device,
            )
            eps_dual = self._admm_abs_rel_tol(
                batch_size=batch,
                dim=int(self.height * self.width),
                abs_tol=float(opts["abs_tol"]),
                rel_tol=float(opts["rel_tol"]),
                reference_terms=[rho_d * self.adjoint(u_data) + rho_r * u_reg],
                device=b.device,
            )
            last_primal_norm = primal_norm
            last_dual_norm = dual_norm
            last_eps_pri = eps_pri
            last_eps_dual = eps_dual
            actual_iter = it + 1
            should_stop, stop_candidate = self._admm_should_stop(
                opts=opts,
                iteration=it,
                primal_norm=primal_norm,
                dual_norm=dual_norm,
                eps_pri=eps_pri,
                eps_dual=eps_dual,
            )
            if should_stop:
                stop_reason = stop_candidate
                break
        if bool(torch.any(zero_active)):
            x = x.clone()
            x[zero_active.view(-1, 1, 1, 1).expand_as(x)] = 0.0
        self._record_admm_stats(
            method="l1_l1_admm",
            iterations=int(actual_iter),
            coeff=x,
            b=b,
            lambda_reg=lam,
            stop_reason=stop_reason,
            primal_norm=last_primal_norm,
            dual_norm=last_dual_norm,
            eps_pri=last_eps_pri,
            eps_dual=last_eps_dual,
        )
        return x

    @torch.no_grad()
    def solve_l2_l1_morozov_admm(
        self,
        b: torch.Tensor,
        noise_radius: float | torch.Tensor,
        *,
        max_iter: Optional[int] = None,
        cg_iters: Optional[int] = None,
        cg_tol: Optional[float] = None,
        rho_data: Optional[float] = None,
        rho_reg: Optional[float] = None,
    ) -> torch.Tensor:
        """Solve ``min ||x||_1`` subject to ``||A x - b||_2 <= noise_radius``."""
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        opts = self._admm_options(max_iter=max_iter, cg_iters=cg_iters, cg_tol=cg_tol, rho_data=rho_data, rho_reg=rho_reg)
        batch = int(b.shape[0])
        radius = self._normalize_lambda_reg(noise_radius, batch_size=batch, target_device=b.device)
        rho_d = float(opts["rho_data"])
        rho_r = float(opts["rho_reg"])
        x = torch.zeros((batch, 1, self.height, self.width), dtype=torch.float32, device=b.device)
        r = _project_l2_ball(-b, radius)
        u_data = torch.zeros_like(b)
        z = torch.zeros_like(x)
        u_reg = torch.zeros_like(x)
        threshold_reg = 1.0 / rho_r
        for _ in range(int(opts["max_iter"])):
            rhs = rho_d * self.adjoint(b + r - u_data) + rho_r * (z - u_reg)
            x = self._solve_weighted_normal_cg(
                rhs,
                normal_weight=rho_d,
                ridge_weight=rho_r,
                max_iter=int(opts["cg_iters"]),
                tol=float(opts["cg_tol"]),
                x0=x,
            )
            residual = self(x) - b
            r = _project_l2_ball(residual + u_data, radius)
            z = _soft_threshold(x + u_reg, threshold_reg)
            u_data = u_data + residual - r
            u_reg = u_reg + x - z
        self._record_constrained_admm_stats(
            method="l2_l1_morozov_admm",
            iterations=int(opts["max_iter"]),
            coeff=x,
            b=b,
            noise_radius=radius,
            residual_norm="l2",
        )
        return x

    @torch.no_grad()
    def solve_l1_l1_morozov_admm(
        self,
        b: torch.Tensor,
        noise_radius: float | torch.Tensor,
        *,
        max_iter: Optional[int] = None,
        cg_iters: Optional[int] = None,
        cg_tol: Optional[float] = None,
        rho_data: Optional[float] = None,
        rho_reg: Optional[float] = None,
    ) -> torch.Tensor:
        """Solve ``min ||x||_1`` subject to ``||A x - b||_1 <= noise_radius``."""
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        opts = self._admm_options(max_iter=max_iter, cg_iters=cg_iters, cg_tol=cg_tol, rho_data=rho_data, rho_reg=rho_reg)
        batch = int(b.shape[0])
        radius = self._normalize_lambda_reg(noise_radius, batch_size=batch, target_device=b.device)
        rho_d = float(opts["rho_data"])
        rho_r = float(opts["rho_reg"])
        x = torch.zeros((batch, 1, self.height, self.width), dtype=torch.float32, device=b.device)
        r = _project_l1_ball(-b, radius)
        u_data = torch.zeros_like(b)
        z = torch.zeros_like(x)
        u_reg = torch.zeros_like(x)
        threshold_reg = 1.0 / rho_r
        for _ in range(int(opts["max_iter"])):
            rhs = rho_d * self.adjoint(b + r - u_data) + rho_r * (z - u_reg)
            x = self._solve_weighted_normal_cg(
                rhs,
                normal_weight=rho_d,
                ridge_weight=rho_r,
                max_iter=int(opts["cg_iters"]),
                tol=float(opts["cg_tol"]),
                x0=x,
            )
            residual = self(x) - b
            r = _project_l1_ball(residual + u_data, radius)
            z = _soft_threshold(x + u_reg, threshold_reg)
            u_data = u_data + residual - r
            u_reg = u_reg + x - z
        self._record_constrained_admm_stats(
            method="l1_l1_morozov_admm",
            iterations=int(opts["max_iter"]),
            coeff=x,
            b=b,
            noise_radius=radius,
            residual_norm="l1",
        )
        return x

    @torch.no_grad()
    def solve_l2_tv_morozov_admm(
        self,
        b: torch.Tensor,
        noise_radius: float | torch.Tensor,
        *,
        max_iter: Optional[int] = None,
        cg_iters: Optional[int] = None,
        cg_tol: Optional[float] = None,
        rho_data: Optional[float] = None,
        rho_reg: Optional[float] = None,
    ) -> torch.Tensor:
        """Solve ``min TV(x)`` subject to ``||A x - b||_2 <= noise_radius``."""
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        opts = self._admm_options(max_iter=max_iter, cg_iters=cg_iters, cg_tol=cg_tol, rho_data=rho_data, rho_reg=rho_reg)
        batch = int(b.shape[0])
        radius = self._normalize_lambda_reg(noise_radius, batch_size=batch, target_device=b.device)
        rho_d = float(opts["rho_data"])
        rho_r = float(opts["rho_reg"])
        x = torch.zeros((batch, 1, self.height, self.width), dtype=torch.float32, device=b.device)
        r = _project_l2_ball(-b, radius)
        u_data = torch.zeros_like(b)
        z = torch.zeros((batch, 2, self.height, self.width), dtype=torch.float32, device=b.device)
        u_reg = torch.zeros_like(z)
        threshold_reg = 1.0 / rho_r
        for _ in range(int(opts["max_iter"])):
            rhs = rho_d * self.adjoint(b + r - u_data) + rho_r * self.tv_divergence_adjoint(z - u_reg)
            x = self._solve_weighted_normal_cg(
                rhs,
                normal_weight=rho_d,
                ridge_weight=0.0,
                tv_weight=rho_r,
                max_iter=int(opts["cg_iters"]),
                tol=float(opts["cg_tol"]),
                x0=x,
            )
            residual = self(x) - b
            r = _project_l2_ball(residual + u_data, radius)
            grad = self.tv_gradient(x)
            z = _soft_threshold(grad + u_reg, threshold_reg)
            u_data = u_data + residual - r
            u_reg = u_reg + grad - z
        self._record_constrained_admm_stats(
            method="l2_tv_morozov_admm",
            iterations=int(opts["max_iter"]),
            coeff=x,
            b=b,
            noise_radius=radius,
            residual_norm="l2",
        )
        return x

    @torch.no_grad()
    def solve_tikhonov_direct(self, b: torch.Tensor, lambda_reg: float | torch.Tensor) -> torch.Tensor:
        self.last_split_admm_stats = None
        context = self._prepare_gram_context(b)
        coeff = _solve_tikhonov_from_gram_spectrum(context["rhs"], eigvals=context["eigvals"], eigvecs=context["eigvecs"], lambda_reg=lambda_reg, rhs_proj=context["rhs_proj"])
        return coeff.to(device=self.sampling_points.device, dtype=torch.float32).view(-1, 1, self.height, self.width)

    @torch.no_grad()
    def solve_tikhonov_cg(self, b: torch.Tensor, lambda_reg: float | torch.Tensor, max_iter: int, tol: float = 1e-4, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.sampling_points.device)
        rhs = self.adjoint(b)
        x = torch.zeros_like(rhs) if x0 is None else x0.to(dtype=torch.float32, device=rhs.device).clone()
        if torch.is_tensor(lambda_reg):
            lam = lambda_reg.detach().to(dtype=torch.float32, device=rhs.device).view(-1)
            if int(lam.numel()) == 1 and int(rhs.shape[0]) > 1:
                lam = lam.expand(int(rhs.shape[0]))
            elif int(lam.numel()) != int(rhs.shape[0]):
                raise ValueError(f"lambda_reg has {int(lam.numel())} entries, expected 1 or batch={int(rhs.shape[0])}.")
        else:
            lam = torch.full((int(rhs.shape[0]),), float(lambda_reg), dtype=torch.float32, device=rhs.device)
        lam = lam.view(-1, 1, 1, 1)
        r = rhs - (self.apply_normal(x) + lam * x)
        p = r.clone()
        rr = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)
        eps = rhs.new_tensor(1e-12)
        for _ in range(int(max_iter)):
            Ap = self.apply_normal(p) + lam * p
            denom = torch.sum(p * Ap, dim=(1, 2, 3), keepdim=True).clamp_min(eps)
            alpha = rr / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)
            if torch.sqrt(rr_new.max()).item() < float(tol):
                break
            cg_ratio = rr_new / (rr + eps)
            p = r + cg_ratio * p
            rr = rr_new
        return x

    @torch.no_grad()
    def choose_lambda_morozov(self, b: torch.Tensor, noise_norm: torch.Tensor, tau: float = 1.0, max_iter: int = 8, lambda_min: float = 1e-12, lambda_max: float = 1e12) -> torch.Tensor:
        context = self._prepare_gram_context(b)
        settings = _morozov_settings(max_iter=max_iter, lambda_min=lambda_min, lambda_max=lambda_max)
        return _choose_lambda_morozov_from_gram_spectrum(
            b=context["b"],
            rhs=context["rhs"],
            noise_norm=noise_norm.to(dtype=torch.float32, device=context["b"].device),
            eigvals=context["eigvals"],
            eigvecs=context["eigvecs"],
            tau=float(tau),
            settings=settings,
            rhs_proj=context["rhs_proj"],
            b_norm2=context["b_norm2"],
        )


def _resolve_theoretical_formula_mode(formula_mode: str | None = None, solver_mode: str | None = None) -> str:
    resolved = "alpha_continuous" if formula_mode is None else str(formula_mode).strip().lower()
    if resolved in {"", "auto"}:
        return "alpha_continuous"
    if resolved == "alpha_continuous":
        return resolved
    raise ValueError(f"Unsupported theoretical_formula_mode={formula_mode!r}; only 'alpha_continuous' is supported.")


def _time_domain_sampling_kwargs() -> dict[str, object]:
    sampling_mode = str(TIME_DOMAIN_CONFIG.get("sampling_mode", "shifted_lattice") or "shifted_lattice").strip().lower().replace("-", "_")
    return {
        "sampling_mode": sampling_mode,
        "num_detector_samples": int(TIME_DOMAIN_CONFIG.get("num_detector_samples", IMAGE_SIZE * IMAGE_SIZE)),
        "detector_phase": float(TIME_DOMAIN_CONFIG.get("detector_phase", 0.5)),
        "detector_margin_ratio": float(TIME_DOMAIN_CONFIG.get("detector_margin_ratio", 0.0)),
        "subset_selection": str(TIME_DOMAIN_CONFIG.get("subset_selection", "uniform") or "uniform"),
        "edge_boundary_fraction": float(TIME_DOMAIN_CONFIG.get("edge_boundary_fraction", 0.2)),
        "edge_weight": int(TIME_DOMAIN_CONFIG.get("edge_weight", 3)),
        "middle_weight": int(TIME_DOMAIN_CONFIG.get("middle_weight", 1)),
    }


def _detector_sampling_points_from_records(records, *, expected_count: int, num_detector_samples: int):
    record_list = list(records or [])
    if not record_list:
        return None
    if int(len(record_list)) != int(expected_count):
        return None
    points_per_angle = []
    for item in record_list:
        values = item.get("detector_sampling_points") if isinstance(item, dict) else None
        if values is None:
            return None
        tensor = torch.as_tensor(values, dtype=torch.float64).view(-1)
        if int(tensor.numel()) != int(num_detector_samples):
            raise ValueError(
                f"detector_sampling_points length={int(tensor.numel())} but "
                f"num_detector_samples={int(num_detector_samples)}."
            )
        points_per_angle.append(tensor)
    return points_per_angle


def build_time_domain_operator(height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> torch.nn.Module:
    """Build the single retained alpha-continuous operator."""
    alpha_values = TIME_DOMAIN_CONFIG.get("alpha_values") or []
    tau_offsets = TIME_DOMAIN_CONFIG.get("alpha_tau_offsets") or []
    sampling_kwargs = _time_domain_sampling_kwargs()
    if not alpha_values:
        raise ValueError("alpha_continuous operator requires TIME_DOMAIN_CONFIG['alpha_values'].")
    if str(sampling_kwargs["sampling_mode"]) == "ct_detector_grid":
        tau_offsets_for_operator = None
        sampling_points_per_angle = _detector_sampling_points_from_records(
            TIME_DOMAIN_CONFIG.get("alpha_condition_constrained_records") or [],
            expected_count=len(alpha_values),
            num_detector_samples=int(sampling_kwargs["num_detector_samples"]),
        )
    else:
        if not tau_offsets:
            raise ValueError("shifted_lattice operator requires TIME_DOMAIN_CONFIG['alpha_tau_offsets'].")
        if len(alpha_values) != len(tau_offsets):
            raise ValueError(f"alpha_values and alpha_tau_offsets length mismatch: {len(alpha_values)} vs {len(tau_offsets)}.")
        tau_offsets_for_operator = tau_offsets
        sampling_points_per_angle = None
    return AlphaContinuousB1B1Operator2D(
        alpha_values=alpha_values,
        tau_offsets=tau_offsets_for_operator,
        sampling_points_per_angle=sampling_points_per_angle,
        height=int(height),
        width=int(width),
        **sampling_kwargs,
    ).to(device)


def build_time_domain_operator_from_alpha_records(records, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> torch.nn.Module:
    """Build an alpha-continuous operator from selected JSON-style records."""
    record_list = list(records or [])
    alpha_values = [float(item["alpha"]) for item in record_list]
    sampling_kwargs = _time_domain_sampling_kwargs()
    if not alpha_values:
        raise ValueError("init alpha records must contain at least one selected angle.")
    if str(sampling_kwargs["sampling_mode"]) == "ct_detector_grid":
        tau_offsets = None
        sampling_points_per_angle = _detector_sampling_points_from_records(
            record_list,
            expected_count=len(alpha_values),
            num_detector_samples=int(sampling_kwargs["num_detector_samples"]),
        )
    else:
        tau_offsets = [float(item["tau_star"] if "tau_star" in item else item["tau"]) for item in record_list]
        if len(alpha_values) != len(tau_offsets):
            raise ValueError(f"init alpha/tau length mismatch: {len(alpha_values)} vs {len(tau_offsets)}.")
        sampling_points_per_angle = None
    return AlphaContinuousB1B1Operator2D(
        alpha_values=alpha_values,
        tau_offsets=tau_offsets,
        sampling_points_per_angle=sampling_points_per_angle,
        height=int(height),
        width=int(width),
        **sampling_kwargs,
    ).to(device)


def _resolve_data_formula_mode(reconstruction_formula_mode: str) -> str:
    raw = str(TIME_DOMAIN_CONFIG.get("data_formula_mode", "auto_complete") or "").strip().lower()
    if raw in {"", "auto", "auto_complete", "alpha_continuous"}:
        return "alpha_continuous"
    raise ValueError(f"Unsupported data_formula_mode={raw!r}; only alpha_continuous data generation is supported.")


class TheoreticalDataGenerator:
    """Generate coefficient maps, alpha-continuous observations, and Tikhonov initializations."""

    def __init__(self, img_size=IMAGE_SIZE, data_source: Optional[str] = None, time_operator: Optional[torch.nn.Module] = None):
        self.img_size = int(img_size)
        self.N = self.img_size * self.img_size
        self.data_source = str(data_source or DATA_CONFIG.get("data_source", "random_ellipses")).strip().lower()
        self.noise_mode = str(DATA_CONFIG.get("noise_mode", "multiplicative")).strip().lower()
        self.noise_level = float(DATA_CONFIG.get("noise_level", 0.1))
        self.target_snr_db = float(DATA_CONFIG.get("target_snr_db", 30.0))
        self.image_gen = DifferentiableImageGenerator(self.img_size)
        self._phantom_cache: Optional[torch.Tensor] = None
        self.time_operator = time_operator if time_operator is not None else build_time_domain_operator(height=self.img_size, width=self.img_size)
        reconstruction_formula_mode = str(TIME_DOMAIN_CONFIG.get("theoretical_formula_mode", "alpha_continuous")).strip().lower()
        self.data_formula_mode = _resolve_data_formula_mode(reconstruction_formula_mode)
        self.data_time_operator = self.time_operator
        init_records = TIME_DOMAIN_CONFIG.get("init_alpha_condition_constrained_records", None)
        if init_records:
            self.init_time_operator = build_time_domain_operator_from_alpha_records(
                init_records,
                height=self.img_size,
                width=self.img_size,
            )
        else:
            self.init_time_operator = self.time_operator
        self.feature_time_operator = None
        self.M = int(getattr(self.time_operator, "M", int(TIME_DOMAIN_CONFIG.get("num_detector_samples", self.N))))
        self.last_lambda: Optional[float | torch.Tensor] = None
        self.last_lambda_info: Optional[dict[str, object]] = None
        self._chol_lambda: Optional[float] = None
        self._chol_factor: Optional[torch.Tensor] = None
        self._ata_factor: Optional[torch.Tensor] = None
        self._first_batch_progress_logged = False

    def _normalize_lambda_reg(self, lambda_reg: float | torch.Tensor, batch_size: int, *, dtype: torch.dtype = torch.float32, target_device: Optional[torch.device] = None) -> torch.Tensor:
        if target_device is None:
            target_device = device
        if torch.is_tensor(lambda_reg):
            lam = lambda_reg.detach().to(dtype=dtype, device=target_device).view(-1)
            if int(lam.numel()) == 1 and batch_size > 1:
                lam = lam.expand(batch_size)
            elif int(lam.numel()) != batch_size:
                raise ValueError(f"lambda_reg has {int(lam.numel())} entries, expected 1 or batch={batch_size}.")
        else:
            lam = torch.full((batch_size,), float(lambda_reg), dtype=dtype, device=target_device)
        return lam

    def _l1_init_lambda_to_solver_scale(self, lambda_reg: float | torch.Tensor, batch_size: int, *, measurement_count: int, target_device: torch.device) -> torch.Tensor:
        lam = self._normalize_lambda_reg(lambda_reg, batch_size=batch_size, dtype=torch.float32, target_device=target_device)
        return lam * float(max(int(measurement_count), 1))

    def forward_operator(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.time_operator.forward(coeff_matrix)

    def data_forward_operator(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.data_time_operator.forward(coeff_matrix)

    def init_forward_operator(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.init_time_operator.forward(coeff_matrix)

    def adjoint_operator(self, residual: torch.Tensor) -> torch.Tensor:
        return self.time_operator.adjoint(residual)

    @contextmanager
    def _using_init_operator(self):
        old_operator = self.time_operator
        self.time_operator = self.init_time_operator
        try:
            yield
        finally:
            self.time_operator = old_operator

    @torch.no_grad()
    def solve_tikhonov_direct_init(self, g_obs: torch.Tensor, lambda_reg: float | torch.Tensor) -> torch.Tensor:
        return self._tikhonov_direct_init(g_obs, lambda_reg=lambda_reg)

    @torch.no_grad()
    def solve_regularized_init(
        self,
        g_obs: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        *,
        init_method: Optional[str] = None,
    ) -> torch.Tensor:
        method = normalize_init_method(str(init_method or TIME_DOMAIN_CONFIG.get("init_method", "cg")))
        if method == "tikhonov_direct":
            return self._tikhonov_direct_init(g_obs, lambda_reg=lambda_reg)
        if method == "cg":
            init_cg_iters = int(TIME_DOMAIN_CONFIG.get("init_cg_iters", 0))
            if init_cg_iters <= 0:
                raise ValueError("init_method='cg' requires TIME_DOMAIN_CONFIG['init_cg_iters'] > 0.")
            return self._tikhonov_cg_init(g_obs, lambda_reg=lambda_reg, max_iter=init_cg_iters)
        if method == "l2_l1_admm":
            if not hasattr(self.time_operator, "solve_l2_l1_admm"):
                raise ValueError("Active operator does not expose solve_l2_l1_admm().")
            if g_obs.dim() == 1:
                g_obs = g_obs.unsqueeze(0)
            g_obs = g_obs.to(device=device, dtype=torch.float32)
            lam_solver = self._l1_init_lambda_to_solver_scale(
                lambda_reg,
                batch_size=int(g_obs.shape[0]),
                measurement_count=int(g_obs.shape[-1]),
                target_device=g_obs.device,
            )
            return self.time_operator.solve_l2_l1_admm(g_obs, lambda_reg=lam_solver)
        if method == "l1_l1_admm":
            if not hasattr(self.time_operator, "solve_l1_l1_admm"):
                raise ValueError("Active operator does not expose solve_l1_l1_admm().")
            if g_obs.dim() == 1:
                g_obs = g_obs.unsqueeze(0)
            g_obs = g_obs.to(device=device, dtype=torch.float32)
            lam_solver = self._l1_init_lambda_to_solver_scale(
                lambda_reg,
                batch_size=int(g_obs.shape[0]),
                measurement_count=int(g_obs.shape[-1]),
                target_device=g_obs.device,
            )
            return self.time_operator.solve_l1_l1_admm(g_obs, lambda_reg=lam_solver)
        if method == "l2_tv_admm":
            if not hasattr(self.time_operator, "solve_l2_tv_admm"):
                raise ValueError("Active operator does not expose solve_l2_tv_admm().")
            if g_obs.dim() == 1:
                g_obs = g_obs.unsqueeze(0)
            g_obs = g_obs.to(device=device, dtype=torch.float32)
            lam_solver = self._l1_init_lambda_to_solver_scale(
                lambda_reg,
                batch_size=int(g_obs.shape[0]),
                measurement_count=int(g_obs.shape[-1]),
                target_device=g_obs.device,
            )
            return self.time_operator.solve_l2_tv_admm(g_obs, lambda_reg=lam_solver)
        if method == "l2_tv_pdhg":
            if not hasattr(self.time_operator, "solve_l2_tv_pdhg"):
                raise ValueError("Active operator does not expose solve_l2_tv_pdhg().")
            if g_obs.dim() == 1:
                g_obs = g_obs.unsqueeze(0)
            g_obs = g_obs.to(device=device, dtype=torch.float32)
            lam_solver = self._l1_init_lambda_to_solver_scale(
                lambda_reg,
                batch_size=int(g_obs.shape[0]),
                measurement_count=int(g_obs.shape[-1]),
                target_device=g_obs.device,
            )
            x0 = self._tikhonov_direct_init(g_obs, lambda_reg=lambda_reg)
            coeff = self.time_operator.solve_l2_tv_pdhg(
                g_obs,
                lambda_reg=lam_solver,
                max_iter=int(DATA_CONFIG.get("tv_pdhg_iters", 10)),
                theta=float(DATA_CONFIG.get("tv_pdhg_theta", 1.0)),
                nonnegative=bool(DATA_CONFIG.get("tv_pdhg_nonnegative", False)),
                x0=x0,
                power_iters=int(DATA_CONFIG.get("tv_pdhg_power_iters", 8)),
            )
            info = dict(self.last_lambda_info or {})
            if info:
                info["solver_stats"] = dict(getattr(self.time_operator, "last_pdhg_stats", None) or {})
                self.last_lambda_info = info
            return coeff
        raise ValueError(
            f"Unsupported init_method={method!r}; expected one of {list(INIT_METHOD_CHOICES)!r}."
        )

    @torch.no_grad()
    def solve_constrained_init(
        self,
        g_obs: torch.Tensor,
        noise_radius: float | torch.Tensor,
        *,
        init_method: Optional[str] = None,
    ) -> torch.Tensor:
        method = normalize_init_method(str(init_method or TIME_DOMAIN_CONFIG.get("init_method", "cg")))
        if g_obs.dim() == 1:
            g_obs = g_obs.unsqueeze(0)
        g_obs = g_obs.to(device=device, dtype=torch.float32)
        radius = self._normalize_lambda_reg(noise_radius, batch_size=int(g_obs.shape[0]), dtype=torch.float32, target_device=g_obs.device)
        if method == "l2_l1_admm":
            if not hasattr(self.time_operator, "solve_l2_l1_morozov_admm"):
                raise ValueError("Active operator does not expose solve_l2_l1_morozov_admm().")
            return self.time_operator.solve_l2_l1_morozov_admm(g_obs, noise_radius=radius)
        if method == "l1_l1_admm":
            if not hasattr(self.time_operator, "solve_l1_l1_morozov_admm"):
                raise ValueError("Active operator does not expose solve_l1_l1_morozov_admm().")
            return self.time_operator.solve_l1_l1_morozov_admm(g_obs, noise_radius=radius)
        if method == "l2_tv_admm":
            if not hasattr(self.time_operator, "solve_l2_tv_morozov_admm"):
                raise ValueError("Active operator does not expose solve_l2_tv_morozov_admm().")
            return self.time_operator.solve_l2_tv_morozov_admm(g_obs, noise_radius=radius)
        raise ValueError(
            "Constrained Morozov initialization is only supported for "
            f"'l2_l1_admm', 'l1_l1_admm', and 'l2_tv_admm', got {method!r}."
        )

    @torch.no_grad()
    def solve_morozov_constrained_init(
        self,
        g_obs: torch.Tensor,
        *,
        init_method: Optional[str] = None,
    ) -> torch.Tensor:
        method = normalize_init_method(str(init_method or TIME_DOMAIN_CONFIG.get("init_method", "cg")))
        if method == "l2_l1_admm":
            norm_type = "l2"
        elif method == "l1_l1_admm":
            norm_type = "l1"
        elif method == "l2_tv_admm":
            norm_type = "l2"
        else:
            raise ValueError(
                "Constrained Morozov initialization is only supported for "
                f"'l2_l1_admm', 'l1_l1_admm', and 'l2_tv_admm', got {method!r}."
            )
        radius_base, radius_source = self._estimate_morozov_noise_norm_from_observed(g_obs, norm_type=norm_type)
        radius = radius_base * float(DATA_CONFIG.get("morozov_tau", 1.0))
        info = {
            "mode": "morozov_constrained_radius",
            "method": method,
            "residual_norm": norm_type,
            "target_norm": [float(v) for v in radius.detach().cpu().view(-1).tolist()],
            "constraint_radius": [float(v) for v in radius.detach().cpu().view(-1).tolist()],
            "noise_radius_source": radius_source,
            "noise_mode": str(self.noise_mode),
            "noise_level": float(self.noise_level),
        }
        coeff = self.solve_constrained_init(g_obs, noise_radius=radius, init_method=method)
        stats = dict(getattr(self.time_operator, "last_split_admm_stats", None) or {})
        info.update(
            {
                "mode": "morozov_constrained",
                "constraint_radius": [float(v) for v in radius.detach().cpu().view(-1).tolist()] if torch.is_tensor(radius) else [float(radius)],
                "solver_stats": stats,
            }
        )
        self.last_lambda = radius
        self.last_lambda_info = info
        return coeff

    @torch.no_grad()
    def _tikhonov_direct_init(self, g_obs: torch.Tensor, lambda_reg: float | torch.Tensor) -> torch.Tensor:
        if g_obs.dim() == 1:
            g_obs = g_obs.unsqueeze(0)
        g_obs = g_obs.to(device=device, dtype=torch.float32)
        lam_batch = self._normalize_lambda_reg(lambda_reg, batch_size=int(g_obs.shape[0]), dtype=torch.float32, target_device=g_obs.device)
        if hasattr(self.time_operator, "solve_tikhonov_direct") and not hasattr(self.time_operator, "A"):
            return self.time_operator.solve_tikhonov_direct(g_obs, lambda_reg=lam_batch)
        if hasattr(self.time_operator, "solve_tikhonov_cg") and not hasattr(self.time_operator, "A"):
            cg_iters = max(int(TIME_DOMAIN_CONFIG.get("init_cg_iters", 40)), 40)
            cg_tol = float(TIME_DOMAIN_CONFIG.get("init_cg_tol", 1e-4))
            return self.time_operator.solve_tikhonov_cg(g_obs, lambda_reg=lam_batch, max_iter=cg_iters, tol=cg_tol)
        raise ValueError("Active operator does not expose a Tikhonov solver.")

    @torch.no_grad()
    def _tikhonov_cg_init(self, g_obs: torch.Tensor, lambda_reg: float | torch.Tensor, max_iter: int, tol: float = 1e-6) -> torch.Tensor:
        if g_obs.dim() == 1:
            g_obs = g_obs.unsqueeze(0)
        g_obs = g_obs.to(device=device, dtype=torch.float32)
        lam_batch = self._normalize_lambda_reg(lambda_reg, batch_size=int(g_obs.shape[0]), dtype=torch.float32, target_device=g_obs.device)
        return self.time_operator.solve_tikhonov_cg(g_obs, lambda_reg=lam_batch, max_iter=max_iter, tol=tol)

    def _sample_coefficients(self, batch_size: int = 1) -> torch.Tensor:
        if self.data_source == "shepp_logan":
            if self._phantom_cache is None:
                phantom = generate_shepp_logan_phantom(image_size=self.img_size, modified=True, device=device, dtype=torch.float32)
                self._phantom_cache = phantom.view(1, 1, self.img_size, self.img_size)
            return self._phantom_cache.expand(batch_size, -1, -1, -1)
        if self.data_source in ("random_ellipses", "random_ellipse", "ellipse"):
            phantom_list = [generate_random_ellipse_phantom(image_size=self.img_size) for _ in range(int(batch_size))]
            return torch.stack(phantom_list, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)
        raise ValueError(f"Unsupported data_source={self.data_source!r}; expected 'random_ellipses' or 'shepp_logan'.")

    def _apply_noise(self, g_clean: torch.Tensor) -> torch.Tensor:
        if self.noise_mode == "multiplicative":
            rand_u = 2.0 * torch.rand_like(g_clean) - 1.0
            return g_clean + (self.noise_level * g_clean * rand_u)
        if self.noise_mode == "additive":
            return g_clean + (self.noise_level * torch.randn_like(g_clean))
        if self.noise_mode == "snr":
            if g_clean.dim() == 1:
                signal_energy = torch.sum(g_clean ** 2)
                numel = g_clean.numel()
            else:
                signal_energy = torch.sum(g_clean ** 2, dim=-1, keepdim=True)
                numel = g_clean.shape[-1]
            sigma_squared = signal_energy / (numel * (10 ** (self.target_snr_db / 10.0)))
            sigma = torch.sqrt(sigma_squared).to(g_clean)
            return g_clean + (torch.randn_like(g_clean) * sigma)
        raise ValueError(f"Unsupported noise_mode={self.noise_mode!r}; expected 'additive', 'multiplicative', or 'snr'.")

    @torch.no_grad()
    def _measurement_residual_norm(self, coeff: torch.Tensor, observed: torch.Tensor, *, norm_type: str) -> torch.Tensor:
        residual = self.forward_operator(coeff) - observed.to(dtype=torch.float32, device=coeff.device)
        norm_type = str(norm_type).strip().lower()
        if norm_type == "l1":
            return torch.sum(torch.abs(residual), dim=-1)
        if norm_type == "l2":
            return torch.norm(residual, dim=-1)
        raise ValueError(f"Unsupported residual norm_type={norm_type!r}; expected 'l1' or 'l2'.")

    @torch.no_grad()
    def _observed_data_norm(self, g_observed: torch.Tensor, *, norm_type: str) -> torch.Tensor:
        if g_observed.dim() == 1:
            g_observed = g_observed.unsqueeze(0)
        g_observed = g_observed.to(device=device, dtype=torch.float32)
        norm_type = str(norm_type).strip().lower()
        if norm_type == "l1":
            return torch.sum(torch.abs(g_observed), dim=-1)
        if norm_type == "l2":
            return torch.norm(g_observed, dim=-1)
        raise ValueError(f"Unsupported norm_type={norm_type!r}; expected 'l1' or 'l2'.")

    @torch.no_grad()
    def _estimate_morozov_noise_norm_from_observed(self, g_observed: torch.Tensor, *, norm_type: str) -> tuple[torch.Tensor, str]:
        """Estimate Morozov noise size from observed data only.

        For multiplicative noise

            g_delta_i = g_i * (1 + alpha * xi_i),  xi_i in [-1, 1],

        the clean datum ``g`` is unknown at reconstruction time.  Two observed
        data-only modes are supported:

        * ``rms`` (default): use the second moment of xi ~ U(-1, 1),

              ||noise|| ~= alpha / sqrt(3 + alpha^2) * ||g_delta||.

        * ``conservative``: use the deterministic upper bound

              ||noise|| <= alpha / (1 - alpha) * ||g_delta||.

        Neither mode uses ``g_clean`` as reconstruction-time input.
        """
        if g_observed.dim() == 1:
            g_observed = g_observed.unsqueeze(0)
        g_observed = g_observed.to(device=device, dtype=torch.float32)
        mode = str(self.noise_mode).strip().lower()
        norm_type = str(norm_type).strip().lower()
        if mode == "multiplicative":
            alpha = float(self.noise_level)
            if alpha < 0.0:
                raise ValueError(f"multiplicative noise_level must be non-negative, got {alpha!r}.")
            radius_mode = str(DATA_CONFIG.get("morozov_noise_radius_mode", "rms")).strip().lower()
            if radius_mode == "rms":
                scale = alpha / math.sqrt(3.0 + alpha * alpha)
                return scale * self._observed_data_norm(g_observed, norm_type=norm_type), "observed_multiplicative_rms"
            if radius_mode == "conservative":
                if alpha >= 1.0:
                    raise ValueError(
                        "Conservative observed-data Morozov bound for multiplicative noise requires noise_level < 1. "
                        f"Got noise_level={alpha!r}."
                    )
                scale = alpha / max(1.0 - alpha, 1.0e-12)
                return scale * self._observed_data_norm(g_observed, norm_type=norm_type), "observed_multiplicative_conservative"
            raise ValueError(
                f"Unsupported morozov_noise_radius_mode={radius_mode!r}; expected 'rms' or 'conservative'."
            )
        if mode == "snr":
            radius_mode = str(DATA_CONFIG.get("morozov_noise_radius_mode", "rms")).strip().lower()
            epsilon = 10.0 ** (-float(self.target_snr_db) / 20.0)
            if radius_mode == "rms":
                return epsilon * self._observed_data_norm(g_observed, norm_type=norm_type), "observed_snr_rms"
            if radius_mode == "conservative":
                if epsilon >= 1.0:
                    raise ValueError(
                        "Conservative observed-data Morozov bound for SNR noise requires target_snr_db > 0. "
                        f"Got target_snr_db={float(self.target_snr_db)!r}."
                    )
                scale = epsilon / max(1.0 - epsilon, 1.0e-12)
                return scale * self._observed_data_norm(g_observed, norm_type=norm_type), "observed_snr_conservative"
            raise ValueError(
                f"Unsupported morozov_noise_radius_mode={radius_mode!r}; expected 'rms' or 'conservative'."
            )
        if mode == "additive":
            sigma = max(float(self.noise_level), 0.0)
            batch = int(g_observed.shape[0])
            m = max(int(g_observed.shape[-1]), 1)
            if norm_type == "l2":
                value = sigma * math.sqrt(float(m))
            elif norm_type == "l1":
                value = sigma * float(m) * math.sqrt(2.0 / math.pi)
            else:
                raise ValueError(f"Unsupported norm_type={norm_type!r}; expected 'l1' or 'l2'.")
            return torch.full((batch,), float(value), dtype=torch.float32, device=g_observed.device), "known_additive_expected"
        raise ValueError(f"Unsupported noise_mode={mode!r}; expected 'additive', 'multiplicative', or 'snr'.")

    @torch.no_grad()
    def _morozov_noise_radius(self, g_observed: torch.Tensor, *, norm_type: str) -> torch.Tensor:
        radius, _ = self._estimate_morozov_noise_norm_from_observed(g_observed, norm_type=norm_type)
        return radius * float(DATA_CONFIG.get("morozov_tau", 1.0))

    @torch.no_grad()
    def _choose_lambda_morozov_iterative(
        self,
        g_observed: torch.Tensor,
        *,
        init_method: str,
        residual_norm: str,
    ) -> torch.Tensor:
        if g_observed.dim() == 1:
            g_observed = g_observed.unsqueeze(0)
        g_observed = g_observed.to(device=device, dtype=torch.float32)
        batch = int(g_observed.shape[0])
        residual_norm = str(residual_norm).strip().lower()
        noise_norm, noise_radius_source = self._estimate_morozov_noise_norm_from_observed(
            g_observed,
            norm_type=residual_norm,
        )
        target = noise_norm * float(DATA_CONFIG.get("morozov_tau", 1.0))

        lam_min = max(float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12)), 1.0e-30)
        configured_lam_max = max(float(DATA_CONFIG.get("morozov_lambda_max", 1.0e12)), lam_min * 10.0)
        max_iter = max(0, int(DATA_CONFIG.get("morozov_max_iter", 8)))
        lambda_max_source = "configured"
        lambda_scale = "raw"
        measurement_count = int(g_observed.shape[-1])
        natural_lam_max: Optional[torch.Tensor] = None
        natural_lam_max_is_exact_zero_threshold = False
        if init_method == "l2_l1_admm" and hasattr(self.time_operator, "l2_l1_zero_threshold"):
            natural_lam_max = self.time_operator.l2_l1_zero_threshold(g_observed).to(dtype=torch.float32, device=g_observed.device) / float(max(measurement_count, 1))
            lambda_max_source = "zero_solution_threshold"
            lambda_scale = "normalized_by_measurements"
            natural_lam_max_is_exact_zero_threshold = True
        elif init_method == "l1_l1_admm" and hasattr(self.time_operator, "l1_l1_zero_threshold"):
            natural_lam_max = self.time_operator.l1_l1_zero_threshold(g_observed).to(dtype=torch.float32, device=g_observed.device) / float(max(measurement_count, 1))
            lambda_max_source = "zero_solution_threshold"
            lambda_scale = "normalized_by_measurements"
            natural_lam_max_is_exact_zero_threshold = True
        elif init_method == "l2_tv_admm" and hasattr(self.time_operator, "l2_l1_zero_threshold"):
            # TV has a constant-image nullspace, so the exact zero-solution
            # threshold used by pixel L1 does not apply.  Use the L2/L1
            # threshold only as a finite, data-scaled upper-bound proxy so
            # Morozov does not start from the global 1e12 configuration.
            natural_lam_max = self.time_operator.l2_l1_zero_threshold(g_observed).to(dtype=torch.float32, device=g_observed.device) / float(max(measurement_count, 1))
            lambda_max_source = "l2_l1_zero_threshold_proxy"
            lambda_scale = "normalized_by_measurements"
        elif init_method in MEASUREMENT_NORMALIZED_REGULARIZED_INIT_METHODS:
            lambda_scale = "normalized_by_measurements"
        lam_values: list[float] = []
        final_residuals: list[float] = []
        statuses: list[str] = []
        lambda_max_values: list[float] = []

        for idx in range(batch):
            observed_i = g_observed[idx : idx + 1]
            target_i = float(target[idx].item())
            natural_hi_i = None
            if natural_lam_max is not None:
                natural_hi_i = max(float(natural_lam_max[idx].item()), lam_min * 10.0)

            def evaluate(lam_value: float) -> float:
                if natural_lam_max_is_exact_zero_threshold and natural_hi_i is not None and lam_value >= natural_hi_i * (1.0 - 1.0e-7):
                    if residual_norm == "l1":
                        return float(torch.sum(torch.abs(observed_i), dim=-1).view(-1)[0].item())
                    return float(torch.norm(observed_i, dim=-1).view(-1)[0].item())
                lam_tensor = torch.tensor([float(lam_value)], dtype=torch.float32, device=observed_i.device)
                coeff = self.solve_regularized_init(observed_i, lambda_reg=lam_tensor, init_method=init_method)
                return float(self._measurement_residual_norm(coeff, observed_i, norm_type=residual_norm).view(-1)[0].item())

            lo = float(lam_min)
            hi = float(configured_lam_max if natural_hi_i is None else min(configured_lam_max, natural_hi_i))
            hi = max(hi, lo * 10.0)
            res_lo = evaluate(lo)
            res_hi = evaluate(hi)
            candidates = [(abs(res_lo - target_i), lo, res_lo), (abs(res_hi - target_i), hi, res_hi)]

            if init_method == "l2_tv_pdhg":
                # Low-iteration PDHG is intentionally inexact for speed.  Its
                # residual as a function of lambda can be non-monotone near
                # tiny lambda values, so endpoint-only Morozov bracketing may
                # miss a good interior lambda and jump to the configured upper
                # limit.  Scan the configured log range first, then refine a
                # sign-changing interval if one is found.
                grid_size = max(3, int(max_iter))
                log_lo = math.log10(max(lo, 1.0e-30))
                log_hi = math.log10(max(hi, lo * 10.0))
                if log_hi > log_lo:
                    for pos in range(grid_size):
                        frac = 0.0 if grid_size == 1 else float(pos) / float(grid_size - 1)
                        lam_grid = 10.0 ** (log_lo + frac * (log_hi - log_lo))
                        if lam_grid <= lo * (1.0 + 1.0e-12) or lam_grid >= hi * (1.0 - 1.0e-12):
                            continue
                        res_grid = evaluate(lam_grid)
                        candidates.append((abs(res_grid - target_i), lam_grid, res_grid))
                ordered = sorted((lam, res) for _, lam, res in candidates)
                brackets = []
                for (lam_a, res_a), (lam_b, res_b) in zip(ordered, ordered[1:]):
                    diff_a = res_a - target_i
                    diff_b = res_b - target_i
                    if diff_a == 0.0:
                        brackets.append((lam_a, lam_a, res_a, res_a))
                    if diff_a * diff_b <= 0.0:
                        brackets.append((lam_a, lam_b, res_a, res_b))
                if brackets:
                    bracket = max(brackets, key=lambda item: item[1])
                    a, b_hi, res_a, res_b = bracket
                    if a == b_hi:
                        best_lam = a
                        best_res = res_a
                    else:
                        bracket_candidates = [
                            (abs(res_a - target_i), a, res_a),
                            (abs(res_b - target_i), b_hi, res_b),
                        ]
                        for _ in range(max_iter):
                            mid = math.sqrt(max(a, 1.0e-30) * max(b_hi, 1.0e-30))
                            res_mid = evaluate(mid)
                            candidates.append((abs(res_mid - target_i), mid, res_mid))
                            bracket_candidates.append((abs(res_mid - target_i), mid, res_mid))
                            if (res_a - target_i) * (res_mid - target_i) <= 0.0:
                                b_hi = mid
                                res_b = res_mid
                            else:
                                a = mid
                                res_a = res_mid
                        _, best_lam, best_res = min(bracket_candidates, key=lambda item: item[0])
                    status = "bracketed_log_grid"
                else:
                    _, best_lam, best_res = min(candidates, key=lambda item: item[0])
                    status = "log_grid_best"
            elif target_i <= res_lo:
                _, best_lam, best_res = min(candidates, key=lambda item: item[0])
                status = "target_below_or_equal_lambda_min"
            elif target_i >= res_hi:
                _, best_lam, best_res = min(candidates, key=lambda item: item[0])
                status = "target_above_or_equal_lambda_max"
                if natural_lam_max_is_exact_zero_threshold and natural_hi_i is not None and hi <= natural_hi_i * (1.0 + 1.0e-7):
                    status = "target_above_or_equal_zero_solution"
            else:
                status = "bracketed"
                best_lam = lo
                best_res = res_lo
                for _ in range(max_iter):
                    mid = math.sqrt(lo * hi)
                    res_mid = evaluate(mid)
                    candidates.append((abs(res_mid - target_i), mid, res_mid))
                    if res_mid < target_i:
                        lo = mid
                    else:
                        hi = mid
                _, best_lam, best_res = min(candidates, key=lambda item: item[0])

            lam_values.append(float(best_lam))
            final_residuals.append(float(best_res))
            statuses.append(status)
            lambda_max_values.append(float(hi))

        lam = torch.tensor(lam_values, dtype=torch.float32, device=g_observed.device)
        self.last_lambda_info = {
            "mode": "morozov_iterative",
            "method": str(init_method),
            "residual_norm": residual_norm,
            "lambda_min": float(lam_min),
            "lambda_max": lambda_max_values,
            "lambda_max_source": lambda_max_source,
            "lambda_scale": lambda_scale,
            "measurement_count": int(measurement_count),
            "lambda_raw_value": [float(v) * float(max(measurement_count, 1)) if lambda_scale == "normalized_by_measurements" else float(v) for v in lam_values],
            "lambda_raw_max": [float(v) * float(max(measurement_count, 1)) if lambda_scale == "normalized_by_measurements" else float(v) for v in lambda_max_values],
            "max_iter": int(max_iter),
            "target_norm": [float(v) for v in target.detach().cpu().view(-1).tolist()],
            "noise_radius_source": noise_radius_source,
            "noise_mode": str(self.noise_mode),
            "noise_level": float(self.noise_level),
            "residual_norm_value": final_residuals,
            "status": statuses,
        }
        return lam

    def select_lambda_for_init_method(
        self,
        g_observed: torch.Tensor,
        *,
        init_method: Optional[str] = None,
        lambda_reg: float | torch.Tensor = None,
    ) -> float | torch.Tensor:
        if lambda_reg is not None:
            if torch.is_tensor(lambda_reg):
                provided_method = normalize_init_method(str(init_method or TIME_DOMAIN_CONFIG.get("init_method", "cg")))
                self.last_lambda_info = {
                    "mode": "provided",
                    "method": provided_method,
                    "lambda_scale": "normalized_by_measurements" if provided_method in MEASUREMENT_NORMALIZED_REGULARIZED_INIT_METHODS else "raw",
                }
                return lambda_reg.to(dtype=torch.float32, device=g_observed.device)
            provided_method = normalize_init_method(str(init_method or TIME_DOMAIN_CONFIG.get("init_method", "cg")))
            self.last_lambda_info = {
                "mode": "provided",
                "method": provided_method,
                "lambda_scale": "normalized_by_measurements" if provided_method in MEASUREMENT_NORMALIZED_REGULARIZED_INIT_METHODS else "raw",
            }
            return float(lambda_reg)
        mode = str(DATA_CONFIG.get("lambda_select_mode", "fixed")).strip().lower()
        method = normalize_init_method(str(init_method or TIME_DOMAIN_CONFIG.get("init_method", "cg")))
        if mode == "fixed":
            self.last_lambda_info = {
                "mode": "fixed",
                "method": method,
                "residual_norm": "none",
                "lambda_scale": "normalized_by_measurements" if method in MEASUREMENT_NORMALIZED_REGULARIZED_INIT_METHODS else "raw",
            }
            return float(DATA_CONFIG.get("lambda_reg", 1e-2))
        if mode == "morozov":
            morozov_form = str(DATA_CONFIG.get("morozov_form", "regularized")).strip().lower()
            if morozov_form not in {"regularized", "constrained"}:
                raise ValueError(
                    f"Unsupported morozov_form={morozov_form!r}; expected 'regularized' or 'constrained'."
                )
            if morozov_form == "constrained":
                if method == "l2_l1_admm":
                    norm_type = "l2"
                elif method == "l1_l1_admm":
                    norm_type = "l1"
                elif method == "l2_tv_admm":
                    norm_type = "l2"
                else:
                    raise ValueError(
                        "morozov_form='constrained' is only supported for "
                        f"'l2_l1_admm', 'l1_l1_admm', and 'l2_tv_admm', got {method!r}."
                    )
                radius_base, radius_source = self._estimate_morozov_noise_norm_from_observed(g_observed, norm_type=norm_type)
                radius = radius_base * float(DATA_CONFIG.get("morozov_tau", 1.0))
                self.last_lambda_info = {
                    "mode": "morozov_constrained_radius",
                    "method": method,
                    "residual_norm": norm_type,
                    "target_norm": [float(v) for v in radius.detach().cpu().view(-1).tolist()],
                    "constraint_radius": [float(v) for v in radius.detach().cpu().view(-1).tolist()],
                    "noise_radius_source": radius_source,
                    "noise_mode": str(self.noise_mode),
                    "noise_level": float(self.noise_level),
                }
                return radius.to(dtype=torch.float32, device=g_observed.device)
            if method in MEASUREMENT_NORMALIZED_REGULARIZED_INIT_METHODS:
                norm_type = "l1" if method == "l1_l1_admm" else "l2"
                return self._choose_lambda_morozov_iterative(
                    g_observed,
                    init_method=method,
                    residual_norm=norm_type,
                )
            noise_norm, noise_radius_source = self._estimate_morozov_noise_norm_from_observed(g_observed, norm_type="l2")
            lam = self.time_operator.choose_lambda_morozov(
                g_observed,
                noise_norm=noise_norm,
                tau=float(DATA_CONFIG.get("morozov_tau", 1.0)),
                max_iter=int(DATA_CONFIG.get("morozov_max_iter", 8)),
                lambda_min=float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12)),
                lambda_max=float(DATA_CONFIG.get("morozov_lambda_max", 1.0e12)),
            ).to(dtype=torch.float32, device=g_observed.device)
            self.last_lambda_info = {
                "mode": "morozov_spectral",
                "method": method,
                "residual_norm": "l2",
                "target_norm": [float(v) for v in (noise_norm * float(DATA_CONFIG.get("morozov_tau", 1.0))).detach().cpu().view(-1).tolist()],
                "noise_radius_source": noise_radius_source,
                "noise_mode": str(self.noise_mode),
                "noise_level": float(self.noise_level),
            }
            return lam
        return float(DATA_CONFIG.get("lambda_reg", 1e-2))

    def _select_lambda(self, g_observed: torch.Tensor, lambda_reg: float | torch.Tensor = None) -> float | torch.Tensor:
        return self.select_lambda_for_init_method(g_observed, init_method=None, lambda_reg=lambda_reg)

    def generate_training_sample(self, random_seed=None, lambda_reg: float | torch.Tensor = None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        coeff_true = self._sample_coefficients()
        f_true = self.image_gen(coeff_true).squeeze(0)
        with torch.no_grad():
            g_clean = self.data_forward_operator(coeff_true).to(torch.float32)
            g_observed = self._apply_noise(g_clean)
            if self.init_time_operator is self.data_time_operator:
                g_init_observed = g_observed
            else:
                g_init_clean = self.init_forward_operator(coeff_true).to(torch.float32)
                g_init_observed = self._apply_noise(g_init_clean)
            init_method = normalize_init_method(str(TIME_DOMAIN_CONFIG.get("init_method", "cg")))
            with self._using_init_operator():
                lambda_eff = self._select_lambda(g_init_observed, lambda_reg=lambda_reg)
                self.last_lambda = lambda_eff
                if dict(self.last_lambda_info or {}).get("mode") == "morozov_constrained_radius":
                    coeff_initial = self.solve_constrained_init(g_init_observed, noise_radius=lambda_eff, init_method=init_method)
                    info = dict(self.last_lambda_info or {})
                    info["mode"] = "morozov_constrained"
                    info["solver_stats"] = dict(getattr(self.time_operator, "last_split_admm_stats", None) or {})
                    self.last_lambda_info = info
                else:
                    coeff_initial = self.solve_regularized_init(g_init_observed, lambda_reg=lambda_eff, init_method=init_method)
        return coeff_true.squeeze(0).squeeze(0), f_true.squeeze(0), g_observed.squeeze(0), coeff_initial.squeeze(0).squeeze(0)

    def generate_batch(self, batch_size, random_seed=None, lambda_reg: float | torch.Tensor = None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        batch_started = time.perf_counter()
        init_method = normalize_init_method(str(TIME_DOMAIN_CONFIG.get("init_method", "cg")))
        lambda_mode = "provided" if lambda_reg is not None else str(DATA_CONFIG.get("lambda_select_mode", "fixed")).strip().lower()
        morozov_form = str(DATA_CONFIG.get("morozov_form", "regularized")).strip().lower()
        progress_enabled = (not self._first_batch_progress_logged) and init_method != "cg"
        if progress_enabled:
            solver_name = "constrained_init" if lambda_mode == "morozov" and morozov_form == "constrained" else "regularized_init"
            print(
                "[init] first batch start "
                f"batch_size={int(batch_size)} data_angles={int(getattr(self.data_time_operator, 'num_angles', 1) or 1)} "
                f"init_angles={int(getattr(self.init_time_operator, 'num_angles', 1) or 1)} "
                f"lambda_mode={lambda_mode} morozov_form={morozov_form} init_method={init_method} solver={solver_name}"
            )
        coeff_true = self._sample_coefficients(batch_size)
        f_true = self.image_gen(coeff_true)
        with torch.no_grad():
            g_clean = self.data_forward_operator(coeff_true).to(torch.float32)
            g_observed = self._apply_noise(g_clean)
            if self.init_time_operator is self.data_time_operator:
                g_init_observed = g_observed
            else:
                g_init_clean = self.init_forward_operator(coeff_true).to(torch.float32)
                g_init_observed = self._apply_noise(g_init_clean)
        coeff_init_started = time.perf_counter()
        with self._using_init_operator():
            lambda_eff = self._select_lambda(g_init_observed, lambda_reg=lambda_reg)
            self.last_lambda = lambda_eff
            if dict(self.last_lambda_info or {}).get("mode") == "morozov_constrained_radius":
                coeff_initial = self.solve_constrained_init(g_init_observed, noise_radius=lambda_eff, init_method=init_method)
                info = dict(self.last_lambda_info or {})
                info["mode"] = "morozov_constrained"
                info["solver_stats"] = dict(getattr(self.time_operator, "last_split_admm_stats", None) or {})
                self.last_lambda_info = info
            else:
                coeff_initial = self.solve_regularized_init(g_init_observed, lambda_reg=lambda_eff, init_method=init_method)
        if progress_enabled:
            print(f"[init] coefficient init finished in {time.perf_counter() - coeff_init_started:.2f}s")
            print(f"[init] first batch ready in {time.perf_counter() - batch_started:.2f}s")
            self._first_batch_progress_logged = True
        return coeff_true, f_true, g_observed, coeff_initial
