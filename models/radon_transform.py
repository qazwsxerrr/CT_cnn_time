"""Time-domain single-angle operator and data generator for CT inverse problems.

This module implements the ill-posed measurement model in Example 1.1 using the
generator phi(x, y) = B1(x) * B1(y) on the active coefficient grid.

Observation system (fixed across samples):
- Choose sampling points X via paper_grid_t0:
    X_j=(t0+j)/||beta||_2, j=0..M-1 (paper Sec. 5.1.1, t0=0.5)
- Build A in Theorem-3.5 form:
    A = L P,
  where L is lower-triangular Toeplitz (Eq. (3.11)), and P maps
  lexicographical coefficients c to d indexed by beta·k.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import DATA_CONFIG, IMAGE_SIZE, THEORETICAL_CONFIG, TIME_DOMAIN_CONFIG, device
from image_generator import (
    DifferentiableImageGenerator,
    generate_random_ellipse_phantom,
    generate_shepp_logan_phantom,
)
from b_spline.b2b1_spline import (
    phi_support_bounds_b1b1,
    radon_phi_b1b1,
    radon_phi_b2b1,
)


def _morozov_settings(
    max_iter: int,
    lambda_min: float,
    lambda_max: float,
) -> dict[str, float]:
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
    # Use a monotone bracketed search in log-space. The residual curve can have
    # a flat near-zero plateau for tiny lambda, where Newton updates become
    # numerically meaningless even though a valid bracket already exists.
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


def _choose_lambda_morozov_from_explicit_svd(
    b: torch.Tensor,
    noise_norm: torch.Tensor,
    U: torch.Tensor,
    s: torch.Tensor,
    tau: float,
    settings: dict[str, float],
) -> torch.Tensor:
    if b.dim() == 1:
        b = b.unsqueeze(0)
    batch = int(b.shape[0])
    noise_norm = noise_norm.view(-1)
    if int(noise_norm.numel()) == 1 and batch > 1:
        noise_norm = noise_norm.expand(batch)

    U_cpu = U.detach().to(dtype=torch.float64, device="cpu")
    s2 = s.detach().to(dtype=torch.float64, device="cpu").square().numpy()
    b_cpu = b.detach().to(dtype=torch.float64, device="cpu")
    noise_cpu = noise_norm.detach().to(dtype=torch.float64, device="cpu")

    lam_list = []
    for idx in range(batch):
        b_i = b_cpu[idx]
        c = torch.mv(U_cpu.t(), b_i)
        c2 = c.square().numpy()
        b_norm2 = float(torch.dot(b_i, b_i).item())
        c_norm2 = float(torch.sum(c.square()).item())
        perp_norm2 = max(0.0, b_norm2 - c_norm2)
        target2 = float(float(tau) * float(noise_cpu[idx].item())) ** 2

        def residual2_fn(lam: float) -> float:
            w = lam / (s2 + lam)
            return float(np.sum((w * w) * c2) + perp_norm2)

        def derivative_fn(lam: float) -> float:
            return float(np.sum((2.0 * lam * s2 / ((s2 + lam) ** 3)) * c2))

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
                min_residual2=perp_norm2,
                max_residual2=b_norm2,
            )
        )

    return torch.tensor(lam_list, dtype=b.dtype, device=b.device)


@torch.no_grad()
def _build_implicit_normal_matrix(
    operator,
    chunk_size: int = 64,
) -> torch.Tensor:
    n = int(operator.N)
    op_device = getattr(getattr(operator, "r_vectors", None), "device", None)
    if op_device is None:
        op_device = getattr(getattr(operator, "alphas", None), "device", device)
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


def _ensure_implicit_gram_spectrum(
    operator,
    fingerprint: dict[str, object],
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(operator, "_morozov_gram_eigvals", None) is not None and getattr(operator, "_morozov_gram_eigvecs", None) is not None:
        return operator._morozov_gram_eigvals, operator._morozov_gram_eigvecs

    cache_dir = str(DATA_CONFIG.get("morozov_cache_dir", "")).strip()
    if not cache_dir:
        raise ValueError("DATA_CONFIG['morozov_cache_dir'] must be a non-empty path.")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _morozov_cache_path(cache_dir, fingerprint)
    setattr(operator, "_morozov_cache_path", cache_path)

    cache_hit = False
    build_seconds = None
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("fingerprint") == fingerprint:
            eigvals = cached["eigvals"].to(dtype=torch.float32, device="cpu")
            eigvecs = cached["eigvecs"].to(dtype=torch.float32, device="cpu")
            cache_hit = True
        else:
            cached = None
    else:
        cached = None

    if cached is None:
        started = time.perf_counter()
        print(f"[Morozov] building exact Gram spectrum cache: {cache_path}")
        gram = _build_implicit_normal_matrix(operator, chunk_size=chunk_size)
        eigvals, eigvecs = torch.linalg.eigh(gram)
        eigvals = eigvals.clamp_min_(0.0).to(dtype=torch.float32, device="cpu")
        eigvecs = eigvecs.to(dtype=torch.float32, device="cpu")
        torch.save(
            {
                "fingerprint": fingerprint,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
            },
            cache_path,
        )
        build_seconds = float(time.perf_counter() - started)
        print(f"[Morozov] cached exact Gram spectrum in {build_seconds:.2f}s")
    else:
        print(f"[Morozov] loaded exact Gram spectrum cache: {cache_path}")

    operator._morozov_gram_eigvals = eigvals
    operator._morozov_gram_eigvecs = eigvecs
    operator.last_morozov_cache_hit = bool(cache_hit)
    operator.last_morozov_cache_build_seconds = build_seconds
    return eigvals, eigvecs


def _ensure_runtime_gram_spectrum(
    operator,
    fingerprint: dict[str, object],
    target_device: torch.device | str,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_device = torch.device(target_device)
    eigvals_cpu, eigvecs_cpu = _ensure_implicit_gram_spectrum(
        operator,
        fingerprint,
        chunk_size=chunk_size,
    )
    if target_device.type != "cuda":
        return (
            eigvals_cpu.to(dtype=torch.float32, device=target_device),
            eigvecs_cpu.to(dtype=torch.float32, device=target_device),
        )

    runtime_device = getattr(operator, "_morozov_gram_runtime_device", None)
    cached_vals = getattr(operator, "_morozov_gram_eigvals_gpu", None)
    cached_vecs = getattr(operator, "_morozov_gram_eigvecs_gpu", None)
    if (
        runtime_device == str(target_device)
        and cached_vals is not None
        and cached_vecs is not None
    ):
        return cached_vals, cached_vecs

    try:
        eigvals_gpu = eigvals_cpu.to(dtype=torch.float32, device=target_device, non_blocking=True)
        eigvecs_gpu = eigvecs_cpu.to(dtype=torch.float32, device=target_device, non_blocking=True)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message:
            raise
        if not bool(getattr(operator, "_morozov_gpu_fallback_warned", False)):
            print(
                "[Morozov] GPU runtime Gram cache allocation failed; "
                "falling back to CPU spectrum for this run."
            )
            operator._morozov_gpu_fallback_warned = True
        operator._morozov_gram_eigvals_gpu = None
        operator._morozov_gram_eigvecs_gpu = None
        operator._morozov_gram_runtime_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return eigvals_cpu, eigvecs_cpu

    operator._morozov_gram_eigvals_gpu = eigvals_gpu
    operator._morozov_gram_eigvecs_gpu = eigvecs_gpu
    operator._morozov_gram_runtime_device = str(target_device)
    return eigvals_gpu, eigvecs_gpu


def _solve_tikhonov_from_gram_spectrum(
    rhs: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    lambda_reg: float | torch.Tensor,
    *,
    rhs_proj: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Keep the spectral Tikhonov solve on the same deterministic CPU float32
    # path as the legacy 8-angle implementation.  The Gram problem is very
    # ill-conditioned; moving this projection to GPU (or silently promoting the
    # CPU path to float64) changes the learned optimizer's first-step gradients
    # enough to alter the training trajectory, even when the underlying
    # operator is mathematically equivalent.
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
            raise ValueError(
                f"lambda_reg has {int(lam_cpu.numel())} entries, expected 1 or batch={batch}."
            )
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
    # Match the legacy CPU/NumPy scalar Morozov path for reproducibility.  The
    # vectorized GPU path was faster, but it changed the selected lambdas and
    # the resulting initialization enough to perturb the learned optimizer.
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


def _normalize_backbone_beta_vectors(beta_vectors) -> list[tuple[int, int]]:
    if beta_vectors is None or len(list(beta_vectors)) == 0:
        raise ValueError("use_multi_angle=True but TIME_DOMAIN_CONFIG['beta_vectors'] is empty.")
    normalized = []
    for beta in list(beta_vectors):
        beta_i = _to_integer_beta(beta)
        normalized.append((int(beta_i[0].item()), int(beta_i[1].item())))
    return normalized


def _effective_angle_t0(alpha: torch.Tensor, beta_norm: float, base_t0: float, auto_shift: bool) -> float:
    if not bool(auto_shift):
        return float(base_t0)
    support_lo, _ = phi_support_bounds_b1b1(alpha)
    return float(support_lo * float(beta_norm) + float(base_t0))


def _beta_support_bounds_b1b1(beta: torch.Tensor) -> tuple[float, float]:
    """Return the exact beta-domain support [A_beta, B_beta] for phi=B1(x)B1(y)."""
    beta = beta.to(torch.int64).view(-1)
    if int(beta.numel()) != 2:
        raise ValueError(f"beta must have shape (2,), got {tuple(beta.shape)}")
    b1 = int(beta[0].item())
    b2 = int(beta[1].item())
    vals = [0, b1, b2, b1 + b2]
    return float(min(vals)), float(max(vals))


def _formula_mode_from_solver_mode(solver_mode: str) -> str:
    profile = str(TIME_DOMAIN_CONFIG.get("experiment_profile", "")).strip().lower()
    if profile == "same8_shifted_support_triangular_pi":
        return "legacy_injective_extension"
    return "condition_constrained_offset"


def _resolve_theoretical_formula_mode(formula_mode: str | None, solver_mode: str) -> str:
    resolved = "auto" if formula_mode is None else str(formula_mode).strip().lower()
    if resolved in {"", "auto"}:
        return _formula_mode_from_solver_mode(solver_mode)
    if resolved in {
        "legacy_injective_extension",
        "condition_constrained_offset",
    }:
        return resolved
    raise ValueError(
        f"Unsupported theoretical_formula_mode={formula_mode!r}; "
        "expected 'auto', 'legacy_injective_extension', or "
        "'condition_constrained_offset'."
    )


def _kernel_support_length(r: torch.Tensor, tol: float = 1.0e-8) -> int:
    nz = torch.nonzero(torch.abs(r.detach().to(dtype=torch.float64)) > float(tol)).view(-1)
    if int(nz.numel()) == 0:
        return 1
    return int(nz[-1].item()) + 1


def _trim_lower_banded_ab(lower_ab: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    if lower_ab.ndim != 2:
        raise ValueError(f"lower_ab must be 2D, got shape {lower_ab.shape}")
    keep = np.where(np.max(np.abs(lower_ab), axis=1) > float(tol))[0]
    if keep.size == 0:
        return lower_ab[:1].copy()
    return lower_ab[: int(keep[-1]) + 1].copy()


def _lower_banded_apply(lower_bands: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if lower_bands.dim() != 2:
        raise ValueError(f"lower_bands must have shape (bw, N), got {tuple(lower_bands.shape)}")
    bw = int(lower_bands.shape[0])
    n = int(lower_bands.shape[1])
    y = torch.zeros((x.shape[0], n), dtype=x.dtype, device=x.device)
    for offset in range(bw):
        length = n - offset
        if length <= 0:
            break
        coeff = lower_bands[offset, :length].to(dtype=x.dtype, device=x.device)
        y[:, offset:] = y[:, offset:] + x[:, :length] * coeff.unsqueeze(0)
    return y


def _lower_banded_adjoint_apply(lower_bands: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if lower_bands.dim() != 2:
        raise ValueError(f"lower_bands must have shape (bw, N), got {tuple(lower_bands.shape)}")
    bw = int(lower_bands.shape[0])
    n = int(lower_bands.shape[1])
    y = torch.zeros((x.shape[0], n), dtype=x.dtype, device=x.device)
    for offset in range(bw):
        length = n - offset
        if length <= 0:
            break
        coeff = lower_bands[offset, :length].to(dtype=x.dtype, device=x.device)
        y[:, :length] = y[:, :length] + x[:, offset:] * coeff.unsqueeze(0)
    return y


def _lower_banded_apply_batched(lower_bands: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"x must have shape (B,K,N), got {tuple(x.shape)}")
    if lower_bands.dim() != 3:
        raise ValueError(f"lower_bands must have shape (K,bw,N), got {tuple(lower_bands.shape)}")
    batch, num_angles, n = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if int(lower_bands.shape[0]) != num_angles or int(lower_bands.shape[2]) != n:
        raise ValueError(
            "lower_bands/x shape mismatch: "
            f"lower_bands={tuple(lower_bands.shape)}, x={tuple(x.shape)}"
        )
    bw = int(lower_bands.shape[1])
    y = torch.zeros((batch, num_angles, n), dtype=x.dtype, device=x.device)
    lower_bands = lower_bands.to(dtype=x.dtype, device=x.device)
    for offset in range(bw):
        length = n - offset
        if length <= 0:
            break
        coeff = lower_bands[:, offset, :length].unsqueeze(0)
        y[:, :, offset:] = y[:, :, offset:] + x[:, :, :length] * coeff
    return y


def _lower_banded_adjoint_apply_batched(lower_bands: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"x must have shape (B,K,N), got {tuple(x.shape)}")
    if lower_bands.dim() != 3:
        raise ValueError(f"lower_bands must have shape (K,bw,N), got {tuple(lower_bands.shape)}")
    batch, num_angles, n = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if int(lower_bands.shape[0]) != num_angles or int(lower_bands.shape[2]) != n:
        raise ValueError(
            "lower_bands/x shape mismatch: "
            f"lower_bands={tuple(lower_bands.shape)}, x={tuple(x.shape)}"
        )
    bw = int(lower_bands.shape[1])
    y = torch.zeros((batch, num_angles, n), dtype=x.dtype, device=x.device)
    lower_bands = lower_bands.to(dtype=x.dtype, device=x.device)
    for offset in range(bw):
        length = n - offset
        if length <= 0:
            break
        coeff = lower_bands[:, offset, :length].unsqueeze(0)
        y[:, :, :length] = y[:, :, :length] + x[:, :, offset:] * coeff
    return y


def _sparse_blocks_apply_batched(
    rows: torch.Tensor,
    cols: torch.Tensor,
    values: torch.Tensor,
    nnz: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Apply per-angle sparse d-order matrices stored as padded COO buffers."""
    if x.dim() != 3:
        raise ValueError(f"x must have shape (B,K,N), got {tuple(x.shape)}")
    batch, num_angles, n = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if int(rows.shape[0]) != num_angles or int(cols.shape[0]) != num_angles or int(values.shape[0]) != num_angles:
        raise ValueError(
            "sparse block/x shape mismatch: "
            f"rows={tuple(rows.shape)}, cols={tuple(cols.shape)}, values={tuple(values.shape)}, x={tuple(x.shape)}"
        )
    y = torch.zeros((batch, num_angles, n), dtype=x.dtype, device=x.device)
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
) -> torch.Tensor:
    """Apply adjoints of per-angle sparse d-order matrices stored as padded COO buffers."""
    if x.dim() != 3:
        raise ValueError(f"x must have shape (B,K,N), got {tuple(x.shape)}")
    batch, num_angles, n = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if int(rows.shape[0]) != num_angles or int(cols.shape[0]) != num_angles or int(values.shape[0]) != num_angles:
        raise ValueError(
            "sparse block/x shape mismatch: "
            f"rows={tuple(rows.shape)}, cols={tuple(cols.shape)}, values={tuple(values.shape)}, x={tuple(x.shape)}"
        )
    y = torch.zeros((batch, num_angles, n), dtype=x.dtype, device=x.device)
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


def _to_integer_beta(beta) -> torch.Tensor:
    """Validate and convert a 2D beta vector to int64 tensor."""
    beta_t = torch.as_tensor(beta, dtype=torch.float64).view(-1)
    if int(beta_t.numel()) != 2:
        raise ValueError(f"beta must contain exactly 2 entries, got {tuple(beta_t.tolist())}")
    beta_round = torch.round(beta_t)
    if torch.max(torch.abs(beta_t - beta_round)).item() > 1e-9:
        raise ValueError(
            f"beta must be integer-valued for Theorem 3.5 construction, got {tuple(beta_t.tolist())}"
        )
    beta_i = beta_round.to(torch.int64)
    if int(torch.sum(beta_i.abs()).item()) == 0:
        raise ValueError("beta must be non-zero.")
    return beta_i


def _lex_lattice_indices(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lexicographical lattice order: k1 major, k2 minor."""
    k1 = torch.arange(int(height), dtype=torch.int64).repeat_interleave(int(width))
    k2 = torch.arange(int(width), dtype=torch.int64).repeat(int(height))
    return k1, k2


def _build_lower_toeplitz_from_r(r: torch.Tensor) -> torch.Tensor:
    """Build lower-triangular Toeplitz matrix from r_k, k=0..m."""
    n = int(r.numel())
    row = torch.arange(n, dtype=torch.int64).view(-1, 1)
    col = torch.arange(n, dtype=torch.int64).view(1, -1)
    diff = row - col
    L = torch.zeros((n, n), dtype=r.dtype)
    mask = diff >= 0
    L[mask] = r[diff[mask]]
    return L


def _build_sparse_b1b1_block_from_sorted_proj(
    *,
    sorted_proj: torch.Tensor,
    beta_norm: float,
    alpha: torch.Tensor,
    support_lo_beta: float,
    support_hi_beta: float,
    band_t0: float,
) -> dict[str, torch.Tensor]:
    """Build the full sparse d-order matrix in the paper's beta-scale.

    In code we evaluate ``R_alpha phi((tau+kappa_i-kappa_j)/||beta||)``,
    with ``alpha=beta/||beta||``.  Equivalently, using the paper's scaled
    convention ``R_beta g(t):=R_alpha g(t/||beta||)``, this is
    ``A_ij=R_beta phi(tau+kappa_i-kappa_j)``.

    Unlike the legacy lower-banded construction, this supports arbitrary tau in
    (A_beta, B_beta), so nonzero entries may lie above and below the diagonal.
    """
    proj_np = sorted_proj.detach().to(dtype=torch.int64, device="cpu").numpy()
    n = int(proj_np.shape[0])
    proj_min = int(proj_np[0])
    proj_max = int(proj_np[-1])
    lookup = np.full((proj_max - proj_min + 1,), -1, dtype=np.int64)
    lookup[proj_np - proj_min] = np.arange(n, dtype=np.int64)

    delta_lo = int(math.ceil(float(support_lo_beta) - float(band_t0) - 1.0e-12))
    delta_hi = int(math.floor(float(support_hi_beta) - float(band_t0) + 1.0e-12))

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    lower_bw = 0
    upper_bw = 0
    alpha64 = alpha.detach().to(dtype=torch.float64, device="cpu")

    for delta in range(delta_lo, delta_hi + 1):
        kernel_value = float(
            radon_phi_b1b1(
                torch.tensor([(float(band_t0) + float(delta)) / float(beta_norm)], dtype=torch.float64),
                alpha64,
            )[0].item()
        )
        if abs(kernel_value) <= 1.0e-15:
            continue
        targets = proj_np - int(delta)
        mask = (targets >= proj_min) & (targets <= proj_max)
        if not np.any(mask):
            continue
        rows = np.nonzero(mask)[0].astype(np.int64, copy=False)
        cols = lookup[targets[mask] - proj_min]
        valid = cols >= 0
        if not np.any(valid):
            continue
        rows = rows[valid]
        cols = cols[valid]
        if rows.size == 0:
            continue
        row_parts.append(rows)
        col_parts.append(cols)
        val_parts.append(np.full((rows.size,), kernel_value, dtype=np.float64))
        lower_bw = max(lower_bw, int(np.max(rows - cols, initial=0)) + 1)
        upper_bw = max(upper_bw, int(np.max(cols - rows, initial=0)) + 1)

    if row_parts:
        rows_np = np.concatenate(row_parts).astype(np.int64, copy=False)
        cols_np = np.concatenate(col_parts).astype(np.int64, copy=False)
        vals_np = np.concatenate(val_parts).astype(np.float64, copy=False)
    else:
        rows_np = np.empty((0,), dtype=np.int64)
        cols_np = np.empty((0,), dtype=np.int64)
        vals_np = np.empty((0,), dtype=np.float64)

    diag0 = float(
        radon_phi_b1b1(
            torch.tensor([float(band_t0) / float(beta_norm)], dtype=torch.float64),
            alpha64,
        )[0].item()
    )
    return {
        "sparse_rows": torch.from_numpy(rows_np),
        "sparse_cols": torch.from_numpy(cols_np),
        "sparse_values": torch.from_numpy(vals_np),
        "sparse_nnz": torch.tensor(int(vals_np.shape[0]), dtype=torch.int64),
        "lower_bandwidth": torch.tensor(int(lower_bw), dtype=torch.int64),
        "upper_bandwidth": torch.tensor(int(upper_bw), dtype=torch.int64),
        "delta_min": torch.tensor(int(delta_lo), dtype=torch.int64),
        "delta_max": torch.tensor(int(delta_hi), dtype=torch.int64),
        "diag0": torch.tensor(float(diag0), dtype=torch.float64),
    }


def _theorem35_block(
    beta,
    height: int,
    width: int,
    t0: float,
) -> dict[str, torch.Tensor]:
    """Construct one per-direction block A = L P in Theorem 3.5 format."""
    h = int(height)
    w = int(width)
    n = int(h * w)
    beta_i = _to_integer_beta(beta)
    beta_f = beta_i.to(torch.float64)
    beta_norm = float(torch.norm(beta_f, p=2).item())
    alpha = beta_f / beta_norm

    k1, k2 = _lex_lattice_indices(h, w)
    beta_dot_k = beta_i[0] * k1 + beta_i[1] * k2  # (N,), int64

    uniq_sorted = torch.sort(torch.unique(beta_dot_k)).values
    if int(uniq_sorted.numel()) != n:
        raise ValueError(
            f"beta={tuple(int(x) for x in beta_i.tolist())} does not make P_beta injective on E^+."
        )

    kappa0 = int(uniq_sorted[0].item())
    kappa_m = int(uniq_sorted[-1].item())
    m = int(kappa_m - kappa0)
    if (m + 1) != n:
        raise ValueError(
            f"range(P_beta) is not contiguous with size N={n}: got [{kappa0}, {kappa_m}] (m+1={m+1})."
        )
    expected = torch.arange(kappa0, kappa_m + 1, dtype=torch.int64)
    if not torch.equal(uniq_sorted, expected):
        raise ValueError(
            f"range(P_beta) must be consecutive integers for Theorem 3.5 (beta={tuple(int(x) for x in beta_i.tolist())})."
        )

    lex_to_d = (beta_dot_k - kappa0).to(torch.int64)  # (N,)
    d_to_lex = torch.empty(n, dtype=torch.int64)
    d_to_lex[lex_to_d] = torch.arange(n, dtype=torch.int64)

    P = torch.zeros((n, n), dtype=torch.float64)
    P[lex_to_d, torch.arange(n, dtype=torch.int64)] = 1.0

    k = torch.arange(n, dtype=torch.float64)
    sampling_points = (float(t0) + k) / beta_norm
    r = radon_phi_b2b1((float(t0) + k - float(kappa0)) / beta_norm, alpha).to(torch.float64)  # (N,)
    L = _build_lower_toeplitz_from_r(r)  # (N,N)
    A = L @ P  # (N,N)

    return {
        "A": A,
        "L": L,
        "P": P,
        "r": r,
        "alpha": alpha,
        "beta": beta_i,
        "sampling_points": sampling_points,
        "lex_to_d": lex_to_d,
        "d_to_lex": d_to_lex,
        "kappa0": torch.tensor(kappa0, dtype=torch.int64),
        "kappa_m": torch.tensor(kappa_m, dtype=torch.int64),
    }


def _theoretical_b1b1_block(
    beta,
    height: int,
    width: int,
    t0: float,
    formula_mode: str = "condition_constrained_offset",
    auto_shift_t0: bool = True,
) -> dict[str, torch.Tensor]:
    """Construct one per-direction theoretical B1*B1 block metadata for injective beta·k orderings."""
    h = int(height)
    w = int(width)
    n = int(h * w)
    beta_i = _to_integer_beta(beta)
    beta_f = beta_i.to(torch.float64)
    beta_norm = float(torch.norm(beta_f, p=2).item())
    alpha = beta_f / beta_norm

    k1, k2 = _lex_lattice_indices(h, w)
    beta_dot_k = beta_i[0] * k1 + beta_i[1] * k2

    sort_perm = torch.argsort(beta_dot_k, stable=True)
    uniq_sorted = beta_dot_k.index_select(0, sort_perm)
    if int(uniq_sorted.numel()) != n:
        raise ValueError(
            f"beta={tuple(int(x) for x in beta_i.tolist())} does not make beta·k injective on [0,{h-1}]x[0,{w-1}]."
        )
    if int(torch.unique(uniq_sorted).numel()) != n:
        raise ValueError(
            f"beta={tuple(int(x) for x in beta_i.tolist())} does not make beta·k injective on [0,{h-1}]x[0,{w-1}]."
        )

    lex_to_d = torch.empty(n, dtype=torch.int64)
    lex_to_d[sort_perm] = torch.arange(n, dtype=torch.int64)
    d_to_lex = sort_perm.to(torch.int64)

    sorted_proj = uniq_sorted.to(torch.int64)
    kappa0 = int(sorted_proj[0].item())
    support_lo_beta_exact, support_hi_beta_exact = _beta_support_bounds_b1b1(beta_i)
    support_hi_beta = float(support_hi_beta_exact)
    formula_mode = str(formula_mode).strip().lower()
    if formula_mode == "legacy_injective_extension":
        # Correct triangular injective extension in the beta-domain.  The
        # first beta-domain sampling point is
        #
        #   t_0^beta = kappa_0 + A_beta + t0,
        #   A_beta = min{0, beta_1, beta_2, beta_1 + beta_2}.
        #
        # The ordered beta-domain sampling grid is therefore
        #
        #   t_0^beta,
        #   t_0^beta + kappa_1 - kappa_0,
        #   ...,
        #   t_0^beta + kappa_{N-1} - kappa_0.
        #
        # Only when evaluating R_alpha phi do we divide the beta-domain
        # argument by ||beta||.  The stored ``sampling_points`` buffer is this
        # normalized alpha-domain coordinate for the numerical Radon kernel.
        #
        # Hence the d-order block entries are
        #
        #   A_ij = R_beta phi(A_beta + t0 + kappa_i - kappa_j).
        #
        # For i < j the integer gap is at most -1, so
        # A_beta + t0 + kappa_i - kappa_j <= A_beta - 0.5
        # when t0=0.5; this lies strictly outside the left support.  The
        # complete physical matrix is therefore lower triangular for every
        # signed injective beta, and the lower-banded representation is exact.
        support_lo_beta = float(support_lo_beta_exact)
        effective_t0 = float(support_lo_beta_exact) + float(t0)
        band_t0 = float(effective_t0)
        theory_t0_abs = float(kappa0 + effective_t0)
    elif formula_mode == "condition_constrained_offset":
        # Condition-number-constrained sampling:
        #
        #   X_beta = {(kappa_i + tau_beta) / ||beta||}_i
        #
        # The condition-optimized method always builds the complete sparse
        # matrix. No triangular truncation or lower-banded shortcut is allowed
        # for this path, so observation generation and reconstruction both use
        # the full operator.
        support_lo_beta = float(support_lo_beta_exact)
        effective_t0 = float(t0)
        band_t0 = float(t0)
        theory_t0_abs = float(t0)
    else:
        raise ValueError(
            f"Unknown B1*B1 formula_mode={formula_mode!r}; expected "
            "'legacy_injective_extension' or 'condition_constrained_offset'."
        )

    sampling_points = (float(effective_t0) + sorted_proj.to(torch.float64)) / beta_norm
    if formula_mode == "condition_constrained_offset":
        sparse_info = _build_sparse_b1b1_block_from_sorted_proj(
            sorted_proj=sorted_proj,
            beta_norm=float(beta_norm),
            alpha=alpha,
            support_lo_beta=float(support_lo_beta),
            support_hi_beta=float(support_hi_beta),
            band_t0=float(band_t0),
        )
        r = torch.zeros((n,), dtype=torch.float64)
        if int(sparse_info["lower_bandwidth"].item()) > 0:
            r[0] = sparse_info["diag0"].to(dtype=torch.float64)
        return {
            "r": r,
            "sorted_proj": sorted_proj,
            "alpha": alpha,
            "beta": beta_i,
            "sampling_points": sampling_points,
            "lex_to_d": lex_to_d,
            "d_to_lex": d_to_lex,
            "kappa0": torch.tensor(kappa0, dtype=torch.int64),
            "effective_t0": torch.tensor(float(effective_t0), dtype=torch.float64),
            "band_t0": torch.tensor(float(band_t0), dtype=torch.float64),
            "support_lo_beta": torch.tensor(float(support_lo_beta), dtype=torch.float64),
            "theory_t0_abs": torch.tensor(float(theory_t0_abs), dtype=torch.float64),
            **sparse_info,
        }

    max_gap = max(0, int(math.ceil(support_hi_beta - float(band_t0) + 1.0e-12)))
    band_limit = min(n, max_gap + 1)
    lower_ab = np.zeros((band_limit, n), dtype=np.float64)
    proj_np = sorted_proj.detach().to(dtype=torch.float64, device="cpu")
    for offset in range(band_limit):
        length = n - offset
        if length <= 0:
            break
        diffs = proj_np[offset:] - proj_np[:length]
        values = radon_phi_b1b1((float(band_t0) + diffs) / beta_norm, alpha).to(dtype=torch.float64, device="cpu")
        lower_ab[offset, :length] = values.numpy()
    lower_ab = _trim_lower_banded_ab(lower_ab)
    band_width = int(lower_ab.shape[0])
    r = np.zeros((n,), dtype=np.float64)
    r[:band_width] = lower_ab[:, 0]

    return {
        "r": torch.from_numpy(r),
        "lower_bands": torch.from_numpy(lower_ab),
        "sorted_proj": sorted_proj,
        "lower_bandwidth": torch.tensor(band_width, dtype=torch.int64),
        "alpha": alpha,
        "beta": beta_i,
        "sampling_points": sampling_points,
        "lex_to_d": lex_to_d,
        "d_to_lex": d_to_lex,
        "kappa0": torch.tensor(kappa0, dtype=torch.int64),
        "effective_t0": torch.tensor(float(effective_t0), dtype=torch.float64),
        "band_t0": torch.tensor(float(band_t0), dtype=torch.float64),
        "support_lo_beta": torch.tensor(float(support_lo_beta), dtype=torch.float64),
        "theory_t0_abs": torch.tensor(float(theory_t0_abs), dtype=torch.float64),
    }

# Provide discrete forward/adjoint Jacobian products explicitly so training does
# not rely on unsupported second-order derivatives through grid_sample.
class _ImplicitForwardProjectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff_matrix: torch.Tensor, operator):
        ctx.operator = operator
        return operator._forward_numeric(coeff_matrix)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_coeff = ctx.operator._adjoint_numeric(grad_output)
        return grad_coeff, None


class _ImplicitAdjointProjectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual: torch.Tensor, operator):
        ctx.operator = operator
        return operator._adjoint_numeric(residual)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_residual = ctx.operator._forward_numeric(grad_output)
        return grad_residual, None


class _ImplicitAdjointPerAngleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual_per_angle: torch.Tensor, operator):
        ctx.operator = operator
        return operator._adjoint_per_angle_numeric(residual_per_angle)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_residual = ctx.operator._forward_from_per_angle_coeff_numeric(grad_output)
        return grad_residual, None


class ImplicitPixelRadonOperator2D(torch.nn.Module):
    """
    Implicit multi-angle Radon operator for B1*B1 / pixel basis on a square grid.

    The forward model rotates the image to each angle and sums along one axis,
    which avoids constructing the dense matrix A explicitly.
    """

    def __init__(
        self,
        beta_vectors: list[tuple[int, int]],
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        num_detector_samples_per_angle: Optional[int] = None,
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.N = self.height * self.width
        if not beta_vectors:
            raise ValueError("beta_vectors must be a non-empty list of integer pairs.")

        beta_list = []
        for beta in beta_vectors:
            beta_i = _to_integer_beta(beta)
            beta_list.append((int(beta_i[0].item()), int(beta_i[1].item())))
        self.beta_vectors = beta_list
        self.num_angles = int(len(self.beta_vectors))

        if num_detector_samples_per_angle is None:
            num_detector_samples_per_angle = int(TIME_DOMAIN_CONFIG.get("num_detector_samples", self.width))
        self.M_per_angle = int(num_detector_samples_per_angle)
        self.M = int(self.num_angles * self.M_per_angle)

        self.pad_size = int(math.ceil(math.sqrt(float(self.height * self.height + self.width * self.width))))
        if (self.pad_size % 2) != (self.height % 2):
            self.pad_size += 1

        pad_h = self.pad_size - self.height
        pad_w = self.pad_size - self.width
        self.pad_top = pad_h // 2
        self.pad_bottom = pad_h - self.pad_top
        self.pad_left = pad_w // 2
        self.pad_right = pad_w - self.pad_left

        betas = torch.tensor(self.beta_vectors, dtype=torch.float32)
        beta_norm = torch.norm(betas, dim=1, keepdim=True).clamp_min(1e-8)
        alphas = betas / beta_norm
        thetas = torch.atan2(alphas[:, 1], alphas[:, 0])

        forward_grids = []
        inverse_grids = []
        for theta in thetas.tolist():
            forward_grids.append(self._build_rotation_grid(float(theta)))
            inverse_grids.append(self._build_rotation_grid(float(-theta)))

        detector_positions = torch.linspace(-1.0, 1.0, self.M_per_angle, dtype=torch.float32)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("thetas", thetas)
        self.register_buffer("forward_grids", torch.stack(forward_grids, dim=0))
        self.register_buffer("inverse_grids", torch.stack(inverse_grids, dim=0))
        self.register_buffer("sampling_points", detector_positions.repeat(self.num_angles))
        self.register_buffer("sampling_points_per_angle", detector_positions.repeat(self.num_angles, 1))
        self._morozov_gram_eigvals: Optional[torch.Tensor] = None
        self._morozov_gram_eigvecs: Optional[torch.Tensor] = None
        self.last_morozov_cache_hit: Optional[bool] = None
        self.last_morozov_cache_build_seconds: Optional[float] = None
        self.last_split_admm_stats: Optional[dict[str, object]] = None

    def _morozov_cache_fingerprint(self) -> dict[str, object]:
        return {
            "class_name": self.__class__.__name__,
            "height": int(self.height),
            "width": int(self.width),
            "num_angles": int(self.num_angles),
            "num_detector_samples_per_angle": int(self.M_per_angle),
            "beta_vectors": [list(beta) for beta in self.beta_vectors],
        }

    def _build_rotation_grid(self, theta: float) -> torch.Tensor:
        c = math.cos(theta)
        s = math.sin(theta)
        affine = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32).unsqueeze(0)
        return F.affine_grid(
            affine,
            size=(1, 1, self.pad_size, self.pad_size),
            align_corners=False,
        ).squeeze(0)

    def _pad_image(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return F.pad(
            coeff_matrix,
            (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom),
            mode="constant",
            value=0.0,
        )

    def _crop_image(self, padded: torch.Tensor) -> torch.Tensor:
        return padded[
            :,
            :,
            self.pad_top:self.pad_top + self.height,
            self.pad_left:self.pad_left + self.width,
        ]

    def _resize_projection(self, proj: torch.Tensor, out_len: int) -> torch.Tensor:
        if int(proj.shape[-1]) == int(out_len):
            return proj
        return F.interpolate(proj.unsqueeze(1), size=int(out_len), mode="linear", align_corners=False).squeeze(1)

    def _rotate_batch(self, padded: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            padded,
            grid.unsqueeze(0).expand(padded.shape[0], -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def split_measurements(self, g: torch.Tensor) -> torch.Tensor:
        if g.dim() == 3 and g.shape[1] == 1:
            g = g.squeeze(1)
        if g.dim() != 2:
            raise ValueError(f"Expected g with shape (B,M), got {tuple(g.shape)}")
        if int(g.shape[1]) != int(self.M):
            raise ValueError(f"Expected measurement length M={self.M}, got {g.shape[1]}")
        return g.view(g.shape[0], self.num_angles, self.M_per_angle)

    def _forward_single_angle_numeric(self, coeff_matrix: torch.Tensor, angle_idx: int) -> torch.Tensor:
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(1)
        coeff_matrix = coeff_matrix.to(dtype=torch.float32, device=self.alphas.device)
        padded = self._pad_image(coeff_matrix)
        rotated = self._rotate_batch(padded, self.forward_grids[int(angle_idx)])
        proj_full = rotated.sum(dim=2).squeeze(1)
        return self._resize_projection(proj_full, self.M_per_angle)

    def _forward_per_angle_numeric(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        proj_list = []
        for idx in range(self.num_angles):
            proj_list.append(self._forward_single_angle_numeric(coeff_matrix, idx))
        return torch.stack(proj_list, dim=1)

    def _forward_numeric(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self._forward_per_angle_numeric(coeff_matrix).reshape(coeff_matrix.shape[0], self.M)

    def _adjoint_single_angle_numeric(
        self,
        residual_angle: torch.Tensor,
        angle_idx: int,
    ) -> torch.Tensor:
        if residual_angle.dim() == 3 and residual_angle.shape[1] == 1:
            residual_angle = residual_angle.squeeze(1)
        if residual_angle.dim() != 2:
            raise ValueError(
                f"Expected residual_angle with shape (B,M_per_angle), got {tuple(residual_angle.shape)}"
            )
        residual_angle = residual_angle.to(dtype=torch.float32, device=self.alphas.device)
        with torch.enable_grad():
            x = torch.zeros(
                (residual_angle.shape[0], 1, self.height, self.width),
                device=self.alphas.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            proj = self._forward_single_angle_numeric(x, angle_idx)
            weighted_sum = torch.sum(proj * residual_angle)
            grad = torch.autograd.grad(weighted_sum, x, retain_graph=False, create_graph=False)[0]
        return grad.detach()

    def _adjoint_per_angle_numeric(self, residual_per_angle: torch.Tensor) -> torch.Tensor:
        if residual_per_angle.dim() == 4 and residual_per_angle.shape[2] == 1:
            residual_per_angle = residual_per_angle.squeeze(2)
        if residual_per_angle.dim() != 3:
            raise ValueError(
                f"Expected residual_per_angle with shape (B,K,M_per_angle), got {tuple(residual_per_angle.shape)}"
            )
        residual_per_angle = residual_per_angle.to(dtype=torch.float32, device=self.alphas.device)
        out = []
        for idx in range(self.num_angles):
            out.append(self._adjoint_single_angle_numeric(residual_per_angle[:, idx, :], idx))
        return torch.stack(out, dim=1)

    def _adjoint_numeric(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.dim() == 1:
            residual = residual.unsqueeze(0)
        residual_pa = self.split_measurements(residual)
        return self._adjoint_per_angle_numeric(residual_pa).sum(dim=1)

    def _forward_from_per_angle_coeff_numeric(self, coeff_per_angle: torch.Tensor) -> torch.Tensor:
        if coeff_per_angle.dim() == 4:
            coeff_per_angle = coeff_per_angle.unsqueeze(2)
        if coeff_per_angle.dim() != 5 or coeff_per_angle.shape[2] != 1:
            raise ValueError(
                "Expected coeff_per_angle with shape (B,K,1,H,W), "
                f"got {tuple(coeff_per_angle.shape)}"
            )
        outputs = []
        for idx in range(self.num_angles):
            outputs.append(self._forward_single_angle_numeric(coeff_per_angle[:, idx, :, :, :], idx))
        return torch.stack(outputs, dim=1)

    def forward_per_angle(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self._forward_per_angle_numeric(coeff_matrix)

    def forward(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and coeff_matrix.requires_grad:
            return _ImplicitForwardProjectFunction.apply(coeff_matrix, self)
        return self._forward_numeric(coeff_matrix)

    def adjoint_per_angle(self, residual_per_angle: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and residual_per_angle.requires_grad:
            return _ImplicitAdjointPerAngleFunction.apply(residual_per_angle, self)
        return self._adjoint_per_angle_numeric(residual_per_angle)

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and residual.requires_grad:
            return _ImplicitAdjointProjectFunction.apply(residual, self)
        return self._adjoint_numeric(residual)

    def apply_normal(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(coeff_matrix))

    @torch.no_grad()
    def solve_tikhonov_direct(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
    ) -> torch.Tensor:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.alphas.device)
        self.last_split_admm_stats = None
        rhs = self.adjoint(b).view(b.shape[0], self.N)
        eigvals, eigvecs = _ensure_implicit_gram_spectrum(self, self._morozov_cache_fingerprint())
        coeff = _solve_tikhonov_from_gram_spectrum(rhs, eigvals=eigvals, eigvecs=eigvecs, lambda_reg=lambda_reg)
        return coeff.to(device=self.alphas.device, dtype=torch.float32).view(-1, 1, self.height, self.width)

    @torch.no_grad()
    def solve_tikhonov_cg(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        max_iter: int,
        tol: float = 1e-4,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.alphas.device)
        rhs = self.adjoint(b)
        if x0 is None:
            x = torch.zeros_like(rhs)
        else:
            x = x0.to(dtype=torch.float32, device=rhs.device).clone()

        if torch.is_tensor(lambda_reg):
            lam = lambda_reg.detach().to(dtype=torch.float32, device=rhs.device).view(-1)
            if int(lam.numel()) == 1 and int(rhs.shape[0]) > 1:
                lam = lam.expand(int(rhs.shape[0]))
            elif int(lam.numel()) != int(rhs.shape[0]):
                raise ValueError(
                    f"lambda_reg has {int(lam.numel())} entries, expected 1 or batch={int(rhs.shape[0])}."
                )
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
            beta = rr_new / (rr + eps)
            p = r + beta * p
            rr = rr_new
        return x

    @torch.no_grad()
    def choose_lambda_morozov(
        self,
        b: torch.Tensor,
        noise_norm: torch.Tensor,
        tau: float = 1.0,
        max_iter: int = 8,
        lambda_min: float = 1e-12,
        lambda_max: float = 1e12,
    ) -> torch.Tensor:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.alphas.device)
        rhs = self.adjoint(b).view(b.shape[0], self.N)
        eigvals, eigvecs = _ensure_implicit_gram_spectrum(self, self._morozov_cache_fingerprint())
        settings = _morozov_settings(max_iter=max_iter, lambda_min=lambda_min, lambda_max=lambda_max)
        return _choose_lambda_morozov_from_gram_spectrum(
            b=b,
            rhs=rhs,
            noise_norm=noise_norm.to(dtype=torch.float32, device=b.device),
            eigvals=eigvals,
            eigvecs=eigvecs,
            tau=float(tau),
            settings=settings,
        )


class TheoreticalB1B1Operator2D(torch.nn.Module):
    """
    Theoretical B1*B1 operator on the pixel coefficient lattice:

        g_stack = [A_1; ...; A_K] c,   A_i = L_i P_i

    where each per-angle block is applied implicitly through Toeplitz convolution
    in the d-order induced by beta·k.
    """

    def __init__(
        self,
        beta_vectors: list[tuple[int, int]],
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        t0: float = 0.5,
        formula_mode: str = "condition_constrained_offset",
        auto_shift_t0: bool = True,
        t0_per_angle: Optional[list[float]] = None,
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.N = int(self.height * self.width)
        if not beta_vectors:
            raise ValueError("beta_vectors must be a non-empty list of integer pairs.")
        beta_list = []
        for beta in beta_vectors:
            beta_i = _to_integer_beta(beta)
            beta_list.append((int(beta_i[0].item()), int(beta_i[1].item())))
        self.beta_vectors = beta_list
        self.num_angles = int(len(self.beta_vectors))
        self.M_per_angle = int(self.N)
        self.M = int(self.num_angles * self.M_per_angle)
        self.t0 = float(t0)
        self.formula_mode = str(formula_mode).strip().lower()
        self.auto_shift_t0 = bool(auto_shift_t0)
        if t0_per_angle is None:
            self.t0_per_angle = None
        else:
            self.t0_per_angle = [float(v) for v in list(t0_per_angle)]
            if len(self.t0_per_angle) != int(self.num_angles):
                raise ValueError(
                    "t0_per_angle length must match beta_vectors length; "
                    f"got {len(self.t0_per_angle)} offsets for {int(self.num_angles)} angles."
                )

        with torch.no_grad():
            blocks = [
                _theoretical_b1b1_block(
                    beta=beta,
                    height=self.height,
                    width=self.width,
                    t0=(self.t0 if self.t0_per_angle is None else self.t0_per_angle[idx]),
                    formula_mode=self.formula_mode,
                    auto_shift_t0=self.auto_shift_t0,
                )
                for idx, beta in enumerate(self.beta_vectors)
            ]
            uses_sparse_blocks = any("sparse_rows" in blk for blk in blocks)
            if uses_sparse_blocks and not all("sparse_rows" in blk for blk in blocks):
                raise ValueError("Internal error: sparse and lower-banded theoretical blocks cannot be mixed.")
            r_vectors = torch.stack([blk["r"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            sorted_proj = torch.zeros((self.num_angles, self.N), dtype=torch.int64, device=device)
            if uses_sparse_blocks:
                lower_bandwidths = torch.stack([blk["lower_bandwidth"] for blk in blocks], dim=0).to(
                    dtype=torch.int64, device=device
                )
                upper_bandwidths = torch.stack([blk["upper_bandwidth"] for blk in blocks], dim=0).to(
                    dtype=torch.int64, device=device
                )
                sparse_nnz = torch.stack([blk["sparse_nnz"] for blk in blocks], dim=0).to(
                    dtype=torch.int64, device=device
                )
                max_nnz = int(sparse_nnz.max().item()) if int(sparse_nnz.numel()) > 0 else 0
                sparse_rows = torch.zeros((self.num_angles, max_nnz), dtype=torch.int64, device=device)
                sparse_cols = torch.zeros((self.num_angles, max_nnz), dtype=torch.int64, device=device)
                sparse_values = torch.zeros((self.num_angles, max_nnz), dtype=torch.float32, device=device)
                for idx, blk in enumerate(blocks):
                    count = int(blk["sparse_nnz"].item())
                    if count > 0:
                        sparse_rows[idx, :count] = blk["sparse_rows"].to(dtype=torch.int64, device=device)
                        sparse_cols[idx, :count] = blk["sparse_cols"].to(dtype=torch.int64, device=device)
                        sparse_values[idx, :count] = blk["sparse_values"].to(dtype=torch.float32, device=device)
                    sorted_proj[idx] = blk["sorted_proj"].to(dtype=torch.int64, device=device)
                lower_bands = torch.zeros((self.num_angles, 0, self.N), dtype=torch.float32, device=device)
            else:
                lower_bandwidths = torch.stack([blk["lower_bandwidth"] for blk in blocks], dim=0).to(
                    dtype=torch.int64, device=device
                )
                upper_bandwidths = torch.ones_like(lower_bandwidths)
                max_bandwidth = int(lower_bandwidths.max().item())
                lower_bands = torch.zeros(
                    (self.num_angles, max_bandwidth, self.N),
                    dtype=torch.float32,
                    device=device,
                )
                sparse_nnz = torch.zeros((self.num_angles,), dtype=torch.int64, device=device)
                sparse_rows = torch.zeros((self.num_angles, 0), dtype=torch.int64, device=device)
                sparse_cols = torch.zeros((self.num_angles, 0), dtype=torch.int64, device=device)
                sparse_values = torch.zeros((self.num_angles, 0), dtype=torch.float32, device=device)
                for idx, blk in enumerate(blocks):
                    bw = int(blk["lower_bandwidth"].item())
                    lower_bands[idx, :bw, :] = blk["lower_bands"].to(dtype=torch.float32, device=device)
                    sorted_proj[idx] = blk["sorted_proj"].to(dtype=torch.int64, device=device)
            alphas = torch.stack([blk["alpha"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            betas = torch.stack([blk["beta"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            lex_to_d = torch.stack([blk["lex_to_d"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            d_to_lex = torch.stack([blk["d_to_lex"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            kappa0 = torch.stack([blk["kappa0"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            effective_t0 = torch.stack([blk["effective_t0"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            band_t0 = torch.stack([blk["band_t0"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            support_lo_beta = torch.stack([blk["support_lo_beta"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            theory_t0_abs = torch.stack([blk["theory_t0_abs"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            sampling_points_pa = torch.stack([blk["sampling_points"] for blk in blocks], dim=0).to(
                dtype=torch.float32, device=device
            )

        self.uses_sparse_blocks = bool(uses_sparse_blocks)
        self.register_buffer("r_vectors", r_vectors)
        self.register_buffer("lower_bands", lower_bands)
        self.register_buffer("lower_bandwidths", lower_bandwidths)
        self.register_buffer("upper_bandwidths", upper_bandwidths)
        self.register_buffer("sparse_rows", sparse_rows)
        self.register_buffer("sparse_cols", sparse_cols)
        self.register_buffer("sparse_values", sparse_values)
        self.register_buffer("sparse_nnz", sparse_nnz)
        self.register_buffer("sorted_proj_per_angle", sorted_proj)
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        self.register_buffer("lex_to_d_indices", lex_to_d)
        self.register_buffer("d_to_lex_indices", d_to_lex)
        self.register_buffer("kappa0_per_angle", kappa0)
        self.register_buffer("effective_t0_per_angle", effective_t0)
        self.register_buffer("band_t0_per_angle", band_t0)
        self.register_buffer("support_lo_beta_per_angle", support_lo_beta)
        self.register_buffer("theory_t0_abs_per_angle", theory_t0_abs)
        self.register_buffer("sampling_points_per_angle", sampling_points_pa)
        self.register_buffer("sampling_points", sampling_points_pa.reshape(-1))
        self._morozov_gram_eigvals: Optional[torch.Tensor] = None
        self._morozov_gram_eigvecs: Optional[torch.Tensor] = None
        self._morozov_gram_eigvals_gpu: Optional[torch.Tensor] = None
        self._morozov_gram_eigvecs_gpu: Optional[torch.Tensor] = None
        self._morozov_gram_runtime_device: Optional[str] = None
        self._morozov_gpu_fallback_warned = False
        self.last_morozov_cache_hit: Optional[bool] = None
        self.last_morozov_cache_build_seconds: Optional[float] = None
        self.last_split_admm_stats: Optional[dict[str, object]] = None
        self._last_gram_context_signature: Optional[tuple[object, ...]] = None
        self._last_gram_context: Optional[dict[str, torch.Tensor]] = None

    def _morozov_cache_fingerprint(self) -> dict[str, object]:
        fingerprint = {
            "class_name": self.__class__.__name__,
            "height": int(self.height),
            "width": int(self.width),
            "num_angles": int(self.num_angles),
            "sampling_t0": float(self.t0),
            "t0_per_angle": None if self.t0_per_angle is None else [float(v) for v in self.t0_per_angle],
            "formula_mode": str(self.formula_mode),
            "auto_shift_t0": bool(self.auto_shift_t0),
            "effective_t0_per_angle": [float(v.item()) for v in self.effective_t0_per_angle],
            "beta_vectors": [list(beta) for beta in self.beta_vectors],
            "basis": "b1b1",
        }
        if str(self.formula_mode) == "legacy_injective_extension":
            fingerprint["band_t0_per_angle"] = [float(v.item()) for v in self.band_t0_per_angle]
            fingerprint["implementation_version"] = "legacy_injective_exact_triangular_v1"
        if str(self.formula_mode) == "condition_constrained_offset":
            fingerprint["band_t0_per_angle"] = [float(v.item()) for v in self.band_t0_per_angle]
            fingerprint["sparse_nnz_per_angle"] = [int(v.item()) for v in self.sparse_nnz]
            fingerprint["implementation_version"] = "condition_tau_full_sparse_v3"
        return fingerprint

    def _get_lower_bands(self, angle_idx: int) -> torch.Tensor:
        if bool(self.uses_sparse_blocks):
            raise ValueError("Lower-banded access is unavailable for sparse theoretical blocks.")
        angle_idx = int(angle_idx)
        bw = int(self.lower_bandwidths[angle_idx].item())
        return self.lower_bands[angle_idx, :bw, :]

    def _lower_apply(self, angle_idx: int, x: torch.Tensor) -> torch.Tensor:
        return _lower_banded_apply(self._get_lower_bands(angle_idx), x)

    def _lower_adjoint_apply(self, angle_idx: int, x: torch.Tensor) -> torch.Tensor:
        return _lower_banded_adjoint_apply(self._get_lower_bands(angle_idx), x)

    def _permute_c_to_d(self, coeff_flat: torch.Tensor, angle_idx: int) -> torch.Tensor:
        return coeff_flat.index_select(1, self.d_to_lex_indices[int(angle_idx)])

    def _permute_d_to_c(self, d_vector: torch.Tensor, angle_idx: int) -> torch.Tensor:
        return d_vector.gather(1, self.lex_to_d_indices[int(angle_idx)].view(1, -1).expand(d_vector.shape[0], -1))

    def _gram_context_signature(self, b: torch.Tensor) -> tuple[object, ...]:
        return (
            int(b.data_ptr()),
            tuple(int(v) for v in b.shape),
            str(b.device),
            str(b.dtype),
            int(getattr(b, "_version", 0)),
        )

    @torch.no_grad()
    def _prepare_gram_context(self, b: torch.Tensor) -> dict[str, torch.Tensor]:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.lower_bands.device)
        signature = self._gram_context_signature(b)
        if self._last_gram_context_signature == signature and self._last_gram_context is not None:
            return self._last_gram_context

        rhs = self.adjoint(b).view(b.shape[0], self.N)
        eigvals, eigvecs = _ensure_implicit_gram_spectrum(
            self,
            self._morozov_cache_fingerprint(),
        )
        rhs_cpu = rhs.detach().to(dtype=torch.float32, device="cpu")
        eigvecs_cpu = eigvecs.detach().to(dtype=torch.float32, device="cpu")
        rhs_proj = rhs_cpu @ eigvecs_cpu
        b_norm2 = torch.sum(b.detach().to(dtype=torch.float32, device="cpu").square(), dim=1)
        context = {
            "b": b,
            "rhs": rhs,
            "rhs_proj": rhs_proj,
            "b_norm2": b_norm2,
            "eigvals": eigvals,
            "eigvecs": eigvecs,
        }
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
        coeff_matrix = coeff_matrix.to(dtype=torch.float32, device=self.lower_bands.device)
        batch = int(coeff_matrix.shape[0])
        coeff_flat = coeff_matrix.view(batch, self.N)
        gather_index = self.d_to_lex_indices.view(1, self.num_angles, self.N).expand(batch, -1, -1)
        d_all = coeff_flat.unsqueeze(1).expand(-1, self.num_angles, -1).gather(2, gather_index)
        if bool(self.uses_sparse_blocks):
            return _sparse_blocks_apply_batched(
                self.sparse_rows,
                self.sparse_cols,
                self.sparse_values,
                self.sparse_nnz,
                d_all,
            )
        return _lower_banded_apply_batched(self.lower_bands, d_all)

    def forward(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.forward_per_angle(coeff_matrix).reshape(coeff_matrix.shape[0], self.M)

    def adjoint_per_angle(self, residual_per_angle: torch.Tensor) -> torch.Tensor:
        if residual_per_angle.dim() == 4 and residual_per_angle.shape[2] == 1:
            residual_per_angle = residual_per_angle.squeeze(2)
        if residual_per_angle.dim() != 3:
            raise ValueError(
                f"Expected residual_per_angle with shape (B,K,M_per_angle), got {tuple(residual_per_angle.shape)}"
            )
        residual_per_angle = residual_per_angle.to(dtype=torch.float32, device=self.lower_bands.device)
        batch = int(residual_per_angle.shape[0])
        if bool(self.uses_sparse_blocks):
            grad_d_all = _sparse_blocks_adjoint_apply_batched(
                self.sparse_rows,
                self.sparse_cols,
                self.sparse_values,
                self.sparse_nnz,
                residual_per_angle,
            )
        else:
            grad_d_all = _lower_banded_adjoint_apply_batched(self.lower_bands, residual_per_angle)
        gather_index = self.lex_to_d_indices.view(1, self.num_angles, self.N).expand(batch, -1, -1)
        grad_c_all = grad_d_all.gather(2, gather_index)
        return grad_c_all.view(batch, self.num_angles, 1, self.height, self.width)

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        residual_pa = self.split_measurements(residual)
        return self.adjoint_per_angle(residual_pa).sum(dim=1)

    def apply_normal(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(coeff_matrix))

    @torch.no_grad()
    def _solve_gram_tikhonov_direct(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
    ) -> torch.Tensor:
        context = self._prepare_gram_context(b)
        coeff = _solve_tikhonov_from_gram_spectrum(
            context["rhs"],
            eigvals=context["eigvals"],
            eigvecs=context["eigvecs"],
            lambda_reg=lambda_reg,
            rhs_proj=context["rhs_proj"],
        )
        return coeff.to(device=self.lower_bands.device, dtype=torch.float32).view(-1, 1, self.height, self.width)

    @torch.no_grad()
    def solve_tikhonov_direct(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        *,
        rho: Optional[float] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> torch.Tensor:
        solver_mode = str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "stacked_tikhonov")).strip().lower()
        if solver_mode != "stacked_tikhonov":
            raise ValueError(
                "Unsupported multi_angle_solver_mode="
                f"{solver_mode!r}; only 'stacked_tikhonov' is retained."
            )
        self.last_split_admm_stats = None
        return self._solve_gram_tikhonov_direct(b, lambda_reg=lambda_reg)

    @torch.no_grad()
    def solve_tikhonov_cg(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        max_iter: int,
        tol: float = 1e-4,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.lower_bands.device)
        rhs = self.adjoint(b)
        if x0 is None:
            x = torch.zeros_like(rhs)
        else:
            x = x0.to(dtype=torch.float32, device=rhs.device).clone()

        if torch.is_tensor(lambda_reg):
            lam = lambda_reg.detach().to(dtype=torch.float32, device=rhs.device).view(-1)
            if int(lam.numel()) == 1 and int(rhs.shape[0]) > 1:
                lam = lam.expand(int(rhs.shape[0]))
            elif int(lam.numel()) != int(rhs.shape[0]):
                raise ValueError(
                    f"lambda_reg has {int(lam.numel())} entries, expected 1 or batch={int(rhs.shape[0])}."
                )
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
            beta = rr_new / (rr + eps)
            p = r + beta * p
            rr = rr_new
        return x

    @torch.no_grad()
    def choose_lambda_morozov(
        self,
        b: torch.Tensor,
        noise_norm: torch.Tensor,
        tau: float = 1.0,
        max_iter: int = 8,
        lambda_min: float = 1e-12,
        lambda_max: float = 1e12,
    ) -> torch.Tensor:
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


def _condition_tau_offsets_for_formula(formula_mode: str) -> Optional[list[float]]:
    if str(formula_mode).strip().lower() != "condition_constrained_offset":
        return None
    offsets = TIME_DOMAIN_CONFIG.get("condition_constrained_tau_offsets", None)
    if offsets is None:
        return None
    return [float(v) for v in list(offsets)]


def build_time_domain_operator(
    beta=THEORETICAL_CONFIG["beta_vector"],
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
    *,
    formula_mode_override: Optional[str] = None,
    auto_shift_t0_override: Optional[bool] = None,
) -> torch.nn.Module:
    """Factory for the retained two sampling methods."""
    operator_mode = str(TIME_DOMAIN_CONFIG.get("operator_mode", "")).strip().lower()
    use_multi = TIME_DOMAIN_CONFIG.get("use_multi_angle", False)
    beta_vectors = TIME_DOMAIN_CONFIG.get("beta_vectors", None)
    if operator_mode == "theoretical_b1b1":
        if use_multi:
            backbone_betas = _normalize_backbone_beta_vectors(beta_vectors)
        else:
            backbone_betas = [(1, int(width))]
        total_angles = int(TIME_DOMAIN_CONFIG.get("num_angles_total", TIME_DOMAIN_CONFIG.get("num_angles", len(backbone_betas))))
        t0 = float(TIME_DOMAIN_CONFIG.get("sampling_t0", 0.5))
        solver_mode = str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "stacked_tikhonov")).strip().lower()
        formula_mode_source = (
            TIME_DOMAIN_CONFIG.get("theoretical_formula_mode", "condition_constrained_offset")
            if formula_mode_override is None
            else formula_mode_override
        )
        formula_mode = _resolve_theoretical_formula_mode(formula_mode_source, solver_mode)
        t0_per_angle = _condition_tau_offsets_for_formula(formula_mode)
        auto_shift_t0 = (
            bool(TIME_DOMAIN_CONFIG.get("auto_angle_t0", True))
            if auto_shift_t0_override is None
            else bool(auto_shift_t0_override)
        )
        layout = str(TIME_DOMAIN_CONFIG.get("multi_angle_layout", "full_triangular")).strip().lower()
        if layout != "full_triangular":
            raise ValueError(
                f"Unsupported multi_angle_layout={layout!r}; only 'full_triangular' is retained."
            )
        if int(total_angles) != int(len(backbone_betas)):
            raise ValueError(
                "full_triangular layout requires num_angles_total == len(beta_vectors); "
                f"got total_angles={int(total_angles)} and len(beta_vectors)={len(backbone_betas)}."
            )
        return TheoreticalB1B1Operator2D(
            beta_vectors=backbone_betas,
            height=int(height),
            width=int(width),
            t0=t0,
            formula_mode=formula_mode,
            auto_shift_t0=auto_shift_t0,
            t0_per_angle=t0_per_angle,
        ).to(device)
    if operator_mode == "implicit_b1b1":
        if use_multi and beta_vectors is not None and len(list(beta_vectors)) > 0:
            betas = [tuple(v) for v in list(beta_vectors)]
        elif use_multi:
            raise ValueError("use_multi_angle=True but TIME_DOMAIN_CONFIG['beta_vectors'] is empty.")
        else:
            betas = [tuple(_to_integer_beta(beta).tolist())]
        n_per_angle = int(TIME_DOMAIN_CONFIG.get("num_detector_samples", int(width)))
        return ImplicitPixelRadonOperator2D(
            beta_vectors=betas,
            height=int(height),
            width=int(width),
            num_detector_samples_per_angle=n_per_angle,
        ).to(device)

    if use_multi and beta_vectors is not None and len(list(beta_vectors)) > 0:
        n_per_angle = int(TIME_DOMAIN_CONFIG.get("num_detector_samples", int(height) * int(width)))
        return MultiAngleRadonOperator2D(
            beta_vectors=[tuple(v) for v in list(beta_vectors)],
            height=int(height),
            width=int(width),
            sampling_seed=int(TIME_DOMAIN_CONFIG.get("sampling_seed", 123)),
            num_detector_samples_per_angle=n_per_angle,
        ).to(device)
    if use_multi:
        raise ValueError("use_multi_angle=True but TIME_DOMAIN_CONFIG['beta_vectors'] is empty.")

    n_samples = int(TIME_DOMAIN_CONFIG.get("num_detector_samples", int(height) * int(width)))
    return RadonExample11Operator2D(
        beta=beta,
        height=int(height),
        width=int(width),
        sampling_seed=int(TIME_DOMAIN_CONFIG.get("sampling_seed", 123)),
        num_detector_samples=n_samples,
    ).to(device)


_COMPLETE_SPARSE_DATA_FORMULA_MODES = {
    "condition_constrained_offset",
}

_COMPLETE_BANDED_DATA_FORMULA_MODES = {
    "legacy_injective_extension",
}


def _complete_data_formula_for_reconstruction(reconstruction_formula_mode: str) -> str:
    recon = str(reconstruction_formula_mode).strip().lower()
    if recon == "legacy_injective_extension":
        return "legacy_injective_extension"
    if recon == "condition_constrained_offset":
        return "condition_constrained_offset"
    raise ValueError(
        f"Cannot infer a complete data formula for reconstruction formula {recon!r}. "
        "Set data_formula_mode to one of "
        f"{sorted(_COMPLETE_SPARSE_DATA_FORMULA_MODES | _COMPLETE_BANDED_DATA_FORMULA_MODES)!r}."
    )


def _resolve_data_formula_mode(reconstruction_formula_mode: str) -> str:
    raw = str(TIME_DOMAIN_CONFIG.get("data_formula_mode", "auto_complete") or "").strip().lower()
    if raw in {"", "auto", "auto_complete", "complete", "full"}:
        return _complete_data_formula_for_reconstruction(reconstruction_formula_mode)
    if raw in {
        "same",
        "same_as_reconstruction",
        "reconstruction",
        "legacy",
    }:
        raise ValueError(
            "data_formula_mode must select a complete data operator for generating "
            "R_beta f(t_i). Self-consistent incomplete data generation is forbidden; "
            f"got {raw!r}. Expected one of "
            f"{sorted(_COMPLETE_SPARSE_DATA_FORMULA_MODES | _COMPLETE_BANDED_DATA_FORMULA_MODES)!r} "
            "or 'auto_complete'."
        )
    resolved = _resolve_theoretical_formula_mode(
        raw,
        str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "stacked_tikhonov")).strip().lower(),
    )
    if resolved not in (_COMPLETE_SPARSE_DATA_FORMULA_MODES | _COMPLETE_BANDED_DATA_FORMULA_MODES):
        raise ValueError(
            "data_formula_mode must select a complete data operator for generating "
            "R_beta f(t_i). Self-consistent incomplete data generation is forbidden; "
            f"got {raw!r}. Expected one of "
            f"{sorted(_COMPLETE_SPARSE_DATA_FORMULA_MODES | _COMPLETE_BANDED_DATA_FORMULA_MODES)!r} "
            "or 'auto_complete'."
        )
    return resolved


class TheoreticalDataGenerator:
    """
    Time-domain data generator for the spline model.

    Ground truth is a coefficient field c on the integer lattice, and the measurement
    is g = A(c) (real).

    Init (feeds the learned iterative optimizer):
    - "cg": solve (A^T A + lambda I)c0 = A^T g approximately with a few CG iterations
    - "tikhonov_direct": solve (A^T A + lambda I)c0 = A^T g directly
    """

    def __init__(self, data_source: Optional[str] = None, time_operator: Optional[torch.nn.Module] = None):
        self.img_size = IMAGE_SIZE
        self.N = self.img_size * self.img_size
        if data_source is None:
            data_source = DATA_CONFIG.get("data_source", "random_ellipses")
        self.data_source = str(data_source).strip().lower()
        if self.data_source not in {"random_ellipses", "random_ellipse", "ellipse", "shepp_logan"}:
            raise ValueError(
                f"Unsupported data_source={self.data_source!r}; expected 'random_ellipses' or 'shepp_logan'."
            )

        # Noise configuration
        self.noise_mode = str(DATA_CONFIG.get("noise_mode", "additive")).strip().lower()
        if self.noise_mode not in {"additive", "multiplicative", "snr"}:
            raise ValueError(
                f"Unsupported noise_mode={self.noise_mode!r}; expected 'additive', 'multiplicative', or 'snr'."
            )
        self.noise_level = float(DATA_CONFIG.get("noise_level", 0.05))
        self.target_snr_db = float(DATA_CONFIG.get("target_snr_db", 30.0))
        self._phantom_cache: Optional[torch.Tensor] = None

        # In the current B1*B1 pipeline the image generator is identity on the 128x128 grid.
        self.image_gen = DifferentiableImageGenerator(image_size=self.img_size).to(device)

        if time_operator is None:
            self.time_operator = build_time_domain_operator(
                beta=THEORETICAL_CONFIG["beta_vector"],
                height=self.img_size,
                width=self.img_size,
            )
        else:
            self.time_operator = time_operator.to(device)
        reconstruction_formula_mode = str(
            getattr(
                self.time_operator,
                "formula_mode",
                _resolve_theoretical_formula_mode(
                    TIME_DOMAIN_CONFIG.get("theoretical_formula_mode", "auto"),
                    str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "stacked_tikhonov")).strip().lower(),
                ),
            )
        ).strip().lower()
        self.data_formula_mode = _resolve_data_formula_mode(reconstruction_formula_mode)
        if self.data_formula_mode == reconstruction_formula_mode:
            self.data_time_operator = self.time_operator
        else:
            self.data_time_operator = build_time_domain_operator(
                beta=THEORETICAL_CONFIG["beta_vector"],
                height=self.img_size,
                width=self.img_size,
                formula_mode_override=self.data_formula_mode,
            )
            if int(getattr(self.data_time_operator, "M", -1)) != int(getattr(self.time_operator, "M", -2)):
                raise ValueError(
                    "Data operator and reconstruction operator must have the same measurement length; "
                    f"got data M={getattr(self.data_time_operator, 'M', None)} and "
                    f"reconstruction M={getattr(self.time_operator, 'M', None)}."
                )
        data_formula_resolved = str(self.data_formula_mode).strip().lower()
        data_is_sparse = bool(getattr(self.data_time_operator, "uses_sparse_blocks", False))
        data_is_complete_banded = data_formula_resolved in _COMPLETE_BANDED_DATA_FORMULA_MODES
        if hasattr(self.data_time_operator, "uses_sparse_blocks") and (not data_is_sparse) and (not data_is_complete_banded):
            raise ValueError(
                "Data operator must be a complete sampling operator for generating R_beta f(t_i). "
                "Only exact triangular lower-banded storage is allowed to avoid full sparse work."
            )
        self.feature_time_operator = None
        feature_beta_vectors = TIME_DOMAIN_CONFIG.get("cnn_feature_beta_vectors_override", None)
        if feature_beta_vectors:
            feature_beta_vectors = [tuple(int(v) for v in beta) for beta in list(feature_beta_vectors)]
            self.feature_time_operator = TheoreticalB1B1Operator2D(
                beta_vectors=feature_beta_vectors,
                height=self.img_size,
                width=self.img_size,
                t0=float(TIME_DOMAIN_CONFIG.get("sampling_t0", 0.5)),
                formula_mode=_resolve_theoretical_formula_mode(
                    TIME_DOMAIN_CONFIG.get("theoretical_formula_mode", "auto"),
                    str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "stacked_tikhonov")).strip().lower(),
                ),
                auto_shift_t0=bool(TIME_DOMAIN_CONFIG.get("auto_angle_t0", True)),
            ).to(device)
        self.M = int(getattr(self.time_operator, "M", int(TIME_DOMAIN_CONFIG.get("num_detector_samples", self.N))))
        self.flatten_order = getattr(self.time_operator, "flatten_order", None)
        self.last_lambda: Optional[float | torch.Tensor] = None
        self._chol_lambda: Optional[float] = None
        self._chol_factor: Optional[torch.Tensor] = None
        self._ata_factor: Optional[torch.Tensor] = None
        self._first_batch_progress_logged = False

    def _normalize_lambda_reg(
        self,
        lambda_reg: float | torch.Tensor,
        batch_size: int,
        *,
        dtype: torch.dtype = torch.float32,
        target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if target_device is None:
            target_device = device
        if torch.is_tensor(lambda_reg):
            lam = lambda_reg.detach().to(dtype=dtype, device=target_device).view(-1)
            if int(lam.numel()) == 1 and batch_size > 1:
                lam = lam.expand(batch_size)
            elif int(lam.numel()) != batch_size:
                raise ValueError(
                    f"lambda_reg has {int(lam.numel())} entries, expected 1 or batch={batch_size}."
                )
        else:
            lam = torch.full((batch_size,), float(lambda_reg), dtype=dtype, device=target_device)
        return lam

    def forward_operator(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        """Forward operator, returns g (real)."""
        return self.time_operator.forward(coeff_matrix)

    def data_forward_operator(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        """Physical data forward operator used only for generating clean observations."""
        return self.data_time_operator.forward(coeff_matrix)

    def adjoint_operator(self, residual: torch.Tensor) -> torch.Tensor:
        """Adjoint operator, returns A^T r (real)."""
        return self.time_operator.adjoint(residual)

    @torch.no_grad()
    def solve_tikhonov_direct_init(self, g_obs: torch.Tensor, lambda_reg: float | torch.Tensor) -> torch.Tensor:
        return self._tikhonov_direct_init(g_obs, lambda_reg=lambda_reg)

    @torch.no_grad()
    def _tikhonov_direct_init(self, g_obs: torch.Tensor, lambda_reg: float | torch.Tensor) -> torch.Tensor:
        """
        Direct Tikhonov init by solving the normal equations with a cached Cholesky factor:

            (A^T A + lambda I) c0 = A^T g

        This is much faster and more accurate than running many CG iterations when A is fixed.
        """
        if g_obs.dim() == 1:
            g_obs = g_obs.unsqueeze(0)
        g_obs = g_obs.to(device=device, dtype=torch.float32)
        lam_batch = self._normalize_lambda_reg(
            lambda_reg,
            batch_size=int(g_obs.shape[0]),
            dtype=torch.float32,
            target_device=g_obs.device,
        )

        if hasattr(self.time_operator, "solve_tikhonov_direct") and not hasattr(self.time_operator, "A"):
            return self.time_operator.solve_tikhonov_direct(g_obs, lambda_reg=lam_batch)

        if hasattr(self.time_operator, "solve_tikhonov_cg") and not hasattr(self.time_operator, "A"):
            cg_iters = max(int(TIME_DOMAIN_CONFIG.get("init_cg_iters", 40)), 40)
            cg_tol = float(TIME_DOMAIN_CONFIG.get("init_cg_tol", 1e-4))
            return self.time_operator.solve_tikhonov_cg(
                g_obs,
                lambda_reg=lam_batch,
                max_iter=cg_iters,
                tol=cg_tol,
            )

        A = self.time_operator.A  # (M,N)
        if self._ata_factor is None:
            self._ata_factor = A.t() @ A  # (N,N)
        AtA = self._ata_factor  # (N,N)
        Atb = g_obs @ A  # (B,N)
        I = torch.eye(int(AtA.shape[0]), device=AtA.device, dtype=AtA.dtype)

        if int(lam_batch.numel()) == 1 or bool(torch.allclose(lam_batch, lam_batch[:1])):
            lam = float(lam_batch[0].item())
            if self._chol_factor is None or self._chol_lambda is None or abs(self._chol_lambda - lam) > 1e-12:
                L = torch.linalg.cholesky(AtA + (lam * I))
                self._chol_factor = L
                self._chol_lambda = lam
            x = torch.cholesky_solve(Atb.t(), self._chol_factor).t()  # (B,N)
            return x.view(-1, 1, self.img_size, self.img_size)

        xs = []
        for idx in range(int(g_obs.shape[0])):
            lam_i = float(lam_batch[idx].item())
            L_i = torch.linalg.cholesky(AtA + (lam_i * I))
            x_i = torch.cholesky_solve(Atb[idx:idx + 1].t(), L_i).t()
            xs.append(x_i)
        x = torch.cat(xs, dim=0)
        return x.view(-1, 1, self.img_size, self.img_size)

    @torch.no_grad()
    def _tikhonov_cg_init(
        self,
        g_obs: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        max_iter: int,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute a better coefficient init by (approximately) solving the Tikhonov normal equations:

            (A^T A + lambda I) c0 = A^T g

        using a small number of CG iterations.
        NOTE: the solve is performed in the full coefficient space (including boundary
        indices) so boundary coefficients are treated as unknowns during initialization.
        """
        if g_obs.dim() == 1:
            g_obs = g_obs.unsqueeze(0)
        g_obs = g_obs.to(device=device, dtype=torch.float32)
        lam_batch = self._normalize_lambda_reg(
            lambda_reg,
            batch_size=int(g_obs.shape[0]),
            dtype=torch.float32,
            target_device=g_obs.device,
        )

        if hasattr(self.time_operator, "solve_tikhonov_cg") and not hasattr(self.time_operator, "A"):
            return self.time_operator.solve_tikhonov_cg(
                g_obs,
                lambda_reg=lam_batch,
                max_iter=max_iter,
                tol=tol,
            )

        # CG in coefficient-vector space for speed:
        #   (A^T A + lambda I) x = A^T g
        A = self.time_operator.A  # (M,N) float32 on device
        if self._ata_factor is None:
            self._ata_factor = A.t() @ A  # (N,N)
        AtA = self._ata_factor  # (N,N)

        rhs = g_obs @ A  # (B,N)
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        diag = torch.diagonal(AtA).clamp_min(1e-8)  # (N,)
        diag_inv = 1.0 / (diag.view(1, -1) + lam_batch.view(-1, 1))
        z = r * diag_inv
        p = z.clone()
        rzold = torch.sum(r * z, dim=1, keepdim=True)
        eps = rhs.new_tensor(1e-12)

        lam = lam_batch.view(-1, 1)
        for _ in range(int(max_iter)):
            Ap = (p @ AtA) + (lam * p)
            denom = torch.sum(p * Ap, dim=1, keepdim=True).clamp_min(eps)
            alpha = rzold / denom
            x = x + alpha * p
            r = r - alpha * Ap
            r2 = torch.sum(r * r, dim=1, keepdim=True)
            if torch.sqrt(r2.max()).item() < float(tol):
                break
            z = r * diag_inv
            rznew = torch.sum(r * z, dim=1, keepdim=True)
            beta = rznew / (rzold + eps)
            p = z + beta * p
            rzold = rznew

        return x.view(-1, 1, self.img_size, self.img_size)

    def _sample_coefficients(self, batch_size: int = 1) -> torch.Tensor:
        """Return B1*B1 coefficient maps for the active data source."""
        if self.data_source == "shepp_logan":
            if self._phantom_cache is None:
                phantom = generate_shepp_logan_phantom(
                    image_size=self.img_size,
                    modified=True,
                    device=device,
                    dtype=torch.float32,
                )
                self._phantom_cache = phantom.view(1, 1, self.img_size, self.img_size)
            return self._phantom_cache.expand(batch_size, -1, -1, -1)

        if self.data_source in ("random_ellipses", "random_ellipse", "ellipse"):
            phantom_list = [
                generate_random_ellipse_phantom(image_size=self.img_size)
                for _ in range(int(batch_size))
            ]
            coeff = torch.stack(phantom_list, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)
            return coeff

        raise ValueError(
            f"Unsupported data_source={self.data_source!r}; expected 'random_ellipses' or 'shepp_logan'."
        )

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
        raise ValueError(
            f"Unsupported noise_mode={self.noise_mode!r}; expected 'additive', 'multiplicative', or 'snr'."
        )

    def generate_training_sample(self, random_seed=None, lambda_reg: float | torch.Tensor = None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        coeff_true = self._sample_coefficients()
        f_true = self.image_gen(coeff_true).squeeze(0)

        with torch.no_grad():
            g_clean = self.data_forward_operator(coeff_true).to(torch.float32)
            g_observed = self._apply_noise(g_clean)
            observed = g_observed
            if self.feature_time_operator is not None:
                g_feature_clean = self.feature_time_operator.forward(coeff_true).to(torch.float32)
                g_feature_observed = self._apply_noise(g_feature_clean)
                observed = torch.cat([observed, g_feature_observed], dim=-1)

        # Coefficient init (feeds the learned iterative optimizer).
        if lambda_reg is not None:
            if torch.is_tensor(lambda_reg):
                lambda_eff = float(lambda_reg.detach().view(-1)[0].item())
            else:
                lambda_eff = float(lambda_reg)
        else:
            mode = str(DATA_CONFIG.get("lambda_select_mode", "fixed")).strip().lower()
            if mode == "morozov":
                tau = float(DATA_CONFIG.get("morozov_tau", 1.0))
                max_iter = int(DATA_CONFIG.get("morozov_max_iter", 40))
                lam_min = float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12))
                lam_max = float(DATA_CONFIG.get("morozov_lambda_max", 1.0e12))
                noise_norm = torch.norm(g_observed - g_clean, dim=-1)  # (B,)
                lam = self.time_operator.choose_lambda_morozov(
                    g_observed,
                    noise_norm=noise_norm,
                    tau=tau,
                    max_iter=max_iter,
                    lambda_min=lam_min,
                    lambda_max=lam_max,
                )
                lambda_eff = float(lam.mean().item())
            else:
                lambda_eff = float(DATA_CONFIG.get("lambda_reg", 1e-2))
        self.last_lambda = lambda_eff
        init_method = str(TIME_DOMAIN_CONFIG.get("init_method", "cg")).strip().lower()
        init_cg_iters = int(TIME_DOMAIN_CONFIG.get("init_cg_iters", 0))

        if init_method == "tikhonov_direct":
            coeff_initial = self._tikhonov_direct_init(g_observed, lambda_reg=lambda_eff)
        elif init_method == "cg" and init_cg_iters > 0:
            coeff_initial = self._tikhonov_cg_init(
                g_observed,
                lambda_reg=lambda_eff,
                max_iter=init_cg_iters,
            )
        else:
            raise ValueError(
                f"Unsupported init_method={init_method!r}; expected 'cg' or 'tikhonov_direct'."
            )

        return (
            coeff_true.squeeze(0).squeeze(0),
            f_true.squeeze(0),
            observed.squeeze(0),
            coeff_initial.squeeze(0).squeeze(0),
        )

    def generate_batch(self, batch_size, random_seed=None, lambda_reg: float | torch.Tensor = None):
        """Vectorized batch generation: all samples generated in parallel."""
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        batch_started = time.perf_counter()
        init_method = str(TIME_DOMAIN_CONFIG.get("init_method", "cg")).strip().lower()
        solver_mode = str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "stacked_tikhonov")).strip().lower()
        lambda_mode = "provided" if lambda_reg is not None else str(DATA_CONFIG.get("lambda_select_mode", "fixed")).strip().lower()
        progress_enabled = (
            (not self._first_batch_progress_logged)
            and init_method == "tikhonov_direct"
            and solver_mode == "stacked_tikhonov"
        )
        if progress_enabled:
            print(
                "[init] first batch start "
                f"batch_size={int(batch_size)} angles={int(getattr(self.time_operator, 'num_angles', 1) or 1)} "
                f"lambda_mode={lambda_mode} init_method={init_method} "
                "solver=stacked_tikhonov"
            )

        # 1. Sample coefficients: (B,1,H,W)
        coeff_true = self._sample_coefficients(batch_size)

        # 2. Image from coefficients (batch-capable)
        f_true = self.image_gen(coeff_true)  # (B,1,H,W)

        with torch.no_grad():
            # 3. Forward operator: g_clean = A c, (B,M)
            g_clean = self.data_forward_operator(coeff_true).to(torch.float32)

            # 4. Add noise
            g_observed = self._apply_noise(g_clean)
            g_observed_full = g_observed
            if self.feature_time_operator is not None:
                g_feature_clean = self.feature_time_operator.forward(coeff_true).to(torch.float32)
                g_feature_observed = self._apply_noise(g_feature_clean)
                g_observed_full = torch.cat([g_observed, g_feature_observed], dim=-1)

        # 5. Coefficient init
        if lambda_reg is not None:
            lambda_eff = self._normalize_lambda_reg(
                lambda_reg,
                batch_size=int(batch_size),
                dtype=torch.float32,
                target_device=g_observed.device,
            )
        else:
            mode = lambda_mode
            if mode == "morozov":
                tau = float(DATA_CONFIG.get("morozov_tau", 1.0))
                max_iter = int(DATA_CONFIG.get("morozov_max_iter", 40))
                lam_min = float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12))
                lam_max = float(DATA_CONFIG.get("morozov_lambda_max", 1.0e12))
                noise_norm = torch.norm(g_observed - g_clean, dim=-1)
                lambda_started = time.perf_counter()
                lam = self.time_operator.choose_lambda_morozov(
                    g_observed,
                    noise_norm=noise_norm,
                    tau=tau,
                    max_iter=max_iter,
                    lambda_min=lam_min,
                    lambda_max=lam_max,
                )
                if progress_enabled:
                    print(
                        "[init] Morozov lambda selection finished "
                        f"in {time.perf_counter() - lambda_started:.2f}s"
                    )
                lambda_eff = lam.to(dtype=torch.float32, device=g_observed.device)
            else:
                lambda_eff = float(DATA_CONFIG.get("lambda_reg", 1e-2))
        self.last_lambda = lambda_eff
        init_cg_iters = int(TIME_DOMAIN_CONFIG.get("init_cg_iters", 0))

        coeff_init_started = time.perf_counter()
        if init_method == "tikhonov_direct":
            coeff_initial = self._tikhonov_direct_init(g_observed, lambda_reg=lambda_eff)
        elif init_method == "cg" and init_cg_iters > 0:
            coeff_initial = self._tikhonov_cg_init(
                g_observed, lambda_reg=lambda_eff, max_iter=init_cg_iters
            )
        else:
            raise ValueError(
                f"Unsupported init_method={init_method!r}; expected 'cg' or 'tikhonov_direct'."
            )
        if progress_enabled:
            print(
                "[init] coefficient init finished "
                f"in {time.perf_counter() - coeff_init_started:.2f}s"
            )
            print(f"[init] first batch ready in {time.perf_counter() - batch_started:.2f}s")
            self._first_batch_progress_logged = True

        # coeff_true: (B,1,H,W), f_true: (B,1,H,W), g_observed: (B,M), coeff_initial: (B,1,H,W)
        return coeff_true, f_true, g_observed_full, coeff_initial
