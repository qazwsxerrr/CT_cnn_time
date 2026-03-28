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

try:
    from scipy.linalg import cholesky_banded, cho_solve_banded, solve_banded
except Exception:  # pragma: no cover - exercised only when SciPy is unavailable
    cholesky_banded = None
    cho_solve_banded = None
    solve_banded = None


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


def _normalized_extra_angle_weight(
    base_weight: float,
    num_extra: int,
) -> float:
    _ = int(num_extra)
    return float(base_weight)


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


def _solve_tikhonov_from_gram_spectrum(
    rhs: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    lambda_reg: float | torch.Tensor,
) -> torch.Tensor:
    batch = int(rhs.shape[0])
    rhs_cpu = rhs.detach().to(dtype=torch.float32, device="cpu")
    eigvals_cpu = eigvals.detach().to(dtype=torch.float32, device="cpu")
    eigvecs_cpu = eigvecs.detach().to(dtype=torch.float32, device="cpu")
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
    rhs_proj = rhs_cpu @ eigvecs_cpu
    return (rhs_proj / denom) @ eigvecs_cpu.t()


def _choose_lambda_morozov_from_gram_spectrum(
    b: torch.Tensor,
    rhs: torch.Tensor,
    noise_norm: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    tau: float,
    settings: dict[str, float],
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
    rhs_proj_all = (rhs_cpu @ eigvecs_cpu).to(dtype=torch.float64)

    lam_list = []
    for idx in range(batch):
        rhs_proj2 = rhs_proj_all[idx].square().numpy()
        b_norm2 = float(torch.dot(b_cpu[idx], b_cpu[idx]).item())
        target2 = float(float(tau) * float(noise_cpu[idx].item())) ** 2

        def residual2_fn(lam: float) -> float:
            denom = eigvals_cpu + lam
            x_rhs = float(np.sum(rhs_proj2 / denom))
            x_norm2 = float(np.sum(rhs_proj2 / (denom * denom)))
            return max(0.0, b_norm2 - x_rhs - (lam * x_norm2))

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
                max_residual2=b_norm2,
            )
        )

    return torch.tensor(lam_list, dtype=b.dtype, device=b.device)


def _require_scipy_banded() -> None:
    if cholesky_banded is None or cho_solve_banded is None or solve_banded is None:
        raise ImportError(
            "SciPy banded linear algebra is required for split_triangular_admm. "
            "Install scipy or switch TIME_DOMAIN_CONFIG['multi_angle_solver_mode'] to 'stacked_tikhonov'."
        )


def _canonical_beta_direction(beta) -> tuple[int, int]:
    beta_i = _to_integer_beta(beta)
    a = int(beta_i[0].item())
    b = int(beta_i[1].item())
    g = math.gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    return (a, b)


def _sample_uniform_extra_beta_vectors(
    backbone_betas: list[tuple[int, int]],
    extra_count: int,
    *,
    height: int,
    width: int,
    seed: int,
) -> list[tuple[int, int]]:
    extra_count = int(extra_count)
    if extra_count <= 0:
        return []

    rng = np.random.default_rng(int(seed))
    scale = int(max(height, width))
    used = {_canonical_beta_direction(beta) for beta in backbone_betas}
    extra = []
    attempts = 0
    max_attempts = max(2048, 256 * extra_count)

    while len(extra) < extra_count and attempts < max_attempts:
        theta = float(rng.uniform(0.0, math.pi))
        a = int(round(scale * math.cos(theta)))
        b = int(round(scale * math.sin(theta)))
        if a == 0 and b == 0:
            attempts += 1
            continue
        beta_dir = _canonical_beta_direction((a, b))
        if beta_dir not in used:
            used.add(beta_dir)
            extra.append(beta_dir)
        attempts += 1

    if len(extra) != extra_count:
        raise RuntimeError(
            f"Failed to sample {extra_count} extra beta vectors uniformly after {attempts} attempts."
        )
    return extra


def _normalize_backbone_beta_vectors(beta_vectors) -> list[tuple[int, int]]:
    if beta_vectors is None or len(list(beta_vectors)) == 0:
        raise ValueError("use_multi_angle=True but TIME_DOMAIN_CONFIG['beta_vectors'] is empty.")
    normalized = []
    for beta in list(beta_vectors):
        beta_i = _to_integer_beta(beta)
        normalized.append((int(beta_i[0].item()), int(beta_i[1].item())))
    return normalized


def _validate_multi_angle_backbone(
    backbone_betas: list[tuple[int, int]],
    *,
    total_angles: int,
) -> None:
    if len(backbone_betas) != 8:
        raise ValueError(
            "The structured multi-angle pipeline requires exactly 8 backbone beta_vectors; "
            f"got {len(backbone_betas)}."
        )
    normalized = [tuple(int(v) for v in beta) for beta in backbone_betas]
    if len(set(normalized)) != 8:
        raise ValueError("beta_vectors must contain 8 distinct backbone directions.")
    if int(total_angles) < 8:
        raise ValueError(
            "num_angles_total must be >= 8 for the structured multi-angle pipeline; "
            f"got {int(total_angles)}."
        )


def _effective_angle_t0(alpha: torch.Tensor, beta_norm: float, base_t0: float, auto_shift: bool) -> float:
    if not bool(auto_shift):
        return float(base_t0)
    support_lo, _ = phi_support_bounds_b1b1(alpha)
    return float(support_lo * float(beta_norm) + float(base_t0))


def _formula_mode_from_solver_mode(solver_mode: str) -> str:
    solver_mode = str(solver_mode).strip().lower()
    if solver_mode == "stacked_tikhonov":
        return "legacy"
    if solver_mode == "split_triangular_admm":
        return "shifted_support"
    raise ValueError(
        f"Unsupported multi_angle_solver_mode={solver_mode!r}; "
        "expected 'stacked_tikhonov' or 'split_triangular_admm'."
    )


def _kernel_support_length(r: torch.Tensor, tol: float = 1.0e-8) -> int:
    nz = torch.nonzero(torch.abs(r.detach().to(dtype=torch.float64)) > float(tol)).view(-1)
    if int(nz.numel()) == 0:
        return 1
    return int(nz[-1].item()) + 1


def _build_lower_banded_ab_from_kernel(kernel: np.ndarray, n: int) -> np.ndarray:
    bw = int(kernel.shape[0])
    ab = np.zeros((bw, int(n)), dtype=np.float64)
    for offset in range(bw):
        ab[offset, : int(n) - offset] = float(kernel[offset])
    return ab


def _build_upper_normal_banded_from_kernel(
    kernel: np.ndarray,
    n: int,
    *,
    rho: float,
    weight: float,
) -> np.ndarray:
    bw = int(kernel.shape[0])
    n = int(n)
    ab = np.zeros((bw, n), dtype=np.float64)
    for offset in range(bw):
        prod = kernel[offset:] * kernel[: bw - offset]
        prefix = np.cumsum(prod, dtype=np.float64)
        diag = np.full(n - offset, prefix[-1], dtype=np.float64)
        tail_len = bw - offset - 1
        if tail_len > 0:
            diag[-tail_len:] = prefix[tail_len - 1 :: -1]
        diag *= (2.0 * float(weight))
        if offset == 0:
            diag += float(rho)
        ab[bw - 1 - offset, offset:] = diag
    return ab


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
    formula_mode: str = "legacy",
    auto_shift_t0: bool = True,
) -> dict[str, torch.Tensor]:
    """Construct one per-direction theoretical B1*B1 block metadata without forming dense matrices."""
    h = int(height)
    w = int(width)
    n = int(h * w)
    beta_i = _to_integer_beta(beta)
    beta_f = beta_i.to(torch.float64)
    beta_norm = float(torch.norm(beta_f, p=2).item())
    alpha = beta_f / beta_norm

    k1, k2 = _lex_lattice_indices(h, w)
    beta_dot_k = beta_i[0] * k1 + beta_i[1] * k2

    uniq_sorted = torch.sort(torch.unique(beta_dot_k)).values
    if int(uniq_sorted.numel()) != n:
        raise ValueError(
            f"beta={tuple(int(x) for x in beta_i.tolist())} does not make beta·k injective on [0,{h-1}]x[0,{w-1}]."
        )

    kappa0 = int(uniq_sorted[0].item())
    kappa_m = int(uniq_sorted[-1].item())
    if (kappa_m - kappa0 + 1) != n:
        raise ValueError(
            f"range(beta·k) must be contiguous with size N={n}, got [{kappa0}, {kappa_m}]"
        )
    expected = torch.arange(kappa0, kappa_m + 1, dtype=torch.int64)
    if not torch.equal(uniq_sorted, expected):
        raise ValueError("range(beta·k) must be consecutive integers for theoretical B1*B1 operator.")

    kappa0 = int(uniq_sorted[0].item())
    lex_to_d = (beta_dot_k - kappa0).to(torch.int64)
    d_to_lex = torch.empty(n, dtype=torch.int64)
    d_to_lex[lex_to_d] = torch.arange(n, dtype=torch.int64)

    k = torch.arange(n, dtype=torch.float64)
    formula_mode = str(formula_mode).strip().lower()
    if formula_mode == "legacy":
        effective_t0 = float(t0)
        sampling_points = (float(t0) + k) / beta_norm
        r = radon_phi_b1b1((float(t0) + k - float(kappa0)) / beta_norm, alpha).to(torch.float64)
    elif formula_mode == "shifted_support":
        effective_t0 = _effective_angle_t0(alpha, beta_norm=beta_norm, base_t0=float(t0), auto_shift=auto_shift_t0)
        sampling_points = (float(effective_t0) + float(kappa0) + k) / beta_norm
        r = radon_phi_b1b1((float(effective_t0) + k) / beta_norm, alpha).to(torch.float64)
    else:
        raise ValueError(
            f"Unknown B1*B1 formula_mode={formula_mode!r}; expected 'legacy' or 'shifted_support'."
        )

    return {
        "r": r,
        "alpha": alpha,
        "beta": beta_i,
        "sampling_points": sampling_points,
        "lex_to_d": lex_to_d,
        "d_to_lex": d_to_lex,
        "kappa0": torch.tensor(kappa0, dtype=torch.int64),
        "effective_t0": torch.tensor(float(effective_t0), dtype=torch.float64),
    }


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

    def forward_per_angle(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(1)
        coeff_matrix = coeff_matrix.to(dtype=torch.float32, device=self.alphas.device)
        padded = self._pad_image(coeff_matrix)
        proj_list = []
        for idx in range(self.num_angles):
            rotated = self._rotate_batch(padded, self.forward_grids[idx])
            proj_full = rotated.sum(dim=2).squeeze(1)
            proj = self._resize_projection(proj_full, self.M_per_angle)
            proj_list.append(proj)
        return torch.stack(proj_list, dim=1)

    def forward(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.forward_per_angle(coeff_matrix).reshape(coeff_matrix.shape[0], self.M)

    def adjoint_per_angle(self, residual_per_angle: torch.Tensor) -> torch.Tensor:
        if residual_per_angle.dim() == 4 and residual_per_angle.shape[2] == 1:
            residual_per_angle = residual_per_angle.squeeze(2)
        if residual_per_angle.dim() != 3:
            raise ValueError(
                f"Expected residual_per_angle with shape (B,K,M_per_angle), got {tuple(residual_per_angle.shape)}"
            )
        residual_per_angle = residual_per_angle.to(dtype=torch.float32, device=self.alphas.device)
        out = []
        with torch.enable_grad():
            for idx in range(self.num_angles):
                x = torch.zeros(
                    (residual_per_angle.shape[0], 1, self.height, self.width),
                    device=self.alphas.device,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                proj = self.forward_per_angle(x)[:, idx, :]
                weighted_sum = torch.sum(proj * residual_per_angle[:, idx, :])
                grad = torch.autograd.grad(weighted_sum, x, retain_graph=False, create_graph=False)[0]
                out.append(grad.detach())
        return torch.stack(out, dim=1)

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.dim() == 1:
            residual = residual.unsqueeze(0)
        residual = residual.to(dtype=torch.float32, device=self.alphas.device)
        with torch.enable_grad():
            x = torch.zeros(
                (residual.shape[0], 1, self.height, self.width),
                device=self.alphas.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            weighted_sum = torch.sum(self.forward(x) * residual)
            grad = torch.autograd.grad(weighted_sum, x, retain_graph=False, create_graph=False)[0]
        return grad.detach()

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
        formula_mode: str = "legacy",
        auto_shift_t0: bool = True,
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

        with torch.no_grad():
            blocks = [
                _theoretical_b1b1_block(
                    beta=beta,
                    height=self.height,
                    width=self.width,
                    t0=self.t0,
                    formula_mode=self.formula_mode,
                    auto_shift_t0=self.auto_shift_t0,
                )
                for beta in self.beta_vectors
            ]
            r_vectors = torch.stack([blk["r"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            alphas = torch.stack([blk["alpha"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            betas = torch.stack([blk["beta"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            lex_to_d = torch.stack([blk["lex_to_d"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            d_to_lex = torch.stack([blk["d_to_lex"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            kappa0 = torch.stack([blk["kappa0"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            effective_t0 = torch.stack([blk["effective_t0"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            sampling_points_pa = torch.stack([blk["sampling_points"] for blk in blocks], dim=0).to(
                dtype=torch.float32, device=device
            )

        conv_size = 1 << (int(2 * self.N - 1).bit_length())
        r_fft = torch.fft.rfft(r_vectors, n=conv_size, dim=1)

        self.register_buffer("r_vectors", r_vectors)
        self.register_buffer("r_fft", r_fft)
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        self.register_buffer("lex_to_d_indices", lex_to_d)
        self.register_buffer("d_to_lex_indices", d_to_lex)
        self.register_buffer("kappa0_per_angle", kappa0)
        self.register_buffer("effective_t0_per_angle", effective_t0)
        self.register_buffer("sampling_points_per_angle", sampling_points_pa)
        self.register_buffer("sampling_points", sampling_points_pa.reshape(-1))
        self.conv_size = int(conv_size)
        self._morozov_gram_eigvals: Optional[torch.Tensor] = None
        self._morozov_gram_eigvecs: Optional[torch.Tensor] = None
        self.last_morozov_cache_hit: Optional[bool] = None
        self.last_morozov_cache_build_seconds: Optional[float] = None
        self.last_split_admm_stats: Optional[dict[str, object]] = None
        self._split_linear_cache: dict[tuple[int, float, float], dict[str, np.ndarray]] = {}

    def _morozov_cache_fingerprint(self) -> dict[str, object]:
        return {
            "class_name": self.__class__.__name__,
            "height": int(self.height),
            "width": int(self.width),
            "num_angles": int(self.num_angles),
            "sampling_t0": float(self.t0),
            "formula_mode": str(self.formula_mode),
            "auto_shift_t0": bool(self.auto_shift_t0),
            "effective_t0_per_angle": [float(v.item()) for v in self.effective_t0_per_angle],
            "beta_vectors": [list(beta) for beta in self.beta_vectors],
            "basis": "b1b1",
        }

    def _toeplitz_apply(self, r_fft: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_full = torch.fft.irfft(
            torch.fft.rfft(x.to(torch.float32), n=self.conv_size, dim=1) * r_fft.unsqueeze(0),
            n=self.conv_size,
            dim=1,
        )
        return y_full[:, : self.N]

    def _toeplitz_adjoint_apply(self, r_fft: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        rev = torch.flip(x, dims=[1])
        rev_y = self._toeplitz_apply(r_fft, rev)
        return torch.flip(rev_y, dims=[1])

    def _permute_c_to_d(self, coeff_flat: torch.Tensor, angle_idx: int) -> torch.Tensor:
        return coeff_flat.index_select(1, self.d_to_lex_indices[int(angle_idx)])

    def _permute_d_to_c(self, d_vector: torch.Tensor, angle_idx: int) -> torch.Tensor:
        return d_vector.gather(1, self.lex_to_d_indices[int(angle_idx)].view(1, -1).expand(d_vector.shape[0], -1))

    def _split_kernel_np(self, angle_idx: int) -> np.ndarray:
        support_len = _kernel_support_length(self.r_vectors[int(angle_idx)])
        return self.r_vectors[int(angle_idx), :support_len].detach().to(dtype=torch.float64, device="cpu").numpy()

    def _solve_single_lower_direct(self, angle_idx: int, rhs: torch.Tensor) -> torch.Tensor:
        _require_scipy_banded()
        angle_idx = int(angle_idx)
        if rhs.dim() == 1:
            rhs = rhs.unsqueeze(0)
        rhs_np = rhs.detach().to(dtype=torch.float64, device="cpu").numpy().T
        kernel = self._split_kernel_np(angle_idx)
        lower_ab = _build_lower_banded_ab_from_kernel(kernel, self.N)
        solved = solve_banded((kernel.shape[0] - 1, 0), lower_ab, rhs_np, check_finite=False)
        return torch.from_numpy(np.asarray(solved.T, dtype=np.float32)).to(device=self.r_vectors.device, dtype=torch.float32)

    def _get_split_linear_cache(self, angle_idx: int, rho: float, weight: float) -> dict[str, np.ndarray]:
        key = (int(angle_idx), round(float(rho), 12), round(float(weight), 12))
        cached = self._split_linear_cache.get(key, None)
        if cached is not None:
            return cached
        _require_scipy_banded()
        kernel = self._split_kernel_np(angle_idx)
        upper_ab = _build_upper_normal_banded_from_kernel(kernel, self.N, rho=float(rho), weight=float(weight))
        chol = cholesky_banded(upper_ab, lower=False, check_finite=False)
        cached = {
            "kernel": kernel,
            "chol": chol,
            "bandwidth": np.asarray([kernel.shape[0]], dtype=np.int64),
        }
        self._split_linear_cache[key] = cached
        return cached

    def _solve_single_split_quadratic(
        self,
        angle_idx: int,
        rhs: torch.Tensor,
        *,
        rho: float,
        weight: float,
    ) -> torch.Tensor:
        cache = self._get_split_linear_cache(angle_idx, rho=float(rho), weight=float(weight))
        if rhs.dim() == 1:
            rhs = rhs.unsqueeze(0)
        rhs_np = rhs.detach().to(dtype=torch.float64, device="cpu").numpy().T
        solved = cho_solve_banded((cache["chol"], False), rhs_np, check_finite=False)
        return torch.from_numpy(np.asarray(solved.T, dtype=np.float32)).to(device=self.r_vectors.device, dtype=torch.float32)

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
        coeff_matrix = coeff_matrix.to(dtype=torch.float32, device=self.r_vectors.device)
        batch = int(coeff_matrix.shape[0])
        coeff_flat = coeff_matrix.view(batch, self.N)
        outputs = []
        for idx in range(self.num_angles):
            d = coeff_flat.index_select(1, self.d_to_lex_indices[idx])
            outputs.append(self._toeplitz_apply(self.r_fft[idx], d))
        return torch.stack(outputs, dim=1)

    def forward(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.forward_per_angle(coeff_matrix).reshape(coeff_matrix.shape[0], self.M)

    def adjoint_per_angle(self, residual_per_angle: torch.Tensor) -> torch.Tensor:
        if residual_per_angle.dim() == 4 and residual_per_angle.shape[2] == 1:
            residual_per_angle = residual_per_angle.squeeze(2)
        if residual_per_angle.dim() != 3:
            raise ValueError(
                f"Expected residual_per_angle with shape (B,K,M_per_angle), got {tuple(residual_per_angle.shape)}"
            )
        residual_per_angle = residual_per_angle.to(dtype=torch.float32, device=self.r_vectors.device)
        batch = int(residual_per_angle.shape[0])
        grads = []
        for idx in range(self.num_angles):
            grad_d = self._toeplitz_adjoint_apply(self.r_fft[idx], residual_per_angle[:, idx, :])
            grad_c = grad_d.gather(1, self.lex_to_d_indices[idx].view(1, -1).expand(batch, -1))
            grads.append(grad_c.view(batch, 1, self.height, self.width))
        return torch.stack(grads, dim=1)

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
        if b.dim() == 1:
            b = b.unsqueeze(0)
        rhs = self.adjoint(b.to(dtype=torch.float32, device=self.r_vectors.device)).view(b.shape[0], self.N)
        eigvals, eigvecs = _ensure_implicit_gram_spectrum(self, self._morozov_cache_fingerprint())
        coeff = _solve_tikhonov_from_gram_spectrum(rhs, eigvals=eigvals, eigvecs=eigvecs, lambda_reg=lambda_reg)
        return coeff.to(device=self.r_vectors.device, dtype=torch.float32).view(-1, 1, self.height, self.width)

    @torch.no_grad()
    def solve_split_triangular_admm(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        *,
        rho: Optional[float] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        weights: Optional[torch.Tensor] = None,
        extra_operator: Optional[torch.nn.Module] = None,
        extra_measurements: Optional[torch.Tensor] = None,
        extra_weight: Optional[float] = None,
    ) -> torch.Tensor:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        if int(self.num_angles) != 8:
            raise ValueError(
                "split_triangular_admm requires exactly 8 backbone angles; "
                f"got {int(self.num_angles)}."
            )

        _require_scipy_banded()
        self.last_split_admm_stats = None
        batch = int(b.shape[0])
        b = b.to(dtype=torch.float32, device=self.r_vectors.device)
        g_backbone = self.split_measurements(b)  # (B,K,N)
        lam_batch = lambda_reg
        if torch.is_tensor(lam_batch):
            lam = lam_batch.detach().to(dtype=torch.float32, device=self.r_vectors.device).view(-1)
            if int(lam.numel()) == 1 and batch > 1:
                lam = lam.expand(batch)
            elif int(lam.numel()) != batch:
                raise ValueError(
                    f"lambda_reg has {int(lam.numel())} entries, expected 1 or batch={batch}."
                )
        else:
            lam = torch.full((batch,), float(lambda_reg), dtype=torch.float32, device=self.r_vectors.device)

        rho_eff = float(TIME_DOMAIN_CONFIG.get("split_admm_rho", 1.0) if rho is None else rho)
        max_iter_eff = int(TIME_DOMAIN_CONFIG.get("split_admm_max_iter", 20) if max_iter is None else max_iter)
        tol_eff = float(TIME_DOMAIN_CONFIG.get("split_admm_tol", 1.0e-4) if tol is None else tol)
        if weights is None:
            weight_vec = torch.ones((self.num_angles,), dtype=torch.float32, device=self.r_vectors.device)
        else:
            weight_vec = weights.detach().to(dtype=torch.float32, device=self.r_vectors.device).view(-1)
            if int(weight_vec.numel()) != int(self.num_angles):
                raise ValueError(
                    f"weights has {int(weight_vec.numel())} entries, expected num_angles={int(self.num_angles)}."
                )

        if extra_operator is not None or extra_measurements is not None or extra_weight is not None:
            raise ValueError(
                "solve_split_triangular_admm only supports the backbone 8-angle problem. "
                "Extra-angle refinement must be applied outside the backbone ADMM."
            )

        d_stack = []
        for angle_idx in range(self.num_angles):
            d_init = self._solve_single_lower_direct(angle_idx, g_backbone[:, angle_idx, :])
            d_stack.append(d_init)
        d_stack = torch.stack(d_stack, dim=1)  # (B,K,N)

        c_flat = torch.zeros((batch, self.N), dtype=torch.float32, device=self.r_vectors.device)
        for angle_idx in range(self.num_angles):
            c_flat = c_flat + self._permute_d_to_c(d_stack[:, angle_idx, :], angle_idx)
        c_flat = c_flat / float(self.num_angles)
        u_stack = torch.zeros_like(d_stack)

        sqrt_k = math.sqrt(float(self.num_angles))
        primal = float("inf")
        dual = float("inf")
        primal_rel = float("inf")
        dual_rel = float("inf")
        converged = False
        iterations_run = 0
        for iter_idx in range(max_iter_eff):
            c_prev = c_flat.clone()

            for angle_idx in range(self.num_angles):
                consensus = self._permute_c_to_d(c_flat, angle_idx) - u_stack[:, angle_idx, :]
                rhs = (
                    2.0
                    * float(weight_vec[angle_idx].item())
                    * self._toeplitz_adjoint_apply(self.r_fft[angle_idx], g_backbone[:, angle_idx, :])
                    + rho_eff * consensus
                )
                d_stack[:, angle_idx, :] = self._solve_single_split_quadratic(
                    angle_idx,
                    rhs,
                    rho=rho_eff,
                    weight=float(weight_vec[angle_idx].item()),
                )

            consensus_sum = torch.zeros_like(c_flat)
            for angle_idx in range(self.num_angles):
                consensus_sum = consensus_sum + self._permute_d_to_c(
                    d_stack[:, angle_idx, :] + u_stack[:, angle_idx, :],
                    angle_idx,
                )

            denom = (2.0 * lam) + (rho_eff * float(self.num_angles))
            c_flat = (rho_eff * consensus_sum) / denom.view(-1, 1).clamp_min(1.0e-8)

            primal_sq = c_flat.new_zeros((batch,))
            d_norm_sq = c_flat.new_zeros((batch,))
            pc_norm_sq = c_flat.new_zeros((batch,))
            for angle_idx in range(self.num_angles):
                permuted = self._permute_c_to_d(c_flat, angle_idx)
                diff = d_stack[:, angle_idx, :] - permuted
                u_stack[:, angle_idx, :] = u_stack[:, angle_idx, :] + diff
                primal_sq = primal_sq + torch.sum(diff * diff, dim=1)
                d_norm_sq = d_norm_sq + torch.sum(d_stack[:, angle_idx, :] * d_stack[:, angle_idx, :], dim=1)
                pc_norm_sq = pc_norm_sq + torch.sum(permuted * permuted, dim=1)

            primal_abs_batch = torch.sqrt(primal_sq).div(math.sqrt(float(self.N) * float(self.num_angles)))
            dual_abs_batch = (
                rho_eff * sqrt_k * torch.norm(c_flat - c_prev, dim=1).div(math.sqrt(float(self.N)))
            )
            primal_scale = torch.maximum(
                torch.sqrt(d_norm_sq).div(math.sqrt(float(self.N) * float(self.num_angles))),
                torch.sqrt(pc_norm_sq).div(math.sqrt(float(self.N) * float(self.num_angles))),
            ).clamp_min(1.0e-8)
            dual_scale = (
                rho_eff
                * sqrt_k
                * torch.norm(c_flat, dim=1).div(math.sqrt(float(self.N)))
            ).clamp_min(1.0e-8)
            primal_rel_batch = primal_abs_batch / primal_scale
            dual_rel_batch = dual_abs_batch / dual_scale

            primal = primal_abs_batch.max().item()
            dual = dual_abs_batch.max().item()
            primal_rel = primal_rel_batch.max().item()
            dual_rel = dual_rel_batch.max().item()
            iterations_run = int(iter_idx + 1)
            if max(primal_rel, dual_rel) <= tol_eff:
                converged = True
                break

        self.last_split_admm_stats = {
            "iterations": int(iterations_run),
            "max_iter": int(max_iter_eff),
            "converged": bool(converged),
            "primal_residual": float(primal),
            "dual_residual": float(dual),
            "relative_primal_residual": float(primal_rel),
            "relative_dual_residual": float(dual_rel),
            "rho": float(rho_eff),
            "tol": float(tol_eff),
            "used_extra_angles": False,
            "num_angles": int(self.num_angles),
        }
        return c_flat.view(-1, 1, self.height, self.width)

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
        solver_mode = str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "split_triangular_admm")).strip().lower()
        if solver_mode == "split_triangular_admm" and int(self.num_angles) == 8:
            return self.solve_split_triangular_admm(
                b,
                lambda_reg=lambda_reg,
                rho=rho,
                max_iter=max_iter,
                tol=tol,
            )
        if solver_mode not in {"split_triangular_admm", "stacked_tikhonov"}:
            raise ValueError(
                "Unsupported multi_angle_solver_mode="
                f"{solver_mode!r}; expected 'split_triangular_admm' or 'stacked_tikhonov'."
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
        b = b.to(dtype=torch.float32, device=self.r_vectors.device)
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
        b = b.to(dtype=torch.float32, device=self.r_vectors.device)
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


class StructuredMultiAngleB1B1Operator2D(torch.nn.Module):
    """
    Multi-angle B1*B1 operator with:
    - 8 fixed backbone eligible directions solved via split triangular blocks
    - extra directions treated as generic refinement views
    """

    def __init__(
        self,
        backbone_beta_vectors: list[tuple[int, int]],
        extra_beta_vectors: list[tuple[int, int]],
        *,
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        t0: float = 0.5,
        formula_mode: str = "legacy",
        auto_shift_t0: bool = True,
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.N = int(self.height * self.width)
        self.backbone_operator = TheoreticalB1B1Operator2D(
            beta_vectors=list(backbone_beta_vectors),
            height=self.height,
            width=self.width,
            t0=float(t0),
            formula_mode=str(formula_mode),
            auto_shift_t0=bool(auto_shift_t0),
        )
        self.extra_beta_vectors = [tuple(int(v) for v in beta) for beta in list(extra_beta_vectors)]
        self.num_backbone = int(self.backbone_operator.num_angles)
        self.num_extra = int(len(self.extra_beta_vectors))
        self.formula_mode = str(self.backbone_operator.formula_mode)
        self.extra_operator = None
        if self.num_extra > 0:
            self.extra_operator = ImplicitPixelRadonOperator2D(
                beta_vectors=self.extra_beta_vectors,
                height=self.height,
                width=self.width,
                num_detector_samples_per_angle=self.N,
            ).to(device)
            if int(self.extra_operator.M_per_angle) != int(self.backbone_operator.M_per_angle):
                raise ValueError(
                    "Structured multi-angle operator requires equal detector counts per angle "
                    f"(backbone={self.backbone_operator.M_per_angle}, extra={self.extra_operator.M_per_angle})."
                )

        self.num_angles = int(self.num_backbone + self.num_extra)
        self.M_per_angle = int(self.backbone_operator.M_per_angle)
        self.M = int(self.num_angles * self.M_per_angle)
        self.beta_vectors = list(self.backbone_operator.beta_vectors) + list(self.extra_beta_vectors)
        self._morozov_gram_eigvals: Optional[torch.Tensor] = None
        self._morozov_gram_eigvecs: Optional[torch.Tensor] = None
        self.last_morozov_cache_hit: Optional[bool] = None
        self.last_morozov_cache_build_seconds: Optional[float] = None
        self.last_split_admm_stats: Optional[dict[str, object]] = None

        sampling_parts = [self.backbone_operator.sampling_points_per_angle]
        if self.extra_operator is not None:
            sampling_parts.append(self.extra_operator.sampling_points_per_angle)
        sampling_points_pa = torch.cat(sampling_parts, dim=0)
        self.register_buffer("sampling_points_per_angle", sampling_points_pa)
        self.register_buffer("sampling_points", sampling_points_pa.reshape(-1))

    def _morozov_cache_fingerprint(self) -> dict[str, object]:
        return {
            "class_name": self.__class__.__name__,
            "height": int(self.height),
            "width": int(self.width),
            "num_angles": int(self.num_angles),
            "num_backbone": int(self.num_backbone),
            "formula_mode": str(self.formula_mode),
            "backbone_beta_vectors": [list(beta) for beta in self.backbone_operator.beta_vectors],
            "extra_beta_vectors": [list(beta) for beta in self.extra_beta_vectors],
            "basis": "b1b1_backbone_plus_extra",
        }

    def split_measurements(self, g: torch.Tensor) -> torch.Tensor:
        if g.dim() == 3 and g.shape[1] == 1:
            g = g.squeeze(1)
        if g.dim() != 2:
            raise ValueError(f"Expected g with shape (B,M), got {tuple(g.shape)}")
        if int(g.shape[1]) != int(self.M):
            raise ValueError(f"Expected measurement length M={self.M}, got {g.shape[1]}")
        return g.view(g.shape[0], self.num_angles, self.M_per_angle)

    def _split_backbone_extra(self, g: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        g_pa = self.split_measurements(g)
        g_backbone = g_pa[:, : self.num_backbone, :]
        g_extra = None
        if self.num_extra > 0:
            g_extra = g_pa[:, self.num_backbone :, :].reshape(g_pa.shape[0], self.num_extra * self.M_per_angle)
        return g_backbone, g_extra

    def forward_per_angle(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        parts = [self.backbone_operator.forward_per_angle(coeff_matrix)]
        if self.extra_operator is not None:
            parts.append(self.extra_operator.forward_per_angle(coeff_matrix))
        return torch.cat(parts, dim=1)

    def forward(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.forward_per_angle(coeff_matrix).reshape(coeff_matrix.shape[0], self.M)

    def adjoint_per_angle(self, residual_per_angle: torch.Tensor) -> torch.Tensor:
        if residual_per_angle.dim() == 4 and residual_per_angle.shape[2] == 1:
            residual_per_angle = residual_per_angle.squeeze(2)
        if residual_per_angle.dim() != 3:
            raise ValueError(
                f"Expected residual_per_angle with shape (B,K,M_per_angle), got {tuple(residual_per_angle.shape)}"
            )
        parts = [self.backbone_operator.adjoint_per_angle(residual_per_angle[:, : self.num_backbone, :])]
        if self.extra_operator is not None:
            parts.append(self.extra_operator.adjoint_per_angle(residual_per_angle[:, self.num_backbone :, :]))
        return torch.cat(parts, dim=1)

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        g_backbone, g_extra = self._split_backbone_extra(residual)
        grad = self.backbone_operator.adjoint(g_backbone.reshape(g_backbone.shape[0], -1))
        if self.extra_operator is not None and g_extra is not None:
            grad = grad + self.extra_operator.adjoint(g_extra)
        return grad

    def apply_normal(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(coeff_matrix))

    @torch.no_grad()
    def _solve_extra_delta_refinement(
        self,
        c_backbone: torch.Tensor,
        g_extra: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        extra_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float | str]]:
        if self.extra_operator is None or int(self.num_extra) <= 0:
            return c_backbone, {
                "extra_refine_gamma": 0.0,
                "extra_refine_backbone_gamma": 0.0,
                "extra_refine_residual_gamma": 0.0,
                "extra_refine_delta_norm": 0.0,
                "extra_refine_solver": str(TIME_DOMAIN_CONFIG.get("extra_refine_solver", "direct")),
                "extra_refine_residual_norm": 0.0,
            }
        if g_extra.dim() == 1:
            g_extra = g_extra.unsqueeze(0)
        batch = int(g_extra.shape[0])
        g_extra = g_extra.to(dtype=torch.float32, device=device)
        mu = max(float(extra_weight), 1.0e-8)
        extra_solver = str(TIME_DOMAIN_CONFIG.get("extra_refine_solver", "direct")).strip().lower()
        if extra_solver not in {"direct", "cg"}:
            raise ValueError(
                "Unsupported extra_refine_solver="
                f"{extra_solver!r}; expected 'direct' or 'cg'."
            )
        g_extra_pred = self.extra_operator.forward(c_backbone).view(batch, -1)
        extra_residual = g_extra - g_extra_pred

        if torch.is_tensor(lambda_reg):
            lambda_backbone = lambda_reg.detach().to(dtype=torch.float32, device=device).view(-1)
            if int(lambda_backbone.numel()) == 1 and batch > 1:
                lambda_backbone = lambda_backbone.expand(batch)
            elif int(lambda_backbone.numel()) != batch:
                raise ValueError(
                    f"lambda_reg has {int(lambda_backbone.numel())} entries, expected 1 or batch={batch}."
                )
        else:
            lambda_backbone = torch.full((batch,), float(lambda_reg), dtype=torch.float32, device=device)

        gamma_scale = float(TIME_DOMAIN_CONFIG.get("extra_refine_gamma_scale", 8.0))
        residual_scale = float(TIME_DOMAIN_CONFIG.get("extra_refine_residual_scale", 8.0))
        residual_norm = torch.norm(extra_residual.view(batch, -1), dim=1) / math.sqrt(float(extra_residual.shape[1]))
        gamma_backbone = gamma_scale * lambda_backbone
        gamma_residual = residual_scale * residual_norm
        gamma = gamma_backbone + gamma_residual
        lambda_eff = gamma / mu
        if extra_solver == "direct":
            delta_c = self.extra_operator.solve_tikhonov_direct(
                extra_residual,
                lambda_reg=lambda_eff,
            )
        else:
            delta_c = self.extra_operator.solve_tikhonov_cg(
                extra_residual,
                lambda_reg=lambda_eff,
                max_iter=int(TIME_DOMAIN_CONFIG.get("extra_refine_cg_iters", 20)),
                tol=float(TIME_DOMAIN_CONFIG.get("extra_refine_cg_tol", 1.0e-4)),
                x0=torch.zeros_like(c_backbone),
            )
        refined = c_backbone + delta_c.to(device=device, dtype=torch.float32)
        delta_norm = torch.norm(delta_c.view(batch, -1), dim=1).mean().item() / math.sqrt(float(self.N))
        stats = {
            "extra_refine_gamma": float(gamma.mean().item()),
            "extra_refine_backbone_gamma": float(gamma_backbone.mean().item()),
            "extra_refine_residual_gamma": float(gamma_residual.mean().item()),
            "extra_refine_delta_norm": float(delta_norm),
            "extra_refine_solver": str(extra_solver),
            "extra_refine_residual_norm": float(residual_norm.mean().item()),
        }
        return (
            refined.to(device=device, dtype=torch.float32).view(-1, 1, self.height, self.width),
            stats,
        )

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
        solver_mode = str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "split_triangular_admm")).strip().lower()
        if solver_mode == "split_triangular_admm":
            g_backbone, g_extra = self._split_backbone_extra(b)
            base_extra_weight = float(TIME_DOMAIN_CONFIG.get("extra_angle_weight_mu", 1.0))
            normalized_extra_weight = _normalized_extra_angle_weight(base_extra_weight, self.num_extra)
            rho_eff = float(TIME_DOMAIN_CONFIG.get("split_admm_rho", 1.0) if rho is None else rho)
            coeff_backbone = self.backbone_operator.solve_split_triangular_admm(
                g_backbone.reshape(g_backbone.shape[0], -1),
                lambda_reg=lambda_reg,
                rho=rho,
                max_iter=max_iter,
                tol=tol,
            )
            coeff = coeff_backbone
            extra_stats = {
                "extra_refine_gamma": 0.0,
                "extra_refine_backbone_gamma": 0.0,
                "extra_refine_residual_gamma": 0.0,
                "extra_refine_delta_norm": 0.0,
                "extra_refine_solver": str(TIME_DOMAIN_CONFIG.get("extra_refine_solver", "direct")),
                "extra_refine_residual_norm": 0.0,
            }
            if self.extra_operator is not None and g_extra is not None and float(normalized_extra_weight) > 0.0:
                coeff, extra_stats = self._solve_extra_delta_refinement(
                    coeff_backbone,
                    g_extra,
                    lambda_reg=lambda_reg,
                    extra_weight=normalized_extra_weight,
                )
            stats = getattr(self.backbone_operator, "last_split_admm_stats", None)
            self.last_split_admm_stats = dict(stats) if stats is not None else None
            if self.last_split_admm_stats is not None:
                self.last_split_admm_stats["num_extra_angles"] = int(self.num_extra)
                self.last_split_admm_stats["base_extra_weight_mu"] = float(base_extra_weight)
                self.last_split_admm_stats["effective_extra_weight_mu"] = float(normalized_extra_weight)
                self.last_split_admm_stats["solve_path"] = (
                    "backbone_split_admm_plus_extra_delta_refinement"
                    if self.extra_operator is not None and g_extra is not None and float(normalized_extra_weight) > 0.0
                    else "backbone_split_admm"
                )
                self.last_split_admm_stats.update(extra_stats)
            return coeff
        if solver_mode != "stacked_tikhonov":
            raise ValueError(
                "Unsupported multi_angle_solver_mode="
                f"{solver_mode!r}; expected 'split_triangular_admm' or 'stacked_tikhonov'."
            )
        if b.dim() == 1:
            b = b.unsqueeze(0)
        self.last_split_admm_stats = None
        rhs = self.adjoint(b.to(dtype=torch.float32, device=device)).view(b.shape[0], self.N)
        eigvals, eigvecs = _ensure_implicit_gram_spectrum(self, self._morozov_cache_fingerprint())
        coeff = _solve_tikhonov_from_gram_spectrum(rhs, eigvals=eigvals, eigvecs=eigvecs, lambda_reg=lambda_reg)
        return coeff.to(device=device, dtype=torch.float32).view(-1, 1, self.height, self.width)

    @torch.no_grad()
    def solve_tikhonov_cg(
        self,
        b: torch.Tensor,
        lambda_reg: float | torch.Tensor,
        max_iter: int,
        tol: float = 1.0e-4,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=device)
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
        lambda_min: float = 1.0e-12,
        lambda_max: float = 1.0e12,
    ) -> torch.Tensor:
        solver_mode = str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "split_triangular_admm")).strip().lower()
        if b.dim() == 1:
            b = b.unsqueeze(0)
        if solver_mode == "split_triangular_admm":
            g_backbone, _ = self._split_backbone_extra(b)
            noise_norm = noise_norm.to(dtype=torch.float32, device=b.device).view(-1)
            scale = math.sqrt(float(self.num_backbone) / float(self.num_angles))
            noise_norm_backbone = noise_norm * float(scale)
            return self.backbone_operator.choose_lambda_morozov(
                g_backbone.reshape(g_backbone.shape[0], -1),
                noise_norm=noise_norm_backbone,
                tau=tau,
                max_iter=max_iter,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
            )
        b = b.to(dtype=torch.float32, device=device)
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


class RadonExample11Operator2D(torch.nn.Module):
    """
    Theorem-3.5 operator for the ill-posed system in Example 1.1:

      A = L P,   g = A c

    where:
      - L is lower-triangular Toeplitz as in Eq. (3.11),
      - P maps lexicographical coefficients c to d indexed by beta·k.
    """

    def __init__(
        self,
        beta=THEORETICAL_CONFIG["beta_vector"],
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        sampling_seed: Optional[int] = None,
        sampling_points: Optional[torch.Tensor] = None,
        num_detector_samples: Optional[int] = None,
    ):
        super().__init__()

        self.height = int(height)
        self.width = int(width)
        self.N = self.height * self.width

        if num_detector_samples is None:
            num_detector_samples = int(TIME_DOMAIN_CONFIG.get("num_detector_samples", self.N))
        self.M = int(num_detector_samples)

        if self.M != self.N:
            raise ValueError(
                "Theorem 3.5 lower-triangular form requires num_detector_samples == N "
                f"(got M={self.M}, N={self.N})."
            )

        scheme = str(TIME_DOMAIN_CONFIG.get("sampling_scheme", "paper_grid_t0")).strip().lower()
        if scheme != "paper_grid_t0":
            raise ValueError(
                f"sampling_scheme={scheme!r} is incompatible with Theorem-3.5 operator; use 'paper_grid_t0'."
            )
        t0 = float(TIME_DOMAIN_CONFIG.get("sampling_t0", 0.5))

        with torch.no_grad():
            blk = _theorem35_block(beta=beta, height=self.height, width=self.width, t0=t0)
            A_cpu = blk["A"]  # (N,N), float64
            A = A_cpu.to(dtype=torch.float32, device=device)

            sampling_points_theorem = blk["sampling_points"].to(dtype=torch.float64, device="cpu")
            if sampling_points is None:
                sampling_points = sampling_points_theorem
            else:
                sampling_points = sampling_points.to(dtype=torch.float64, device="cpu").view(-1)
                if int(sampling_points.numel()) != self.M:
                    raise ValueError(f"sampling_points must have length M={self.M}, got {sampling_points.numel()}")
                if not torch.allclose(sampling_points, sampling_points_theorem, atol=1e-9, rtol=0.0):
                    raise ValueError(
                        "Provided sampling_points do not match theorem grid X_j=(t0+j)/||beta|| required by Theorem 3.5."
                    )

            L = blk["L"].to(dtype=torch.float32, device=device)
            P = blk["P"].to(dtype=torch.float32, device=device)
            r = blk["r"].to(dtype=torch.float32, device=device)
            alpha = blk["alpha"].to(dtype=torch.float32, device=device)
            beta_i = blk["beta"].to(dtype=torch.int64, device=device)
            lex_to_d = blk["lex_to_d"].to(dtype=torch.int64, device=device)
            d_to_lex = blk["d_to_lex"].to(dtype=torch.int64, device=device)
            kappa0 = blk["kappa0"].to(dtype=torch.int64, device=device)
            kappa_m = blk["kappa_m"].to(dtype=torch.int64, device=device)

        # Precompute a thin SVD of A (on CPU, float64) to support Morozov discrepancy principle
        # for choosing the Tikhonov regularization parameter lambda.
        #
        # We only need U and singular values s to evaluate the residual norm
        #   ||(I - A(A^T A + lambda I)^{-1}A^T) b||_2
        # efficiently for many right-hand sides b.
        with torch.no_grad():
            U_cpu, s_cpu, _ = torch.linalg.svd(A_cpu, full_matrices=False)  # U:(M,K), s:(K,)
            U = U_cpu.to(dtype=torch.float32, device=device)
            s = s_cpu.to(dtype=torch.float32, device=device)

        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta_i)
        self.register_buffer("sampling_points", sampling_points.to(dtype=torch.float32, device=device))
        self.register_buffer("A_lower", L)
        self.register_buffer("A_perm", P)
        self.register_buffer("r_vector", r)
        self.register_buffer("lex_to_d_indices", lex_to_d)
        self.register_buffer("d_to_lex_indices", d_to_lex)
        self.register_buffer("kappa0", kappa0)
        self.register_buffer("kappa_m", kappa_m)
        self.register_buffer("A", A)
        self.register_buffer("svd_U", U)
        self.register_buffer("svd_s", s)

    def forward(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(1)
        b = coeff_matrix.shape[0]
        d = coeff_matrix.view(b, -1)  # (B,N) in lexicographical order
        g = d @ self.A.t()  # (B,M)
        return g

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.dim() == 3 and residual.shape[1] == 1:
            residual = residual.squeeze(1)
        b = residual.shape[0]
        d_grad = residual @ self.A  # (B,N)
        return d_grad.view(b, 1, self.height, self.width)

    @torch.no_grad()
    def choose_lambda_morozov(
        self,
        b: torch.Tensor,
        noise_norm: torch.Tensor,
        tau: float = 1.0,
        max_iter: int = 40,
        lambda_min: float = 1e-12,
        lambda_max: float = 1e12,
    ) -> torch.Tensor:
        """
        Choose the Tikhonov regularization parameter lambda by Morozov discrepancy principle:

            ||A x_lambda - b||_2 = tau * ||noise||_2,

        where x_lambda solves (A^T A + lambda I) x = A^T b.

        Args:
            b: (B,M) or (M,) observed right-hand side.
            noise_norm: (B,) or scalar, ||b - b_clean||_2 estimated from the noise model.
            tau: safety factor (>1 in some literature; paper does not specify, default 1.0).
            max_iter: bisection iterations.
        Returns:
            lambda: (B,) tensor on the same device as b.
        """
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.A.device)
        settings = _morozov_settings(max_iter=max_iter, lambda_min=lambda_min, lambda_max=lambda_max)
        return _choose_lambda_morozov_from_explicit_svd(
            b=b,
            noise_norm=noise_norm.to(dtype=torch.float32, device=b.device),
            U=self.svd_U,
            s=self.svd_s,
            tau=float(tau),
            settings=settings,
        )


class MultiAngleRadonOperator2D(torch.nn.Module):
    """
    Stacked multi-angle operator with per-angle Theorem-3.5 blocks:

        g_stack = [A_1; ...; A_K] c,   A_i = L_i P_i
    """

    def __init__(
        self,
        beta_vectors: list[tuple[int, int]],
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        sampling_seed: Optional[int] = None,
        sampling_points: Optional[torch.Tensor] = None,
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
            num_detector_samples_per_angle = int(TIME_DOMAIN_CONFIG.get("num_detector_samples", self.N))
        self.M_per_angle = int(num_detector_samples_per_angle)
        self.M = int(self.num_angles * self.M_per_angle)
        if self.M_per_angle != self.N:
            raise ValueError(
                "Theorem 3.5 per-angle lower-triangular form requires num_detector_samples_per_angle == N "
                f"(got M_per_angle={self.M_per_angle}, N={self.N})."
            )

        scheme = str(TIME_DOMAIN_CONFIG.get("sampling_scheme", "paper_grid_t0")).strip().lower()
        if scheme != "paper_grid_t0":
            raise ValueError(
                f"sampling_scheme={scheme!r} is incompatible with Theorem-3.5 operator; use 'paper_grid_t0'."
            )
        t0 = float(TIME_DOMAIN_CONFIG.get("sampling_t0", 0.5))

        with torch.no_grad():
            blocks = [
                _theorem35_block(beta=beta, height=self.height, width=self.width, t0=t0)
                for beta in self.beta_vectors
            ]

            A_blocks_cpu_t = torch.stack([blk["A"] for blk in blocks], dim=0)  # (K,N,N), float64
            A_cpu = A_blocks_cpu_t.reshape(self.M, self.N)  # (M,N), float64
            A_blocks = A_blocks_cpu_t.to(dtype=torch.float32, device=device)
            A = A_blocks.reshape(self.M, self.N)

            L_blocks = torch.stack([blk["L"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            P_blocks = torch.stack([blk["P"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            r_blocks = torch.stack([blk["r"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            alphas = torch.stack([blk["alpha"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            betas = torch.stack([blk["beta"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            lex_to_d = torch.stack([blk["lex_to_d"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            d_to_lex = torch.stack([blk["d_to_lex"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            kappa0 = torch.stack([blk["kappa0"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            kappa_m = torch.stack([blk["kappa_m"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)

            sampling_points_theorem = torch.stack([blk["sampling_points"] for blk in blocks], dim=0).to(
                dtype=torch.float64, device="cpu"
            )  # (K,N)

        # Resolve sampling points per angle: shape (K, M_per_angle) on CPU float64.
        sampling_points_pa: torch.Tensor
        if sampling_points is None:
            sampling_points_pa = sampling_points_theorem
        else:
            sp = sampling_points.to(dtype=torch.float64, device="cpu")
            if sp.dim() == 1:
                if int(sp.numel()) != self.M:
                    raise ValueError(f"sampling_points must have length M={self.M}, got {sp.numel()}")
                sampling_points_pa = sp.view(self.num_angles, self.M_per_angle)
            elif sp.dim() == 2:
                expected = (self.num_angles, self.M_per_angle)
                if tuple(sp.shape) != expected:
                    raise ValueError(f"sampling_points must have shape {expected}, got {tuple(sp.shape)}")
                sampling_points_pa = sp
            else:
                raise ValueError(
                    f"sampling_points must be 1D (flattened) or 2D (per-angle), got {tuple(sp.shape)}"
                )
            if not torch.allclose(sampling_points_pa, sampling_points_theorem, atol=1e-9, rtol=0.0):
                raise ValueError(
                    "Provided sampling_points do not match theorem grid X_j=(t0+j)/||beta|| for all angles."
                )

        sampling_points_total = sampling_points_pa.reshape(-1)  # (M,)

        # Precompute thin SVD of A (on CPU, float64) for Morozov discrepancy principle.
        with torch.no_grad():
            U_cpu, s_cpu, _ = torch.linalg.svd(A_cpu, full_matrices=False)  # U:(M,K), s:(K,)
            U = U_cpu.to(dtype=torch.float32, device=device)
            s = s_cpu.to(dtype=torch.float32, device=device)

        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        self.register_buffer("sampling_points", sampling_points_total.to(dtype=torch.float32, device=device))
        self.register_buffer("sampling_points_per_angle", sampling_points_pa.to(dtype=torch.float32, device=device))
        self.register_buffer("A_lower_blocks", L_blocks)  # (K,N,N)
        self.register_buffer("A_perm_blocks", P_blocks)  # (K,N,N)
        self.register_buffer("r_vectors", r_blocks)  # (K,N)
        self.register_buffer("lex_to_d_indices", lex_to_d)  # (K,N)
        self.register_buffer("d_to_lex_indices", d_to_lex)  # (K,N)
        self.register_buffer("kappa0", kappa0)  # (K,)
        self.register_buffer("kappa_m", kappa_m)  # (K,)
        self.register_buffer("A_blocks", A_blocks)  # (K,M_per_angle,N)
        self.register_buffer("A", A)
        self.register_buffer("svd_U", U)
        self.register_buffer("svd_s", s)

    def forward(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(1)
        b = coeff_matrix.shape[0]
        d = coeff_matrix.view(b, -1)  # (B,N)
        return d @ self.A.t()  # (B,M)

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.dim() == 3 and residual.shape[1] == 1:
            residual = residual.squeeze(1)
        b = residual.shape[0]
        d_grad = residual @ self.A  # (B,N)
        return d_grad.view(b, 1, self.height, self.width)

    def split_measurements(self, g: torch.Tensor) -> torch.Tensor:
        """Reshape stacked measurements (B,M) into per-angle blocks (B,K,M_per_angle)."""
        if g.dim() == 3 and g.shape[1] == 1:
            g = g.squeeze(1)
        if g.dim() != 2:
            raise ValueError(f"Expected g with shape (B,M), got {tuple(g.shape)}")
        if int(g.shape[1]) != int(self.M):
            raise ValueError(f"Expected measurement length M={self.M}, got {g.shape[1]}")
        return g.view(g.shape[0], self.num_angles, self.M_per_angle)

    def forward_per_angle(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        """Per-angle forward projections, returns (B,K,M_per_angle)."""
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(1)
        b = coeff_matrix.shape[0]
        d = coeff_matrix.view(b, -1)  # (B,N)
        return torch.einsum("bn,kmn->bkm", d, self.A_blocks)

    def adjoint_per_angle(self, residual_per_angle: torch.Tensor) -> torch.Tensor:
        """Per-angle adjoint, returns (B,K,1,H,W)."""
        if residual_per_angle.dim() == 4 and residual_per_angle.shape[2] == 1:
            residual_per_angle = residual_per_angle.squeeze(2)
        if residual_per_angle.dim() != 3:
            raise ValueError(
                f"Expected residual_per_angle with shape (B,K,M_per_angle), got {tuple(residual_per_angle.shape)}"
            )
        if int(residual_per_angle.shape[1]) != int(self.num_angles):
            raise ValueError(f"Expected K={self.num_angles}, got {residual_per_angle.shape[1]}")
        if int(residual_per_angle.shape[2]) != int(self.M_per_angle):
            raise ValueError(
                f"Expected M_per_angle={self.M_per_angle}, got {residual_per_angle.shape[2]}"
            )
        d_grad = torch.einsum("bkm,kmn->bkn", residual_per_angle, self.A_blocks)  # (B,K,N)
        return d_grad.view(-1, self.num_angles, 1, self.height, self.width)

    @torch.no_grad()
    def choose_lambda_morozov(
        self,
        b: torch.Tensor,
        noise_norm: torch.Tensor,
        tau: float = 1.0,
        max_iter: int = 40,
        lambda_min: float = 1e-12,
        lambda_max: float = 1e12,
    ) -> torch.Tensor:
        """See RadonExample11Operator2D.choose_lambda_morozov for details."""
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.A.device)
        settings = _morozov_settings(max_iter=max_iter, lambda_min=lambda_min, lambda_max=lambda_max)
        return _choose_lambda_morozov_from_explicit_svd(
            b=b,
            noise_norm=noise_norm.to(dtype=torch.float32, device=b.device),
            U=self.svd_U,
            s=self.svd_s,
            tau=float(tau),
            settings=settings,
        )


def build_time_domain_operator(
    beta=THEORETICAL_CONFIG["beta_vector"],
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> torch.nn.Module:
    """Factory for the active forward/adjoint operator (implicit pixel or legacy dense)."""
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
        formula_mode = _formula_mode_from_solver_mode(solver_mode)
        auto_shift_t0 = bool(TIME_DOMAIN_CONFIG.get("auto_angle_t0", True))
        if use_multi:
            _validate_multi_angle_backbone(
                backbone_betas,
                total_angles=total_angles,
            )
        if use_multi and total_angles > len(backbone_betas):
            extra_betas = _sample_uniform_extra_beta_vectors(
                backbone_betas=backbone_betas,
                extra_count=int(total_angles - len(backbone_betas)),
                height=int(height),
                width=int(width),
                seed=int(TIME_DOMAIN_CONFIG.get("extra_angle_seed", 20260322)),
            )
            return StructuredMultiAngleB1B1Operator2D(
                backbone_beta_vectors=backbone_betas,
                extra_beta_vectors=extra_betas,
                height=int(height),
                width=int(width),
                t0=t0,
                formula_mode=formula_mode,
                auto_shift_t0=auto_shift_t0,
            ).to(device)
        active_betas = backbone_betas if use_multi else backbone_betas
        return TheoreticalB1B1Operator2D(
            beta_vectors=active_betas,
            height=int(height),
            width=int(width),
            t0=t0,
            formula_mode=formula_mode,
            auto_shift_t0=auto_shift_t0,
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


class TheoreticalDataGenerator:
    """
    Time-domain data generator for the spline model.

    Ground truth is a coefficient field c on the integer lattice, and the measurement
    is g = A(c) (real).

    Init (feeds the learned iterative optimizer):
    - "cg": solve (A^T A + lambda I)c0 = A^T g approximately with a few CG iterations
    - "tikhonov_direct": solve (A^T A + lambda I)c0 = A^T g directly
    """

    def __init__(self, data_source: Optional[str] = None):
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

        self.time_operator = build_time_domain_operator(
            beta=THEORETICAL_CONFIG["beta_vector"],
            height=self.img_size,
            width=self.img_size,
        )
        self.M = int(getattr(self.time_operator, "M", int(TIME_DOMAIN_CONFIG.get("num_detector_samples", self.N))))
        self.flatten_order = getattr(self.time_operator, "flatten_order", None)
        self.last_lambda: Optional[float | torch.Tensor] = None
        self._chol_lambda: Optional[float] = None
        self._chol_factor: Optional[torch.Tensor] = None
        self._ata_factor: Optional[torch.Tensor] = None

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

    def adjoint_operator(self, residual: torch.Tensor) -> torch.Tensor:
        """Adjoint operator, returns A^T r (real)."""
        return self.time_operator.adjoint(residual)

    @torch.no_grad()
    def _resolve_split_admm_solver_kwargs(self) -> dict[str, float | int]:
        solver_mode = str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "split_triangular_admm")).strip().lower()
        if solver_mode == "stacked_tikhonov":
            return {}
        if solver_mode != "split_triangular_admm":
            raise ValueError(
                "Unsupported multi_angle_solver_mode="
                f"{solver_mode!r}; expected 'split_triangular_admm' or 'stacked_tikhonov'."
            )

        return {
            "rho": float(TIME_DOMAIN_CONFIG.get("split_admm_rho", 1.0)),
            "max_iter": int(TIME_DOMAIN_CONFIG.get("split_admm_max_iter", 200)),
            "tol": float(TIME_DOMAIN_CONFIG.get("split_admm_tol", 1.0e-4)),
        }

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

        solver_kwargs = self._resolve_split_admm_solver_kwargs()
        if hasattr(self.time_operator, "solve_tikhonov_direct") and not hasattr(self.time_operator, "A"):
            return self.time_operator.solve_tikhonov_direct(g_obs, lambda_reg=lam_batch, **solver_kwargs)

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
            g_clean = self.forward_operator(coeff_true).to(torch.float32)
            g_observed = self._apply_noise(g_clean)

            observed = g_observed

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
                noise_norm = torch.norm(observed - g_clean, dim=-1)  # (B,)
                lam = self.time_operator.choose_lambda_morozov(
                    observed,
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
            coeff_initial = self._tikhonov_direct_init(observed, lambda_reg=lambda_eff)
        elif init_method == "cg" and init_cg_iters > 0:
            coeff_initial = self._tikhonov_cg_init(
                observed,
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

        # 1. Sample coefficients: (B,1,H,W)
        coeff_true = self._sample_coefficients(batch_size)

        # 2. Image from coefficients (batch-capable)
        f_true = self.image_gen(coeff_true)  # (B,1,H,W)

        with torch.no_grad():
            # 3. Forward operator: g_clean = A c, (B,M)
            g_clean = self.forward_operator(coeff_true).to(torch.float32)

            # 4. Add noise
            g_observed = self._apply_noise(g_clean)

        # 5. Coefficient init
        if lambda_reg is not None:
            lambda_eff = self._normalize_lambda_reg(
                lambda_reg,
                batch_size=int(batch_size),
                dtype=torch.float32,
                target_device=g_observed.device,
            )
        else:
            mode = str(DATA_CONFIG.get("lambda_select_mode", "fixed")).strip().lower()
            if mode == "morozov":
                tau = float(DATA_CONFIG.get("morozov_tau", 1.0))
                max_iter = int(DATA_CONFIG.get("morozov_max_iter", 40))
                lam_min = float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12))
                lam_max = float(DATA_CONFIG.get("morozov_lambda_max", 1.0e12))
                noise_norm = torch.norm(g_observed - g_clean, dim=-1)
                lam = self.time_operator.choose_lambda_morozov(
                    g_observed,
                    noise_norm=noise_norm,
                    tau=tau,
                    max_iter=max_iter,
                    lambda_min=lam_min,
                    lambda_max=lam_max,
                )
                lambda_eff = lam.to(dtype=torch.float32, device=g_observed.device)
            else:
                lambda_eff = float(DATA_CONFIG.get("lambda_reg", 1e-2))
        self.last_lambda = lambda_eff

        init_method = str(TIME_DOMAIN_CONFIG.get("init_method", "cg")).strip().lower()
        init_cg_iters = int(TIME_DOMAIN_CONFIG.get("init_cg_iters", 0))

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

        # coeff_true: (B,1,H,W), f_true: (B,1,H,W), g_observed: (B,M), coeff_initial: (B,1,H,W)
        return coeff_true, f_true, g_observed, coeff_initial
