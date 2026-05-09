"""Alpha-continuous time-domain Radon operator and data generator.

For each continuous angle alpha in [0, pi), this module sorts lattice
projections

    s_k(alpha) = k1*cos(alpha) + k2*sin(alpha)

and samples at

    t_i = s_(i) + tau.

The single-angle matrix is

    A_alpha_tau[i,j] = R_alpha phi(s_(i) + tau - s_(j)).

Multiple angles are stacked vertically and solved with Tikhonov / Morozov.
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

from config import DATA_CONFIG, IMAGE_SIZE, TIME_DOMAIN_CONFIG, device
from image_generator import (
    DifferentiableImageGenerator,
    generate_random_ellipse_phantom,
    generate_shepp_logan_phantom,
)
from b_spline.b2b1_spline import (
    phi_support_bounds_b1b1,
    radon_phi_b1b1,
)


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


def _sparse_blocks_apply_batched(rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, nnz: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply per-angle sparse projection-order matrices stored as padded COO buffers."""
    if x.dim() != 3:
        raise ValueError(f"x must have shape (B,K,N), got {tuple(x.shape)}")
    batch, num_angles, n = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
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


def _sparse_blocks_adjoint_apply_batched(rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, nnz: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply adjoints of per-angle sparse projection-order matrices stored as padded COO buffers."""
    if x.dim() != 3:
        raise ValueError(f"x must have shape (B,K,N), got {tuple(x.shape)}")
    batch, num_angles, n = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
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


def _build_sparse_b1b1_block_from_continuous_proj(
    *,
    sorted_proj: torch.Tensor,
    direction: torch.Tensor,
    tau: float,
    value_tol: float = 1.0e-15,
) -> dict[str, torch.Tensor]:
    """Build the full sparse projection-order block for alpha-continuous sampling."""
    sorted_proj = sorted_proj.detach().to(dtype=torch.float64, device="cpu")
    direction = direction.detach().to(dtype=torch.float64, device="cpu")
    tau = float(tau)
    proj_np = sorted_proj.numpy()
    n = int(proj_np.shape[0])
    support_lo, support_hi = phi_support_bounds_b1b1(direction)
    support_lo = float(support_lo)
    support_hi = float(support_hi)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    lower_width = 0
    upper_width = 0

    for row_idx in range(n):
        t_i = float(proj_np[row_idx] + tau)
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

    diag0 = float(radon_phi_b1b1(torch.tensor([tau], dtype=torch.float64), direction)[0].item())
    return {
        "sparse_rows": torch.from_numpy(rows_np),
        "sparse_cols": torch.from_numpy(cols_np),
        "sparse_values": torch.from_numpy(vals_np),
        "sparse_nnz": torch.tensor(int(vals_np.shape[0]), dtype=torch.int64),
        "lower_width": torch.tensor(int(lower_width), dtype=torch.int64),
        "upper_width": torch.tensor(int(upper_width), dtype=torch.int64),
        "diag0": torch.tensor(float(diag0), dtype=torch.float64),
        "support_lo": torch.tensor(float(support_lo), dtype=torch.float64),
        "support_hi": torch.tensor(float(support_hi), dtype=torch.float64),
    }


class AlphaContinuousB1B1Operator2D(torch.nn.Module):
    """Continuous-alpha B1*B1 operator with one full sparse block per angle."""

    def __init__(
        self,
        alpha_values,
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        tau_offsets=None,
        t0: float = 0.5,
        injective_tol: float = 1.0e-12,
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.N = int(self.height * self.width)
        self.alpha_values = [float(v) % math.pi for v in list(alpha_values or [])]
        if not self.alpha_values:
            raise ValueError("alpha_values must be a non-empty list of angles in radians.")
        self.num_angles = int(len(self.alpha_values))
        self.M_per_angle = int(self.N)
        self.M = int(self.num_angles * self.M_per_angle)
        self.t0 = float(t0)
        self.formula_mode = "alpha_continuous"
        self.uses_sparse_blocks = True
        self._gram_cache_dir_override = str(
            DATA_CONFIG.get(
                "alpha_gram_cache_dir",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "alpha_gram_cache"),
            )
        )
        if tau_offsets is None:
            tau_list = None
        else:
            tau_list = [float(v) for v in list(tau_offsets)]
            if len(tau_list) != self.num_angles:
                raise ValueError(f"tau_offsets length={len(tau_list)} but num_angles={self.num_angles}.")
        self.tau_offsets = tau_list

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

            for angle_idx, alpha in enumerate(self.alpha_values):
                info = _alpha_projection_order(alpha, self.height, self.width, injective_tol=float(injective_tol))
                direction = info["direction"]
                support_lo, support_hi = phi_support_bounds_b1b1(direction)
                tau = float(support_lo) + float(t0) * (float(support_hi) - float(support_lo)) if tau_list is None else float(tau_list[angle_idx])
                block = _build_sparse_b1b1_block_from_continuous_proj(sorted_proj=info["sorted_proj"], direction=direction, tau=tau)
                directions.append(direction)
                sorted_proj_list.append(info["sorted_proj"])
                sampling_points_list.append(info["sorted_proj"] + float(tau))
                lex_to_order_list.append(info["lex_to_order"])
                order_to_lex_list.append(info["order_to_lex"])
                min_gap_list.append(info["min_gap"])
                support_lo_list.append(torch.tensor(float(support_lo), dtype=torch.float64))
                support_hi_list.append(torch.tensor(float(support_hi), dtype=torch.float64))
                blocks.append(block)
                effective_tau.append(float(tau))

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

    def _morozov_cache_fingerprint(self) -> dict[str, object]:
        return {
            "class_name": self.__class__.__name__,
            "height": int(self.height),
            "width": int(self.width),
            "num_angles": int(self.num_angles),
            "alpha_values": [round(float(v), 15) for v in self.alpha_values],
            "tau_offsets": [round(float(v), 15) for v in self.tau_offsets_tensor.detach().cpu().tolist()],
            "sparse_nnz_per_angle": [int(v.item()) for v in self.sparse_nnz],
            "formula_mode": "alpha_continuous",
            "basis": "b1b1",
            "implementation_version": "alpha_continuous_full_sparse_v2",
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
        return _sparse_blocks_apply_batched(self.sparse_rows, self.sparse_cols, self.sparse_values, self.sparse_nnz, ordered)

    def forward(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.forward_per_angle(coeff_matrix).reshape(coeff_matrix.shape[0], self.M)

    def adjoint_per_angle(self, residual_per_angle: torch.Tensor) -> torch.Tensor:
        if residual_per_angle.dim() == 4 and residual_per_angle.shape[2] == 1:
            residual_per_angle = residual_per_angle.squeeze(2)
        if residual_per_angle.dim() != 3:
            raise ValueError(f"Expected residual_per_angle with shape (B,K,M_per_angle), got {tuple(residual_per_angle.shape)}")
        residual_per_angle = residual_per_angle.to(dtype=torch.float32, device=self.sampling_points.device)
        batch = int(residual_per_angle.shape[0])
        grad_ordered = _sparse_blocks_adjoint_apply_batched(self.sparse_rows, self.sparse_cols, self.sparse_values, self.sparse_nnz, residual_per_angle)
        gather_index = self.lex_to_order_indices.view(1, self.num_angles, self.N).expand(batch, -1, -1)
        grad_lex = grad_ordered.gather(2, gather_index)
        return grad_lex.view(batch, self.num_angles, 1, self.height, self.width)

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        return self.adjoint_per_angle(self.split_measurements(residual)).sum(dim=1)

    def apply_normal(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(coeff_matrix))

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
        )
        grad_ordered = _sparse_blocks_adjoint_apply_batched(
            self.sparse_rows,
            self.sparse_cols,
            self.sparse_values,
            self.sparse_nnz,
            measurement_pa,
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


def build_time_domain_operator(height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> torch.nn.Module:
    """Build the single retained alpha-continuous operator."""
    alpha_values = TIME_DOMAIN_CONFIG.get("alpha_values") or []
    tau_offsets = TIME_DOMAIN_CONFIG.get("alpha_tau_offsets") or []
    if not alpha_values or not tau_offsets:
        raise ValueError("alpha_continuous operator requires TIME_DOMAIN_CONFIG['alpha_values'] and TIME_DOMAIN_CONFIG['alpha_tau_offsets'].")
    if len(alpha_values) != len(tau_offsets):
        raise ValueError(f"alpha_values and alpha_tau_offsets length mismatch: {len(alpha_values)} vs {len(tau_offsets)}.")
    return AlphaContinuousB1B1Operator2D(alpha_values=alpha_values, tau_offsets=tau_offsets, height=int(height), width=int(width)).to(device)


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
        self.feature_time_operator = None
        self.M = int(getattr(self.time_operator, "M", int(TIME_DOMAIN_CONFIG.get("num_detector_samples", self.N))))
        self.last_lambda: Optional[float | torch.Tensor] = None
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

    def forward_operator(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.time_operator.forward(coeff_matrix)

    def data_forward_operator(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        return self.data_time_operator.forward(coeff_matrix)

    def adjoint_operator(self, residual: torch.Tensor) -> torch.Tensor:
        return self.time_operator.adjoint(residual)

    @torch.no_grad()
    def solve_tikhonov_direct_init(self, g_obs: torch.Tensor, lambda_reg: float | torch.Tensor) -> torch.Tensor:
        return self._tikhonov_direct_init(g_obs, lambda_reg=lambda_reg)

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

    def _select_lambda(self, g_observed: torch.Tensor, g_clean: torch.Tensor, lambda_reg: float | torch.Tensor = None) -> float | torch.Tensor:
        if lambda_reg is not None:
            if torch.is_tensor(lambda_reg):
                return lambda_reg.to(dtype=torch.float32, device=g_observed.device)
            return float(lambda_reg)
        mode = str(DATA_CONFIG.get("lambda_select_mode", "fixed")).strip().lower()
        if mode == "morozov":
            noise_norm = torch.norm(g_observed - g_clean, dim=-1)
            return self.time_operator.choose_lambda_morozov(
                g_observed,
                noise_norm=noise_norm,
                tau=float(DATA_CONFIG.get("morozov_tau", 1.0)),
                max_iter=int(DATA_CONFIG.get("morozov_max_iter", 8)),
                lambda_min=float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12)),
                lambda_max=float(DATA_CONFIG.get("morozov_lambda_max", 1.0e12)),
            ).to(dtype=torch.float32, device=g_observed.device)
        return float(DATA_CONFIG.get("lambda_reg", 1e-2))

    def generate_training_sample(self, random_seed=None, lambda_reg: float | torch.Tensor = None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        coeff_true = self._sample_coefficients()
        f_true = self.image_gen(coeff_true).squeeze(0)
        with torch.no_grad():
            g_clean = self.data_forward_operator(coeff_true).to(torch.float32)
            g_observed = self._apply_noise(g_clean)
        lambda_eff = self._select_lambda(g_observed, g_clean, lambda_reg=lambda_reg)
        self.last_lambda = lambda_eff
        init_method = str(TIME_DOMAIN_CONFIG.get("init_method", "cg")).strip().lower()
        init_cg_iters = int(TIME_DOMAIN_CONFIG.get("init_cg_iters", 0))
        if init_method == "tikhonov_direct":
            coeff_initial = self._tikhonov_direct_init(g_observed, lambda_reg=lambda_eff)
        elif init_method == "cg" and init_cg_iters > 0:
            coeff_initial = self._tikhonov_cg_init(g_observed, lambda_reg=lambda_eff, max_iter=init_cg_iters)
        else:
            raise ValueError(f"Unsupported init_method={init_method!r}; expected 'cg' or 'tikhonov_direct'.")
        return coeff_true.squeeze(0).squeeze(0), f_true.squeeze(0), g_observed.squeeze(0), coeff_initial.squeeze(0).squeeze(0)

    def generate_batch(self, batch_size, random_seed=None, lambda_reg: float | torch.Tensor = None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        batch_started = time.perf_counter()
        init_method = str(TIME_DOMAIN_CONFIG.get("init_method", "cg")).strip().lower()
        lambda_mode = "provided" if lambda_reg is not None else str(DATA_CONFIG.get("lambda_select_mode", "fixed")).strip().lower()
        progress_enabled = (not self._first_batch_progress_logged) and init_method == "tikhonov_direct"
        if progress_enabled:
            print(
                "[init] first batch start "
                f"batch_size={int(batch_size)} angles={int(getattr(self.time_operator, 'num_angles', 1) or 1)} "
                f"lambda_mode={lambda_mode} init_method={init_method} solver=stacked_tikhonov"
            )
        coeff_true = self._sample_coefficients(batch_size)
        f_true = self.image_gen(coeff_true)
        with torch.no_grad():
            g_clean = self.data_forward_operator(coeff_true).to(torch.float32)
            g_observed = self._apply_noise(g_clean)
        lambda_eff = self._select_lambda(g_observed, g_clean, lambda_reg=lambda_reg)
        self.last_lambda = lambda_eff
        init_cg_iters = int(TIME_DOMAIN_CONFIG.get("init_cg_iters", 0))
        coeff_init_started = time.perf_counter()
        if init_method == "tikhonov_direct":
            coeff_initial = self._tikhonov_direct_init(g_observed, lambda_reg=lambda_eff)
        elif init_method == "cg" and init_cg_iters > 0:
            coeff_initial = self._tikhonov_cg_init(g_observed, lambda_reg=lambda_eff, max_iter=init_cg_iters)
        else:
            raise ValueError(f"Unsupported init_method={init_method!r}; expected 'cg' or 'tikhonov_direct'.")
        if progress_enabled:
            print(f"[init] coefficient init finished in {time.perf_counter() - coeff_init_started:.2f}s")
            print(f"[init] first batch ready in {time.perf_counter() - batch_started:.2f}s")
            self._first_batch_progress_logged = True
        return coeff_true, f_true, g_observed, coeff_initial
