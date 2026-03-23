"""Pure-Tikhonov comparison entrypoint: paper B2*B1 vs current B1*B1."""

from __future__ import annotations

import argparse
import math
import os
import sys
from contextlib import contextmanager
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from b_spline.b2b1_spline import radon_phi_b1b1, synthesize_f_from_coeff_b2b1
from config import DATA_CONFIG, IMAGE_SIZE, RESULTS_DIR, TIME_DOMAIN_CONFIG, device
from image_generator import generate_random_ellipse_phantom
from radon_transform import (
    ImplicitPixelRadonOperator2D,
    MultiAngleRadonOperator2D,
    RadonExample11Operator2D,
    TheoreticalB1B1Operator2D as RuntimeTheoreticalB1B1Operator2D,
)

matplotlib.use("Agg")

PAPER_COEFF_SIZE = 21
PAPER_BETA = (1, 21)
PAPER_T0 = 0.5
DEFAULT_COMPARE_MODE = "both"
DISPLAY_SIZE = 256
PAPER_COEFF_NPY_PATH = os.path.join(PROJECT_ROOT, "compared", "paper_coeff_fig5_2A.npy")
PAPER_FIXED_SEED = 0


def _coeff_res(coeff_est: torch.Tensor, coeff_true: torch.Tensor) -> float:
    coeff_est = coeff_est.to(dtype=torch.float32)
    coeff_true = coeff_true.to(dtype=torch.float32, device=coeff_est.device)
    diff_norm = torch.norm(coeff_est - coeff_true)
    true_norm = torch.norm(coeff_true).clamp_min(1e-12)
    return float((diff_norm / true_norm).item())


def _to_integer_beta(beta) -> torch.Tensor:
    beta_t = torch.as_tensor(beta, dtype=torch.float64).view(-1)
    if int(beta_t.numel()) != 2:
        raise ValueError(f"beta must contain exactly 2 entries, got {tuple(beta_t.tolist())}")
    beta_round = torch.round(beta_t)
    if torch.max(torch.abs(beta_t - beta_round)).item() > 1e-9:
        raise ValueError(f"beta must be integer-valued, got {tuple(beta_t.tolist())}")
    beta_i = beta_round.to(torch.int64)
    if int(torch.sum(beta_i.abs()).item()) == 0:
        raise ValueError("beta must be non-zero.")
    return beta_i


def _lex_lattice_indices(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    k1 = torch.arange(int(height), dtype=torch.int64).repeat_interleave(int(width))
    k2 = torch.arange(int(width), dtype=torch.int64).repeat(int(height))
    return k1, k2


def _build_current_b1b1_theory_block(
    beta,
    height: int,
    width: int,
    t0: float,
) -> dict[str, torch.Tensor]:
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
        raise ValueError("range(beta·k) must be consecutive integers for the theoretical B1*B1 operator.")

    lex_to_d = (beta_dot_k - kappa0).to(torch.int64)
    d_to_lex = torch.empty(n, dtype=torch.int64)
    d_to_lex[lex_to_d] = torch.arange(n, dtype=torch.int64)

    k = torch.arange(n, dtype=torch.float64)
    sampling_points = (float(t0) + k) / beta_norm
    r = radon_phi_b1b1((float(t0) + k - float(kappa0)) / beta_norm, alpha).to(torch.float64)
    return {
        "r": r,
        "alpha": alpha,
        "beta": beta_i,
        "sampling_points": sampling_points,
        "lex_to_d": lex_to_d,
        "d_to_lex": d_to_lex,
    }


class TheoreticalB1B1Operator2D(torch.nn.Module):
    """Theoretical B1*B1 operator with A_i = L_i P_i, applied implicitly via Toeplitz convolution."""

    def __init__(
        self,
        beta_vectors: Iterable[Tuple[int, int]],
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        t0: float = PAPER_T0,
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.N = int(self.height * self.width)
        self.beta_vectors = [tuple(int(vv) for vv in beta) for beta in beta_vectors]
        if not self.beta_vectors:
            raise ValueError("beta_vectors must be a non-empty list.")
        self.num_angles = int(len(self.beta_vectors))
        self.M_per_angle = int(self.N)
        self.M = int(self.num_angles * self.M_per_angle)
        self.t0 = float(t0)

        with torch.no_grad():
            blocks = [
                _build_current_b1b1_theory_block(beta=beta, height=self.height, width=self.width, t0=self.t0)
                for beta in self.beta_vectors
            ]
            r_vectors = torch.stack([blk["r"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            alphas = torch.stack([blk["alpha"] for blk in blocks], dim=0).to(dtype=torch.float32, device=device)
            betas = torch.stack([blk["beta"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            lex_to_d = torch.stack([blk["lex_to_d"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            d_to_lex = torch.stack([blk["d_to_lex"] for blk in blocks], dim=0).to(dtype=torch.int64, device=device)
            sampling_points_pa = torch.stack([blk["sampling_points"] for blk in blocks], dim=0).to(
                dtype=torch.float32, device=device
            )

        conv_size = 1 << (int(2 * self.N - 1).bit_length())
        r_fft = torch.fft.rfft(r_vectors.to(torch.float32), n=conv_size, dim=1)

        self.register_buffer("r_vectors", r_vectors)
        self.register_buffer("r_fft", r_fft)
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        self.register_buffer("lex_to_d_indices", lex_to_d)
        self.register_buffer("d_to_lex_indices", d_to_lex)
        self.register_buffer("sampling_points_per_angle", sampling_points_pa)
        self.register_buffer("sampling_points", sampling_points_pa.reshape(-1))
        self.conv_size = int(conv_size)

    def _toeplitz_apply(self, r_fft: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_full = torch.fft.irfft(torch.fft.rfft(x.to(torch.float32), n=self.conv_size, dim=1) * r_fft.unsqueeze(0), n=self.conv_size, dim=1)
        return y_full[:, : self.N]

    def _toeplitz_adjoint_apply(self, r_fft: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        rev = torch.flip(x, dims=[1])
        rev_y = self._toeplitz_apply(r_fft, rev)
        return torch.flip(rev_y, dims=[1])

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
    def solve_tikhonov_cg(
        self,
        b: torch.Tensor,
        lambda_reg: float,
        max_iter: int,
        tol: float = 1e-4,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.r_vectors.device)
        rhs = self.adjoint(b)
        if x0 is None:
            x = torch.zeros_like(rhs)
        else:
            x = x0.to(dtype=torch.float32, device=rhs.device).clone()

        lam = float(lambda_reg)
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
        lambda_min: float = 1e-6,
        lambda_max: float = 1e2,
    ) -> torch.Tensor:
        if b.dim() == 1:
            b = b.unsqueeze(0)
        b = b.to(dtype=torch.float32, device=self.r_vectors.device)
        batch = int(b.shape[0])

        if noise_norm.dim() == 0:
            noise_norm = noise_norm.expand(batch)
        noise_norm = noise_norm.to(dtype=torch.float32, device=b.device).view(batch)
        target = float((float(tau) * noise_norm).mean().item())
        cg_iters = int(DATA_CONFIG.get("morozov_cg_iters", 12))
        cg_tol = float(DATA_CONFIG.get("morozov_cg_tol", 1e-4))

        def _residual_mean(lam_value: float) -> float:
            x = self.solve_tikhonov_cg(b, lambda_reg=lam_value, max_iter=cg_iters, tol=cg_tol)
            residual = self.forward(x) - b
            return float(torch.norm(residual, dim=1).mean().item())

        lo = float(lambda_min)
        hi = float(max(lo * 10.0, 1.0))
        r_lo = _residual_mean(lo)
        if r_lo >= target:
            return b.new_full((batch,), lo)

        r_hi = _residual_mean(hi)
        while (r_hi < target) and (hi < float(lambda_max)):
            hi = min(hi * 10.0, float(lambda_max))
            r_hi = _residual_mean(hi)
        if r_hi < target:
            return b.new_full((batch,), hi)

        for _ in range(int(max_iter)):
            mid = math.sqrt(lo * hi)
            r_mid = _residual_mean(mid)
            if r_mid < target:
                lo = mid
            else:
                hi = mid
        return b.new_full((batch,), hi)


def _build_current_b1b1_operator(
    beta_vectors: Iterable[Tuple[int, int]],
) -> RuntimeTheoreticalB1B1Operator2D:
    return RuntimeTheoreticalB1B1Operator2D(beta_vectors=beta_vectors, height=IMAGE_SIZE, width=IMAGE_SIZE, t0=PAPER_T0)


def _current_b1b1_lambda_floor() -> float:
    return float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12))


@torch.no_grad()
def _solve_current_b1b1_tikhonov(
    operator: ImplicitPixelRadonOperator2D,
    g_obs: torch.Tensor,
    lambda_reg: float,
) -> torch.Tensor:
    if hasattr(operator, "solve_tikhonov_direct"):
        return operator.solve_tikhonov_direct(g_obs, lambda_reg=float(lambda_reg))
    solver = str(DATA_CONFIG.get("implicit_eval_solver", "cg")).strip().lower()
    if solver != "cg":
        raise ValueError(f"Unsupported implicit_eval_solver={solver!r}; expected 'cg'.")
    cg_iters = int(DATA_CONFIG.get("implicit_eval_cg_iters", 80))
    cg_tol = float(DATA_CONFIG.get("implicit_eval_cg_tol", 1.0e-4))
    lambda_eff = max(float(lambda_reg), _current_b1b1_lambda_floor())
    return operator.solve_tikhonov_cg(
        g_obs,
        lambda_reg=lambda_eff,
        max_iter=cg_iters,
        tol=cg_tol,
    )


def _apply_multiplicative_noise(g_clean: torch.Tensor, delta: float) -> Tuple[torch.Tensor, float]:
    rand_u = 2.0 * torch.rand_like(g_clean) - 1.0
    g_obs = g_clean + (float(delta) * g_clean * rand_u)
    noise_norm = float(torch.norm(g_obs - g_clean).item())
    return g_obs, noise_norm


def _paper_multi8_betas() -> List[Tuple[int, int]]:
    n = int(PAPER_COEFF_SIZE)
    return [
        (1, n),
        (-1, n),
        (1, -n),
        (-1, -n),
        (n, 1),
        (-n, 1),
        (n, -1),
        (-n, -1),
    ]


def _paper_interior_mask() -> np.ndarray:
    mask = np.zeros((PAPER_COEFF_SIZE, PAPER_COEFF_SIZE), dtype=bool)
    mask[2:19, 1:20] = True
    return mask


def _load_paper_coefficients() -> Tuple[np.ndarray, str]:
    mask = _paper_interior_mask().reshape(-1)
    if os.path.exists(PAPER_COEFF_NPY_PATH):
        coeff = np.load(PAPER_COEFF_NPY_PATH).astype(np.float64).reshape(-1)
        if coeff.size != PAPER_COEFF_SIZE * PAPER_COEFF_SIZE:
            raise ValueError(
                f"Expected {PAPER_COEFF_SIZE * PAPER_COEFF_SIZE} paper coefficients, got {coeff.size}"
            )
        coeff[~mask] = 0.0
        return coeff.reshape(PAPER_COEFF_SIZE, PAPER_COEFF_SIZE).astype(np.float32), "fig5_2A_npy"

    coeff = np.random.RandomState(PAPER_FIXED_SEED).randn(PAPER_COEFF_SIZE * PAPER_COEFF_SIZE).astype(np.float64)
    coeff[~mask] = 0.0
    return coeff.reshape(PAPER_COEFF_SIZE, PAPER_COEFF_SIZE).astype(np.float32), "fixed_seed_0_fallback"


def _direct_tikhonov_solve_explicit(
    A: torch.Tensor,
    g_obs: torch.Tensor,
    lambda_reg: float,
    height: int,
    width: int,
) -> torch.Tensor:
    if g_obs.dim() == 1:
        g_obs = g_obs.unsqueeze(0)
    g_obs = g_obs.to(dtype=A.dtype, device=A.device)
    ata = A.t() @ A
    eye = torch.eye(int(ata.shape[0]), device=ata.device, dtype=ata.dtype)
    chol = torch.linalg.cholesky(ata + (float(lambda_reg) * eye))
    atb = g_obs @ A
    coeff = torch.cholesky_solve(atb.t(), chol).t()
    return coeff.view(-1, 1, height, width)


def _direct_tikhonov_solve_explicit_masked(
    A: torch.Tensor,
    g_obs: torch.Tensor,
    lambda_reg: float,
    support_mask: np.ndarray,
    height: int,
    width: int,
) -> torch.Tensor:
    idx = torch.from_numpy(np.flatnonzero(np.asarray(support_mask).reshape(-1))).to(device=A.device, dtype=torch.long)
    A_solve = A[:, idx]
    coeff_inner = _direct_tikhonov_solve_explicit(
        A=A_solve,
        g_obs=g_obs,
        lambda_reg=lambda_reg,
        height=1,
        width=int(idx.numel()),
    ).view(g_obs.shape[0], -1)
    coeff_full = torch.zeros((g_obs.shape[0], int(height * width)), device=A.device, dtype=A.dtype)
    coeff_full[:, idx] = coeff_inner
    return coeff_full.view(-1, 1, height, width)


@torch.no_grad()
def _build_measurement_gram_implicit(
    operator: ImplicitPixelRadonOperator2D,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Build G = A A^T for an implicit operator by probing measurement-space basis vectors.
    """
    m = int(operator.M)
    eye = torch.eye(m, device=operator.alphas.device, dtype=torch.float32)
    rows: List[torch.Tensor] = []
    for start in range(0, m, int(chunk_size)):
        basis = eye[start : start + int(chunk_size)]
        rows.append(operator.forward(operator.adjoint(basis)))
    return torch.cat(rows, dim=0)


@torch.no_grad()
def _direct_tikhonov_solve_implicit(
    operator: ImplicitPixelRadonOperator2D,
    g_obs: torch.Tensor,
    lambda_reg: float,
    gram_eigvals: torch.Tensor,
    gram_eigvecs: torch.Tensor,
) -> torch.Tensor:
    """
    Exact Tikhonov direct solve in measurement space:

        c = A^T (A A^T + lambda I)^{-1} g
    """
    if g_obs.dim() == 1:
        g_obs = g_obs.unsqueeze(0)
    eigvals = gram_eigvals.to(dtype=torch.float64, device=gram_eigvals.device)
    eigvecs = gram_eigvecs.to(dtype=torch.float64, device=gram_eigvecs.device)
    rhs = g_obs.to(dtype=torch.float64, device=eigvecs.device)
    inv_diag = 1.0 / (eigvals + float(lambda_reg))
    y = ((rhs @ eigvecs) * inv_diag.unsqueeze(0)) @ eigvecs.t()
    return operator.adjoint(y.to(dtype=torch.float32))


def _morozov_choose_lambda_newton(
    A: torch.Tensor,
    b: torch.Tensor,
    noise_norm: float,
    tau: float = 1.0,
    max_iter: int = 30,
    lam_min: float = 1e-16,
    lam_max: float = 1e16,
) -> float:
    a_cpu = A.detach().to(dtype=torch.float64, device="cpu")
    b_cpu = b.detach().view(-1).to(dtype=torch.float64, device="cpu")
    u, s, _ = torch.linalg.svd(a_cpu, full_matrices=False)

    c = u.t().mv(b_cpu)
    c2 = c.square().numpy()
    s2 = s.square().numpy()
    b_norm2 = float(torch.dot(b_cpu, b_cpu).item())
    c_norm2 = float(np.sum(c2))
    perp_norm2 = max(0.0, b_norm2 - c_norm2)
    target2 = float(tau * noise_norm) ** 2

    if target2 <= perp_norm2 + 1e-18:
        return float(lam_min)
    if target2 >= b_norm2 - 1e-18:
        return float(lam_max)

    newton_tol = float(DATA_CONFIG.get("morozov_newton_tol", 1.0e-10))
    initial_lambda = float(DATA_CONFIG.get("morozov_initial_lambda", 1.0))

    def phi(lam: float) -> float:
        w = lam / (s2 + lam)
        return float(np.sum((w * w) * c2) + perp_norm2)

    def dphi(lam: float) -> float:
        return float(np.sum((2.0 * lam * s2 / ((s2 + lam) ** 3)) * c2))

    lo = float(lam_min)
    hi = 1.0
    while phi(hi) < target2 and hi < float(lam_max):
        hi = min(hi * 10.0, float(lam_max))
    if phi(hi) < target2:
        return float(hi)

    lam = float(initial_lambda)
    if not (lo < lam < hi):
        lam = np.sqrt(lo * hi) if lo > 0.0 else min(1.0, hi)
    for _ in range(int(max_iter)):
        value = phi(lam) - target2
        if abs(value) <= float(newton_tol) * max(1.0, target2):
            return float(lam)

        deriv = dphi(lam)
        if deriv <= 0.0:
            candidate = np.sqrt(lo * hi)
        else:
            candidate = lam - (value / deriv)
            if not (lo < candidate < hi):
                candidate = np.sqrt(lo * hi)

        if value < 0.0:
            lo = lam
        else:
            hi = lam
        lam = float(candidate)

    return float(lam)


@contextmanager
def _paper_time_domain_config():
    old = dict(TIME_DOMAIN_CONFIG)
    try:
        TIME_DOMAIN_CONFIG.update(
            {
                "sampling_scheme": "paper_grid_t0",
                "sampling_t0": PAPER_T0,
                "num_detector_samples": PAPER_COEFF_SIZE * PAPER_COEFF_SIZE,
            }
        )
        yield
    finally:
        TIME_DOMAIN_CONFIG.clear()
        TIME_DOMAIN_CONFIG.update(old)


def _build_current_multi8_betas() -> List[Tuple[int, int]]:
    size = int(IMAGE_SIZE)
    return [
        (1, size),
        (-1, size),
        (1, -size),
        (-1, -size),
        (size, 1),
        (-size, 1),
        (size, -1),
        (-size, -1),
    ]


def _build_current_single_beta() -> Tuple[int, int]:
    return (1, int(IMAGE_SIZE))


def _render_coeff(coeff: np.ndarray, mode_name: str) -> np.ndarray:
    if mode_name.startswith("paper_b2b1"):
        return synthesize_f_from_coeff_b2b1(coeff, image_size=PAPER_COEFF_SIZE, out_size=DISPLAY_SIZE)
    return coeff.astype(np.float32)


def _plot_mode_triptych(
    coeff_true: np.ndarray,
    coeff_est: np.ndarray,
    mode_name: str,
    delta: float,
    lambda_reg: float,
    coeff_res: float,
) -> str:
    img_true = _render_coeff(coeff_true, mode_name)
    img_est = _render_coeff(coeff_est, mode_name)
    diff = img_est - img_true

    vmin = float(min(np.min(img_true), np.min(img_est)))
    vmax = float(max(np.max(img_true), np.max(img_est)))
    dv = float(np.max(np.abs(diff))) if diff.size else 1.0
    if dv <= 0.0:
        dv = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    im0 = axes[0].imshow(img_true, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("True")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(img_est, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Tikhonov")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, cmap="coolwarm", origin="lower", vmin=-dv, vmax=dv)
    axes[2].set_title("Difference")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.suptitle(f"{mode_name} | delta={delta:.3g} | lambda={lambda_reg:.4e} | coeff_RES={coeff_res:.6f}", y=1.05)
    out_path = os.path.join(RESULTS_DIR, f"{mode_name}_delta_{str(delta).replace('.', '_')}.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _evaluate_paper_b2b1(
    delta: float,
    num_trials: int,
    base_seed: int,
    beta_vectors: Iterable[Tuple[int, int]],
    mode_name: str,
) -> Dict[str, object]:
    coeff_true_np, coeff_source = _load_paper_coefficients()
    coeff_true = torch.from_numpy(coeff_true_np).view(1, 1, PAPER_COEFF_SIZE, PAPER_COEFF_SIZE).to(device=device)
    support_mask = _paper_interior_mask()

    with _paper_time_domain_config():
        betas = [tuple(int(vv) for vv in beta) for beta in beta_vectors]
        if len(betas) == 1:
            operator = RadonExample11Operator2D(
                beta=betas[0],
                height=PAPER_COEFF_SIZE,
                width=PAPER_COEFF_SIZE,
                num_detector_samples=PAPER_COEFF_SIZE * PAPER_COEFF_SIZE,
            )
        else:
            operator = MultiAngleRadonOperator2D(
                beta_vectors=betas,
                height=PAPER_COEFF_SIZE,
                width=PAPER_COEFF_SIZE,
                num_detector_samples_per_angle=PAPER_COEFF_SIZE * PAPER_COEFF_SIZE,
            )

    g_clean = operator.forward(coeff_true).to(dtype=torch.float32)

    lambdas: List[float] = []
    noise_norms: List[float] = []
    residual_norms: List[float] = []
    coeff_res_list: List[float] = []
    last_est = coeff_true.squeeze().detach().cpu().numpy()
    last_lambda = 0.0
    solve_matrix = operator.A[:, torch.from_numpy(np.flatnonzero(support_mask.reshape(-1))).to(device=operator.A.device, dtype=torch.long)]

    for idx in range(int(num_trials)):
        torch.manual_seed(int(base_seed + idx))
        g_obs, noise_norm = _apply_multiplicative_noise(g_clean, delta=float(delta))
        lam = _morozov_choose_lambda_newton(
            solve_matrix,
            g_obs.squeeze(0),
            noise_norm=noise_norm,
            tau=float(DATA_CONFIG.get("morozov_tau", 1.0)),
            max_iter=int(DATA_CONFIG.get("morozov_max_iter", 30)),
            lam_min=float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12)),
            lam_max=float(DATA_CONFIG.get("morozov_lambda_max", 1.0e12)),
        )
        coeff_est = _direct_tikhonov_solve_explicit_masked(
            A=operator.A,
            g_obs=g_obs,
            lambda_reg=lam,
            support_mask=support_mask,
            height=PAPER_COEFF_SIZE,
            width=PAPER_COEFF_SIZE,
        )
        residual = operator.forward(coeff_est) - g_obs

        lambdas.append(float(lam))
        noise_norms.append(float(noise_norm))
        residual_norms.append(float(torch.norm(residual).item()))
        coeff_res_list.append(_coeff_res(coeff_est.squeeze(), coeff_true.squeeze()))
        last_est = coeff_est.squeeze().detach().cpu().numpy()
        last_lambda = float(lam)

    plot_path = _plot_mode_triptych(
        coeff_true_np,
        last_est,
        mode_name=mode_name,
        delta=float(delta),
        lambda_reg=last_lambda,
        coeff_res=coeff_res_list[-1],
    )
    return {
        "mode": mode_name,
        "coeff_true": coeff_true_np,
        "coeff_est": last_est,
        "lambda_mean": float(np.mean(lambdas)),
        "noise_norm_mean": float(np.mean(noise_norms)),
        "meas_res_mean": float(np.mean(residual_norms)),
        "coeff_res_mean": float(np.mean(coeff_res_list)),
        "plot_path": plot_path,
        "coeff_source": coeff_source,
    }


def _evaluate_paper_b2b1_single(
    delta: float,
    num_trials: int,
    base_seed: int,
) -> Dict[str, object]:
    return _evaluate_paper_b2b1(
        delta=delta,
        num_trials=num_trials,
        base_seed=base_seed,
        beta_vectors=[PAPER_BETA],
        mode_name="paper_b2b1_single",
    )


def _evaluate_paper_b2b1_multi8(
    delta: float,
    num_trials: int,
    base_seed: int,
) -> Dict[str, object]:
    return _evaluate_paper_b2b1(
        delta=delta,
        num_trials=num_trials,
        base_seed=base_seed,
        beta_vectors=_paper_multi8_betas(),
        mode_name="paper_b2b1_multi8",
    )


def _evaluate_current_b1b1(
    phantom_128: torch.Tensor,
    delta: float,
    num_trials: int,
    base_seed: int,
    beta_vectors: Iterable[Tuple[int, int]],
    mode_name: str,
) -> Dict[str, object]:
    coeff_true = phantom_128.view(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device=device, dtype=torch.float32)
    operator = _build_current_b1b1_operator(beta_vectors)
    g_clean = operator.forward(coeff_true).to(dtype=torch.float32)

    lambdas: List[float] = []
    noise_norms: List[float] = []
    residual_norms: List[float] = []
    coeff_res_list: List[float] = []
    last_est = coeff_true.squeeze().detach().cpu().numpy()
    last_lambda = 0.0

    tau = float(DATA_CONFIG.get("morozov_tau", 1.0))
    max_iter = int(DATA_CONFIG.get("morozov_max_iter", 8))
    lambda_floor = _current_b1b1_lambda_floor()

    for idx in range(int(num_trials)):
        torch.manual_seed(int(base_seed + idx))
        g_obs, noise_norm = _apply_multiplicative_noise(g_clean, delta=float(delta))
        lam = operator.choose_lambda_morozov(
            g_obs,
            noise_norm=torch.tensor([noise_norm], device=g_obs.device, dtype=g_obs.dtype),
            tau=tau,
            max_iter=max_iter,
            lambda_min=lambda_floor,
        )
        lambda_eff = max(float(lam.view(-1)[0].item()), lambda_floor)
        coeff_est = _solve_current_b1b1_tikhonov(
            operator=operator,
            g_obs=g_obs,
            lambda_reg=lambda_eff,
        )
        residual = operator.forward(coeff_est) - g_obs

        lambdas.append(lambda_eff)
        noise_norms.append(float(noise_norm))
        residual_norms.append(float(torch.norm(residual).item()))
        coeff_res_list.append(_coeff_res(coeff_est.squeeze(), coeff_true.squeeze()))
        last_est = coeff_est.squeeze().detach().cpu().numpy()
        last_lambda = lambda_eff

    plot_path = _plot_mode_triptych(
        coeff_true.squeeze().detach().cpu().numpy(),
        last_est,
        mode_name=mode_name,
        delta=float(delta),
        lambda_reg=last_lambda,
        coeff_res=coeff_res_list[-1],
    )
    return {
        "mode": mode_name,
        "coeff_true": coeff_true.squeeze().detach().cpu().numpy(),
        "coeff_est": last_est,
        "lambda_mean": float(np.mean(lambdas)),
        "noise_norm_mean": float(np.mean(noise_norms)),
        "meas_res_mean": float(np.mean(residual_norms)),
        "coeff_res_mean": float(np.mean(coeff_res_list)),
        "plot_path": plot_path,
    }


def _print_summary(results: List[Dict[str, object]], delta: float, num_trials: int) -> None:
    print("=" * 92)
    print(f"Pure Tikhonov comparison | delta={delta:.3g} | trials={num_trials}")
    print("=" * 92)
    print(f"{'Mode':<24} | {'Mean lambda':>12} | {'Mean noise':>12} | {'Mean meas res':>14} | {'Mean coeff RES':>14}")
    print("-" * 92)
    for item in results:
        print(
            f"{item['mode']:<24} | "
            f"{float(item['lambda_mean']):>12.6e} | "
            f"{float(item['noise_norm_mean']):>12.6f} | "
            f"{float(item['meas_res_mean']):>14.6f} | "
            f"{float(item['coeff_res_mean']):>14.6f}"
        )
        if "coeff_source" in item:
            print(f"  coeff_source: {item['coeff_source']}")
        print(f"  plot: {item['plot_path']}")
    print("-" * 92)
    print("Note: coefficient RES is reported inside each mode's own coefficient space.")


def _run_mode(mode: str, phantom_128: torch.Tensor, delta: float, num_trials: int, base_seed: int) -> Dict[str, object]:
    if mode == "paper_b2b1_single":
        return _evaluate_paper_b2b1_single(delta=delta, num_trials=num_trials, base_seed=base_seed)
    if mode == "paper_b2b1_multi8":
        return _evaluate_paper_b2b1_multi8(delta=delta, num_trials=num_trials, base_seed=base_seed)
    if mode in ("current_b1b1_single", "ellipse_b1b1_single"):
        return _evaluate_current_b1b1(
            phantom_128=phantom_128,
            delta=delta,
            num_trials=num_trials,
            base_seed=base_seed,
            beta_vectors=[_build_current_single_beta()],
            mode_name="ellipse_b1b1_single",
        )
    if mode in ("current_b1b1_multi8", "ellipse_b1b1_multi8"):
        return _evaluate_current_b1b1(
            phantom_128=phantom_128,
            delta=delta,
            num_trials=num_trials,
            base_seed=base_seed,
            beta_vectors=_build_current_multi8_betas(),
            mode_name="ellipse_b1b1_multi8",
        )
    raise ValueError(f"Unknown mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure Tikhonov comparison: paper B2*B1 vs current B1*B1")
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=[
            "compare",
            "paper_b2b1_single",
            "paper_b2b1_multi8",
            "ellipse_b1b1_single",
            "ellipse_b1b1_multi8",
            "current_b1b1_single",
            "current_b1b1_multi8",
        ],
        help="Run a single mode or the paper-vs-current comparison bundle.",
    )
    parser.add_argument(
        "--compare_mode",
        type=str,
        default=DEFAULT_COMPARE_MODE,
        choices=["ellipse_b1b1_single", "ellipse_b1b1_multi8", "both"],
        help="When mode=compare, choose which current-mode baselines to include.",
    )
    parser.add_argument("--delta", type=float, default=0.1, help="Multiplicative noise level delta.")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of noise trials on the same ellipse phantom.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for the shared ellipse phantom.")
    args = parser.parse_args()

    print(f"Running pure Tikhonov comparison on device: {device}")
    print(f"mode={args.mode}, compare_mode={args.compare_mode}, delta={args.delta}, num_trials={args.num_trials}, seed={args.seed}")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    phantom_128 = generate_random_ellipse_phantom(image_size=IMAGE_SIZE).to(dtype=torch.float32)

    if args.mode == "compare":
        modes = ["paper_b2b1_single", "paper_b2b1_multi8"]
        if args.compare_mode in ("ellipse_b1b1_single", "both"):
            modes.append("ellipse_b1b1_single")
        if args.compare_mode in ("ellipse_b1b1_multi8", "both"):
            modes.append("ellipse_b1b1_multi8")
    else:
        modes = [args.mode]

    results = [
        _run_mode(mode=mode, phantom_128=phantom_128, delta=float(args.delta), num_trials=int(args.num_trials), base_seed=int(args.seed) + 1000)
        for mode in modes
    ]
    _print_summary(results, delta=float(args.delta), num_trials=int(args.num_trials))


if __name__ == "__main__":
    main()
