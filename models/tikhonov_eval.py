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

from b_spline.b2b1_spline import phi_support_bounds_b1b1, radon_phi_b1b1, synthesize_f_from_coeff_b2b1
from config import DATA_CONFIG, IMAGE_SIZE, RESULTS_DIR, TIME_DOMAIN_CONFIG, device
from image_generator import generate_random_ellipse_phantom
from radon_transform import (
    FrequencyDualB1B1Operator2D,
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


def _noise_label(
    noise_mode: str,
    *,
    delta: float,
    target_snr_db: float,
) -> str:
    mode = str(noise_mode).strip().lower()
    if mode == "multiplicative":
        return f"delta={float(delta):.3g}"
    if mode == "snr":
        return f"SNR={float(target_snr_db):g}dB"
    raise ValueError(f"Unsupported noise_mode={noise_mode!r}; expected 'multiplicative' or 'snr'.")


def _noise_suffix(
    noise_mode: str,
    *,
    delta: float,
    target_snr_db: float,
) -> str:
    mode = str(noise_mode).strip().lower()
    if mode == "multiplicative":
        token = f"{float(delta):g}".replace("-", "m").replace(".", "_")
        return f"delta_{token}"
    if mode == "snr":
        token = f"{float(target_snr_db):g}".replace("-", "m").replace(".", "_")
        return f"snr_{token}_db"
    raise ValueError(f"Unsupported noise_mode={noise_mode!r}; expected 'multiplicative' or 'snr'.")


def _apply_real_measurement_noise(
    g_clean: torch.Tensor,
    *,
    noise_mode: str,
    delta: float,
    target_snr_db: float,
) -> Tuple[torch.Tensor, float]:
    mode = str(noise_mode).strip().lower()
    if mode == "multiplicative":
        rand_u = 2.0 * torch.rand_like(g_clean) - 1.0
        noise = float(delta) * g_clean * rand_u
    elif mode == "snr":
        snr_linear = 10.0 ** (float(target_snr_db) / 10.0)
        if g_clean.dim() == 1:
            signal_energy = torch.sum(g_clean.square())
            numel = int(g_clean.numel())
        else:
            signal_energy = torch.sum(g_clean.square(), dim=-1, keepdim=True)
            numel = int(g_clean.shape[-1])
        sigma_squared = signal_energy / (float(numel) * snr_linear)
        sigma = torch.sqrt(sigma_squared).to(dtype=g_clean.dtype, device=g_clean.device)
        noise = torch.randn_like(g_clean) * sigma
    else:
        raise ValueError(f"Unsupported noise_mode={noise_mode!r}; expected 'multiplicative' or 'snr'.")

    g_obs = g_clean + noise
    if g_clean.dim() == 1:
        noise_norm = float(torch.linalg.vector_norm(noise).item())
    else:
        noise_norm = float(torch.linalg.vector_norm(noise.reshape(noise.shape[0], -1), dim=1).mean().item())
    return g_obs, noise_norm


def _apply_frequency_sample_noise(
    g_clean: torch.Tensor,
    *,
    noise_mode: str,
    delta: float,
    target_snr_db: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    mode = str(noise_mode).strip().lower()
    if mode == "multiplicative":
        rand_u = (2.0 * torch.rand_like(g_clean.real) - 1.0).to(dtype=torch.float32, device=g_clean.device)
        noise = float(delta) * g_clean * rand_u.to(dtype=g_clean.dtype)
    elif mode == "snr":
        snr_linear = 10.0 ** (float(target_snr_db) / 10.0)
        if g_clean.dim() == 1:
            signal_energy = torch.sum(torch.abs(g_clean).square())
            numel = int(g_clean.numel())
        else:
            signal_energy = torch.sum(torch.abs(g_clean).square(), dim=-1, keepdim=True)
            numel = int(g_clean.shape[-1])
        sigma_squared = signal_energy / (float(numel) * snr_linear)
        sigma = torch.sqrt(sigma_squared).to(dtype=torch.float32, device=g_clean.device)
        noise_real = torch.randn_like(g_clean.real)
        noise_imag = torch.randn_like(g_clean.real)
        noise = (sigma / math.sqrt(2.0)) * (noise_real + (1j * noise_imag))
        noise = noise.to(dtype=g_clean.dtype)
    else:
        raise ValueError(
            f"Unsupported noise_mode={noise_mode!r}; expected 'multiplicative' or 'snr'."
        )
    g_obs = g_clean + noise
    if g_clean.dim() == 1:
        noise_power = torch.mean(torch.abs(noise).square()).view(1)
        noise_norm = float(torch.linalg.vector_norm(noise).item())
    else:
        noise_power = torch.mean(torch.abs(noise).square(), dim=1)
        noise_norm = float(torch.linalg.vector_norm(noise.reshape(noise.shape[0], -1), dim=1).mean().item())
    return g_obs, noise_power.to(dtype=torch.float32), noise_norm


def _current_dual_noise_domain() -> str:
    noise_domain = str(DATA_CONFIG.get("dual_noise_domain", "spectral_samples")).strip().lower()
    if noise_domain not in {"spectral_samples", "time_radon_samples"}:
        raise ValueError(
            f"Unsupported dual_noise_domain={noise_domain!r}; expected 'spectral_samples' or 'time_radon_samples'."
        )
    return noise_domain


def _mean_complex_noise_power(noise: torch.Tensor) -> torch.Tensor:
    if noise.dim() == 1:
        return torch.mean(torch.abs(noise).square()).view(1).to(dtype=torch.float32)
    return torch.mean(torch.abs(noise).square(), dim=1).to(dtype=torch.float32)


def _complex_noise_norm(noise: torch.Tensor) -> float:
    if noise.dim() == 1:
        return float(torch.linalg.vector_norm(noise).item())
    return float(torch.linalg.vector_norm(noise.reshape(noise.shape[0], -1), dim=1).mean().item())


def _build_dense_uniform_time_grid(
    reference_samples: torch.Tensor,
    *,
    num_samples: int,
) -> tuple[torch.Tensor, float]:
    ref = reference_samples.view(-1).to(dtype=torch.float32)
    num_samples = int(num_samples)
    if int(ref.numel()) < 2:
        raise ValueError("Need at least two reference theorem-grid samples to define the dense time interval.")
    if num_samples < 2:
        raise ValueError(f"dual_time_num_samples must be at least 2, got {num_samples}.")

    t_min = float(ref[0].item())
    t_max = float(ref[-1].item())
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError(
            f"Invalid theorem-grid interval [{t_min}, {t_max}] for dense time-domain sampling."
        )

    t_samples = torch.linspace(
        t_min,
        t_max,
        steps=num_samples,
        dtype=torch.float32,
        device=ref.device,
    )
    delta_t = float((t_samples[1] - t_samples[0]).item())
    return t_samples, delta_t


def _beta_phi_support_bounds_b1b1(beta: torch.Tensor) -> tuple[float, float]:
    beta64 = beta.to(dtype=torch.float64).view(-1)
    if int(beta64.numel()) != 2:
        raise ValueError(f"Expected beta with shape (2,), got {tuple(beta.shape)}")
    b1 = float(beta64[0].item())
    b2 = float(beta64[1].item())
    vals = [0.0, b1, b2, b1 + b2]
    return float(min(vals)), float(max(vals))


def _build_dual_time_sampling_grid_for_frequency_frontend(
    operator_dual: FrequencyDualB1B1Operator2D,
    *,
    num_samples: int,
    device: torch.device | str,
    reference_s_samples: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, Dict[str, object]]:
    num_samples = int(num_samples)
    if num_samples < 2:
        raise ValueError(f"dual_time_num_samples must be at least 2, got {num_samples}.")

    mode = str(DATA_CONFIG.get("dual_time_sampling_interval_mode", "theorem_grid")).strip().lower()
    beta_norm = float(torch.linalg.vector_norm(operator_dual.beta.to(dtype=torch.float64), ord=2).item())
    if mode == "theorem_grid":
        if reference_s_samples is None:
            raise ValueError("reference_s_samples is required when dual_time_sampling_interval_mode='theorem_grid'.")
        s_samples, delta_s = _build_dense_uniform_time_grid(
            reference_s_samples.to(dtype=torch.float32, device=device),
            num_samples=num_samples,
        )
        t_beta_samples = s_samples * float(beta_norm)
        delta_t = float(delta_s * float(beta_norm))
        return t_beta_samples, delta_t, {
            "mode": mode,
            "t_min": float(t_beta_samples[0].item()),
            "t_max": float(t_beta_samples[-1].item()),
            "delta_t": float(delta_t),
        }

    phi_lo, phi_hi = _beta_phi_support_bounds_b1b1(operator_dual.beta)
    t_min = float(operator_dual.kappa0.item()) + float(phi_lo)
    t_max = float(operator_dual.kappa_m.item()) + float(phi_hi)
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError(f"Invalid R_beta f support interval [{t_min}, {t_max}].")

    if mode == "beta_support_midpoint":
        delta_t = float((t_max - t_min) / float(num_samples))
        idx = torch.arange(num_samples, dtype=torch.float32, device=device)
        t_beta_samples = float(t_min) + (idx + 0.5) * float(delta_t)
        return t_beta_samples, delta_t, {
            "mode": mode,
            "t_min": float(t_min),
            "t_max": float(t_max),
            "delta_t": float(delta_t),
            "phi_support_t_min": float(phi_lo),
            "phi_support_t_max": float(phi_hi),
        }

    raise ValueError(
        f"Unsupported dual_time_sampling_interval_mode={mode!r}; expected 'theorem_grid', "
        "or 'beta_support_midpoint'."
    )


def _evaluate_dense_single_angle_time_samples(
    coeff_true: torch.Tensor,
    *,
    beta: Tuple[int, int],
    kappa0: int,
    d_to_lex_indices: torch.Tensor,
    t_samples: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if coeff_true.dim() == 3:
        coeff_true = coeff_true.unsqueeze(1)
    if coeff_true.dim() != 4 or int(coeff_true.shape[1]) != 1:
        raise ValueError(f"Expected coeff_true with shape (B,1,H,W), got {tuple(coeff_true.shape)}")

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"dual_time_forward_chunk_size must be positive, got {chunk_size}.")

    batch = int(coeff_true.shape[0])
    n = int(coeff_true.shape[-2] * coeff_true.shape[-1])
    coeff_flat = coeff_true.to(dtype=torch.float32).view(batch, n)
    d_order = d_to_lex_indices.view(-1).to(dtype=torch.int64, device=coeff_flat.device)
    if int(d_order.numel()) != n:
        raise ValueError(f"Expected d_to_lex_indices of length N={n}, got {int(d_order.numel())}.")
    d_vector = coeff_flat.index_select(1, d_order)

    beta_i = _to_integer_beta(beta).to(dtype=torch.float64, device=coeff_flat.device)
    beta_norm = float(torch.linalg.vector_norm(beta_i, ord=2).item())
    alpha = beta_i / beta_norm
    support_lo, support_hi = phi_support_bounds_b1b1(alpha)

    t = t_samples.view(-1).to(dtype=torch.float32, device=coeff_flat.device)
    out = torch.zeros((batch, int(t.numel())), dtype=torch.float32, device=coeff_flat.device)
    kappa0_float = float(int(kappa0))
    margin = 2

    for start in range(0, int(t.numel()), chunk_size):
        stop = min(start + chunk_size, int(t.numel()))
        t_chunk = t[start:stop]
        t_lo = float(t_chunk[0].item())
        t_hi = float(t_chunk[-1].item())
        n_lo = max(0, int(math.floor(beta_norm * (t_lo - support_hi) - kappa0_float)) - margin)
        n_hi = min(n - 1, int(math.ceil(beta_norm * (t_hi - support_lo) - kappa0_float)) + margin)
        if n_lo > n_hi:
            continue

        local_n = torch.arange(n_lo, n_hi + 1, dtype=torch.float64, device=coeff_flat.device)
        local_shifts = (local_n + kappa0_float) / beta_norm
        kernel = radon_phi_b1b1(
            t_chunk.to(dtype=torch.float64).view(-1, 1) - local_shifts.view(1, -1),
            alpha,
        ).to(dtype=torch.float32, device=coeff_flat.device)
        out[:, start:stop] = d_vector[:, n_lo : n_hi + 1] @ kernel.transpose(0, 1)

    return out


def _interpolate_time_samples(
    g_source: torch.Tensor,
    *,
    source_t: torch.Tensor,
    target_t: torch.Tensor,
) -> torch.Tensor:
    if g_source.dim() == 1:
        g_source = g_source.unsqueeze(0)
    if g_source.dim() != 2:
        raise ValueError(f"Expected g_source with shape (B,N_s), got {tuple(g_source.shape)}")

    source = source_t.view(-1).to(dtype=torch.float32, device=g_source.device)
    target = target_t.view(-1).to(dtype=torch.float32, device=g_source.device)
    if int(source.numel()) != int(g_source.shape[1]):
        raise ValueError(
            f"Expected source_t length {int(g_source.shape[1])}, got {int(source.numel())}."
        )
    if int(source.numel()) < 2:
        raise ValueError("Need at least two source time samples for interpolation.")
    source_step = torch.min(source[1:] - source[:-1]).clamp_min(1.0e-12)
    edge_tolerance = torch.clamp(0.51 * source_step, min=torch.tensor(1.0e-6, dtype=source.dtype, device=source.device))
    if bool((target < (source[0] - edge_tolerance)).any()) or bool((target > (source[-1] + edge_tolerance)).any()):
        raise ValueError("target_t must stay within the source_t interval for interpolation.")
    target = target.clamp(min=float(source[0].item()), max=float(source[-1].item()))
    if int(target.numel()) == int(source.numel()) and torch.allclose(target, source, atol=1.0e-7, rtol=0.0):
        return g_source.clone()

    idx_right = torch.searchsorted(source, target)
    idx_right = idx_right.clamp(1, int(source.numel()) - 1)
    idx_left = idx_right - 1
    x0 = source[idx_left]
    x1 = source[idx_right]
    weight = ((target - x0) / (x1 - x0).clamp_min(1.0e-12)).view(1, -1)
    y0 = g_source[:, idx_left]
    y1 = g_source[:, idx_right]
    return (1.0 - weight) * y0 + weight * y1


def _time_radon_samples_to_frequency_samples(
    g_time: torch.Tensor,
    *,
    t_samples: torch.Tensor,
    xi_samples: torch.Tensor,
    delta_t: float,
    chunk_size: int,
) -> torch.Tensor:
    if g_time.dim() == 1:
        g_time = g_time.unsqueeze(0)
    if g_time.dim() != 2:
        raise ValueError(f"Expected g_time with shape (B,N_t), got {tuple(g_time.shape)}")
    if int(t_samples.numel()) != int(g_time.shape[1]):
        raise ValueError(
            f"Expected t_samples length {int(g_time.shape[1])}, got {int(t_samples.numel())}."
        )

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"dual_time_fourier_chunk_size must be positive, got {chunk_size}.")

    device_time = g_time.device
    t = t_samples.view(-1).to(dtype=torch.float32, device=device_time)
    xi = xi_samples.view(-1).to(dtype=torch.float32, device=device_time)
    g_complex = g_time.to(dtype=torch.complex64, device=device_time)
    quadrature = str(DATA_CONFIG.get("dual_time_fourier_quadrature", "rectangle")).strip().lower()
    if quadrature not in {"rectangle", "midpoint_cell"}:
        raise ValueError(
            f"Unsupported dual_time_fourier_quadrature={quadrature!r}; "
            "expected 'rectangle' or 'midpoint_cell'."
        )

    out = torch.zeros((g_complex.shape[0], int(xi.numel())), dtype=torch.complex64, device=device_time)
    t_col = t.view(-1, 1)
    for start in range(0, int(xi.numel()), chunk_size):
        stop = min(start + chunk_size, int(xi.numel()))
        xi_chunk = xi[start:stop].view(1, -1)
        phase = torch.exp(-1j * (t_col * xi_chunk))
        freq_chunk = g_complex @ phase
        if quadrature == "midpoint_cell":
            z = 0.5 * float(delta_t) * xi_chunk.view(-1)
            cell_factor = torch.ones_like(z)
            nonzero = torch.abs(z) > 1.0e-7
            cell_factor[nonzero] = torch.sin(z[nonzero]) / z[nonzero]
            freq_chunk = freq_chunk * cell_factor.view(1, -1).to(dtype=torch.complex64)
        out[:, start:stop] = freq_chunk
    return out * float(delta_t)


def _build_dual_time_alias_covariance_inverse(
    *,
    t_beta_samples: torch.Tensor,
    delta_t: float,
    alias_offsets: torch.Tensor,
) -> torch.Tensor:
    alias_offsets = alias_offsets.view(-1).to(dtype=torch.float64, device=t_beta_samples.device)
    if int(alias_offsets.numel()) <= 0:
        raise ValueError("alias_offsets must be non-empty.")
    t = t_beta_samples.view(-1).to(dtype=torch.float64, device=t_beta_samples.device)
    diff = alias_offsets.view(-1, 1) - alias_offsets.view(1, -1)
    phase = torch.exp(-1j * ((2.0 * math.pi) * t.view(-1, 1, 1) * diff.view(1, *diff.shape)))
    cov = (float(delta_t) * float(delta_t)) * torch.sum(phase, dim=0).to(dtype=torch.complex128)
    ridge_rel = float(DATA_CONFIG.get("dual_time_gls_covariance_ridge_rel", 1.0e-10))
    diag_scale = float(torch.max(torch.abs(torch.diagonal(cov))).item())
    ridge = max(diag_scale * float(ridge_rel), 1.0e-12)
    cov = cov + ridge * torch.eye(int(alias_offsets.numel()), dtype=torch.complex128, device=cov.device)
    return torch.linalg.pinv(cov, hermitian=True)


def _build_dual_frequency_mask_info(
    reliability: torch.Tensor,
) -> Dict[str, object]:
    reliability = reliability.to(dtype=torch.float32)
    max_reliability = float(torch.max(reliability).item()) if int(reliability.numel()) > 0 else 0.0
    threshold_rel = float(DATA_CONFIG.get("dual_time_frequency_mask_rel", 0.0))
    mode = str(DATA_CONFIG.get("dual_time_frequency_mask_mode", "soft")).strip().lower()
    if mode not in {"hard", "soft"}:
        raise ValueError(
            f"Unsupported dual_time_frequency_mask_mode={mode!r}; expected 'hard' or 'soft'."
        )

    weights = torch.ones_like(reliability, dtype=torch.float32, device=reliability.device)
    threshold = 0.0
    if threshold_rel > 0.0 and max_reliability > 0.0:
        threshold = float(threshold_rel * max_reliability)
        if mode == "hard":
            weights = (reliability >= float(threshold)).to(dtype=torch.float32)
        else:
            weights = (reliability / float(threshold)).clamp(min=0.0, max=1.0).to(dtype=torch.float32)

    keep_ratio = float((weights > 0.0).to(dtype=torch.float32).mean().item()) if int(weights.numel()) > 0 else 1.0
    mean_weight = float(weights.mean().item()) if int(weights.numel()) > 0 else 1.0
    return {
        "weights": weights,
        "threshold": float(threshold),
        "threshold_rel": float(threshold_rel),
        "mode": mode,
        "keep_ratio": float(keep_ratio),
        "mean_weight": float(mean_weight),
        "max_reliability": float(max_reliability),
    }


def _build_direct_ck_gramian_mask_info(
    gramian: torch.Tensor,
) -> Dict[str, object]:
    """Build the sampling-constraint mask used by the direct c_k formula."""

    reliability = gramian.view(-1).to(dtype=torch.float32)
    if int(reliability.numel()) <= 0:
        raise ValueError("gramian must be non-empty.")

    mask_quantile = float(DATA_CONFIG.get("dual_direct_ck_gramian_mask_quantile", 0.0))
    if not math.isfinite(mask_quantile) or mask_quantile < 0.0 or mask_quantile >= 1.0:
        raise ValueError(
            "dual_direct_ck_gramian_mask_quantile must be in [0,1), "
            f"got {mask_quantile}."
        )
    mode = str(DATA_CONFIG.get("dual_direct_ck_mask_mode", "hard")).strip().lower()
    if mode not in {"hard", "soft"}:
        raise ValueError(f"Unsupported dual_direct_ck_mask_mode={mode!r}; expected 'hard' or 'soft'.")
    weight_norm = str(DATA_CONFIG.get("dual_direct_ck_weight_norm", "none")).strip().lower()
    if weight_norm not in {"none", "renorm"}:
        raise ValueError(f"Unsupported dual_direct_ck_weight_norm={weight_norm!r}; expected 'none' or 'renorm'.")

    weights = torch.ones_like(reliability, dtype=torch.float32, device=reliability.device)
    threshold = 0.0
    if mask_quantile > 0.0:
        threshold = float(torch.quantile(reliability, float(mask_quantile)).item())
        if mode == "hard":
            weights = (reliability > float(threshold)).to(dtype=torch.float32)
            if float(weights.sum().item()) <= 0.0:
                weights = torch.ones_like(reliability, dtype=torch.float32, device=reliability.device)
        else:
            tau = max(float(threshold), 1.0e-30)
            weights = (reliability / float(tau)).clamp(min=0.0, max=1.0).to(dtype=torch.float32)

    keep_ratio = float((weights > 0.0).to(dtype=torch.float32).mean().item())
    mean_weight = float(weights.mean().item())
    norm_scale = 1.0
    if weight_norm == "renorm":
        norm_scale = 1.0 / max(float(mean_weight), 1.0e-12)

    max_reliability = float(torch.max(reliability).item())
    min_reliability = float(torch.min(reliability).item())
    return {
        "weights": weights,
        "threshold": float(threshold),
        "threshold_rel": float(mask_quantile),
        "mode": mode,
        "weight_norm": weight_norm,
        "norm_scale": float(norm_scale),
        "keep_ratio": float(keep_ratio),
        "mean_weight": float(mean_weight),
        "max_reliability": float(max_reliability),
        "min_reliability": float(min_reliability),
    }


def _direct_ck_stability_disabled_diagnostics(
    kernel: torch.Tensor,
) -> Dict[str, object]:
    abs_kernel = torch.abs(kernel)
    return {
        "stability_mode": "none",
        "stability_alias_mode": "soft",
        "stability_clip_threshold": 0.0,
        "stability_clip_ratio": 0.0,
        "stability_alias_residual_mean": 0.0,
        "stability_alias_residual_q95": 0.0,
        "stability_alias_weight_mean": 1.0,
        "stability_alias_weight_min": 1.0,
        "stability_norm_ratio_threshold": 0.0,
        "stability_norm_ratio_mean": 0.0,
        "stability_norm_ratio_q95": 0.0,
        "stability_norm_ratio_weight_mean": 1.0,
        "stability_norm_ratio_weight_min": 1.0,
        "stability_kernel_abs_raw_max": float(torch.max(abs_kernel).item()) if int(abs_kernel.numel()) > 0 else 0.0,
        "stability_kernel_abs_after_max": float(torch.max(abs_kernel).item()) if int(abs_kernel.numel()) > 0 else 0.0,
    }


def _stabilize_direct_ck_kernel(
    g_alias: torch.Tensor,
    h_alias: torch.Tensor,
    numerator: torch.Tensor,
    gramian_den: torch.Tensor,
    *,
    lambda_eff: float,
) -> Dict[str, object]:
    """Apply optional g/h-alias stability controls to the direct_ck kernel.

    ``g_alias`` and ``h_alias`` describe the actually sampled alias block used in
    the numerator.  ``gramian_den`` is the denominator Gramian controlled by L.
    These are intentionally kept separate because experiments may use Q != L.
    """

    if g_alias.dim() != 3:
        raise ValueError(f"Expected g_alias with shape (B,Q,P), got {tuple(g_alias.shape)}")
    if h_alias.dim() != 2:
        raise ValueError(f"Expected h_alias with shape (Q,P), got {tuple(h_alias.shape)}")
    if int(g_alias.shape[1]) != int(h_alias.shape[0]) or int(g_alias.shape[2]) != int(h_alias.shape[1]):
        raise ValueError(
            "g_alias and h_alias dimensions disagree: "
            f"g_alias={tuple(g_alias.shape)}, h_alias={tuple(h_alias.shape)}"
        )
    if numerator.dim() == 1:
        numerator = numerator.unsqueeze(0)
    if numerator.dim() != 2 or int(numerator.shape[0]) != int(g_alias.shape[0]) or int(numerator.shape[1]) != int(g_alias.shape[2]):
        raise ValueError(
            f"Expected numerator with shape (B,P) matching g_alias, got {tuple(numerator.shape)}."
        )

    mode = str(DATA_CONFIG.get("dual_direct_ck_stability_mode", "none")).strip().lower()
    allowed_modes = {"none", "kclip", "alias_consistency", "kclip_alias", "norm_ratio"}
    if mode not in allowed_modes:
        raise ValueError(
            f"Unsupported dual_direct_ck_stability_mode={mode!r}; "
            "expected 'none', 'kclip', 'alias_consistency', 'kclip_alias', or 'norm_ratio'."
        )
    gramian_view = gramian_den.view(1, -1).to(dtype=torch.float32, device=numerator.device)
    if int(gramian_view.shape[1]) != int(numerator.shape[1]):
        raise ValueError(
            f"Expected gramian_den width P={int(numerator.shape[1])}, got {int(gramian_view.shape[1])}."
        )
    eps = float(DATA_CONFIG.get("dual_direct_ck_clip_eps", 1.0e-12))
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"dual_direct_ck_clip_eps must be positive and finite, got {eps}.")

    denominator = gramian_view.to(dtype=torch.complex64) + torch.tensor(
        float(lambda_eff),
        dtype=torch.complex64,
        device=numerator.device,
    )
    kernel_raw = numerator.to(dtype=torch.complex64, device=numerator.device) / denominator
    if mode == "none":
        diagnostics = _direct_ck_stability_disabled_diagnostics(kernel_raw)
        return {"kernel": kernel_raw, **diagnostics}

    kernel = kernel_raw
    clip_threshold = torch.zeros((int(kernel.shape[0]), 1), dtype=torch.float32, device=kernel.device)
    clip_ratio = 0.0
    good_rel = float(DATA_CONFIG.get("dual_direct_ck_clip_good_g_rel", 0.1))
    clip_quantile = float(DATA_CONFIG.get("dual_direct_ck_clip_quantile", 0.99))
    clip_scale = float(DATA_CONFIG.get("dual_direct_ck_clip_scale", 1.2))
    if not math.isfinite(good_rel) or good_rel < 0.0 or good_rel > 1.0:
        raise ValueError(f"dual_direct_ck_clip_good_g_rel must be in [0,1], got {good_rel}.")
    if not math.isfinite(clip_quantile) or clip_quantile < 0.0 or clip_quantile >= 1.0:
        raise ValueError(f"dual_direct_ck_clip_quantile must be in [0,1), got {clip_quantile}.")
    if not math.isfinite(clip_scale) or clip_scale <= 0.0:
        raise ValueError(f"dual_direct_ck_clip_scale must be positive and finite, got {clip_scale}.")

    if mode in {"kclip", "kclip_alias"}:
        reliability = gramian_view.view(-1)
        max_reliability = float(torch.max(reliability).item())
        good = reliability >= float(good_rel * max_reliability)
        if int(good.to(dtype=torch.int64).sum().item()) < 10:
            median_reliability = torch.quantile(reliability, 0.5)
            good = reliability >= median_reliability
        if int(good.to(dtype=torch.int64).sum().item()) < 1:
            good = torch.ones_like(reliability, dtype=torch.bool, device=reliability.device)
        abs_good = torch.abs(kernel_raw[:, good])
        clip_threshold = (
            torch.quantile(abs_good.to(dtype=torch.float32), float(clip_quantile), dim=1, keepdim=True)
            * float(clip_scale)
        ).clamp_min(float(eps))
        shrink = torch.minimum(
            torch.ones_like(torch.abs(kernel), dtype=torch.float32),
            clip_threshold / (torch.abs(kernel).to(dtype=torch.float32) + float(eps)),
        )
        clip_ratio = float((shrink < 0.999999).to(dtype=torch.float32).mean().item())
        kernel = kernel * shrink.to(dtype=torch.complex64)

    alias_mode = str(DATA_CONFIG.get("dual_direct_ck_alias_mode", "soft")).strip().lower()
    if alias_mode not in {"soft", "hard"}:
        raise ValueError(f"Unsupported dual_direct_ck_alias_mode={alias_mode!r}; expected 'soft' or 'hard'.")
    alias_tau = float(DATA_CONFIG.get("dual_direct_ck_alias_tau", 0.3))
    if not math.isfinite(alias_tau) or alias_tau <= 0.0:
        raise ValueError(f"dual_direct_ck_alias_tau must be positive and finite, got {alias_tau}.")

    alias_residual = torch.zeros(
        (int(g_alias.shape[0]), int(g_alias.shape[2])),
        dtype=torch.float32,
        device=g_alias.device,
    )
    alias_weights = torch.ones_like(alias_residual, dtype=torch.float32, device=g_alias.device)
    norm_ratio = torch.zeros_like(alias_residual, dtype=torch.float32, device=g_alias.device)
    norm_ratio_weights = torch.ones_like(alias_residual, dtype=torch.float32, device=g_alias.device)
    norm_ratio_threshold = torch.zeros((int(kernel.shape[0]), 1), dtype=torch.float32, device=kernel.device)
    if mode in {"alias_consistency", "kclip_alias"} and int(h_alias.shape[0]) > 1:
        h_complex = h_alias.to(dtype=torch.complex64, device=g_alias.device)
        g_complex = g_alias.to(dtype=torch.complex64, device=g_alias.device)
        gramian_alias = torch.sum(torch.abs(h_complex).square(), dim=0).to(dtype=torch.float32).clamp_min(float(eps))
        common_alias = numerator.to(dtype=torch.complex64, device=g_alias.device) / gramian_alias.view(1, -1).to(dtype=torch.complex64)
        residual = g_complex - common_alias.unsqueeze(1) * h_complex.unsqueeze(0)
        residual_norm = torch.sqrt(torch.sum(torch.abs(residual).square(), dim=1).to(dtype=torch.float32))
        g_norm = torch.sqrt(torch.sum(torch.abs(g_complex).square(), dim=1).to(dtype=torch.float32))
        alias_residual = residual_norm / (g_norm + float(eps))
        if alias_mode == "soft":
            alias_weights = 1.0 / (1.0 + torch.square(alias_residual / float(alias_tau)))
        else:
            alias_weights = (alias_residual <= float(alias_tau)).to(dtype=torch.float32)
        kernel = kernel * alias_weights.to(dtype=torch.complex64)

    if mode == "norm_ratio":
        h_complex = h_alias.to(dtype=torch.complex64, device=g_alias.device)
        g_complex = g_alias.to(dtype=torch.complex64, device=g_alias.device)
        g_norm = torch.sqrt(torch.sum(torch.abs(g_complex).square(), dim=1).to(dtype=torch.float32))
        h_norm = torch.sqrt(torch.sum(torch.abs(h_complex).square(), dim=0).to(dtype=torch.float32)).clamp_min(float(eps))
        norm_ratio = g_norm / (h_norm.view(1, -1) + float(eps))

        if int(h_alias.shape[0]) > 1:
            gramian_alias = torch.sum(torch.abs(h_complex).square(), dim=0).to(dtype=torch.float32).clamp_min(float(eps))
            common_alias = numerator.to(dtype=torch.complex64, device=g_alias.device) / gramian_alias.view(1, -1).to(dtype=torch.complex64)
            residual = g_complex - common_alias.unsqueeze(1) * h_complex.unsqueeze(0)
            residual_norm = torch.sqrt(torch.sum(torch.abs(residual).square(), dim=1).to(dtype=torch.float32))
            alias_residual = residual_norm / (g_norm + float(eps))

        # Local robust norm-bound mode:
        #
        #   ||g_j||_2 <= gamma_j * ||h_j||_2
        #   gamma_j = scale * local_median(||g_j||_2 / ||h_j||_2)
        #
        # With alias_gate enabled, only apply the norm shrink when the same
        # frequency also violates the g_j ~= C_j h_j alias direction test.
        local_window = int(DATA_CONFIG.get("dual_direct_ck_norm_ratio_window", 512))
        if local_window <= 0:
            raise ValueError(f"dual_direct_ck_norm_ratio_window must be positive, got {local_window}.")
        radius = int(local_window // 2)
        if radius > 0 and int(norm_ratio.shape[1]) > 1:
            radius = min(radius, int(norm_ratio.shape[1]) - 1)
            ratio_padded = torch.nn.functional.pad(
                norm_ratio.unsqueeze(1),
                (radius, radius),
                mode="circular",
            ).squeeze(1)
            ratio_windows = ratio_padded.unfold(1, int(2 * radius + 1), 1)
            local_scale = torch.median(ratio_windows, dim=-1).values
        else:
            local_scale = norm_ratio
        norm_ratio_threshold = (local_scale * float(clip_scale)).clamp_min(float(eps))
        if alias_mode == "soft":
            raw_norm_ratio_weights = torch.minimum(
                torch.ones_like(norm_ratio, dtype=torch.float32),
                norm_ratio_threshold / (norm_ratio + float(eps)),
            )
        else:
            raw_norm_ratio_weights = (norm_ratio <= norm_ratio_threshold).to(dtype=torch.float32)
        use_alias_gate = bool(DATA_CONFIG.get("dual_direct_ck_norm_ratio_alias_gate", True)) and int(h_alias.shape[0]) > 1
        if use_alias_gate:
            gate = alias_residual > float(alias_tau)
            norm_ratio_weights = torch.where(
                gate,
                raw_norm_ratio_weights,
                torch.ones_like(raw_norm_ratio_weights, dtype=torch.float32),
            )
        else:
            norm_ratio_weights = raw_norm_ratio_weights
        kernel = kernel * norm_ratio_weights.to(dtype=torch.complex64)

    abs_raw = torch.abs(kernel_raw)
    abs_after = torch.abs(kernel)
    diagnostics = {
        "stability_mode": mode,
        "stability_alias_mode": alias_mode,
        "stability_clip_threshold": float(torch.mean(clip_threshold).item()),
        "stability_clip_ratio": float(clip_ratio),
        "stability_alias_residual_mean": float(torch.mean(alias_residual).item()),
        "stability_alias_residual_q95": float(torch.quantile(alias_residual.reshape(-1), 0.95).item()),
        "stability_alias_weight_mean": float(torch.mean(alias_weights).item()),
        "stability_alias_weight_min": float(torch.min(alias_weights).item()),
        "stability_norm_ratio_threshold": float(torch.mean(norm_ratio_threshold).item()),
        "stability_norm_ratio_mean": float(torch.mean(norm_ratio).item()),
        "stability_norm_ratio_q95": float(torch.quantile(norm_ratio.reshape(-1), 0.95).item()),
        "stability_norm_ratio_weight_mean": float(torch.mean(norm_ratio_weights).item()),
        "stability_norm_ratio_weight_min": float(torch.min(norm_ratio_weights).item()),
        "stability_kernel_abs_raw_max": float(torch.max(abs_raw).item()) if int(abs_raw.numel()) > 0 else 0.0,
        "stability_kernel_abs_after_max": float(torch.max(abs_after).item()) if int(abs_after.numel()) > 0 else 0.0,
    }
    return {"kernel": kernel, **diagnostics}


def _estimate_dual_common_factor_simple_ratio(
    g_estimator_freq: torch.Tensor,
    *,
    operator_dual: FrequencyDualB1B1Operator2D,
    alias_subset_indices: torch.Tensor,
) -> Dict[str, object]:
    if g_estimator_freq.dim() == 1:
        g_estimator_freq = g_estimator_freq.unsqueeze(0)
    if g_estimator_freq.dim() != 2:
        raise ValueError(
            f"Expected g_estimator_freq with shape (B,P*(2Qe+1)), got {tuple(g_estimator_freq.shape)}"
        )

    alias_indices = alias_subset_indices.to(dtype=torch.int64, device=operator_dual.hat_radon_phi_aliases.device).view(-1)
    if int(alias_indices.numel()) <= 0:
        raise ValueError("alias_subset_indices must be non-empty.")
    hat_subset = operator_dual.hat_radon_phi_aliases.index_select(0, alias_indices).to(
        dtype=torch.complex64,
        device=g_estimator_freq.device,
    )
    expected_width = int(hat_subset.shape[0] * hat_subset.shape[1])
    if int(g_estimator_freq.shape[1]) != expected_width:
        raise ValueError(
            f"Expected estimator frequency length {expected_width}, got {int(g_estimator_freq.shape[1])}."
        )
    g_subset = g_estimator_freq.to(dtype=torch.complex64, device=hat_subset.device).reshape(
        g_estimator_freq.shape[0],
        hat_subset.shape[0],
        hat_subset.shape[1],
    )
    numerator = torch.sum(g_subset * hat_subset.conj().unsqueeze(0), dim=1)
    subset_reliability = torch.sum(hat_subset.abs().square(), dim=0).to(dtype=torch.float32, device=hat_subset.device)
    subset_gramian = subset_reliability.view(1, -1).to(dtype=torch.complex64, device=hat_subset.device)
    rel_floor = float(DATA_CONFIG.get("dual_kernel_lambda_rel_floor", 3.0e-15))
    gramian_floor = float(torch.max(torch.abs(subset_gramian)).item()) * float(rel_floor)
    common_factor = numerator / (subset_gramian + float(gramian_floor))
    mask_info = _build_dual_frequency_mask_info(subset_reliability)
    common_factor = (
        common_factor
        * mask_info["weights"].view(1, -1).to(dtype=torch.complex64, device=hat_subset.device)
        / max(float(mask_info["mean_weight"]), 1.0e-12)
    )
    return {
        "common_factor": common_factor,
        "reliability": subset_reliability,
        "mask_weights": mask_info["weights"],
        "mask_threshold": float(mask_info["threshold"]),
        "mask_threshold_rel": float(mask_info["threshold_rel"]),
        "mask_mode": str(mask_info["mode"]),
        "mask_keep_ratio": float(mask_info["keep_ratio"]),
        "mask_mean_weight": float(mask_info["mean_weight"]),
        "max_reliability": float(mask_info["max_reliability"]),
    }


def _estimate_dual_common_factor_gls(
    g_estimator_freq: torch.Tensor,
    *,
    operator_dual: FrequencyDualB1B1Operator2D,
    alias_subset_indices: torch.Tensor,
    alias_covariance_inv: torch.Tensor,
) -> Dict[str, object]:
    if g_estimator_freq.dim() == 1:
        g_estimator_freq = g_estimator_freq.unsqueeze(0)
    if g_estimator_freq.dim() != 2:
        raise ValueError(
            f"Expected g_estimator_freq with shape (B,P*(2Qe+1)), got {tuple(g_estimator_freq.shape)}"
        )

    alias_indices = alias_subset_indices.to(dtype=torch.int64, device=operator_dual.hat_radon_phi_aliases.device).view(-1)
    if int(alias_indices.numel()) <= 0:
        raise ValueError("alias_subset_indices must be non-empty.")
    hat_subset = operator_dual.hat_radon_phi_aliases.index_select(0, alias_indices).to(
        dtype=torch.complex64,
        device=g_estimator_freq.device,
    )  # (m,P)
    expected_width = int(hat_subset.shape[0] * hat_subset.shape[1])
    if int(g_estimator_freq.shape[1]) != expected_width:
        raise ValueError(
            f"Expected estimator frequency length {expected_width}, got {int(g_estimator_freq.shape[1])}."
        )
    g_subset = g_estimator_freq.to(dtype=torch.complex64, device=hat_subset.device).reshape(
        g_estimator_freq.shape[0],
        hat_subset.shape[0],
        hat_subset.shape[1],
    )  # (B,m,P)
    cov_inv = alias_covariance_inv.to(dtype=torch.complex64, device=hat_subset.device)
    weighted_h = torch.einsum("ab,bp->ap", cov_inv, hat_subset)
    weighted_g = torch.einsum("ab,nbp->nap", cov_inv, g_subset)
    numerator = torch.sum(hat_subset.conj().unsqueeze(0) * weighted_g, dim=1)  # (B,P)
    denominator_raw = torch.sum(hat_subset.conj() * weighted_h, dim=0).real.to(dtype=torch.float32).clamp_min(0.0)  # (P,)
    lambda_rel = float(DATA_CONFIG.get("dual_time_gls_lambda_rel", 1.0e-6))
    lambda_floor = float(torch.max(denominator_raw).item()) * float(lambda_rel)
    denominator = denominator_raw.clamp_min(max(lambda_floor, 1.0e-12))
    common_factor = numerator / denominator.view(1, -1).to(dtype=torch.complex64, device=hat_subset.device)
    mask_info = _build_dual_frequency_mask_info(denominator_raw)
    common_factor = (
        common_factor
        * mask_info["weights"].view(1, -1).to(dtype=torch.complex64, device=hat_subset.device)
        / max(float(mask_info["mean_weight"]), 1.0e-12)
    )
    return {
        "common_factor": common_factor,
        "reliability": denominator_raw,
        "mask_weights": mask_info["weights"],
        "mask_threshold": float(mask_info["threshold"]),
        "mask_threshold_rel": float(mask_info["threshold_rel"]),
        "mask_mode": str(mask_info["mode"]),
        "mask_keep_ratio": float(mask_info["keep_ratio"]),
        "mask_mean_weight": float(mask_info["mean_weight"]),
        "max_reliability": float(mask_info["max_reliability"]),
    }


def _estimate_dual_common_factor_from_low_alias_subset(
    g_estimator_freq: torch.Tensor,
    *,
    operator_dual: FrequencyDualB1B1Operator2D,
    alias_subset_indices: torch.Tensor,
    alias_covariance_inv: torch.Tensor | None = None,
) -> Dict[str, object]:
    mode = str(DATA_CONFIG.get("dual_time_recovery_mode", "gls_low_alias")).strip().lower()
    if mode == "simple_ratio":
        return _estimate_dual_common_factor_simple_ratio(
            g_estimator_freq,
            operator_dual=operator_dual,
            alias_subset_indices=alias_subset_indices,
        )
    if mode == "gls_low_alias":
        if alias_covariance_inv is None:
            raise ValueError("alias_covariance_inv is required when dual_time_recovery_mode='gls_low_alias'.")
        return _estimate_dual_common_factor_gls(
            g_estimator_freq,
            operator_dual=operator_dual,
            alias_subset_indices=alias_subset_indices,
            alias_covariance_inv=alias_covariance_inv,
        )
    raise ValueError(
        f"Unsupported dual_time_recovery_mode={mode!r}; expected 'simple_ratio' or 'gls_low_alias' "
        "in the low-alias estimator path. Use the direct_ck path for direct c_k recovery."
    )


def _synthesize_dual_alias_blocks_from_common_factor(
    common_factor: torch.Tensor,
    *,
    operator_dual: FrequencyDualB1B1Operator2D,
) -> torch.Tensor:
    if common_factor.dim() == 1:
        common_factor = common_factor.unsqueeze(0)
    if common_factor.dim() != 2:
        raise ValueError(
            f"Expected common_factor with shape (B,P), got {tuple(common_factor.shape)}"
        )
    alias_hat = operator_dual.hat_radon_phi_aliases.to(
        dtype=torch.complex64,
        device=common_factor.device,
    )
    if int(common_factor.shape[1]) != int(alias_hat.shape[1]):
        raise ValueError(
            f"Expected common_factor width P={int(alias_hat.shape[1])}, got {int(common_factor.shape[1])}."
        )
    g_alias = alias_hat.unsqueeze(0) * common_factor.to(dtype=torch.complex64, device=alias_hat.device).unsqueeze(1)
    return g_alias.reshape(g_alias.shape[0], -1)


def _solve_dual_coeff_from_common_factor(
    common_factor: torch.Tensor,
    *,
    operator_dual: FrequencyDualB1B1Operator2D,
) -> torch.Tensor:
    if common_factor.dim() == 1:
        common_factor = common_factor.unsqueeze(0)
    if common_factor.dim() != 2:
        raise ValueError(f"Expected common_factor with shape (B,P), got {tuple(common_factor.shape)}")
    kernel = common_factor.to(dtype=torch.complex64, device=operator_dual.hat_radon_phi_aliases.device)
    if int(kernel.shape[1]) != int(operator_dual.base_num_frequency_samples):
        raise ValueError(
            f"Expected common_factor width P={int(operator_dual.base_num_frequency_samples)}, got {int(kernel.shape[1])}."
        )
    if operator_dual.use_direct_backend:
        d_vector = operator_dual._direct_sum_base_to_modes(
            kernel,
            scale=(1.0 / float(operator_dual.base_num_frequency_samples)),
        )
        return operator_dual._d_to_coeff(d_vector.real.to(dtype=torch.float32))

    weighted = operator_dual.dual_phase_pos_base.view(1, -1) * kernel
    d_vector = (
        torch.fft.ifft(weighted, dim=1)[:, : operator_dual.N]
        * operator_dual.dual_alt_sign_complex.view(1, -1)
        * operator_dual.dual_mode_shift_pos.view(1, -1)
    )
    return operator_dual._d_to_coeff(d_vector.real.to(dtype=torch.float32))


def _solve_dual_coeff_direct_ck_from_frequency_samples(
    g_freq: torch.Tensor,
    *,
    operator_dual: FrequencyDualB1B1Operator2D,
) -> Dict[str, object]:
    """Recover coefficients by directly applying the discretized c_k integral.

    This is the route used by the time-domain SNR sampling-constraint plan:
    time-domain samples are converted to the full alias frequency block
    \widetilde g_{q,j}, while \hat{R_\beta\phi} and G_{\beta,L} stay analytic.
    Optional masks only multiply the final principal-cell kernel.
    """

    if g_freq.dim() == 1:
        g_freq = g_freq.unsqueeze(0)
    g_alias = operator_dual._reshape_frequency_measurements(g_freq).to(
        dtype=torch.complex64,
        device=operator_dual.hat_radon_phi_aliases.device,
    )
    numerator = torch.sum(
        g_alias * operator_dual.hat_radon_phi_aliases.conj().unsqueeze(0),
        dim=1,
    )
    gramian = operator_dual.gramian_diag.view(1, -1).to(dtype=torch.float32, device=numerator.device)
    lambda_rel = float(DATA_CONFIG.get("dual_direct_ck_lambda_rel", 0.0))
    if not math.isfinite(lambda_rel) or lambda_rel < 0.0:
        raise ValueError(f"dual_direct_ck_lambda_rel must be non-negative and finite, got {lambda_rel}.")
    lambda_abs = float(torch.max(gramian).item()) * float(lambda_rel)
    floor_rel = float(DATA_CONFIG.get("dual_kernel_lambda_rel_floor", 3.0e-15))
    lambda_floor = float(torch.max(gramian).item()) * max(float(floor_rel), 0.0)
    lambda_eff = max(float(lambda_abs), float(lambda_floor))

    stability_info = _stabilize_direct_ck_kernel(
        g_alias,
        operator_dual.hat_radon_phi_aliases.to(dtype=torch.complex64, device=numerator.device),
        numerator,
        gramian,
        lambda_eff=float(lambda_eff),
    )
    mask_info = _build_direct_ck_gramian_mask_info(gramian.view(-1))
    weights = mask_info["weights"].view(1, -1).to(dtype=torch.complex64, device=numerator.device)
    kernel = stability_info["kernel"]
    kernel = kernel * weights * float(mask_info["norm_scale"])
    coeff = _solve_dual_coeff_from_common_factor(kernel, operator_dual=operator_dual)
    return {
        "coeff": coeff,
        "kernel": kernel,
        "numerator": numerator,
        "reliability": gramian.view(-1).to(dtype=torch.float32),
        "mask_weights": mask_info["weights"],
        "mask_threshold": float(mask_info["threshold"]),
        "mask_threshold_rel": float(mask_info["threshold_rel"]),
        "mask_mode": str(mask_info["mode"]),
        "mask_weight_norm": str(mask_info["weight_norm"]),
        "mask_keep_ratio": float(mask_info["keep_ratio"]),
        "mask_mean_weight": float(mask_info["mean_weight"]),
        "max_reliability": float(mask_info["max_reliability"]),
        "min_reliability": float(mask_info["min_reliability"]),
        "lambda_rel": float(lambda_rel),
        "lambda": float(lambda_eff),
        "stability_mode": str(stability_info["stability_mode"]),
        "stability_alias_mode": str(stability_info["stability_alias_mode"]),
        "stability_clip_threshold": float(stability_info["stability_clip_threshold"]),
        "stability_clip_ratio": float(stability_info["stability_clip_ratio"]),
        "stability_alias_residual_mean": float(stability_info["stability_alias_residual_mean"]),
        "stability_alias_residual_q95": float(stability_info["stability_alias_residual_q95"]),
        "stability_alias_weight_mean": float(stability_info["stability_alias_weight_mean"]),
        "stability_alias_weight_min": float(stability_info["stability_alias_weight_min"]),
        "stability_norm_ratio_threshold": float(stability_info["stability_norm_ratio_threshold"]),
        "stability_norm_ratio_mean": float(stability_info["stability_norm_ratio_mean"]),
        "stability_norm_ratio_q95": float(stability_info["stability_norm_ratio_q95"]),
        "stability_norm_ratio_weight_mean": float(stability_info["stability_norm_ratio_weight_mean"]),
        "stability_norm_ratio_weight_min": float(stability_info["stability_norm_ratio_weight_min"]),
        "stability_kernel_abs_raw_max": float(stability_info["stability_kernel_abs_raw_max"]),
        "stability_kernel_abs_after_max": float(stability_info["stability_kernel_abs_after_max"]),
    }


def _prepare_dual_time_domain_frontend(
    coeff_true: torch.Tensor,
    *,
    beta_vectors: Iterable[Tuple[int, int]],
    operator_time: RuntimeTheoreticalB1B1Operator2D | None = None,
    operator_dual: FrequencyDualB1B1Operator2D | None = None,
) -> Dict[str, object]:
    betas = [tuple(int(vv) for vv in beta) for beta in beta_vectors]
    if len(betas) != 1:
        raise ValueError(
            "Time-domain Dual front-end currently supports exactly one beta vector; "
            f"got {len(betas)}."
        )

    if operator_time is None:
        operator_time = _build_current_b1b1_operator(betas)
    if operator_dual is None:
        operator_dual = _build_current_dual_frequency_operator(betas)

    expected_num_samples = int(DATA_CONFIG.get("dual_time_num_samples", IMAGE_SIZE * IMAGE_SIZE))
    reference_s_samples = operator_time.sampling_points.view(-1).to(dtype=torch.float32, device=coeff_true.device)
    if int(reference_s_samples.numel()) < 2:
        raise ValueError("Need at least two theorem-grid time samples to define the dense time-domain interval.")

    reference_ds_all = reference_s_samples[1:] - reference_s_samples[:-1]
    reference_delta_s = float(reference_ds_all[0].item())
    if not torch.allclose(reference_ds_all, reference_ds_all[0], atol=1.0e-5, rtol=0.0):
        raise ValueError("Current theorem-grid time-domain sampling is not uniform; expected a uniform reference grid.")

    beta_norm = float(torch.linalg.vector_norm(operator_dual.beta.to(dtype=torch.float32), ord=2).item())
    t_beta_samples, delta_t, time_sampling_info = _build_dual_time_sampling_grid_for_frequency_frontend(
        operator_dual,
        num_samples=expected_num_samples,
        device=coeff_true.device,
        reference_s_samples=reference_s_samples,
    )
    s_samples = t_beta_samples / float(beta_norm)
    delta_s = float(delta_t / float(beta_norm))

    recovery_mode = str(DATA_CONFIG.get("dual_time_recovery_mode", "gls_low_alias")).strip().lower()
    if recovery_mode == "direct_ck":
        alias_estimator_truncation = int(operator_dual.integral_alias_truncation)
    else:
        alias_estimator_truncation = int(DATA_CONFIG.get("dual_time_alias_estimator_truncation", 1))
        alias_estimator_truncation = max(0, min(alias_estimator_truncation, int(operator_dual.integral_alias_truncation)))
    alias_estimator_mask = (
        operator_dual.frequency_alias_offsets.abs()
        <= int(alias_estimator_truncation)
    )
    alias_estimator_indices = torch.nonzero(alias_estimator_mask, as_tuple=False).view(-1)
    alias_estimator_offsets = operator_dual.frequency_alias_offsets.index_select(
        0,
        alias_estimator_indices.to(dtype=torch.int64, device=operator_dual.frequency_alias_offsets.device),
    ).to(dtype=torch.int64, device=coeff_true.device)
    xi_estimator_samples = operator_dual.xi_alias_grid.index_select(
        0,
        alias_estimator_indices.to(dtype=torch.int64, device=operator_dual.xi_alias_grid.device),
    ).reshape(-1).to(dtype=torch.float32, device=coeff_true.device)
    xi_samples = operator_dual.sampling_points.view(-1).to(dtype=torch.float32, device=coeff_true.device)
    forward_chunk_size = int(DATA_CONFIG.get("dual_time_forward_chunk_size", 512))
    chunk_size = int(DATA_CONFIG.get("dual_time_fourier_chunk_size", 512))
    beta_single = tuple(int(v.item()) for v in operator_dual.beta.view(-1))
    g_clean_time = _evaluate_dense_single_angle_time_samples(
        coeff_true,
        beta=beta_single,
        kappa0=int(operator_dual.kappa0.item()),
        d_to_lex_indices=operator_dual.d_to_lex_indices,
        t_samples=s_samples,
        chunk_size=forward_chunk_size,
    )
    g_clean_freq_estimator = _time_radon_samples_to_frequency_samples(
        g_clean_time,
        t_samples=t_beta_samples,
        xi_samples=xi_estimator_samples,
        delta_t=delta_t,
        chunk_size=chunk_size,
    )
    alias_covariance_inv = None
    if recovery_mode == "direct_ck":
        common_factor_clean_info = _solve_dual_coeff_direct_ck_from_frequency_samples(
            g_clean_freq_estimator,
            operator_dual=operator_dual,
        )
        common_factor_clean = common_factor_clean_info["kernel"]
        g_clean_freq = g_clean_freq_estimator
    else:
        alias_covariance_inv = _build_dual_time_alias_covariance_inverse(
            t_beta_samples=t_beta_samples,
            delta_t=delta_t,
            alias_offsets=alias_estimator_offsets,
        )
        common_factor_clean_info = _estimate_dual_common_factor_from_low_alias_subset(
            g_clean_freq_estimator,
            operator_dual=operator_dual,
            alias_subset_indices=alias_estimator_indices,
            alias_covariance_inv=alias_covariance_inv,
        )
        common_factor_clean = common_factor_clean_info["common_factor"]
        g_clean_freq = _synthesize_dual_alias_blocks_from_common_factor(
            common_factor_clean,
            operator_dual=operator_dual,
        )

    return {
        "operator_time": operator_time,
        "operator_dual": operator_dual,
        "s_num_samples": int(s_samples.numel()),
        "delta_t": float(delta_t),
        "delta_s": float(delta_s),
        "reference_s_num_samples": int(reference_s_samples.numel()),
        "reference_delta_s": float(reference_delta_s),
        "time_sampling_mode": str(time_sampling_info["mode"]),
        "time_sampling_t_min": float(time_sampling_info["t_min"]),
        "time_sampling_t_max": float(time_sampling_info["t_max"]),
        "s_samples": s_samples,
        "t_beta_samples": t_beta_samples,
        "alias_estimator_truncation": int(alias_estimator_truncation),
        "alias_estimator_indices": alias_estimator_indices.to(dtype=torch.int64, device=coeff_true.device),
        "alias_estimator_offsets": alias_estimator_offsets,
        "alias_covariance_inv": (
            alias_covariance_inv.to(dtype=torch.complex64, device=coeff_true.device)
            if alias_covariance_inv is not None
            else None
        ),
        "xi_estimator_samples": xi_estimator_samples,
        "xi_samples": xi_samples,
        "forward_chunk_size": int(forward_chunk_size),
        "chunk_size": int(chunk_size),
        "g_clean_time": g_clean_time,
        "g_clean_freq_estimator": g_clean_freq_estimator,
        "g_clean_common_factor": common_factor_clean,
        "g_clean_freq": g_clean_freq,
        "frequency_reliability": common_factor_clean_info["reliability"],
        "frequency_mask_weights": common_factor_clean_info["mask_weights"],
        "frequency_mask_threshold": float(common_factor_clean_info["mask_threshold"]),
        "frequency_mask_threshold_rel": float(common_factor_clean_info["mask_threshold_rel"]),
        "frequency_mask_mode": str(common_factor_clean_info["mask_mode"]),
        "frequency_mask_keep_ratio": float(common_factor_clean_info["mask_keep_ratio"]),
        "frequency_mask_mean_weight": float(common_factor_clean_info["mask_mean_weight"]),
        "frequency_max_reliability": float(common_factor_clean_info["max_reliability"]),
        "frequency_build_mode": recovery_mode,
    }


def _build_dual_frequency_observation_from_time_domain(
    frontend: Dict[str, object],
    *,
    noise_mode: str,
    delta: float,
    target_snr_db: float,
) -> Dict[str, object]:
    g_clean_time = frontend["g_clean_time"]
    g_clean_freq = frontend["g_clean_freq"]
    g_obs_time, noise_norm_time = _apply_real_measurement_noise(
        g_clean_time,
        noise_mode=noise_mode,
        delta=float(delta),
        target_snr_db=float(target_snr_db),
    )
    g_obs_freq_estimator = _time_radon_samples_to_frequency_samples(
        g_obs_time,
        t_samples=frontend["t_beta_samples"],
        xi_samples=frontend["xi_estimator_samples"],
        delta_t=float(frontend["delta_t"]),
        chunk_size=int(frontend["chunk_size"]),
    )
    recovery_mode = str(frontend["frequency_build_mode"]).strip().lower()
    if recovery_mode == "direct_ck":
        common_factor_obs_info = _solve_dual_coeff_direct_ck_from_frequency_samples(
            g_obs_freq_estimator,
            operator_dual=frontend["operator_dual"],
        )
        common_factor_obs = common_factor_obs_info["kernel"]
        g_obs_freq = g_obs_freq_estimator
    else:
        common_factor_obs_info = _estimate_dual_common_factor_from_low_alias_subset(
            g_obs_freq_estimator,
            operator_dual=frontend["operator_dual"],
            alias_subset_indices=frontend["alias_estimator_indices"],
            alias_covariance_inv=frontend["alias_covariance_inv"],
        )
        common_factor_obs = common_factor_obs_info["common_factor"]
        g_obs_freq = _synthesize_dual_alias_blocks_from_common_factor(
            common_factor_obs,
            operator_dual=frontend["operator_dual"],
        )
    noise_freq = g_obs_freq - g_clean_freq
    noise_power_freq = _mean_complex_noise_power(noise_freq)
    noise_norm_freq = _complex_noise_norm(noise_freq)
    signal_power_freq = torch.mean(torch.abs(g_clean_freq).square(), dim=1).clamp_min(1.0e-12).to(dtype=torch.float32)
    relative_noise_power = noise_power_freq / signal_power_freq
    return {
        "g_obs_time": g_obs_time,
        "g_obs_freq": g_obs_freq,
        "g_obs_freq_estimator": g_obs_freq_estimator,
        "g_obs_common_factor": common_factor_obs,
        "noise_norm_time": float(noise_norm_time),
        "noise_power_freq": noise_power_freq,
        "noise_norm_freq": float(noise_norm_freq),
        "relative_noise_power": relative_noise_power,
        "s_num_samples": int(frontend["s_num_samples"]),
        "delta_t": float(frontend["delta_t"]),
        "delta_s": float(frontend["delta_s"]),
        "alias_estimator_truncation": int(frontend["alias_estimator_truncation"]),
        "frequency_reliability": common_factor_obs_info["reliability"],
        "frequency_mask_weights": common_factor_obs_info["mask_weights"],
        "frequency_mask_threshold": float(common_factor_obs_info["mask_threshold"]),
        "frequency_mask_threshold_rel": float(common_factor_obs_info["mask_threshold_rel"]),
        "frequency_mask_mode": str(common_factor_obs_info["mask_mode"]),
        "frequency_mask_weight_norm": str(common_factor_obs_info.get("mask_weight_norm", "renorm")),
        "frequency_mask_keep_ratio": float(common_factor_obs_info["mask_keep_ratio"]),
        "frequency_mask_mean_weight": float(common_factor_obs_info["mask_mean_weight"]),
        "frequency_max_reliability": float(common_factor_obs_info["max_reliability"]),
        "frequency_build_mode": str(frontend["frequency_build_mode"]),
        "direct_ck_stability_mode": str(common_factor_obs_info.get("stability_mode", "none")),
        "direct_ck_stability_alias_mode": str(common_factor_obs_info.get("stability_alias_mode", "soft")),
        "direct_ck_stability_clip_threshold": float(common_factor_obs_info.get("stability_clip_threshold", 0.0)),
        "direct_ck_stability_clip_ratio": float(common_factor_obs_info.get("stability_clip_ratio", 0.0)),
        "direct_ck_stability_alias_residual_mean": float(common_factor_obs_info.get("stability_alias_residual_mean", 0.0)),
        "direct_ck_stability_alias_residual_q95": float(common_factor_obs_info.get("stability_alias_residual_q95", 0.0)),
        "direct_ck_stability_alias_weight_mean": float(common_factor_obs_info.get("stability_alias_weight_mean", 1.0)),
        "direct_ck_stability_alias_weight_min": float(common_factor_obs_info.get("stability_alias_weight_min", 1.0)),
        "direct_ck_stability_norm_ratio_threshold": float(common_factor_obs_info.get("stability_norm_ratio_threshold", 0.0)),
        "direct_ck_stability_norm_ratio_mean": float(common_factor_obs_info.get("stability_norm_ratio_mean", 0.0)),
        "direct_ck_stability_norm_ratio_q95": float(common_factor_obs_info.get("stability_norm_ratio_q95", 0.0)),
        "direct_ck_stability_norm_ratio_weight_mean": float(common_factor_obs_info.get("stability_norm_ratio_weight_mean", 1.0)),
        "direct_ck_stability_norm_ratio_weight_min": float(common_factor_obs_info.get("stability_norm_ratio_weight_min", 1.0)),
        "direct_ck_stability_kernel_abs_raw_max": float(common_factor_obs_info.get("stability_kernel_abs_raw_max", 0.0)),
        "direct_ck_stability_kernel_abs_after_max": float(common_factor_obs_info.get("stability_kernel_abs_after_max", 0.0)),
    }


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


def _build_current_dual_frequency_operator(
    beta_vectors: Iterable[Tuple[int, int]],
) -> FrequencyDualB1B1Operator2D:
    return FrequencyDualB1B1Operator2D(
        beta_vectors=[tuple(int(vv) for vv in beta) for beta in beta_vectors],
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_frequency_samples=int(DATA_CONFIG.get("dual_num_frequency_samples", 12 * IMAGE_SIZE * IMAGE_SIZE)),
        integral_alias_truncation=int(DATA_CONFIG.get("dual_integral_alias_truncation", 1)),
        gramian_truncation_L=int(DATA_CONFIG.get("dual_gramian_truncation_L", 8)),
        xi_grid=str(DATA_CONFIG.get("dual_xi_grid", "uniform_periodic")),
        xi_phase_shift=float(DATA_CONFIG.get("dual_xi_phase_shift", 0.0)),
    )


def _render_coeff(coeff: np.ndarray, mode_name: str) -> np.ndarray:
    if mode_name.startswith("paper_b2b1"):
        return synthesize_f_from_coeff_b2b1(coeff, image_size=PAPER_COEFF_SIZE, out_size=DISPLAY_SIZE)
    return coeff.astype(np.float32)


def _plot_mode_triptych(
    coeff_true: np.ndarray,
    coeff_est: np.ndarray,
    mode_name: str,
    noise_label: str,
    noise_suffix: str,
    lambda_reg: float,
    coeff_res: float,
    estimate_title: str = "Tikhonov",
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
    axes[1].set_title(str(estimate_title))
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, cmap="coolwarm", origin="lower", vmin=-dv, vmax=dv)
    axes[2].set_title("Difference")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.suptitle(f"{mode_name} | {noise_label} | lambda={lambda_reg:.4e} | coeff_RES={coeff_res:.6f}", y=1.05)
    out_path = os.path.join(RESULTS_DIR, f"{mode_name}_{noise_suffix}.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _evaluate_paper_b2b1(
    delta: float,
    noise_mode: str,
    target_snr_db: float,
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
    noise_label = _noise_label(noise_mode, delta=delta, target_snr_db=target_snr_db)
    noise_suffix = _noise_suffix(noise_mode, delta=delta, target_snr_db=target_snr_db)

    for idx in range(int(num_trials)):
        torch.manual_seed(int(base_seed + idx))
        g_obs, noise_norm = _apply_real_measurement_noise(
            g_clean,
            noise_mode=noise_mode,
            delta=float(delta),
            target_snr_db=float(target_snr_db),
        )
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
        noise_label=noise_label,
        noise_suffix=noise_suffix,
        lambda_reg=last_lambda,
        coeff_res=coeff_res_list[-1],
        estimate_title="Tikhonov",
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
    noise_mode: str,
    target_snr_db: float,
    num_trials: int,
    base_seed: int,
) -> Dict[str, object]:
    return _evaluate_paper_b2b1(
        delta=delta,
        noise_mode=noise_mode,
        target_snr_db=target_snr_db,
        num_trials=num_trials,
        base_seed=base_seed,
        beta_vectors=[PAPER_BETA],
        mode_name="paper_b2b1_single",
    )


def _evaluate_paper_b2b1_multi8(
    delta: float,
    noise_mode: str,
    target_snr_db: float,
    num_trials: int,
    base_seed: int,
) -> Dict[str, object]:
    return _evaluate_paper_b2b1(
        delta=delta,
        noise_mode=noise_mode,
        target_snr_db=target_snr_db,
        num_trials=num_trials,
        base_seed=base_seed,
        beta_vectors=_paper_multi8_betas(),
        mode_name="paper_b2b1_multi8",
    )


def _evaluate_current_b1b1(
    phantom_128: torch.Tensor,
    delta: float,
    noise_mode: str,
    target_snr_db: float,
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
    noise_label = _noise_label(noise_mode, delta=delta, target_snr_db=target_snr_db)
    noise_suffix = _noise_suffix(noise_mode, delta=delta, target_snr_db=target_snr_db)

    for idx in range(int(num_trials)):
        torch.manual_seed(int(base_seed + idx))
        g_obs, noise_norm = _apply_real_measurement_noise(
            g_clean,
            noise_mode=noise_mode,
            delta=float(delta),
            target_snr_db=float(target_snr_db),
        )
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
        noise_label=noise_label,
        noise_suffix=noise_suffix,
        lambda_reg=last_lambda,
        coeff_res=coeff_res_list[-1],
        estimate_title="Tikhonov",
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


def _evaluate_current_b1b1_dual_frequency(
    phantom_128: torch.Tensor,
    delta: float,
    noise_mode: str,
    target_snr_db: float,
    num_trials: int,
    base_seed: int,
    beta_vectors: Iterable[Tuple[int, int]],
    mode_name: str,
) -> Dict[str, object]:
    coeff_true = phantom_128.view(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device=device, dtype=torch.float32)
    noise_domain = _current_dual_noise_domain()
    operator = _build_current_dual_frequency_operator(beta_vectors)
    time_frontend: Dict[str, object] | None = None
    s_num_samples = 0
    delta_s = float("nan")
    delta_t = float("nan")
    if noise_domain == "time_radon_samples":
        time_frontend = _prepare_dual_time_domain_frontend(
            coeff_true,
            beta_vectors=beta_vectors,
            operator_dual=operator,
        )
        g_clean = time_frontend["g_clean_freq"].to(dtype=torch.complex64)
        s_num_samples = int(time_frontend["s_num_samples"])
        delta_s = float(time_frontend["delta_s"])
        delta_t = float(time_frontend["delta_t"])
    else:
        g_clean = operator.forward(coeff_true).to(dtype=torch.complex64)

    stabilizers: List[float] = []
    noise_norms: List[float] = []
    noise_norms_time: List[float] = []
    noise_norms_freq: List[float] = []
    residual_norms: List[float] = []
    coeff_res_list: List[float] = []
    last_est = coeff_true.squeeze().detach().cpu().numpy()
    last_stabilizer = 0.0
    noise_label = _noise_label(noise_mode, delta=delta, target_snr_db=target_snr_db)
    noise_suffix = _noise_suffix(noise_mode, delta=delta, target_snr_db=target_snr_db)

    for idx in range(int(num_trials)):
        torch.manual_seed(int(base_seed + idx))
        if noise_domain == "time_radon_samples":
            assert time_frontend is not None
            obs = _build_dual_frequency_observation_from_time_domain(
                time_frontend,
                noise_mode=noise_mode,
                delta=float(delta),
                target_snr_db=float(target_snr_db),
            )
            g_obs = obs["g_obs_freq"].to(dtype=torch.complex64)
            relative_noise_power = obs["relative_noise_power"].to(dtype=torch.float32)
            noise_norm = float(obs["noise_norm_time"])
            noise_norm_time = float(obs["noise_norm_time"])
            noise_norm_freq = float(obs["noise_norm_freq"])
            coeff_est = _solve_dual_coeff_from_common_factor(
                obs["g_obs_common_factor"].to(dtype=torch.complex64),
                operator_dual=operator,
            )
            residual = operator.forward(coeff_est) - g_obs
            stabilizer = 0.0
        else:
            g_obs, noise_power, noise_norm_freq = _apply_frequency_sample_noise(
                g_clean,
                noise_mode=noise_mode,
                delta=float(delta),
                target_snr_db=float(target_snr_db),
            )
            signal_power = torch.mean(torch.abs(g_clean).square(), dim=1).clamp_min(1.0e-12)
            relative_noise_power = noise_power / signal_power
            noise_norm = float(noise_norm_freq)
            noise_norm_time = float("nan")
            coeff_est = operator.solve_dual_frame_direct(
                g_obs,
                noise_power=relative_noise_power,
            )
            residual = operator.forward(coeff_est) - g_obs
            stabilizer = float(operator.last_dual_lambda.mean().item()) if operator.last_dual_lambda is not None else 0.0
        stabilizers.append(stabilizer)
        noise_norms.append(float(noise_norm))
        noise_norms_time.append(float(noise_norm_time))
        noise_norms_freq.append(float(noise_norm_freq))
        residual_norms.append(float(torch.linalg.vector_norm(residual.reshape(residual.shape[0], -1), dim=1).mean().item()))
        coeff_res_list.append(_coeff_res(coeff_est.squeeze(), coeff_true.squeeze()))
        last_est = coeff_est.squeeze().detach().cpu().numpy()
        last_stabilizer = stabilizer

    plot_path = _plot_mode_triptych(
        coeff_true.squeeze().detach().cpu().numpy(),
        last_est,
        mode_name=mode_name,
        noise_label=noise_label,
        noise_suffix=noise_suffix,
        lambda_reg=last_stabilizer,
        coeff_res=coeff_res_list[-1],
        estimate_title="Dual Frequency",
    )
    return {
        "mode": mode_name,
        "noise_domain": noise_domain,
        "coeff_true": coeff_true.squeeze().detach().cpu().numpy(),
        "coeff_est": last_est,
        "lambda_mean": float(np.mean(stabilizers)),
        "noise_norm_mean": float(np.mean(noise_norms)),
        "noise_norm_time_mean": float(np.mean(noise_norms_time)),
        "noise_norm_freq_mean": float(np.mean(noise_norms_freq)),
        "meas_res_mean": float(np.mean(residual_norms)),
        "coeff_res_mean": float(np.mean(coeff_res_list)),
        "s_num_samples": int(s_num_samples),
        "delta_s": float(delta_s),
        "delta_t": float(delta_t),
        "plot_path": plot_path,
    }


def _print_summary(results: List[Dict[str, object]], noise_label: str, num_trials: int) -> None:
    print("=" * 92)
    print(f"Pure Tikhonov comparison | {noise_label} | trials={num_trials}")
    print("=" * 92)
    print(f"{'Mode':<36} | {'Mean lambda':>12} | {'Mean noise':>12} | {'Mean meas res':>14} | {'Mean coeff RES':>14}")
    print("-" * 92)
    for item in results:
        print(
            f"{item['mode']:<36} | "
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


def _run_mode(
    mode: str,
    phantom_128: torch.Tensor,
    delta: float,
    noise_mode: str,
    target_snr_db: float,
    num_trials: int,
    base_seed: int,
) -> Dict[str, object]:
    if mode == "paper_b2b1_single":
        return _evaluate_paper_b2b1_single(
            delta=delta,
            noise_mode=noise_mode,
            target_snr_db=target_snr_db,
            num_trials=num_trials,
            base_seed=base_seed,
        )
    if mode == "paper_b2b1_multi8":
        return _evaluate_paper_b2b1_multi8(
            delta=delta,
            noise_mode=noise_mode,
            target_snr_db=target_snr_db,
            num_trials=num_trials,
            base_seed=base_seed,
        )
    if mode in ("current_b1b1_single", "ellipse_b1b1_single"):
        return _evaluate_current_b1b1(
            phantom_128=phantom_128,
            delta=delta,
            noise_mode=noise_mode,
            target_snr_db=target_snr_db,
            num_trials=num_trials,
            base_seed=base_seed,
            beta_vectors=[_build_current_single_beta()],
            mode_name="ellipse_b1b1_single",
        )
    if mode in ("current_b1b1_multi8", "ellipse_b1b1_multi8"):
        return _evaluate_current_b1b1(
            phantom_128=phantom_128,
            delta=delta,
            noise_mode=noise_mode,
            target_snr_db=target_snr_db,
            num_trials=num_trials,
            base_seed=base_seed,
            beta_vectors=_build_current_multi8_betas(),
            mode_name="ellipse_b1b1_multi8",
        )
    if mode == "current_b1b1_dual_frequency_single":
        return _evaluate_current_b1b1_dual_frequency(
            phantom_128=phantom_128,
            delta=delta,
            noise_mode=noise_mode,
            target_snr_db=target_snr_db,
            num_trials=num_trials,
            base_seed=base_seed,
            beta_vectors=[_build_current_single_beta()],
            mode_name="current_b1b1_dual_frequency_single",
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
            "current_b1b1_dual_frequency_single",
            "dual_compare_single",
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
    parser.add_argument(
        "--noise_mode",
        type=str,
        default=str(DATA_CONFIG.get("noise_mode", "multiplicative")).strip().lower(),
        choices=["multiplicative", "snr"],
        help="Noise mode for the comparison chain.",
    )
    parser.add_argument("--delta", type=float, default=0.1, help="Multiplicative noise level delta.")
    parser.add_argument(
        "--target_snr_db",
        type=float,
        default=float(DATA_CONFIG.get("target_snr_db", 30.0)),
        help="Target SNR in dB when noise_mode=snr.",
    )
    parser.add_argument("--num_trials", type=int, default=1, help="Number of noise trials on the same ellipse phantom.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for the shared ellipse phantom.")
    args = parser.parse_args()

    print(f"Running pure Tikhonov comparison on device: {device}")
    noise_mode = str(args.noise_mode).strip().lower()
    noise_label = _noise_label(noise_mode, delta=float(args.delta), target_snr_db=float(args.target_snr_db))
    print(
        f"mode={args.mode}, compare_mode={args.compare_mode}, noise={noise_label}, "
        f"num_trials={args.num_trials}, seed={args.seed}"
    )

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    phantom_128 = generate_random_ellipse_phantom(image_size=IMAGE_SIZE).to(dtype=torch.float32)

    if args.mode == "compare":
        modes = ["paper_b2b1_single", "paper_b2b1_multi8"]
        if args.compare_mode in ("ellipse_b1b1_single", "both"):
            modes.append("ellipse_b1b1_single")
        if args.compare_mode in ("ellipse_b1b1_multi8", "both"):
            modes.append("ellipse_b1b1_multi8")
    elif args.mode == "dual_compare_single":
        modes = ["ellipse_b1b1_single", "current_b1b1_dual_frequency_single"]
    else:
        modes = [args.mode]

    results = [
        _run_mode(
            mode=mode,
            phantom_128=phantom_128,
            delta=float(args.delta),
            noise_mode=noise_mode,
            target_snr_db=float(args.target_snr_db),
            num_trials=int(args.num_trials),
            base_seed=int(args.seed) + 1000,
        )
        for mode in modes
    ]
    _print_summary(results, noise_label=noise_label, num_trials=int(args.num_trials))


if __name__ == "__main__":
    main()
