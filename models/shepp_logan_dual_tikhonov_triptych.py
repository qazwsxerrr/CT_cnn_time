"""Generate a 1x3 triptych: True / Dual Frequency / Tikhonov on Shepp-Logan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_CONFIG, IMAGE_SIZE, RESULTS_DIR
from image_generator import generate_shepp_logan_phantom
import tikhonov_eval as te


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Shepp-Logan 1x3 triptych: True / Dual Frequency / Tikhonov."
    )
    parser.add_argument("--beta_x", type=int, default=1, help="First entry of the single-angle beta vector.")
    parser.add_argument(
        "--beta_y",
        type=int,
        default=int(IMAGE_SIZE),
        help="Second entry of the single-angle beta vector.",
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default=str(DATA_CONFIG.get("noise_mode", "multiplicative")).strip().lower(),
        choices=["multiplicative", "snr"],
        help="Noise mode for both Tikhonov and Dual Frequency pipelines.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=float(DATA_CONFIG.get("noise_level", 0.1)),
        help="Multiplicative noise level delta when noise_mode=multiplicative.",
    )
    parser.add_argument(
        "--target_snr_db",
        type=float,
        default=float(DATA_CONFIG.get("target_snr_db", 30.0)),
        help="Target SNR in dB when noise_mode=snr.",
    )
    parser.add_argument(
        "--P",
        type=int,
        default=int(DATA_CONFIG.get("dual_num_frequency_samples", 12 * IMAGE_SIZE * IMAGE_SIZE)),
        help="Principal-cell frequency sample count P.",
    )
    parser.add_argument(
        "--dual_xi_phase_shift",
        type=float,
        default=float(DATA_CONFIG.get("dual_xi_phase_shift", 0.0)),
        help="Principal-cell phase shift theta in xi_j=-pi+theta+2*pi*j/P.",
    )
    parser.add_argument(
        "--Q",
        type=int,
        default=int(DATA_CONFIG.get("dual_integral_alias_truncation", 1)),
        help="Alias-period truncation Q for the Dual Frequency numerator.",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=int(DATA_CONFIG.get("dual_gramian_truncation_L", 8)),
        help="Gramian truncation level L for the Dual Frequency denominator.",
    )
    parser.add_argument(
        "--dual_noise_domain",
        type=str,
        default=str(DATA_CONFIG.get("dual_noise_domain", "spectral_samples")).strip().lower(),
        choices=["spectral_samples", "time_radon_samples"],
        help="Where to inject Dual noise: directly on spectral samples or first on time-domain Radon samples.",
    )
    parser.add_argument(
        "--dual_time_num_samples",
        type=int,
        default=int(DATA_CONFIG.get("dual_time_num_samples", IMAGE_SIZE * IMAGE_SIZE)),
        help="Dense time-domain sample count N_t used when dual_noise_domain=time_radon_samples.",
    )
    parser.add_argument(
        "--Qe",
        type=int,
        default=int(DATA_CONFIG.get("dual_time_alias_estimator_truncation", 1)),
        help="Low-alias subset truncation Q_e used by the time-domain Dual estimator.",
    )
    parser.add_argument(
        "--dual_time_recovery_mode",
        type=str,
        default=str(DATA_CONFIG.get("dual_time_recovery_mode", "gls_low_alias")).strip().lower(),
        choices=["simple_ratio", "gls_low_alias", "direct_ck"],
        help="Recovery mode for time-domain-noise Dual reconstruction.",
    )
    parser.add_argument(
        "--dual_time_sampling_interval_mode",
        type=str,
        default=str(DATA_CONFIG.get("dual_time_sampling_interval_mode", "theorem_grid")).strip().lower(),
        choices=[
            "theorem_grid",
            "beta_support_midpoint",
        ],
        help="Time grid interval used before time-to-frequency conversion.",
    )
    parser.add_argument(
        "--dual_time_fourier_quadrature",
        type=str,
        default=str(DATA_CONFIG.get("dual_time_fourier_quadrature", "rectangle")).strip().lower(),
        choices=["rectangle", "midpoint_cell"],
        help="Quadrature rule for converting time-domain Radon samples to frequency samples.",
    )
    parser.add_argument(
        "--dual_time_gls_lambda_rel",
        type=float,
        default=float(DATA_CONFIG.get("dual_time_gls_lambda_rel", 1.0e-3)),
        help="Relative ridge used in the GLS denominator for time-domain Dual recovery.",
    )
    parser.add_argument(
        "--dual_time_gls_covariance_ridge_rel",
        type=float,
        default=float(DATA_CONFIG.get("dual_time_gls_covariance_ridge_rel", 1.0e-10)),
        help="Relative ridge used when inverting the low-alias GLS covariance block.",
    )
    parser.add_argument(
        "--dual_time_frequency_mask_rel",
        type=float,
        default=float(DATA_CONFIG.get("dual_time_frequency_mask_rel", 0.0)),
        help="Relative reliability threshold for suppressing bad principal-cell frequencies.",
    )
    parser.add_argument(
        "--dual_time_frequency_mask_mode",
        type=str,
        default=str(DATA_CONFIG.get("dual_time_frequency_mask_mode", "soft")).strip().lower(),
        choices=["hard", "soft"],
        help="How to suppress bad principal-cell frequencies: hard zeroing or soft attenuation.",
    )
    parser.add_argument(
        "--dual_direct_ck_gramian_mask_quantile",
        type=float,
        default=float(DATA_CONFIG.get("dual_direct_ck_gramian_mask_quantile", 0.0)),
        help="direct_ck only: discard/downweight the lowest rho_G fraction of Gramian frequencies.",
    )
    parser.add_argument(
        "--dual_direct_ck_mask_mode",
        type=str,
        default=str(DATA_CONFIG.get("dual_direct_ck_mask_mode", "hard")).strip().lower(),
        choices=["hard", "soft"],
        help="direct_ck only: hard mask or soft G/tau weight for low-Gramian frequencies.",
    )
    parser.add_argument(
        "--dual_direct_ck_weight_norm",
        type=str,
        default=str(DATA_CONFIG.get("dual_direct_ck_weight_norm", "none")).strip().lower(),
        choices=["none", "renorm"],
        help="direct_ck only: keep no-renorm integral scale or renormalize by mean mask weight.",
    )
    parser.add_argument(
        "--dual_direct_ck_lambda_rel",
        type=float,
        default=float(DATA_CONFIG.get("dual_direct_ck_lambda_rel", 0.0)),
        help="direct_ck only: relative stabilizer added to G_{beta,L}(xi_j).",
    )
    parser.add_argument(
        "--dual_direct_ck_stability_mode",
        type=str,
        default=str(DATA_CONFIG.get("dual_direct_ck_stability_mode", "none")).strip().lower(),
        choices=["none", "kclip", "alias_consistency", "kclip_alias", "norm_ratio"],
        help="direct_ck only: optional g/h alias stability control.",
    )
    parser.add_argument(
        "--dual_direct_ck_clip_good_g_rel",
        type=float,
        default=float(DATA_CONFIG.get("dual_direct_ck_clip_good_g_rel", 0.1)),
        help="direct_ck stability: high-Gramian relative threshold used to estimate the clipping bound.",
    )
    parser.add_argument(
        "--dual_direct_ck_clip_quantile",
        type=float,
        default=float(DATA_CONFIG.get("dual_direct_ck_clip_quantile", 0.99)),
        help="direct_ck stability: quantile of |K_raw| on good frequencies used for clipping.",
    )
    parser.add_argument(
        "--dual_direct_ck_clip_scale",
        type=float,
        default=float(DATA_CONFIG.get("dual_direct_ck_clip_scale", 1.2)),
        help="direct_ck stability: multiplicative scale applied to the clipping quantile.",
    )
    parser.add_argument(
        "--dual_direct_ck_alias_tau",
        type=float,
        default=float(DATA_CONFIG.get("dual_direct_ck_alias_tau", 0.3)),
        help="direct_ck stability: tau for alias consistency soft/hard weighting.",
    )
    parser.add_argument(
        "--dual_direct_ck_alias_mode",
        type=str,
        default=str(DATA_CONFIG.get("dual_direct_ck_alias_mode", "soft")).strip().lower(),
        choices=["soft", "hard"],
        help="direct_ck stability: alias consistency weighting mode.",
    )
    parser.add_argument(
        "--dual_direct_ck_norm_ratio_window",
        type=int,
        default=int(DATA_CONFIG.get("dual_direct_ck_norm_ratio_window", 512)),
        help="direct_ck norm_ratio: local median window size used to estimate the robust norm-ratio bound.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed used for deterministic noise generation.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path. Defaults to results/shepp_logan_true_dual_tikhonov_<...>.png",
    )
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Run the experiment and print JSON metrics without writing the triptych image.",
    )
    return parser.parse_args()


def _resolve_output_path(args: argparse.Namespace) -> str:
    if args.output is not None and str(args.output).strip():
        return os.path.abspath(str(args.output))
    mode = str(args.noise_mode).strip().lower()
    noise_domain = str(args.dual_noise_domain).strip().lower()
    if mode == "multiplicative":
        token = f"{float(args.delta):g}".replace("-", "m").replace(".", "_")
        noise_suffix = f"delta_{token}"
    elif mode == "snr":
        token = f"{float(args.target_snr_db):g}".replace("-", "m").replace(".", "_")
        noise_suffix = f"snr_{token}db"
    else:
        raise ValueError(f"Unsupported noise_mode={args.noise_mode!r}; expected 'multiplicative' or 'snr'.")
    shift_suffix = ""
    if abs(float(args.dual_xi_phase_shift)) > 0.0:
        token = f"{float(args.dual_xi_phase_shift):g}".replace("-", "m").replace(".", "_")
        shift_suffix = f"_shift_{token}"
    mask_suffix = ""
    if noise_domain == "time_radon_samples" and float(args.dual_time_frequency_mask_rel) > 0.0:
        token = f"{float(args.dual_time_frequency_mask_rel):g}".replace("-", "m").replace(".", "_")
        mask_suffix = f"_mask_{str(args.dual_time_frequency_mask_mode).strip().lower()}_{token}"
    grid_suffix = ""
    if noise_domain == "time_radon_samples":
        grid_token = str(args.dual_time_sampling_interval_mode).strip().lower()
        if grid_token not in {"theorem_grid"}:
            grid_suffix = f"_grid_{grid_token}"
    quadrature_suffix = ""
    if noise_domain == "time_radon_samples" and str(args.dual_time_fourier_quadrature).strip().lower() != "rectangle":
        quadrature_suffix = f"_quad_{str(args.dual_time_fourier_quadrature).strip().lower()}"
    direct_ck_suffix = ""
    if noise_domain == "time_radon_samples" and str(args.dual_time_recovery_mode).strip().lower() == "direct_ck":
        rho_token = f"{float(args.dual_direct_ck_gramian_mask_quantile):g}".replace("-", "m").replace(".", "_")
        direct_ck_suffix = (
            f"_gq_{rho_token}_{str(args.dual_direct_ck_mask_mode).strip().lower()}"
            f"_{str(args.dual_direct_ck_weight_norm).strip().lower()}"
        )
        if float(args.dual_direct_ck_lambda_rel) > 0.0:
            lam_token = f"{float(args.dual_direct_ck_lambda_rel):g}".replace("-", "m").replace(".", "_")
            direct_ck_suffix += f"_lrel_{lam_token}"
        stability_mode = str(args.dual_direct_ck_stability_mode).strip().lower()
        if stability_mode != "none":
            q_token = f"{float(args.dual_direct_ck_clip_quantile):g}".replace("-", "m").replace(".", "_")
            s_token = f"{float(args.dual_direct_ck_clip_scale):g}".replace("-", "m").replace(".", "_")
            tau_token = f"{float(args.dual_direct_ck_alias_tau):g}".replace("-", "m").replace(".", "_")
            direct_ck_suffix += (
                f"_stab_{stability_mode}_q{q_token}_s{s_token}"
                f"_tau{tau_token}_{str(args.dual_direct_ck_alias_mode).strip().lower()}"
            )
    return os.path.join(
        RESULTS_DIR,
        "shepp_logan_true_dual_tikhonov_"
        f"{noise_suffix}_noise_domain_{noise_domain}"
        + (f"_Nt_{int(args.dual_time_num_samples)}" if noise_domain == "time_radon_samples" else "")
        + (f"_Qe_{int(args.Qe)}" if noise_domain == "time_radon_samples" else "")
        + (f"_{str(args.dual_time_recovery_mode).strip().lower()}" if noise_domain == "time_radon_samples" else "")
        + shift_suffix
        + grid_suffix
        + mask_suffix
        + quadrature_suffix
        + direct_ck_suffix
        + f"_P_{int(args.P)}_Q_{int(args.Q)}_L_{int(args.L)}.png",
    )


def _build_beta(args: argparse.Namespace) -> Tuple[int, int]:
    beta = (int(args.beta_x), int(args.beta_y))
    te._to_integer_beta(beta)
    return beta


def _configure_dual_frequency(args: argparse.Namespace) -> Dict[str, object]:
    old = {
        "dual_num_frequency_samples": DATA_CONFIG.get("dual_num_frequency_samples"),
        "dual_integral_alias_truncation": DATA_CONFIG.get("dual_integral_alias_truncation"),
        "dual_gramian_truncation_L": DATA_CONFIG.get("dual_gramian_truncation_L"),
        "dual_xi_grid": DATA_CONFIG.get("dual_xi_grid"),
        "dual_xi_phase_shift": DATA_CONFIG.get("dual_xi_phase_shift"),
        "dual_noise_domain": DATA_CONFIG.get("dual_noise_domain"),
        "dual_time_num_samples": DATA_CONFIG.get("dual_time_num_samples"),
        "dual_time_recovery_mode": DATA_CONFIG.get("dual_time_recovery_mode"),
        "dual_time_sampling_interval_mode": DATA_CONFIG.get("dual_time_sampling_interval_mode"),
        "dual_time_fourier_quadrature": DATA_CONFIG.get("dual_time_fourier_quadrature"),
        "dual_time_alias_estimator_truncation": DATA_CONFIG.get("dual_time_alias_estimator_truncation"),
        "dual_time_gls_lambda_rel": DATA_CONFIG.get("dual_time_gls_lambda_rel"),
        "dual_time_gls_covariance_ridge_rel": DATA_CONFIG.get("dual_time_gls_covariance_ridge_rel"),
        "dual_time_frequency_mask_rel": DATA_CONFIG.get("dual_time_frequency_mask_rel"),
        "dual_time_frequency_mask_mode": DATA_CONFIG.get("dual_time_frequency_mask_mode"),
        "dual_direct_ck_gramian_mask_quantile": DATA_CONFIG.get("dual_direct_ck_gramian_mask_quantile"),
        "dual_direct_ck_mask_mode": DATA_CONFIG.get("dual_direct_ck_mask_mode"),
        "dual_direct_ck_weight_norm": DATA_CONFIG.get("dual_direct_ck_weight_norm"),
        "dual_direct_ck_lambda_rel": DATA_CONFIG.get("dual_direct_ck_lambda_rel"),
        "dual_direct_ck_stability_mode": DATA_CONFIG.get("dual_direct_ck_stability_mode"),
        "dual_direct_ck_clip_good_g_rel": DATA_CONFIG.get("dual_direct_ck_clip_good_g_rel"),
        "dual_direct_ck_clip_quantile": DATA_CONFIG.get("dual_direct_ck_clip_quantile"),
        "dual_direct_ck_clip_scale": DATA_CONFIG.get("dual_direct_ck_clip_scale"),
        "dual_direct_ck_alias_tau": DATA_CONFIG.get("dual_direct_ck_alias_tau"),
        "dual_direct_ck_alias_mode": DATA_CONFIG.get("dual_direct_ck_alias_mode"),
        "dual_direct_ck_norm_ratio_window": DATA_CONFIG.get("dual_direct_ck_norm_ratio_window"),
    }
    DATA_CONFIG["dual_num_frequency_samples"] = int(args.P)
    DATA_CONFIG["dual_integral_alias_truncation"] = int(args.Q)
    DATA_CONFIG["dual_gramian_truncation_L"] = int(args.L)
    DATA_CONFIG["dual_xi_grid"] = "uniform_periodic"
    DATA_CONFIG["dual_xi_phase_shift"] = float(args.dual_xi_phase_shift)
    DATA_CONFIG["dual_noise_domain"] = str(args.dual_noise_domain).strip().lower()
    DATA_CONFIG["dual_time_num_samples"] = int(args.dual_time_num_samples)
    DATA_CONFIG["dual_time_recovery_mode"] = str(args.dual_time_recovery_mode).strip().lower()
    DATA_CONFIG["dual_time_sampling_interval_mode"] = str(args.dual_time_sampling_interval_mode).strip().lower()
    DATA_CONFIG["dual_time_fourier_quadrature"] = str(args.dual_time_fourier_quadrature).strip().lower()
    DATA_CONFIG["dual_time_alias_estimator_truncation"] = int(args.Qe)
    DATA_CONFIG["dual_time_gls_lambda_rel"] = float(args.dual_time_gls_lambda_rel)
    DATA_CONFIG["dual_time_gls_covariance_ridge_rel"] = float(args.dual_time_gls_covariance_ridge_rel)
    DATA_CONFIG["dual_time_frequency_mask_rel"] = float(args.dual_time_frequency_mask_rel)
    DATA_CONFIG["dual_time_frequency_mask_mode"] = str(args.dual_time_frequency_mask_mode).strip().lower()
    DATA_CONFIG["dual_direct_ck_gramian_mask_quantile"] = float(args.dual_direct_ck_gramian_mask_quantile)
    DATA_CONFIG["dual_direct_ck_mask_mode"] = str(args.dual_direct_ck_mask_mode).strip().lower()
    DATA_CONFIG["dual_direct_ck_weight_norm"] = str(args.dual_direct_ck_weight_norm).strip().lower()
    DATA_CONFIG["dual_direct_ck_lambda_rel"] = float(args.dual_direct_ck_lambda_rel)
    DATA_CONFIG["dual_direct_ck_stability_mode"] = str(args.dual_direct_ck_stability_mode).strip().lower()
    DATA_CONFIG["dual_direct_ck_clip_good_g_rel"] = float(args.dual_direct_ck_clip_good_g_rel)
    DATA_CONFIG["dual_direct_ck_clip_quantile"] = float(args.dual_direct_ck_clip_quantile)
    DATA_CONFIG["dual_direct_ck_clip_scale"] = float(args.dual_direct_ck_clip_scale)
    DATA_CONFIG["dual_direct_ck_alias_tau"] = float(args.dual_direct_ck_alias_tau)
    DATA_CONFIG["dual_direct_ck_alias_mode"] = str(args.dual_direct_ck_alias_mode).strip().lower()
    DATA_CONFIG["dual_direct_ck_norm_ratio_window"] = int(args.dual_direct_ck_norm_ratio_window)
    return old


def _restore_dual_frequency(old: Dict[str, object]) -> None:
    for key, value in old.items():
        DATA_CONFIG[key] = value


@torch.no_grad()
def _run_once(args: argparse.Namespace) -> Dict[str, object]:
    noise_mode = str(args.noise_mode).strip().lower()
    noise_domain = str(args.dual_noise_domain).strip().lower()
    beta = _build_beta(args)
    noise_seed = int(args.seed) + 1000

    phantom = generate_shepp_logan_phantom(image_size=IMAGE_SIZE).to(dtype=torch.float32)
    coeff_true = phantom.view(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device=te.device, dtype=torch.float32)

    operator_tikh = te._build_current_b1b1_operator([beta])
    operator_dual = te._build_current_dual_frequency_operator([beta])

    dual_s_num_samples = 0
    dual_reference_s_num_samples = 0
    dual_delta_s = float("nan")
    dual_reference_delta_s = float("nan")
    dual_delta_t = float("nan")
    dual_time_sampling_mode = str(args.dual_time_sampling_interval_mode).strip().lower()
    dual_time_sampling_t_min = float("nan")
    dual_time_sampling_t_max = float("nan")
    dual_alias_estimator_truncation = 0
    dual_frequency_build_mode = "direct_spectral_samples"
    noise_norm_time = float("nan")
    freq_rel_err_time_vs_direct_clean = float("nan")
    res_direct_noiseless = float("nan")
    frequency_mask_threshold = 0.0
    frequency_mask_threshold_rel = 0.0
    frequency_mask_keep_ratio = 1.0
    frequency_mask_mean_weight = 1.0
    frequency_mask_weight_norm = "renorm"
    frequency_mask_mode = str(args.dual_time_frequency_mask_mode).strip().lower()
    direct_ck_stability_mode = str(args.dual_direct_ck_stability_mode).strip().lower()
    direct_ck_stability_alias_mode = str(args.dual_direct_ck_alias_mode).strip().lower()
    direct_ck_stability_clip_threshold = 0.0
    direct_ck_stability_clip_ratio = 0.0
    direct_ck_stability_alias_residual_mean = 0.0
    direct_ck_stability_alias_residual_q95 = 0.0
    direct_ck_stability_alias_weight_mean = 1.0
    direct_ck_stability_alias_weight_min = 1.0
    direct_ck_stability_norm_ratio_threshold = 0.0
    direct_ck_stability_norm_ratio_mean = 0.0
    direct_ck_stability_norm_ratio_q95 = 0.0
    direct_ck_stability_norm_ratio_weight_mean = 1.0
    direct_ck_stability_norm_ratio_weight_min = 1.0
    direct_ck_stability_kernel_abs_raw_max = 0.0
    direct_ck_stability_kernel_abs_after_max = 0.0
    if noise_domain == "time_radon_samples":
        dual_frontend = te._prepare_dual_time_domain_frontend(
            coeff_true,
            beta_vectors=[beta],
            operator_time=operator_tikh,
            operator_dual=operator_dual,
        )
        tikh_t_samples = operator_tikh.sampling_points.view(-1).to(dtype=torch.float32, device=te.device)
        g_clean_time_dense = dual_frontend["g_clean_time"].to(dtype=torch.float32)
        g_clean_real = te._interpolate_time_samples(
            g_clean_time_dense,
            source_t=dual_frontend["s_samples"],
            target_t=tikh_t_samples,
        ).to(dtype=torch.float32)
        torch.manual_seed(noise_seed)
        dual_obs = te._build_dual_frequency_observation_from_time_domain(
            dual_frontend,
            noise_mode=noise_mode,
            delta=float(args.delta),
            target_snr_db=float(args.target_snr_db),
        )
        g_obs_time_dense = dual_obs["g_obs_time"].to(dtype=torch.float32)
        g_obs_real = te._interpolate_time_samples(
            g_obs_time_dense,
            source_t=dual_frontend["s_samples"],
            target_t=tikh_t_samples,
        ).to(dtype=torch.float32)
        noise_real = g_obs_real - g_clean_real
        noise_norm_real = float(
            torch.linalg.vector_norm(noise_real.reshape(noise_real.shape[0], -1), dim=1).mean().item()
        )
        noise_norm_time = float(dual_obs["noise_norm_time"])
        g_obs_freq = dual_obs["g_obs_freq"].to(dtype=torch.complex64)
        relative_noise_power = dual_obs["relative_noise_power"].to(dtype=torch.float32)
        noise_norm_freq = float(dual_obs["noise_norm_freq"])
        dual_s_num_samples = int(dual_obs["s_num_samples"])
        dual_delta_s = float(dual_obs["delta_s"])
        dual_delta_t = float(dual_obs["delta_t"])
        dual_reference_s_num_samples = int(dual_frontend["reference_s_num_samples"])
        dual_reference_delta_s = float(dual_frontend["reference_delta_s"])
        dual_time_sampling_mode = str(dual_frontend["time_sampling_mode"])
        dual_time_sampling_t_min = float(dual_frontend["time_sampling_t_min"])
        dual_time_sampling_t_max = float(dual_frontend["time_sampling_t_max"])
        dual_alias_estimator_truncation = int(dual_obs["alias_estimator_truncation"])
        dual_frequency_build_mode = str(dual_obs["frequency_build_mode"])
        frequency_mask_threshold = float(dual_obs["frequency_mask_threshold"])
        frequency_mask_threshold_rel = float(dual_obs["frequency_mask_threshold_rel"])
        frequency_mask_keep_ratio = float(dual_obs["frequency_mask_keep_ratio"])
        frequency_mask_mean_weight = float(dual_obs["frequency_mask_mean_weight"])
        frequency_mask_weight_norm = str(dual_obs.get("frequency_mask_weight_norm", "renorm"))
        frequency_mask_mode = str(dual_obs["frequency_mask_mode"])
        direct_ck_stability_mode = str(dual_obs.get("direct_ck_stability_mode", direct_ck_stability_mode))
        direct_ck_stability_alias_mode = str(dual_obs.get("direct_ck_stability_alias_mode", direct_ck_stability_alias_mode))
        direct_ck_stability_clip_threshold = float(dual_obs.get("direct_ck_stability_clip_threshold", 0.0))
        direct_ck_stability_clip_ratio = float(dual_obs.get("direct_ck_stability_clip_ratio", 0.0))
        direct_ck_stability_alias_residual_mean = float(dual_obs.get("direct_ck_stability_alias_residual_mean", 0.0))
        direct_ck_stability_alias_residual_q95 = float(dual_obs.get("direct_ck_stability_alias_residual_q95", 0.0))
        direct_ck_stability_alias_weight_mean = float(dual_obs.get("direct_ck_stability_alias_weight_mean", 1.0))
        direct_ck_stability_alias_weight_min = float(dual_obs.get("direct_ck_stability_alias_weight_min", 1.0))
        direct_ck_stability_norm_ratio_threshold = float(dual_obs.get("direct_ck_stability_norm_ratio_threshold", 0.0))
        direct_ck_stability_norm_ratio_mean = float(dual_obs.get("direct_ck_stability_norm_ratio_mean", 0.0))
        direct_ck_stability_norm_ratio_q95 = float(dual_obs.get("direct_ck_stability_norm_ratio_q95", 0.0))
        direct_ck_stability_norm_ratio_weight_mean = float(dual_obs.get("direct_ck_stability_norm_ratio_weight_mean", 1.0))
        direct_ck_stability_norm_ratio_weight_min = float(dual_obs.get("direct_ck_stability_norm_ratio_weight_min", 1.0))
        direct_ck_stability_kernel_abs_raw_max = float(dual_obs.get("direct_ck_stability_kernel_abs_raw_max", 0.0))
        direct_ck_stability_kernel_abs_after_max = float(dual_obs.get("direct_ck_stability_kernel_abs_after_max", 0.0))
    else:
        torch.manual_seed(noise_seed)
        g_clean_real = operator_tikh.forward(coeff_true).to(dtype=torch.float32)
        g_obs_real, noise_norm_real = te._apply_real_measurement_noise(
            g_clean_real,
            noise_mode=noise_mode,
            delta=float(args.delta),
            target_snr_db=float(args.target_snr_db),
        )
    lam = operator_tikh.choose_lambda_morozov(
        g_obs_real,
        noise_norm=torch.tensor([noise_norm_real], device=g_obs_real.device, dtype=g_obs_real.dtype),
        tau=float(DATA_CONFIG.get("morozov_tau", 1.0)),
        max_iter=int(DATA_CONFIG.get("morozov_max_iter", 8)),
        lambda_min=te._current_b1b1_lambda_floor(),
    )
    lambda_eff = max(float(lam.view(-1)[0].item()), te._current_b1b1_lambda_floor())
    coeff_tikh = te._solve_current_b1b1_tikhonov(
        operator=operator_tikh,
        g_obs=g_obs_real,
        lambda_reg=lambda_eff,
    )
    tikh_res = te._coeff_res(coeff_tikh.squeeze(), coeff_true.squeeze())

    if noise_domain == "time_radon_samples":
        g_clean_freq = dual_frontend["g_clean_freq"].to(dtype=torch.complex64)
        g_direct_freq = operator_dual.forward(coeff_true).to(dtype=torch.complex64)
        freq_rel_err_time_vs_direct_clean = float(
            (
                torch.linalg.vector_norm((g_clean_freq - g_direct_freq).reshape(g_clean_freq.shape[0], -1), dim=1)
                / torch.linalg.vector_norm(g_direct_freq.reshape(g_direct_freq.shape[0], -1), dim=1).clamp_min(1.0e-12)
            )
            .mean()
            .item()
        )
        coeff_direct_noiseless = operator_dual.solve_dual_frame_direct(
            g_direct_freq,
            lambda_reg=0.0,
        )
        res_direct_noiseless = te._coeff_res(coeff_direct_noiseless.squeeze(), coeff_true.squeeze())
    else:
        torch.manual_seed(noise_seed)
        g_clean_freq = operator_dual.forward(coeff_true).to(dtype=torch.complex64)
        g_obs_freq, noise_power, noise_norm_freq = te._apply_frequency_sample_noise(
            g_clean_freq,
            noise_mode=noise_mode,
            delta=float(args.delta),
            target_snr_db=float(args.target_snr_db),
        )
        signal_power = torch.mean(torch.abs(g_clean_freq).square(), dim=1).clamp_min(1.0e-12)
        relative_noise_power = noise_power / signal_power
    if noise_domain == "time_radon_samples":
        if str(dual_frequency_build_mode).strip().lower() == "direct_ck":
            dual_direct_info = te._solve_dual_coeff_direct_ck_from_frequency_samples(
                dual_obs["g_obs_freq_estimator"].to(dtype=torch.complex64),
                operator_dual=operator_dual,
            )
            coeff_dual = dual_direct_info["coeff"]
            dual_lambda = float(dual_direct_info["lambda"])
            direct_ck_stability_mode = str(dual_direct_info.get("stability_mode", direct_ck_stability_mode))
            direct_ck_stability_alias_mode = str(dual_direct_info.get("stability_alias_mode", direct_ck_stability_alias_mode))
            direct_ck_stability_clip_threshold = float(dual_direct_info.get("stability_clip_threshold", direct_ck_stability_clip_threshold))
            direct_ck_stability_clip_ratio = float(dual_direct_info.get("stability_clip_ratio", direct_ck_stability_clip_ratio))
            direct_ck_stability_alias_residual_mean = float(dual_direct_info.get("stability_alias_residual_mean", direct_ck_stability_alias_residual_mean))
            direct_ck_stability_alias_residual_q95 = float(dual_direct_info.get("stability_alias_residual_q95", direct_ck_stability_alias_residual_q95))
            direct_ck_stability_alias_weight_mean = float(dual_direct_info.get("stability_alias_weight_mean", direct_ck_stability_alias_weight_mean))
            direct_ck_stability_alias_weight_min = float(dual_direct_info.get("stability_alias_weight_min", direct_ck_stability_alias_weight_min))
            direct_ck_stability_norm_ratio_threshold = float(dual_direct_info.get("stability_norm_ratio_threshold", direct_ck_stability_norm_ratio_threshold))
            direct_ck_stability_norm_ratio_mean = float(dual_direct_info.get("stability_norm_ratio_mean", direct_ck_stability_norm_ratio_mean))
            direct_ck_stability_norm_ratio_q95 = float(dual_direct_info.get("stability_norm_ratio_q95", direct_ck_stability_norm_ratio_q95))
            direct_ck_stability_norm_ratio_weight_mean = float(dual_direct_info.get("stability_norm_ratio_weight_mean", direct_ck_stability_norm_ratio_weight_mean))
            direct_ck_stability_norm_ratio_weight_min = float(dual_direct_info.get("stability_norm_ratio_weight_min", direct_ck_stability_norm_ratio_weight_min))
            direct_ck_stability_kernel_abs_raw_max = float(dual_direct_info.get("stability_kernel_abs_raw_max", direct_ck_stability_kernel_abs_raw_max))
            direct_ck_stability_kernel_abs_after_max = float(dual_direct_info.get("stability_kernel_abs_after_max", direct_ck_stability_kernel_abs_after_max))
        else:
            coeff_dual = te._solve_dual_coeff_from_common_factor(
                dual_obs["g_obs_common_factor"].to(dtype=torch.complex64),
                operator_dual=operator_dual,
            )
            dual_lambda = 0.0
    else:
        coeff_dual = operator_dual.solve_dual_frame_direct(
            g_obs_freq,
            noise_power=relative_noise_power,
        )
        dual_lambda = (
            float(operator_dual.last_dual_lambda.mean().item())
            if operator_dual.last_dual_lambda is not None
            else float("nan")
        )
    dual_res = te._coeff_res(coeff_dual.squeeze(), coeff_true.squeeze())

    return {
        "beta": beta,
        "dual_noise_domain": noise_domain,
        "dual_s_num_samples": int(dual_s_num_samples),
        "dual_reference_s_num_samples": int(dual_reference_s_num_samples),
        "dual_delta_s": float(dual_delta_s),
        "dual_reference_delta_s": float(dual_reference_delta_s),
        "dual_delta_t": float(dual_delta_t),
        "dual_time_sampling_mode": str(dual_time_sampling_mode),
        "dual_time_sampling_t_min": float(dual_time_sampling_t_min),
        "dual_time_sampling_t_max": float(dual_time_sampling_t_max),
        "dual_alias_estimator_truncation": int(dual_alias_estimator_truncation),
        "dual_frequency_build_mode": str(dual_frequency_build_mode),
        "dual_xi_phase_shift": float(args.dual_xi_phase_shift),
        "frequency_mask_threshold": float(frequency_mask_threshold),
        "frequency_mask_threshold_rel": float(frequency_mask_threshold_rel),
        "frequency_mask_keep_ratio": float(frequency_mask_keep_ratio),
        "frequency_mask_mean_weight": float(frequency_mask_mean_weight),
        "frequency_mask_weight_norm": str(frequency_mask_weight_norm),
        "frequency_mask_mode": str(frequency_mask_mode),
        "direct_ck_stability_mode": str(direct_ck_stability_mode),
        "direct_ck_stability_alias_mode": str(direct_ck_stability_alias_mode),
        "direct_ck_stability_clip_threshold": float(direct_ck_stability_clip_threshold),
        "direct_ck_stability_clip_ratio": float(direct_ck_stability_clip_ratio),
        "direct_ck_stability_alias_residual_mean": float(direct_ck_stability_alias_residual_mean),
        "direct_ck_stability_alias_residual_q95": float(direct_ck_stability_alias_residual_q95),
        "direct_ck_stability_alias_weight_mean": float(direct_ck_stability_alias_weight_mean),
        "direct_ck_stability_alias_weight_min": float(direct_ck_stability_alias_weight_min),
        "direct_ck_stability_norm_ratio_threshold": float(direct_ck_stability_norm_ratio_threshold),
        "direct_ck_stability_norm_ratio_mean": float(direct_ck_stability_norm_ratio_mean),
        "direct_ck_stability_norm_ratio_q95": float(direct_ck_stability_norm_ratio_q95),
        "direct_ck_stability_norm_ratio_weight_mean": float(direct_ck_stability_norm_ratio_weight_mean),
        "direct_ck_stability_norm_ratio_weight_min": float(direct_ck_stability_norm_ratio_weight_min),
        "direct_ck_stability_kernel_abs_raw_max": float(direct_ck_stability_kernel_abs_raw_max),
        "direct_ck_stability_kernel_abs_after_max": float(direct_ck_stability_kernel_abs_after_max),
        "freq_rel_err_time_vs_direct_clean": float(freq_rel_err_time_vs_direct_clean),
        "res_direct_noiseless": float(res_direct_noiseless),
        "coeff_true": coeff_true.squeeze().detach().cpu().numpy().astype(np.float32),
        "coeff_tikh": coeff_tikh.squeeze().detach().cpu().numpy().astype(np.float32),
        "coeff_dual": coeff_dual.squeeze().detach().cpu().numpy().astype(np.float32),
        "tikhonov_res": float(tikh_res),
        "dual_res": float(dual_res),
        "tikhonov_lambda": float(lambda_eff),
        "dual_lambda_mean": float(dual_lambda),
        "dual_backend": str(operator_dual.dual_backend_name),
        "noise_norm_real": float(noise_norm_real),
        "noise_norm_time": float(noise_norm_time),
        "noise_norm_freq": float(noise_norm_freq),
    }


def _plot_triptych(args: argparse.Namespace, result: Dict[str, object], output_path: str) -> None:
    img_true = np.asarray(result["coeff_true"], dtype=np.float32)
    img_dual = np.asarray(result["coeff_dual"], dtype=np.float32)
    img_tikh = np.asarray(result["coeff_tikh"], dtype=np.float32)

    vmin = float(min(np.min(img_true), np.min(img_dual), np.min(img_tikh)))
    vmax = float(max(np.max(img_true), np.max(img_dual), np.max(img_tikh)))
    noise_label = te._noise_label(
        str(args.noise_mode).strip().lower(),
        delta=float(args.delta),
        target_snr_db=float(args.target_snr_db),
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    panels = [
        ("True", img_true, "shepp_logan"),
        (
            f"Dual Frequency\nRES={float(result['dual_res']):.6f}",
            img_dual,
            (
                f"P={int(args.P)}, Q={int(args.Q)}, L={int(args.L)}"
                f" | backend={result['dual_backend']}"
                f" | noise_domain={result['dual_noise_domain']}"
                f" | build={result['dual_frequency_build_mode']}"
                f" | Qe={int(result['dual_alias_estimator_truncation'])}"
                f" | shift={float(result['dual_xi_phase_shift']):.3e}"
                f" | mask={result['frequency_mask_mode']}:{float(result['frequency_mask_threshold_rel']):.3e}"
                f" | mean_w={float(result['frequency_mask_mean_weight']):.3f}"
                f" | norm={result['frequency_mask_weight_norm']}"
                f" | gls_lambda_rel={float(args.dual_time_gls_lambda_rel):.3e}"
                f" | lambda={float(result['dual_lambda_mean']):.3e}"
            ),
        ),
        (
            f"Tikhonov\nRES={float(result['tikhonov_res']):.6f}",
            img_tikh,
            f"single-angle baseline | lambda={float(result['tikhonov_lambda']):.3e}",
        ),
    ]
    for ax, (title, img, subtitle) in zip(axes, panels):
        ax.imshow(img, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        ax.text(0.5, -0.08, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=8)

    fig.suptitle(
        "shepp_logan | "
        f"beta={tuple(result['beta'])} | {noise_label} | noise_domain={result['dual_noise_domain']} | "
        f"P={int(args.P)} | Q={int(args.Q)} | L={int(args.L)} | shift={float(args.dual_xi_phase_shift):.3e}",
        y=0.98,
        fontsize=12,
    )
    fig.subplots_adjust(wspace=0.04, top=0.82, bottom=0.20)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if int(args.P) <= 0:
        raise ValueError(f"P must be positive, got {args.P}.")
    if int(args.Q) < 0:
        raise ValueError(f"Q must be non-negative, got {args.Q}.")
    if int(args.L) < 0:
        raise ValueError(f"L must be non-negative, got {args.L}.")
    if int(args.dual_time_num_samples) < 2:
        raise ValueError(f"dual_time_num_samples must be at least 2, got {args.dual_time_num_samples}.")

    old_dual = _configure_dual_frequency(args)
    try:
        output_path = _resolve_output_path(args)
        result = _run_once(args)
        if not bool(args.skip_plot):
            _plot_triptych(args, result, output_path)
    finally:
        _restore_dual_frequency(old_dual)

    summary = {
        "output": output_path,
        "beta": list(result["beta"]),
        "noise_mode": str(args.noise_mode).strip().lower(),
        "delta": float(args.delta),
        "target_snr_db": float(args.target_snr_db),
        "dual_noise_domain": str(args.dual_noise_domain).strip().lower(),
        "dual_requested_s_num_samples": int(args.dual_time_num_samples),
        "P": int(args.P),
        "Q": int(args.Q),
        "L": int(args.L),
        "dual_xi_phase_shift": float(args.dual_xi_phase_shift),
        "dual_backend": str(result["dual_backend"]),
        "dual_frequency_build_mode": str(result["dual_frequency_build_mode"]),
        "dual_alias_estimator_truncation": int(result["dual_alias_estimator_truncation"]),
        "dual_time_gls_lambda_rel": float(args.dual_time_gls_lambda_rel),
        "dual_time_gls_covariance_ridge_rel": float(args.dual_time_gls_covariance_ridge_rel),
        "dual_time_sampling_interval_mode": str(args.dual_time_sampling_interval_mode).strip().lower(),
        "dual_time_fourier_quadrature": str(args.dual_time_fourier_quadrature).strip().lower(),
        "dual_time_frequency_mask_rel": float(args.dual_time_frequency_mask_rel),
        "dual_time_frequency_mask_mode": str(args.dual_time_frequency_mask_mode).strip().lower(),
        "dual_direct_ck_gramian_mask_quantile": float(args.dual_direct_ck_gramian_mask_quantile),
        "dual_direct_ck_mask_mode": str(args.dual_direct_ck_mask_mode).strip().lower(),
        "dual_direct_ck_weight_norm": str(args.dual_direct_ck_weight_norm).strip().lower(),
        "dual_direct_ck_lambda_rel": float(args.dual_direct_ck_lambda_rel),
        "dual_direct_ck_stability_mode": str(args.dual_direct_ck_stability_mode).strip().lower(),
        "dual_direct_ck_clip_good_g_rel": float(args.dual_direct_ck_clip_good_g_rel),
        "dual_direct_ck_clip_quantile": float(args.dual_direct_ck_clip_quantile),
        "dual_direct_ck_clip_scale": float(args.dual_direct_ck_clip_scale),
        "dual_direct_ck_alias_tau": float(args.dual_direct_ck_alias_tau),
        "dual_direct_ck_alias_mode": str(args.dual_direct_ck_alias_mode).strip().lower(),
        "dual_direct_ck_norm_ratio_window": int(args.dual_direct_ck_norm_ratio_window),
        "direct_ck_stability_mode": str(result["direct_ck_stability_mode"]),
        "direct_ck_stability_alias_mode": str(result["direct_ck_stability_alias_mode"]),
        "direct_ck_stability_clip_threshold": float(result["direct_ck_stability_clip_threshold"]),
        "direct_ck_stability_clip_ratio": float(result["direct_ck_stability_clip_ratio"]),
        "direct_ck_stability_alias_residual_mean": float(result["direct_ck_stability_alias_residual_mean"]),
        "direct_ck_stability_alias_residual_q95": float(result["direct_ck_stability_alias_residual_q95"]),
        "direct_ck_stability_alias_weight_mean": float(result["direct_ck_stability_alias_weight_mean"]),
        "direct_ck_stability_alias_weight_min": float(result["direct_ck_stability_alias_weight_min"]),
        "direct_ck_stability_norm_ratio_threshold": float(result["direct_ck_stability_norm_ratio_threshold"]),
        "direct_ck_stability_norm_ratio_mean": float(result["direct_ck_stability_norm_ratio_mean"]),
        "direct_ck_stability_norm_ratio_q95": float(result["direct_ck_stability_norm_ratio_q95"]),
        "direct_ck_stability_norm_ratio_weight_mean": float(result["direct_ck_stability_norm_ratio_weight_mean"]),
        "direct_ck_stability_norm_ratio_weight_min": float(result["direct_ck_stability_norm_ratio_weight_min"]),
        "direct_ck_stability_kernel_abs_raw_max": float(result["direct_ck_stability_kernel_abs_raw_max"]),
        "direct_ck_stability_kernel_abs_after_max": float(result["direct_ck_stability_kernel_abs_after_max"]),
        "frequency_mask_threshold": float(result["frequency_mask_threshold"]),
        "frequency_mask_keep_ratio": float(result["frequency_mask_keep_ratio"]),
        "frequency_mask_mean_weight": float(result["frequency_mask_mean_weight"]),
        "frequency_mask_weight_norm": str(result["frequency_mask_weight_norm"]),
        "dual_s_num_samples": int(result["dual_s_num_samples"]),
        "dual_delta_s": float(result["dual_delta_s"]),
        "dual_delta_t": float(result["dual_delta_t"]),
        "dual_time_sampling_mode": str(result["dual_time_sampling_mode"]),
        "dual_time_sampling_t_min": float(result["dual_time_sampling_t_min"]),
        "dual_time_sampling_t_max": float(result["dual_time_sampling_t_max"]),
        "dual_reference_s_num_samples": int(result["dual_reference_s_num_samples"]),
        "dual_reference_delta_s": float(result["dual_reference_delta_s"]),
        "freq_rel_err_time_vs_direct_clean": float(result["freq_rel_err_time_vs_direct_clean"]),
        "res_direct_noiseless": float(result["res_direct_noiseless"]),
        "dual_res": float(result["dual_res"]),
        "tikhonov_res": float(result["tikhonov_res"]),
        "dual_lambda_mean": float(result["dual_lambda_mean"]),
        "tikhonov_lambda": float(result["tikhonov_lambda"]),
        "noise_norm_real": float(result["noise_norm_real"]),
        "noise_norm_time": float(result["noise_norm_time"]),
        "noise_norm_freq": float(result["noise_norm_freq"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
