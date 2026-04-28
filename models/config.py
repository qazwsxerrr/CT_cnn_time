# -*- coding: utf-8 -*-
"""Project configuration for CT_cnn."""

from __future__ import annotations

import os
import sys
import torch
import numpy as np
import math

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# -----------------------------------------------------------------------------
# Main experiment setting
# -----------------------------------------------------------------------------
# 128x128 Shepp-Logan phantom with B1*B1 pixel basis.
IMAGE_SIZE = 128

# Ensure project root is importable even when running scripts from within `models/`.
# This keeps project-local imports available without requiring package installs.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Theoretical training parameters (paper uses Tikhonov regularization).
THEORETICAL_CONFIG = {
    "beta_vector": (1, IMAGE_SIZE),
    "regularizer_type": "tikhonov",
    "n_iter": 15,
    "n_memory_units": 16,
}

# Optional runtime override: N_ITER_OVERRIDE
_n_iter_override = os.environ.get("N_ITER_OVERRIDE", None)
if _n_iter_override is not None:
    _s = str(_n_iter_override).strip()
    if _s:
        try:
            _n_iter = int(_s)
        except ValueError as e:
            raise ValueError(f"Invalid N_ITER_OVERRIDE={_n_iter_override!r}; expected an integer.") from e
        if _n_iter <= 0:
            raise ValueError(f"Invalid N_ITER_OVERRIDE={_n_iter_override!r}; expected a positive integer.")
        THEORETICAL_CONFIG["n_iter"] = _n_iter

n_data = 8
n_train = 5000

BACKBONE_MULTI8_BETA_VECTORS = [
    (1, IMAGE_SIZE),
    (-1, IMAGE_SIZE),
    (1, -IMAGE_SIZE),
    (-1, -IMAGE_SIZE),
    (IMAGE_SIZE, 1),
    (-IMAGE_SIZE, 1),
    (IMAGE_SIZE, -1),
    (-IMAGE_SIZE, -1),
]

DEFAULT_EXPERIMENT_PROFILE = "injective16_pi_best"
FULL_TRIANGULAR_16ANGLE_COUNT = 16
INJECTIVE16_PI_BEST_CNN_ANGLE_INDICES = (0, 5, 6, 8, 9, 10, 13, 14)

# Best-performing 16-angle full-triangular set observed in the archived
# "(0,pi)" experiment. We keep it fixed for reproducible Tikhonov
# initialization and learned-optimizer training/testing.
BEST_PI16_BETA_VECTORS = [
    (128, 1),
    (1, -128),
    (127, 128),
    (128, -127),
    (128, -53),
    (53, 128),
    (53, -128),
    (128, 53),
    (25, 128),
    (128, -25),
    (128, -85),
    (85, 128),
    (128, 85),
    (85, -128),
    (25, -128),
    (128, 25),
]


def _primitive_signed_beta(beta) -> tuple[int, int]:
    a = int(beta[0])
    b = int(beta[1])
    g = math.gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
    return (a, b)


def _is_injective_beta_for_square(beta, image_size: int) -> bool:
    a, b = _primitive_signed_beta(beta)
    return a != 0 and b != 0 and max(abs(a), abs(b)) >= int(image_size)


def _injective_boundary_beta_candidates(image_size: int) -> list[tuple[int, int]]:
    size = int(image_size)
    used = set()
    candidates = []
    for a in range(-size, size + 1):
        for b in (-size, size):
            beta = _primitive_signed_beta((a, b))
            if (
                beta in used
                or max(abs(beta[0]), abs(beta[1])) != size
                or not _is_injective_beta_for_square(beta, size)
            ):
                continue
            used.add(beta)
            candidates.append(beta)
    for b in range(-size + 1, size):
        for a in (-size, size):
            beta = _primitive_signed_beta((a, b))
            if (
                beta in used
                or max(abs(beta[0]), abs(beta[1])) != size
                or not _is_injective_beta_for_square(beta, size)
            ):
                continue
            used.add(beta)
            candidates.append(beta)
    candidates.sort(key=lambda beta: (beta[0], beta[1]))
    return candidates


def _current_random_beta_seed() -> int:
    override = os.environ.get("EXTRA_ANGLE_SEED_OVERRIDE", None)
    if override is not None and str(override).strip():
        return int(str(override).strip())
    return int(TIME_DOMAIN_CONFIG.get("extra_angle_seed", 20260322))


def _select_random_beta_vectors(
    *,
    image_size: int,
    count: int,
    excluded_betas=None,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    count = int(count)
    if count <= 0:
        return []

    excluded = {_primitive_signed_beta(beta) for beta in list(excluded_betas or [])}
    pool = [
        beta
        for beta in _injective_boundary_beta_candidates(int(image_size))
        if _primitive_signed_beta(beta) not in excluded
    ]
    if len(pool) < count:
        raise ValueError(
            f"Not enough injective beta candidates for count={count}; only {len(pool)} available after exclusions."
        )

    rng = np.random.default_rng(int(_current_random_beta_seed() if seed is None else seed))
    picked = rng.choice(len(pool), size=count, replace=False)
    return [pool[int(idx)] for idx in picked.tolist()]


def _apply_full_triangular_16angle_profile(
    *,
    beta_vectors,
    solver_mode: str,
    formula_mode: str | None = None,
) -> None:
    beta_list = [tuple(int(v) for v in beta) for beta in list(beta_vectors)]
    if len(beta_list) != FULL_TRIANGULAR_16ANGLE_COUNT:
        raise ValueError(
            "Full-triangular 16-angle profile requires exactly "
            f"{FULL_TRIANGULAR_16ANGLE_COUNT} beta vectors; got {len(beta_list)}."
        )
    solver_mode = str(solver_mode).strip().lower()
    if solver_mode not in {"stacked_tikhonov", "split_triangular_admm"}:
        raise ValueError(
            f"Unsupported solver_mode={solver_mode!r}; expected 'stacked_tikhonov' "
            "or 'split_triangular_admm'."
        )

    TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = solver_mode
    TIME_DOMAIN_CONFIG["multi_angle_layout"] = "full_triangular"
    TIME_DOMAIN_CONFIG["beta_vectors"] = beta_list
    TIME_DOMAIN_CONFIG["explicit_extra_beta_vectors"] = None
    TIME_DOMAIN_CONFIG["num_angles_total"] = FULL_TRIANGULAR_16ANGLE_COUNT
    TIME_DOMAIN_CONFIG["num_angles"] = FULL_TRIANGULAR_16ANGLE_COUNT
    TIME_DOMAIN_CONFIG["cnn_backbone_only"] = False
    TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = FULL_TRIANGULAR_16ANGLE_COUNT
    TIME_DOMAIN_CONFIG["cnn_angle_indices_override"] = None
    TIME_DOMAIN_CONFIG["cnn_feature_beta_vectors_override"] = None
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_enabled"] = False
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_mode"] = "disabled"
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_output_channels"] = 8
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_hidden_channels"] = 8
    if formula_mode is not None:
        TIME_DOMAIN_CONFIG["theoretical_formula_mode"] = str(formula_mode).strip().lower()


def _apply_experiment_profile(profile_name: str) -> None:
    profile = str(profile_name or "").strip().lower()
    if profile in ("", "default", "none"):
        TIME_DOMAIN_CONFIG["experiment_profile"] = "default"
        TIME_DOMAIN_CONFIG["multi_angle_layout"] = "structured_backbone_extra"
        TIME_DOMAIN_CONFIG["explicit_extra_beta_vectors"] = None
        return

    TIME_DOMAIN_CONFIG["experiment_profile"] = profile
    TIME_DOMAIN_CONFIG["operator_mode"] = "theoretical_b1b1"
    TIME_DOMAIN_CONFIG["use_multi_angle"] = True
    TIME_DOMAIN_CONFIG["auto_angle_t0"] = True
    TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = "split_triangular_admm"

    if profile == "structured16_injective_extra":
        _apply_full_triangular_16angle_profile(
            beta_vectors=list(BACKBONE_MULTI8_BETA_VECTORS)
            + _select_random_beta_vectors(
                image_size=IMAGE_SIZE,
                count=8,
                excluded_betas=BACKBONE_MULTI8_BETA_VECTORS,
            ),
            solver_mode="split_triangular_admm",
            formula_mode="legacy_injective_extension",
        )
        return

    if profile == "injective16_full_triangular":
        _apply_full_triangular_16angle_profile(
            beta_vectors=_select_random_beta_vectors(
                image_size=IMAGE_SIZE,
                count=FULL_TRIANGULAR_16ANGLE_COUNT,
                excluded_betas=None,
            ),
            solver_mode="split_triangular_admm",
            formula_mode="legacy_injective_extension",
        )
        return

    if profile == "injective16_pi_best":
        _apply_full_triangular_16angle_profile(
            beta_vectors=BEST_PI16_BETA_VECTORS,
            solver_mode="stacked_tikhonov",
            formula_mode="legacy_injective_extension",
        )
        TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = FULL_TRIANGULAR_16ANGLE_COUNT
        TIME_DOMAIN_CONFIG["cnn_feature_beta_vectors_override"] = None
        return

    raise ValueError(
        f"Unsupported EXPERIMENT_PROFILE_OVERRIDE={profile_name!r}; "
        "expected one of 'structured16_injective_extra', 'injective16_full_triangular', "
        "'injective16_pi_best'."
    )


def _get_env_override(name: str):
    value = os.environ.get(name, None)
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def _apply_string_override(target: dict, key: str, env_name: str, allowed_values=None):
    value = _get_env_override(env_name)
    if value is None:
        return
    value = value.lower()
    if allowed_values is not None and value not in allowed_values:
        raise ValueError(
            f"Invalid {env_name}={value!r}; expected one of {sorted(allowed_values)!r}."
        )
    target[key] = value


def _apply_float_override(target: dict, key: str, env_name: str):
    value = _get_env_override(env_name)
    if value is None:
        return
    try:
        target[key] = float(value)
    except ValueError as e:
        raise ValueError(f"Invalid {env_name}={value!r}; expected a float.") from e


def _apply_int_list_override(target: dict, key: str, env_name: str):
    value = _get_env_override(env_name)
    if value is None:
        return
    tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
    if not tokens:
        target[key] = None
        return
    try:
        target[key] = [int(token) for token in tokens]
    except ValueError as e:
        raise ValueError(f"Invalid {env_name}={value!r}; expected a comma-separated integer list.") from e


def _apply_beta_vector_list_override(target: dict, key: str, env_name: str):
    value = _get_env_override(env_name)
    if value is None:
        return
    tokens = [token.strip() for token in value.split(";") if token.strip()]
    if not tokens:
        target[key] = None
        return
    parsed = []
    try:
        for token in tokens:
            token = token.replace(":", ",")
            parts = [part.strip() for part in token.split(",") if part.strip()]
            if len(parts) != 2:
                raise ValueError(token)
            parsed.append((int(parts[0]), int(parts[1])))
    except ValueError as e:
        raise ValueError(
            f"Invalid {env_name}={value!r}; expected semicolon-separated integer pairs."
        ) from e
    target[key] = parsed


def _apply_int_override(target: dict, key: str, env_name: str):
    value = _get_env_override(env_name)
    if value is None:
        return
    try:
        target[key] = int(value)
    except ValueError as e:
        raise ValueError(f"Invalid {env_name}={value!r}; expected an integer.") from e


def _apply_bool_override(target: dict, key: str, env_name: str):
    value = _get_env_override(env_name)
    if value is None:
        return
    value = value.lower()
    if value not in ("1", "0", "true", "false", "yes", "no", "y", "n"):
        raise ValueError(
            f"Invalid {env_name}={value!r}; expected a boolean-like value."
        )
    target[key] = value in ("1", "true", "yes", "y")

DATA_CONFIG = {
    # Main experiment data source: B1*B1 coefficient maps from random ellipses or Shepp-Logan.
    "data_source": "random_ellipses",
    "train_data_source": "random_ellipses",
    "val_data_source": "shepp_logan",
    "test_data_source": "shepp_logan",

    # Noise Configuration
    # Options: "additive", "multiplicative", "snr"
    "noise_mode": "multiplicative",
    "noise_level": 0.1,
    "target_snr_db": 30.0,

    # Tikhonov regularization parameter selection:
    # - "morozov": choose lambda by Morozov discrepancy principle (paper setting)
    # - "fixed": use the manual lambda_reg value below
    "lambda_select_mode": "morozov",
    # Safety factor in Morozov: ||A d_lambda - b^δ|| = tau * ||noise||
    "morozov_tau": 1.0,
    # Iterations for the outer safeguarded Newton root search.
    "morozov_max_iter": 8,
    # Lower / upper bounds, Newton tolerance, and initial guess used by the
    # paper-style Morozov discrepancy solver.
    "morozov_lambda_min": 1.0e-12,
    "morozov_lambda_max": 1.0e12,
    "morozov_newton_tol": 1.0e-10,
    "morozov_initial_lambda": 1.0,
    # Disk cache for exact implicit Gram/spectral data used by paper-style Morozov.
    "morozov_cache_dir": os.path.join(DATA_DIR, "morozov_cache"),
    # Retired compatibility knobs from the previous CG-inside-Morozov implementation.
    # They are kept only so older scripts importing DATA_CONFIG do not break.
    "morozov_cg_iters": 12,
    "morozov_cg_tol": 1.0e-4,
    # Manual lambda used only when lambda_select_mode == "fixed".
    # Oracle-optimal value ~0.06 (empirically determined via grid search).
    "lambda_reg": 1.0e-02,
    "implicit_eval_solver": "cg",
    "implicit_eval_lambda_min": 1.0e-02,
    "implicit_eval_cg_iters": 80,
    "implicit_eval_cg_tol": 1.0e-4,
    # Frequency-domain dual-frame recovery defaults (single-angle v2):
    # Use a periodic base grid xi_j = -pi + 2*pi*j/P on the principal cell and
    # include a small number of neighboring 2*pi-shifted cells in the outer
    # frequency integral.  The actual sampled frequency count is therefore
    # P * (2 * dual_integral_alias_truncation + 1).
    #
    # Empirically for the single-angle beta=(1,128) test:
    # - P = 8N is still slightly under-resolved in the no-noise roundtrip;
    # - P = 12N with one alias layer on each side already brings the relative
    #   roundtrip error below 1e-2 while keeping the total sample count moderate.
    "dual_xi_grid": "uniform_periodic",
    # Optional principal-cell phase shift:
    # xi_j = -pi + dual_xi_phase_shift + 2*pi*j/P.
    # This is used to avoid "bad" resonance points on the unshifted grid.
    "dual_xi_phase_shift": 0.0,
    "dual_num_frequency_samples": 12 * IMAGE_SIZE * IMAGE_SIZE,
    "dual_integral_alias_truncation": 1,
    "dual_gramian_truncation_L": 8,
    # Chunk size used by the direct exponential backend that supports P < N.
    # The direct backend avoids FFT aliasing but costs O(NP), so we keep the
    # working set bounded by processing the exponential sums in chunks.
    "dual_direct_chunk_size": 512,
    # Time-domain front-end for Dual Frequency:
    # - keep the current single-angle theorem-grid interval as the reference
    #   time window;
    # - rebuild a dense uniform grid on that same interval with N_t points;
    # - evaluate only a small low-alias subset |q| <= dual_time_alias_estimator_truncation
    #   by discrete Fourier summation;
    # - when dual_time_recovery_mode == "gls_low_alias", use the induced
    #   alias-block covariance to form a GLS estimate of the shared periodic
    #   factor on the principal cell, then synthesize the requested alias blocks;
    # - when dual_time_recovery_mode == "simple_ratio", keep the legacy
    #   unweighted ratio estimator;
    # - when dual_time_recovery_mode == "direct_ck", keep the full time-to-frequency
    #   alias samples and insert them directly into the c_k discretized integral.
    "dual_time_num_samples": IMAGE_SIZE * IMAGE_SIZE,
    "dual_time_forward_chunk_size": 512,
    "dual_time_fourier_chunk_size": 512,
    # Quadrature used by the time-domain Fourier integral:
    # - "rectangle": legacy Delta_t * sum_r g(t_r) exp(-i xi t_r);
    # - "midpoint_cell": exact integral of piecewise-constant midpoint cells,
    #   i.e. the rectangle rule multiplied by sinc(xi * Delta_t / 2).
    "dual_time_fourier_quadrature": "rectangle",
    # Time grid used before converting time-domain Radon samples to frequency samples:
    # - "theorem_grid": legacy interval inherited from the B1*B1 theorem grid;
    # - "beta_support_midpoint": midpoint rule on the full R_beta f support interval.
    "dual_time_sampling_interval_mode": "theorem_grid",
    "dual_time_recovery_mode": "gls_low_alias",
    "dual_time_alias_estimator_truncation": 1,
    "dual_time_gls_covariance_ridge_rel": 1.0e-10,
    # For the current single-angle beta=(1,128) time-domain SNR experiments,
    # a relative GLS denominator floor around 1e-3 is substantially more stable
    # than the previous near-zero default 1e-6.
    "dual_time_gls_lambda_rel": 1.0e-3,
    # Additional reliability-aware suppression on the recovered principal-cell
    # common factor. The threshold is relative to max_j h_j^* Sigma^{-1} h_j
    # (or its simple-ratio counterpart when not using GLS).
    "dual_time_frequency_mask_rel": 0.0,
    "dual_time_frequency_mask_mode": "soft",
    # direct_ck-only controls.  gramian_mask_quantile discards/downweights the
    # lowest rho_G fraction of G_{beta,L}(xi_j); weight_norm="none" implements
    # the no-renorm variant in the direct-c_k sampling-constraint plan.
    "dual_direct_ck_gramian_mask_quantile": 0.0,
    "dual_direct_ck_mask_mode": "hard",
    "dual_direct_ck_weight_norm": "none",
    "dual_direct_ck_lambda_rel": 0.0,
    # direct_ck-only g/h alias stability controls.  These are disabled by
    # default so the existing direct_ck baseline remains unchanged unless the
    # experiment explicitly opts in.
    "dual_direct_ck_stability_mode": "none",
    "dual_direct_ck_clip_good_g_rel": 0.1,
    "dual_direct_ck_clip_quantile": 0.99,
    "dual_direct_ck_clip_scale": 1.2,
    "dual_direct_ck_clip_eps": 1.0e-12,
    "dual_direct_ck_alias_tau": 0.3,
    "dual_direct_ck_alias_mode": "soft",
    "dual_direct_ck_norm_ratio_window": 512,
    "dual_direct_ck_norm_ratio_alias_gate": True,
    "dual_kernel_lambda_rel_floor": 3.0e-15,
    "dual_noise_domain": "spectral_samples",

    # Data fidelity modeling inside the learned optimizer:
    # - "standard": use residual r = (A c - b) (homoscedastic least squares)
    # - "irls": multiplicative-noise-aware weighted residual
    #          r_w = (A c - b) / (|A c| + eps)^2   (weights computed from current prediction)
    # This does NOT add more measurements; it changes how the optimizer treats heteroscedastic noise.
    "data_fidelity_mode": "standard",
    # eps = irls_eps_factor * median(|A c|) for numerical stability.
    "irls_eps_factor": 3.0e-03,
    # Detach weights from autograd (treat as IRLS-style fixed weights for stability).
    "irls_detach_weights": True,
    # Detach physically-derived gradients (A^T(Ac-b), regularizer grad) from backprop-through-time.
    # This often stabilizes training for deep unrolled optimizers on ill-conditioned operators.
    "detach_physical_grads": False,
    # Ablation: keep the CNN architecture unchanged, but zero the per-angle
    # data-fidelity gradient channels.  Then Dual only supplies coeff_initial
    # and does not influence the learned update through measurement residuals.
    "zero_data_grad_channels": False,

    # Learned-optimizer initialization (network-side, does not affect data generation):
    # A smaller lambda often works better for multiplicative-noise-aware (IRLS) data fidelity.
    "learned_reg_lambda_init": 1.0e-02,
    # Step size for the (preconditioned) learned optimizer.
    # NOTE: step initialization is scaled by num_angles inside the model, so this value is per-angle.
    # For 8 angles: 6.25e-4 * 8 = 5e-3 (matches the stable setting).
    "learned_step_init": 2.0e-03,
    # Positive floor for the learned step size. Must stay below learned_step_init
    # if the initialization is expected to be honored exactly.
    "learned_step_min": 1.0e-06,
    # Safety caps for the learned optimizer scalars (0 disables a cap).
    "learned_step_max": 1.0e-02,
    "learned_reg_lambda_max": 1.0e-01,
    "learned_correction_max": 0.0,
    # Trust-region style cap for each unrolled update (L2 norm in coefficient space, per sample).
    # Helps prevent occasional training steps from pushing coefficients to a worse basin.
    "update_max_norm": 0.0,
    "validation_seed": 42,
    "val_batch_size": n_data,
    # Validation reproducibility:
    # - True: keep validation set fixed so RES curves are comparable across iterations (recommended for debugging)
    # - False: re-sample validation data each time (estimates expected RES but will look noisy/flat)
    "val_reproducible": True,

}

_apply_string_override(
    DATA_CONFIG,
    "train_data_source",
    "TRAIN_DATA_SOURCE_OVERRIDE",
    allowed_values={"shepp_logan", "random_ellipses", "random_ellipse", "ellipse"},
)
_apply_string_override(
    DATA_CONFIG,
    "val_data_source",
    "VAL_DATA_SOURCE_OVERRIDE",
    allowed_values={"shepp_logan", "random_ellipses", "random_ellipse", "ellipse"},
)
_apply_string_override(
    DATA_CONFIG,
    "test_data_source",
    "TEST_DATA_SOURCE_OVERRIDE",
    allowed_values={"shepp_logan", "random_ellipses", "random_ellipse", "ellipse"},
)
_apply_string_override(
    DATA_CONFIG,
    "noise_mode",
    "NOISE_MODE_OVERRIDE",
    allowed_values={"additive", "multiplicative", "snr"},
)
_apply_float_override(DATA_CONFIG, "noise_level", "NOISE_LEVEL_OVERRIDE")
_apply_float_override(DATA_CONFIG, "target_snr_db", "TARGET_SNR_DB_OVERRIDE")
_apply_string_override(
    DATA_CONFIG,
    "lambda_select_mode",
    "LAMBDA_SELECT_MODE_OVERRIDE",
    allowed_values={"fixed", "morozov"},
)
_apply_float_override(DATA_CONFIG, "lambda_reg", "LAMBDA_REG_OVERRIDE")
_apply_float_override(DATA_CONFIG, "morozov_tau", "MOROZOV_TAU_OVERRIDE")
_apply_float_override(DATA_CONFIG, "morozov_lambda_min", "MOROZOV_LAMBDA_MIN_OVERRIDE")
_apply_float_override(DATA_CONFIG, "morozov_lambda_max", "MOROZOV_LAMBDA_MAX_OVERRIDE")
_apply_float_override(DATA_CONFIG, "morozov_newton_tol", "MOROZOV_NEWTON_TOL_OVERRIDE")
_apply_float_override(DATA_CONFIG, "morozov_initial_lambda", "MOROZOV_INITIAL_LAMBDA_OVERRIDE")
_apply_string_override(DATA_CONFIG, "morozov_cache_dir", "MOROZOV_CACHE_DIR_OVERRIDE")
_apply_string_override(
    DATA_CONFIG,
    "dual_xi_grid",
    "DUAL_XI_GRID_OVERRIDE",
    allowed_values={"uniform_periodic"},
)
_apply_float_override(DATA_CONFIG, "dual_xi_phase_shift", "DUAL_XI_PHASE_SHIFT_OVERRIDE")
_apply_int_override(DATA_CONFIG, "dual_num_frequency_samples", "DUAL_NUM_FREQUENCY_SAMPLES_OVERRIDE")
_apply_int_override(DATA_CONFIG, "dual_integral_alias_truncation", "DUAL_INTEGRAL_ALIAS_TRUNCATION_OVERRIDE")
_apply_int_override(DATA_CONFIG, "dual_gramian_truncation_L", "DUAL_GRAMIAN_TRUNCATION_L_OVERRIDE")
_apply_int_override(DATA_CONFIG, "dual_time_num_samples", "DUAL_TIME_NUM_SAMPLES_OVERRIDE")
_apply_int_override(DATA_CONFIG, "dual_time_forward_chunk_size", "DUAL_TIME_FORWARD_CHUNK_SIZE_OVERRIDE")
_apply_int_override(DATA_CONFIG, "dual_time_fourier_chunk_size", "DUAL_TIME_FOURIER_CHUNK_SIZE_OVERRIDE")
_apply_string_override(
    DATA_CONFIG,
    "dual_time_fourier_quadrature",
    "DUAL_TIME_FOURIER_QUADRATURE_OVERRIDE",
    allowed_values={"rectangle", "midpoint_cell"},
)
_apply_string_override(
    DATA_CONFIG,
    "dual_time_sampling_interval_mode",
    "DUAL_TIME_SAMPLING_INTERVAL_MODE_OVERRIDE",
    allowed_values={"theorem_grid", "beta_support_midpoint"},
)
_apply_string_override(
    DATA_CONFIG,
    "dual_time_recovery_mode",
    "DUAL_TIME_RECOVERY_MODE_OVERRIDE",
    allowed_values={"simple_ratio", "gls_low_alias", "direct_ck"},
)
_apply_int_override(DATA_CONFIG, "dual_time_alias_estimator_truncation", "DUAL_TIME_ALIAS_ESTIMATOR_TRUNCATION_OVERRIDE")
_apply_float_override(DATA_CONFIG, "dual_time_gls_covariance_ridge_rel", "DUAL_TIME_GLS_COVARIANCE_RIDGE_REL_OVERRIDE")
_apply_float_override(DATA_CONFIG, "dual_time_gls_lambda_rel", "DUAL_TIME_GLS_LAMBDA_REL_OVERRIDE")
_apply_float_override(DATA_CONFIG, "dual_time_frequency_mask_rel", "DUAL_TIME_FREQUENCY_MASK_REL_OVERRIDE")
_apply_string_override(
    DATA_CONFIG,
    "dual_time_frequency_mask_mode",
    "DUAL_TIME_FREQUENCY_MASK_MODE_OVERRIDE",
    allowed_values={"hard", "soft"},
)
_apply_float_override(DATA_CONFIG, "dual_direct_ck_gramian_mask_quantile", "DUAL_DIRECT_CK_GRAMIAN_MASK_QUANTILE_OVERRIDE")
_apply_string_override(
    DATA_CONFIG,
    "dual_direct_ck_mask_mode",
    "DUAL_DIRECT_CK_MASK_MODE_OVERRIDE",
    allowed_values={"hard", "soft"},
)
_apply_string_override(
    DATA_CONFIG,
    "dual_direct_ck_weight_norm",
    "DUAL_DIRECT_CK_WEIGHT_NORM_OVERRIDE",
    allowed_values={"none", "renorm"},
)
_apply_float_override(DATA_CONFIG, "dual_direct_ck_lambda_rel", "DUAL_DIRECT_CK_LAMBDA_REL_OVERRIDE")
_apply_string_override(
    DATA_CONFIG,
    "dual_direct_ck_stability_mode",
    "DUAL_DIRECT_CK_STABILITY_MODE_OVERRIDE",
    allowed_values={"none", "kclip", "alias_consistency", "kclip_alias", "norm_ratio"},
)
_apply_float_override(DATA_CONFIG, "dual_direct_ck_clip_good_g_rel", "DUAL_DIRECT_CK_CLIP_GOOD_G_REL_OVERRIDE")
_apply_float_override(DATA_CONFIG, "dual_direct_ck_clip_quantile", "DUAL_DIRECT_CK_CLIP_QUANTILE_OVERRIDE")
_apply_float_override(DATA_CONFIG, "dual_direct_ck_clip_scale", "DUAL_DIRECT_CK_CLIP_SCALE_OVERRIDE")
_apply_float_override(DATA_CONFIG, "dual_direct_ck_clip_eps", "DUAL_DIRECT_CK_CLIP_EPS_OVERRIDE")
_apply_float_override(DATA_CONFIG, "dual_direct_ck_alias_tau", "DUAL_DIRECT_CK_ALIAS_TAU_OVERRIDE")
_apply_string_override(
    DATA_CONFIG,
    "dual_direct_ck_alias_mode",
    "DUAL_DIRECT_CK_ALIAS_MODE_OVERRIDE",
    allowed_values={"soft", "hard"},
)
_apply_int_override(DATA_CONFIG, "dual_direct_ck_norm_ratio_window", "DUAL_DIRECT_CK_NORM_RATIO_WINDOW_OVERRIDE")
_apply_bool_override(DATA_CONFIG, "dual_direct_ck_norm_ratio_alias_gate", "DUAL_DIRECT_CK_NORM_RATIO_ALIAS_GATE_OVERRIDE")
_apply_float_override(DATA_CONFIG, "dual_kernel_lambda_rel_floor", "DUAL_KERNEL_LAMBDA_REL_FLOOR_OVERRIDE")
_apply_string_override(
    DATA_CONFIG,
    "dual_noise_domain",
    "DUAL_NOISE_DOMAIN_OVERRIDE",
    allowed_values={"spectral_samples", "time_radon_samples"},
)
_apply_string_override(
    DATA_CONFIG,
    "data_fidelity_mode",
    "DATA_FIDELITY_MODE_OVERRIDE",
    allowed_values={"standard", "irls"},
)
_apply_bool_override(DATA_CONFIG, "detach_physical_grads", "DETACH_PHYSICAL_GRADS_OVERRIDE")
_apply_bool_override(DATA_CONFIG, "zero_data_grad_channels", "ZERO_DATA_GRAD_CHANNELS_OVERRIDE")

TIME_DOMAIN_CONFIG = {
    # Main experiment uses the theoretical B1*B1 time-domain operator.
    "operator_mode": "theoretical_b1b1",
    "experiment_profile": DEFAULT_EXPERIMENT_PROFILE,
    "sampling_scheme": "paper_grid_t0",
    "sampling_t0": 0.5,
    # Used only by the split_triangular_admm-internal shifted-support B1*B1 formula.
    # For the 8 backbone eligible directions we shift each per-angle sampling grid by
    # the lower support bound of R_alpha phi so every theoretical block remains valid.
    "auto_angle_t0": True,
    # Used only when sampling_scheme == "uniform_random_support"
    "sampling_seed": 123,

    # Angle mode switch:
    # - True:  multi-angle, uses beta_vectors list below
    # - False: single-angle, uses the paper's eligible direction beta=(1,21)
    "use_multi_angle": True,

    # Multi-angle configuration (only used when use_multi_angle is True).
    # beta_vectors are the fixed 8 backbone directions that must always be present.
    # The effective num_angles is derived later from num_angles_total.
    "num_angles_total": 8,
    "beta_vectors": list(BACKBONE_MULTI8_BETA_VECTORS),
    # - "structured_backbone_extra": first 8 backbone angles are theoretical split blocks,
    #   any additional angles are handled by the legacy extra-angle refinement path.
    # - "full_triangular": every listed beta_vector is treated as a theoretical
    #   split-triangular angle (requires each beta to satisfy injectivity).
    "multi_angle_layout": "structured_backbone_extra",
    # Optional explicit extra angles used only by structured_backbone_extra.
    "explicit_extra_beta_vectors": None,
    # Multi-angle solver:
    # - "stacked_tikhonov": reduced stacked solve in the common c-coordinate
    # - "split_triangular_admm": backbone 8 angles solved in split variables d_i=P_i c
    "multi_angle_solver_mode": "stacked_tikhonov",
    # Theoretical B1*B1 per-angle block construction:
    # - "auto": derive from multi_angle_solver_mode for backward compatibility
    # - "legacy": effective_t0 = t0 - kappa0
    # - "shifted_support": choose effective_t0 = support_left * ||beta|| + t0
    # - "legacy_injective_extension": keep t0=0.5 on the reordered d-axis and
    #   build lower bands from relative sorted injective gaps n_j - n_i
    "theoretical_formula_mode": "auto",
    # Random seed for extra angles when num_angles_total > len(beta_vectors).
    "extra_angle_seed": 20260322,
    # CNN-side angle channels:
    # - True:  only expose the fixed backbone angles to the learned update network
    # - False: expose all physical angles to the learned update network
    "cnn_backbone_only": True,
    # Optional hard cap for the learned optimizer angle subset. When set to a
    # positive integer, the learned stage uses only the first K angles even if
    # the physical operator / initialization uses more views.
    "cnn_num_angles_override": None,
    # Optional explicit CNN channel selection from the learned operator's
    # per-angle gradients. This only changes what the CNN sees; it does not
    # change the physical 16-angle operator itself.
    "cnn_angle_indices_override": None,
    # Optional explicit beta vectors used only to generate CNN feature
    # observations/gradients. When provided, the physical operator remains
    # unchanged and only the learned feature stream switches to this angle set.
    "cnn_feature_beta_vectors_override": None,
    # Optional learned-stage adapter applied only to the per-angle gradient
    # channels before they enter the CNN update network. The physical 16-angle
    # operator and Tikhonov/Morozov initialization stay unchanged.
    "cnn_angle_adapter_enabled": False,
    "cnn_angle_adapter_mode": "disabled",
    "cnn_angle_adapter_output_channels": 8,
    "cnn_angle_adapter_hidden_channels": 8,
    # Weight mu for extra (non-backbone) generic refinement views.
    "extra_angle_weight_mu": 1.0,
    # ADMM controls for the split-triangular backbone solver.
    "split_admm_rho": 0.5,
    # 16-angle full-triangular initialization becomes prohibitively slow with
    # very large iteration budgets. Use a moderate default and override it
    # explicitly for high-accuracy runs when needed.
    "split_admm_max_iter": 400,
    "split_admm_tol": 5.0e-5,
    # Progress heartbeat for the first split-ADMM initialization call.
    "split_admm_progress_interval": 50,
    # Extra-angle refinement controls for K > 8 structured solves.
    # The refinement solves for delta_c around the backbone solution:
    #   mu ||A_extra delta_c - r_extra||^2 + gamma ||delta_c||^2
    # with gamma = extra_refine_gamma_scale * lambda_backbone
    #           + extra_refine_residual_scale * ||r_extra|| / sqrt(M_extra).
    "extra_refine_gamma_scale": 32.0,
    "extra_refine_residual_scale": 32.0,
    "extra_refine_solver": "direct",
    "extra_refine_cg_iters": 20,
    "extra_refine_cg_tol": 1.0e-4,

    # Initialization strategy used by train.py / test.py before the learned updates.
    # Supported values:
    # - "cg": iterative Tikhonov solve
    # - "tikhonov_direct": direct Tikhonov solve
    # - "dual_frequency": single-angle time-domain-noise -> frequency-domain dual init
    "init_method": "tikhonov_direct",
    "init_cg_iters": 40,
    "init_cg_tol": 1.0e-4,

    # Operator uses alpha = beta / ||beta|| where beta = THEORETICAL_CONFIG["beta_vector"].
    #
    # Tensor shapes (convention):
    # - coeff: (B,1,IMAGE_SIZE,IMAGE_SIZE) real
    # - g_observed: (B,M) real
    #   * single-angle: M = num_detector_samples
    #   * multi-angle:  M = num_angles * num_detector_samples  (stacked per-angle observations)
    # - adjoint output: (B,1,IMAGE_SIZE,IMAGE_SIZE) real
    "num_detector_samples": IMAGE_SIZE * IMAGE_SIZE,
}

_experiment_profile_raw = os.environ.get("EXPERIMENT_PROFILE_OVERRIDE", None)
_apply_experiment_profile(
    DEFAULT_EXPERIMENT_PROFILE
    if _experiment_profile_raw is None
    else str(_experiment_profile_raw).strip() or DEFAULT_EXPERIMENT_PROFILE
)

# Optional runtime override: BETA_VECTORS_OVERRIDE
# - unset: use TIME_DOMAIN_CONFIG["beta_vectors"] as defined above
# - "single"/"none"/"" : disable multi-angle (fall back to single-angle beta direction)
# - "1,21;-1,21;..." or "1:21;-1:21;..." : semicolon-separated integer beta pairs
_betas_override = os.environ.get("BETA_VECTORS_OVERRIDE", None)
_total_angles_override_raw = os.environ.get("NUM_ANGLES_TOTAL_OVERRIDE", None)
_explicit_extra_betas_override = os.environ.get("EXPLICIT_EXTRA_BETA_VECTORS_OVERRIDE", None)
if _betas_override is not None:
    _s = str(_betas_override).strip()
    if _s == "" or _s.lower() in ("single", "none", "0"):
        TIME_DOMAIN_CONFIG["use_multi_angle"] = False
    else:
        _parsed = []
        try:
            for token in [t.strip() for t in _s.split(";") if t.strip()]:
                token = token.replace(":", ",")
                parts = [p.strip() for p in token.split(",") if p.strip()]
                if len(parts) != 2:
                    raise ValueError(f"Invalid beta token: {token!r}")
                _parsed.append((int(parts[0]), int(parts[1])))
        except ValueError as e:
            raise ValueError(
                "Invalid BETA_VECTORS_OVERRIDE="
                f"{_betas_override!r}; expected semicolon-separated integer pairs "
                "like '1,21;-1,21;21,1'."
            ) from e

        if len(_parsed) == 0:
            raise ValueError(
                f"Invalid BETA_VECTORS_OVERRIDE={_betas_override!r}; at least one beta pair is required."
            )
        TIME_DOMAIN_CONFIG["beta_vectors"] = _parsed
        TIME_DOMAIN_CONFIG["num_angles"] = int(len(_parsed))
        if _total_angles_override_raw is None:
            TIME_DOMAIN_CONFIG["num_angles_total"] = int(len(_parsed))
        TIME_DOMAIN_CONFIG["use_multi_angle"] = True

if _explicit_extra_betas_override is not None:
    _s = str(_explicit_extra_betas_override).strip()
    if _s:
        _parsed_extra = []
        try:
            for token in [t.strip() for t in _s.split(";") if t.strip()]:
                token = token.replace(":", ",")
                parts = [p.strip() for p in token.split(",") if p.strip()]
                if len(parts) != 2:
                    raise ValueError(f"Invalid extra beta token: {token!r}")
                _parsed_extra.append((int(parts[0]), int(parts[1])))
        except ValueError as e:
            raise ValueError(
                "Invalid EXPLICIT_EXTRA_BETA_VECTORS_OVERRIDE="
                f"{_explicit_extra_betas_override!r}; expected semicolon-separated integer pairs."
            ) from e
        TIME_DOMAIN_CONFIG["explicit_extra_beta_vectors"] = _parsed_extra

if _total_angles_override_raw is not None:
    _s = str(_total_angles_override_raw).strip()
    if _s:
        try:
            TIME_DOMAIN_CONFIG["num_angles_total"] = int(_s)
        except ValueError as e:
            raise ValueError(
                f"Invalid NUM_ANGLES_TOTAL_OVERRIDE={_total_angles_override_raw!r}; expected an integer."
            ) from e
        TIME_DOMAIN_CONFIG["num_angles"] = int(TIME_DOMAIN_CONFIG["num_angles_total"])

# Optional runtime override: NUM_DETECTOR_SAMPLES_OVERRIDE
# Controls the number of sampling points per angle (M_per_angle). For multi-angle, total M = K * M_per_angle.
_m_override = os.environ.get("NUM_DETECTOR_SAMPLES_OVERRIDE", None)
if _m_override is not None:
    _s = str(_m_override).strip()
    if _s:
        try:
            TIME_DOMAIN_CONFIG["num_detector_samples"] = int(_s)
        except ValueError as e:
            raise ValueError(
                f"Invalid NUM_DETECTOR_SAMPLES_OVERRIDE={_m_override!r}; expected an integer."
            ) from e

_apply_string_override(
    TIME_DOMAIN_CONFIG,
    "operator_mode",
    "OPERATOR_MODE_OVERRIDE",
    allowed_values={"theoretical_b1b1", "implicit_b1b1"},
)
_apply_string_override(
    TIME_DOMAIN_CONFIG,
    "init_method",
    "INIT_METHOD_OVERRIDE",
    allowed_values={"cg", "tikhonov_direct", "dual_frequency"},
)
_apply_string_override(
    TIME_DOMAIN_CONFIG,
    "multi_angle_solver_mode",
    "MULTI_ANGLE_SOLVER_MODE_OVERRIDE",
    allowed_values={"stacked_tikhonov", "split_triangular_admm"},
)
_apply_string_override(
    TIME_DOMAIN_CONFIG,
    "multi_angle_layout",
    "MULTI_ANGLE_LAYOUT_OVERRIDE",
    allowed_values={"structured_backbone_extra", "full_triangular"},
)
_apply_string_override(
    TIME_DOMAIN_CONFIG,
    "theoretical_formula_mode",
    "THEORETICAL_FORMULA_MODE_OVERRIDE",
    allowed_values={"auto", "legacy", "shifted_support", "legacy_injective_extension"},
)
_apply_bool_override(TIME_DOMAIN_CONFIG, "cnn_backbone_only", "CNN_BACKBONE_ONLY_OVERRIDE")
_apply_int_override(TIME_DOMAIN_CONFIG, "cnn_num_angles_override", "CNN_NUM_ANGLES_OVERRIDE")
_apply_int_list_override(TIME_DOMAIN_CONFIG, "cnn_angle_indices_override", "CNN_ANGLE_INDICES_OVERRIDE")
_apply_beta_vector_list_override(
    TIME_DOMAIN_CONFIG,
    "cnn_feature_beta_vectors_override",
    "CNN_FEATURE_BETA_VECTORS_OVERRIDE",
)
_apply_bool_override(TIME_DOMAIN_CONFIG, "cnn_angle_adapter_enabled", "CNN_ANGLE_ADAPTER_ENABLED_OVERRIDE")
_apply_string_override(
    TIME_DOMAIN_CONFIG,
    "cnn_angle_adapter_mode",
    "CNN_ANGLE_ADAPTER_MODE_OVERRIDE",
    allowed_values={"disabled", "adaptive_attention_mix"},
)
_apply_int_override(
    TIME_DOMAIN_CONFIG,
    "cnn_angle_adapter_output_channels",
    "CNN_ANGLE_ADAPTER_OUTPUT_CHANNELS_OVERRIDE",
)
_apply_int_override(
    TIME_DOMAIN_CONFIG,
    "cnn_angle_adapter_hidden_channels",
    "CNN_ANGLE_ADAPTER_HIDDEN_CHANNELS_OVERRIDE",
)
_apply_bool_override(TIME_DOMAIN_CONFIG, "auto_angle_t0", "AUTO_ANGLE_T0_OVERRIDE")
_apply_float_override(TIME_DOMAIN_CONFIG, "extra_angle_weight_mu", "EXTRA_ANGLE_WEIGHT_MU_OVERRIDE")
_apply_float_override(TIME_DOMAIN_CONFIG, "split_admm_rho", "SPLIT_ADMM_RHO_OVERRIDE")
_apply_float_override(TIME_DOMAIN_CONFIG, "split_admm_tol", "SPLIT_ADMM_TOL_OVERRIDE")
_apply_float_override(TIME_DOMAIN_CONFIG, "extra_refine_gamma_scale", "EXTRA_REFINE_GAMMA_SCALE_OVERRIDE")
_apply_float_override(TIME_DOMAIN_CONFIG, "extra_refine_residual_scale", "EXTRA_REFINE_RESIDUAL_SCALE_OVERRIDE")
_apply_string_override(
    TIME_DOMAIN_CONFIG,
    "extra_refine_solver",
    "EXTRA_REFINE_SOLVER_OVERRIDE",
    allowed_values={"direct", "cg"},
)
_split_iter_override = os.environ.get("SPLIT_ADMM_MAX_ITER_OVERRIDE", None)
if _split_iter_override is not None and str(_split_iter_override).strip():
    TIME_DOMAIN_CONFIG["split_admm_max_iter"] = int(str(_split_iter_override).strip())
_split_progress_interval_override = os.environ.get("SPLIT_ADMM_PROGRESS_INTERVAL_OVERRIDE", None)
if _split_progress_interval_override is not None and str(_split_progress_interval_override).strip():
    TIME_DOMAIN_CONFIG["split_admm_progress_interval"] = int(str(_split_progress_interval_override).strip())
_extra_refine_iters_override = os.environ.get("EXTRA_REFINE_CG_ITERS_OVERRIDE", None)
if _extra_refine_iters_override is not None and str(_extra_refine_iters_override).strip():
    TIME_DOMAIN_CONFIG["extra_refine_cg_iters"] = int(str(_extra_refine_iters_override).strip())
_apply_float_override(TIME_DOMAIN_CONFIG, "extra_refine_cg_tol", "EXTRA_REFINE_CG_TOL_OVERRIDE")
_seed_override = os.environ.get("EXTRA_ANGLE_SEED_OVERRIDE", None)
if _seed_override is not None and str(_seed_override).strip():
    TIME_DOMAIN_CONFIG["extra_angle_seed"] = int(str(_seed_override).strip())
TIME_DOMAIN_CONFIG["num_angles"] = int(TIME_DOMAIN_CONFIG.get("num_angles_total", TIME_DOMAIN_CONFIG.get("num_angles", 1)))

TRAINING_CONFIG = {
    "batch_size": n_data,
    "validation_interval": 10,
    "save_interval": 1000,
    "early_stopping_patience": 500,
    "gradient_clip_value": 5.0,
    # Optimizer LR for training the learned iterative network (keep conservative; unrolled optimizers are sensitive).
    "optimizer_learning_rate": 1.0e-02,
    # Relative LR multipliers for parameter sub-groups
    "scalar_lr_ratio": 0.1,
    "use_mixed_precision": False,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

_default_profile_tag = {
    "structured16_injective_extra": "structured16_injective_extra",
    "injective16_full_triangular": "injective16_full_triangular",
    "injective16_pi_best": "injective16_pi_best",
}.get(str(TIME_DOMAIN_CONFIG.get("experiment_profile", "default")).strip().lower(), "")
EXPERIMENT_OUTPUT_TAG = str(os.environ.get("OUTPUT_TAG_OVERRIDE", "") or _default_profile_tag).strip()
_model_stem = "theoretical_ct"
if EXPERIMENT_OUTPUT_TAG:
    _model_stem = f"{_model_stem}_{EXPERIMENT_OUTPUT_TAG}"

MODEL_PATH = os.path.join(MODEL_DIR, f"{_model_stem}_model.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, f"{_model_stem}_best_model.pth")
CHECKPOINT_DIR = (
    os.path.join(MODEL_DIR, f"checkpoints_{EXPERIMENT_OUTPUT_TAG}")
    if EXPERIMENT_OUTPUT_TAG
    else os.path.join(MODEL_DIR, "checkpoints")
)

LOG_DIR = (
    os.path.join(PROJECT_ROOT, "logs", EXPERIMENT_OUTPUT_TAG)
    if EXPERIMENT_OUTPUT_TAG
    else os.path.join(PROJECT_ROOT, "logs")
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOGGING_CONFIG = {
    "log_dir": LOG_DIR,
    "log_level": "INFO",
    "log_to_file": True,
    "log_to_console": True,
}


def print_config():
    """Print the current configuration for quick inspection."""
    print("=" * 60)
    print("THEORETICAL CT RECONSTRUCTION CONFIGURATION")
    print("=" * 60)
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Beta vector: {THEORETICAL_CONFIG['beta_vector']}")
    print(f"Regularizer type: {THEORETICAL_CONFIG['regularizer_type']}")
    print(f"Optimization iterations: {THEORETICAL_CONFIG['n_iter']}")
    print(f"Memory units: {THEORETICAL_CONFIG['n_memory_units']}")
    print(f"Device: {device}")
    print(f"Train data source: {DATA_CONFIG['train_data_source']}")
    print(f"Val data source: {DATA_CONFIG['val_data_source']}")
    print(f"Noise Mode: {DATA_CONFIG['noise_mode']}")
    if DATA_CONFIG['noise_mode'] == "snr":
        print(f"Target SNR (dB): {DATA_CONFIG['target_snr_db']}")
    else:
        print(f"Noise Level (delta): {DATA_CONFIG['noise_level']}")
    print(f"Data fidelity mode: {DATA_CONFIG['data_fidelity_mode']}")
    print(f"Operator mode: {TIME_DOMAIN_CONFIG['operator_mode']}")
    print(f"Experiment profile: {TIME_DOMAIN_CONFIG.get('experiment_profile', 'default')}")
    print(f"Multi-angle layout: {TIME_DOMAIN_CONFIG.get('multi_angle_layout', 'structured_backbone_extra')}")
    print(f"Lambda mode: {DATA_CONFIG['lambda_select_mode']}")
    print(f"Init method: {TIME_DOMAIN_CONFIG['init_method']}")
    print(f"Multi-angle solver mode: {TIME_DOMAIN_CONFIG['multi_angle_solver_mode']}")
    print(f"Theoretical formula mode: {TIME_DOMAIN_CONFIG.get('theoretical_formula_mode', 'auto')}")
    print(f"Backbone angles: {len(TIME_DOMAIN_CONFIG['beta_vectors'])}")
    print(f"Total angles: {TIME_DOMAIN_CONFIG['num_angles_total']}")
    print(f"CNN backbone only: {TIME_DOMAIN_CONFIG['cnn_backbone_only']}")
    print(f"CNN angle adapter enabled: {TIME_DOMAIN_CONFIG['cnn_angle_adapter_enabled']}")
    print(f"CNN angle adapter mode: {TIME_DOMAIN_CONFIG['cnn_angle_adapter_mode']}")
    print(f"CNN angle adapter output channels: {TIME_DOMAIN_CONFIG['cnn_angle_adapter_output_channels']}")
    print(f"CNN angle adapter hidden channels: {TIME_DOMAIN_CONFIG['cnn_angle_adapter_hidden_channels']}")
    print(f"Output tag: {EXPERIMENT_OUTPUT_TAG or '(default)'}")
    print(f"Training iterations: {n_train}")
    print(f"Batch size: {n_data}")
    print(f"Learning rate: {TRAINING_CONFIG['optimizer_learning_rate']}")
    print(f"Training patience: {TRAINING_CONFIG['early_stopping_patience']}")
    print(f"Model save path: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
