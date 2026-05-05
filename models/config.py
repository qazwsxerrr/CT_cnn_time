# -*- coding: utf-8 -*-
"""Project configuration for CT_cnn."""

from __future__ import annotations

import os
import sys
import json
import torch

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

DEFAULT_EXPERIMENT_PROFILE = "condition_constrained8_pi"
CONDITION_CONSTRAINED8_PI_JSON = os.path.join(MODEL_DIR, "best_condition_constrained8_pi.json")


def _condition_record_float(record: dict, *keys: str) -> float:
    for key in keys:
        if key in record:
            return float(record[key])
    raise ValueError(f"Condition-constrained record is missing one of keys={keys!r}: {record!r}")


def _extract_condition_constrained_records(payload) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("top8", "best8", "results"):
            records = payload.get(key, None)
            if isinstance(records, list):
                return [dict(item) for item in records]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError(
        "Condition-constrained JSON must be a list or contain a list under 'top8', 'best8', or 'results'."
    )


def _load_condition_constrained8_records(path: str | None = None) -> list[dict]:
    json_path = str(
        path
        or os.environ.get("CONDITION_CONSTRAINED8_PI_JSON_OVERRIDE", "").strip()
        or CONDITION_CONSTRAINED8_PI_JSON
    )
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "condition_constrained8_pi profile requires best-condition JSON. "
            f"Missing file: {json_path}. Run models/condition_constrained_sampling.py first "
            "or set CONDITION_CONSTRAINED8_PI_JSON_OVERRIDE."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = _extract_condition_constrained_records(payload)
    if len(records) < 8:
        raise ValueError(f"condition_constrained8_pi requires at least 8 records, got {len(records)} from {json_path}.")

    normalized: list[dict] = []
    for idx, record in enumerate(records[:8], start=1):
        if "beta" not in record:
            raise ValueError(f"Condition-constrained record #{idx} is missing 'beta': {record!r}")
        beta = tuple(int(v) for v in list(record["beta"]))
        if len(beta) != 2:
            raise ValueError(f"Condition-constrained record #{idx} beta must have length 2, got {record['beta']!r}")
        normalized.append(
            {
                "beta": beta,
                "tau_star": _condition_record_float(record, "tau_star", "tau", "band_t0"),
                "cond": _condition_record_float(record, "cond", "condition_number"),
                "sigma_min": float(record.get("sigma_min", float("nan"))),
                "sigma_max": float(record.get("sigma_max", float("nan"))),
            }
        )
    if len({item["beta"] for item in normalized}) != 8:
        raise ValueError("condition_constrained8_pi JSON must contain 8 distinct beta vectors.")
    return normalized


def _apply_condition_constrained8_pi_profile(json_path: str | None = None) -> None:
    records = _load_condition_constrained8_records(path=json_path)
    beta_list = [item["beta"] for item in records]
    tau_offsets = [float(item["tau_star"]) for item in records]

    # Keep the best absolute Tikhonov initialization as the only retained
    # initialization path for the condition-constrained experiment.
    TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = "stacked_tikhonov"
    TIME_DOMAIN_CONFIG["multi_angle_layout"] = "full_triangular"
    TIME_DOMAIN_CONFIG["beta_vectors"] = beta_list
    TIME_DOMAIN_CONFIG["num_angles_total"] = 8
    TIME_DOMAIN_CONFIG["num_angles"] = 8
    TIME_DOMAIN_CONFIG["cnn_backbone_only"] = False
    TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = 8
    TIME_DOMAIN_CONFIG["cnn_angle_indices_override"] = None
    TIME_DOMAIN_CONFIG["cnn_feature_beta_vectors_override"] = None
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_enabled"] = False
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_mode"] = "disabled"
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_output_channels"] = 8
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_hidden_channels"] = 8
    TIME_DOMAIN_CONFIG["theoretical_formula_mode"] = "condition_constrained_offset"
    TIME_DOMAIN_CONFIG["data_formula_mode"] = "auto_complete"
    TIME_DOMAIN_CONFIG["condition_constrained_tau_offsets"] = tau_offsets
    TIME_DOMAIN_CONFIG["condition_constrained_records"] = records
    TIME_DOMAIN_CONFIG["condition_constrained_json"] = str(
        json_path
        or os.environ.get("CONDITION_CONSTRAINED8_PI_JSON_OVERRIDE", "").strip()
        or CONDITION_CONSTRAINED8_PI_JSON
    )
    TIME_DOMAIN_CONFIG["auto_angle_t0"] = False


def _apply_same8_shifted_support_triangular_pi_profile(json_path: str | None = None) -> None:
    """Same 8 angles as condition_constrained8_pi, using the exact triangular legacy baseline."""
    records = _load_condition_constrained8_records(
        path=json_path
        or os.environ.get("CONDITION_CONSTRAINED8_PI_JSON_OVERRIDE", "").strip()
        or CONDITION_CONSTRAINED8_PI_JSON
    )
    beta_list = [item["beta"] for item in records]

    TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = "stacked_tikhonov"
    TIME_DOMAIN_CONFIG["multi_angle_layout"] = "full_triangular"
    TIME_DOMAIN_CONFIG["beta_vectors"] = beta_list
    TIME_DOMAIN_CONFIG["num_angles_total"] = 8
    TIME_DOMAIN_CONFIG["num_angles"] = 8
    TIME_DOMAIN_CONFIG["cnn_backbone_only"] = False
    TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = 8
    TIME_DOMAIN_CONFIG["cnn_angle_indices_override"] = None
    TIME_DOMAIN_CONFIG["cnn_feature_beta_vectors_override"] = None
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_enabled"] = False
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_mode"] = "disabled"
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_output_channels"] = 8
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_hidden_channels"] = 8
    TIME_DOMAIN_CONFIG["theoretical_formula_mode"] = "legacy_injective_extension"
    TIME_DOMAIN_CONFIG["data_formula_mode"] = "auto_complete"
    TIME_DOMAIN_CONFIG["condition_constrained_tau_offsets"] = None
    TIME_DOMAIN_CONFIG["condition_constrained_records"] = None
    TIME_DOMAIN_CONFIG["condition_constrained_json"] = None
    TIME_DOMAIN_CONFIG["auto_angle_t0"] = True


def _apply_experiment_profile(profile_name: str) -> None:
    profile = str(profile_name or "").strip().lower()
    if profile in ("", "default", "none"):
        profile = DEFAULT_EXPERIMENT_PROFILE

    TIME_DOMAIN_CONFIG["experiment_profile"] = profile
    TIME_DOMAIN_CONFIG["operator_mode"] = "theoretical_b1b1"
    TIME_DOMAIN_CONFIG["use_multi_angle"] = True
    TIME_DOMAIN_CONFIG["multi_angle_layout"] = "full_triangular"
    TIME_DOMAIN_CONFIG["auto_angle_t0"] = True
    TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = "stacked_tikhonov"
    TIME_DOMAIN_CONFIG["data_formula_mode"] = "auto_complete"

    if profile == "condition_constrained8_pi":
        _apply_condition_constrained8_pi_profile(
            json_path=os.environ.get("CONDITION_CONSTRAINED8_PI_JSON_OVERRIDE", "").strip()
            or CONDITION_CONSTRAINED8_PI_JSON
        )
        return

    if profile == "same8_shifted_support_triangular_pi":
        _apply_same8_shifted_support_triangular_pi_profile()
        return

    raise ValueError(
        f"Unsupported EXPERIMENT_PROFILE_OVERRIDE={profile_name!r}; "
        "expected one of 'condition_constrained8_pi' or "
        "'same8_shifted_support_triangular_pi'."
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
    "data_fidelity_mode",
    "DATA_FIDELITY_MODE_OVERRIDE",
    allowed_values={"standard", "irls"},
)
_apply_bool_override(DATA_CONFIG, "detach_physical_grads", "DETACH_PHYSICAL_GRADS_OVERRIDE")

TIME_DOMAIN_CONFIG = {
    # Main experiment uses the theoretical B1*B1 time-domain operator.
    "operator_mode": "theoretical_b1b1",
    "experiment_profile": DEFAULT_EXPERIMENT_PROFILE,
    "sampling_scheme": "paper_grid_t0",
    "sampling_t0": 0.5,
    # legacy_injective_extension uses the exact triangular support-left shift,
    # while condition_constrained_offset uses per-angle tau offsets.
    "auto_angle_t0": False,
    "sampling_seed": 123,

    # Angle mode switch:
    # - True:  multi-angle, uses beta_vectors list below
    # - False: single-angle, uses the paper's eligible direction beta=(1,21)
    "use_multi_angle": True,

    # Only the 8-angle full-triangular layout is retained.
    "num_angles_total": 8,
    "beta_vectors": [],
    "multi_angle_layout": "full_triangular",
    "multi_angle_solver_mode": "stacked_tikhonov",
    # Only the two retained sampling/matrix constructions remain:
    # - condition_constrained_offset: full sparse matrix with per-angle tau offsets
    # - legacy_injective_extension: exact lower-banded triangular matrix
    "theoretical_formula_mode": "condition_constrained_offset",
    # Observation generation reuses the same exact operator for each retained method.
    "data_formula_mode": "auto_complete",
    "condition_constrained_tau_offsets": None,
    "condition_constrained_records": None,
    "condition_constrained_json": CONDITION_CONSTRAINED8_PI_JSON,
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
    # Initialization strategy used by train.py / test.py before the learned updates.
    # Supported values:
    # - "cg": iterative Tikhonov solve
    # - "tikhonov_direct": direct Tikhonov solve
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
    allowed_values={"cg", "tikhonov_direct"},
)
_apply_string_override(
    TIME_DOMAIN_CONFIG,
    "multi_angle_solver_mode",
    "MULTI_ANGLE_SOLVER_MODE_OVERRIDE",
    allowed_values={"stacked_tikhonov"},
)
_apply_string_override(
    TIME_DOMAIN_CONFIG,
    "multi_angle_layout",
    "MULTI_ANGLE_LAYOUT_OVERRIDE",
    allowed_values={"full_triangular"},
)
_apply_string_override(
    TIME_DOMAIN_CONFIG,
    "theoretical_formula_mode",
    "THEORETICAL_FORMULA_MODE_OVERRIDE",
    allowed_values={
        "legacy_injective_extension",
        "condition_constrained_offset",
    },
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
    "condition_constrained8_pi": "condition_constrained8_pi",
    "same8_shifted_support_triangular_pi": "same8_shifted_support_triangular_pi",
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
    print(f"Multi-angle layout: {TIME_DOMAIN_CONFIG.get('multi_angle_layout', 'full_triangular')}")
    print(f"Lambda mode: {DATA_CONFIG['lambda_select_mode']}")
    print(f"Init method: {TIME_DOMAIN_CONFIG['init_method']}")
    print(f"Multi-angle solver mode: {TIME_DOMAIN_CONFIG['multi_angle_solver_mode']}")
    print(f"Theoretical formula mode: {TIME_DOMAIN_CONFIG.get('theoretical_formula_mode', 'auto')}")
    print(f"Data formula mode: {TIME_DOMAIN_CONFIG.get('data_formula_mode', 'auto_complete')}")
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
