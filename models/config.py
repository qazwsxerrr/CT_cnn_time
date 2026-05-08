# -*- coding: utf-8 -*-
"""Alpha-only project configuration for CT_cnn."""

from __future__ import annotations

import json
import os
import sys

import torch

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CODE_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "deep_learn")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 128x128 Shepp-Logan phantom with B1*B1 pixel basis.
IMAGE_SIZE = 128

# Ensure project root is importable even when running scripts from within `models/`.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

THEORETICAL_CONFIG = {
    "regularizer_type": "tikhonov",
    "n_iter": 15,
    "n_memory_units": 16,
}

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

DEFAULT_EXPERIMENT_PROFILE = "alpha_condition"
DEFAULT_ALPHA_CONDITION_TOP_K = int(os.environ.get("ALPHA_CONDITION_TOP_K_OVERRIDE", "16"))
DEFAULT_ALPHA_CONDITION_JSON = os.path.join(
    DATA_DIR,
    "alpha_search_cache",
    f"alpha_selected{DEFAULT_ALPHA_CONDITION_TOP_K}.json",
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
    value = value.lower() if allowed_values is not None else value
    if allowed_values is not None and value not in allowed_values:
        raise ValueError(f"Invalid {env_name}={value!r}; expected one of {sorted(allowed_values)!r}.")
    target[key] = value


def _apply_float_override(target: dict, key: str, env_name: str):
    value = _get_env_override(env_name)
    if value is None:
        return
    try:
        target[key] = float(value)
    except ValueError as e:
        raise ValueError(f"Invalid {env_name}={value!r}; expected a float.") from e


def _apply_int_override(target: dict, key: str, env_name: str):
    value = _get_env_override(env_name)
    if value is None:
        return
    try:
        target[key] = int(value)
    except ValueError as e:
        raise ValueError(f"Invalid {env_name}={value!r}; expected an integer.") from e


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


def _apply_float_list_override(target: dict, key: str, env_name: str):
    value = _get_env_override(env_name)
    if value is None:
        return
    tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
    if not tokens:
        target[key] = []
        return
    try:
        target[key] = [float(token) for token in tokens]
    except ValueError as e:
        raise ValueError(f"Invalid {env_name}={value!r}; expected a comma-separated float list.") from e


def _apply_bool_override(target: dict, key: str, env_name: str):
    value = _get_env_override(env_name)
    if value is None:
        return
    value = value.lower()
    if value not in ("1", "0", "true", "false", "yes", "no", "y", "n"):
        raise ValueError(f"Invalid {env_name}={value!r}; expected a boolean-like value.")
    target[key] = value in ("1", "true", "yes", "y")


def _alpha_record_float(record: dict, *keys: str) -> float:
    for key in keys:
        if key in record:
            return float(record[key])
    raise ValueError(f"Alpha condition record is missing one of keys={keys!r}: {record!r}")


def _extract_alpha_condition_records(payload) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("selected", "top8", "best8", "results"):
            records = payload.get(key, None)
            if isinstance(records, list) and records:
                return [dict(item) for item in records]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError(
        "Alpha condition JSON must be a list or contain a non-empty list under "
        "'selected', 'top8', 'best8', or 'results'."
    )


def _load_alpha_condition_records(path: str | None = None) -> tuple[list[dict], str]:
    json_path = str(path or os.environ.get("ALPHA_CONDITION_JSON_OVERRIDE", "").strip() or DEFAULT_ALPHA_CONDITION_JSON)
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "alpha_condition profile requires an alpha-selected JSON. "
            f"Missing file: {json_path}. Run models/α_condition/alpha_condition_constrained_sampling.py "
            "or set ALPHA_CONDITION_JSON_OVERRIDE."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = _extract_alpha_condition_records(payload)
    normalized: list[dict] = []
    for record in records:
        normalized.append(
            {
                **record,
                "alpha": float(record["alpha"]),
                "tau_star": _alpha_record_float(record, "tau_star", "tau"),
                "cond": _alpha_record_float(record, "cond", "condition_number"),
                "sigma_min": float(record.get("sigma_min", float("nan"))),
                "sigma_max": float(record.get("sigma_max", float("nan"))),
            }
        )
    return normalized, json_path


DATA_CONFIG = {
    "data_source": "random_ellipses",
    "train_data_source": "random_ellipses",
    "val_data_source": "shepp_logan",
    "test_data_source": "shepp_logan",
    "noise_mode": "multiplicative",
    "noise_level": 0.1,
    "target_snr_db": 30.0,
    "lambda_select_mode": "morozov",
    "morozov_tau": 1.0,
    "morozov_max_iter": 8,
    "morozov_lambda_min": 1.0e-12,
    "morozov_lambda_max": 1.0e12,
    "morozov_newton_tol": 1.0e-10,
    "morozov_initial_lambda": 1.0,
    "morozov_cache_dir": os.path.join(DATA_DIR, "morozov_cache"),
    "alpha_gram_cache_dir": os.path.join(DATA_DIR, "alpha_gram_cache"),
    "morozov_cg_iters": 12,
    "morozov_cg_tol": 1.0e-4,
    "lambda_reg": 1.0e-02,
    "implicit_eval_solver": "cg",
    "implicit_eval_lambda_min": 1.0e-02,
    "implicit_eval_cg_iters": 80,
    "implicit_eval_cg_tol": 1.0e-4,
    "data_fidelity_mode": "standard",
    "irls_eps_factor": 3.0e-03,
    "irls_detach_weights": True,
    "detach_physical_grads": False,
    "learned_reg_lambda_init": 1.0e-02,
    "learned_step_init": 2.0e-03,
    "learned_step_min": 1.0e-06,
    "learned_step_max": 1.0e-02,
    "learned_reg_lambda_max": 1.0e-01,
    "learned_correction_max": 0.0,
    "update_max_norm": 0.0,
    "validation_seed": 42,
    "val_batch_size": n_data,
    "val_reproducible": True,
    "intermediate_supervision_enabled": False,
    "intermediate_supervision_weight_start": 0.2,
    "intermediate_supervision_weight_end": 1.0,
}

_apply_string_override(DATA_CONFIG, "train_data_source", "TRAIN_DATA_SOURCE_OVERRIDE", allowed_values={"shepp_logan", "random_ellipses", "random_ellipse", "ellipse"})
_apply_string_override(DATA_CONFIG, "val_data_source", "VAL_DATA_SOURCE_OVERRIDE", allowed_values={"shepp_logan", "random_ellipses", "random_ellipse", "ellipse"})
_apply_string_override(DATA_CONFIG, "test_data_source", "TEST_DATA_SOURCE_OVERRIDE", allowed_values={"shepp_logan", "random_ellipses", "random_ellipse", "ellipse"})
_apply_string_override(DATA_CONFIG, "noise_mode", "NOISE_MODE_OVERRIDE", allowed_values={"additive", "multiplicative", "snr"})
_apply_float_override(DATA_CONFIG, "noise_level", "NOISE_LEVEL_OVERRIDE")
_apply_float_override(DATA_CONFIG, "target_snr_db", "TARGET_SNR_DB_OVERRIDE")
_apply_string_override(DATA_CONFIG, "lambda_select_mode", "LAMBDA_SELECT_MODE_OVERRIDE", allowed_values={"fixed", "morozov"})
_apply_float_override(DATA_CONFIG, "lambda_reg", "LAMBDA_REG_OVERRIDE")
_apply_float_override(DATA_CONFIG, "morozov_tau", "MOROZOV_TAU_OVERRIDE")
_apply_float_override(DATA_CONFIG, "morozov_lambda_min", "MOROZOV_LAMBDA_MIN_OVERRIDE")
_apply_float_override(DATA_CONFIG, "morozov_lambda_max", "MOROZOV_LAMBDA_MAX_OVERRIDE")
_apply_float_override(DATA_CONFIG, "morozov_newton_tol", "MOROZOV_NEWTON_TOL_OVERRIDE")
_apply_float_override(DATA_CONFIG, "morozov_initial_lambda", "MOROZOV_INITIAL_LAMBDA_OVERRIDE")
_apply_string_override(DATA_CONFIG, "morozov_cache_dir", "MOROZOV_CACHE_DIR_OVERRIDE")
_apply_string_override(DATA_CONFIG, "alpha_gram_cache_dir", "ALPHA_GRAM_CACHE_DIR_OVERRIDE")
_apply_string_override(DATA_CONFIG, "data_fidelity_mode", "DATA_FIDELITY_MODE_OVERRIDE", allowed_values={"standard", "irls"})
_apply_bool_override(DATA_CONFIG, "detach_physical_grads", "DETACH_PHYSICAL_GRADS_OVERRIDE")
_apply_bool_override(DATA_CONFIG, "intermediate_supervision_enabled", "INTERMEDIATE_SUPERVISION_ENABLED_OVERRIDE")
_apply_float_override(DATA_CONFIG, "intermediate_supervision_weight_start", "INTERMEDIATE_SUPERVISION_WEIGHT_START_OVERRIDE")
_apply_float_override(DATA_CONFIG, "intermediate_supervision_weight_end", "INTERMEDIATE_SUPERVISION_WEIGHT_END_OVERRIDE")

TIME_DOMAIN_CONFIG = {
    "operator_mode": "theoretical_b1b1",
    "experiment_profile": DEFAULT_EXPERIMENT_PROFILE,
    "use_multi_angle": True,
    "num_angles_total": DEFAULT_ALPHA_CONDITION_TOP_K,
    "num_angles": DEFAULT_ALPHA_CONDITION_TOP_K,
    "alpha_values": [],
    "alpha_tau_offsets": [],
    "alpha_condition_constrained_records": None,
    "alpha_condition_constrained_json": None,
    "theoretical_formula_mode": "alpha_continuous",
    "data_formula_mode": "auto_complete",
    "multi_angle_solver_mode": "stacked_tikhonov",
    "cnn_backbone_only": False,
    "cnn_num_angles_override": None,
    "cnn_angle_indices_override": None,
    "cnn_angle_adapter_enabled": False,
    "cnn_angle_adapter_mode": "disabled",
    "cnn_angle_adapter_output_channels": DEFAULT_ALPHA_CONDITION_TOP_K,
    "cnn_angle_adapter_hidden_channels": min(DEFAULT_ALPHA_CONDITION_TOP_K, 8),
    "physics_residual_channel_enabled": True,
    "physics_residual_mode": "stacked_cg",
    "physics_residual_damping": 1.0e-2,
    "physics_residual_cg_iters": 8,
    "physics_residual_detach": True,
    "physics_residual_normalize": True,
    "physics_explicit_update_enabled": False,
    "physics_explicit_update_alpha_init": 0.02,
    "physics_explicit_update_max": 0.10,
    "init_method": "tikhonov_direct",
    "init_cg_iters": 40,
    "init_cg_tol": 1.0e-4,
    "num_detector_samples": IMAGE_SIZE * IMAGE_SIZE,
}


def _apply_alpha_condition_profile(json_path: str | None = None) -> None:
    records, resolved_json = _load_alpha_condition_records(path=json_path)
    alpha_values = [float(item["alpha"]) for item in records]
    tau_offsets = [float(item["tau_star"]) for item in records]
    num_angles = int(len(alpha_values))

    TIME_DOMAIN_CONFIG["experiment_profile"] = "alpha_condition"
    TIME_DOMAIN_CONFIG["operator_mode"] = "theoretical_b1b1"
    TIME_DOMAIN_CONFIG["use_multi_angle"] = True
    TIME_DOMAIN_CONFIG["alpha_values"] = alpha_values
    TIME_DOMAIN_CONFIG["alpha_tau_offsets"] = tau_offsets
    TIME_DOMAIN_CONFIG["alpha_condition_constrained_records"] = records
    TIME_DOMAIN_CONFIG["alpha_condition_constrained_json"] = str(resolved_json)
    TIME_DOMAIN_CONFIG["num_angles_total"] = num_angles
    TIME_DOMAIN_CONFIG["num_angles"] = num_angles
    TIME_DOMAIN_CONFIG["theoretical_formula_mode"] = "alpha_continuous"
    TIME_DOMAIN_CONFIG["data_formula_mode"] = "auto_complete"
    TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = "stacked_tikhonov"
    TIME_DOMAIN_CONFIG["cnn_backbone_only"] = False
    TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = num_angles
    TIME_DOMAIN_CONFIG["cnn_angle_indices_override"] = None
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_enabled"] = False
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_mode"] = "disabled"
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_output_channels"] = num_angles
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_hidden_channels"] = min(num_angles, 8)
    TIME_DOMAIN_CONFIG["physics_residual_channel_enabled"] = True
    TIME_DOMAIN_CONFIG["physics_residual_mode"] = "stacked_cg"
    TIME_DOMAIN_CONFIG["physics_residual_damping"] = 1.0e-2
    TIME_DOMAIN_CONFIG["physics_residual_cg_iters"] = 8
    TIME_DOMAIN_CONFIG["physics_residual_detach"] = True
    TIME_DOMAIN_CONFIG["physics_residual_normalize"] = True
    TIME_DOMAIN_CONFIG["physics_explicit_update_enabled"] = False
    TIME_DOMAIN_CONFIG["physics_explicit_update_alpha_init"] = 0.02
    TIME_DOMAIN_CONFIG["physics_explicit_update_max"] = 0.10
    TIME_DOMAIN_CONFIG["init_method"] = "tikhonov_direct"


def _apply_runtime_alpha_profile() -> None:
    TIME_DOMAIN_CONFIG["experiment_profile"] = "runtime_alpha"
    TIME_DOMAIN_CONFIG["operator_mode"] = "theoretical_b1b1"
    TIME_DOMAIN_CONFIG["use_multi_angle"] = False
    TIME_DOMAIN_CONFIG["num_angles_total"] = 1
    TIME_DOMAIN_CONFIG["num_angles"] = 1
    TIME_DOMAIN_CONFIG["alpha_values"] = []
    TIME_DOMAIN_CONFIG["alpha_tau_offsets"] = []
    TIME_DOMAIN_CONFIG["alpha_condition_constrained_records"] = None
    TIME_DOMAIN_CONFIG["alpha_condition_constrained_json"] = None
    TIME_DOMAIN_CONFIG["theoretical_formula_mode"] = "alpha_continuous"
    TIME_DOMAIN_CONFIG["data_formula_mode"] = "auto_complete"
    TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = "stacked_tikhonov"
    TIME_DOMAIN_CONFIG["cnn_backbone_only"] = False
    TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = None


def _apply_experiment_profile(profile_name: str) -> None:
    profile = str(profile_name or "").strip().lower()
    if profile in ("", "default", "none"):
        profile = DEFAULT_EXPERIMENT_PROFILE
    if profile in {"runtime_alpha", "runtime", "minimal"}:
        _apply_runtime_alpha_profile()
        return
    if profile in {"alpha_condition", "alpha_condition_constrained"}:
        _apply_alpha_condition_profile(json_path=os.environ.get("ALPHA_CONDITION_JSON_OVERRIDE", "").strip() or None)
        return
    raise ValueError(
        f"Unsupported EXPERIMENT_PROFILE_OVERRIDE={profile_name!r}; "
        "expected 'alpha_condition' or 'runtime_alpha'."
    )


_experiment_profile_raw = os.environ.get("EXPERIMENT_PROFILE_OVERRIDE", None)
_apply_experiment_profile(DEFAULT_EXPERIMENT_PROFILE if _experiment_profile_raw is None else str(_experiment_profile_raw).strip() or DEFAULT_EXPERIMENT_PROFILE)

_total_angles_override_raw = os.environ.get("NUM_ANGLES_TOTAL_OVERRIDE", None)
if _total_angles_override_raw is not None:
    _s = str(_total_angles_override_raw).strip()
    if _s:
        try:
            TIME_DOMAIN_CONFIG["num_angles_total"] = int(_s)
            TIME_DOMAIN_CONFIG["num_angles"] = int(_s)
        except ValueError as e:
            raise ValueError(f"Invalid NUM_ANGLES_TOTAL_OVERRIDE={_total_angles_override_raw!r}; expected an integer.") from e

_apply_float_list_override(TIME_DOMAIN_CONFIG, "alpha_values", "ALPHA_VALUES_OVERRIDE")
_apply_float_list_override(TIME_DOMAIN_CONFIG, "alpha_tau_offsets", "ALPHA_TAU_OFFSETS_OVERRIDE")
if TIME_DOMAIN_CONFIG.get("alpha_values"):
    _alpha_k = int(len(TIME_DOMAIN_CONFIG.get("alpha_values") or []))
    if TIME_DOMAIN_CONFIG.get("alpha_tau_offsets") and len(TIME_DOMAIN_CONFIG["alpha_tau_offsets"]) != _alpha_k:
        raise ValueError("ALPHA_VALUES_OVERRIDE and ALPHA_TAU_OFFSETS_OVERRIDE must have the same length.")
    TIME_DOMAIN_CONFIG["num_angles_total"] = _alpha_k
    TIME_DOMAIN_CONFIG["num_angles"] = _alpha_k
    TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = _alpha_k
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_output_channels"] = _alpha_k
    TIME_DOMAIN_CONFIG["cnn_angle_adapter_hidden_channels"] = min(_alpha_k, 8)

_m_override = os.environ.get("NUM_DETECTOR_SAMPLES_OVERRIDE", None)
if _m_override is not None:
    _s = str(_m_override).strip()
    if _s:
        try:
            TIME_DOMAIN_CONFIG["num_detector_samples"] = int(_s)
        except ValueError as e:
            raise ValueError(f"Invalid NUM_DETECTOR_SAMPLES_OVERRIDE={_m_override!r}; expected an integer.") from e

_apply_string_override(TIME_DOMAIN_CONFIG, "operator_mode", "OPERATOR_MODE_OVERRIDE", allowed_values={"theoretical_b1b1"})
_apply_string_override(TIME_DOMAIN_CONFIG, "init_method", "INIT_METHOD_OVERRIDE", allowed_values={"cg", "tikhonov_direct"})
_apply_string_override(TIME_DOMAIN_CONFIG, "multi_angle_solver_mode", "MULTI_ANGLE_SOLVER_MODE_OVERRIDE", allowed_values={"stacked_tikhonov"})
_apply_string_override(TIME_DOMAIN_CONFIG, "theoretical_formula_mode", "THEORETICAL_FORMULA_MODE_OVERRIDE", allowed_values={"alpha_continuous"})
_apply_bool_override(TIME_DOMAIN_CONFIG, "cnn_backbone_only", "CNN_BACKBONE_ONLY_OVERRIDE")
_apply_int_override(TIME_DOMAIN_CONFIG, "cnn_num_angles_override", "CNN_NUM_ANGLES_OVERRIDE")
_apply_int_list_override(TIME_DOMAIN_CONFIG, "cnn_angle_indices_override", "CNN_ANGLE_INDICES_OVERRIDE")
_apply_bool_override(TIME_DOMAIN_CONFIG, "cnn_angle_adapter_enabled", "CNN_ANGLE_ADAPTER_ENABLED_OVERRIDE")
_apply_string_override(TIME_DOMAIN_CONFIG, "cnn_angle_adapter_mode", "CNN_ANGLE_ADAPTER_MODE_OVERRIDE", allowed_values={"disabled", "adaptive_attention_mix"})
_apply_int_override(TIME_DOMAIN_CONFIG, "cnn_angle_adapter_output_channels", "CNN_ANGLE_ADAPTER_OUTPUT_CHANNELS_OVERRIDE")
_apply_int_override(TIME_DOMAIN_CONFIG, "cnn_angle_adapter_hidden_channels", "CNN_ANGLE_ADAPTER_HIDDEN_CHANNELS_OVERRIDE")
_apply_bool_override(TIME_DOMAIN_CONFIG, "physics_residual_channel_enabled", "PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE")
_apply_string_override(TIME_DOMAIN_CONFIG, "physics_residual_mode", "PHYSICS_RESIDUAL_MODE_OVERRIDE", allowed_values={"stacked_cg"})
_apply_float_override(TIME_DOMAIN_CONFIG, "physics_residual_damping", "PHYSICS_RESIDUAL_DAMPING_OVERRIDE")
_apply_int_override(TIME_DOMAIN_CONFIG, "physics_residual_cg_iters", "PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE")
_apply_bool_override(TIME_DOMAIN_CONFIG, "physics_residual_detach", "PHYSICS_RESIDUAL_DETACH_OVERRIDE")
_apply_bool_override(TIME_DOMAIN_CONFIG, "physics_residual_normalize", "PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE")
_apply_bool_override(TIME_DOMAIN_CONFIG, "physics_explicit_update_enabled", "PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE")
_apply_float_override(TIME_DOMAIN_CONFIG, "physics_explicit_update_alpha_init", "PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE")
_apply_float_override(TIME_DOMAIN_CONFIG, "physics_explicit_update_max", "PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE")
TIME_DOMAIN_CONFIG["num_angles"] = int(TIME_DOMAIN_CONFIG.get("num_angles_total", TIME_DOMAIN_CONFIG.get("num_angles", 1)))

TRAINING_CONFIG = {
    "batch_size": n_data,
    "validation_interval": 10,
    "save_interval": 1000,
    "early_stopping_patience": 500,
    "gradient_clip_value": 5.0,
    "optimizer_learning_rate": 1.0e-02,
    "scalar_lr_ratio": 0.1,
    "use_mixed_precision": False,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

_default_profile_tag = {
    "alpha_condition": "alpha_condition",
    "alpha_condition_constrained": "alpha_condition",
    "runtime_alpha": "runtime_alpha",
}.get(str(TIME_DOMAIN_CONFIG.get("experiment_profile", "default")).strip().lower(), "")
EXPERIMENT_OUTPUT_TAG = str(os.environ.get("OUTPUT_TAG_OVERRIDE", "") or _default_profile_tag).strip()
_model_stem = "theoretical_ct"
if EXPERIMENT_OUTPUT_TAG:
    _model_stem = f"{_model_stem}_{EXPERIMENT_OUTPUT_TAG}"

MODEL_PATH = os.path.join(MODEL_DIR, f"{_model_stem}_model.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, f"{_model_stem}_best_model.pth")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, f"checkpoints_{EXPERIMENT_OUTPUT_TAG}") if EXPERIMENT_OUTPUT_TAG else os.path.join(MODEL_DIR, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", EXPERIMENT_OUTPUT_TAG) if EXPERIMENT_OUTPUT_TAG else os.path.join(PROJECT_ROOT, "logs")

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
    """Print the current alpha-only configuration for quick inspection."""
    print("=" * 60)
    print("ALPHA-ONLY CT RECONSTRUCTION CONFIGURATION")
    print("=" * 60)
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
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
    print(f"Lambda mode: {DATA_CONFIG['lambda_select_mode']}")
    print(f"Init method: {TIME_DOMAIN_CONFIG['init_method']}")
    print(f"Solver mode: {TIME_DOMAIN_CONFIG['multi_angle_solver_mode']}")
    print(f"Formula mode: {TIME_DOMAIN_CONFIG.get('theoretical_formula_mode', 'alpha_continuous')}")
    print(f"Alpha angles: {len(TIME_DOMAIN_CONFIG.get('alpha_values') or [])}")
    if TIME_DOMAIN_CONFIG.get("alpha_condition_constrained_json"):
        print(f"Alpha JSON: {TIME_DOMAIN_CONFIG['alpha_condition_constrained_json']}")
    print(f"CNN angle adapter enabled: {TIME_DOMAIN_CONFIG['cnn_angle_adapter_enabled']}")
    print(f"CNN angle adapter mode: {TIME_DOMAIN_CONFIG['cnn_angle_adapter_mode']}")
    print(f"CNN angle adapter output channels: {TIME_DOMAIN_CONFIG['cnn_angle_adapter_output_channels']}")
    print(f"CNN angle adapter hidden channels: {TIME_DOMAIN_CONFIG['cnn_angle_adapter_hidden_channels']}")
    print(f"Physics residual channel: {TIME_DOMAIN_CONFIG['physics_residual_channel_enabled']}")
    print(f"Output tag: {EXPERIMENT_OUTPUT_TAG or '(default)'}")
    print(f"Training iterations: {n_train}")
    print(f"Batch size: {n_data}")
    print(f"Learning rate: {TRAINING_CONFIG['optimizer_learning_rate']}")
    print(f"Training patience: {TRAINING_CONFIG['early_stopping_patience']}")
    print(f"Model save path: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
