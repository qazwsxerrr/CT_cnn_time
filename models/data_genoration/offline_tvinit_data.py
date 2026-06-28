from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = ("coeff_true", "g_observed", "coeff_initial")
METADATA_KEY = "offline_tvinit_metadata"
ANGLE_ENV_KEYS = (
    "ALPHA_CONDITION_TOP_K_OVERRIDE",
    "ALPHA_CONDITION_JSON_OVERRIDE",
)
# Only keys that change the offline tensors themselves belong in metadata.
# CNN/refiner/physics-residual feature settings are NN-only and may vary at
# training time without invalidating a precomputed TV-init dataset.
TVINIT_ENV_KEYS = (
    "EXPERIMENT_PROFILE_OVERRIDE",
    *ANGLE_ENV_KEYS,
    "SAMPLING_MODE_OVERRIDE",
    "NUM_DETECTOR_SAMPLES_OVERRIDE",
    "DETECTOR_PHASE_OVERRIDE",
    "DETECTOR_MARGIN_RATIO_OVERRIDE",
    "ALPHA_GRAM_CACHE_DIR_OVERRIDE",
    "MULTI_ANGLE_SOLVER_MODE_OVERRIDE",
    "THEORETICAL_FORMULA_MODE_OVERRIDE",
    "INIT_METHOD_OVERRIDE",
    "LAMBDA_SELECT_MODE_OVERRIDE",
    "MOROZOV_FORM_OVERRIDE",
    "MOROZOV_NOISE_RADIUS_MODE_OVERRIDE",
    "MOROZOV_TAU_OVERRIDE",
    "L1_INIT_ADMM_ITERS_OVERRIDE",
    "L1_INIT_ADMM_CG_ITERS_OVERRIDE",
    "L1_INIT_ADMM_CG_TOL_OVERRIDE",
    "L1_INIT_ADMM_RHO_DATA_OVERRIDE",
    "L1_INIT_ADMM_RHO_REG_OVERRIDE",
    "REGULARIZER_TYPE_OVERRIDE",
    "NOISE_MODE_OVERRIDE",
    "NOISE_LEVEL_OVERRIDE",
)

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
PROJECT_ROOT = MODELS_DIR.parent
DEFAULT_OUTPUT_PATH = THIS_DIR / "offline_tvinit_dataset.pt"
DEFAULT_SPLIT_OUTPUT_DIR = PROJECT_ROOT / "data" / "data_genoration"


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _load_torch_file(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a torch-saved mapping, got {type(data).__name__}.")
    return data


def _as_coeff_batch(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().to(dtype=torch.float32)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.dim() != 4 or int(tensor.shape[1]) != 1:
        raise ValueError(f"{name} must have shape (N,H,W) or (N,1,H,W), got {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _as_observation_batch(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().to(dtype=torch.float32)
    if tensor.dim() == 3 and int(tensor.shape[1]) == 1:
        tensor = tensor.squeeze(1)
    if tensor.dim() != 2:
        raise ValueError(f"g_observed must have shape (N,M) or (N,1,M), got {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _validate_same_batch_size(tensors: Mapping[str, torch.Tensor]) -> None:
    sizes = {key: int(tensors[key].shape[0]) for key in REQUIRED_KEYS}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Offline tensors have inconsistent batch sizes: {sizes!r}.")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return _json_safe(dict(metadata))


def load_offline_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return embedded TV-init metadata, or an empty dict for legacy files."""
    raw = _load_torch_file(path)
    metadata = raw.get(METADATA_KEY, {})
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{METADATA_KEY} must be a mapping, got {type(metadata).__name__}.")
    return dict(metadata)


def _metadata_env(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    env = metadata.get("env", {})
    if isinstance(env, Mapping):
        return env
    return {}


def _env_value_for_compare(key: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if key in {"ALPHA_CONDITION_JSON_OVERRIDE", "ALPHA_GRAM_CACHE_DIR_OVERRIDE"}:
        return os.path.normcase(os.path.abspath(str(_resolve_project_path(text))))
    return text


def _float_lists_match(stored: Any, current: Any) -> bool:
    if not isinstance(stored, list) or not isinstance(current, list):
        return False
    if len(stored) != len(current):
        return False
    try:
        return all(abs(float(a) - float(b)) <= 1.0e-12 for a, b in zip(stored, current))
    except (TypeError, ValueError):
        return False


def _alpha_json_matches_metadata(metadata: Mapping[str, Any], current_path: Any) -> bool:
    """Accept moved alpha JSON files when their selected angles are unchanged."""
    angle_selection = metadata.get("angle_selection", {})
    stored_path = _metadata_env(metadata).get("ALPHA_CONDITION_JSON_OVERRIDE", "")
    if isinstance(angle_selection, Mapping):
        stored_path = angle_selection.get("alpha_json_path", stored_path)
    if _env_value_for_compare("ALPHA_CONDITION_JSON_OVERRIDE", current_path) == _env_value_for_compare(
        "ALPHA_CONDITION_JSON_OVERRIDE", stored_path
    ):
        return True

    if not isinstance(angle_selection, Mapping):
        return False
    stored_summary = angle_selection.get("alpha_json_summary", {})
    if not isinstance(stored_summary, Mapping):
        return False
    current_summary = _read_alpha_json_summary(current_path)
    if not current_summary.get("exists", False):
        return False

    stored_count = stored_summary.get("selected_count", stored_summary.get("top_k"))
    current_count = current_summary.get("selected_count", current_summary.get("top_k"))
    if stored_count is not None and current_count is not None and int(stored_count) != int(current_count):
        return False
    if not _float_lists_match(stored_summary.get("alpha_values", []), current_summary.get("alpha_values", [])):
        return False
    stored_tau = stored_summary.get("tau_offsets", [])
    current_tau = current_summary.get("tau_offsets", [])
    if stored_tau or current_tau:
        return _float_lists_match(stored_tau, current_tau)
    return True


def validate_offline_metadata_against_env(metadata: Mapping[str, Any]) -> None:
    """Fail early when a metadata-tagged offline file conflicts with active env overrides."""
    stored_env = _metadata_env(metadata)
    if not stored_env and not isinstance(metadata.get("resolved_config", None), Mapping):
        return
    mismatches: list[str] = []
    for key in TVINIT_ENV_KEYS:
        current = os.environ.get(key, None)
        if current is None or str(current).strip() == "":
            continue
        stored = stored_env.get(key, None)
        if stored is None or str(stored).strip() == "":
            continue
        if key == "ALPHA_CONDITION_JSON_OVERRIDE":
            if not _alpha_json_matches_metadata(metadata, current):
                mismatches.append(f"{key}: file={stored!r}, env={current!r}")
            continue
        if _env_value_for_compare(key, current) != _env_value_for_compare(key, stored):
            mismatches.append(f"{key}: file={stored!r}, env={current!r}")
    mismatches.extend(_active_config_mismatches(metadata))
    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(f"Offline TV-init metadata conflicts with active environment overrides: {detail}")


def _active_config_mismatches(metadata: Mapping[str, Any]) -> list[str]:
    stored_config = metadata.get("resolved_config", None)
    if not isinstance(stored_config, Mapping):
        return []
    config_module = sys.modules.get("config") or sys.modules.get("models.config")
    if config_module is None:
        return []
    try:
        current_config = _config_snapshot_from_module(config_module)
    except Exception as exc:  # pragma: no cover - validation should report, not hide, config issues.
        return [f"resolved_config: unable to read active config ({exc})"]
    return _compare_metadata_config(stored_config, current_config)


def _flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value} if prefix else {}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        next_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, next_prefix))
        else:
            flattened[next_prefix] = item
    return flattened


def _values_match_for_config(path: str, stored: Any, current: Any) -> bool:
    if stored is None or current is None:
        return stored is current
    if path.endswith("alpha_condition_constrained_json"):
        return _env_value_for_compare("ALPHA_CONDITION_JSON_OVERRIDE", stored) == _env_value_for_compare(
            "ALPHA_CONDITION_JSON_OVERRIDE", current
        )
    if isinstance(stored, (int, float)) or isinstance(current, (int, float)):
        try:
            return abs(float(stored) - float(current)) <= 1.0e-12
        except (TypeError, ValueError):
            return False
    return str(stored).strip() == str(current).strip()


def _compare_metadata_config(stored_config: Mapping[str, Any], current_config: Mapping[str, Any]) -> list[str]:
    current_flat = _flatten_mapping(current_config)
    mismatches: list[str] = []
    for path, stored_value in _flatten_mapping(stored_config).items():
        if path not in current_flat:
            continue
        current_value = current_flat[path]
        if not _values_match_for_config(path, stored_value, current_value):
            mismatches.append(f"resolved_config.{path}: file={stored_value!r}, active={current_value!r}")
    return mismatches


def load_offline_tensors(
    path: str | os.PathLike[str],
    *,
    validate_metadata: bool = True,
) -> dict[str, torch.Tensor]:
    """Load the minimal tensor payload needed by the neural network.

    Returned keys are exactly ``coeff_true``, ``g_observed`` and
    ``coeff_initial``. Extra fields in older experiment files are ignored.
    """
    raw = _load_torch_file(path)
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise KeyError(f"Offline dataset is missing keys: {missing!r}.")
    if bool(validate_metadata):
        metadata = raw.get(METADATA_KEY, {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TypeError(f"{METADATA_KEY} must be a mapping, got {type(metadata).__name__}.")
        validate_offline_metadata_against_env(metadata)

    tensors = {
        "coeff_true": _as_coeff_batch(raw["coeff_true"], name="coeff_true"),
        "g_observed": _as_observation_batch(raw["g_observed"]),
        "coeff_initial": _as_coeff_batch(raw["coeff_initial"], name="coeff_initial"),
    }
    _validate_same_batch_size(tensors)
    return tensors


def save_offline_tensors(
    path: str | os.PathLike[str],
    *,
    coeff_true: torch.Tensor,
    g_observed: torch.Tensor,
    coeff_initial: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save tensors consumed by the network plus optional provenance metadata."""
    tensors = {
        "coeff_true": _as_coeff_batch(coeff_true, name="coeff_true"),
        "g_observed": _as_observation_batch(g_observed),
        "coeff_initial": _as_coeff_batch(coeff_initial, name="coeff_initial"),
    }
    _validate_same_batch_size(tensors)

    payload: dict[str, Any] = dict(tensors)
    normalized_metadata = _normalize_metadata(metadata)
    if normalized_metadata is not None:
        normalized_metadata.setdefault(
            "tensor_shapes",
            {key: list(value.shape) for key, value in tensors.items()},
        )
        payload[METADATA_KEY] = normalized_metadata

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


class OfflineCTDataset(Dataset):
    """Torch Dataset returning ``(coeff_true, g_observed, coeff_initial)``."""

    def __init__(self, path: str | os.PathLike[str], *, validate_metadata: bool = True):
        self.path = Path(path)
        self.metadata = load_offline_metadata(self.path)
        if bool(validate_metadata):
            validate_offline_metadata_against_env(self.metadata)
        self.tensors = load_offline_tensors(self.path, validate_metadata=False)

    def __len__(self) -> int:
        return int(self.tensors["coeff_true"].shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.tensors["coeff_true"][index],
            self.tensors["g_observed"][index],
            self.tensors["coeff_initial"][index],
        )


class OfflineBatchProvider:
    """Small adapter with the same batch API shape as ``TheoreticalDataGenerator``.

    ``generate_batch`` returns ``(coeff_true, f_true, g_observed, coeff_initial)``;
    the second item is kept only for call-site compatibility and is not saved in
    the dataset file.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        shuffle: bool = True,
        target_device: torch.device | str | None = None,
        validate_metadata: bool = True,
    ):
        self.dataset = OfflineCTDataset(path, validate_metadata=validate_metadata)
        self.metadata = self.dataset.metadata
        self.shuffle = bool(shuffle)
        self.target_device = torch.device(target_device) if target_device is not None else None
        self._order = torch.arange(len(self.dataset), dtype=torch.long)
        self._cursor = 0
        if self.shuffle:
            self._reshuffle()

    def _reshuffle(self) -> None:
        self._order = self._order[torch.randperm(len(self._order))]
        self._cursor = 0

    def _next_indices(self, batch_size: int) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        remaining = int(batch_size)
        while remaining > 0:
            if self._cursor >= len(self.dataset):
                if self.shuffle:
                    self._reshuffle()
                else:
                    self._cursor = 0
            take = min(remaining, len(self.dataset) - self._cursor)
            pieces.append(self._order[self._cursor : self._cursor + take])
            self._cursor += take
            remaining -= take
        return torch.cat(pieces, dim=0)

    def __len__(self) -> int:
        return len(self.dataset)

    def generate_batch_by_indices(
        self,
        indices,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.as_tensor(indices, dtype=torch.long).view(-1)
        if int(indices.numel()) <= 0:
            raise ValueError("indices must contain at least one item.")
        invalid_mask = (indices < 0) | (indices >= len(self.dataset))
        if bool(torch.any(invalid_mask)):
            invalid = indices[invalid_mask].detach().cpu().tolist()
            raise IndexError(f"indices contain out-of-range values {invalid!r} for dataset size={len(self.dataset)}.")
        tensors = self.dataset.tensors
        coeff_true = tensors["coeff_true"].index_select(0, indices)
        g_observed = tensors["g_observed"].index_select(0, indices)
        coeff_initial = tensors["coeff_initial"].index_select(0, indices)
        if self.target_device is not None:
            coeff_true = coeff_true.to(self.target_device)
            g_observed = g_observed.to(self.target_device)
            coeff_initial = coeff_initial.to(self.target_device)
        return coeff_true, coeff_true, g_observed, coeff_initial

    def generate_random_batch(
        self,
        batch_size: int,
        random_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}.")
        if batch_size > len(self.dataset):
            raise ValueError(f"batch_size={batch_size} exceeds dataset size={len(self.dataset)}.")
        generator = torch.Generator(device="cpu")
        if random_seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(random_seed))
        indices = torch.randperm(len(self.dataset), generator=generator)[:batch_size]
        return self.generate_batch_by_indices(indices)

    def generate_batch(
        self,
        batch_size: int,
        random_seed: int | None = None,
        lambda_reg: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del lambda_reg
        if random_seed is not None:
            torch.manual_seed(int(random_seed))
            if self.shuffle:
                self._reshuffle()
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}.")
        if batch_size > len(self.dataset):
            raise ValueError(f"batch_size={batch_size} exceeds dataset size={len(self.dataset)}.")
        indices = self._next_indices(batch_size)
        return self.generate_batch_by_indices(indices)


def _first_existing_path(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def default_alpha_json_path() -> Path:
    return _first_existing_path(
        [
            PROJECT_ROOT / "data" / "alpha8_tv" / "alpha_selected8_dopt_soft_g25.json",
            PROJECT_ROOT / "汇报" / "正则化改动" / "angle" / "alpha_selected8_dopt_soft_g25.json",
            PROJECT_ROOT
            / "results"
            / "shepp_logan_condition_vs_dopt_tv_noise01_8"
            / "alpha_selected8_dopt_soft_g25.json",
            PROJECT_ROOT / "data" / "alpha_search_cache" / "alpha_selected8.json",
        ]
    )


def default_alpha16_json_path() -> Path:
    return _first_existing_path(
        [
            PROJECT_ROOT / "data" / "alpha16_tv" / "alpha_selected16_dopt_hard_gap_9_14.json",
        ]
    )


def _resolve_project_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _default_alpha_json_for_num_angles(num_angles: int) -> Path:
    if int(num_angles) == 16:
        return default_alpha16_json_path()
    if int(num_angles) == 8:
        return default_alpha_json_path()
    return PROJECT_ROOT / "data" / "alpha_search_cache" / f"alpha_selected{int(num_angles)}.json"


def apply_tvinit_env_defaults(
    *,
    num_angles: int = 8,
    alpha_json_path: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Set TV-init defaults while allowing the selected angle JSON to vary."""
    num_angles = int(num_angles)
    if num_angles <= 0:
        raise ValueError(f"num_angles must be positive, got {num_angles!r}.")
    resolved_alpha_json = _resolve_project_path(
        _default_alpha_json_for_num_angles(num_angles) if alpha_json_path is None else alpha_json_path
    )
    defaults = {
        "EXPERIMENT_PROFILE_OVERRIDE": "alpha_condition",
        "ALPHA_CONDITION_TOP_K_OVERRIDE": str(num_angles),
        "ALPHA_CONDITION_JSON_OVERRIDE": str(resolved_alpha_json),
        "ALPHA_GRAM_CACHE_DIR_OVERRIDE": str(PROJECT_ROOT / "data" / "alpha_gram_cache"),
        "MULTI_ANGLE_SOLVER_MODE_OVERRIDE": "stacked_tikhonov",
        "THEORETICAL_FORMULA_MODE_OVERRIDE": "alpha_continuous",
        "INIT_METHOD_OVERRIDE": "l2_tv_admm",
        "LAMBDA_SELECT_MODE_OVERRIDE": "morozov",
        "MOROZOV_FORM_OVERRIDE": "constrained",
        "MOROZOV_NOISE_RADIUS_MODE_OVERRIDE": "rms",
        "MOROZOV_TAU_OVERRIDE": "1.0",
        "L1_INIT_ADMM_ITERS_OVERRIDE": "40",
        "L1_INIT_ADMM_CG_ITERS_OVERRIDE": "15",
        "L1_INIT_ADMM_CG_TOL_OVERRIDE": "1e-4",
        "L1_INIT_ADMM_RHO_DATA_OVERRIDE": "1.0",
        "L1_INIT_ADMM_RHO_REG_OVERRIDE": "1.0",
        "REGULARIZER_TYPE_OVERRIDE": "dirichlet",
        "NOISE_MODE_OVERRIDE": "multiplicative",
        "NOISE_LEVEL_OVERRIDE": "0.1",
    }
    explicit_angle_selection = alpha_json_path is not None or num_angles != 8
    for key, value in defaults.items():
        if key in ANGLE_ENV_KEYS and (explicit_angle_selection or num_angles != 8):
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in defaults}


def apply_alpha8_tvinit_env_defaults() -> dict[str, str]:
    """Set run-script-compatible defaults before importing ``models/config.py``."""
    return apply_tvinit_env_defaults(num_angles=8)


def apply_alpha16_tvinit_env_defaults() -> dict[str, str]:
    """Set alpha16 TV-init defaults with the same TV parameters as alpha8."""
    return apply_tvinit_env_defaults(num_angles=16, alpha_json_path=default_alpha16_json_path())


def _default_generator_factory(
    *,
    num_angles: int = 8,
    alpha_json_path: str | os.PathLike[str] | None = None,
):
    apply_tvinit_env_defaults(num_angles=num_angles, alpha_json_path=alpha_json_path)
    if str(MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(MODELS_DIR))

    from radon_transform import TheoreticalDataGenerator  # noqa: WPS433

    return TheoreticalDataGenerator


def _read_alpha_json_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = _resolve_project_path(path)
    summary: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if not resolved.exists():
        return summary
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        summary["type"] = type(raw).__name__
        return summary
    selected = raw.get("selected", [])
    if not isinstance(selected, list):
        selected = []
    meta = raw.get("meta", {})
    if not isinstance(meta, Mapping):
        meta = {}
    summary.update(
        {
            "selection_objective": meta.get("selection_objective"),
            "top_k": meta.get("top_k", len(selected)),
            "selected_count": len(selected),
            "min_gap_deg": meta.get("min_gap_deg"),
            "max_gap_deg": meta.get("max_gap_deg"),
            "final_min_gap_deg": meta.get("final_min_gap_deg"),
            "final_max_gap_deg": meta.get("final_max_gap_deg"),
            "sampling_formula": meta.get("sampling_formula"),
            "alpha_values": [float(item["alpha"]) for item in selected if isinstance(item, Mapping) and "alpha" in item],
            "tau_offsets": [float(item["tau_star"]) for item in selected if isinstance(item, Mapping) and "tau_star" in item],
        }
    )
    return summary


def _config_snapshot_from_module(config_module) -> dict[str, Any]:
    data_config = getattr(config_module, "DATA_CONFIG")
    image_size = getattr(config_module, "IMAGE_SIZE")
    theoretical_config = getattr(config_module, "THEORETICAL_CONFIG")
    time_domain_config = getattr(config_module, "TIME_DOMAIN_CONFIG")
    time_keys = (
        "experiment_profile",
        "num_angles_total",
        "num_angles",
        "sampling_mode",
        "num_detector_samples",
        "detector_phase",
        "detector_margin_ratio",
        "init_method",
        "multi_angle_solver_mode",
        "theoretical_formula_mode",
    )
    data_keys = (
        "lambda_select_mode",
        "morozov_form",
        "morozov_noise_radius_mode",
        "morozov_tau",
        "l1_init_admm_iters",
        "l1_init_admm_cg_iters",
        "l1_init_admm_cg_tol",
        "l1_init_admm_rho_data",
        "l1_init_admm_rho_reg",
        "noise_mode",
        "noise_level",
    )
    theoretical_keys = ("regularizer_type",)
    return {
        "image_size": int(image_size),
        "time_domain": {key: _json_safe(time_domain_config.get(key)) for key in time_keys},
        "data": {key: _json_safe(data_config.get(key)) for key in data_keys},
        "theoretical": {key: _json_safe(theoretical_config.get(key)) for key in theoretical_keys},
    }


def _snapshot_active_config() -> dict[str, Any]:
    try:
        import config as config_module  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - metadata should not block tensor saving.
        return {"error": str(exc)}
    return _config_snapshot_from_module(config_module)


def _current_tvinit_env() -> dict[str, str]:
    return {key: str(os.environ.get(key, "")) for key in TVINIT_ENV_KEYS}


def build_offline_tvinit_metadata(
    *,
    output_path: str | os.PathLike[str],
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    env = _current_tvinit_env()
    alpha_json = env.get("ALPHA_CONDITION_JSON_OVERRIDE", "")
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generator_script": str(Path(__file__).resolve()),
        "output_path": str(Path(output_path)),
        "generation": _json_safe(dict(generation)),
        "env": env,
        "angle_selection": {
            "num_angles": int(env.get("ALPHA_CONDITION_TOP_K_OVERRIDE") or generation.get("num_angles") or 0),
            "alpha_json_path": alpha_json,
            "alpha_json_summary": _read_alpha_json_summary(alpha_json) if alpha_json else {},
        },
        "tv_regularization": {
            "init_method": env.get("INIT_METHOD_OVERRIDE", ""),
            "lambda_select_mode": env.get("LAMBDA_SELECT_MODE_OVERRIDE", ""),
            "morozov_form": env.get("MOROZOV_FORM_OVERRIDE", ""),
            "morozov_noise_radius_mode": env.get("MOROZOV_NOISE_RADIUS_MODE_OVERRIDE", ""),
            "morozov_tau": env.get("MOROZOV_TAU_OVERRIDE", ""),
            "l1_init_admm_iters": env.get("L1_INIT_ADMM_ITERS_OVERRIDE", ""),
            "l1_init_admm_cg_iters": env.get("L1_INIT_ADMM_CG_ITERS_OVERRIDE", ""),
            "l1_init_admm_cg_tol": env.get("L1_INIT_ADMM_CG_TOL_OVERRIDE", ""),
            "l1_init_admm_rho_data": env.get("L1_INIT_ADMM_RHO_DATA_OVERRIDE", ""),
            "l1_init_admm_rho_reg": env.get("L1_INIT_ADMM_RHO_REG_OVERRIDE", ""),
            "regularizer_type": env.get("REGULARIZER_TYPE_OVERRIDE", ""),
            "noise_mode": env.get("NOISE_MODE_OVERRIDE", ""),
            "noise_level": env.get("NOISE_LEVEL_OVERRIDE", ""),
        },
        "resolved_config": _snapshot_active_config(),
    }


def _append_generated_segment(
    *,
    generator,
    num_samples: int,
    batch_size: int,
    seed_offset: int,
    label: str,
    coeff_true_parts: list[torch.Tensor],
    g_observed_parts: list[torch.Tensor],
    coeff_initial_parts: list[torch.Tensor],
) -> int:
    produced = 0
    while produced < num_samples:
        current = min(batch_size, num_samples - produced)
        seed = int(seed_offset) + produced
        coeff_true, _f_true, g_observed, coeff_initial = generator.generate_batch(
            batch_size=current,
            random_seed=seed,
        )
        coeff_true_parts.append(coeff_true.detach().cpu())
        g_observed_parts.append(g_observed.detach().cpu())
        coeff_initial_parts.append(coeff_initial.detach().cpu())
        produced += current
        _log(f"[offline-tvinit] {label}: generated {produced}/{num_samples}")
    return int(num_samples)


def generate_offline_dataset(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_PATH,
    *,
    num_samples: int,
    data_source: str = "random_ellipses",
    seed_offset: int = 0,
    batch_size: int = 1,
    num_angles: int = 8,
    alpha_json_path: str | os.PathLike[str] | None = None,
    dataset_role: str | None = None,
    generator_factory=None,
) -> Path:
    """Generate TV-initialized offline data compatible with ``deep_learn/model.py``."""
    num_samples = int(num_samples)
    batch_size = int(batch_size)
    num_angles = int(num_angles)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples!r}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}.")
    if num_angles <= 0:
        raise ValueError(f"num_angles must be positive, got {num_angles!r}.")

    apply_tvinit_env_defaults(num_angles=num_angles, alpha_json_path=alpha_json_path)
    if generator_factory is None:
        generator_factory = _default_generator_factory(num_angles=num_angles, alpha_json_path=alpha_json_path)
    generator_cls = generator_factory
    generator = generator_cls(data_source=data_source)
    coeff_true_parts: list[torch.Tensor] = []
    g_observed_parts: list[torch.Tensor] = []
    coeff_initial_parts: list[torch.Tensor] = []

    _append_generated_segment(
        generator=generator,
        num_samples=num_samples,
        batch_size=batch_size,
        seed_offset=seed_offset,
        label=data_source,
        coeff_true_parts=coeff_true_parts,
        g_observed_parts=g_observed_parts,
        coeff_initial_parts=coeff_initial_parts,
    )

    generation = {
        "mode": "single_source",
        "num_samples": num_samples,
        "data_source": data_source,
        "seed_offset": seed_offset,
        "batch_size": batch_size,
        "num_angles": num_angles,
    }
    if dataset_role is not None:
        generation["dataset_role"] = str(dataset_role)

    return save_offline_tensors(
        output_path,
        coeff_true=torch.cat(coeff_true_parts, dim=0),
        g_observed=torch.cat(g_observed_parts, dim=0),
        coeff_initial=torch.cat(coeff_initial_parts, dim=0),
        metadata=build_offline_tvinit_metadata(
            output_path=output_path,
            generation=generation,
        ),
    )


def generate_mixed_offline_dataset(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_PATH,
    *,
    random_ellipses_samples: int = 3000,
    shepp_logan_samples: int = 500,
    seed_offset: int = 0,
    batch_size: int = 1,
    num_angles: int = 8,
    alpha_json_path: str | os.PathLike[str] | None = None,
    generator_factory=None,
) -> Path:
    """Generate one ordered file: random ellipses first, Shepp-Logan second."""
    random_ellipses_samples = int(random_ellipses_samples)
    shepp_logan_samples = int(shepp_logan_samples)
    batch_size = int(batch_size)
    num_angles = int(num_angles)
    if random_ellipses_samples < 0 or shepp_logan_samples < 0:
        raise ValueError("sample counts must be non-negative.")
    if random_ellipses_samples + shepp_logan_samples <= 0:
        raise ValueError("At least one sample must be requested.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}.")
    if num_angles <= 0:
        raise ValueError(f"num_angles must be positive, got {num_angles!r}.")

    apply_tvinit_env_defaults(num_angles=num_angles, alpha_json_path=alpha_json_path)

    if generator_factory is None:
        generator_factory = _default_generator_factory(num_angles=num_angles, alpha_json_path=alpha_json_path)

    coeff_true_parts: list[torch.Tensor] = []
    g_observed_parts: list[torch.Tensor] = []
    coeff_initial_parts: list[torch.Tensor] = []
    produced_total = 0

    if random_ellipses_samples > 0:
        produced_total += _append_generated_segment(
            generator=generator_factory(data_source="random_ellipses"),
            num_samples=random_ellipses_samples,
            batch_size=batch_size,
            seed_offset=int(seed_offset) + produced_total,
            label="random_ellipses",
            coeff_true_parts=coeff_true_parts,
            g_observed_parts=g_observed_parts,
            coeff_initial_parts=coeff_initial_parts,
        )
    if shepp_logan_samples > 0:
        produced_total += _append_generated_segment(
            generator=generator_factory(data_source="shepp_logan"),
            num_samples=shepp_logan_samples,
            batch_size=batch_size,
            seed_offset=int(seed_offset) + produced_total,
            label="shepp_logan",
            coeff_true_parts=coeff_true_parts,
            g_observed_parts=g_observed_parts,
            coeff_initial_parts=coeff_initial_parts,
        )

    return save_offline_tensors(
        output_path,
        coeff_true=torch.cat(coeff_true_parts, dim=0),
        g_observed=torch.cat(g_observed_parts, dim=0),
        coeff_initial=torch.cat(coeff_initial_parts, dim=0),
        metadata=build_offline_tvinit_metadata(
            output_path=output_path,
            generation={
                "mode": "mixed_ordered",
                "random_ellipses_samples": random_ellipses_samples,
                "shepp_logan_samples": shepp_logan_samples,
                "seed_offset": seed_offset,
                "batch_size": batch_size,
                "num_angles": num_angles,
            },
        ),
    )


def _resolve_split_output_path(
    *,
    output_dir: str | os.PathLike[str],
    explicit_path: str | os.PathLike[str] | None,
    default_name: str,
) -> Path:
    if explicit_path:
        return _resolve_project_path(explicit_path)
    return _resolve_project_path(Path(output_dir) / default_name)


def generate_train_val_test_offline_datasets(
    *,
    output_dir: str | os.PathLike[str] = DEFAULT_SPLIT_OUTPUT_DIR,
    train_output: str | os.PathLike[str] | None = None,
    val_output: str | os.PathLike[str] | None = None,
    test_output: str | os.PathLike[str] | None = None,
    train_samples: int = 8000,
    val_random_ellipses_samples: int = 500,
    test_shepp_logan_samples: int = 500,
    seed_offset: int = 0,
    train_batch_size: int = 480,
    val_batch_size: int = 32,
    test_batch_size: int = 32,
    num_angles: int = 8,
    alpha_json_path: str | os.PathLike[str] | None = None,
    generator_factory=None,
) -> dict[str, Path]:
    """Generate train/validation/test split files in one command.

    The validation split intentionally uses random ellipses so it matches the
    training distribution while still sampling independent random shapes.
    """
    train_samples = int(train_samples)
    val_random_ellipses_samples = int(val_random_ellipses_samples)
    test_shepp_logan_samples = int(test_shepp_logan_samples)
    seed_offset = int(seed_offset)
    train_batch_size = int(train_batch_size)
    val_batch_size = int(val_batch_size)
    test_batch_size = int(test_batch_size)
    num_angles = int(num_angles)
    if train_samples <= 0:
        raise ValueError(f"train_samples must be positive, got {train_samples!r}.")
    if val_random_ellipses_samples <= 0:
        raise ValueError(f"val_random_ellipses_samples must be positive, got {val_random_ellipses_samples!r}.")
    if test_shepp_logan_samples <= 0:
        raise ValueError(f"test_shepp_logan_samples must be positive, got {test_shepp_logan_samples!r}.")
    for name, value in (
        ("train_batch_size", train_batch_size),
        ("val_batch_size", val_batch_size),
        ("test_batch_size", test_batch_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}.")
    if num_angles <= 0:
        raise ValueError(f"num_angles must be positive, got {num_angles!r}.")

    apply_tvinit_env_defaults(num_angles=num_angles, alpha_json_path=alpha_json_path)
    if generator_factory is None:
        generator_factory = _default_generator_factory(num_angles=num_angles, alpha_json_path=alpha_json_path)

    tag = f"alpha{num_angles}_noise01"
    paths = {
        "train": _resolve_split_output_path(
            output_dir=output_dir,
            explicit_path=train_output,
            default_name=f"train{train_samples}_random_ellipses_tvinit_{tag}.pt",
        ),
        "val": _resolve_split_output_path(
            output_dir=output_dir,
            explicit_path=val_output,
            default_name=f"val{val_random_ellipses_samples}_random_ellipses_tvinit_{tag}.pt",
        ),
        "test": _resolve_split_output_path(
            output_dir=output_dir,
            explicit_path=test_output,
            default_name=f"test{test_shepp_logan_samples}_shepp_logan_tvinit_{tag}.pt",
        ),
    }

    generated: dict[str, Path] = {}
    generated["train"] = generate_offline_dataset(
        paths["train"],
        num_samples=train_samples,
        data_source="random_ellipses",
        seed_offset=seed_offset,
        batch_size=train_batch_size,
        num_angles=num_angles,
        alpha_json_path=alpha_json_path,
        dataset_role="train",
        generator_factory=generator_factory,
    )
    generated["val"] = generate_offline_dataset(
        paths["val"],
        num_samples=val_random_ellipses_samples,
        data_source="random_ellipses",
        seed_offset=seed_offset + train_samples,
        batch_size=val_batch_size,
        num_angles=num_angles,
        alpha_json_path=alpha_json_path,
        dataset_role="val",
        generator_factory=generator_factory,
    )
    generated["test"] = generate_offline_dataset(
        paths["test"],
        num_samples=test_shepp_logan_samples,
        data_source="shepp_logan",
        seed_offset=seed_offset + train_samples + val_random_ellipses_samples,
        batch_size=test_batch_size,
        num_angles=num_angles,
        alpha_json_path=alpha_json_path,
        dataset_role="test",
        generator_factory=generator_factory,
    )
    return generated


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TV-initialized offline CT data.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output .pt path.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples for single-source generation.")
    parser.add_argument("--random-ellipses-samples", type=int, default=None, help="Number of random_ellipses samples in ordered mixed generation.")
    parser.add_argument("--shepp-logan-samples", type=int, default=None, help="Number of shepp_logan samples in ordered mixed generation.")
    parser.add_argument("--data-source", default="random_ellipses", choices=["random_ellipses", "random_ellipse", "ellipse", "shepp_logan"])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size; use small values if GPU memory is tight.")
    parser.add_argument("--num-angles", type=int, default=None, help="Selected alpha-angle count. If omitted with --alpha-json, infer it from that JSON.")
    parser.add_argument("--alpha-json", default=None, help="Selected alpha JSON path. Defaults to the alpha8/alpha16 TV path for known counts.")
    parser.add_argument("--train-val-test-splits", action="store_true", help="Generate train random-ellipses, validation random-ellipses, and test Shepp-Logan .pt files.")
    parser.add_argument("--split-output-dir", default=str(DEFAULT_SPLIT_OUTPUT_DIR), help="Output directory for --train-val-test-splits when split paths are not set.")
    parser.add_argument("--train-output", default=None, help="Explicit train .pt path for --train-val-test-splits.")
    parser.add_argument("--val-output", default=None, help="Explicit validation .pt path for --train-val-test-splits.")
    parser.add_argument("--test-output", default=None, help="Explicit test .pt path for --train-val-test-splits.")
    parser.add_argument("--train-samples", type=int, default=8000, help="Train random_ellipses sample count for --train-val-test-splits.")
    parser.add_argument("--val-random-ellipses-samples", type=int, default=500, help="Validation random_ellipses sample count for --train-val-test-splits.")
    parser.add_argument("--test-shepp-logan-samples", type=int, default=500, help="Test shepp_logan sample count for --train-val-test-splits.")
    parser.add_argument("--train-batch-size", type=int, default=None, help="Train generation batch size for --train-val-test-splits; default 480.")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Validation generation batch size for --train-val-test-splits; default 32.")
    parser.add_argument("--test-batch-size", type=int, default=None, help="Test generation batch size for --train-val-test-splits; default 32.")
    return parser.parse_args(argv)


def _resolve_cli_num_angles(args: argparse.Namespace) -> int:
    if args.num_angles is not None:
        return int(args.num_angles)
    if args.alpha_json:
        summary = _read_alpha_json_summary(args.alpha_json)
        for key in ("selected_count", "top_k"):
            value = summary.get(key)
            if value is None:
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    return 8


def _resolve_generation_counts(args: argparse.Namespace) -> tuple[int, int]:
    if args.num_samples is not None:
        raise ValueError("--num-samples uses single-source generation and has no mixed train/val counts.")
    if args.random_ellipses_samples is None and args.shepp_logan_samples is None:
        return 3000, 500
    return (
        0 if args.random_ellipses_samples is None else int(args.random_ellipses_samples),
        0 if args.shepp_logan_samples is None else int(args.shepp_logan_samples),
    )


def _log_saved_dataset(output: str | os.PathLike[str]) -> None:
    tensors = load_offline_tensors(output)
    _log(
        "[offline-tvinit] saved "
        f"{output} | coeff_true={tuple(tensors['coeff_true'].shape)} "
        f"g_observed={tuple(tensors['g_observed'].shape)} "
        f"coeff_initial={tuple(tensors['coeff_initial'].shape)}"
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    num_angles = _resolve_cli_num_angles(args)
    mixed_requested = args.random_ellipses_samples is not None or args.shepp_logan_samples is not None
    if args.train_val_test_splits:
        outputs = generate_train_val_test_offline_datasets(
            output_dir=args.split_output_dir,
            train_output=args.train_output,
            val_output=args.val_output,
            test_output=args.test_output,
            train_samples=args.train_samples,
            val_random_ellipses_samples=args.val_random_ellipses_samples,
            test_shepp_logan_samples=args.test_shepp_logan_samples,
            seed_offset=args.seed_offset,
            train_batch_size=480 if args.train_batch_size is None else args.train_batch_size,
            val_batch_size=32 if args.val_batch_size is None else args.val_batch_size,
            test_batch_size=32 if args.test_batch_size is None else args.test_batch_size,
            num_angles=num_angles,
            alpha_json_path=args.alpha_json,
        )
        for role in ("train", "val", "test"):
            _log_saved_dataset(outputs[role])
        return
    if mixed_requested:
        random_count, shepp_count = _resolve_generation_counts(args)
        output = generate_mixed_offline_dataset(
            args.output,
            random_ellipses_samples=random_count,
            shepp_logan_samples=shepp_count,
            seed_offset=args.seed_offset,
            batch_size=args.batch_size,
            num_angles=num_angles,
            alpha_json_path=args.alpha_json,
        )
    elif args.num_samples is not None:
        output = generate_offline_dataset(
            args.output,
            num_samples=args.num_samples,
            data_source=args.data_source,
            seed_offset=args.seed_offset,
            batch_size=args.batch_size,
            num_angles=num_angles,
            alpha_json_path=args.alpha_json,
        )
    else:
        output = generate_mixed_offline_dataset(
            args.output,
            random_ellipses_samples=3000,
            shepp_logan_samples=500,
            seed_offset=args.seed_offset,
            batch_size=args.batch_size,
            num_angles=num_angles,
            alpha_json_path=args.alpha_json,
        )
    _log_saved_dataset(output)


if __name__ == "__main__":
    main()
