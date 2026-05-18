"""Cascade data/model helpers for the alpha16 -> extra8 second-stage training."""

from __future__ import annotations

import os
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(__file__).resolve().parents[1]
DEEP_LEARN_DIR = MODELS_DIR / "deep_learn"
for path in (THIS_DIR, DEEP_LEARN_DIR, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config as config_module
from config import DATA_CONFIG, IMAGE_SIZE, TIME_DOMAIN_CONFIG, device
from model import initialize_model, load_trainable_state_dict
from radon_transform import TheoreticalDataGenerator


def normalize_runtime_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    value = str(path).strip()
    if not value:
        return None
    if os.name == "nt":
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", value)
        if match is not None:
            drive = match.group(1).upper()
            tail = match.group(2).replace("/", "\\")
            return f"{drive}:\\{tail}"
    return value


def parse_int_list(raw: str | Iterable[int] | None) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return [int(token.strip()) for token in text.replace(";", ",").split(",") if token.strip()]
    return [int(item) for item in raw]


def configure_alpha_condition_runtime(
    *,
    alpha_json: str | Path,
    cnn_angle_indices: str | Iterable[int] | None = None,
    cnn_num_angles: int | None = None,
    physics_residual_channel_enabled: bool = True,
    physics_residual_mode: str = "per_angle_cg",
    physics_residual_damping: float = 1.0e-2,
    physics_residual_cg_iters: int = 8,
    physics_residual_detach: bool = True,
    physics_residual_normalize: bool = True,
    physics_explicit_update_enabled: bool = True,
    physics_explicit_update_alpha_init: float = 0.05,
    physics_explicit_update_max: float = 0.25,
    init_method: str = "tikhonov_direct",
    multi_angle_solver_mode: str = "stacked_tikhonov",
) -> dict:
    """Mutate the shared config dictionaries before constructing a model."""

    records, resolved_json = config_module._load_alpha_condition_records(path=str(alpha_json))
    alpha_values = [float(item["alpha"]) for item in records]
    tau_offsets = [float(item["tau_star"]) for item in records]
    indices = parse_int_list(cnn_angle_indices)
    if cnn_num_angles is None:
        cnn_num_angles = len(indices) if indices is not None else len(alpha_values)

    TIME_DOMAIN_CONFIG.update(
        {
            "experiment_profile": "alpha_condition",
            "operator_mode": "theoretical_b1b1",
            "use_multi_angle": True,
            "alpha_values": alpha_values,
            "alpha_tau_offsets": tau_offsets,
            "alpha_condition_constrained_records": records,
            "alpha_condition_constrained_json": str(resolved_json),
            "num_angles_total": int(len(alpha_values)),
            "num_angles": int(len(alpha_values)),
            "theoretical_formula_mode": "alpha_continuous",
            "data_formula_mode": "auto_complete",
            "multi_angle_solver_mode": str(multi_angle_solver_mode),
            "cnn_backbone_only": False,
            "cnn_num_angles_override": int(cnn_num_angles),
            "cnn_angle_indices_override": indices,
            "physics_residual_channel_enabled": bool(physics_residual_channel_enabled),
            "physics_residual_mode": str(physics_residual_mode),
            "physics_residual_damping": float(physics_residual_damping),
            "physics_residual_cg_iters": int(physics_residual_cg_iters),
            "physics_residual_detach": bool(physics_residual_detach),
            "physics_residual_normalize": bool(physics_residual_normalize),
            "physics_explicit_update_enabled": bool(physics_explicit_update_enabled),
            "physics_explicit_update_alpha_init": float(physics_explicit_update_alpha_init),
            "physics_explicit_update_max": float(physics_explicit_update_max),
            "init_method": str(init_method),
        }
    )
    return {
        "alpha_json": str(resolved_json),
        "alpha_values": alpha_values,
        "alpha_tau_offsets": tau_offsets,
        "cnn_angle_indices": indices if indices is not None else list(range(int(cnn_num_angles))),
        "cnn_num_angles": int(cnn_num_angles),
    }


def _torch_load_checkpoint(path: str | Path):
    load_path = normalize_runtime_path(path)
    if load_path is None or not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(load_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(load_path, map_location=device)


def build_model_for_alpha_json(
    *,
    alpha_json: str | Path,
    cnn_angle_indices: str | Iterable[int] | None,
    cnn_num_angles: int | None,
    checkpoint_path: str | Path | None = None,
    frozen: bool = False,
):
    configure_alpha_condition_runtime(
        alpha_json=alpha_json,
        cnn_angle_indices=cnn_angle_indices,
        cnn_num_angles=cnn_num_angles,
    )
    model = initialize_model()
    checkpoint_metadata = {}
    if checkpoint_path:
        checkpoint = _torch_load_checkpoint(checkpoint_path)
        checkpoint_metadata = checkpoint.get("experiment_metadata", {}) if isinstance(checkpoint, dict) else {}
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        load_trainable_state_dict(model, state)
    if frozen:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, checkpoint_metadata


@contextmanager
def preserve_rng_state():
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class CascadeBatchGenerator:
    """Generate one true coefficient map and two observations on alpha16/extra8."""

    def __init__(
        self,
        *,
        stage1_model,
        stage2_model,
        data_source: str | None = None,
        img_size: int = IMAGE_SIZE,
    ):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.img_size = int(img_size)
        source = str(data_source or DATA_CONFIG.get("data_source", "random_ellipses")).strip().lower()
        self.stage1_generator = TheoreticalDataGenerator(
            img_size=self.img_size,
            data_source=source,
            time_operator=self.stage1_model.optimizer.operator,
        )
        self.stage2_generator = TheoreticalDataGenerator(
            img_size=self.img_size,
            data_source=source,
            time_operator=self.stage2_model.optimizer.operator,
        )

    def _observed_pair(self, generator: TheoreticalDataGenerator, coeff_true: torch.Tensor):
        with torch.no_grad():
            g_clean = generator.data_forward_operator(coeff_true).to(torch.float32)
            g_observed = generator._apply_noise(g_clean)
        return g_clean, g_observed

    def _stage1_initial(self, g_observed: torch.Tensor, g_clean: torch.Tensor):
        lambda_eff = self.stage1_generator._select_lambda(g_observed, g_clean, lambda_reg=None)
        self.stage1_generator.last_lambda = lambda_eff
        return self.stage1_generator.solve_tikhonov_direct_init(g_observed, lambda_reg=lambda_eff)

    def generate_batch(self, batch_size: int, random_seed: int | None = None):
        if random_seed is not None:
            torch.manual_seed(int(random_seed))
            np.random.seed(int(random_seed))
            random.seed(int(random_seed))
        coeff_true = self.stage1_generator._sample_coefficients(int(batch_size))
        f_true = self.stage1_generator.image_gen(coeff_true)
        g16_clean, g16_observed = self._observed_pair(self.stage1_generator, coeff_true)
        _, g8_observed = self._observed_pair(self.stage2_generator, coeff_true)
        coeff_initial16 = self._stage1_initial(g16_observed, g16_clean)
        self.stage1_model.eval()
        with torch.no_grad():
            coeff_stage1, _, _ = self.stage1_model(
                coeff_initial16.to(device),
                g16_observed.to(device),
            )
        return {
            "coeff_true": coeff_true.to(device),
            "f_true": f_true.to(device),
            "g16_observed": g16_observed.to(device),
            "g8_observed": g8_observed.to(device),
            "coeff_initial16": coeff_initial16.to(device),
            "coeff_stage1": coeff_stage1.detach().to(device),
        }
