import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import time
import os
import sys
import logging
import random
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from model import (
    initialize_model,
    count_parameters,
    export_trainable_state_dict,
    load_trainable_state_dict,
)
from radon_transform import TheoreticalDataGenerator
from data_genoration import OfflineBatchProvider
from config import (
    n_data, n_train,
    device, MODEL_PATH, BEST_MODEL_PATH, CHECKPOINT_DIR,
    TRAINING_CONFIG, DATA_CONFIG, LOGGING_CONFIG, TIME_DOMAIN_CONFIG, THEORETICAL_CONFIG, EXPERIMENT_OUTPUT_TAG
)

# Optional quick overrides for debugging (do not affect config.py).
N_TRAIN = int(os.environ.get("N_TRAIN_OVERRIDE", n_train))
N_DATA = int(os.environ.get("N_DATA_OVERRIDE", n_data))


def _env_flag(name, default=False):
    value = os.environ.get(name, None)
    if value is None:
        return bool(default)
    value = str(value).strip().lower()
    if value == "":
        return bool(default)
    return value in {"1", "true", "yes", "y", "on"}


def _resolve_resume_checkpoint_path():
    resume_path = str(os.environ.get("RESUME_CHECKPOINT_OVERRIDE", "") or "").strip()
    if resume_path:
        return resume_path
    auto_resume = str(os.environ.get("AUTO_RESUME_OVERRIDE", "") or "").strip().lower()
    if auto_resume not in {"1", "true", "yes", "y", "on"}:
        return None
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.is_dir():
        return None
    candidates = []
    for checkpoint_path in checkpoint_dir.glob("checkpoint_iter_*.pth"):
        raw_iter = checkpoint_path.stem.replace("checkpoint_iter_", "", 1)
        try:
            iter_idx = int(raw_iter)
        except ValueError:
            continue
        candidates.append((iter_idx, checkpoint_path))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


def _build_train_or_val_generator(*, data_source, shared_time_operator, offline_env_name, shuffle_offline):
    offline_path = str(os.environ.get(offline_env_name, "") or "").strip()
    if offline_path:
        return OfflineBatchProvider(
            offline_path,
            shuffle=bool(shuffle_offline),
            target_device=None,
        )
    return TheoreticalDataGenerator(
        data_source=data_source,
        time_operator=shared_time_operator,
    )


def _next_training_start_iter(current_iter):
    return max(int(current_iter) + 1, 0)


def _relative_l2_loss(pred, target, eps=None):
    if eps is None:
        eps = float(TRAINING_CONFIG.get("loss_eps", 1.0e-12))
    diff_sq_sum = torch.sum(torch.abs(pred - target) ** 2)
    true_sq_sum = torch.sum(torch.abs(target) ** 2).clamp_min(float(eps))
    return torch.sqrt(diff_sq_sum / true_sq_sum)


def _forward_gradient_pair(x):
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    return grad_y, grad_x


def _gradient_relative_l2_loss(pred, target, eps=None):
    if eps is None:
        eps = float(TRAINING_CONFIG.get("loss_eps", 1.0e-12))
    err_y, err_x = _forward_gradient_pair(pred - target)
    true_y, true_x = _forward_gradient_pair(target)
    diff_sq_sum = torch.sum(err_y.pow(2)) + torch.sum(err_x.pow(2))
    true_sq_sum = (torch.sum(true_y.pow(2)) + torch.sum(true_x.pow(2))).clamp_min(float(eps))
    return torch.sqrt(diff_sq_sum / true_sq_sum)


def _laplacian(x):
    base_kernel = x.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).view(1, 1, 3, 3)
    channels = int(x.shape[1])
    kernel = base_kernel.repeat(channels, 1, 1, 1)
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel, groups=channels)


def _laplacian_relative_l2_loss(pred, target, eps=None):
    if eps is None:
        eps = float(TRAINING_CONFIG.get("loss_eps", 1.0e-12))
    err_lap = _laplacian(pred - target)
    true_lap = _laplacian(target)
    diff_sq_sum = torch.sum(err_lap.pow(2))
    true_sq_sum = torch.sum(true_lap.pow(2)).clamp_min(float(eps))
    return torch.sqrt(diff_sq_sum / true_sq_sum)


def _aux_loss_decay_factor(iter_idx):
    start_fraction = float(TRAINING_CONFIG.get("aux_loss_decay_start_fraction", 1.0))
    end_fraction = float(TRAINING_CONFIG.get("aux_loss_decay_end_fraction", 1.0))
    start_fraction = min(max(start_fraction, 0.0), 1.0)
    end_fraction = min(max(end_fraction, 0.0), 1.0)
    if end_fraction <= start_fraction:
        return 1.0
    start_step = int(round(float(N_TRAIN) * start_fraction))
    end_step = int(round(float(N_TRAIN) * end_fraction))
    iter_idx = int(iter_idx)
    if iter_idx <= start_step:
        return 1.0
    if iter_idx >= end_step:
        return 0.0
    progress = float(iter_idx - start_step) / float(max(end_step - start_step, 1))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _reconstruction_objective(pred, target, *, aux_factor=1.0):
    res = _relative_l2_loss(pred, target)
    res_weight = float(TRAINING_CONFIG.get("res_loss_weight", 1.0))
    grad_weight = float(TRAINING_CONFIG.get("gradres_loss_weight", 0.0)) * float(aux_factor)
    lap_weight = float(TRAINING_CONFIG.get("lapres_loss_weight", 0.0)) * float(aux_factor)
    loss = pred.new_zeros(())
    if res_weight != 0.0:
        loss = loss + res_weight * res
    if grad_weight != 0.0:
        loss = loss + grad_weight * _gradient_relative_l2_loss(pred, target)
    if lap_weight != 0.0:
        loss = loss + lap_weight * _laplacian_relative_l2_loss(pred, target)
    return loss, res


def _objective_description():
    return (
        "RES+aux "
        f"res={float(TRAINING_CONFIG.get('res_loss_weight', 1.0)):.3g} "
        f"gradres={float(TRAINING_CONFIG.get('gradres_loss_weight', 0.0)):.3g} "
        f"lapres={float(TRAINING_CONFIG.get('lapres_loss_weight', 0.0)):.3g} "
        f"aux_decay=({float(TRAINING_CONFIG.get('aux_loss_decay_start_fraction', 1.0)):.3g},"
        f"{float(TRAINING_CONFIG.get('aux_loss_decay_end_fraction', 1.0)):.3g})"
    )


def _select_by_indices(values, indices):
    values_list = list(values or [])
    selected = []
    for idx in indices:
        idx = int(idx)
        if 0 <= idx < len(values_list):
            selected.append(float(values_list[idx]))
    return selected


def _build_cnn_angle_selection_summary(
    cnn_angle_indices,
    alpha_values,
    tau_offsets,
    physics_residual_enabled,
    physics_residual_mode,
):
    indices = [int(idx) for idx in list(cnn_angle_indices or [])]
    physics_mode = str(physics_residual_mode or "").strip().lower()
    physics_indices = (
        indices
        if bool(physics_residual_enabled) and physics_mode in {"per_angle_cg", "stacked_selected_cg"}
        else []
    )
    return {
        "count": int(len(indices)),
        "indices": indices,
        "alpha_values": _select_by_indices(alpha_values, indices),
        "tau_offsets": _select_by_indices(tau_offsets, indices),
        "data_fidelity_gradient_channel_indices": indices,
        "physics_residual_channel_indices": physics_indices,
    }


def _set_global_seed_from_env():
    seed_raw = os.environ.get("GLOBAL_SEED_OVERRIDE", None)
    if seed_raw is None:
        return None
    seed_str = str(seed_raw).strip()
    if seed_str == "":
        return None
    seed = int(seed_str)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

class TheoreticalTrainer:
    def __init__(self):
        self._setup_logging()
        self.model = initialize_model()
        self.experiment_metadata = self._build_experiment_metadata()
        train_data_source = str(
            DATA_CONFIG.get("train_data_source", DATA_CONFIG.get("data_source", "random_ellipses"))
        ).strip().lower()
        val_data_source = str(
            DATA_CONFIG.get("val_data_source", DATA_CONFIG.get("data_source", train_data_source))
        ).strip().lower()
        shared_time_operator = getattr(self.model.optimizer, "operator", None)
        self.data_generator = _build_train_or_val_generator(
            data_source=train_data_source,
            shared_time_operator=shared_time_operator,
            offline_env_name="OFFLINE_TRAIN_DATASET_OVERRIDE",
            shuffle_offline=True,
        )
        self.val_data_generator = _build_train_or_val_generator(
            data_source=val_data_source,
            shared_time_operator=shared_time_operator,
            offline_env_name="OFFLINE_VAL_DATASET_OVERRIDE",
            shuffle_offline=False,
        )
        secondary_val_dataset = os.environ.get("OFFLINE_SECONDARY_VAL_DATASET_OVERRIDE", "").strip()
        self.secondary_val_data_generator = None
        self.secondary_val_data_source = ""
        self.secondary_val_label = os.environ.get("SECONDARY_VAL_LABEL_OVERRIDE", "secondary").strip() or "secondary"
        if secondary_val_dataset:
            self.secondary_val_data_source = os.environ.get(
                "SECONDARY_VAL_DATA_SOURCE_OVERRIDE", "shepp_logan"
            ).strip().lower()
            self.secondary_val_data_generator = _build_train_or_val_generator(
                data_source=self.secondary_val_data_source,
                shared_time_operator=shared_time_operator,
                offline_env_name="OFFLINE_SECONDARY_VAL_DATASET_OVERRIDE",
                shuffle_offline=False,
            )
        self.logger.info("Train data source: %s", train_data_source)
        self.logger.info("Validation data source: %s", val_data_source)
        if self.secondary_val_data_generator is not None:
            self.logger.info(
                "Secondary validation data source: %s label=%s",
                self.secondary_val_data_source,
                self.secondary_val_label,
            )
        if os.environ.get("OFFLINE_TRAIN_DATASET_OVERRIDE", "").strip():
            self.logger.info("Offline train dataset: %s", os.environ["OFFLINE_TRAIN_DATASET_OVERRIDE"])
        if os.environ.get("OFFLINE_VAL_DATASET_OVERRIDE", "").strip():
            self.logger.info("Offline validation dataset: %s", os.environ["OFFLINE_VAL_DATASET_OVERRIDE"])
        if secondary_val_dataset:
            self.logger.info("Secondary offline validation dataset: %s", secondary_val_dataset)
        self.logger.info("Experiment tag: %s", self.experiment_metadata["output_tag"])
        self.logger.info("Operator mode: %s", self.experiment_metadata["operator_mode"])
        self.logger.info("Operator class: %s", self.experiment_metadata["operator_class"])
        self.logger.info("Theoretical formula mode: %s", self.experiment_metadata["theoretical_formula_mode"])
        self.logger.info("Data formula mode: %s", self.experiment_metadata["data_formula_mode"])
        self.logger.info(
            "Resolved data formula mode: %s | data/recon operator reused=%s",
            getattr(self.data_generator, "data_formula_mode", self.experiment_metadata["data_formula_mode"]),
            bool(getattr(self.data_generator, "data_time_operator", None) is getattr(self.data_generator, "time_operator", None)),
        )
        self.logger.info(
            "Generator/model operator shared: train=%s val=%s",
            bool(getattr(self.data_generator, "time_operator", None) is shared_time_operator),
            bool(getattr(self.val_data_generator, "time_operator", None) is shared_time_operator),
        )
        self.logger.info(
            "Angle usage: total=%d, learned=%d, cnn_channels=%d",
            self.experiment_metadata["num_angles"],
            self.experiment_metadata["learned_num_angles"],
            self.experiment_metadata["cnn_num_angles"],
        )
        self.logger.info(
            "CNN input angle selection: count=%d indices=%s",
            self.experiment_metadata["cnn_angle_selection"]["count"],
            self.experiment_metadata["cnn_angle_selection"]["indices"],
        )
        self.logger.info(
            "CNN input alpha values: %s",
            self.experiment_metadata["cnn_angle_selection"]["alpha_values"],
        )
        self.logger.info(
            "CNN input tau offsets: %s",
            self.experiment_metadata["cnn_angle_selection"]["tau_offsets"],
        )
        self.logger.info(
            "Data fidelity gradient channel angle indices: %s",
            self.experiment_metadata["cnn_angle_selection"]["data_fidelity_gradient_channel_indices"],
        )
        self.logger.info(
            "Data fidelity channel mode: %s channels=%d",
            self.experiment_metadata["data_fidelity_channel_mode"],
            self.experiment_metadata["data_fidelity_channels"],
        )
        if self.experiment_metadata["cnn_angle_selection"]["physics_residual_channel_indices"]:
            self.logger.info(
                "Physics residual channel angle indices: %s",
                self.experiment_metadata["cnn_angle_selection"]["physics_residual_channel_indices"],
            )
        self.logger.info(
            "Physics residual: enabled=%s mode=%s channels=%d explicit_update=%s",
            self.experiment_metadata["physics_residual_channel_enabled"],
            self.experiment_metadata["physics_residual_mode"],
            self.experiment_metadata["physics_residual_channels"],
            self.experiment_metadata["physics_explicit_update_enabled"],
        )
        self.logger.info(
            "Model architecture: %s refiner_input=%s unet_backbone=%s unet_base=%s unet_depth=%s stages=%s gate=%s stage_dc=%s",
            self.experiment_metadata.get("model_arch"),
            self.experiment_metadata.get("refiner_input_mode"),
            self.experiment_metadata.get("unet_backbone"),
            self.experiment_metadata.get("unet_base_channels"),
            self.experiment_metadata.get("unet_depth"),
            self.experiment_metadata.get("refiner_stages"),
            self.experiment_metadata.get("physics_gate_mode"),
            self.experiment_metadata.get("refiner_stage_dc_enabled"),
        )
        self.logger.info(
            "Detail head: enabled=%s input=%s hidden=%s depth=%s residual_max=%s stage_policy=%s share_weights=%s",
            self.experiment_metadata.get("detail_head_enabled"),
            self.experiment_metadata.get("detail_head_input_mode"),
            self.experiment_metadata.get("detail_head_hidden_channels"),
            self.experiment_metadata.get("detail_head_depth"),
            self.experiment_metadata.get("detail_head_residual_max"),
            self.experiment_metadata.get("detail_head_stage_policy"),
            self.experiment_metadata.get("detail_head_share_weights"),
        )
        self.logger.info("Training objective: %s", _objective_description())
        self.logger.info(
            "Active alpha JSON: %s",
            self.experiment_metadata.get("alpha_condition_constrained_json"),
        )
        self.logger.info(
            "Init alpha JSON: %s",
            self.experiment_metadata.get("init_alpha_condition_constrained_json"),
        )
        self.logger.info(
            "Generator init operator shared with data operator: train=%s val=%s",
            bool(getattr(self.data_generator, "init_time_operator", None) is getattr(self.data_generator, "data_time_operator", None)),
            bool(getattr(self.val_data_generator, "init_time_operator", None) is getattr(self.val_data_generator, "data_time_operator", None)),
        )
        self.logger.info(
            "Active alpha angles (%d): %s",
            len(self.experiment_metadata.get("alpha_values") or []),
            self.experiment_metadata.get("alpha_values") or [],
        )
        self.logger.info(
            "Active alpha tau offsets (%d): %s",
            len(self.experiment_metadata.get("alpha_tau_offsets") or []),
            self.experiment_metadata.get("alpha_tau_offsets") or [],
        )
        noise_mode = str(DATA_CONFIG.get("noise_mode", "additive")).strip().lower()
        if noise_mode == "snr":
            self.logger.info("Noise setting: SNR=%sdB", DATA_CONFIG.get("target_snr_db", 30.0))
        else:
            self.logger.info(
                "Noise setting: %s delta=%s",
                noise_mode,
                DATA_CONFIG.get("noise_level", 0.1),
            )
        self.logger.info(
            "Validation: interval=%d samples=%d micro_batch=%d random_subsample=%s reproducible=%s",
            int(TRAINING_CONFIG.get("validation_interval", 10)),
            int(DATA_CONFIG.get("val_subsample_size", DATA_CONFIG.get("val_batch_size", N_DATA))),
            int(DATA_CONFIG.get("val_batch_size", N_DATA)),
            bool(DATA_CONFIG.get("val_random_subsample", False)),
            bool(DATA_CONFIG.get("val_reproducible", False)),
        )

        # Separate parameter groups: zero weight_decay and lower LR for per-iteration scalars
        scalar_params = []
        other_params = []
        scalar_names = (
            "step_size_raw",
            "reg_lambda_raw",
            "physics_alpha_raw",
            "stage_dc_alpha_raw",
        )
        for name, param in self.model.named_parameters():
            if any(key in name for key in scalar_names):
                scalar_params.append(param)
            else:
                other_params.append(param)

        base_lr = float(os.environ.get(
            "BASE_LR_OVERRIDE",
            TRAINING_CONFIG["optimizer_learning_rate"],
        ))
        scalar_lr = base_lr * float(TRAINING_CONFIG.get("scalar_lr_ratio", 0.1))
        param_groups = []
        if other_params:
            param_groups.append({'params': other_params, 'weight_decay': 1e-4})
        if scalar_params:
            param_groups.append({'params': scalar_params, 'weight_decay': 0.0, 'lr': scalar_lr})
        if not param_groups:
            raise RuntimeError("Model has no trainable parameters.")
        self.optimizer = optim.AdamW(param_groups, lr=base_lr)

        schedule_mode = str(TRAINING_CONFIG.get("lr_schedule", "inverse")).strip().lower()
        inverse_decay_steps = max(float(TRAINING_CONFIG.get("lr_inverse_decay_steps", 500.0)), 1.0)
        constant_steps = max(int(TRAINING_CONFIG.get("lr_constant_steps", 0)), 0)
        min_factor = min(max(float(TRAINING_CONFIG.get("lr_min_factor", 0.1)), 0.0), 1.0)
        warmup_steps = max(int(TRAINING_CONFIG.get("lr_warmup_steps", 0)), 0)

        def lr_lambda(step):
            step = int(step)
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step + 1) / float(warmup_steps), 1.0e-8)
            adjusted_step = max(step - warmup_steps, 0)
            if schedule_mode == "constant":
                return 1.0
            if schedule_mode == "inverse":
                return 1.0 / (1.0 + adjusted_step / inverse_decay_steps)
            if schedule_mode == "constant_cosine":
                if adjusted_step <= constant_steps:
                    return 1.0
                cosine_step = adjusted_step - constant_steps
                cosine_total = max(int(N_TRAIN) - warmup_steps - constant_steps, 1)
            elif schedule_mode == "cosine":
                cosine_step = adjusted_step
                cosine_total = max(int(N_TRAIN) - warmup_steps, 1)
            else:
                raise ValueError(f"Unsupported lr_schedule={schedule_mode!r}.")
            progress = min(max(float(cosine_step) / float(cosine_total), 0.0), 1.0)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.current_iter = -1
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_res': [],
            'val_res': [],
            'learning_rate': [],
            'data_fidelity_error': [],
            'update_difference': []
        }
        resume_checkpoint_path = _resolve_resume_checkpoint_path()
        if resume_checkpoint_path is not None:
            self._load_resume_checkpoint(resume_checkpoint_path)
        self.logger.info("Theoretical trainer initialized successfully")

    def _load_resume_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Resume checkpoint must be a dict, got {type(checkpoint).__name__}.")

        model_state = checkpoint.get('model_state_dict', None)
        if model_state is None:
            raise KeyError(f"Resume checkpoint {checkpoint_path} does not contain 'model_state_dict'.")
        load_info = load_trainable_state_dict(self.model, model_state)

        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except (RuntimeError, ValueError) as e:
                self.logger.warning(
                    "Could not load optimizer_state_dict from resume checkpoint; optimizer state starts fresh. Error: %s",
                    e,
                )
        else:
            self.logger.warning("Resume checkpoint has no optimizer_state_dict; optimizer state starts fresh.")
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except (RuntimeError, ValueError) as e:
                self.logger.warning(
                    "Could not load scheduler_state_dict from resume checkpoint; scheduler state starts fresh. Error: %s",
                    e,
                )
        else:
            self.logger.warning("Resume checkpoint has no scheduler_state_dict; scheduler state starts fresh.")

        self.current_iter = int(checkpoint.get('iter', -1))
        self.best_val_loss = float(checkpoint.get('best_val_loss', self.best_val_loss))
        saved_history = checkpoint.get('training_history', None)
        if isinstance(saved_history, dict):
            for key in self.training_history:
                self.training_history[key] = list(saved_history.get(key, self.training_history[key]))
        ignored_count = len(load_info.get('ignored_non_parameter_keys', [])) if isinstance(load_info, dict) else 0
        initialized_count = len(load_info.get('initialized_missing_parameter_keys', [])) if isinstance(load_info, dict) else 0
        self.logger.info(
            "Resumed training from %s at iter=%d best_val_loss=%.6f loaded_params=%s ignored_non_params=%d initialized_missing_params=%d",
            str(checkpoint_path),
            int(self.current_iter),
            float(self.best_val_loss),
            load_info.get('loaded_parameter_count', 'unknown') if isinstance(load_info, dict) else 'unknown',
            int(ignored_count),
            int(initialized_count),
        )

    def _build_experiment_metadata(self):
        operator = self.model.optimizer.operator
        cnn_angle_indices = [int(idx) for idx in getattr(self.model.optimizer, "cnn_channel_indices", [])]
        alpha_values = list(TIME_DOMAIN_CONFIG.get("alpha_values") or getattr(operator, "alpha_values", []) or [])
        tau_offsets = list(TIME_DOMAIN_CONFIG.get("alpha_tau_offsets") or [])
        if not tau_offsets and hasattr(operator, "tau_offsets_tensor"):
            tau_offsets = [float(v) for v in operator.tau_offsets_tensor.detach().cpu().view(-1).tolist()]
        physics_residual_enabled = bool(getattr(self.model.optimizer, "physics_residual_enabled", False))
        physics_residual_mode = str(
            getattr(
                self.model.optimizer,
                "physics_residual_mode",
                TIME_DOMAIN_CONFIG.get("physics_residual_mode", "per_angle_cg"),
            )
        )
        init_records = list(TIME_DOMAIN_CONFIG.get("init_alpha_condition_constrained_records") or [])
        angle_selection = _build_cnn_angle_selection_summary(
            cnn_angle_indices=cnn_angle_indices,
            alpha_values=alpha_values,
            tau_offsets=tau_offsets,
            physics_residual_enabled=physics_residual_enabled,
            physics_residual_mode=physics_residual_mode,
        )
        return {
            "output_tag": EXPERIMENT_OUTPUT_TAG or "default",
            "model_arch": str(THEORETICAL_CONFIG.get("model_arch", "unrolled_cnn")),
            "refiner_input_mode": str(THEORETICAL_CONFIG.get("refiner_input_mode", "u2_stacked")),
            "unet_backbone": str(THEORETICAL_CONFIG.get("unet_backbone", "plain")),
            "unet_base_channels": int(THEORETICAL_CONFIG.get("unet_base_channels", 32)),
            "unet_depth": int(THEORETICAL_CONFIG.get("unet_depth", 4)),
            "unet_residual_max": float(THEORETICAL_CONFIG.get("unet_residual_max", 0.0)),
            "physics_gate_mode": str(THEORETICAL_CONFIG.get("physics_gate_mode", "scalar")),
            "refiner_stages": int(THEORETICAL_CONFIG.get("refiner_stages", 1)),
            "refiner_share_weights": bool(THEORETICAL_CONFIG.get("refiner_share_weights", True)),
            "refiner_stage_dc_enabled": bool(THEORETICAL_CONFIG.get("refiner_stage_dc_enabled", False)),
            "refiner_stage_dc_cg_iters": int(THEORETICAL_CONFIG.get("refiner_stage_dc_cg_iters", 4)),
            "refiner_stage_dc_damping": float(THEORETICAL_CONFIG.get("refiner_stage_dc_damping", 1.0e-2)),
            "refiner_stage_dc_detach": bool(THEORETICAL_CONFIG.get("refiner_stage_dc_detach", True)),
            "refiner_stage_dc_normalize": bool(THEORETICAL_CONFIG.get("refiner_stage_dc_normalize", True)),
            "detail_head_enabled": bool(THEORETICAL_CONFIG.get("detail_head_enabled", False)),
            "detail_head_input_mode": str(THEORETICAL_CONFIG.get("detail_head_input_mode", "features")),
            "detail_head_hidden_channels": int(THEORETICAL_CONFIG.get("detail_head_hidden_channels", 16)),
            "detail_head_depth": int(THEORETICAL_CONFIG.get("detail_head_depth", 2)),
            "detail_head_residual_max": float(THEORETICAL_CONFIG.get("detail_head_residual_max", 0.0)),
            "detail_head_stage_policy": str(THEORETICAL_CONFIG.get("detail_head_stage_policy", "last")),
            "detail_head_share_weights": bool(THEORETICAL_CONFIG.get("detail_head_share_weights", True)),
            "detail_head_zero_init": bool(THEORETICAL_CONFIG.get("detail_head_zero_init", True)),
            "lr_schedule": str(TRAINING_CONFIG.get("lr_schedule", "inverse")),
            "lr_inverse_decay_steps": float(TRAINING_CONFIG.get("lr_inverse_decay_steps", 500.0)),
            "lr_constant_steps": int(TRAINING_CONFIG.get("lr_constant_steps", 0)),
            "lr_min_factor": float(TRAINING_CONFIG.get("lr_min_factor", 0.1)),
            "lr_warmup_steps": int(TRAINING_CONFIG.get("lr_warmup_steps", 0)),
            "scalar_lr_ratio": float(TRAINING_CONFIG.get("scalar_lr_ratio", 0.1)),
            "res_loss_weight": float(TRAINING_CONFIG.get("res_loss_weight", 1.0)),
            "gradres_loss_weight": float(TRAINING_CONFIG.get("gradres_loss_weight", 0.0)),
            "lapres_loss_weight": float(TRAINING_CONFIG.get("lapres_loss_weight", 0.0)),
            "loss_eps": float(TRAINING_CONFIG.get("loss_eps", 1.0e-12)),
            "aux_loss_decay_start_fraction": float(TRAINING_CONFIG.get("aux_loss_decay_start_fraction", 1.0)),
            "aux_loss_decay_end_fraction": float(TRAINING_CONFIG.get("aux_loss_decay_end_fraction", 1.0)),
            "regularizer_type": str(getattr(self.model.optimizer.theoretical_gd, "regularizer_type", "")),
            "init_method": str(TIME_DOMAIN_CONFIG.get("init_method", "")),
            "lambda_select_mode": str(DATA_CONFIG.get("lambda_select_mode", "")),
            "l1_init_admm_iters": int(DATA_CONFIG.get("l1_init_admm_iters", 80)),
            "l1_init_admm_cg_iters": int(DATA_CONFIG.get("l1_init_admm_cg_iters", 30)),
            "l1_init_admm_cg_tol": float(DATA_CONFIG.get("l1_init_admm_cg_tol", 1.0e-4)),
            "l1_init_admm_rho_data": float(DATA_CONFIG.get("l1_init_admm_rho_data", 1.0)),
            "l1_init_admm_rho_reg": float(DATA_CONFIG.get("l1_init_admm_rho_reg", 1.0)),
            "admm_stop_mode": str(DATA_CONFIG.get("admm_stop_mode", "fixed")),
            "tv_pdhg_iters": int(DATA_CONFIG.get("tv_pdhg_iters", 10)),
            "tv_pdhg_theta": float(DATA_CONFIG.get("tv_pdhg_theta", 1.0)),
            "tv_pdhg_nonnegative": bool(DATA_CONFIG.get("tv_pdhg_nonnegative", False)),
            "tv_pdhg_power_iters": int(DATA_CONFIG.get("tv_pdhg_power_iters", 8)),
            "experiment_profile": str(TIME_DOMAIN_CONFIG.get("experiment_profile", "default")),
            "operator_mode": str(TIME_DOMAIN_CONFIG.get("operator_mode", "")),
            "operator_class": operator.__class__.__name__,
            "num_angles": int(getattr(operator, "num_angles", 1) or 1),
            "num_angles_total": int(TIME_DOMAIN_CONFIG.get("num_angles_total", getattr(operator, "num_angles", 1) or 1)),
            "cnn_backbone_only": bool(TIME_DOMAIN_CONFIG.get("cnn_backbone_only", True)),
            "learned_num_angles": int(getattr(self.model.optimizer, "learned_num_angles", 1) or 1),
            "raw_cnn_angle_channels": int(getattr(self.model.optimizer, "raw_cnn_num_angles", getattr(self.model.optimizer, "cnn_num_angles", 1)) or 1),
            "cnn_num_angles": int(getattr(self.model.optimizer, "cnn_num_angles", 1) or 1),
            "cnn_angle_indices": cnn_angle_indices,
            "cnn_angle_alpha_values": list(angle_selection["alpha_values"]),
            "cnn_angle_tau_offsets": list(angle_selection["tau_offsets"]),
            "cnn_angle_selection": angle_selection,
            "data_fidelity_channel_mode": str(getattr(self.model.optimizer, "data_fidelity_channel_mode", DATA_CONFIG.get("data_fidelity_channel_mode", "per_angle"))),
            "data_fidelity_channels": int(getattr(self.model.optimizer, "data_fidelity_channels", getattr(self.model.optimizer, "cnn_num_angles", 1)) or 1),
            "physics_residual_channel_enabled": physics_residual_enabled,
            "physics_residual_mode": physics_residual_mode,
            "physics_residual_channels": int(getattr(self.model.optimizer, "physics_residual_channels", 0) or 0),
            "physics_residual_cg_iters": int(getattr(self.model.optimizer, "physics_residual_cg_iters", TIME_DOMAIN_CONFIG.get("physics_residual_cg_iters", 8)) or 0),
            "physics_residual_damping": float(getattr(self.model.optimizer, "physics_residual_damping", TIME_DOMAIN_CONFIG.get("physics_residual_damping", 1.0e-2))),
            "physics_residual_detach": bool(getattr(self.model.optimizer, "physics_residual_detach", TIME_DOMAIN_CONFIG.get("physics_residual_detach", True))),
            "physics_residual_normalize": bool(getattr(self.model.optimizer, "physics_residual_normalize", TIME_DOMAIN_CONFIG.get("physics_residual_normalize", True))),
            "physics_explicit_update_enabled": bool(getattr(self.model.optimizer, "physics_explicit_update_enabled", False)),
            "physics_explicit_update_max": float(getattr(self.model.optimizer, "physics_explicit_update_max", TIME_DOMAIN_CONFIG.get("physics_explicit_update_max", 0.10))),
            "alpha_values": list(TIME_DOMAIN_CONFIG.get("alpha_values") or []),
            "alpha_tau_offsets": list(TIME_DOMAIN_CONFIG.get("alpha_tau_offsets") or []),
            "alpha_condition_constrained_json": TIME_DOMAIN_CONFIG.get("alpha_condition_constrained_json", None),
            "init_alpha_values": [float(item["alpha"]) for item in init_records],
            "init_alpha_tau_offsets": [float(item["tau_star"] if "tau_star" in item else item["tau"]) for item in init_records],
            "init_alpha_condition_constrained_json": TIME_DOMAIN_CONFIG.get("init_alpha_condition_constrained_json", None),
            "theoretical_formula_mode": str(TIME_DOMAIN_CONFIG.get("theoretical_formula_mode", "auto")),
            "data_formula_mode": str(TIME_DOMAIN_CONFIG.get("data_formula_mode", "auto_complete")),
        }

    def _setup_logging(self):
        log_dir = LOGGING_CONFIG['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger('TheoreticalCTTrainer')
        self.logger.setLevel(getattr(logging, LOGGING_CONFIG['log_level']))
        self.logger.handlers.clear()
        if LOGGING_CONFIG['log_to_console']:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        if LOGGING_CONFIG['log_to_file']:
            file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def _generate_training_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = N_DATA
        # Option 3: fully random batches (do not reseed by iteration).
        return self.data_generator.generate_batch(batch_size, random_seed=None)

    def _validate(
        self,
        *,
        generator=None,
        val_bs=None,
        val_sample_count=None,
        random_subsample=None,
        reproducible=None,
        seed_override=None,
    ):
        val_generator = generator if generator is not None else self.val_data_generator
        val_bs = max(int(DATA_CONFIG.get('val_batch_size', N_DATA) if val_bs is None else val_bs), 1)
        val_sample_count = max(
            int(DATA_CONFIG.get('val_subsample_size', val_bs) if val_sample_count is None else val_sample_count),
            1,
        )
        # Optional override to make validation fixed/reproducible for debugging.
        # This keeps the validation set (including noise) fixed so RES across iterations is comparable.
        if reproducible is None:
            val_repro = bool(DATA_CONFIG.get("val_reproducible", False))
            env_val_repro = os.environ.get("VAL_REPRODUCIBLE_OVERRIDE", None)
            if env_val_repro is not None:
                val_repro = env_val_repro.strip().lower() in ("1", "true", "yes", "y")
        else:
            val_repro = bool(reproducible)

        seed = None
        if val_repro:
            if seed_override is None:
                seed = int(os.environ.get("VAL_SEED_OVERRIDE", DATA_CONFIG.get("validation_seed", 42)))
            else:
                seed = int(seed_override)
        val_random_subsample = bool(DATA_CONFIG.get("val_random_subsample", False) if random_subsample is None else random_subsample)
        if val_random_subsample and hasattr(val_generator, "generate_random_batch"):
            subset_seed = None
            if seed is not None:
                subset_seed = seed + max(int(getattr(self, "current_iter", 0)), 0)
            coeff_true_val, _, g_observed_val, coeff_initial_val = val_generator.generate_random_batch(
                batch_size=val_sample_count,
                random_seed=subset_seed,
            )
        elif seed is not None:
            # NOTE: generate_batch(random_seed=...) calls torch.manual_seed/np.random.seed.
            # If we don't restore RNG states, this will reset the global RNG and make subsequent
            # training batches partially deterministic (hurts generalization and confuses curves).
            py_state = random.getstate()
            np_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                coeff_true_val, _, g_observed_val, coeff_initial_val = val_generator.generate_batch(
                    batch_size=val_sample_count, random_seed=seed
                )
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.random.set_rng_state(torch_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
        else:
            coeff_true_val, _, g_observed_val, coeff_initial_val = val_generator.generate_batch(
                batch_size=val_sample_count, random_seed=None
            )
        self.model.eval()
        total_diff_sq = None
        total_true_sq = None
        metric_sums = {}
        metric_weight = 0
        with torch.no_grad():
            total = int(coeff_true_val.shape[0])
            for start in range(0, total, val_bs):
                end = min(start + val_bs, total)
                coeff_true_chunk = coeff_true_val[start:end].to(device)
                g_observed_chunk = g_observed_val[start:end].to(device)
                coeff_initial_chunk = coeff_initial_val[start:end].to(device)
                coeff_pred, _, metrics = self.model(coeff_initial_chunk, g_observed_chunk)
                diff_sq_sum = torch.sum(torch.abs(coeff_pred - coeff_true_chunk) ** 2)
                true_sq_sum = torch.sum(torch.abs(coeff_true_chunk) ** 2)
                total_diff_sq = diff_sq_sum if total_diff_sq is None else total_diff_sq + diff_sq_sum
                total_true_sq = true_sq_sum if total_true_sq is None else total_true_sq + true_sq_sum
                chunk_weight = end - start
                metric_weight += chunk_weight
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * float(chunk_weight)
            if total_diff_sq is None or total_true_sq is None:
                raise RuntimeError("Validation batch is empty.")
            val_loss = torch.sqrt(total_diff_sq / total_true_sq.clamp_min(1e-12))
            metrics = {key: value / max(metric_weight, 1) for key, value in metric_sums.items()}
        self.model.train()
        return val_loss.item(), metrics

    def train(self):
        self.logger.info("Starting theoretical CT reconstruction training...")
        self.logger.info(f"Total iterations: {N_TRAIN}")
        self.logger.info(f"Batch size: {N_DATA}")
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")
        self.logger.info("Objective mode: %s", _objective_description())
        total_start_time = time.time()
        start_iter = _next_training_start_iter(self.current_iter)
        self.logger.info(f"Starting iteration: {start_iter}")
        if start_iter >= N_TRAIN:
            self.logger.info(
                "No training iterations to run: checkpoint iter=%d, target N_TRAIN=%d",
                int(self.current_iter),
                int(N_TRAIN),
            )
        for self.current_iter in range(start_iter, N_TRAIN):
            iter_start_time = time.time()
            coeff_true, _, g_observed, coeff_initial = self._generate_training_batch()
            coeff_true = coeff_true.to(device)
            g_observed = g_observed.to(device)
            coeff_initial = coeff_initial.to(device)
            self.optimizer.zero_grad()
            coeff_pred, history, metrics = self.model(coeff_initial, g_observed)

            aux_factor = _aux_loss_decay_factor(self.current_iter)
            loss, train_res_tensor = _reconstruction_objective(coeff_pred, coeff_true, aux_factor=aux_factor)
            if bool(DATA_CONFIG.get("intermediate_supervision_enabled", False)):
                # history[0] is the initialization. Supervise only actual unrolled updates.
                hist = history[1:]
                if len(hist) == 0:
                    pass
                else:
                    step_losses = torch.stack([
                        _reconstruction_objective(x, coeff_true, aux_factor=aux_factor)[0]
                        for x in hist
                    ])
                    w_start = float(DATA_CONFIG.get("intermediate_supervision_weight_start", 0.2))
                    w_end = float(DATA_CONFIG.get("intermediate_supervision_weight_end", 1.0))
                    weights = torch.linspace(
                        w_start,
                        w_end,
                        steps=len(step_losses),
                        device=step_losses.device,
                        dtype=step_losses.dtype,
                    )
                    weights = weights / weights.sum().clamp_min(1.0e-12)
                    loss = torch.sum(weights * step_losses)
            loss.backward()
            if TRAINING_CONFIG.get('gradient_clip_value', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    TRAINING_CONFIG['gradient_clip_value']
                )
            self.optimizer.step()
            self.scheduler.step()

            # Diagnostic logging for learnable optimization scalars
            if self.current_iter % 500 == 0 and self.current_iter > 0:
                lgd = self.model.optimizer
                if hasattr(lgd, "current_step_size") and hasattr(lgd, "current_reg_lambda"):
                    self.logger.info(
                        "  Learned scalars: step=%.6f lambda=%.6f",
                        float(lgd.current_step_size().item()),
                        float(lgd.current_reg_lambda().item()),
                    )
                elif hasattr(lgd, "current_physics_alpha"):
                    diagnostics = None
                    if hasattr(lgd, "physics_gate_diagnostics"):
                        diagnostics = lgd.physics_gate_diagnostics(coeff_initial, g_observed)
                    if diagnostics and diagnostics.get("gate_mode") == "spatial" and "gate_mean" in diagnostics:
                        self.logger.info(
                            "  Refiner spatial gate: mean=%.6f std=%.6f min=%.6f max=%.6f legacy_alpha=%.6f physics_update_norm=%.3e physics_corr_norm=%.3e",
                            float(diagnostics["gate_mean"]),
                            float(diagnostics["gate_std"]),
                            float(diagnostics["gate_min"]),
                            float(diagnostics["gate_max"]),
                            float(diagnostics["legacy_alpha"]),
                            float(diagnostics["physics_update_norm"]),
                            float(diagnostics["physics_corr_norm"]),
                        )
                    else:
                        alpha = float(
                            diagnostics.get("legacy_alpha", lgd.current_physics_alpha().item())
                            if diagnostics
                            else lgd.current_physics_alpha().item()
                        )
                        self.logger.info("  Refiner physics alpha=%.6f", alpha)
                    if diagnostics and "stage_dc_alpha" in diagnostics:
                        stage_dc_alpha = ", ".join(f"{float(value):.6f}" for value in diagnostics["stage_dc_alpha"])
                        self.logger.info("  Refiner stage DC alpha=[%s]", stage_dc_alpha)
            # 记录训练指标用于画图
            self.training_history['train_loss'].append(loss.item())
            self.training_history['train_res'].append(float(train_res_tensor.item()))
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            if metrics is not None:
                self.training_history['data_fidelity_error'].append(metrics.get('data_fidelity_error', 0.0))
                self.training_history['update_difference'].append(metrics.get('update_difference', 0.0))
            iter_time = time.time() - iter_start_time
            if self.current_iter % TRAINING_CONFIG['validation_interval'] == 0:
                val_loss, val_metrics = self._validate()
                # 记录验证损失
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_res'].append(val_loss)

                # For logging, compute Train RES with the *updated* model so Train/Val are comparable.
                self.model.eval()
                with torch.no_grad():
                    coeff_pred_post, _, _ = self.model(coeff_initial, g_observed)
                    train_res_post = _relative_l2_loss(coeff_pred_post, coeff_true).item()
                self.model.train()

                data_err = float('nan')
                upd_diff = float('nan')
                if metrics is not None:
                    data_err = float(metrics.get('data_fidelity_error', float('nan')))
                    upd_diff = float(metrics.get('update_difference', float('nan')))
                self.logger.info(
                    f"Iter: {self.current_iter:4d} | "
                    f"Train RES: {train_res_post:.6f} | Val RES: {val_loss:.6f} | "
                    f"Loss(obj): {loss.item():.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.8f} | "
                    f"Time: {iter_time:.3f}s | "
                    f"Data Fidelity Error: {data_err:.6f} | "
                    f"Coeff Change: {upd_diff:.3e}"
                )
                if self.secondary_val_data_generator is not None:
                    secondary_val_loss, _ = self._validate(
                        generator=self.secondary_val_data_generator,
                        val_bs=int(os.environ.get("SECONDARY_VAL_BATCH_SIZE_OVERRIDE", DATA_CONFIG.get("val_batch_size", N_DATA))),
                        val_sample_count=int(os.environ.get("SECONDARY_VAL_SUBSAMPLE_SIZE_OVERRIDE", 100)),
                        random_subsample=_env_flag("SECONDARY_VAL_RANDOM_SUBSAMPLE_OVERRIDE", True),
                        reproducible=_env_flag("SECONDARY_VAL_REPRODUCIBLE_OVERRIDE", True),
                        seed_override=int(os.environ.get("SECONDARY_VAL_SEED_OVERRIDE", DATA_CONFIG.get("validation_seed", 42))),
                    )
                    self.logger.info(
                        "Iter: %4d | Secondary Val (%s) RES: %.6f | samples=%d",
                        self.current_iter,
                        self.secondary_val_label,
                        secondary_val_loss,
                        int(os.environ.get("SECONDARY_VAL_SUBSAMPLE_SIZE_OVERRIDE", 100)),
                    )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                if (self.patience_counter >= TRAINING_CONFIG['early_stopping_patience'] and
                    TRAINING_CONFIG['early_stopping_patience'] > 0):
                    self.logger.info(f"Early stopping triggered after {self.current_iter} iterations")
                    break
            if self.current_iter % TRAINING_CONFIG['save_interval'] == 0:
                self._save_checkpoint()
            if self.current_iter % 500 == 0 and self.current_iter > 0:
                self._save_training_plots()
        total_time = time.time() - total_start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self._save_checkpoint()
        self._save_training_plots()

    def _build_checkpoint_payload(self, *, include_training_state):
        checkpoint = {
            'iter': self.current_iter,
            'model_state_dict': export_trainable_state_dict(self.model, move_to_cpu=True),
            'model_state_format': 'trainable_parameters_only',
            'best_val_loss': self.best_val_loss,
            'experiment_metadata': self.experiment_metadata,
            'compact_checkpoint': not bool(include_training_state),
        }
        if include_training_state:
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'training_history': self.training_history,
            })
        return checkpoint

    def _save_checkpoint(self, is_best=False):
        # Default to compact model checkpoints for training outputs used by evaluation.
        # Set COMPACT_CHECKPOINTS_OVERRIDE=0 only when optimizer/scheduler state is needed for exact resume.
        compact_checkpoints = _env_flag("COMPACT_CHECKPOINTS_OVERRIDE", True)
        checkpoint = self._build_checkpoint_payload(include_training_state=not compact_checkpoints)
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, f'checkpoint_iter_{self.current_iter}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            torch.save(checkpoint, BEST_MODEL_PATH)
            self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.6f}")
        torch.save(checkpoint, MODEL_PATH)

    def _save_training_plots(self):
        if len(self.training_history['train_loss']) == 0:
            return
        fig = None
        try:
            start_idx = 150 if len(self.training_history['train_loss']) > 150 else 0

            def _slice(seq):
                return seq[start_idx:] if len(seq) > start_idx else seq

            train_loss = _slice(self.training_history['train_loss'])
            val_loss = _slice(self.training_history['val_loss'])
            lr_hist = _slice(self.training_history['learning_rate'])
            data_err = _slice(self.training_history['data_fidelity_error'])
            upd_diff = _slice(self.training_history['update_difference'])

            if len(train_loss) == 0:
                return
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes[0, 0].plot(train_loss, label='Train Loss')
            if len(val_loss) > 0:
                axes[0, 0].plot(val_loss, label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            axes[0, 1].plot(lr_hist)
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)
            axes[1, 0].plot(data_err)
            axes[1, 0].set_title('Data Fidelity Error')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].grid(True)
            axes[1, 1].plot(upd_diff)
            axes[1, 1].set_title('Coefficient Change (Init -> Final)')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Norm')
            axes[1, 1].grid(True)
            plt.tight_layout()
            plot_path = os.path.join(LOGGING_CONFIG['log_dir'], 'training_progress.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Training plots saved to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error saving training plots: {e}")
        finally:
            if fig is not None:
                plt.close(fig)

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            load_info = load_trainable_state_dict(self.model, checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_iter = checkpoint['iter']
            self.best_val_loss = checkpoint['best_val_loss']
            self.training_history = checkpoint['training_history']
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            self.logger.info(f"Resuming from iteration {self.current_iter}")
            self.logger.info(
                "Loaded %d trainable parameters from checkpoint; ignored %d non-parameter keys and %d buffer keys",
                load_info["loaded_parameter_count"],
                len(load_info["ignored_non_parameter_keys"]),
                len(load_info["missing_buffer_keys"]),
            )
        else:
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")


def main():
    print("=" * 60)
    print("THEORETICAL CT RECONSTRUCTION TRAINING")
    print("=" * 60)
    seed = _set_global_seed_from_env()
    if seed is not None:
        print(f"Using GLOBAL_SEED_OVERRIDE={seed}")
    trainer = TheoreticalTrainer()
    resume_path = _resolve_resume_checkpoint_path()
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"RESUME_CHECKPOINT_OVERRIDE not found: {resume_path}")
        trainer.load_checkpoint(resume_path)
    trainer.train()
    print("Theoretical training completed successfully!")


if __name__ == '__main__':
    main()
