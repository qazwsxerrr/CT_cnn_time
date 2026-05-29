import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import time
import os
import sys
import logging
import random
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
    TRAINING_CONFIG, DATA_CONFIG, LOGGING_CONFIG, TIME_DOMAIN_CONFIG, EXPERIMENT_OUTPUT_TAG
)

# Optional quick overrides for debugging (do not affect config.py).
N_TRAIN = int(os.environ.get("N_TRAIN_OVERRIDE", n_train))
N_DATA = int(os.environ.get("N_DATA_OVERRIDE", n_data))


def _resolve_resume_checkpoint_path():
    resume_path = str(os.environ.get("RESUME_CHECKPOINT_OVERRIDE", "") or "").strip()
    return resume_path or None


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
        self.logger.info("Train data source: %s", train_data_source)
        self.logger.info("Validation data source: %s", val_data_source)
        if os.environ.get("OFFLINE_TRAIN_DATASET_OVERRIDE", "").strip():
            self.logger.info("Offline train dataset: %s", os.environ["OFFLINE_TRAIN_DATASET_OVERRIDE"])
        if os.environ.get("OFFLINE_VAL_DATASET_OVERRIDE", "").strip():
            self.logger.info("Offline validation dataset: %s", os.environ["OFFLINE_VAL_DATASET_OVERRIDE"])
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
            "Validation: interval=%d batch_size=%d random_subsample=%s reproducible=%s",
            int(TRAINING_CONFIG.get("validation_interval", 10)),
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
        self.optimizer = optim.AdamW([
            {'params': other_params, 'weight_decay': 1e-4},
            {'params': scalar_params, 'weight_decay': 0.0, 'lr': scalar_lr},
        ], lr=base_lr)

        # Match the reference project: mild inverse decay is less aggressive than cosine here.
        def lr_lambda(step):
            return 1.0 / (1.0 + step / 500.0)

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
        self.logger.info("Theoretical trainer initialized successfully")

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

    def _validate(self):
        val_bs = DATA_CONFIG.get('val_batch_size', N_DATA)
        # Optional override to make validation fixed/reproducible for debugging.
        # This keeps the validation set (including noise) fixed so RES across iterations is comparable.
        val_repro = bool(DATA_CONFIG.get("val_reproducible", False))
        env_val_repro = os.environ.get("VAL_REPRODUCIBLE_OVERRIDE", None)
        if env_val_repro is not None:
            val_repro = env_val_repro.strip().lower() in ("1", "true", "yes", "y")

        seed = None
        if val_repro:
            seed = int(os.environ.get("VAL_SEED_OVERRIDE", DATA_CONFIG.get("validation_seed", 42)))
        val_random_subsample = bool(DATA_CONFIG.get("val_random_subsample", False))
        if val_random_subsample and hasattr(self.val_data_generator, "generate_random_batch"):
            subset_seed = None
            if seed is not None:
                subset_seed = seed + max(int(getattr(self, "current_iter", 0)), 0)
            coeff_true_val, _, g_observed_val, coeff_initial_val = self.val_data_generator.generate_random_batch(
                batch_size=val_bs,
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
                coeff_true_val, _, g_observed_val, coeff_initial_val = self.val_data_generator.generate_batch(
                    batch_size=val_bs, random_seed=seed
                )
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.random.set_rng_state(torch_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
        else:
            coeff_true_val, _, g_observed_val, coeff_initial_val = self.val_data_generator.generate_batch(
                batch_size=val_bs, random_seed=None
            )
        coeff_true_val = coeff_true_val.to(device)
        g_observed_val = g_observed_val.to(device)
        coeff_initial_val = coeff_initial_val.to(device)
        self.model.eval()
        with torch.no_grad():
            coeff_pred, _, metrics = self.model(coeff_initial_val, g_observed_val)
            diff_sq_sum = torch.sum(torch.abs(coeff_pred - coeff_true_val) ** 2)
            true_sq_sum = torch.sum(torch.abs(coeff_true_val) ** 2).clamp_min(1e-12)
            val_loss = torch.sqrt(diff_sq_sum / true_sq_sum)
        self.model.train()
        return val_loss.item(), metrics

    def train(self):
        self.logger.info("Starting theoretical CT reconstruction training...")
        self.logger.info(f"Total iterations: {N_TRAIN}")
        self.logger.info(f"Batch size: {N_DATA}")
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")
        self.logger.info("Objective mode: RES")
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

            true_sq_sum = torch.sum(torch.abs(coeff_true) ** 2).clamp_min(1e-12)

            def _res_loss(x):
                return torch.sqrt(torch.sum(torch.abs(x - coeff_true) ** 2) / true_sq_sum)

            train_res_tensor = _res_loss(coeff_pred)
            if bool(DATA_CONFIG.get("intermediate_supervision_enabled", False)):
                # history[0] is the initialization. Supervise only actual unrolled updates.
                hist = history[1:]
                if len(hist) == 0:
                    loss = train_res_tensor
                else:
                    step_losses = torch.stack([_res_loss(x) for x in hist])
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
            else:
                loss = train_res_tensor
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
                self.logger.info(
                    "  Learned scalars: step=%.6f lambda=%.6f",
                    float(lgd.current_step_size().item()),
                    float(lgd.current_reg_lambda().item()),
                )
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
                    diff_sq_sum_post = torch.sum(torch.abs(coeff_pred_post - coeff_true) ** 2)
                    train_res_post = torch.sqrt(diff_sq_sum_post / true_sq_sum).item()
                self.model.train()

                data_err = float('nan')
                upd_diff = float('nan')
                if metrics is not None:
                    data_err = float(metrics.get('data_fidelity_error', float('nan')))
                    upd_diff = float(metrics.get('update_difference', float('nan')))
                self.logger.info(
                    f"Iter: {self.current_iter:4d} | "
                    f"Train RES: {train_res_post:.6f} | Val RES: {val_loss:.6f} | "
                    f"Loss(RES): {loss.item():.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.8f} | "
                    f"Time: {iter_time:.3f}s | "
                    f"Data Fidelity Error: {data_err:.6f} | "
                    f"Coeff Change: {upd_diff:.3e}"
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

    def _save_checkpoint(self, is_best=False):
        checkpoint = {
            'iter': self.current_iter,
            'model_state_dict': export_trainable_state_dict(self.model, move_to_cpu=True),
            'model_state_format': 'trainable_parameters_only',
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'experiment_metadata': self.experiment_metadata,
        }
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
