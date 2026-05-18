"""Train the second-stage extra8 network initialized by frozen alpha16 network."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(__file__).resolve().parents[1]
DEEP_LEARN_DIR = MODELS_DIR / "deep_learn"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (THIS_DIR, DEEP_LEARN_DIR, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from angle_selection import write_selection_outputs


def _prepare_angle_files() -> dict[str, Path]:
    cache_dir = PROJECT_ROOT / "data" / "alpha_search_cache"
    full_json = Path(os.environ.get("ALPHA_FULL_JSON_OVERRIDE", cache_dir / "alpha_full_resume.json"))
    original16_json = Path(os.environ.get("ALPHA16_JSON_OVERRIDE", cache_dir / "alpha_selected16.json"))
    output_dir = Path(os.environ.get("ANGLE_SELECTION_OUTPUT_DIR_OVERRIDE", cache_dir))
    extra8_json = Path(os.environ.get("EXTRA8_JSON_OVERRIDE", output_dir / "alpha_extra8_from24_excluding16.json"))

    force = str(os.environ.get("FORCE_EXTRA8_RESELECT", "0")).strip().lower() in {"1", "true", "yes", "y"}
    comparison_json = output_dir / "alpha_selected16_no_exclude_comparison.json"
    selected24_json = output_dir / "alpha_selected24_no_exclude.json"
    generated16_json = output_dir / "alpha_selected16_no_exclude_regenerated.json"
    if force or not extra8_json.exists() or not comparison_json.exists() or not selected24_json.exists():
        paths = write_selection_outputs(
            full_json=full_json,
            selected16_json=original16_json,
            output_dir=output_dir,
            generated16_name=generated16_json.name,
            selected24_name=selected24_json.name,
            extra8_name=extra8_json.name,
            comparison_name=comparison_json.name,
        )
    else:
        paths = {
            "generated16": generated16_json,
            "selected24": selected24_json,
            "extra8": extra8_json,
            "comparison": comparison_json,
        }

    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    if not bool(comparison.get("all_match", False)):
        raise RuntimeError(
            "Regenerated no-exclusion alpha_selected16 does not match the reference "
            f"{original16_json}. See {paths['comparison']}."
        )
    return paths


ANGLE_PATHS = _prepare_angle_files()

os.environ.setdefault("EXPERIMENT_PROFILE_OVERRIDE", "alpha_condition")
os.environ.setdefault("ALPHA_CONDITION_JSON_OVERRIDE", str(ANGLE_PATHS["extra8"]))
os.environ.setdefault("MULTI_ANGLE_SOLVER_MODE_OVERRIDE", "stacked_tikhonov")
os.environ.setdefault("THEORETICAL_FORMULA_MODE_OVERRIDE", "alpha_continuous")
os.environ.setdefault("INIT_METHOD_OVERRIDE", "tikhonov_direct")
os.environ.setdefault("LAMBDA_SELECT_MODE_OVERRIDE", "morozov")
os.environ.setdefault("ALPHA_GRAM_CACHE_DIR_OVERRIDE", str(PROJECT_ROOT / "data" / "alpha_gram_cache"))
os.environ.setdefault("CNN_ANGLE_INDICES_OVERRIDE", "0,1,2,3,4,5,6,7")
os.environ.setdefault("CNN_NUM_ANGLES_OVERRIDE", "8")
os.environ.setdefault("CNN_ANGLE_ADAPTER_ENABLED_OVERRIDE", "0")
os.environ.setdefault("PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE", "1")
os.environ.setdefault("PHYSICS_RESIDUAL_MODE_OVERRIDE", "per_angle_cg")
os.environ.setdefault("PHYSICS_RESIDUAL_DAMPING_OVERRIDE", "1e-2")
os.environ.setdefault("PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE", "8")
os.environ.setdefault("PHYSICS_RESIDUAL_DETACH_OVERRIDE", "1")
os.environ.setdefault("PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE", "1")
os.environ.setdefault("PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE", "1")
os.environ.setdefault("PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE", "0.05")
os.environ.setdefault("PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE", "0.25")
os.environ.setdefault("INTERMEDIATE_SUPERVISION_ENABLED_OVERRIDE", "1")
os.environ.setdefault("INTERMEDIATE_SUPERVISION_WEIGHT_START_OVERRIDE", "0.2")
os.environ.setdefault("INTERMEDIATE_SUPERVISION_WEIGHT_END_OVERRIDE", "1.0")
os.environ.setdefault("NOISE_MODE_OVERRIDE", "multiplicative")
os.environ.setdefault("NOISE_LEVEL_OVERRIDE", "0.1")
os.environ.setdefault("N_DATA_OVERRIDE", "4")
os.environ.setdefault("N_TRAIN_OVERRIDE", "5000")
os.environ.setdefault("BASE_LR_OVERRIDE", "0.001")
os.environ.setdefault("OUTPUT_TAG_OVERRIDE", "alpha16_plus_extra8_continue_grad_phys_morozov_direct_noise01")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from cascade_data import (
    CascadeBatchGenerator,
    build_model_for_alpha_json,
    configure_alpha_condition_runtime,
    normalize_runtime_path,
    preserve_rng_state,
)
from config import (
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    DATA_CONFIG,
    EXPERIMENT_OUTPUT_TAG,
    LOGGING_CONFIG,
    MODEL_PATH,
    TIME_DOMAIN_CONFIG,
    TRAINING_CONFIG,
    device,
    n_data,
    n_train,
)
from model import count_parameters, export_trainable_state_dict, load_trainable_state_dict


N_TRAIN = int(os.environ.get("N_TRAIN_OVERRIDE", n_train))
N_DATA = int(os.environ.get("N_DATA_OVERRIDE", n_data))


def _set_global_seed_from_env():
    seed_raw = os.environ.get("GLOBAL_SEED_OVERRIDE", None)
    if seed_raw is None or str(seed_raw).strip() == "":
        return None
    seed = int(str(seed_raw).strip())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def _next_training_start_iter(current_iter: int) -> int:
    return max(int(current_iter) + 1, 0)


def _resolve_stage1_checkpoint_path() -> str:
    stage1_tag = "alpha16_even8_grad_phys_morozov_direct_noise01"
    default_path = PROJECT_ROOT / "checkpoints" / stage1_tag / (
        "theoretical_ct_alpha16_even8_grad_phys_morozov_direct_noise01_best_model.pth"
    )
    legacy_path = PROJECT_ROOT / "checkpoints" / "deep_learn" / (
        "theoretical_ct_alpha16_even8_grad_phys_morozov_direct_noise01_best_model.pth"
    )
    raw_override = os.environ.get("STAGE1_CHECKPOINT_PATH_OVERRIDE", "")
    candidates = [raw_override] if str(raw_override).strip() else [str(default_path), str(legacy_path)]
    if str(raw_override).strip():
        candidates.append(str(legacy_path))
    for candidate in candidates:
        path = normalize_runtime_path(candidate)
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Stage1 checkpoint not found. Tried: "
        + ", ".join(str(candidate) for candidate in candidates if str(candidate).strip())
    )


def _resolve_continue_checkpoint_path() -> str | None:
    path = normalize_runtime_path(os.environ.get("CONTINUE8_RESUME_CHECKPOINT_OVERRIDE", ""))
    return path if path else None


def _select_by_indices(values, indices):
    values_list = list(values or [])
    selected = []
    for idx in list(indices or []):
        idx = int(idx)
        if 0 <= idx < len(values_list):
            selected.append(float(values_list[idx]))
    return selected


class ContinueTrain8Trainer:
    def __init__(self):
        self._setup_logging()
        self.stage1_json = str(Path(os.environ.get("ALPHA16_JSON_OVERRIDE", PROJECT_ROOT / "data" / "alpha_search_cache" / "alpha_selected16.json")))
        self.stage2_json = str(ANGLE_PATHS["extra8"])
        self.stage1_checkpoint = _resolve_stage1_checkpoint_path()

        self.logger.info("Building frozen stage1 alpha16 model")
        self.stage1_model, self.stage1_checkpoint_metadata = build_model_for_alpha_json(
            alpha_json=self.stage1_json,
            cnn_angle_indices=os.environ.get("STAGE1_CNN_ANGLE_INDICES_OVERRIDE", "0,2,4,6,8,10,12,14"),
            cnn_num_angles=int(os.environ.get("STAGE1_CNN_NUM_ANGLES_OVERRIDE", "8")),
            checkpoint_path=self.stage1_checkpoint,
            frozen=True,
        )
        self.logger.info("Building trainable stage2 extra8 model")
        self.model, _ = build_model_for_alpha_json(
            alpha_json=self.stage2_json,
            cnn_angle_indices=os.environ.get("CNN_ANGLE_INDICES_OVERRIDE", "0,1,2,3,4,5,6,7"),
            cnn_num_angles=int(os.environ.get("CNN_NUM_ANGLES_OVERRIDE", "8")),
            checkpoint_path=None,
            frozen=False,
        )
        configure_alpha_condition_runtime(
            alpha_json=self.stage2_json,
            cnn_angle_indices=os.environ.get("CNN_ANGLE_INDICES_OVERRIDE", "0,1,2,3,4,5,6,7"),
            cnn_num_angles=int(os.environ.get("CNN_NUM_ANGLES_OVERRIDE", "8")),
        )

        train_data_source = str(DATA_CONFIG.get("train_data_source", DATA_CONFIG.get("data_source", "random_ellipses"))).strip().lower()
        val_data_source = str(DATA_CONFIG.get("val_data_source", DATA_CONFIG.get("data_source", train_data_source))).strip().lower()
        self.data_generator = CascadeBatchGenerator(
            stage1_model=self.stage1_model,
            stage2_model=self.model,
            data_source=train_data_source,
        )
        self.val_data_generator = CascadeBatchGenerator(
            stage1_model=self.stage1_model,
            stage2_model=self.model,
            data_source=val_data_source,
        )

        self.experiment_metadata = self._build_experiment_metadata(train_data_source, val_data_source)
        self._setup_optimizer()
        self.current_iter = -1
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_res": [],
            "val_res": [],
            "learning_rate": [],
            "data_fidelity_error": [],
            "update_difference": [],
        }
        self._log_startup_summary()

    def _setup_logging(self):
        log_dir = LOGGING_CONFIG["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger("ContinueTrain8Trainer")
        self.logger.setLevel(getattr(logging, LOGGING_CONFIG["log_level"]))
        self.logger.handlers.clear()
        if LOGGING_CONFIG["log_to_console"]:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(console_handler)
        if LOGGING_CONFIG["log_to_file"]:
            file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"))
            self.logger.addHandler(file_handler)

    def _setup_optimizer(self):
        scalar_names = ("step_size_raw", "reg_lambda_raw", "physics_alpha_raw")
        scalar_params = []
        other_params = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(key in name for key in scalar_names):
                scalar_params.append(parameter)
            else:
                other_params.append(parameter)
        base_lr = float(os.environ.get("BASE_LR_OVERRIDE", TRAINING_CONFIG["optimizer_learning_rate"]))
        scalar_lr = base_lr * float(TRAINING_CONFIG.get("scalar_lr_ratio", 0.1))
        self.optimizer = optim.AdamW(
            [
                {"params": other_params, "weight_decay": 1e-4},
                {"params": scalar_params, "weight_decay": 0.0, "lr": scalar_lr},
            ],
            lr=base_lr,
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0 / (1.0 + step / 500.0))

    def _build_experiment_metadata(self, train_data_source: str, val_data_source: str) -> dict:
        operator = self.model.optimizer.operator
        alpha_values = list(TIME_DOMAIN_CONFIG.get("alpha_values") or [])
        tau_offsets = list(TIME_DOMAIN_CONFIG.get("alpha_tau_offsets") or [])
        cnn_indices = [int(idx) for idx in getattr(self.model.optimizer, "cnn_channel_indices", [])]
        return {
            "output_tag": EXPERIMENT_OUTPUT_TAG,
            "cascade_mode": "alpha16_frozen_to_extra8",
            "train_data_source": train_data_source,
            "val_data_source": val_data_source,
            "stage1_alpha_json": self.stage1_json,
            "stage1_checkpoint": self.stage1_checkpoint,
            "stage1_checkpoint_metadata": self.stage1_checkpoint_metadata,
            "stage1_cnn_angle_indices": [0, 2, 4, 6, 8, 10, 12, 14],
            "stage2_alpha_json": self.stage2_json,
            "stage2_angle_selection_paths": {key: str(value) for key, value in ANGLE_PATHS.items()},
            "operator_class": operator.__class__.__name__,
            "num_angles": int(getattr(operator, "num_angles", 1) or 1),
            "cnn_num_angles": int(getattr(self.model.optimizer, "cnn_num_angles", 1) or 1),
            "cnn_angle_indices": cnn_indices,
            "cnn_angle_alpha_values": _select_by_indices(alpha_values, cnn_indices),
            "cnn_angle_tau_offsets": _select_by_indices(tau_offsets, cnn_indices),
            "alpha_values": alpha_values,
            "alpha_tau_offsets": tau_offsets,
            "physics_residual_channel_enabled": bool(getattr(self.model.optimizer, "physics_residual_enabled", False)),
            "physics_residual_mode": str(getattr(self.model.optimizer, "physics_residual_mode", "")),
            "physics_explicit_update_enabled": bool(getattr(self.model.optimizer, "physics_explicit_update_enabled", False)),
            "noise_mode": str(DATA_CONFIG.get("noise_mode", "")),
            "noise_level": float(DATA_CONFIG.get("noise_level", 0.0)),
        }

    def _log_startup_summary(self):
        self.logger.info("Experiment tag: %s", EXPERIMENT_OUTPUT_TAG)
        self.logger.info("Angle comparison JSON: %s", ANGLE_PATHS["comparison"])
        self.logger.info("No-exclusion top24 JSON: %s", ANGLE_PATHS["selected24"])
        self.logger.info("Extra8 JSON: %s", ANGLE_PATHS["extra8"])
        self.logger.info("Stage1 checkpoint: %s", self.stage1_checkpoint)
        self.logger.info("Stage1 frozen parameters: %d", sum(p.numel() for p in self.stage1_model.parameters() if not p.requires_grad))
        self.logger.info("Stage2 trainable parameters: %d", count_parameters(self.model))
        self.logger.info("Stage2 alpha values: %s", self.experiment_metadata["alpha_values"])
        self.logger.info("Stage2 tau offsets: %s", self.experiment_metadata["alpha_tau_offsets"])

    def _generate_training_batch(self, batch_size: int | None = None):
        return self.data_generator.generate_batch(int(batch_size or N_DATA), random_seed=None)

    def _validate(self):
        val_bs = int(os.environ.get("VAL_BATCH_SIZE_OVERRIDE", DATA_CONFIG.get("val_batch_size", N_DATA)))
        val_repro = bool(DATA_CONFIG.get("val_reproducible", False))
        env_val_repro = os.environ.get("VAL_REPRODUCIBLE_OVERRIDE", None)
        if env_val_repro is not None:
            val_repro = env_val_repro.strip().lower() in {"1", "true", "yes", "y"}
        seed = int(os.environ.get("VAL_SEED_OVERRIDE", DATA_CONFIG.get("validation_seed", 42))) if val_repro else None
        if seed is not None:
            with preserve_rng_state():
                batch = self.val_data_generator.generate_batch(batch_size=val_bs, random_seed=seed)
        else:
            batch = self.val_data_generator.generate_batch(batch_size=val_bs, random_seed=None)

        coeff_true = batch["coeff_true"].to(device)
        coeff_stage1 = batch["coeff_stage1"].to(device)
        g8_observed = batch["g8_observed"].to(device)
        self.model.eval()
        with torch.no_grad():
            coeff_pred, _, metrics = self.model(coeff_stage1, g8_observed)
            diff_sq_sum = torch.sum(torch.abs(coeff_pred - coeff_true) ** 2)
            true_sq_sum = torch.sum(torch.abs(coeff_true) ** 2).clamp_min(1.0e-12)
            val_loss = torch.sqrt(diff_sq_sum / true_sq_sum)
        self.model.train()
        return float(val_loss.item()), metrics

    def _res_loss(self, coeff_pred: torch.Tensor, coeff_true: torch.Tensor) -> torch.Tensor:
        true_sq_sum = torch.sum(torch.abs(coeff_true) ** 2).clamp_min(1.0e-12)
        return torch.sqrt(torch.sum(torch.abs(coeff_pred - coeff_true) ** 2) / true_sq_sum)

    def _training_loss(self, coeff_pred, history, coeff_true):
        train_res_tensor = self._res_loss(coeff_pred, coeff_true)
        if bool(DATA_CONFIG.get("intermediate_supervision_enabled", False)):
            hist = history[1:]
            if hist:
                step_losses = torch.stack([self._res_loss(item, coeff_true) for item in hist])
                weights = torch.linspace(
                    float(DATA_CONFIG.get("intermediate_supervision_weight_start", 0.2)),
                    float(DATA_CONFIG.get("intermediate_supervision_weight_end", 1.0)),
                    steps=len(step_losses),
                    device=step_losses.device,
                    dtype=step_losses.dtype,
                )
                weights = weights / weights.sum().clamp_min(1.0e-12)
                return torch.sum(weights * step_losses), train_res_tensor
        return train_res_tensor, train_res_tensor

    def train(self):
        self.logger.info("Starting alpha16 -> extra8 cascade training")
        self.logger.info("Total iterations: %d", N_TRAIN)
        self.logger.info("Batch size: %d", N_DATA)
        start_iter = _next_training_start_iter(self.current_iter)
        total_start_time = time.time()
        if start_iter >= N_TRAIN:
            self.logger.info("No training iterations to run: checkpoint iter=%d target=%d", self.current_iter, N_TRAIN)
        for self.current_iter in range(start_iter, N_TRAIN):
            iter_start = time.time()
            batch = self._generate_training_batch()
            coeff_true = batch["coeff_true"].to(device)
            coeff_stage1 = batch["coeff_stage1"].to(device)
            g8_observed = batch["g8_observed"].to(device)

            self.optimizer.zero_grad()
            coeff_pred, history, metrics = self.model(coeff_stage1, g8_observed)
            loss, train_res_tensor = self._training_loss(coeff_pred, history, coeff_true)
            loss.backward()
            if TRAINING_CONFIG.get("gradient_clip_value", 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING_CONFIG["gradient_clip_value"])
            self.optimizer.step()
            self.scheduler.step()

            self.training_history["train_loss"].append(float(loss.item()))
            self.training_history["train_res"].append(float(train_res_tensor.item()))
            self.training_history["learning_rate"].append(float(self.optimizer.param_groups[0]["lr"]))
            if metrics is not None:
                self.training_history["data_fidelity_error"].append(float(metrics.get("data_fidelity_error", 0.0)))
                self.training_history["update_difference"].append(float(metrics.get("update_difference", 0.0)))

            if self.current_iter % TRAINING_CONFIG["validation_interval"] == 0:
                val_loss, val_metrics = self._validate()
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_res"].append(val_loss)
                self.model.eval()
                with torch.no_grad():
                    coeff_pred_post, _, _ = self.model(coeff_stage1, g8_observed)
                    train_res_post = float(self._res_loss(coeff_pred_post, coeff_true).item())
                self.model.train()
                data_err = float(metrics.get("data_fidelity_error", float("nan"))) if metrics is not None else float("nan")
                upd_diff = float(metrics.get("update_difference", float("nan"))) if metrics is not None else float("nan")
                self.logger.info(
                    "Iter: %4d | Train RES: %.6f | Val RES: %.6f | Loss: %.6f | LR: %.8f | Time: %.3fs | Data Fidelity: %.6f | Coeff Change: %.3e",
                    self.current_iter,
                    train_res_post,
                    val_loss,
                    float(loss.item()),
                    float(self.optimizer.param_groups[0]["lr"]),
                    time.time() - iter_start,
                    data_err,
                    upd_diff,
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                if self.patience_counter >= TRAINING_CONFIG["early_stopping_patience"] > 0:
                    self.logger.info("Early stopping triggered after iteration %d", self.current_iter)
                    break

            if self.current_iter % TRAINING_CONFIG["save_interval"] == 0:
                self._save_checkpoint()
            if self.current_iter % 500 == 0 and self.current_iter > 0:
                self._save_training_plots()

        self.logger.info("Training completed in %.2f seconds", time.time() - total_start_time)
        self.logger.info("Best validation loss: %.6f", self.best_val_loss)
        self._save_checkpoint()
        self._save_training_plots()

    def _save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            "iter": int(self.current_iter),
            "model_state_dict": export_trainable_state_dict(self.model, move_to_cpu=True),
            "model_state_format": "trainable_parameters_only",
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": float(self.best_val_loss),
            "training_history": self.training_history,
            "experiment_metadata": self.experiment_metadata,
        }
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_iter_{self.current_iter}.pth")
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            torch.save(checkpoint, BEST_MODEL_PATH)
            self.logger.info("New best model saved: %s", BEST_MODEL_PATH)
        torch.save(checkpoint, MODEL_PATH)

    def _save_training_plots(self):
        if not self.training_history["train_loss"]:
            return
        fig = None
        try:
            start_idx = 150 if len(self.training_history["train_loss"]) > 150 else 0

            def _slice(values):
                return values[start_idx:] if len(values) > start_idx else values

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes[0, 0].plot(_slice(self.training_history["train_loss"]), label="Train Loss")
            if self.training_history["val_loss"]:
                axes[0, 0].plot(_slice(self.training_history["val_loss"]), label="Val Loss")
            axes[0, 0].set_title("Continue8 Training and Validation Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            axes[0, 1].plot(_slice(self.training_history["learning_rate"]))
            axes[0, 1].set_title("Learning Rate")
            axes[0, 1].grid(True)
            axes[1, 0].plot(_slice(self.training_history["data_fidelity_error"]))
            axes[1, 0].set_title("Extra8 Data Fidelity Error")
            axes[1, 0].grid(True)
            axes[1, 1].plot(_slice(self.training_history["update_difference"]))
            axes[1, 1].set_title("Coeff Change")
            axes[1, 1].grid(True)
            plt.tight_layout()
            plot_path = os.path.join(LOGGING_CONFIG["log_dir"], "training_progress.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            self.logger.info("Training plots saved to %s", plot_path)
        finally:
            if fig is not None:
                plt.close(fig)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint_path = normalize_runtime_path(checkpoint_path)
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"CONTINUE8_RESUME_CHECKPOINT_OVERRIDE not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        load_trainable_state_dict(self.model, checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_iter = int(checkpoint["iter"])
        self.best_val_loss = float(checkpoint["best_val_loss"])
        self.training_history = checkpoint["training_history"]
        self.logger.info("Loaded second-stage checkpoint from %s", checkpoint_path)


def main():
    print("=" * 72)
    print("ALPHA16 -> EXTRA8 CONTINUE TRAINING")
    print("=" * 72)
    seed = _set_global_seed_from_env()
    if seed is not None:
        print(f"Using GLOBAL_SEED_OVERRIDE={seed}")
    trainer = ContinueTrain8Trainer()
    resume_path = _resolve_continue_checkpoint_path()
    if resume_path:
        trainer.load_checkpoint(resume_path)
    trainer.train()
    print("Continue8 training completed.")


if __name__ == "__main__":
    main()
