import copy
import os
import re
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from model import initialize_model, load_trainable_state_dict
from radon_transform import TheoreticalDataGenerator
from config import (
    device,
    BEST_MODEL_PATH,
    MODEL_PATH,
    MODEL_DIR,
    RESULTS_DIR,
    IMAGE_SIZE,
    DATA_CONFIG,
    EXPERIMENT_OUTPUT_TAG,
    TIME_DOMAIN_CONFIG,
    _apply_experiment_profile,
)


def _normalize_runtime_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path = str(path).strip()
    if not path:
        return None
    if os.name == "nt":
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", path)
        if match is not None:
            drive = match.group(1).upper()
            tail = match.group(2).replace("/", "\\")
            return f"{drive}:\\{tail}"
    return path


def load_model(load_path: Optional[str] = None, checkpoint=None):
    env_load_path = str(os.environ.get("MODEL_LOAD_PATH_OVERRIDE", "") or "").strip()
    if load_path is None and env_load_path:
        load_path = env_load_path
    load_path = _normalize_runtime_path(load_path)
    if load_path is None:
        load_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}")

    if checkpoint is None:
        checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    experiment_metadata = checkpoint.get("experiment_metadata", {}) if isinstance(checkpoint, dict) else {}
    model = initialize_model()
    loaded_state = checkpoint.get("model_state_dict", checkpoint)
    load_info = load_trainable_state_dict(model, loaded_state)
    model.eval()
    print(f"Loaded checkpoint: {load_path}")
    skipped = load_info.get("ignored_non_parameter_keys", [])
    missing_buffers = load_info.get("missing_buffer_keys", [])
    if skipped:
        print(f"Ignored non-parameter keys: {skipped}")
    if missing_buffers:
        print(f"Rebuilt buffer keys from current config: {missing_buffers}")
    return model, experiment_metadata


def coeff_to_display_image(coeff: np.ndarray) -> np.ndarray:
    """B1*B1 主链下，系数图本身就是显示图。"""
    return coeff


def plot_result(idx, f_true, f_tikhonov, f_pred, res_tikhonov, res_pred, save_path, noise_desc, lambda_reg):
    """Plot synthesized f images; RES shown is still coefficient-domain RES."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = min(float(np.min(f_true)), float(np.min(f_tikhonov)), float(np.min(f_pred)))
    vmax = max(float(np.max(f_true)), float(np.max(f_tikhonov)), float(np.max(f_pred)))
    extent = (0.0, float(IMAGE_SIZE - 1), 0.0, float(IMAGE_SIZE - 1))

    im0 = axes[0].imshow(f_true, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title("True f(x,y)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(f_tikhonov, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent)
    axes[1].set_title(f"Tikhonov f(x,y)\nCoeff RES={res_tikhonov:.4f}")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(f_pred, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent)
    axes[2].set_title(f"Reconstruction f(x,y)\nCoeff RES={res_pred:.4f}")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.suptitle(f"Sample {idx} | {noise_desc} | lambda={lambda_reg:.4e}", y=1.08)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{idx}] saved: {save_path}")


def build_sample(generator: TheoreticalDataGenerator, seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    coeff_true, _, g_obs, coeff_init = generator.generate_training_sample(random_seed=seed)
    return coeff_true, g_obs, coeff_init


@torch.no_grad()
def compute_tikhonov_baseline(generator: TheoreticalDataGenerator, g_obs: torch.Tensor, lambda_reg) -> torch.Tensor:
    """Compute an explicit Tikhonov baseline for reporting/evaluation.

    This intentionally does not reuse the neural network input tensor named
    ``coeff_init``.  The network may use a different initialization path during
    experiments, while the reported baseline should stay the pure Tikhonov
    reconstruction for the same observation and lambda.
    """
    method = str(os.environ.get("EVAL_TIKHONOV_BASELINE_METHOD_OVERRIDE", "tikhonov_direct")).strip().lower()
    g_batch = g_obs.unsqueeze(0) if g_obs.dim() == 1 else g_obs
    if method in {"", "direct", "tikhonov_direct"}:
        coeff = generator.solve_tikhonov_direct_init(g_batch, lambda_reg=lambda_reg)
    elif method in {"cg", "tikhonov_cg"}:
        max_iter = int(os.environ.get("EVAL_TIKHONOV_BASELINE_CG_ITERS_OVERRIDE", TIME_DOMAIN_CONFIG.get("init_cg_iters", 40)))
        tol = float(os.environ.get("EVAL_TIKHONOV_BASELINE_CG_TOL_OVERRIDE", TIME_DOMAIN_CONFIG.get("init_cg_tol", 1.0e-4)))
        coeff = generator._tikhonov_cg_init(g_batch, lambda_reg=lambda_reg, max_iter=max_iter, tol=tol)
    else:
        raise ValueError(
            f"Unsupported EVAL_TIKHONOV_BASELINE_METHOD_OVERRIDE={method!r}; "
            "expected 'tikhonov_direct' or 'tikhonov_cg'."
        )
    return coeff.squeeze(0).squeeze(0)


def _resolve_checkpoint_from_tag(tag: str) -> str:
    tag = str(tag or "").strip()
    if not tag or tag.lower() == "default":
        best_path = BEST_MODEL_PATH
        model_path = MODEL_PATH
    else:
        stem = f"theoretical_ct_{tag}"
        best_path = os.path.join(MODEL_DIR, f"{stem}_best_model.pth")
        model_path = os.path.join(MODEL_DIR, f"{stem}_model.pth")
    return best_path if os.path.exists(best_path) else model_path


@contextmanager
def _temporary_experiment_config(experiment_metadata: dict):
    if not experiment_metadata:
        yield
        return

    backup = copy.deepcopy(TIME_DOMAIN_CONFIG)
    backup_cnn_angle_indices = backup.get("cnn_angle_indices_override", None)
    try:
        profile_name = str(experiment_metadata.get("experiment_profile", "") or "").strip()
        if profile_name:
            try:
                _apply_experiment_profile(profile_name)
            except ValueError:
                TIME_DOMAIN_CONFIG["experiment_profile"] = profile_name
        if experiment_metadata.get("operator_mode"):
            TIME_DOMAIN_CONFIG["operator_mode"] = str(experiment_metadata["operator_mode"])
        if experiment_metadata.get("theoretical_formula_mode"):
            TIME_DOMAIN_CONFIG["theoretical_formula_mode"] = str(
                experiment_metadata["theoretical_formula_mode"]
            )
        if experiment_metadata.get("data_formula_mode"):
            TIME_DOMAIN_CONFIG["data_formula_mode"] = str(
                experiment_metadata["data_formula_mode"]
            )
        if "alpha_values" in experiment_metadata:
            alpha_values = [float(v) for v in (experiment_metadata.get("alpha_values") or [])]
            TIME_DOMAIN_CONFIG["alpha_values"] = alpha_values
            TIME_DOMAIN_CONFIG["use_multi_angle"] = len(alpha_values) > 1
            TIME_DOMAIN_CONFIG["num_angles"] = int(len(alpha_values))
            TIME_DOMAIN_CONFIG["num_angles_total"] = int(len(alpha_values))
        if "alpha_tau_offsets" in experiment_metadata:
            TIME_DOMAIN_CONFIG["alpha_tau_offsets"] = [
                float(v) for v in (experiment_metadata.get("alpha_tau_offsets") or [])
            ]
        if "alpha_condition_constrained_json" in experiment_metadata:
            TIME_DOMAIN_CONFIG["alpha_condition_constrained_json"] = experiment_metadata.get(
                "alpha_condition_constrained_json"
            )
        if "cnn_backbone_only" in experiment_metadata:
            TIME_DOMAIN_CONFIG["cnn_backbone_only"] = bool(experiment_metadata["cnn_backbone_only"])
        selected_angle_indices = None
        if experiment_metadata.get("cnn_angle_indices") is not None:
            selected_angle_indices = [int(idx) for idx in experiment_metadata.get("cnn_angle_indices") or []]
        elif backup_cnn_angle_indices is not None:
            selected_angle_indices = [int(idx) for idx in list(backup_cnn_angle_indices)]

        if selected_angle_indices:
            TIME_DOMAIN_CONFIG["cnn_angle_indices_override"] = selected_angle_indices
            TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = int(len(selected_angle_indices))
        else:
            TIME_DOMAIN_CONFIG["cnn_angle_indices_override"] = None
            if "raw_cnn_angle_channels" in experiment_metadata:
                raw_cnn_angle_channels = int(experiment_metadata["raw_cnn_angle_channels"])
                TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = (
                    raw_cnn_angle_channels if raw_cnn_angle_channels > 0 else None
                )
            elif "cnn_num_angles" in experiment_metadata:
                cnn_num_angles = int(experiment_metadata["cnn_num_angles"])
                TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = (
                    cnn_num_angles if cnn_num_angles > 0 else None
                )
            elif "learned_num_angles" in experiment_metadata:
                learned_num_angles = int(experiment_metadata["learned_num_angles"])
                TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = (
                    learned_num_angles if learned_num_angles > 0 else None
                )
        if "cnn_angle_adapter_enabled" in experiment_metadata:
            TIME_DOMAIN_CONFIG["cnn_angle_adapter_enabled"] = bool(
                experiment_metadata["cnn_angle_adapter_enabled"]
            )
        if "cnn_angle_adapter_mode" in experiment_metadata:
            TIME_DOMAIN_CONFIG["cnn_angle_adapter_mode"] = str(
                experiment_metadata["cnn_angle_adapter_mode"]
            )
        if "cnn_angle_adapter_hidden_channels" in experiment_metadata:
            hidden_channels = int(experiment_metadata["cnn_angle_adapter_hidden_channels"])
            if hidden_channels > 0:
                TIME_DOMAIN_CONFIG["cnn_angle_adapter_hidden_channels"] = hidden_channels
        if bool(TIME_DOMAIN_CONFIG.get("cnn_angle_adapter_enabled", False)) and "cnn_num_angles" in experiment_metadata:
            adapter_output_channels = int(experiment_metadata["cnn_num_angles"])
            if adapter_output_channels > 0:
                TIME_DOMAIN_CONFIG["cnn_angle_adapter_output_channels"] = adapter_output_channels
        if "physics_residual_channel_enabled" in experiment_metadata:
            TIME_DOMAIN_CONFIG["physics_residual_channel_enabled"] = bool(experiment_metadata["physics_residual_channel_enabled"])
        if experiment_metadata.get("physics_residual_mode"):
            TIME_DOMAIN_CONFIG["physics_residual_mode"] = str(experiment_metadata["physics_residual_mode"])
        if "physics_residual_cg_iters" in experiment_metadata:
            TIME_DOMAIN_CONFIG["physics_residual_cg_iters"] = int(experiment_metadata["physics_residual_cg_iters"])
        if "physics_residual_damping" in experiment_metadata:
            TIME_DOMAIN_CONFIG["physics_residual_damping"] = float(experiment_metadata["physics_residual_damping"])
        if "physics_residual_detach" in experiment_metadata:
            TIME_DOMAIN_CONFIG["physics_residual_detach"] = bool(experiment_metadata["physics_residual_detach"])
        if "physics_residual_normalize" in experiment_metadata:
            TIME_DOMAIN_CONFIG["physics_residual_normalize"] = bool(experiment_metadata["physics_residual_normalize"])
        if "physics_explicit_update_enabled" in experiment_metadata:
            TIME_DOMAIN_CONFIG["physics_explicit_update_enabled"] = bool(experiment_metadata["physics_explicit_update_enabled"])
        yield
    finally:
        TIME_DOMAIN_CONFIG.clear()
        TIME_DOMAIN_CONFIG.update(backup)


def evaluate(
    noise_mode: Optional[str] = None,
    noise_level: Optional[float] = None,
    target_snr_db: Optional[float] = None,
    num_samples: int = 50,
    load_path: Optional[str] = None,
    result_prefix: Optional[str] = None,
    result_dir: Optional[str] = None,
):
    resolved_load_path = load_path
    env_load_path = str(os.environ.get("MODEL_LOAD_PATH_OVERRIDE", "") or "").strip()
    if resolved_load_path is None and env_load_path:
        resolved_load_path = env_load_path
    resolved_load_path = _normalize_runtime_path(resolved_load_path)
    if resolved_load_path is None:
        resolved_load_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH

    checkpoint = torch.load(resolved_load_path, map_location=device, weights_only=True)
    experiment_metadata = checkpoint.get("experiment_metadata", {}) if isinstance(checkpoint, dict) else {}

    with _temporary_experiment_config(experiment_metadata):
        model, experiment_metadata = load_model(load_path=resolved_load_path, checkpoint=checkpoint)
        lambda_reg_fixed = DATA_CONFIG.get("lambda_reg", 0.01)
        lambda_mode = str(DATA_CONFIG.get("lambda_select_mode", "fixed")).strip().lower()

        mode = noise_mode if noise_mode is not None else DATA_CONFIG.get("noise_mode", "additive")
        level = noise_level if noise_level is not None else DATA_CONFIG.get("noise_level", 0.1)
        snr_db = target_snr_db if target_snr_db is not None else DATA_CONFIG.get("target_snr_db", 30.0)
        if str(mode).strip().lower() == "snr":
            noise_desc = f"SNR={float(snr_db):g}dB"
        else:
            noise_desc = f"{mode} delta={level}"

        test_data_source = str(
            DATA_CONFIG.get("test_data_source", DATA_CONFIG.get("data_source", "random_ellipses"))
        ).strip().lower()
        generator = TheoreticalDataGenerator(data_source=test_data_source)
        generator.noise_mode = mode
        generator.noise_level = level
        generator.target_snr_db = float(snr_db)
        generator.image_gen = generator.image_gen.to(device)

        res_tikhonov_list = []
        res_pred_list = []
        last_plot_data = None

        for _ in range(num_samples):
            coeff_true, g_obs, coeff_init = build_sample(generator, seed=None)
            lam_used = float(generator.last_lambda) if generator.last_lambda is not None else float(lambda_reg_fixed)
            coeff_tikhonov = compute_tikhonov_baseline(generator, g_obs, lambda_reg=lam_used)
            coeff_true_cpu = coeff_true.squeeze().cpu()
            coeff_tikhonov_cpu = coeff_tikhonov.squeeze().cpu()

            g_obs_batch = g_obs.unsqueeze(0).to(device)
            coeff_init_batch = coeff_init.unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                coeff_pred_batch, _, _ = model(coeff_init_batch, g_obs_batch)

            coeff_pred = coeff_pred_batch.squeeze().detach().cpu()
            diff_sq_sum_pred = torch.sum(torch.abs(coeff_pred - coeff_true_cpu) ** 2)
            diff_sq_sum_tikhonov = torch.sum(torch.abs(coeff_tikhonov_cpu - coeff_true_cpu) ** 2)
            true_sq_sum = torch.sum(torch.abs(coeff_true_cpu) ** 2).clamp_min(1e-12)
            res_pred = torch.sqrt(diff_sq_sum_pred / true_sq_sum).item()
            res_tikhonov = torch.sqrt(diff_sq_sum_tikhonov / true_sq_sum).item()

            res_tikhonov_list.append(res_tikhonov)
            res_pred_list.append(res_pred)
            last_plot_data = (
                coeff_true_cpu.numpy(),
                coeff_tikhonov_cpu.numpy(),
                coeff_pred.numpy(),
                res_tikhonov,
                res_pred,
                lam_used,
            )

        mean_res_tikhonov = sum(res_tikhonov_list) / len(res_tikhonov_list)
        mean_res_pred = sum(res_pred_list) / len(res_pred_list)

        if last_plot_data is not None:
            coeff_true_np, coeff_tikhonov_np, coeff_pred_np, res_tikhonov_last, res_pred_last, lam_used_last = last_plot_data
            f_true_np = coeff_to_display_image(coeff_true_np)
            f_tikhonov_np = coeff_to_display_image(coeff_tikhonov_np)
            f_pred_np = coeff_to_display_image(coeff_pred_np)
            save_name = "shepp_logan_last.png" if test_data_source == "shepp_logan" else "random_ellipses_last.png"
            prefix = str(
                result_prefix or experiment_metadata.get("output_tag") or EXPERIMENT_OUTPUT_TAG or ""
            ).strip()
            if prefix:
                save_name = f"{prefix}_{save_name}"
            resolved_result_dir = _normalize_runtime_path(
                result_dir or os.environ.get("RESULT_DIR_OVERRIDE", "") or RESULTS_DIR
            )
            save_path = os.path.join(resolved_result_dir, save_name)
            plot_result(
                idx="MainExperiment_last",
                f_true=f_true_np,
                f_tikhonov=f_tikhonov_np,
                f_pred=f_pred_np,
                res_tikhonov=res_tikhonov_last,
                res_pred=res_pred_last,
                save_path=save_path,
                noise_desc=noise_desc,
                lambda_reg=lam_used_last if lambda_mode == "morozov" else lambda_reg_fixed,
            )

        print("==== Main Experiment Evaluation (Mean over samples) ====")
        print(f"Noise: {noise_desc}")
        print(f"Mean RES (tikhonov): {mean_res_tikhonov:.6f}")
        print(f"Mean RES (pred): {mean_res_pred:.6f}")
        return {
            "noise_desc": noise_desc,
            "mean_res_init": float(mean_res_tikhonov),
            "mean_res_tikhonov": float(mean_res_tikhonov),
            "mean_res_pred": float(mean_res_pred),
            "model_path": resolved_load_path,
            "output_tag": str(result_prefix or experiment_metadata.get("output_tag") or EXPERIMENT_OUTPUT_TAG or "default"),
        }


def compare_saved_models(tags: List[str], num_samples: int = 50, result_dir: Optional[str] = None):
    print("==== Saved Model Comparison ====")
    results = []
    for raw_tag in tags:
        tag = str(raw_tag).strip()
        if not tag:
            continue
        load_path = _resolve_checkpoint_from_tag(tag)
        print(f"[compare] tag={tag} | load_path={load_path}")
        result = evaluate(
            num_samples=num_samples,
            load_path=load_path,
            result_prefix=tag,
            result_dir=result_dir,
        )
        results.append(result)

    print("==== Comparison Summary ====")
    for item in results:
        print(
            f"{item['output_tag']}: Mean RES(tikhonov)={item['mean_res_tikhonov']:.6f} | "
            f"Mean RES(pred)={item['mean_res_pred']:.6f} | "
            f"model={item['model_path']}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved CT reconstruction models.")
    parser.add_argument("--model-path", type=str, default="", help="Checkpoint path to evaluate.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of evaluation samples.")
    parser.add_argument("--result-dir", type=str, default="", help="Directory for evaluation plots.")
    parser.add_argument("--result-prefix", type=str, default="", help="Prefix for saved evaluation plot names.")
    parser.add_argument(
        "--compare-tags",
        type=str,
        default="",
        help="Comma-separated experiment tags to compare, e.g. special8,random8_seed20260327",
    )
    args = parser.parse_args()

    num_samples = (
        int(args.num_samples)
        if args.num_samples is not None
        else int(os.environ.get("EVAL_NUM_SAMPLES_OVERRIDE", "50"))
    )
    compare_tags = str(args.compare_tags or os.environ.get("COMPARE_MODEL_TAGS_OVERRIDE", "") or "").strip()
    if compare_tags:
        compare_saved_models(
            tags=[token.strip() for token in compare_tags.split(",") if token.strip()],
            num_samples=num_samples,
            result_dir=str(args.result_dir or "").strip() or None,
        )
    else:
        evaluate(
            num_samples=num_samples,
            load_path=str(args.model_path or "").strip() or None,
            result_prefix=str(args.result_prefix or "").strip() or None,
            result_dir=str(args.result_dir or "").strip() or None,
        )
