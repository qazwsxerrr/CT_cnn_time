import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import initialize_model
from radon_transform import TheoreticalDataGenerator
from config import device, BEST_MODEL_PATH, MODEL_PATH, RESULTS_DIR, IMAGE_SIZE, DATA_CONFIG


def load_model():
    model = initialize_model()
    load_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {BEST_MODEL_PATH} or {MODEL_PATH}")

    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    loaded_state = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    filtered = {k: v for k, v in loaded_state.items() if k in model_state and model_state[k].shape == v.shape}
    skipped = [k for k in loaded_state.keys() if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {load_path}")
    if skipped:
        print(f"Skipped mismatched keys: {skipped}")
    return model


def coeff_to_display_image(coeff: np.ndarray) -> np.ndarray:
    """B1*B1 主链下，系数图本身就是显示图。"""
    return coeff


def plot_result(idx, f_true, f_init, f_pred, res_init, res_pred, save_path, noise_desc, lambda_reg):
    """Plot synthesized f images; RES shown is still coefficient-domain RES."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = min(float(np.min(f_true)), float(np.min(f_init)), float(np.min(f_pred)))
    vmax = max(float(np.max(f_true)), float(np.max(f_init)), float(np.max(f_pred)))
    extent = (0.0, float(IMAGE_SIZE - 1), 0.0, float(IMAGE_SIZE - 1))

    im0 = axes[0].imshow(f_true, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title("True f(x,y)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(f_init, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent)
    axes[1].set_title(f"Init f(x,y)\nCoeff RES={res_init:.4f}")
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


def evaluate(
    noise_mode: Optional[str] = None,
    noise_level: Optional[float] = None,
    target_snr_db: Optional[float] = None,
    num_samples: int = 50,
):
    model = load_model()
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

    res_init_list = []
    res_pred_list = []
    last_plot_data = None

    for _ in range(num_samples):
        coeff_true, g_obs, coeff_init = build_sample(generator, seed=None)
        lam_used = float(generator.last_lambda) if generator.last_lambda is not None else float(lambda_reg_fixed)
        coeff_true_cpu = coeff_true.squeeze().cpu()
        coeff_init_cpu = coeff_init.squeeze().cpu()

        g_obs_batch = g_obs.unsqueeze(0).to(device)
        coeff_init_batch = coeff_init.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            coeff_pred_batch, _, _ = model(coeff_init_batch, g_obs_batch)

        coeff_pred = coeff_pred_batch.squeeze().detach().cpu()
        diff_sq_sum_pred = torch.sum(torch.abs(coeff_pred - coeff_true_cpu) ** 2)
        diff_sq_sum_init = torch.sum(torch.abs(coeff_init_cpu - coeff_true_cpu) ** 2)
        true_sq_sum = torch.sum(torch.abs(coeff_true_cpu) ** 2).clamp_min(1e-12)
        res_pred = torch.sqrt(diff_sq_sum_pred / true_sq_sum).item()
        res_init = torch.sqrt(diff_sq_sum_init / true_sq_sum).item()

        res_init_list.append(res_init)
        res_pred_list.append(res_pred)
        last_plot_data = (
            coeff_true_cpu.numpy(),
            coeff_init_cpu.numpy(),
            coeff_pred.numpy(),
            res_init,
            res_pred,
            lam_used,
        )

    mean_res_init = sum(res_init_list) / len(res_init_list)
    mean_res_pred = sum(res_pred_list) / len(res_pred_list)

    if last_plot_data is not None:
        coeff_true_np, coeff_init_np, coeff_pred_np, res_init_last, res_pred_last, lam_used_last = last_plot_data
        f_true_np = coeff_to_display_image(coeff_true_np)
        f_init_np = coeff_to_display_image(coeff_init_np)
        f_pred_np = coeff_to_display_image(coeff_pred_np)
        save_name = "shepp_logan_last.png" if test_data_source == "shepp_logan" else "random_ellipses_last.png"
        save_path = os.path.join(RESULTS_DIR, save_name)
        plot_result(
            idx="MainExperiment_last",
            f_true=f_true_np,
            f_init=f_init_np,
            f_pred=f_pred_np,
            res_init=res_init_last,
            res_pred=res_pred_last,
            save_path=save_path,
            noise_desc=noise_desc,
            lambda_reg=lam_used_last if lambda_mode == "morozov" else lambda_reg_fixed,
        )

    print("==== Main Experiment Evaluation (Mean over samples) ====")
    print(f"Noise: {noise_desc}")
    print(f"Mean RES (init): {mean_res_init:.6f}")
    print(f"Mean RES (pred): {mean_res_pred:.6f}")


if __name__ == "__main__":
    evaluate(num_samples=50)
