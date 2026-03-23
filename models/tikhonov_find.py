"""Grid search helper for Tikhonov lambda on the current B1*B1 pipeline."""

import argparse
import os
from typing import Optional

import matplotlib
import numpy as np
import torch

from config import DATA_CONFIG, RESULTS_DIR, device
from radon_transform import TheoreticalDataGenerator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def tikhonov_cg_solve(
    generator: TheoreticalDataGenerator,
    g_obs: torch.Tensor,
    lambda_reg: float,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Solve (A^T A + lambda I)c = A^T g with CG in coefficient space."""
    if g_obs.dim() == 1:
        g_obs = g_obs.unsqueeze(0)
    g_obs = g_obs.to(device=device, dtype=torch.float32)

    b = generator.adjoint_operator(g_obs).to(dtype=torch.float32)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.sum(r * r)
    eps = b.new_tensor(1e-12)

    for _ in range(int(max_iter)):
        Ap = generator.adjoint_operator(generator.forward_operator(p)) + (lambda_reg * p)
        denom = torch.sum(p * Ap).clamp_min(eps)
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)
        if torch.sqrt(rsnew).item() < tol:
            break
        p = r + (rsnew / (rsold + eps)) * p
        rsold = rsnew

    return x.squeeze(0).squeeze(0)


def build_generator(
    data_source: str,
    noise_mode: str,
    noise_level: float,
    target_snr_db: float,
) -> TheoreticalDataGenerator:
    generator = TheoreticalDataGenerator(data_source=data_source)
    generator.noise_mode = noise_mode
    generator.noise_level = noise_level
    generator.target_snr_db = target_snr_db
    return generator


def get_sample(generator: TheoreticalDataGenerator, seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    coeff_true, _, g_obs, _ = generator.generate_training_sample(random_seed=seed)
    return coeff_true, g_obs


def search_lambda(
    generator: TheoreticalDataGenerator,
    coeff_true: torch.Tensor,
    g_obs: torch.Tensor,
    lambda_list: list[float],
):
    best_res = float("inf")
    best_lambda = None
    best_est = None

    for lam in lambda_list:
        coeff_est = tikhonov_cg_solve(generator, g_obs, lambda_reg=float(lam))
        diff_norm = torch.norm(coeff_est - coeff_true.to(device))
        true_norm = torch.norm(coeff_true.to(device)).clamp_min(1e-12)
        res = (diff_norm / true_norm).item()
        print(f"{lam:<12.4e} | {res:.6f}")
        if res < best_res:
            best_res = res
            best_lambda = float(lam)
            best_est = coeff_est

    return best_lambda, best_res, best_est


def save_heatmap(
    coeff_true: torch.Tensor,
    coeff_est: torch.Tensor,
    lambda_reg: float,
    res: float,
    data_source: str,
    noise_mode: str,
    noise_level: float,
    target_snr_db: float,
):
    coeff_true_np = coeff_true.detach().cpu().numpy()
    coeff_est_np = coeff_est.detach().cpu().numpy()
    vmin = min(float(coeff_true_np.min()), float(coeff_est_np.min()))
    vmax = max(float(coeff_true_np.max()), float(coeff_est_np.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(coeff_true_np, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"True Coeff ({data_source})")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(coeff_est_np, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if noise_mode == "snr":
        noise_desc = f"SNR={target_snr_db:g}dB"
    else:
        noise_desc = f"{noise_mode} delta={noise_level}"
    axes[1].set_title(
        f"Best Tikhonov\nlambda={lambda_reg:.2e}, RES={res:.4f}\n{noise_desc}"
    )
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"tikhonov_best_res_{data_source}.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {out_path}")


def main(
    data_source: str,
    noise_mode: str,
    noise_level: float,
    target_snr_db: float,
    seed: int,
):
    print(f"Running Tikhonov lambda search on device: {device}")
    print(f"Data source: {data_source}")
    if noise_mode == "snr":
        print(f"Noise: SNR={target_snr_db:g}dB")
    else:
        print(f"Noise: {noise_mode} delta={noise_level}")

    lambda_list = [
        1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
        1e-3, 1e-2, 5e-2, 1e-1, 5e-1,
        1.0, 5.0, 10.0, 50.0, 100.0,
    ]

    generator = build_generator(
        data_source=data_source,
        noise_mode=noise_mode,
        noise_level=noise_level,
        target_snr_db=target_snr_db,
    )
    coeff_true, g_obs = get_sample(generator, seed=seed)
    print(f"Sample shape: coeff={tuple(coeff_true.shape)}, g={tuple(g_obs.shape)}")
    print(f"{'Lambda':<12} | RES")
    print("-" * 24)

    best_lambda, best_res, best_est = search_lambda(generator, coeff_true, g_obs, lambda_list)
    print("-" * 24)
    print(f"Best lambda: {best_lambda:.6e}")
    print(f"Best RES   : {best_res:.6f}")

    if best_est is not None:
        save_heatmap(
            coeff_true=coeff_true,
            coeff_est=best_est,
            lambda_reg=best_lambda,
            res=best_res,
            data_source=data_source,
            noise_mode=noise_mode,
            noise_level=noise_level,
            target_snr_db=target_snr_db,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search a Tikhonov lambda on the current B1*B1 pipeline.")
    parser.add_argument(
        "--data_source",
        type=str,
        default=str(DATA_CONFIG.get("test_data_source", "shepp_logan")),
        choices=["random_ellipses", "shepp_logan"],
        help="Ground-truth source used to generate the sample.",
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default=str(DATA_CONFIG.get("noise_mode", "additive")),
        choices=["additive", "multiplicative", "snr"],
        help="Noise model.",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=float(DATA_CONFIG.get("noise_level", 0.1)),
        help="Noise delta used by the chosen noise model.",
    )
    parser.add_argument(
        "--target_snr_db",
        type=float,
        default=float(DATA_CONFIG.get("target_snr_db", 30.0)),
        help="Target SNR in dB when noise_mode=snr.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sample seed.")
    args = parser.parse_args()

    main(
        data_source=args.data_source,
        noise_mode=args.noise_mode,
        noise_level=args.noise_level,
        target_snr_db=args.target_snr_db,
        seed=args.seed,
    )
