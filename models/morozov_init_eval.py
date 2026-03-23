"""Evaluate Morozov-selected direct Tikhonov initialization on 128x128 Shepp-Logan."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import matplotlib
import numpy as np
import torch

from config import DATA_CONFIG, RESULTS_DIR, TIME_DOMAIN_CONFIG, device
from radon_transform import TheoreticalDataGenerator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "morozov_init_eval_remote")
DEFAULT_BASELINE_RESULTS_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "morozov_init_eval_remote_results.txt")


def _coeff_res(coeff_est: torch.Tensor, coeff_true: torch.Tensor) -> float:
    coeff_est = coeff_est.to(dtype=torch.float32)
    coeff_true = coeff_true.to(dtype=torch.float32, device=coeff_est.device)
    diff_norm = torch.norm(coeff_est - coeff_true)
    true_norm = torch.norm(coeff_true).clamp_min(1e-12)
    return float((diff_norm / true_norm).item())


def _scenario_defs() -> List[Dict[str, object]]:
    return [
        {
            "name": "clean",
            "noise_mode": "additive",
            "noise_level": 0.0,
            "display_noise": "additive",
            "display_level": "0.0",
        },
        {
            "name": "mult_0_01",
            "noise_mode": "multiplicative",
            "noise_level": 0.01,
            "display_noise": "multiplicative",
            "display_level": "0.01",
        },
        {
            "name": "mult_0_1",
            "noise_mode": "multiplicative",
            "noise_level": 0.1,
            "display_noise": "multiplicative",
            "display_level": "0.1",
        },
    ]


def _select_scenarios(requested: str) -> List[Dict[str, object]]:
    requested = str(requested).strip().lower()
    scenarios = _scenario_defs()
    if requested == "all":
        return scenarios
    for item in scenarios:
        if item["name"] == requested:
            return [item]
    allowed = ["all"] + [item["name"] for item in scenarios]
    raise ValueError(f"Unsupported scenario={requested!r}; expected one of {allowed!r}.")


def _configure_morozov(cache_dir: str) -> None:
    DATA_CONFIG["lambda_select_mode"] = "morozov"
    DATA_CONFIG["morozov_tau"] = 1.0
    DATA_CONFIG["morozov_max_iter"] = 8
    DATA_CONFIG["morozov_lambda_min"] = 1.0e-12
    DATA_CONFIG["morozov_lambda_max"] = 1.0e12
    DATA_CONFIG["morozov_newton_tol"] = 1.0e-8
    DATA_CONFIG["morozov_initial_lambda"] = 1.0
    DATA_CONFIG["morozov_cache_dir"] = cache_dir
    TIME_DOMAIN_CONFIG["init_method"] = "tikhonov_direct"


def _build_observation(
    generator: TheoreticalDataGenerator,
    g_clean: torch.Tensor,
    scenario: Dict[str, object],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(seed))
    generator.noise_mode = str(scenario["noise_mode"])
    generator.noise_level = float(scenario["noise_level"])
    if float(scenario["noise_level"]) == 0.0:
        observed = g_clean.clone()
        noise_norm = torch.zeros(1, device=g_clean.device, dtype=g_clean.dtype)
    else:
        observed = generator._apply_noise(g_clean)
        noise_norm = torch.norm(observed - g_clean, dim=-1)
    return observed, noise_norm


def _plot_scenario(
    coeff_true: np.ndarray,
    coeff_init: np.ndarray,
    scenario: Dict[str, object],
    result: Dict[str, object],
    save_path: str,
) -> None:
    diff = coeff_init - coeff_true
    vmin = min(float(np.min(coeff_true)), float(np.min(coeff_init)))
    vmax = max(float(np.max(coeff_true)), float(np.max(coeff_init)))
    dv = float(np.max(np.abs(diff))) if diff.size else 1.0
    if dv <= 0.0:
        dv = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(coeff_true, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("True Shepp-Logan")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(coeff_init, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(
        "Morozov Init\n"
        f"noise={scenario['display_noise']}, level={scenario['display_level']}\n"
        f"lambda={result['lambda']:.4e}, coeff_RES={result['coeff_res']:.4f}"
    )
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, cmap="coolwarm", origin="lower", vmin=-dv, vmax=dv)
    axes[2].set_title(
        "Difference\n"
        f"noise_norm={result['noise_norm']:.4f}\n"
        f"meas_res={result['measurement_residual']:.4f}"
    )
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.suptitle(
        f"Morozov Direct Init | 128x128 | shepp_logan | "
        f"noise={scenario['display_noise']} | level={scenario['display_level']}",
        y=1.08,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_results_text(
    results: List[Dict[str, object]],
    output_prefix: str,
) -> str:
    lines = [
        f"Morozov Direct Init Evaluation | prefix={output_prefix}",
        "Configuration: data_source=shepp_logan, image_size=128x128, lambda_select_mode=morozov",
        "=" * 96,
        f"{'Scenario':<12} | {'Noise':<14} | {'Level':>8} | {'Lambda':>14} | {'NoiseNorm':>12} | {'MeasRes':>12} | {'CoeffRES':>10}",
        "-" * 96,
    ]
    for item in results:
        lines.append(
            f"{str(item['name']):<12} | "
            f"{str(item['noise_mode']):<14} | "
            f"{float(item['noise_level']):>8.4f} | "
            f"{float(item['lambda']):>14.6e} | "
            f"{float(item['noise_norm']):>12.6f} | "
            f"{float(item['measurement_residual']):>12.6f} | "
            f"{float(item['coeff_res']):>10.6f}"
        )
        lines.append(
            f"  timing: lambda_seconds={float(item['lambda_seconds']):.6f}, "
            f"solve_seconds={float(item['solve_seconds']):.6f}, "
            f"cache_hit={item['cache_hit']}, cache_build_seconds={item['cache_build_seconds']}"
        )
        split_stats = item.get("split_admm_stats", None)
        if isinstance(split_stats, dict):
            lines.append(
                f"  split_admm: iterations={split_stats.get('iterations')}, "
                f"max_iter={split_stats.get('max_iter')}, "
                f"converged={split_stats.get('converged')}, "
                f"rho={split_stats.get('rho')}, "
                f"primal={split_stats.get('primal_residual')}, "
                f"dual={split_stats.get('dual_residual')}"
            )
    return "\n".join(lines) + "\n"


def _load_baseline_thresholds(results_path: str) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    with open(results_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("Morozov") or line.startswith("Configuration"):
                continue
            if line.startswith("=") or line.startswith("-") or line.startswith("timing:") or line.startswith("timing"):
                continue
            if "|" not in line:
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 7:
                continue
            scenario = parts[0]
            try:
                coeff_res = float(parts[6])
            except ValueError:
                continue
            thresholds[scenario] = coeff_res
    if not thresholds:
        raise ValueError(f"No baseline thresholds found in {results_path!r}.")
    return thresholds


def _check_against_baseline(
    results: List[Dict[str, object]],
    baseline_thresholds: Dict[str, float],
    *,
    atol: float = 1.0e-6,
) -> List[str]:
    failures: List[str] = []
    for item in results:
        scenario = str(item["name"])
        if scenario not in baseline_thresholds:
            failures.append(f"baseline missing scenario={scenario}")
            continue
        coeff_res = float(item["coeff_res"])
        threshold = float(baseline_thresholds[scenario])
        if coeff_res > (threshold + float(atol)):
            failures.append(
                f"{scenario}: coeff_res={coeff_res:.6f} exceeds baseline={threshold:.6f}"
            )
    return failures


def evaluate(
    scenario: str,
    base_seed: int,
    output_prefix: str,
    output_dir: str,
    cache_dir: str,
    enforce_baseline: bool = False,
    baseline_results_path: str = DEFAULT_BASELINE_RESULTS_PATH,
) -> List[Dict[str, object]]:
    _configure_morozov(cache_dir=cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    generator = TheoreticalDataGenerator(data_source="shepp_logan")
    coeff_true = generator._sample_coefficients(batch_size=1)
    g_clean = generator.forward_operator(coeff_true).to(torch.float32)
    coeff_true_2d = coeff_true.squeeze(0).squeeze(0)

    results: List[Dict[str, object]] = []
    scenarios = _select_scenarios(scenario)
    baseline_thresholds = None
    if enforce_baseline:
        baseline_thresholds = _load_baseline_thresholds(baseline_results_path)
        print(f"Loaded baseline thresholds from: {baseline_results_path}")
    print(f"Running Morozov direct-init evaluation on device: {device}")
    print("Configuration: data_source=shepp_logan, image_size=128x128, lambda_select_mode=morozov")

    for idx, item in enumerate(scenarios):
        observed, noise_norm = _build_observation(
            generator=generator,
            g_clean=g_clean,
            scenario=item,
            seed=int(base_seed + idx),
        )

        t0 = time.perf_counter()
        lam = generator.time_operator.choose_lambda_morozov(
            observed,
            noise_norm=noise_norm,
            tau=float(DATA_CONFIG["morozov_tau"]),
            max_iter=int(DATA_CONFIG["morozov_max_iter"]),
            lambda_min=float(DATA_CONFIG["morozov_lambda_min"]),
            lambda_max=float(DATA_CONFIG["morozov_lambda_max"]),
        )
        t_lambda = time.perf_counter()
        coeff_init = generator.solve_tikhonov_direct_init(
            observed,
            lambda_reg=float(lam[0].item()),
        )
        t_solve = time.perf_counter()

        meas_res = torch.norm(generator.forward_operator(coeff_init) - observed, dim=-1)
        result = {
            "name": str(item["name"]),
            "noise_mode": str(item["noise_mode"]),
            "noise_level": float(item["noise_level"]),
            "lambda": float(lam[0].item()),
            "noise_norm": float(noise_norm[0].item()),
            "measurement_residual": float(meas_res[0].item()),
            "coeff_res": _coeff_res(coeff_init.squeeze(0).squeeze(0), coeff_true_2d),
            "lambda_seconds": float(t_lambda - t0),
            "solve_seconds": float(t_solve - t_lambda),
            "cache_hit": getattr(generator.time_operator, "last_morozov_cache_hit", None),
            "cache_build_seconds": getattr(generator.time_operator, "last_morozov_cache_build_seconds", None),
            "split_admm_stats": getattr(generator.time_operator, "last_split_admm_stats", None),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

        plot_path = os.path.join(output_dir, f"{output_prefix}_{item['name']}.png")
        _plot_scenario(
            coeff_true=coeff_true_2d.detach().cpu().numpy(),
            coeff_init=coeff_init.squeeze(0).squeeze(0).detach().cpu().numpy(),
            scenario=item,
            result=result,
            save_path=plot_path,
        )
        print(f"Saved figure: {plot_path}")

    txt_path = os.path.join(output_dir, f"{output_prefix}_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_format_results_text(results, output_prefix=output_prefix))
    print(f"Saved summary: {txt_path}")
    if enforce_baseline:
        failures = _check_against_baseline(results, baseline_thresholds)
        if failures:
            raise RuntimeError(
                "Baseline check failed:\n" + "\n".join(failures)
            )
        print("Baseline check passed.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Morozov-selected direct Tikhonov initialization on 128x128 shepp_logan."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "clean", "mult_0_01", "mult_0_1"],
        help="Which noise scenario to evaluate.",
    )
    parser.add_argument("--base_seed", type=int, default=1234, help="Base seed used for noisy scenarios.")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="morozov_init_eval_remote",
        help="Prefix for generated txt and png files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to store generated json and png files.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join(os.path.dirname(RESULTS_DIR), "data", "morozov_cache"),
        help="Cache directory for Morozov exact-spectrum files.",
    )
    parser.add_argument(
        "--enforce_baseline",
        action="store_true",
        help="Fail if current coeff_res is worse than the baseline results file.",
    )
    parser.add_argument(
        "--baseline_results_path",
        type=str,
        default=DEFAULT_BASELINE_RESULTS_PATH,
        help="Baseline txt file used by --enforce_baseline.",
    )
    args = parser.parse_args()
    evaluate(
        scenario=args.scenario,
        base_seed=args.base_seed,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        enforce_baseline=bool(args.enforce_baseline),
        baseline_results_path=args.baseline_results_path,
    )


if __name__ == "__main__":
    main()
