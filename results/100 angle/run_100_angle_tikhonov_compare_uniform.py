"""Run pure-Tikhonov comparison experiments for multiple multi-angle settings (uniform injective variant)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

for _path in (str(MODELS_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import (  # noqa: E402
    BACKBONE_MULTI8_BETA_VECTORS,
    DATA_CONFIG,
    IMAGE_SIZE,
    TIME_DOMAIN_CONFIG,
    _injective_boundary_beta_candidates,
    device,
)
from image_generator import generate_shepp_logan_phantom  # noqa: E402
from radon_transform import (  # noqa: E402
    ImplicitPixelRadonOperator2D,
    StructuredMultiAngleB1B1Operator2D,
    TheoreticalB1B1Operator2D,
    _sample_uniform_extra_beta_vectors,
)


DEFAULT_NOISE_LEVELS_100 = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
DEFAULT_NOISE_LEVELS_STRUCTURED16 = [0.1]
DEFAULT_NUM_SEEDS_100 = 5
DEFAULT_NUM_SEEDS_STRUCTURED16 = 1


def _beta_orientation_theta_signed(beta: tuple[int, int]) -> float:
    a = int(beta[0])
    b = int(beta[1])
    theta = math.atan2(float(b), float(a))
    if theta < 0.0:
        theta += 2.0 * math.pi
    return float(theta)


def _angular_distance_mod_2pi(theta_a: float, theta_b: float) -> float:
    dist = abs(float(theta_a) - float(theta_b))
    return min(dist, (2.0 * math.pi) - dist)


def _select_uniform_beta_vectors(
    *,
    image_size: int,
    count: int,
    excluded_betas=None,
) -> list[tuple[int, int]]:
    count = int(count)
    if count <= 0:
        return []

    excluded = {tuple(int(v) for v in beta) for beta in list(excluded_betas or [])}
    pool = [
        tuple(int(v) for v in beta)
        for beta in _injective_boundary_beta_candidates(int(image_size))
        if tuple(int(v) for v in beta) not in excluded
    ]
    if len(pool) < count:
        raise ValueError(
            f"Not enough injective beta candidates for count={count}; only {len(pool)} available after exclusions."
        )

    anchors = list(excluded)
    selected: list[tuple[int, int]] = []
    while len(selected) < count:
        best_beta = None
        best_score = None
        active = anchors + selected
        for beta in pool:
            if beta in selected:
                continue
            theta = _beta_orientation_theta_signed(beta)
            if active:
                min_dist = min(
                    _angular_distance_mod_2pi(theta, _beta_orientation_theta_signed(other)) for other in active
                )
                score = (min_dist, -abs(theta - (0.5 * math.pi)), -abs(beta[1]), -abs(beta[0]), beta[0], beta[1])
            else:
                score = (-abs(theta - math.pi), -abs(beta[1]), -abs(beta[0]), beta[0], beta[1])
            if best_score is None or score > best_score:
                best_score = score
                best_beta = beta
        if best_beta is None:
            raise RuntimeError("Failed to select uniform injective beta vectors.")
        selected.append(best_beta)
    return selected


def _coeff_res(coeff_est: torch.Tensor, coeff_true: torch.Tensor) -> float:
    coeff_est = coeff_est.to(dtype=torch.float32)
    coeff_true = coeff_true.to(dtype=torch.float32, device=coeff_est.device)
    diff_norm = torch.norm(coeff_est - coeff_true)
    true_norm = torch.norm(coeff_true).clamp_min(1.0e-12)
    return float((diff_norm / true_norm).item())


def _measurement_residual(operator: torch.nn.Module, coeff_est: torch.Tensor, g_obs: torch.Tensor) -> float:
    pred = operator.forward(coeff_est.to(device=device, dtype=torch.float32))
    return float(torch.norm(pred - g_obs.to(device=device, dtype=pred.dtype), dim=-1).mean().item())


def _mean_std_min_max(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _apply_multiplicative_noise(g_clean: torch.Tensor, noise_level: float) -> tuple[torch.Tensor, torch.Tensor]:
    if float(noise_level) <= 0.0:
        noise = torch.zeros_like(g_clean)
        return g_clean.clone(), torch.norm(noise, dim=-1)
    rand_u = 2.0 * torch.rand_like(g_clean) - 1.0
    noise = float(noise_level) * g_clean * rand_u
    g_obs = g_clean + noise
    noise_norm = torch.norm(noise, dim=-1)
    return g_obs, noise_norm


@contextmanager
def _temporary_solver_mode(solver_mode: str):
    old = TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", None)
    TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = str(solver_mode)
    try:
        yield
    finally:
        if old is None:
            TIME_DOMAIN_CONFIG.pop("multi_angle_solver_mode", None)
        else:
            TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = old


def _build_coeff_true() -> torch.Tensor:
    phantom = generate_shepp_logan_phantom(
        image_size=IMAGE_SIZE,
        modified=True,
        device=device,
        dtype=torch.float32,
    )
    return phantom.view(1, 1, IMAGE_SIZE, IMAGE_SIZE)


def _build_structured_operator(solver_mode: str, total_angles: int, angle_seed: int) -> StructuredMultiAngleB1B1Operator2D:
    backbone = [tuple(beta) for beta in BACKBONE_MULTI8_BETA_VECTORS]
    extra = _sample_uniform_extra_beta_vectors(
        backbone_betas=backbone,
        extra_count=int(total_angles - len(backbone)),
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        seed=int(angle_seed),
    )
    formula_mode = "legacy" if str(solver_mode).strip().lower() == "stacked_tikhonov" else "shifted_support"
    return StructuredMultiAngleB1B1Operator2D(
        backbone_beta_vectors=backbone,
        extra_beta_vectors=extra,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        t0=float(TIME_DOMAIN_CONFIG.get("sampling_t0", 0.5)),
        formula_mode=formula_mode,
        auto_shift_t0=bool(TIME_DOMAIN_CONFIG.get("auto_angle_t0", True)),
    ).to(device)


def _build_random_implicit_operator(total_angles: int, angle_seed: int) -> ImplicitPixelRadonOperator2D:
    betas = _sample_uniform_extra_beta_vectors(
        backbone_betas=[],
        extra_count=int(total_angles),
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        seed=int(angle_seed),
    )
    return ImplicitPixelRadonOperator2D(
        beta_vectors=betas,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_detector_samples_per_angle=IMAGE_SIZE * IMAGE_SIZE,
    ).to(device)


def _build_structured16_random_extra_operator(
    angle_seed: int,
) -> tuple[StructuredMultiAngleB1B1Operator2D, list[tuple[int, int]], list[tuple[int, int]]]:
    backbone = [tuple(beta) for beta in BACKBONE_MULTI8_BETA_VECTORS]
    extra = _sample_uniform_extra_beta_vectors(
        backbone_betas=backbone,
        extra_count=8,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        seed=int(angle_seed),
    )
    operator = StructuredMultiAngleB1B1Operator2D(
        backbone_beta_vectors=backbone,
        extra_beta_vectors=extra,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        t0=float(TIME_DOMAIN_CONFIG.get("sampling_t0", 0.5)),
        formula_mode="shifted_support",
        auto_shift_t0=bool(TIME_DOMAIN_CONFIG.get("auto_angle_t0", True)),
    ).to(device)
    return operator, backbone, extra


def _build_structured16_injective_extra_operator() -> tuple[TheoreticalB1B1Operator2D, list[tuple[int, int]]]:
    backbone = [tuple(beta) for beta in BACKBONE_MULTI8_BETA_VECTORS]
    extra = _select_uniform_beta_vectors(
        image_size=IMAGE_SIZE,
        count=8,
        excluded_betas=backbone,
    )
    beta_vectors = list(backbone) + list(extra)
    operator = TheoreticalB1B1Operator2D(
        beta_vectors=beta_vectors,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        t0=float(TIME_DOMAIN_CONFIG.get("sampling_t0", 0.5)),
        formula_mode="shifted_support",
        auto_shift_t0=bool(TIME_DOMAIN_CONFIG.get("auto_angle_t0", True)),
    ).to(device)
    return operator, beta_vectors


def _build_injective16_full_triangular_operator() -> tuple[TheoreticalB1B1Operator2D, list[tuple[int, int]]]:
    beta_vectors = _select_uniform_beta_vectors(
        image_size=IMAGE_SIZE,
        count=16,
        excluded_betas=None,
    )
    operator = TheoreticalB1B1Operator2D(
        beta_vectors=beta_vectors,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        t0=float(TIME_DOMAIN_CONFIG.get("sampling_t0", 0.5)),
        formula_mode="shifted_support",
        auto_shift_t0=bool(TIME_DOMAIN_CONFIG.get("auto_angle_t0", True)),
    ).to(device)
    return operator, beta_vectors


def _evaluate_operator(
    *,
    label: str,
    operator: torch.nn.Module,
    coeff_true: torch.Tensor,
    noise_level: float,
    noise_seed: int,
    solver_mode: str | None,
    keep_coeff_est: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(int(noise_seed))
    np.random.seed(int(noise_seed))
    g_clean = operator.forward(coeff_true)
    g_obs, noise_norm = _apply_multiplicative_noise(g_clean, noise_level=float(noise_level))

    tau = float(DATA_CONFIG.get("morozov_tau", 1.0))
    max_iter = int(DATA_CONFIG.get("morozov_max_iter", 8))
    lambda_min = float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12))
    lambda_max = float(DATA_CONFIG.get("morozov_lambda_max", 1.0e12))

    if solver_mode is None:
        t0 = time.perf_counter()
        lam = operator.choose_lambda_morozov(
            g_obs,
            noise_norm=noise_norm,
            tau=tau,
            max_iter=max_iter,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )
        lambda_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        coeff_est = operator.solve_tikhonov_direct(g_obs, lambda_reg=lam)
        solve_seconds = time.perf_counter() - t1
    else:
        with _temporary_solver_mode(solver_mode):
            t0 = time.perf_counter()
            lam = operator.choose_lambda_morozov(
                g_obs,
                noise_norm=noise_norm,
                tau=tau,
                max_iter=max_iter,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
            )
            lambda_seconds = time.perf_counter() - t0

            t1 = time.perf_counter()
            coeff_est = operator.solve_tikhonov_direct(g_obs, lambda_reg=lam)
            solve_seconds = time.perf_counter() - t1

    result: dict[str, Any] = {
        "name": str(label),
        "noise_mode": "multiplicative",
        "noise_level": float(noise_level),
        "noise_seed": int(noise_seed),
        "lambda": float(lam.view(-1)[0].item()),
        "noise_norm": float(noise_norm.view(-1)[0].item()),
        "measurement_residual": _measurement_residual(operator, coeff_est, g_obs),
        "coeff_res": _coeff_res(coeff_est, coeff_true),
        "lambda_seconds": float(lambda_seconds),
        "solve_seconds": float(solve_seconds),
    }
    split_stats = getattr(operator, "last_split_admm_stats", None)
    if isinstance(split_stats, dict):
        result["split_admm_stats"] = {
            key: (value.item() if torch.is_tensor(value) else value) for key, value in split_stats.items()
        }
    if keep_coeff_est:
        result["_coeff_est_cpu"] = coeff_est.detach().to(device="cpu", dtype=torch.float32)
    return result


def _save_structured16_comparison_plot(
    *,
    coeff_true: torch.Tensor,
    plot_cases: dict[str, dict[str, Any]],
    output_path: str,
    noise_level: float,
    noise_seed: int,
    split_max_iter: int,
) -> None:
    ordered_names = [
        "baseline_structured16_random_extra",
        "structured16_injective_extra",
        "injective16_full_triangular",
    ]
    display_names = {
        "baseline_structured16_random_extra": "Baseline 8+8 Random",
        "structured16_injective_extra": "A: 8 Fixed + 8 Injective (uniform)",
        "injective16_full_triangular": "B: 16 Injective (uniform)",
    }
    coeff_true_np = coeff_true.squeeze().detach().cpu().numpy()
    recon_arrays = []
    for name in ordered_names:
        case = plot_cases.get(name)
        if not isinstance(case, dict) or "coeff_est_cpu" not in case:
            raise ValueError(f"Missing plot case for {name!r}.")
        recon_arrays.append(case["coeff_est_cpu"].squeeze().detach().cpu().numpy())

    vmin = min(float(np.min(coeff_true_np)), *(float(np.min(arr)) for arr in recon_arrays))
    vmax = max(float(np.max(coeff_true_np)), *(float(np.max(arr)) for arr in recon_arrays))
    extent = (0.0, float(IMAGE_SIZE - 1), 0.0, float(IMAGE_SIZE - 1))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    im0 = axes[0].imshow(coeff_true_np, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title("True Coeff")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    for ax, name, coeff_np in zip(axes[1:], ordered_names, recon_arrays):
        case = plot_cases[name]
        image = ax.imshow(coeff_np, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(
            f"{display_names[name]}\n"
            f"RES={float(case['coeff_res']):.4f}, lambda={float(case['lambda']):.2e}"
        )
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.suptitle(
        f"Structured16 Pure Tikhonov | multiplicative delta={float(noise_level):g} | "
        f"noise seed={int(noise_seed)} | split_admm_max_iter={int(split_max_iter)}",
        y=1.03,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _summarize_results(results: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for item in results:
        key = (str(item["name"]), float(item["noise_level"]))
        grouped.setdefault(key, []).append(item)

    summary: dict[tuple[str, float], dict[str, Any]] = {}
    for key, items in grouped.items():
        summary[key] = {
            "count": int(len(items)),
            "coeff_res": _mean_std_min_max([float(item["coeff_res"]) for item in items]),
            "lambda": _mean_std_min_max([float(item["lambda"]) for item in items]),
            "noise_norm": _mean_std_min_max([float(item["noise_norm"]) for item in items]),
            "measurement_residual": _mean_std_min_max([float(item["measurement_residual"]) for item in items]),
            "lambda_seconds": _mean_std_min_max([float(item["lambda_seconds"]) for item in items]),
            "solve_seconds": _mean_std_min_max([float(item["solve_seconds"]) for item in items]),
        }
        split_items = [item for item in items if isinstance(item.get("split_admm_stats"), dict)]
        if split_items:
            summary[key]["split_admm"] = {
                "iterations": _mean_std_min_max(
                    [float(item["split_admm_stats"].get("iterations", 0.0)) for item in split_items]
                ),
                "converged_count": int(
                    sum(1 for item in split_items if bool(item["split_admm_stats"].get("converged", False)))
                ),
            }
    return summary


def _format_markdown(
    *,
    results: list[dict[str, Any]],
    summary: dict[tuple[str, float], dict[str, Any]],
    total_angles: int,
    angle_seed: int,
    noise_levels: list[float],
    num_seeds: int,
    base_seed: int,
) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(str(item["name"]), []).append(item)

    ordered_names = [
        "fixed8_plus_random92_stacked",
        "fixed8_plus_random92_split",
        "random100_implicit_direct",
    ]

    lines = [
        "# 100角度纯 Tikhonov 对比实验结果",
        "",
        "## 1. 实验设置",
        "",
        f"- 总角度数：`{int(total_angles)}`",
        f"- 额外角度随机种子：`{int(angle_seed)}`",
        f"- 噪声基准种子：`{int(base_seed)}`",
        f"- 每个噪声水平的 seed 数：`{int(num_seeds)}`",
        "- 数据源：`shepp_logan`",
        "- 噪声模型：`multiplicative`",
        "- 正则参数：`Morozov`",
        f"- 噪声水平：`{' / '.join(f'{x:g}' for x in noise_levels)}`",
        "",
        "## 2. CoeffRES 均值对比",
        "",
        "| 噪声水平 | 固定8+92随机 `stacked_tikhonov` | 固定8+92随机 `split_triangular_admm` | 纯随机100 `implicit + direct` |",
        "| --- | ---: | ---: | ---: |",
    ]
    for level in noise_levels:
        s1 = summary[("fixed8_plus_random92_stacked", float(level))]["coeff_res"]
        s2 = summary[("fixed8_plus_random92_split", float(level))]["coeff_res"]
        s3 = summary[("random100_implicit_direct", float(level))]["coeff_res"]
        lines.append(
            f"| `{level:g}` | "
            f"`{s1['mean']:.6f} ± {s1['std']:.6f}` | "
            f"`{s2['mean']:.6f} ± {s2['std']:.6f}` | "
            f"`{s3['mean']:.6f} ± {s3['std']:.6f}` |"
        )

    lines.extend(
        [
            "",
            "## 3. 汇总统计",
            "",
            f"共 `3 x {len(noise_levels)} x {int(num_seeds)} = {len(results)}` 条原始记录。",
            "",
            "| 实验 | 噪声水平 | Lambda(mean±std) | NoiseNorm(mean±std) | MeasRes(mean±std) | CoeffRES(mean±std) | CoeffRES[min,max] | 备注 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name in ordered_names:
        levels_for_name = sorted({float(item["noise_level"]) for item in grouped.get(name, [])}, reverse=True)
        for level in levels_for_name:
            stats = summary[(name, level)]
            remark = ""
            split_stats = stats.get("split_admm", None)
            if isinstance(split_stats, dict):
                remark = (
                    f"iter_mean={split_stats['iterations']['mean']:.1f}, "
                    f"converged={split_stats['converged_count']}/{stats['count']}"
                )
            lines.append(
                f"| `{name}` | `{level:g}` | "
                f"`{stats['lambda']['mean']:.6e} ± {stats['lambda']['std']:.6e}` | "
                f"`{stats['noise_norm']['mean']:.6f} ± {stats['noise_norm']['std']:.6f}` | "
                f"`{stats['measurement_residual']['mean']:.6f} ± {stats['measurement_residual']['std']:.6f}` | "
                f"`{stats['coeff_res']['mean']:.6f} ± {stats['coeff_res']['std']:.6f}` | "
                f"`[{stats['coeff_res']['min']:.6f}, {stats['coeff_res']['max']:.6f}]` | {remark} |"
            )

    lines.extend(
        [
            "",
            "## 4. 原始明细",
            "",
            "| 实验 | 噪声水平 | Seed | Lambda | NoiseNorm | MeasRes | CoeffRES | 选参耗时(s) | 求解耗时(s) | 备注 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name in ordered_names:
        ordered_items = sorted(
            grouped.get(name, []),
            key=lambda x: (float(x["noise_level"]), int(x["noise_seed"])),
            reverse=True,
        )
        for item in ordered_items:
            remark = ""
            split_stats = item.get("split_admm_stats", None)
            if isinstance(split_stats, dict):
                remark = (
                    f"iter={split_stats.get('iterations')}, "
                    f"max_iter={split_stats.get('max_iter')}, "
                    f"converged={split_stats.get('converged')}"
                )
            lines.append(
                f"| `{name}` | `{float(item['noise_level']):g}` | `{int(item['noise_seed'])}` | "
                f"`{float(item['lambda']):.6e}` | `{float(item['noise_norm']):.6f}` | "
                f"`{float(item['measurement_residual']):.6f}` | `{float(item['coeff_res']):.6f}` | "
                f"`{float(item['lambda_seconds']):.3f}` | `{float(item['solve_seconds']):.3f}` | {remark} |"
            )
    return "\n".join(lines) + "\n"


def _format_beta_lines(beta_vectors: list[tuple[int, int]]) -> list[str]:
    return [f"  - {tuple(int(v) for v in beta)}" for beta in beta_vectors]


def _format_markdown_structured16_three_way(
    *,
    results: list[dict[str, Any]],
    summary: dict[tuple[str, float], dict[str, Any]],
    old_random_seed: int,
    noise_levels: list[float],
    num_seeds: int,
    base_seed: int,
    method_beta_vectors: dict[str, list[tuple[int, int]]],
) -> str:
    ordered_names = [
        "baseline_structured16_random_extra",
        "structured16_injective_extra",
        "injective16_full_triangular",
    ]
    name_to_display = {
        "baseline_structured16_random_extra": "旧基线：8特定+8随机",
        "structured16_injective_extra": "实验A：8特定+8自动单射（均匀）",
        "injective16_full_triangular": "实验B：16自动单射（均匀）",
    }
    name_to_solver = {
        "baseline_structured16_random_extra": "structured_backbone_extra + split_triangular_admm",
        "structured16_injective_extra": "full_triangular + split_triangular_admm",
        "injective16_full_triangular": "full_triangular + split_triangular_admm",
    }

    lines = [
        "# 16角度三方法纯 Tikhonov 对比实验结果",
        "",
        "## 1. 实验设置",
        "",
        "- 总角度数：`16`",
        f"- 旧基线额外随机角度种子：`{int(old_random_seed)}`",
        f"- 噪声基准种子：`{int(base_seed)}`",
        f"- 每个噪声水平的 seed 数：`{int(num_seeds)}`",
        "- 数据源：`shepp_logan`",
        "- 噪声模型：`multiplicative`",
        "- 正则参数：`Morozov`",
        "- 初始化/求解：`tikhonov_direct + split_triangular_admm`",
        f"- 噪声水平：`{' / '.join(f'{x:g}' for x in noise_levels)}`",
        "",
        "## 2. 三种方法说明",
        "",
        "- 旧基线：`8` 个特定骨干角 + `8` 个旧随机角；求解路径仍是 `structured_backbone_extra`。",
        "- 实验A：`8` 个特定骨干角 + `8` 个满足单射条件的自动角；`16` 角全部下三角化。",
        "- 实验B：`16` 个满足单射条件的自动角；`16` 角全部下三角化。",
        "",
        "## 3. 角度列表",
        "",
    ]
    for idx, name in enumerate(ordered_names, start=1):
        lines.append(f"### 3.{idx} {name_to_display[name]}")
        lines.append("")
        lines.append(f"- machine_name: `{name}`")
        lines.append(f"- solver_path: `{name_to_solver[name]}`")
        lines.append("")
        lines.extend(_format_beta_lines(method_beta_vectors[name]))
        lines.append("")

    lines.extend(
        [
            "## 4. CoeffRES 均值对比",
            "",
            "| 噪声水平 | 旧基线：8特定+8随机 | 实验A：8特定+8自动单射 | 实验B：16自动单射 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for level in noise_levels:
        s0 = summary[("baseline_structured16_random_extra", float(level))]["coeff_res"]
        s1 = summary[("structured16_injective_extra", float(level))]["coeff_res"]
        s2 = summary[("injective16_full_triangular", float(level))]["coeff_res"]
        lines.append(
            f"| `{level:g}` | "
            f"`{s0['mean']:.6f} ± {s0['std']:.6f}` | "
            f"`{s1['mean']:.6f} ± {s1['std']:.6f}` | "
            f"`{s2['mean']:.6f} ± {s2['std']:.6f}` |"
        )

    lines.extend(
        [
            "",
            "## 5. Lambda 均值对比",
            "",
            "| 噪声水平 | 旧基线：8特定+8随机 | 实验A：8特定+8自动单射 | 实验B：16自动单射 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for level in noise_levels:
        s0 = summary[("baseline_structured16_random_extra", float(level))]["lambda"]
        s1 = summary[("structured16_injective_extra", float(level))]["lambda"]
        s2 = summary[("injective16_full_triangular", float(level))]["lambda"]
        lines.append(
            f"| `{level:g}` | "
            f"`{s0['mean']:.6e} ± {s0['std']:.6e}` | "
            f"`{s1['mean']:.6e} ± {s1['std']:.6e}` | "
            f"`{s2['mean']:.6e} ± {s2['std']:.6e}` |"
        )

    lines.extend(
        [
            "",
            "## 6. 汇总统计",
            "",
            "| 实验 | 噪声水平 | Lambda(mean±std) | MeasRes(mean±std) | CoeffRES(mean±std) | SolveSeconds(mean±std) | ADMM备注 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name in ordered_names:
        for level in noise_levels:
            stats = summary[(name, float(level))]
            remark = ""
            split_stats = stats.get("split_admm", None)
            if isinstance(split_stats, dict):
                remark = (
                    f"iter_mean={split_stats['iterations']['mean']:.1f}, "
                    f"converged={split_stats['converged_count']}/{stats['count']}"
                )
            lines.append(
                f"| `{name}` | `{level:g}` | "
                f"`{stats['lambda']['mean']:.6e} ± {stats['lambda']['std']:.6e}` | "
                f"`{stats['measurement_residual']['mean']:.6f} ± {stats['measurement_residual']['std']:.6f}` | "
                f"`{stats['coeff_res']['mean']:.6f} ± {stats['coeff_res']['std']:.6f}` | "
                f"`{stats['solve_seconds']['mean']:.3f} ± {stats['solve_seconds']['std']:.3f}` | "
                f"{remark} |"
            )

    lines.extend(
        [
            "",
            "## 7. 原始明细",
            "",
            "| 实验 | 噪声水平 | Seed | Lambda | NoiseNorm | MeasRes | CoeffRES | 选参耗时(s) | 求解耗时(s) | 备注 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(str(item["name"]), []).append(item)
    for name in ordered_names:
        ordered_items = sorted(
            grouped.get(name, []),
            key=lambda x: (float(x["noise_level"]), int(x["noise_seed"])),
            reverse=True,
        )
        for item in ordered_items:
            remark = ""
            split_stats = item.get("split_admm_stats", None)
            if isinstance(split_stats, dict):
                remark = (
                    f"iter={split_stats.get('iterations')}, "
                    f"max_iter={split_stats.get('max_iter')}, "
                    f"converged={split_stats.get('converged')}"
                )
            lines.append(
                f"| `{name}` | `{float(item['noise_level']):g}` | `{int(item['noise_seed'])}` | "
                f"`{float(item['lambda']):.6e}` | `{float(item['noise_norm']):.6f}` | "
                f"`{float(item['measurement_residual']):.6f}` | `{float(item['coeff_res']):.6f}` | "
                f"`{float(item['lambda_seconds']):.3f}` | `{float(item['solve_seconds']):.3f}` | {remark} |"
            )
    return "\n".join(lines) + "\n"


def _run_experiments(
    noise_levels: list[float],
    total_angles: int,
    angle_seed: int,
    base_seed: int,
    num_seeds: int,
) -> list[dict[str, Any]]:
    coeff_true = _build_coeff_true()

    structured_stacked = _build_structured_operator(
        solver_mode="stacked_tikhonov",
        total_angles=total_angles,
        angle_seed=angle_seed,
    )
    structured_split = _build_structured_operator(
        solver_mode="split_triangular_admm",
        total_angles=total_angles,
        angle_seed=angle_seed,
    )
    random_implicit = _build_random_implicit_operator(total_angles=total_angles, angle_seed=angle_seed)

    experiments = [
        ("fixed8_plus_random92_stacked", structured_stacked, "stacked_tikhonov"),
        ("fixed8_plus_random92_split", structured_split, "split_triangular_admm"),
        ("random100_implicit_direct", random_implicit, None),
    ]

    results: list[dict[str, Any]] = []
    for level_idx, level in enumerate(noise_levels):
        for seed_idx in range(int(num_seeds)):
            noise_seed = int(base_seed + (1000 * level_idx) + (100 * seed_idx))
            for label, operator, solver_mode in experiments:
                item = _evaluate_operator(
                    label=label,
                    operator=operator,
                    coeff_true=coeff_true,
                    noise_level=float(level),
                    noise_seed=noise_seed,
                    solver_mode=solver_mode,
                )
                item["seed_index"] = int(seed_idx)
                results.append(item)
    return results


def _run_structured16_three_way_experiments(
    noise_levels: list[float],
    old_random_seed: int,
    base_seed: int,
    num_seeds: int,
    plot_noise_level: float | None = None,
    plot_noise_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[int, int]]], dict[str, dict[str, Any]]]:
    coeff_true = _build_coeff_true()
    baseline_operator, baseline_backbone, baseline_extra = _build_structured16_random_extra_operator(
        angle_seed=old_random_seed
    )
    injective_extra_operator, injective_extra_betas = _build_structured16_injective_extra_operator()
    injective_full_operator, injective_full_betas = _build_injective16_full_triangular_operator()

    experiments = [
        (
            "baseline_structured16_random_extra",
            baseline_operator,
            "split_triangular_admm",
            list(baseline_backbone) + list(baseline_extra),
        ),
        (
            "structured16_injective_extra",
            injective_extra_operator,
            "split_triangular_admm",
            list(injective_extra_betas),
        ),
        (
            "injective16_full_triangular",
            injective_full_operator,
            "split_triangular_admm",
            list(injective_full_betas),
        ),
    ]

    results: list[dict[str, Any]] = []
    plot_cases: dict[str, dict[str, Any]] = {}
    beta_vectors_by_name = {str(name): list(beta_vectors) for name, _, _, beta_vectors in experiments}
    for level_idx, level in enumerate(noise_levels):
        for seed_idx in range(int(num_seeds)):
            noise_seed = int(base_seed + (1000 * level_idx) + (100 * seed_idx))
            keep_plot_case = (
                plot_noise_level is not None
                and abs(float(level) - float(plot_noise_level)) <= 1.0e-12
                and plot_noise_seed is not None
                and int(noise_seed) == int(plot_noise_seed)
            )
            for label, operator, solver_mode, _ in experiments:
                item = _evaluate_operator(
                    label=label,
                    operator=operator,
                    coeff_true=coeff_true,
                    noise_level=float(level),
                    noise_seed=noise_seed,
                    solver_mode=solver_mode,
                    keep_coeff_est=bool(keep_plot_case),
                )
                coeff_est_cpu = item.pop("_coeff_est_cpu", None)
                item["seed_index"] = int(seed_idx)
                results.append(item)
                if coeff_est_cpu is not None:
                    plot_cases[str(label)] = {**item, "coeff_est_cpu": coeff_est_cpu}
    return results, beta_vectors_by_name, plot_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pure-Tikhonov results for multiple multi-angle experiment groups.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="100angle_reference",
        choices=["100angle_reference", "structured16_three_way"],
        help="Which comparison bundle to run.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=None,
        help="Multiplicative noise levels to evaluate. If omitted, the experiment-specific defaults are used.",
    )
    parser.add_argument("--total-angles", type=int, default=100, help="Total number of angles.")
    parser.add_argument(
        "--angle-seed",
        type=int,
        default=None,
        help="Angle seed. For 100angle_reference it defaults to TIME_DOMAIN_CONFIG['extra_angle_seed']; "
        "for structured16_three_way it defaults to 20260327 to match the previous 8特定+8随机 baseline.",
    )
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for noise realizations.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of seeds per noise level. If omitted, the experiment-specific defaults are used.",
    )
    parser.add_argument(
        "--split-max-iter",
        type=int,
        default=None,
        help="Optional override for TIME_DOMAIN_CONFIG['split_admm_max_iter'] during this script.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Markdown output path. If omitted, the experiment-specific default filename is used.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="JSON output path. If omitted, the experiment-specific default filename is used.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Optional reconstruction comparison plot path. For structured16_three_way, defaults to a PNG beside the markdown output.",
    )
    parser.add_argument(
        "--plot-noise-level",
        type=float,
        default=None,
        help="Noise level used for the saved comparison plot. Defaults to the first evaluated noise level.",
    )
    parser.add_argument(
        "--plot-noise-seed",
        type=int,
        default=None,
        help="Noise seed used for the saved comparison plot. Defaults to --base-seed.",
    )
    args = parser.parse_args()

    DATA_CONFIG["lambda_select_mode"] = "morozov"
    TIME_DOMAIN_CONFIG["init_method"] = "tikhonov_direct"
    if args.split_max_iter is not None:
        if int(args.split_max_iter) <= 0:
            raise ValueError(f"split-max-iter must be positive, got {int(args.split_max_iter)}.")
        TIME_DOMAIN_CONFIG["split_admm_max_iter"] = int(args.split_max_iter)

    if args.experiment == "100angle_reference":
        total_angles = int(args.total_angles)
        if total_angles != 100:
            raise ValueError(f"100angle_reference requires total_angles=100, got {total_angles}.")
        angle_seed = int(TIME_DOMAIN_CONFIG.get("extra_angle_seed", 20260322) if args.angle_seed is None else args.angle_seed)
        noise_levels = [float(v) for v in (list(DEFAULT_NOISE_LEVELS_100) if args.noise_levels is None else args.noise_levels)]
        num_seeds = int(DEFAULT_NUM_SEEDS_100 if args.num_seeds is None else args.num_seeds)
        output_md = str(THIS_DIR / "100角度Tikhonov对比实验结果.md") if args.output_md is None else str(args.output_md)
        output_json = str(THIS_DIR / "100角度Tikhonov对比实验结果.json") if args.output_json is None else str(args.output_json)

        if num_seeds <= 0:
            raise ValueError(f"num_seeds must be positive, got {num_seeds}.")
        os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

        results = _run_experiments(
            noise_levels=noise_levels,
            total_angles=total_angles,
            angle_seed=angle_seed,
            base_seed=int(args.base_seed),
            num_seeds=num_seeds,
        )
        summary = _summarize_results(results)

        markdown = _format_markdown(
            results=results,
            summary=summary,
            total_angles=total_angles,
            angle_seed=angle_seed,
            noise_levels=noise_levels,
            num_seeds=num_seeds,
            base_seed=int(args.base_seed),
        )
        meta = {
            "experiment": str(args.experiment),
            "total_angles": total_angles,
            "angle_seed": angle_seed,
            "base_seed": int(args.base_seed),
            "num_seeds": num_seeds,
            "noise_levels": noise_levels,
            "split_admm_max_iter": int(TIME_DOMAIN_CONFIG.get("split_admm_max_iter", 0)),
        }
    else:
        total_angles = 16
        if int(args.total_angles) != 16:
            raise ValueError(
                f"structured16_three_way requires total_angles=16, got {int(args.total_angles)}."
            )
        angle_seed = int(20260327 if args.angle_seed is None else args.angle_seed)
        noise_levels = [
            float(v)
            for v in (list(DEFAULT_NOISE_LEVELS_STRUCTURED16) if args.noise_levels is None else args.noise_levels)
        ]
        num_seeds = int(DEFAULT_NUM_SEEDS_STRUCTURED16 if args.num_seeds is None else args.num_seeds)
        output_md = (
            str(THIS_DIR / "16角度三方法Tikhonov对比实验结果_uniformsigned.md")
            if args.output_md is None
            else str(args.output_md)
        )
        output_json = (
            str(THIS_DIR / "16角度三方法Tikhonov对比实验结果_uniformsigned.json")
            if args.output_json is None
            else str(args.output_json)
        )
        plot_output = (
            str(THIS_DIR / "16角度三方法Tikhonov对比实验结果_uniformsigned.png")
            if args.plot_output is None
            else str(args.plot_output)
        )
        plot_noise_level = float(noise_levels[0] if args.plot_noise_level is None else args.plot_noise_level)
        plot_noise_seed = int(args.base_seed if args.plot_noise_seed is None else args.plot_noise_seed)

        if num_seeds <= 0:
            raise ValueError(f"num_seeds must be positive, got {num_seeds}.")
        os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(plot_output)), exist_ok=True)

        results, method_beta_vectors, plot_cases = _run_structured16_three_way_experiments(
            noise_levels=noise_levels,
            old_random_seed=angle_seed,
            base_seed=int(args.base_seed),
            num_seeds=num_seeds,
            plot_noise_level=plot_noise_level,
            plot_noise_seed=plot_noise_seed,
        )
        coeff_true = _build_coeff_true()
        summary = _summarize_results(results)
        markdown = _format_markdown_structured16_three_way(
            results=results,
            summary=summary,
            old_random_seed=angle_seed,
            noise_levels=noise_levels,
            num_seeds=num_seeds,
            base_seed=int(args.base_seed),
            method_beta_vectors=method_beta_vectors,
        )
        if plot_cases:
            _save_structured16_comparison_plot(
                coeff_true=coeff_true,
                plot_cases=plot_cases,
                output_path=plot_output,
                noise_level=plot_noise_level,
                noise_seed=plot_noise_seed,
                split_max_iter=int(TIME_DOMAIN_CONFIG.get("split_admm_max_iter", 0)),
            )
        meta = {
            "experiment": str(args.experiment),
            "total_angles": total_angles,
            "old_random_seed": angle_seed,
            "base_seed": int(args.base_seed),
            "num_seeds": num_seeds,
            "noise_levels": noise_levels,
            "split_admm_max_iter": int(TIME_DOMAIN_CONFIG.get("split_admm_max_iter", 0)),
            "method_beta_vectors": {
                name: [list(beta) for beta in betas] for name, betas in method_beta_vectors.items()
            },
            "injective_selection_mode": "uniform_signed",
            "plot_output": plot_output,
            "plot_noise_level": plot_noise_level,
            "plot_noise_seed": plot_noise_seed,
        }

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "summary": [
                    {
                        "name": key[0],
                        "noise_level": key[1],
                        **stats,
                    }
                    for key, stats in sorted(summary.items(), key=lambda kv: (kv[0][0], kv[0][1]))
                ],
                "raw_results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(markdown)
    print(f"Markdown saved to: {output_md}")
    print(f"JSON saved to: {output_json}")
    if args.experiment == "structured16_three_way":
        print(f"Plot saved to: {plot_output}")


if __name__ == "__main__":
    main()
