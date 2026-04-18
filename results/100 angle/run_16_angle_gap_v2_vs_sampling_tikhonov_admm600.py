"""Compare old gap_v2 and new sampling Tikhonov results with split_triangular_admm=600.

This script intentionally keeps the experiment structure close to
``run_100_angle_tikhonov_compare_olduniform.py`` but narrows the comparison to
two per-angle lower-band constructions on the same injective 16-angle set:

1. ``gap_v2``: old empirical construction, using ``band_t0 = 0.5`` on the
   reordered beta-dot-k axis.
2. ``sampling``: current theory-aligned construction, i.e.
   ``formula_mode='legacy_injective_extension'`` in ``radon_transform.py``.

Both methods use Morozov-selected Tikhonov lambda and solve the final Tikhonov
problem through ``split_triangular_admm``.  The default ADMM iteration budget is
fixed to 600 as requested.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

for _path in (str(MODELS_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import DATA_CONFIG, IMAGE_SIZE, TIME_DOMAIN_CONFIG, device  # noqa: E402
from image_generator import generate_shepp_logan_phantom  # noqa: E402
import radon_transform as rt  # noqa: E402


DEFAULT_NOISE_LEVELS = [0.1]
DEFAULT_NUM_SEEDS = 1
DEFAULT_BASE_SEED = 42
DEFAULT_SPLIT_MAX_ITER = 600

# This is the same active 16-angle list shown by injective16_pi_best and by the
# archived old-uniform "B" experiment.
INJECTIVE16_PI_BEST_BETAS: list[tuple[int, int]] = [
    (128, 1),
    (1, -128),
    (127, 128),
    (128, -127),
    (128, -53),
    (53, 128),
    (53, -128),
    (128, 53),
    (25, 128),
    (128, -25),
    (128, -85),
    (85, 128),
    (128, 85),
    (85, -128),
    (25, -128),
    (128, 25),
]

# The archived old-uniform "A" list from the reference script.  It is useful if
# we want to compare the 8 fixed + 8 old-uniform injective-extra setting.
ARCHIVED_OLDUNIFORM_A_BETAS: list[tuple[int, int]] = [
    (1, 128),
    (-1, 128),
    (1, -128),
    (-1, -128),
    (128, 1),
    (-128, 1),
    (128, -1),
    (-128, -1),
    (127, -128),
    (127, 128),
    (53, 128),
    (53, -128),
    (128, 53),
    (128, -53),
    (128, -85),
    (85, 128),
]


def _select_beta_vectors(name: str) -> list[tuple[int, int]]:
    key = str(name).strip().lower()
    if key in {"injective16_pi_best", "olduniform_b", "b"}:
        return [tuple(beta) for beta in INJECTIVE16_PI_BEST_BETAS]
    if key in {"olduniform_a", "a"}:
        return [tuple(beta) for beta in ARCHIVED_OLDUNIFORM_A_BETAS]
    raise ValueError(
        f"Unsupported beta-set={name!r}; expected 'injective16_pi_best'/'olduniform_b' or 'olduniform_a'."
    )


def _coeff_res(coeff_est: torch.Tensor, coeff_true: torch.Tensor) -> float:
    coeff_est = coeff_est.to(dtype=torch.float32)
    coeff_true = coeff_true.to(dtype=torch.float32, device=coeff_est.device)
    return float((torch.norm(coeff_est - coeff_true) / torch.norm(coeff_true).clamp_min(1.0e-12)).item())


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
    return g_clean + noise, torch.norm(noise, dim=-1)


def _build_coeff_true() -> torch.Tensor:
    phantom = generate_shepp_logan_phantom(
        image_size=IMAGE_SIZE,
        modified=True,
        device=device,
        dtype=torch.float32,
    )
    return phantom.view(1, 1, IMAGE_SIZE, IMAGE_SIZE)


def _theoretical_b1b1_block_gap_v2(
    beta,
    height: int,
    width: int,
    t0: float,
    formula_mode: str = "legacy_injective_gap_v2",
    auto_shift_t0: bool = True,
) -> dict[str, torch.Tensor]:
    """Build the old gap_v2 lower-band construction for an injective beta.

    The important difference from the current sampling construction is:

    - gap_v2: ``band_t0 = t0``;
    - sampling: ``band_t0 = A_beta + t0``.

    Both use the same sorted ``beta·k`` gaps, so this isolates the sampling
    origin choice rather than changing the beta order or ADMM solver.
    """
    del formula_mode, auto_shift_t0
    h = int(height)
    w = int(width)
    n = int(h * w)
    beta_i = rt._to_integer_beta(beta)
    beta_f = beta_i.to(torch.float64)
    beta_norm = float(torch.norm(beta_f, p=2).item())
    alpha = beta_f / beta_norm

    k1, k2 = rt._lex_lattice_indices(h, w)
    beta_dot_k = beta_i[0] * k1 + beta_i[1] * k2
    sort_perm = torch.argsort(beta_dot_k, stable=True)
    sorted_proj = beta_dot_k.index_select(0, sort_perm).to(torch.int64)
    if int(torch.unique(sorted_proj).numel()) != n:
        raise ValueError(
            f"beta={tuple(int(x) for x in beta_i.tolist())} does not make beta·k injective "
            f"on [0,{h - 1}]x[0,{w - 1}]."
        )

    lex_to_d = torch.empty(n, dtype=torch.int64)
    lex_to_d[sort_perm] = torch.arange(n, dtype=torch.int64)
    d_to_lex = sort_perm.to(torch.int64)

    kappa0 = int(sorted_proj[0].item())
    support_lo_beta_exact, support_hi_beta_exact = rt._beta_support_bounds_b1b1(beta_i)
    support_lo_beta = float(support_lo_beta_exact)
    support_hi_beta = float(support_hi_beta_exact)

    # Old gap_v2: keep the reordered-axis sample origin at 0.5.
    effective_t0 = float(t0)
    band_t0 = float(t0)
    theory_t0_abs = float(kappa0 + effective_t0)
    sampling_points = (float(effective_t0) + sorted_proj.to(torch.float64)) / beta_norm

    max_gap = max(0, int(math.ceil(support_hi_beta - float(band_t0) + 1.0e-12)))
    band_limit = min(n, max_gap + 1)
    lower_ab = np.zeros((band_limit, n), dtype=np.float64)
    proj_np = sorted_proj.detach().to(dtype=torch.float64, device="cpu")
    for offset in range(band_limit):
        length = n - offset
        if length <= 0:
            break
        diffs = proj_np[offset:] - proj_np[:length]
        values = rt.radon_phi_b1b1((float(band_t0) + diffs) / beta_norm, alpha).to(
            dtype=torch.float64,
            device="cpu",
        )
        lower_ab[offset, :length] = values.numpy()
    lower_ab = rt._trim_lower_banded_ab(lower_ab)

    band_width = int(lower_ab.shape[0])
    r = np.zeros((n,), dtype=np.float64)
    r[:band_width] = lower_ab[:, 0]

    return {
        "r": torch.from_numpy(r),
        "lower_bands": torch.from_numpy(lower_ab),
        "sorted_proj": sorted_proj,
        "lower_bandwidth": torch.tensor(band_width, dtype=torch.int64),
        "alpha": alpha,
        "beta": beta_i,
        "sampling_points": sampling_points,
        "lex_to_d": lex_to_d,
        "d_to_lex": d_to_lex,
        "kappa0": torch.tensor(kappa0, dtype=torch.int64),
        "effective_t0": torch.tensor(float(effective_t0), dtype=torch.float64),
        "band_t0": torch.tensor(float(band_t0), dtype=torch.float64),
        "support_lo_beta": torch.tensor(float(support_lo_beta), dtype=torch.float64),
        "theory_t0_abs": torch.tensor(float(theory_t0_abs), dtype=torch.float64),
    }


def _theoretical_b1b1_block_sampling_v3(
    beta,
    height: int,
    width: int,
    t0: float,
    formula_mode: str = "legacy_injective_sampling_v3",
    auto_shift_t0: bool = True,
) -> dict[str, torch.Tensor]:
    """Build the theory-aligned sampling_v3 lower-band construction locally.

    This keeps the new sampling interface isolated inside this comparison script,
    even if the main codebase restores legacy_injective_extension to the old
    gap_v2 behavior.
    """
    del formula_mode, auto_shift_t0
    h = int(height)
    w = int(width)
    n = int(h * w)
    beta_i = rt._to_integer_beta(beta)
    beta_f = beta_i.to(torch.float64)
    beta_norm = float(torch.norm(beta_f, p=2).item())
    alpha = beta_f / beta_norm

    k1, k2 = rt._lex_lattice_indices(h, w)
    beta_dot_k = beta_i[0] * k1 + beta_i[1] * k2
    sort_perm = torch.argsort(beta_dot_k, stable=True)
    sorted_proj = beta_dot_k.index_select(0, sort_perm).to(torch.int64)
    if int(torch.unique(sorted_proj).numel()) != n:
        raise ValueError(
            f"beta={tuple(int(x) for x in beta_i.tolist())} does not make beta·k injective "
            f"on [0,{h - 1}]x[0,{w - 1}]."
        )

    lex_to_d = torch.empty(n, dtype=torch.int64)
    lex_to_d[sort_perm] = torch.arange(n, dtype=torch.int64)
    d_to_lex = sort_perm.to(torch.int64)

    kappa0 = int(sorted_proj[0].item())
    support_lo_beta_exact, support_hi_beta_exact = rt._beta_support_bounds_b1b1(beta_i)
    support_lo_beta = float(support_lo_beta_exact)
    support_hi_beta = float(support_hi_beta_exact)

    # New sampling_v3: shift the per-angle sampling origin to A_beta + 0.5.
    effective_t0 = float(support_lo_beta + float(t0))
    band_t0 = float(effective_t0)
    theory_t0_abs = float(kappa0 + effective_t0)
    sampling_points = (float(effective_t0) + sorted_proj.to(torch.float64)) / beta_norm

    max_gap = max(0, int(math.ceil(support_hi_beta - float(band_t0) + 1.0e-12)))
    band_limit = min(n, max_gap + 1)
    lower_ab = np.zeros((band_limit, n), dtype=np.float64)
    proj_np = sorted_proj.detach().to(dtype=torch.float64, device="cpu")
    for offset in range(band_limit):
        length = n - offset
        if length <= 0:
            break
        diffs = proj_np[offset:] - proj_np[:length]
        values = rt.radon_phi_b1b1((float(band_t0) + diffs) / beta_norm, alpha).to(
            dtype=torch.float64,
            device="cpu",
        )
        lower_ab[offset, :length] = values.numpy()
    lower_ab = rt._trim_lower_banded_ab(lower_ab)

    band_width = int(lower_ab.shape[0])
    r = np.zeros((n,), dtype=np.float64)
    r[:band_width] = lower_ab[:, 0]

    return {
        "r": torch.from_numpy(r),
        "lower_bands": torch.from_numpy(lower_ab),
        "sorted_proj": sorted_proj,
        "lower_bandwidth": torch.tensor(band_width, dtype=torch.int64),
        "alpha": alpha,
        "beta": beta_i,
        "sampling_points": sampling_points,
        "lex_to_d": lex_to_d,
        "d_to_lex": d_to_lex,
        "kappa0": torch.tensor(kappa0, dtype=torch.int64),
        "effective_t0": torch.tensor(float(effective_t0), dtype=torch.float64),
        "band_t0": torch.tensor(float(band_t0), dtype=torch.float64),
        "support_lo_beta": torch.tensor(float(support_lo_beta), dtype=torch.float64),
        "theory_t0_abs": torch.tensor(float(theory_t0_abs), dtype=torch.float64),
    }


@contextlib.contextmanager
def _gap_v2_block_patch() -> Iterator[None]:
    original = rt._theoretical_b1b1_block
    rt._theoretical_b1b1_block = _theoretical_b1b1_block_gap_v2
    try:
        yield
    finally:
        rt._theoretical_b1b1_block = original


@contextlib.contextmanager
def _sampling_v3_block_patch() -> Iterator[None]:
    original = rt._theoretical_b1b1_block
    rt._theoretical_b1b1_block = _theoretical_b1b1_block_sampling_v3
    try:
        yield
    finally:
        rt._theoretical_b1b1_block = original


def _build_operator(method: str, beta_vectors: list[tuple[int, int]]) -> rt.TheoreticalB1B1Operator2D:
    method_key = str(method).strip().lower()
    common_kwargs = dict(
        beta_vectors=list(beta_vectors),
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        t0=float(TIME_DOMAIN_CONFIG.get("sampling_t0", 0.5)),
        auto_shift_t0=bool(TIME_DOMAIN_CONFIG.get("auto_angle_t0", True)),
    )
    if method_key == "gap_v2":
        return rt.TheoreticalB1B1Operator2D(
            **common_kwargs,
            formula_mode="legacy_injective_extension",
        ).to(device)
    if method_key == "sampling":
        with _sampling_v3_block_patch():
            operator = rt.TheoreticalB1B1Operator2D(
                **common_kwargs,
                formula_mode="legacy_injective_sampling_v3",
            ).to(device)
        operator.formula_mode = "legacy_injective_sampling_v3"
        return operator
    raise ValueError(f"Unsupported method={method!r}; expected 'gap_v2' or 'sampling'.")


def _operator_sampling_stats(operator: rt.TheoreticalB1B1Operator2D) -> dict[str, Any]:
    diag0 = operator.lower_bands[:, 0, 0].detach().to(dtype=torch.float64, device="cpu")
    bandwidths = operator.lower_bandwidths.detach().to(dtype=torch.float64, device="cpu")
    band_t0 = operator.band_t0_per_angle.detach().to(dtype=torch.float64, device="cpu")
    effective_t0 = operator.effective_t0_per_angle.detach().to(dtype=torch.float64, device="cpu")
    return {
        "formula_mode": str(operator.formula_mode),
        "num_angles": int(operator.num_angles),
        "band_t0_per_angle": [float(v) for v in band_t0.tolist()],
        "effective_t0_per_angle": [float(v) for v in effective_t0.tolist()],
        "diag0": _mean_std_min_max([float(v) for v in diag0.tolist()]),
        "lower_bandwidth": _mean_std_min_max([float(v) for v in bandwidths.tolist()]),
    }


def _evaluate_operator(
    *,
    label: str,
    operator: rt.TheoreticalB1B1Operator2D,
    coeff_true: torch.Tensor,
    noise_level: float,
    noise_seed: int,
    split_max_iter: int,
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
    coeff_est = operator.solve_tikhonov_direct(
        g_obs,
        lambda_reg=lam,
        max_iter=int(split_max_iter),
    )
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


def _summarize_results(results: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault((str(item["name"]), float(item["noise_level"])), []).append(item)

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


def _format_beta_lines(beta_vectors: list[tuple[int, int]]) -> list[str]:
    return [f"  - {tuple(int(v) for v in beta)}" for beta in beta_vectors]


def _format_markdown(
    *,
    results: list[dict[str, Any]],
    summary: dict[tuple[str, float], dict[str, Any]],
    beta_set: str,
    beta_vectors: list[tuple[int, int]],
    operator_stats: dict[str, dict[str, Any]],
    noise_levels: list[float],
    num_seeds: int,
    base_seed: int,
    split_max_iter: int,
) -> str:
    ordered_names = ["gap_v2", "sampling"]
    lines = [
        "# gap_v2 vs sampling 纯 Tikhonov 对比（split_triangular_admm=600）",
        "",
        "## 1. 实验设置",
        "",
        f"- beta_set：`{beta_set}`",
        "- 角度数：`16`",
        f"- 噪声基准种子：`{int(base_seed)}`",
        f"- 每个噪声水平的 seed 数：`{int(num_seeds)}`",
        "- 数据源：`shepp_logan`",
        "- 噪声模型：`multiplicative`",
        "- 正则参数：`Morozov`",
        "- 求解器：`solve_tikhonov_direct -> split_triangular_admm`",
        f"- ADMM 最大迭代次数：`{int(split_max_iter)}`",
        f"- 噪声水平：`{' / '.join(f'{x:g}' for x in noise_levels)}`",
        "",
        "## 2. 方法说明",
        "",
        "- `gap_v2`：旧经验构造，固定 `band_t0=0.5`，再按 `sorted beta·k` 的相对 gap 构造 lower bands。",
        "- `sampling`：当前理论采样构造，即 `legacy_injective_extension`，使用 `band_t0=A_beta+0.5`。",
        "- 两组方法使用完全相同的 `beta` 列表、噪声、Morozov 设置和 ADMM 迭代数。",
        "",
        "## 3. 角度列表",
        "",
    ]
    lines.extend(_format_beta_lines(beta_vectors))
    lines.extend(
        [
            "",
            "## 4. lower-band 采样统计",
            "",
            "| 方法 | formula_mode | diag0(mean/min/max) | bandwidth(mean/min/max) |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for name in ordered_names:
        stats = operator_stats[name]
        diag = stats["diag0"]
        bw = stats["lower_bandwidth"]
        lines.append(
            f"| `{name}` | `{stats['formula_mode']}` | "
            f"`{diag['mean']:.6e}/{diag['min']:.6e}/{diag['max']:.6e}` | "
            f"`{bw['mean']:.2f}/{bw['min']:.0f}/{bw['max']:.0f}` |"
        )

    lines.extend(
        [
            "",
            "## 5. CoeffRES 均值对比",
            "",
            "| 噪声水平 | gap_v2 | sampling | 更优 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for level in noise_levels:
        s_gap = summary[("gap_v2", float(level))]["coeff_res"]
        s_sampling = summary[("sampling", float(level))]["coeff_res"]
        best = "gap_v2" if float(s_gap["mean"]) < float(s_sampling["mean"]) else "sampling"
        lines.append(
            f"| `{level:g}` | "
            f"`{s_gap['mean']:.6f} ± {s_gap['std']:.6f}` | "
            f"`{s_sampling['mean']:.6f} ± {s_sampling['std']:.6f}` | `{best}` |"
        )

    lines.extend(
        [
            "",
            "## 6. 汇总统计",
            "",
            "| 方法 | 噪声水平 | Lambda(mean±std) | MeasRes(mean±std) | CoeffRES(mean±std) | SolveSeconds(mean±std) | ADMM备注 |",
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
            "| 方法 | 噪声水平 | Seed | Lambda | NoiseNorm | MeasRes | CoeffRES | 选参耗时(s) | 求解耗时(s) | 备注 |",
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


def _save_plot(
    *,
    coeff_true: torch.Tensor,
    plot_cases: dict[str, dict[str, Any]],
    output_path: str,
    noise_level: float,
    noise_seed: int,
    split_max_iter: int,
) -> None:
    ordered_names = ["gap_v2", "sampling"]
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    image0 = axes[0].imshow(coeff_true_np, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title("True Coeff")
    plt.colorbar(image0, ax=axes[0], fraction=0.046, pad=0.04)

    for ax, name, coeff_np in zip(axes[1:], ordered_names, recon_arrays):
        case = plot_cases[name]
        image = ax.imshow(coeff_np, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(f"{name}\nRES={float(case['coeff_res']):.4f}, lambda={float(case['lambda']):.2e}")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.suptitle(
        f"gap_v2 vs sampling | delta={float(noise_level):g} | "
        f"seed={int(noise_seed)} | split_admm_max_iter={int(split_max_iter)}",
        y=1.03,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _run_experiments(
    *,
    beta_vectors: list[tuple[int, int]],
    noise_levels: list[float],
    base_seed: int,
    num_seeds: int,
    split_max_iter: int,
    plot_noise_level: float,
    plot_noise_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    coeff_true = _build_coeff_true()
    operators = {
        "gap_v2": _build_operator("gap_v2", beta_vectors),
        "sampling": _build_operator("sampling", beta_vectors),
    }
    operator_stats = {name: _operator_sampling_stats(operator) for name, operator in operators.items()}

    results: list[dict[str, Any]] = []
    plot_cases: dict[str, dict[str, Any]] = {}
    for level_idx, level in enumerate(noise_levels):
        for seed_idx in range(int(num_seeds)):
            noise_seed = int(base_seed + (1000 * level_idx) + (100 * seed_idx))
            keep_plot_case = (
                abs(float(level) - float(plot_noise_level)) <= 1.0e-12
                and int(noise_seed) == int(plot_noise_seed)
            )
            for label, operator in operators.items():
                item = _evaluate_operator(
                    label=label,
                    operator=operator,
                    coeff_true=coeff_true,
                    noise_level=float(level),
                    noise_seed=noise_seed,
                    split_max_iter=int(split_max_iter),
                    keep_coeff_est=bool(keep_plot_case),
                )
                coeff_est_cpu = item.pop("_coeff_est_cpu", None)
                item["seed_index"] = int(seed_idx)
                results.append(item)
                if coeff_est_cpu is not None:
                    plot_cases[str(label)] = {**item, "coeff_est_cpu": coeff_est_cpu}
    return results, operator_stats, plot_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare old gap_v2 and new sampling Tikhonov results with split_triangular_admm=600."
    )
    parser.add_argument(
        "--beta-set",
        type=str,
        default="injective16_pi_best",
        choices=["injective16_pi_best", "olduniform_b", "b", "olduniform_a", "a"],
        help="16-angle beta list to evaluate. Default matches current injective16_pi_best.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=None,
        help="Multiplicative noise levels. Defaults to 0.1.",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="Base seed for noise realizations.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Number of noise seeds per noise level.",
    )
    parser.add_argument(
        "--split-max-iter",
        type=int,
        default=DEFAULT_SPLIT_MAX_ITER,
        help="ADMM max iterations for split_triangular_admm. Default: 600.",
    )
    parser.add_argument("--output-md", type=str, default=None, help="Markdown output path.")
    parser.add_argument("--output-json", type=str, default=None, help="JSON output path.")
    parser.add_argument("--plot-output", type=str, default=None, help="Optional reconstruction comparison plot path.")
    parser.add_argument(
        "--plot-noise-level",
        type=float,
        default=None,
        help="Noise level used for the saved plot. Defaults to the first evaluated noise level.",
    )
    parser.add_argument(
        "--plot-noise-seed",
        type=int,
        default=None,
        help="Noise seed used for the saved plot. Defaults to --base-seed.",
    )
    parser.add_argument("--skip-plot", action="store_true", help="Do not write the reconstruction PNG.")
    args = parser.parse_args()

    if int(args.num_seeds) <= 0:
        raise ValueError(f"num-seeds must be positive, got {int(args.num_seeds)}.")
    if int(args.split_max_iter) <= 0:
        raise ValueError(f"split-max-iter must be positive, got {int(args.split_max_iter)}.")

    DATA_CONFIG["lambda_select_mode"] = "morozov"
    TIME_DOMAIN_CONFIG["init_method"] = "tikhonov_direct"
    TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = "split_triangular_admm"
    TIME_DOMAIN_CONFIG["split_admm_max_iter"] = int(args.split_max_iter)
    TIME_DOMAIN_CONFIG["theoretical_formula_mode"] = "legacy_injective_extension"

    beta_vectors = _select_beta_vectors(args.beta_set)
    noise_levels = [float(v) for v in (DEFAULT_NOISE_LEVELS if args.noise_levels is None else args.noise_levels)]
    output_stem = f"gap_v2_vs_sampling_tikhonov_admm{int(args.split_max_iter)}_{args.beta_set}"
    output_md = str(THIS_DIR / f"{output_stem}.md") if args.output_md is None else str(args.output_md)
    output_json = str(THIS_DIR / f"{output_stem}.json") if args.output_json is None else str(args.output_json)
    plot_output = str(THIS_DIR / f"{output_stem}.png") if args.plot_output is None else str(args.plot_output)
    plot_noise_level = float(noise_levels[0] if args.plot_noise_level is None else args.plot_noise_level)
    plot_noise_seed = int(args.base_seed if args.plot_noise_seed is None else args.plot_noise_seed)

    os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    if not bool(args.skip_plot):
        os.makedirs(os.path.dirname(os.path.abspath(plot_output)), exist_ok=True)

    results, operator_stats, plot_cases = _run_experiments(
        beta_vectors=beta_vectors,
        noise_levels=noise_levels,
        base_seed=int(args.base_seed),
        num_seeds=int(args.num_seeds),
        split_max_iter=int(args.split_max_iter),
        plot_noise_level=plot_noise_level,
        plot_noise_seed=plot_noise_seed,
    )
    summary = _summarize_results(results)
    markdown = _format_markdown(
        results=results,
        summary=summary,
        beta_set=str(args.beta_set),
        beta_vectors=beta_vectors,
        operator_stats=operator_stats,
        noise_levels=noise_levels,
        num_seeds=int(args.num_seeds),
        base_seed=int(args.base_seed),
        split_max_iter=int(args.split_max_iter),
    )

    if not bool(args.skip_plot) and plot_cases:
        _save_plot(
            coeff_true=_build_coeff_true(),
            plot_cases=plot_cases,
            output_path=plot_output,
            noise_level=plot_noise_level,
            noise_seed=plot_noise_seed,
            split_max_iter=int(args.split_max_iter),
        )

    meta = {
        "comparison": "gap_v2_vs_sampling",
        "beta_set": str(args.beta_set),
        "beta_vectors": [list(beta) for beta in beta_vectors],
        "base_seed": int(args.base_seed),
        "num_seeds": int(args.num_seeds),
        "noise_levels": noise_levels,
        "split_admm_max_iter": int(args.split_max_iter),
        "multi_angle_solver_mode": str(TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode", "")),
        "operator_stats": operator_stats,
        "output_md": output_md,
        "output_json": output_json,
        "plot_output": None if bool(args.skip_plot) else plot_output,
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

    print(f"Wrote markdown: {output_md}")
    print(f"Wrote json: {output_json}")
    if not bool(args.skip_plot):
        print(f"Wrote plot: {plot_output}")


if __name__ == "__main__":
    main()
