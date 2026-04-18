"""Compare per-angle condition numbers for legacy_injective_extension vs sampling.

This script analyzes the 16-angle injective set used by the current `models/`
pipeline and compares two per-angle B1*B1 matrix constructions:

1. ``legacy_injective_extension`` (current models/"gap_v2"):
   ``t0 - kappa_0 = 0.5``
2. ``sampling`` (theory-aligned sampling scheme):
   ``t0 - kappa_0 = A_beta + 0.5``

For each angle, the script constructs the full reordered matrix

    A_{ij} = [R_beta Phi](t0 + kappa_i - kappa_0 - kappa_j)

and computes the matrix condition number

    cond(A) = sigma_max(A) / sigma_min(A)

where ``sigma`` are singular values of the whole matrix rather than only the
lower-triangular/banded part.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


IMAGE_SIZE = 128
DEFAULT_T0 = 0.5
DEFAULT_EIGEN_TOL = 1.0e-14
DEFAULT_SVD_METHOD = "svds"
DEFAULT_SVDS_TOL = 1.0e-6
DEFAULT_SVDS_MAXITER = 4000
DEFAULT_DISPLAY_PRECISION = 12

# Same 16-angle set currently used by models/config.py -> injective16_pi_best.
BEST_PI16_BETA_VECTORS: list[tuple[int, int]] = [
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


def _primitive_signed_beta(beta: tuple[int, int]) -> tuple[int, int]:
    a = int(beta[0])
    b = int(beta[1])
    g = math.gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
    return (a, b)


def _lex_lattice_indices(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    k1 = np.repeat(np.arange(int(height), dtype=np.int64), int(width))
    k2 = np.tile(np.arange(int(width), dtype=np.int64), int(height))
    return k1, k2


def _beta_support_bounds_b1b1(beta: tuple[int, int]) -> tuple[float, float]:
    b1 = int(beta[0])
    b2 = int(beta[1])
    vals = (0, b1, b2, b1 + b2)
    return float(min(vals)), float(max(vals))


def _integral_b1_numpy(u: np.ndarray) -> np.ndarray:
    return np.clip(u, 0.0, 1.0)


def _b1_numpy(u: np.ndarray) -> np.ndarray:
    return ((u > 0.0) & (u <= 1.0)).astype(np.float64)


def _radon_phi_b1b1_numpy(s: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if alpha.shape != (2,):
        raise ValueError(f"alpha must have shape (2,), got {alpha.shape}")

    s = np.asarray(s, dtype=np.float64)
    a1 = float(alpha[0])
    a2 = float(alpha[1])
    eps = 1.0e-12

    if abs(a1) <= eps and abs(a2) <= eps:
        raise ValueError("alpha must be non-zero.")

    if abs(a2) <= eps:
        return _b1_numpy(s / a1) / abs(a1)

    if abs(a1) <= eps:
        return _b1_numpy(s / a2) / abs(a2)

    u0 = s / a1
    u1 = (s - a2) / a1
    u_lo = np.minimum(u0, u1)
    u_hi = np.maximum(u0, u1)
    return (_integral_b1_numpy(u_hi) - _integral_b1_numpy(u_lo)) / abs(a2)


def _build_angle_block(
    *,
    beta: tuple[int, int],
    height: int,
    width: int,
    band_t0: float,
) -> dict[str, Any]:
    h = int(height)
    w = int(width)
    n = int(h * w)

    beta_i = _primitive_signed_beta(beta)
    beta_f = np.asarray(beta_i, dtype=np.float64)
    beta_norm = float(np.linalg.norm(beta_f))
    alpha = beta_f / beta_norm

    k1, k2 = _lex_lattice_indices(h, w)
    beta_dot_k = int(beta_i[0]) * k1 + int(beta_i[1]) * k2
    sort_perm = np.argsort(beta_dot_k, kind="stable")
    sorted_proj = beta_dot_k[sort_perm].astype(np.int64, copy=False)

    if int(np.unique(sorted_proj).size) != n:
        raise ValueError(f"beta={beta_i!r} is not injective on [0,{h - 1}]x[0,{w - 1}]")

    support_lo_beta, support_hi_beta = _beta_support_bounds_b1b1(beta_i)
    return {
        "beta": [int(beta_i[0]), int(beta_i[1])],
        "beta_norm": float(beta_norm),
        "alpha": alpha.tolist(),
        "sorted_proj": sorted_proj,
        "sorted_proj_first": int(sorted_proj[0]),
        "sorted_proj_last": int(sorted_proj[-1]),
        "support_lo_beta": float(support_lo_beta),
        "support_hi_beta": float(support_hi_beta),
        "band_t0": float(band_t0),
    }


def _build_full_sparse_matrix(
    *,
    sorted_proj: np.ndarray,
    beta_norm: float,
    alpha: np.ndarray,
    support_lo_beta: float,
    support_hi_beta: float,
    band_t0: float,
) -> tuple[csr_matrix, dict[str, int | float]]:
    sorted_proj = np.asarray(sorted_proj, dtype=np.int64).reshape(-1)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    n = int(sorted_proj.shape[0])
    proj_min = int(sorted_proj[0])
    proj_max = int(sorted_proj[-1])

    lookup = np.full((proj_max - proj_min + 1,), -1, dtype=np.int64)
    lookup[sorted_proj - proj_min] = np.arange(n, dtype=np.int64)

    delta_lo = int(math.ceil(float(support_lo_beta) - float(band_t0) - 1.0e-12))
    delta_hi = int(math.floor(float(support_hi_beta) - float(band_t0) + 1.0e-12))

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    lower_bandwidth = 0
    upper_bandwidth = 0

    for delta in range(delta_lo, delta_hi + 1):
        kernel_value = float(_radon_phi_b1b1_numpy(np.asarray([(float(band_t0) + delta) / beta_norm]), alpha)[0])
        if abs(kernel_value) <= 1.0e-15:
            continue

        targets = sorted_proj - delta
        mask = (targets >= proj_min) & (targets <= proj_max)
        if not np.any(mask):
            continue

        rows = np.nonzero(mask)[0].astype(np.int64, copy=False)
        cols = lookup[targets[mask] - proj_min]
        valid = cols >= 0
        if not np.any(valid):
            continue

        rows = rows[valid]
        cols = cols[valid]
        if rows.size == 0:
            continue

        row_parts.append(rows)
        col_parts.append(cols)
        data_parts.append(np.full((rows.size,), kernel_value, dtype=np.float64))

        lower_bandwidth = max(lower_bandwidth, int(np.max(rows - cols, initial=0)))
        upper_bandwidth = max(upper_bandwidth, int(np.max(cols - rows, initial=0)))

    if row_parts:
        all_rows = np.concatenate(row_parts)
        all_cols = np.concatenate(col_parts)
        all_data = np.concatenate(data_parts)
    else:
        all_rows = np.empty((0,), dtype=np.int64)
        all_cols = np.empty((0,), dtype=np.int64)
        all_data = np.empty((0,), dtype=np.float64)

    matrix = csr_matrix((all_data, (all_rows, all_cols)), shape=(n, n), dtype=np.float64)
    diag0 = float(_radon_phi_b1b1_numpy(np.asarray([float(band_t0) / beta_norm]), alpha)[0])
    stats: dict[str, int | float] = {
        "nnz": int(matrix.nnz),
        "lower_bandwidth": int(lower_bandwidth + 1),
        "upper_bandwidth": int(upper_bandwidth + 1),
        "diag0": float(diag0),
        "delta_min": int(delta_lo),
        "delta_max": int(delta_hi),
    }
    return matrix, stats


def _extreme_singular_values_from_full_matrix_svds(
    matrix: csr_matrix,
    *,
    eigen_tol: float,
    svds_tol: float,
    svds_maxiter: int,
) -> dict[str, Any]:
    normal_matrix = (matrix.T @ matrix).asfptype()
    eigsh_kwargs = {
        "k": 1,
        "return_eigenvectors": False,
        "tol": float(svds_tol),
        "maxiter": int(svds_maxiter),
    }

    lambda_max = float(eigsh(normal_matrix, which="LA", **eigsh_kwargs)[0])
    lambda_min_source = "shift_invert_sigma0"

    try:
        lambda_min = float(eigsh(normal_matrix, sigma=0.0, which="LM", **eigsh_kwargs)[0])
    except (ArpackNoConvergence, RuntimeError, ValueError):
        try:
            lambda_min = float(eigsh(normal_matrix, which="SA", **eigsh_kwargs)[0])
            lambda_min_source = "smallest_algebraic"
        except (ArpackNoConvergence, RuntimeError, ValueError):
            lambda_min = 0.0
            lambda_min_source = "fallback_zero"

    sigma_max = math.sqrt(max(lambda_max, 0.0))
    sigma_min = math.sqrt(max(lambda_min, 0.0))
    cond_lower_bound = None
    if sigma_min <= float(eigen_tol):
        cond = math.inf
        if float(eigen_tol) > 0.0:
            cond_lower_bound = sigma_max / float(eigen_tol)
    else:
        cond = sigma_max / sigma_min
    return {
        "lambda_min": float(lambda_min),
        "lambda_max": float(lambda_max),
        "lambda_min_source": str(lambda_min_source),
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "condition_number": float(cond) if math.isfinite(cond) else "inf",
        "condition_number_lower_bound": (
            float(cond_lower_bound) if cond_lower_bound is not None and math.isfinite(cond_lower_bound) else None
        ),
        "is_condition_infinite": bool(not math.isfinite(cond)),
    }


def _analyze_beta(
    beta: tuple[int, int],
    *,
    image_size: int,
    t0: float,
    eigen_tol: float,
    svd_method: str,
    svds_tol: float,
    svds_maxiter: int,
) -> dict[str, Any]:
    support_lo_beta, _ = _beta_support_bounds_b1b1(beta)
    methods = {
        "legacy_injective_extension": float(t0),
        "sampling": float(support_lo_beta + t0),
    }

    method_results: dict[str, Any] = {}
    for method_name, band_t0 in methods.items():
        block = _build_angle_block(beta=beta, height=image_size, width=image_size, band_t0=band_t0)
        if str(svd_method).strip().lower() != "svds":
            raise ValueError("This script now computes the full sparse matrix condition number; only --svd-method=svds is supported.")
        matrix, matrix_stats = _build_full_sparse_matrix(
            sorted_proj=np.asarray(block["sorted_proj"], dtype=np.int64),
            beta_norm=float(block["beta_norm"]),
            alpha=np.asarray(block["alpha"], dtype=np.float64),
            support_lo_beta=float(block["support_lo_beta"]),
            support_hi_beta=float(block["support_hi_beta"]),
            band_t0=float(block["band_t0"]),
        )
        spectral_info = _extreme_singular_values_from_full_matrix_svds(
            matrix,
            eigen_tol=float(eigen_tol),
            svds_tol=float(svds_tol),
            svds_maxiter=int(svds_maxiter),
        )
        method_results[method_name] = {
            "band_t0": float(block["band_t0"]),
            "lower_bandwidth": int(matrix_stats["lower_bandwidth"]),
            "upper_bandwidth": int(matrix_stats["upper_bandwidth"]),
            "matrix_nnz": int(matrix_stats["nnz"]),
            "delta_min": int(matrix_stats["delta_min"]),
            "delta_max": int(matrix_stats["delta_max"]),
            "diag0": float(matrix_stats["diag0"]),
            "lambda_min": float(spectral_info["lambda_min"]),
            "lambda_max": float(spectral_info["lambda_max"]),
            "lambda_min_source": str(spectral_info["lambda_min_source"]),
            "sigma_min": float(spectral_info["sigma_min"]),
            "sigma_max": float(spectral_info["sigma_max"]),
            "condition_number": spectral_info["condition_number"],
            "condition_number_lower_bound": spectral_info["condition_number_lower_bound"],
            "is_condition_infinite": bool(spectral_info["is_condition_infinite"]),
        }

    return {
        "beta": [int(beta[0]), int(beta[1])],
        "support_lo_beta": float(support_lo_beta),
        "legacy_injective_extension": method_results["legacy_injective_extension"],
        "sampling": method_results["sampling"],
    }


def _default_output_paths(this_file: Path) -> tuple[Path, Path]:
    stem = "legacyext_vs_sampling_condition_numbers_injective16_pi_best"
    return (
        this_file.with_name(f"{stem}.json"),
        this_file.with_name(f"{stem}.md"),
    )


def _format_float(value: float | str | None, precision: int = DEFAULT_DISPLAY_PRECISION) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if math.isinf(value):
        return "inf"
    return f"{float(value):.{int(precision)}e}"


def _format_condition_display(row: dict[str, Any]) -> str:
    cond_value = row.get("condition_number")
    if isinstance(cond_value, str) or (isinstance(cond_value, (int, float)) and math.isinf(float(cond_value))):
        lower_bound = row.get("condition_number_lower_bound")
        if lower_bound is not None:
            return f">={_format_float(float(lower_bound))}"
        return "inf"
    return _format_float(float(cond_value))


def _render_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# legacy_injective_extension vs sampling 条件数对比",
        "",
        "- `legacy_injective_extension`：`t0 - kappa_0 = 0.5`",
        "- `sampling`：`t0 - kappa_0 = A_beta + 0.5`",
        "- 这里计算的是整个矩阵 `A_{beta,phi,X}` 的条件数，不再是仅下带矩阵的条件数。",
        "- 条件数定义：`sigma_max / sigma_min`",
        "- 当 `cond(A)=inf` 时，额外给出 `cond_lb`，表示基于 `eigen_tol` 的条件数下界。",
        "",
        "| # | beta | A_beta | method | band_t0 | lower_bw | upper_bw | nnz | diag0 | lambda_min | lambda_max | sigma_min | sigma_max | cond(A) | cond_lb | lambda_min_source |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for idx, item in enumerate(results, start=1):
        beta_text = f"({item['beta'][0]}, {item['beta'][1]})"
        a_beta_text = f"{float(item['support_lo_beta']):.1f}"
        for method_name in ("legacy_injective_extension", "sampling"):
            row = item[method_name]
            lines.append(
                f"| {idx} | `{beta_text}` | `{a_beta_text}` | `{method_name}` | "
                f"`{float(row['band_t0']):.6f}` | `{int(row['lower_bandwidth'])}` | "
                f"`{int(row['upper_bandwidth'])}` | `{int(row['matrix_nnz'])}` | "
                f"`{_format_float(float(row['diag0']))}` | `{_format_float(row['lambda_min'])}` | "
                f"`{_format_float(row['lambda_max'])}` | `{_format_float(row['sigma_min'])}` | "
                f"`{_format_float(row['sigma_max'])}` | `{_format_condition_display(row)}` | "
                f"`{_format_float(row['condition_number_lower_bound'])}` | `{row['lambda_min_source']}` |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-angle condition numbers for legacy_injective_extension vs sampling."
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Square image size. Default: 128.")
    parser.add_argument("--t0", type=float, default=DEFAULT_T0, help="Base sampling t0. Default: 0.5.")
    parser.add_argument(
        "--eigen-tol",
        type=float,
        default=DEFAULT_EIGEN_TOL,
        help="Threshold below which sigma_min is treated as zero.",
    )
    parser.add_argument(
        "--svd-method",
        type=str,
        default=DEFAULT_SVD_METHOD,
        choices=["svds"],
        help="Singular-value estimation method for the full sparse matrix. Default: svds.",
    )
    parser.add_argument(
        "--svds-tol",
        type=float,
        default=DEFAULT_SVDS_TOL,
        help="Tolerance for scipy.sparse.linalg.svds.",
    )
    parser.add_argument(
        "--svds-maxiter",
        type=int,
        default=DEFAULT_SVDS_MAXITER,
        help="Max iteration count for scipy.sparse.linalg.svds.",
    )
    parser.add_argument(
        "--beta-index",
        type=int,
        default=None,
        help="Optional 1-based beta index to run only one angle.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="JSON output path.")
    parser.add_argument("--output-md", type=str, default=None, help="Markdown output path.")
    args = parser.parse_args()

    this_file = Path(__file__).resolve()
    default_json, default_md = _default_output_paths(this_file)
    output_json = Path(args.output_json) if args.output_json else default_json
    output_md = Path(args.output_md) if args.output_md else default_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    if args.beta_index is None:
        beta_items = list(enumerate(BEST_PI16_BETA_VECTORS, start=1))
    else:
        beta_idx = int(args.beta_index)
        if beta_idx < 1 or beta_idx > len(BEST_PI16_BETA_VECTORS):
            raise ValueError(
                f"beta-index must be in [1, {len(BEST_PI16_BETA_VECTORS)}], got {beta_idx}."
            )
        beta_items = [(beta_idx, BEST_PI16_BETA_VECTORS[beta_idx - 1])]

    results: list[dict[str, Any]] = []
    total = len(beta_items)
    for local_idx, (beta_idx, beta) in enumerate(beta_items, start=1):
        support_lo_beta, _ = _beta_support_bounds_b1b1(beta)
        print(
            f"[{local_idx}/{total}] start beta_index={beta_idx} beta={beta} A_beta={support_lo_beta:.1f}",
            flush=True,
        )
        item = _analyze_beta(
            beta=beta,
            image_size=int(args.image_size),
            t0=float(args.t0),
            eigen_tol=float(args.eigen_tol),
            svd_method=str(args.svd_method),
            svds_tol=float(args.svds_tol),
            svds_maxiter=int(args.svds_maxiter),
        )
        if abs(float(support_lo_beta)) <= 1.0e-12:
            item["sampling"] = dict(item["legacy_injective_extension"])
            item["sampling"]["band_t0"] = float(support_lo_beta + float(args.t0))
        results.append(item)
        print(
            f"[{local_idx}/{total}] done beta_index={beta_idx} "
            f"legacy_cond={_format_condition_display(item['legacy_injective_extension'])} "
            f"sampling_cond={_format_condition_display(item['sampling'])}",
            flush=True,
        )

    payload = {
        "meta": {
            "image_size": int(args.image_size),
            "t0": float(args.t0),
            "eigen_tol": float(args.eigen_tol),
            "svd_method": str(args.svd_method),
            "svds_tol": float(args.svds_tol),
            "svds_maxiter": int(args.svds_maxiter),
            "beta_vectors": [list(beta) for beta in BEST_PI16_BETA_VECTORS],
        },
        "results": results,
    }

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_render_markdown(results), encoding="utf-8")

    print(f"Wrote json: {output_json}")
    print(f"Wrote markdown: {output_md}")
    print("")
    for idx, item in enumerate(results, start=1):
        print(
            f"[{idx:02d}] beta={tuple(item['beta'])} "
            f"A_beta={item['support_lo_beta']:.1f} "
            f"legacy_cond={_format_condition_display(item['legacy_injective_extension'])} "
            f"sampling_cond={_format_condition_display(item['sampling'])}"
        )


if __name__ == "__main__":
    main()
