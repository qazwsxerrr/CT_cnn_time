# -*- coding: utf-8 -*-
"""Continuous-alpha condition-constrained sampling search for B1*B1 CT.

For each candidate angle alpha in [0, pi), this script sorts continuous lattice
projections

    s_k(alpha) = k1*cos(alpha) + k2*sin(alpha)

and searches a shift tau so that the single-angle sparse matrix

    A_alpha_tau[i,j] = R_alpha phi(s_(i) + tau - s_(j))

has a small condition number.  The selected angles are bucketed over [0, pi)
so the final set stays approximately uniform instead of clustering only where
single-angle conditioning is best.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

IMAGE_SIZE = 128
DEFAULT_NUM_ALPHA_GRID = 2048
DEFAULT_NUM_ALPHA_RANDOM = 2048
DEFAULT_NUM_ALPHA_GOLDEN = 2048
DEFAULT_TOP_K = 8
DEFAULT_PER_BUCKET_KEEP = 20
DEFAULT_BEAM_SIZE = 200
DEFAULT_LAMBDA_UNIFORM = 0.25
DEFAULT_INJECTIVE_TOL = 1.0e-12
DEFAULT_VALUE_TOL = 1.0e-15
DEFAULT_EIGEN_TOL = 1.0e-14
DEFAULT_TAU_XATOL = 1.0e-3
DEFAULT_MAXITER = 32
DEFAULT_SVDS_TOL = 1.0e-6
DEFAULT_SVDS_MAXITER = 4000


def lex_lattice_indices(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    k1 = np.repeat(np.arange(int(height), dtype=np.float64), int(width))
    k2 = np.tile(np.arange(int(width), dtype=np.float64), int(height))
    return k1, k2


def alpha_direction(alpha: float) -> np.ndarray:
    alpha = float(alpha) % math.pi
    return np.asarray([math.cos(alpha), math.sin(alpha)], dtype=np.float64)


def support_bounds_b1b1(direction: np.ndarray) -> tuple[float, float]:
    c, s = float(direction[0]), float(direction[1])
    values = (0.0, c, s, c + s)
    return float(min(values)), float(max(values))


def integral_b1_numpy(u: np.ndarray) -> np.ndarray:
    return np.clip(u, 0.0, 1.0)


def b1_numpy(u: np.ndarray) -> np.ndarray:
    return ((u > 0.0) & (u <= 1.0)).astype(np.float64)


def radon_phi_b1b1_numpy(s_values: np.ndarray, direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    s_values = np.asarray(s_values, dtype=np.float64)
    a1 = float(direction[0])
    a2 = float(direction[1])
    eps = 1.0e-12
    if abs(a1) <= eps and abs(a2) <= eps:
        raise ValueError("direction must be non-zero")
    if abs(a2) <= eps:
        return b1_numpy(s_values / a1) / abs(a1)
    if abs(a1) <= eps:
        return b1_numpy(s_values / a2) / abs(a2)
    u0 = s_values / a1
    u1 = (s_values - a2) / a1
    u_lo = np.minimum(u0, u1)
    u_hi = np.maximum(u0, u1)
    return (integral_b1_numpy(u_hi) - integral_b1_numpy(u_lo)) / abs(a2)


def sorted_alpha_projections(
    alpha: float,
    image_size: int,
    *,
    injective_tol: float,
) -> tuple[np.ndarray, float, bool]:
    direction = alpha_direction(float(alpha))
    k1, k2 = lex_lattice_indices(int(image_size), int(image_size))
    proj = k1 * direction[0] + k2 * direction[1]
    sorted_proj = np.sort(proj, kind="stable")
    gaps = np.diff(sorted_proj)
    min_gap = float(np.min(gaps)) if gaps.size else math.inf
    return sorted_proj, min_gap, bool(min_gap > float(injective_tol))


def build_sparse_matrix(
    *,
    sorted_proj: np.ndarray,
    direction: np.ndarray,
    tau: float,
    value_tol: float,
) -> tuple[csr_matrix, dict[str, int | float]]:
    sorted_proj = np.asarray(sorted_proj, dtype=np.float64).reshape(-1)
    n = int(sorted_proj.size)
    support_lo, support_hi = support_bounds_b1b1(direction)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    lower_bw = 0
    upper_bw = 0

    for row_idx in range(n):
        t_i = float(sorted_proj[row_idx] + float(tau))
        left = t_i - support_hi
        right = t_i - support_lo
        col0 = int(np.searchsorted(sorted_proj, left, side="left"))
        col1 = int(np.searchsorted(sorted_proj, right, side="right"))
        if col1 <= col0:
            continue
        cols = np.arange(col0, col1, dtype=np.int64)
        vals = radon_phi_b1b1_numpy(t_i - sorted_proj[col0:col1], direction)
        mask = np.abs(vals) > float(value_tol)
        if not np.any(mask):
            continue
        rows = np.full((int(np.count_nonzero(mask)),), row_idx, dtype=np.int64)
        cols = cols[mask]
        vals = vals[mask].astype(np.float64, copy=False)
        row_parts.append(rows)
        col_parts.append(cols)
        data_parts.append(vals)
        lower_bw = max(lower_bw, int(np.max(rows - cols, initial=0)) + 1)
        upper_bw = max(upper_bw, int(np.max(cols - rows, initial=0)) + 1)

    if row_parts:
        rows_all = np.concatenate(row_parts)
        cols_all = np.concatenate(col_parts)
        data_all = np.concatenate(data_parts)
    else:
        rows_all = np.empty((0,), dtype=np.int64)
        cols_all = np.empty((0,), dtype=np.int64)
        data_all = np.empty((0,), dtype=np.float64)
    matrix = csr_matrix((data_all, (rows_all, cols_all)), shape=(n, n), dtype=np.float64)
    diag0 = float(radon_phi_b1b1_numpy(np.asarray([float(tau)]), direction)[0])
    return matrix, {
        "nnz": int(matrix.nnz),
        "lower_bandwidth": int(lower_bw),
        "upper_bandwidth": int(upper_bw),
        "diag0": float(diag0),
        "support_lo": float(support_lo),
        "support_hi": float(support_hi),
    }


def extreme_singular_values(
    matrix: csr_matrix,
    *,
    eigen_tol: float,
    svds_tol: float,
    svds_maxiter: int,
) -> dict[str, float | str | bool | None]:
    n = int(matrix.shape[0])
    if n <= 512:
        singular = np.linalg.svd(matrix.toarray(), compute_uv=False)
        sigma_max = float(singular[0]) if singular.size else 0.0
        sigma_min = float(singular[-1]) if singular.size else 0.0
        lambda_max = sigma_max * sigma_max
        lambda_min = sigma_min * sigma_min
        source = "dense_svd"
    else:
        normal = (matrix.T @ matrix).asfptype()
        v0 = np.linspace(1.0, 2.0, int(n), dtype=np.float64)
        v0 /= float(np.linalg.norm(v0))
        kwargs = {
            "k": 1,
            "return_eigenvectors": False,
            "tol": float(svds_tol),
            "maxiter": int(svds_maxiter),
            "v0": v0,
        }
        lambda_max = float(eigsh(normal, which="LA", **kwargs)[0])
        source = "shift_invert_sigma0"
        try:
            lambda_min = float(eigsh(normal, sigma=0.0, which="LM", **kwargs)[0])
        except (ArpackNoConvergence, RuntimeError, ValueError):
            try:
                lambda_min = float(eigsh(normal, which="SA", **kwargs)[0])
                source = "smallest_algebraic"
            except (ArpackNoConvergence, RuntimeError, ValueError):
                lambda_min = 0.0
                source = "fallback_zero"
        sigma_max = math.sqrt(max(lambda_max, 0.0))
        sigma_min = math.sqrt(max(lambda_min, 0.0))

    if sigma_min <= float(eigen_tol):
        cond = math.inf
        lower_bound = sigma_max / float(eigen_tol) if float(eigen_tol) > 0.0 else None
    else:
        cond = sigma_max / sigma_min
        lower_bound = None
    return {
        "lambda_min": float(lambda_min),
        "lambda_max": float(lambda_max),
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "condition_number": float(cond) if math.isfinite(cond) else "inf",
        "condition_number_lower_bound": lower_bound,
        "is_condition_infinite": bool(not math.isfinite(cond)),
        "lambda_min_source": str(source),
    }


def condition_metrics_for_alpha_tau(
    *,
    alpha: float,
    tau: float,
    sorted_proj: np.ndarray,
    direction: np.ndarray,
    eigen_tol: float,
    value_tol: float,
    svds_tol: float,
    svds_maxiter: int,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    matrix, matrix_stats = build_sparse_matrix(
        sorted_proj=sorted_proj,
        direction=direction,
        tau=float(tau),
        value_tol=float(value_tol),
    )
    spectral = extreme_singular_values(
        matrix,
        eigen_tol=float(eigen_tol),
        svds_tol=float(svds_tol),
        svds_maxiter=int(svds_maxiter),
    )
    cond_value = spectral["condition_number"]
    if isinstance(cond_value, str) or not math.isfinite(float(cond_value)) or float(cond_value) <= 0.0:
        log_cond = float("inf")
    else:
        log_cond = math.log(float(cond_value))
    return log_cond, matrix_stats, spectral


def optimize_tau_for_alpha(
    alpha: float,
    *,
    image_size: int,
    injective_tol: float,
    value_tol: float,
    eigen_tol: float,
    tau_xatol: float,
    maxiter: int,
    svds_tol: float,
    svds_maxiter: int,
) -> dict[str, Any]:
    sorted_proj, min_gap, injective = sorted_alpha_projections(
        alpha,
        int(image_size),
        injective_tol=float(injective_tol),
    )
    direction = alpha_direction(float(alpha))
    support_lo, support_hi = support_bounds_b1b1(direction)
    if not injective:
        return {
            "alpha": float(alpha),
            "tau_star": None,
            "cond": "inf",
            "is_valid": False,
            "reason": "alpha_projection_not_injective",
            "min_gap": float(min_gap),
        }

    cache: dict[float, tuple[float, dict[str, Any], dict[str, Any]]] = {}

    def eval_tau(tau: float) -> tuple[float, dict[str, Any], dict[str, Any]]:
        key = round(float(tau), 12)
        if key not in cache:
            cache[key] = condition_metrics_for_alpha_tau(
                alpha=float(alpha),
                tau=float(tau),
                sorted_proj=sorted_proj,
                direction=direction,
                eigen_tol=float(eigen_tol),
                value_tol=float(value_tol),
                svds_tol=float(svds_tol),
                svds_maxiter=int(svds_maxiter),
            )
        return cache[key]

    eps = 1.0e-6
    lo = float(support_lo + eps)
    hi = float(support_hi - eps)
    tau_baseline = float(support_lo + 0.5 * (support_hi - support_lo))
    baseline_log, baseline_stats, baseline_spectral = eval_tau(tau_baseline)
    if hi <= lo:
        tau_star = tau_baseline
        opt_log, opt_stats, opt_spectral = baseline_log, baseline_stats, baseline_spectral
        success = False
    else:
        result = minimize_scalar(
            lambda tau: eval_tau(float(tau))[0],
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": float(tau_xatol), "maxiter": int(maxiter)},
        )
        tau_star = float(result.x) if bool(result.success) else tau_baseline
        opt_log, opt_stats, opt_spectral = eval_tau(tau_star)
        if opt_log > baseline_log:
            tau_star = tau_baseline
            opt_log, opt_stats, opt_spectral = baseline_log, baseline_stats, baseline_spectral
        success = bool(result.success)

    cond_value = opt_spectral["condition_number"]
    cond = float(cond_value) if not isinstance(cond_value, str) else math.inf
    return {
        "alpha": float(alpha) % math.pi,
        "tau_baseline": float(tau_baseline),
        "tau_star": float(tau_star),
        "support_lo": float(support_lo),
        "support_hi": float(support_hi),
        "cond": float(cond) if math.isfinite(cond) else "inf",
        "sigma_min": float(opt_spectral["sigma_min"]),
        "sigma_max": float(opt_spectral["sigma_max"]),
        "lambda_min": float(opt_spectral["lambda_min"]),
        "lambda_max": float(opt_spectral["lambda_max"]),
        "lambda_min_source": str(opt_spectral["lambda_min_source"]),
        "is_condition_infinite": bool(opt_spectral["is_condition_infinite"]),
        "matrix_nnz": int(opt_stats["nnz"]),
        "lower_bandwidth": int(opt_stats["lower_bandwidth"]),
        "upper_bandwidth": int(opt_stats["upper_bandwidth"]),
        "diag0": float(opt_stats["diag0"]),
        "min_gap": float(min_gap),
        "log_cond": float(opt_log),
        "optimizer_success": bool(success),
        "is_valid": bool(math.isfinite(cond)),
        "sampling_formula": "t_i = sorted(k1*cos(alpha)+k2*sin(alpha))[i] + tau_star",
    }


def _optimize_tau_for_alpha_task(task: dict[str, Any]) -> dict[str, Any]:
    alpha = float(task["alpha"])
    try:
        return optimize_tau_for_alpha(
            alpha,
            image_size=int(task["image_size"]),
            injective_tol=float(task["injective_tol"]),
            value_tol=float(task["value_tol"]),
            eigen_tol=float(task["eigen_tol"]),
            tau_xatol=float(task["tau_xatol"]),
            maxiter=int(task["maxiter"]),
            svds_tol=float(task["svds_tol"]),
            svds_maxiter=int(task["svds_maxiter"]),
        )
    except Exception as exc:
        return {
            "alpha": float(alpha),
            "tau_star": None,
            "cond": "inf",
            "is_valid": False,
            "reason": repr(exc),
        }


def alpha_result_key(alpha: float) -> float:
    return round(float(alpha) % math.pi, 15)


def evaluate_alpha_candidates(
    alphas,
    *,
    workers: int,
    image_size: int,
    injective_tol: float,
    value_tol: float,
    eigen_tol: float,
    tau_xatol: float,
    maxiter: int,
    svds_tol: float,
    svds_maxiter: int,
    executor_cls=ProcessPoolExecutor,
    existing_results: list[dict[str, Any]] | None = None,
    save_every: int = 0,
    save_callback=None,
) -> list[dict[str, Any]]:
    alpha_list = [float(alpha) for alpha in list(alphas)]
    results: list[dict[str, Any]] = [dict(item) for item in list(existing_results or [])]
    seen = {
        alpha_result_key(float(item["alpha"]))
        for item in results
        if isinstance(item, dict) and "alpha" in item
    }
    tasks = [
        {
            "alpha": alpha,
            "image_size": int(image_size),
            "injective_tol": float(injective_tol),
            "value_tol": float(value_tol),
            "eigen_tol": float(eigen_tol),
            "tau_xatol": float(tau_xatol),
            "maxiter": int(maxiter),
            "svds_tol": float(svds_tol),
            "svds_maxiter": int(svds_maxiter),
        }
        for alpha in alpha_list
        if alpha_result_key(alpha) not in seen
    ]
    worker_count = max(1, int(workers))
    total = int(len(tasks))
    completed_since_save = 0
    if total == 0:
        if save_callback is not None:
            save_callback(results)
        return results

    if worker_count == 1:
        try:
            for idx, task in enumerate(tasks, start=1):
                print(f"[{idx}/{total}] alpha={float(task['alpha']):.12f} start", flush=True)
                item = _optimize_tau_for_alpha_task(task)
                results.append(item)
                completed_since_save += 1
                print(
                    f"[{idx}/{total}] alpha={float(task['alpha']):.12f} "
                    f"cond={format_float(item.get('cond'))} tau={item.get('tau_star')}",
                    flush=True,
                )
                if save_callback is not None and int(save_every) > 0 and completed_since_save >= int(save_every):
                    save_callback(results)
                    completed_since_save = 0
        except KeyboardInterrupt:
            if save_callback is not None:
                save_callback(results)
            raise
        if save_callback is not None and completed_since_save > 0:
            save_callback(results)
        return results

    print(f"Evaluating {total} alpha candidates with workers={worker_count}", flush=True)
    try:
        with executor_cls(max_workers=worker_count) as executor:
            future_to_alpha = {
                executor.submit(_optimize_tau_for_alpha_task, task): float(task["alpha"])
                for task in tasks
            }
            for idx, future in enumerate(as_completed(future_to_alpha), start=1):
                alpha = float(future_to_alpha[future])
                try:
                    item = future.result()
                except Exception as exc:
                    item = {
                        "alpha": alpha,
                        "tau_star": None,
                        "cond": "inf",
                        "is_valid": False,
                        "reason": repr(exc),
                    }
                results.append(item)
                completed_since_save += 1
                print(
                    f"[{idx}/{total}] alpha={float(item.get('alpha', alpha)):.12f} "
                    f"cond={format_float(item.get('cond'))} tau={item.get('tau_star')}",
                    flush=True,
                )
                if save_callback is not None and int(save_every) > 0 and completed_since_save >= int(save_every):
                    save_callback(results)
                    completed_since_save = 0
    except KeyboardInterrupt:
        if save_callback is not None:
            save_callback(results)
        raise
    if save_callback is not None and completed_since_save > 0:
        save_callback(results)
    return results


def generate_alpha_candidates(
    num_grid: int,
    num_random: int,
    num_golden: int,
    *,
    seed: int,
) -> np.ndarray:
    parts = []
    if int(num_grid) > 0:
        parts.append(np.linspace(0.0, math.pi, int(num_grid), endpoint=False))
    if int(num_random) > 0:
        rng = np.random.default_rng(int(seed))
        parts.append(rng.uniform(0.0, math.pi, size=int(num_random)))
    if int(num_golden) > 0:
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        parts.append(np.mod(np.arange(int(num_golden)) * golden, 1.0) * math.pi)
    if not parts:
        raise ValueError("At least one alpha candidate source must be non-empty.")
    alphas = np.mod(np.concatenate(parts), math.pi)
    alphas = np.unique(np.round(alphas, decimals=15))
    return np.sort(alphas)


def slice_alpha_candidates(
    alphas: np.ndarray,
    *,
    candidate_start: int | None,
    candidate_stop: int | None,
    max_candidates: int | None,
) -> np.ndarray:
    alphas = np.asarray(alphas, dtype=np.float64)
    start = 0 if candidate_start is None else int(candidate_start)
    stop = int(alphas.shape[0]) if candidate_stop is None else int(candidate_stop)
    if start < 0:
        raise ValueError(f"candidate_start must be non-negative, got {start}.")
    if stop < start:
        raise ValueError(f"candidate_stop must be >= candidate_start, got start={start}, stop={stop}.")
    sliced = alphas[start:stop]
    if max_candidates is not None:
        sliced = sliced[: int(max_candidates)]
    return sliced


def circular_uniformity_penalty(alphas: list[float], period: float = math.pi) -> float:
    alphas_np = np.sort(np.mod(np.asarray(alphas, dtype=np.float64), float(period)))
    if alphas_np.size <= 1:
        return 0.0
    gaps = np.diff(np.r_[alphas_np, alphas_np[0] + float(period)])
    target = float(period) / float(alphas_np.size)
    return float(np.mean(((gaps - target) / target) ** 2))


def parse_angle_centers(raw: str) -> list[float]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    values: list[float] = []
    aliases = {
        "pi/4": math.pi / 4.0,
        "π/4": math.pi / 4.0,
        "3pi/4": 3.0 * math.pi / 4.0,
        "3π/4": 3.0 * math.pi / 4.0,
        "pi2": math.pi / 2.0,
        "pi/2": math.pi / 2.0,
        "π/2": math.pi / 2.0,
    }
    for token in raw.replace(";", ",").split(","):
        token = token.strip().lower().replace(" ", "")
        if not token:
            continue
        if token in aliases:
            values.append(float(aliases[token]))
        else:
            values.append(float(token))
    return values


def circular_angle_distance(alpha: float, center: float, period: float = math.pi) -> float:
    diff = abs((float(alpha) - float(center)) % float(period))
    return float(min(diff, float(period) - diff))


def filter_excluded_angle_windows(
    candidates: list[dict[str, Any]],
    *,
    exclude_centers: list[float],
    exclude_window: float,
) -> list[dict[str, Any]]:
    centers = [float(center) % math.pi for center in list(exclude_centers or [])]
    window = float(exclude_window)
    if not centers or window <= 0.0:
        return list(candidates)
    filtered: list[dict[str, Any]] = []
    for item in candidates:
        alpha = float(item["alpha"]) % math.pi
        if any(circular_angle_distance(alpha, center) <= window for center in centers):
            continue
        filtered.append(item)
    return filtered


def allowed_angle_intervals(
    *,
    exclude_centers: list[float],
    exclude_window: float,
    period: float = math.pi,
) -> list[tuple[float, float]]:
    period = float(period)
    centers = [float(center) % period for center in list(exclude_centers or [])]
    window = float(exclude_window)
    if not centers or window <= 0.0:
        return [(0.0, period)]
    if window >= 0.5 * period:
        return []

    excluded: list[tuple[float, float]] = []
    for center in centers:
        lo = float(center) - window
        hi = float(center) + window
        if lo < 0.0:
            excluded.append((0.0, hi))
            excluded.append((lo + period, period))
        elif hi > period:
            excluded.append((lo, period))
            excluded.append((0.0, hi - period))
        else:
            excluded.append((lo, hi))

    excluded.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = []
    for lo, hi in excluded:
        lo = max(0.0, float(lo))
        hi = min(period, float(hi))
        if hi <= lo:
            continue
        if not merged or lo > merged[-1][1]:
            merged.append((lo, hi))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))

    allowed: list[tuple[float, float]] = []
    cursor = 0.0
    for lo, hi in merged:
        if lo > cursor:
            allowed.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < period:
        allowed.append((cursor, period))
    return [(lo, hi) for lo, hi in allowed if hi > lo]


def angle_allowed_coordinate(alpha: float, allowed_intervals: list[tuple[float, float]]) -> tuple[float, float] | None:
    alpha = float(alpha) % math.pi
    offset = 0.0
    total = 0.0
    for lo, hi in allowed_intervals:
        lo = float(lo)
        hi = float(hi)
        length = max(0.0, hi - lo)
        if lo <= alpha < hi or (alpha == math.pi and hi == math.pi):
            return offset + (alpha - lo), sum(max(0.0, b - a) for a, b in allowed_intervals)
        offset += length
        total += length
    return None


def bucket_candidates(
    candidates: list[dict[str, Any]],
    *,
    k: int,
    keep: int,
    allowed_intervals: list[tuple[float, float]] | None = None,
) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(int(k))]
    intervals = allowed_intervals or [(0.0, math.pi)]
    total_allowed = float(sum(max(0.0, hi - lo) for lo, hi in intervals))
    if total_allowed <= 0.0:
        return buckets
    for item in candidates:
        if not item.get("is_valid", False):
            continue
        alpha = float(item["alpha"]) % math.pi
        coord = angle_allowed_coordinate(alpha, intervals)
        if coord is None:
            continue
        allowed_pos, allowed_total = coord
        bucket = int(math.floor(float(allowed_pos) / float(allowed_total) * int(k)))
        bucket = min(max(bucket, 0), int(k) - 1)
        copied = dict(item)
        copied["bucket"] = bucket
        buckets[bucket].append(copied)
    for bucket in buckets:
        bucket.sort(key=lambda x: float(x["log_cond"]))
        del bucket[int(keep) :]
    return buckets


def set_score(group: list[dict[str, Any]], *, lambda_uniform: float) -> float:
    log_cond_mean = float(np.mean([float(item["log_cond"]) for item in group]))
    uniformity = circular_uniformity_penalty([float(item["alpha"]) for item in group])
    return log_cond_mean + float(lambda_uniform) * uniformity


def select_uniform_condition_best(
    candidates: list[dict[str, Any]],
    *,
    k: int,
    per_bucket_keep: int,
    beam_size: int,
    lambda_uniform: float,
    allowed_intervals: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    buckets = bucket_candidates(
        candidates,
        k=int(k),
        keep=int(per_bucket_keep),
        allowed_intervals=allowed_intervals,
    )
    for idx, bucket in enumerate(buckets):
        if not bucket:
            raise RuntimeError(f"No valid alpha candidates in bucket {idx}.")
    beams: list[list[dict[str, Any]]] = [[]]
    for bucket in buckets:
        new_beams = [beam + [item] for beam in beams for item in bucket]
        new_beams.sort(key=lambda group: set_score(group, lambda_uniform=float(lambda_uniform)))
        beams = new_beams[: int(beam_size)]
    return sorted(beams[0], key=lambda x: float(x["alpha"]))


def _split_reuse_json_paths(path: str | Path) -> list[Path]:
    raw = str(path)
    parts = [part.strip() for part in raw.split(";") if part.strip()]
    return [Path(part) for part in (parts or [raw])]


def _load_reusable_alpha_results_one(path: str | Path) -> list[dict[str, Any]]:
    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("results")
        if not isinstance(records, list):
            records = payload.get("selected") or payload.get("top8") or payload.get("best8")
    else:
        records = None
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"Reusable alpha JSON must contain a non-empty 'results', 'selected', 'top8', or 'best8' list: {json_path}"
        )
    normalized: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Alpha cache record #{idx} must be an object, got {record!r}.")
        if "alpha" not in record:
            raise ValueError(f"Alpha cache record #{idx} is missing 'alpha': {record!r}.")
        item = dict(record)
        cond_value = item.get("cond", item.get("condition_number", math.inf))
        try:
            cond_float = float(cond_value)
        except (TypeError, ValueError):
            cond_float = math.inf
        if "log_cond" not in item:
            item["log_cond"] = math.log(cond_float) if math.isfinite(cond_float) and cond_float > 0.0 else math.inf
        item["is_valid"] = bool(item.get("is_valid", math.isfinite(cond_float)))
        normalized.append(item)
    return normalized


def load_reusable_alpha_results(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[float] = set()
    for json_path in _split_reuse_json_paths(path):
        for item in _load_reusable_alpha_results_one(json_path):
            alpha_key = alpha_result_key(float(item["alpha"]))
            if alpha_key in seen:
                continue
            seen.add(alpha_key)
            records.append(item)
    return records


def load_resume_alpha_results(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    json_path = Path(path)
    if not json_path.exists():
        return []
    return load_reusable_alpha_results(json_path)


def _format_exclude_window_for_filename(exclude_window: float) -> str:
    return f"{float(exclude_window):g}"


def default_output_path(args) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "alpha_search_cache"
    suffix = ""
    if float(getattr(args, "exclude_window", 0.0)) > 0.0:
        suffix = f"_exclude{_format_exclude_window_for_filename(float(args.exclude_window))}"
    return output_dir / f"alpha_selected{int(args.top_k)}{suffix}.json"


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if math.isinf(float(value)):
        return "inf"
    return f"{float(value):.12e}"


def build_output_payload(
    *,
    args,
    results: list[dict[str, Any]],
    top: list[dict[str, Any]],
    candidate_count: int,
    generated_candidate_count: int,
    selection_source: str,
    elapsed: float,
) -> dict[str, Any]:
    valid = [item for item in results if item.get("is_valid", False)]
    return {
        "meta": {
            "image_size": int(args.image_size),
            "top_k": int(args.top_k),
            "angle_interval": "[0, pi)",
            "parameterization": "alpha",
            "candidate_count": int(candidate_count),
            "generated_candidate_count": int(generated_candidate_count),
            "candidate_start": None if args.reuse_json else args.candidate_start,
            "candidate_stop": None if args.reuse_json else args.candidate_stop,
            "valid_candidate_count": int(len(valid)),
            "num_alpha_grid": int(args.num_alpha_grid),
            "num_alpha_random": int(args.num_alpha_random),
            "num_alpha_golden": int(args.num_alpha_golden),
            "workers": int(args.workers) if not args.reuse_json else 0,
            "selection_source": selection_source,
            "reuse_json": str(args.reuse_json) if args.reuse_json else None,
            "resume_json": str(args.resume_json) if getattr(args, "resume_json", None) else None,
            "save_every": int(getattr(args, "save_every", 0)),
            "tau_selection": "bounded minimize log(cond(A_alpha_tau)) on support interval",
            "selection_method": "bucketed [0,pi) candidates + beam search with circular uniformity penalty",
            "lambda_uniform": float(args.lambda_uniform),
            "exclude_centers": str(getattr(args, "exclude_centers", "")),
            "exclude_window": float(getattr(args, "exclude_window", 0.0)),
            "sampling_formula": "t_i = sorted(k1*cos(alpha)+k2*sin(alpha))[i] + tau_star",
            "elapsed_seconds": float(elapsed),
        },
        "selected": top,
        "top8": top,
        "results": results,
    }


def write_partial_alpha_results(
    path: str | Path,
    *,
    args,
    results: list[dict[str, Any]],
    candidate_count: int,
    generated_candidate_count: int,
    selection_source: str,
    started: float,
) -> None:
    json_path = Path(path)
    valid = [item for item in results if item.get("is_valid", False)]
    try:
        top = select_uniform_condition_best(
            valid,
            k=int(args.top_k),
            per_bucket_keep=int(args.per_bucket_keep),
            beam_size=int(args.beam_size),
            lambda_uniform=float(args.lambda_uniform),
        )
    except RuntimeError:
        top = []
    payload = build_output_payload(
        args=args,
        results=results,
        top=top,
        candidate_count=int(candidate_count),
        generated_candidate_count=int(generated_candidate_count),
        selection_source=selection_source,
        elapsed=time.perf_counter() - float(started),
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[checkpoint] saved {len(results)} alpha records to {json_path}", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search continuous-alpha B1*B1 sampling offsets.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num-alpha-grid", type=int, default=DEFAULT_NUM_ALPHA_GRID)
    parser.add_argument("--num-alpha-random", type=int, default=DEFAULT_NUM_ALPHA_RANDOM)
    parser.add_argument("--num-alpha-golden", type=int, default=DEFAULT_NUM_ALPHA_GOLDEN)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--per-bucket-keep", type=int, default=DEFAULT_PER_BUCKET_KEEP)
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE)
    parser.add_argument("--lambda-uniform", type=float, default=DEFAULT_LAMBDA_UNIFORM)
    parser.add_argument(
        "--exclude-centers",
        type=str,
        default="",
        help="Comma-separated angle centers to exclude before selection, e.g. 'pi/4,3pi/4'.",
    )
    parser.add_argument(
        "--exclude-window",
        type=float,
        default=0.0,
        help="Exclude candidates within this radians window around --exclude-centers before bucket/beam selection.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-start", type=int, default=None, help="Start index in the generated alpha candidate list.")
    parser.add_argument("--candidate-stop", type=int, default=None, help="Stop index in the generated alpha candidate list.")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for fresh alpha evaluation.")
    parser.add_argument("--injective-tol", type=float, default=DEFAULT_INJECTIVE_TOL)
    parser.add_argument("--value-tol", type=float, default=DEFAULT_VALUE_TOL)
    parser.add_argument("--eigen-tol", type=float, default=DEFAULT_EIGEN_TOL)
    parser.add_argument("--tau-xatol", type=float, default=DEFAULT_TAU_XATOL)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--svds-tol", type=float, default=DEFAULT_SVDS_TOL)
    parser.add_argument("--svds-maxiter", type=int, default=DEFAULT_SVDS_MAXITER)
    parser.add_argument(
        "--reuse-json",
        type=str,
        default=None,
        help="Reuse an existing alpha JSON cache and only re-run bucket/beam selection for --top-k.",
    )
    parser.add_argument(
        "--resume-json",
        type=str,
        default=None,
        help="Resume fresh alpha evaluation from an existing JSON. Defaults to --output-json when that file exists.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=200,
        help="During fresh alpha evaluation, checkpoint results every N newly evaluated alpha candidates. Use 0 to disable.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    started = time.perf_counter()
    output_json = Path(args.output_json) if args.output_json else default_output_path(args)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_json:
        results = load_reusable_alpha_results(args.reuse_json)
        selection_source = "reuse_json"
        candidate_count = int(len(results))
        generated_candidate_count = int(candidate_count)
        print(f"Reusing {candidate_count} cached alpha records from: {args.reuse_json}", flush=True)
    else:
        alphas = generate_alpha_candidates(
            int(args.num_alpha_grid),
            int(args.num_alpha_random),
            int(args.num_alpha_golden),
            seed=int(args.seed),
        )
        total_generated_count = int(len(alphas))
        alphas = slice_alpha_candidates(
            alphas,
            candidate_start=args.candidate_start,
            candidate_stop=args.candidate_stop,
            max_candidates=args.max_candidates,
        )
        resume_json = Path(args.resume_json) if args.resume_json else output_json
        existing_results = load_resume_alpha_results(resume_json)
        if existing_results:
            print(f"Resuming from {resume_json}: loaded {len(existing_results)} existing alpha records", flush=True)
        candidate_count = int(len(alphas))
        generated_candidate_count = int(total_generated_count)
        selection_source = "fresh_search_resume" if existing_results else "fresh_search"

        def save_checkpoint(current_results: list[dict[str, Any]]) -> None:
            write_partial_alpha_results(
                output_json,
                args=args,
                results=current_results,
                candidate_count=candidate_count,
                generated_candidate_count=generated_candidate_count,
                selection_source=selection_source,
                started=started,
            )

        results = evaluate_alpha_candidates(
            alphas,
            workers=int(args.workers),
            image_size=int(args.image_size),
            injective_tol=float(args.injective_tol),
            value_tol=float(args.value_tol),
            eigen_tol=float(args.eigen_tol),
            tau_xatol=float(args.tau_xatol),
            maxiter=int(args.maxiter),
            svds_tol=float(args.svds_tol),
            svds_maxiter=int(args.svds_maxiter),
            existing_results=existing_results,
            save_every=int(args.save_every),
            save_callback=save_checkpoint if int(args.save_every) > 0 else None,
        )

    valid_all = [item for item in results if item.get("is_valid", False)]
    exclude_centers = parse_angle_centers(args.exclude_centers)
    valid = filter_excluded_angle_windows(
        valid_all,
        exclude_centers=exclude_centers,
        exclude_window=float(args.exclude_window),
    )
    allowed_intervals = allowed_angle_intervals(
        exclude_centers=exclude_centers,
        exclude_window=float(args.exclude_window),
    )
    if len(valid) != len(valid_all):
        print(
            f"Excluded {len(valid_all) - len(valid)} valid alpha candidates "
            f"using centers={exclude_centers} window={float(args.exclude_window)}; "
            f"rebucketing on allowed intervals={allowed_intervals}",
            flush=True,
        )
    top = select_uniform_condition_best(
        valid,
        k=int(args.top_k),
        per_bucket_keep=int(args.per_bucket_keep),
        beam_size=int(args.beam_size),
        lambda_uniform=float(args.lambda_uniform),
        allowed_intervals=allowed_intervals,
    )
    elapsed = time.perf_counter() - started
    payload = build_output_payload(
        args=args,
        results=results,
        top=top,
        candidate_count=int(candidate_count),
        generated_candidate_count=int(generated_candidate_count),
        selection_source=selection_source,
        elapsed=float(elapsed),
    )
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote json: {output_json}")
    print("Selected angles:")
    for item in top:
        print(
            f"bucket={item.get('bucket')} alpha={float(item['alpha']):.12f} "
            f"tau={float(item['tau_star']):.6f} cond={format_float(item.get('cond'))}"
        )


if __name__ == "__main__":
    main()
