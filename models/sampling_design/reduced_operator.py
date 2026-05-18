# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
PROJECT_ROOT = MODELS_DIR.parent
ALPHA_CONDITION_DIR = MODELS_DIR / "α_condition"
for _path in (PROJECT_ROOT, MODELS_DIR, ALPHA_CONDITION_DIR):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from alpha_condition_constrained_sampling import (  # noqa: E402
    alpha_direction,
    build_sparse_matrix,
    lex_lattice_indices,
)


def _as_finite_float(value: Any, default: float = math.inf) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _payload_records(payload: Any, *, prefer_raw_results: bool) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None

    keys = ("results", "selected", "top8", "best8") if prefer_raw_results else ("selected", "top8", "best8", "results")
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list) and value:
            return value
    return None


def load_candidate_records(path: str | Path, *, prefer_raw_results: bool = True) -> list[dict[str, Any]]:
    """Load valid alpha/tau candidates from an alpha-search JSON file.

    ``alpha_full_resume.json`` contains both final ``selected`` records and the
    raw ``results`` pool.  D-optimal selection should normally run over
    ``results`` so the current condition-number method remains only a baseline.
    """
    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    records = _payload_records(payload, prefer_raw_results=bool(prefer_raw_results))
    if not isinstance(records, list):
        raise ValueError(f"No candidate records found in {json_path}")

    cleaned: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        if "alpha" not in item:
            continue
        tau = item.get("tau_star", item.get("tau", None))
        if tau is None:
            continue
        cond = _as_finite_float(item.get("cond", item.get("condition_number", math.inf)))
        is_valid = bool(item.get("is_valid", math.isfinite(cond)))
        if not is_valid:
            continue
        if not math.isfinite(float(tau)):
            continue
        alpha = float(item["alpha"]) % math.pi
        out = dict(item)
        out["alpha"] = float(alpha)
        out["tau_star"] = float(tau)
        if math.isfinite(cond):
            out["cond"] = float(cond)
            log_cond = _as_finite_float(item.get("log_cond", math.inf))
            out["log_cond"] = float(log_cond) if math.isfinite(log_cond) else math.log(float(cond))
        out["is_valid"] = True
        cleaned.append(out)

    if not cleaned:
        raise ValueError(f"No valid candidate records found in {json_path}")
    return cleaned


def sort_candidate_records(records: list[dict[str, Any]], order: str) -> list[dict[str, Any]]:
    order_key = str(order or "input").strip().lower().replace("_", "-")
    items = [dict(item) for item in records]
    if order_key in {"input", "none"}:
        return items
    if order_key in {"alpha", "angle"}:
        return sorted(items, key=lambda item: float(item["alpha"]))
    if order_key in {"log-cond", "condition", "cond"}:
        return sorted(items, key=lambda item: _as_finite_float(item.get("log_cond", math.inf)))
    raise ValueError(f"Unknown candidate order {order!r}; expected input, alpha, or log-cond.")


def make_random_sketch_basis(n: int, rank: int, seed: int = 0) -> np.ndarray:
    n = int(n)
    rank = int(rank)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}.")
    if rank <= 0 or rank > n:
        raise ValueError(f"rank must be in [1, n], got rank={rank!r}, n={n!r}.")
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal((n, rank)).astype(np.float64)
    q, _ = np.linalg.qr(z, mode="reduced")
    return np.asarray(q, dtype=np.float64)


def sorted_projection_order(
    alpha: float,
    image_size: int,
    injective_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Return direction, sorted projections, and sorted-order -> lex-order map."""
    direction = alpha_direction(float(alpha))
    k1, k2 = lex_lattice_indices(int(image_size), int(image_size))
    proj = k1 * direction[0] + k2 * direction[1]
    order_to_lex = np.argsort(proj, kind="stable").astype(np.int64)
    sorted_proj = np.asarray(proj[order_to_lex], dtype=np.float64)
    gaps = np.diff(sorted_proj)
    min_gap = float(np.min(gaps)) if gaps.size else math.inf
    return direction, sorted_proj, order_to_lex, min_gap, bool(min_gap > float(injective_tol))


def build_order_to_lex(alpha: float, image_size: int, injective_tol: float) -> np.ndarray:
    _direction, _sorted_proj, order_to_lex, min_gap, injective = sorted_projection_order(
        alpha=float(alpha),
        image_size=int(image_size),
        injective_tol=float(injective_tol),
    )
    if not injective:
        raise ValueError(f"alpha={float(alpha)} is numerically non-injective; min_gap={min_gap:.6e}")
    return order_to_lex


def reduced_information_for_record(
    record: dict[str, Any],
    *,
    z_basis: np.ndarray,
    image_size: int,
    injective_tol: float,
    value_tol: float,
) -> dict[str, Any]:
    alpha = float(record["alpha"]) % math.pi
    tau = float(record["tau_star"])
    direction, sorted_proj, order_to_lex, min_gap, injective = sorted_projection_order(
        alpha=alpha,
        image_size=int(image_size),
        injective_tol=float(injective_tol),
    )
    if not injective:
        raise ValueError(f"alpha={alpha} is numerically non-injective; min_gap={min_gap:.6e}")

    z_basis = np.asarray(z_basis, dtype=np.float64)
    n = int(image_size) * int(image_size)
    if z_basis.ndim != 2 or int(z_basis.shape[0]) != n:
        raise ValueError(f"z_basis must have shape ({n}, rank), got {z_basis.shape!r}.")

    block, matrix_stats = build_sparse_matrix(
        sorted_proj=sorted_proj,
        direction=direction,
        tau=tau,
        value_tol=float(value_tol),
    )
    z_ordered = z_basis[order_to_lex, :]
    b_mat = block @ z_ordered
    g_mat = np.asarray(b_mat.T @ b_mat, dtype=np.float64)
    g_mat = 0.5 * (g_mat + g_mat.T)

    out = dict(record)
    out["alpha"] = float(alpha)
    out["tau_star"] = float(tau)
    out["min_gap_rebuilt"] = float(min_gap)
    out["reduced_info"] = g_mat
    out["reduced_info_trace"] = float(np.trace(g_mat))
    out["matrix_nnz_rebuilt"] = int(matrix_stats["nnz"])
    return out


def clean_record_for_json(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "reduced_info"}
