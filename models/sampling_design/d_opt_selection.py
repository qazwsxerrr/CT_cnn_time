# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
PROJECT_ROOT = MODELS_DIR.parent
ALPHA_CONDITION_DIR = MODELS_DIR / "α_condition"
for _path in (PROJECT_ROOT, MODELS_DIR, ALPHA_CONDITION_DIR, THIS_DIR):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from alpha_condition_constrained_sampling import circular_uniformity_penalty  # noqa: E402

try:  # noqa: E402
    from models.sampling_design.reduced_operator import (
        clean_record_for_json,
        load_candidate_records,
        make_random_sketch_basis,
        reduced_information_for_record,
        sort_candidate_records,
    )
except ModuleNotFoundError:  # pragma: no cover - script-by-path fallback
    from reduced_operator import (  # type: ignore
        clean_record_for_json,
        load_candidate_records,
        make_random_sketch_basis,
        reduced_information_for_record,
        sort_candidate_records,
    )


def logdet_spd(mat: np.ndarray, jitter: float = 1.0e-10) -> float:
    sym = 0.5 * (np.asarray(mat, dtype=np.float64) + np.asarray(mat, dtype=np.float64).T)
    eye = np.eye(int(sym.shape[0]), dtype=np.float64)
    jitter_value = float(jitter)
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(sym + jitter_value * eye)
            return float(2.0 * np.sum(np.log(np.diag(chol))))
        except np.linalg.LinAlgError:
            jitter_value = max(jitter_value * 10.0, 1.0e-14)
    sign, value = np.linalg.slogdet(sym + jitter_value * eye)
    if sign <= 0:
        raise ValueError("matrix is not positive definite")
    return float(value)


def d_opt_greedy_select(
    candidates: list[dict[str, Any]],
    *,
    k: int,
    sketch_rank: int,
    lambda_info: float,
    gamma_uniform: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if int(k) <= 0:
        raise ValueError(f"k must be positive, got {k!r}.")
    if int(k) > len(candidates):
        raise ValueError(f"k={int(k)} exceeds candidate count {len(candidates)}.")
    if float(lambda_info) <= 0.0:
        raise ValueError(f"lambda_info must be positive, got {lambda_info!r}.")

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    f_mat = np.eye(int(sketch_rank), dtype=np.float64)
    current_logdet = logdet_spd(f_mat)
    trace: list[dict[str, Any]] = []

    for step in range(1, int(k) + 1):
        best_idx: int | None = None
        best_score = -math.inf
        best_payload: dict[str, Any] | None = None
        current_uniform = circular_uniformity_penalty([float(item["alpha"]) for item in selected])

        for idx, item in enumerate(remaining):
            g_mat = np.asarray(item["reduced_info"], dtype=np.float64)
            if g_mat.shape != (int(sketch_rank), int(sketch_rank)):
                raise ValueError(
                    f"candidate reduced_info has shape {g_mat.shape!r}; "
                    f"expected {(int(sketch_rank), int(sketch_rank))!r}."
                )
            trial_f = f_mat + (1.0 / float(lambda_info)) * g_mat
            trial_logdet = logdet_spd(trial_f)

            trial_alphas = [float(x["alpha"]) for x in selected] + [float(item["alpha"])]
            trial_uniform = circular_uniformity_penalty(trial_alphas)
            gain = trial_logdet - current_logdet
            uniform_delta = trial_uniform - current_uniform
            score = gain - float(gamma_uniform) * uniform_delta

            if score > best_score:
                best_idx = int(idx)
                best_score = float(score)
                best_payload = {
                    "step": int(step),
                    "alpha": float(item["alpha"]),
                    "tau_star": float(item["tau_star"]),
                    "gain": float(gain),
                    "uniform_delta": float(uniform_delta),
                    "score": float(score),
                    "trial_logdet": float(trial_logdet),
                    "trial_uniformity": float(trial_uniform),
                    "source_log_cond": float(item["log_cond"]) if "log_cond" in item else None,
                    "source_cond": item.get("cond"),
                }

        if best_idx is None or best_payload is None:
            raise RuntimeError("failed to select next D-optimal candidate")

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        f_mat = f_mat + (1.0 / float(lambda_info)) * np.asarray(chosen["reduced_info"], dtype=np.float64)
        current_logdet = logdet_spd(f_mat)
        trace.append(best_payload)
        print(
            f"[select {step}/{int(k)}] alpha={best_payload['alpha']:.12f} "
            f"tau={best_payload['tau_star']:.6e} gain={best_payload['gain']:.6e} "
            f"uniform_delta={best_payload['uniform_delta']:.6e} score={best_payload['score']:.6e}",
            flush=True,
        )

    return selected, trace


def _enrich_candidates(
    records: list[dict[str, Any]],
    *,
    z_basis: np.ndarray,
    image_size: int,
    injective_tol: float,
    value_tol: float,
    progress_every: int,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    total = int(len(records))
    started = time.perf_counter()
    for idx, record in enumerate(records, start=1):
        try:
            item = reduced_information_for_record(
                record,
                z_basis=z_basis,
                image_size=int(image_size),
                injective_tol=float(injective_tol),
                value_tol=float(value_tol),
            )
        except Exception as exc:
            print(f"[{idx}/{total}] skip alpha={float(record.get('alpha', math.nan)):.12f}: {exc!r}", flush=True)
            continue
        enriched.append(item)
        if int(progress_every) <= 1 or idx == total or idx % int(progress_every) == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[{idx}/{total}] alpha={float(item['alpha']):.12f} "
                f"trace={float(item['reduced_info_trace']):.6e} "
                f"nnz={int(item['matrix_nnz_rebuilt'])} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return enriched


def build_output_payload(
    *,
    selected: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    args: argparse.Namespace,
    source_count: int,
    enriched_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    selected_clean = [clean_record_for_json(item) for item in selected]
    return {
        "meta": {
            "selection_objective": "sketched_bayesian_d_optimal",
            "candidate_json": str(args.candidate_json),
            "source_candidate_count": int(source_count),
            "enriched_candidate_count": int(enriched_count),
            "top_k": int(args.top_k),
            "image_size": int(args.image_size),
            "sketch_rank": int(args.sketch_rank),
            "sketch_seed": int(args.sketch_seed),
            "lambda_info": float(args.lambda_info),
            "gamma_uniform": float(args.gamma_uniform),
            "candidate_order": str(args.candidate_order),
            "max_candidates": int(args.max_candidates),
            "injective_tol": float(args.injective_tol),
            "value_tol": float(args.value_tol),
            "tau_selection": "reuse tau_star from candidate JSON",
            "sampling_formula": "t_i = sorted(k1*cos(alpha)+k2*sin(alpha))[i] + tau_star",
            "elapsed_seconds": float(elapsed_seconds),
        },
        "selected": selected_clean,
        "top8": selected_clean,
        "selection_trace": trace,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    default_candidate = PROJECT_ROOT / "data" / "alpha_search_cache" / "alpha_full_resume.json"
    default_output = PROJECT_ROOT / "data" / "alpha_search_cache" / "alpha_selected16_dopt.json"
    parser = argparse.ArgumentParser(description="Greedy sketched Bayesian D-optimal alpha/tau selection.")
    parser.add_argument("--candidate-json", type=str, default=str(default_candidate))
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sketch-rank", type=int, default=128)
    parser.add_argument("--sketch-seed", type=int, default=0)
    parser.add_argument("--lambda-info", type=float, default=1.0e-2)
    parser.add_argument("--gamma-uniform", type=float, default=0.25)
    parser.add_argument("--injective-tol", type=float, default=1.0e-12)
    parser.add_argument("--value-tol", type=float, default=1.0e-15)
    parser.add_argument(
        "--candidate-order",
        type=str,
        default="log-cond",
        choices=["input", "alpha", "log-cond"],
        help="Order before optional --max-candidates truncation.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="0 means all valid candidates; positive values preselect after --candidate-order.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-json", type=str, default=str(default_output))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    records = load_candidate_records(args.candidate_json, prefer_raw_results=True)
    records = sort_candidate_records(records, args.candidate_order)
    if int(args.max_candidates) > 0:
        records = records[: int(args.max_candidates)]
    if int(args.top_k) > len(records):
        raise ValueError(f"--top-k={int(args.top_k)} exceeds candidate count {len(records)}.")

    n = int(args.image_size) * int(args.image_size)
    z_basis = make_random_sketch_basis(n=n, rank=int(args.sketch_rank), seed=int(args.sketch_seed))
    print(
        f"Loaded {len(records)} candidate records; image_size={int(args.image_size)} "
        f"sketch_rank={int(args.sketch_rank)}",
        flush=True,
    )
    enriched = _enrich_candidates(
        records,
        z_basis=z_basis,
        image_size=int(args.image_size),
        injective_tol=float(args.injective_tol),
        value_tol=float(args.value_tol),
        progress_every=int(args.progress_every),
    )
    if int(args.top_k) > len(enriched):
        raise ValueError(f"Only {len(enriched)} candidates were rebuilt, but --top-k={int(args.top_k)} was requested.")

    selected, trace = d_opt_greedy_select(
        enriched,
        k=int(args.top_k),
        sketch_rank=int(args.sketch_rank),
        lambda_info=float(args.lambda_info),
        gamma_uniform=float(args.gamma_uniform),
    )
    elapsed = time.perf_counter() - started
    payload = build_output_payload(
        selected=selected,
        trace=trace,
        args=args,
        source_count=len(records),
        enriched_count=len(enriched),
        elapsed_seconds=float(elapsed),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()

