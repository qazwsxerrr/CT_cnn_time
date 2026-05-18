"""Select the second-stage extra8 alpha angles without excluding middle angles.

This module intentionally re-implements only the cached JSON re-selection part
of ``models/α_condition/alpha_condition_constrained_sampling.py`` in pure
Python.  It does not evaluate new candidates; it only re-buckets the existing
``alpha_full_resume.json`` records.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PER_BUCKET_KEEP = 20
DEFAULT_BEAM_SIZE = 200
DEFAULT_LAMBDA_UNIFORM = 0.25
DEFAULT_DUPLICATE_DECIMALS = 12


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("selected", "top8", "best8", "results"):
            records = payload.get(key)
            if isinstance(records, list) and records:
                return [dict(item) for item in records]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError("Alpha JSON must be a list or contain records under selected/top8/best8/results.")


def load_alpha_records(path: str | Path, *, prefer_results: bool = False) -> list[dict[str, Any]]:
    payload = json.loads(_as_path(path).read_text(encoding="utf-8"))
    if prefer_results and isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [dict(item) for item in payload["results"]]
    return _extract_records(payload)


def circular_uniformity_penalty(alphas: Iterable[float], period: float = math.pi) -> float:
    values = sorted(float(alpha) % float(period) for alpha in alphas)
    if len(values) <= 1:
        return 0.0
    gaps = [values[idx + 1] - values[idx] for idx in range(len(values) - 1)]
    gaps.append(values[0] + float(period) - values[-1])
    target = float(period) / float(len(values))
    return sum(((gap - target) / target) ** 2 for gap in gaps) / float(len(gaps))


def bucket_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    k: int,
    keep: int = DEFAULT_PER_BUCKET_KEEP,
) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(int(k))]
    if int(k) <= 0:
        raise ValueError(f"k must be positive, got {k!r}.")
    for item in candidates:
        if not bool(item.get("is_valid", False)):
            continue
        alpha = float(item["alpha"]) % math.pi
        bucket_idx = int(math.floor(alpha / math.pi * int(k)))
        bucket_idx = min(max(bucket_idx, 0), int(k) - 1)
        copied = dict(item)
        copied["bucket"] = bucket_idx
        buckets[bucket_idx].append(copied)
    for bucket in buckets:
        bucket.sort(key=lambda record: float(record["log_cond"]))
        del bucket[int(keep) :]
    return buckets


def set_score(group: Iterable[dict[str, Any]], *, lambda_uniform: float = DEFAULT_LAMBDA_UNIFORM) -> float:
    records = list(group)
    if not records:
        return float("inf")
    log_cond_mean = sum(float(record["log_cond"]) for record in records) / float(len(records))
    uniformity = circular_uniformity_penalty(float(record["alpha"]) for record in records)
    return log_cond_mean + float(lambda_uniform) * uniformity


def select_uniform_condition_best(
    candidates: Iterable[dict[str, Any]],
    *,
    k: int,
    per_bucket_keep: int = DEFAULT_PER_BUCKET_KEEP,
    beam_size: int = DEFAULT_BEAM_SIZE,
    lambda_uniform: float = DEFAULT_LAMBDA_UNIFORM,
) -> list[dict[str, Any]]:
    """Select one low-condition candidate from each alpha bucket.

    This is equivalent to the no-exclusion cached re-selection branch used by
    the alpha condition search script.
    """

    buckets = bucket_candidates(candidates, k=int(k), keep=int(per_bucket_keep))
    for idx, bucket in enumerate(buckets):
        if not bucket:
            raise RuntimeError(f"No valid alpha candidates in bucket {idx}.")

    beams: list[list[dict[str, Any]]] = [[]]
    for bucket in buckets:
        expanded = [beam + [item] for beam in beams for item in bucket]
        expanded.sort(key=lambda group: set_score(group, lambda_uniform=float(lambda_uniform)))
        beams = expanded[: int(beam_size)]
    return sorted(beams[0], key=lambda record: float(record["alpha"]))


def alpha_key(alpha: float, *, decimals: int = DEFAULT_DUPLICATE_DECIMALS) -> float:
    return round(float(alpha) % math.pi, int(decimals))


def circular_angle_distance(alpha: float, beta: float, period: float = math.pi) -> float:
    diff = abs((float(alpha) - float(beta)) % float(period))
    return min(diff, float(period) - diff)


def min_distance_to_original(record: dict[str, Any], original_alphas: Iterable[float]) -> float:
    alpha = float(record["alpha"])
    return min(circular_angle_distance(alpha, original) for original in original_alphas)


def _sort_by_alpha(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(item) for item in records), key=lambda record: float(record["alpha"]))


def _select_by_distance(
    candidates: list[dict[str, Any]],
    *,
    original_alphas: list[float],
    extra_k: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda record: (
            -min_distance_to_original(record, original_alphas),
            float(record.get("log_cond", math.log(float(record.get("cond", 1.0))))),
            float(record["alpha"]),
        ),
    )
    return _sort_by_alpha(ranked[: int(extra_k)])


def _select_uniform_distance_combo(
    candidates: list[dict[str, Any]],
    *,
    original_alphas: list[float],
    extra_k: int,
) -> list[dict[str, Any]]:
    """Select a uniformly spread extra set when no top24 angle repeats original16."""

    if len(candidates) < int(extra_k):
        raise ValueError(f"Need at least {extra_k} candidates, got {len(candidates)}.")

    best_group: tuple[float, float, float, tuple[dict[str, Any], ...]] | None = None
    for combo in itertools.combinations(candidates, int(extra_k)):
        distances = [min_distance_to_original(record, original_alphas) for record in combo]
        mean_negative_distance = -sum(distances) / float(len(distances))
        uniformity = circular_uniformity_penalty(float(record["alpha"]) for record in combo)
        mean_log_cond = sum(
            float(record.get("log_cond", math.log(float(record.get("cond", 1.0)))))
            for record in combo
        ) / float(len(combo))
        score = (uniformity, mean_negative_distance, mean_log_cond, tuple(combo))
        if best_group is None or score[:3] < best_group[:3]:
            best_group = score
    if best_group is None:
        raise RuntimeError("Failed to select a uniform extra angle combination.")
    return _sort_by_alpha(best_group[3])


def select_extra8_from_full(
    *,
    full_json: str | Path,
    selected16_json: str | Path,
    top24_k: int = 24,
    extra_k: int = 8,
    per_bucket_keep: int = DEFAULT_PER_BUCKET_KEEP,
    beam_size: int = DEFAULT_BEAM_SIZE,
    lambda_uniform: float = DEFAULT_LAMBDA_UNIFORM,
    duplicate_decimals: int = DEFAULT_DUPLICATE_DECIMALS,
) -> dict[str, Any]:
    full_records = load_alpha_records(full_json, prefer_results=True)
    original16 = _sort_by_alpha(load_alpha_records(selected16_json))
    original_alphas = [float(record["alpha"]) for record in original16]
    original_keys = {alpha_key(alpha, decimals=int(duplicate_decimals)) for alpha in original_alphas}

    selected24 = select_uniform_condition_best(
        full_records,
        k=int(top24_k),
        per_bucket_keep=int(per_bucket_keep),
        beam_size=int(beam_size),
        lambda_uniform=float(lambda_uniform),
    )
    repeats = [
        record for record in selected24
        if alpha_key(float(record["alpha"]), decimals=int(duplicate_decimals)) in original_keys
    ]
    non_original = [
        record for record in selected24
        if alpha_key(float(record["alpha"]), decimals=int(duplicate_decimals)) not in original_keys
    ]

    if repeats:
        if len(non_original) < int(extra_k):
            raise RuntimeError(
                f"Top{top24_k} has {len(non_original)} non-original angles after duplicate removal; "
                f"cannot select extra_k={extra_k}."
            )
        extra = _select_by_distance(non_original, original_alphas=original_alphas, extra_k=int(extra_k))
        strategy = "remove_repeats_then_max_min_distance"
    else:
        extra = _select_uniform_distance_combo(selected24, original_alphas=original_alphas, extra_k=int(extra_k))
        strategy = "no_repeats_uniform_combo_max_distance"

    return {
        "original16": original16,
        "selected24": _sort_by_alpha(selected24),
        "repeats": _sort_by_alpha(repeats),
        "non_original24": _sort_by_alpha(non_original),
        "extra8": extra,
        "repeat_count": int(len(repeats)),
        "strategy": strategy,
        "selection_params": {
            "top24_k": int(top24_k),
            "extra_k": int(extra_k),
            "per_bucket_keep": int(per_bucket_keep),
            "beam_size": int(beam_size),
            "lambda_uniform": float(lambda_uniform),
            "duplicate_decimals": int(duplicate_decimals),
            "exclude_centers": "",
            "exclude_window": 0.0,
        },
    }


def build_alpha_payload(
    *,
    selected: list[dict[str, Any]],
    results: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_sorted = _sort_by_alpha(selected)
    payload: dict[str, Any] = {
        "meta": dict(meta or {}),
        "selected": selected_sorted,
        "top8": selected_sorted[:8],
    }
    if results is not None:
        payload["results"] = list(results)
    return payload


def compare_selected_records(
    generated: Iterable[dict[str, Any]],
    reference: Iterable[dict[str, Any]],
    *,
    alpha_tol: float = 1.0e-12,
    tau_tol: float = 1.0e-12,
    cond_rel_tol: float = 1.0e-12,
) -> dict[str, Any]:
    generated_list = list(generated)
    reference_list = list(reference)
    rows: list[dict[str, Any]] = []
    all_match = len(generated_list) == len(reference_list)
    for idx, (gen, ref) in enumerate(zip(generated_list, reference_list)):
        alpha_diff = abs(float(gen["alpha"]) - float(ref["alpha"]))
        tau_diff = abs(float(gen["tau_star"]) - float(ref["tau_star"]))
        cond_ref = max(1.0, abs(float(ref["cond"])))
        cond_rel_diff = abs(float(gen["cond"]) - float(ref["cond"])) / cond_ref
        match = alpha_diff <= alpha_tol and tau_diff <= tau_tol and cond_rel_diff <= cond_rel_tol
        all_match = all_match and match
        rows.append(
            {
                "index": int(idx),
                "match": bool(match),
                "generated_alpha": float(gen["alpha"]),
                "reference_alpha": float(ref["alpha"]),
                "generated_degree": float(gen["alpha"]) * 180.0 / math.pi,
                "reference_degree": float(ref["alpha"]) * 180.0 / math.pi,
                "alpha_diff": alpha_diff,
                "tau_diff": tau_diff,
                "cond_rel_diff": cond_rel_diff,
            }
        )
    return {
        "all_match": bool(all_match),
        "generated_count": int(len(generated_list)),
        "reference_count": int(len(reference_list)),
        "rows": rows,
    }


def write_selection_outputs(
    *,
    full_json: str | Path,
    selected16_json: str | Path,
    output_dir: str | Path,
    generated16_name: str = "alpha_selected16_no_exclude_regenerated.json",
    selected24_name: str = "alpha_selected24_no_exclude.json",
    extra8_name: str = "alpha_extra8_from24_excluding16.json",
    comparison_name: str = "alpha_selected16_no_exclude_comparison.json",
) -> dict[str, Path]:
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_payload = json.loads(_as_path(full_json).read_text(encoding="utf-8"))
    full_records = full_payload.get("results", [])
    reference16 = _sort_by_alpha(load_alpha_records(selected16_json))
    generated16 = select_uniform_condition_best(full_records, k=16)
    extra_result = select_extra8_from_full(full_json=full_json, selected16_json=selected16_json)

    common_meta = {
        "selection_source": "reuse_json",
        "reuse_json": str(full_json),
        "selection_method": "bucketed [0,pi) candidates + beam search with circular uniformity penalty",
        "lambda_uniform": DEFAULT_LAMBDA_UNIFORM,
        "exclude_centers": "",
        "exclude_window": 0.0,
        "note": "No middle-angle exclusion is applied.",
    }
    generated16_payload = build_alpha_payload(
        selected=generated16,
        results=full_records,
        meta={**common_meta, "top_k": 16},
    )
    selected24_payload = build_alpha_payload(
        selected=extra_result["selected24"],
        results=full_records,
        meta={**common_meta, "top_k": 24},
    )
    extra8_payload = build_alpha_payload(
        selected=extra_result["extra8"],
        results=extra_result["non_original24"],
        meta={
            **common_meta,
            "top_k": 8,
            "source_top24_json": selected24_name,
            "source_original16_json": str(selected16_json),
            "repeat_count": extra_result["repeat_count"],
            "strategy": extra_result["strategy"],
            "selection_params": extra_result["selection_params"],
        },
    )
    comparison_payload = compare_selected_records(generated16, reference16)

    paths = {
        "generated16": output_dir / generated16_name,
        "selected24": output_dir / selected24_name,
        "extra8": output_dir / extra8_name,
        "comparison": output_dir / comparison_name,
    }
    paths["generated16"].write_text(json.dumps(generated16_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["selected24"].write_text(json.dumps(selected24_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["extra8"].write_text(json.dumps(extra8_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["comparison"].write_text(json.dumps(comparison_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    root = _default_project_root()
    parser = argparse.ArgumentParser(description="Select no-exclusion alpha16/top24/extra8 JSON files.")
    parser.add_argument("--full-json", default=str(root / "data" / "alpha_search_cache" / "alpha_full_resume.json"))
    parser.add_argument("--selected16-json", default=str(root / "data" / "alpha_search_cache" / "alpha_selected16.json"))
    parser.add_argument("--output-dir", default=str(root / "data" / "alpha_search_cache"))
    args = parser.parse_args(argv)

    paths = write_selection_outputs(
        full_json=args.full_json,
        selected16_json=args.selected16_json,
        output_dir=args.output_dir,
    )
    print("Wrote angle-selection outputs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    print(f"Generated alpha_selected16 matches reference: {comparison['all_match']}")


if __name__ == "__main__":
    main()
