# -*- coding: utf-8 -*-
"""Pure Tikhonov evaluation for alpha condition-constrained sampling.

This entrypoint intentionally does not load or run the learned neural network.
It only:

1. loads alpha/tau records produced by ``alpha_condition_constrained_sampling.py``;
2. configures the runtime alpha-continuous stacked operator;
3. generates observations; and
4. solves the direct Tikhonov normal equations, with either fixed lambda or
   Morozov-selected lambda.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from alpha_condition_constrained_sampling import (
    load_reusable_alpha_results,
    select_uniform_condition_best,
)
from config import DATA_CONFIG, DATA_DIR, IMAGE_SIZE, RESULTS_DIR, TIME_DOMAIN_CONFIG, device
from radon_transform import TheoreticalDataGenerator

DEFAULT_ALPHA_JSON = ""
DEFAULT_OUTPUT_DIR = ""


def coeff_res(coeff_est: torch.Tensor, coeff_true: torch.Tensor) -> float:
    coeff_est = coeff_est.to(dtype=torch.float32)
    coeff_true = coeff_true.to(dtype=torch.float32, device=coeff_est.device)
    diff = torch.norm(coeff_est - coeff_true)
    denom = torch.norm(coeff_true).clamp_min(1.0e-12)
    return float((diff / denom).item())


def _as_float_cond(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = math.inf
    return out if math.isfinite(out) else math.inf


def _format_exclude_window_for_filename(exclude_window: float) -> str:
    return f"{float(exclude_window):g}"


def resolve_default_alpha_json(
    alpha_json: str | Path,
    *,
    top_k: int,
    exclude_window: float = 0.0,
) -> Path:
    raw = str(alpha_json or "").strip()
    if raw:
        return Path(raw)
    suffix = ""
    if float(exclude_window) > 0.0:
        suffix = f"_exclude{_format_exclude_window_for_filename(float(exclude_window))}"
    return Path(DATA_DIR) / "alpha_search_cache" / f"alpha_selected{int(top_k)}{suffix}.json"


def _alpha_json_has_exclusion(alpha_json: str | Path) -> bool:
    json_path = Path(alpha_json)
    if "exclude" in json_path.stem.lower():
        return True
    if not json_path.exists():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return False
    try:
        return float(meta.get("exclude_window", 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


def resolve_default_output_prefix(
    output_prefix: str,
    *,
    alpha_json: str | Path,
    top_k: int,
    lambda_mode: str,
) -> str:
    raw = str(output_prefix or "").strip()
    if raw:
        return raw
    diag_tag = "exclude_diag" if _alpha_json_has_exclusion(alpha_json) else "diag"
    return f"alpha{int(top_k)}_{diag_tag}_{str(lambda_mode).strip().lower()}"


def resolve_default_output_dir(output_dir: str | Path, *, output_prefix: str) -> Path:
    raw = str(output_dir or "").strip()
    if raw:
        return Path(raw)
    return Path(RESULTS_DIR) / str(output_prefix)


def _normalize_alpha_record(record: dict[str, Any]) -> dict[str, Any]:
    tau_value = record.get("tau_star", record.get("tau", None))
    if tau_value is None:
        raise ValueError(f"Alpha record is missing tau_star/tau: {record!r}")
    cond = _as_float_cond(record.get("cond", record.get("condition_number", math.inf)))
    item = dict(record)
    item["alpha"] = float(record["alpha"]) % math.pi
    item["tau_star"] = float(tau_value)
    item["cond"] = float(cond) if math.isfinite(cond) else "inf"
    item["log_cond"] = (
        float(record["log_cond"])
        if "log_cond" in record and math.isfinite(float(record["log_cond"]))
        else (math.log(cond) if math.isfinite(cond) and cond > 0.0 else math.inf)
    )
    item["is_valid"] = bool(record.get("is_valid", math.isfinite(cond)))
    return item


def _selected_records_from_payload(path: str | Path) -> list[dict[str, Any]] | None:
    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [_normalize_alpha_record(item) for item in payload]
    if not isinstance(payload, dict):
        return None
    records = payload.get("selected") or payload.get("top8") or payload.get("best8")
    if isinstance(records, list) and records:
        return [_normalize_alpha_record(item) for item in records]
    return None


def load_alpha_records(
    alpha_json: str | Path,
    *,
    top_k: int,
    per_bucket_keep: int,
    beam_size: int,
    lambda_uniform: float,
) -> list[dict[str, Any]]:
    """Load alpha/tau records, selecting from raw results when necessary."""
    selected = _selected_records_from_payload(alpha_json)
    if selected is not None:
        if len(selected) < int(top_k):
            raise ValueError(f"Selected alpha JSON has {len(selected)} records, but top_k={int(top_k)} was requested.")
        return selected[: int(top_k)]

    candidates = [_normalize_alpha_record(item) for item in load_reusable_alpha_results(alpha_json)]
    valid = [item for item in candidates if item.get("is_valid", False)]
    if len(valid) < int(top_k):
        raise ValueError(f"Alpha JSON has only {len(valid)} valid records, but top_k={int(top_k)} was requested.")
    return select_uniform_condition_best(
        valid,
        k=int(top_k),
        per_bucket_keep=int(per_bucket_keep),
        beam_size=int(beam_size),
        lambda_uniform=float(lambda_uniform),
    )


def apply_alpha_runtime_config(
    records: list[dict[str, Any]],
    *,
    alpha_json: str | Path,
    lambda_mode: str,
    lambda_reg: float,
    alpha_gram_cache_dir: str | Path | None = None,
) -> None:
    alpha_values = [float(item["alpha"]) % math.pi for item in records]
    tau_offsets = [float(item["tau_star"]) for item in records]
    num_angles = int(len(alpha_values))

    TIME_DOMAIN_CONFIG["experiment_profile"] = "alpha_condition"
    TIME_DOMAIN_CONFIG["alpha_values"] = alpha_values
    TIME_DOMAIN_CONFIG["alpha_tau_offsets"] = tau_offsets
    TIME_DOMAIN_CONFIG["alpha_condition_constrained_records"] = list(records)
    TIME_DOMAIN_CONFIG["alpha_condition_constrained_json"] = str(alpha_json)
    TIME_DOMAIN_CONFIG["num_angles_total"] = num_angles
    TIME_DOMAIN_CONFIG["num_angles"] = num_angles
    TIME_DOMAIN_CONFIG["operator_mode"] = "theoretical_b1b1"
    TIME_DOMAIN_CONFIG["use_multi_angle"] = True
    TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = "stacked_tikhonov"
    TIME_DOMAIN_CONFIG["theoretical_formula_mode"] = "alpha_continuous"
    TIME_DOMAIN_CONFIG["data_formula_mode"] = "auto_complete"
    TIME_DOMAIN_CONFIG["init_method"] = "tikhonov_direct"
    TIME_DOMAIN_CONFIG["cnn_backbone_only"] = False
    TIME_DOMAIN_CONFIG["cnn_num_angles_override"] = num_angles

    mode = str(lambda_mode).strip().lower()
    if mode not in {"morozov", "fixed"}:
        raise ValueError(f"Unsupported lambda_mode={lambda_mode!r}; expected 'morozov' or 'fixed'.")
    DATA_CONFIG["lambda_select_mode"] = mode
    DATA_CONFIG["lambda_reg"] = float(lambda_reg)
    if alpha_gram_cache_dir:
        DATA_CONFIG["alpha_gram_cache_dir"] = str(alpha_gram_cache_dir)


def scenario_defs() -> list[dict[str, Any]]:
    return [
        {"name": "mult_0_1", "noise_mode": "multiplicative", "noise_level": 0.1},
        {"name": "mult_0_05", "noise_mode": "multiplicative", "noise_level": 0.05},
        {"name": "mult_0_04", "noise_mode": "multiplicative", "noise_level": 0.04},
        {"name": "mult_0_03", "noise_mode": "multiplicative", "noise_level": 0.03},
        {"name": "mult_0_02", "noise_mode": "multiplicative", "noise_level": 0.02},
        {"name": "mult_0_01", "noise_mode": "multiplicative", "noise_level": 0.01},
    ]


def select_scenarios(requested: str) -> list[dict[str, Any]]:
    requested = str(requested).strip().lower()
    scenarios = scenario_defs()
    if requested == "all":
        return scenarios
    for item in scenarios:
        if item["name"] == requested:
            return [item]
    allowed = ["all"] + [item["name"] for item in scenarios]
    raise ValueError(f"Unsupported scenario={requested!r}; expected one of {allowed!r}.")


def build_observation(
    generator: TheoreticalDataGenerator,
    g_clean: torch.Tensor,
    scenario: dict[str, Any],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(seed))
    generator.noise_mode = str(scenario["noise_mode"]).strip().lower()
    generator.noise_level = float(scenario["noise_level"])
    if float(scenario["noise_level"]) == 0.0:
        observed = g_clean.clone()
    else:
        observed = generator._apply_noise(g_clean)
    noise_norm = torch.norm(observed - g_clean, dim=-1)
    return observed, noise_norm


def choose_lambda(
    generator: TheoreticalDataGenerator,
    observed: torch.Tensor,
    noise_norm: torch.Tensor,
    *,
    lambda_mode: str,
    lambda_reg: float,
) -> torch.Tensor:
    mode = str(lambda_mode).strip().lower()
    if mode == "fixed":
        return torch.full((int(observed.shape[0]),), float(lambda_reg), device=observed.device, dtype=torch.float32)
    if mode != "morozov":
        raise ValueError(f"Unsupported lambda_mode={lambda_mode!r}; expected 'morozov' or 'fixed'.")
    lam = generator.time_operator.choose_lambda_morozov(
        observed,
        noise_norm=noise_norm,
        tau=float(DATA_CONFIG.get("morozov_tau", 1.0)),
        max_iter=int(DATA_CONFIG.get("morozov_max_iter", 8)),
        lambda_min=float(DATA_CONFIG.get("morozov_lambda_min", 1.0e-12)),
        lambda_max=float(DATA_CONFIG.get("morozov_lambda_max", 1.0e12)),
    )
    return lam.to(device=observed.device, dtype=torch.float32).view(-1)


def plot_triptych(
    *,
    coeff_true: np.ndarray,
    coeff_est: np.ndarray,
    title: str,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    diff = coeff_est - coeff_true
    vmin = min(float(np.min(coeff_true)), float(np.min(coeff_est)))
    vmax = max(float(np.max(coeff_true)), float(np.max(coeff_est)))
    dv = float(np.max(np.abs(diff))) if diff.size else 1.0
    if dv <= 0.0:
        dv = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(coeff_true, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("true coeff")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(coeff_est, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Tikhonov coeff")
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(diff, cmap="bwr", origin="lower", vmin=-dv, vmax=dv)
    axes[2].set_title("error")
    plt.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_summary(results: list[dict[str, Any]], *, alpha_json: str, output_prefix: str) -> str:
    lines = [
        f"output_prefix: {output_prefix}",
        f"alpha_json: {alpha_json}",
        f"device: {device}",
        "",
        "scenario | trial | lambda | noise_norm | meas_res | coeff_res | solve_seconds",
    ]
    for item in results:
        lines.append(
            f"{item['scenario']} | {item['trial']} | {float(item['lambda']):.6e} | "
            f"{float(item['noise_norm']):.6e} | {float(item['measurement_residual']):.6e} | "
            f"{float(item['coeff_res']):.6e} | {float(item['solve_seconds']):.3f}"
        )
    return "\n".join(lines) + "\n"


def evaluate(
    *,
    alpha_json: str | Path,
    top_k: int,
    per_bucket_keep: int,
    beam_size: int,
    lambda_uniform: float,
    scenario: str,
    lambda_mode: str,
    lambda_reg: float,
    data_source: str,
    num_trials: int,
    base_seed: int,
    output_dir: str | Path,
    output_prefix: str,
    alpha_gram_cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    records = load_alpha_records(
        alpha_json,
        top_k=int(top_k),
        per_bucket_keep=int(per_bucket_keep),
        beam_size=int(beam_size),
        lambda_uniform=float(lambda_uniform),
    )
    apply_alpha_runtime_config(
        records,
        alpha_json=alpha_json,
        lambda_mode=lambda_mode,
        lambda_reg=float(lambda_reg),
        alpha_gram_cache_dir=alpha_gram_cache_dir,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = TheoreticalDataGenerator(data_source=data_source)

    print(f"Running alpha pure Tikhonov on device: {device}")
    print(f"alpha_json={alpha_json}")
    print(f"num_angles={len(records)} lambda_mode={lambda_mode} data_source={data_source}")
    for idx, item in enumerate(records, start=1):
        print(
            f"[alpha {idx:02d}] alpha={float(item['alpha']):.12f} "
            f"tau={float(item['tau_star']):.12f} cond={item.get('cond')}"
        )

    results: list[dict[str, Any]] = []
    scenarios = select_scenarios(scenario)
    for scenario_idx, scenario_item in enumerate(scenarios):
        for trial in range(int(num_trials)):
            seed = int(base_seed) + scenario_idx * 1000 + int(trial)
            torch.manual_seed(seed)
            np.random.seed(seed)
            coeff_true = generator._sample_coefficients(batch_size=1)
            g_clean = generator.data_forward_operator(coeff_true).to(torch.float32)
            observed, noise_norm = build_observation(generator, g_clean, scenario_item, seed=seed)

            t0 = time.perf_counter()
            lam = choose_lambda(
                generator,
                observed,
                noise_norm,
                lambda_mode=lambda_mode,
                lambda_reg=float(lambda_reg),
            )
            t1 = time.perf_counter()
            coeff_est = generator.solve_tikhonov_direct_init(observed, lambda_reg=lam)
            t2 = time.perf_counter()
            residual = generator.forward_operator(coeff_est) - observed
            result = {
                "scenario": str(scenario_item["name"]),
                "trial": int(trial),
                "seed": int(seed),
                "lambda": float(lam.view(-1)[0].item()),
                "noise_norm": float(noise_norm.view(-1)[0].item()),
                "measurement_residual": float(torch.norm(residual, dim=-1).view(-1)[0].item()),
                "coeff_res": coeff_res(coeff_est.squeeze(0).squeeze(0), coeff_true.squeeze(0).squeeze(0)),
                "lambda_seconds": float(t1 - t0),
                "solve_seconds": float(t2 - t1),
                "num_angles": int(len(records)),
                "alpha_values": [float(item["alpha"]) for item in records],
                "tau_offsets": [float(item["tau_star"]) for item in records],
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))

            if trial == int(num_trials) - 1:
                plot_triptych(
                    coeff_true=coeff_true.squeeze(0).squeeze(0).detach().cpu().numpy(),
                    coeff_est=coeff_est.squeeze(0).squeeze(0).detach().cpu().numpy(),
                    title=(
                        f"{scenario_item['name']} | lambda={result['lambda']:.3e} | "
                        f"coeff_RES={result['coeff_res']:.6f}"
                    ),
                    save_path=output_dir / f"{output_prefix}_{scenario_item['name']}.png",
                )

    json_path = output_dir / f"{output_prefix}_results.json"
    txt_path = output_dir / f"{output_prefix}_results.txt"
    payload = {
        "meta": {
            "alpha_json": str(alpha_json),
            "top_k": int(top_k),
            "lambda_mode": str(lambda_mode),
            "lambda_reg": float(lambda_reg),
            "data_source": str(data_source),
            "num_trials": int(num_trials),
            "image_size": int(IMAGE_SIZE),
        },
        "alpha_records": records,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(_format_summary(results, alpha_json=str(alpha_json), output_prefix=output_prefix), encoding="utf-8")
    print(f"Saved json: {json_path}")
    print(f"Saved summary: {txt_path}")
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure stacked Tikhonov for alpha condition-constrained sampling.")
    parser.add_argument("--alpha-json", type=str, default=DEFAULT_ALPHA_JSON)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--per-bucket-keep", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=80)
    parser.add_argument("--lambda-uniform", type=float, default=0.25)
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "mult_0_1", "mult_0_05", "mult_0_04", "mult_0_03", "mult_0_02", "mult_0_01"],
    )
    parser.add_argument("--lambda-mode", type=str, default="morozov", choices=["morozov", "fixed"])
    parser.add_argument("--lambda-reg", type=float, default=1.0e-2)
    parser.add_argument("--data-source", type=str, default="shepp_logan", choices=["shepp_logan", "random_ellipses"])
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument(
        "--exclude-window",
        type=float,
        default=0.0,
        help="When --alpha-json is omitted, read alpha_selected{top_k}_exclude{exclude_window}.json.",
    )
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha-gram-cache-dir", type=str, default="")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    alpha_json = resolve_default_alpha_json(
        args.alpha_json,
        top_k=int(args.top_k),
        exclude_window=float(args.exclude_window),
    )
    output_prefix = resolve_default_output_prefix(
        args.output_prefix,
        alpha_json=alpha_json,
        top_k=int(args.top_k),
        lambda_mode=args.lambda_mode,
    )
    output_dir = resolve_default_output_dir(args.output_dir, output_prefix=output_prefix)
    evaluate(
        alpha_json=alpha_json,
        top_k=int(args.top_k),
        per_bucket_keep=int(args.per_bucket_keep),
        beam_size=int(args.beam_size),
        lambda_uniform=float(args.lambda_uniform),
        scenario=args.scenario,
        lambda_mode=args.lambda_mode,
        lambda_reg=float(args.lambda_reg),
        data_source=args.data_source,
        num_trials=int(args.num_trials),
        base_seed=int(args.base_seed),
        output_dir=output_dir,
        output_prefix=output_prefix,
        alpha_gram_cache_dir=str(args.alpha_gram_cache_dir).strip() or None,
    )


if __name__ == "__main__":
    main()
