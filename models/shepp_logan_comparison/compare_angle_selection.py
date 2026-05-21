# -*- coding: utf-8 -*-
r"""Compare condition-selected angles against uniformly random angles.

Default experiment matches the recent Shepp-Logan TV comparison:

* condition-selected ``alpha_selected{k}.json`` versus random ``k`` angles;
* multiplicative noise level ``0.1``;
* Morozov lambda selection for TV initialization;
* ADMM iterations ``80``;
* configurable angle counts via ``--angle-counts``.

Example from the project root through Windows Python:

```
D:\python_code\anaconda_mini\envs\pytorch_env\python.exe ^
  D:\ai_code\ai_project\ct_time\models\shepp_logan_comparison\compare_angle_selection.py ^
  --angle-counts 8,4 --init-method tv --admm-iters 80
```
"""

from __future__ import annotations

import argparse
import json
import math
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
PROJECT_ROOT = MODELS_DIR.parent
ALPHA_CONDITION_DIR = MODELS_DIR / "α_condition"
for path in (str(MODELS_DIR), str(ALPHA_CONDITION_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import DATA_CONFIG, DATA_DIR, RESULTS_DIR, TIME_DOMAIN_CONFIG, device
from initialization_methods import method_spec_from_init_method, method_names, reconstruction_method_defs
from radon_transform import AlphaContinuousB1B1Operator2D, TheoreticalDataGenerator

import alpha_tikhonov_eval as evalmod


def parse_angle_counts(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        tokens = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
        if not tokens:
            raise ValueError("--angle-counts must contain at least one positive integer.")
        counts = [int(item) for item in tokens]
    else:
        counts = [int(item) for item in raw]
    if not counts or any(item <= 0 for item in counts):
        raise ValueError(f"Invalid angle counts {counts!r}; every count must be positive.")
    deduped: list[int] = []
    for item in counts:
        if item not in deduped:
            deduped.append(item)
    return deduped


def random_uniform_alphas(count: int, seed: int, *, trial_index: int = 0) -> list[float]:
    """Draw sorted random angles in ``[0, pi)`` independently for each count.

    The seed is derived from ``(seed, count, trial_index)`` so that the
    random baseline for a given angle count does not depend on whether other
    counts were evaluated earlier in the same run.
    """
    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be positive, got {count!r}.")
    seed_seq = np.random.SeedSequence([int(seed), count, int(trial_index)])
    rng = np.random.default_rng(seed_seq)
    return sorted(float(v) for v in rng.uniform(0.0, math.pi, size=count))


def resolve_selected_alpha_json(template: str | Path, count: int) -> Path:
    raw = str(template or "").strip()
    if not raw:
        return Path(DATA_DIR) / "alpha_search_cache" / f"alpha_selected{int(count)}.json"
    if "{count}" in raw or "{k}" in raw:
        return Path(raw.format(count=int(count), k=int(count)))
    path = Path(raw)
    if path.exists() and path.is_dir():
        return path / f"alpha_selected{int(count)}.json"
    return path


def resolve_method_spec(method: str) -> dict[str, str]:
    return method_spec_from_init_method(method)


def parse_method_specs(raw: str) -> list[dict[str, str]]:
    text = str(raw or "tv").strip()
    if text.lower() == "all":
        return reconstruction_method_defs()
    tokens = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
    if not tokens:
        raise ValueError("--init-method must contain at least one method or 'all'.")
    return [resolve_method_spec(item) for item in tokens]


def _load_selected_records(
    *,
    selected_json: str | Path,
    source_json: str | Path,
    count: int,
    per_bucket_keep: int,
    beam_size: int,
    lambda_uniform: float,
) -> tuple[list[dict[str, Any]], Path, str]:
    selected_path = Path(selected_json)
    if selected_path.exists():
        records = evalmod.load_alpha_records(
            selected_path,
            top_k=int(count),
            per_bucket_keep=int(per_bucket_keep),
            beam_size=int(beam_size),
            lambda_uniform=float(lambda_uniform),
        )
        return records, selected_path, "selected_json"

    source_path = Path(str(source_json or "").strip())
    if source_path.exists():
        records = evalmod.load_alpha_records(
            source_path,
            top_k=int(count),
            per_bucket_keep=int(per_bucket_keep),
            beam_size=int(beam_size),
            lambda_uniform=float(lambda_uniform),
        )
        return records, source_path, "source_json_selected"

    raise FileNotFoundError(
        f"Cannot find selected angle JSON {selected_path} or source JSON {source_path}. "
        "Create alpha_selected{k}.json first, or pass --source-json."
    )


def _configure_runtime(
    *,
    alphas: list[float],
    tau_offsets: list[float] | None,
    init_method: str,
    lambda_mode: str,
    lambda_reg: float,
    noise_mode: str,
    noise_level: float,
    admm_iters: int,
    admm_cg_iters: int,
    admm_cg_tol: float,
    rho_data: float,
    rho_reg: float,
) -> None:
    num_angles = int(len(alphas))
    TIME_DOMAIN_CONFIG.update(
        {
            "experiment_profile": "alpha_condition",
            "alpha_values": [float(v) % math.pi for v in alphas],
            "alpha_tau_offsets": [] if tau_offsets is None else [float(v) for v in tau_offsets],
            "num_angles_total": num_angles,
            "num_angles": num_angles,
            "operator_mode": "theoretical_b1b1",
            "use_multi_angle": True,
            "multi_angle_solver_mode": "stacked_tikhonov",
            "theoretical_formula_mode": "alpha_continuous",
            "data_formula_mode": "auto_complete",
            "init_method": str(init_method),
            "cnn_num_angles_override": num_angles,
        }
    )
    DATA_CONFIG.update(
        {
            "lambda_select_mode": str(lambda_mode).strip().lower(),
            "lambda_reg": float(lambda_reg),
            "noise_mode": str(noise_mode).strip().lower(),
            "noise_level": float(noise_level),
            "l1_init_admm_iters": int(admm_iters),
            "l1_init_admm_cg_iters": int(admm_cg_iters),
            "l1_init_admm_cg_tol": float(admm_cg_tol),
            "l1_init_admm_rho_data": float(rho_data),
            "l1_init_admm_rho_reg": float(rho_reg),
        }
    )


def _build_operator(
    *,
    alphas: list[float],
    tau_offsets: list[float] | None,
    sampling_points_per_angle: list[list[float]] | None = None,
    height: int,
    width: int,
) -> AlphaContinuousB1B1Operator2D:
    return AlphaContinuousB1B1Operator2D(
        alpha_values=[float(v) % math.pi for v in alphas],
        height=int(height),
        width=int(width),
        tau_offsets=None if tau_offsets is None else [float(v) for v in tau_offsets],
        sampling_points_per_angle=sampling_points_per_angle,
    ).to(device)


def _residual_ratio(result: dict[str, Any], *, method: dict[str, str], lambda_info: dict[str, Any]) -> float:
    norm_type = str(lambda_info.get("residual_norm") or method.get("morozov_residual_norm") or "l2").strip().lower()
    target_values = lambda_info.get("target_norm") or []
    if isinstance(target_values, (int, float)):
        target = float(target_values)
    elif target_values:
        target = float(target_values[0])
    else:
        target = float(result["noise_l1"] if norm_type == "l1" else result["noise_l2"])
    observed_residual = float(result["measurement_l1"] if norm_type == "l1" else result["measurement_l2"])
    return observed_residual / max(target, 1.0e-12)


def _solve_case(
    *,
    case_name: str,
    count: int,
    alphas: list[float],
    tau_offsets: list[float] | None,
    sampling_points_per_angle: list[list[float]] | None = None,
    coeff_true: torch.Tensor,
    method: dict[str, str],
    lambda_mode: str,
    lambda_reg: float,
    noise_mode: str,
    noise_level: float,
    noise_seed: int,
    random_angle_seed: int | None,
    selected_json: str | Path | None,
    selected_source: str | None,
    admm_iters: int,
    admm_cg_iters: int,
    admm_cg_tol: float,
    rho_data: float,
    rho_reg: float,
) -> tuple[dict[str, Any], np.ndarray]:
    _configure_runtime(
        alphas=alphas,
        tau_offsets=tau_offsets,
        init_method=str(method["init_method"]),
        lambda_mode=lambda_mode,
        lambda_reg=float(lambda_reg),
        noise_mode=noise_mode,
        noise_level=float(noise_level),
        admm_iters=int(admm_iters),
        admm_cg_iters=int(admm_cg_iters),
        admm_cg_tol=float(admm_cg_tol),
        rho_data=float(rho_data),
        rho_reg=float(rho_reg),
    )
    op = _build_operator(
        alphas=alphas,
        tau_offsets=tau_offsets,
        sampling_points_per_angle=sampling_points_per_angle,
        height=int(coeff_true.shape[-2]),
        width=int(coeff_true.shape[-1]),
    )
    generator = TheoreticalDataGenerator(data_source="shepp_logan", time_operator=op)
    g_clean = generator.data_forward_operator(coeff_true).to(torch.float32)
    observed, noise_l2 = evalmod.build_observation(
        generator,
        g_clean,
        {"name": f"{noise_mode}_{noise_level:g}", "noise_mode": noise_mode, "noise_level": float(noise_level)},
        seed=int(noise_seed),
    )
    noise_l1 = torch.sum(torch.abs(observed - g_clean), dim=-1)

    started = time.perf_counter()
    lam_tensor = evalmod.choose_lambda(
        generator,
        observed,
        lambda_mode=lambda_mode,
        lambda_reg=float(lambda_reg),
        method=method,
    )
    coeff_est = evalmod.solve_method(generator, observed, lambda_reg=lam_tensor, method=method)
    solve_seconds = time.perf_counter() - started

    lambda_info = dict(generator.last_lambda_info or {})
    lam = float(lam_tensor.view(-1)[0].item()) if torch.is_tensor(lam_tensor) else float(lam_tensor)
    is_constrained = str(lambda_info.get("mode", "")).strip().lower() == "morozov_constrained"
    constraint_values = lambda_info.get("constraint_radius") or lambda_info.get("target_norm") or []
    constraint_radius = float(constraint_values[0]) if is_constrained and constraint_values else None
    norms = evalmod.measurement_norms(generator, coeff_est, observed)
    result = {
        "case": str(case_name),
        "num_angles": int(count),
        "selection": "random" if selected_json is None else "condition_selected",
        "method": str(method["name"]),
        "init_method": str(method["init_method"]),
        "objective": str(method["objective"]),
        "lambda_mode": str(lambda_mode),
        "morozov_form": str(DATA_CONFIG.get("morozov_form", "regularized")),
        "lambda": None if is_constrained else lam,
        "constraint_radius": constraint_radius,
        "lambda_info": lambda_info,
        "noise_mode": str(noise_mode),
        "noise_level": float(noise_level),
        "noise_seed": int(noise_seed),
        "random_angle_seed": random_angle_seed,
        "selected_json": None if selected_json is None else str(selected_json),
        "selected_source": selected_source,
        "alpha_values": [float(v) for v in alphas],
        "tau_offsets": None if tau_offsets is None else [float(v) for v in tau_offsets],
        "sampling_mode": str(getattr(op, "sampling_mode", "shifted_lattice")),
        "noise_l2": float(noise_l2.view(-1)[0].item()),
        "noise_l1": float(noise_l1.view(-1)[0].item()),
        **norms,
        "coeff_res": evalmod.coeff_res(coeff_est.squeeze(0).squeeze(0), coeff_true.squeeze(0).squeeze(0)),
        "tv_value": float(op.anisotropic_tv_norm(coeff_est).view(-1)[0].item()),
        "solve_seconds": float(solve_seconds),
        "admm_iters": int(admm_iters),
        "admm_cg_iters": int(admm_cg_iters),
        "rho_data": float(rho_data),
        "rho_reg": float(rho_reg),
    }
    result["residual_ratio"] = _residual_ratio(result, method=method, lambda_info=lambda_info)
    return result, coeff_est.squeeze(0).squeeze(0).detach().cpu().numpy()


def _plot_method_panels(
    *,
    coeff_true: np.ndarray,
    estimates: dict[str, np.ndarray],
    title: str,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [("true", coeff_true)] + [(name, estimates[name]) for name in estimates]
    vmin = min(float(np.min(arr)) for _, arr in panels)
    vmax = max(float(np.max(arr)) for _, arr in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (panel_title, arr) in zip(axes, panels):
        im = ax.imshow(arr, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(panel_title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_error_panels(
    *,
    coeff_true: np.ndarray,
    estimates: dict[str, np.ndarray],
    title: str,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    errors = {name: arr - coeff_true for name, arr in estimates.items()}
    dv = max(float(np.max(np.abs(arr))) for arr in errors.values()) if errors else 1.0
    dv = max(dv, 1.0e-6)
    fig, axes = plt.subplots(1, len(errors), figsize=(4 * len(errors), 4))
    if len(errors) == 1:
        axes = [axes]
    for ax, (name, arr) in zip(axes, errors.items()):
        im = ax.imshow(arr, cmap="bwr", origin="lower", vmin=-dv, vmax=dv)
        ax.set_title(f"{name} error")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_summary(results: list[dict[str, Any]]) -> str:
    lines = [
        "case | num_angles | method | morozov_form | lambda | constraint_radius | residual_ratio | measurement_l2 | measurement_l1 | coeff_res | tv_value | solve_seconds"
    ]
    for item in results:
        lam = item.get("lambda")
        lam_text = "None" if lam is None else f"{float(lam):.6e}"
        radius = item.get("constraint_radius")
        radius_text = "None" if radius is None else f"{float(radius):.6e}"
        lines.append(
            f"{item['case']} | {item['num_angles']} | {item['method']} | "
            f"{item.get('morozov_form', 'regularized')} | {lam_text} | {radius_text} | {float(item['residual_ratio']):.6f} | "
            f"{float(item['measurement_l2']):.6e} | {float(item['measurement_l1']):.6e} | "
            f"{float(item['coeff_res']):.6e} | {float(item['tv_value']):.6e} | {float(item['solve_seconds']):.3f}"
        )
    return "\n".join(lines) + "\n"


def run_comparison(
    *,
    angle_counts: list[int],
    selected_json_template: str | Path,
    source_json: str | Path,
    methods: list[dict[str, str]],
    noise_mode: str,
    noise_level: float,
    lambda_mode: str,
    lambda_reg: float,
    phantom_seed: int,
    noise_seed: int,
    random_angle_seed: int,
    per_bucket_keep: int,
    beam_size: int,
    lambda_uniform: float,
    admm_iters: int,
    admm_cg_iters: int,
    admm_cg_tol: float,
    rho_data: float,
    rho_reg: float,
    output_dir: str | Path,
    output_prefix: str,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(phantom_seed))
    np.random.seed(int(phantom_seed))
    base_generator = TheoreticalDataGenerator(data_source="shepp_logan")
    coeff_true = base_generator._sample_coefficients(batch_size=1).to(device=device, dtype=torch.float32)
    coeff_true_np = coeff_true.squeeze(0).squeeze(0).detach().cpu().numpy()

    results: list[dict[str, Any]] = []
    estimates_by_method: dict[str, dict[str, np.ndarray]] = {str(method["name"]): {} for method in methods}
    selected_meta: dict[str, Any] = {}

    for count in angle_counts:
        selected_json = resolve_selected_alpha_json(selected_json_template, int(count))
        records, resolved_json, selected_source = _load_selected_records(
            selected_json=selected_json,
            source_json=source_json,
            count=int(count),
            per_bucket_keep=int(per_bucket_keep),
            beam_size=int(beam_size),
            lambda_uniform=float(lambda_uniform),
        )
        selected_alphas = [float(item["alpha"]) % math.pi for item in records]
        selected_taus = [float(item["tau_star"]) for item in records]
        random_alphas = random_uniform_alphas(int(count), int(random_angle_seed), trial_index=0)
        selected_meta[str(count)] = {
            "resolved_json": str(resolved_json),
            "selected_source": selected_source,
            "selected_records": records,
            "random_alphas": random_alphas,
        }

        case_defs = [
            (f"selected{int(count)}_uniform_condition", selected_alphas, selected_taus, str(resolved_json), selected_source, None),
            (f"random{int(count)}_uniform", random_alphas, None, None, None, int(random_angle_seed)),
        ]
        for method in methods:
            for case_name, alphas, taus, case_selected_json, selected_source_name, case_random_seed in case_defs:
                result, estimate = _solve_case(
                    case_name=case_name,
                    count=int(count),
                    alphas=alphas,
                    tau_offsets=taus,
                    coeff_true=coeff_true,
                    method=method,
                    lambda_mode=lambda_mode,
                    lambda_reg=float(lambda_reg),
                    noise_mode=noise_mode,
                    noise_level=float(noise_level),
                    noise_seed=int(noise_seed),
                    random_angle_seed=case_random_seed,
                    selected_json=case_selected_json,
                    selected_source=selected_source_name,
                    admm_iters=int(admm_iters),
                    admm_cg_iters=int(admm_cg_iters),
                    admm_cg_tol=float(admm_cg_tol),
                    rho_data=float(rho_data),
                    rho_reg=float(rho_reg),
                )
                results.append(result)
                estimates_by_method[str(method["name"])][str(case_name)] = estimate
                print(json.dumps(result, ensure_ascii=False), flush=True)

    payload = {
        "meta": {
            "angle_counts": [int(v) for v in angle_counts],
            "selected_json_template": str(selected_json_template),
            "source_json": str(source_json),
            "methods": methods,
            "noise_mode": str(noise_mode),
            "noise_level": float(noise_level),
            "lambda_mode": str(lambda_mode),
            "phantom_seed": int(phantom_seed),
            "noise_seed": int(noise_seed),
            "random_angle_seed": int(random_angle_seed),
            "admm_iters": int(admm_iters),
        },
        "angle_selection": selected_meta,
        "results": results,
    }
    json_path = output_dir / f"{output_prefix}_results.json"
    txt_path = output_dir / f"{output_prefix}_results.txt"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(_format_summary(results), encoding="utf-8")

    for method_name, estimates in estimates_by_method.items():
        safe_method = method_name.replace("/", "_")
        _plot_method_panels(
            coeff_true=coeff_true_np,
            estimates=estimates,
            title=f"Shepp-Logan {method_name} | selected vs random | noise {noise_level:g}",
            save_path=output_dir / f"{output_prefix}_{safe_method}_comparison.png",
        )
        _plot_error_panels(
            coeff_true=coeff_true_np,
            estimates=estimates,
            title=f"Shepp-Logan {method_name} error maps",
            save_path=output_dir / f"{output_prefix}_{safe_method}_error_maps.png",
        )

    print(f"Saved json: {json_path}", flush=True)
    print(f"Saved summary: {txt_path}", flush=True)
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    default_source = Path(DATA_DIR) / "alpha_search_cache" / "alpha_full_resume.json"
    parser = argparse.ArgumentParser(description="Shepp-Logan selected-vs-random angle comparison with configurable angle counts.")
    parser.add_argument("--angle-counts", type=str, default="8,4", help="Comma-separated angle counts, e.g. '8,4' or '16'.")
    parser.add_argument(
        "--selected-json-template",
        type=str,
        default="",
        help="Selected-angle JSON template. Empty means data/alpha_search_cache/alpha_selected{count}.json.",
    )
    parser.add_argument("--source-json", type=str, default=str(default_source), help="Raw alpha search JSON used if selected JSON is missing.")
    parser.add_argument("--per-bucket-keep", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=200)
    parser.add_argument("--lambda-uniform", type=float, default=0.25)
    parser.add_argument(
        "--init-method",
        type=str,
        default="tv",
        help=f"One method alias, comma-separated aliases, or 'all'. Method names: {method_names()!r}.",
    )
    parser.add_argument("--noise-mode", type=str, default="multiplicative", choices=["multiplicative", "additive", "snr"])
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--lambda-mode", type=str, default="morozov", choices=["morozov", "fixed"])
    parser.add_argument("--lambda-reg", type=float, default=1.0e-2)
    parser.add_argument("--phantom-seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--random-angle-seed", type=int, default=20260517)
    parser.add_argument("--admm-iters", type=int, default=80)
    parser.add_argument("--admm-cg-iters", type=int, default=30)
    parser.add_argument("--admm-cg-tol", type=float, default=1.0e-4)
    parser.add_argument("--rho-data", type=float, default=4.0)
    parser.add_argument("--rho-reg", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(RESULTS_DIR) / "shepp_logan_angle_selection_compare"),
    )
    parser.add_argument("--output-prefix", type=str, default="")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    angle_counts = parse_angle_counts(args.angle_counts)
    methods = parse_method_specs(args.init_method)
    noise_tag = str(args.noise_level).replace(".", "p")
    counts_tag = "_".join(str(v) for v in angle_counts)
    method_tag = "all" if str(args.init_method).strip().lower() == "all" else "_".join(item["name"] for item in methods)
    output_prefix = str(args.output_prefix or "").strip() or f"shepp_logan_selected_vs_random_k{counts_tag}_{method_tag}_noise{noise_tag}_admm{int(args.admm_iters)}"

    print(f"Using device: {device}", flush=True)
    print(f"angle_counts={angle_counts} methods={[item['name'] for item in methods]}", flush=True)
    print(f"lambda_mode={args.lambda_mode} admm_iters={args.admm_iters}", flush=True)

    run_comparison(
        angle_counts=angle_counts,
        selected_json_template=args.selected_json_template,
        source_json=args.source_json,
        methods=methods,
        noise_mode=args.noise_mode,
        noise_level=float(args.noise_level),
        lambda_mode=args.lambda_mode,
        lambda_reg=float(args.lambda_reg),
        phantom_seed=int(args.phantom_seed),
        noise_seed=int(args.noise_seed),
        random_angle_seed=int(args.random_angle_seed),
        per_bucket_keep=int(args.per_bucket_keep),
        beam_size=int(args.beam_size),
        lambda_uniform=float(args.lambda_uniform),
        admm_iters=int(args.admm_iters),
        admm_cg_iters=int(args.admm_cg_iters),
        admm_cg_tol=float(args.admm_cg_tol),
        rho_data=float(args.rho_data),
        rho_reg=float(args.rho_reg),
        output_dir=args.output_dir,
        output_prefix=output_prefix,
    )


if __name__ == "__main__":
    main()
