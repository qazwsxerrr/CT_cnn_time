# -*- coding: utf-8 -*-
r"""Plot condition-selected angle-count reconstructions without random baselines.

For each reconstruction method, this script solves the Shepp-Logan selected
angle cases for several angle counts and writes one row figure:

``true | k=4 | k=8 | k=10 | k=12 | k=16``

Each reconstructed panel title includes the coefficient relative error
(``RES``).
"""

from __future__ import annotations

import argparse
import json
import sys
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
for path in (str(THIS_DIR), str(MODELS_DIR), str(ALPHA_CONDITION_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from compare_angle_selection import (  # noqa: E402
    _format_summary,
    _load_selected_records,
    _solve_case,
    parse_angle_counts,
    parse_method_specs,
    resolve_selected_alpha_json,
)
from config import DATA_DIR, RESULTS_DIR, device  # noqa: E402
from radon_transform import TheoreticalDataGenerator  # noqa: E402


def selected_case_name(count: int) -> str:
    return f"selected{int(count)}_uniform_condition"


def format_panel_title(count: int, coeff_res: float) -> str:
    return f"k={int(count)}\nRES={float(coeff_res):.6f}"


def safe_method_filename(method_name: str) -> str:
    return str(method_name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def plot_method_angle_count_panels(
    *,
    coeff_true: np.ndarray,
    angle_counts: list[int],
    estimates_by_count: dict[int, np.ndarray],
    results_by_count: dict[int, dict[str, Any]],
    method_name: str,
    title: str,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, np.ndarray]] = [("true", coeff_true)]
    for count in angle_counts:
        result = results_by_count[int(count)]
        panels.append((format_panel_title(int(count), float(result["coeff_res"])), estimates_by_count[int(count)]))

    vmin = min(float(np.min(arr)) for _, arr in panels)
    vmax = max(float(np.max(arr)) for _, arr in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (panel_title, arr) in zip(axes, panels):
        ax.imshow(arr, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(panel_title)
        ax.axis("off")
    fig.suptitle(f"{title} | {method_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_selected_angle_count_plots(
    *,
    angle_counts: list[int],
    selected_json_template: str | Path,
    source_json: str | Path,
    methods: list[dict[str, str]],
    noise_mode: str,
    noise_level: float,
    lambda_mode: str,
    lambda_reg: float,
    morozov_form: str,
    phantom_seed: int,
    noise_seed: int,
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

    selected_meta: dict[str, Any] = {}
    angle_defs: dict[int, tuple[list[float], list[float], Path, str, list[dict[str, Any]]]] = {}
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
        selected_alphas = [float(item["alpha"]) % np.pi for item in records]
        selected_taus = [float(item["tau_star"]) for item in records]
        angle_defs[int(count)] = (selected_alphas, selected_taus, resolved_json, selected_source, records)
        selected_meta[str(int(count))] = {
            "resolved_json": str(resolved_json),
            "selected_source": selected_source,
            "selected_records": records,
        }

    results: list[dict[str, Any]] = []
    estimates_by_method: dict[str, dict[int, np.ndarray]] = {str(method["name"]): {} for method in methods}
    results_by_method: dict[str, dict[int, dict[str, Any]]] = {str(method["name"]): {} for method in methods}

    for method in methods:
        method_name = str(method["name"])
        for count in angle_counts:
            alphas, taus, resolved_json, selected_source, _records = angle_defs[int(count)]
            result, estimate = _solve_case(
                case_name=selected_case_name(int(count)),
                count=int(count),
                alphas=alphas,
                tau_offsets=taus,
                coeff_true=coeff_true,
                method=method,
                lambda_mode=lambda_mode,
                lambda_reg=float(lambda_reg),
                morozov_form=morozov_form,
                noise_mode=noise_mode,
                noise_level=float(noise_level),
                noise_seed=int(noise_seed),
                random_angle_seed=None,
                selected_json=str(resolved_json),
                selected_source=selected_source,
                admm_iters=int(admm_iters),
                admm_cg_iters=int(admm_cg_iters),
                admm_cg_tol=float(admm_cg_tol),
                rho_data=float(rho_data),
                rho_reg=float(rho_reg),
            )
            results.append(result)
            estimates_by_method[method_name][int(count)] = estimate
            results_by_method[method_name][int(count)] = result
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
            "morozov_form": str(morozov_form),
            "phantom_seed": int(phantom_seed),
            "noise_seed": int(noise_seed),
            "admm_iters": int(admm_iters),
            "plot_layout": f"1x{len(angle_counts) + 1}",
        },
        "angle_selection": selected_meta,
        "results": results,
    }
    json_path = output_dir / f"{output_prefix}_results.json"
    txt_path = output_dir / f"{output_prefix}_results.txt"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(_format_summary(results), encoding="utf-8")

    for method in methods:
        method_name = str(method["name"])
        safe_method = safe_method_filename(method_name)
        plot_method_angle_count_panels(
            coeff_true=coeff_true_np,
            angle_counts=angle_counts,
            estimates_by_count=estimates_by_method[method_name],
            results_by_count=results_by_method[method_name],
            method_name=method_name,
            title=f"Shepp-Logan selected angle counts | noise {noise_level:g}",
            save_path=output_dir / f"{output_prefix}_{safe_method}_selected_counts.png",
        )

    print(f"Saved json: {json_path}", flush=True)
    print(f"Saved summary: {txt_path}", flush=True)
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    default_source = Path(DATA_DIR) / "alpha_search_cache" / "alpha_full_resume.json"
    parser = argparse.ArgumentParser(description="Plot selected angle-count Shepp-Logan reconstructions without random baselines.")
    parser.add_argument("--angle-counts", type=str, default="4,8,10,12,16")
    parser.add_argument(
        "--selected-json-template",
        type=str,
        default="",
        help="Selected-angle JSON template. Empty means data/alpha_search_cache/alpha_selected{count}.json.",
    )
    parser.add_argument("--source-json", type=str, default=str(default_source))
    parser.add_argument("--per-bucket-keep", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=200)
    parser.add_argument("--lambda-uniform", type=float, default=0.25)
    parser.add_argument("--init-method", type=str, default="all")
    parser.add_argument("--noise-mode", type=str, default="multiplicative", choices=["multiplicative", "additive", "snr"])
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--lambda-mode", type=str, default="morozov", choices=["morozov", "fixed"])
    parser.add_argument("--lambda-reg", type=float, default=1.0e-2)
    parser.add_argument("--morozov-form", type=str, default="constrained", choices=["constrained", "regularized"])
    parser.add_argument("--phantom-seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--admm-iters", type=int, default=80)
    parser.add_argument("--admm-cg-iters", type=int, default=30)
    parser.add_argument("--admm-cg-tol", type=float, default=1.0e-4)
    parser.add_argument("--rho-data", type=float, default=4.0)
    parser.add_argument("--rho-reg", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(RESULTS_DIR) / "shepp_logan_selected_angle_counts"),
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
    output_prefix = str(args.output_prefix or "").strip() or f"shepp_logan_selected_counts_k{counts_tag}_{method_tag}_noise{noise_tag}_admm{int(args.admm_iters)}"

    print(f"Using device: {device}", flush=True)
    print(f"angle_counts={angle_counts} methods={[item['name'] for item in methods]}", flush=True)
    print(f"selected-only morozov_form={args.morozov_form} lambda_mode={args.lambda_mode} admm_iters={args.admm_iters}", flush=True)

    run_selected_angle_count_plots(
        angle_counts=angle_counts,
        selected_json_template=args.selected_json_template,
        source_json=args.source_json,
        methods=methods,
        noise_mode=args.noise_mode,
        noise_level=float(args.noise_level),
        lambda_mode=args.lambda_mode,
        lambda_reg=float(args.lambda_reg),
        morozov_form=args.morozov_form,
        phantom_seed=int(args.phantom_seed),
        noise_seed=int(args.noise_seed),
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
