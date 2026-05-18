"""Evaluate the frozen alpha16 -> trained extra8 cascade."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(__file__).resolve().parents[1]
DEEP_LEARN_DIR = MODELS_DIR / "deep_learn"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (THIS_DIR, DEEP_LEARN_DIR, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from angle_selection import write_selection_outputs


def _prepare_angle_files() -> dict[str, Path]:
    cache_dir = PROJECT_ROOT / "data" / "alpha_search_cache"
    full_json = Path(os.environ.get("ALPHA_FULL_JSON_OVERRIDE", cache_dir / "alpha_full_resume.json"))
    original16_json = Path(os.environ.get("ALPHA16_JSON_OVERRIDE", cache_dir / "alpha_selected16.json"))
    output_dir = Path(os.environ.get("ANGLE_SELECTION_OUTPUT_DIR_OVERRIDE", cache_dir))
    extra8_json = Path(os.environ.get("EXTRA8_JSON_OVERRIDE", output_dir / "alpha_extra8_from24_excluding16.json"))
    comparison_json = output_dir / "alpha_selected16_no_exclude_comparison.json"
    selected24_json = output_dir / "alpha_selected24_no_exclude.json"
    generated16_json = output_dir / "alpha_selected16_no_exclude_regenerated.json"
    if not extra8_json.exists() or not comparison_json.exists() or not selected24_json.exists():
        paths = write_selection_outputs(
            full_json=full_json,
            selected16_json=original16_json,
            output_dir=output_dir,
            generated16_name=generated16_json.name,
            selected24_name=selected24_json.name,
            extra8_name=extra8_json.name,
            comparison_name=comparison_json.name,
        )
    else:
        paths = {
            "generated16": generated16_json,
            "selected24": selected24_json,
            "extra8": extra8_json,
            "comparison": comparison_json,
        }
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    if not bool(comparison.get("all_match", False)):
        raise RuntimeError(f"Regenerated no-exclusion alpha16 does not match reference. See {paths['comparison']}.")
    return paths


ANGLE_PATHS = _prepare_angle_files()

os.environ.setdefault("EXPERIMENT_PROFILE_OVERRIDE", "alpha_condition")
os.environ.setdefault("ALPHA_CONDITION_JSON_OVERRIDE", str(ANGLE_PATHS["extra8"]))
os.environ.setdefault("MULTI_ANGLE_SOLVER_MODE_OVERRIDE", "stacked_tikhonov")
os.environ.setdefault("THEORETICAL_FORMULA_MODE_OVERRIDE", "alpha_continuous")
os.environ.setdefault("INIT_METHOD_OVERRIDE", "tikhonov_direct")
os.environ.setdefault("LAMBDA_SELECT_MODE_OVERRIDE", "morozov")
os.environ.setdefault("ALPHA_GRAM_CACHE_DIR_OVERRIDE", str(PROJECT_ROOT / "data" / "alpha_gram_cache"))
os.environ.setdefault("CNN_ANGLE_INDICES_OVERRIDE", "0,1,2,3,4,5,6,7")
os.environ.setdefault("CNN_NUM_ANGLES_OVERRIDE", "8")
os.environ.setdefault("PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE", "1")
os.environ.setdefault("PHYSICS_RESIDUAL_MODE_OVERRIDE", "per_angle_cg")
os.environ.setdefault("PHYSICS_RESIDUAL_DAMPING_OVERRIDE", "1e-2")
os.environ.setdefault("PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE", "8")
os.environ.setdefault("PHYSICS_RESIDUAL_DETACH_OVERRIDE", "1")
os.environ.setdefault("PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE", "1")
os.environ.setdefault("PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE", "1")
os.environ.setdefault("PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE", "0.05")
os.environ.setdefault("PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE", "0.25")
os.environ.setdefault("NOISE_MODE_OVERRIDE", "multiplicative")
os.environ.setdefault("NOISE_LEVEL_OVERRIDE", "0.1")
os.environ.setdefault("BASE_LR_OVERRIDE", "0.001")
os.environ.setdefault("OUTPUT_TAG_OVERRIDE", "alpha16_plus_extra8_continue_grad_phys_morozov_direct_noise01")

import matplotlib.pyplot as plt
import numpy as np
import torch

from cascade_data import (
    CascadeBatchGenerator,
    build_model_for_alpha_json,
    configure_alpha_condition_runtime,
    normalize_runtime_path,
)
from config import BEST_MODEL_PATH, DATA_CONFIG, RESULTS_DIR, device


def _resolve_stage1_checkpoint_path() -> str:
    stage1_tag = "alpha16_even8_grad_phys_morozov_direct_noise01"
    default_path = PROJECT_ROOT / "checkpoints" / stage1_tag / (
        "theoretical_ct_alpha16_even8_grad_phys_morozov_direct_noise01_best_model.pth"
    )
    legacy_path = PROJECT_ROOT / "checkpoints" / "deep_learn" / (
        "theoretical_ct_alpha16_even8_grad_phys_morozov_direct_noise01_best_model.pth"
    )
    raw_override = os.environ.get("STAGE1_CHECKPOINT_PATH_OVERRIDE", "")
    candidates = [raw_override] if str(raw_override).strip() else [str(default_path), str(legacy_path)]
    if str(raw_override).strip():
        candidates.append(str(legacy_path))
    for candidate in candidates:
        path = normalize_runtime_path(candidate)
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Stage1 checkpoint not found. Tried: "
        + ", ".join(str(candidate) for candidate in candidates if str(candidate).strip())
    )


def _resolve_stage2_checkpoint_path() -> str:
    path = normalize_runtime_path(os.environ.get("STAGE2_CHECKPOINT_PATH_OVERRIDE", BEST_MODEL_PATH))
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Stage2 checkpoint not found: {path or BEST_MODEL_PATH}")
    return path


def _to_image(tensor: torch.Tensor, index: int = 0) -> np.ndarray:
    item = tensor.detach().cpu()[index]
    while item.dim() > 2:
        item = item.squeeze(0)
    return item.numpy()


def _relative_error(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.norm(pred - truth) / torch.norm(truth).clamp_min(1.0e-12))


def _mse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.mean((pred - truth) ** 2))


def _psnr(pred: torch.Tensor, truth: torch.Tensor) -> float:
    mse = torch.mean((pred - truth) ** 2).clamp_min(1.0e-12)
    data_range = (truth.max() - truth.min()).clamp_min(1.0e-6)
    return float(20.0 * torch.log10(data_range / torch.sqrt(mse)))


def _data_fidelity(operator, coeff: torch.Tensor, observed: torch.Tensor) -> float:
    pred = operator(coeff)
    return float(torch.norm(pred - observed, dim=-1).mean())


def compute_metrics(stage1_model, stage2_model, batch: dict, coeff_stage2: torch.Tensor) -> list[dict[str, float | str]]:
    coeff_true = batch["coeff_true"].to(device)
    coeff_initial16 = batch["coeff_initial16"].to(device)
    coeff_stage1 = batch["coeff_stage1"].to(device)
    g16 = batch["g16_observed"].to(device)
    g8 = batch["g8_observed"].to(device)
    rows = []
    for name, coeff in (
        ("alpha16_tikhonov_c0", coeff_initial16),
        ("stage1_alpha16_c1", coeff_stage1),
        ("stage2_extra8_c2", coeff_stage2),
    ):
        rows.append(
            {
                "name": name,
                "mse": _mse(coeff, coeff_true),
                "relative_error": _relative_error(coeff, coeff_true),
                "psnr": _psnr(coeff, coeff_true),
                "data_fidelity_g16": _data_fidelity(stage1_model.optimizer.operator, coeff, g16),
                "data_fidelity_g8": _data_fidelity(stage2_model.optimizer.operator, coeff, g8),
            }
        )
    return rows


def save_sample_figure(batch: dict, coeff_stage2: torch.Tensor, output_path: Path):
    coeff_true = batch["coeff_true"].to(device)
    coeff_initial16 = batch["coeff_initial16"].to(device)
    coeff_stage1 = batch["coeff_stage1"].to(device)
    panels = [
        ("True", coeff_true),
        ("c0: alpha16 Tikhonov", coeff_initial16),
        ("c1: stage1 alpha16", coeff_stage1),
        ("c2: stage2 extra8", coeff_stage2),
        ("|c0-true|", torch.abs(coeff_initial16 - coeff_true)),
        ("|c1-true|", torch.abs(coeff_stage1 - coeff_true)),
        ("|c2-true|", torch.abs(coeff_stage2 - coeff_true)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.ravel()
    for ax, (title, tensor) in zip(axes_flat, panels):
        im = ax.imshow(_to_image(tensor), cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes_flat[-1].axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(num_samples: int, output_dir: Path):
    stage1_json = Path(os.environ.get("ALPHA16_JSON_OVERRIDE", PROJECT_ROOT / "data" / "alpha_search_cache" / "alpha_selected16.json"))
    stage1_model, _ = build_model_for_alpha_json(
        alpha_json=stage1_json,
        cnn_angle_indices=os.environ.get("STAGE1_CNN_ANGLE_INDICES_OVERRIDE", "0,2,4,6,8,10,12,14"),
        cnn_num_angles=int(os.environ.get("STAGE1_CNN_NUM_ANGLES_OVERRIDE", "8")),
        checkpoint_path=_resolve_stage1_checkpoint_path(),
        frozen=True,
    )
    stage2_model, _ = build_model_for_alpha_json(
        alpha_json=ANGLE_PATHS["extra8"],
        cnn_angle_indices=os.environ.get("CNN_ANGLE_INDICES_OVERRIDE", "0,1,2,3,4,5,6,7"),
        cnn_num_angles=int(os.environ.get("CNN_NUM_ANGLES_OVERRIDE", "8")),
        checkpoint_path=_resolve_stage2_checkpoint_path(),
        frozen=False,
    )
    stage2_model.eval()
    configure_alpha_condition_runtime(
        alpha_json=ANGLE_PATHS["extra8"],
        cnn_angle_indices=os.environ.get("CNN_ANGLE_INDICES_OVERRIDE", "0,1,2,3,4,5,6,7"),
        cnn_num_angles=int(os.environ.get("CNN_NUM_ANGLES_OVERRIDE", "8")),
    )
    data_source = str(os.environ.get("TEST_DATA_SOURCE_OVERRIDE", DATA_CONFIG.get("test_data_source", "shepp_logan"))).strip().lower()
    generator = CascadeBatchGenerator(stage1_model=stage1_model, stage2_model=stage2_model, data_source=data_source)
    batch = generator.generate_batch(batch_size=int(num_samples), random_seed=int(os.environ.get("TEST_SEED_OVERRIDE", "123")))
    with torch.no_grad():
        coeff_stage2, _, _ = stage2_model(batch["coeff_stage1"].to(device), batch["g8_observed"].to(device))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(stage1_model, stage2_model, batch, coeff_stage2)
    csv_path = output_dir / "metrics.csv"
    json_path = output_dir / "metrics.json"
    figure_path = output_dir / "cascade_reconstruction.png"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_sample_figure(batch, coeff_stage2, figure_path)
    return {"metrics_csv": csv_path, "metrics_json": json_path, "figure": figure_path}


def main(argv: list[str] | None = None):
    default_output_dir = Path(RESULTS_DIR) / "alpha16_plus_extra8_continue_grad_phys_morozov_direct_noise01_eval"
    parser = argparse.ArgumentParser(description="Evaluate alpha16 -> extra8 cascade model.")
    parser.add_argument("--num-samples", type=int, default=int(os.environ.get("TEST_NUM_SAMPLES_OVERRIDE", "1")))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    args = parser.parse_args(argv)
    paths = evaluate(num_samples=int(args.num_samples), output_dir=Path(args.output_dir))
    print("Evaluation outputs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
