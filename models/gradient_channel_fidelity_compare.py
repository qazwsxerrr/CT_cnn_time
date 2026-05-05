# -*- coding: utf-8 -*-
"""Compare per-angle data-fidelity gradient channels for two sampling schemes.

This diagnostic is intentionally not another stacked Tikhonov recovery score.
It evaluates the channels that are actually fed to the learned optimizer:

    grad_i(c) = 2 A_i^T (A_i c - g_i)

For a fixed current estimate c, it compares noisy-observation gradients against
clean-observation gradients and also measures whether the negative gradient
channel points toward the desired update c_true - c.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional for headless runs
    plt = None

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

import config  # noqa: E402
from image_generator import generate_shepp_logan_phantom  # noqa: E402
from radon_transform import TheoreticalB1B1Operator2D  # noqa: E402


TOP_JSON = MODELS / "top8_condition_constrained_vs_triangular_condition_numbers.json"
OUT_DIR = ROOT / "results" / "gradient_channel_fidelity_compare"


def _flatten_angle_channels(x: torch.Tensor) -> torch.Tensor:
    """Return a tensor with shape (B,K,D) from per-angle image channels."""
    if x.dim() == 5 and int(x.shape[2]) == 1:
        return x.detach().to(dtype=torch.float64, device="cpu").reshape(int(x.shape[0]), int(x.shape[1]), -1)
    if x.dim() == 4:
        return x.detach().to(dtype=torch.float64, device="cpu").reshape(int(x.shape[0]), int(x.shape[1]), -1)
    raise ValueError(f"Expected per-angle tensor with shape (B,K,1,H,W) or (B,K,H,W), got {tuple(x.shape)}")


def _float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().to(dtype=torch.float64, device="cpu").view(-1).tolist()]


def compute_gradient_channel_metrics(
    grad_clean_pa: torch.Tensor,
    grad_noisy_pa: torch.Tensor,
    eps: float = 1.0e-12,
) -> Dict[str, object]:
    """Measure noisy-vs-clean fidelity of per-angle gradient channels.

    Args:
        grad_clean_pa: clean per-angle gradients, shape (B,K,1,H,W).
        grad_noisy_pa: noisy per-angle gradients, same shape as grad_clean_pa.
        eps: numerical floor for divisions.

    Returns:
        JSON-serializable metrics with per-angle and aggregate values.
    """
    clean = _flatten_angle_channels(grad_clean_pa)
    noisy = _flatten_angle_channels(grad_noisy_pa)
    if tuple(clean.shape) != tuple(noisy.shape):
        raise ValueError(f"Gradient shape mismatch: clean={tuple(clean.shape)}, noisy={tuple(noisy.shape)}")

    diff = noisy - clean
    clean_norm = torch.linalg.vector_norm(clean, dim=2)
    noisy_norm = torch.linalg.vector_norm(noisy, dim=2)
    diff_norm = torch.linalg.vector_norm(diff, dim=2)
    dot = torch.sum(clean * noisy, dim=2)
    denom = (clean_norm * noisy_norm).clamp_min(float(eps))
    cosine = torch.clamp(dot / denom, min=-1.0, max=1.0)
    rel_error = diff_norm / clean_norm.clamp_min(float(eps))
    snr_db = 20.0 * torch.log10(clean_norm.clamp_min(float(eps)) / diff_norm.clamp_min(float(eps)))

    rel_flat = rel_error.reshape(-1)
    cos_flat = cosine.reshape(-1)
    snr_flat = snr_db.reshape(-1)
    return {
        "batch": int(clean.shape[0]),
        "num_angles": int(clean.shape[1]),
        "clean_norm_per_angle": _float_list(clean_norm[0]),
        "noisy_norm_per_angle": _float_list(noisy_norm[0]),
        "diff_norm_per_angle": _float_list(diff_norm[0]),
        "relative_error_per_angle": _float_list(rel_error[0]),
        "cosine_per_angle": _float_list(cosine[0]),
        "snr_db_per_angle": _float_list(snr_db[0]),
        "relative_error_mean": float(torch.mean(rel_flat).item()),
        "relative_error_median": float(torch.median(rel_flat).item()),
        "cosine_mean": float(torch.mean(cos_flat).item()),
        "cosine_median": float(torch.median(cos_flat).item()),
        "snr_db_mean": float(torch.mean(snr_flat).item()),
        "snr_db_median": float(torch.median(snr_flat).item()),
    }


def compute_channel_target_metrics(
    channel_pa: torch.Tensor,
    target_update: torch.Tensor,
    eps: float = 1.0e-12,
) -> Dict[str, object]:
    """Measure how well per-angle channels align with a desired coefficient update.

    The learned optimizer sees the raw channel and can learn a scale, so this
    reports both cosine and the best scalar-fit residual per angle.
    """
    channel = _flatten_angle_channels(channel_pa)
    target = target_update.detach().to(dtype=torch.float64, device="cpu").reshape(int(target_update.shape[0]), -1)
    if int(channel.shape[0]) != int(target.shape[0]):
        raise ValueError(f"Batch mismatch: channel={tuple(channel.shape)}, target={tuple(target.shape)}")

    target = target.unsqueeze(1).expand(-1, int(channel.shape[1]), -1)
    channel_norm = torch.linalg.vector_norm(channel, dim=2)
    target_norm = torch.linalg.vector_norm(target, dim=2)
    dot = torch.sum(channel * target, dim=2)
    cosine = torch.clamp(dot / (channel_norm * target_norm).clamp_min(float(eps)), min=-1.0, max=1.0)
    scale = dot / torch.sum(channel * channel, dim=2).clamp_min(float(eps))
    fitted = channel * scale.unsqueeze(2)
    best_scaled_res = torch.linalg.vector_norm(fitted - target, dim=2) / target_norm.clamp_min(float(eps))

    cos_flat = cosine.reshape(-1)
    res_flat = best_scaled_res.reshape(-1)
    return {
        "target_norm_per_angle": _float_list(target_norm[0]),
        "channel_norm_per_angle": _float_list(channel_norm[0]),
        "cosine_per_angle": _float_list(cosine[0]),
        "best_scale_per_angle": _float_list(scale[0]),
        "best_scaled_res_per_angle": _float_list(best_scaled_res[0]),
        "cosine_mean": float(torch.mean(cos_flat).item()),
        "cosine_median": float(torch.median(cos_flat).item()),
        "best_scaled_res_mean": float(torch.mean(res_flat).item()),
        "best_scaled_res_median": float(torch.median(res_flat).item()),
    }


def load_rows(path: Path = TOP_JSON) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("rows", None)
    if not isinstance(rows, list) or len(rows) < 8:
        raise ValueError(f"{path} must contain at least 8 rows under key 'rows'.")
    return [dict(row) for row in rows[:8]]


def build_operator(mode: str, rows: List[dict]) -> TheoreticalB1B1Operator2D:
    betas = [tuple(int(v) for v in row["beta"]) for row in rows]
    if mode == "condition_constrained":
        taus = [float(row["condition_constrained"]["tau"]) for row in rows]
        return TheoreticalB1B1Operator2D(
            beta_vectors=betas,
            height=config.IMAGE_SIZE,
            width=config.IMAGE_SIZE,
            t0=0.5,
            formula_mode="condition_constrained_offset",
            auto_shift_t0=False,
            t0_per_angle=taus,
        ).to(config.device)
    if mode == "triangular":
        return TheoreticalB1B1Operator2D(
            beta_vectors=betas,
            height=config.IMAGE_SIZE,
            width=config.IMAGE_SIZE,
            t0=0.5,
            formula_mode="legacy_injective_extension",
            auto_shift_t0=True,
        ).to(config.device)
    raise ValueError(f"Unsupported mode={mode!r}")


def build_single_angle_operators(mode: str, rows: List[dict]) -> List[TheoreticalB1B1Operator2D]:
    operators = []
    for row in rows:
        beta = tuple(int(v) for v in row["beta"])
        if mode == "condition_constrained":
            operators.append(
                TheoreticalB1B1Operator2D(
                    beta_vectors=[beta],
                    height=config.IMAGE_SIZE,
                    width=config.IMAGE_SIZE,
                    t0=0.5,
                    formula_mode="condition_constrained_offset",
                    auto_shift_t0=False,
                    t0_per_angle=[float(row["condition_constrained"]["tau"])],
                ).to(config.device)
            )
        elif mode == "triangular":
            operators.append(
                TheoreticalB1B1Operator2D(
                    beta_vectors=[beta],
                    height=config.IMAGE_SIZE,
                    width=config.IMAGE_SIZE,
                    t0=0.5,
                    formula_mode="legacy_injective_extension",
                    auto_shift_t0=True,
                ).to(config.device)
            )
        else:
            raise ValueError(f"Unsupported mode={mode!r}")
    return operators


def apply_noise_for_diagnostic(
    g_clean: torch.Tensor,
    *,
    noise_mode: str,
    noise_level: float,
    target_snr_db: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate noisy data for offline diagnostics.

    The compared gradient extraction method receives only g_noisy.  The
    noise parameters are used here only to create reproducible benchmark data.
    """
    noise_mode = str(noise_mode).strip().lower()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if noise_mode == "multiplicative":
        noise = float(noise_level) * g_clean * (2.0 * torch.rand_like(g_clean) - 1.0)
    elif noise_mode == "additive":
        noise = float(noise_level) * torch.randn_like(g_clean)
    elif noise_mode == "snr":
        signal_energy = torch.sum(g_clean.square(), dim=1, keepdim=True)
        sigma = torch.sqrt(signal_energy / (g_clean.shape[1] * (10.0 ** (float(target_snr_db) / 10.0))))
        noise = torch.randn_like(g_clean) * sigma.to(g_clean)
    else:
        raise ValueError("noise_mode must be one of: multiplicative, additive, snr")
    return g_clean + noise, noise


@torch.no_grad()
def solve_shifted_single_angle_normal_cg(
    op: TheoreticalB1B1Operator2D,
    rhs: torch.Tensor,
    *,
    rho: float,
    max_iter: int,
    tol: float,
    x0: torch.Tensor = None,
) -> torch.Tensor:
    """Solve (2 A_i^T A_i + rho I)x = rhs by CG."""
    if x0 is None:
        x = torch.zeros_like(rhs)
    else:
        x = x0.to(dtype=rhs.dtype, device=rhs.device).clone()

    rho = float(rho)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return 2.0 * op.apply_normal(v) + rho * v

    r = rhs - matvec(x)
    p = r.clone()
    rr = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)
    eps = rhs.new_tensor(1.0e-12)
    for _ in range(int(max_iter)):
        ap = matvec(p)
        alpha = rr / torch.sum(p * ap, dim=(1, 2, 3), keepdim=True).clamp_min(eps)
        x = x + alpha * p
        r = r - alpha * ap
        rr_new = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)
        if torch.sqrt(rr_new.max()).item() < float(tol):
            break
        p = r + (rr_new / rr.clamp_min(eps)) * p
        rr = rr_new
    return x


def make_current_estimates(coeff_true: torch.Tensor, seed: int) -> Dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    perturb = torch.randn_like(coeff_true)
    perturb = perturb / torch.linalg.vector_norm(perturb.reshape(perturb.shape[0], -1), dim=1).view(-1, 1, 1, 1)
    true_norm = torch.linalg.vector_norm(coeff_true.reshape(coeff_true.shape[0], -1), dim=1).view(-1, 1, 1, 1)
    perturb = 0.20 * true_norm * perturb
    return {
        "zero": torch.zeros_like(coeff_true),
        "true_plus_20pct_random_perturb": coeff_true + perturb,
    }


@torch.no_grad()
def evaluate_one_mode(
    *,
    mode: str,
    rows: List[dict],
    coeff_true: torch.Tensor,
    current_estimates: Dict[str, torch.Tensor],
    noise_mode: str,
    noise_level: float,
    target_snr_db: float,
    seed: int,
    output_dir: Path,
    make_plots: bool,
) -> dict:
    started = time.perf_counter()
    op = build_operator(mode, rows)
    g_clean = op.forward(coeff_true)
    g_noisy, noise = apply_noise_for_diagnostic(
        g_clean,
        noise_mode=noise_mode,
        noise_level=float(noise_level),
        target_snr_db=float(target_snr_db),
        seed=int(seed),
    )
    g_clean_pa = op.split_measurements(g_clean)
    g_noisy_pa = op.split_measurements(g_noisy)
    noise_pa = op.split_measurements(noise)
    noise_grad_pa = 2.0 * op.adjoint_per_angle(noise_pa)

    cases = []
    for case_name, coeff_current in current_estimates.items():
        g_pred_pa = op.forward_per_angle(coeff_current)
        grad_clean_pa = 2.0 * op.adjoint_per_angle(g_pred_pa - g_clean_pa)
        grad_noisy_pa = 2.0 * op.adjoint_per_angle(g_pred_pa - g_noisy_pa)
        target_update = coeff_true - coeff_current
        noisy_descent = -grad_noisy_pa
        clean_descent = -grad_clean_pa

        grad_metrics = compute_gradient_channel_metrics(grad_clean_pa, grad_noisy_pa)
        clean_target_metrics = compute_channel_target_metrics(clean_descent, target_update)
        noisy_target_metrics = compute_channel_target_metrics(noisy_descent, target_update)
        noise_metrics = compute_channel_target_metrics(-noise_grad_pa, target_update)

        plot_paths = []
        if make_plots:
            plot_path = output_dir / f"{mode}_{case_name}_{noise_mode}_{str(noise_level).replace('.', '_')}_channels.png"
            if plot_gradient_channel_mosaic(
                coeff_true=coeff_true,
                target_update=target_update,
                clean_descent_pa=clean_descent,
                noisy_descent_pa=noisy_descent,
                path=plot_path,
                title=f"{mode} | {case_name} | {noise_mode}={noise_level}",
            ):
                plot_paths.append(str(plot_path))

        cases.append(
            {
                "case": case_name,
                "gradient_noisy_vs_clean": grad_metrics,
                "clean_descent_vs_target": clean_target_metrics,
                "noisy_descent_vs_target": noisy_target_metrics,
                "noise_descent_vs_target_debug": noise_metrics,
                "plot_paths": plot_paths,
            }
        )

    result = {
        "mode": mode,
        "betas": [list(row["beta"]) for row in rows],
        "formula_mode": str(op.formula_mode),
        "uses_sparse_blocks": bool(op.uses_sparse_blocks),
        "band_t0_per_angle": _float_list(op.band_t0_per_angle),
        "condition_numbers": [
            float(row["condition_constrained" if mode == "condition_constrained" else "triangular"]["cond"])
            for row in rows
        ],
        "noise_norm": float(torch.linalg.vector_norm(noise.reshape(noise.shape[0], -1), dim=1)[0].item()),
        "clean_measurement_norm": float(torch.linalg.vector_norm(g_clean.reshape(g_clean.shape[0], -1), dim=1)[0].item()),
        "cases": cases,
        "seconds": float(time.perf_counter() - started),
    }
    del op
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _normalize_image_for_plot(x: torch.Tensor) -> np.ndarray:
    arr = x.squeeze().detach().to(dtype=torch.float32, device="cpu").numpy()
    denom = float(np.max(np.abs(arr)))
    if denom <= 1.0e-12:
        return arr
    return arr / denom


def plot_gradient_channel_mosaic(
    *,
    coeff_true: torch.Tensor,
    target_update: torch.Tensor,
    clean_descent_pa: torch.Tensor,
    noisy_descent_pa: torch.Tensor,
    path: Path,
    title: str,
) -> bool:
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    num_angles = int(noisy_descent_pa.shape[1])
    cols = num_angles
    fig, axes = plt.subplots(4, cols, figsize=(2.2 * cols, 8.2), squeeze=False)
    rows = [
        ("true", coeff_true.expand(-1, num_angles, -1, -1, -1)),
        ("target", target_update.expand(-1, num_angles, -1, -1, -1)),
        ("clean -grad", clean_descent_pa),
        ("noisy -grad", noisy_descent_pa),
    ]
    for row_idx, (label, tensor) in enumerate(rows):
        for angle_idx in range(cols):
            ax = axes[row_idx][angle_idx]
            arr = _normalize_image_for_plot(tensor[0, angle_idx] if tensor.dim() == 5 else tensor[0])
            im = ax.imshow(arr, cmap="coolwarm" if row_idx > 0 else "gray", origin="lower", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"angle {angle_idx}", fontsize=8)
            if angle_idx == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def write_reports(payload: dict, output_dir: Path, tag: str) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"gradient_channel_fidelity_{tag}.json"
    md_path = output_dir / f"gradient_channel_fidelity_{tag}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Per-angle gradient channel fidelity comparison",
        "",
        f"- noise mode used only for benchmark generation: `{payload['meta']['noise_mode']}`",
        f"- noise level used only for benchmark generation: `{payload['meta']['noise_level']}`",
        f"- target SNR dB used only when noise_mode=snr: `{payload['meta']['target_snr_db']}`",
        f"- random seed: `{payload['meta']['seed']}`",
        "",
        "## Summary",
        "",
        "| mode | case | cond mean | rel err noisy-vs-clean ↓ | cosine noisy-vs-clean ↑ | grad SNR dB ↑ | noisy -grad vs target cosine ↑ | noisy -grad best-scaled RES ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in payload["results"]:
        cond_mean = float(np.mean(result["condition_numbers"]))
        for case in result["cases"]:
            gm = case["gradient_noisy_vs_clean"]
            tm = case["noisy_descent_vs_target"]
            lines.append(
                f"| `{result['mode']}` | `{case['case']}` | `{cond_mean:.6e}` | "
                f"`{gm['relative_error_mean']:.6f}` | `{gm['cosine_mean']:.6f}` | "
                f"`{gm['snr_db_mean']:.3f}` | `{tm['cosine_mean']:.6f}` | "
                f"`{tm['best_scaled_res_mean']:.6f}` |"
            )
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- `rel err noisy-vs-clean` measures how much the noisy observation perturbs each per-angle gradient channel.",
            "- `noisy -grad vs target` measures the channel as a learned-optimizer input: a CNN can rescale/mix channels, so the best-scaled residual is often more relevant than raw Tikhonov RES.",
            "- The experiment does not pass noise kind or noise level to the extraction method; they are only used to generate this offline benchmark.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def parse_args(argv: Iterable[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-mode", choices=["multiplicative", "additive", "snr"], default="multiplicative")
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--target-snr-db", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] = None) -> int:
    args = parse_args(argv)
    rows = load_rows()
    coeff_true = generate_shepp_logan_phantom(
        image_size=config.IMAGE_SIZE,
        device=config.device,
        dtype=torch.float32,
    ).view(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    current_estimates = make_current_estimates(coeff_true, seed=int(args.seed) + 17)
    tag = f"{args.noise_mode}_{str(args.noise_level).replace('.', '_')}_seed_{args.seed}"

    print(
        json.dumps(
            {
                "event": "start",
                "noise_mode": args.noise_mode,
                "noise_level": float(args.noise_level),
                "seed": int(args.seed),
                "device": str(config.device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results = []
    for mode in ("condition_constrained", "triangular"):
        print(json.dumps({"event": "mode_start", "mode": mode}, ensure_ascii=False), flush=True)
        result = evaluate_one_mode(
            mode=mode,
            rows=rows,
            coeff_true=coeff_true,
            current_estimates=current_estimates,
            noise_mode=str(args.noise_mode),
            noise_level=float(args.noise_level),
            target_snr_db=float(args.target_snr_db),
            seed=int(args.seed),
            output_dir=OUT_DIR,
            make_plots=not bool(args.no_plots),
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "mode_done",
                    "mode": mode,
                    "seconds": result["seconds"],
                    "case_summary": [
                        {
                            "case": case["case"],
                            "rel_error_mean": case["gradient_noisy_vs_clean"]["relative_error_mean"],
                            "noisy_target_cosine_mean": case["noisy_descent_vs_target"]["cosine_mean"],
                            "noisy_target_res_mean": case["noisy_descent_vs_target"]["best_scaled_res_mean"],
                        }
                        for case in result["cases"]
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "meta": {
            "method": "per-angle data-fidelity gradient channel noisy-vs-clean diagnostic",
            "noise_mode": str(args.noise_mode),
            "noise_level": float(args.noise_level),
            "target_snr_db": float(args.target_snr_db),
            "seed": int(args.seed),
            "top_json": str(TOP_JSON),
        },
        "results": results,
    }
    json_path, md_path = write_reports(payload, OUT_DIR, tag)
    print(json.dumps({"event": "wrote", "json": str(json_path), "md": str(md_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
