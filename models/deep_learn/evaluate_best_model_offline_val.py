from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
PROJECT_ROOT = MODELS_DIR.parent
for candidate in (THIS_DIR, MODELS_DIR, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

DEFAULT_ALPHA_JSON = PROJECT_ROOT / "data" / "alpha8_tv" / "alpha_selected8_dopt_soft_g25.json"
os.environ.setdefault("EXPERIMENT_PROFILE_OVERRIDE", "alpha_condition")
os.environ.setdefault("ALPHA_CONDITION_TOP_K_OVERRIDE", "8")
if "ALPHA_CONDITION_JSON_OVERRIDE" not in os.environ and DEFAULT_ALPHA_JSON.exists():
    os.environ["ALPHA_CONDITION_JSON_OVERRIDE"] = str(DEFAULT_ALPHA_JSON)

from config import IMAGE_SIZE, RESULTS_DIR, device  # noqa: E402
from data_genoration import load_offline_tensors  # noqa: E402
from test import _temporary_experiment_config, load_model  # noqa: E402


def _resolve_path(raw_path: str | os.PathLike[str], *, default: Path | None = None) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        if default is None:
            raise ValueError("A non-empty path is required.")
        path = default
    else:
        path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _default_prefix(model_path: Path) -> str:
    stem = model_path.stem
    stem = re.sub(r"^theoretical_ct_", "", stem)
    stem = re.sub(r"_best_model$", "", stem)
    stem = re.sub(r"_model$", "", stem)
    return stem or "offline_val"


def _select_indices(total: int, sample_count: int, *, selection: str, seed: int) -> torch.Tensor:
    total = int(total)
    sample_count = int(sample_count)
    if total <= 0:
        raise ValueError("Offline validation dataset is empty.")
    if sample_count <= 0:
        raise ValueError(f"num_samples must be positive, got {sample_count!r}.")
    if sample_count > total:
        raise ValueError(f"num_samples={sample_count} exceeds offline dataset size={total}.")
    mode = str(selection).strip().lower()
    if mode == "first":
        return torch.arange(sample_count, dtype=torch.long)
    if mode == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return torch.randperm(total, generator=generator)[:sample_count]
    raise ValueError(f"Unsupported selection={selection!r}; expected 'first' or 'random'.")


def _relative_l2_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.detach().float()
    target = target.detach().float()
    dims = tuple(range(1, pred.dim()))
    diff_sq = torch.sum(torch.abs(pred - target).pow(2), dim=dims)
    true_sq = torch.sum(torch.abs(target).pow(2), dim=dims).clamp_min(1.0e-12)
    return torch.sqrt(diff_sq / true_sq)


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().float().view(-1)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "median": float(values.median().item()),
    }


def _parse_plot_sample_orders(raw_value: str, *, fallback: int) -> list[int]:
    raw = str(raw_value or "").strip()
    if not raw:
        return [int(fallback)]
    orders: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            order = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --plot-sample-orders value {raw_value!r}; expected comma-separated integers."
            ) from exc
        if order not in seen:
            orders.append(order)
            seen.add(order)
    if not orders:
        raise ValueError("--plot-sample-orders must contain at least one integer when provided.")
    return orders


def _plot_comparison(
    *,
    true_img: torch.Tensor,
    tv_img: torch.Tensor,
    pred_img: torch.Tensor,
    tv_res: float,
    pred_res: float,
    original_index: int,
    save_path: Path,
) -> None:
    arrays = [
        true_img.detach().cpu().squeeze().numpy(),
        tv_img.detach().cpu().squeeze().numpy(),
        pred_img.detach().cpu().squeeze().numpy(),
    ]
    vmin = min(float(array.min()) for array in arrays)
    vmax = max(float(array.max()) for array in arrays)
    extent = (0.0, float(IMAGE_SIZE - 1), 0.0, float(IMAGE_SIZE - 1))
    titles = [
        f"True\nval index={original_index}",
        f"TV init\nRES={tv_res:.6f}",
        f"NN reconstruction\nRES={pred_res:.6f}",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, array, title in zip(axes, arrays, titles):
        image = ax.imshow(array, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Offline val Shepp-Logan reconstruction comparison", y=1.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_offline_val(
    *,
    model_path: Path,
    offline_val_path: Path,
    result_dir: Path,
    result_prefix: str,
    num_samples: int,
    batch_size: int,
    selection: str,
    seed: int,
    plot_sample_orders: list[int],
    make_plots: bool = True,
) -> dict[str, object]:
    if not model_path.is_file():
        raise FileNotFoundError(f"best_model checkpoint not found: {model_path}")
    if not offline_val_path.is_file():
        raise FileNotFoundError(f"offline validation dataset not found: {offline_val_path}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}.")

    checkpoint = _load_checkpoint(model_path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}.")
    experiment_metadata = checkpoint.get("experiment_metadata", {})
    tensors = load_offline_tensors(offline_val_path)
    dataset_size = int(tensors["coeff_true"].shape[0])
    indices = _select_indices(dataset_size, int(num_samples), selection=selection, seed=int(seed))
    plot_orders = [int(value) for value in plot_sample_orders]
    if make_plots and not plot_orders:
        raise ValueError("plot_sample_orders must contain at least one sample order.")
    if make_plots:
        for plot_order in plot_orders:
            if not (0 <= int(plot_order) < int(num_samples)):
                raise ValueError(f"plot_sample_order must be in [0, {int(num_samples) - 1}], got {plot_order!r}.")

    with _temporary_experiment_config(experiment_metadata):
        model, _ = load_model(load_path=str(model_path), checkpoint=checkpoint)
        pred_chunks: list[torch.Tensor] = []
        for start in range(0, int(num_samples), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            coeff_init = tensors["coeff_initial"].index_select(0, batch_indices).to(device)
            g_observed = tensors["g_observed"].index_select(0, batch_indices).to(device)
            coeff_pred, _, _ = model(coeff_init, g_observed)
            pred_chunks.append(coeff_pred.detach().cpu())

    coeff_true = tensors["coeff_true"].index_select(0, indices).cpu()
    coeff_tv = tensors["coeff_initial"].index_select(0, indices).cpu()
    coeff_pred = torch.cat(pred_chunks, dim=0).cpu()

    tv_res = _relative_l2_per_sample(coeff_tv, coeff_true)
    pred_res = _relative_l2_per_sample(coeff_pred, coeff_true)
    tv_stats = _stats(tv_res)
    pred_stats = _stats(pred_res)

    result_dir.mkdir(parents=True, exist_ok=True)
    summary_json = result_dir / f"{result_prefix}_val{int(num_samples)}_summary.json"
    per_sample_csv = result_dir / f"{result_prefix}_val{int(num_samples)}_per_sample_res.csv"

    comparison_paths: list[Path] = []
    plot_val_indices: list[int] = []
    if make_plots:
        for plot_pos in plot_orders:
            original_index = int(indices[plot_pos].item())
            comparison_path = result_dir / (
                f"{result_prefix}_val{int(num_samples)}_sample{int(plot_pos):03d}_idx{original_index:03d}_comparison.png"
            )
            _plot_comparison(
                true_img=coeff_true[plot_pos],
                tv_img=coeff_tv[plot_pos],
                pred_img=coeff_pred[plot_pos],
                tv_res=float(tv_res[plot_pos].item()),
                pred_res=float(pred_res[plot_pos].item()),
                original_index=original_index,
                save_path=comparison_path,
            )
            comparison_paths.append(comparison_path)
            plot_val_indices.append(original_index)

    with per_sample_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["sample_order", "val_index", "tv_init_res", "nn_res"])
        writer.writeheader()
        for sample_order, original_index in enumerate(indices.tolist()):
            writer.writerow(
                {
                    "sample_order": int(sample_order),
                    "val_index": int(original_index),
                    "tv_init_res": float(tv_res[sample_order].item()),
                    "nn_res": float(pred_res[sample_order].item()),
                }
            )

    payload = {
        "model_path": str(model_path),
        "offline_val_path": str(offline_val_path),
        "result_dir": str(result_dir),
        "num_samples": int(num_samples),
        "dataset_size": int(dataset_size),
        "selection": str(selection),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "device": str(device),
        "selected_indices": [int(value) for value in indices.tolist()],
        "plot_sample_order": int(plot_orders[0]) if plot_orders else None,
        "plot_val_index": int(plot_val_indices[0]) if plot_val_indices else None,
        "plot_sample_orders": [int(value) for value in plot_orders],
        "plot_val_indices": [int(value) for value in plot_val_indices],
        "tv_init_res": tv_stats,
        "nn_res": pred_stats,
        "outputs": {
            "comparison_png": str(comparison_paths[0]) if comparison_paths else None,
            "comparison_pngs": [str(path) for path in comparison_paths],
            "summary_json": str(summary_json),
            "per_sample_csv": str(per_sample_csv),
        },
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    default_offline_val = PROJECT_ROOT / "data" / "data_genoration" / "val500_tvinit_alpha8_noise01.pt"
    parser = argparse.ArgumentParser(description="Evaluate a best_model checkpoint on offline val500 TV-init samples.")
    parser.add_argument("--model-path", required=True, help="Path to the selected *best_model.pth checkpoint.")
    parser.add_argument("--offline-val", default=str(default_offline_val), help="Path to val500_tvinit_alpha8_noise01.pt.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of offline validation samples to evaluate.")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for neural network evaluation.")
    parser.add_argument("--selection", choices=["first", "random"], default="first", help="How to select samples from val500.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --selection=random.")
    parser.add_argument("--plot-sample-order", type=int, default=0, help="Which evaluated sample to plot, by order in the selected subset.")
    parser.add_argument(
        "--plot-sample-orders",
        default="",
        help="Comma-separated evaluated sample orders to plot, e.g. 0,10,20,30,40. Overrides --plot-sample-order.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip comparison PNG generation for faster sweeps.")
    parser.add_argument("--result-dir", default=str(Path(RESULTS_DIR) / "unet" / "offline_val100"), help="Output directory.")
    parser.add_argument("--result-prefix", default="", help="Output file prefix. Defaults to checkpoint tag.")
    args = parser.parse_args()

    model_path = _resolve_path(args.model_path)
    offline_val_path = _resolve_path(args.offline_val)
    result_dir = _resolve_path(args.result_dir)
    result_prefix = str(args.result_prefix or _default_prefix(model_path)).strip()
    plot_sample_orders = [] if args.no_plots else _parse_plot_sample_orders(args.plot_sample_orders, fallback=int(args.plot_sample_order))

    payload = evaluate_offline_val(
        model_path=model_path,
        offline_val_path=offline_val_path,
        result_dir=result_dir,
        result_prefix=result_prefix,
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
        selection=str(args.selection),
        seed=int(args.seed),
        plot_sample_orders=plot_sample_orders,
        make_plots=not bool(args.no_plots),
    )

    print("==== Offline val evaluation ====")
    print(f"Model: {payload['model_path']}")
    print(f"Offline val: {payload['offline_val_path']}")
    print(f"Samples: {payload['num_samples']} / {payload['dataset_size']} selection={payload['selection']}")
    print(f"TV-init mean RES: {payload['tv_init_res']['mean']:.6f}")
    print(f"NN mean RES: {payload['nn_res']['mean']:.6f}")
    if payload["outputs"]["comparison_pngs"]:
        print(f"Comparison figures ({len(payload['outputs']['comparison_pngs'])}):")
        for figure_path in payload["outputs"]["comparison_pngs"]:
            print(f"  - {figure_path}")
    else:
        print("Comparison figures: disabled")
    print(f"Summary JSON: {payload['outputs']['summary_json']}")
    print(f"Per-sample CSV: {payload['outputs']['per_sample_csv']}")


if __name__ == "__main__":
    main()
