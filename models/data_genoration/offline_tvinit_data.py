from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = ("coeff_true", "g_observed", "coeff_initial")

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
PROJECT_ROOT = MODELS_DIR.parent
DEFAULT_OUTPUT_PATH = THIS_DIR / "offline_tvinit_dataset.pt"


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _load_torch_file(path: str | os.PathLike[str]) -> Mapping[str, torch.Tensor]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a torch-saved mapping, got {type(data).__name__}.")
    return data


def _as_coeff_batch(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().to(dtype=torch.float32)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.dim() != 4 or int(tensor.shape[1]) != 1:
        raise ValueError(f"{name} must have shape (N,H,W) or (N,1,H,W), got {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _as_observation_batch(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().to(dtype=torch.float32)
    if tensor.dim() == 3 and int(tensor.shape[1]) == 1:
        tensor = tensor.squeeze(1)
    if tensor.dim() != 2:
        raise ValueError(f"g_observed must have shape (N,M) or (N,1,M), got {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _validate_same_batch_size(tensors: Mapping[str, torch.Tensor]) -> None:
    sizes = {key: int(tensors[key].shape[0]) for key in REQUIRED_KEYS}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Offline tensors have inconsistent batch sizes: {sizes!r}.")


def load_offline_tensors(path: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    """Load the minimal tensor payload needed by the neural network.

    Returned keys are exactly ``coeff_true``, ``g_observed`` and
    ``coeff_initial``. Extra fields in older experiment files are ignored.
    """
    raw = _load_torch_file(path)
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise KeyError(f"Offline dataset is missing keys: {missing!r}.")

    tensors = {
        "coeff_true": _as_coeff_batch(raw["coeff_true"], name="coeff_true"),
        "g_observed": _as_observation_batch(raw["g_observed"]),
        "coeff_initial": _as_coeff_batch(raw["coeff_initial"], name="coeff_initial"),
    }
    _validate_same_batch_size(tensors)
    return tensors


def save_offline_tensors(
    path: str | os.PathLike[str],
    *,
    coeff_true: torch.Tensor,
    g_observed: torch.Tensor,
    coeff_initial: torch.Tensor,
) -> Path:
    """Save only the tensors consumed by the training/test network."""
    tensors = {
        "coeff_true": _as_coeff_batch(coeff_true, name="coeff_true"),
        "g_observed": _as_observation_batch(g_observed),
        "coeff_initial": _as_coeff_batch(coeff_initial, name="coeff_initial"),
    }
    _validate_same_batch_size(tensors)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, output_path)
    return output_path


class OfflineCTDataset(Dataset):
    """Torch Dataset returning ``(coeff_true, g_observed, coeff_initial)``."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.tensors = load_offline_tensors(self.path)

    def __len__(self) -> int:
        return int(self.tensors["coeff_true"].shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.tensors["coeff_true"][index],
            self.tensors["g_observed"][index],
            self.tensors["coeff_initial"][index],
        )


class OfflineBatchProvider:
    """Small adapter with the same batch API shape as ``TheoreticalDataGenerator``.

    ``generate_batch`` returns ``(coeff_true, f_true, g_observed, coeff_initial)``;
    the second item is kept only for call-site compatibility and is not saved in
    the dataset file.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        shuffle: bool = True,
        target_device: torch.device | str | None = None,
    ):
        self.dataset = OfflineCTDataset(path)
        self.shuffle = bool(shuffle)
        self.target_device = torch.device(target_device) if target_device is not None else None
        self._order = torch.arange(len(self.dataset), dtype=torch.long)
        self._cursor = 0
        if self.shuffle:
            self._reshuffle()

    def _reshuffle(self) -> None:
        self._order = self._order[torch.randperm(len(self._order))]
        self._cursor = 0

    def _next_indices(self, batch_size: int) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        remaining = int(batch_size)
        while remaining > 0:
            if self._cursor >= len(self.dataset):
                if self.shuffle:
                    self._reshuffle()
                else:
                    self._cursor = 0
            take = min(remaining, len(self.dataset) - self._cursor)
            pieces.append(self._order[self._cursor : self._cursor + take])
            self._cursor += take
            remaining -= take
        return torch.cat(pieces, dim=0)

    def __len__(self) -> int:
        return len(self.dataset)

    def generate_batch_by_indices(
        self,
        indices,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.as_tensor(indices, dtype=torch.long).view(-1)
        if int(indices.numel()) <= 0:
            raise ValueError("indices must contain at least one item.")
        invalid_mask = (indices < 0) | (indices >= len(self.dataset))
        if bool(torch.any(invalid_mask)):
            invalid = indices[invalid_mask].detach().cpu().tolist()
            raise IndexError(f"indices contain out-of-range values {invalid!r} for dataset size={len(self.dataset)}.")
        tensors = self.dataset.tensors
        coeff_true = tensors["coeff_true"].index_select(0, indices)
        g_observed = tensors["g_observed"].index_select(0, indices)
        coeff_initial = tensors["coeff_initial"].index_select(0, indices)
        if self.target_device is not None:
            coeff_true = coeff_true.to(self.target_device)
            g_observed = g_observed.to(self.target_device)
            coeff_initial = coeff_initial.to(self.target_device)
        return coeff_true, coeff_true, g_observed, coeff_initial

    def generate_random_batch(
        self,
        batch_size: int,
        random_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}.")
        if batch_size > len(self.dataset):
            raise ValueError(f"batch_size={batch_size} exceeds dataset size={len(self.dataset)}.")
        generator = torch.Generator(device="cpu")
        if random_seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(random_seed))
        indices = torch.randperm(len(self.dataset), generator=generator)[:batch_size]
        return self.generate_batch_by_indices(indices)

    def generate_batch(
        self,
        batch_size: int,
        random_seed: int | None = None,
        lambda_reg: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del lambda_reg
        if random_seed is not None:
            torch.manual_seed(int(random_seed))
            if self.shuffle:
                self._reshuffle()
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}.")
        if batch_size > len(self.dataset):
            raise ValueError(f"batch_size={batch_size} exceeds dataset size={len(self.dataset)}.")
        indices = self._next_indices(batch_size)
        return self.generate_batch_by_indices(indices)


def _first_existing_path(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def default_alpha_json_path() -> Path:
    return _first_existing_path(
        [
            PROJECT_ROOT / "data" / "alpha8_tv" / "alpha_selected8_dopt_soft_g25.json",
            PROJECT_ROOT / "汇报" / "正则化改动" / "angle" / "alpha_selected8_dopt_soft_g25.json",
            PROJECT_ROOT
            / "results"
            / "shepp_logan_condition_vs_dopt_tv_noise01_8"
            / "alpha_selected8_dopt_soft_g25.json",
            PROJECT_ROOT / "data" / "alpha_search_cache" / "alpha_selected8.json",
        ]
    )


def apply_alpha8_tvinit_env_defaults() -> dict[str, str]:
    """Set run-script-compatible defaults before importing ``models/config.py``."""
    defaults = {
        "EXPERIMENT_PROFILE_OVERRIDE": "alpha_condition",
        "ALPHA_CONDITION_TOP_K_OVERRIDE": "8",
        "ALPHA_CONDITION_JSON_OVERRIDE": str(default_alpha_json_path()),
        "ALPHA_GRAM_CACHE_DIR_OVERRIDE": str(PROJECT_ROOT / "data" / "alpha_gram_cache"),
        "MULTI_ANGLE_SOLVER_MODE_OVERRIDE": "stacked_tikhonov",
        "THEORETICAL_FORMULA_MODE_OVERRIDE": "alpha_continuous",
        "CNN_ANGLE_INDICES_OVERRIDE": "0,1,2,3,4,5,6,7",
        "CNN_NUM_ANGLES_OVERRIDE": "8",
        "INIT_METHOD_OVERRIDE": "l2_tv_admm",
        "LAMBDA_SELECT_MODE_OVERRIDE": "morozov",
        "MOROZOV_FORM_OVERRIDE": "constrained",
        "MOROZOV_NOISE_RADIUS_MODE_OVERRIDE": "rms",
        "MOROZOV_TAU_OVERRIDE": "1.0",
        "L1_INIT_ADMM_ITERS_OVERRIDE": "40",
        "L1_INIT_ADMM_CG_ITERS_OVERRIDE": "15",
        "L1_INIT_ADMM_CG_TOL_OVERRIDE": "1e-4",
        "L1_INIT_ADMM_RHO_DATA_OVERRIDE": "1.0",
        "L1_INIT_ADMM_RHO_REG_OVERRIDE": "1.0",
        "REGULARIZER_TYPE_OVERRIDE": "dirichlet",
        "NOISE_MODE_OVERRIDE": "multiplicative",
        "NOISE_LEVEL_OVERRIDE": "0.1",
        "DATA_FIDELITY_CHANNEL_MODE_OVERRIDE": "stacked_selected",
        "PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE": "1",
        "PHYSICS_RESIDUAL_MODE_OVERRIDE": "stacked_selected_cg",
        "PHYSICS_RESIDUAL_DAMPING_OVERRIDE": "1e-2",
        "PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE": "8",
        "PHYSICS_RESIDUAL_DETACH_OVERRIDE": "1",
        "PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE": "1",
        "PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE": "1",
        "PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE": "0.1",
        "PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE": "0.25",
        "N_ITER_OVERRIDE": "8",
        "DETACH_PHYSICAL_GRADS_OVERRIDE": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in defaults}


def _default_generator_factory():
    apply_alpha8_tvinit_env_defaults()
    if str(MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(MODELS_DIR))

    from radon_transform import TheoreticalDataGenerator  # noqa: WPS433

    return TheoreticalDataGenerator


def _append_generated_segment(
    *,
    generator,
    num_samples: int,
    batch_size: int,
    seed_offset: int,
    label: str,
    coeff_true_parts: list[torch.Tensor],
    g_observed_parts: list[torch.Tensor],
    coeff_initial_parts: list[torch.Tensor],
) -> int:
    produced = 0
    while produced < num_samples:
        current = min(batch_size, num_samples - produced)
        seed = int(seed_offset) + produced
        coeff_true, _f_true, g_observed, coeff_initial = generator.generate_batch(
            batch_size=current,
            random_seed=seed,
        )
        coeff_true_parts.append(coeff_true.detach().cpu())
        g_observed_parts.append(g_observed.detach().cpu())
        coeff_initial_parts.append(coeff_initial.detach().cpu())
        produced += current
        _log(f"[offline-tvinit] {label}: generated {produced}/{num_samples}")
    return int(num_samples)


def generate_offline_dataset(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_PATH,
    *,
    num_samples: int,
    data_source: str = "random_ellipses",
    seed_offset: int = 0,
    batch_size: int = 1,
) -> Path:
    """Generate TV-initialized offline data compatible with ``deep_learn/model.py``."""
    num_samples = int(num_samples)
    batch_size = int(batch_size)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples!r}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}.")

    generator_cls = _default_generator_factory()
    generator = generator_cls(data_source=data_source)
    coeff_true_parts: list[torch.Tensor] = []
    g_observed_parts: list[torch.Tensor] = []
    coeff_initial_parts: list[torch.Tensor] = []

    _append_generated_segment(
        generator=generator,
        num_samples=num_samples,
        batch_size=batch_size,
        seed_offset=seed_offset,
        label=data_source,
        coeff_true_parts=coeff_true_parts,
        g_observed_parts=g_observed_parts,
        coeff_initial_parts=coeff_initial_parts,
    )

    return save_offline_tensors(
        output_path,
        coeff_true=torch.cat(coeff_true_parts, dim=0),
        g_observed=torch.cat(g_observed_parts, dim=0),
        coeff_initial=torch.cat(coeff_initial_parts, dim=0),
    )


def generate_mixed_offline_dataset(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_PATH,
    *,
    random_ellipses_samples: int = 3000,
    shepp_logan_samples: int = 500,
    seed_offset: int = 0,
    batch_size: int = 1,
    generator_factory=None,
) -> Path:
    """Generate one ordered file: random ellipses first, Shepp-Logan second."""
    random_ellipses_samples = int(random_ellipses_samples)
    shepp_logan_samples = int(shepp_logan_samples)
    batch_size = int(batch_size)
    if random_ellipses_samples < 0 or shepp_logan_samples < 0:
        raise ValueError("sample counts must be non-negative.")
    if random_ellipses_samples + shepp_logan_samples <= 0:
        raise ValueError("At least one sample must be requested.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}.")

    if generator_factory is None:
        generator_factory = _default_generator_factory()

    coeff_true_parts: list[torch.Tensor] = []
    g_observed_parts: list[torch.Tensor] = []
    coeff_initial_parts: list[torch.Tensor] = []
    produced_total = 0

    if random_ellipses_samples > 0:
        produced_total += _append_generated_segment(
            generator=generator_factory(data_source="random_ellipses"),
            num_samples=random_ellipses_samples,
            batch_size=batch_size,
            seed_offset=int(seed_offset) + produced_total,
            label="random_ellipses",
            coeff_true_parts=coeff_true_parts,
            g_observed_parts=g_observed_parts,
            coeff_initial_parts=coeff_initial_parts,
        )
    if shepp_logan_samples > 0:
        produced_total += _append_generated_segment(
            generator=generator_factory(data_source="shepp_logan"),
            num_samples=shepp_logan_samples,
            batch_size=batch_size,
            seed_offset=int(seed_offset) + produced_total,
            label="shepp_logan",
            coeff_true_parts=coeff_true_parts,
            g_observed_parts=g_observed_parts,
            coeff_initial_parts=coeff_initial_parts,
        )

    return save_offline_tensors(
        output_path,
        coeff_true=torch.cat(coeff_true_parts, dim=0),
        g_observed=torch.cat(g_observed_parts, dim=0),
        coeff_initial=torch.cat(coeff_initial_parts, dim=0),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate alpha8 TV-initialized offline CT data.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output .pt path.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples for single-source generation.")
    parser.add_argument("--random-ellipses-samples", type=int, default=None, help="Number of random_ellipses samples in ordered mixed generation.")
    parser.add_argument("--shepp-logan-samples", type=int, default=None, help="Number of shepp_logan samples in ordered mixed generation.")
    parser.add_argument("--data-source", default="random_ellipses", choices=["random_ellipses", "random_ellipse", "ellipse", "shepp_logan"])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size; use small values if GPU memory is tight.")
    return parser.parse_args(argv)


def _resolve_generation_counts(args: argparse.Namespace) -> tuple[int, int]:
    if args.num_samples is not None:
        raise ValueError("--num-samples uses single-source generation and has no mixed train/val counts.")
    if args.random_ellipses_samples is None and args.shepp_logan_samples is None:
        return 3000, 500
    return (
        0 if args.random_ellipses_samples is None else int(args.random_ellipses_samples),
        0 if args.shepp_logan_samples is None else int(args.shepp_logan_samples),
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    mixed_requested = args.random_ellipses_samples is not None or args.shepp_logan_samples is not None
    if mixed_requested:
        random_count, shepp_count = _resolve_generation_counts(args)
        output = generate_mixed_offline_dataset(
            args.output,
            random_ellipses_samples=random_count,
            shepp_logan_samples=shepp_count,
            seed_offset=args.seed_offset,
            batch_size=args.batch_size,
        )
    elif args.num_samples is not None:
        output = generate_offline_dataset(
            args.output,
            num_samples=args.num_samples,
            data_source=args.data_source,
            seed_offset=args.seed_offset,
            batch_size=args.batch_size,
        )
    else:
        output = generate_mixed_offline_dataset(
            args.output,
            random_ellipses_samples=3000,
            shepp_logan_samples=500,
            seed_offset=args.seed_offset,
            batch_size=args.batch_size,
        )
    tensors = load_offline_tensors(output)
    _log(
        "[offline-tvinit] saved "
        f"{output} | coeff_true={tuple(tensors['coeff_true'].shape)} "
        f"g_observed={tuple(tensors['g_observed'].shape)} "
        f"coeff_initial={tuple(tensors['coeff_initial'].shape)}"
    )


if __name__ == "__main__":
    main()
