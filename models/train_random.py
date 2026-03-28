import math
import os
from typing import List, Tuple

import numpy as np


DEFAULT_IMAGE_SIZE = 128
DEFAULT_NUM_ANGLES = 8
DEFAULT_RANDOM_ANGLE_SEED = 20260327


def _canonical_beta_direction(beta) -> Tuple[int, int]:
    a = int(beta[0])
    b = int(beta[1])
    g = math.gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    return (a, b)


def _sample_random_beta_vectors(
    *,
    extra_count: int,
    height: int,
    width: int,
    seed: int,
) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    scale = int(max(height, width))
    used = set()
    extra = []
    attempts = 0
    max_attempts = max(2048, 256 * int(extra_count))

    while len(extra) < int(extra_count) and attempts < max_attempts:
        theta = float(rng.uniform(0.0, math.pi))
        a = int(round(scale * math.cos(theta)))
        b = int(round(scale * math.sin(theta)))
        if a == 0 and b == 0:
            attempts += 1
            continue
        beta_dir = _canonical_beta_direction((a, b))
        if beta_dir not in used:
            used.add(beta_dir)
            extra.append(beta_dir)
        attempts += 1

    if len(extra) != int(extra_count):
        raise RuntimeError(
            f"Failed to sample {int(extra_count)} random beta vectors after {attempts} attempts."
        )
    return extra


def _configure_random_angle_environment() -> Tuple[int, List[Tuple[int, int]]]:
    seed = int(os.environ.get("RANDOM_ANGLE_SEED_OVERRIDE", str(DEFAULT_RANDOM_ANGLE_SEED)))
    num_angles = int(os.environ.get("RANDOM_NUM_ANGLES_OVERRIDE", str(DEFAULT_NUM_ANGLES)))
    if num_angles <= 0:
        raise ValueError(f"RANDOM_NUM_ANGLES_OVERRIDE must be positive, got {num_angles}.")

    betas = _sample_random_beta_vectors(
        extra_count=num_angles,
        height=DEFAULT_IMAGE_SIZE,
        width=DEFAULT_IMAGE_SIZE,
        seed=seed,
    )

    os.environ.setdefault("OPERATOR_MODE_OVERRIDE", "implicit_b1b1")
    os.environ.setdefault("NUM_ANGLES_TOTAL_OVERRIDE", str(num_angles))
    os.environ.setdefault(
        "OUTPUT_TAG_OVERRIDE",
        f"random{num_angles}_seed{seed}",
    )
    os.environ["BETA_VECTORS_OVERRIDE"] = ";".join(f"{a},{b}" for a, b in betas)
    return seed, betas


def main():
    seed, betas = _configure_random_angle_environment()
    print("=" * 60)
    print("RANDOM-ANGLE CT RECONSTRUCTION TRAINING")
    print("=" * 60)
    print(f"Random angle seed: {seed}")
    print(f"Output tag: {os.environ.get('OUTPUT_TAG_OVERRIDE', '')}")
    print("Active random beta vectors:")
    for beta in betas:
        print(f"  {beta}")

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
