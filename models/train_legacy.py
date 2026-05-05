# -*- coding: utf-8 -*-
"""Full training entry for same8_shifted_support_triangular_pi legacy sampling."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(__file__).resolve().parent

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ENV = {
    "EXPERIMENT_PROFILE_OVERRIDE": "same8_shifted_support_triangular_pi",
    "OUTPUT_TAG_OVERRIDE": "strict_same8_legacy_full_data_noise01_5000_seed20260503",
    "NOISE_MODE_OVERRIDE": "multiplicative",
    "NOISE_LEVEL_OVERRIDE": "0.1",
    "GLOBAL_SEED_OVERRIDE": "20260503",
    "VAL_REPRODUCIBLE_OVERRIDE": "1",
    "VAL_SEED_OVERRIDE": "42",
    "N_TRAIN_OVERRIDE": "5000",
    "N_DATA_OVERRIDE": "8",
}


def apply_default_env() -> None:
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)


def main() -> None:
    apply_default_env()

    print("=" * 80)
    print("FULL TRAINING: same8_shifted_support_triangular_pi")
    print("=" * 80)
    print("Expected recon formula: legacy_injective_extension (exact lower-banded triangular)")
    print("Expected data formula : auto_complete -> legacy_injective_extension (reuse)")
    print(f"Output tag            : {os.environ['OUTPUT_TAG_OVERRIDE']}")
    print("=" * 80)

    import train  # noqa: WPS433,E402

    train.main()


if __name__ == "__main__":
    main()
