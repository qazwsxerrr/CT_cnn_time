# -*- coding: utf-8 -*-
"""Shared reconstruction-initialization method definitions.

The project uses the same four regularized initialization families in
training, testing, and standalone Shepp-Logan comparison scripts:

* Tikhonov: ``||Ax-b||_2^2 + lambda ||x||_2^2``
* L2/L1: ``||Ax-b||_2^2 + lambda ||x||_1``
* L1/L1: ``||Ax-b||_1 + lambda ||x||_1``
* L2/TV ADMM: ``||Ax-b||_2^2 + lambda TV(x)``
* L2/TV PDHG: few primal-dual steps for fast TV-informed initialization

Keep the canonical names centralized here so command-line overrides,
experiment scripts, and data generators dispatch to the same methods.
"""

from __future__ import annotations

CG_INIT_METHOD = "cg"
TIKHONOV_INIT_METHOD = "tikhonov_direct"
L2_L1_INIT_METHOD = "l2_l1_admm"
L1_L1_INIT_METHOD = "l1_l1_admm"
L2_TV_INIT_METHOD = "l2_tv_admm"
L2_TV_PDHG_INIT_METHOD = "l2_tv_pdhg"

REGULARIZED_INIT_METHOD_CHOICES = (
    TIKHONOV_INIT_METHOD,
    L2_L1_INIT_METHOD,
    L1_L1_INIT_METHOD,
    L2_TV_INIT_METHOD,
    L2_TV_PDHG_INIT_METHOD,
)

SPLIT_ADMM_INIT_METHODS = (
    L2_L1_INIT_METHOD,
    L1_L1_INIT_METHOD,
    L2_TV_INIT_METHOD,
)

MEASUREMENT_NORMALIZED_REGULARIZED_INIT_METHODS = (
    L2_L1_INIT_METHOD,
    L1_L1_INIT_METHOD,
    L2_TV_INIT_METHOD,
    L2_TV_PDHG_INIT_METHOD,
)

INIT_METHOD_CHOICES = (
    CG_INIT_METHOD,
    *REGULARIZED_INIT_METHOD_CHOICES,
)

_ALIASES = {
    "cg": CG_INIT_METHOD,
    "tikhonov": TIKHONOV_INIT_METHOD,
    "tikhonov_direct": TIKHONOV_INIT_METHOD,
    "tikhonov_l2_l2": TIKHONOV_INIT_METHOD,
    "l2_l2": TIKHONOV_INIT_METHOD,
    "l2/l2": TIKHONOV_INIT_METHOD,
    "l2-l2": TIKHONOV_INIT_METHOD,
    "l2_l1": L2_L1_INIT_METHOD,
    "l2/l1": L2_L1_INIT_METHOD,
    "l2-l1": L2_L1_INIT_METHOD,
    "l2_l1_admm": L2_L1_INIT_METHOD,
    "l1_l1": L1_L1_INIT_METHOD,
    "l1/l1": L1_L1_INIT_METHOD,
    "l1-l1": L1_L1_INIT_METHOD,
    "l1_l1_admm": L1_L1_INIT_METHOD,
    "tv": L2_TV_INIT_METHOD,
    "l2_tv": L2_TV_INIT_METHOD,
    "l2/tv": L2_TV_INIT_METHOD,
    "l2-tv": L2_TV_INIT_METHOD,
    "l2_tv_admm": L2_TV_INIT_METHOD,
    "tv_pdhg": L2_TV_PDHG_INIT_METHOD,
    "l2_tv_pdhg": L2_TV_PDHG_INIT_METHOD,
    "pdhg_tv": L2_TV_PDHG_INIT_METHOD,
    "pdhg-tv": L2_TV_PDHG_INIT_METHOD,
    "l2/tv/pdhg": L2_TV_PDHG_INIT_METHOD,
}

_RECONSTRUCTION_METHODS: tuple[dict[str, str], ...] = (
    {
        "name": "tikhonov_l2_l2",
        "init_method": TIKHONOV_INIT_METHOD,
        "objective": "l2_l2",
        "morozov_residual_norm": "l2",
    },
    {
        "name": L2_L1_INIT_METHOD,
        "init_method": L2_L1_INIT_METHOD,
        "objective": "l2_l1",
        "morozov_residual_norm": "l2",
    },
    {
        "name": L1_L1_INIT_METHOD,
        "init_method": L1_L1_INIT_METHOD,
        "objective": "l1_l1",
        "morozov_residual_norm": "l1",
    },
    {
        "name": L2_TV_INIT_METHOD,
        "init_method": L2_TV_INIT_METHOD,
        "objective": "l2_tv",
        "morozov_residual_norm": "l2",
    },
    {
        "name": L2_TV_PDHG_INIT_METHOD,
        "init_method": L2_TV_PDHG_INIT_METHOD,
        "objective": "l2_tv",
        "morozov_residual_norm": "l2",
    },
)


def normalize_init_method(value: str) -> str:
    """Return the canonical initialization method name.

    User-facing scripts often use short labels such as ``tv`` or ``L2/L1``.
    Internally all dispatch goes through the canonical names in
    ``INIT_METHOD_CHOICES``.
    """
    key = str(value or "").strip().lower().replace(" ", "_")
    if key in _ALIASES:
        return _ALIASES[key]
    allowed = sorted(set(_ALIASES) | set(INIT_METHOD_CHOICES))
    raise ValueError(f"Unsupported init_method={value!r}; expected one of {allowed!r}.")


def reconstruction_method_defs() -> list[dict[str, str]]:
    """Return copy-safe metadata for the four regularized initializers."""
    return [dict(item) for item in _RECONSTRUCTION_METHODS]


def method_spec_from_init_method(value: str) -> dict[str, str]:
    """Resolve an init-method alias or method display name to its metadata."""
    raw = str(value or "").strip().lower()
    canonical = normalize_init_method(raw)
    for item in _RECONSTRUCTION_METHODS:
        if item["init_method"] == canonical or item["name"] == raw:
            return dict(item)
    raise ValueError(f"Initialization method {value!r} is not a regularized reconstruction method.")


def is_split_admm_init_method(value: str) -> bool:
    return normalize_init_method(value) in SPLIT_ADMM_INIT_METHODS


def is_regularized_init_method(value: str) -> bool:
    return normalize_init_method(value) in REGULARIZED_INIT_METHOD_CHOICES


def method_names() -> list[str]:
    return [item["name"] for item in _RECONSTRUCTION_METHODS]


def init_method_names() -> list[str]:
    return list(INIT_METHOD_CHOICES)


__all__ = [
    "CG_INIT_METHOD",
    "TIKHONOV_INIT_METHOD",
    "L2_L1_INIT_METHOD",
    "L1_L1_INIT_METHOD",
    "L2_TV_INIT_METHOD",
    "L2_TV_PDHG_INIT_METHOD",
    "REGULARIZED_INIT_METHOD_CHOICES",
    "SPLIT_ADMM_INIT_METHODS",
    "MEASUREMENT_NORMALIZED_REGULARIZED_INIT_METHODS",
    "INIT_METHOD_CHOICES",
    "normalize_init_method",
    "reconstruction_method_defs",
    "method_spec_from_init_method",
    "is_split_admm_init_method",
    "is_regularized_init_method",
    "method_names",
    "init_method_names",
]
