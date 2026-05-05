# -*- coding: utf-8 -*-
"""Lightweight regression tests for condition_constrained8_pi sampling support.

Run from the repository root with:
    D:\\python_code\\anaconda_mini\\envs\\pytorch_env\\python.exe models/test_condition_constrained_sampling.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))


class ConditionConstrainedConfigTests(unittest.TestCase):
    def test_condition_constrained8_pi_profile_loads_betas_and_tau_offsets_from_json(self) -> None:
        records = [
            {"beta": [1, 128], "tau_star": -127.5, "cond": 10.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-1, 128], "tau_star": -128.5, "cond": 11.0, "sigma_min": 0.1, "sigma_max": 1.1},
            {"beta": [1, -128], "tau_star": 0.5, "cond": 12.0, "sigma_min": 0.1, "sigma_max": 1.2},
            {"beta": [-1, -128], "tau_star": -129.5, "cond": 13.0, "sigma_min": 0.1, "sigma_max": 1.3},
            {"beta": [128, 1], "tau_star": 0.5, "cond": 14.0, "sigma_min": 0.1, "sigma_max": 1.4},
            {"beta": [-128, 1], "tau_star": -128.5, "cond": 15.0, "sigma_min": 0.1, "sigma_max": 1.5},
            {"beta": [128, -1], "tau_star": -0.5, "cond": 16.0, "sigma_min": 0.1, "sigma_max": 1.6},
            {"beta": [-128, -1], "tau_star": -129.5, "cond": 17.0, "sigma_min": 0.1, "sigma_max": 1.7},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best_condition_constrained8_pi.json"
            path.write_text(json.dumps({"top8": records}, ensure_ascii=False), encoding="utf-8")
            old_env = dict(os.environ)
            try:
                os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "condition_constrained8_pi"
                os.environ["CONDITION_CONSTRAINED8_PI_JSON_OVERRIDE"] = str(path)
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                cfg = config.TIME_DOMAIN_CONFIG
                self.assertEqual(cfg["experiment_profile"], "condition_constrained8_pi")
                self.assertEqual(cfg["num_angles_total"], 8)
                self.assertEqual(cfg["multi_angle_solver_mode"], "stacked_tikhonov")
                self.assertEqual(cfg["theoretical_formula_mode"], "condition_constrained_offset")
                self.assertEqual(cfg["data_formula_mode"], "auto_complete")
                self.assertEqual(cfg["beta_vectors"], [tuple(r["beta"]) for r in records])
                self.assertEqual(cfg["condition_constrained_tau_offsets"], [r["tau_star"] for r in records])
            finally:
                os.environ.clear()
                os.environ.update(old_env)
                sys.modules.pop("config", None)

    def test_condition_constrained8_pi_profile_uses_pi_json_override(self) -> None:
        records = [
            {"beta": [128, 1], "tau_star": 58.05, "cond": 1.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-128, 1], "tau_star": -69.95, "cond": 2.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [1, 128], "tau_star": 70.95, "cond": 3.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-1, 128], "tau_star": 69.95, "cond": 4.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [128, 3], "tau_star": 72.05, "cond": 5.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-128, 3], "tau_star": -55.95, "cond": 6.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [3, 128], "tau_star": 58.95, "cond": 7.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-3, 128], "tau_star": 55.95, "cond": 8.0, "sigma_min": 0.1, "sigma_max": 1.0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best_condition_constrained8_pi.json"
            path.write_text(json.dumps({"top8": records}, ensure_ascii=False), encoding="utf-8")
            old_env = dict(os.environ)
            try:
                os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "condition_constrained8_pi"
                os.environ["CONDITION_CONSTRAINED8_PI_JSON_OVERRIDE"] = str(path)
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                cfg = config.TIME_DOMAIN_CONFIG
                self.assertEqual(cfg["experiment_profile"], "condition_constrained8_pi")
                self.assertEqual(cfg["num_angles_total"], 8)
                self.assertEqual(cfg["multi_angle_solver_mode"], "stacked_tikhonov")
                self.assertEqual(cfg["theoretical_formula_mode"], "condition_constrained_offset")
                self.assertEqual(cfg["data_formula_mode"], "auto_complete")
                self.assertEqual(cfg["beta_vectors"], [tuple(r["beta"]) for r in records])
                self.assertEqual(cfg["condition_constrained_tau_offsets"], [r["tau_star"] for r in records])
                self.assertEqual(cfg["condition_constrained_json"], str(path))
            finally:
                os.environ.clear()
                os.environ.update(old_env)
                sys.modules.pop("config", None)

    def test_same8_shifted_support_triangular_pi_profile_uses_complete_triangular_baseline(self) -> None:
        records = [
            {"beta": [128, 1], "tau_star": 58.05, "cond": 1.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-128, 1], "tau_star": -69.95, "cond": 2.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [1, 128], "tau_star": 70.95, "cond": 3.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-1, 128], "tau_star": 69.95, "cond": 4.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [128, 3], "tau_star": 72.05, "cond": 5.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-128, 3], "tau_star": -55.95, "cond": 6.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [3, 128], "tau_star": 58.95, "cond": 7.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-3, 128], "tau_star": 55.95, "cond": 8.0, "sigma_min": 0.1, "sigma_max": 1.0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best_condition_constrained8_pi.json"
            path.write_text(json.dumps({"top8": records}, ensure_ascii=False), encoding="utf-8")
            old_env = dict(os.environ)
            try:
                os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "same8_shifted_support_triangular_pi"
                os.environ["CONDITION_CONSTRAINED8_PI_JSON_OVERRIDE"] = str(path)
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                cfg = config.TIME_DOMAIN_CONFIG
                self.assertEqual(cfg["experiment_profile"], "same8_shifted_support_triangular_pi")
                self.assertEqual(cfg["num_angles_total"], 8)
                self.assertEqual(cfg["multi_angle_solver_mode"], "stacked_tikhonov")
                self.assertEqual(cfg["theoretical_formula_mode"], "legacy_injective_extension")
                self.assertEqual(cfg["data_formula_mode"], "auto_complete")
                self.assertTrue(cfg["auto_angle_t0"])
                self.assertEqual(cfg["beta_vectors"], [tuple(r["beta"]) for r in records])
                self.assertIsNone(cfg["condition_constrained_tau_offsets"])
                self.assertIsNone(cfg["condition_constrained_records"])
                self.assertIsNone(cfg["condition_constrained_json"])
            finally:
                os.environ.clear()
                os.environ.update(old_env)
                sys.modules.pop("config", None)

    def test_formula_override_to_legacy_does_not_leak_condition_tau_offsets(self) -> None:
        records = [
            {"beta": [128, 1], "tau_star": 58.05, "cond": 1.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-128, 1], "tau_star": -69.95, "cond": 2.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [1, 128], "tau_star": 70.95, "cond": 3.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-1, 128], "tau_star": 69.95, "cond": 4.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [128, 3], "tau_star": 72.05, "cond": 5.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-128, 3], "tau_star": -55.95, "cond": 6.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [3, 128], "tau_star": 58.95, "cond": 7.0, "sigma_min": 0.1, "sigma_max": 1.0},
            {"beta": [-3, 128], "tau_star": 55.95, "cond": 8.0, "sigma_min": 0.1, "sigma_max": 1.0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best_condition_constrained8_pi.json"
            path.write_text(json.dumps({"top8": records}, ensure_ascii=False), encoding="utf-8")
            old_env = dict(os.environ)
            try:
                os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "condition_constrained8_pi"
                os.environ["CONDITION_CONSTRAINED8_PI_JSON_OVERRIDE"] = str(path)
                os.environ["THEORETICAL_FORMULA_MODE_OVERRIDE"] = "legacy_injective_extension"
                sys.modules.pop("config", None)
                sys.modules.pop("radon_transform", None)
                config = importlib.import_module("config")
                radon_transform = importlib.import_module("radon_transform")

                self.assertEqual(config.TIME_DOMAIN_CONFIG["theoretical_formula_mode"], "legacy_injective_extension")
                op = radon_transform.build_time_domain_operator(height=4, width=4)

                self.assertIsNone(op.t0_per_angle)
                expected_band_t0 = torch.tensor(
                    [
                        min(0, int(beta[0]), int(beta[1]), int(beta[0]) + int(beta[1])) + 0.5
                        for beta in config.TIME_DOMAIN_CONFIG["beta_vectors"]
                    ],
                    dtype=op.band_t0_per_angle.cpu().dtype,
                )
                self.assertTrue(torch.allclose(op.band_t0_per_angle.cpu(), expected_band_t0, atol=1e-6))
            finally:
                os.environ.clear()
                os.environ.update(old_env)
                sys.modules.pop("config", None)
                sys.modules.pop("radon_transform", None)

    def test_temporary_experiment_config_restores_retained_formula_from_metadata(self) -> None:
        old_env = dict(os.environ)
        try:
            os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "condition_constrained8_pi"
            sys.modules.pop("config", None)
            sys.modules.pop("test", None)
            config = importlib.import_module("config")
            test_module = importlib.import_module("test")

            metadata = {
                "experiment_profile": "condition_constrained8_pi",
                "theoretical_formula_mode": "legacy_injective_extension",
                "auto_angle_t0": True,
                "condition_constrained_tau_offsets": None,
                "condition_constrained_json": None,
                "beta_vectors": [(128, 1), (-128, 1), (1, 128), (-1, 128), (128, 3), (-128, 3), (3, 128), (-3, 128)],
                "num_angles": 8,
                "num_angles_total": 8,
            }
            with test_module._temporary_experiment_config(metadata):
                cfg = config.TIME_DOMAIN_CONFIG
                self.assertEqual(cfg["theoretical_formula_mode"], "legacy_injective_extension")
                self.assertTrue(cfg["auto_angle_t0"])
                self.assertIsNone(cfg["condition_constrained_tau_offsets"])
                self.assertIsNone(cfg["condition_constrained_json"])
        finally:
            os.environ.clear()
            os.environ.update(old_env)
            sys.modules.pop("config", None)
            sys.modules.pop("test", None)


class ConditionConstrainedRadonTests(unittest.TestCase):
    def test_internal_data_formula_mode_rejects_lower_banded_or_self_consistent_generation(self) -> None:
        old_env = dict(os.environ)
        try:
            os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "default"
            sys.modules.pop("config", None)
            sys.modules.pop("radon_transform", None)
            config = importlib.import_module("config")
            radon_transform = importlib.import_module("radon_transform")

            config.TIME_DOMAIN_CONFIG.update(
                {
                    "operator_mode": "theoretical_b1b1",
                    "use_multi_angle": True,
                    "multi_angle_layout": "full_triangular",
                    "beta_vectors": [(1, 128)],
                    "num_angles": 1,
                    "num_angles_total": 1,
                    "theoretical_formula_mode": "legacy_injective_extension",
                    "auto_angle_t0": True,
                    "condition_constrained_tau_offsets": None,
                }
            )

            for invalid_mode in ("same_as_reconstruction",):
                with self.subTest(invalid_mode=invalid_mode):
                    config.TIME_DOMAIN_CONFIG["data_formula_mode"] = invalid_mode
                    with self.assertRaisesRegex(ValueError, "complete data operator"):
                        radon_transform.TheoreticalDataGenerator(data_source="shepp_logan")
        finally:
            os.environ.clear()
            os.environ.update(old_env)
            sys.modules.pop("config", None)
            sys.modules.pop("radon_transform", None)

    def test_removed_data_formula_modes_raise_value_error(self) -> None:
        old_env = dict(os.environ)
        try:
            os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "default"
            sys.modules.pop("config", None)
            sys.modules.pop("radon_transform", None)
            config = importlib.import_module("config")
            radon_transform = importlib.import_module("radon_transform")

            config.TIME_DOMAIN_CONFIG.update(
                {
                    "operator_mode": "theoretical_b1b1",
                    "use_multi_angle": True,
                    "multi_angle_layout": "full_triangular",
                    "beta_vectors": [(1, 128)],
                    "num_angles": 1,
                    "num_angles_total": 1,
                    "theoretical_formula_mode": "legacy_injective_extension",
                    "auto_angle_t0": True,
                    "condition_constrained_tau_offsets": None,
                }
            )
            for removed_mode in ("legacy_injective_full", "condition_constrained_full", "shifted_support"):
                with self.subTest(removed_mode=removed_mode):
                    config.TIME_DOMAIN_CONFIG["data_formula_mode"] = removed_mode
                    with self.assertRaises(ValueError):
                        radon_transform.TheoreticalDataGenerator(data_source="shepp_logan")
        finally:
            os.environ.clear()
            os.environ.update(old_env)
            sys.modules.pop("config", None)
            sys.modules.pop("radon_transform", None)

    def test_condition_full_reconstruction_reuses_same_operator_for_data_generation(self) -> None:
        old_env = dict(os.environ)
        try:
            os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "default"
            sys.modules.pop("config", None)
            sys.modules.pop("radon_transform", None)
            config = importlib.import_module("config")
            radon_transform = importlib.import_module("radon_transform")

            config.TIME_DOMAIN_CONFIG.update(
                {
                    "operator_mode": "theoretical_b1b1",
                    "use_multi_angle": True,
                    "multi_angle_layout": "full_triangular",
                    "beta_vectors": [(1, 128)],
                    "num_angles": 1,
                    "num_angles_total": 1,
                    "theoretical_formula_mode": "condition_constrained_offset",
                    "data_formula_mode": "auto_complete",
                    "auto_angle_t0": False,
                    "condition_constrained_tau_offsets": [70.95],
                }
            )

            generator = radon_transform.TheoreticalDataGenerator(data_source="shepp_logan")

            self.assertTrue(bool(generator.time_operator.uses_sparse_blocks))
            self.assertIs(generator.data_time_operator, generator.time_operator)
            self.assertEqual(str(generator.data_formula_mode), "condition_constrained_offset")
        finally:
            os.environ.clear()
            os.environ.update(old_env)
            sys.modules.pop("config", None)
            sys.modules.pop("radon_transform", None)

    def test_data_generator_can_share_existing_reconstruction_operator(self) -> None:
        old_env = dict(os.environ)
        try:
            os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "default"
            sys.modules.pop("config", None)
            sys.modules.pop("radon_transform", None)
            config = importlib.import_module("config")
            radon_transform = importlib.import_module("radon_transform")

            config.TIME_DOMAIN_CONFIG.update(
                {
                    "operator_mode": "theoretical_b1b1",
                    "use_multi_angle": True,
                    "multi_angle_layout": "full_triangular",
                    "beta_vectors": [(1, 128)],
                    "num_angles": 1,
                    "num_angles_total": 1,
                    "theoretical_formula_mode": "condition_constrained_offset",
                    "data_formula_mode": "auto_complete",
                    "auto_angle_t0": False,
                    "condition_constrained_tau_offsets": [70.95],
                }
            )

            shared_operator = radon_transform.build_time_domain_operator(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE)
            generator = radon_transform.TheoreticalDataGenerator(
                data_source="shepp_logan",
                time_operator=shared_operator,
            )

            self.assertIs(generator.time_operator, shared_operator)
            self.assertIs(generator.data_time_operator, shared_operator)
            self.assertEqual(str(generator.data_formula_mode), "condition_constrained_offset")
        finally:
            os.environ.clear()
            os.environ.update(old_env)
            sys.modules.pop("config", None)
            sys.modules.pop("radon_transform", None)

    def test_legacy_injective_extension_data_generation_uses_exact_lower_banded_operator(self) -> None:
        old_env = dict(os.environ)
        try:
            os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "default"
            sys.modules.pop("config", None)
            sys.modules.pop("radon_transform", None)
            config = importlib.import_module("config")
            radon_transform = importlib.import_module("radon_transform")

            config.TIME_DOMAIN_CONFIG.update(
                {
                    "operator_mode": "theoretical_b1b1",
                    "use_multi_angle": True,
                    "multi_angle_layout": "full_triangular",
                    "beta_vectors": [(1, 128)],
                    "num_angles": 1,
                    "num_angles_total": 1,
                    "theoretical_formula_mode": "legacy_injective_extension",
                    "data_formula_mode": "auto_complete",
                    "auto_angle_t0": True,
                    "condition_constrained_tau_offsets": None,
                }
            )

            generator = radon_transform.TheoreticalDataGenerator(data_source="shepp_logan")

            self.assertFalse(bool(generator.time_operator.uses_sparse_blocks))
            self.assertIs(generator.data_time_operator, generator.time_operator)
            self.assertEqual(str(generator.data_formula_mode), "legacy_injective_extension")
            self.assertEqual(int(generator.time_operator.upper_bandwidths.max().item()), 1)
        finally:
            os.environ.clear()
            os.environ.update(old_env)
            sys.modules.pop("config", None)
            sys.modules.pop("radon_transform", None)

    def test_condition_offset_uses_complete_sparse_matrix_with_tau_sampling_points(self) -> None:
        # condition_constrained_offset is now the actual condition sampling
        # operator, not a lower-banded ablation.  It keeps
        # X_beta={(kappa_i+tau)/||beta||} and builds the complete sparse
        # matrix for that sampling grid.
        os.environ.setdefault("EXPERIMENT_PROFILE_OVERRIDE", "default")
        sys.modules.pop("config", None)
        sys.modules.pop("radon_transform", None)
        radon_transform = importlib.import_module("radon_transform")
        beta = (1, -4)
        tau = -1.5
        op = radon_transform.TheoreticalB1B1Operator2D(
            beta_vectors=[beta],
            height=4,
            width=4,
            t0=0.5,
            formula_mode="condition_constrained_offset",
            t0_per_angle=[tau],
        )
        self.assertTrue(bool(op.uses_sparse_blocks))
        self.assertGreater(int(op.upper_bandwidths[0].item()), 1)

        expected = (op.sorted_proj_per_angle[0].to(torch.float32) + tau) / torch.linalg.vector_norm(
            op.betas[0].to(torch.float32)
        )
        self.assertTrue(torch.allclose(op.sampling_points_per_angle[0].cpu(), expected.cpu(), atol=1e-6))

        x = torch.randn(2, 1, 4, 4, device=op.sampling_points.device)
        y = torch.randn(2, op.M, device=op.sampling_points.device)
        lhs = torch.sum(op.forward(x) * y)
        rhs = torch.sum(x * op.adjoint(y))
        self.assertLess(float(torch.abs(lhs - rhs).item()), 1e-4)

    def test_removed_formula_modes_raise_value_error_in_operator_construction(self) -> None:
        os.environ.setdefault("EXPERIMENT_PROFILE_OVERRIDE", "default")
        sys.modules.pop("config", None)
        sys.modules.pop("radon_transform", None)
        radon_transform = importlib.import_module("radon_transform")
        for removed_mode in ("legacy_injective_full", "condition_constrained_full", "shifted_support"):
            with self.subTest(removed_mode=removed_mode):
                with self.assertRaises(ValueError):
                    radon_transform.TheoreticalB1B1Operator2D(
                        beta_vectors=[(1, -4)],
                        height=4,
                        width=4,
                        t0=0.5,
                        formula_mode=removed_mode,
                    )

    def test_legacy_injective_extension_is_true_triangular_for_signed_betas(self) -> None:
        os.environ.setdefault("EXPERIMENT_PROFILE_OVERRIDE", "default")
        sys.modules.pop("config", None)
        sys.modules.pop("radon_transform", None)
        radon_transform = importlib.import_module("radon_transform")
        betas = [(-4, 1), (-1, 4), (1, -4), (4, -1)]
        banded = radon_transform.TheoreticalB1B1Operator2D(
            beta_vectors=betas,
            height=4,
            width=4,
            t0=0.5,
            formula_mode="legacy_injective_extension",
        )

        expected_band_t0 = torch.tensor(
            [min(0, beta[0], beta[1], beta[0] + beta[1]) + 0.5 for beta in betas],
            dtype=banded.band_t0_per_angle.cpu().dtype,
        )
        self.assertTrue(torch.allclose(banded.band_t0_per_angle.cpu(), expected_band_t0, atol=1e-6))
        self.assertEqual(int(banded.upper_bandwidths.max().item()), 1)

        for angle_idx in range(len(betas)):
            beta_norm = torch.linalg.vector_norm(banded.betas[angle_idx].to(torch.float32)).cpu()
            beta_domain_points = banded.sampling_points_per_angle[angle_idx].cpu() * beta_norm
            expected_beta_domain_points = (
                banded.theory_t0_abs_per_angle[angle_idx].cpu()
                + banded.sorted_proj_per_angle[angle_idx].cpu().to(torch.float32)
                - banded.kappa0_per_angle[angle_idx].cpu().to(torch.float32)
            )
            self.assertTrue(torch.allclose(beta_domain_points, expected_beta_domain_points, atol=1e-5))

            lower = banded.lower_bands[angle_idx].cpu()
            band_width = int(banded.lower_bandwidths[angle_idx].item())
            self.assertGreaterEqual(band_width, 1)
            self.assertGreater(float(torch.abs(lower[0]).max().item()), 0.0)

        x = torch.randn(2, 1, 4, 4, device=banded.sampling_points.device)
        y = torch.randn(2, banded.M, device=banded.sampling_points.device)
        lhs = torch.sum(banded.forward(x) * y)
        rhs = torch.sum(x * banded.adjoint(y))
        self.assertLess(float(torch.abs(lhs - rhs).item()), 1e-4)

    def test_removed_solver_modes_raise_value_error(self) -> None:
        os.environ.setdefault("EXPERIMENT_PROFILE_OVERRIDE", "default")
        sys.modules.pop("config", None)
        sys.modules.pop("radon_transform", None)
        config = importlib.import_module("config")
        radon_transform = importlib.import_module("radon_transform")
        old_mode = config.TIME_DOMAIN_CONFIG.get("multi_angle_solver_mode")
        try:
            for removed_mode in ("consensus_admm", "split_triangular_admm"):
                with self.subTest(removed_mode=removed_mode):
                    config.TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = removed_mode
                    op = radon_transform.TheoreticalB1B1Operator2D(
                        beta_vectors=[(1, 4), (4, 1)],
                        height=4,
                        width=4,
                        t0=0.5,
                        formula_mode="condition_constrained_offset",
                        t0_per_angle=[2.5, 2.5],
                    )
                    coeff_true = torch.randn(1, 1, 4, 4, device=op.sampling_points.device)
                    g_clean = op.forward(coeff_true)
                    with self.assertRaises(ValueError):
                        op.solve_tikhonov_direct(g_clean, lambda_reg=1.0)
        finally:
            config.TIME_DOMAIN_CONFIG["multi_angle_solver_mode"] = old_mode


class GradientChannelFidelityMetricTests(unittest.TestCase):
    def test_metric_function_reports_relative_error_cosine_and_snr_per_angle(self) -> None:
        module = importlib.import_module("gradient_channel_fidelity_compare")
        clean = torch.tensor(
            [
                [
                    [[[3.0, 4.0]]],
                    [[[1.0, 0.0]]],
                ]
            ]
        )
        noisy = torch.tensor(
            [
                [
                    [[[6.0, 8.0]]],
                    [[[1.0, 1.0]]],
                ]
            ]
        )

        metrics = module.compute_gradient_channel_metrics(clean, noisy)

        self.assertEqual(metrics["num_angles"], 2)
        self.assertAlmostEqual(metrics["relative_error_per_angle"][0], 1.0, places=6)
        self.assertAlmostEqual(metrics["cosine_per_angle"][0], 1.0, places=6)
        self.assertAlmostEqual(metrics["relative_error_per_angle"][1], 1.0, places=6)
        self.assertAlmostEqual(metrics["cosine_per_angle"][1], 2.0 ** -0.5, places=6)
        self.assertAlmostEqual(metrics["relative_error_mean"], 1.0, places=6)


class PdhgTvUtilityTests(unittest.TestCase):
    def test_finite_difference_adjoint_matches_inner_product(self) -> None:
        module = importlib.import_module("_tmp_compare_condition_vs_tri_tikhonov")
        x = torch.randn(2, 1, 5, 6)
        q = torch.randn(2, 2, 5, 6)

        grad = module.finite_difference_gradient(x)
        adj = module.finite_difference_adjoint(q)

        lhs = torch.sum(grad * q)
        rhs = torch.sum(x * adj)
        self.assertLess(float(torch.abs(lhs - rhs).item()), 1.0e-5)

    def test_project_isotropic_tv_dual_clamps_pixelwise_norm(self) -> None:
        module = importlib.import_module("_tmp_compare_condition_vs_tri_tikhonov")
        q = torch.tensor([[[[3.0]], [[4.0]]]])

        projected = module.project_isotropic_tv_dual(q, radius=2.0)

        norm = torch.linalg.vector_norm(projected, dim=1)
        self.assertLessEqual(float(norm.max().item()), 2.0 + 1.0e-6)
        self.assertAlmostEqual(float(projected[0, 0, 0, 0].item()), 1.2, places=6)
        self.assertAlmostEqual(float(projected[0, 1, 0, 0].item()), 1.6, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
