import inspect
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parents[0]
for path in (THIS_DIR, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class AlphaOnlyRefactorTests(unittest.TestCase):
    def test_config_has_no_beta_or_triangular_runtime_keys(self):
        from config import THEORETICAL_CONFIG, TIME_DOMAIN_CONFIG, _apply_experiment_profile

        self.assertNotIn("beta_vector", THEORETICAL_CONFIG)
        forbidden_keys = {
            "beta_vectors",
            "angle_parameterization",
            "multi_angle_layout",
            "auto_angle_t0",
            "condition_constrained_tau_offsets",
            "condition_constrained_records",
            "condition_constrained_json",
            "cnn_feature_beta_vectors_override",
            "triangular_residual_channel_enabled",
            "triangular_explicit_update_enabled",
            "triangular_angle_attention_enabled",
            "cnn_angle_adapter_enabled",
            "cnn_angle_adapter_mode",
            "cnn_angle_adapter_output_channels",
            "cnn_angle_adapter_hidden_channels",
        }
        self.assertTrue(forbidden_keys.isdisjoint(TIME_DOMAIN_CONFIG.keys()))

        config_source = Path(MODELS_DIR / "config.py").read_text(encoding="utf-8")
        self.assertNotIn("CNN_ANGLE_ADAPTER", config_source)

        _apply_experiment_profile("runtime_alpha")
        _apply_experiment_profile("alpha_condition")
        self.assertTrue(forbidden_keys.isdisjoint(TIME_DOMAIN_CONFIG.keys()))
        with self.assertRaises(ValueError):
            _apply_experiment_profile("condition_constrained8_pi")

    def test_radon_transform_exposes_only_alpha_operator(self):
        import radon_transform
        from radon_transform import AlphaContinuousB1B1Operator2D, build_time_domain_operator

        self.assertFalse(hasattr(radon_transform, "TheoreticalB1B1Operator2D"))
        self.assertFalse(hasattr(radon_transform, "ImplicitPixelRadonOperator2D"))
        self.assertFalse(hasattr(radon_transform, "_to_integer_beta"))
        self.assertFalse(hasattr(radon_transform, "_lower_banded_apply"))
        self.assertFalse(hasattr(radon_transform, "_build_lower_toeplitz_from_r"))

        build_params = inspect.signature(build_time_domain_operator).parameters
        self.assertNotIn("beta", build_params)
        self.assertNotIn("formula_mode_override", build_params)
        self.assertTrue(issubclass(AlphaContinuousB1B1Operator2D, radon_transform.torch.nn.Module))

    def test_learned_model_has_no_triangular_or_feature_beta_path(self):
        from config import TIME_DOMAIN_CONFIG
        import model
        from model import LearnedGradientDescent

        self.assertFalse(hasattr(model, "AdaptiveAngleFeatureAdapter"))

        backup = dict(TIME_DOMAIN_CONFIG)
        try:
            TIME_DOMAIN_CONFIG.update(
                {
                    "experiment_profile": "runtime_alpha",
                    "alpha_values": [0.23, 0.57, 1.11, 1.43],
                    "alpha_tau_offsets": [0.15, 0.25, 0.35, 0.45],
                    "num_angles_total": 4,
                    "num_angles": 4,
                    "cnn_num_angles_override": 4,
                    # Legacy adapter knobs must be ignored if injected by an old caller.
                    "cnn_angle_adapter_enabled": True,
                    "cnn_angle_adapter_mode": "adaptive_attention_mix",
                    "cnn_angle_adapter_output_channels": 2,
                    "cnn_angle_adapter_hidden_channels": 2,
                    "physics_residual_channel_enabled": False,
                    "physics_explicit_update_enabled": False,
                }
            )
            lgd = LearnedGradientDescent(height=4, width=4, n_iter=1, n_memory=1)
        finally:
            TIME_DOMAIN_CONFIG.clear()
            TIME_DOMAIN_CONFIG.update(backup)

        self.assertFalse(hasattr(lgd, "triangular_residual_enabled"))
        self.assertFalse(hasattr(lgd, "triangular_alpha_raw"))
        self.assertFalse(hasattr(lgd, "feature_beta_vectors"))
        self.assertFalse(hasattr(lgd, "angle_feature_adapter"))
        self.assertFalse(hasattr(lgd, "angle_adapter_enabled"))
        self.assertFalse(hasattr(lgd, "get_angle_adapter_diagnostics"))
        self.assertEqual(lgd.raw_cnn_num_angles, 4)
        self.assertEqual(lgd.cnn_num_angles, 4)
        self.assertEqual(lgd.input_channels, 2 + 4 + 0 + 1)
        self.assertFalse(lgd.physics_residual_enabled)

    def test_checkpoint_config_restore_preserves_explicit_cnn_angle_indices(self):
        from config import DATA_CONFIG, THEORETICAL_CONFIG, TIME_DOMAIN_CONFIG
        from test import _temporary_experiment_config

        data_backup = dict(DATA_CONFIG)
        backup = dict(TIME_DOMAIN_CONFIG)
        regularizer_backup = dict(THEORETICAL_CONFIG)
        try:
            DATA_CONFIG["l1_init_admm_iters"] = 12
            THEORETICAL_CONFIG["regularizer_type"] = "tikhonov"
            TIME_DOMAIN_CONFIG.update(
                {
                    "cnn_angle_indices_override": [0, 2],
                    "cnn_num_angles_override": 2,
                    "physics_residual_mode": "per_angle_cg",
                    "init_method": "tikhonov_direct",
                }
            )
            metadata = {
                "experiment_profile": "runtime_alpha",
                "operator_mode": "theoretical_b1b1",
                "alpha_values": [0.23, 0.57, 1.11, 1.43],
                "alpha_tau_offsets": [0.15, 0.25, 0.35, 0.45],
                "learned_num_angles": 4,
                "raw_cnn_angle_channels": 2,
                "cnn_num_angles": 2,
                "theoretical_formula_mode": "alpha_continuous",
                "data_formula_mode": "auto_complete",
                "physics_residual_mode": "per_angle_cg",
                "regularizer_type": "tv",
                "init_method": "l2_tv_admm",
                "l1_init_admm_iters": 80,
            }
            with _temporary_experiment_config(metadata):
                self.assertEqual(TIME_DOMAIN_CONFIG["cnn_angle_indices_override"], [0, 2])
                self.assertEqual(TIME_DOMAIN_CONFIG["cnn_num_angles_override"], 2)
                self.assertEqual(TIME_DOMAIN_CONFIG["physics_residual_mode"], "per_angle_cg")
                self.assertEqual(THEORETICAL_CONFIG["regularizer_type"], "tv")
                self.assertEqual(TIME_DOMAIN_CONFIG["init_method"], "l2_tv_admm")
                self.assertEqual(DATA_CONFIG["l1_init_admm_iters"], 80)
        finally:
            DATA_CONFIG.clear()
            DATA_CONFIG.update(data_backup)
            TIME_DOMAIN_CONFIG.clear()
            TIME_DOMAIN_CONFIG.update(backup)
            THEORETICAL_CONFIG.clear()
            THEORETICAL_CONFIG.update(regularizer_backup)

    def test_config_regularizer_type_env_override_accepts_tv_and_rejects_invalid(self):
        import importlib
        import config

        env_names = ("EXPERIMENT_PROFILE_OVERRIDE", "REGULARIZER_TYPE_OVERRIDE")
        env_backup = {name: os.environ.get(name) for name in env_names}
        try:
            os.environ["EXPERIMENT_PROFILE_OVERRIDE"] = "runtime_alpha"
            os.environ["REGULARIZER_TYPE_OVERRIDE"] = "tv"
            cfg = importlib.reload(config)
            self.assertEqual(cfg.THEORETICAL_CONFIG["regularizer_type"], "tv")

            os.environ["REGULARIZER_TYPE_OVERRIDE"] = "unsupported_regularizer"
            with self.assertRaises(ValueError):
                importlib.reload(config)
        finally:
            for name, value in env_backup.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            importlib.reload(config)

    def test_theoretical_gd_tv_regularization_gradient_matches_smoothed_tv_derivative(self):
        from model import TheoreticalGradientDescent

        x = torch.tensor(
            [[[[0.0, 2.0], [1.0, 4.0]]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        gd = TheoreticalGradientDescent(
            height=2,
            width=2,
            regularizer_type="tv",
            operator=torch.nn.Identity(),
        )

        grad = gd.compute_regularization_gradient(x.detach())

        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(x)
        grad_y[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        grad_x[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]
        tv_value = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1.0e-6).sum()
        expected = torch.autograd.grad(tv_value, x)[0]

        self.assertTrue(
            torch.allclose(grad, expected, atol=1.0e-5, rtol=1.0e-4),
            f"got {grad.detach().cpu().tolist()}, expected {expected.detach().cpu().tolist()}",
        )

    def test_train_helpers_resume_after_loaded_checkpoint_iteration(self):
        import train as train_module

        self.assertEqual(train_module._next_training_start_iter(-1), 0)
        self.assertEqual(train_module._next_training_start_iter(0), 1)
        self.assertEqual(train_module._next_training_start_iter(137), 138)

        backup = os.environ.get("RESUME_CHECKPOINT_OVERRIDE")
        try:
            os.environ["RESUME_CHECKPOINT_OVERRIDE"] = "/root/checkpoints/deep_learn/example_model.pth"
            self.assertEqual(
                train_module._resolve_resume_checkpoint_path(),
                "/root/checkpoints/deep_learn/example_model.pth",
            )
            os.environ["RESUME_CHECKPOINT_OVERRIDE"] = "   "
            self.assertIsNone(train_module._resolve_resume_checkpoint_path())
        finally:
            if backup is None:
                os.environ.pop("RESUME_CHECKPOINT_OVERRIDE", None)
            else:
                os.environ["RESUME_CHECKPOINT_OVERRIDE"] = backup

    def test_train_helper_summarizes_selected_cnn_angle_channels(self):
        import train as train_module

        summary = train_module._build_cnn_angle_selection_summary(
            cnn_angle_indices=[0, 2],
            alpha_values=[0.1, 0.2, 0.3],
            tau_offsets=[1.0, 1.1, 1.2],
            physics_residual_enabled=True,
            physics_residual_mode="per_angle_cg",
        )

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["indices"], [0, 2])
        self.assertEqual(summary["alpha_values"], [0.1, 0.3])
        self.assertEqual(summary["tau_offsets"], [1.0, 1.2])
        self.assertEqual(summary["data_fidelity_gradient_channel_indices"], [0, 2])
        self.assertEqual(summary["physics_residual_channel_indices"], [0, 2])

    def test_eval_tikhonov_baseline_uses_solver_not_network_input(self):
        from test import compute_tikhonov_baseline

        class FakeGenerator:
            def __init__(self):
                self.calls = []

            def solve_tikhonov_direct_init(self, g_obs, lambda_reg):
                self.calls.append(("direct", tuple(g_obs.shape), float(lambda_reg)))
                return torch.full((1, 1, 2, 2), 3.0)

        generator = FakeGenerator()
        baseline = compute_tikhonov_baseline(
            generator,
            g_obs=torch.arange(4, dtype=torch.float32),
            lambda_reg=0.25,
        )

        self.assertEqual(generator.calls, [("direct", (1, 4), 0.25)])
        self.assertTrue(torch.equal(baseline, torch.full((2, 2), 3.0)))

    def test_morozov_selects_lambda_against_full_measurement_residual(self):
        from config import DATA_CONFIG, device
        from radon_transform import AlphaContinuousB1B1Operator2D

        data_backup = dict(DATA_CONFIG)
        try:
            with tempfile.TemporaryDirectory() as cache_dir:
                DATA_CONFIG["alpha_gram_cache_dir"] = cache_dir
                torch.manual_seed(7)
                op = AlphaContinuousB1B1Operator2D(
                    alpha_values=[0.23, 1.11],
                    height=4,
                    width=4,
                    tau_offsets=[0.15, 0.35],
                ).to(device)
                coeff_true = torch.randn(1, 1, 4, 4, device=device)
                g_clean = op(coeff_true)
                noise = 0.05 * torch.randn_like(g_clean)
                g_observed = g_clean + noise
                noise_norm = torch.norm(noise, dim=-1)

                lam = op.choose_lambda_morozov(
                    g_observed,
                    noise_norm=noise_norm,
                    tau=1.0,
                    max_iter=40,
                    lambda_min=1.0e-12,
                    lambda_max=1.0e12,
                )
                coeff_est = op.solve_tikhonov_direct(g_observed, lambda_reg=lam)
                measurement_residual = torch.norm(op(coeff_est) - g_observed, dim=-1)

                self.assertLess(
                    torch.abs(measurement_residual - noise_norm).item() / noise_norm.item(),
                    1.0e-3,
                )
        finally:
            DATA_CONFIG.clear()
            DATA_CONFIG.update(data_backup)

    def test_l1_tv_admm_and_pdhg_initializers_reduce_their_objectives_and_dispatch(self):
        from config import DATA_CONFIG, TIME_DOMAIN_CONFIG, device
        from radon_transform import AlphaContinuousB1B1Operator2D, TheoreticalDataGenerator

        data_backup = dict(DATA_CONFIG)
        time_backup = dict(TIME_DOMAIN_CONFIG)
        try:
            DATA_CONFIG.update(
                {
                    "lambda_select_mode": "fixed",
                    "l1_init_admm_iters": 16,
                    "l1_init_admm_cg_iters": 12,
                    "l1_init_admm_cg_tol": 1.0e-5,
                    "l1_init_admm_rho_data": 4.0,
                    "l1_init_admm_rho_reg": 1.0,
                    "tv_pdhg_iters": 6,
                    "tv_pdhg_theta": 0.0,
                    "tv_pdhg_nonnegative": False,
                    "tv_pdhg_power_iters": 3,
                }
            )
            TIME_DOMAIN_CONFIG.update(
                {
                    "experiment_profile": "runtime_alpha",
                    "operator_mode": "theoretical_b1b1",
                    "use_multi_angle": True,
                    "alpha_values": [0.23, 1.11],
                    "alpha_tau_offsets": [0.15, 0.35],
                    "num_angles_total": 2,
                    "num_angles": 2,
                    "theoretical_formula_mode": "alpha_continuous",
                    "data_formula_mode": "auto_complete",
                }
            )
            torch.manual_seed(11)
            op = AlphaContinuousB1B1Operator2D(
                alpha_values=[0.23, 1.11],
                height=4,
                width=4,
                tau_offsets=[0.15, 0.35],
            ).to(device)
            generator = TheoreticalDataGenerator(img_size=4, data_source="shepp_logan", time_operator=op)
            coeff_true = torch.randn(1, 1, 4, 4, device=device)
            g_observed = op(coeff_true) + 0.03 * torch.randn(1, op.M, device=device)
            lam = torch.tensor([0.08], dtype=torch.float32, device=device)

            zero = torch.zeros_like(coeff_true)

            def objective_l2_l1(x):
                residual = op(x) - g_observed
                return 0.5 * torch.sum(residual.square()) + lam[0] * torch.sum(torch.abs(x))

            def objective_l1_l1(x):
                residual = op(x) - g_observed
                return torch.sum(torch.abs(residual)) + lam[0] * torch.sum(torch.abs(x))

            def objective_l2_tv(x):
                residual = op(x) - g_observed
                return 0.5 * torch.sum(residual.square()) + lam[0] * op.anisotropic_tv_norm(x).sum()

            coeff_l2_l1 = op.solve_l2_l1_admm(g_observed, lambda_reg=lam)
            coeff_l1_l1 = op.solve_l1_l1_admm(g_observed, lambda_reg=lam)
            coeff_l2_tv = op.solve_l2_tv_admm(g_observed, lambda_reg=lam)
            coeff_l2_tv_pdhg = op.solve_l2_tv_pdhg(
                g_observed,
                lambda_reg=lam,
                max_iter=6,
                theta=0.0,
                x0=zero,
                power_iters=3,
            )

            self.assertEqual(tuple(coeff_l2_l1.shape), (1, 1, 4, 4))
            self.assertEqual(tuple(coeff_l1_l1.shape), (1, 1, 4, 4))
            self.assertEqual(tuple(coeff_l2_tv.shape), (1, 1, 4, 4))
            self.assertEqual(tuple(coeff_l2_tv_pdhg.shape), (1, 1, 4, 4))
            self.assertTrue(torch.isfinite(coeff_l2_l1).all())
            self.assertTrue(torch.isfinite(coeff_l1_l1).all())
            self.assertTrue(torch.isfinite(coeff_l2_tv).all())
            self.assertTrue(torch.isfinite(coeff_l2_tv_pdhg).all())
            self.assertLess(float(objective_l2_l1(coeff_l2_l1)), float(objective_l2_l1(zero)))
            self.assertLess(float(objective_l1_l1(coeff_l1_l1)), float(objective_l1_l1(zero)))
            self.assertLess(float(objective_l2_tv(coeff_l2_tv)), float(objective_l2_tv(zero)))
            self.assertLess(float(objective_l2_tv(coeff_l2_tv_pdhg)), float(objective_l2_tv(zero)))

            lam_normalized = lam / float(op.M)
            dispatched_l2_l1 = generator.solve_regularized_init(
                g_observed,
                lambda_reg=lam_normalized,
                init_method="l2_l1_admm",
            )
            dispatched_l1_l1 = generator.solve_regularized_init(
                g_observed,
                lambda_reg=lam_normalized,
                init_method="l1_l1_admm",
            )
            dispatched_l2_tv = generator.solve_regularized_init(
                g_observed,
                lambda_reg=lam_normalized,
                init_method="l2_tv_admm",
            )
            dispatched_l2_tv_pdhg = generator.solve_regularized_init(
                g_observed,
                lambda_reg=lam_normalized,
                init_method="l2_tv_pdhg",
            )
            self.assertEqual(tuple(dispatched_l2_l1.shape), (1, 1, 4, 4))
            self.assertEqual(tuple(dispatched_l1_l1.shape), (1, 1, 4, 4))
            self.assertEqual(tuple(dispatched_l2_tv.shape), (1, 1, 4, 4))
            self.assertEqual(tuple(dispatched_l2_tv_pdhg.shape), (1, 1, 4, 4))
            self.assertTrue(torch.isfinite(dispatched_l2_tv_pdhg).all())
            self.assertTrue(torch.allclose(dispatched_l2_l1, coeff_l2_l1, atol=1.0e-4, rtol=1.0e-3))
            self.assertTrue(torch.allclose(dispatched_l1_l1, coeff_l1_l1, atol=1.0e-4, rtol=1.0e-3))
            self.assertTrue(torch.allclose(dispatched_l2_tv, coeff_l2_tv, atol=1.0e-4, rtol=1.0e-3))
        finally:
            DATA_CONFIG.clear()
            DATA_CONFIG.update(data_backup)
            TIME_DOMAIN_CONFIG.clear()
            TIME_DOMAIN_CONFIG.update(time_backup)

    def test_tv_gradient_adjoint_identity(self):
        from config import device
        from radon_transform import AlphaContinuousB1B1Operator2D

        torch.manual_seed(12)
        op = AlphaContinuousB1B1Operator2D(
            alpha_values=[0.23, 1.11],
            height=5,
            width=4,
            tau_offsets=[0.15, 0.35],
        ).to(device)
        x = torch.randn(2, 1, 5, 4, device=device)
        p = torch.randn(2, 2, 5, 4, device=device)

        lhs = torch.sum(op.tv_gradient(x) * p)
        rhs = torch.sum(x * op.tv_divergence_adjoint(p))

        self.assertLess(float(torch.abs(lhs - rhs).item()), 1.0e-5)

    def test_l1_morozov_uses_method_specific_residual_norms(self):
        from config import DATA_CONFIG, TIME_DOMAIN_CONFIG, device
        from radon_transform import AlphaContinuousB1B1Operator2D, TheoreticalDataGenerator

        data_backup = dict(DATA_CONFIG)
        time_backup = dict(TIME_DOMAIN_CONFIG)
        try:
            DATA_CONFIG.update(
                {
                    "lambda_select_mode": "morozov",
                    "morozov_max_iter": 2,
                    "morozov_lambda_min": 1.0e-4,
                    "morozov_lambda_max": 1.0,
                    "l1_init_admm_iters": 4,
                    "l1_init_admm_cg_iters": 4,
                    "l1_init_admm_cg_tol": 1.0e-4,
                }
            )
            TIME_DOMAIN_CONFIG.update(
                {
                    "experiment_profile": "runtime_alpha",
                    "operator_mode": "theoretical_b1b1",
                    "use_multi_angle": True,
                    "alpha_values": [0.23, 1.11],
                    "alpha_tau_offsets": [0.15, 0.35],
                    "num_angles_total": 2,
                    "num_angles": 2,
                    "theoretical_formula_mode": "alpha_continuous",
                    "data_formula_mode": "auto_complete",
                }
            )
            torch.manual_seed(13)
            op = AlphaContinuousB1B1Operator2D(
                alpha_values=[0.23, 1.11],
                height=4,
                width=4,
                tau_offsets=[0.15, 0.35],
            ).to(device)
            generator = TheoreticalDataGenerator(img_size=4, data_source="shepp_logan", time_operator=op)
            coeff_true = torch.randn(1, 1, 4, 4, device=device)
            g_clean = op(coeff_true)
            g_observed = g_clean + 0.04 * torch.randn_like(g_clean)

            TIME_DOMAIN_CONFIG["init_method"] = "l2_l1_admm"
            lam_l2 = generator._select_lambda(g_observed)
            info_l2 = dict(generator.last_lambda_info)
            self.assertTrue(torch.isfinite(lam_l2).all())
            self.assertEqual(info_l2["method"], "l2_l1_admm")
            self.assertEqual(info_l2["residual_norm"], "l2")
            self.assertEqual(info_l2["mode"], "morozov_iterative")
            self.assertEqual(info_l2["lambda_max_source"], "zero_solution_threshold")
            self.assertEqual(info_l2["lambda_scale"], "normalized_by_measurements")
            self.assertLess(float(lam_l2.view(-1)[0].item()), 1.0e6)

            TIME_DOMAIN_CONFIG["init_method"] = "l1_l1_admm"
            lam_l1 = generator._select_lambda(g_observed)
            info_l1 = dict(generator.last_lambda_info)
            self.assertTrue(torch.isfinite(lam_l1).all())
            self.assertEqual(info_l1["method"], "l1_l1_admm")
            self.assertEqual(info_l1["residual_norm"], "l1")
            self.assertEqual(info_l1["mode"], "morozov_iterative")
            self.assertEqual(info_l1["lambda_max_source"], "zero_solution_threshold")
            self.assertEqual(info_l1["lambda_scale"], "normalized_by_measurements")
            self.assertLess(float(lam_l1.view(-1)[0].item()), 1.0e6)

            TIME_DOMAIN_CONFIG["init_method"] = "l2_tv_admm"
            lam_tv = generator._select_lambda(g_observed)
            info_tv = dict(generator.last_lambda_info)
            self.assertTrue(torch.isfinite(lam_tv).all())
            self.assertEqual(info_tv["method"], "l2_tv_admm")
            self.assertEqual(info_tv["residual_norm"], "l2")
            self.assertEqual(info_tv["mode"], "morozov_iterative")
            self.assertEqual(info_tv["lambda_max_source"], "l2_l1_zero_threshold_proxy")
            self.assertEqual(info_tv["lambda_scale"], "normalized_by_measurements")
            self.assertLess(float(lam_tv.view(-1)[0].item()), 1.0e6)

            TIME_DOMAIN_CONFIG["init_method"] = "l2_tv_pdhg"
            lam_pdhg = generator._select_lambda(g_observed)
            info_pdhg = dict(generator.last_lambda_info)
            self.assertTrue(torch.isfinite(lam_pdhg).all())
            self.assertEqual(info_pdhg["method"], "l2_tv_pdhg")
            self.assertEqual(info_pdhg["residual_norm"], "l2")
            self.assertEqual(info_pdhg["mode"], "morozov_iterative")
            self.assertEqual(info_pdhg["lambda_max_source"], "configured")
            self.assertEqual(info_pdhg["lambda_scale"], "normalized_by_measurements")
        finally:
            DATA_CONFIG.clear()
            DATA_CONFIG.update(data_backup)
            TIME_DOMAIN_CONFIG.clear()
            TIME_DOMAIN_CONFIG.update(time_backup)

    def test_pdhg_morozov_search_checks_interior_lambdas_when_endpoint_residuals_are_large(self):
        from config import DATA_CONFIG
        from radon_transform import TheoreticalDataGenerator

        class DummyPDHGGenerator(TheoreticalDataGenerator):
            def __init__(self):
                self.time_operator = object()
                self.noise_mode = "additive"
                self.noise_level = 1.0
                self.last_lambda_info = None

            def solve_regularized_init(self, g_obs, lambda_reg, *, init_method=None):
                return torch.as_tensor(lambda_reg, dtype=torch.float32, device=g_obs.device).view(1, 1, 1, 1)

            @torch.no_grad()
            def _measurement_residual_norm(self, coeff, observed, *, norm_type: str):
                lam_value = max(float(coeff.detach().view(-1)[0].item()), 1.0e-30)
                residual = 1.0 + abs(math.log10(lam_value) - 1.0)
                return torch.tensor([residual], dtype=torch.float32, device=observed.device)

        data_backup = dict(DATA_CONFIG)
        try:
            DATA_CONFIG.update(
                {
                    "noise_mode": "additive",
                    "noise_level": 1.0,
                    "morozov_tau": 1.0,
                    "morozov_max_iter": 13,
                    "morozov_lambda_min": 1.0e-6,
                    "morozov_lambda_max": 1.0e6,
                }
            )
            generator = DummyPDHGGenerator()
            observed = torch.zeros(1, 1)
            lam = generator._choose_lambda_morozov_iterative(
                observed,
                init_method="l2_tv_pdhg",
                residual_norm="l2",
            )
            info = dict(generator.last_lambda_info)

            self.assertAlmostEqual(float(lam.view(-1)[0].item()), 10.0, delta=1.0e-4)
            self.assertEqual(info["lambda_max_source"], "configured")
            self.assertEqual(info["lambda_scale"], "normalized_by_measurements")
            self.assertIn(info["status"][0], {"log_grid_best", "bracketed_log_grid"})
        finally:
            DATA_CONFIG.clear()
            DATA_CONFIG.update(data_backup)

    def test_pdhg_morozov_prefers_largest_discrepancy_feasible_lambda_branch(self):
        from config import DATA_CONFIG
        from radon_transform import TheoreticalDataGenerator

        class DummyPDHGGenerator(TheoreticalDataGenerator):
            def __init__(self):
                self.time_operator = object()
                self.noise_mode = "additive"
                self.noise_level = 2.0
                self.last_lambda_info = None

            def solve_regularized_init(self, g_obs, lambda_reg, *, init_method=None):
                return torch.as_tensor(lambda_reg, dtype=torch.float32, device=g_obs.device).view(1, 1, 1, 1)

            @torch.no_grad()
            def _measurement_residual_norm(self, coeff, observed, *, norm_type: str):
                lam_value = max(float(coeff.detach().view(-1)[0].item()), 1.0e-30)
                residual = 1.0 + abs(math.log10(lam_value) - 1.0)
                return torch.tensor([residual], dtype=torch.float32, device=observed.device)

        data_backup = dict(DATA_CONFIG)
        try:
            DATA_CONFIG.update(
                {
                    "noise_mode": "additive",
                    "noise_level": 2.0,
                    "morozov_tau": 1.0,
                    "morozov_max_iter": 17,
                    "morozov_lambda_min": 1.0e-6,
                    "morozov_lambda_max": 1.0e6,
                }
            )
            generator = DummyPDHGGenerator()
            observed = torch.zeros(1, 1)
            lam = generator._choose_lambda_morozov_iterative(
                observed,
                init_method="l2_tv_pdhg",
                residual_norm="l2",
            )
            info = dict(generator.last_lambda_info)

            self.assertAlmostEqual(float(lam.view(-1)[0].item()), 100.0, delta=1.0e-3)
            self.assertEqual(info["status"][0], "bracketed_log_grid")
        finally:
            DATA_CONFIG.clear()
            DATA_CONFIG.update(data_backup)

    def test_morozov_lambda_uses_observed_multiplicative_noise_bound_not_clean_data(self):
        from config import DATA_CONFIG, TIME_DOMAIN_CONFIG, device
        from radon_transform import AlphaContinuousB1B1Operator2D, TheoreticalDataGenerator

        data_backup = dict(DATA_CONFIG)
        time_backup = dict(TIME_DOMAIN_CONFIG)
        try:
            DATA_CONFIG.update(
                {
                    "lambda_select_mode": "morozov",
                    "noise_mode": "multiplicative",
                    "noise_level": 0.1,
                    "morozov_noise_radius_mode": "rms",
                    "morozov_tau": 1.0,
                    "morozov_max_iter": 1,
                    "l1_init_admm_iters": 2,
                    "l1_init_admm_cg_iters": 2,
                    "l1_init_admm_cg_tol": 1.0e-4,
                }
            )
            TIME_DOMAIN_CONFIG.update(
                {
                    "experiment_profile": "runtime_alpha",
                    "operator_mode": "theoretical_b1b1",
                    "use_multi_angle": True,
                    "alpha_values": [0.23, 1.11],
                    "alpha_tau_offsets": [0.15, 0.35],
                    "num_angles_total": 2,
                    "num_angles": 2,
                    "theoretical_formula_mode": "alpha_continuous",
                    "data_formula_mode": "auto_complete",
                }
            )
            op = AlphaContinuousB1B1Operator2D(
                alpha_values=[0.23, 1.11],
                height=4,
                width=4,
                tau_offsets=[0.15, 0.35],
            ).to(device)
            generator = TheoreticalDataGenerator(img_size=4, data_source="shepp_logan", time_operator=op)
            g_observed = torch.linspace(0.2, 1.1, steps=op.M, dtype=torch.float32, device=device).view(1, -1)
            self.assertNotIn("g_clean", inspect.signature(generator.select_lambda_for_init_method).parameters)

            expected_l2 = (0.1 / (3.0 + 0.1 * 0.1) ** 0.5) * torch.norm(g_observed, dim=-1)
            lam_a = generator.select_lambda_for_init_method(g_observed, init_method="tikhonov_direct")
            info_a = dict(generator.last_lambda_info)
            lam_b = generator.select_lambda_for_init_method(g_observed, init_method="tikhonov_direct")
            info_b = dict(generator.last_lambda_info)

            self.assertTrue(torch.allclose(lam_a, lam_b, atol=1.0e-7, rtol=1.0e-6))
            self.assertAlmostEqual(float(info_a["target_norm"][0]), float(expected_l2[0].item()), places=6)
            self.assertEqual(info_a["target_norm"], info_b["target_norm"])
            self.assertEqual(info_a["noise_radius_source"], "observed_multiplicative_rms")

            expected_l1 = (0.1 / (3.0 + 0.1 * 0.1) ** 0.5) * torch.sum(torch.abs(g_observed), dim=-1)
            lam_l1_a = generator.select_lambda_for_init_method(g_observed, init_method="l1_l1_admm")
            info_l1_a = dict(generator.last_lambda_info)
            lam_l1_b = generator.select_lambda_for_init_method(g_observed, init_method="l1_l1_admm")
            info_l1_b = dict(generator.last_lambda_info)

            self.assertTrue(torch.allclose(lam_l1_a, lam_l1_b, atol=1.0e-7, rtol=1.0e-6))
            self.assertAlmostEqual(float(info_l1_a["target_norm"][0]), float(expected_l1[0].item()), places=6)
            self.assertEqual(info_l1_a["target_norm"], info_l1_b["target_norm"])
            self.assertEqual(info_l1_a["noise_radius_source"], "observed_multiplicative_rms")
        finally:
            DATA_CONFIG.clear()
            DATA_CONFIG.update(data_backup)
            TIME_DOMAIN_CONFIG.clear()
            TIME_DOMAIN_CONFIG.update(time_backup)

    def test_morozov_form_config_and_constrained_api_are_available(self):
        import importlib
        import config
        import radon_transform
        from radon_transform import AlphaContinuousB1B1Operator2D, TheoreticalDataGenerator

        config_source = Path(MODELS_DIR / "config.py").read_text(encoding="utf-8")

        self.assertIn('"morozov_form": "regularized"', config_source)
        self.assertIn("MOROZOV_FORM_OVERRIDE", config_source)
        self.assertIn("MOROZOV_NOISE_RADIUS_MODE_OVERRIDE", config_source)
        self.assertNotIn("ADMM_MOROZOV_CHECK_OVERRIDE", config_source)
        self.assertNotIn("ADMM_MOROZOV_TAU_OVERRIDE", config_source)
        self.assertNotIn("admm_morozov_check", config_source)
        self.assertNotIn("admm_morozov_tau", config_source)
        self.assertTrue(hasattr(radon_transform.AlphaContinuousB1B1Operator2D, "solve_l2_l1_morozov_admm"))
        self.assertTrue(hasattr(radon_transform.AlphaContinuousB1B1Operator2D, "solve_l1_l1_morozov_admm"))
        self.assertTrue(hasattr(radon_transform.AlphaContinuousB1B1Operator2D, "solve_l2_tv_morozov_admm"))
        self.assertTrue(hasattr(TheoreticalDataGenerator, "solve_constrained_init"))
        self.assertTrue(hasattr(TheoreticalDataGenerator, "solve_morozov_constrained_init"))

        env_backup = os.environ.get("MOROZOV_FORM_OVERRIDE")
        try:
            os.environ["MOROZOV_FORM_OVERRIDE"] = "constrained"
            reloaded = importlib.reload(config)
            self.assertEqual(reloaded.DATA_CONFIG["morozov_form"], "constrained")
        finally:
            if env_backup is None:
                os.environ.pop("MOROZOV_FORM_OVERRIDE", None)
            else:
                os.environ["MOROZOV_FORM_OVERRIDE"] = env_backup
            importlib.reload(config)
            importlib.reload(radon_transform)

    def test_constrained_morozov_form_uses_radius_instead_of_lambda_search(self):
        from config import DATA_CONFIG, TIME_DOMAIN_CONFIG, device
        from radon_transform import AlphaContinuousB1B1Operator2D, TheoreticalDataGenerator

        data_backup = dict(DATA_CONFIG)
        time_backup = dict(TIME_DOMAIN_CONFIG)
        try:
            DATA_CONFIG.update(
                {
                    "lambda_select_mode": "morozov",
                    "morozov_form": "constrained",
                    "noise_mode": "additive",
                    "noise_level": 0.25,
                    "morozov_tau": 2.0,
                    "l1_init_admm_iters": 2,
                    "l1_init_admm_cg_iters": 2,
                    "l1_init_admm_cg_tol": 1.0e-4,
                    "l1_init_admm_rho_data": 1.0,
                    "l1_init_admm_rho_reg": 1.0,
                }
            )
            TIME_DOMAIN_CONFIG.update(
                {
                    "experiment_profile": "runtime_alpha",
                    "operator_mode": "theoretical_b1b1",
                    "use_multi_angle": True,
                    "alpha_values": [0.23, 1.11],
                    "alpha_tau_offsets": [0.15, 0.35],
                    "num_angles_total": 2,
                    "num_angles": 2,
                    "theoretical_formula_mode": "alpha_continuous",
                    "data_formula_mode": "auto_complete",
                    "init_method": "l2_tv_admm",
                }
            )
            op = AlphaContinuousB1B1Operator2D(
                alpha_values=[0.23, 1.11],
                height=4,
                width=4,
                tau_offsets=[0.15, 0.35],
            ).to(device)
            generator = TheoreticalDataGenerator(img_size=4, data_source="shepp_logan", time_operator=op)
            g_observed = torch.linspace(0.1, 0.9, steps=op.M, dtype=torch.float32, device=device).view(1, -1)
            expected_radius = 2.0 * 0.25 * math.sqrt(float(op.M))

            radius = generator._select_lambda(g_observed)
            info = dict(generator.last_lambda_info)
            self.assertEqual(info["mode"], "morozov_constrained_radius")
            self.assertEqual(info["method"], "l2_tv_admm")
            self.assertEqual(info["residual_norm"], "l2")
            self.assertAlmostEqual(float(radius.view(-1)[0].item()), expected_radius, places=5)
            self.assertAlmostEqual(float(info["constraint_radius"][0]), expected_radius, places=5)

            coeff = generator.solve_morozov_constrained_init(g_observed, init_method="l2_tv_admm")
            info = dict(generator.last_lambda_info)
            self.assertEqual(tuple(coeff.shape), (1, 1, 4, 4))
            self.assertTrue(torch.isfinite(coeff).all())
            self.assertEqual(info["mode"], "morozov_constrained")
            self.assertEqual(info["solver_stats"]["method"], "l2_tv_morozov_admm")
            self.assertIn("constraint_radius", info["solver_stats"])
        finally:
            DATA_CONFIG.clear()
            DATA_CONFIG.update(data_backup)
            TIME_DOMAIN_CONFIG.clear()
            TIME_DOMAIN_CONFIG.update(time_backup)

    def test_morozov_noise_radius_mode_can_use_conservative_bound(self):
        from config import DATA_CONFIG, TIME_DOMAIN_CONFIG, device
        from radon_transform import AlphaContinuousB1B1Operator2D, TheoreticalDataGenerator

        data_backup = dict(DATA_CONFIG)
        time_backup = dict(TIME_DOMAIN_CONFIG)
        try:
            DATA_CONFIG.update(
                {
                    "lambda_select_mode": "morozov",
                    "noise_mode": "multiplicative",
                    "noise_level": 0.1,
                    "morozov_noise_radius_mode": "conservative",
                    "morozov_tau": 1.0,
                    "morozov_max_iter": 1,
                }
            )
            TIME_DOMAIN_CONFIG.update(
                {
                    "experiment_profile": "runtime_alpha",
                    "operator_mode": "theoretical_b1b1",
                    "use_multi_angle": True,
                    "alpha_values": [0.23, 1.11],
                    "alpha_tau_offsets": [0.15, 0.35],
                    "num_angles_total": 2,
                    "num_angles": 2,
                    "theoretical_formula_mode": "alpha_continuous",
                    "data_formula_mode": "auto_complete",
                }
            )
            op = AlphaContinuousB1B1Operator2D(
                alpha_values=[0.23, 1.11],
                height=4,
                width=4,
                tau_offsets=[0.15, 0.35],
            ).to(device)
            generator = TheoreticalDataGenerator(img_size=4, data_source="shepp_logan", time_operator=op)
            g_observed = torch.linspace(0.2, 1.1, steps=op.M, dtype=torch.float32, device=device).view(1, -1)
            expected_l2 = (0.1 / 0.9) * torch.norm(g_observed, dim=-1)

            _ = generator.select_lambda_for_init_method(g_observed, init_method="tikhonov_direct")
            info = dict(generator.last_lambda_info)

            self.assertAlmostEqual(float(info["target_norm"][0]), float(expected_l2[0].item()), places=6)
            self.assertEqual(info["noise_radius_source"], "observed_multiplicative_conservative")
        finally:
            DATA_CONFIG.clear()
            DATA_CONFIG.update(data_backup)
            TIME_DOMAIN_CONFIG.clear()
            TIME_DOMAIN_CONFIG.update(time_backup)

    def test_alpha_eval_defines_regularized_baseline_methods(self):
        alpha_dir = MODELS_DIR / "α_condition"
        if str(alpha_dir) not in sys.path:
            sys.path.insert(0, str(alpha_dir))
        import alpha_tikhonov_eval

        methods = alpha_tikhonov_eval.reconstruction_method_defs()
        by_name = {item["name"]: item for item in methods}

        self.assertEqual(
            list(by_name.keys()),
            ["tikhonov_l2_l2", "l2_l1_admm", "l1_l1_admm", "l2_tv_admm", "l2_tv_pdhg"],
        )
        self.assertEqual(by_name["tikhonov_l2_l2"]["init_method"], "tikhonov_direct")
        self.assertEqual(by_name["tikhonov_l2_l2"]["morozov_residual_norm"], "l2")
        self.assertEqual(by_name["l2_l1_admm"]["init_method"], "l2_l1_admm")
        self.assertEqual(by_name["l2_l1_admm"]["morozov_residual_norm"], "l2")
        self.assertEqual(by_name["l1_l1_admm"]["init_method"], "l1_l1_admm")
        self.assertEqual(by_name["l1_l1_admm"]["morozov_residual_norm"], "l1")
        self.assertEqual(by_name["l2_tv_admm"]["init_method"], "l2_tv_admm")
        self.assertEqual(by_name["l2_tv_admm"]["objective"], "l2_tv")
        self.assertEqual(by_name["l2_tv_admm"]["morozov_residual_norm"], "l2")
        self.assertEqual(by_name["l2_tv_pdhg"]["init_method"], "l2_tv_pdhg")
        self.assertEqual(by_name["l2_tv_pdhg"]["objective"], "l2_tv")
        self.assertEqual(by_name["l2_tv_pdhg"]["morozov_residual_norm"], "l2")

    def test_initialization_methods_are_available_as_shared_model_choices(self):
        import initialization_methods as init_methods
        from config import INIT_METHOD_CHOICES

        self.assertEqual(
            list(init_methods.REGULARIZED_INIT_METHOD_CHOICES),
            ["tikhonov_direct", "l2_l1_admm", "l1_l1_admm", "l2_tv_admm", "l2_tv_pdhg"],
        )
        self.assertTrue(set(init_methods.REGULARIZED_INIT_METHOD_CHOICES).issubset(set(INIT_METHOD_CHOICES)))
        self.assertEqual(
            list(init_methods.MEASUREMENT_NORMALIZED_REGULARIZED_INIT_METHODS),
            ["l2_l1_admm", "l1_l1_admm", "l2_tv_admm", "l2_tv_pdhg"],
        )
        self.assertEqual(init_methods.normalize_init_method("tikhonov"), "tikhonov_direct")
        self.assertEqual(init_methods.normalize_init_method("L2/L1"), "l2_l1_admm")
        self.assertEqual(init_methods.normalize_init_method("l1-l1"), "l1_l1_admm")
        self.assertEqual(init_methods.normalize_init_method("tv"), "l2_tv_admm")
        self.assertEqual(init_methods.normalize_init_method("tv_pdhg"), "l2_tv_pdhg")
        self.assertEqual(init_methods.normalize_init_method("pdhg_tv"), "l2_tv_pdhg")

        methods = init_methods.reconstruction_method_defs()
        self.assertEqual([item["name"] for item in methods], ["tikhonov_l2_l2", "l2_l1_admm", "l1_l1_admm", "l2_tv_admm", "l2_tv_pdhg"])
        self.assertEqual(init_methods.method_spec_from_init_method("tv")["objective"], "l2_tv")
        self.assertEqual(init_methods.method_spec_from_init_method("tv_pdhg")["init_method"], "l2_tv_pdhg")

    def test_shepp_logan_comparison_exposes_adjustable_angle_count_helpers(self):
        comparison_dir = MODELS_DIR / "shepp_logan_comparison"
        if str(comparison_dir) not in sys.path:
            sys.path.insert(0, str(comparison_dir))
        import compare_angle_selection

        self.assertEqual(compare_angle_selection.parse_angle_counts("8,4"), [8, 4])
        self.assertEqual(compare_angle_selection.parse_angle_counts(" 16 "), [16])
        with self.assertRaises(ValueError):
            compare_angle_selection.parse_angle_counts("8,0")

        selected_path = compare_angle_selection.resolve_selected_alpha_json("", 8)
        self.assertEqual(selected_path.name, "alpha_selected8.json")
        self.assertIn("alpha_search_cache", str(selected_path))
        self.assertEqual(compare_angle_selection.resolve_method_spec("tv")["init_method"], "l2_tv_admm")

    def test_random_uniform_angles_are_independent_of_angle_count_order(self):
        comparison_dir = MODELS_DIR / "shepp_logan_comparison"
        if str(comparison_dir) not in sys.path:
            sys.path.insert(0, str(comparison_dir))
        import compare_angle_selection

        seed = 20260517
        forward = {
            count: compare_angle_selection.random_uniform_alphas(count, seed, trial_index=0)
            for count in [10, 12, 16]
        }
        reverse = {
            count: compare_angle_selection.random_uniform_alphas(count, seed, trial_index=0)
            for count in [16, 12, 10]
        }

        self.assertEqual(forward, reverse)
        self.assertEqual(forward[16], sorted(forward[16]))
        self.assertEqual(len(forward[16]), 16)
        self.assertNotEqual(
            compare_angle_selection.random_uniform_alphas(16, seed, trial_index=0),
            compare_angle_selection.random_uniform_alphas(16, seed, trial_index=1),
        )

    def test_selected_angle_count_plot_titles_include_res_values(self):
        comparison_dir = MODELS_DIR / "shepp_logan_comparison"
        if str(comparison_dir) not in sys.path:
            sys.path.insert(0, str(comparison_dir))
        import plot_selected_angle_counts

        self.assertEqual(plot_selected_angle_counts.format_panel_title(4, 0.123456789), "k=4\nRES=0.123457")
        self.assertEqual(plot_selected_angle_counts.selected_case_name(16), "selected16_uniform_condition")
        self.assertEqual(
            plot_selected_angle_counts.safe_method_filename("l2_tv_admm"),
            "l2_tv_admm",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
