import inspect
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
        from config import TIME_DOMAIN_CONFIG
        from test import _temporary_experiment_config

        backup = dict(TIME_DOMAIN_CONFIG)
        try:
            TIME_DOMAIN_CONFIG.update(
                {
                    "cnn_angle_indices_override": [0, 2],
                    "cnn_num_angles_override": 2,
                    "physics_residual_mode": "per_angle_cg",
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
            }
            with _temporary_experiment_config(metadata):
                self.assertEqual(TIME_DOMAIN_CONFIG["cnn_angle_indices_override"], [0, 2])
                self.assertEqual(TIME_DOMAIN_CONFIG["cnn_num_angles_override"], 2)
                self.assertEqual(TIME_DOMAIN_CONFIG["physics_residual_mode"], "per_angle_cg")
        finally:
            TIME_DOMAIN_CONFIG.clear()
            TIME_DOMAIN_CONFIG.update(backup)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
