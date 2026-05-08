import inspect
import sys
import unittest
from pathlib import Path

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
        }
        self.assertTrue(forbidden_keys.isdisjoint(TIME_DOMAIN_CONFIG.keys()))

        _apply_experiment_profile("runtime_alpha")
        _apply_experiment_profile("alpha_condition")
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
        from model import LearnedGradientDescent

        backup = dict(TIME_DOMAIN_CONFIG)
        try:
            TIME_DOMAIN_CONFIG.update(
                {
                    "experiment_profile": "runtime_alpha",
                    "alpha_values": [0.23, 1.11],
                    "alpha_tau_offsets": [0.15, 0.35],
                    "num_angles_total": 2,
                    "num_angles": 2,
                    "cnn_num_angles_override": 2,
                    "cnn_angle_adapter_output_channels": 2,
                    "cnn_angle_adapter_hidden_channels": 2,
                    "physics_residual_channel_enabled": True,
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
        self.assertTrue(lgd.physics_residual_enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
