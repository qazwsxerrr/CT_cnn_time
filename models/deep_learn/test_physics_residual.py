import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parents[0]
for path in (THIS_DIR, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import TIME_DOMAIN_CONFIG, device
from radon_transform import AlphaContinuousB1B1Operator2D


class ConfigPatch:
    def __init__(self, **updates):
        self.updates = updates
        self.original = None

    def __enter__(self):
        self.original = dict(TIME_DOMAIN_CONFIG)
        TIME_DOMAIN_CONFIG.update(self.updates)
        return TIME_DOMAIN_CONFIG

    def __exit__(self, exc_type, exc, tb):
        TIME_DOMAIN_CONFIG.clear()
        TIME_DOMAIN_CONFIG.update(self.original)


class ZeroUpdateNetwork(nn.Module):
    def __init__(self, n_memory):
        super().__init__()
        self.n_memory = int(n_memory)

    def forward(self, x):
        return torch.zeros(
            x.shape[0],
            1 + self.n_memory,
            x.shape[2],
            x.shape[3],
            dtype=x.dtype,
            device=x.device,
        )


class PhysicsResidualChannelTests(unittest.TestCase):
    def test_alpha_operator_per_angle_residual_inverse_correction_solves_each_angle_system(self):
        torch.manual_seed(0)
        op = AlphaContinuousB1B1Operator2D(
            alpha_values=[0.23, 1.11],
            height=4,
            width=4,
            tau_offsets=[0.15, 0.35],
        ).to(device)

        coeff_true = torch.randn(1, 1, 4, 4, device=device)
        coeff_current = torch.randn(1, 1, 4, 4, device=device)
        observed = op(coeff_true)

        correction_pa = op.residual_inverse_correction_per_angle(
            coeff_current,
            observed,
            damping=1.0e-2,
            cg_iters=16,
            detach=True,
            normalize=False,
        )

        self.assertEqual(tuple(correction_pa.shape), (1, 2, 1, 4, 4))
        rhs_pa = op.adjoint_per_angle(op.split_measurements(observed - op(coeff_current)))
        normal_residual_pa = op.apply_normal_per_angle(correction_pa) + 1.0e-2 * correction_pa - rhs_pa
        self.assertLess(
            torch.norm(normal_residual_pa).item(),
            1.0e-3 * torch.norm(rhs_pa).item(),
        )

        normalized_pa = op.residual_inverse_correction_per_angle(
            coeff_current,
            observed,
            damping=1.0e-2,
            cg_iters=4,
            detach=True,
            normalize=True,
        )
        norms = torch.norm(normalized_pa.view(1, 2, -1), dim=2)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5, rtol=1.0e-5))

    def test_alpha_operator_residual_inverse_correction_solves_shifted_normal_system(self):
        torch.manual_seed(0)
        op = AlphaContinuousB1B1Operator2D(
            alpha_values=[0.23, 1.11],
            height=4,
            width=4,
            tau_offsets=[0.15, 0.35],
        ).to(device)

        coeff_true = torch.randn(1, 1, 4, 4, device=device)
        coeff_current = torch.randn(1, 1, 4, 4, device=device)
        observed = op(coeff_true)

        correction = op.residual_inverse_correction(
            coeff_current,
            observed,
            damping=1.0e-2,
            cg_iters=16,
            detach=True,
            normalize=False,
        )

        self.assertEqual(tuple(correction.shape), tuple(coeff_current.shape))
        rhs = op.adjoint(observed - op(coeff_current))
        normal_residual = op.apply_normal(correction) + 1.0e-2 * correction - rhs
        self.assertLess(
            torch.norm(normal_residual).item(),
            1.0e-3 * torch.norm(rhs).item(),
        )

        normalized = op.residual_inverse_correction(
            coeff_current,
            observed,
            damping=1.0e-2,
            cg_iters=4,
            detach=True,
            normalize=True,
        )
        self.assertAlmostEqual(
            torch.norm(normalized.view(normalized.shape[0], -1), dim=1).item(),
            1.0,
            places=5,
        )

    def test_lgd_physics_channel_adds_input_channel_and_explicit_update_uses_positive_error_direction(self):
        with ConfigPatch(
            operator_mode="theoretical_b1b1",
            use_multi_angle=True,
            alpha_values=[0.23, 1.11],
            alpha_tau_offsets=[0.15, 0.35],
            num_angles_total=2,
            num_angles=2,
            cnn_backbone_only=False,
            cnn_num_angles_override=2,
            cnn_angle_indices_override=None,
            cnn_angle_adapter_enabled=False,
            cnn_angle_adapter_mode="disabled",
            cnn_angle_adapter_output_channels=2,
            cnn_angle_adapter_hidden_channels=2,
            theoretical_formula_mode="alpha_continuous",
            multi_angle_solver_mode="stacked_tikhonov",
            physics_residual_channel_enabled=True,
            physics_residual_mode="per_angle_cg",
            physics_residual_damping=1.0e-2,
            physics_residual_cg_iters=4,
            physics_residual_detach=True,
            physics_residual_normalize=False,
            physics_explicit_update_enabled=True,
            physics_explicit_update_alpha_init=0.02,
            physics_explicit_update_max=0.10,
        ):
            from model import LearnedGradientDescent

            torch.manual_seed(1)
            lgd = LearnedGradientDescent(height=4, width=4, n_iter=1, n_memory=1).to(device)
            lgd.update_network = ZeroUpdateNetwork(n_memory=1).to(device)

            self.assertTrue(lgd.physics_residual_enabled)
            self.assertEqual(lgd.physics_residual_channels, 2)
            self.assertEqual(lgd.input_channels, 2 + 2 + 2 + 1)
            self.assertIn("physics_alpha_raw", dict(lgd.named_parameters()))

            coeff_initial = torch.zeros(1, 1, 4, 4, device=device)
            coeff_true = torch.randn(1, 1, 4, 4, device=device)
            observed = lgd.operator(coeff_true)
            expected_corr_pa = lgd.operator.residual_inverse_correction_per_angle(
                coeff_initial,
                observed,
                damping=lgd.physics_residual_damping,
                cg_iters=lgd.physics_residual_cg_iters,
                detach=lgd.physics_residual_detach,
                normalize=lgd.physics_residual_normalize,
            )
            expected_corr = expected_corr_pa.mean(dim=1)
            expected = coeff_initial + lgd.current_physics_alpha() * expected_corr

            coeff_final, history = lgd(coeff_initial, observed)

            self.assertEqual(len(history), 2)
            self.assertTrue(torch.allclose(coeff_final, expected, atol=1.0e-5, rtol=1.0e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
