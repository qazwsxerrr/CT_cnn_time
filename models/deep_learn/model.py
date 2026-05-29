from collections import OrderedDict

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).resolve().parents[1]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from config import device, THEORETICAL_CONFIG, DATA_CONFIG, TIME_DOMAIN_CONFIG, IMAGE_SIZE
from radon_transform import build_time_domain_operator


# ============================================================================
# 1. Coefficient mapping (row-major flatten)
# ============================================================================
class CoefficientMapping:
    def __init__(self, E_plus_shape=(IMAGE_SIZE, IMAGE_SIZE)):
        self.E_plus_shape = E_plus_shape
        self.height, self.width = E_plus_shape
        self.N = self.height * self.width
        self._create_one_to_one_mapping()

    def _create_one_to_one_mapping(self):
        self.k_to_d_mapping = {}
        self.d_to_k_mapping = {}
        for kx in range(self.height):
            for ky in range(self.width):
                k = (kx, ky)
                d_index = kx * self.width + ky
                self.k_to_d_mapping[k] = d_index
                self.d_to_k_mapping[d_index] = k

    def coeff_to_vector(self, coeff_matrix):
        return coeff_matrix.flatten()

    def vector_to_coeff(self, d_vector):
        return d_vector.view(self.height, self.width)

    def flatten_batch(self, coeff_batch):
        return coeff_batch.view(coeff_batch.shape[0], -1)

    def unflatten_batch(self, d_batch):
        return d_batch.view(d_batch.shape[0], 1, self.height, self.width)

    def verify_mapping_consistency(self):
        coeff_matrix = torch.randn(self.E_plus_shape)
        d_vector = self.coeff_to_vector(coeff_matrix)
        recovered_coeff = self.vector_to_coeff(d_vector)
        error = torch.norm(coeff_matrix - recovered_coeff)
        return error.item()


# ============================================================================
# 2. Theoretical gradient channels
# ============================================================================
class TheoreticalGradientDescent(nn.Module):
    def __init__(self, height=IMAGE_SIZE, width=IMAGE_SIZE, regularizer_type="tikhonov", lambda_reg=0.01, operator=None):
        super().__init__()
        self.operator = operator if operator is not None else build_time_domain_operator(height=height, width=width)
        self.regularizer_type = regularizer_type
        self.lambda_reg = lambda_reg
        self.step_size = 1e-2
        self.register_buffer(
            "laplace_kernel",
            torch.tensor(
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
            ).view(1, 1, 3, 3),
        )

    def _compute_weighted_residual(self, g_pred, g_obs):
        mode = str(DATA_CONFIG.get("data_fidelity_mode", "standard")).strip().lower()
        residual = g_pred - g_obs
        if mode != "irls":
            return residual
        abs_pred = torch.abs(g_pred)
        eps_factor = float(DATA_CONFIG.get("irls_eps_factor", 3.0e-3))
        median_abs = torch.median(abs_pred.view(abs_pred.shape[0], -1), dim=1).values.clamp_min(1e-6)
        eps = (eps_factor * median_abs).view(-1, 1)
        denom = (abs_pred + eps).pow(2)
        if bool(DATA_CONFIG.get("irls_detach_weights", True)):
            denom = denom.detach()
        return residual / denom.clamp_min(1e-8)

    def compute_data_fidelity_gradient(self, coeff_matrix, g_observed, return_per_angle=False):
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        g_pred = self.operator(coeff_matrix)
        g_obs = g_observed.to(dtype=g_pred.dtype, device=g_pred.device)
        residual = self._compute_weighted_residual(g_pred, g_obs)
        num_angles = int(getattr(self.operator, "num_angles", 1) or 1)
        if return_per_angle and num_angles > 1 and hasattr(self.operator, "split_measurements") and hasattr(self.operator, "adjoint_per_angle"):
            residual_pa = self.operator.split_measurements(residual)
            gradient_pa = self.operator.adjoint_per_angle(residual_pa)
            gradient = gradient_pa.mean(dim=1)
            return 2.0 * gradient, 2.0 * gradient_pa
        gradient = self.operator.adjoint(residual)
        if num_angles > 1:
            gradient = gradient / float(num_angles)
        if return_per_angle:
            return 2.0 * gradient, (2.0 * gradient).unsqueeze(1)
        return 2.0 * gradient

    def compute_regularization_gradient(self, coeff_matrix):
        if self.regularizer_type == "dirichlet":
            return self._dirichlet_gradient(coeff_matrix)
        if self.regularizer_type == "tikhonov":
            return 2 * coeff_matrix
        if self.regularizer_type == "tv":
            return self._tv_gradient(coeff_matrix)
        return torch.zeros_like(coeff_matrix)

    def _tv_gradient(self, coeff_matrix):
        eps = coeff_matrix.new_tensor(1e-6)
        grad_x, grad_y = self._forward_gradient(coeff_matrix)
        grad_norm = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)
        return -self._divergence(grad_x / grad_norm, grad_y / grad_norm)

    def _dirichlet_gradient(self, coeff_matrix):
        padded = F.pad(coeff_matrix, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, self.laplace_kernel.to(coeff_matrix), padding=0)

    def _forward_gradient(self, x):
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(x)
        grad_y[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        grad_x[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]
        return grad_x, grad_y

    def _divergence(self, grad_x, grad_y):
        div = torch.zeros_like(grad_x)
        div[:, :, 0, :] += grad_y[:, :, 0, :]
        div[:, :, 1:, :] += grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :]
        div[:, :, :, 0] += grad_x[:, :, :, 0]
        div[:, :, :, 1:] += grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1]
        return div

    def gradient_descent_step(self, coeff_matrix, g_observed):
        data_grad = self.compute_data_fidelity_gradient(coeff_matrix, g_observed)
        reg_grad = self.compute_regularization_gradient(coeff_matrix)
        total_grad = data_grad + self.lambda_reg * reg_grad
        return coeff_matrix - self.step_size * total_grad


# ============================================================================
# 3. Learned gradient descent (CNN updates)
# ============================================================================
class LearnedGradientDescent(nn.Module):
    def __init__(self, height=IMAGE_SIZE, width=IMAGE_SIZE, regularizer_type="tikhonov", n_iter=10, n_memory=5):
        super().__init__()
        self.n_iter = int(n_iter)
        self.n_memory = int(n_memory)
        self.height = int(height)
        self.width = int(width)
        self.operator = build_time_domain_operator(height=height, width=width)
        self.num_angles = int(getattr(self.operator, "num_angles", 1) or 1)
        if not hasattr(self.operator, "split_measurements") or not hasattr(self.operator, "adjoint_per_angle"):
            raise ValueError("Alpha-only learned optimizer requires per-angle operator support.")

        self.requested_cnn_num_angles = TIME_DOMAIN_CONFIG.get("cnn_num_angles_override", None)
        if self.requested_cnn_num_angles is not None:
            self.requested_cnn_num_angles = int(self.requested_cnn_num_angles)
            if self.requested_cnn_num_angles <= 0:
                raise ValueError("cnn_num_angles_override must be positive when provided.")
            if self.requested_cnn_num_angles > self.num_angles:
                raise ValueError(f"cnn_num_angles_override={self.requested_cnn_num_angles} exceeds num_angles={self.num_angles}.")
        self.requested_cnn_angle_indices = TIME_DOMAIN_CONFIG.get("cnn_angle_indices_override", None)
        self.learned_operator = self.operator
        self.learned_num_angles = self.num_angles
        self.cnn_channel_indices = self._resolve_cnn_channel_indices()
        self.raw_cnn_num_angles = len(self.cnn_channel_indices)
        self.cnn_num_angles = self.raw_cnn_num_angles
        self.theoretical_gd = TheoreticalGradientDescent(height, width, regularizer_type, operator=self.learned_operator)

        self.data_fidelity_channel_mode = str(
            DATA_CONFIG.get("data_fidelity_channel_mode", "per_angle")
        ).strip().lower()
        if self.data_fidelity_channel_mode == "per_angle":
            self.data_fidelity_channels = self.raw_cnn_num_angles
        elif self.data_fidelity_channel_mode == "stacked_selected":
            self.data_fidelity_channels = 1
        elif self.data_fidelity_channel_mode == "both_selected":
            self.data_fidelity_channels = self.raw_cnn_num_angles + 1
        else:
            raise ValueError(
                f"data_fidelity_channel_mode={self.data_fidelity_channel_mode!r}; "
                "expected 'per_angle', 'stacked_selected', or 'both_selected'."
            )

        self.physics_residual_enabled = bool(TIME_DOMAIN_CONFIG.get("physics_residual_channel_enabled", False))
        self.physics_residual_mode = str(TIME_DOMAIN_CONFIG.get("physics_residual_mode", "per_angle_cg")).strip().lower()
        if self.physics_residual_mode not in {"stacked_cg", "stacked_selected_cg", "per_angle_cg"}:
            raise ValueError(
                f"physics_residual_mode={self.physics_residual_mode!r}; "
                "expected 'per_angle_cg', 'stacked_cg', or 'stacked_selected_cg'."
            )
        self.physics_residual_damping = float(TIME_DOMAIN_CONFIG.get("physics_residual_damping", 1.0e-2))
        self.physics_residual_cg_iters = int(TIME_DOMAIN_CONFIG.get("physics_residual_cg_iters", 8))
        self.physics_residual_detach = bool(TIME_DOMAIN_CONFIG.get("physics_residual_detach", True))
        self.physics_residual_normalize = bool(TIME_DOMAIN_CONFIG.get("physics_residual_normalize", True))
        if self.physics_residual_enabled and self.physics_residual_mode == "per_angle_cg":
            self.physics_residual_channels = self.raw_cnn_num_angles
        else:
            self.physics_residual_channels = 1 if self.physics_residual_enabled else 0
        self.physics_explicit_update_enabled = bool(TIME_DOMAIN_CONFIG.get("physics_explicit_update_enabled", False))
        self.physics_explicit_update_max = float(TIME_DOMAIN_CONFIG.get("physics_explicit_update_max", 0.10))
        phys_alpha_init = max(float(TIME_DOMAIN_CONFIG.get("physics_explicit_update_alpha_init", 0.02)), 1.0e-8)
        self.physics_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(phys_alpha_init) - 1.0), dtype=torch.float32))

        self.input_channels = 1 + self.data_fidelity_channels + self.physics_residual_channels + 1 + self.n_memory
        self.detach_physical_grads = bool(DATA_CONFIG.get("detach_physical_grads", True))
        self.learned_correction_max = float(DATA_CONFIG.get("learned_correction_max", 0.0))
        self.update_max_norm = float(DATA_CONFIG.get("update_max_norm", 0.0))
        self.learned_step_max = float(DATA_CONFIG.get("learned_step_max", 0.0))
        self.learned_reg_lambda_max = float(DATA_CONFIG.get("learned_reg_lambda_max", 0.0))
        self.feature_channels = 64
        self.update_network = self._build_update_network(self.input_channels)

        step_min = float(DATA_CONFIG.get("learned_step_min", 1.0e-6))
        self.step_min = step_min
        target_init = max(float(DATA_CONFIG.get("learned_step_init", 1.0e-2)) - step_min, 1e-8)
        self.step_size_raw = nn.Parameter(torch.tensor(math.log(math.exp(target_init) - 1.0), dtype=torch.float32))

        lambda_min = 1e-5
        self.lambda_min = lambda_min
        target_lambda = max(float(DATA_CONFIG.get("learned_reg_lambda_init", 1.0e-3)) - lambda_min, 1e-8)
        self.reg_lambda_raw = nn.Parameter(torch.tensor(math.log(math.exp(target_lambda) - 1.0), dtype=torch.float32))

    def _resolve_cnn_channel_indices(self):
        if self.requested_cnn_angle_indices is not None:
            indices = [int(idx) for idx in list(self.requested_cnn_angle_indices)]
            if not indices:
                raise ValueError("cnn_angle_indices_override must not be empty when provided.")
            if len(set(indices)) != len(indices):
                raise ValueError(f"cnn_angle_indices_override contains duplicates: {indices!r}.")
            invalid = [idx for idx in indices if idx < 0 or idx >= self.learned_num_angles]
            if invalid:
                raise ValueError(f"cnn_angle_indices_override contains out-of-range indices {invalid!r} for learned_num_angles={self.learned_num_angles}.")
            return indices
        if self.requested_cnn_num_angles is not None:
            return list(range(int(self.requested_cnn_num_angles)))
        return list(range(self.learned_num_angles))

    def _cnn_angle_index_tensor(self, target_device):
        return torch.as_tensor(self.cnn_channel_indices, device=target_device, dtype=torch.long)

    def _select_cnn_angle_channels(self, per_angle_tensor: torch.Tensor) -> torch.Tensor:
        if int(per_angle_tensor.shape[1]) < int(max(self.cnn_channel_indices) + 1):
            raise ValueError(
                f"Per-angle tensor has {int(per_angle_tensor.shape[1])} channels, "
                f"but requested indices are {self.cnn_channel_indices!r}."
            )
        return torch.index_select(
            per_angle_tensor,
            dim=1,
            index=self._cnn_angle_index_tensor(per_angle_tensor.device),
        )

    def _build_update_network(self, input_channels):
        return nn.Sequential(
            nn.InstanceNorm2d(input_channels, affine=True),
            nn.Conv2d(input_channels, self.feature_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(self.feature_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(self.feature_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=4, dilation=4),
            nn.InstanceNorm2d(self.feature_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=8, dilation=8),
            nn.InstanceNorm2d(self.feature_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels, 1 + self.n_memory, kernel_size=3, padding=1),
        )

    def _compose_data_fidelity_channels(self, data_grad=None, data_grad_pa=None):
        if data_grad_pa is None:
            raise ValueError(f"data_grad_pa is required for data_fidelity_channel_mode={self.data_fidelity_channel_mode!r}.")
        selected_pa = self._select_cnn_angle_channels(data_grad_pa).squeeze(2)
        if self.data_fidelity_channel_mode == "per_angle":
            return selected_pa
        if self.data_fidelity_channel_mode == "stacked_selected":
            return selected_pa.mean(dim=1, keepdim=True)
        if self.data_fidelity_channel_mode == "both_selected":
            return torch.cat([selected_pa, selected_pa.mean(dim=1, keepdim=True)], dim=1)
        raise ValueError(f"Unsupported data_fidelity_channel_mode={self.data_fidelity_channel_mode!r}.")

    def _compose_cnn_input(
        self,
        coeff_current,
        g_observed,
        reg_grad,
        memory,
        data_grad=None,
        data_grad_pa=None,
        physics_corr=None,
    ):
        if data_grad_pa is None:
            computed_data_grad, computed_data_grad_pa = self.theoretical_gd.compute_data_fidelity_gradient(
                coeff_current,
                g_observed,
                return_per_angle=True,
            )
            if data_grad is None:
                data_grad = computed_data_grad
            data_grad_pa = computed_data_grad_pa
        grad_channels = self._compose_data_fidelity_channels(
            data_grad=data_grad,
            data_grad_pa=data_grad_pa,
        )
        if int(grad_channels.shape[1]) != int(self.data_fidelity_channels):
            raise ValueError(
                f"data fidelity channels has {int(grad_channels.shape[1])} channels, "
                f"expected {int(self.data_fidelity_channels)}."
            )
        parts = [coeff_current, grad_channels]
        if self.physics_residual_enabled:
            if physics_corr is None:
                raise ValueError("physics_residual_channel_enabled=True, but physics_corr is None.")
            if self.physics_residual_mode == "per_angle_cg" and int(physics_corr.shape[1]) != int(self.physics_residual_channels):
                physics_corr = self._select_cnn_angle_channels(physics_corr)
            if int(physics_corr.shape[1]) != int(self.physics_residual_channels):
                raise ValueError(
                    f"physics residual has {int(physics_corr.shape[1])} channels, "
                    f"expected {int(self.physics_residual_channels)}."
                )
            parts.append(physics_corr)
        parts.extend([reg_grad, memory])
        return torch.cat(parts, dim=1)

    def _select_learned_measurements(self, g_observed):
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        return g_observed

    def _split_observations(self, g_observed):
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        return g_observed, None

    def _cap_correction(self, correction):
        if self.learned_correction_max <= 0:
            return correction
        return self.learned_correction_max * torch.tanh(correction / self.learned_correction_max)

    def _clip_update_norm(self, update):
        if self.update_max_norm <= 0:
            return update
        flat = update.view(update.shape[0], -1)
        norms = torch.norm(flat, dim=1, keepdim=True).clamp_min(1e-8)
        scales = torch.clamp(self.update_max_norm / norms, max=1.0)
        return update * scales.view(-1, 1, 1, 1)

    def current_step_size(self):
        step = self.step_min + F.softplus(self.step_size_raw)
        if self.learned_step_max > 0:
            step = torch.clamp(step, max=self.learned_step_max)
        return step

    def current_reg_lambda(self):
        lam = self.lambda_min + F.softplus(self.reg_lambda_raw)
        if self.learned_reg_lambda_max > 0:
            lam = torch.clamp(lam, max=self.learned_reg_lambda_max)
        return lam

    def current_physics_alpha(self):
        alpha = F.softplus(self.physics_alpha_raw)
        if self.physics_explicit_update_max > 0:
            alpha = torch.clamp(alpha, max=self.physics_explicit_update_max)
        return alpha

    def forward(self, coeff_initial, g_observed):
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        batch_size = coeff_initial.shape[0]
        coeff_current = coeff_initial.clone()
        g_observed_learned = self._select_learned_measurements(g_observed)
        memory = torch.zeros(batch_size, self.n_memory, self.height, self.width, device=coeff_initial.device)
        history = [coeff_current.clone()]
        for _ in range(self.n_iter):
            lambda_i = self.current_reg_lambda()
            reg_grad_base = self.theoretical_gd.compute_regularization_gradient(coeff_current)
            data_grad, data_grad_pa = self.theoretical_gd.compute_data_fidelity_gradient(
                coeff_current,
                g_observed_learned,
                return_per_angle=True,
            )

            physics_corr = None
            physics_update_corr = None
            if self.physics_residual_enabled or self.physics_explicit_update_enabled:
                op = self.theoretical_gd.operator
                if self.physics_residual_mode == "per_angle_cg":
                    if not hasattr(op, "residual_inverse_correction_per_angle"):
                        raise ValueError("The active operator does not implement residual_inverse_correction_per_angle().")
                    physics_corr_pa = op.residual_inverse_correction_per_angle(
                        coeff_current,
                        g_observed_learned,
                        damping=self.physics_residual_damping,
                        cg_iters=self.physics_residual_cg_iters,
                        detach=self.physics_residual_detach,
                        normalize=self.physics_residual_normalize,
                    )
                    selected_physics_corr_pa = self._select_cnn_angle_channels(physics_corr_pa)
                    physics_corr = selected_physics_corr_pa.squeeze(2)
                    physics_update_corr = selected_physics_corr_pa.mean(dim=1)
                elif self.physics_residual_mode == "stacked_selected_cg":
                    if not hasattr(op, "residual_inverse_correction_selected_angles"):
                        raise ValueError("The active operator does not implement residual_inverse_correction_selected_angles().")
                    physics_corr = op.residual_inverse_correction_selected_angles(
                        coeff_current,
                        g_observed_learned,
                        angle_indices=self.cnn_channel_indices,
                        damping=self.physics_residual_damping,
                        cg_iters=self.physics_residual_cg_iters,
                        detach=self.physics_residual_detach,
                        normalize=self.physics_residual_normalize,
                    )
                    physics_update_corr = physics_corr
                else:
                    if not hasattr(op, "residual_inverse_correction"):
                        raise ValueError("The active operator does not implement residual_inverse_correction().")
                    physics_corr = op.residual_inverse_correction(
                        coeff_current,
                        g_observed_learned,
                        damping=self.physics_residual_damping,
                        cg_iters=self.physics_residual_cg_iters,
                        detach=self.physics_residual_detach,
                        normalize=self.physics_residual_normalize,
                    )
                    physics_update_corr = physics_corr
            if self.detach_physical_grads:
                reg_grad_base = reg_grad_base.detach()
                data_grad = data_grad.detach()
                data_grad_pa = data_grad_pa.detach()
            if self.physics_residual_detach and physics_corr is not None:
                physics_corr = physics_corr.detach()
                if physics_update_corr is not None:
                    physics_update_corr = physics_update_corr.detach()
            reg_grad = reg_grad_base * lambda_i
            cnn_input = self._compose_cnn_input(
                coeff_current,
                g_observed_learned,
                reg_grad,
                memory,
                data_grad=data_grad,
                data_grad_pa=data_grad_pa,
                physics_corr=physics_corr,
            )
            cnn_output = self.update_network(cnn_input)
            raw_update = cnn_output[:, 0:1, :, :]
            new_memory = cnn_output[:, 1:, :, :]
            learned_update = self._cap_correction(raw_update) * self.current_step_size()
            phys_update = torch.zeros_like(learned_update)
            if self.physics_explicit_update_enabled and physics_update_corr is not None:
                phys_update = self.current_physics_alpha() * physics_update_corr
            total_update = self._clip_update_norm(learned_update - phys_update)
            coeff_current = coeff_current - total_update
            memory = torch.relu(new_memory)
            history.append(coeff_current.clone())
        return coeff_current, history


# ============================================================================
# 5. Full CT network
# ============================================================================
class TheoreticalCTNet(nn.Module):
    def __init__(self, height=IMAGE_SIZE, width=IMAGE_SIZE, regularizer_type="tikhonov", n_iter=10, n_memory=5):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.optimizer = LearnedGradientDescent(height, width, regularizer_type, n_iter, n_memory)
        self.mapping = CoefficientMapping((height, width))

    def forward(self, coeff_initial, g_observed):
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        coeff_final, history = self.optimizer(coeff_initial, g_observed)
        metrics = self._compute_optimization_metrics(coeff_initial, coeff_final, g_observed, history)
        return coeff_final, history, metrics

    def _compute_optimization_metrics(self, coeff_initial, coeff_final, g_observed, history):
        metrics = {}
        with torch.no_grad():
            g_observed_main, _ = self.optimizer._split_observations(g_observed)
            g_final = self.optimizer.operator(coeff_final)
            data_fidelity_error = torch.norm(g_final - g_observed_main, dim=-1).mean()
            metrics["data_fidelity_error"] = data_fidelity_error.item()
            coeff_change = torch.norm(coeff_final - coeff_initial, dim=(2, 3)).mean()
            metrics["coefficient_change"] = coeff_change.item()
            if self.optimizer.theoretical_gd.regularizer_type == "tikhonov":
                reg_value = torch.norm(coeff_final, dim=(2, 3)) ** 2
                metrics["regularization_value"] = reg_value.mean().item()
            elif self.optimizer.theoretical_gd.regularizer_type == "dirichlet":
                grad_y = torch.diff(coeff_final, dim=2, prepend=coeff_final[:, :, -1:])
                grad_x = torch.diff(coeff_final, dim=3, prepend=coeff_final[:, :, :, -1:])
                reg_value = 0.5 * (grad_x.pow(2) + grad_y.pow(2)).sum(dim=(2, 3))
                metrics["regularization_value"] = reg_value.mean().item()
            metrics["update_difference"] = coeff_change.item()
        return metrics


# ============================================================================
# 6. Helpers
# ============================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_trainable_state_dict(model: nn.Module, *, move_to_cpu: bool = True):
    state = OrderedDict()
    for name, param in model.named_parameters():
        tensor = param.detach()
        if move_to_cpu:
            tensor = tensor.cpu()
        else:
            tensor = tensor.clone()
        state[name] = tensor.clone()
    return state


def load_trainable_state_dict(model: nn.Module, state_dict):
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict to be a dict-like object, got {type(state_dict).__name__}.")
    parameter_map = OrderedDict(model.named_parameters())
    parameter_names = set(parameter_map.keys())
    filtered = OrderedDict()
    unexpected = []
    for key, value in state_dict.items():
        if key in parameter_names:
            filtered[key] = value
        else:
            unexpected.append(key)
    missing_parameters = [name for name in parameter_map.keys() if name not in filtered]
    if missing_parameters:
        preview = missing_parameters[:8]
        raise RuntimeError(f"Checkpoint is missing trainable parameters required by the current model: {preview}{' ...' if len(missing_parameters) > len(preview) else ''}")
    incompatible = model.load_state_dict(filtered, strict=False)
    missing_buffers = [name for name in incompatible.missing_keys if name not in parameter_names]
    missing_named_parameters = [name for name in incompatible.missing_keys if name in parameter_names]
    if missing_named_parameters:
        preview = missing_named_parameters[:8]
        raise RuntimeError(f"load_state_dict reported missing trainable parameters after filtering: {preview}{' ...' if len(missing_named_parameters) > len(preview) else ''}")
    if incompatible.unexpected_keys:
        preview = list(incompatible.unexpected_keys)[:8]
        raise RuntimeError(f"load_state_dict reported unexpected keys after filtering: {preview}{' ...' if len(incompatible.unexpected_keys) > len(preview) else ''}")
    return {"loaded_parameter_count": int(len(filtered)), "ignored_non_parameter_keys": unexpected, "missing_buffer_keys": missing_buffers}


def initialize_model():
    regularizer_type = THEORETICAL_CONFIG["regularizer_type"]
    n_iter = THEORETICAL_CONFIG["n_iter"]
    n_memory = THEORETICAL_CONFIG["n_memory_units"]
    model = TheoreticalCTNet(height=IMAGE_SIZE, width=IMAGE_SIZE, regularizer_type=regularizer_type, n_iter=n_iter, n_memory=n_memory).to(device)
    print(f"Model initialized on device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print("Using alpha-continuous theoretical GD block")
    print(f"Regularizer type: {regularizer_type}")
    print(f"Optimization iterations: {n_iter}")
    print(f"Memory units: {n_memory}")
    print(
        "Physical angles / learned data angles / CNN angle channels: "
        f"{model.optimizer.num_angles} / {model.optimizer.learned_num_angles} / "
        f"{model.optimizer.cnn_num_angles}"
    )
    print(
        "Data fidelity channel mode: %s channels=%d"
        % (
            str(getattr(model.optimizer, "data_fidelity_channel_mode", "per_angle")),
            int(getattr(model.optimizer, "data_fidelity_channels", 0) or 0),
        )
    )
    print(
        "Physics residual: enabled=%s mode=%s damping=%.3g cg_iters=%d channels=%d explicit_update=%s alpha=%.4f"
        % (
            bool(getattr(model.optimizer, "physics_residual_enabled", False)),
            str(getattr(model.optimizer, "physics_residual_mode", "disabled")),
            float(getattr(model.optimizer, "physics_residual_damping", 0.0)),
            int(getattr(model.optimizer, "physics_residual_cg_iters", 0) or 0),
            int(getattr(model.optimizer, "physics_residual_channels", 0) or 0),
            bool(getattr(model.optimizer, "physics_explicit_update_enabled", False)),
            float(model.optimizer.current_physics_alpha().detach().cpu().item()) if hasattr(model.optimizer, "current_physics_alpha") else 0.0,
        )
    )
    return model


if __name__ == "__main__":
    model = initialize_model()
    batch_size = 2
    mapping = CoefficientMapping()
    x_0 = torch.randn(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
    M = model.optimizer.operator.M
    y_fake = torch.randn(batch_size, M, dtype=torch.float32).to(device)
    with torch.no_grad():
        coeff_pred, history, metrics = model(x_0, y_fake)
        print(f"input shape: {x_0.shape}")
        print(f"output shape: {coeff_pred.shape}")
        print(f"observed shape: {y_fake.shape}")
        print(f"iterations: {len(history)-1}")
        print(f"data fidelity error: {metrics['data_fidelity_error']:.6f}")
        print(f"update difference: {metrics['update_difference']:.6f}")
        mapping_error = mapping.verify_mapping_consistency()
        print(f"mapping error: {mapping_error:.6f} (should be ~0)")
    print("Alpha-only LGD model test successful!")
