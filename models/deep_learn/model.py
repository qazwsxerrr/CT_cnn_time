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
# 3. Non-iterative residual U-Net refiner blocks
# ============================================================================
def _group_norm(num_channels):
    groups = min(8, int(num_channels))
    while groups > 1 and int(num_channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(num_channels))


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvNormAct(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvNormAct(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = int(skip.shape[-2]) - int(x.shape[-2])
        diff_x = int(skip.shape[-1]) - int(x.shape[-1])
        if diff_x != 0 or diff_y != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class ResidualUNet(nn.Module):
    def __init__(self, input_channels, base_channels=32, depth=4, residual_max=0.0):
        super().__init__()
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.residual_max = float(residual_max)
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")

        channels = [self.base_channels * (2 ** i) for i in range(self.depth + 1)]
        self.input_norm = nn.InstanceNorm2d(self.input_channels, affine=True)
        self.inc = ConvNormAct(self.input_channels, channels[0])
        self.downs = nn.ModuleList(
            DownBlock(channels[i], channels[i + 1]) for i in range(self.depth)
        )
        self.ups = nn.ModuleList(
            UpBlock(channels[i + 1], channels[i], channels[i]) for i in range(self.depth - 1, -1, -1)
        )
        self.out_conv = nn.Conv2d(channels[0], 1, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.input_norm(x)
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        residual = self.out_conv(x)
        if self.residual_max > 0:
            residual = self.residual_max * torch.tanh(residual / self.residual_max)
        return residual


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, growth_rate=None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        if growth_rate is None:
            growth_rate = min(max(self.out_channels // 4, 16), 128)
        self.growth_rate = int(growth_rate)
        if self.in_channels <= 0 or self.out_channels <= 0 or self.num_layers <= 0:
            raise ValueError("DenseResidualBlock channel counts and num_layers must be positive.")

        layers = []
        current_channels = self.in_channels
        for _ in range(self.num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, self.growth_rate, kernel_size=3, padding=1),
                    _group_norm(self.growth_rate),
                    nn.SiLU(inplace=True),
                )
            )
            current_channels += self.growth_rate
        self.layers = nn.ModuleList(layers)
        self.compress = nn.Sequential(
            nn.Conv2d(current_channels, self.out_channels, kernel_size=1),
            _group_norm(self.out_channels),
        )
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        dense = self.compress(torch.cat(features, dim=1))
        return self.act(dense + self.shortcut(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        channels = int(channels)
        hidden = max(channels // int(reduction), 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        weight = torch.sigmoid(self.mlp(avg) + self.mlp(max_pool))
        return x * weight


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = int(kernel_size) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.amax(x, dim=1, keepdim=True)
        weight = torch.sigmoid(self.conv(torch.cat([avg, max_pool], dim=1)))
        return x * weight


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


class RADBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense_residual = DenseResidualBlock(in_channels, out_channels)
        self.attention = CBAM(out_channels)

    def forward(self, x):
        x = self.dense_residual(x)
        return self.attention(x) + x


class AttentionGate(nn.Module):
    def __init__(self, skip_channels, gate_channels, inter_channels=None):
        super().__init__()
        skip_channels = int(skip_channels)
        gate_channels = int(gate_channels)
        if inter_channels is None:
            inter_channels = max(min(skip_channels, gate_channels) // 2, 8)
        inter_channels = int(inter_channels)
        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)

    def forward(self, skip, gate):
        gate_resized = gate
        if gate.shape[-2:] != skip.shape[-2:]:
            gate_resized = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.psi(F.silu(self.skip_proj(skip) + self.gate_proj(gate_resized)))
        return skip * torch.sigmoid(logits)


class RADDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = RADBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))


class RADUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attention_gate = AttentionGate(skip_channels, in_channels)
        self.block = RADBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = int(skip.shape[-2]) - int(x.shape[-2])
        diff_x = int(skip.shape[-1]) - int(x.shape[-1])
        if diff_x != 0 or diff_y != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        skip = self.attention_gate(skip, x)
        return self.block(torch.cat([skip, x], dim=1))


class RADUNet(nn.Module):
    """Residual-attention-dense U-Net that predicts one residual image."""

    def __init__(self, input_channels, base_channels=32, depth=4, residual_max=0.0):
        super().__init__()
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.residual_max = float(residual_max)
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")

        channels = [self.base_channels * (2 ** i) for i in range(self.depth + 1)]
        self.input_norm = nn.InstanceNorm2d(self.input_channels, affine=True)
        self.inc = RADBlock(self.input_channels, channels[0])
        self.downs = nn.ModuleList(
            RADDownBlock(channels[i], channels[i + 1]) for i in range(self.depth)
        )
        self.ups = nn.ModuleList(
            RADUpBlock(channels[i + 1], channels[i], channels[i]) for i in range(self.depth - 1, -1, -1)
        )
        self.out_conv = nn.Conv2d(channels[0], 1, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.input_norm(x)
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        residual = self.out_conv(x)
        if self.residual_max > 0:
            residual = self.residual_max * torch.tanh(residual / self.residual_max)
        return residual


class FullResolutionDetailHead(nn.Module):
    """Small same-resolution residual head for sharpening local details."""

    def __init__(self, input_channels, hidden_channels=16, depth=2, residual_max=0.0, zero_init=True):
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.depth = int(depth)
        self.residual_max = float(residual_max)
        if self.input_channels <= 0:
            raise ValueError("detail head input_channels must be positive.")
        if self.hidden_channels <= 0:
            raise ValueError("detail head hidden_channels must be positive.")
        if self.depth <= 0:
            raise ValueError("detail head depth must be positive.")

        layers = [
            nn.InstanceNorm2d(self.input_channels, affine=True),
            nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, padding=1),
            _group_norm(self.hidden_channels),
            nn.SiLU(inplace=True),
        ]
        for _ in range(self.depth - 1):
            layers.extend(
                [
                    nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                    _group_norm(self.hidden_channels),
                    nn.SiLU(inplace=True),
                ]
            )
        out_conv = nn.Conv2d(self.hidden_channels, 1, kernel_size=3, padding=1)
        if bool(zero_init):
            nn.init.zeros_(out_conv.weight)
            nn.init.zeros_(out_conv.bias)
        layers.append(out_conv)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        detail = self.net(x)
        if self.residual_max > 0:
            detail = self.residual_max * torch.tanh(detail / self.residual_max)
        return detail


# ============================================================================
# 4. Learned gradient descent (CNN updates)
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
# 5. TV-init physics-conditioned non-iterative U-Net refiner
# ============================================================================
class TVPCUNetRefiner(nn.Module):
    """One-shot residual U-Net conditioned on TV-init physics features.

    B variant: x_pred = x_tv + U-Net(features).
    C variant: x_pred = x_tv + alpha * physics_corr0 + U-Net(features), enabled
    by PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE=1.
    """

    def __init__(self, height=IMAGE_SIZE, width=IMAGE_SIZE, regularizer_type="tikhonov"):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.n_iter = max(int(THEORETICAL_CONFIG.get("refiner_stages", 1)), 1)
        self.n_memory = 0
        self.model_arch = str(THEORETICAL_CONFIG.get("model_arch", "tv_pc_unet")).strip().lower()
        self.refiner_stages = self.n_iter
        self.refiner_share_weights = bool(THEORETICAL_CONFIG.get("refiner_share_weights", True))
        self.stage_dc_enabled = bool(THEORETICAL_CONFIG.get("refiner_stage_dc_enabled", False))
        self.stage_dc_cg_iters = int(THEORETICAL_CONFIG.get("refiner_stage_dc_cg_iters", 4))
        self.stage_dc_damping = float(THEORETICAL_CONFIG.get("refiner_stage_dc_damping", 1.0e-2))
        self.stage_dc_detach = bool(THEORETICAL_CONFIG.get("refiner_stage_dc_detach", True))
        self.stage_dc_normalize = bool(THEORETICAL_CONFIG.get("refiner_stage_dc_normalize", True))
        self.refiner_input_mode = str(THEORETICAL_CONFIG.get("refiner_input_mode", "u2_stacked")).strip().lower()
        if self.refiner_input_mode not in {"u2", "u2_stacked", "physics_conditioned_u2", "u2_alpha_stack"}:
            raise ValueError(
                f"refiner_input_mode={self.refiner_input_mode!r}; "
                "expected 'u2_stacked' or 'u2_alpha_stack' for TV-PC U-Net."
            )

        self.operator = build_time_domain_operator(height=height, width=width)
        self.num_angles = int(getattr(self.operator, "num_angles", 1) or 1)
        if not hasattr(self.operator, "split_measurements") or not hasattr(self.operator, "adjoint_per_angle"):
            raise ValueError("TV-PC U-Net requires per-angle operator support for U2 stacked features.")

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
            DATA_CONFIG.get("data_fidelity_channel_mode", "stacked_selected")
        ).strip().lower()
        if self.refiner_input_mode == "u2_alpha_stack":
            if self.data_fidelity_channel_mode not in {"per_angle", "stacked_selected"}:
                raise ValueError(
                    f"TV-PC U-Net u2_alpha_stack expects data_fidelity_channel_mode='per_angle' "
                    f"or 'stacked_selected', got {self.data_fidelity_channel_mode!r}."
                )
            self.data_fidelity_channels = self.raw_cnn_num_angles
        elif self.data_fidelity_channel_mode != "stacked_selected":
            raise ValueError(
                f"TV-PC U-Net U2 expects data_fidelity_channel_mode='stacked_selected', "
                f"got {self.data_fidelity_channel_mode!r}."
            )
        else:
            self.data_fidelity_channels = 1

        self.physics_residual_enabled = bool(TIME_DOMAIN_CONFIG.get("physics_residual_channel_enabled", True))
        if not self.physics_residual_enabled:
            raise ValueError("TV-PC U-Net U2 requires physics_residual_channel_enabled=True.")
        self.physics_residual_mode = str(TIME_DOMAIN_CONFIG.get("physics_residual_mode", "stacked_selected_cg")).strip().lower()
        if self.physics_residual_mode not in {"stacked_cg", "stacked_selected_cg", "per_angle_cg"}:
            raise ValueError(
                f"physics_residual_mode={self.physics_residual_mode!r}; "
                "expected 'per_angle_cg', 'stacked_cg', or 'stacked_selected_cg'."
            )
        self.physics_residual_damping = float(TIME_DOMAIN_CONFIG.get("physics_residual_damping", 1.0e-2))
        self.physics_residual_cg_iters = int(TIME_DOMAIN_CONFIG.get("physics_residual_cg_iters", 8))
        self.physics_residual_detach = bool(TIME_DOMAIN_CONFIG.get("physics_residual_detach", True))
        self.physics_residual_normalize = bool(TIME_DOMAIN_CONFIG.get("physics_residual_normalize", True))
        self.physics_residual_channels = 1

        self.physics_explicit_update_enabled = bool(TIME_DOMAIN_CONFIG.get("physics_explicit_update_enabled", False))
        self.physics_explicit_update_max = float(TIME_DOMAIN_CONFIG.get("physics_explicit_update_max", 0.10))
        phys_alpha_init = max(float(TIME_DOMAIN_CONFIG.get("physics_explicit_update_alpha_init", 0.02)), 1.0e-8)
        self.physics_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(phys_alpha_init) - 1.0), dtype=torch.float32))
        self.physics_gate_mode = str(THEORETICAL_CONFIG.get("physics_gate_mode", "scalar")).strip().lower()
        if self.physics_gate_mode not in {"scalar", "spatial"}:
            raise ValueError("physics_gate_mode must be 'scalar' or 'spatial'.")
        stage_alpha_count = max(self.refiner_stages - 1, 1)
        stage_alpha_raw_init = math.log(math.exp(phys_alpha_init) - 1.0)
        self.stage_dc_alpha_raw = nn.Parameter(torch.full((stage_alpha_count,), stage_alpha_raw_init, dtype=torch.float32))

        self.detach_physical_grads = bool(DATA_CONFIG.get("detach_physical_grads", True))
        self.input_channels = 1 + self.data_fidelity_channels + self.physics_residual_channels + 1
        base_channels = int(THEORETICAL_CONFIG.get("unet_base_channels", 32))
        depth = int(THEORETICAL_CONFIG.get("unet_depth", 4))
        residual_max = float(THEORETICAL_CONFIG.get("unet_residual_max", 0.0))
        self.unet_backbone = str(THEORETICAL_CONFIG.get("unet_backbone", "plain")).strip().lower()
        self.update_network = self._build_refiner_network(
            input_channels=self.input_channels,
            base_channels=base_channels,
            depth=depth,
            residual_max=residual_max,
        )
        self.extra_stage_networks = nn.ModuleList()
        if self.refiner_stages > 1 and not self.refiner_share_weights:
            self.extra_stage_networks.extend(
                self._build_refiner_network(
                    input_channels=self.input_channels,
                    base_channels=base_channels,
                    depth=depth,
                    residual_max=residual_max,
                )
                for _ in range(self.refiner_stages - 1)
            )
        self.detail_head_enabled = bool(THEORETICAL_CONFIG.get("detail_head_enabled", False))
        self.detail_head_input_mode = str(THEORETICAL_CONFIG.get("detail_head_input_mode", "features")).strip().lower()
        if self.detail_head_input_mode not in {"features", "features_residual", "features_residual_coeff"}:
            raise ValueError(
                f"detail_head_input_mode={self.detail_head_input_mode!r}; "
                "expected 'features', 'features_residual', or 'features_residual_coeff'."
            )
        self.detail_head_stage_policy = str(THEORETICAL_CONFIG.get("detail_head_stage_policy", "last")).strip().lower()
        if self.detail_head_stage_policy not in {"last", "all"}:
            raise ValueError("detail_head_stage_policy must be 'last' or 'all'.")
        self.detail_head_share_weights = bool(THEORETICAL_CONFIG.get("detail_head_share_weights", True))
        self.detail_head_input_channels = self._detail_head_input_channels()
        self.detail_head = None
        self.extra_detail_heads = nn.ModuleList()
        if self.detail_head_enabled:
            self.detail_head = self._build_detail_head(self.detail_head_input_channels)
            if self.refiner_stages > 1 and not self.detail_head_share_weights:
                self.extra_detail_heads.extend(
                    self._build_detail_head(self.detail_head_input_channels)
                    for _ in range(self.refiner_stages - 1)
                )
        self.physics_gate_network = None
        if self.physics_gate_mode == "spatial":
            hidden = max(min(base_channels, 64), 16)
            self.physics_gate_network = nn.Sequential(
                nn.InstanceNorm2d(self.input_channels, affine=True),
                nn.Conv2d(self.input_channels, hidden, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 1, kernel_size=1),
            )
            final_conv = self.physics_gate_network[-1]
            nn.init.zeros_(final_conv.weight)
            ratio = phys_alpha_init / max(self.physics_explicit_update_max, phys_alpha_init + 1.0e-6)
            ratio = min(max(ratio, 1.0e-4), 1.0 - 1.0e-4)
            nn.init.constant_(final_conv.bias, math.log(ratio / (1.0 - ratio)))

    def _build_refiner_network(self, input_channels, base_channels, depth, residual_max):
        if self.unet_backbone in {"plain", "residual_unet"}:
            return ResidualUNet(
                input_channels=input_channels,
                base_channels=base_channels,
                depth=depth,
                residual_max=residual_max,
            )
        if self.unet_backbone == "rad_unet":
            return RADUNet(
                input_channels=input_channels,
                base_channels=base_channels,
                depth=depth,
                residual_max=residual_max,
            )
        raise ValueError(
            f"Unsupported unet_backbone={self.unet_backbone!r}; "
            "expected 'plain', 'residual_unet', or 'rad_unet'."
        )

    def _detail_head_input_channels(self):
        channels = int(self.input_channels)
        if self.detail_head_input_mode == "features_residual":
            channels += 1
        elif self.detail_head_input_mode == "features_residual_coeff":
            channels += 2
        return channels

    def _build_detail_head(self, input_channels):
        return FullResolutionDetailHead(
            input_channels=input_channels,
            hidden_channels=int(THEORETICAL_CONFIG.get("detail_head_hidden_channels", 16)),
            depth=int(THEORETICAL_CONFIG.get("detail_head_depth", 2)),
            residual_max=float(THEORETICAL_CONFIG.get("detail_head_residual_max", 0.0)),
            zero_init=bool(THEORETICAL_CONFIG.get("detail_head_zero_init", True)),
        )

    def _get_stage_network(self, stage_idx):
        if self.refiner_share_weights or stage_idx == 0:
            return self.update_network
        return self.extra_stage_networks[int(stage_idx) - 1]

    def _get_detail_head(self, stage_idx):
        if self.detail_head is None:
            raise ValueError("detail_head_enabled=True requires a detail head module.")
        if self.detail_head_share_weights or stage_idx == 0:
            return self.detail_head
        return self.extra_detail_heads[int(stage_idx) - 1]

    def _detail_head_active_for_stage(self, stage_idx):
        if not self.detail_head_enabled:
            return False
        if self.detail_head_stage_policy == "all":
            return True
        return int(stage_idx) == int(self.refiner_stages) - 1

    def _compose_detail_features(self, features, residual, coeff_current):
        if self.detail_head_input_mode == "features":
            return features
        if self.detail_head_input_mode == "features_residual":
            return torch.cat([features, residual], dim=1)
        if self.detail_head_input_mode == "features_residual_coeff":
            return torch.cat([features, residual, coeff_current], dim=1)
        raise ValueError(f"Unsupported detail_head_input_mode={self.detail_head_input_mode!r}.")

    def _apply_detail_head(self, stage_idx, features, residual, coeff_current):
        if not self._detail_head_active_for_stage(stage_idx):
            return torch.zeros_like(residual)
        detail_features = self._compose_detail_features(features, residual, coeff_current)
        if int(detail_features.shape[1]) != int(self.detail_head_input_channels):
            raise ValueError(
                f"Detail head features have {int(detail_features.shape[1])} channels, "
                f"expected {int(self.detail_head_input_channels)}."
            )
        return self._get_detail_head(stage_idx)(detail_features)

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

    def _select_learned_measurements(self, g_observed):
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        return g_observed

    def _split_observations(self, g_observed):
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        return g_observed, None

    def current_physics_alpha(self):
        alpha = F.softplus(self.physics_alpha_raw)
        if self.physics_explicit_update_max > 0:
            alpha = torch.clamp(alpha, max=self.physics_explicit_update_max)
        return alpha

    def current_stage_dc_alpha(self, stage_idx=0):
        idx = min(max(int(stage_idx), 0), int(self.stage_dc_alpha_raw.numel()) - 1)
        alpha = F.softplus(self.stage_dc_alpha_raw[idx])
        if self.physics_explicit_update_max > 0:
            alpha = torch.clamp(alpha, max=self.physics_explicit_update_max)
        return alpha

    def _current_physics_gate(self, features):
        if self.physics_gate_mode == "spatial":
            if self.physics_gate_network is None:
                raise ValueError("physics_gate_mode='spatial' requires physics_gate_network.")
            scale = self.physics_explicit_update_max if self.physics_explicit_update_max > 0 else 1.0
            return float(scale) * torch.sigmoid(self.physics_gate_network(features))
        return self.current_physics_alpha().view(1, 1, 1, 1)

    def _apply_explicit_physics_update(self, features, physics_corr):
        if not self.physics_explicit_update_enabled:
            return torch.zeros_like(physics_corr)
        return self._current_physics_gate(features) * physics_corr

    @torch.no_grad()
    def physics_gate_diagnostics(self, coeff_initial=None, g_observed=None):
        diagnostics = {
            "gate_mode": self.physics_gate_mode,
            "legacy_alpha": float(self.current_physics_alpha().item()),
        }
        if self.stage_dc_enabled and hasattr(self, "stage_dc_alpha_raw"):
            diagnostics["stage_dc_alpha"] = [
                float(self.current_stage_dc_alpha(idx).item())
                for idx in range(max(self.refiner_stages - 1, 1))
            ]
        if self.physics_gate_mode != "spatial" or coeff_initial is None or g_observed is None:
            return diagnostics
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        g_observed_learned = self._select_learned_measurements(g_observed)
        features, physics_corr = self._compose_features(coeff_initial, g_observed_learned)
        gate = self._current_physics_gate(features)
        physics_update = gate * physics_corr
        diagnostics.update(
            {
                "gate_mean": float(gate.mean().item()),
                "gate_std": float(gate.std(unbiased=False).item()),
                "gate_min": float(gate.min().item()),
                "gate_max": float(gate.max().item()),
                "physics_corr_norm": float(torch.norm(physics_corr).item()),
                "physics_update_norm": float(torch.norm(physics_update).item()),
            }
        )
        return diagnostics

    def _compose_data_fidelity_channels(self, data_grad_pa):
        selected_pa = self._select_cnn_angle_channels(data_grad_pa).squeeze(2)
        if self.refiner_input_mode == "u2_alpha_stack":
            return selected_pa
        return selected_pa.mean(dim=1, keepdim=True)

    def _compute_inverse_residual_correction(
        self,
        coeff_initial,
        g_observed_learned,
        *,
        damping,
        cg_iters,
        detach,
        normalize,
    ):
        op = self.theoretical_gd.operator
        if self.physics_residual_mode == "per_angle_cg":
            if not hasattr(op, "residual_inverse_correction_per_angle"):
                raise ValueError("The active operator does not implement residual_inverse_correction_per_angle().")
            physics_corr_pa = op.residual_inverse_correction_per_angle(
                coeff_initial,
                g_observed_learned,
                damping=damping,
                cg_iters=cg_iters,
                detach=detach,
                normalize=normalize,
            )
            selected = self._select_cnn_angle_channels(physics_corr_pa)
            return selected.mean(dim=1)
        if self.physics_residual_mode == "stacked_selected_cg":
            if not hasattr(op, "residual_inverse_correction_selected_angles"):
                raise ValueError("The active operator does not implement residual_inverse_correction_selected_angles().")
            return op.residual_inverse_correction_selected_angles(
                coeff_initial,
                g_observed_learned,
                angle_indices=self.cnn_channel_indices,
                damping=damping,
                cg_iters=cg_iters,
                detach=detach,
                normalize=normalize,
            )
        if not hasattr(op, "residual_inverse_correction"):
            raise ValueError("The active operator does not implement residual_inverse_correction().")
        return op.residual_inverse_correction(
            coeff_initial,
            g_observed_learned,
            damping=damping,
            cg_iters=cg_iters,
            detach=detach,
            normalize=normalize,
        )

    def _compute_physics_correction(self, coeff_initial, g_observed_learned):
        return self._compute_inverse_residual_correction(
            coeff_initial,
            g_observed_learned,
            damping=self.physics_residual_damping,
            cg_iters=self.physics_residual_cg_iters,
            detach=self.physics_residual_detach,
            normalize=self.physics_residual_normalize,
        )

    def _compute_stage_dc_correction(self, coeff_current, g_observed_learned):
        return self._compute_inverse_residual_correction(
            coeff_current,
            g_observed_learned,
            damping=self.stage_dc_damping,
            cg_iters=self.stage_dc_cg_iters,
            detach=self.stage_dc_detach,
            normalize=self.stage_dc_normalize,
        )

    def _compose_features(self, coeff_initial, g_observed_learned):
        reg_grad = self.theoretical_gd.compute_regularization_gradient(coeff_initial)
        _data_grad, data_grad_pa = self.theoretical_gd.compute_data_fidelity_gradient(
            coeff_initial,
            g_observed_learned,
            return_per_angle=True,
        )
        grad_channels = self._compose_data_fidelity_channels(data_grad_pa)
        physics_corr = self._compute_physics_correction(coeff_initial, g_observed_learned)
        if self.detach_physical_grads:
            reg_grad = reg_grad.detach()
            grad_channels = grad_channels.detach()
        if self.physics_residual_detach:
            physics_corr = physics_corr.detach()
        if int(physics_corr.shape[1]) != 1:
            raise ValueError(f"TV-PC U-Net expects one physics residual channel, got {int(physics_corr.shape[1])}.")
        features = torch.cat([coeff_initial, grad_channels, physics_corr, reg_grad], dim=1)
        if int(features.shape[1]) != int(self.input_channels):
            raise ValueError(f"TV-PC U-Net features have {int(features.shape[1])} channels, expected {int(self.input_channels)}.")
        return features, physics_corr

    def forward(self, coeff_initial, g_observed):
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        x_tv = coeff_initial.clone()
        g_observed_learned = self._select_learned_measurements(g_observed)
        coeff_current = x_tv
        history = [x_tv.clone()]
        for stage_idx in range(self.refiner_stages):
            features, physics_corr = self._compose_features(coeff_current, g_observed_learned)
            residual = self._get_stage_network(stage_idx)(features)
            detail = self._apply_detail_head(stage_idx, features, residual, coeff_current)
            coeff_next = coeff_current + self._apply_explicit_physics_update(features, physics_corr) + residual + detail
            if self.stage_dc_enabled and stage_idx < self.refiner_stages - 1:
                dc_corr = self._compute_stage_dc_correction(coeff_next, g_observed_learned)
                coeff_next = coeff_next + self.current_stage_dc_alpha(stage_idx).view(1, 1, 1, 1) * dc_corr
            history.append(coeff_next.clone())
            coeff_current = coeff_next
        return coeff_current, history


# ============================================================================
# 6. Full CT network
# ============================================================================
class TheoreticalCTNet(nn.Module):
    def __init__(self, height=IMAGE_SIZE, width=IMAGE_SIZE, regularizer_type="tikhonov", n_iter=10, n_memory=5):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.model_arch = str(THEORETICAL_CONFIG.get("model_arch", "unrolled_cnn")).strip().lower()
        if self.model_arch in {"unrolled_cnn", "learned_gradient_descent", "lgd"}:
            self.optimizer = LearnedGradientDescent(height, width, regularizer_type, n_iter, n_memory)
        elif self.model_arch in {"tv_pc_unet", "tv_pc_refiner", "physics_unet", "tv_pc_cascade_unet"}:
            self.optimizer = TVPCUNetRefiner(height, width, regularizer_type)
        else:
            raise ValueError(
                f"Unsupported model_arch={self.model_arch!r}; "
                "expected 'unrolled_cnn', 'tv_pc_unet', or 'tv_pc_cascade_unet'."
            )
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


def _is_optional_detail_head_parameter(name: str) -> bool:
    return ".detail_head." in str(name) or ".extra_detail_heads." in str(name)


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
    initialized_missing_parameters = [
        name for name in missing_parameters if _is_optional_detail_head_parameter(name)
    ]
    required_missing_parameters = [
        name for name in missing_parameters if not _is_optional_detail_head_parameter(name)
    ]
    if required_missing_parameters:
        preview = required_missing_parameters[:8]
        suffix = " ..." if len(required_missing_parameters) > len(preview) else ""
        raise RuntimeError(
            "Checkpoint is missing trainable parameters required by the current model: "
            f"{preview}{suffix}"
        )
    incompatible = model.load_state_dict(filtered, strict=False)
    missing_buffers = [name for name in incompatible.missing_keys if name not in parameter_names]
    missing_named_parameters = [
        name
        for name in incompatible.missing_keys
        if name in parameter_names and not _is_optional_detail_head_parameter(name)
    ]
    if missing_named_parameters:
        preview = missing_named_parameters[:8]
        suffix = " ..." if len(missing_named_parameters) > len(preview) else ""
        raise RuntimeError(
            "load_state_dict reported missing trainable parameters after filtering: "
            f"{preview}{suffix}"
        )
    if incompatible.unexpected_keys:
        preview = list(incompatible.unexpected_keys)[:8]
        suffix = " ..." if len(incompatible.unexpected_keys) > len(preview) else ""
        raise RuntimeError(
            f"load_state_dict reported unexpected keys after filtering: {preview}{suffix}"
        )
    return {
        "loaded_parameter_count": int(len(filtered)),
        "ignored_non_parameter_keys": unexpected,
        "missing_buffer_keys": missing_buffers,
        "initialized_missing_parameter_keys": initialized_missing_parameters,
    }


def initialize_model():
    regularizer_type = THEORETICAL_CONFIG["regularizer_type"]
    n_iter = THEORETICAL_CONFIG["n_iter"]
    n_memory = THEORETICAL_CONFIG["n_memory_units"]
    model_arch = str(THEORETICAL_CONFIG.get("model_arch", "unrolled_cnn")).strip().lower()
    model = TheoreticalCTNet(height=IMAGE_SIZE, width=IMAGE_SIZE, regularizer_type=regularizer_type, n_iter=n_iter, n_memory=n_memory).to(device)
    print(f"Model initialized on device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print("Using alpha-continuous theoretical GD block")
    print(f"Regularizer type: {regularizer_type}")
    print(f"Model architecture: {model_arch}")
    if model_arch in {"unrolled_cnn", "learned_gradient_descent", "lgd"}:
        print(f"Optimization iterations: {n_iter}")
        print(f"Memory units: {n_memory}")
    else:
        print("Optimization iterations: non-iterative refiner")
        print(
            "U-Net refiner: input_mode=%s backbone=%s base_channels=%d depth=%d residual_max=%.3g"
            % (
                str(THEORETICAL_CONFIG.get("refiner_input_mode", "u2_stacked")),
                str(THEORETICAL_CONFIG.get("unet_backbone", "plain")),
                int(THEORETICAL_CONFIG.get("unet_base_channels", 32)),
                int(THEORETICAL_CONFIG.get("unet_depth", 4)),
                float(THEORETICAL_CONFIG.get("unet_residual_max", 0.0)),
            )
        )
        print(
            "Refiner cascade: stages=%d share_weights=%s stage_dc=%s stage_dc_iters=%d"
            % (
                int(getattr(model.optimizer, "refiner_stages", THEORETICAL_CONFIG.get("refiner_stages", 1))),
                bool(getattr(model.optimizer, "refiner_share_weights", THEORETICAL_CONFIG.get("refiner_share_weights", True))),
                bool(getattr(model.optimizer, "stage_dc_enabled", THEORETICAL_CONFIG.get("refiner_stage_dc_enabled", False))),
                int(getattr(model.optimizer, "stage_dc_cg_iters", THEORETICAL_CONFIG.get("refiner_stage_dc_cg_iters", 4))),
            )
        )
        print(
            "Detail head: enabled=%s input=%s hidden=%d depth=%d residual_max=%.3g stage_policy=%s share_weights=%s"
            % (
                bool(getattr(model.optimizer, "detail_head_enabled", False)),
                str(getattr(model.optimizer, "detail_head_input_mode", THEORETICAL_CONFIG.get("detail_head_input_mode", "features"))),
                int(THEORETICAL_CONFIG.get("detail_head_hidden_channels", 16)),
                int(THEORETICAL_CONFIG.get("detail_head_depth", 2)),
                float(THEORETICAL_CONFIG.get("detail_head_residual_max", 0.0)),
                str(getattr(model.optimizer, "detail_head_stage_policy", THEORETICAL_CONFIG.get("detail_head_stage_policy", "last"))),
                bool(getattr(model.optimizer, "detail_head_share_weights", THEORETICAL_CONFIG.get("detail_head_share_weights", True))),
            )
        )
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
    print("Alpha-only CT model smoke test successful!")
