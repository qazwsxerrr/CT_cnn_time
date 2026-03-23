import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import device, THEORETICAL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, IMAGE_SIZE
from radon_transform import build_time_domain_operator


# ============================================================================
# 1. Coefficient mapping (row-major flatten)
# ============================================================================
class CoefficientMapping:
    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], E_plus_shape=(IMAGE_SIZE, IMAGE_SIZE)):
        self.beta = torch.tensor(beta, dtype=torch.long)
        self.beta_norm = torch.norm(self.beta.float(), p=2)
        self.alpha = self.beta.float() / self.beta_norm
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
# 3. Theoretical gradient descent
# ============================================================================
class TheoreticalGradientDescent(nn.Module):
    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], height=IMAGE_SIZE, width=IMAGE_SIZE,
                 regularizer_type='tikhonov', lambda_reg=0.01, operator=None):
        super().__init__()
        # Example 1.1 dense system operator A_{alpha,phi,X}. We intentionally avoid the
        # paper's Toeplitz reduction and solve the ill-posed system via optimization + learning.
        self.operator = operator if operator is not None else build_time_domain_operator(
            beta=beta, height=height, width=width
        )
        self.regularizer_type = regularizer_type
        self.lambda_reg = lambda_reg
        self.step_size = 1e-2
        self.register_buffer('laplace_kernel', torch.tensor(
            [[0.0, -1.0, 0.0],
             [-1.0, 4.0, -1.0],
             [0.0, -1.0, 0.0]]
        ).view(1, 1, 3, 3))

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
        g_obs = g_observed.to(dtype=g_pred.dtype)
        residual = self._compute_weighted_residual(g_pred, g_obs)
        num_angles = int(getattr(self.operator, "num_angles", 1) or 1)

        if (
            return_per_angle
            and num_angles > 1
            and hasattr(self.operator, "split_measurements")
            and hasattr(self.operator, "adjoint_per_angle")
        ):
            residual_pa = self.operator.split_measurements(residual)  # (B,K,M_per_angle)
            gradient_pa = self.operator.adjoint_per_angle(residual_pa)  # (B,K,1,H,W)
            gradient = gradient_pa.mean(dim=1)  # (B,1,H,W)
            return 2.0 * gradient, 2.0 * gradient_pa

        gradient = self.operator.adjoint(residual)
        if num_angles > 1:
            gradient = gradient / float(num_angles)

        if return_per_angle:
            return 2.0 * gradient, (2.0 * gradient).unsqueeze(1)
        return 2.0 * gradient

    def compute_regularization_gradient(self, coeff_matrix):
        if self.regularizer_type == 'dirichlet':
            return self._dirichlet_gradient(coeff_matrix)
        elif self.regularizer_type == 'tikhonov':
            return 2 * coeff_matrix
        elif self.regularizer_type == 'tv':
            return self._tv_gradient(coeff_matrix)
        else:
            return torch.zeros_like(coeff_matrix)

    def _tv_gradient(self, coeff_matrix):
        eps = coeff_matrix.new_tensor(1e-6)
        grad_x, grad_y = self._forward_gradient(coeff_matrix)
        grad_norm = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)
        grad_x_norm = grad_x / grad_norm
        grad_y_norm = grad_y / grad_norm
        div_grad = self._divergence(grad_x_norm, grad_y_norm)
        return div_grad

    def _dirichlet_gradient(self, coeff_matrix):
        padded = F.pad(coeff_matrix, (1, 1, 1, 1), mode='replicate')
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
        updated_coeff = coeff_matrix - self.step_size * total_grad
        return updated_coeff


# ============================================================================
# 4. Learned gradient descent (CNN updates)
# ============================================================================
class LearnedGradientDescent(nn.Module):
    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], height=IMAGE_SIZE, width=IMAGE_SIZE,
                 regularizer_type='tikhonov', n_iter=10, n_memory=5):
        super().__init__()
        self.n_iter = n_iter
        self.n_memory = n_memory
        self.height = height
        self.width = width
        self.operator = build_time_domain_operator(beta=beta, height=height, width=width)
        self.theoretical_gd = TheoreticalGradientDescent(
            beta, height, width, regularizer_type, operator=self.operator
        )
        self.num_angles = int(getattr(self.operator, "num_angles", 1) or 1)
        if not hasattr(self.operator, "split_measurements") or not hasattr(self.operator, "adjoint_per_angle"):
            raise ValueError("Current B1*B1 8-angle pipeline requires per-angle operator support.")
        self.input_channels = (2 + self.num_angles + self.n_memory)
        self.detach_physical_grads = bool(DATA_CONFIG.get("detach_physical_grads", True))
        self.learned_correction_max = float(DATA_CONFIG.get("learned_correction_max", 0.0))
        self.update_max_norm = float(DATA_CONFIG.get("update_max_norm", 0.0))
        self.learned_step_max = float(DATA_CONFIG.get("learned_step_max", 0.0))
        self.learned_reg_lambda_max = float(DATA_CONFIG.get("learned_reg_lambda_max", 0.0))
        self.feature_channels = 64

        # Update network learns the whole correction; physical gradients are inputs, not mandatory updates.
        self.update_network = self._build_update_network(self.input_channels)

        # Shared learned step size across iterations.
        step_min = float(DATA_CONFIG.get("learned_step_min", 1.0e-6))
        self.step_min = step_min
        target_init = max(float(DATA_CONFIG.get("learned_step_init", 1.0e-2)) - step_min, 1e-8)
        raw_init = math.log(math.exp(target_init) - 1.0)
        self.step_size_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # Shared learned regularization weight used to scale the regularizer-gradient channel.
        lambda_min = 1e-5
        self.lambda_min = lambda_min
        target_lambda = max(float(DATA_CONFIG.get("learned_reg_lambda_init", 1.0e-3)) - lambda_min, 1e-8)
        raw_lambda_init = math.log(math.exp(target_lambda) - 1.0)
        self.reg_lambda_raw = nn.Parameter(torch.tensor(raw_lambda_init, dtype=torch.float32))

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

    def _compose_cnn_input(self, coeff_current, g_observed, reg_grad, memory, data_grad_pa=None):
        if data_grad_pa is None:
            _, data_grad_pa = self.theoretical_gd.compute_data_fidelity_gradient(
                coeff_current, g_observed, return_per_angle=True
            )
        grad_channels = data_grad_pa.squeeze(2)
        return torch.cat([coeff_current, grad_channels, reg_grad, memory], dim=1)

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

    def forward(self, coeff_initial, g_observed):
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        batch_size = coeff_initial.shape[0]
        coeff_current = coeff_initial.clone()
        memory = torch.zeros(batch_size, self.n_memory, self.height, self.width,
                             device=coeff_initial.device)
        history = [coeff_current.clone()]

        for _ in range(self.n_iter):
            lambda_i = self.current_reg_lambda()
            reg_grad_base = self.theoretical_gd.compute_regularization_gradient(coeff_current)

            data_grad, data_grad_pa = self.theoretical_gd.compute_data_fidelity_gradient(
                coeff_current, g_observed, return_per_angle=True
            )

            if self.detach_physical_grads:
                data_grad = data_grad.detach()
                reg_grad_base = reg_grad_base.detach()
                if data_grad_pa is not None:
                    data_grad_pa = data_grad_pa.detach()

            reg_grad = reg_grad_base * lambda_i

            cnn_input = self._compose_cnn_input(
                coeff_current, g_observed, reg_grad, memory, data_grad_pa=data_grad_pa
            )

            cnn_output = self.update_network(cnn_input)
            raw_update = cnn_output[:, 0:1, :, :]
            new_memory = cnn_output[:, 1:, :, :]

            step_i = self.current_step_size()
            learned_update = self._cap_correction(raw_update) * step_i
            total_update = self._clip_update_norm(learned_update)

            coeff_current = coeff_current - total_update
            memory = torch.relu(new_memory)
            history.append(coeff_current.clone())

        return coeff_current, history


# ============================================================================
# 5. Full CT network
# ============================================================================
class TheoreticalCTNet(nn.Module):
    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], height=IMAGE_SIZE, width=IMAGE_SIZE,
                 regularizer_type='tikhonov', n_iter=10, n_memory=5):
        super().__init__()
        self.beta = beta
        self.height = height
        self.width = width
        self.optimizer = LearnedGradientDescent(
            beta, height, width, regularizer_type, n_iter, n_memory
        )
        self.mapping = CoefficientMapping(beta, (height, width))

    def forward(self, coeff_initial, g_observed):
        if g_observed.dim() == 3 and g_observed.shape[1] == 1:
            g_observed = g_observed.squeeze(1)
        coeff_final, history = self.optimizer(coeff_initial, g_observed)
        metrics = self._compute_optimization_metrics(
            coeff_initial, coeff_final, g_observed, history
        )
        return coeff_final, history, metrics

    def _compute_optimization_metrics(self, coeff_initial, coeff_final, g_observed, history):
        metrics = {}
        with torch.no_grad():
            g_final = self.optimizer.operator(coeff_final)
            data_fidelity_error = torch.norm(g_final - g_observed, dim=-1).mean()
            metrics['data_fidelity_error'] = data_fidelity_error.item()

            coeff_change = torch.norm(coeff_final - coeff_initial, dim=(2, 3)).mean()
            metrics['coefficient_change'] = coeff_change.item()

            if self.optimizer.theoretical_gd.regularizer_type == 'tikhonov':
                reg_value = torch.norm(coeff_final, dim=(2, 3)) ** 2
                metrics['regularization_value'] = reg_value.mean().item()
            elif self.optimizer.theoretical_gd.regularizer_type == 'dirichlet':
                grad_y = torch.diff(coeff_final, dim=2, prepend=coeff_final[:, :, -1:])
                grad_x = torch.diff(coeff_final, dim=3, prepend=coeff_final[:, :, :, -1:])
                reg_value = 0.5 * (grad_x.pow(2) + grad_y.pow(2)).sum(dim=(2, 3))
                metrics['regularization_value'] = reg_value.mean().item()

            # Cheap diagnostic: magnitude of the coefficient update from init -> final.
            metrics['update_difference'] = coeff_change.item()

        return metrics


# ============================================================================
# 6. Helpers
# ============================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_model():
    beta = THEORETICAL_CONFIG['beta_vector']
    regularizer_type = THEORETICAL_CONFIG['regularizer_type']
    n_iter = THEORETICAL_CONFIG['n_iter']
    n_memory = THEORETICAL_CONFIG['n_memory_units']

    model = TheoreticalCTNet(
        beta=beta,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        regularizer_type=regularizer_type,
        n_iter=n_iter,
        n_memory=n_memory
    ).to(device)

    print(f"Model initialized on device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Using theoretical GD block")
    print(f"Regularizer type: {regularizer_type}")
    print(f"Optimization iterations: {n_iter}")
    print(f"Memory units: {n_memory}")

    return model


if __name__ == "__main__":
    model = initialize_model()
    batch_size = 2
    beta = THEORETICAL_CONFIG['beta_vector']
    mapping = CoefficientMapping(beta)
    N = mapping.N
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
    print("Simplified LGD model test successful!")
