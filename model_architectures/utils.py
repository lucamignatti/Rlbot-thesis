import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import math
from typing import Union, Tuple

def load_partial_state_dict(model, state_dict):
    """Load a state dict, skipping missing and mismatched parameters"""
    model_state = model.state_dict()
    filtered_state = {}
    skipped_params = []

    for name, param in state_dict.items():
        # Remove _orig_mod prefix if it exists (from torch.compile)
        if name.startswith('_orig_mod.'):
            name = name[len('_orig_mod.'):]

        if name not in model_state:
            skipped_params.append(name)
            continue

        if param.shape != model_state[name].shape:
            skipped_params.append(f"{name}: size mismatch - expected {model_state[name].shape}, got {param.shape}")
            continue

        filtered_state[name] = param

    # Load the filtered state dict
    model.load_state_dict(filtered_state, strict=False)

    if len(skipped_params) > 0:
        print(f"Warning: Skipped {len(skipped_params)} mismatched parameters:")
        # Only print first 5 mismatched parameters to avoid cluttering output
        for param in skipped_params[:5]:
            print(f"  - {param}")
        if len(skipped_params) > 5:
            print(f"  ... and {len(skipped_params) - 5} more")

    return len(skipped_params)

def print_model_info(model, model_name, print_amp=False, debug=False):
    """Print information about a model including parameter count and device"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = next(model.parameters()).device

    if debug:
        print(f"\n{model_name} Info:")
        print(f"- Device: {device}")
        print(f"- Total parameters: {total_params:,}")
        print(f"- Trainable parameters: {trainable_params:,}")
        if print_amp:
            print(f"- AMP enabled: {True}")
        print(f"- Using torch.compile: {hasattr(model, '_orig_mod')}")
    else:
        # More concise output when not in debug mode
        print(f"{model_name}: {trainable_params:,} params on {device}")

def extract_model_dimensions(state_dict):
    """Extract model dimensions from a state dict"""
    # Find input dimension from first layer
    first_layer = next((key for key in state_dict.keys() if 'embedding.weight' in key or 'embedding_linear.weight' in key), None)
    if first_layer is None:
        raise ValueError("Could not find embedding layer weight in state_dict to determine obs_shape.")
    obs_shape = state_dict[first_layer].shape[1]
    if 'embedding_linear' in first_layer: # Adjust for SimbaV2 input dim
        obs_shape -= 1

    # Find hidden dimension
    hidden_dim = state_dict[first_layer].shape[0]

    # Find output dimension from last layer
    output_layers = [key for key in state_dict.keys() if ('output.weight' in key or 'output_linear.weight' in key)]
    action_shape = None
    if output_layers:
        # Heuristic: find the layer with the largest output dimension if multiple exist
        output_layer_key = max(output_layers, key=lambda k: state_dict[k].shape[0])
        action_shape = state_dict[output_layer_key].shape[0]
        # Adjust for continuous action spaces (mean + log_std)
        if action_shape % 2 == 0:
             potential_action_dim = action_shape // 2
             # This is heuristic, might need refinement based on actual usage
             action_shape = potential_action_dim # Assume it's mean+std

    # Find number of blocks from number of residual layers
    # Count unique block indices
    block_indices = set()
    for key in state_dict.keys():
        if 'blocks.' in key:
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    num_blocks = len(block_indices)

    return obs_shape, hidden_dim, action_shape, num_blocks


def fix_compiled_state_dict(state_dict):
    """Fix state dict keys for compiled models by removing _orig_mod prefix"""
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value
    return fixed_state_dict

class RSNorm(nn.Module):
    """
    Running Statistics Normalization (RSNorm) from the SimBa paper
    https://arxiv.org/abs/2410.09754

    Standardizes input features using running estimates of mean and variance.
    """
    def __init__(self, num_features, momentum=None, eps=1e-5):
        super(RSNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Register buffers for running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Flag for cuda graphs compatibility
        self._cuda_graphs_fixed = False # Keep this flag internal

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, num_features]
        Returns:
            Normalized tensor of same shape
        """
        # Use running stats directly
        mean = self.running_mean
        var = self.running_var

        # Update running statistics during training if not fixed
        if self.training and not self._cuda_graphs_fixed:
            # Calculate batch statistics
            batch_mean = x.mean(dim=0, keepdim=False)
            batch_var = x.var(dim=0, unbiased=False, keepdim=False)

            # Update running statistics using welford's online algorithm
            with torch.no_grad():
                self.num_batches_tracked += 1
                if self.momentum is None:  # Use cumulative moving average
                    momentum = 1.0 / float(self.num_batches_tracked)
                else:  # Use exponential moving average
                    momentum = self.momentum

                # Update running mean and variance
                self.running_mean.copy_(self.running_mean * (1 - momentum) + batch_mean * momentum)
                self.running_var.copy_(self.running_var * (1 - momentum) + batch_var * momentum)

        # --- End Stat Update ---

        # Ensure stats are on the same device as input
        if mean.device != x.device:
            mean = mean.to(x.device)
            var = var.to(x.device)

        # Normalize using appropriate statistics
        return (x - mean) / torch.sqrt(var + self.eps)

    def reset_running_stats(self):
        """Reset running statistics"""
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    # Keep the fix_rsnorm_cuda_graphs function separate for clarity
def fix_rsnorm_cuda_graphs(model):
    """Fix CUDA graphs compilation issues for RSNorm modules

    This function should be called before using torch.compile() on models
    that contain RSNorm modules.

    Args:
        model: PyTorch model that may contain RSNorm layers

    Returns:
        The patched model with more stable compilation behavior
    """
    for module in model.modules():
        if isinstance(module, RSNorm):
            # Mark the module as fixed
            module._cuda_graphs_fixed = True
    return model


class MPSFriendlyDropout(nn.Module):
    """Dropout layer that's compatible with MPS backend by avoiding native_dropout"""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # Generate binary mask using bernoulli distribution
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
        # Scale output during training
        return x * mask / (1 - self.p)

class ResidualFFBlock(nn.Module):
    """Residual feed-forward block with pre-layer normalization"""
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.norm1 = RSNorm(hidden_dim)
        self.ff1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ff2 = nn.Linear(hidden_dim * 4, hidden_dim)
        if torch.backends.mps.is_available():
            self.dropout = MPSFriendlyDropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        # Pre-layer normalization
        norm_x = self.norm1(x)

        # Feed-forward network
        h = self.ff1(norm_x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.ff2(h)
        h = self.dropout(h)

        # Residual connection
        return x + h

# --- SimbaV2 Specific Components ---

# Helper function for L2 normalization
def l2_norm(t, eps=1e-8):
    return t / (torch.norm(t, p=2, dim=-1, keepdim=True) + eps)

# Scaler module from SimbaV2 paper (Appendix B, Listing 1 adaptation)
# Updated to use decoupled init/scale (Section 4.4, Appendix A.2)
class Scaler(nn.Module):
    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        # Store init and scale factors
        self.init_val = init
        self.scale_val = scale
        # Initialize the learnable parameter based on scale
        self.scaler_param = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply decoupled scaling (Eq 21)
        effective_scaler = self.scaler_param * (self.init_val / self.scale_val) if self.scale_val != 0 else self.scaler_param
        return effective_scaler * x # Element-wise scaling

# Linear layer with orthogonal initialization and weight projection
class OrthogonalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # Orthogonal initialization
        nn.init.orthogonal_(self.weight, gain=scale) # Use specified scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Projection happens during optimization, not here
        return F.linear(x, self.weight)

# Learnable Linear Interpolation (LERP) module
# Updated to use decoupled init/scale (Section 4.4, Appendix A.4)
class LERP(nn.Module):
    def __init__(self, dim: int, init: float = 0.5, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.init_val = init
        self.scale_val = scale
        # Initialize learnable parameter based on scale
        self.alpha_param = nn.Parameter(torch.full((dim,), scale))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Apply decoupled scaling for alpha (similar to Scaler's Eq 21)
        effective_alpha = self.alpha_param * (self.init_val / self.scale_val) if self.scale_val != 0 else self.alpha_param
        # Interpolate directly using effective_alpha (matches JAX reference code structure)
        # Reference: residual + alpha * (h_tilde - residual)
        # Equivalent: (1 - alpha) * residual + alpha * h_tilde
        interpolated = (1.0 - effective_alpha) * x1 + effective_alpha * x2
        # Apply final L2 normalization as per Eq 12 and reference code
        return l2_norm(interpolated)
