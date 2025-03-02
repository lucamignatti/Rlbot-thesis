import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

class BasicModel(nn.Module):
    def __init__(self, input_size = 784, output_size = 10, hidden_size = 64, dropout_rate = 0.5):
        super(BasicModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size),
        )

    def forward(self, x):
        return self.network(x)

class MPSFriendlyDropout(nn.Module):
    """Dropout layer that's compatible with MPS backend by avoiding native_dropout"""
    def __init__(self, p=0.5):
        super(MPSFriendlyDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # Implementation using binary mask
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p)) / (1 - self.p)
        return x * mask

def fix_compiled_state_dict(state_dict):
    """
    Fixes state dictionaries saved from compiled models by removing the '_orig_mod.' prefix
    from keys, allowing them to be loaded into non-compiled models.
    """
    if not any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        return state_dict  # No fix needed

    fixed_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            fixed_dict[new_key] = value
        else:
            fixed_dict[key] = value
    return fixed_dict


class SimBa(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim=512, num_blocks=2,
                 dropout_rate=0.1, device="cpu"):
        super(SimBa, self).__init__()
        self.device = device

        # save IO shapes
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # Running Statistics Normalization
        self.rsnorm = RSNorm(obs_shape)

        # Initial embedding
        self.embedding = nn.Linear(obs_shape, hidden_dim)
        if torch.backends.mps.is_available():
            self.dropout = MPSFriendlyDropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)

        # Residual feedforward blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResidualFFBlock(hidden_dim, dropout_rate, device=self.device))

        # Post-layer normalization
        self.post_ln = nn.LayerNorm(hidden_dim)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, action_shape)

        self.to(self.device)

    def forward(self, x):
        # For CUDA graphs, explicitly mark step before we begin
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Convert input to correct device and format, but preserve grad history
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)

        # Just clone, don't detach (to preserve gradients)
        x = x.clone()

        # Mark another step after input preparation
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Normalize input with running statistics
        x_normalized = self.rsnorm(x)

        # Mark step after normalization which modifies buffers
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Initial embedding
        h = self.embedding(x_normalized)
        h = self.dropout(h)

        # Apply residual blocks with explicit steps between blocks
        for i, block in enumerate(self.blocks):
            # Mark step before each block
            if "cuda" in str(self.device):
                torch.compiler.cudagraph_mark_step_begin()
            h = block(h)

        # Mark step before normalization
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Apply post-layer normalization
        h = self.post_ln(h)

        # Mark step before final layer
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Output layer
        output = self.output_layer(h)

        # Mark step after the model completes
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Clone for CUDA graph safety but don't detach (preserve gradients)
        return output.clone()

    def to(self, device=None, dtype=None, non_blocking=False):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.device = device
        return self


class ResidualFFBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, device=None):
        super(ResidualFFBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.device = device

        # Pre-layer normalization
        self.ln = nn.LayerNorm(hidden_dim)

        # MLP with inverted bottleneck (4x expansion)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)

        # Use device-appropriate dropout
        if torch.backends.mps.is_available():
            self.dropout1 = MPSFriendlyDropout(dropout_rate)
            self.dropout2 = MPSFriendlyDropout(dropout_rate)
        else:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x):
        # Use clone() but not detach() to preserve gradient history
        x = x.clone()

        # Mark CUDA graph step if on CUDA device
        if self.device and "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Pre-layer normalization
        norm_x = self.ln(x)

        # Mark step after normalization
        if self.device and "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Apply MLP with dropout
        h = self.linear1(norm_x)
        h = F.relu(h)
        h = self.dropout1(h)

        # Mark step between linear layers
        if self.device and "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        h = self.linear2(h)
        h = self.dropout2(h)

        # Mark step before residual connection
        if self.device and "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Combine with residual connection
        output = x + h

        # Mark step after computation completes
        if self.device and "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Clone but preserve gradients
        return output.clone()

class RSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(RSNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Initialize running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        device = x.device

        # Use clone without detach to preserve gradients
        x = x.clone()

        # Move buffers to the correct device if needed
        if self.running_mean.device != device:
            self.running_mean = self.running_mean.to(device)
            self.running_var = self.running_var.to(device)
            self.num_batches_tracked = self.num_batches_tracked.to(device)

        # Mark CUDA graph step if on CUDA device
        if "cuda" in str(device):
            torch.compiler.cudagraph_mark_step_begin()

        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_var = x.var(dim=0, unbiased=False, keepdim=True)

            # Mark step before modifying buffers
            if "cuda" in str(device):
                torch.compiler.cudagraph_mark_step_begin()

            # Update batch counter
            cloned_num_batches = self.num_batches_tracked.clone()
            cloned_num_batches += 1

            # Mark step before updating buffer
            if "cuda" in str(device):
                torch.compiler.cudagraph_mark_step_begin()

            self.num_batches_tracked.copy_(cloned_num_batches)

            # Calculate momentum
            update_factor = self.momentum
            if self.num_batches_tracked == 1:
                update_factor = 1.0

            # Mark step before modifying running stats
            if "cuda" in str(device):
                torch.compiler.cudagraph_mark_step_begin()

            # Update running statistics with proper cloning
            new_running_mean = (1 - update_factor) * self.running_mean + update_factor * batch_mean.squeeze()
            new_running_var = (1 - update_factor) * self.running_var + update_factor * batch_var.squeeze()

            # Mark step before updating buffers
            if "cuda" in str(device):
                torch.compiler.cudagraph_mark_step_begin()

            # Copy updated values to buffers
            self.running_mean.copy_(new_running_mean)
            self.running_var.copy_(new_running_var)

            # Mark step after buffer updates
            if "cuda" in str(device):
                torch.compiler.cudagraph_mark_step_begin()

            # Use batch statistics for normalization
            normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Use running statistics for normalization
            normalized = (x - self.running_mean.unsqueeze(0)) / torch.sqrt(self.running_var.unsqueeze(0) + self.eps)

        # Mark step after normalization
        if "cuda" in str(device):
            torch.compiler.cudagraph_mark_step_begin()

        # Clone but don't detach to preserve gradient flow
        return normalized.clone()
