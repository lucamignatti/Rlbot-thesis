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
        # Add cudagraph_mark_step_begin call to prevent CUDA graph overwriting issues
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        elif x.device != self.device:
            x = x.to(self.device)

        # Normalize input with running statistics
        x = self.rsnorm(x)

        # Initial embedding
        x = self.embedding(x)
        x = self.dropout(x)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Apply post-layer normalization
        x = self.post_ln(x)

        # Output layer
        x = self.output_layer(x)

        # Clone the output tensor to avoid CUDA graph overwriting
        x = x.clone()

        return x

    def to(self, device=None, dtype=None, non_blocking=False):
        # Call parent's to() method with all parameters
        super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        # Update instance device attribute
        self.device = device
        return self


class ResidualFFBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, device=None):
        super(ResidualFFBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

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
        # Pre-layer normalization
        norm_x = self.ln(x)

        # Apply MLP with dropout
        h = self.linear1(norm_x)
        h = F.relu(h)
        h = self.dropout1(h)
        h = self.linear2(h)
        h = self.dropout2(h)

        # Residual connection with clone to avoid CUDA graph issues
        return (x + h).clone()

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

        if self.running_mean.device != device:
            self.running_mean = self.running_mean.to(device)
            self.running_var = self.running_var.to(device)
            self.num_batches_tracked = self.num_batches_tracked.to(device)

        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Clone the tensor to avoid CUDA graph issues
            cloned_num_batches = self.num_batches_tracked.clone()
            cloned_num_batches += 1
            self.num_batches_tracked.copy_(cloned_num_batches)

            update_factor = self.momentum
            if self.num_batches_tracked == 1:
                update_factor = 1.0

            # Use clones to avoid CUDA graph overwrite issues
            new_running_mean = (1 - update_factor) * self.running_mean.clone() + update_factor * batch_mean
            new_running_var = (1 - update_factor) * self.running_var.clone() + update_factor * batch_var

            self.running_mean.copy_(new_running_mean)
            self.running_var.copy_(new_running_var)

            # Clone to avoid overwriting
            normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            return normalized.clone()
        else:
            normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            return normalized.clone()
