import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

def load_partial_state_dict(model, state_dict):
    """
    Loads parts of the state dictionary that match the model's
    parameter names, skipping layers that don't match.
    """
    # Fix compiled state if needed
    state_dict = fix_compiled_state_dict(state_dict)

    model_dict = model.state_dict()

    # Filter out mismatched keys
    filtered_dict = {}
    mismatched = []
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                mismatched.append(f"{k}: saved {v.shape}, model {model_dict[k].shape}")
        else:
            mismatched.append(f"{k}: not in model")

    if mismatched:
        print(f"Warning: Skipped {len(mismatched)} mismatched parameters:")
        for msg in mismatched[:5]:  # Print first few mismatches
            print(f"  - {msg}")
        if len(mismatched) > 5:
            print(f"  ... and {len(mismatched) - 5} more")

    # Update the model with the filtered state dict
    model.load_state_dict(filtered_dict, strict=False)
    return len(mismatched)

def print_model_info(model, model_name, print_amp=False):
    """
    Prints model information in a clean, multi-line format
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Get device
    device = next(model.parameters()).device

    # Print in a consistent multi-line format
    print(f"\n{model_name}:")
    print(f"Parameters: {total_params:,}")

    if hasattr(model, 'obs_shape'):
        print(f"Obs Shape: {model.obs_shape}")

    if hasattr(model, 'action_shape'):
        print(f"Act Shape: {model.action_shape}")

    print(f"Device: {device}")

    if hasattr(model, '_orig_mod'):
        print("Status: Compiled")
    else:
        print("Status: Not compiled")

    print("AMP: Enabled" if print_amp else "AMP: Disabled")

def extract_model_dimensions(state_dict):
    """
    Extracts model dimensions from a saved state dictionary.

    Returns:
        tuple: (observation_shape, hidden_dim, action_shape, num_blocks)
    """
    # Remove _orig_mod prefix if present for easier access to keys
    fixed_dict = fix_compiled_state_dict(state_dict)

    # First, find embedding layer to get input shape and hidden dimension
    hidden_dim = None
    obs_shape = None
    action_shape = None
    num_blocks = 0

    # Find key patterns to extract dimensions
    for key in fixed_dict.keys():
        if 'embedding.weight' in key:
            if hidden_dim is None:
                hidden_dim = fixed_dict[key].shape[0]  # First dimension is hidden_dim
            if obs_shape is None:
                obs_shape = fixed_dict[key].shape[1]   # Second dimension is input shape

        if 'output_layer.weight' in key:
            if action_shape is None:
                action_shape = fixed_dict[key].shape[0]  # First dimension is output shape

        # Count blocks by looking for block-specific parameters
        if 'blocks.' in key:
            parts = key.split('.')
            if len(parts) > 2 and parts[0] == 'blocks' and parts[1].isdigit():
                block_num = int(parts[1])
                num_blocks = max(num_blocks, block_num + 1)  # +1 because 0-indexed

    return (obs_shape, hidden_dim, action_shape, num_blocks)

def fix_compiled_state_dict(state_dict):
    """
    Fixes state dictionaries saved from compiled models.

    This handles both simple cases (prefix '_orig_mod.') and
    cases where there are multiple levels of compilation wrappers.
    """
    # If no prefixes to fix, return as is
    if not any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        # Check if the keys match what a compiled model would expect
        if any('.' in k for k in state_dict.keys()):
            # This is likely a non-compiled state dict being loaded into a compiled model
            # We need to add the '_orig_mod.' prefix
            fixed_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}
            return fixed_dict
        return state_dict  # No fix needed

    # Remove one level of '_orig_mod.' prefix
    fixed_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            fixed_dict[new_key] = value
        else:
            fixed_dict[key] = value

    # Recursively fix in case of multiple levels
    if any(k.startswith('_orig_mod.') for k in fixed_dict.keys()):
        return fix_compiled_state_dict(fixed_dict)

    return fixed_dict

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

        # Create a mask with the same shape as input x, filled with (1-p).
        # Then, randomly set some elements of the mask to 0 based on Bernoulli distribution.
        # Finally, scale the mask by 1/(1-p) to preserve the expected value of input elements.
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p)) / (1 - self.p)
        return x * mask

class SimBa(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim=1536, num_blocks=6,
                 dropout_rate=0.1, device="cpu"):
        super(SimBa, self).__init__()
        self.device = device

        # Store input and output shapes for later use.
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # Use Running Statistics Normalization for input.
        self.rsnorm = RSNorm(obs_shape)

        # Initial embedding to project input to a higher-dimensional space.
        self.embedding = nn.Linear(obs_shape, hidden_dim)
        # Use a dropout layer that works on the MPS backend, if available.
        if torch.backends.mps.is_available():
            self.dropout = MPSFriendlyDropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)

        # Stack residual feedforward blocks for the main part of the network.
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResidualFFBlock(hidden_dim, dropout_rate, device=self.device))

        # Layer normalization after the residual blocks.
        self.post_ln = nn.LayerNorm(hidden_dim)

        # Output layer to project back to the action space.
        self.output_layer = nn.Linear(hidden_dim, action_shape)

        self.to(self.device)

    def forward(self, x):
        # For CUDA graphs, explicitly mark step before we begin.
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Ensure input is a tensor on the correct device, preserving gradient history.
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)

        # Clone to avoid in-place modification of the original input.
        x = x.clone()

        # Mark another step after input preparation.
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Normalize the input using running statistics.
        x_normalized = self.rsnorm(x)

        # Mark CUDA graph step after normalization.
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Initial embedding and dropout.
        h = self.embedding(x_normalized)
        if self.training:
            h = self.dropout(h)

        # Apply the residual blocks.
        for i, block in enumerate(self.blocks):
            # Mark CUDA graph step before each block.
            if "cuda" in str(self.device):
                torch.compiler.cudagraph_mark_step_begin()
            h = block(h)

        # Mark CUDA graph step before normalization.
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Apply layer normalization after the residual blocks.
        h = self.post_ln(h)

        # Mark CUDA graph step before the final layer.
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Get the final output.
        output = self.output_layer(h)

        # Mark CUDA graph step after the model completes.
        if "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Clone the output for CUDA graph safety, preserving gradients.
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

        # Layer normalization before the MLP.
        self.ln = nn.LayerNorm(hidden_dim)

        # MLP with an inverted bottleneck structure (4x expansion).
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)

        # Choose dropout based on device to handle MPS.
        if torch.backends.mps.is_available():
            self.dropout1 = MPSFriendlyDropout(dropout_rate)
            self.dropout2 = MPSFriendlyDropout(dropout_rate)
        else:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x):
        # Clone the input to avoid in-place modifications, but preserve gradients.
        x = x.clone()

        # Mark step for CUDA graph, if applicable.
        if self.device and "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Apply layer normalization.
        norm_x = self.ln(x)

        # Mark step after normalization.
        if self.device and "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # First linear layer, ReLU activation, and dropout.
        h = self.linear1(norm_x)
        h = F.relu(h)
        if self.training:
            h = self.dropout1(h)

        # Mark step between linear layers.
        if self.device and "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Second linear layer and dropout.
        h = self.linear2(h)
        if self.training:
            h = self.dropout2(h)

        # Mark step before residual connection.
        if self.device and "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Add the residual connection.
        output = x + h

        # Mark step after computation completes
        if self.device and "cuda" in str(self.device):
            torch.compiler.cudagraph_mark_step_begin()

        # Clone the output to ensure no in-place modification, while keeping gradient flow.
        return output.clone()

class RSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(RSNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps  # Small constant to avoid division by zero.
        self.momentum = momentum  # Momentum for updating running statistics.

        # Initialize running statistics (mean and variance) as buffers.
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Disable compilation on MPS device - it's causing issues
        if "mps" in str(x.device):
            torch._dynamo.config.suppress_errors = True

        # Clone to avoid in-place modifications.
        x = x.clone()

        # Ensure running statistics are on the same device as the input.
        device = x.device
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)
        self.num_batches_tracked = self.num_batches_tracked.to(device)

        # Get batch size and feature dimensions
        if x.dim() == 1:  # Case for single vector input
            batch_size = 1
            feature_dim = x.size(0)
            x = x.view(1, -1)  # Add batch dimension for consistency
        else:  # Case for batched input
            batch_size = x.size(0)
            feature_dim = x.size(1)

        # Ensure feature dimensions match
        if feature_dim != self.num_features:
            # If feature dimensions don't match, try to handle it gracefully
            if self.training:
                # During training, resize running stats to match input
                self.num_features = feature_dim
                self.running_mean = torch.zeros(feature_dim, device=device)
                self.running_var = torch.ones(feature_dim, device=device)
            else:
                # During evaluation, this is an error
                raise ValueError(f"Expected input features {self.num_features}, got {feature_dim}")

        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Update tracker count
            with torch.no_grad():
                self.num_batches_tracked += 1

            # Update factor depends on whether this is the first batch
            update_factor = self.momentum
            if self.num_batches_tracked == 1:
                update_factor = 1.0

            # Update running statistics with safe operations
            with torch.no_grad():
                self.running_mean = (1 - update_factor) * self.running_mean + update_factor * batch_mean.detach()
                self.running_var = (1 - update_factor) * self.running_var + update_factor * batch_var.detach()

            # Normalize using batch statistics
            normalized = (x - batch_mean.unsqueeze(0)) / torch.sqrt(batch_var.unsqueeze(0) + self.eps)
        else:
            # Normalize using running statistics
            normalized = (x - self.running_mean.unsqueeze(0)) / torch.sqrt(self.running_var.unsqueeze(0) + self.eps)

        # Return a clone to prevent in-place operations but still maintain gradient information.
        return normalized
