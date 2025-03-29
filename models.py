import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

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
    first_layer = next(key for key in state_dict.keys() if 'embedding.weight' in key)
    obs_shape = state_dict[first_layer].shape[1]
    
    # Find hidden dimension
    hidden_dim = state_dict[first_layer].shape[0]
    
    # Find output dimension from last layer
    output_layers = [key for key in state_dict.keys() if 'output' in key and 'weight' in key]
    if output_layers:
        action_shape = state_dict[output_layers[0]].shape[0]
    else:
        action_shape = None
    
    # Find number of blocks from number of residual layers
    num_blocks = len([key for key in state_dict.keys() if 'blocks.' in key]) // 4
    
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

class BasicModel(nn.Module):
    """Base model that all policy/value models should inherit from"""
    def __init__(self, obs_shape, action_shape, hidden_dim=512, num_blocks=4,
                 dropout_rate=0.1, device="cpu"):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_dim = hidden_dim
        self.device = device

        # Input norm and embedding
        self.input_norm = RSNorm(obs_shape)
        self.embedding = nn.Linear(obs_shape, hidden_dim)
        
        # Dropout (MPS-friendly if needed)
        if torch.backends.mps.is_available():
            self.dropout = MPSFriendlyDropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)

        # Stack residual blocks
        self.blocks = nn.ModuleList([
            ResidualFFBlock(hidden_dim, dropout_rate)
            for _ in range(num_blocks)
        ])

        # Output head
        self.output = nn.Linear(hidden_dim, action_shape)

    def forward(self, x, return_features=False):
        # Input normalization and embedding
        x = self.input_norm(x)
        features = self.embedding(x)
        features = self.dropout(features)

        # Process through residual blocks
        for block in self.blocks:
            features = block(features)

        # Get output
        output = self.output(features)

        if return_features:
            return output, features
        return output

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

class SimBa(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim=1536, num_blocks=6,
                 dropout_rate=0.1, device="cpu"):
        super(SimBa, self).__init__()
        self.device = device

        # Store input and output shapes for later use.
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_dim = hidden_dim  # Explicitly store hidden_dim

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
            self.blocks.append(ResidualFFBlock(hidden_dim, dropout_rate))

        # Final output layer to map to action space.
        # For continuous action spaces, output both mean and log std.
        if isinstance(action_shape, tuple):
            self.output = nn.Linear(hidden_dim, np.prod(action_shape) * 2)
        else:
            self.output = nn.Linear(hidden_dim, action_shape)

    def forward(self, x, return_features=False):
        # Normalize input using running statistics.
        x = self.rsnorm(x)

        # Initial embedding and dropout.
        features = self.embedding(x)
        features = self.dropout(features)

        # Process through residual blocks.
        for block in self.blocks:
            features = block(features)

        # Get raw output.
        output = self.output(features)

        # Return features if requested (for auxiliary tasks).
        if return_features:
            return output, features
        return output

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

class RSNorm(nn.Module):
    """
    Running Statistics Normalization (RSNorm) from the SimBa paper
    https://arxiv.org/abs/2410.09754
    
    Standardizes input features using running estimates of mean and variance.
    """
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(RSNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Register buffers for running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Flag for cuda graphs compatibility
        self._cuda_graphs_fixed = False
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, num_features]
        Returns:
            Normalized tensor of same shape
        """
        # Don't modify original input
        x = x.clone()

        # Ensure running statistics are on the same device as the input.
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)
            self.num_batches_tracked = self.num_batches_tracked.to(x.device)
            
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
                self.running_mean = self.running_mean * (1 - momentum) + batch_mean * momentum
                self.running_var = self.running_var * (1 - momentum) + batch_var * momentum
                
        # Normalize using running statistics as described in the paper (equation 4)
        # (x_t - μ_t) / sqrt(σ_t^2 + ε)
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
    
    def reset_running_stats(self):
        """Reset running statistics"""
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
