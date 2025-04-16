import torch
import torch.nn as nn
import numpy as np
import math
from typing import Union, Tuple
from .utils import RSNorm, Scaler, OrthogonalLinear, SimbaV2Block, l2_norm

# SimbaV2 Model Implementation
class SimbaV2(nn.Module):
    def __init__(self, obs_shape: int, action_shape: Union[int, Tuple[int]],
                 hidden_dim: int = 512, num_blocks: int = 4,
                 shift_constant: float = 3.0, device: str = "cpu"):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_dim = hidden_dim
        self.shift_constant = shift_constant
        self.device = device

        # Input Embedding (Section 4.1)
        self.input_norm = RSNorm(obs_shape)
        # Shift + L2 Norm (Eq 9) - dimension increases by 1
        self.input_embed_dim = obs_shape + 1
        # Linear + Scaler (Eq 10)
        self.embedding_linear = OrthogonalLinear(self.input_embed_dim, hidden_dim)
        self.embedding_scaler = Scaler(hidden_dim)

        # Feature Encoding (Section 4.2)
        self.blocks = nn.ModuleList([
            SimbaV2Block(hidden_dim) for _ in range(num_blocks)
        ])

        # Output Prediction (Section 4.3 - simplified for standard actor/critic)
        # The paper uses a distributional critic (Eq 13-16).
        # We'll use a standard linear output for now.
        # For continuous actions, output mean and std dev
        if isinstance(action_shape, tuple):
             output_dim = np.prod(action_shape) * 2
        else:
             output_dim = action_shape
        self.output_linear = OrthogonalLinear(hidden_dim, output_dim)
        # Note: The paper uses another MLP block for output (Eq 14), simplified here.

    def forward(self, x: torch.Tensor, return_features=False):
        # 1. Input Embedding
        x_norm = self.input_norm(x) # RSNorm (Eq 4)

        # Shift + L2 Norm (Eq 9)
        # Create shift tensor on the correct device
        shift = torch.full((x_norm.size(0), 1), self.shift_constant, device=x.device)
        x_shifted = torch.cat([x_norm, shift], dim=-1)
        x_tilde = l2_norm(x_shifted)

        # Linear + Scaler (Eq 10)
        h0_linear = self.embedding_linear(x_tilde)
        h0_scaled = self.embedding_scaler(h0_linear)
        # Eq 10 includes a final l2_norm after scaling
        h = l2_norm(h0_scaled)

        # 2. Feature Encoding
        for block in self.blocks:
            h = block(h)
        features = h # Use the output of the last block as features

        # 3. Output Prediction
        output = self.output_linear(features)

        if return_features:
            return output, features
        return output
