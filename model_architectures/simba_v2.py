import torch
import torch.nn as nn
import numpy as np
import math
from typing import Union, Tuple
# Add LERP to the import list
from .utils import RSNorm, Scaler, OrthogonalLinear, l2_norm, LERP

# SimbaV2 Model Implementation
class SimbaV2(nn.Module):
    def __init__(self, obs_shape: int, action_shape: Union[int, Tuple[int]],
                 hidden_dim: int = 512, num_blocks: int = 4,
                 shift_constant: float = 3.0, device: str = "cpu"):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks # Store num_blocks
        self.shift_constant = shift_constant
        self.device = device

        # Input Embedding (Section 4.1)
        self.input_norm = RSNorm(obs_shape)
        # Shift + L2 Norm (Eq 9) - dimension increases by 1
        self.input_embed_dim = obs_shape + 1
        # Linear + Scaler (Eq 10)
        self.embedding_linear = OrthogonalLinear(self.input_embed_dim, hidden_dim)
        # Use decoupled Scaler initialization (Appendix A.2)
        embed_scaler_init_scale = math.sqrt(2.0 / hidden_dim)
        self.embedding_scaler = Scaler(hidden_dim, init=embed_scaler_init_scale, scale=embed_scaler_init_scale)

        # Feature Encoding (Section 4.2)
        self.blocks = nn.ModuleList([
            # Pass num_blocks to SimbaV2Block for LERP init
            SimbaV2Block(hidden_dim, num_total_blocks=num_blocks) for _ in range(num_blocks)
        ])

        # Output Prediction (Section 4.3)
        # Paper Eq 14 describes an MLP block before final output.
        # Implementing a similar structure (Linear -> Scaler -> ReLU) before the final layer.
        self.output_mlp_linear = OrthogonalLinear(hidden_dim, hidden_dim)
        # Use decoupled Scaler initialization (Appendix A.2 discussion)
        output_scaler_init_scale = math.sqrt(2.0 / hidden_dim)
        self.output_mlp_scaler = Scaler(hidden_dim, init=output_scaler_init_scale, scale=output_scaler_init_scale)
        self.output_mlp_relu = nn.ReLU()

        # Final linear layer for policy/value output
        if isinstance(action_shape, tuple):
            # Continuous actions: output mean and std dev (or flattened parameters)
            output_dim = np.prod(action_shape) * 2 # Assuming mean and std dev
        else:
            # Discrete actions: output logits or value distribution atoms
            output_dim = action_shape
        self.output_linear = OrthogonalLinear(hidden_dim, output_dim)

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
        features = h # Initialize features with initial embedding
        for block in self.blocks:
            h = block(h)
            features = h # Update features after each block

        # 3. Output Prediction
        # Pass features through the intermediate MLP block (inspired by Eq 14)
        h_out = self.output_mlp_linear(features)
        h_out = self.output_mlp_scaler(h_out)
        h_out = self.output_mlp_relu(h_out)
        # Note: Paper Eq 14 uses another Linear+Scaler after MLP, simplified here.
        # The final l2_norm projection from the paper's MLP (Eq 11) is also omitted here
        # as the final output usually doesn't need to be on the hypersphere.

        # Final output layer
        output = self.output_linear(h_out)

        if return_features:
            return output, features
        return output

# SimbaV2 Residual Block
class SimbaV2Block(nn.Module):
    def __init__(self, hidden_dim: int, expansion_factor: int = 4, num_total_blocks: int = 4): # Add num_total_blocks
        super().__init__()
        bottleneck_dim = hidden_dim * expansion_factor

        # First part: MLP + L2 Norm
        self.linear1 = OrthogonalLinear(hidden_dim, bottleneck_dim)
        # Use decoupled Scaler initialization (Appendix A.2, adapted for bottleneck dim)
        # Note: Paper init s_init=s_scale=sqrt(2/d_h). Here d_h corresponds to bottleneck_dim for the scaler.
        mlp_scaler_init_scale = math.sqrt(2.0 / bottleneck_dim) if bottleneck_dim > 0 else 1.0 # Avoid division by zero
        self.scaler1 = Scaler(bottleneck_dim, init=mlp_scaler_init_scale, scale=mlp_scaler_init_scale)
        self.activation = nn.ReLU() # Paper uses ReLU here (Eq 11)
        self.linear2 = OrthogonalLinear(bottleneck_dim, hidden_dim)

        # Second part: LERP + L2 Norm
        # Use decoupled LERP initialization based on total blocks (Appendix C / A.4)
        lerp_init = 1.0 / (num_total_blocks + 1) if num_total_blocks > -1 else 0.5 # Avoid div by zero if blocks= -1
        lerp_scale = 1.0 / math.sqrt(hidden_dim) if hidden_dim > 0 else 1.0 # Avoid sqrt(0)
        self.lerp = LERP(hidden_dim, init=lerp_init, scale=lerp_scale)

    def forward(self, h_in: torch.Tensor) -> torch.Tensor:
        # MLP + L2 Norm part (Eq 11)
        h_mlp = self.linear1(h_in)
        h_mlp_scaled = self.scaler1(h_mlp)
        h_mlp_activated = self.activation(h_mlp_scaled)
        h_mlp_out = self.linear2(h_mlp_activated)
        h_tilde = l2_norm(h_mlp_out) # Project MLP output

        # LERP + L2 Norm part (Eq 12)
        h_out = self.lerp(h_in, h_tilde) # LERP handles the final L2 norm

        return h_out
