import torch
import torch.nn as nn
import numpy as np
import math
from typing import Union, Tuple, Optional
from .utils import RSNorm, Scaler, OrthogonalLinear, l2_norm, LERP

# SimbaV2 Model Implementation
class SimbaV2(nn.Module):
    def __init__(self, obs_shape: int, action_shape: Union[int, Tuple[int]],
                 hidden_dim: int = 512, num_blocks: int = 4,
                 shift_constant: float = 3.0, device: str = "cpu",
                 # --- Add these parameters ---
                 is_critic: bool = False,
                 num_atoms: Optional[int] = None):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape # Still needed for actor role
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks # Store num_blocks
        self.shift_constant = shift_constant
        self.device = device
        # Store critic flag and num_atoms
        self.is_critic = is_critic
        self.num_atoms = num_atoms

        # Input Embedding (Section 4.1)
        self.input_norm = RSNorm(obs_shape)
        # Shift + L2 Norm (Eq 9) - dimension increases by 1
        self.input_embed_dim = obs_shape + 1
        # Linear + Scaler (Eq 10)
        self.embedding_linear = OrthogonalLinear(self.input_embed_dim, hidden_dim)
        # Use decoupled Scaler initialization (Appendix A.2)
        embed_scaler_init_scale = math.sqrt(2.0 / hidden_dim) if hidden_dim > 0 else 1.0
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
        output_mlp_scaler_init_scale = math.sqrt(2.0 / hidden_dim) if hidden_dim > 0 else 1.0
        self.output_mlp_scaler = Scaler(hidden_dim, init=output_mlp_scaler_init_scale, scale=output_mlp_scaler_init_scale)

        # Final linear layer for policy/value output
        # --- Modify output_dim calculation ---
        if self.is_critic:
            # For distributional critic, output num_atoms logits
            if self.num_atoms is not None:
                if self.num_atoms <= 0:
                    raise ValueError("num_atoms must be positive if provided")
                output_dim = self.num_atoms
            else:
                # For standard value critic, output 1 value
                output_dim = 1
            # Add the output scaler for the critic head as well (similar to paper's Eq 14 structure)
            # This scaler wasn't explicitly in your original code for the output layer, but fits the pattern
            output_final_scaler_init_scale = math.sqrt(2.0 / hidden_dim) if hidden_dim > 0 else 1.0
            self.output_final_scaler = Scaler(hidden_dim, init=output_final_scaler_init_scale, scale=output_final_scaler_init_scale)

        else: # Actor output
            if isinstance(action_shape, tuple):
                # Continuous actions: output mean and std dev (or flattened parameters)
                # Note: SAC often outputs mean/log_std directly. PPO might need logits if using Categorical/Normal later.
                # Assuming mean and log_std for now.
                output_dim = np.prod(action_shape) * 2
            else:
                # Discrete actions: output logits
                output_dim = action_shape
            # Actor doesn't necessarily need the final scaler from Eq 14's structure, depends on how distribution is parameterized.
            # We'll omit it for the actor for now, matching your original code more closely.
            self.output_final_scaler = None

        self.output_linear = OrthogonalLinear(hidden_dim, output_dim)
        # --- End modification ---

    def forward(self, x: torch.Tensor, return_features=False):
        # 1. Input Embedding
        
        # Ensure input has correct dimensionality - add batch dimension if missing
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        x_norm = self.input_norm(x) # RSNorm (Eq 4)

        # Shift + L2 Norm (Eq 9)
        shift = torch.full((x_norm.size(0), 1), self.shift_constant, device=x.device)
        x_shifted = torch.cat([x_norm, shift], dim=-1)
        x_tilde = l2_norm(x_shifted)

        # Linear + Scaler (Eq 10)
        h0_linear = self.embedding_linear(x_tilde)
        h0_scaled = self.embedding_scaler(h0_linear)
        h = l2_norm(h0_scaled)

        # 2. Feature Encoding
        features = h
        for block in self.blocks:
            h = block(h)
            features = h

        # 3. Output Prediction
        h_out = self.output_mlp_linear(features)
        h_out = self.output_mlp_scaler(h_out)

        # --- Apply final scaler only if critic ---
        if self.is_critic and self.output_final_scaler is not None:
            h_out = self.output_final_scaler(h_out)
        # --- End modification ---

        # Final output layer
        output = self.output_linear(h_out) # Shape: [batch, output_dim]

        if return_features:
            return output, features
        return output

# SimbaV2 Residual Block (No changes needed here)
class SimbaV2Block(nn.Module):
    def __init__(self, hidden_dim: int, expansion_factor: int = 4, num_total_blocks: int = 4): # Add num_total_blocks
        super().__init__()
        bottleneck_dim = hidden_dim * expansion_factor

        # First part: MLP + L2 Norm
        self.linear1 = OrthogonalLinear(hidden_dim, bottleneck_dim)
        mlp_scaler_init_scale = math.sqrt(2.0 / bottleneck_dim) if bottleneck_dim > 0 else 1.0
        self.scaler1 = Scaler(bottleneck_dim, init=mlp_scaler_init_scale, scale=mlp_scaler_init_scale)
        self.activation = nn.ReLU()
        self.linear2 = OrthogonalLinear(bottleneck_dim, hidden_dim)

        # Second part: LERP + L2 Norm
        lerp_init = 1.0 / (num_total_blocks + 1) if num_total_blocks > -1 else 0.5
        lerp_scale = 1.0 / math.sqrt(hidden_dim) if hidden_dim > 0 else 1.0
        self.lerp = LERP(hidden_dim, init=lerp_init, scale=lerp_scale)

    def forward(self, h_in: torch.Tensor) -> torch.Tensor:
        # MLP + L2 Norm part (Eq 11)
        h_mlp = self.linear1(h_in)
        h_mlp_scaled = self.scaler1(h_mlp)
        h_mlp_activated = self.activation(h_mlp_scaled)
        h_mlp_out = self.linear2(h_mlp_activated)
        h_tilde = l2_norm(h_mlp_out)

        # LERP + L2 Norm part (Eq 12)
        h_out = self.lerp(h_in, h_tilde) # LERP handles the final L2 norm

        return h_out
