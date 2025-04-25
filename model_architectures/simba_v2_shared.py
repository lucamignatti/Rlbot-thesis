import torch
import torch.nn as nn
import numpy as np
import math
from typing import Union, Tuple, Dict, Optional
from .utils import RSNorm, Scaler, OrthogonalLinear, l2_norm
from .simba_v2 import SimbaV2Block

# SimbaV2 Shared Body Model Implementation
class SimbaV2Shared(nn.Module):
    def __init__(self, obs_shape: int, action_shape: Union[int, Tuple[int]],
                 hidden_dim: int = 512, num_blocks: int = 4,
                 shift_constant: float = 3.0, device: str = "cpu"):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_dim = hidden_dim
        self.shift_constant = shift_constant
        self.device = device

        # --- Shared Body ---
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
        # --- End Shared Body ---

        # --- Actor Head ---
        # For continuous actions, output mean and std dev
        if isinstance(action_shape, tuple):
             actor_output_dim = np.prod(action_shape) * 2
        else:
             actor_output_dim = action_shape
        self.actor_head = OrthogonalLinear(hidden_dim, actor_output_dim)
        # --- End Actor Head ---

        # --- Critic Head ---
        critic_output_dim = 1 # Critic always outputs a single value
        self.critic_head = OrthogonalLinear(hidden_dim, critic_output_dim)
        # --- End Critic Head ---

    def forward(self, x: torch.Tensor,
                return_features: bool = False,
                return_actor: bool = True,
                return_critic: bool = True) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through the shared model.

        Args:
            x: Input tensor.
            return_features: Whether to return the features from the shared body.
            return_actor: Whether to compute and return the actor head output.
            return_critic: Whether to compute and return the critic head output.

        Returns:
            A dictionary containing the requested outputs ('actor_out', 'critic_out', 'features').
            Keys will be absent if the corresponding output was not requested.
        """
        # 1. Input Embedding (Shared)
        x_norm = self.input_norm(x) # RSNorm (Eq 4)
        shift = torch.full((x_norm.size(0), 1), self.shift_constant, device=x.device)
        x_shifted = torch.cat([x_norm, shift], dim=-1)
        x_tilde = l2_norm(x_shifted) # Eq 9
        h0_linear = self.embedding_linear(x_tilde)
        h0_scaled = self.embedding_scaler(h0_linear)
        h = l2_norm(h0_scaled) # Eq 10

        # 2. Feature Encoding (Shared)
        for block in self.blocks:
            h = block(h)
        features = h # Output of the last block

        # 3. Output Heads
        results = {}
        if return_actor:
            results['actor_out'] = self.actor_head(features)
        if return_critic:
            results['critic_out'] = self.critic_head(features)
        if return_features:
            results['features'] = features

        return results
