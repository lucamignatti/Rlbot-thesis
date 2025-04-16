import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple
from .utils import RSNorm, MPSFriendlyDropout, ResidualFFBlock

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
