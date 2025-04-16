import torch
import torch.nn as nn
from .utils import RSNorm, MPSFriendlyDropout, ResidualFFBlock

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
