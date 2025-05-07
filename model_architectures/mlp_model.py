import torch
import torch.nn as nn

class MLPModel(nn.Module):
    """
    Basic Multi-Layer Perceptron (MLP) model matching BasicModel API and style.
    """
    def __init__(self, obs_shape, action_shape, hidden_dim=1024, num_blocks=4, dropout_rate=0.1, device="cpu"):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_dim = hidden_dim
        self.device = device

        # Input embedding
        self.embedding = nn.Linear(obs_shape, hidden_dim)

        # Dropout (MPS-friendly if needed)
        if torch.backends.mps.is_available():
            self.dropout = MLPModel._mps_friendly_dropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)

        # Hidden layers (fully connected, no residuals)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])

        # Output head
        self.output = nn.Linear(hidden_dim, action_shape)

        # Activation
        self.activation = nn.GELU()

    @staticmethod
    def _mps_friendly_dropout(p):
        class MPSFriendlyDropout(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p
            def forward(self, x):
                if not self.training or self.p == 0:
                    return x
                mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
                return x * mask / (1 - self.p)
        return MPSFriendlyDropout(p)

    def forward(self, x, return_features=False):
        # Input embedding
        features = self.embedding(x)
        features = self.activation(features)
        features = self.dropout(features)

        # Hidden layers
        for block in self.blocks:
            features = block(features)
            features = self.activation(features)
            features = self.dropout(features)

        # Output
        output = self.output(features)

        if return_features:
            return output, features
        return output
