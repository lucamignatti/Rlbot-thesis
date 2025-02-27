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


class SimBa(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim=512, num_blocks=2,
                 dropout_rate=0.1, device="cpu"):
        super(SimBa, self).__init__()
        self.device = torch.device(device)

        # save IO shapes
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # Running Statistics Normalization
        self.rsnorm = RSNorm(obs_shape)

        # Initial embedding
        self.embedding = nn.Linear(obs_shape, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Residual feedforward blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResidualFFBlock(hidden_dim, dropout_rate))

        # Post-layer normalization
        self.post_ln = nn.LayerNorm(hidden_dim)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, action_shape)

        self.to(self.device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
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

        return x


class ResidualFFBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualFFBlock, self).__init__()

        # Pre-layer normalization
        self.ln = nn.LayerNorm(hidden_dim)

        # MLP with inverted bottleneck (4x expansion)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Pre-layer normalization
        norm_x = self.ln(x)

        # Apply MLP with dropout
        h = self.linear1(norm_x)
        h = F.relu(h)
        h = self.dropout1(h)
        h = self.linear2(h)
        h = self.dropout2(h)

        # Residual connection
        return x + h

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
        device = self.running_mean.device
        x = x.to(device)

        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            self.num_batches_tracked += 1

            if self.num_batches_tracked == 1:
                update_factor = 1
            else:
                update_factor = self.momentum

            self.running_mean = (1 - update_factor) * self.running_mean + update_factor * batch_mean
            self.running_var = (1 - update_factor) * self.running_var + update_factor * batch_var

            return (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
