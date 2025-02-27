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
    def __init__(self, obs_shape, action_shape, device: str ="cpu", dropout_rate: float = 0.1):
        super(SimBa, self).__init__()
        self.device = torch.device(device)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        
        # Create everything on CPU first, then move the whole model at once
        self.rsnorm = RSNorm(obs_shape)
        self.linear1 = nn.Linear(obs_shape, obs_shape*4)
        self.layernorm1 = nn.LayerNorm(obs_shape*4)
        self.linear2 = nn.Linear(obs_shape*4, obs_shape*8)
        self.linear3 = nn.Linear(obs_shape*8, obs_shape*4)
        self.layernorm2 = nn.LayerNorm(obs_shape*4)
        self.outputlayer = nn.Linear(obs_shape*4, action_shape)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Move everything to the target device at once
        self.to(self.device)
        
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        x = self.rsnorm(x)  # Normalize input
        x = self.dropout(self.linear1(x))  # Apply first linear layer with dropout

        @torch.compiler.disable(recursive=False)
        def checkpoint_fn(x):
            # Use checkpointing to reduce memory usage during training
            y = self.layernorm1(x)
            y = self.dropout(self.linear2(y))
            y = F.relu(y)
            y = self.dropout(self.linear3(y))
            return y

        y = checkpoint(checkpoint_fn, x, use_reentrant=False)
        x = x + y  # Residual connection to combine input and transformed features
        x = self.layernorm2(x)
        x = self.dropout(self.outputlayer(x))
        return x

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
