import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, Union, Any

class BaseAlgorithm:
    """Base class for RL algorithms"""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        action_space_type: str = "discrete",
        action_dim: Union[int, Tuple[int]] = None,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        device: str = "cuda",
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        use_amp: bool = False,
        debug: bool = False,
        use_wandb: bool = False,
        **kwargs # Accept extra args for subclasses
    ):
        self.actor = actor
        self.critic = critic
        self.action_space_type = action_space_type
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.device = device
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.debug = debug
        self.use_wandb = use_wandb

        # Initialize basic metrics dictionary
        self.metrics = {}

    def get_action(self, obs, deterministic=False, return_features=False):
        """Get action for a given observation"""
        raise NotImplementedError

    def store_experience(self, *args, **kwargs):
        """Store experience in the buffer"""
        raise NotImplementedError

    def update(self):
        """Update policy based on collected experiences"""
        raise NotImplementedError

    def reset(self):
        """Reset algorithm state (e.g., memory)"""
        pass # Default implementation does nothing

    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics"""
        return self.metrics

    def get_state_dict(self) -> Dict[str, Any]:
        """Return state dictionary for saving"""
        # Base implementation returns an empty dict, subclasses should extend this
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary for resuming"""
        # Base implementation does nothing, subclasses should extend this
        pass
