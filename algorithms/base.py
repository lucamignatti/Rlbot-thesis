import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Optional, Union, Any, Callable

class BaseAlgorithm:
    """Base class for RL algorithms like PPO and StreamAC"""
    
    def __init__(
            self,
            actor: nn.Module,
            critic: nn.Module,
            action_space_type: str = "discrete",
            action_dim: int = 8,
            device: str = "cpu",
            lr_actor: float = 3e-4,
            lr_critic: float = 1e-3,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_epsilon: float = 0.2,
            critic_coef: float = 0.5,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            **kwargs
        ):
        """Initialize the base algorithm with common parameters
        
        Args:
            actor: Actor network
            critic: Critic network
            action_space_type: Type of action space (discrete or continuous)
            action_dim: Dimension of the action space
            device: Device to use for computation
            lr_actor: Learning rate for the actor network
            lr_critic: Learning rate for the critic network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            critic_coef: Coefficient for the critic loss
            entropy_coef: Coefficient for the entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.actor = actor
        self.critic = critic
        self.action_space_type = action_space_type
        self.action_dim = action_dim
        self.device = device
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize additional attributes from kwargs
        self.ppo_epochs = kwargs.get('ppo_epochs', 10)
        self.batch_size = kwargs.get('batch_size', 64)
        self.use_amp = kwargs.get('use_amp', False)
        self.debug = kwargs.get('debug', False)
        
        # Initialize optimizers
        self.actor_optimizer = None
        self.critic_optimizer = None
        self._init_optimizers()
        
        # Metrics tracking
        self.metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0
        }
        
    def _init_optimizers(self):
        """Initialize optimizers based on the selected learning rates"""
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
    
    def get_action(self, obs, deterministic=False):
        """Get action, log probability, and value for a given observation
        
        Args:
            obs: Observation tensor
            deterministic: If True, return the most likely action without sampling
            
        Returns:
            action: Action tensor
            log_prob: Log probability of the action
            value: Value estimate from the critic
        """
        raise NotImplementedError("Subclasses must implement get_action")
    
    def update(self, experiences):
        """Update policy using the given experiences
        
        Args:
            experiences: List of experience tuples
            
        Returns:
            dict: Dictionary of metrics from the update
        """
        raise NotImplementedError("Subclasses must implement update")
    
    def store_experience(self, *args):
        """Store experience for later updates
        
        Args:
            Experience components (observation, action, reward, etc.)
        """
        raise NotImplementedError("Subclasses must implement store_experience")
    
    def reset(self):
        """Reset any algorithm-specific state"""
        pass
    
    def get_metrics(self):
        """Get the latest training metrics
        
        Returns:
            dict: Dictionary of metrics
        """
        return self.metrics.copy()