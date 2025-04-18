import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class BaseAlgorithm:
    """Base class for RL algorithms"""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        action_space_type: str = "discrete",
        action_dim: Optional[int] = None,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: str = "cuda",
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: Optional[float] = None, # PPO specific
        clip_epsilon: Optional[float] = None, # PPO specific
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = None,
        ppo_epochs: Optional[int] = None, # PPO specific
        batch_size: Optional[int] = None, # PPO/SAC specific
        use_amp: bool = False, # Add use_amp
        debug: bool = False,
        use_wandb: bool = False,
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
        self.use_amp = use_amp and "cuda" in str(device) # Enable AMP only if flag is true and device is CUDA
        self.debug = debug
        self.use_wandb = use_wandb

        # Ensure models are on the correct device
        self.actor.to(self.device)
        self.critic.to(self.device)

    def get_action(self, obs, deterministic=False, return_features=False):
        """Get action for a given observation"""
        raise NotImplementedError

    def store_experience(self, obs, action, log_prob, reward, value, done, env_id=0):
        """Store experience in the buffer"""
        raise NotImplementedError

    def update(self):
        """Update policy based on stored experiences"""
        raise NotImplementedError

    def reset(self):
        """Reset algorithm state (e.g., memory)"""
        pass # Optional: Implement in subclasses if needed

    def get_metrics(self) -> Dict[str, float]:
        """Return current training metrics"""
        # Return a copy to prevent modification
        return getattr(self, 'metrics', {}).copy()

    def get_state_dict(self) -> Dict:
        """Get state dict for saving algorithm state"""
        # Base implementation saves model states
        # Subclasses should call super().get_state_dict() and add their specific state
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            # Add hyperparameters that might change (like entropy_coef)
            'entropy_coef': self.entropy_coef,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict for resuming algorithm state"""
        # Base implementation loads model states
        # Subclasses should call super().load_state_dict() and load their specific state
        if 'actor' in state_dict:
            # Handle potential issues with compiled models or architecture changes
            from model_architectures import fix_compiled_state_dict, load_partial_state_dict
            actor_state = fix_compiled_state_dict(state_dict['actor'])
            load_partial_state_dict(self.actor, actor_state)
        if 'critic' in state_dict:
            from model_architectures import fix_compiled_state_dict, load_partial_state_dict
            critic_state = fix_compiled_state_dict(state_dict['critic'])
            load_partial_state_dict(self.critic, critic_state)

        # Load hyperparameters
        if 'entropy_coef' in state_dict:
            self.entropy_coef = state_dict['entropy_coef']

        # Ensure models are on the correct device after loading
        self.actor.to(self.device)
        self.critic.to(self.device)
