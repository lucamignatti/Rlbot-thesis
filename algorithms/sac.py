import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Categorical
from typing import Dict, Tuple, List, Optional, Union, Any
from collections import deque
from .base import BaseAlgorithm

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity, observation_shape, action_shape, device="cpu"):
        """Initialize a replay buffer for SAC.
        
        Args:
            capacity: Maximum number of transitions to store
            observation_shape: Shape of the observation space
            action_shape: Shape of the action space
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.observations = torch.zeros((capacity, *observation_shape), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((capacity, *observation_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0
        
    def add(self, obs, action, reward, next_obs, done):
        """Add a new transition to the buffer."""
        # Convert numpy arrays to tensors if needed
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        # Fix: Handle both numpy arrays and scalar numpy values
        if isinstance(action, (int, np.integer)):
            # Convert scalar integer to tensor with batch dimension
            action = torch.tensor([float(action)], dtype=torch.float32, device=self.device)
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        
        if isinstance(reward, (float, int)):
            reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        elif isinstance(reward, np.ndarray):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        if isinstance(next_obs, np.ndarray):
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        
        if isinstance(done, bool):
            done = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        elif isinstance(done, np.ndarray):
            done = torch.tensor(done, dtype=torch.float32, device=self.device)
            
        # Ensure all tensors have the right shape
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0).unsqueeze(0)
        elif reward.dim() == 1:
            reward = reward.unsqueeze(0)
        if next_obs.dim() == 1:
            next_obs = next_obs.unsqueeze(0)
        if done.dim() == 0:
            done = done.unsqueeze(0).unsqueeze(0)
        elif done.dim() == 1:
            done = done.unsqueeze(0)
            
        # Store the transition
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )
        
    def __len__(self):
        return self.size


class SACAlgorithm(BaseAlgorithm):
    """
    Soft Actor-Critic (SAC) implementation.
    
    SAC is an off-policy actor-critic algorithm that:
    1. Uses entropy regularization for exploration
    2. Employs twin Q-networks to mitigate overestimation bias
    3. Features automatic entropy tuning (optional)
    4. Leverages experience replay for sample efficiency
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,  # First Q-network
        critic2: nn.Module,  # Second Q-network for twin Q-learning
        action_space_type: str = "continuous",
        action_dim: int = None,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        device: str = "cpu",
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,  # Target network update rate
        alpha: float = 0.2,  # Temperature parameter for entropy
        auto_alpha_tuning: bool = True,  # Automatically tune alpha
        target_entropy: float = None,  # Target entropy when auto-tuning
        buffer_size: int = 1000000,  # Replay buffer size
        batch_size: int = 256,
        warmup_steps: int = 1000,  # Number of steps before starting to train
        update_freq: int = 1,  # How often to update networks
        updates_per_step: int = 1,  # Number of gradient updates per step
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
        use_wandb: bool = False,  # Add wandb parameter
        debug: bool = False,
        **kwargs
    ):
        # Set critic references BEFORE calling super().__init__
        self.critic1 = critic1
        self.critic2 = critic2
        
        # Initialize with a dummy critic and critic_coef for BaseAlgorithm
        # We will override the critic functionality in this class
        super().__init__(
            actor=actor,
            critic=critic1,  # Just pass the first critic to the parent
            action_space_type=action_space_type,
            action_dim=action_dim,
            device=device,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            gae_lambda=0.0,  # Not used in SAC
            clip_epsilon=0.0,  # Not used in SAC
            critic_coef=1.0,   # Not used in SAC
            entropy_coef=0.0,  # SAC uses alpha instead
            max_grad_norm=max_grad_norm,
            batch_size=batch_size,
            use_amp=use_amp,
            use_wandb=use_wandb,  # Pass wandb parameter
            debug=debug,
            **kwargs
        )
        
        # SAC-specific attributes
        self.action_bounds = action_bounds
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha_tuning = auto_alpha_tuning
        self.buffer_size = buffer_size
        self.warmup_steps = warmup_steps
        self.update_freq = update_freq
        self.updates_per_step = updates_per_step
        
        # Create target networks
        self.critic1_target = self._create_target_network(critic1)
        self.critic2_target = self._create_target_network(critic2)
        
        # Set target entropy if not provided
        if target_entropy is None:
            if action_space_type == "discrete":
                self.target_entropy = -0.98 * np.log(1.0 / action_dim)  # Default for discrete
            else:
                self.target_entropy = -action_dim  # Default for continuous
        else:
            self.target_entropy = target_entropy
            
        # Initialize log_alpha if using auto-tuning
        if self.auto_alpha_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
        
        # Create replay buffer
        obs_shape = self._get_observation_shape()
        action_shape = self._get_action_shape()
        self.memory = ReplayBuffer(buffer_size, obs_shape, action_shape, device)
        
        # Override optimizers
        self._init_optimizers()
        
        # Step counter
        self.steps = 0
        
        # Metrics tracking
        self.metrics.update({
            'actor_loss': 0.0,
            'critic1_loss': 0.0,
            'critic2_loss': 0.0,
            'alpha_loss': 0.0,
            'alpha': self.alpha,
            'mean_q_value': 0.0,
            'mean_entropy': 0.0,
            'mean_return': 0.0,
        })
        
        # Episode return tracking
        self.current_episode_rewards = []
        self.episode_returns = deque(maxlen=100)
        
        if self.debug:
            print(f"[DEBUG] Initialized SAC algorithm with auto_alpha={auto_alpha_tuning}, "
                  f"target_entropy={self.target_entropy}")
    
    def _create_target_network(self, network):
        """Create a target network as a copy of the original network."""
        # For SimBa models, reconstruct with parameters from the original network
        if hasattr(network, 'obs_shape') and hasattr(network, 'action_shape'):
            # Extract network parameters
            kwargs = {
                'obs_shape': network.obs_shape,
                'action_shape': network.action_shape,
                'device': self.device
            }
            
            # Get additional parameters if available
            if hasattr(network, 'hidden_dim'):
                kwargs['hidden_dim'] = network.hidden_dim
            if hasattr(network, 'blocks'):
                kwargs['num_blocks'] = len(network.blocks)
            if hasattr(network, 'dropout_rate'):
                kwargs['dropout_rate'] = network.dropout_rate
                
            # Create a new instance of the same type
            target_net = type(network)(**kwargs).to(self.device)
        else:
            # Fallback for other model types
            import copy
            target_net = copy.deepcopy(network).to(self.device)
        
        # Copy weights from original network
        target_net.load_state_dict(network.state_dict())
        
        # Freeze the target network parameters
        for param in target_net.parameters():
            param.requires_grad = False
            
        return target_net
    
    def _get_observation_shape(self):
        """Get the shape of the observation space."""
        # Try to infer from the actor network
        if hasattr(self.actor, 'input_shape'):
            return self.actor.input_shape
        elif hasattr(self.actor, 'obs_shape'):
            return (self.actor.obs_shape,)
        else:
            # Fallback to a reasonable default
            return (self.action_dim * 2,)
    
    def _get_action_shape(self):
        """Get the shape of the action space."""
        if self.action_space_type == "discrete":
            # For discrete actions, use one-hot encoding shape
            return (self.action_dim,)  # Changed from (1,) to (self.action_dim,)
        else:
            if isinstance(self.action_dim, tuple):
                return self.action_dim
            else:
                return (self.action_dim,)
    
    def _init_optimizers(self):
        """Initialize SAC-specific optimizers."""
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic)
    
    def _soft_update_target_networks(self):
        """Perform soft update of target network parameters."""
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_action(self, obs, deterministic=False, return_features=False):
        """Get an action for a given observation."""
        # Convert observation to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(self.device)
            
        # Ensure observation has batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        # Set models to evaluation mode
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        
        with torch.no_grad():
            if self.action_space_type == "discrete":
                # Get raw action probabilities from the actor
                action_logits = self.actor(obs)
                
                # Apply softmax to get proper probabilities
                action_probs = F.softmax(action_logits, dim=-1)
                
                # Ensure valid probabilities (handle NaN, inf, negative values)
                # Replace any invalid values with small positive value
                action_probs = torch.where(
                    torch.isfinite(action_probs) & (action_probs >= 0),
                    action_probs,
                    torch.tensor(1e-8, device=action_probs.device)
                )
                
                # Normalize to ensure sum = 1
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                
                # Clamp small values for numerical stability
                action_probs = torch.clamp(action_probs, min=1e-8)
                
                if deterministic:
                    # Select the action with highest probability
                    action_idx = torch.argmax(action_probs, dim=-1)
                    # Convert to one-hot encoding
                    action = F.one_hot(action_idx, num_classes=self.action_dim).float()
                    log_prob = torch.log(action_probs.gather(-1, action_idx.unsqueeze(-1))).squeeze(-1)
                else:
                    try:
                        # Create proper categorical distribution and sample
                        dist = Categorical(probs=action_probs)
                        action_idx = dist.sample()
                        # Convert to one-hot encoding
                        action = F.one_hot(action_idx, num_classes=self.action_dim).float()
                        log_prob = dist.log_prob(action_idx)
                    except RuntimeError as e:
                        # Fall back to argmax in case of any sampling errors
                        print(f"Warning: Sampling error, falling back to argmax: {e}")
                        action_idx = torch.argmax(action_probs, dim=-1)
                        action = F.one_hot(action_idx, num_classes=self.action_dim).float()
                        log_prob = torch.log(action_probs + 1e-8).gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
            else:
                # Continuous action space handling
                mean, log_std = self.actor(obs).chunk(2, dim=-1)
                std = torch.exp(log_std)
                
                # Create normal distribution
                dist = Normal(mean, std)
                
                if deterministic:
                    action = mean
                else:
                    action = dist.sample()
                    
                # Apply tanh squashing for bounded actions
                action = torch.tanh(action)
                
                # Scale to action range
                action_scale = (self.action_bounds[1] - self.action_bounds[0]) / 2.0
                action_bias = (self.action_bounds[1] + self.action_bounds[0]) / 2.0
                action = action * action_scale + action_bias
                
                # Compute log probability with squashing correction
                log_prob = dist.log_prob(action)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            # For Q-values, just use observations (don't concat with actions)
            # This is a workaround to avoid changing the model architecture
            q1 = self.critic1(obs)
            q2 = self.critic2(obs)
            
            # Take minimum Q-value 
            value = torch.min(q1, q2)
        
        # Return features if requested
        if return_features:
            _, features = self.actor(obs, return_features=True)
            return action, log_prob, value, features
            
        return action, log_prob, value
    
    def store_experience(self, obs, action, log_prob, reward, value, done, env_id=0):
        """Store experience in the replay buffer."""
        # Track episode rewards for statistics
        if isinstance(reward, torch.Tensor):
            reward_item = reward.item()
        else:
            reward_item = reward
            
        self.current_episode_rewards.append(reward_item)
        
        # If we have a next observation stored from a previous call
        if hasattr(self, 'prev_obs') and self.prev_obs is not None:
            # Convert action to tensor if it's one-hot encoded or an index
            if self.action_space_type == "discrete":
                # Properly handle various action formats for discrete actions
                if isinstance(action, (int, np.integer)):
                    # For integer actions (scalar or numpy), convert to one-hot
                    action_tensor = np.zeros(self.action_dim)
                    action_tensor[int(action)] = 1.0
                    action = action_tensor
                elif isinstance(action, np.ndarray):
                    # Handle numpy arrays that might be indices or one-hot
                    if action.size == 1:
                        # Single value array (index)
                        action_tensor = np.zeros(self.action_dim)
                        action_tensor[int(action.item())] = 1.0
                        action = action_tensor
                    elif len(action.shape) == 1 and action.shape[0] > 1:
                        # Already one-hot encoded as flat array
                        pass
                elif isinstance(action, torch.Tensor):
                    # Handle tensor actions
                    if action.dim() == 0 or (action.dim() == 1 and action.shape[0] == 1):
                        # Single index
                        action_idx = action.item()
                        action_tensor = np.zeros(self.action_dim)
                        action_tensor[int(action_idx)] = 1.0
                        action = action_tensor
            
            # Add experience to replay buffer
            self.memory.add(self.prev_obs, action, reward, obs, done)
            
            # Update model if we have enough samples and it's time to update
            if (self.steps >= self.warmup_steps and 
                len(self.memory) >= self.batch_size and 
                self.steps % self.update_freq == 0):
                
                for _ in range(self.updates_per_step):
                    self.update()
        
        # Store current observation for next step
        self.prev_obs = obs
        
        # If episode ended, reset episode tracking and update statistics
        if done:
            episode_return = sum(self.current_episode_rewards)
            self.episode_returns.append(episode_return)
            self.metrics['mean_return'] = np.mean(self.episode_returns) if self.episode_returns else 0
            
            # Reset episode rewards
            self.current_episode_rewards = []
            
            # Reset previous observation
            self.prev_obs = None
        
        # Increment step counter
        self.steps += 1
    
    def update(self):
        """Update the SAC networks using a batch from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return self.metrics
            
        # Update step counter for proper tracking
        self.train_steps = getattr(self, 'train_steps', 0) + 1
        
        # Sample a batch from the replay buffer
        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)
        
        # Get current alpha value
        if self.auto_alpha_tuning:
            alpha = self.log_alpha.exp().item()
        else:
            alpha = self.alpha
            
        # Update critic networks
        critic_loss1, critic_loss2 = self._update_critics(
            observations, actions, rewards, next_observations, dones, alpha
        )
        
        # Update actor network and alpha (if auto-tuning)
        actor_loss, entropy, alpha_loss = self._update_actor_and_alpha(observations, alpha)
        
        # Perform soft update of target networks
        self._soft_update_target_networks()
        
        # Update metrics
        self.metrics.update({
            'actor_loss': actor_loss,
            'critic1_loss': critic_loss1,
            'critic2_loss': critic_loss2,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'mean_q_value': 0.0,
            'mean_entropy': entropy,
            'step': self.train_steps,
        })
        
        return self.metrics
        
    def _update_critics(self, obs, actions, rewards, next_obs, dones, alpha):
        with torch.no_grad():
            # Get next actions and log probs for target calculation
            if self.action_space_type == "discrete":
                next_action_probs = F.softmax(self.actor(next_obs), dim=-1)
                next_action_probs = torch.clamp(next_action_probs, min=1e-8)
                dist = Categorical(probs=next_action_probs)
                next_action_idx = dist.sample()
                next_actions = F.one_hot(next_action_idx, num_classes=self.action_dim).float()
                next_log_probs = dist.log_prob(next_action_idx).unsqueeze(-1)
            else:
                next_mean, next_log_std = self.actor(next_obs).chunk(2, dim=-1)
                next_std = torch.exp(next_log_std)
                dist = Normal(next_mean, next_std)
                next_actions_raw = dist.rsample()
                next_actions = torch.tanh(next_actions_raw)
                next_log_probs = dist.log_prob(next_actions_raw) - torch.log(1 - next_actions.pow(2) + 1e-6)
                next_log_probs = next_log_probs.sum(1, keepdim=True)
                
                # Scale to action range
                action_scale = (self.action_bounds[1] - self.action_bounds[0]) / 2.0
                action_bias = (self.action_bounds[1] + self.action_bounds[0]) / 2.0
                next_actions = next_actions * action_scale + action_bias
            
            # Target Q-values - use next_obs directly instead of concatenating 
            next_q1 = self.critic1_target(next_obs)
            next_q2 = self.critic2_target(next_obs)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Current Q-values - use obs directly
        current_q1 = self.critic1(obs)
        current_q2 = self.critic2(obs)
        
        # Compute critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()
        
        return critic1_loss.item(), critic2_loss.item()
    
    def _update_actor_and_alpha(self, obs, alpha):
        """Update the actor network and alpha parameter."""
        if self.action_space_type == "discrete":
            # For discrete actions
            action_probs = F.softmax(self.actor(obs), dim=-1)
            action_probs = torch.clamp(action_probs, min=1e-8)
            
            # Calculate the Q-values for all actions - use obs directly
            q1 = self.critic1(obs)
            q2 = self.critic2(obs)
            q = torch.min(q1, q2)
            
            # Calculate log probabilities
            log_probs = torch.log(action_probs)
            
            # Calculate entropy
            entropy = -torch.sum(action_probs * log_probs, dim=1).mean()
            
            # Actor loss is expectation of Q values weighted by probabilities, minus entropy term
            inside_term = action_probs * (alpha * log_probs - q)
            actor_loss = torch.mean(torch.sum(inside_term, dim=1))
        else:
            # For continuous actions - implementation details will vary
            mean, log_std = self.actor(obs).chunk(2, dim=-1)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            
            # Sample actions
            actions_raw = dist.rsample()
            actions = torch.tanh(actions_raw)
            
            # Scale to action range
            action_scale = (self.action_bounds[1] - self.action_bounds[0]) / 2.0
            action_bias = (self.action_bounds[1] + self.action_bounds[0]) / 2.0
            actions = actions * action_scale + action_bias
            
            # Calculate log probabilities with squashing correction
            log_probs = dist.log_prob(actions_raw) - torch.log(1 - actions.pow(2) + 1e-6)
            log_probs = log_probs.sum(1, keepdim=True)
            
            # Calculate Q-values - use obs directly
            q1 = self.critic1(obs)
            q2 = self.critic2(obs)
            q = torch.min(q1, q2)
            
            # Calculate entropy
            entropy = -log_probs.mean()
            
            # Actor loss: maximize Q - alpha * log_prob
            actor_loss = (alpha * log_probs - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update alpha if auto-tuning
        if self.auto_alpha_tuning:
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            return actor_loss.item(), entropy.item(), alpha_loss.item()
        else:
            # If not auto-tuning, just return 0 for alpha loss
            return actor_loss.item(), entropy.item(), 0.0
    
    def reset(self):
        """Reset episode-specific state at the end of an episode."""
        # Reset episode rewards
        self.current_episode_rewards = []
        
        # Reset previous observation
        self.prev_obs = None
        
    def get_metrics(self):
        """Get current metrics."""
        return self.metrics