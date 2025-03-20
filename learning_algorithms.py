import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical, Normal
import numpy as np
import time
import abc
import math
from collections import deque
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


class PPOAlgorithm(BaseAlgorithm):
    """PPO algorithm implementation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # PPO-specific attributes
        self.buffer_size = kwargs.get('buffer_size', 10000)
        
        # Initialize experience buffer
        self.memory = self.PPOMemory(self.buffer_size, self.device)
        
        # Define scaler for mixed precision training if enabled
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
    
    class PPOMemory:
        """Memory buffer for PPO to store experiences"""
        
        def __init__(self, buffer_size, device):
            self.obs = None
            self.actions = None
            self.log_probs = None
            self.rewards = None
            self.values = None
            self.dones = None
            
            self.buffer_size = buffer_size
            self.device = device
            self.pos = 0
            self.full = False
            
            # Initialize buffers as empty tensors
            self._reset_buffers()
        
        def _reset_buffers(self):
            """Reset all buffers to empty tensors"""
            self.obs = None
            self.actions = None
            self.log_probs = None
            self.rewards = None
            self.values = None
            self.dones = None
            self.pos = 0
            self.full = False
        
        def store(self, obs, action, log_prob, reward, value, done):
            """Store a single experience in the buffer"""
            # Initialize buffers on first call based on sample shapes
            if self.obs is None:
                # Single sample, convert to batch of 1 for shape inference
                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs, dtype=torch.float32)
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32)
                if not isinstance(log_prob, torch.Tensor):
                    log_prob = torch.tensor(log_prob, dtype=torch.float32)
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, dtype=torch.float32)
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32)
                if not isinstance(done, torch.Tensor):
                    done = torch.tensor(done, dtype=torch.bool)
                
                # Ensure all tensors have batch dimension
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                if action.dim() == 1:
                    action = action.unsqueeze(0)
                if log_prob.dim() == 0:
                    log_prob = log_prob.unsqueeze(0)
                if value.dim() == 0:
                    value = value.unsqueeze(0)
                if done.dim() == 0:
                    done = done.unsqueeze(0)
                
                # Initialize buffers with correct shapes
                self.obs = torch.zeros((self.buffer_size, *obs.shape[1:]), dtype=torch.float32, device=self.device)
                self.actions = torch.zeros((self.buffer_size, *action.shape[1:]), dtype=torch.float32, device=self.device)
                self.log_probs = torch.zeros((self.buffer_size, *log_prob.shape[1:]), dtype=torch.float32, device=self.device)
                self.rewards = torch.zeros((self.buffer_size, *reward.shape[1:]), dtype=torch.float32, device=self.device)
                self.values = torch.zeros((self.buffer_size, *value.shape[1:]), dtype=torch.float32, device=self.device)
                self.dones = torch.zeros((self.buffer_size, *done.shape[1:]), dtype=torch.bool, device=self.device)
            
            # Convert inputs to tensors if they're not already
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)
            if not isinstance(log_prob, torch.Tensor):
                log_prob = torch.tensor(log_prob, dtype=torch.float32, device=self.device)
            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32, device=self.device)
            if not isinstance(done, torch.Tensor):
                done = torch.tensor(done, dtype=torch.bool, device=self.device)
            
            # Ensure all tensors are on the correct device
            obs = obs.to(self.device)
            action = action.to(self.device)
            log_prob = log_prob.to(self.device)
            reward = reward.to(self.device)
            value = value.to(self.device)
            done = done.to(self.device)
            
            # Store the experience
            if obs.dim() < self.obs[0].dim() + 1:  # +1 for batch dimension in stored value
                self.obs[self.pos] = obs
            else:
                # Handle different batch dimensions
                self.obs[self.pos] = obs.squeeze(0)
                
            if action.dim() < self.actions[0].dim() + 1:
                self.actions[self.pos] = action
            else:
                self.actions[self.pos] = action.squeeze(0)
                
            if log_prob.dim() < self.log_probs[0].dim() + 1:
                self.log_probs[self.pos] = log_prob
            else:
                self.log_probs[self.pos] = log_prob.squeeze(0)
                
            if reward.dim() < self.rewards[0].dim() + 1:
                self.rewards[self.pos] = reward
            else:
                self.rewards[self.pos] = reward.squeeze(0)
                
            if value.dim() < self.values[0].dim() + 1:
                self.values[self.pos] = value
            else:
                self.values[self.pos] = value.squeeze(0)
                
            if done.dim() < self.dones[0].dim() + 1:
                self.dones[self.pos] = done
            else:
                self.dones[self.pos] = done.squeeze(0)
            
            # Update position and full flag
            self.pos = (self.pos + 1) % self.buffer_size
            self.full = self.full or self.pos == 0
        
        def store_at_idx(self, idx, obs=None, action=None, log_prob=None, reward=None, value=None, done=None):
            """Store values at a specific index, only updating the provided fields"""
            if idx < 0 or idx >= self.buffer_size:
                raise IndexError(f"Index {idx} out of bounds for buffer size {self.buffer_size}")
                
            if obs is not None:
                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                self.obs[idx] = obs
                
            if action is not None:
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32, device=self.device)
                self.actions[idx] = action
                
            if log_prob is not None:
                if not isinstance(log_prob, torch.Tensor):
                    log_prob = torch.tensor(log_prob, dtype=torch.float32, device=self.device)
                self.log_probs[idx] = log_prob
                
            if reward is not None:
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
                self.rewards[idx] = reward
                
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32, device=self.device)
                self.values[idx] = value
                
            if done is not None:
                if not isinstance(done, torch.Tensor):
                    done = torch.tensor(done, dtype=torch.bool, device=self.device)
                self.dones[idx] = done
        
        def get_generator(self, batch_size, compute_returns=True, gamma=0.99, gae_lambda=0.95):
            """Get a generator that yields batches of experiences
            
            Args:
                batch_size: Size of each batch
                compute_returns: Whether to compute returns and advantages
                gamma: Discount factor for computing returns
                gae_lambda: GAE lambda parameter
                
            Returns:
                Generator yielding batches of experiences
            """
            if self.obs is None:
                return
                
            # Number of samples to use (all if buffer is full, or position if not)
            n_samples = self.buffer_size if self.full else self.pos
            
            # If no samples, return empty generator
            if n_samples == 0:
                return
                
            # Compute returns and advantages if requested
            if compute_returns:
                returns, advantages = self._compute_returns_and_advantages(
                    self.rewards[:n_samples],
                    self.values[:n_samples],
                    self.dones[:n_samples],
                    gamma,
                    gae_lambda
                )
            else:
                # If not computing returns, use stored values
                returns = self.rewards[:n_samples]  # Use immediate rewards as returns
                advantages = torch.zeros_like(self.rewards[:n_samples])  # Zero advantages
                
            # Create indices for all samples
            indices = torch.randperm(n_samples)
            
            # Yield batches
            start_idx = 0
            while start_idx < n_samples:
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Extract batch data
                batch = {
                    'obs': self.obs[batch_indices],
                    'actions': self.actions[batch_indices],
                    'log_probs': self.log_probs[batch_indices],
                    'returns': returns[batch_indices],
                    'advantages': advantages[batch_indices]
                }
                
                yield batch
                
                start_idx += batch_size
        
        def _compute_returns_and_advantages(self, rewards, values, dones, gamma, gae_lambda):
            """Compute returns and advantages using GAE
            
            Args:
                rewards: Tensor of rewards [T]
                values: Tensor of values [T]
                dones: Tensor of done flags [T]
                gamma: Discount factor
                gae_lambda: GAE lambda parameter
                
            Returns:
                returns: Tensor of returns [T]
                advantages: Tensor of advantages [T]
            """
            n_samples = len(rewards)
            
            # Initialize returns and advantages
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)
            
            # Initialize gae
            gae = 0
            
            # Initialize next_value and next_done for bootstrapping
            next_value = 0
            next_done = False
            
            # Compute returns and advantages in reverse order
            for t in reversed(range(n_samples)):
                # Get next value and done
                if t < n_samples - 1:
                    next_value = values[t + 1]
                    next_done = dones[t + 1]
                else:
                    next_value = 0
                    next_done = False
                    
                # Compute TD error
                delta = rewards[t] + gamma * next_value * (~next_done) - values[t]
                
                # Compute GAE
                gae = delta + gamma * gae_lambda * (~next_done) * gae
                
                # Store advantages and returns
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]
                
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return returns, advantages
        
        def clear(self):
            """Clear the buffer"""
            self._reset_buffers()
            
        def size(self):
            """Get the number of samples in the buffer"""
            return self.buffer_size if self.full else self.pos
    
    def get_action(self, obs, deterministic=False, return_features=False):
        """Get action for a given observation 
        
        Args:
            obs: Observation tensor [B, obs_dim]
            deterministic: If True, return the most likely action instead of sampling
            return_features: If True, return the features from the last hidden layer
            
        Returns:
            action: Action tensor [B, action_dim]
            log_prob: Log probability of the action [B]
            value: Value estimate [B]
            features: Features from the last hidden layer (if return_features=True)
        """
        # Ensure observation is a torch tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
        # Get action distribution from actor
        with torch.no_grad():
            if return_features:
                action_dist, actor_features = self.actor(obs, return_features=True)
                value, critic_features = self.critic(obs, return_features=True)
                features = torch.cat([actor_features, critic_features], dim=-1) if hasattr(self, 'hidden_dim') else actor_features
            else:
                action_dist = self.actor(obs)
                value = self.critic(obs)
            
            # Get action based on deterministic flag
            if deterministic:
                if self.action_space_type == "discrete":
                    # For discrete actions, take the most likely action
                    probs = action_dist
                    action = torch.argmax(probs, dim=-1)
                    
                    # One-hot encode the action
                    action_one_hot = torch.zeros_like(probs)
                    action_one_hot.scatter_(-1, action.unsqueeze(-1), 1)
                    
                    # Calculate log probability
                    log_prob = torch.log(torch.sum(probs * action_one_hot, dim=-1) + 1e-10)
                else:
                    # For continuous actions, use the mean of the distribution
                    action = action_dist.loc
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
            else:
                if self.action_space_type == "discrete":
                    # For discrete actions, sample from the categorical distribution
                    probs = action_dist
                    
                    # Handle numerical instability
                    probs = torch.clamp(probs, 1e-10, 1.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    
                    # Sample action
                    action_dist = torch.distributions.Categorical(probs=probs)
                    action = action_dist.sample()
                    
                    # One-hot encode the action
                    action_one_hot = torch.zeros_like(probs)
                    action_one_hot.scatter_(-1, action.unsqueeze(-1), 1)
                    
                    # Set action to one-hot vector for consistency with continuous case
                    action = action_one_hot
                    
                    # Calculate log probability
                    log_prob = action_dist.log_prob(torch.argmax(action, dim=-1))
                else:
                    # For continuous actions, sample from the normal distribution
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        # Reshape value to match action dimensionality
        if value.dim() < action.dim():
            value = value.unsqueeze(-1)
            
        # Return action, log_prob, and value
        if return_features:
            return action, log_prob, value, features
        else:
            return action, log_prob, value
    
    def store_experience(self, obs, action, log_prob, reward, value, done):
        """Store experience in the buffer
        
        Args:
            obs: Observation
            action: Action
            log_prob: Log probability of the action
            reward: Reward
            value: Value estimate
            done: Done flag
        """
        self.memory.store(obs, action, log_prob, reward, value, done)
    
    def store_experience_at_idx(self, idx, obs=None, action=None, log_prob=None, reward=None, value=None, done=None):
        """Store experience at a specific index in the buffer
        
        Args:
            idx: Index to store at
            obs: Observation (optional)
            action: Action (optional)
            log_prob: Log probability of the action (optional)
            reward: Reward (optional)
            value: Value estimate (optional)
            done: Done flag (optional)
        """
        self.memory.store_at_idx(idx, obs, action, log_prob, reward, value, done)
    
    def update(self):
        """Update policy using PPO
        
        Returns:
            dict: Dictionary of metrics from the update
        """
        # Calculate buffer size
        buffer_size = self.memory.size()
        
        if buffer_size == 0:
            if self.debug:
                print("[DEBUG] Buffer is empty, skipping update")
            return self.metrics
            
        # Calculate number of batches
        n_batches = math.ceil(buffer_size / self.batch_size)
        
        # Initialize metrics
        actor_loss_epoch = 0
        critic_loss_epoch = 0
        entropy_loss_epoch = 0
        total_loss_epoch = 0
        
        # Get data generator with precomputed returns and advantages
        generator = self.memory.get_generator(
            self.batch_size,
            compute_returns=True,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # Perform PPO epochs
        for epoch in range(self.ppo_epochs):
            # Reset generator for each epoch
            generator = self.memory.get_generator(
                self.batch_size,
                compute_returns=True if epoch == 0 else False,  # Only compute returns in first epoch
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )
            
            # Iterate over batches
            for batch in generator:
                # Extract batch data
                obs = batch['obs']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                returns = batch['returns']
                advantages = batch['advantages']
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Get new action distribution and values
                    if self.action_space_type == "discrete":
                        action_probs = self.actor(obs)
                        
                        # Handle numerical instability
                        action_probs = torch.clamp(action_probs, 1e-10, 1.0)
                        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                        
                        # Calculate log probability of actions
                        dist = torch.distributions.Categorical(probs=action_probs)
                        
                        # Get one-hot action indices
                        action_idx = torch.argmax(actions, dim=-1)
                        
                        # Calculate log probability
                        new_log_probs = dist.log_prob(action_idx)
                        
                        # Calculate entropy
                        entropy = dist.entropy().mean()
                    else:
                        dist = self.actor(obs)
                        new_log_probs = dist.log_prob(actions).sum(dim=-1)
                        entropy = dist.entropy().mean()
                    
                    # Get new values
                    new_values = self.critic(obs).squeeze(-1)
                    
                    # Calculate ratio and clipped ratio for PPO
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # Clamp ratio for numerical stability
                    ratio = torch.clamp(ratio, 0.01, 100.0)
                    
                    # Calculate surrogate objectives
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                    
                    # Calculate actor loss (negative because we're maximizing)
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Calculate critic loss
                    critic_loss = F.mse_loss(new_values, returns)
                    
                    # Calculate entropy loss (negative because we're maximizing)
                    entropy_loss = -entropy * self.entropy_coef
                    
                    # Calculate total loss
                    total_loss = actor_loss + self.critic_coef * critic_loss + entropy_loss
                    
                # Backward pass with mixed precision
                if self.use_amp:
                    # Scale loss to avoid underflow
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    
                    # Backward pass with loss scaling
                    self.scaler.scale(total_loss).backward()
                    
                    # Clip gradients
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.actor_optimizer)
                        self.scaler.unscale_(self.critic_optimizer)
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    
                    # Update weights with scaled gradients
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    
                    # Update scaler for next batch
                    self.scaler.update()
                else:
                    # Standard backward pass
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Clip gradients
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    
                    # Update weights
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                
                # Accumulate losses
                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                total_loss_epoch += total_loss.item()
        
        # Average losses over batches and epochs
        n_updates = n_batches * self.ppo_epochs
        actor_loss_avg = actor_loss_epoch / n_updates
        critic_loss_avg = critic_loss_epoch / n_updates
        entropy_loss_avg = entropy_loss_epoch / n_updates
        total_loss_avg = total_loss_epoch / n_updates
        
        # Update metrics
        self.metrics.update({
            'actor_loss': actor_loss_avg,
            'critic_loss': critic_loss_avg,
            'entropy_loss': entropy_loss_avg,
            'total_loss': total_loss_avg
        })
        
        # Clear memory
        self.memory.clear()
        
        return self.metrics
    
    def reset(self):
        """Reset memory"""
        self.memory.clear()


class StreamACAlgorithm(BaseAlgorithm):
    """Implementation of StreamAC from the paper https://arxiv.org/pdf/2410.14606"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize StreamAC-specific parameters
        self.adaptive_learning_rate = kwargs.get('adaptive_learning_rate', True)
        self.target_step_size = kwargs.get('target_step_size', 0.025)
        self.backtracking_patience = kwargs.get('backtracking_patience', 10)
        self.backtracking_zeta = kwargs.get('backtracking_zeta', 0.85)
        self.min_lr_factor = kwargs.get('min_lr_factor', 0.1)
        self.max_lr_factor = kwargs.get('max_lr_factor', 10.0)
        self.use_obgd = kwargs.get('use_obgd', True)
        self.stream_buffer_size = kwargs.get('stream_buffer_size', 32)
        self.use_sparse_init = kwargs.get('use_sparse_init', True)
        self.update_freq = kwargs.get('update_freq', 1)
        
        # Initialize metrics with StreamAC-specific metrics
        self.metrics.update({
            'effective_step_size': 0.0,
            'actor_lr': self.lr_actor,
            'critic_lr': self.lr_critic,
            'backtracking_count': 0
        })
        
        # Reset optimizers with proper initialization
        if self.use_sparse_init:
            self.apply_sparse_init(self.actor)
            self.apply_sparse_init(self.critic)
            
        # Re-init optimizers with potentially new network weights
        self._init_optimizers()
        
        # Initialize experience buffer
        self.experience_buffer = deque(maxlen=self.stream_buffer_size)
        
        # Initialize tracking variables for step size and backtracking
        self.steps_since_backtrack = 0
        self.base_lr_actor = self.lr_actor
        self.base_lr_critic = self.lr_critic
        self.effective_step_size_history = []
        self.last_parameter_update = None
        self.update_counter = 0
        
        # Initialize gradient accumulators for OBGD
        if self.use_obgd:
            self.actor_grad_accum = {}
            self.critic_grad_accum = {}
            self._init_grad_accumulators()
    
    def apply_sparse_init(self, model):
        """Apply SparseInit initialization as described in the paper
        
        Args:
            model: The model to initialize
        """
        # Initialize all parameters with a smaller variance than standard initialization
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Check if this is a convolutional or linear layer
                if param.dim() > 1:
                    # For Conv or Linear layers, use scaled initialization
                    fan_in = param.size(1)
                    if param.dim() > 2:
                        # Convolutional layers have additional dimensions
                        fan_in *= param.size(2) * param.size(3)
                    
                    # Scale factor based on fan-in (as described in the paper)
                    scale = 1.0 / math.sqrt(fan_in)
                    
                    # Apply sparse-style initialization
                    # The paper suggests 90% sparsity
                    sparsity = 0.9
                    
                    # Initialize with zeros
                    param.data.zero_()
                    
                    # Set a fraction of the weights to non-zero values
                    mask = torch.rand_like(param) > sparsity
                    num_nonzero = mask.sum().item()
                    
                    if num_nonzero > 0:
                        # Initialize non-zero weights with proper scaling
                        param.data[mask] = torch.randn(num_nonzero, device=param.device) * scale * 3.0
                    else:
                        # Ensure at least one non-zero weight per row
                        if param.dim() == 2:  # Linear layers
                            for i in range(param.size(0)):
                                idx = torch.randint(0, param.size(1), (1,))
                                param.data[i, idx] = torch.randn(1, device=param.device) * scale * 3.0
                        else:
                            # For Conv layers, ensure at least one non-zero
                            idx = torch.randint(0, param.numel(), (1,))
                            param.data.view(-1)[idx] = torch.randn(1, device=param.device) * scale * 3.0
            
            # Initialize bias terms to zero
            elif 'bias' in name:
                param.data.zero_()
    
    def _init_optimizers(self):
        """Initialize optimizers with the current learning rates"""
        # For StreamAC, we use Adam optimizer with the current learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
    
    def _init_grad_accumulators(self):
        """Initialize gradient accumulators for OBGD"""
        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                self.actor_grad_accum[name] = torch.zeros_like(param.data)
                
        for name, param in self.critic.named_parameters():
            if param.requires_grad:
                self.critic_grad_accum[name] = torch.zeros_like(param.data)
    
    def get_action(self, obs, deterministic=False, return_features=False):
        """Get action for a given observation
        
        Args:
            obs: Observation tensor [B, obs_dim]
            deterministic: If True, return the most likely action instead of sampling
            return_features: If True, return the features from the last hidden layer
            
        Returns:
            action: Action tensor [B, action_dim]
            log_prob: Log probability of the action [B]
            value: Value estimate [B]
            features: Features from the last hidden layer (if return_features=True)
        """
        # Implementation is the same as PPO
        # Ensure observation is a torch tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
        # Get action distribution from actor
        with torch.no_grad():
            if return_features:
                action_dist, actor_features = self.actor(obs, return_features=True)
                value, critic_features = self.critic(obs, return_features=True)
                features = torch.cat([actor_features, critic_features], dim=-1) if hasattr(self, 'hidden_dim') else actor_features
            else:
                action_dist = self.actor(obs)
                value = self.critic(obs)
            
            # Get action based on deterministic flag
            if deterministic:
                if self.action_space_type == "discrete":
                    # For discrete actions, take the most likely action
                    probs = action_dist
                    action = torch.argmax(probs, dim=-1)
                    
                    # One-hot encode the action
                    action_one_hot = torch.zeros_like(probs)
                    action_one_hot.scatter_(-1, action.unsqueeze(-1), 1)
                    
                    # Calculate log probability
                    log_prob = torch.log(torch.sum(probs * action_one_hot, dim=-1) + 1e-10)
                else:
                    # For continuous actions, use the mean of the distribution
                    action = action_dist.loc
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
            else:
                if self.action_space_type == "discrete":
                    # For discrete actions, sample from the categorical distribution
                    probs = action_dist
                    
                    # Handle numerical instability
                    probs = torch.clamp(probs, 1e-10, 1.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    
                    # Sample action
                    action_dist = torch.distributions.Categorical(probs=probs)
                    action = action_dist.sample()
                    
                    # One-hot encode the action
                    action_one_hot = torch.zeros_like(probs)
                    action_one_hot.scatter_(-1, action.unsqueeze(-1), 1)
                    
                    # Set action to one-hot vector for consistency with continuous case
                    action = action_one_hot
                    
                    # Calculate log probability
                    log_prob = action_dist.log_prob(torch.argmax(action, dim=-1))
                else:
                    # For continuous actions, sample from the normal distribution
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        # Reshape value to match action dimensionality
        if value.dim() < action.dim():
            value = value.unsqueeze(-1)
            
        # Return action, log_prob, and value
        if return_features:
            return action, log_prob, value, features
        else:
            return action, log_prob, value
    
    def store_experience(self, obs, action, log_prob, reward, value, done):
        """Store experience in the buffer (limited size)
        
        Args:
            obs: Observation
            action: Action
            log_prob: Log probability of the action
            reward: Reward
            value: Value estimate
            done: Done flag
        """
        # Convert all inputs to torch tensors
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob, dtype=torch.float32, device=self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        # Store experience as a dictionary
        experience = {
            'obs': obs,
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'value': value,
            'done': done
        }
        
        # Add experience to buffer
        self.experience_buffer.append(experience)
        
        # Update after every experience if update_freq is 1
        self.update_counter += 1
        if self.update_counter >= self.update_freq:
            self.update()
            self.update_counter = 0
        
        return self.metrics
    
    def update(self, experiences=None):
        """Update policy using the latest experience
        
        Args:
            experiences: Optional list of experiences (not used in StreamAC, which uses the buffer)
            
        Returns:
            dict: Dictionary of metrics from the update
        """
        # Skip if buffer is empty
        if len(self.experience_buffer) == 0:
            return self.metrics
        
        # Use only the latest experience for the StreamAC update
        # The paper recommends using just the latest experience for true online learning
        exp = self.experience_buffer[-1]
        obs = exp['obs']
        action = exp['action']
        old_log_prob = exp['log_prob']
        reward = exp['reward']
        old_value = exp['value']
        done = exp['done']
        
        # Compute returns and advantages for the single experience
        returns = reward
        if not done:
            with torch.no_grad():
                next_value = self.critic(obs).squeeze(-1)
                returns = reward + self.gamma * next_value
        
        advantages = returns - old_value
        
        # Save current parameters for step size calculation
        if self.last_parameter_update is None:
            self.last_parameter_update = self._get_parameter_vector()
        
        # Forward pass
        # Get new action distribution and values
        if self.action_space_type == "discrete":
            action_probs = self.actor(obs)
            
            # Handle numerical instability
            action_probs = torch.clamp(action_probs, 1e-10, 1.0)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            
            # Get one-hot action indices
            action_idx = torch.argmax(action, dim=-1)
            
            # Calculate log probability
            dist = torch.distributions.Categorical(probs=action_probs)
            new_log_prob = dist.log_prob(action_idx)
            
            # Calculate entropy
            entropy = dist.entropy().mean()
        else:
            dist = self.actor(obs)
            new_log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().mean()
        
        # Get new value
        new_value = self.critic(obs).squeeze(-1)
        
        # Calculate ratio and clipped ratio for PPO-style update
        ratio = torch.exp(new_log_prob - old_log_prob)
        
        # Clamp ratio for numerical stability
        ratio = torch.clamp(ratio, 0.01, 100.0)
        
        # Calculate surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        
        # Calculate actor loss (negative because we're maximizing)
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate critic loss
        critic_loss = F.mse_loss(new_value, returns)
        
        # Calculate entropy loss (negative because we're maximizing)
        entropy_loss = -entropy * self.entropy_coef
        
        # Calculate total loss
        total_loss = actor_loss + self.critic_coef * critic_loss + entropy_loss
        
        # Perform gradient update with OBGD or standard update
        if self.use_obgd:
            # Online Backpropagation Gradient Descent (OBGD)
            # Clear gradients
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            # Compute gradients
            total_loss.backward()
            
            # Update accumulated gradients
            for name, param in self.actor.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Simple moving average of gradients
                    self.actor_grad_accum[name] = 0.9 * self.actor_grad_accum[name] + 0.1 * param.grad
                    
                    # Overwrite gradient with accumulated gradient
                    param.grad = self.actor_grad_accum[name].clone()
            
            for name, param in self.critic.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Simple moving average of gradients
                    self.critic_grad_accum[name] = 0.9 * self.critic_grad_accum[name] + 0.1 * param.grad
                    
                    # Overwrite gradient with accumulated gradient
                    param.grad = self.critic_grad_accum[name].clone()
            
            # Apply gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            # Step optimizers
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        else:
            # Standard update
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            
            # Apply gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            # Step optimizers
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
        # Calculate effective step size
        current_params = self._get_parameter_vector()
        param_diff = current_params - self.last_parameter_update
        effective_step_size = torch.norm(param_diff).item()
        self.last_parameter_update = current_params
        
        # Store effective step size for backtracking
        self.effective_step_size_history.append(effective_step_size)
        
        # Implement adaptive learning rate adjustment
        if self.adaptive_learning_rate:
            # Only adapt the learning rate after we have enough history
            if len(self.effective_step_size_history) > 10:
                # Check if we need to adjust the learning rate
                # If step size is too large or too small compared to target
                if effective_step_size > self.target_step_size * 1.1:
                    # Step size too large, decrease learning rate
                    self.lr_actor *= 0.95
                    self.lr_critic *= 0.95
                    
                    # Update optimizer learning rates
                    for param_group in self.actor_optimizer.param_groups:
                        param_group['lr'] = self.lr_actor
                    for param_group in self.critic_optimizer.param_groups:
                        param_group['lr'] = self.lr_critic
                elif effective_step_size < self.target_step_size * 0.9:
                    # Step size too small, increase learning rate
                    self.lr_actor *= 1.05
                    self.lr_critic *= 1.05
                    
                    # Update optimizer learning rates
                    for param_group in self.actor_optimizer.param_groups:
                        param_group['lr'] = self.lr_actor
                    for param_group in self.critic_optimizer.param_groups:
                        param_group['lr'] = self.lr_critic
                
                # Enforce limits on learning rate
                self.lr_actor = max(self.base_lr_actor * self.min_lr_factor, 
                                  min(self.base_lr_actor * self.max_lr_factor, self.lr_actor))
                self.lr_critic = max(self.base_lr_critic * self.min_lr_factor, 
                                   min(self.base_lr_critic * self.max_lr_factor, self.lr_critic))
        
        # Implement backtracking for better stability
        self.steps_since_backtrack += 1
        
        # Check if we need to apply backtracking
        if (self.steps_since_backtrack >= self.backtracking_patience and 
            len(self.effective_step_size_history) >= self.backtracking_patience):
            
            # Calculate average of recent step sizes
            recent_avg = sum(self.effective_step_size_history[-self.backtracking_patience:]) / self.backtracking_patience
            
            # If average step size is much larger than target, reduce learning rate more aggressively
            if recent_avg > self.target_step_size * 2.0:
                self.lr_actor *= self.backtracking_zeta
                self.lr_critic *= self.backtracking_zeta
                
                # Update optimizer learning rates
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.lr_actor
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.lr_critic
                
                # Update metrics
                self.metrics['backtracking_count'] += 1
                
                # Reset backtracking counter
                self.steps_since_backtrack = 0
        
        # Update metrics
        self.metrics.update({
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'effective_step_size': effective_step_size,
            'actor_lr': self.lr_actor,
            'critic_lr': self.lr_critic
        })
        
        return self.metrics
    
    def _get_parameter_vector(self):
        """Get the current parameters as a single vector for step size calculation
        
        Returns:
            torch.Tensor: Vector of all parameters
        """
        # Get all parameters as a flat vector
        actor_params = torch.cat([p.data.view(-1) for p in self.actor.parameters()])
        critic_params = torch.cat([p.data.view(-1) for p in self.critic.parameters()])
        return torch.cat([actor_params, critic_params])
    
    def reset(self):
        """Reset the algorithm state"""
        self.experience_buffer.clear()
        self.steps_since_backtrack = 0
        self.effective_step_size_history = []
        if self.use_obgd:
            self._init_grad_accumulators()