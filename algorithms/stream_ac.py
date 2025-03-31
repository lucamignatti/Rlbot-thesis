import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions import Categorical, Normal
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
from .base import BaseAlgorithm
from collections import deque

class ObGD(torch.optim.Optimizer):
    """
    overshooting-Bounded Gradient Descent optimizer with eligibility traces.
    """
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)
        self.env_traces = {}  # Store eligibility traces per environment
        
    def get_traces(self, env_id):
        """Get eligibility traces for a specific environment."""
        if env_id not in self.env_traces:
            self.env_traces[env_id] = {}
            for group in self.param_groups:
                for p in group["params"]:
                    self.env_traces[env_id][p] = torch.zeros_like(p.data)
        return self.env_traces[env_id]
        
    def step(self, delta, env_id=0, reset=False):
        """
        Perform a parameter update step using TD error and eligibility traces.
        Combines trace update and parameter update loops for efficiency.
        """
        z_sum = 0.0
        env_traces = self.get_traces(env_id)
        param_trace_list = [] # To store parameters and traces for the second pass

        # First pass: Update traces, calculate z_sum, and store refs
        for group in self.param_groups:
            # Store group hyperparams locally for efficiency if needed later
            lr = group["lr"]
            gamma = group["gamma"]
            lamda = group["lamda"]
            kappa = group["kappa"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get trace for this param
                e = env_traces[p]

                # Update eligibility trace: e = λγe + ∇θ
                e.mul_(gamma * lamda).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

                # Store param, trace, and group info for the update step
                param_trace_list.append({'param': p, 'trace': e, 'group': group})

        # Calculate adaptive step size (using hyperparams from the last group, assumes they are the same)
        # If hyperparams can differ per group, this calculation needs adjustment.
        delta_bar = max(abs(delta), 1.0)
        # Using lr and kappa from the last processed group. Ensure this is intended behavior.
        # If groups can have different lr/kappa, the step_size calculation might need refinement.
        dot_product = delta_bar * z_sum * lr * kappa
        if dot_product > 1:
            step_size = lr / dot_product
        else:
            step_size = lr

        # Second pass: Apply updates to parameters
        for item in param_trace_list:
            p = item['param']
            e = item['trace']
            # Update parameter: θ = θ - α·δ·e
            p.data.add_(delta * e, alpha=-step_size)
            if reset:
                e.zero_()

        return step_size

    def reset_traces(self, env_id=None):
        """Reset eligibility traces for a specific environment or all environments."""
        if env_id is not None and env_id in self.env_traces:
            for e in self.env_traces[env_id].values():
                e.zero_()
        elif env_id is None:
            for env_traces in self.env_traces.values():
                for e in env_traces.values():
                    e.zero_()

def sparse_init(tensor, sparsity, init_type='uniform'):
    """Initialize tensor with controlled sparsity."""
    if tensor.ndimension() == 2:
        fan_out, fan_in = tensor.shape
        num_zeros = int(math.ceil(sparsity * fan_in))

        with torch.no_grad():
            if init_type == 'uniform':
                tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
            elif init_type == 'normal':
                tensor.normal_(0, math.sqrt(1.0 / fan_in))
            else:
                raise ValueError("Unknown initialization type")
                
            for col_idx in range(fan_out):
                row_indices = torch.randperm(fan_in)
                zero_indices = row_indices[:num_zeros]
                tensor[col_idx, zero_indices] = 0
        return tensor
    elif tensor.ndimension() == 4:
        channels_out, channels_in, h, w = tensor.shape
        fan_in = channels_in*h*w

        num_zeros = int(math.ceil(sparsity * fan_in))

        with torch.no_grad():
            if init_type == 'uniform':
                tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
            elif init_type == 'normal':
                tensor.normal_(0, math.sqrt(1.0 / fan_in))
            else:
                raise ValueError("Unknown initialization type")
                
            for out_channel_idx in range(channels_out):
                indices = torch.randperm(fan_in)
                zero_indices = indices[:num_zeros]
                tensor[out_channel_idx].reshape(channels_in*h*w)[zero_indices] = 0
        return tensor
    else:
        raise ValueError("Only tensors with 2 or 4 dimensions are supported")

class StreamACAlgorithm(BaseAlgorithm):
    """
    Implementation of Stream-based Actor-Critic algorithm.
    Uses online updates with eligibility traces and adaptive learning rates.
    """
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        action_space_type: str = "discrete",
        action_dim: int = None,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        device: str = "cpu",
        lr_actor: float = 1.0,
        lr_critic: float = 1.0,
        gamma: float = 0.99,
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        adaptive_learning_rate: bool = True,
        target_step_size: float = 0.025,
        backtracking_patience: int = 10,
        backtracking_zeta: float = 0.85,
        min_lr_factor: float = 0.1,
        max_lr_factor: float = 10.0,
        use_obgd: bool = True,
        buffer_size: int = 32,
        use_sparse_init: bool = True,
        update_freq: int = 1,
        **kwargs
    ):
        # Set StreamAC specific parameters before calling the parent initialization
        self.action_bounds = action_bounds
        self.adaptive_learning_rate = adaptive_learning_rate
        self.target_step_size = target_step_size
        self.backtracking_patience = backtracking_patience
        self.backtracking_zeta = backtracking_zeta
        self.min_lr_factor = min_lr_factor
        self.max_lr_factor = max_lr_factor
        self.use_obgd = use_obgd  # Set this before parent initialization
        self.buffer_size = buffer_size
        self.update_freq = update_freq
        self.use_sparse_init = use_sparse_init  # Set this before parent initialization
        self.lamda = kwargs.get('lamda', 0.8)
        self.kappa_actor = kwargs.get('kappa_actor', 3.0)
        self.kappa_critic = kwargs.get('kappa_critic', 2.0)
        
        # Initialize the base class
        super().__init__(
            actor=actor,
            critic=critic,
            action_space_type=action_space_type,
            action_dim=action_dim,
            device=device,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            gae_lambda=self.lamda,  # Use lambda as gae_lambda for consistency
            clip_epsilon=0.0,  # Not used in StreamAC
            critic_coef=critic_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            **kwargs
        )
        
        # Add counters for tracking successful updates
        self.successful_steps = 0
        self.last_step_sizes = [0.0]  # Initialize with a default value
        
        # Initialize experience buffer
        self.experience_buffers = {}  # Per-environment buffers
        
        # Metrics for adaptive learning rate
        self.actor_lr_factor = 1.0
        self.critic_lr_factor = 1.0
        self.backtracking_count = 0
        self.effective_step_size_history = []
        
        # Apply sparse initialization if requested
        if self.use_sparse_init:
            self._apply_sparse_init()
            
        # Create optimizers - either ObGD or Adam
        self._init_optimizers()
        
        # Initialize additional metrics
        self.metrics.update({
            'effective_step_size': 0.0,
            'backtracking_count': 0,
            'mean_return': 0.0,
            'td_error_mean': 0.0,  # Add TD error tracking
            'td_error_max': 0.0,   # Add max TD error
            'td_error_min': 0.0    # Add min TD error
        })
        
        # Track episode data
        self.current_episode_rewards_per_env = {}  # Per-environment rewards
        
        # Add episode return tracking
        self.episode_returns = deque(maxlen=100)  # Store last 100 episode returns
        
        # Track TD errors for logging
        self.td_error_buffer = deque(maxlen=1000)  # Store recent TD errors for statistics
        
        # Debug info
        if self.debug:
            print(f"[DEBUG] Initialized StreamAC with adaptive_lr={adaptive_learning_rate}, "
                  f"use_obgd={use_obgd}, buffer_size={buffer_size}")
    
    def _apply_sparse_init(self):
        """Apply sparse initialization to network weights"""
        def init_module(module):
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    sparse_init(param, sparsity=0.5, init_type='uniform')
        
        # Apply to both actor and critic
        self.actor.apply(lambda m: init_module(m) if hasattr(m, 'weight') else None)
        self.critic.apply(lambda m: init_module(m) if hasattr(m, 'weight') else None)
        
        if self.debug:
            print("[DEBUG] Applied sparse initialization to actor and critic networks")
    
    def _init_optimizers(self):
        """Initialize optimizers based on algorithm configuration"""
        if self.use_obgd:
            # Create ObGD optimizers with eligibility traces
            self.actor_optimizer = ObGD(
                self.actor.parameters(), 
                lr=self.lr_actor,
                gamma=self.gamma,
                lamda=self.lamda,
                kappa=self.kappa_actor
            )
            
            self.critic_optimizer = ObGD(
                self.critic.parameters(),
                lr=self.lr_critic,
                gamma=self.gamma,
                lamda=self.lamda,
                kappa=self.kappa_critic
            )
        else:
            # Use standard Adam optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
            
        if self.debug:
            print(f"[DEBUG] Initialized {'ObGD' if self.use_obgd else 'Adam'} optimizers")
    
    def get_action(self, obs, deterministic=False, return_features=False):
        """Get action for a given observation."""
        # Convert observation to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(self.device)
            
        # Ensure observation has batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        # Set models to evaluation mode
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            # Get value estimate
            value = self.critic(obs)
            
            # Get action distribution
            if self.action_space_type == "discrete":
                # For discrete actions
                action_logits = self.actor(obs)
                
                # Add this line to ensure valid probabilities:
                action_probs = F.softmax(action_logits, dim=-1)
                
                dist = Categorical(action_probs)
                
                if deterministic:
                    action = torch.argmax(action_probs, dim=1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                    
            else:
                # For continuous actions
                mu, log_std = self.actor(obs)
                std = torch.exp(log_std)
                dist = Normal(mu, std)
                
                if deterministic:
                    action = mu  # Use mean for deterministic action
                else:
                    action = dist.sample()
                    
                    # Apply action bounds if provided
                    if self.action_bounds:
                        action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])
                        
                log_prob = dist.log_prob(action)
                
                # Sum log probs for each action dimension
                if log_prob.dim() > 1:
                    log_prob = log_prob.sum(dim=1)
        
        # Set models back to training mode
        self.actor.train()
        self.critic.train()
        
        # Extract tensors from device if needed
        action_np = action.cpu().numpy()[0] if action.shape[0] == 1 else action.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()[0] if log_prob.shape[0] == 1 else log_prob.cpu().numpy()
        value_np = value.cpu().numpy()[0] if value.shape[0] == 1 else value.cpu().numpy()
        
        # Return features if requested
        if return_features:
            features = None
            if hasattr(self.actor, 'extract_features'):
                features = self.actor.extract_features(obs)
            elif hasattr(self.actor, 'features'):
                features = self.actor.features
                
            # Add checks for invalid values in logits
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                print("[ERROR] Invalid values (NaN or Inf) detected in actor logits:")
                print(action_logits)
                # Optionally, raise an error or handle it
                raise ValueError("Actor produced invalid logits (NaN or Inf)")
            
            return action_np, log_prob_np, value_np, features
            
        return action_np, log_prob_np, value_np
    
    def update_reward_tracking(self, reward, env_id=0):
        """Update reward tracking for episode return calculation after delay-updating an experience."""
        # Update for specific environment
        if env_id in self.current_episode_rewards_per_env and len(self.current_episode_rewards_per_env[env_id]) > 0:
            # Update the last stored reward (replace placeholder with actual reward)
            self.current_episode_rewards_per_env[env_id][-1] = reward.item() if hasattr(reward, 'item') else reward

    def store_experience(self, obs, action, log_prob, reward, value, done, env_id=0):
        """Store experience and perform online update if needed."""
        # Convert inputs to tensors if needed - Fix slow tensor creation warnings
        if not isinstance(obs, torch.Tensor):
            if isinstance(obs, list):
                obs = np.array(obs, dtype=np.float32)
            # No need to wrap in a list if already a numpy array
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
        
        if not isinstance(action, torch.Tensor):
            if self.action_space_type == "discrete":
                if isinstance(action, (list, np.ndarray)) and len(action) == 1:
                    action = action[0]  # Extract scalar from single-item list/array
                action = torch.tensor(action, dtype=torch.long, device=self.device)
                if action.dim() == 0:  # If scalar, add batch dimension
                    action = action.unsqueeze(0)
            else:
                if isinstance(action, list):
                    action = np.array(action, dtype=np.float32)
                action = torch.tensor(action, dtype=torch.float32, device=self.device)
                if action.dim() == 1:
                    action = action.unsqueeze(0)
                    
        if not isinstance(log_prob, torch.Tensor):
            if isinstance(log_prob, (list, np.ndarray)) and len(log_prob) == 1:
                log_prob = log_prob[0]  # Extract scalar
            log_prob = torch.tensor(log_prob, dtype=torch.float32, device=self.device)
            if log_prob.dim() == 0:  # If scalar, add batch dimension
                log_prob = log_prob.unsqueeze(0)
        
        if not isinstance(reward, torch.Tensor):
            if isinstance(reward, (list, np.ndarray)) and len(reward) == 1:
                reward = reward[0]  # Extract scalar
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            if reward.dim() == 0:  # If scalar, add batch dimension
                reward = reward.unsqueeze(0)
        
        if not isinstance(value, torch.Tensor):
            if isinstance(value, (list, np.ndarray)) and len(value) == 1:
                value = value[0]  # Extract scalar
            value = torch.tensor(value, dtype=torch.float32, device=self.device)
            if value.dim() == 0:  # If scalar, add batch dimension
                value = value.unsqueeze(0)
        
        if not isinstance(done, torch.Tensor):
            if isinstance(done, (list, np.ndarray)) and len(done) == 1:
                done = done[0]  # Extract scalar
            done = torch.tensor(float(done), dtype=torch.float32, device=self.device)
            if done.dim() == 0:  # If scalar, add batch dimension
                done = done.unsqueeze(0)
        
        # Make sure reward and done are float32
        if isinstance(reward, torch.Tensor) and reward.dtype != torch.float32:
            reward = reward.float()
        elif not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
            
        if isinstance(done, torch.Tensor) and done.dtype != torch.float32:
            done = done.float()
        elif not isinstance(done, torch.Tensor):
            done = torch.tensor([done], dtype=torch.float32, device=self.device)
        
        # Create experience dictionary
        experience = {
            'obs': obs.detach(),
            'action': action.detach(),
            'log_prob': log_prob.detach(),
            'reward': reward.detach(),
            'value': value.detach(),
            'done': done.detach()
        }
        
        # Initialize buffers for this environment if they don't exist yet
        if env_id not in self.experience_buffers:
            self.experience_buffers[env_id] = []
            self.current_episode_rewards_per_env[env_id] = []
            
        # Store in episode history for return calculation
        reward_value = reward.item() if hasattr(reward, 'item') else reward
        self.current_episode_rewards_per_env[env_id].append(reward_value)
        
        # Add to experience buffer, keeping only the most recent experiences
        self.experience_buffers[env_id].append(experience)
        if len(self.experience_buffers[env_id]) > self.buffer_size:
            self.experience_buffers[env_id].pop(0)
        
        # Determine if we should update
        did_update = False
        metrics = {}
        
        # Update based on frequency or buffer fullness
        if (len(self.experience_buffers[env_id]) >= self.buffer_size or 
            len(self.experience_buffers[env_id]) % self.update_freq == 0 and len(self.experience_buffers[env_id]) > 1):
            did_update = True
            metrics = self._update_online(env_id=env_id)
            
        # Always update on episode completion
        if done.item() > 0.5:
            did_update = True
            metrics = self._update_online(env_id=env_id, end_of_episode=True)
            
            # Calculate episode return for metrics
            if len(self.current_episode_rewards_per_env[env_id]) > 0:
                episode_return = sum(self.current_episode_rewards_per_env[env_id])
                self.episode_returns.append(episode_return)
                
                # Calculate true mean over multiple episodes
                mean_return = sum(self.episode_returns) / len(self.episode_returns)
                metrics['mean_return'] = mean_return
                
                # Reset episode tracking
                self.current_episode_rewards_per_env[env_id] = []
                
                if self.debug:
                    print(f"[DEBUG] Episode complete, return: {episode_return:.2f}")
        
        # Return metrics and whether we updated
        return metrics, did_update
    
    def _calculate_td_error(self, reward, next_value, done, value):
        """Calculate TD error: r + γV(s') - V(s). Optimized for batch processing."""
        # Convert all inputs to float32 tensors if they aren't already
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        elif reward.dtype != torch.float32:
            reward = reward.float()
        
        if not isinstance(next_value, torch.Tensor):
            next_value = torch.tensor(next_value, dtype=torch.float32, device=self.device)
        elif next_value.dtype != torch.float32:
            next_value = next_value.float()
        
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, dtype=torch.float32, device=self.device)
        elif done.dtype != torch.float32:
            done = done.float()
        
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=self.device)
        elif value.dtype != torch.float32:
            value = value.float()
        
        # Ensure all tensors have the same shape
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        if next_value.dim() == 0:
            next_value = next_value.unsqueeze(0)
        if done.dim() == 0:
            done = done.unsqueeze(0)
        if value.dim() == 0:
            value = value.unsqueeze(0)
        
        # done mask: 0 if done, 1 otherwise
        done_mask = 1.0 - done
        
        # TD target: r + γV(s') if not done, otherwise just r
        td_target = reward + self.gamma * next_value * done_mask
        
        # TD error
        delta = td_target - value
        
        # Store TD errors for metrics
        if not delta.requires_grad:
            # Convert to numpy and add to buffer for metrics
            delta_np = delta.cpu().numpy()
            for d in delta_np:
                self.td_error_buffer.append(float(d))
            
            # Update TD error metrics
            if len(self.td_error_buffer) > 0:
                self.metrics['td_error_mean'] = float(np.mean(self.td_error_buffer))
                self.metrics['td_error_max'] = float(np.max(self.td_error_buffer))
                self.metrics['td_error_min'] = float(np.min(self.td_error_buffer))
        
        return delta, td_target

    def _update_online(self, env_id=0, end_of_episode=False):
        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'effective_step_size': 0.0,
            'backtracking_count': self.backtracking_count,
            'sr_loss': 0.0,
            'rp_loss': 0.0
        }
        
        # Nothing to update if buffer is empty
        if env_id not in self.experience_buffers or len(self.experience_buffers[env_id]) <= 1:
            return metrics
        
        # Process experiences in order
        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        
        # For OBGD, we can use mini-batches to speed up forward passes
        if self.use_obgd:
            # Define mini-batch size - too large might harm sample efficiency, too small is inefficient
            mini_batch_size = min(8, len(self.experience_buffers[env_id]) - 1)
            
            for i in range(0, len(self.experience_buffers[env_id]) - 1, mini_batch_size):
                # Get batch of experiences
                batch_indices = range(i, min(i + mini_batch_size, len(self.experience_buffers[env_id]) - 1))
                
                # Pre-allocate arrays for batch data
                batch_obs = []
                batch_actions = []
                batch_rewards = []
                batch_values = []
                batch_dones = []
                batch_next_values = []
                
                # Collect batch data
                for idx in batch_indices:
                    current_exp = self.experience_buffers[env_id][idx]
                    next_exp = self.experience_buffers[env_id][idx + 1]
                    
                    batch_obs.append(current_exp['obs'])
                    batch_actions.append(current_exp['action'])
                    batch_rewards.append(current_exp['reward'])
                    batch_values.append(current_exp['value'])
                    batch_dones.append(current_exp['done'])
                    batch_next_values.append(next_exp['value'])
                
                # Stack tensors if all are same shape, otherwise process individually
                try:
                    # Try to stack tensors for batch processing
                    batch_obs = torch.cat(batch_obs, dim=0)
                    
                    # Handle discrete vs continuous actions differently
                    if self.action_space_type == "discrete":
                        batch_actions = torch.cat(batch_actions, dim=0).long()
                    else:
                        batch_actions = torch.cat(batch_actions, dim=0).float()
                        
                    batch_rewards = torch.cat(batch_rewards, dim=0)
                    batch_values = torch.cat(batch_values, dim=0)
                    batch_dones = torch.cat(batch_dones, dim=0)
                    batch_next_values = torch.cat(batch_next_values, dim=0)
                    
                    # Calculate TD errors for batch
                    batch_deltas, batch_td_targets = self._calculate_td_error(
                        batch_rewards, batch_next_values, batch_dones, batch_values)
                    
                    # Forward pass (batched)
                    self.actor.train()
                    self.critic.train()
                    
                    if self.action_space_type == "discrete":
                        # Discrete actions
                        action_probs = self.actor(batch_obs)
                        dist = Categorical(action_probs)
                        log_probs = dist.log_prob(batch_actions.squeeze())
                        entropies = dist.entropy()
                    else:
                        # Continuous actions
                        mu, log_std = self.actor(batch_obs)
                        std = torch.exp(log_std)
                        dist = Normal(mu, std)
                        log_probs = dist.log_prob(batch_actions).sum(-1)
                        entropies = dist.entropy().sum(-1)
                    
                    # Compute critic outputs (batched)
                    current_values = self.critic(batch_obs)
                    
                    # Now we need to process each experience individually for OBGD
                    for j in range(len(batch_indices)):
                        # Get individual components from the batch
                        idx = batch_indices[j]
                        delta = batch_deltas[j]
                        td_target = batch_td_targets[j]
                        log_prob = log_probs[j]
                        entropy = entropies[j]
                        current_value = current_values[j]
                        done = batch_dones[j]
                        
                        # Compute individual losses
                        current_value_flat = current_value.reshape(-1)
                        td_target_flat = td_target.reshape(-1)
                        value_loss = F.mse_loss(current_value_flat, td_target_flat.detach())
                        policy_loss = -log_prob.mean()
                        entropy_loss = -entropy.mean() * self.entropy_coef * torch.sign(delta.detach()) if self.entropy_coef != 0 else torch.tensor(0.0, device=self.device)
                        
                        # Zero gradients
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        
                        # Backpropagate losses
                        actor_loss = policy_loss + entropy_loss
                        if actor_loss.requires_grad:
                            actor_loss.backward(retain_graph=True)  # Need retain_graph for batch processing
                        value_loss.backward(retain_graph=(j < len(batch_indices) - 1))  # No retain on last item
                        
                        # Apply OBGD updates
                        delta_value = delta.item()
                        actor_step_size = self.actor_optimizer.step(delta_value, env_id=env_id, reset=done.item() > 0.5)
                        critic_step_size = self.critic_optimizer.step(delta_value, env_id=env_id, reset=done.item() > 0.5)
                        
                        # Track metrics
                        metrics['effective_step_size'] = float((actor_step_size + critic_step_size) / 2.0)
                        self.last_step_sizes.append(metrics['effective_step_size'])
                        if len(self.last_step_sizes) > 100:
                            self.last_step_sizes.pop(0)
                        self.successful_steps += 1
                        
                        # Track losses
                        total_value_loss += value_loss.item()
                        total_policy_loss += policy_loss.item()
                        total_entropy_loss += entropy_loss.item() if hasattr(entropy_loss, 'item') else 0.0
                    
                except (RuntimeError, ValueError) as e:
                    # Fallback to original individual processing if stacking fails
                    for idx in batch_indices:
                        # Get current and next experience
                        current_exp = self.experience_buffers[env_id][idx]
                        next_exp = self.experience_buffers[env_id][idx + 1]
                        
                        # Get experience components and ensure they're all float32
                        obs = current_exp['obs'].float() if hasattr(current_exp['obs'], 'float') else current_exp['obs']
                        action = current_exp['action']
                        reward = current_exp['reward'].float() if hasattr(current_exp['reward'], 'float') else current_exp['reward']
                        value = current_exp['value'].float() if hasattr(current_exp['value'], 'float') else current_exp['value']
                        done = current_exp['done'].float() if hasattr(current_exp['done'], 'float') else current_exp['done']
                        next_value = next_exp['value'].float() if hasattr(next_exp['value'], 'float') else next_exp['value']
                        
                        # For discrete action spaces, ensure action is the right type
                        if self.action_space_type == "discrete" and hasattr(action, 'long'):
                            action = action.long()
                        elif hasattr(action, 'float'):
                            action = action.float()
                            
                        # Calculate TD error
                        delta, td_target = self._calculate_td_error(reward, next_value, done, value)
                        
                        # Train networks
                        self.actor.train()
                        self.critic.train()
                        
                        # Forward pass
                        if self.action_space_type == "discrete":
                            action_probs = self.actor(obs)
                            dist = Categorical(action_probs)
                            log_prob = dist.log_prob(action)
                            entropy = dist.entropy()
                        else:
                            mu, log_std = self.actor(obs)
                            std = torch.exp(log_std)
                            dist = Normal(mu, std)
                            log_prob = dist.log_prob(action).sum(-1)
                            entropy = dist.entropy().sum(-1)
                        
                        # Compute critic output
                        current_value = self.critic(obs)
                        
                        # Compute losses
                        current_value_flat = current_value.reshape(-1)
                        td_target_flat = td_target.reshape(-1)
                        value_loss = F.mse_loss(current_value_flat, td_target_flat.detach())
                        policy_loss = -log_prob.mean()
                        entropy_loss = -entropy.mean() * self.entropy_coef * torch.sign(delta.detach()) if self.entropy_coef != 0 else torch.tensor(0.0, device=self.device)
                        
                        # Zero gradients
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        
                        # Backpropagation - Combined actor loss
                        actor_loss = policy_loss + entropy_loss
                        if actor_loss.requires_grad:
                            actor_loss.backward()
                        value_loss.backward()
                        
                        # Apply updates with OBGD optimizers
                        delta_value = delta.item()
                        actor_step_size = self.actor_optimizer.step(delta_value, env_id=env_id, reset=done.item() > 0.5)
                        critic_step_size = self.critic_optimizer.step(delta_value, env_id=env_id, reset=done.item() > 0.5)
                        
                        # Track metrics
                        metrics['effective_step_size'] = float((actor_step_size + critic_step_size) / 2.0)
                        self.last_step_sizes.append(metrics['effective_step_size'])
                        if len(self.last_step_sizes) > 100:
                            self.last_step_sizes.pop(0)
                        self.successful_steps += 1
                        
                        # Track losses
                        total_value_loss += value_loss.item()
                        total_policy_loss += policy_loss.item()
                        total_entropy_loss += entropy_loss.item() if hasattr(entropy_loss, 'item') else 0.0
        
        else:
            # Batch update with standard optimizers - more efficient implementation
            batch_size = min(64, len(self.experience_buffers[env_id]) - 1)  # Increased batch size
            batch_indices = np.random.choice(len(self.experience_buffers[env_id]) - 1, batch_size, replace=False)
            
            # Pre-allocate lists for batching
            batch_obs = []
            batch_actions = []
            batch_rewards = []
            batch_values = []
            batch_dones = []
            batch_next_values = []
            
            # Collect batch data
            for i in batch_indices:
                current_exp = self.experience_buffers[env_id][i]
                next_exp = self.experience_buffers[env_id][i + 1]
                
                batch_obs.append(current_exp['obs'])
                batch_actions.append(current_exp['action'])
                batch_rewards.append(current_exp['reward'])
                batch_values.append(current_exp['value'])
                batch_dones.append(current_exp['done'])
                batch_next_values.append(next_exp['value'])
            
            # Stack tensors for batch processing
            batch_obs = torch.cat(batch_obs, dim=0)
            batch_rewards = torch.cat(batch_rewards, dim=0)
            batch_values = torch.cat(batch_values, dim=0)
            batch_dones = torch.cat(batch_dones, dim=0)
            batch_next_values = torch.cat(batch_next_values, dim=0)
            
            if self.action_space_type == "discrete":
                batch_actions = torch.cat(batch_actions, dim=0).long()
            else:
                batch_actions = torch.cat(batch_actions, dim=0).float()
            
            # Calculate TD error for the entire batch
            batch_deltas, batch_td_targets = self._calculate_td_error(
                batch_rewards, batch_next_values, batch_dones, batch_values)
            
            # Forward pass (batched)
            if self.action_space_type == "discrete":
                action_probs = self.actor(batch_obs)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions.squeeze())
                entropy = dist.entropy().mean()
            else:
                mu, log_std = self.actor(batch_obs)
                std = torch.exp(log_std)
                dist = Normal(mu, std)
                log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
            
            # Compute critic output (batched)
            current_values = self.critic(batch_obs)
            
            # Compute losses (batched)
            current_value_flat = current_values.reshape(-1)
            td_target_flat = batch_td_targets.reshape(-1)
            value_loss = F.mse_loss(current_value_flat, td_target_flat.detach())
            policy_loss = -log_probs.mean()
            entropy_loss = -entropy * self.entropy_coef
            
            # Compute total loss
            loss = policy_loss + self.critic_coef * value_loss + entropy_loss
            
            # Zero gradients and update
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping if enabled
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            # Apply updates
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            # Track losses for metrics
            total_value_loss = value_loss.item() * batch_size
            total_policy_loss = policy_loss.item() * batch_size
            total_entropy_loss = entropy_loss.item() * batch_size
        
        # Calculate average losses
        num_updates = len(self.experience_buffers[env_id]) - 1 if self.use_obgd else batch_size
        avg_value_loss = total_value_loss / max(1, num_updates)
        avg_policy_loss = total_policy_loss / max(1, num_updates)
        avg_entropy_loss = total_entropy_loss / max(1, num_updates)
        
        # Adaptive learning rate adjustment
        if self.adaptive_learning_rate:
            # Record effective step size
            if metrics['effective_step_size'] > 0:
                self.effective_step_size_history.append(metrics['effective_step_size'])
                
                # Maintain a fixed-length history
                if len(self.effective_step_size_history) > 100:
                    self.effective_step_size_history.pop(0)
                
                # Calculate average effective step size
                avg_effective_step_size = sum(self.effective_step_size_history) / len(self.effective_step_size_history)
                
                # Adjust learning rates if needed
                if avg_effective_step_size < self.target_step_size * 0.8:
                    # Step size too small, increase learning rates
                    self.backtracking_count = 0
                    self.actor_lr_factor = min(self.actor_lr_factor * 1.1, self.max_lr_factor)
                    self.critic_lr_factor = min(self.critic_lr_factor * 1.1, self.max_lr_factor)
                    self._update_learning_rates()
                    
                elif avg_effective_step_size > self.target_step_size * 1.2:
                    # Step size too large, increase backtracking count
                    self.backtracking_count += 1
                    
                    # If backtracking count exceeds patience, reduce learning rates
                    if self.backtracking_count >= self.backtracking_patience:
                        self.actor_lr_factor = max(self.actor_lr_factor * self.backtracking_zeta, self.min_lr_factor)
                        self.critic_lr_factor = max(self.critic_lr_factor * self.backtracking_zeta, self.min_lr_factor)
                        self._update_learning_rates()
                        self.backtracking_count = 0
                        
                        if self.debug:
                            print(f"[DEBUG] Reduced learning rates - actor: {self.lr_actor * self.actor_lr_factor:.6f}, "
                                  f"critic: {self.lr_critic * self.critic_lr_factor:.6f}")
                else:
                    # Step size in target range, reset backtracking count
                    self.backtracking_count = 0
        
        # Update metrics
        metrics.update({
            'actor_loss': avg_policy_loss,
            'critic_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_policy_loss + avg_value_loss + avg_entropy_loss,
            'backtracking_count': self.backtracking_count,
            'actor_lr': self.lr_actor * self.actor_lr_factor,
            'critic_lr': self.lr_critic * self.critic_lr_factor
        })
        
        # Update the main metrics dictionary
        self.metrics.update(metrics)
        
        # Update auxiliary tasks if available
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'use_auxiliary_tasks') and self.trainer.use_auxiliary_tasks:
            if hasattr(self.trainer, 'aux_task_manager'):
                # Compute auxiliary losses and update models
                aux_losses = self.trainer.aux_task_manager.compute_losses()
                if aux_losses:
                    metrics.update(aux_losses)
                    # Also update in main metrics
                    self.metrics.update(aux_losses)
                    
                    if self.debug:
                        print(f"[DEBUG StreamAC] Auxiliary task losses - SR: {aux_losses.get('sr_loss', 0):.6f}, RP: {aux_losses.get('rp_loss', 0):.6f}")
        
        # Clear buffer if requested
        if end_of_episode:
            self.experience_buffers[env_id] = []
            
        return metrics
    
    def _update_learning_rates(self):
        """Update learning rates based on current factors"""
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.lr_actor * self.actor_lr_factor
            
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.lr_critic * self.critic_lr_factor
    
    def update(self):
        """Perform a manual update if needed."""
        # If we have enough experiences, do a full update
        metrics = {}
        
        if len(self.experience_buffers) > 1:
            metrics = self._update_online(end_of_episode=True)
            
        # Include the latest metrics
        metrics.update(self.metrics)
        
        return metrics
    
    def reset(self, env_id=None):
        """Reset the algorithm state at the end of an episode"""
        # Reset eligibility traces if using OBGD
        if self.use_obgd:
            if env_id is not None:
                self.actor_optimizer.reset_traces(env_id)
                self.critic_optimizer.reset_traces(env_id)
                self.current_episode_rewards_per_env[env_id] = []
                self.experience_buffers[env_id] = []
            else:
                self.actor_optimizer.reset_traces()
                self.critic_optimizer.reset_traces()
                self.current_episode_rewards_per_env = {}
                self.experience_buffers = {}
        
        # Clear episode tracking data
        self.current_episode_rewards = []
        self.current_episode_values = []
        
        # Clear experience buffer
        self.experience_buffer = []