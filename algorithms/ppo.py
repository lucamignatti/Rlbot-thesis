import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import math
from typing import Dict, Tuple, List, Optional, Union, Any
from .base import BaseAlgorithm

class PPOAlgorithm(BaseAlgorithm):
    """PPO algorithm implementation"""

    def __init__(
        self,
        actor,
        critic,
        action_space_type="discrete",
        action_dim=None,
        action_bounds=(-1.0, 1.0),
        device="cuda",
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        critic_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        batch_size=64,
        use_amp=False,
        debug=False,
        use_wandb=False,
        # --- Weight Clipping Params ---
        # Unsupported using SimBa v2
        use_weight_clipping=False,
        weight_clip_kappa=1.0,
        adaptive_kappa=False,
        kappa_update_freq=10,
        kappa_update_rate=0.01,
        target_clip_fraction=0.05,
        min_kappa=0.1,
        max_kappa=10.0,
        # ---------------------------
    ):
        super().__init__(
            actor,
            critic,
            action_space_type=action_space_type,
            action_dim=action_dim,
            action_bounds=action_bounds,
            device=device,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            critic_coef=critic_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            use_amp=use_amp,
            debug=debug,
            use_wandb=use_wandb,
        )

        # Weight clipping parameters
        self.use_weight_clipping = use_weight_clipping
        self.weight_clip_kappa = weight_clip_kappa
        self.adaptive_kappa = adaptive_kappa
        self.kappa_update_freq = kappa_update_freq
        self.kappa_update_rate = kappa_update_rate
        self.target_clip_fraction = target_clip_fraction
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa
        self._update_counter = 0 # Counter for kappa updates

        # Store initial weight ranges if weight clipping is enabled
        if self.use_weight_clipping:
            self.init_weight_ranges()

        # Initialize memory with buffer size and device
        self.memory = self.PPOMemory(batch_size=batch_size, buffer_size=10000, device=device)

        # Initialize optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        # Combined optimizer for single backward pass
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr_actor
        )

        # Tracking metrics
        self.metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'clip_fraction': 0.0,
            'explained_variance': 0.0,
            'kl_divergence': 0.0,
            'mean_advantage': 0.0,
            'mean_return': 0.0,
            'weight_clip_fraction': 0.0,
            'current_kappa': self.weight_clip_kappa,
        }

        # Add episode return tracking
        self.current_episode_rewards = []
        self.episode_returns = deque(maxlen=100)

    class PPOMemory:
        """Memory buffer for PPO to store experiences"""

        def __init__(self, batch_size, buffer_size, device):
            self.batch_size = batch_size

            self.buffer_size = buffer_size
            self.device = device
            self.pos = 0
            self.size = 0
            self.full = False
            self.use_device_tensors = device != "cpu"

            # Initialize buffers as empty tensors
            self._reset_buffers()

        def _reset_buffers(self):
            """Initialize all buffer tensors with the correct shapes"""
            buffer_size = self.buffer_size
            device = self.device
            use_device_tensors = self.use_device_tensors

            # Determine tensor type based on device
            if use_device_tensors:
                # Initialize empty tensors on the specified device
                self.obs = None  # Will be initialized on first store() call
                self.actions = None  # Will be initialized on first store() call
                self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
                self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
                self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
                self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
            else:
                # Initialize empty numpy arrays for CPU
                self.obs = None  # Will be initialized on first store() call
                self.actions = None  # Will be initialized on first store() call
                self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
                self.rewards = np.zeros((buffer_size,), dtype=np.float32)
                self.values = np.zeros((buffer_size,), dtype=np.float32)
                self.dones = np.zeros((buffer_size,), dtype=np.bool_)

            # Reset position and full indicator
            self.pos = 0
            self.full = False
            self.size = 0

        def store(self, obs, action, log_prob, reward, value, done):
            """Store a single experience in the buffer"""
            # Initialize buffers on first call based on sample shapes
            if self.obs is None:
                # Initialize tensors
                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs, dtype=torch.float32)
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32)

                # Ensure all tensors have batch dimension
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                if action.dim() == 1 and action.shape[0] > 1:  # If it's a vector action
                    action = action.unsqueeze(0)

                # Initialize buffers with correct shapes
                if self.use_device_tensors:
                    # Store tensors directly on the target device
                    self.obs = torch.zeros((self.buffer_size, *obs.shape[1:]), dtype=torch.float32, device=self.device)
                    self.actions = torch.zeros((self.buffer_size, *action.shape[1:]), dtype=torch.float32, device=self.device)
                else:
                    # Store tensors on CPU
                    self.obs = torch.zeros((self.buffer_size, *obs.shape[1:]), dtype=torch.float32)
                    self.actions = torch.zeros((self.buffer_size, *action.shape[1:]), dtype=torch.float32)

            # Convert inputs to tensors
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device if self.use_device_tensors else "cpu")
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=self.device if self.use_device_tensors else "cpu")
            if not isinstance(log_prob, torch.Tensor):
                log_prob = torch.tensor(log_prob, dtype=torch.float32, device=self.device if self.use_device_tensors else "cpu")
            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device if self.use_device_tensors else "cpu")
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32, device=self.device if self.use_device_tensors else "cpu")
            if not isinstance(done, torch.Tensor):
                done = torch.tensor(done, dtype=torch.bool, device=self.device if self.use_device_tensors else "cpu")

            # Ensure tensors are on the correct device
            if self.use_device_tensors:
                obs = obs.to(self.device)
                action = action.to(self.device)
                log_prob = log_prob.to(self.device)
                reward = reward.to(self.device)
                value = value.to(self.device)
                done = done.to(self.device)

            # Store the experience, detaching obs and action
            if obs.dim() > 1:  # If obs is batched
                self.obs[self.pos] = obs.squeeze(0).detach()
            else:
                self.obs[self.pos] = obs.detach()

            if action.dim() > 1:  # If action is batched
                self.actions[self.pos] = action.squeeze(0).detach()
            else:
                self.actions[self.pos] = action.detach()

            if log_prob.dim() > 0:  # If log_prob has dimensions
                self.log_probs[self.pos] = log_prob.item()
            else:
                self.log_probs[self.pos] = log_prob

            if reward.dim() > 0:  # If reward has dimensions
                self.rewards[self.pos] = reward.item()
            else:
                self.rewards[self.pos] = reward

            if value.dim() > 0:  # If value has dimensions
                self.values[self.pos] = value.item()
            else:
                self.values[self.pos] = value

            if done.dim() > 0:  # If done has dimensions
                self.dones[self.pos] = done.item()
            else:
                self.dones[self.pos] = done

            # Update position and full flag
            self.pos = (self.pos + 1) % self.buffer_size
            if not self.full and self.pos == 0:
                self.full = True
            self.size = self.buffer_size if self.full else self.pos

        def store_experience_at_idx(self, idx, state=None, action=None, log_prob=None, reward=None, value=None, done=None):
            """Update only specific values at an index, rather than a complete experience."""
            if idx >= self.buffer_size:
                return  # Index out of range

            # Only update the specified fields (non-None values)
            if state is not None and self.obs is not None:
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32, device=self.device if self.use_device_tensors else "cpu")
                # Detach state before storing
                state_detached = state.detach()
                if state_detached.dim() > 1:  # If state is batched
                    self.obs[idx] = state_detached.squeeze(0).to(self.device) if self.use_device_tensors else state_detached.squeeze(0)
                else:
                    self.obs[idx] = state_detached.to(self.device) if self.use_device_tensors else state_detached

            if action is not None and self.actions is not None:
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32, device=self.device if self.use_device_tensors else "cpu")
                # Detach action before storing
                action_detached = action.detach()
                if action_detached.dim() > 1:  # If action is batched
                    self.actions[idx] = action_detached.squeeze(0).to(self.device) if self.use_device_tensors else action_detached.squeeze(0)
                else:
                    self.actions[idx] = action_detached.to(self.device) if self.use_device_tensors else action_detached

            if log_prob is not None and self.log_probs is not None:
                if not isinstance(log_prob, torch.Tensor):
                    log_prob = torch.tensor(log_prob, dtype=torch.float32, device=self.device if self.use_device_tensors else "cpu")
                if log_prob.dim() > 0:  # If log_prob has dimensions
                    self.log_probs[idx] = log_prob.item()
                else:
                    self.log_probs[idx] = log_prob

            if reward is not None and self.rewards is not None:
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, dtype=torch.float32, device=self.device if self.use_device_tensors else "cpu")
                if reward.dim() > 0:  # If reward has dimensions
                    self.rewards[idx] = reward.item()
                else:
                    self.rewards[idx] = reward

            if value is not None and self.values is not None:
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32, device=self.device if self.use_device_tensors else "cpu")
                if value.dim() > 0:  # If value has dimensions
                    self.values[idx] = value.item()
                else:
                    self.values[idx] = value

            if done is not None and self.dones is not None:
                if not isinstance(done, torch.Tensor):
                    done = torch.tensor(done, dtype=torch.bool, device=self.device if self.use_device_tensors else "cpu")
                if done.dim() > 0:  # If done has dimensions
                    self.dones[idx] = done.item()
                else:
                    self.dones[idx] = done

        def get(self):
            """Get all data currently stored in the buffer."""
            if self.size == 0 or self.obs is None:
                return None, None, None, None, None, None

            return (
                self.obs[:self.size],
                self.actions[:self.size],
                self.log_probs[:self.size],
                self.rewards[:self.size],
                self.values[:self.size],
                self.dones[:self.size]
            )

        def generate_batches(self):
            """Generate batches of indices for training."""
            if self.size == 0:
                return []

            # Generate random indices
            if self.use_device_tensors:
                indices = torch.randperm(self.size, device=self.device)
            else:
                indices = np.random.permutation(self.size)

            # Create batches
            batch_start = 0
            batches = []
            while batch_start < self.size:
                batch_end = min(batch_start + self.batch_size, self.size)
                batch_idx = indices[batch_start:batch_end]
                batches.append(batch_idx)
                batch_start = batch_end

            return batches

        def clear(self):
            """Reset the buffer."""
            self._reset_buffers()

    def store_experience(self, obs, action, log_prob, reward, value, done):
        """Store experience in the buffer"""
        if self.debug:
            print(f"[DEBUG PPO] Storing experience - reward: {reward}")

        # Forward to memory buffer
        self.memory.store(obs, action, log_prob, reward, value, done)

        # Track episode rewards for calculating returns
        if isinstance(reward, torch.Tensor):
            reward_val = reward.item()
        else:
            reward_val = float(reward)

        self.current_episode_rewards.append(reward_val)

        # When episode is done, calculate return
        if done:
            if self.current_episode_rewards:
                episode_return = sum(self.current_episode_rewards)
                self.episode_returns.append(episode_return)
                if self.debug:
                    print(f"[DEBUG PPO] Episode done with return: {episode_return}")
                self.current_episode_rewards = []  # Reset for next episode

    def store_experience_at_idx(self, idx, obs=None, action=None, log_prob=None, reward=None, value=None, done=None):
        """Update specific values of an experience at a given index"""
        if self.debug and reward is not None:
            print(f"[DEBUG PPO] Updating reward at idx {idx}: {reward}")

        # Forward to memory buffer
        self.memory.store_experience_at_idx(idx, obs, action, log_prob, reward, value, done)

    def get_action(self, obs, deterministic=False, return_features=False):
        """Get action for a given observation"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if return_features:
                action_dist, actor_features = self.actor(obs, return_features=True)
                value, critic_features = self.critic(obs, return_features=True)
                features = torch.cat([actor_features, critic_features], dim=-1) if hasattr(self, 'hidden_dim') else actor_features
            else:
                action_dist = self.actor(obs)
                value = self.critic(obs)

            if deterministic:
                if self.action_space_type == "discrete":
                    probs = action_dist
                    action = torch.argmax(probs, dim=-1)
                    action_one_hot = torch.zeros_like(probs)
                    action_one_hot.scatter_(-1, action.unsqueeze(-1), 1)
                    log_prob = torch.log(torch.sum(probs * action_one_hot, dim=-1) + 1e-10)
                else:
                    action = action_dist.loc
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
            else:
                if self.action_space_type == "discrete":
                    probs = action_dist
                    probs = torch.clamp(probs, 1e-10, 1.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    action_dist = torch.distributions.Categorical(probs=probs)
                    action = action_dist.sample()
                    action_one_hot = torch.zeros_like(probs)
                    action_one_hot.scatter_(-1, action.unsqueeze(-1), 1)
                    action = action_one_hot
                    log_prob = action_dist.log_prob(torch.argmax(action, dim=-1))
                else:
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum(dim=-1)

        if value.dim() < action.dim():
            value = value.unsqueeze(-1)

        # Only move to CPU if using MPS (Apple Silicon)
        if self.device == "mps":
            action = action.cpu()
            log_prob = log_prob.cpu()
            value = value.cpu()
            if return_features:
                features = features.cpu()

        if return_features:
            return action, log_prob, value, features
        else:
            return action, log_prob, value

    def reset(self):
        """Reset memory"""
        self.memory.clear()

    def update(self):
        """Update policy using PPO"""
        buffer_size = self.memory.size

        if buffer_size == 0:
            if self.debug:
                print("[DEBUG] Buffer is empty, skipping update")
            return self.metrics

        # Get experiences from buffer
        states, actions, old_log_probs, rewards, values, dones = self.memory.get()

        if states is None:
            if self.debug:
                print("[DEBUG] Failed to get experiences from buffer")
            return self.metrics

        # Convert to tensors if not already
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        if not isinstance(old_log_probs, torch.Tensor):
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32, device=self.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, values, dones)

        # Check if advantages has NaNs
        if torch.isnan(advantages).any():
            if self.debug:
                print("[DEBUG] NaN detected in advantages, skipping update")
            # Clear memory even if update is skipped due to NaNs
            self.memory.clear()
            return self.metrics

        # Update the policy and value networks using PPO
        metrics = self._update_policy(states, actions, old_log_probs, returns, advantages)

        # Clear the memory buffer after using the data for updates
        self.memory.clear()

        # Update the metrics
        self.metrics = {**self.metrics, **metrics}

        # If we have episode returns, update the mean return metric
        if len(self.episode_returns) > 0:
            self.metrics['mean_return'] = sum(self.episode_returns) / len(self.episode_returns)

        # Increment update counter for adaptive kappa
        self._update_counter += 1

        return self.metrics

    def _compute_gae(self, rewards, values, dones):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE)

        Args:
            rewards: rewards tensor [buffer_size]
            values: value predictions tensor [buffer_size]
            dones: done flags tensor [buffer_size]

        Returns:
            tuple of (returns, advantages) tensors
        """
        buffer_size = len(rewards)

        # Initialize return and advantage arrays
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        # Last value is 0 if trajectory ended, otherwise use the value prediction
        # Get the last index that's valid
        last_idx = buffer_size - 1
        next_value = 0 if dones[last_idx] else values[last_idx]
        next_advantage = 0

        # Compute GAE for each timestep, going backwards
        for t in reversed(range(buffer_size)):
            # If this is the end of an episode, next_value is 0
            if t < buffer_size - 1:
                next_value = 0 if dones[t] else values[t + 1]

            # Calculate TD error: r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]

            # Compute GAE advantage
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * (1 - dones[t].float())
            next_advantage = advantages[t]

            # Compute returns as advantage + value
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def init_weight_ranges(self):
        """Store the BASE initialization ranges (kappa=1) of all network parameters"""
        self.actor_base_bounds = {}
        self.critic_base_bounds = {}

        # Store base bounds for actor network weights
        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                # Use the He initialization range for the parameter (kappa=1)
                if 'weight' in name:
                    fan_in = param.shape[1] * (param.shape[2] if len(param.shape) > 2 else 1)
                    bound = (6 / fan_in) ** 0.5 # Base bound (kappa=1)
                    self.actor_base_bounds[name] = (-bound, bound)
                else:
                    # For biases, use a smaller base range
                    bound = 0.01 # Base bound (kappa=1)
                    self.actor_base_bounds[name] = (-bound, bound)

        # Store base bounds for critic network weights
        for name, param in self.critic.named_parameters():
            if param.requires_grad:
                if 'weight' in name:
                    fan_in = param.shape[1] * (param.shape[2] if len(param.shape) > 2 else 1)
                    bound = (6 / fan_in) ** 0.5 # Base bound (kappa=1)
                    self.critic_base_bounds[name] = (-bound, bound)
                else:
                    bound = 0.01 # Base bound (kappa=1)
                    self.critic_base_bounds[name] = (-bound, bound)

    def clip_weights(self):
        """Clip weights based on current kappa and return clipped fraction."""
        if not self.use_weight_clipping:
            return 0.0

        total_params = 0
        clipped_params = 0
        current_kappa = self.weight_clip_kappa # Use the current kappa value

        # Clip actor network weights
        for name, param in self.actor.named_parameters():
            if name in self.actor_base_bounds and param.requires_grad:
                base_lower, base_upper = self.actor_base_bounds[name]
                lower_bound = base_lower * current_kappa
                upper_bound = base_upper * current_kappa

                # Store original values to check for clipping
                original_param = param.data.clone()
                param.data.clamp_(lower_bound, upper_bound)

                # Count clipped parameters
                num_params = param.numel()
                total_params += num_params
                clipped_params += torch.sum(param.data != original_param).item()

        # Clip critic network weights
        for name, param in self.critic.named_parameters():
            if name in self.critic_base_bounds and param.requires_grad:
                base_lower, base_upper = self.critic_base_bounds[name]
                lower_bound = base_lower * current_kappa
                upper_bound = base_upper * current_kappa

                # Store original values to check for clipping
                original_param = param.data.clone()
                param.data.clamp_(lower_bound, upper_bound)

                # Count clipped parameters
                num_params = param.numel()
                total_params += num_params
                clipped_params += torch.sum(param.data != original_param).item()

        # Return the fraction of clipped parameters
        return float(clipped_params) / total_params if total_params > 0 else 0.0


    def _update_policy(self, states, actions, old_log_probs, returns, advantages):
        """
        Update policy and value networks using PPO algorithm

        Args:
            states: batch of states [buffer_size, state_dim]
            actions: batch of actions [buffer_size, action_dim]
            old_log_probs: batch of log probabilities from old policy [buffer_size]
            returns: batch of returns [buffer_size]
            advantages: batch of advantages [buffer_size]

        Returns:
            dict: metrics from the update
        """
        # Track metrics for this update cycle
        update_metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'clip_fraction': 0.0, # PPO policy clip fraction
            'weight_clip_fraction': 0.0, # Fraction of weights clipped this update
        }

        # Calculate explained variance (once before updates)
        with torch.no_grad():
            y_pred = self.critic(states).squeeze()
        y_true = returns
        var_y = torch.var(y_true)
        explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
        update_metrics['explained_variance'] = explained_var.item()
        update_metrics['mean_advantage'] = advantages.mean().item()
        update_metrics['kl_divergence'] = 0.0 # Will be averaged later

        total_weight_clipped_fraction_epoch = 0.0
        num_batches_processed = 0

        # Multiple epochs of PPO update
        for epoch in range(self.ppo_epochs):
            # Generate random batches
            batch_indices = self.memory.generate_batches()

            # Skip if no batches
            if not batch_indices:
                continue

            # Process each batch
            for batch_idx in batch_indices:
                num_batches_processed += 1
                # Get batch data
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Get current policy distribution and values
                if self.action_space_type == "discrete":
                    # For discrete actions
                    action_probs = self.actor(batch_states)
                    action_probs = torch.clamp(action_probs, 1e-10, 1.0)
                    # Ensure probabilities sum to 1
                    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                    dist = torch.distributions.Categorical(probs=action_probs)

                    # Calculate log probabilities and entropy
                    # Ensure batch_actions are indices if they are one-hot
                    if batch_actions.shape[-1] > 1 and batch_actions.dtype == torch.float32:
                         actions_indices = torch.argmax(batch_actions, dim=-1)
                    else: # Assume actions are already indices
                         actions_indices = batch_actions.long()

                    curr_log_probs = dist.log_prob(actions_indices)
                    entropy = dist.entropy().mean()
                else:
                    # For continuous actions
                    action_dist = self.actor(batch_states)
                    curr_log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = action_dist.entropy().mean()

                # Calculate critic values
                values = self.critic(batch_states).squeeze()

                # Calculate ratio and surrogates for PPO
                ratio = torch.exp(curr_log_probs - batch_old_log_probs)

                # Clipped surrogate function
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Calculate critic loss (MSE)
                critic_loss = F.mse_loss(values, batch_returns)

                # Calculate entropy loss
                entropy_loss = -entropy * self.entropy_coef

                # Total loss
                total_loss = actor_loss + self.critic_coef * critic_loss + entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()

                # Clip gradients
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.optimizer.step()

                # Apply weight clipping after optimization step and get clipped fraction
                current_weight_clip_fraction = self.clip_weights()
                total_weight_clipped_fraction_epoch += current_weight_clip_fraction

                # Update metrics (accumulate)
                update_metrics['actor_loss'] += actor_loss.detach().item()
                update_metrics['critic_loss'] += critic_loss.detach().item()
                update_metrics['entropy_loss'] += entropy_loss.detach().item()
                update_metrics['total_loss'] += total_loss.detach().item()

                # Calculate PPO policy clipping fraction
                policy_clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().detach().item()
                update_metrics['clip_fraction'] += policy_clip_fraction

                # Calculate KL divergence (approximate)
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - curr_log_probs.detach()).mean().detach().item()
                    update_metrics['kl_divergence'] += kl_div

        # Calculate averages over all batches and epochs
        if num_batches_processed > 0:
            update_metrics['actor_loss'] /= num_batches_processed
            update_metrics['critic_loss'] /= num_batches_processed
            update_metrics['entropy_loss'] /= num_batches_processed
            update_metrics['total_loss'] /= num_batches_processed
            update_metrics['clip_fraction'] /= num_batches_processed
            update_metrics['kl_divergence'] /= num_batches_processed
            update_metrics['weight_clip_fraction'] = total_weight_clipped_fraction_epoch / num_batches_processed

        # --- Adaptive Kappa Update Logic ---
        if self.use_weight_clipping and self.adaptive_kappa and (self._update_counter % self.kappa_update_freq == 0):
            actual_clip_fraction = update_metrics['weight_clip_fraction']
            if actual_clip_fraction > self.target_clip_fraction:
                self.weight_clip_kappa *= (1 + self.kappa_update_rate)
            elif actual_clip_fraction < self.target_clip_fraction:
                self.weight_clip_kappa *= (1 - self.kappa_update_rate)

            # Clamp kappa within bounds
            self.weight_clip_kappa = max(self.min_kappa, min(self.max_kappa, self.weight_clip_kappa))

            if self.debug:
                print(f"[DEBUG PPO] Kappa updated. New kappa: {self.weight_clip_kappa:.4f} (Clip fraction: {actual_clip_fraction:.4f})")

        # Store current kappa value in metrics
        update_metrics['current_kappa'] = self.weight_clip_kappa

        return update_metrics
