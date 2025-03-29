import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import math
from typing import Dict, Tuple, List, Optional, Union, Any
from .base import BaseAlgorithm

class PPOAlgorithm(BaseAlgorithm):
    """PPO algorithm implementation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # PPO-specific attributes
        self.buffer_size = kwargs.get('buffer_size', 10000)
        
        # Initialize experience buffer
        self.memory = self.PPOMemory(self.buffer_size, self.device)
        
        # Initialize GradScaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        # Add episode return tracking
        self.current_episode_rewards = []
        self.episode_returns = deque(maxlen=100)  # Store last 100 episode returns
    
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
            self.use_device_tensors = device != "cpu"
            
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
                # Initialize tensors
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
                if self.use_device_tensors:
                    # Store tensors directly on the target device
                    self.obs = torch.zeros((self.buffer_size, *obs.shape[1:]), dtype=torch.float32, device=self.device)
                    self.actions = torch.zeros((self.buffer_size, *action.shape[1:]), dtype=torch.float32, device=self.device)
                    self.log_probs = torch.zeros((self.buffer_size, *log_prob.shape[1:]), dtype=torch.float32, device=self.device)
                    self.rewards = torch.zeros((self.buffer_size, *reward.shape[1:]), dtype=torch.float32, device=self.device)
                    self.values = torch.zeros((self.buffer_size, *value.shape[1:]), dtype=torch.float32, device=self.device)
                    self.dones = torch.zeros((self.buffer_size, *done.shape[1:]), dtype=torch.bool, device=self.device)
                else:
                    # Store tensors on CPU
                    self.obs = torch.zeros((self.buffer_size, *obs.shape[1:]), dtype=torch.float32)
                    self.actions = torch.zeros((self.buffer_size, *action.shape[1:]), dtype=torch.float32)
                    self.log_probs = torch.zeros((self.buffer_size, *log_prob.shape[1:]), dtype=torch.float32)
                    self.rewards = torch.zeros((self.buffer_size, *reward.shape[1:]), dtype=torch.float32)
                    self.values = torch.zeros((self.buffer_size, *value.shape[1:]), dtype=torch.float32)
                    self.dones = torch.zeros((self.buffer_size, *done.shape[1:]), dtype=torch.bool)
            
            # Convert inputs to tensors and move to device
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
            
            # Store the experience
            if obs.dim() < self.obs[0].dim() + 1:
                self.obs[self.pos] = obs
            else:
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
        
        def store_experience_at_idx(self, idx, state=None, action=None, log_prob=None, reward=None, value=None, done=None):
            """Update only specific values at an index, rather than a complete experience."""
            if idx >= self.buffer_size:
                return  # Index out of range
                
            # Only update the specified fields (non-None values)
            if state is not None and self.obs is not None:
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                if state.dim() < self.obs[0].dim() + 1:
                    self.obs[idx] = state.to(self.device)
                else:
                    self.obs[idx] = state.squeeze(0).to(self.device)
                    
            if action is not None and self.actions is not None:
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32, device=self.device)
                if action.dim() < self.actions[0].dim() + 1:
                    self.actions[idx] = action.to(self.device)
                else:
                    self.actions[idx] = action.squeeze(0).to(self.device)
                    
            if log_prob is not None and self.log_probs is not None:
                if not isinstance(log_prob, torch.Tensor):
                    log_prob = torch.tensor(log_prob, dtype=torch.float32, device=self.device)
                if log_prob.dim() < self.log_probs[0].dim() + 1:
                    self.log_probs[idx] = log_prob.to(self.device)
                else:
                    self.log_probs[idx] = log_prob.squeeze(0).to(self.device)
                    
            if reward is not None and self.rewards is not None:
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
                if reward.dim() < self.rewards[0].dim() + 1:
                    self.rewards[idx] = reward.to(self.device)
                else:
                    self.rewards[idx] = reward.squeeze(0).to(self.device)
                    
            if value is not None and self.values is not None:
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32, device=self.device)
                if value.dim() < self.values[0].dim() + 1:
                    self.values[idx] = value.to(self.device)
                else:
                    self.values[idx] = value.squeeze(0).to(self.device)
                    
            if done is not None and self.dones is not None:
                if not isinstance(done, torch.Tensor):
                    done = torch.tensor(done, dtype=torch.bool, device=self.device)
                if done.dim() < self.dones[0].dim() + 1:
                    self.dones[idx] = done.to(self.device)
                else:
                    self.dones[idx] = done.squeeze(0).to(self.device)
        
        def get_generator(self, batch_size, compute_returns=True, gamma=0.99, gae_lambda=0.95):
            """Get a generator that yields batches of experiences"""
            if self.obs is None:
                return
                
            # Number of samples to use
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
                returns = self.rewards[:n_samples]
                advantages = torch.zeros_like(self.rewards[:n_samples])
                
            # Create indices for all samples
            indices = torch.randperm(n_samples)
            
            # Yield batches
            start_idx = 0
            while start_idx < n_samples:
                batch_indices = indices[start_idx:start_idx + batch_size]
                
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
            """Compute returns and advantages using GAE"""
            n_samples = len(rewards)
            
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)
            
            gae = 0
            next_value = 0
            next_done = False
            
            for t in reversed(range(n_samples)):
                if t < n_samples - 1:
                    next_value = values[t + 1]
                    next_done = dones[t + 1]
                
                delta = rewards[t] + gamma * next_value * (~next_done) - values[t]
                gae = delta + gamma * gae_lambda * (~next_done) * gae
                
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return returns, advantages
        
        def clear(self):
            """Clear the buffer"""
            # Add debug logging that will work in this context
            import inspect
            # Check if we're called from the update method
            stack = inspect.stack()
            caller_name = stack[1].function if len(stack) > 1 else "unknown"

            self._reset_buffers()
            
        def size(self):
            """Get the number of samples in the buffer"""
            return self.buffer_size if self.full else self.pos
    
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
    
    def store_experience(self, obs, action, log_prob, reward, value, done):
        """Store experience in the buffer"""
        # Add debug logging to track experience storage
        if getattr(self, 'debug', False) and isinstance(self.memory.pos, int) and self.memory.pos % 100 == 0:
            buffer_size = getattr(self.memory, 'buffer_size', 0)
            current_pos = getattr(self.memory, 'pos', 0)
            print(f"[DEBUG PPO] Storing experience at position {current_pos}/{buffer_size}")
            if hasattr(self.memory, 'rewards') and self.memory.rewards is not None:
                reward_sum = self.memory.rewards.sum().item() if isinstance(self.memory.rewards, torch.Tensor) else 0
                print(f"[DEBUG PPO] Current reward sum: {reward_sum:.4f}")
                
        self.memory.store(obs, action, log_prob, reward, value, done)
        
        # Track episode reward for return calculation
        if hasattr(self, 'current_episode_rewards'):
            # Convert reward to a scalar if it's a tensor
            reward_value = reward.item() if isinstance(reward, torch.Tensor) else reward
            self.current_episode_rewards.append(reward_value)
            
        # When episode is done, calculate the return
        if done and hasattr(self, 'current_episode_rewards') and hasattr(self, 'episode_returns'):
            episode_return = sum(self.current_episode_rewards)
            self.episode_returns.append(episode_return)
            self.current_episode_rewards = []  # Reset for next episode
    
    def update(self):
        """Update policy using PPO"""
        buffer_size = self.memory.size()
        
        if buffer_size == 0:
            if self.debug:
                print("[DEBUG] Buffer is empty, skipping update")
            return self.metrics
        
        n_batches = math.ceil(buffer_size / self.batch_size)
        
        actor_loss_epoch = 0
        critic_loss_epoch = 0
        entropy_loss_epoch = 0
        total_loss_epoch = 0
        clip_fraction_sum = 0
        
        generator = self.memory.get_generator(
            self.batch_size,
            compute_returns=True,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        advantages_all = []
        returns_all = []
        values_all = []
        old_log_probs_all = []
        states_all = []
        
        for epoch in range(self.ppo_epochs):
            generator = self.memory.get_generator(
                self.batch_size,
                compute_returns=True if epoch == 0 else False,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )
            
            for batch in generator:
                obs = batch['obs']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                returns = batch['returns']
                advantages = batch['advantages']
                
                if epoch == 0:
                    if isinstance(advantages, torch.Tensor):
                        advantages_all.append(advantages.detach())
                    if isinstance(returns, torch.Tensor):
                        returns_all.append(returns.detach())
                    if isinstance(old_log_probs, torch.Tensor):
                        old_log_probs_all.append(old_log_probs.detach())
                    if isinstance(obs, torch.Tensor):
                        states_all.append(obs.detach())
                
                with torch.amp.autocast(enabled=self.use_amp, device_type="cuda"):
                    if self.action_space_type == "discrete":
                        action_probs = self.actor(obs)
                        action_probs = torch.clamp(action_probs, 1e-10, 1.0)
                        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                        dist = torch.distributions.Categorical(probs=action_probs)
                        action_idx = torch.argmax(actions, dim=-1)
                        new_log_probs = dist.log_prob(action_idx)
                        entropy = dist.entropy().mean()
                    else:
                        dist = self.actor(obs)
                        new_log_probs = dist.log_prob(actions).sum(dim=-1)
                        entropy = dist.entropy().mean()
                    
                    new_values = self.critic(obs).squeeze(-1)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    ratio = torch.clamp(ratio, 0.01, 100.0)
                    
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(new_values, returns)
                    entropy_loss = -entropy * self.entropy_coef
                    total_loss = actor_loss + self.critic_coef * critic_loss + entropy_loss
                    
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    clip_fraction_sum += clip_fraction
                
                if self.use_amp:
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.actor_optimizer)
                        self.scaler.unscale_(self.critic_optimizer)
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    total_loss.backward()
                    
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                
                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                total_loss_epoch += total_loss.item()
        
        n_updates = n_batches * self.ppo_epochs
        self.metrics.update({
            'actor_loss': actor_loss_epoch / n_updates,
            'critic_loss': critic_loss_epoch / n_updates,
            'entropy_loss': entropy_loss_epoch / n_updates,
            'total_loss': total_loss_epoch / n_updates,
            'clip_fraction': clip_fraction_sum / n_updates,
            'explained_variance': self._compute_explained_variance(values_all, returns_all) if values_all and returns_all else 0.0,
            'kl_divergence': self._compute_approx_kl(states_all, old_log_probs_all) if states_all and old_log_probs_all else 0.0,
            'mean_advantage': torch.cat(advantages_all).mean().item() if advantages_all else 0.0,
            'mean_return': torch.cat(returns_all).mean().item() if returns_all else 0.0
        })
        
        # Calculate mean return from episode returns
        if len(self.episode_returns) > 0:
            self.metrics['mean_return'] = sum(self.episode_returns) / len(self.episode_returns)
        
        buffer_pos = self.memory.pos
        self.memory.clear()
        if self.debug:
            print(f"[DEBUG] PPO Memory buffer cleared from update(). Was at position {buffer_pos}/{self.memory.buffer_size}")
        return self.metrics
    
    def _compute_explained_variance(self, values_all, returns_all):
        """Compute the explained variance metric"""
        if not values_all or not returns_all:
            return 0.0
            
        try:
            all_values = torch.cat(values_all)
            all_returns = torch.cat(returns_all)
            
            values_mean = all_values.mean()
            returns_mean = all_returns.mean()
            
            values_var = ((all_values - values_mean) ** 2).mean()
            returns_var = ((all_returns - returns_mean) ** 2).mean()
            diff_var = ((all_values - all_returns) ** 2).mean()
            
            return (1 - (diff_var / (returns_var + 1e-8))).item()
        except:
            return 0.0
    
    def _compute_approx_kl(self, states_all, old_log_probs_all):
        """Compute approximate KL divergence"""
        if not states_all or not old_log_probs_all:
            return 0.0
            
        try:
            all_states = torch.cat(states_all)
            all_old_log_probs = torch.cat(old_log_probs_all)
            
            with torch.no_grad():
                if self.action_space_type == "discrete":
                    action_probs = self.actor(all_states)
                    action_probs = torch.clamp(action_probs, 1e-10, 1.0)
                    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                    dist = torch.distributions.Categorical(probs=action_probs)
                    new_actions = dist.sample()
                    new_log_probs = dist.log_prob(new_actions)
                else:
                    dist = self.actor(all_states)
                    new_actions = dist.sample()
                    new_log_probs = dist.log_prob(new_actions).sum(dim=-1)
                
                return (all_old_log_probs - new_log_probs).mean().item()
        except:
            return 0.0
    
    def reset(self):
        """Reset memory"""
        self.memory.clear()