import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import math
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any
from .base import BaseAlgorithm

class StreamACAlgorithm(BaseAlgorithm):
    """Implementation of StreamAC from the paper https://arxiv.org/pdf/2410.14606"""
    
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
            entropy_coef: float = 0.01,
            critic_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            adaptive_learning_rate: bool = True,
            target_step_size: float = 0.01,
            min_lr_factor: float = 0.1,
            max_lr_factor: float = 10.0,
            experience_buffer_size: int = 100,
            update_freq: int = 1,
            use_sparse_init: bool = True,
            use_obgd: bool = True,
            buffer_size: int = 32,
            debug: bool = False,
            **kwargs
        ):
        self.use_sparse_init = use_sparse_init
        self.use_obgd = use_obgd

        """Initialize StreamAC algorithm"""
        super().__init__(
            actor=actor,
            critic=critic,
            action_space_type=action_space_type,
            action_dim=action_dim,
            device=device,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            max_grad_norm=max_grad_norm,
            **kwargs
        )
        
        # Store additional configuration parameters
        self.adaptive_learning_rate = adaptive_learning_rate
        self.target_step_size = target_step_size
        self.min_lr_factor = min_lr_factor
        self.max_lr_factor = max_lr_factor
        self.experience_buffer_size = experience_buffer_size
        self.update_freq = update_freq
        self.debug = debug
        
        # Store base learning rates for adaptive LR
        self.base_lr_actor = lr_actor
        self.base_lr_critic = lr_critic
        
        # Initialize metrics and counters
        self.metrics = {}
        self.training_steps = 0
        self.update_counter = 0
        self.backtracking_count = 0
        
        # Initialize experience buffer
        self.experience_buffer = deque(maxlen=experience_buffer_size)
        
        # Initialize effective step size history
        self.effective_step_size_history = []
        
        # Initialize eligibility traces with small random values
        actor_param_count = sum(p.numel() for p in self.actor.parameters())
        critic_param_count = sum(p.numel() for p in self.critic.parameters())
        self.actor_trace = torch.randn(actor_param_count, device=self.device) * 0.01
        self.critic_trace = torch.randn(critic_param_count, device=self.device) * 0.01
        
        # Apply SparseInit if requested
        if self.use_sparse_init:
            if self.debug:
                print("[DEBUG] Applying SparseInit to actor and critic networks")
            self.apply_sparse_init(self.actor)
            self.apply_sparse_init(self.critic)
            
        # Initialize optimizer momentum buffers
        self._init_optimizer_buffers()
    
    def apply_sparse_init(self, model):
        """Apply SparseInit initialization as described in the paper"""
        for name, param in model.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    fan_in = param.size(1)
                    if param.dim() > 2:
                        fan_in *= param.size(2) * param.size(3)
                    
                    scale = 1.0 / math.sqrt(fan_in)
                    sparsity = 0.9
                    
                    param.data.zero_()
                    mask = torch.rand_like(param) > sparsity
                    num_nonzero = mask.sum().item()
                    
                    if num_nonzero > 0:
                        param.data[mask] = torch.randn(num_nonzero, device=param.device) * scale * 3.0
                    else:
                        if param.dim() == 2:
                            for i in range(param.size(0)):
                                idx = torch.randint(0, param.size(1), (1,))
                                param.data[i, idx] = torch.randn(1, device=param.device) * scale * 3.0
                        else:
                            idx = torch.randint(0, param.numel(), (1,))
                            param.data.view(-1)[idx] = torch.randn(1, device=param.device) * scale * 3.0
            
            elif 'bias' in name:
                param.data.zero_()
    
    def _init_optimizer_buffers(self):
        """Initialize optimizer momentum buffers with dummy passes"""
        try:
            if hasattr(self, 'obs_dim'):
                dummy_obs = torch.randn((1, self.obs_dim), device=self.device)
            else:
                first_layer = next(self.actor.parameters()).shape
                if len(first_layer) > 1:
                    input_size = first_layer[1]
                    dummy_obs = torch.randn((1, input_size), device=self.device)
                else:
                    dummy_obs = None
                    
            if dummy_obs is not None:
                # Actor dummy update
                self.actor_optimizer.zero_grad()
                if self.action_space_type == "discrete":
                    action_probs = self.actor(dummy_obs)
                    actor_loss = -action_probs.mean()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                
                # Critic dummy update
                self.critic_optimizer.zero_grad()
                value = self.critic(dummy_obs)
                critic_loss = value.mean()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                # Reset optimizers
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                if self.debug:
                    print("[StreamAC] Traces and optimizer state initialized")
        except Exception as e:
            if self.debug:
                print(f"[StreamAC] Warning: Could not initialize optimizer state: {e}")
            pass

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
            
        if return_features:
            return action, log_prob, value, features
        else:
            return action, log_prob, value
    
    def store_experience(self, obs, action, log_prob, reward, value, done):
        """Store experience and directly perform an update"""
        # Convert inputs to tensors if needed
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
        
        experience = {
            'obs': obs,
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'value': value,
            'done': done
        }
        
        # Add to buffer
        self.experience_buffer.append(experience)
        
        # Keep buffer size limited
        if len(self.experience_buffer) > self.experience_buffer_size:
            self.experience_buffer.pop(0)
        
        # Update immediately
        metrics = self.update()
        return metrics, True
    
    def update(self):
        """Update policy using StreamAC"""
        # Increment step counter
        self.training_steps += 1
        debug_this_step = self.debug and (self.training_steps % 10 == 0)
        
        # Get latest experience
        if not self.experience_buffer:
            return self.metrics
            
        exp = self.experience_buffer[-1]
        obs = exp['obs']
        action = exp['action']
        reward = exp['reward']
        old_value = exp['value']
        done = exp['done']
        
        # Ensure all inputs are tensors
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        if not isinstance(old_value, torch.Tensor):
            old_value = torch.tensor(old_value, dtype=torch.float32, device=self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        # Compute TD error
        with torch.no_grad():
            if not done:
                next_value = self.critic(obs).squeeze(-1)
                delta = reward + self.gamma * next_value - old_value
            else:
                delta = reward - old_value
        
        # Force minimum magnitude for delta
        delta_scalar = delta.item()
        if abs(delta_scalar) < 0.01:
            delta_scalar = 0.01 if delta_scalar >= 0 else -0.01
        
        # Compute losses and gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # Forward passes
        value = self.critic(obs).squeeze(-1)
        critic_loss = ((value - old_value) ** 2).mean()
        
        if self.action_space_type == "discrete":
            action_probs = self.actor(obs)
            dist = torch.distributions.Categorical(probs=action_probs)
            log_prob = dist.log_prob(action.long())
            entropy = dist.entropy()
        else:
            dist = self.actor(obs)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        
        # Actor loss with entropy bonus
        actor_loss = -log_prob.mean()
        entropy_loss = -self.entropy_coef * entropy.mean()
        total_actor_loss = actor_loss + entropy_loss
        
        # Compute gradients
        critic_loss.backward()
        critic_grads = [p.grad.clone() for p in self.critic.parameters() if p.grad is not None]
        
        total_actor_loss.backward()
        actor_grads = [p.grad.clone() for p in self.actor.parameters() if p.grad is not None]
        
        # Update eligibility traces
        if not hasattr(self, 'actor_traces') or not self.actor_traces:
            self.actor_traces = [torch.zeros_like(p) for p in actor_grads]
            self.critic_traces = [torch.zeros_like(p) for p in critic_grads]
        
        for i in range(len(self.actor_traces)):
            self.actor_traces[i] = self.gamma * self.gae_lambda * self.actor_traces[i] + actor_grads[i]
        
        for i in range(len(self.critic_traces)):
            self.critic_traces[i] = self.gamma * self.gae_lambda * self.critic_traces[i] + critic_grads[i]
        
        # Get trace norms
        actor_trace_norm = sum(t.norm().item() for t in self.actor_traces)
        critic_trace_norm = sum(t.norm().item() for t in self.critic_traces)
        
        # Calculate effective step size
        effective_step_size = self.lr_actor * abs(delta_scalar) * actor_trace_norm
        
        # Force minimum update magnitude
        min_update = 0.001
        
        # Apply actor updates
        actor_param_idx = 0
        for param, trace in zip(self.actor.parameters(), self.actor_traces):
            if not param.requires_grad:
                continue
            
            step_size = self.lr_actor
            update = -step_size * delta_scalar * trace
            
            update_norm = update.norm().item()
            if 0 < update_norm < min_update * param.norm().item():
                update = update * (min_update * param.norm().item() / update_norm)
            
            if self.training_steps % 100 == 0:
                update = update + torch.randn_like(update) * 0.001 * param.norm().item()
            
            param.data.add_(update)
            actor_param_idx += 1
        
        # Apply critic updates
        critic_param_idx = 0
        for param, trace in zip(self.critic.parameters(), self.critic_traces):
            if not param.requires_grad:
                continue
            
            step_size = self.lr_critic
            update = -step_size * delta_scalar * trace
            
            update_norm = update.norm().item()
            if 0 < update_norm < min_update * param.norm().item():
                update = update * (min_update * param.norm().item() / update_norm)
            
            if self.training_steps % 100 == 0:
                update = update + torch.randn_like(update) * 0.001 * param.norm().item()
            
            param.data.add_(update)
            critic_param_idx += 1
        
        # Store effective step size
        self.effective_step_size_history.append(effective_step_size)
        if len(self.effective_step_size_history) > 100:
            self.effective_step_size_history.pop(0)
        
        # Periodic learning rate reset
        if self.training_steps % 1000 == 0 and self.adaptive_learning_rate:
            self.lr_actor = self.base_lr_actor
            self.lr_critic = self.base_lr_critic
            
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.lr_actor
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.lr_critic
                
            if debug_this_step:
                print(f"[DEBUG] RESET learning rates: actor={self.lr_actor:.6f}, critic={self.lr_critic:.6f}")
        
        # Update metrics
        self.metrics.update({
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_actor_loss.item() + critic_loss.item(),
            'effective_step_size': effective_step_size,
            'mean_return': reward.mean().item(),
            'mean_advantage': delta_scalar,
            'policy_kl_divergence': 0.0,
            'actor_trace_norm': actor_trace_norm,
            'critic_trace_norm': critic_trace_norm,
            'backtracking_count': 0,
            'actor_learning_rate': self.lr_actor,
            'critic_learning_rate': self.lr_critic,
        })
        
        # Reset traces if episode is done
        if done and self.gae_lambda < 1.0:
            for trace in self.actor_traces:
                trace.zero_()
            for trace in self.critic_traces:
                trace.zero_()
        
        return self.metrics

    def reset(self):
        """Reset the algorithm state"""
        self.experience_buffer.clear()
        self.effective_step_size_history = []