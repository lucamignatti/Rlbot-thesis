import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import math
from typing import Dict, Tuple, List, Optional, Union, Any
from .base import BaseAlgorithm
# Import AuxiliaryTaskManager to use its methods
from auxiliary import AuxiliaryTaskManager # Assuming this exists and works as intended
# Import GradScaler for AMP
from torch.amp import GradScaler, autocast
# Removed SimbaV2 l2_norm import

class PPOAlgorithm(BaseAlgorithm):
    """
    Standard PPO algorithm implementation with support for Auxiliary Tasks.

    Key features:
    1. Standard Actor-Critic architecture with scalar value prediction.
    2. Clipped Surrogate Objective for policy updates.
    3. Mean Squared Error (MSE) loss for the critic.
    4. Generalized Advantage Estimation (GAE).
    5. Optional integration with AuxiliaryTaskManager.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        # Add aux_task_manager parameter
        aux_task_manager: Optional[AuxiliaryTaskManager] = None,
        action_space_type: str = "discrete",
        action_dim: Optional[int] = None,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        device: str = "cuda",
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4, # Can be the same or different from lr_actor
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
        # --- Removed Distributional/Simba/WeightClip Parameters ---
    ):
        # Initialize BaseAlgorithm with relevant parameters
        super().__init__(
            actor=actor,
            critic=critic,
            action_space_type=action_space_type,
            action_dim=action_dim,
            action_bounds=action_bounds,
            device=device,
            lr_actor=lr_actor,
            lr_critic=lr_critic, # Pass critic LR to base
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

        # Store the auxiliary task manager
        self.aux_task_manager = aux_task_manager
        # Check if actor and critic are the same instance
        self.shared_model = (actor is critic)
        if self.debug and self.shared_model:
            print("[DEBUG PPO] Actor and Critic are the same instance (shared model).")

        # --- Removed Distributional Critic Params ---
        # --- Removed Distributional PPO Uncertainty Weighting Params ---
        # --- Removed Weight Clipping Params ---

        # Initialize memory with buffer size and device
        # Note: PPOMemory class needs to be adapted slightly or confirmed compatible
        self.memory = self.PPOMemory(
            batch_size=batch_size,
            # Increase buffer size to match update_interval typically used
            buffer_size=131072, # Example size, adjust if needed
            device=device,
            debug=debug,
            action_space_type=self.action_space_type,
            # Removed algorithm_instance reference as complex normalization is gone
        )

        # Initialize optimizer(s)
        # Option 1: Shared Optimizer (if lr_actor == lr_critic or models are shared)
        if self.shared_model or lr_actor == lr_critic:
             combined_params = list(self.actor.parameters()) # Covers both if shared
             if not self.shared_model:
                 combined_params += list(self.critic.parameters()) # Add critic if separate

             # Add auxiliary task parameters if they exist
             if self.aux_task_manager:
                 aux_params = self.aux_task_manager.get_parameters() # Assume manager provides its params
                 combined_params += aux_params

             self.optimizer = torch.optim.Adam(combined_params, lr=lr_actor)
             self.actor_optimizer = self.optimizer # Point to the same optimizer
             self.critic_optimizer = self.optimizer # Point to the same optimizer
             if self.debug: print("[DEBUG PPO] Using combined optimizer.")

        # Option 2: Separate Optimizers
        else:
            actor_params = list(self.actor.parameters())
            critic_params = list(self.critic.parameters())

            # Add auxiliary task parameters ONLY to the optimizer corresponding to the shared backbone
            # Typically, aux tasks branch off the actor's backbone features
            if self.aux_task_manager:
                aux_params = self.aux_task_manager.get_parameters()
                # Decide where aux params go - usually actor if features come from there
                actor_params += aux_params # Assuming aux tasks depend on actor features

            self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr_actor)
            self.critic_optimizer = torch.optim.Adam(critic_params, lr=lr_critic)
            # Create a dummy combined optimizer reference for scaler (only one needed)
            self.optimizer = self.actor_optimizer # Or critic_optimizer, scaler works with any
            if self.debug: print("[DEBUG PPO] Using separate optimizers.")


        # Initialize GradScaler if AMP is enabled
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking metrics - Simplified for standard PPO + Aux
        self.metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy_loss': 0.0,
            'sr_loss_scalar': 0.0, # Aux metric
            'rp_loss_scalar': 0.0, # Aux metric
            'total_loss': 0.0,     # Combined loss for logging
            'clip_fraction': 0.0,  # PPO policy clip fraction
            'explained_variance': 0.0,
            'mean_advantage': 0.0,
            'mean_return': 0.0,    # Average episodic return
        }

        # Add episode return tracking
        self.current_episode_rewards = []
        self.episode_returns = deque(maxlen=100)

        # --- Removed SimbaV2 Reward Normalization Params ---


    class PPOMemory:
        """Memory buffer for PPO to store experiences (Standard Version)"""

        def __init__(self, batch_size, buffer_size, device, debug=False, action_space_type="discrete"):
            # Removed algorithm_instance reference
            self.batch_size = batch_size
            self.debug = debug
            self.action_space_type = action_space_type

            self.buffer_size = buffer_size
            self.device = device
            self.pos = 0
            self.size = 0
            self.full = False

            # Initialize buffers as empty tensors
            self._reset_buffers()

        def _reset_buffers(self):
            """Initialize all buffer tensors with the correct shapes"""
            buffer_size = self.buffer_size
            device = self.device

            # Initialize empty tensors on the specified device
            self.obs = None         # Will be initialized on first store() call
            self.actions = None     # Will be initialized on first store() call
            self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device) # Store value estimates used for GAE
            self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)

            self.pos = 0
            self.full = False
            self.size = 0

        def _initialize_buffers_if_needed(self, obs_sample, action_sample):
            """Initialize buffer tensors based on sample shapes if not already done."""
            if self.obs is None:
                if not isinstance(obs_sample, torch.Tensor):
                    obs_sample = torch.tensor(obs_sample, dtype=torch.float32)
                if not isinstance(action_sample, torch.Tensor):
                    action_dtype = torch.long if self.action_space_type == "discrete" else torch.float32
                    action_sample = torch.tensor(action_sample, dtype=action_dtype)

                if obs_sample.dim() == 1: obs_sample = obs_sample.unsqueeze(0)

                if self.action_space_type == "discrete":
                    action_shape = ()
                    action_dtype = torch.long
                else: # Continuous
                    if action_sample.numel() == 1 and action_sample.dim() == 0:
                        action_shape = (1,)
                    elif action_sample.dim() == 1:
                        action_shape = action_sample.shape
                    elif action_sample.dim() > 1:
                        action_shape = action_sample.shape[1:]
                    else:
                        action_shape = (1,) # Fallback
                    action_dtype = torch.float32

                obs_shape = obs_sample.shape[1:]

                if self.debug:
                    print(f"[DEBUG PPOMemory] Initializing buffers: obs_shape={obs_shape}, action_shape={action_shape}, action_dtype={action_dtype}")

                self.obs = torch.zeros((self.buffer_size, *obs_shape), dtype=torch.float32, device=self.device)
                self.actions = torch.zeros((self.buffer_size, *action_shape), dtype=action_dtype, device=self.device)


        def store_initial_batch(self, obs_batch, action_batch, log_prob_batch, value_batch):
            """Store the initial part of experiences in batch (obs, action, log_prob, value)."""
            batch_size = obs_batch.shape[0]
            if batch_size == 0:
                return torch.tensor([], dtype=torch.long, device=self.device)

            expected_action_dtype = torch.long if self.action_space_type == "discrete" else torch.float32
            action_sample_for_init = action_batch[0].clone().to(expected_action_dtype)
            self._initialize_buffers_if_needed(obs_batch[0], action_sample_for_init)

            if not isinstance(obs_batch, torch.Tensor): obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
            if not isinstance(action_batch, torch.Tensor):
                action_batch = torch.tensor(action_batch, dtype=expected_action_dtype)
            elif action_batch.dtype != expected_action_dtype:
                 if self.debug: print(f"[DEBUG PPOMemory] Casting action_batch from {action_batch.dtype} to {expected_action_dtype}")
                 action_batch = action_batch.to(expected_action_dtype)

            if not isinstance(log_prob_batch, torch.Tensor): log_prob_batch = torch.tensor(log_prob_batch, dtype=torch.float32)
            # Ensure value_batch is float32
            if not isinstance(value_batch, torch.Tensor): value_batch = torch.tensor(value_batch, dtype=torch.float32)
            elif value_batch.dtype != torch.float32: value_batch = value_batch.to(torch.float32)


            target_device = self.device
            obs_batch = obs_batch.to(target_device)
            action_batch = action_batch.to(target_device)
            log_prob_batch = log_prob_batch.to(target_device)
            value_batch = value_batch.to(target_device) # Store the value estimate from collection time

            indices = torch.arange(self.pos, self.pos + batch_size, device=target_device) % self.buffer_size

            self.obs.index_copy_(0, indices, obs_batch.detach())
            # Handle action shapes
            if self.action_space_type == "discrete":
                 if action_batch.dim() == 1 and self.actions.dim() == 1:
                     self.actions.index_copy_(0, indices, action_batch.detach())
                 elif action_batch.dim() == 2 and action_batch.shape[1] == 1 and self.actions.dim() == 1:
                     self.actions.index_copy_(0, indices, action_batch.detach().squeeze(1))
                 else:
                      if self.debug: print(f"[DEBUG PPOMemory] Discrete action shape mismatch. Buffer: {self.actions.shape}, Batch: {action_batch.shape}")
            else: # Continuous
                 if self.actions.shape[1:] == action_batch.shape[1:]:
                     self.actions.index_copy_(0, indices, action_batch.detach())
                 else:
                      if self.debug: print(f"[DEBUG PPOMemory] Continuous action shape mismatch. Buffer: {self.actions.shape}, Batch: {action_batch.shape}")

            self.log_probs.index_copy_(0, indices, log_prob_batch.detach())
            self.values.index_copy_(0, indices, value_batch.detach()) # Store the actual scalar value estimate

            new_pos = (self.pos + batch_size) % self.buffer_size
            if not self.full and (self.pos + batch_size >= self.buffer_size):
                self.full = True
            self.pos = new_pos
            self.size = min(self.size + batch_size, self.buffer_size)

            if self.debug and batch_size > 0:
                 print(f"[DEBUG PPOMemory] Stored initial batch of size {batch_size}. New pos: {self.pos}, size: {self.size}")

            return indices

        def update_rewards_dones_batch(self, indices, rewards_batch, dones_batch):
            """Update rewards and dones for experiences at given indices."""
            if len(indices) == 0:
                return

            if not isinstance(rewards_batch, torch.Tensor): rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
            if not isinstance(dones_batch, torch.Tensor): dones_batch = torch.tensor(dones_batch, dtype=torch.bool)

            target_device = self.device
            indices = indices.to(target_device)
            rewards_batch = rewards_batch.to(target_device)
            dones_batch = dones_batch.to(target_device)

            # Store raw rewards - normalization happens during GAE calculation if needed (advantage normalization)
            self.rewards.index_copy_(0, indices, rewards_batch)
            self.dones.index_copy_(0, indices, dones_batch)

            if self.debug and len(indices) > 0:
                 print(f"[DEBUG PPOMemory] Updated rewards/dones for {len(indices)} indices. First reward: {rewards_batch[0].item():.4f}")

        # store() and store_experience_at_idx() can be simplified or removed if only batch methods are used externally
        # For now, let's assume they might still be called and adapt them

        def store(self, obs, action, log_prob, reward, value, done):
             """Store a single experience (obs, action, log_prob, reward, value, done)"""
             expected_action_dtype = torch.long if self.action_space_type == "discrete" else torch.float32

             obs_b = torch.tensor([obs], dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs.unsqueeze(0)
             act_b = torch.tensor([action], dtype=expected_action_dtype) if not isinstance(action, torch.Tensor) else action.unsqueeze(0).to(expected_action_dtype)
             lp_b = torch.tensor([log_prob], dtype=torch.float32) if not isinstance(log_prob, torch.Tensor) else log_prob.unsqueeze(0)
             # Ensure value is float32
             val_b = torch.tensor([value], dtype=torch.float32) if not isinstance(value, torch.Tensor) else value.unsqueeze(0).to(torch.float32)
             rew_b = torch.tensor([reward], dtype=torch.float32) if not isinstance(reward, torch.Tensor) else reward.unsqueeze(0)
             done_b = torch.tensor([done], dtype=torch.bool) if not isinstance(done, torch.Tensor) else done.unsqueeze(0)

             indices = self.store_initial_batch(obs_b, act_b, lp_b, val_b)
             if len(indices) > 0:
                 self.update_rewards_dones_batch(indices, rew_b, done_b)


        def get(self):
            """Get all data currently stored in the buffer."""
            if self.size == 0 or self.obs is None:
                if self.debug:
                    print("[DEBUG PPOMemory] get() called but buffer is empty or uninitialized.")
                return None, None, None, None, None, None

            obs_data = self.obs[:self.size]
            actions_data = self.actions[:self.size]
            log_probs_data = self.log_probs[:self.size]
            rewards_data = self.rewards[:self.size]
            values_data = self.values[:self.size] # These are the value estimates from collection time
            dones_data = self.dones[:self.size]

            if self.debug:
                print(f"[DEBUG PPOMemory] get() returning data of size {self.size}")

            return (
                obs_data,
                actions_data,
                log_probs_data,
                rewards_data,
                values_data,
                dones_data
            )

        def generate_batches(self):
            """Generate batches of indices for training."""
            if self.size == 0:
                if self.debug:
                    print("[DEBUG PPOMemory] generate_batches() called but buffer is empty.")
                return []

            indices = torch.randperm(self.size, device=self.device)
            batch_start = 0
            batches = []
            while batch_start < self.size:
                batch_end = min(batch_start + self.batch_size, self.size)
                batch_idx = indices[batch_start:batch_end]
                batches.append(batch_idx)
                batch_start = batch_end

            if self.debug:
                print(f"[DEBUG PPOMemory] Generated {len(batches)} batches for size {self.size}")

            return batches

        def clear(self):
            """Reset the buffer."""
            self.pos = 0
            self.size = 0
            self.full = False
            # Optionally zero out tensors if needed, but overwriting is usually fine
            if self.debug:
                print("[DEBUG PPOMemory] Buffer cleared.")

    # --- Batch Storage Methods ---
    def store_initial_batch(self, obs_batch, action_batch, log_prob_batch, value_batch):
        """Store the initial part of experiences (obs, action, log_prob, value) in batch."""
        if self.debug:
            print(f"[DEBUG PPO] Storing initial batch of size {obs_batch.shape[0]}")
        # Ensure value_batch is float32 before storing
        if not isinstance(value_batch, torch.Tensor):
             value_batch = torch.tensor(value_batch, dtype=torch.float32)
        elif value_batch.dtype != torch.float32:
             value_batch = value_batch.to(torch.float32)
        return self.memory.store_initial_batch(obs_batch, action_batch, log_prob_batch, value_batch)

    def update_rewards_dones_batch(self, indices, rewards_batch, dones_batch):
        """Update rewards and dones for experiences at given indices in batch."""
        if self.debug:
            print(f"[DEBUG PPO] Updating rewards/dones for batch of size {len(indices)}")
        self.memory.update_rewards_dones_batch(indices, rewards_batch, dones_batch)

        # Track episode returns when dones are received
        done_indices_local = torch.where(dones_batch.cpu())[0] # Find dones on CPU
        if len(done_indices_local) > 0:
             # Process each completed episode in the batch
             # This requires careful handling if batches contain partial episodes
             # For simplicity, this assumes reward tracking happens externally or per-step
             # Or, we reconstruct episode rewards if possible (more complex)
             # Let's keep the simple per-step reward accumulation for now
             pass # Logic moved to store_experience for simplicity

    # --- Single Experience Method (Legacy/Optional) ---
    def store_experience(self, obs, action, log_prob, reward, value, done):
        """
        Store a single experience in the buffer.
        Value should be the scalar value estimate V(s) from the critic at collection time.
        """
        if self.debug:
            print(f"[DEBUG PPO] Storing single experience - reward: {reward}, value: {value}")

        # Store raw reward, normalization happens via advantage normalization
        # Ensure value is float
        value_float = float(value.item() if isinstance(value, torch.Tensor) else value)
        self.memory.store(obs, action, log_prob, reward, value_float, done)

        # Track episode rewards for calculating returns
        if isinstance(reward, torch.Tensor):
            reward_val = reward.item()
        else:
            reward_val = float(reward)
        self.current_episode_rewards.append(reward_val)

        if done:
            if self.current_episode_rewards:
                episode_return = sum(self.current_episode_rewards)
                self.episode_returns.append(episode_return)
                if self.debug:
                    print(f"[DEBUG PPO] Episode done with return: {episode_return}")
                self.current_episode_rewards = [] # Reset for next episode

    # --- Action Selection ---
    def get_action(self, obs, deterministic=False, return_features=False):
        """Get action and value for a given observation"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        # Add batch dimension if missing
        needs_unsqueeze = obs.dim() == len(self.actor.input_shape) # Example check, adapt to your model
        if needs_unsqueeze:
             obs = obs.unsqueeze(0)

        action = None
        log_prob = None
        value = None # Scalar value
        features = None

        with torch.no_grad():
            if self.shared_model:
                # Request actor output, critic output (scalar value), and features
                model_output = self.actor(obs, return_actor=True, return_critic=True, return_features=return_features)
                actor_output = model_output.get('actor_out')
                value = model_output.get('critic_out') # Expecting scalar value [batch, 1] or [batch]
                if return_features:
                    features = model_output.get('features')
            else:
                # Separate actor and critic calls
                actor_result = self.actor(obs, return_features=return_features)
                value = self.critic(obs) # Expecting scalar value [batch, 1] or [batch]

                if return_features:
                    if isinstance(actor_result, tuple):
                         actor_output, features = actor_result
                    else: # Assume actor only returns output if return_features=False
                         actor_output = actor_result
                         features = None # Need features from somewhere if aux tasks are used
                         # Maybe call actor again? Or modify actor to always return features?
                         # For now, assume actor returns features if needed.
                         # A cleaner way is self.actor(obs, return_features=True) always if aux is used.
                         _, features = self.actor(obs, return_features=True) # Re-call if features needed separately
                else:
                    actor_output = actor_result # Actor output only

            if actor_output is None:
                if self.debug: print("[DEBUG PPO get_action] Error: Actor output not found.")
                # Handle error: return dummy values or raise exception
                dummy_action = torch.zeros(obs.shape[0], dtype=torch.long if self.action_space_type == "discrete" else torch.float, device=self.device)
                dummy_log_prob = torch.zeros(obs.shape[0], device=self.device)
                dummy_value = torch.zeros(obs.shape[0], 1, device=self.device) # Match expected shape
                if needs_unsqueeze: # Remove batch dim if added
                     dummy_action, dummy_log_prob, dummy_value = dummy_action.squeeze(0), dummy_log_prob.squeeze(0), dummy_value.squeeze(0)
                if return_features: return dummy_action, dummy_log_prob, dummy_value, features
                else: return dummy_action, dummy_log_prob, dummy_value

            # Ensure value is correctly shaped [batch, 1] or [batch] -> [batch, 1]
            if value is None: # Handle critic error
                 if self.debug: print("[DEBUG PPO get_action] Error: Critic output (value) not found.")
                 value = torch.zeros(obs.shape[0], 1, device=self.device) # Dummy value
            elif value.dim() == 1:
                 value = value.unsqueeze(1) # Ensure shape [batch, 1]
            elif value.dim() == 2 and value.shape[1] != 1:
                 if self.debug: print(f"[DEBUG PPO get_action] Warning: Critic output has shape {value.shape}, expected [batch, 1]. Taking first element.")
                 value = value[:, 0].unsqueeze(1)


            # --- Action Sampling / Deterministic Choice ---
            dist = None
            if self.action_space_type == "discrete":
                # Assuming actor_output contains logits or probabilities
                # If logits, apply softmax
                if torch.is_floating_point(actor_output[0]) and actor_output.shape[-1] > 1 : # Heuristic: check if looks like logits/probs
                    probs = F.softmax(actor_output, dim=-1)
                    # Clamp and re-normalize for stability
                    probs = torch.clamp(probs, min=1e-8)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    try:
                        dist = torch.distributions.Categorical(probs=probs)
                    except ValueError as e:
                         if self.debug: print(f"[DEBUG PPO get_action] Error creating Categorical distribution: {e}")
                         # Fallback: deterministic action
                         action = torch.argmax(probs, dim=-1)
                         log_prob = torch.zeros_like(action, dtype=torch.float32) # Placeholder log_prob

                else: # If actor_output is already sampled action index (less common)
                     action = actor_output.long()
                     log_prob = torch.zeros_like(action, dtype=torch.float32) # Cannot compute log_prob easily

                if dist is not None:
                    if deterministic:
                        action = torch.argmax(dist.probs, dim=-1)
                    else:
                        action = dist.sample()
                    log_prob = dist.log_prob(action)

            else: # Continuous
                # Assuming actor_output is a distribution object (e.g., Normal)
                action_dist = actor_output
                if deterministic:
                    action = action_dist.loc # Mean action
                else:
                    action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1) # Sum across action dimensions

                # --- Clip continuous actions to bounds ---
                # Assuming action_bounds are defined and relevant
                action = torch.clamp(action, min=self.action_bounds[0], max=self.action_bounds[1])


        # Remove batch dimension if we added it
        if needs_unsqueeze:
             action = action.squeeze(0)
             log_prob = log_prob.squeeze(0)
             value = value.squeeze(0) # Shape [1]
             if return_features and features is not None:
                 features = features.squeeze(0)

        # Return features if requested
        if return_features:
             return action, log_prob, value, features
        else:
             return action, log_prob, value


    def reset(self):
        """Reset memory"""
        self.memory.clear()

    # --- Update Function ---
    def update(self):
        """Update policy using standard PPO"""
        buffer_size = self.memory.size

        if buffer_size < self.batch_size: # Don't update if buffer has less than one batch
            if self.debug:
                print(f"[DEBUG PPO] Buffer size ({buffer_size}) < batch size ({self.batch_size}), skipping update")
            # Return previous metrics, ensuring aux losses are included if manager exists
            # Use get() to safely retrieve last known values
            metrics_to_return = self.metrics.copy()
            if self.aux_task_manager:
                metrics_to_return['sr_loss_scalar'] = self.aux_task_manager.last_sr_loss
                metrics_to_return['rp_loss_scalar'] = self.aux_task_manager.last_rp_loss
            return metrics_to_return

        if self.debug:
            print(f"[DEBUG PPO] Starting update with buffer size: {buffer_size}")

        # Get experiences from buffer
        states, actions, old_log_probs, rewards, values, dones = self.memory.get() # `values` are V(s_t) from collection time

        if states is None:
            if self.debug:
                print("[DEBUG PPO] Failed to get experiences from buffer, skipping update")
            return self.metrics

        # Compute returns and advantages using GAE
        # We need V(s_last) which might not be in `values`.
        # Recompute last value or use 0 if last state was `done`.
        with torch.no_grad():
            last_value = torch.tensor([[0.0]], device=self.device) # Default if buffer empty or last state is done
            if buffer_size > 0:
                 last_state = states[-1].unsqueeze(0) # Get last state, add batch dim
                 if not dones[-1]: # If the trajectory didn't end
                      if self.shared_model:
                          model_output = self.critic(last_state, return_actor=False, return_critic=True, return_features=False)
                          last_value = model_output.get('critic_out', torch.tensor([[0.0]], device=self.device))
                      else:
                          last_value = self.critic(last_state) # Get V(s_last)

                      # Ensure scalar value shape [1, 1]
                      if last_value.dim() == 0: last_value = last_value.view(1, 1)
                      elif last_value.dim() == 1: last_value = last_value.unsqueeze(1)
                      elif last_value.shape[1] != 1: last_value = last_value[:, 0].unsqueeze(1)


        # `values` from buffer are V(s_t), `last_value` is V(s_{t+1}) for the last step.
        returns, advantages = self._compute_gae(rewards, values, dones, last_value.item()) # Pass scalar last_value

        # Check for NaNs/Infs in advantages or returns
        if torch.isnan(advantages).any() or torch.isinf(advantages).any() or \
           torch.isnan(returns).any() or torch.isinf(returns).any():
            if self.debug:
                print("[DEBUG PPO] NaN or Inf detected in advantages or returns, skipping update")
                print(f"Advantages mean: {advantages.mean()}, std: {advantages.std()}")
                print(f"Returns mean: {returns.mean()}, std: {returns.std()}")
                print(f"Values mean: {values.mean()}, std: {values.std()}")
            self.memory.clear() # Clear memory
            return self.metrics # Return previous metrics

        # Update the policy and value networks using PPO
        metrics = self._update_policy(states, actions, old_log_probs, returns, advantages, rewards)

        # Clear the memory buffer after using the data for updates
        self.memory.clear()

        # Update the metrics dictionary with results from _update_policy
        self.metrics.update(metrics)

        # If we have episode returns, update the mean return metric
        if len(self.episode_returns) > 0:
            self.metrics['mean_return'] = sum(self.episode_returns) / len(self.episode_returns)
        else:
             self.metrics['mean_return'] = 0.0 # Or keep previous value?

        if self.debug:
            print(f"[DEBUG PPO] Update finished. Actor Loss: {metrics.get('actor_loss', 0):.4f}, Critic Loss: {metrics.get('critic_loss', 0):.4f}, "
                  f"SR Loss: {metrics.get('sr_loss_scalar', 0):.4f}, RP Loss: {metrics.get('rp_loss_scalar', 0):.4f}")

        return self.metrics


    # --- GAE Calculation ---
    def _compute_gae(self, rewards, values, dones, last_value):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE) - Standard Version.

        Args:
            rewards: rewards tensor [buffer_size]
            values: value predictions tensor V(s_t) [buffer_size] (or [buffer_size, 1])
            dones: done flags tensor [buffer_size]
            last_value: Scalar value estimate V(s_N) for the state after the last one in the buffer.

        Returns:
            tuple of (returns, advantages) tensors, both shape [buffer_size]
        """
        buffer_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        gae = 0.0

        # Ensure values has shape [buffer_size] for calculation
        if values.dim() > 1:
             values = values.squeeze(-1) # Convert [buffer_size, 1] to [buffer_size]

        # Calculate advantages
        for t in reversed(range(buffer_size)):
            # TD error: delta = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            next_val = values[t+1] if t < buffer_size - 1 else last_value # V(s_{t+1})
            next_non_terminal = 1.0 - dones[t].float() # 1 if not done, 0 if done

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]

            # GAE advantage: delta_t + gamma * lambda * A_{t+1} * (1 - done_t)
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        # Compute returns R_t = A_t + V(s_t)
        returns = advantages + values

        # Normalize advantages across the entire buffer (standard practice)
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        if self.debug:
            print(f"[DEBUG PPO _compute_gae] Returns mean: {returns.mean():.4f}, std: {returns.std():.4f}")
            print(f"[DEBUG PPO _compute_gae] Advantages (normalized) mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")

        return returns, advantages


    # --- Update Policy and Value Network ---
    def _update_policy(self, states, actions, old_log_probs, returns, advantages, rewards):
        """
        Update policy and value networks using standard PPO algorithm, including auxiliary losses.
        Handles both shared and separate actor/critic models. Uses scalar critic with MSE loss.

        Args:
            states: batch of states [buffer_size, state_dim]
            actions: batch of actions [buffer_size, action_dim or buffer_size]
            old_log_probs: batch of log probabilities from old policy [buffer_size]
            returns: batch of target returns (for critic) [buffer_size]
            advantages: batch of advantages [buffer_size]
            rewards: batch of rewards [buffer_size] (needed for aux tasks)

        Returns:
            dict: metrics from the update
        """
        update_metrics_tensors = {
            'actor_loss': torch.tensor(0.0, device=self.device),
            'critic_loss': torch.tensor(0.0, device=self.device),
            'entropy_loss': torch.tensor(0.0, device=self.device),
            'total_loss': torch.tensor(0.0, device=self.device), # Tracks combined loss of main optimizer
            'clip_fraction': torch.tensor(0.0, device=self.device),
        }
        update_metrics_scalars = { # For metrics not averaged over batches easily
            'sr_loss_scalar': 0.0,
            'rp_loss_scalar': 0.0,
        }

        num_batches_processed = 0

        # Calculate explained variance once before updates using initial values
        explained_var_scalar = 0.0
        mean_advantage_scalar = 0.0
        with torch.no_grad():
             # Use values computed during GAE calculation phase (need to recompute or pass them)
             # Let's recompute values batch-wise here for consistency with update loop
             initial_values_list = []
             buffer_size = states.shape[0]
             for i in range(0, buffer_size, self.batch_size):
                  chunk_states = states[i:min(i + self.batch_size, buffer_size)]
                  with autocast("cuda", enabled=self.use_amp):
                       if self.shared_model:
                            model_output = self.critic(chunk_states, return_actor=False, return_critic=True, return_features=False)
                            chunk_values = model_output.get('critic_out')
                       else:
                            chunk_values = self.critic(chunk_states)

                       # Ensure scalar value, shape [chunk_size, 1] or [chunk_size]
                       if chunk_values is None: chunk_values = torch.zeros(chunk_states.shape[0], device=self.device)
                       if chunk_values.dim() == 2 and chunk_values.shape[1] > 1: chunk_values = chunk_values[:, 0] # Take first if multiple outputs
                       initial_values_list.append(chunk_values.squeeze()) # Ensure shape [chunk_size]

             if initial_values_list:
                  y_pred = torch.cat(initial_values_list) # Predicted V(s) before updates
                  y_true = returns # Target returns
                  var_y = torch.var(y_true)
                  explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
                  explained_var_scalar = explained_var.item()
             else:
                  explained_var_scalar = 0.0

             # Use normalized advantage mean
             mean_advantage_scalar = advantages.mean().item() # Advantages are already normalized

        # Multiple epochs of PPO update
        for epoch in range(self.ppo_epochs):
            batch_indices = self.memory.generate_batches()
            if not batch_indices:
                if self.debug: print(f"[DEBUG PPO _update_policy] Epoch {epoch}: No batches generated, skipping.")
                continue

            if self.debug: print(f"[DEBUG PPO _update_policy] Epoch {epoch}: Processing {len(batch_indices)} batches.")

            for batch_idx in batch_indices:
                num_batches_processed += 1
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx] # Target for critic
                batch_advantages = advantages[batch_idx]
                batch_rewards = rewards[batch_idx] # For aux tasks


                # --- Forward pass for current policy and value ---
                actor_output = None
                predicted_values = None # Scalar values
                current_features = None
                entropy = torch.tensor(0.0, device=self.device)
                curr_log_probs = None

                try:
                    with autocast("cuda", enabled=self.use_amp):
                        if self.shared_model:
                            # Need actor out, critic out (scalar), features
                            model_output = self.actor(batch_states, return_actor=True, return_critic=True, return_features=True)
                            actor_output = model_output.get('actor_out')
                            predicted_values = model_output.get('critic_out') # Expecting [batch, 1] or [batch]
                            current_features = model_output.get('features')
                        else:
                            # Need to call both actor and critic
                            # Call actor requesting features if aux tasks are enabled
                            request_features = self.aux_task_manager is not None
                            actor_result = self.actor(batch_states, return_features=request_features)
                            predicted_values = self.critic(batch_states) # Expecting [batch, 1] or [batch]

                            if request_features:
                                if isinstance(actor_result, tuple):
                                    actor_output, current_features = actor_result
                                else: # Should not happen if features requested
                                     actor_output = actor_result; current_features = None
                                     if self.debug: print("[DEBUG PPO] Warning: Actor did not return features when requested.")
                            else:
                                actor_output = actor_result; current_features = None

                        # Validate outputs
                        if actor_output is None: raise ValueError("Actor output missing")
                        if predicted_values is None: raise ValueError("Critic output (value) missing")

                        # Ensure predicted_values is [batch] for loss calculation
                        if predicted_values.dim() > 1:
                             predicted_values = predicted_values.squeeze(-1)

                        # --- Calculate Actor Loss components ---
                        dist = None
                        if self.action_space_type == "discrete":
                             # Assuming actor_output is logits or probabilities
                             probs = F.softmax(actor_output, dim=-1)
                             probs = torch.clamp(probs, min=1e-8)
                             probs = probs / probs.sum(dim=-1, keepdim=True)
                             dist = torch.distributions.Categorical(probs=probs)

                             # Ensure batch_actions are Long
                             if batch_actions.dtype != torch.long:
                                  batch_actions = batch_actions.long()

                             curr_log_probs = dist.log_prob(batch_actions)
                             entropy = dist.entropy().mean()

                        else: # Continuous
                             # Assuming actor_output is a distribution object
                             action_dist = actor_output
                             # Ensure batch_actions are Float
                             if batch_actions.dtype != torch.float32:
                                  batch_actions = batch_actions.float()

                             curr_log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)
                             entropy = action_dist.entropy().mean()

                        if curr_log_probs is None: raise ValueError("Could not compute current log probs")

                        # Ratio and surrogate objectives
                        ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                        actor_loss = -torch.min(surr1, surr2).mean()

                        # --- Calculate Critic Loss (MSE) ---
                        critic_loss = F.mse_loss(predicted_values, batch_returns)

                        # --- Calculate Entropy Loss ---
                        entropy_loss = -entropy * self.entropy_coef

                        # --- Auxiliary Loss Calculation ---
                        sr_loss = torch.tensor(0.0, device=self.device)
                        rp_loss = torch.tensor(0.0, device=self.device)
                        sr_loss_scalar = 0.0
                        rp_loss_scalar = 0.0
                        if self.aux_task_manager is not None and current_features is not None and \
                           (self.aux_task_manager.sr_weight > 0 or self.aux_task_manager.rp_weight > 0):
                            try:
                                aux_losses = self.aux_task_manager.compute_losses_for_batch(
                                    obs_batch=batch_states,
                                    rewards_batch=batch_rewards,
                                    features_batch=current_features
                                )
                                sr_loss = aux_losses.get("sr_loss", torch.tensor(0.0, device=self.device))
                                rp_loss = aux_losses.get("rp_loss", torch.tensor(0.0, device=self.device))
                                sr_loss_scalar = aux_losses.get("sr_loss_scalar", 0.0)
                                rp_loss_scalar = aux_losses.get("rp_loss_scalar", 0.0)
                            except Exception as e:
                                if self.debug: print(f"[DEBUG PPO Aux] Error computing aux losses: {e}")
                                # Reset losses if error
                                sr_loss = torch.tensor(0.0, device=self.device); rp_loss = torch.tensor(0.0, device=self.device)
                                sr_loss_scalar = 0.0; rp_loss_scalar = 0.0
                        elif self.debug and self.aux_task_manager is not None and current_features is None:
                             print("[DEBUG PPO Aux] Skipping aux loss calculation as features were not obtained.")


                        # --- Combine Losses ---
                        # Decide which optimizer handles which loss based on model structure
                        actor_total_loss = actor_loss + entropy_loss
                        critic_total_loss = self.critic_coef * critic_loss

                        # Add aux losses to the optimizer managing the shared backbone/actor
                        if self.aux_task_manager is not None:
                             actor_total_loss += sr_loss + rp_loss

                        # Overall loss for logging (can be misleading if separate optimizers)
                        total_combined_loss_for_log = actor_total_loss + critic_total_loss


                except Exception as e:
                    if self.debug:
                        import traceback
                        print(f"[DEBUG PPO _update_policy] Error during loss calculation for batch: {e}")
                        traceback.print_exc()
                    continue # Skip this batch if forward/loss calculation fails

                # --- Optimization Step ---
                # Zero gradients for relevant optimizer(s)
                self.actor_optimizer.zero_grad()
                if not self.shared_model and self.critic_optimizer is not self.actor_optimizer:
                     self.critic_optimizer.zero_grad()

                # Scale and backward pass
                # If combined optimizer, scale combined loss
                if self.actor_optimizer is self.critic_optimizer:
                    self.scaler.scale(actor_total_loss + critic_total_loss).backward()
                # If separate, scale and backward each loss separately
                else:
                    self.scaler.scale(actor_total_loss).backward(retain_graph=False) # Free graph if possible
                    self.scaler.scale(critic_total_loss).backward()

                # Unscale and Clip Gradients for each optimizer
                if self.max_grad_norm > 0:
                    # Unscale actor grads
                    self.scaler.unscale_(self.actor_optimizer)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) # Clip actor specific params
                    # If aux tasks have separate params managed by actor_optimizer, clip them too
                    if self.aux_task_manager:
                         nn.utils.clip_grad_norm_(self.aux_task_manager.get_parameters(), self.max_grad_norm)


                    # Unscale and clip critic grads if separate
                    if not self.shared_model and self.critic_optimizer is not self.actor_optimizer:
                        self.scaler.unscale_(self.critic_optimizer)
                        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                # Optimizer Step(s)
                self.scaler.step(self.actor_optimizer)
                if not self.shared_model and self.critic_optimizer is not self.actor_optimizer:
                     self.scaler.step(self.critic_optimizer)

                # Update scaler
                self.scaler.update()

                # --- NO Weight Projection or Clipping ---

                # --- Update Metrics ---
                update_metrics_tensors['actor_loss'] += actor_loss.detach()
                update_metrics_tensors['critic_loss'] += critic_loss.detach() # Store raw critic loss
                update_metrics_tensors['entropy_loss'] += entropy_loss.detach()
                update_metrics_tensors['total_loss'] += total_combined_loss_for_log.detach() # Log combined loss

                update_metrics_scalars['sr_loss_scalar'] += sr_loss_scalar
                update_metrics_scalars['rp_loss_scalar'] += rp_loss_scalar

                with torch.no_grad():
                    policy_clip_fraction_tensor = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    update_metrics_tensors['clip_fraction'] += policy_clip_fraction_tensor

        # --- Finalize Metrics ---
        final_metrics = {}
        if num_batches_processed > 0:
            for key, tensor_val in update_metrics_tensors.items():
                final_metrics[key] = (tensor_val / num_batches_processed).item()
            for key, scalar_val in update_metrics_scalars.items():
                 final_metrics[key] = scalar_val / num_batches_processed # Average accumulated scalars
        else: # Handle case where no batches were processed
             if self.debug: print("[DEBUG PPO _update_policy] No batches processed, returning zero metrics.")
             for key in list(update_metrics_tensors.keys()) + list(update_metrics_scalars.keys()):
                  final_metrics[key] = 0.0

        # Add pre-calculated metrics
        final_metrics['explained_variance'] = explained_var_scalar
        final_metrics['mean_advantage'] = mean_advantage_scalar

        return final_metrics

    # --- Removed _project_weights, clip_weights, init_weight_ranges, _compute_uncertainty_weight ---

    # --- Simplified normalize_reward ---
    def normalize_reward(self, reward, done):
        """
        Placeholder for reward normalization. Currently returns reward unchanged.
        Standard implementations might normalize based on running mean/std of returns
        or use environment wrappers for normalization.
        """
        # For now, return raw reward. Advantage normalization handles scaling differences.
        return reward

    # --- Save/Load State ---
    def get_state_dict(self):
        """Get state dict for saving algorithm state (Standard PPO Version)"""
        state = super().get_state_dict() # Get model states from base

        # Store optimizer states
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        # Store critic optimizer only if it's different
        if self.critic_optimizer is not self.actor_optimizer:
             state['critic_optimizer'] = self.critic_optimizer.state_dict()

        state.update({
            'scaler': self.scaler.state_dict(),
            # 'memory_state': self.memory.get_state_dict() if hasattr(self.memory, 'get_state_dict') else None, # Memory usually not saved
            'episode_returns': list(self.episode_returns),
            'current_episode_rewards': self.current_episode_rewards,
            # Removed non-standard params like kappa, G_running etc.
        })
        # Add aux task manager state if needed
        if self.aux_task_manager and hasattr(self.aux_task_manager, 'get_state_dict'):
             state['aux_task_manager'] = self.aux_task_manager.get_state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load state dict for resuming algorithm state (Standard PPO Version)"""
        super().load_state_dict(state_dict) # Load model states from base

        # Load optimizer states carefully
        try:
            if 'actor_optimizer' in state_dict:
                self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
            # Load critic optimizer only if it exists in state_dict (i.e., was saved separately)
            if 'critic_optimizer' in state_dict and self.critic_optimizer is not self.actor_optimizer:
                self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
            # If only combined optimizer was saved previously, load it into both if they are the same now
            elif 'optimizer' in state_dict and self.actor_optimizer is self.critic_optimizer:
                 self.actor_optimizer.load_state_dict(state_dict['optimizer'])

        except ValueError as e:
             print(f"Warning: Could not load optimizer state, likely due to parameter mismatch. Optimizer state reset. Error: {e}")
             # Re-initialize optimizer(s) - requires knowing the original LRs stored elsewhere or in config
             # This part might need adjustment based on how LRs are managed.


        if 'scaler' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler'])
        # if 'memory_state' in state_dict and hasattr(self.memory, 'load_state_dict'): # Memory usually not loaded
        #     self.memory.load_state_dict(state_dict['memory_state'])
        if 'episode_returns' in state_dict:
            self.episode_returns = deque(state_dict['episode_returns'], maxlen=self.episode_returns.maxlen)
        if 'current_episode_rewards' in state_dict:
            self.current_episode_rewards = state_dict['current_episode_rewards']

        # Load aux task manager state if needed
        if 'aux_task_manager' in state_dict and self.aux_task_manager and hasattr(self.aux_task_manager, 'load_state_dict'):
            self.aux_task_manager.load_state_dict(state_dict['aux_task_manager'])

        # Removed loading of non-standard params
