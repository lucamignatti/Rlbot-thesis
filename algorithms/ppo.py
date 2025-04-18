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
        self.memory = self.PPOMemory(
            batch_size=batch_size,
            buffer_size=10000,
            device=device,
            debug=debug,
            action_space_type=self.action_space_type # Pass action space type
        )

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

        def __init__(self, batch_size, buffer_size, device, debug=False, action_space_type="discrete"):
            self.batch_size = batch_size
            self.debug = debug
            self.action_space_type = action_space_type # Store action space type

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

        def _initialize_buffers_if_needed(self, obs_sample, action_sample):
            """Initialize buffer tensors based on sample shapes if not already done."""
            if self.obs is None:
                if not isinstance(obs_sample, torch.Tensor):
                    obs_sample = torch.tensor(obs_sample, dtype=torch.float32)
                if not isinstance(action_sample, torch.Tensor):
                    # Determine dtype based on action space type
                    action_dtype = torch.long if self.action_space_type == "discrete" else torch.float32
                    action_sample = torch.tensor(action_sample, dtype=action_dtype)

                # Ensure samples have batch dimension for shape extraction
                if obs_sample.dim() == 1: obs_sample = obs_sample.unsqueeze(0)

                # Determine action shape and dtype based on action space type
                if self.action_space_type == "discrete":
                    action_shape = () # Store indices, shape is just (buffer_size,)
                    action_dtype = torch.long
                else: # Continuous
                    # Handle scalar actions vs vector actions
                    if action_sample.numel() == 1 and action_sample.dim() == 0:
                        action_shape = (1,) # Treat scalar as vector of size 1
                    elif action_sample.dim() == 1:
                        action_shape = action_sample.shape # Keep original shape for vector actions
                    elif action_sample.dim() > 1:
                        action_shape = action_sample.shape[1:] # Use shape without batch dim
                    else:
                        action_shape = (1,) # Fallback
                    action_dtype = torch.float32

                obs_shape = obs_sample.shape[1:]

                if self.debug:
                    print(f"[DEBUG PPOMemory] Initializing buffers: obs_shape={obs_shape}, action_shape={action_shape}, action_dtype={action_dtype}")

                if self.use_device_tensors:
                    self.obs = torch.zeros((self.buffer_size, *obs_shape), dtype=torch.float32, device=self.device)
                    self.actions = torch.zeros((self.buffer_size, *action_shape), dtype=action_dtype, device=self.device)
                else:
                    # For CPU, initialize with torch tensors first, then convert if needed?
                    # Let's keep them as torch tensors for consistency, even on CPU.
                    self.obs = torch.zeros((self.buffer_size, *obs_shape), dtype=torch.float32)
                    self.actions = torch.zeros((self.buffer_size, *action_shape), dtype=action_dtype)


        def store_initial_batch(self, obs_batch, action_batch, log_prob_batch, value_batch):
            """Store the initial part of experiences in batch."""
            batch_size = obs_batch.shape[0]
            if batch_size == 0:
                return torch.tensor([], dtype=torch.long, device=self.device)

            # Determine expected action dtype
            expected_action_dtype = torch.long if self.action_space_type == "discrete" else torch.float32

            # Initialize buffers on first call
            # Pass a sample action with the expected dtype for initialization
            action_sample_for_init = action_batch[0].clone().to(expected_action_dtype)
            self._initialize_buffers_if_needed(obs_batch[0], action_sample_for_init)

            # Ensure batches are tensors and on the correct device
            if not isinstance(obs_batch, torch.Tensor): obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
            # Convert action_batch to the expected dtype
            if not isinstance(action_batch, torch.Tensor):
                action_batch = torch.tensor(action_batch, dtype=expected_action_dtype)
            elif action_batch.dtype != expected_action_dtype:
                 if self.debug: print(f"[DEBUG PPOMemory] Casting action_batch from {action_batch.dtype} to {expected_action_dtype}")
                 action_batch = action_batch.to(expected_action_dtype)

            if not isinstance(log_prob_batch, torch.Tensor): log_prob_batch = torch.tensor(log_prob_batch, dtype=torch.float32)
            if not isinstance(value_batch, torch.Tensor): value_batch = torch.tensor(value_batch, dtype=torch.float32)

            target_device = self.device # Always use the target device now
            obs_batch = obs_batch.to(target_device)
            action_batch = action_batch.to(target_device) # Already correct dtype
            log_prob_batch = log_prob_batch.to(target_device)
            value_batch = value_batch.to(target_device)

            # Determine indices to store into
            indices = torch.arange(self.pos, self.pos + batch_size, device=target_device) % self.buffer_size

            # Store data
            self.obs.index_copy_(0, indices, obs_batch.detach())

            # Store actions - handle shape for discrete vs continuous
            if self.action_space_type == "discrete":
                # Action batch should be (batch_size,), buffer is (buffer_size,)
                if action_batch.dim() == 1 and self.actions.dim() == 1:
                    self.actions.index_copy_(0, indices, action_batch.detach())
                elif action_batch.dim() == 2 and action_batch.shape[1] == 1 and self.actions.dim() == 1: # If action batch is (batch_size, 1)
                    self.actions.index_copy_(0, indices, action_batch.detach().squeeze(1))
                else:
                     if self.debug: print(f"[DEBUG PPOMemory] Discrete action shape mismatch. Buffer: {self.actions.shape}, Batch: {action_batch.shape}")
            else: # Continuous
                # Action batch shape should match buffer shape (excluding buffer_size dim)
                if self.actions.shape[1:] == action_batch.shape[1:]:
                    self.actions.index_copy_(0, indices, action_batch.detach())
                else:
                     if self.debug: print(f"[DEBUG PPOMemory] Continuous action shape mismatch. Buffer: {self.actions.shape}, Batch: {action_batch.shape}")


            self.log_probs.index_copy_(0, indices, log_prob_batch.detach())
            self.values.index_copy_(0, indices, value_batch.detach()) # Store placeholder values

            # Update position and size
            new_pos = (self.pos + batch_size) % self.buffer_size
            if not self.full and new_pos < self.pos: # Wrapped around
                self.full = True
            self.pos = new_pos
            self.size = self.buffer_size if self.full else self.pos

            if self.debug and batch_size > 0:
                 print(f"[DEBUG PPOMemory] Stored initial batch of size {batch_size}. New pos: {self.pos}, size: {self.size}")


            return indices # Return the indices where data was stored

        def update_rewards_dones_batch(self, indices, rewards_batch, dones_batch):
            """Update rewards and dones for experiences at given indices."""
            if len(indices) == 0:
                return

            # Ensure batches are tensors and on the correct device
            if not isinstance(rewards_batch, torch.Tensor): rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
            if not isinstance(dones_batch, torch.Tensor): dones_batch = torch.tensor(dones_batch, dtype=torch.bool)

            target_device = self.device # Use target device
            indices = indices.to(target_device) # Ensure indices are on the right device
            rewards_batch = rewards_batch.to(target_device)
            dones_batch = dones_batch.to(target_device)

            # Update data using index_copy_ for potential efficiency
            self.rewards.index_copy_(0, indices, rewards_batch)
            self.dones.index_copy_(0, indices, dones_batch)

            if self.debug and len(indices) > 0:
                print(f"[DEBUG PPOMemory] Updated rewards/dones for {len(indices)} indices. First reward: {rewards_batch[0].item():.4f}")


        def store(self, obs, action, log_prob, reward, value, done):
            """Store a single experience in the buffer (legacy, less efficient)"""
            # Determine expected action dtype
            expected_action_dtype = torch.long if self.action_space_type == "discrete" else torch.float32

            # Use store_initial_batch for single item
            obs_b = torch.tensor([obs], dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs.unsqueeze(0)
            # Convert single action to tensor with correct dtype
            act_b = torch.tensor([action], dtype=expected_action_dtype) if not isinstance(action, torch.Tensor) else action.unsqueeze(0).to(expected_action_dtype)
            lp_b = torch.tensor([log_prob], dtype=torch.float32) if not isinstance(log_prob, torch.Tensor) else log_prob.unsqueeze(0)
            val_b = torch.tensor([value], dtype=torch.float32) if not isinstance(value, torch.Tensor) else value.unsqueeze(0)
            rew_b = torch.tensor([reward], dtype=torch.float32) if not isinstance(reward, torch.Tensor) else reward.unsqueeze(0)
            done_b = torch.tensor([done], dtype=torch.bool) if not isinstance(done, torch.Tensor) else done.unsqueeze(0)

            indices = self.store_initial_batch(obs_b, act_b, lp_b, val_b)
            if len(indices) > 0:
                self.update_rewards_dones_batch(indices, rew_b, done_b)


        def store_experience_at_idx(self, idx, state=None, action=None, log_prob=None, reward=None, value=None, done=None):
            """Update only specific values at an index, rather than a complete experience."""
            # Use batch update methods for single index
            indices = torch.tensor([idx], dtype=torch.long, device=self.device)

            if reward is not None or done is not None:
                rewards_b = torch.tensor([reward if reward is not None else self.rewards[idx]], dtype=torch.float32)
                dones_b = torch.tensor([done if done is not None else self.dones[idx]], dtype=torch.bool)
                self.update_rewards_dones_batch(indices, rewards_b, dones_b)

            # Update others individually if needed (less common now)
            target_device = self.device
            if state is not None and self.obs is not None:
                if not isinstance(state, torch.Tensor): state = torch.tensor(state, dtype=torch.float32)
                self.obs[idx] = state.detach().to(target_device)

            if action is not None and self.actions is not None:
                expected_action_dtype = torch.long if self.action_space_type == "discrete" else torch.float32
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=expected_action_dtype)
                elif action.dtype != expected_action_dtype:
                    action = action.to(expected_action_dtype)
                # Adjust shape for discrete index storage
                if self.action_space_type == "discrete" and action.dim() > 0:
                     action = action.squeeze() # Ensure it's a scalar index if buffer expects it
                self.actions[idx] = action.detach().to(target_device)


            if log_prob is not None and self.log_probs is not None:
                if not isinstance(log_prob, torch.Tensor): log_prob = torch.tensor(log_prob, dtype=torch.float32)
                self.log_probs[idx] = log_prob.detach().to(target_device)

            if value is not None and self.values is not None:
                if not isinstance(value, torch.Tensor): value = torch.tensor(value, dtype=torch.float32)
                self.values[idx] = value.detach().to(target_device)


        def get(self):
            """Get all data currently stored in the buffer."""
            if self.size == 0 or self.obs is None:
                if self.debug:
                    print("[DEBUG PPOMemory] get() called but buffer is empty or uninitialized.")
                return None, None, None, None, None, None

            # Data is already stored as torch tensors on the correct device
            obs_data = self.obs[:self.size]
            actions_data = self.actions[:self.size]
            log_probs_data = self.log_probs[:self.size]
            rewards_data = self.rewards[:self.size]
            values_data = self.values[:self.size] # These are placeholder values
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

            # Generate random indices on the correct device
            indices = torch.randperm(self.size, device=self.device)

            # Create batches
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
            # Don't reallocate tensors, just reset position and size
            self.pos = 0
            self.size = 0
            self.full = False
            # Optionally zero out tensors if memory usage is a concern,
            # but usually overwriting is sufficient.
            # if self.obs is not None: self.obs.zero_()
            # if self.actions is not None: self.actions.zero_()
            # self.log_probs.zero_()
            # self.rewards.zero_()
            # self.values.zero_()
            # self.dones.zero_()
            if self.debug:
                print("[DEBUG PPOMemory] Buffer cleared.")


    def store_initial_batch(self, obs_batch, action_batch, log_prob_batch, value_batch):
        """Store the initial part of experiences (obs, action, log_prob, value) in batch."""
        if self.debug:
            print(f"[DEBUG PPO] Storing initial batch of size {obs_batch.shape[0]}")
        return self.memory.store_initial_batch(obs_batch, action_batch, log_prob_batch, value_batch)

    def update_rewards_dones_batch(self, indices, rewards_batch, dones_batch):
        """Update rewards and dones for experiences at given indices in batch."""
        if self.debug:
            print(f"[DEBUG PPO] Updating rewards/dones for batch of size {len(indices)}")
        self.memory.update_rewards_dones_batch(indices, rewards_batch, dones_batch)

        # Track episode returns when dones are received
        done_indices_local = torch.where(dones_batch.cpu())[0] # Find dones on CPU
        if len(done_indices_local) > 0:
            # This part is tricky with batching, as we don't have the full episode rewards easily accessible here.
            # We rely on the main loop's tracking for mean episode reward calculation.
            # We can still track the number of completed episodes within this update cycle if needed.
            if self.debug:
                 print(f"[DEBUG PPO] {len(done_indices_local)} episodes ended in this batch update.")
            pass


    def store_experience(self, obs, action, log_prob, reward, value, done):
        """
        Store experience in the buffer (legacy single experience method)

        Since we no longer compute value during inference, the value parameter here
        will be a dummy/placeholder (typically zeros). The actual value estimates
        will be computed in batch during the update step.
        """
        if self.debug:
            print(f"[DEBUG PPO] Storing single experience - reward: {reward}")

        # Forward to memory buffer - with placeholder value
        self.memory.store(obs, action, log_prob, reward, value, done)

        # Track episode rewards for calculating returns (still useful for single-env or debugging)
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
        """Update specific values of an experience at a given index (legacy)"""
        if self.debug and reward is not None:
            print(f"[DEBUG PPO] Updating reward at idx {idx}: {reward}")

        # Forward to memory buffer
        self.memory.store_experience_at_idx(idx, obs, action, log_prob, reward, value, done)

    def get_action(self, obs, deterministic=False, return_features=False):
        """Get action for a given observation (using only the actor network)"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if return_features:
                action_dist_output, actor_features = self.actor(obs, return_features=True)
                features = actor_features  # Only use actor features
            else:
                action_dist_output = self.actor(obs)
                features = None # Ensure features is defined
                # No critic call during inference

            if deterministic:
                if self.action_space_type == "discrete":
                    # Assuming actor outputs probabilities for discrete actions
                    probs = action_dist_output
                    action = torch.argmax(probs, dim=-1) # Get index of max probability (Long)
                    # Calculate log_prob of the chosen action index
                    log_prob = torch.log(torch.gather(probs, -1, action.unsqueeze(-1)).squeeze(-1) + 1e-10)
                else: # Continuous
                    action_dist = action_dist_output # Assume output is already a distribution object
                    action = action_dist.loc # Mean action (Float)
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
            else: # Sample action
                if self.action_space_type == "discrete":
                    probs = action_dist_output
                    # Ensure probs are valid before creating Categorical distribution
                    probs = torch.clamp(probs, min=1e-10) # Prevent zeros
                    probs = probs / probs.sum(dim=-1, keepdim=True) # Normalize
                    try:
                        dist = torch.distributions.Categorical(probs=probs)
                        action = dist.sample() # Sample action index (Long)
                        log_prob = dist.log_prob(action) # Log prob of the sampled index
                    except ValueError as e:
                         if self.debug:
                             print(f"[DEBUG PPO get_action] Error creating Categorical distribution: {e}")
                             print(f"Probs shape: {probs.shape}, Probs sum: {probs.sum(dim=-1)}")
                             print(f"Probs sample: {probs[0]}")
                         # Fallback: deterministic action index
                         action = torch.argmax(probs, dim=-1)
                         log_prob = torch.log(torch.gather(probs, -1, action.unsqueeze(-1)).squeeze(-1) + 1e-10)
                else: # Continuous
                    action_dist = action_dist_output # Assume output is already a distribution object
                    action = action_dist.sample() # Float
                    log_prob = action_dist.log_prob(action).sum(dim=-1)

        # Create dummy value tensor (zeros) with the same batch size and device
        dummy_value = torch.zeros_like(log_prob)

        # No need to move to CPU unless specifically required by environment later
        # The main loop handles CPU conversion for vec_env.step

        if return_features:
            return action, log_prob, dummy_value, features
        else:
            return action, log_prob, dummy_value


    def reset(self):
        """Reset memory"""
        self.memory.clear()

    def update(self):
        """Update policy using PPO"""
        buffer_size = self.memory.size

        if buffer_size < self.batch_size: # Don't update if buffer has less than one batch
            if self.debug:
                print(f"[DEBUG PPO] Buffer size ({buffer_size}) < batch size ({self.batch_size}), skipping update")
            return self.metrics # Return previous metrics

        if self.debug:
            print(f"[DEBUG PPO] Starting update with buffer size: {buffer_size}")


        # Get experiences from buffer
        states, actions, old_log_probs, rewards, values_placeholder, dones = self.memory.get()

        if states is None:
            if self.debug:
                print("[DEBUG PPO] Failed to get experiences from buffer, skipping update")
            return self.metrics

        # Since we're using dummy values during get_action, we need to calculate
        # the actual value estimates in batch here
        with torch.no_grad():
            # Calculate values in smaller chunks if buffer is large to avoid OOM
            values = []
            for i in range(0, buffer_size, self.batch_size):
                 chunk_states = states[i:min(i + self.batch_size, buffer_size)]
                 chunk_values = self.critic(chunk_states).squeeze()
                 values.append(chunk_values)
            values = torch.cat(values)


        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, values, dones)

        # Check if advantages has NaNs
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            if self.debug:
                print("[DEBUG PPO] NaN or Inf detected in advantages, skipping update")
                print(f"Advantages mean: {advantages.mean()}, std: {advantages.std()}")
                print(f"Rewards mean: {rewards.mean()}, std: {rewards.std()}")
                print(f"Values mean: {values.mean()}, std: {values.std()}")
            # Clear memory even if update is skipped due to NaNs/Infs
            self.memory.clear()
            return self.metrics # Return previous metrics to avoid logging NaNs

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

        if self.debug:
            print(f"[DEBUG PPO] Update finished. Actor Loss: {metrics.get('actor_loss', 0):.4f}, Critic Loss: {metrics.get('critic_loss', 0):.4f}")


        return self.metrics

    def _compute_gae(self, rewards, values, dones):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE)

        Args:
            rewards: rewards tensor [buffer_size]
            values: value predictions tensor [buffer_size] (calculated in batch during update)
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
        next_value = 0.0 # Default if buffer_size is 0 or last state is done
        if buffer_size > 0:
             # Ensure values tensor is not empty before accessing
             if values.numel() > 0:
                 next_value = 0.0 if dones[last_idx] else values[last_idx].item()
             else:
                 next_value = 0.0 # Handle case where values tensor might be empty


        next_advantage = 0.0

        # Compute GAE for each timestep, going backwards
        for t in reversed(range(buffer_size)):
            # If this is the end of an episode, next_value is 0
            # This logic seems redundant with the (1 - dones[t].float()) multiplier below
            # Let's rely on the multiplier
            current_value = values[t]
            current_reward = rewards[t]
            current_done = dones[t].float() # Convert bool to float (0.0 or 1.0)

            # Calculate TD error: r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            # next_value is V(s_{t+1}) from the previous iteration (or V(s_last) initially)
            delta = current_reward + self.gamma * next_value * (1.0 - current_done) - current_value

            # Compute GAE advantage: delta_t + gamma * lambda * A_{t+1} * (1 - done_t)
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * (1.0 - current_done)

            # Update next_advantage and next_value for the next iteration (t-1)
            next_advantage = advantages[t]
            next_value = current_value # V(s_t) becomes V(s_{t+1}) for the next step back

            # Compute returns as advantage + value (TD(lambda) return)
            # GAE paper: A_t = R_t - V(s_t), so R_t = A_t + V(s_t)
            returns[t] = advantages[t] + values[t]


        # Normalize advantages across the entire buffer
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        if self.debug:
            print(f"[DEBUG PPO _compute_gae] Returns mean: {returns.mean():.4f}, std: {returns.std():.4f}")
            print(f"[DEBUG PPO _compute_gae] Advantages mean: {adv_mean:.4f}, std: {adv_std:.4f}")


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
            actions: batch of actions [buffer_size, action_dim or buffer_size] (dtype depends on space)
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
            # Calculate values in chunks if needed
            y_pred = []
            buffer_size = states.shape[0]
            for i in range(0, buffer_size, self.batch_size):
                 chunk_states = states[i:min(i + self.batch_size, buffer_size)]
                 chunk_values = self.critic(chunk_states).squeeze()
                 y_pred.append(chunk_values)
            y_pred = torch.cat(y_pred)

        y_true = returns
        var_y = torch.var(y_true)
        explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
        update_metrics['explained_variance'] = explained_var.item()
        update_metrics['mean_advantage'] = advantages.mean().item() # Use normalized advantage mean
        update_metrics['kl_divergence'] = 0.0 # Will be averaged later

        total_weight_clipped_fraction_epoch = 0.0
        num_batches_processed = 0

        # Multiple epochs of PPO update
        for epoch in range(self.ppo_epochs):
            # Generate random batches
            batch_indices = self.memory.generate_batches()

            # Skip if no batches
            if not batch_indices:
                if self.debug:
                    print(f"[DEBUG PPO _update_policy] Epoch {epoch}: No batches generated, skipping.")
                continue

            if self.debug:
                 print(f"[DEBUG PPO _update_policy] Epoch {epoch}: Processing {len(batch_indices)} batches.")


            # Process each batch
            for batch_idx in batch_indices:
                num_batches_processed += 1
                # Get batch data
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx] # Dtype depends on action space
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Get current policy distribution and values
                if self.action_space_type == "discrete":
                    # For discrete actions
                    action_probs = self.actor(batch_states)
                    action_probs = torch.clamp(action_probs, min=1e-10) # Prevent zeros
                    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True) # Normalize

                    try:
                        dist = torch.distributions.Categorical(probs=action_probs)
                    except ValueError as e:
                         if self.debug:
                             print(f"[DEBUG PPO _update_policy] Error creating Categorical distribution in batch: {e}")
                             print(f"Probs shape: {action_probs.shape}, Probs sum: {action_probs.sum(dim=-1)}")
                             print(f"Probs sample: {action_probs[0]}")
                         # Skip this batch if distribution is invalid
                         continue

                    # Calculate log probabilities and entropy
                    # batch_actions should be Long indices here
                    if batch_actions.dtype != torch.long:
                         if self.debug: print(f"[DEBUG PPO _update_policy] Warning: batch_actions dtype is {batch_actions.dtype}, expected Long.")
                         # Attempt to cast, but this might indicate an earlier issue
                         try:
                             actions_indices = batch_actions.long()
                         except RuntimeError:
                             if self.debug: print("[DEBUG PPO _update_policy] Failed to cast batch_actions to Long.")
                             continue # Skip batch
                    else:
                         actions_indices = batch_actions

                    # Ensure indices are within the valid range for the distribution
                    if actions_indices.max() >= dist.probs.shape[-1] or actions_indices.min() < 0:
                         if self.debug:
                             print(f"[DEBUG PPO _update_policy] Invalid action indices found.")
                             print(f"Indices shape: {actions_indices.shape}, max: {actions_indices.max()}, min: {actions_indices.min()}")
                             print(f"Probs shape: {dist.probs.shape}")
                         continue # Skip batch


                    try:
                        curr_log_probs = dist.log_prob(actions_indices)
                        entropy = dist.entropy().mean()
                    except (IndexError, ValueError) as e:
                         if self.debug:
                             print(f"[DEBUG PPO _update_policy] Error calculating log_prob/entropy: {e}")
                             print(f"Action indices shape: {actions_indices.shape}, max index: {actions_indices.max()}, min index: {actions_indices.min()}")
                             print(f"Probs shape: {action_probs.shape}")
                         continue # Skip batch

                else:
                    # For continuous actions
                    # batch_actions should be Float
                    if batch_actions.dtype != torch.float:
                         if self.debug: print(f"[DEBUG PPO _update_policy] Warning: batch_actions dtype is {batch_actions.dtype}, expected Float.")
                         try:
                             batch_actions = batch_actions.float()
                         except RuntimeError:
                             if self.debug: print("[DEBUG PPO _update_policy] Failed to cast batch_actions to Float.")
                             continue # Skip batch

                    action_dist = self.actor(batch_states) # Assume actor returns distribution object
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
        else:
             if self.debug:
                 print("[DEBUG PPO _update_policy] No batches were processed in any epoch.")


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
