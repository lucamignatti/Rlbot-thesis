import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import math
from typing import Dict, Tuple, List, Optional, Union, Any
from .base import BaseAlgorithm
# Import AuxiliaryTaskManager to use its methods
from auxiliary import AuxiliaryTaskManager
# Import GradScaler for AMP
from torch.amp import GradScaler, autocast
# Import l2_norm for SimbaV2 weight projection
from model_architectures.utils import l2_norm

class PPOAlgorithm(BaseAlgorithm):
    """PPO algorithm implementation"""

    def __init__(
        self,
        actor,
        critic,
        # Add aux_task_manager parameter
        aux_task_manager: Optional[AuxiliaryTaskManager] = None,
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
        v_min=None,
        v_max=None,
        num_atoms=None,
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
            use_amp=use_amp, # Pass use_amp to base class
            debug=debug,
            use_wandb=use_wandb,
        )

        # Store the auxiliary task manager
        self.aux_task_manager = aux_task_manager
        # Check if actor and critic are the same instance
        self.shared_model = (actor is critic)
        if self.debug and self.shared_model:
            print("[DEBUG PPO] Actor and Critic are the same instance (shared model).")

        # --- ADD DISTRIBUTIONAL CRITIC PARAMS ---
        # Default values, should be configurable via args in main.py
        self.v_min = -10.0 if v_min is None else v_min
        self.v_max = 10.0 if v_max is None else v_max
        self.num_atoms = 51 if num_atoms is None else num_atoms
        # Number of support atoms for the value distribution
        # Precompute support atoms (for efficiency)
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        # -----------------------------------------


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
            # Increase buffer size to match update_interval typically used
            buffer_size=131072, # Example size, adjust if needed
            device=device,
            debug=debug,
            action_space_type=self.action_space_type # Pass action space type
        )

        # Initialize optimizer
        # Combine parameters for a single optimizer step
        # If shared model, actor parameters cover everything.
        # If separate, combine actor and critic parameters.
        if self.shared_model:
            combined_params = list(self.actor.parameters())
        else:
            combined_params = list(self.actor.parameters()) + list(self.critic.parameters())

        # Add auxiliary task parameters if they exist
        if self.aux_task_manager:
            if hasattr(self.aux_task_manager, 'sr_task') and self.aux_task_manager.sr_task is not None:
                combined_params += list(self.aux_task_manager.sr_task.parameters())
            if hasattr(self.aux_task_manager, 'rp_task') and self.aux_task_manager.rp_task is not None:
                combined_params += list(self.aux_task_manager.rp_task.parameters())

        # Use actor LR for the combined optimizer (or a separate LR if needed)
        self.optimizer = torch.optim.Adam(combined_params, lr=lr_actor)

        # Initialize GradScaler if AMP is enabled
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking metrics
        self.metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy_loss': 0.0,
            'sr_loss_scalar': 0.0, # Add aux metrics
            'rp_loss_scalar': 0.0, # Add aux metrics
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
            # Removed use_device_tensors, always use torch tensors on self.device
            # self.use_device_tensors = device != "cpu"

            # Initialize buffers as empty tensors
            self._reset_buffers()

        def _reset_buffers(self):
            """Initialize all buffer tensors with the correct shapes"""
            buffer_size = self.buffer_size
            device = self.device
            # use_device_tensors = self.use_device_tensors # Removed

            # Determine tensor type based on device - Always use torch tensors now
            # if use_device_tensors: # Removed
            # Initialize empty tensors on the specified device
            self.obs = None  # Will be initialized on first store() call
            self.actions = None  # Will be initialized on first store() call
            self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
            # else: # Removed
            #     # Initialize empty numpy arrays for CPU
            #     self.obs = None  # Will be initialized on first store() call
            #     self.actions = None  # Will be initialized on first store() call
            #     self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
            #     self.rewards = np.zeros((buffer_size,), dtype=np.float32)
            #     self.values = np.zeros((buffer_size,), dtype=np.float32)
            #     self.dones = np.zeros((buffer_size,), dtype=np.bool_)

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

                # Always use torch tensors on the specified device
                # if self.use_device_tensors: # Removed
                self.obs = torch.zeros((self.buffer_size, *obs_shape), dtype=torch.float32, device=self.device)
                self.actions = torch.zeros((self.buffer_size, *action_shape), dtype=action_dtype, device=self.device)
                # else: # Removed
                #     # For CPU, initialize with torch tensors first, then convert if needed?
                #     # Let's keep them as torch tensors for consistency, even on CPU.
                #     self.obs = torch.zeros((self.buffer_size, *obs_shape), dtype=torch.float32)
                #     self.actions = torch.zeros((self.buffer_size, *action_shape), dtype=action_dtype)


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
            # Process each completed episode in the batch
            for i in done_indices_local:
                # Get the reward for this episode completion
                if isinstance(rewards_batch, torch.Tensor):
                    reward_value = rewards_batch[i].item()
                else:
                    reward_value = float(rewards_batch[i])

                # Add to current episode's rewards
                self.current_episode_rewards.append(reward_value)

                # Calculate episode return and add to tracking deque
                episode_return = sum(self.current_episode_rewards)
                self.episode_returns.append(episode_return)

                if self.debug:
                    print(f"[DEBUG PPO] Episode completed with return: {episode_return:.4f}")

                # Reset for next episode
                self.current_episode_rewards = []

            if self.debug:
                print(f"[DEBUG PPO] {len(done_indices_local)} episodes ended in this batch update.")


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
        """Get action for a given observation (using only the actor network/head)"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Call the actor model's forward method
            # Conditionally handle shared vs separate model forward call
            if self.shared_model:
                model_output = self.actor(obs, return_actor=True, return_critic=False, return_features=return_features)
                actor_output = model_output.get('actor_out')
                features = model_output.get('features')
            else:
                # Call separate actor model
                actor_result = self.actor(obs, return_features=return_features)
                if return_features:
                    actor_output, features = actor_result
                else:
                    actor_output = actor_result
                    features = None # Explicitly set features to None


            if actor_output is None:
                # Handle error: actor output wasn't returned
                if self.debug: print("[DEBUG PPO get_action] Error: Actor output not found in model return.")
                # Fallback or raise error? For now, return dummy values
                dummy_action = torch.zeros(obs.shape[0], dtype=torch.long if self.action_space_type == "discrete" else torch.float, device=self.device)
                dummy_log_prob = torch.zeros(obs.shape[0], device=self.device)
                dummy_value = torch.zeros_like(dummy_log_prob)
                if return_features: return dummy_action, dummy_log_prob, dummy_value, features
                else: return dummy_action, dummy_log_prob, dummy_value


            if deterministic:
                if self.action_space_type == "discrete":
                    # Assuming actor outputs probabilities for discrete actions
                    probs = actor_output
                    action = torch.argmax(probs, dim=-1) # Get index of max probability (Long)
                    # Calculate log_prob of the chosen action index
                    log_prob = torch.log(torch.gather(probs, -1, action.unsqueeze(-1)).squeeze(-1) + 1e-10)
                else: # Continuous
                    # --- MODIFICATION START ---
                    # Actor output is raw tensor (mean, log_std)
                    mean, log_std = actor_output.chunk(2, dim=-1)
                    # Use mean for deterministic action
                    action = mean 
                    # Calculate log_prob for the mean action
                    log_std_clamped = torch.clamp(log_std, min=-20, max=2)
                    std = torch.exp(log_std_clamped)
                    action_dist = torch.distributions.Normal(mean, std)
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
                    # --- MODIFICATION END ---
            else: # Sample action
                if self.action_space_type == "discrete":
                    probs = actor_output
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
                    # --- MODIFICATION START ---
                    # Actor output is raw tensor (mean, log_std)
                    mean, log_std = actor_output.chunk(2, dim=-1)
                    log_std = torch.clamp(log_std, min=-20, max=2) # Common practice
                    std = torch.exp(log_std)
                    action_dist = torch.distributions.Normal(mean, std) # Create the distribution here
                    action = action_dist.sample() # Sample from the created distribution
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
                    # --- MODIFICATION END ---

        # Create dummy value tensor (zeros) with the same batch size and device
        dummy_value = torch.zeros_like(log_prob)

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
            # Return previous metrics, ensuring aux losses are included if manager exists
            if self.aux_task_manager:
                self.metrics['sr_loss_scalar'] = self.aux_task_manager.last_sr_loss
                self.metrics['rp_loss_scalar'] = self.aux_task_manager.last_rp_loss
            return self.metrics

        if self.debug:
            print(f"[DEBUG PPO] Starting update with buffer size: {buffer_size}")


        # Get experiences from buffer
        # IMPORTANT: 'values_placeholder' here is unused as we compute critic output below
        states, actions, old_log_probs, rewards, values_placeholder, dones = self.memory.get()

        if states is None:
            if self.debug:
                print("[DEBUG PPO] Failed to get experiences from buffer, skipping update")
            return self.metrics

        # --- CALCULATE EXPECTED VALUES FOR GAE ---
        # Calculate the expected value estimates in batch here for GAE calculation
        with torch.no_grad():
            expected_values_list = []
            for i in range(0, buffer_size, self.batch_size):
                chunk_states = states[i:min(i + self.batch_size, buffer_size)]
                # Use autocast for critic evaluation if AMP is enabled
                with autocast("cuda", enabled=self.use_amp):
                    # Request critic logits - handles shared vs separate
                    if self.shared_model:
                        model_output = self.critic(chunk_states, return_actor=False, return_critic=True, return_features=False)
                        # Ensure critic_out has shape [chunk_size, num_atoms]
                        chunk_logits = model_output.get('critic_out', torch.zeros(chunk_states.shape[0], self.num_atoms, device=self.device))
                    else:
                        # Ensure critic model returns logits [chunk_size, num_atoms]
                        chunk_logits = self.critic(chunk_states)

                    # --- Critical Shape Check ---
                    if chunk_logits.shape != (chunk_states.shape[0], self.num_atoms):
                         if self.debug:
                             print(f"[DEBUG PPO GAE] Critic output shape mismatch in GAE calc. Expected {(chunk_states.shape[0], self.num_atoms)}, got {chunk_logits.shape}. Skipping update.")
                         self.memory.clear() # Clear memory as GAE calc failed
                         return self.metrics # Return previous metrics

                    # Calculate expected value: E[V] = sum(softmax(logits) * support)
                    chunk_probs = F.softmax(chunk_logits, dim=1)
                    # Ensure support is on the same device
                    support_device = self.support.to(chunk_probs.device)
                    chunk_expected_values = (chunk_probs * support_device).sum(dim=1) # Shape: [chunk_size]

                expected_values_list.append(chunk_expected_values)

            # Concatenate expected values for the entire buffer
            expected_values_for_gae = torch.cat(expected_values_list) # Shape: [buffer_size]
        # --- END EXPECTED VALUES CALCULATION ---


        # Compute returns and advantages using EXPECTED values
        # Pass expected_values_for_gae instead of the raw critic output
        returns, advantages = self._compute_gae(rewards, expected_values_for_gae, dones)

        # Check if advantages has NaNs
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            if self.debug:
                print("[DEBUG PPO] NaN or Inf detected in advantages, skipping update")
                print(f"Advantages mean: {advantages.mean()}, std: {advantages.std()}")
                print(f"Rewards mean: {rewards.mean()}, std: {rewards.std()}")
                # Print expected values stats instead of raw critic output
                print(f"Expected Values mean: {expected_values_for_gae.mean()}, std: {expected_values_for_gae.std()}")
            # Clear memory even if update is skipped due to NaNs/Infs
            self.memory.clear()
            return self.metrics # Return previous metrics to avoid logging NaNs

        # Update the policy and value networks using PPO
        # Pass scalar returns and advantages, aux loss uses original rewards
        metrics = self._update_policy(states, actions, old_log_probs, returns, advantages, rewards)

        # Clear the memory buffer after using the data for updates
        self.memory.clear()

        # Update the metrics dictionary with results from _update_policy
        self.metrics.update(metrics)

        # If we have episode returns, update the mean return metric
        if len(self.episode_returns) > 0:
            self.metrics['mean_return'] = sum(self.episode_returns) / len(self.episode_returns)

        # Increment update counter for adaptive kappa
        self._update_counter += 1

        if self.debug:
            # Log expected value stats if available
            ev_mean = expected_values_for_gae.mean().item() if 'expected_values_for_gae' in locals() else float('nan')
            ev_std = expected_values_for_gae.std().item() if 'expected_values_for_gae' in locals() else float('nan')
            print(f"[DEBUG PPO] Update finished. Actor Loss: {metrics.get('actor_loss', 0):.4f}, Critic Loss: {metrics.get('critic_loss', 0):.4f}, "
                  f"SR Loss: {metrics.get('sr_loss_scalar', 0):.4f}, RP Loss: {metrics.get('rp_loss_scalar', 0):.4f}, "
                  f"ExpValue Mean: {ev_mean:.4f}, ExpValue Std: {ev_std:.4f}")


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
                 # Use value prediction V(s_last) if not done, otherwise 0
                 # Use .item() here as it's outside the main loop and needed for scalar logic
                 next_value = values[last_idx].item() * (1.0 - dones[last_idx].float().item())
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
            # next_value needs to be the scalar value for the next iteration's calculation
            next_value = current_value.item() # Use .item() here for the loop logic

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

        # Store base bounds for actor network weights (or shared model)
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

        # Store base bounds for critic network weights only if not shared
        if not self.shared_model:
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

        # Clip actor network weights (or shared model)
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
                # Use .item() here as it's summing over potentially many parameters
                clipped_params += torch.sum(param.data != original_param).item()

        # Clip critic network weights only if not shared
        if not self.shared_model:
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
                    # Use .item() here
                    clipped_params += torch.sum(param.data != original_param).item()

        # Return the fraction of clipped parameters
        return float(clipped_params) / total_params if total_params > 0 else 0.0


    def _update_policy(self, states, actions, old_log_probs, returns, advantages, rewards):
        """
        Update policy and value networks using PPO algorithm, including auxiliary losses.
        Handles both shared and separate actor/critic models.
        Uses a distributional critic.

        Args:
            states: batch of states [buffer_size, state_dim]
            actions: batch of actions [buffer_size, action_dim or buffer_size] (dtype depends on space)
            old_log_probs: batch of log probabilities from old policy [buffer_size]
            returns: batch of returns (target scalar values for critic) [buffer_size]
            advantages: batch of advantages [buffer_size]
            rewards: batch of rewards [buffer_size] (needed for aux tasks)

        Returns:
            dict: metrics from the update
        """
        # Track metrics for this update cycle - Initialize tensor metrics on the correct device
        update_metrics_tensors = {
            'actor_loss': torch.tensor(0.0, device=self.device),
            'critic_loss': torch.tensor(0.0, device=self.device),
            'entropy_loss': torch.tensor(0.0, device=self.device),
            'total_loss': torch.tensor(0.0, device=self.device),
            'clip_fraction': torch.tensor(0.0, device=self.device), # PPO policy clip fraction
            'kl_divergence': torch.tensor(0.0, device=self.device),
        }
        # Scalar metrics (accumulated directly)
        update_metrics_scalars = {
            'sr_loss_scalar': 0.0,
            'rp_loss_scalar': 0.0,
            'weight_clip_fraction': 0.0, # Fraction of weights clipped this update
        }

        # Calculate explained variance (once before updates) using expected value from distribution
        explained_var_scalar = 0.0
        mean_advantage_scalar = 0.0
        with torch.no_grad():
            # Calculate expected values from critic distribution in chunks
            y_pred_expected = []
            buffer_size = states.shape[0]
            for i in range(0, buffer_size, self.batch_size):
                chunk_states = states[i:min(i + self.batch_size, buffer_size)]
                with autocast("cuda", enabled=self.use_amp):
                    # Request critic logits - handles shared vs separate
                    if self.shared_model:
                        # Ensure model returns critic logits (shape: [chunk_size, num_atoms])
                        model_output = self.critic(chunk_states, return_actor=False, return_critic=True, return_features=False)
                        # Default to zeros if critic_out missing, check shape
                        chunk_logits = model_output.get('critic_out', torch.zeros(chunk_states.shape[0], self.num_atoms, device=self.device))
                    else:
                        # Ensure critic model returns logits (shape: [chunk_size, num_atoms])
                        chunk_logits = self.critic(chunk_states)

                    # Verify shape - Critical check before proceeding
                    if chunk_logits.shape != (chunk_states.shape[0], self.num_atoms):
                        if self.debug:
                            print(f"[DEBUG PPO EV] Critic output shape mismatch in EV calc. Expected {(chunk_states.shape[0], self.num_atoms)}, got {chunk_logits.shape}. Skipping EV calculation.")
                        y_pred_expected = None # Invalidate EV calculation
                        break # Exit loop if shape is wrong

                    # Calculate expected value from logits
                    chunk_probs = F.softmax(chunk_logits, dim=1)
                    # Ensure self.support is on the correct device
                    support_device = self.support.to(chunk_probs.device)
                    chunk_expected_values = (chunk_probs * support_device).sum(dim=1) # E[V] = sum(p_i * z_i)
                y_pred_expected.append(chunk_expected_values)

            if y_pred_expected is not None:
                y_pred = torch.cat(y_pred_expected) # Expected values [buffer_size]
                y_true = returns # GAE returns [buffer_size]
                var_y = torch.var(y_true)
                explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
                explained_var_scalar = explained_var.item()
            else:
                # Set EV to 0 or NaN if calculation failed due to shape mismatch
                explained_var_scalar = 0.0 # Or np.nan if preferred for logging failures
                if self.debug:
                    print("[DEBUG PPO EV] Explained variance calculation skipped due to critic shape mismatch.")


            # Use normalized advantage mean, convert to scalar here
            mean_advantage_scalar = advantages.mean().item()


        total_weight_clipped_fraction_epoch = 0.0
        num_batches_processed = 0

        # Multiple epochs of PPO update
        for epoch in range(self.ppo_epochs):
            batch_indices = self.memory.generate_batches()
            if not batch_indices:
                if self.debug: print(f"[DEBUG PPO _update_policy] Epoch {epoch}: No batches generated, skipping.")
                continue

            if self.debug: print(f"[DEBUG PPO _update_policy] Epoch {epoch}: Processing {len(batch_indices)} batches.")

            for batch_idx in batch_indices:
                num_batches_processed += 1
                # Get batch data
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx] # Target scalar returns for critic
                batch_advantages = advantages[batch_idx]
                batch_rewards = rewards[batch_idx] # For aux tasks

                # Use autocast for forward passes and loss calculation if AMP is enabled
                with autocast("cuda", enabled=self.use_amp):
                    # Get current policy distribution, features, and critic logits
                    try:
                        if self.shared_model:
                            model_output = self.actor(batch_states, return_actor=True, return_critic=True, return_features=True)
                            actor_output = model_output.get('actor_out')
                            critic_logits = model_output.get('critic_out') # Expecting logits [batch_size, num_atoms]
                            current_features = model_output.get('features')
                        else:
                            actor_result = self.actor(batch_states, return_features=True)
                            critic_logits = self.critic(batch_states) # Expecting logits [batch_size, num_atoms]
                            if isinstance(actor_result, tuple):
                                actor_output, current_features = actor_result
                            else:
                                actor_output = actor_result; current_features = None

                        # Validate outputs
                        if actor_output is None: raise ValueError("Actor output missing from model")
                        if critic_logits is None: raise ValueError("Critic output (logits) missing from model")
                        if critic_logits.shape != (batch_states.shape[0], self.num_atoms):
                             raise ValueError(f"Critic output shape mismatch. Expected {(batch_states.shape[0], self.num_atoms)}, got {critic_logits.shape}")

                        if self.debug and current_features is None and self.aux_task_manager:
                            print("[DEBUG PPO _update_policy] Warning: Features not returned by actor, cannot compute aux loss.")

                    except Exception as e:
                        if self.debug: print(f"[DEBUG PPO _update_policy] Error getting model output/features: {e}")
                        continue # Skip this batch if model forward pass fails

                    # --- Actor Loss (Standard PPO) ---
                    if self.action_space_type == "discrete":
                        action_probs = actor_output # Assume output is probs
                        # Clamp and normalize probabilities
                        action_probs = torch.clamp(action_probs, min=1e-10)
                        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                        try:
                            dist = torch.distributions.Categorical(probs=action_probs)
                        except ValueError as e:
                             if self.debug: print(f"[DEBUG PPO _update_policy] Error creating Categorical distribution in batch: {e}")
                             continue # Skip batch

                        # Ensure action indices are Long type and valid
                        if batch_actions.dtype != torch.long:
                             try: actions_indices = batch_actions.long()
                             except RuntimeError:
                                 if self.debug: print("[DEBUG PPO _update_policy] Failed to cast batch_actions to Long.")
                                 continue # Skip batch
                        else: actions_indices = batch_actions

                        if actions_indices.max() >= dist.probs.shape[-1] or actions_indices.min() < 0:
                             if self.debug: print(f"[DEBUG PPO _update_policy] Invalid action indices found (max={actions_indices.max()}, min={actions_indices.min()}, num_actions={dist.probs.shape[-1]}).")
                             continue # Skip batch

                        # Calculate log prob and entropy
                        try:
                            curr_log_probs = dist.log_prob(actions_indices)
                            entropy = dist.entropy().mean()
                        except (IndexError, ValueError) as e:
                             if self.debug: print(f"[DEBUG PPO _update_policy] Error calculating log_prob/entropy: {e}")
                             continue # Skip batch

                    else: # Continuous action space
                        if batch_actions.dtype != torch.float:
                             try: batch_actions = batch_actions.float()
                             except RuntimeError:
                                 if self.debug: print("[DEBUG PPO _update_policy] Failed to cast batch_actions to Float.")
                                 continue # Skip batch

                        action_dist = actor_output # Assume output is distribution object
                        try:
                             curr_log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)
                             entropy = action_dist.entropy().mean()
                        except Exception as e: # Catch potential errors in distribution methods
                             if self.debug: print(f"[DEBUG PPO _update_policy] Error calculating continuous log_prob/entropy: {e}")
                             continue # Skip batch

                    # Calculate PPO actor loss components
                    ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Calculate entropy loss bonus
                    entropy_loss = -entropy * self.entropy_coef

                    # --- Distributional Critic Loss ---
                    with torch.no_grad(): # Target calculation should not require gradients
                        # Project scalar returns onto the support atoms
                        clamped_returns = batch_returns.clamp(min=self.v_min, max=self.v_max)
                        b = (clamped_returns - self.v_min) / self.delta_z
                        lower_bound_idx = b.floor().long()
                        upper_bound_idx = b.ceil().long()

                        # Calculate probabilities (weights) for lower and upper atoms
                        upper_prob = b - b.floor()
                        lower_prob = 1.0 - upper_prob

                        # Create target distribution tensor (batch_size, num_atoms)
                        target_p = torch.zeros(batch_returns.size(0), self.num_atoms, device=self.device)

                        # --- Use scatter_add_ for distributing probabilities ---
                        # Clamp indices to be safe and reshape for scatter_add
                        clamped_lower_idx = lower_bound_idx.clamp(0, self.num_atoms - 1).unsqueeze(1) # Shape: [batch_size, 1]
                        clamped_upper_idx = upper_bound_idx.clamp(0, self.num_atoms - 1).unsqueeze(1) # Shape: [batch_size, 1]

                        # Reshape probabilities for scatter_add
                        lower_prob_reshaped = lower_prob.unsqueeze(1) # Shape: [batch_size, 1]
                        upper_prob_reshaped = upper_prob.unsqueeze(1) # Shape: [batch_size, 1]

                        # Add lower probabilities
                        # scatter_add_(dimension, index_tensor, source_tensor)
                        target_p.scatter_add_(1, clamped_lower_idx, lower_prob_reshaped)
                        # Add upper probabilities - scatter_add handles accumulation if indices overlap
                        target_p.scatter_add_(1, clamped_upper_idx, upper_prob_reshaped)

                        # Handle the edge case where lower == upper *after* scattering:
                        # If an index appears in both lower and upper (because b was an integer),
                        # scatter_add will correctly sum lower_prob + upper_prob = 1 at that index.
                        # No separate handling for exact matches is needed with scatter_add_.

                        # Optional: Verify target_p sums to 1 (for debugging)
                        # if self.debug and not torch.allclose(target_p.sum(dim=1), torch.tensor(1.0, device=self.device), atol=1e-6):
                        #    print("[DEBUG PPO] Warning: Target distribution does not sum to 1.")
                        #    print(target_p.sum(dim=1))

                    # Get predicted log probabilities from critic logits
                    predicted_log_probs = F.log_softmax(critic_logits, dim=1)

                    predicted_probs = F.softmax(critic_logits, dim=1) # Need probs for KL

                    # Ensure target_p is detached and non-negative, add small epsilon for stability
                    target_p_stable = target_p.detach() + 1e-8

                    # Calculate KL Divergence: KL(target || predicted) = sum(target * log(target / predicted))
                    # Simplifies to: sum(target * (log(target) - log(predicted)))
                    # Note: predicted_log_probs = log(predicted)
                    kl_div = (target_p_stable * (torch.log(target_p_stable) - predicted_log_probs)).sum(dim=1)
                    critic_loss = kl_div.mean() # Minimize KL divergence

                    # --- Auxiliary Loss Calculation ---
                    sr_loss = torch.tensor(0.0, device=self.device) # Ensure tensor type
                    rp_loss = torch.tensor(0.0, device=self.device) # Ensure tensor type
                    sr_loss_scalar = 0.0
                    rp_loss_scalar = 0.0

                    if self.aux_task_manager is not None and current_features is not None and \
                       (self.aux_task_manager.sr_weight > 0 or self.aux_task_manager.rp_weight > 0):
                        try:
                            aux_losses = self.aux_task_manager.compute_losses_for_batch(
                                obs_batch=batch_states,
                                rewards_batch=batch_rewards, # Use original rewards for aux task? Or scaled? Check aux task needs.
                                features_batch=current_features
                            )
                            # Ensure losses are tensors
                            sr_loss = aux_losses.get("sr_loss", torch.tensor(0.0, device=self.device))
                            rp_loss = aux_losses.get("rp_loss", torch.tensor(0.0, device=self.device))
                            sr_loss_scalar = aux_losses.get("sr_loss_scalar", 0.0)
                            rp_loss_scalar = aux_losses.get("rp_loss_scalar", 0.0)

                            if self.debug and (sr_loss_scalar > 0 or rp_loss_scalar > 0):
                                print(f"[DEBUG PPO Aux] Batch Aux Losses - SR: {sr_loss_scalar:.6f}, RP: {rp_loss_scalar:.6f}")
                        except Exception as e:
                            if self.debug:
                                print(f"[DEBUG PPO Aux] Error computing aux losses for batch: {e}")
                                import traceback
                                traceback.print_exc()
                            # Reset losses if error occurs
                            sr_loss = torch.tensor(0.0, device=self.device); rp_loss = torch.tensor(0.0, device=self.device)
                            sr_loss_scalar = 0.0; rp_loss_scalar = 0.0
                    elif self.debug and self.aux_task_manager is not None and current_features is None:
                         print("[DEBUG PPO Aux] Skipping aux loss calculation as features were not obtained.")

                    # --- Total Loss ---
                    # Combine actor, distributional critic, entropy, and auxiliary losses
                    total_loss = actor_loss + (self.critic_coef * critic_loss) + entropy_loss + sr_loss + rp_loss

                # --- Optimization (outside autocast context) ---
                self.optimizer.zero_grad()
                # Scale the loss using GradScaler
                self.scaler.scale(total_loss).backward()

                # Unscale gradients before clipping (required by GradScaler)
                self.scaler.unscale_(self.optimizer)

                # Clip gradients
                if self.max_grad_norm > 0:
                    # Clip gradients for all parameters managed by the optimizer
                    # Check if optimizer has param_groups and access params correctly
                    if self.optimizer.param_groups and 'params' in self.optimizer.param_groups[0]:
                         nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.max_grad_norm)
                    elif self.debug:
                         print("[DEBUG PPO] Could not clip gradients: optimizer param_groups not found or structured as expected.")


                # Optimizer step using GradScaler
                self.scaler.step(self.optimizer)
                # Update the scaler for the next iteration
                self.scaler.update()

                # --- Project Weights (SimbaV2) ---
                self._project_weights()

                # --- Weight Clipping (Should be disabled by default via use_weight_clipping=False) ---
                current_weight_clip_fraction = self.clip_weights()
                total_weight_clipped_fraction_epoch += current_weight_clip_fraction

                # --- Update Metrics (accumulate tensors and scalars) ---
                # Detach tensors before accumulating
                update_metrics_tensors['actor_loss'] += actor_loss.detach()
                update_metrics_tensors['critic_loss'] += critic_loss.detach()
                update_metrics_tensors['entropy_loss'] += entropy_loss.detach()
                update_metrics_tensors['total_loss'] += total_loss.detach()

                # Accumulate scalar aux losses
                update_metrics_scalars['sr_loss_scalar'] += sr_loss_scalar
                update_metrics_scalars['rp_loss_scalar'] += rp_loss_scalar

                # Calculate PPO policy clipping fraction (as tensor)
                with torch.no_grad():
                    policy_clip_fraction_tensor = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    update_metrics_tensors['clip_fraction'] += policy_clip_fraction_tensor

                    # Calculate KL divergence (approximate, as tensor)
                    # Ensure curr_log_probs is detached if used here
                    kl_div_tensor = (batch_old_log_probs - curr_log_probs.detach()).mean()
                    update_metrics_tensors['kl_divergence'] += kl_div_tensor


        # --- Finalize Metrics ---
        final_metrics = {}
        if num_batches_processed > 0:
            # Average accumulated tensors and convert to scalar
            for key, tensor_val in update_metrics_tensors.items():
                final_metrics[key] = (tensor_val / num_batches_processed).item()

            # Average accumulated scalars
            for key, scalar_val in update_metrics_scalars.items():
                final_metrics[key] = scalar_val / num_batches_processed

            # Average weight clip fraction (already accumulated per batch)
            final_metrics['weight_clip_fraction'] = total_weight_clipped_fraction_epoch / num_batches_processed

        else:
             if self.debug:
                 print("[DEBUG PPO _update_policy] No batches were processed in any epoch.")
             # Initialize metrics to zero if no batches processed
             for key in list(update_metrics_tensors.keys()) + list(update_metrics_scalars.keys()):
                 final_metrics[key] = 0.0
             final_metrics['weight_clip_fraction'] = 0.0


        # Add pre-calculated metrics
        final_metrics['explained_variance'] = explained_var_scalar
        final_metrics['mean_advantage'] = mean_advantage_scalar

        # --- Adaptive Kappa Update Logic (Should not run if use_weight_clipping=False) ---
        if self.use_weight_clipping and self.adaptive_kappa and (self._update_counter % self.kappa_update_freq == 0):
            actual_clip_fraction = final_metrics['weight_clip_fraction'] # Use the averaged value
            if actual_clip_fraction > self.target_clip_fraction:
                self.weight_clip_kappa *= (1 + self.kappa_update_rate)
            elif actual_clip_fraction < self.target_clip_fraction:
                self.weight_clip_kappa *= (1 - self.kappa_update_rate)

            # Clamp kappa within bounds
            self.weight_clip_kappa = max(self.min_kappa, min(self.max_kappa, self.weight_clip_kappa))

            if self.debug:
                print(f"[DEBUG PPO] Kappa updated. New kappa: {self.weight_clip_kappa:.4f} (Clip fraction: {actual_clip_fraction:.4f})")

        # Store current kappa value in metrics
        final_metrics['current_kappa'] = self.weight_clip_kappa # This will be the default value if clipping is off

        return final_metrics

    def _project_weights(self):
        """Project weights of OrthogonalLinear layers onto the unit hypersphere (SimbaV2 Eq 20).

        Note: This should ideally happen as part of the optimizer step, but
        doing it immediately after is a common approximation.
        """
        try:
            # Ensure OrthogonalLinear is imported - adjust path if needed
            from model_architectures.utils import OrthogonalLinear
        except ImportError:
             if self.debug:
                 print("[DEBUG _project_weights] Could not import OrthogonalLinear. Projection skipped.")
             return

        models_to_project = [self.actor]
        if not self.shared_model:
            models_to_project.append(self.critic)

        # Also project auxiliary model weights if they use OrthogonalLinear
        if self.aux_task_manager:
            if hasattr(self.aux_task_manager, 'sr_task') and self.aux_task_manager.sr_task:
                models_to_project.append(self.aux_task_manager.sr_task)
            if hasattr(self.aux_task_manager, 'rp_task') and self.aux_task_manager.rp_task:
                models_to_project.append(self.aux_task_manager.rp_task)

        for model in models_to_project:
            if model is None: continue # Skip if a model (e.g., aux task) doesn't exist
            for module in model.modules():
                if isinstance(module, OrthogonalLinear):
                    # Check if the weight parameter exists and requires grad
                    if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                        with torch.no_grad():
                            module.weight.copy_(l2_norm(module.weight))
                    elif self.debug and not (hasattr(module, 'weight') and module.weight is not None):
                         # Optional debug print if weight is missing
                         print(f"[DEBUG _project_weights] Skipping module {module} - missing weight attr.")
                    elif self.debug and not module.weight.requires_grad:
                         # Optional debug print if weight doesn't require grad
                         print(f"[DEBUG _project_weights] Skipping module {module} - weight does not require grad.")



    def get_state_dict(self):
        """Get state dict for saving algorithm state"""
        state = super().get_state_dict() # Get base state if needed
        state.update({
            # 'actor_optimizer': self.actor_optimizer.state_dict(), # Removed individual optimizers
            # 'critic_optimizer': self.critic_optimizer.state_dict(), # Removed individual optimizers
            'optimizer': self.optimizer.state_dict(), # Save combined optimizer
            'scaler': self.scaler.state_dict(), # Save GradScaler state
            'memory_state': self.memory.get_state_dict() if hasattr(self.memory, 'get_state_dict') else None, # Save memory state if possible
            'episode_returns': list(self.episode_returns),
            'current_episode_rewards': self.current_episode_rewards,
            'weight_clip_kappa': self.weight_clip_kappa,
            '_update_counter': self._update_counter
        })
        # Add aux task manager state if it exists and has a method
        if self.aux_task_manager and hasattr(self.aux_task_manager, 'get_state_dict'):
             state['aux_task_manager'] = self.aux_task_manager.get_state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load state dict for resuming algorithm state"""
        super().load_state_dict(state_dict) # Load base state if needed
        # if 'actor_optimizer' in state_dict: # Removed individual optimizers
        #     self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        # if 'critic_optimizer' in state_dict: # Removed individual optimizers
        #     self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        if 'optimizer' in state_dict:
            # Need to handle optimizer loading carefully if parameter lists changed
            # (e.g., due to shared vs separate model)
            # A common approach is to re-initialize the optimizer after loading model params
            # For now, we'll attempt to load, but it might fail if params don't match.
            try:
                self.optimizer.load_state_dict(state_dict['optimizer']) # Load combined optimizer
            except ValueError as e:
                print(f"Warning: Could not load optimizer state, likely due to parameter mismatch (e.g., shared vs separate models). Optimizer state reset. Error: {e}")
                # Re-initialize optimizer with current parameters
                if self.shared_model:
                    current_params = list(self.actor.parameters())
                else:
                    current_params = list(self.actor.parameters()) + list(self.critic.parameters())
                if self.aux_task_manager:
                    if hasattr(self.aux_task_manager, 'sr_task') and self.aux_task_manager.sr_task is not None:
                        current_params += list(self.aux_task_manager.sr_task.parameters())
                    if hasattr(self.aux_task_manager, 'rp_task') and self.aux_task_manager.rp_task is not None:
                        current_params += list(self.aux_task_manager.rp_task.parameters())
                self.optimizer = torch.optim.Adam(current_params, lr=self.lr_actor)


        if 'scaler' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler']) # Load GradScaler state
        if 'memory_state' in state_dict and hasattr(self.memory, 'load_state_dict'):
            self.memory.load_state_dict(state_dict['memory_state'])
        if 'episode_returns' in state_dict:
            self.episode_returns = deque(state_dict['episode_returns'], maxlen=self.episode_returns.maxlen)
        if 'current_episode_rewards' in state_dict:
            self.current_episode_rewards = state_dict['current_episode_rewards']
        if 'weight_clip_kappa' in state_dict:
            self.weight_clip_kappa = state_dict['weight_clip_kappa']
        if '_update_counter' in state_dict:
            self._update_counter = state_dict['_update_counter']
        # Load aux task manager state if it exists and has a method
        if 'aux_task_manager' in state_dict and self.aux_task_manager and hasattr(self.aux_task_manager, 'load_state_dict'):
            self.aux_task_manager.load_state_dict(state_dict['aux_task_manager'])
