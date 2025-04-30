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
    """
    PPO algorithm implementation with SimbaV2 Gaussian Distributional Critic

    This implementation uses a Gaussian distributional critic to model the return distribution.
    Key features:

    1. Gaussian representation of value distribution (mean and variance)
    2. Separate losses for mean (MSE) and variance (NLL)
    3. Uncertainty-weighted PPO objective using Gaussian variance or entropy
    4. Scalar GAE using the mean of the Gaussian critic
    """

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
        buffer_size=131072, # Add buffer_size parameter with default
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
        # --- Gaussian Critic Parameters ---
        variance_loss_coefficient=0.01, # Coefficient for the variance loss term
        # v_min=None, # Removed C51 param
        # v_max=None, # Removed C51 param
        # num_atoms=None, # Removed C51 param
        # --- Distributional PPO Parameters ---
        use_uncertainty_weight=True,  # Whether to use uncertainty weighting in PPO objective
        uncertainty_weight_type="variance",  # "entropy" or "variance" for uncertainty measure
        uncertainty_weight_temp=1.0,  # Temperature parameter for uncertainty weighting
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
        # --- Reward Normalization G_max ---
        reward_norm_G_max=10.0, # Default G_max for reward normalization (adjust based on expected returns)
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

        # --- ADD GAUSSIAN CRITIC PARAMS ---
        self.variance_loss_coefficient = variance_loss_coefficient
        # Add log_std constraints for stability
        self.log_std_min = -20  # Minimum log standard deviation
        self.log_std_max = 2    # Maximum log standard deviation
        # Removed C51 params: v_min, v_max, num_atoms, support, delta_z
        # -----------------------------------------

        # --- ADD DISTRIBUTIONAL PPO UNCERTAINTY WEIGHTING PARAMS ---
        self.use_uncertainty_weight = use_uncertainty_weight
        self.uncertainty_weight_type = uncertainty_weight_type
        self.uncertainty_weight_temp = uncertainty_weight_temp
        # -----------------------------------------


        # Weight clipping parameters (kept for potential future use, but disabled)
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
            algorithm_instance=self, # Pass reference to the algorithm
            batch_size=batch_size,
            # Use the buffer_size parameter
            buffer_size=buffer_size,
            device=device,
            debug=debug,
            action_space_type=self.action_space_type # Pass action space type
        )

        # Initialize optimizer
        # Combine parameters for a single optimizer step
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

        # Use actor LR for the combined optimizer
        self.optimizer = torch.optim.Adam(combined_params, lr=lr_actor)

        # Initialize GradScaler if AMP is enabled
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking metrics
        self.metrics = {
            'actor_loss': 0.0,
            'critic_loss_mean': 0.0, # Split critic loss
            'critic_loss_variance': 0.0, # Split critic loss
            'critic_loss_total': 0.0, # Combined critic loss
            'entropy_loss': 0.0,
            'sr_loss_scalar': 0.0, # Add aux metrics
            'rp_loss_scalar': 0.0, # Add aux metrics
            'total_loss': 0.0,
            'clip_fraction': 0.0,
            'explained_variance': 0.0,
            # 'kl_divergence': 0.0, # Removed C51 metric
            'mean_advantage': 0.0,
            'mean_return': 0.0,
            'weight_clip_fraction': 0.0,
            'current_kappa': self.weight_clip_kappa,
            # Uncertainty weighting metrics
            'mean_uncertainty_weight': 1.0,
            'min_uncertainty_weight': 1.0,
            'max_uncertainty_weight': 1.0,
            # Gaussian critic metrics
            'mean_predicted_variance': 0.0,
        }

        # Add episode return tracking
        self.current_episode_rewards = []
        self.episode_returns = deque(maxlen=100)

        # --- SimbaV2 Reward Normalization Params ---
        self.running_G = 0.0 # Running discounted return
        self.running_G_mean = 0.0 # Running mean of discounted return
        self.running_G_var = 1.0 # Running variance of discounted return (init to 1.0)
        self.running_G_max = 0.0 # Running maximum of discounted return
        self.running_G_count = 0 # Count for Welford's method
        self.reward_norm_epsilon = 1e-8 # Small constant for stability
        self.reward_norm_G_max = reward_norm_G_max # Use configured G_max
        # ------------------------------------------

    class PPOMemory:
        """Memory buffer for PPO to store experiences (No changes needed for Gaussian critic)"""

        def __init__(self, algorithm_instance, batch_size, buffer_size, device, debug=False, action_space_type="discrete"):
            self.algorithm_instance = algorithm_instance # Reference to the PPOAlgorithm instance
            self.batch_size = batch_size
            self.debug = debug
            self.action_space_type = action_space_type # Store action space type

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
            self.obs = None  # Will be initialized on first store() call
            self.actions = None  # Will be initialized on first store() call
            self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device) # Placeholder values
            self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)

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
                    if action_sample.numel() == 1 and action_sample.dim() == 0:
                        action_shape = (1,)
                    elif action_sample.dim() == 1:
                        action_shape = action_sample.shape
                    elif action_sample.dim() > 1:
                        action_shape = action_sample.shape[1:]
                    else:
                        action_shape = (1,)
                    action_dtype = torch.float32

                obs_shape = obs_sample.shape[1:]

                if self.debug:
                    print(f"[DEBUG PPOMemory] Initializing buffers: obs_shape={obs_shape}, action_shape={action_shape}, action_dtype={action_dtype}")

                self.obs = torch.zeros((self.buffer_size, *obs_shape), dtype=torch.float32, device=self.device)
                self.actions = torch.zeros((self.buffer_size, *action_shape), dtype=action_dtype, device=self.device)


        def store_initial_batch(self, obs_batch, action_batch, log_prob_batch, value_batch):
            """Store the initial part of experiences in batch."""
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
            if not isinstance(value_batch, torch.Tensor): value_batch = torch.tensor(value_batch, dtype=torch.float32)

            target_device = self.device
            obs_batch = obs_batch.to(target_device)
            action_batch = action_batch.to(target_device)
            log_prob_batch = log_prob_batch.to(target_device)
            value_batch = value_batch.to(target_device)

            indices = torch.arange(self.pos, self.pos + batch_size, device=target_device) % self.buffer_size

            self.obs.index_copy_(0, indices, obs_batch.detach())

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
            self.values.index_copy_(0, indices, value_batch.detach()) # Store placeholder values

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

            normed_rewards = self.algorithm_instance.normalize_reward(rewards_batch, dones_batch)
            if not isinstance(normed_rewards, torch.Tensor):
                normed_rewards = torch.tensor(normed_rewards, dtype=torch.float32, device=target_device)
            else:
                normed_rewards = normed_rewards.to(target_device)

            self.rewards.index_copy_(0, indices, normed_rewards)
            self.dones.index_copy_(0, indices, dones_batch)

            if self.debug and len(indices) > 0:
                print(f"[DEBUG PPOMemory] Updated rewards/dones for {len(indices)} indices. First reward: {rewards_batch[0].item():.4f}")


        def store(self, obs, action, log_prob, reward, value, done):
            """Store a single experience in the buffer (legacy, less efficient)"""
            expected_action_dtype = torch.long if self.action_space_type == "discrete" else torch.float32

            obs_b = torch.tensor([obs], dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs.unsqueeze(0)
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
            indices = torch.tensor([idx], dtype=torch.long, device=self.device)

            if reward is not None or done is not None:
                rewards_b = torch.tensor([reward if reward is not None else self.rewards[idx]], dtype=torch.float32)
                dones_b = torch.tensor([done if done is not None else self.dones[idx]], dtype=torch.bool)
                self.update_rewards_dones_batch(indices, rewards_b, dones_b)

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
                if self.action_space_type == "discrete" and action.dim() > 0:
                     action = action.squeeze()
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

            obs_data = self.obs[:self.size]
            actions_data = self.actions[:self.size]
            log_probs_data = self.log_probs[:self.size]
            rewards_data = self.rewards[:self.size]
            values_data = self.values[:self.size] # Placeholder values
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
            for i in done_indices_local:
                if isinstance(rewards_batch, torch.Tensor):
                    reward_value = rewards_batch[i].item()
                else:
                    reward_value = float(rewards_batch[i])

                self.current_episode_rewards.append(reward_value)
                episode_return = sum(self.current_episode_rewards)
                self.episode_returns.append(episode_return)

                if self.debug:
                    print(f"[DEBUG PPO] Episode completed with return: {episode_return:.4f}")
                self.current_episode_rewards = []

            if self.debug:
                print(f"[DEBUG PPO] {len(done_indices_local)} episodes ended in this batch update.")


    def store_experience(self, obs, action, log_prob, reward, value, done):
        """Store experience in the buffer (legacy single experience method)"""
        if self.debug:
            print(f"[DEBUG PPO] Storing single experience - reward: {reward}")

        normed_reward = self.normalize_reward(reward, done)
        self.memory.store(obs, action, log_prob, normed_reward, value, done)

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
                self.current_episode_rewards = []

    def store_experience_at_idx(self, idx, obs=None, action=None, log_prob=None, reward=None, value=None, done=None):
        """Update specific values of an experience at a given index (legacy)"""
        if self.debug and reward is not None:
            print(f"[DEBUG PPO] Updating reward at idx {idx}: {reward}")
        self.memory.store_experience_at_idx(idx, obs, action, log_prob, reward, value, done)

    def get_action(self, obs, deterministic=False, return_features=False):
        """Get action for a given observation (using only the actor network/head)"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.shared_model:
                # Shared model needs flags to specify which output(s) are needed
                # Assuming SimbaV2 forward accepts these flags (needs modification if not)
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
                    features = None

            if actor_output is None:
                if self.debug: print("[DEBUG PPO get_action] Error: Actor output not found in model return.")
                dummy_action = torch.zeros(obs.shape[0], dtype=torch.long if self.action_space_type == "discrete" else torch.float, device=self.device)
                dummy_log_prob = torch.zeros(obs.shape[0], device=self.device)
                dummy_value = torch.zeros_like(dummy_log_prob)
                if return_features: return dummy_action, dummy_log_prob, dummy_value, features
                else: return dummy_action, dummy_log_prob, dummy_value

            if deterministic:
                if self.action_space_type == "discrete":
                    probs = actor_output
                    action = torch.argmax(probs, dim=-1)
                    log_prob = torch.log(torch.gather(probs, -1, action.unsqueeze(-1)).squeeze(-1) + 1e-10)
                else: # Continuous
                    action_dist = actor_output
                    action = action_dist.loc
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
            else: # Sample action
                if self.action_space_type == "discrete":
                    probs = actor_output
                    probs = torch.clamp(probs, min=1e-10)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    try:
                        dist = torch.distributions.Categorical(probs=probs)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                    except ValueError as e:
                         if self.debug:
                             print(f"[DEBUG PPO get_action] Error creating Categorical distribution: {e}")
                             print(f"Probs shape: {probs.shape}, Probs sum: {probs.sum(dim=-1)}")
                             print(f"Probs sample: {probs[0]}")
                         action = torch.argmax(probs, dim=-1)
                         log_prob = torch.log(torch.gather(probs, -1, action.unsqueeze(-1)).squeeze(-1) + 1e-10)
                else: # Continuous
                    action_dist = actor_output
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum(dim=-1)

        # Create dummy value tensor (zeros)
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

        if buffer_size < self.batch_size:
            if self.debug:
                print(f"[DEBUG PPO] Buffer size ({buffer_size}) < batch size ({self.batch_size}), skipping update")
            if self.aux_task_manager:
                self.metrics['sr_loss_scalar'] = self.aux_task_manager.last_sr_loss
                self.metrics['rp_loss_scalar'] = self.aux_task_manager.last_rp_loss
            return self.metrics

        if self.debug:
            print(f"[DEBUG PPO] Starting update with buffer size: {buffer_size}")

        states, actions, old_log_probs, rewards, values_placeholder, dones = self.memory.get()

        if states is None:
            if self.debug:
                print("[DEBUG PPO] Failed to get experiences from buffer, skipping update")
            return self.metrics

        # --- CALCULATE EXPECTED VALUES (MEAN) FOR GAE ---
        with torch.no_grad():
            expected_values_list = []
            for i in range(0, buffer_size, self.batch_size):
                chunk_states = states[i:min(i + self.batch_size, buffer_size)]
                with autocast("cuda", enabled=self.use_amp):
                    # Request critic output (mean and log_std)
                    if self.shared_model:
                        # Assuming shared model returns dict with 'critic_out' = [mean, log_std] tensor
                        model_output = self.critic(chunk_states, return_actor=False, return_critic=True, return_features=False)
                        critic_output = model_output.get('critic_out') # Shape: [chunk_size, 2]
                    else:
                        critic_output = self.critic(chunk_states) # Shape: [chunk_size, 2]

                    # --- Critical Shape Check for Gaussian Critic ---
                    if critic_output is None or critic_output.shape[0] != chunk_states.shape[0] or critic_output.shape[1] != 2:
                         if self.debug:
                             expected_shape = (chunk_states.shape[0], 2)
                             actual_shape = critic_output.shape if critic_output is not None else "None"
                             print(f"[DEBUG PPO GAE] Critic output shape mismatch in GAE calc. Expected {expected_shape}, got {actual_shape}. Skipping update.")
                             print(f"[DEBUG PPO GAE] Make sure SimbaV2 critic is properly initialized and returns [mean, log_std] pairs.")
                         self.memory.clear()
                         return self.metrics

                    # Extract the mean from the Gaussian parameters [batch_size, 2] -> [mean, log_std]
                    chunk_expected_values = critic_output[:, 0] # Extract mean (first column)

                expected_values_list.append(chunk_expected_values)

            expected_values_for_gae = torch.cat(expected_values_list) # Shape: [buffer_size]
        # --- END EXPECTED VALUES CALCULATION ---

        returns, advantages = self._compute_gae(rewards, expected_values_for_gae, dones)

        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            if self.debug:
                print("[DEBUG PPO] NaN or Inf detected in advantages, skipping update")
                print(f"Advantages mean: {advantages.mean()}, std: {advantages.std()}")
                print(f"Rewards mean: {rewards.mean()}, std: {rewards.std()}")
                print(f"Expected Values (Mean) mean: {expected_values_for_gae.mean()}, std: {expected_values_for_gae.std()}")
            self.memory.clear()
            return self.metrics

        metrics = self._update_policy(states, actions, old_log_probs, returns, advantages, rewards)
        self.memory.clear()
        self.metrics.update(metrics)

        if len(self.episode_returns) > 0:
            self.metrics['mean_return'] = sum(self.episode_returns) / len(self.episode_returns)

        self._update_counter += 1

        if self.debug:
            ev_mean = expected_values_for_gae.mean().item()
            ev_std = expected_values_for_gae.std().item()
            print(f"[DEBUG PPO] Update finished. Actor Loss: {metrics.get('actor_loss', 0):.4f}, Critic Loss (Total): {metrics.get('critic_loss_total', 0):.4f}, "
                  f"SR Loss: {metrics.get('sr_loss_scalar', 0):.4f}, RP Loss: {metrics.get('rp_loss_scalar', 0):.4f}, "
                  f"ExpValue Mean: {ev_mean:.4f}, ExpValue Std: {ev_std:.4f}")

        return self.metrics


    def _compute_uncertainty_weight(self, predicted_mean, predicted_log_std):
        """
        Compute uncertainty weight based on the Gaussian critic's prediction.

        Args:
            predicted_mean: predicted mean [batch_size]
            predicted_log_std: predicted log_std [batch_size]

        Returns:
            weight tensor [batch_size] for PPO objective weighting
        """
        if not self.use_uncertainty_weight:
            return torch.ones_like(predicted_mean)

        predicted_std = torch.exp(predicted_log_std)
        predicted_variance = predicted_std.pow(2)

        if self.uncertainty_weight_type == "entropy":
            # Entropy of Gaussian: 0.5 * log(2 * pi * e * variance)
            # Constant terms don't affect relative weights, so use log(variance) or log(std)
            # entropy = 0.5 * torch.log(2 * math.pi * math.e * predicted_variance)
            # Simpler: use log_std directly (monotonic transformation)
            # Higher log_std -> higher entropy -> lower weight
            log_std_proxy = predicted_log_std
            weight = torch.sigmoid(-log_std_proxy * self.uncertainty_weight_temp) + 0.5

            if self.debug and torch.rand(1).item() < 0.01:
                print(f"[DEBUG PPO Uncertainty] Entropy (log_std) weight range: {weight.min().item():.4f}-{weight.max().item():.4f}, "
                      f"Mean: {weight.mean().item():.4f}")

        elif self.uncertainty_weight_type == "variance":
            # Use predicted variance directly
            # Higher variance -> more uncertainty -> lower weight
            weight = torch.sigmoid(-predicted_variance * self.uncertainty_weight_temp) + 0.5

            if self.debug and torch.rand(1).item() < 0.01:
                print(f"[DEBUG PPO Uncertainty] Variance weight range: {weight.min().item():.4f}-{weight.max().item():.4f}, "
                      f"Mean: {weight.mean().item():.4f}")
        else:
            if self.debug:
                print(f"[DEBUG PPO Uncertainty] Unknown uncertainty weight type: {self.uncertainty_weight_type}, defaulting to 1.0")
            weight = torch.ones_like(predicted_mean)

        return weight.detach() # Detach weights, they shouldn't influence gradient calculation directly

    def _compute_gae(self, rewards, values, dones):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE)
        (No changes needed here, uses the provided scalar 'values' which are now the critic's mean predictions)
        """
        buffer_size = len(rewards)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_idx = buffer_size - 1
        next_value = 0.0
        if buffer_size > 0 and values.numel() > 0:
             next_value = values[last_idx].item() * (1.0 - dones[last_idx].float().item())

        next_advantage = 0.0
        for t in reversed(range(buffer_size)):
            current_value = values[t]
            current_reward = rewards[t]
            current_done = dones[t].float()
            delta = current_reward + self.gamma * next_value * (1.0 - current_done) - current_value
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * (1.0 - current_done)
            next_advantage = advantages[t]
            next_value = current_value.item()
            returns[t] = advantages[t] + values[t]

        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        if self.debug:
            print(f"[DEBUG PPO _compute_gae] Returns mean: {returns.mean():.4f}, std: {returns.std():.4f}, min: {returns.min():.4f}, max: {returns.max():.4f}")
            # print(f"[DEBUG PPO _compute_gae] Critic Range: N/A for Gaussian") # Removed C51 range
            print(f"[DEBUG PPO _compute_gae] Advantages mean: {adv_mean:.4f}, std: {adv_std:.4f}")

        return returns, advantages

    def init_weight_ranges(self):
        """Store the BASE initialization ranges (kappa=1) of all network parameters (No changes needed)"""
        # This logic remains the same, just applied to the current actor/critic structure
        self.actor_base_bounds = {}
        self.critic_base_bounds = {}
        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                if 'weight' in name:
                    fan_in = param.shape[1] * (param.shape[2] if len(param.shape) > 2 else 1)
                    bound = (6 / fan_in) ** 0.5
                    self.actor_base_bounds[name] = (-bound, bound)
                else:
                    bound = 0.01
                    self.actor_base_bounds[name] = (-bound, bound)
        if not self.shared_model:
            for name, param in self.critic.named_parameters():
                if param.requires_grad:
                    if 'weight' in name:
                        fan_in = param.shape[1] * (param.shape[2] if len(param.shape) > 2 else 1)
                        bound = (6 / fan_in) ** 0.5
                        self.critic_base_bounds[name] = (-bound, bound)
                    else:
                        bound = 0.01
                        self.critic_base_bounds[name] = (-bound, bound)

    def clip_weights(self):
        """Clip weights based on current kappa and return clipped fraction. (No changes needed)"""
        if not self.use_weight_clipping:
            return 0.0
        total_params = 0
        clipped_params = 0
        current_kappa = self.weight_clip_kappa
        for name, param in self.actor.named_parameters():
            if name in self.actor_base_bounds and param.requires_grad:
                base_lower, base_upper = self.actor_base_bounds[name]
                lower_bound = base_lower * current_kappa
                upper_bound = base_upper * current_kappa
                original_param = param.data.clone()
                param.data.clamp_(lower_bound, upper_bound)
                num_params = param.numel()
                total_params += num_params
                clipped_params += torch.sum(param.data != original_param).item()
        if not self.shared_model:
            for name, param in self.critic.named_parameters():
                if name in self.critic_base_bounds and param.requires_grad:
                    base_lower, base_upper = self.critic_base_bounds[name]
                    lower_bound = base_lower * current_kappa
                    upper_bound = base_upper * current_kappa
                    original_param = param.data.clone()
                    param.data.clamp_(lower_bound, upper_bound)
                    num_params = param.numel()
                    total_params += num_params
                    clipped_params += torch.sum(param.data != original_param).item()
        return float(clipped_params) / total_params if total_params > 0 else 0.0


    def _update_policy(self, states, actions, old_log_probs, returns, advantages, rewards):
        """
        Update policy and value networks using PPO algorithm with Gaussian critic.
        """
        update_metrics_tensors = {
            'actor_loss': torch.tensor(0.0, device=self.device),
            'critic_loss_mean': torch.tensor(0.0, device=self.device),
            'critic_loss_variance': torch.tensor(0.0, device=self.device),
            'critic_loss_total': torch.tensor(0.0, device=self.device),
            'entropy_loss': torch.tensor(0.0, device=self.device),
            'total_loss': torch.tensor(0.0, device=self.device),
            'clip_fraction': torch.tensor(0.0, device=self.device),
            # 'kl_divergence': torch.tensor(0.0, device=self.device), # Removed C51 metric
        }
        update_metrics_scalars = {
            'sr_loss_scalar': 0.0,
            'rp_loss_scalar': 0.0,
            'weight_clip_fraction': 0.0,
            'mean_uncertainty_weight': 0.0,
            'min_uncertainty_weight': float('inf'),
            'max_uncertainty_weight': 0.0,
            'mean_predicted_variance': 0.0, # Add Gaussian metric
        }

        explained_var_scalar = 0.0
        mean_advantage_scalar = 0.0
        with torch.no_grad():
            # Calculate expected values (mean) from critic for explained variance
            y_pred_mean_list = []
            buffer_size = states.shape[0]
            for i in range(0, buffer_size, self.batch_size):
                chunk_states = states[i:min(i + self.batch_size, buffer_size)]
                with autocast("cuda", enabled=self.use_amp):
                    if self.shared_model:
                        model_output = self.critic(chunk_states, return_actor=False, return_critic=True, return_features=False)
                        critic_output = model_output.get('critic_out') # Shape: [chunk_size, 2]
                    else:
                        critic_output = self.critic(chunk_states) # Shape: [chunk_size, 2]

                    if critic_output is None or critic_output.shape[0] != chunk_states.shape[0] or critic_output.shape[1] != 2:
                        if self.debug:
                            expected_shape = (chunk_states.shape[0], 2)
                            actual_shape = critic_output.shape if critic_output is not None else "None"
                            print(f"[DEBUG PPO EV] Critic output shape mismatch in EV calc. Expected {expected_shape}, got {actual_shape}.")
                            print(f"[DEBUG PPO EV] Skipping explained variance calculation.")
                        y_pred_mean_list = None
                        break

                    # Extract mean from Gaussian parameters [batch_size, 2]
                    chunk_mean = critic_output[:, 0] # First column is mean
                y_pred_mean_list.append(chunk_mean)

            if y_pred_mean_list is not None:
                y_pred = torch.cat(y_pred_mean_list) # Predicted means [buffer_size]
                y_true = returns # GAE returns [buffer_size]
                var_y = torch.var(y_true)
                explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
                explained_var_scalar = explained_var.item()
            else:
                explained_var_scalar = 0.0
                if self.debug: print("[DEBUG PPO EV] Explained variance calculation skipped due to critic shape mismatch.")

            mean_advantage_scalar = advantages.mean().item()


        total_weight_clipped_fraction_epoch = 0.0
        num_batches_processed = 0

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
                batch_returns = returns[batch_idx] # Target scalar returns for critic mean
                batch_advantages = advantages[batch_idx]
                batch_rewards = rewards[batch_idx] # For aux tasks

                with autocast("cuda", enabled=self.use_amp):
                    # Get current policy distribution, features, and critic output (mean, log_std)
                    try:
                        if self.shared_model:
                            model_output = self.actor(batch_states, return_actor=True, return_critic=True, return_features=True)
                            actor_output = model_output.get('actor_out')
                            critic_output = model_output.get('critic_out') # Expecting [batch_size, 2]
                            current_features = model_output.get('features')
                        else:
                            actor_result = self.actor(batch_states, return_features=True)
                            critic_output = self.critic(batch_states) # Expecting [batch_size, 2]
                            if isinstance(actor_result, tuple):
                                actor_output, current_features = actor_result
                            else:
                                actor_output = actor_result; current_features = None

                        # Validate outputs
                        if actor_output is None: raise ValueError("Actor output missing from model")
                        if critic_output is None: raise ValueError("Critic output missing from model")
                        if critic_output.shape != (batch_states.shape[0], 2):
                             raise ValueError(f"Critic output shape mismatch. Expected {(batch_states.shape[0], 2)}, got {critic_output.shape}")

                        # Get mean and log_std from critic output
                        # Expected shape: [batch_size, 2] for the critic output
                        if critic_output.dim() == 2 and critic_output.shape[1] == 2:
                            # Extract directly from the tensor columns
                            predicted_mean = critic_output[:, 0]  # First column is mean
                            predicted_log_std = critic_output[:, 1]  # Second column is log_std
                        else:
                            # Fall back to chunk if needed (handles different tensor layouts)
                            predicted_mean, predicted_log_std = critic_output.chunk(2, dim=-1)
                            predicted_mean = predicted_mean.squeeze(-1)  # Shape: [batch_size]
                            predicted_log_std = predicted_log_std.squeeze(-1)  # Shape: [batch_size]

                        # Clamp log_std (redundant if done in model, but safe)
                        predicted_log_std = torch.clamp(predicted_log_std, self.log_std_min, self.log_std_max)


                        if self.debug and current_features is None and self.aux_task_manager:
                            print("[DEBUG PPO _update_policy] Warning: Features not returned by actor, cannot compute aux loss.")

                    except Exception as e:
                        if self.debug: print(f"[DEBUG PPO _update_policy] Error getting model output/features: {e}")
                        continue

                    # --- Actor Loss (Standard PPO) ---
                    if self.action_space_type == "discrete":
                        action_probs = actor_output
                        action_probs = torch.clamp(action_probs, min=1e-10)
                        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                        try:
                            dist = torch.distributions.Categorical(probs=action_probs)
                        except ValueError as e:
                             if self.debug: print(f"[DEBUG PPO _update_policy] Error creating Categorical distribution in batch: {e}")
                             continue

                        if batch_actions.dtype != torch.long:
                             try: actions_indices = batch_actions.long()
                             except RuntimeError:
                                 if self.debug: print("[DEBUG PPO _update_policy] Failed to cast batch_actions to Long.")
                                 continue
                        else: actions_indices = batch_actions

                        if actions_indices.max() >= dist.probs.shape[-1] or actions_indices.min() < 0:
                             if self.debug: print(f"[DEBUG PPO _update_policy] Invalid action indices found.")
                             continue

                        try:
                            curr_log_probs = dist.log_prob(actions_indices)
                            entropy = dist.entropy().mean()
                        except (IndexError, ValueError) as e:
                             if self.debug: print(f"[DEBUG PPO _update_policy] Error calculating log_prob/entropy: {e}")
                             continue

                    else: # Continuous action space
                        if batch_actions.dtype != torch.float:
                             try: batch_actions = batch_actions.float()
                             except RuntimeError:
                                 if self.debug: print("[DEBUG PPO _update_policy] Failed to cast batch_actions to Float.")
                                 continue

                        action_dist = actor_output
                        try:
                             curr_log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)
                             entropy = action_dist.entropy().mean()
                        except Exception as e:
                             if self.debug: print(f"[DEBUG PPO _update_policy] Error calculating continuous log_prob/entropy: {e}")
                             continue

                    # Calculate PPO actor loss components
                    ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                    ppo_objectives = torch.min(surr1, surr2)

                    # Apply uncertainty weighting if enabled
                    if self.use_uncertainty_weight:
                        uncertainty_weights = self._compute_uncertainty_weight(predicted_mean, predicted_log_std)
                        weighted_objectives = ppo_objectives * uncertainty_weights
                        actor_loss = -weighted_objectives.mean()

                        batch_mean_weight = uncertainty_weights.mean().item()
                        batch_min_weight = uncertainty_weights.min().item()
                        batch_max_weight = uncertainty_weights.max().item()
                        update_metrics_scalars['mean_uncertainty_weight'] += batch_mean_weight
                        update_metrics_scalars['min_uncertainty_weight'] = min(update_metrics_scalars['min_uncertainty_weight'], batch_min_weight)
                        update_metrics_scalars['max_uncertainty_weight'] = max(update_metrics_scalars['max_uncertainty_weight'], batch_max_weight)

                        if self.debug and torch.rand(1).item() < 0.01:
                            print(f"[DEBUG PPO Uncertainty] Weights: min={batch_min_weight:.4f}, max={batch_max_weight:.4f}, mean={batch_mean_weight:.4f}")
                    else:
                        actor_loss = -ppo_objectives.mean()
                        update_metrics_scalars['mean_uncertainty_weight'] += 1.0
                        update_metrics_scalars['min_uncertainty_weight'] = min(update_metrics_scalars['min_uncertainty_weight'], 1.0)
                        update_metrics_scalars['max_uncertainty_weight'] = max(update_metrics_scalars['max_uncertainty_weight'], 1.0)

                    entropy_loss = -entropy * self.entropy_coef

                    # --- Gaussian Critic Loss ---
                    # 1. Mean Loss (MSE)
                    critic_loss_mean = F.mse_loss(predicted_mean, batch_returns)

                    # 2. Variance Loss (Negative Log Likelihood - NLL)
                    # NLL = 0.5 * [ log(2*pi*var) + (target - mean)^2 / var ]
                    # NLL = 0.5 * [ log(2*pi) + 2*log_std + (target - mean)^2 / exp(2*log_std) ]
                    predicted_variance = torch.exp(predicted_log_std * 2) + 1e-6 # Add epsilon for stability
                    # Use detached mean for variance loss calculation to avoid interference
                    nll = 0.5 * (torch.log(2 * torch.pi * predicted_variance) + \
                                 (batch_returns - predicted_mean.detach()).pow(2) / predicted_variance)
                    critic_loss_variance = nll.mean()

                    # Total critic loss
                    critic_loss_total = critic_loss_mean + self.variance_loss_coefficient * critic_loss_variance

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

                            if self.debug and (sr_loss_scalar > 0 or rp_loss_scalar > 0):
                                print(f"[DEBUG PPO Aux] Batch Aux Losses - SR: {sr_loss_scalar:.6f}, RP: {rp_loss_scalar:.6f}")
                        except Exception as e:
                            if self.debug:
                                print(f"[DEBUG PPO Aux] Error computing aux losses for batch: {e}")
                                import traceback
                                traceback.print_exc()
                            sr_loss = torch.tensor(0.0, device=self.device); rp_loss = torch.tensor(0.0, device=self.device)
                            sr_loss_scalar = 0.0; rp_loss_scalar = 0.0
                    elif self.debug and self.aux_task_manager is not None and current_features is None:
                         print("[DEBUG PPO Aux] Skipping aux loss calculation as features were not obtained.")

                    # --- Total Loss ---
                    total_loss = actor_loss + (self.critic_coef * critic_loss_total) + entropy_loss + sr_loss + rp_loss

                # --- Optimization ---
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0:
                    if self.optimizer.param_groups and 'params' in self.optimizer.param_groups[0]:
                         nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.max_grad_norm)
                    elif self.debug:
                         print("[DEBUG PPO] Could not clip gradients: optimizer param_groups not found or structured as expected.")
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # --- Project Weights (SimbaV2) ---
                self._project_weights()

                # --- Weight Clipping (Disabled by default) ---
                current_weight_clip_fraction = self.clip_weights()
                total_weight_clipped_fraction_epoch += current_weight_clip_fraction

                # --- Update Metrics ---
                update_metrics_tensors['actor_loss'] += actor_loss.detach()
                update_metrics_tensors['critic_loss_mean'] += critic_loss_mean.detach()
                update_metrics_tensors['critic_loss_variance'] += critic_loss_variance.detach()
                update_metrics_tensors['critic_loss_total'] += critic_loss_total.detach()
                update_metrics_tensors['entropy_loss'] += entropy_loss.detach()
                update_metrics_tensors['total_loss'] += total_loss.detach()

                update_metrics_scalars['sr_loss_scalar'] += sr_loss_scalar
                update_metrics_scalars['rp_loss_scalar'] += rp_loss_scalar
                update_metrics_scalars['mean_predicted_variance'] += predicted_variance.mean().item() # Track variance

                with torch.no_grad():
                    policy_clip_fraction_tensor = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    update_metrics_tensors['clip_fraction'] += policy_clip_fraction_tensor
                    # KL divergence metric removed


        # --- Finalize Metrics ---
        final_metrics = {}
        if num_batches_processed > 0:
            for key, tensor_val in update_metrics_tensors.items():
                final_metrics[key] = (tensor_val / num_batches_processed).item()
            for key, scalar_val in update_metrics_scalars.items():
                if key == 'min_uncertainty_weight' or key == 'max_uncertainty_weight':
                    final_metrics[key] = scalar_val
                else:
                    final_metrics[key] = scalar_val / num_batches_processed
            final_metrics['weight_clip_fraction'] = total_weight_clipped_fraction_epoch / num_batches_processed
        else:
             if self.debug: print("[DEBUG PPO _update_policy] No batches were processed in any epoch.")
             for key in list(update_metrics_tensors.keys()): final_metrics[key] = 0.0
             for key in list(update_metrics_scalars.keys()):
                 if key == 'min_uncertainty_weight': final_metrics[key] = 1.0
                 elif key == 'max_uncertainty_weight': final_metrics[key] = 1.0
                 else: final_metrics[key] = 0.0
             final_metrics['weight_clip_fraction'] = 0.0

        final_metrics['explained_variance'] = explained_var_scalar
        final_metrics['mean_advantage'] = mean_advantage_scalar

        # --- Adaptive Kappa Update Logic (Disabled by default) ---
        if self.use_weight_clipping and self.adaptive_kappa and (self._update_counter % self.kappa_update_freq == 0):
            actual_clip_fraction = final_metrics['weight_clip_fraction']
            if actual_clip_fraction > self.target_clip_fraction:
                self.weight_clip_kappa *= (1 + self.kappa_update_rate)
            elif actual_clip_fraction < self.target_clip_fraction:
                self.weight_clip_kappa *= (1 - self.kappa_update_rate)
            self.weight_clip_kappa = max(self.min_kappa, min(self.max_kappa, self.weight_clip_kappa))
            if self.debug: print(f"[DEBUG PPO] Kappa updated. New kappa: {self.weight_clip_kappa:.4f} (Clip fraction: {actual_clip_fraction:.4f})")

        final_metrics['current_kappa'] = self.weight_clip_kappa

        return final_metrics

    def _project_weights(self):
        """Project weights of OrthogonalLinear layers onto the unit hypersphere (SimbaV2 Eq 20). (No changes needed)"""
        try:
            from model_architectures.utils import OrthogonalLinear
        except ImportError:
             if self.debug: print("[DEBUG _project_weights] Could not import OrthogonalLinear. Projection skipped.")
             return

        models_to_project = [self.actor]
        if not self.shared_model: models_to_project.append(self.critic)
        if self.aux_task_manager:
            if hasattr(self.aux_task_manager, 'sr_task') and self.aux_task_manager.sr_task: models_to_project.append(self.aux_task_manager.sr_task)
            if hasattr(self.aux_task_manager, 'rp_task') and self.aux_task_manager.rp_task: models_to_project.append(self.aux_task_manager.rp_task)

        for model in models_to_project:
            if model is None: continue
            for module in model.modules():
                if isinstance(module, OrthogonalLinear):
                    if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                        with torch.no_grad():
                            module.weight.copy_(l2_norm(module.weight))
                    elif self.debug and not (hasattr(module, 'weight') and module.weight is not None):
                         print(f"[DEBUG _project_weights] Skipping module {module} - missing weight attr.")
                    elif self.debug and not module.weight.requires_grad:
                         print(f"[DEBUG _project_weights] Skipping module {module} - weight does not require grad.")

    def normalize_reward(self, reward, done):
        """
        SimbaV2 reward normalization: normalize reward using running discounted return statistics.
        (No changes needed, but ensure reward_norm_G_max is set appropriately)
        """
        gamma = self.gamma
        eps = self.reward_norm_epsilon
        G_max = self.reward_norm_G_max # Use configured G_max

        is_tensor_input = isinstance(reward, torch.Tensor)
        target_device = reward.device if is_tensor_input else self.device

        if is_tensor_input:
            reward_np = reward.detach().cpu().numpy()
            done_np = done.detach().cpu().numpy()
        else:
            reward_np = np.array(reward)
            done_np = np.array(done)

        was_scalar = False
        if reward_np.ndim == 0:
             reward_np = reward_np.reshape(1)
             done_np = done_np.reshape(1)
             was_scalar = True

        normed_rewards = np.empty_like(reward_np)

        for i in range(len(reward_np)):
            r = reward_np[i]
            d = done_np[i]

            if d: self.running_G = 0.0
            self.running_G = (1 - gamma) * self.running_G + r
            self.running_G_count += 1
            delta = self.running_G - self.running_G_mean
            self.running_G_mean += delta / self.running_G_count
            delta2 = self.running_G - self.running_G_mean
            self.running_G_var += delta * delta2
            self.running_G_max = max(self.running_G_max, self.running_G)

            current_variance = self.running_G_var / self.running_G_count if self.running_G_count > 0 else 0.0
            std = np.sqrt(max(0.0, current_variance) + eps)

            denom_max_term = self.running_G_max / G_max if G_max > 0 else 0.0
            denom = max(std, denom_max_term)
            normed_rewards[i] = r / denom if denom > eps else r

            if d: self.running_G_max = 0.0

        if was_scalar: normed_rewards = normed_rewards.item()

        if is_tensor_input:
            return torch.tensor(normed_rewards, dtype=reward.dtype, device=target_device)
        else:
            return normed_rewards


    def get_state_dict(self):
        """Get state dict for saving algorithm state"""
        state = super().get_state_dict()
        state.update({
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'memory_state': self.memory.get_state_dict() if hasattr(self.memory, 'get_state_dict') else None,
            'episode_returns': list(self.episode_returns),
            'current_episode_rewards': self.current_episode_rewards,
            'weight_clip_kappa': self.weight_clip_kappa,
            '_update_counter': self._update_counter,
            # Reward normalization parameters
            'running_G': self.running_G,
            'running_G_mean': self.running_G_mean,
            'running_G_var': self.running_G_var,
            'running_G_max': self.running_G_max,
            'running_G_count': self.running_G_count,
            'reward_norm_G_max': self.reward_norm_G_max,
            'reward_norm_epsilon': self.reward_norm_epsilon,
            # Gaussian PPO parameters
            'variance_loss_coefficient': self.variance_loss_coefficient,
            'use_uncertainty_weight': self.use_uncertainty_weight,
            'uncertainty_weight_type': self.uncertainty_weight_type,
            'uncertainty_weight_temp': self.uncertainty_weight_temp
            # Removed C51 params
        })
        if self.aux_task_manager and hasattr(self.aux_task_manager, 'get_state_dict'):
             state['aux_task_manager'] = self.aux_task_manager.get_state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load state dict for resuming algorithm state"""
        super().load_state_dict(state_dict)
        if 'optimizer' in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
            except ValueError as e:
                print(f"Warning: Could not load optimizer state, likely due to parameter mismatch. Optimizer state reset. Error: {e}")
                # Re-initialize optimizer
                if self.shared_model: current_params = list(self.actor.parameters())
                else: current_params = list(self.actor.parameters()) + list(self.critic.parameters())
                if self.aux_task_manager:
                    if hasattr(self.aux_task_manager, 'sr_task') and self.aux_task_manager.sr_task is not None: current_params += list(self.aux_task_manager.sr_task.parameters())
                    if hasattr(self.aux_task_manager, 'rp_task') and self.aux_task_manager.rp_task is not None: current_params += list(self.aux_task_manager.rp_task.parameters())
                self.optimizer = torch.optim.Adam(current_params, lr=self.lr_actor)

        if 'scaler' in state_dict: self.scaler.load_state_dict(state_dict['scaler'])
        if 'memory_state' in state_dict and hasattr(self.memory, 'load_state_dict'): self.memory.load_state_dict(state_dict['memory_state'])
        if 'episode_returns' in state_dict: self.episode_returns = deque(state_dict['episode_returns'], maxlen=self.episode_returns.maxlen)
        if 'current_episode_rewards' in state_dict: self.current_episode_rewards = state_dict['current_episode_rewards']
        if 'weight_clip_kappa' in state_dict: self.weight_clip_kappa = state_dict['weight_clip_kappa']
        if '_update_counter' in state_dict: self._update_counter = state_dict['_update_counter']

        # Load reward normalization parameters
        if 'running_G' in state_dict: self.running_G = state_dict['running_G']
        if 'running_G_mean' in state_dict: self.running_G_mean = state_dict['running_G_mean']
        if 'running_G_var' in state_dict: self.running_G_var = state_dict['running_G_var']
        if 'running_G_max' in state_dict: self.running_G_max = state_dict['running_G_max']
        if 'running_G_count' in state_dict: self.running_G_count = state_dict['running_G_count']
        if 'reward_norm_G_max' in state_dict: self.reward_norm_G_max = state_dict['reward_norm_G_max']
        if 'reward_norm_epsilon' in state_dict: self.reward_norm_epsilon = state_dict['reward_norm_epsilon']

        # Load Gaussian PPO parameters
        if 'variance_loss_coefficient' in state_dict: self.variance_loss_coefficient = state_dict['variance_loss_coefficient']
        if 'use_uncertainty_weight' in state_dict: self.use_uncertainty_weight = state_dict['use_uncertainty_weight']
        if 'uncertainty_weight_type' in state_dict: self.uncertainty_weight_type = state_dict['uncertainty_weight_type']
        if 'uncertainty_weight_temp' in state_dict: self.uncertainty_weight_temp = state_dict['uncertainty_weight_temp']
        # Removed C51 params

        if 'aux_task_manager' in state_dict and self.aux_task_manager and hasattr(self.aux_task_manager, 'load_state_dict'):
            self.aux_task_manager.load_state_dict(state_dict['aux_task_manager'])
