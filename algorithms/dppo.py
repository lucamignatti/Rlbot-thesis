import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import math
from typing import Dict, Tuple, List, Optional, Union, Any
# Assuming .base exists relative to this file's location
# Adjust the import path based on your project structure
try:
    from .base import BaseAlgorithm
except ImportError:
    # Fallback if running script directly or .base isn't found
    class BaseAlgorithm: # Dummy class for type hinting
        def __init__(self, *args, **kwargs): pass
        def get_state_dict(self): return {}
        def load_state_dict(self, state_dict): pass
# Import AuxiliaryTaskManager to use its methods
# Assuming auxiliary is a module in your project
try:
    from auxiliary import AuxiliaryTaskManager
except ImportError:
    AuxiliaryTaskManager = None # Define as None if not available
# Import GradScaler for AMP
from torch.amp import GradScaler, autocast
# Import l2_norm for SimbaV2 weight projection
from model_architectures.utils import l2_norm

class DPPOAlgorithm(BaseAlgorithm):
    """
    PPO algorithm implementation with Categorical Distributional Critic (C51-style)
    and enhanced features based on the distribution.

    Key features:
    1. Categorical representation of value distribution (logits over fixed support).
    2. KL divergence loss for critic training.
    3. **Quantile Sampling for GAE:** GAE calculation uses values sampled from the
       predicted distribution, introducing stochasticity reflecting the distribution.
    4. **Adaptive Clipping:** PPO clip range epsilon is adapted based on the
       variance of the predicted value distribution (lower epsilon for higher variance).
    5. **Confidence Weighting:** PPO objective is weighted based on the confidence
       (inverse entropy or variance) of the critic's prediction.
    """

    def __init__(
        self,
        actor,
        critic, # Expects critic to output logits for categorical distribution [batch, num_atoms]
        aux_task_manager: Optional['AuxiliaryTaskManager'] = None,
        action_space_type="discrete",
        action_dim=None,
        action_bounds=(-1.0, 1.0),
        device="cuda",
        lr_actor=3e-4,
        lr_critic=3e-4, # Note: If shared_model, only lr_actor used for combined optimizer
        buffer_size=131072,
        gamma=0.99,
        gae_lambda=0.95,
        # --- PPO Clipping Parameters ---
        epsilon_base=0.2,          # Base clipping value if not adaptive
        use_adaptive_epsilon=True, # Whether to adapt epsilon based on variance
        adaptive_epsilon_beta=1.0, # Factor controlling sensitivity to variance (higher beta -> more sensitive)
        epsilon_min=0.05,          # Minimum allowed epsilon
        epsilon_max=0.3,           # Maximum allowed epsilon
        # -----------------------------
        critic_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        batch_size=64,
        use_amp=False,
        debug=False,
        use_wandb=False,
        # --- Categorical Critic Parameters ---
        v_min=-10.0,
        v_max=10.0,
        num_atoms=51,
        # --- Confidence Weighting Parameters ---
        use_confidence_weighting=True,      # Whether to weight PPO loss term by confidence
        confidence_weight_type="entropy",   # "entropy" or "variance" to measure inverse confidence
        confidence_weight_delta=1e-6,       # Epsilon for numerical stability in weighting denominator (if using inverse)
        normalize_confidence_weights=True,  # Normalize weights across batch for stability
        # --- Quantile Sampling (Enabled implicitly by GAE modification, no specific params needed here) ---
        # --- SimbaV2 Reward Normalization G_max ---
        reward_norm_G_max=10.0, # Default G_max for reward normalization
        # --- Deprecated / Unused Weight Clipping Params ---
        use_weight_clipping=False,
        weight_clip_kappa=1.0,
        adaptive_kappa=False,
        kappa_update_freq=10,
        kappa_update_rate=0.01,
        target_clip_fraction=0.05,
        min_kappa=0.1,
        max_kappa=10.0,
    ):
        super().__init__(
            actor=actor, critic=critic, action_space_type=action_space_type,
            action_dim=action_dim, action_bounds=action_bounds, device=device,
            lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, gae_lambda=gae_lambda,
            # Pass base epsilon, critic_coef, etc. to BaseAlgorithm if it uses them
            clip_epsilon=epsilon_base, # BaseAlgorithm might expect this name
            critic_coef=critic_coef, entropy_coef=entropy_coef, max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs, batch_size=batch_size, use_amp=use_amp,
            debug=debug, use_wandb=use_wandb
        )

        self.aux_task_manager = aux_task_manager
        self.shared_model = (actor is critic)
        if self.debug and self.shared_model:
            print("[DEBUG PPO] Actor and Critic are the same instance (shared model).")

        # --- Categorical Critic Params ---
        self.v_min = v_min
        self.v_max = v_max
        if num_atoms <= 1: raise ValueError("num_atoms must be > 1 for Categorical critic")
        self.num_atoms = num_atoms
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # --- Adaptive Clipping Params ---
        self.epsilon_base = epsilon_base
        self.use_adaptive_epsilon = use_adaptive_epsilon
        self.adaptive_epsilon_beta = adaptive_epsilon_beta
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max

        # --- Confidence Weighting Params ---
        if confidence_weight_type not in ["entropy", "variance", "none"]:
             raise ValueError(f"Invalid confidence_weight_type: {confidence_weight_type}")
        self.use_confidence_weighting = use_confidence_weighting if confidence_weight_type != "none" else False
        self.confidence_weight_type = confidence_weight_type
        self.confidence_weight_delta = confidence_weight_delta
        self.normalize_confidence_weights = normalize_confidence_weights

        # --- Deprecated Weight Clipping ---
        self.use_weight_clipping = use_weight_clipping
        self.weight_clip_kappa = weight_clip_kappa
        self.adaptive_kappa = adaptive_kappa
        self.kappa_update_freq = kappa_update_freq
        self.kappa_update_rate = kappa_update_rate
        self.target_clip_fraction = target_clip_fraction
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa
        self._update_counter = 0
        # --- Reward Normalization ---
        self.reward_norm_G_max = reward_norm_G_max if reward_norm_G_max is not None else self.v_max # Use v_max if G_max not set

        # Initialize memory
        self.memory = self.PPOMemory(
            algorithm_instance=self, batch_size=batch_size, buffer_size=buffer_size,
            device=device, debug=debug, action_space_type=self.action_space_type
        )

        # Initialize optimizer
        if self.shared_model: combined_params = list(self.actor.parameters())
        else: combined_params = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.aux_task_manager:
             if hasattr(self.aux_task_manager, 'get_parameters'): combined_params += self.aux_task_manager.get_parameters()
             else: # Fallback
                 if hasattr(self.aux_task_manager, 'sr_task') and self.aux_task_manager.sr_task: combined_params += list(self.aux_task_manager.sr_task.parameters())
                 if hasattr(self.aux_task_manager, 'rp_task') and self.aux_task_manager.rp_task: combined_params += list(self.aux_task_manager.rp_task.parameters())
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, combined_params), lr=lr_actor)

        # Initialize GradScaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Initialize metrics tracking
        self.metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0, # KL Divergence loss
            'entropy_loss': 0.0,
            'sr_loss_scalar': 0.0,
            'rp_loss_scalar': 0.0,
            'total_loss': 0.0,
            'clip_fraction': 0.0,
            'explained_variance': 0.0, # Based on mean prediction vs returns
            'kl_divergence': 0.0, # Critic KL divergence
            'mean_advantage': 0.0, # Mean of the SAMPLED advantage used
            'mean_return': 0.0,
            'weight_clip_fraction': 0.0,
            'current_kappa': self.weight_clip_kappa,
            'mean_confidence_weight': 1.0,
            'min_confidence_weight': 1.0,
            'max_confidence_weight': 1.0,
            'mean_adaptive_epsilon': self.epsilon_base,
            'min_adaptive_epsilon': self.epsilon_base,
            'max_adaptive_epsilon': self.epsilon_base,
            'mean_critic_variance': 0.0, # Variance calculated from categorical distribution
            'mean_critic_entropy': 0.0, # Entropy calculated from categorical distribution
            'mean_sampled_value_gae': 0.0, # Mean of the value sampled for GAE input
        }
        self.current_episode_rewards = []
        self.episode_returns = deque(maxlen=100)

        # Reward normalization state
        self.running_G = 0.0
        self.running_G_mean = 0.0
        self.running_G_var = 1.0
        self.running_G_max = 0.0
        self.running_G_count = 0
        self.reward_norm_epsilon = 1e-8


    # --- PPOMemory Class (No changes needed from previous version, keep as inner class) ---
    class PPOMemory:
        """Memory buffer for PPO to store experiences"""
        def __init__(self, algorithm_instance, batch_size, buffer_size, device, debug=False, action_space_type="discrete"):
            self.algorithm_instance = algorithm_instance
            self.batch_size = batch_size
            self.debug = debug
            self.action_space_type = action_space_type
            self.buffer_size = buffer_size
            self.device = device
            self._reset_buffers()

        def _reset_buffers(self):
            buffer_size = self.buffer_size
            device = self.device
            self.obs = None
            self.actions = None
            self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device) # Stores SAMPLED values used for GAE
            self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
            self.pos = 0
            self.full = False
            self.size = 0

        def _initialize_buffers_if_needed(self, obs_sample, action_sample):
            if self.obs is None:
                if not isinstance(obs_sample, torch.Tensor): obs_sample = torch.tensor(obs_sample, dtype=torch.float32)
                action_dtype = torch.long if self.action_space_type == "discrete" else torch.float32
                if not isinstance(action_sample, torch.Tensor): action_sample = torch.tensor(action_sample, dtype=action_dtype)
                if obs_sample.dim() == 1: obs_sample = obs_sample.unsqueeze(0)
                if self.action_space_type == "discrete": action_shape = ()
                else:
                    if action_sample.numel() == 1 and action_sample.dim() == 0: action_shape = (1,)
                    elif action_sample.dim() == 1: action_shape = action_sample.shape
                    elif action_sample.dim() > 1: action_shape = action_sample.shape[1:]
                    else: action_shape = (1,)
                obs_shape = obs_sample.shape[1:]
                if self.debug: print(f"[DEBUG PPOMemory] Initializing buffers: obs_shape={obs_shape}, action_shape={action_shape}, action_dtype={action_dtype}")
                self.obs = torch.zeros((self.buffer_size, *obs_shape), dtype=torch.float32, device=self.device)
                self.actions = torch.zeros((self.buffer_size, *action_shape), dtype=action_dtype, device=self.device)

        def store_initial_batch(self, obs_batch, action_batch, log_prob_batch, value_batch):
            # value_batch here is the SAMPLED value used for GAE calculation
            batch_size = obs_batch.shape[0]
            if batch_size == 0: return torch.tensor([], dtype=torch.long, device=self.device)
            expected_action_dtype = torch.long if self.action_space_type == "discrete" else torch.float32
            action_sample_for_init = action_batch[0].clone().to(expected_action_dtype)
            self._initialize_buffers_if_needed(obs_batch[0], action_sample_for_init)
            if not isinstance(obs_batch, torch.Tensor): obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
            if not isinstance(action_batch, torch.Tensor): action_batch = torch.tensor(action_batch, dtype=expected_action_dtype)
            elif action_batch.dtype != expected_action_dtype: action_batch = action_batch.to(expected_action_dtype)
            if not isinstance(log_prob_batch, torch.Tensor): log_prob_batch = torch.tensor(log_prob_batch, dtype=torch.float32)
            if not isinstance(value_batch, torch.Tensor): value_batch = torch.tensor(value_batch, dtype=torch.float32)
            target_device = self.device
            obs_batch, action_batch, log_prob_batch, value_batch = (t.to(target_device) for t in (obs_batch, action_batch, log_prob_batch, value_batch))
            indices = torch.arange(self.pos, self.pos + batch_size, device=target_device) % self.buffer_size
            self.obs.index_copy_(0, indices, obs_batch.detach())
            if self.action_space_type == "discrete":
                if action_batch.dim() == 1 and self.actions.dim() == 1: self.actions.index_copy_(0, indices, action_batch.detach())
                elif action_batch.dim() == 2 and action_batch.shape[1] == 1 and self.actions.dim() == 1: self.actions.index_copy_(0, indices, action_batch.detach().squeeze(1))
                else:
                     if self.debug: print(f"[DEBUG PPOMemory] Discrete action shape mismatch. Buffer: {self.actions.shape}, Batch: {action_batch.shape}")
                     if self.actions.shape[1:] == () and action_batch.numel() == batch_size: self.actions.index_copy_(0, indices, action_batch.reshape(batch_size).detach())
            else: # Continuous
                if self.actions.shape[1:] == action_batch.shape[1:]: self.actions.index_copy_(0, indices, action_batch.detach())
                else:
                    if self.actions.numel() // self.buffer_size == action_batch.numel() // batch_size:
                         target_shape = (batch_size, *self.actions.shape[1:])
                         if self.debug: print(f"[DEBUG PPOMemory] Continuous action shape mismatch. Reshaping Batch: {action_batch.shape} to {target_shape}")
                         try: self.actions.index_copy_(0, indices, action_batch.reshape(target_shape).detach())
                         except RuntimeError as e:
                              if self.debug: print(f"[DEBUG PPOMemory] Reshape failed: {e}")
                    elif self.debug: print(f"[DEBUG PPOMemory] Continuous action shape mismatch and incompatible numel.")

            self.log_probs.index_copy_(0, indices, log_prob_batch.detach())
            self.values.index_copy_(0, indices, value_batch.detach()) # Store SAMPLED value
            new_pos = (self.pos + batch_size) % self.buffer_size
            if not self.full and (self.pos + batch_size >= self.buffer_size): self.full = True
            self.pos = new_pos
            self.size = min(self.size + batch_size, self.buffer_size)
            # Removed redundant debug print
            return indices

        def update_rewards_dones_batch(self, indices, rewards_batch, dones_batch):
            if len(indices) == 0: return
            if not isinstance(rewards_batch, torch.Tensor): rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
            if not isinstance(dones_batch, torch.Tensor): dones_batch = torch.tensor(dones_batch, dtype=torch.bool)
            target_device = self.device
            indices, rewards_batch, dones_batch = (t.to(target_device) for t in (indices, rewards_batch, dones_batch))
            normed_rewards = self.algorithm_instance.normalize_reward(rewards_batch, dones_batch)
            if not isinstance(normed_rewards, torch.Tensor): normed_rewards = torch.tensor(normed_rewards, dtype=torch.float32, device=target_device)
            else: normed_rewards = normed_rewards.to(target_device)
            self.rewards.index_copy_(0, indices, normed_rewards)
            self.dones.index_copy_(0, indices, dones_batch)
            # Removed debug print

        def get(self):
            if self.size == 0 or self.obs is None:
                if self.debug: print("[DEBUG PPOMemory] get() called but buffer is empty or uninitialized.")
                return None, None, None, None, None, None
            obs_data, actions_data, log_probs_data, rewards_data, values_data, dones_data = (
                self.obs[:self.size], self.actions[:self.size], self.log_probs[:self.size],
                self.rewards[:self.size], self.values[:self.size], self.dones[:self.size]
            )
            # Removed debug print
            return obs_data, actions_data, log_probs_data, rewards_data, values_data, dones_data

        def generate_batches(self):
            if self.size < self.batch_size: # Return empty if not enough data for even one batch
                if self.debug: print(f"[DEBUG PPOMemory] generate_batches() size {self.size} < batch_size {self.batch_size}.")
                return []
            indices = torch.randperm(self.size, device=self.device)
            batch_start = 0; batches = []
            while batch_start < self.size:
                batch_end = min(batch_start + self.batch_size, self.size)
                # Ensure we don't create batches smaller than needed if size isn't multiple of batch_size
                # For PPO, typically we use all data, so drop last partial batch if desired, or handle it.
                # Current logic includes partial last batch.
                batches.append(indices[batch_start:batch_end])
                batch_start = batch_end
            # Removed debug print
            return batches

        def clear(self):
            self._reset_buffers()
            # Removed debug print

    # --- Method Implementations ---

    def store_initial_batch(self, obs_batch, action_batch, log_prob_batch, value_batch):
        """Store the initial part of experiences (value_batch should be SAMPLED value)."""
        if self.debug: print(f"[DEBUG PPO] Storing initial batch, size {obs_batch.shape[0]}. Sampled value mean: {value_batch.mean().item():.4f}")
        return self.memory.store_initial_batch(obs_batch, action_batch, log_prob_batch, value_batch)

    def update_rewards_dones_batch(self, indices, rewards_batch, dones_batch):
        """Update rewards and dones, track episode returns."""
        # Removed debug print
        self.memory.update_rewards_dones_batch(indices, rewards_batch, dones_batch)
        done_indices_local = torch.where(dones_batch.cpu())[0]
        if len(done_indices_local) > 0:
            reward_values = rewards_batch.cpu().numpy()
            for i in done_indices_local:
                self.current_episode_rewards.append(float(reward_values[i]))
                episode_return = sum(self.current_episode_rewards)
                self.episode_returns.append(episode_return)
                # Removed debug print
                self.current_episode_rewards = []


    def get_action(self, obs, deterministic=False, return_features=False):
        """Get action using actor. Value returned is the MEAN expected value for info."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if self.memory.obs is not None and obs.dim() == len(self.memory.obs.shape) - 1:
            obs = obs.unsqueeze(0)
        elif self.memory.obs is not None and obs.dim() < len(self.memory.obs.shape) - 1:
             target_shape = self.memory.obs.shape[1:]
             try: obs = obs.reshape(1, *target_shape)
             except RuntimeError as e:
                 if self.debug: print(f"[DEBUG PPO get_action] Obs reshape error: {e}"); # Return dummy values
                 dummy_action_shape = (1,) if self.action_space_type == "discrete" else (1, self.action_dim or 1)
                 dummy_action_dtype = torch.long if self.action_space_type == "discrete" else torch.float
                 dummy_action = torch.zeros(dummy_action_shape, dtype=dummy_action_dtype, device=self.device)
                 dummy_log_prob = torch.zeros(1, device=self.device); dummy_value = torch.zeros_like(dummy_log_prob)
                 return (dummy_action, dummy_log_prob, dummy_value, None) if return_features else (dummy_action, dummy_log_prob, dummy_value)


        with torch.no_grad():
            # Actor forward pass
            if self.shared_model:
                model_output = self.actor(obs, return_actor=True, return_critic=False, return_features=return_features)
                actor_output = model_output.get('actor_out'); features = model_output.get('features')
            else:
                actor_result = self.actor(obs, return_features=return_features)
                if return_features: actor_output, features = actor_result
                else: actor_output = actor_result; features = None
            if actor_output is None: # Handle error case
                 if self.debug: print("[DEBUG PPO get_action] Actor output is None."); # Return dummy values
                 dummy_action_shape = (1,) if self.action_space_type == "discrete" else (1, self.action_dim or 1)
                 dummy_action_dtype = torch.long if self.action_space_type == "discrete" else torch.float
                 dummy_action = torch.zeros(dummy_action_shape, dtype=dummy_action_dtype, device=self.device)
                 dummy_log_prob = torch.zeros(1, device=self.device); dummy_value = torch.zeros_like(dummy_log_prob)
                 return (dummy_action, dummy_log_prob, dummy_value, features) if return_features else (dummy_action, dummy_log_prob, dummy_value)


            # Action selection
            if self.action_space_type == "discrete":
                probs = F.softmax(actor_output, dim=-1); probs = torch.clamp(probs, min=1e-10)
                if deterministic: action = torch.argmax(probs, dim=-1)
                else:
                    try: dist = torch.distributions.Categorical(probs=probs); action = dist.sample()
                    except ValueError: action = torch.argmax(probs, dim=-1) # Fallback
                try: dist = torch.distributions.Categorical(probs=probs); log_prob = dist.log_prob(action)
                except ValueError: log_prob = torch.zeros_like(action, dtype=torch.float32) # Fallback
            else: # Continuous
                action_dist = actor_output
                action = action_dist.loc if deterministic else action_dist.sample()
                if self.action_bounds: action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])
                log_prob = action_dist.log_prob(action).sum(dim=-1)

            # Get MEAN value from critic for external use/logging
            with torch.no_grad():
                 if self.shared_model:
                     model_output_critic = self.critic(obs, return_actor=False, return_critic=True, return_features=False)
                     critic_logits = model_output_critic.get('critic_out')
                 else: critic_logits = self.critic(obs)

                 if critic_logits is not None and critic_logits.shape == (obs.shape[0], self.num_atoms):
                     probs = F.softmax(critic_logits, dim=1)
                     support_device = self.support.to(probs.device)
                     value = (probs * support_device).sum(dim=1) # Mean value
                 else: value = torch.zeros_like(log_prob)

            # Squeeze if original input was unsqueezed
            if obs.shape[0] == 1 and action.shape[0] == 1:
                action, log_prob, value = action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)
                if features is not None: features = features.squeeze(0)

        return (action, log_prob, value, features) if return_features else (action, log_prob, value)

    def reset(self):
        self.memory.clear()
        self.current_episode_rewards = []

    def update(self):
        """Update policy using PPO with sampled GAE and adaptive clipping."""
        buffer_size = self.memory.size
        if buffer_size < self.batch_size:
            if self.debug: print(f"[DEBUG PPO] Buffer size ({buffer_size}) < batch size ({self.batch_size}), skipping update")
            if self.aux_task_manager: self.metrics['sr_loss_scalar'] = self.aux_task_manager.last_sr_loss; self.metrics['rp_loss_scalar'] = self.aux_task_manager.last_rp_loss
            return self.metrics

        if self.debug: print(f"[DEBUG PPO] Starting update with buffer size: {buffer_size}")

        states, actions, old_log_probs, rewards, _, dones = self.memory.get() # Ignore stored values

        if states is None:
            if self.debug: print("[DEBUG PPO] Failed to get experiences from buffer, skipping update")
            return self.metrics

        # --- CALCULATE SAMPLED VALUES FOR GAE (Modification 1: Quantile Sampling for GAE) ---
        values_sampled_for_gae_list = []
        all_logits_list = [] # Store all logits for critic loss later
        with torch.no_grad():
            for i in range(0, buffer_size, self.batch_size):
                chunk_states = states[i:min(i + self.batch_size, buffer_size)]
                with autocast("cuda", enabled=self.use_amp):
                    if self.shared_model:
                        model_output = self.critic(chunk_states, return_actor=False, return_critic=True, return_features=False)
                        chunk_logits = model_output.get('critic_out')
                    else:
                        chunk_logits = self.critic(chunk_states)

                    if chunk_logits is None or chunk_logits.shape != (chunk_states.shape[0], self.num_atoms):
                         if self.debug: print(f"[DEBUG PPO GAE] Critic output shape mismatch for GAE calc. Expected {(chunk_states.shape[0], self.num_atoms)}, got {chunk_logits.shape if chunk_logits is not None else 'None'}. Skipping update.")
                         self.memory.clear(); return self.metrics

                    # Sample from the categorical distribution
                    dist = torch.distributions.Categorical(logits=chunk_logits)
                    sampled_indices = dist.sample() # Shape [chunk_size]
                    support_device = self.support.to(sampled_indices.device)
                    value_sample = support_device[sampled_indices] # Shape [chunk_size]

                    values_sampled_for_gae_list.append(value_sample)
                    all_logits_list.append(chunk_logits) # Store logits

            values_sampled_for_gae = torch.cat(values_sampled_for_gae_list)
            all_logits = torch.cat(all_logits_list) # Logits for the whole buffer [buffer_size, num_atoms]

        # Store sampled values back into memory.values for _compute_gae
        self.memory.values = values_sampled_for_gae.detach()
        self.metrics['mean_sampled_value_gae'] = values_sampled_for_gae.mean().item() # Track sampled value mean
        if self.debug: print(f"[DEBUG PPO GAE] Calculated SAMPLED values for GAE. Mean: {values_sampled_for_gae.mean().item():.4f}, Std: {values_sampled_for_gae.std().item():.4f}")
        # --- END SAMPLED VALUES CALCULATION ---

        # --- COMPUTE GAE USING SAMPLED V ---
        # returns_target here is based on the SAMPLED values, used for critic loss target projection
        returns_target, advantages_sampled = self._compute_gae(rewards, values_sampled_for_gae, dones)
        # Note: advantages_sampled now inherently contains noise/info from the distribution sampling

        if torch.isnan(advantages_sampled).any() or torch.isinf(advantages_sampled).any():
            if self.debug: print("[DEBUG PPO] NaN or Inf detected in SAMPLED advantages, skipping update")
            self.memory.clear(); return self.metrics

        # --- Normalize Sampled Advantages ---
        adv_mean = advantages_sampled.mean()
        adv_std = advantages_sampled.std()
        advantages_normalized_sampled = (advantages_sampled - adv_mean) / (adv_std + 1e-8)
        self.metrics['mean_advantage'] = advantages_normalized_sampled.mean().item() # Track final advantage mean used

        # --- Prepare data dictionary for _update_policy ---
        update_data = {
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "returns_target": returns_target, # Target distribution will be projected from these scalar returns
            "advantages_normalized_sampled": advantages_normalized_sampled, # Use sampled advantages
            "all_logits": all_logits, # Pass logits for variance/entropy calculation
            "rewards": rewards # For aux tasks
        }

        metrics = self._update_policy(update_data)
        self.memory.clear()
        self.metrics.update(metrics)

        if len(self.episode_returns) > 0: self.metrics['mean_return'] = np.mean(self.episode_returns)
        self._update_counter += 1
        if self.debug: print(f"[DEBUG PPO] Update finished. Actor Loss: {metrics.get('actor_loss', 0):.4f}, Critic Loss: {metrics.get('critic_loss', 0):.4f}")
        return self.metrics

    # --- _compute_gae remains the same, uses values passed to it (which are now sampled) ---
    def _compute_gae(self, rewards, values_sampled, dones):
        """Compute returns and advantages using GAE with SAMPLED values."""
        buffer_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(buffer_size)):
            if t == buffer_size - 1: next_non_terminal, next_values = 1.0 - dones[t].float(), 0
            else: next_non_terminal, next_values = 1.0 - dones[t+1].float(), values_sampled[t+1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values_sampled[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        returns = advantages + values_sampled # Target returns are based on sampled advantages + sampled values
        if self.debug: print(f"[DEBUG PPO _compute_gae] Using SAMPLED values. Advantage Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}")
        return returns, advantages

    # --- _compute_confidence_weight (modified name and potential logic) ---
    def _compute_confidence_weight(self, predicted_probs):
        """Compute confidence weight (inverse uncertainty)."""
        if not self.use_confidence_weighting:
            return torch.ones(predicted_probs.shape[0], device=self.device)

        if self.confidence_weight_type == "entropy":
            eps = 1e-8
            log_probs = torch.log(predicted_probs + eps)
            entropy = -(predicted_probs * log_probs).sum(dim=1) # Higher entropy = lower confidence
            # Inverse relationship: Lower entropy -> Higher weight
            raw_weights = 1.0 / (entropy + self.confidence_weight_delta)
            if self.debug and torch.rand(1).item() < 0.01: print(f"[DEBUG PPO Confidence] Entropy range: {entropy.min().item():.4f}-{entropy.max().item():.4f}")

        elif self.confidence_weight_type == "variance":
            support = self.support.to(predicted_probs.device)
            expected_value = (predicted_probs * support).sum(dim=1)
            expected_value_squared = (predicted_probs * support.pow(2)).sum(dim=1)
            variance = F.relu(expected_value_squared - expected_value.pow(2)) # Higher variance = lower confidence
            # Inverse relationship: Lower variance -> Higher weight
            raw_weights = 1.0 / (variance + self.confidence_weight_delta)
            if self.debug and torch.rand(1).item() < 0.01: print(f"[DEBUG PPO Confidence] Variance range: {variance.min().item():.4f}-{variance.max().item():.4f}")

        else: # Should not happen
            if self.debug: print(f"[DEBUG PPO Confidence] Unknown type: {self.confidence_weight_type}, using weight 1.0")
            return torch.ones(predicted_probs.shape[0], device=self.device)

        # Normalize weights
        if self.normalize_confidence_weights:
            weights = raw_weights / (raw_weights.mean() + 1e-8)
        else:
            weights = raw_weights

        if self.debug and torch.rand(1).item() < 0.01: print(f"[DEBUG PPO Confidence] Weights ({self.confidence_weight_type}): min={weights.min().item():.4f}, max={weights.max().item():.4f}, mean={weights.mean().item():.4f}")

        return weights.detach() # Detach weights

    # --- _update_policy (Major changes here) ---
    def _update_policy(self, update_data: Dict[str, torch.Tensor]):
        """Update policy and value networks using PPO algorithm."""

        states = update_data["states"]
        actions = update_data["actions"]
        old_log_probs = update_data["old_log_probs"]
        returns_target = update_data["returns_target"] # Target scalar returns for projecting critic target dist
        advantages_normalized_sampled = update_data["advantages_normalized_sampled"] # Use SAMPLED GAE results
        all_logits = update_data["all_logits"] # Logits for the whole buffer
        rewards = update_data["rewards"]

        metrics_accum = {k: 0.0 for k in self.metrics if k not in ['mean_return', 'current_kappa']}
        metrics_accum['min_confidence_weight'] = float('inf'); metrics_accum['max_confidence_weight'] = float('-inf')
        metrics_accum['min_adaptive_epsilon'] = float('inf'); metrics_accum['max_adaptive_epsilon'] = float('-inf')
        num_batches_processed = 0

        # Calculate explained variance using the MEAN prediction vs GAE returns (based on SAMPLED values)
        explained_var_scalar = 0.0
        with torch.no_grad():
             all_probs = F.softmax(all_logits, dim=1)
             support_device = self.support.to(all_probs.device)
             y_pred_mean = (all_probs * support_device).sum(dim=1)
             y_true = returns_target # GAE returns (derived from sampled values)
             var_y = torch.var(y_true)
             explained_var = 1 - torch.var(y_true - y_pred_mean) / (var_y + 1e-8)
             explained_var_scalar = explained_var.item()
             metrics_accum['explained_variance'] = explained_var_scalar # Store EV


        for epoch in range(self.ppo_epochs):
            batch_indices_list = self.memory.generate_batches()
            if not batch_indices_list: continue
            if self.debug: print(f"[DEBUG PPO _update_policy] Epoch {epoch}: Processing {len(batch_indices_list)} batches.")

            for batch_indices in batch_indices_list:
                num_batches_processed += 1
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns_target = returns_target[batch_indices] # Target scalar returns for critic projection
                batch_advantages_sampled = advantages_normalized_sampled[batch_indices] # Use SAMPLED advantage
                batch_rewards = rewards[batch_indices]

                with autocast("cuda", enabled=self.use_amp):
                    # --- Recompute Actor/Critic outputs ---
                    try:
                        if self.shared_model:
                            model_output = self.actor(batch_states, return_actor=True, return_critic=True, return_features=True)
                            actor_output = model_output.get('actor_out'); critic_logits = model_output.get('critic_out'); current_features = model_output.get('features')
                        else:
                            actor_result = self.actor(batch_states, return_features=True)
                            critic_logits = self.critic(batch_states)
                            if isinstance(actor_result, tuple): actor_output, current_features = actor_result
                            else: actor_output = actor_result; current_features = None
                        if actor_output is None or critic_logits is None or critic_logits.shape != (batch_states.shape[0], self.num_atoms):
                            raise ValueError("Model output validation failed")
                    except Exception as e:
                        if self.debug: print(f"[DEBUG PPO _update_policy] Error getting model output: {e}"); continue

                    predicted_log_probs = F.log_softmax(critic_logits, dim=1)
                    predicted_probs = F.softmax(critic_logits, dim=1)

                    # --- Calculate Variance for Adaptive Epsilon ---
                    batch_variance = torch.tensor(0.0, device=self.device)
                    if self.use_adaptive_epsilon or self.confidence_weight_type == "variance":
                        support_device = self.support.to(predicted_probs.device)
                        expected_value = (predicted_probs * support_device).sum(dim=1)
                        expected_value_squared = (predicted_probs * support_device.pow(2)).sum(dim=1)
                        batch_variance = F.relu(expected_value_squared - expected_value.pow(2)) # [batch_size]
                        metrics_accum['mean_critic_variance'] += batch_variance.mean().item()

                    # --- Calculate Adaptive Epsilon (Modification 2) ---
                    if self.use_adaptive_epsilon:
                        # Higher variance -> lower epsilon (more conservative)
                        epsilon_t = self.epsilon_base / (1.0 + self.adaptive_epsilon_beta * batch_variance)
                        epsilon_t = torch.clamp(epsilon_t, self.epsilon_min, self.epsilon_max).detach() # Clamp and detach
                        metrics_accum['mean_adaptive_epsilon'] += epsilon_t.mean().item()
                        metrics_accum['min_adaptive_epsilon'] = min(metrics_accum['min_adaptive_epsilon'], epsilon_t.min().item())
                        metrics_accum['max_adaptive_epsilon'] = max(metrics_accum['max_adaptive_epsilon'], epsilon_t.max().item())
                    else:
                        epsilon_t = torch.tensor(self.epsilon_base, device=self.device) # Use fixed epsilon
                        metrics_accum['mean_adaptive_epsilon'] += self.epsilon_base
                        metrics_accum['min_adaptive_epsilon'] = min(metrics_accum['min_adaptive_epsilon'], self.epsilon_base)
                        metrics_accum['max_adaptive_epsilon'] = max(metrics_accum['max_adaptive_epsilon'], self.epsilon_base)

                    # --- Calculate Confidence Weights (Modification 3) ---
                    confidence_weights = self._compute_confidence_weight(predicted_probs) # Uses variance or entropy based on config
                    # Metrics are updated inside the helper function now

                    # --- Actor Loss Calculation ---
                    # Get current log probs and entropy
                    if self.action_space_type == "discrete":
                         action_probs_actor = F.softmax(actor_output, dim=-1); action_probs_actor = torch.clamp(action_probs_actor, min=1e-10)
                         try: dist = torch.distributions.Categorical(probs=action_probs_actor)
                         except ValueError: continue # Skip batch if probs invalid
                         actions_indices = batch_actions.long() if batch_actions.dtype != torch.long else batch_actions
                         try: curr_log_probs = dist.log_prob(actions_indices); entropy = dist.entropy().mean()
                         except (IndexError, ValueError): continue # Skip batch if action index invalid
                    else: # Continuous
                         action_dist = actor_output
                         if batch_actions.dtype != torch.float: batch_actions = batch_actions.float()
                         try: curr_log_probs = action_dist.log_prob(batch_actions).sum(dim=-1); entropy = action_dist.entropy().mean()
                         except Exception: continue # Skip batch

                    # PPO Objective using SAMPLED advantage and ADAPTIVE epsilon
                    ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages_sampled # Use SAMPLED advantage
                    # Use element-wise adaptive epsilon for clamping
                    surr2 = torch.clamp(ratio, 1.0 - epsilon_t, 1.0 + epsilon_t) * batch_advantages_sampled # Use SAMPLED advantage
                    ppo_objectives = torch.min(surr1, surr2)

                    # Apply confidence weighting
                    weighted_objectives = ppo_objectives * confidence_weights
                    actor_loss = -weighted_objectives.mean()
                    entropy_loss = -entropy * self.entropy_coef

                    # --- Critic Loss Calculation (KL Divergence) ---
                    with torch.no_grad(): # Target calculation
                        clamped_returns = batch_returns_target.clamp(min=self.v_min, max=self.v_max)
                        b = (clamped_returns - self.v_min) / self.delta_z
                        lower_bound_idx = b.floor().long().clamp(0, self.num_atoms - 1)
                        upper_bound_idx = b.ceil().long().clamp(0, self.num_atoms - 1)
                        upper_prob = b - b.floor()
                        lower_prob = 1.0 - upper_prob
                        target_p = torch.zeros_like(predicted_log_probs) # [batch_size, num_atoms]
                        target_p.scatter_add_(1, lower_bound_idx.unsqueeze(1), lower_prob.unsqueeze(1))
                        target_p.scatter_add_(1, upper_bound_idx.unsqueeze(1), upper_prob.unsqueeze(1))
                        target_p_stable = target_p + 1e-8 # Add epsilon for log stability

                    # KL(target || predicted) = sum(target * (log(target) - log(predicted)))
                    kl_div = (target_p_stable * (torch.log(target_p_stable) - predicted_log_probs)).sum(dim=1)
                    critic_loss = kl_div.mean()

                    # --- Auxiliary Loss ---
                    sr_loss = torch.tensor(0.0, device=self.device); rp_loss = torch.tensor(0.0, device=self.device)
                    sr_loss_scalar = 0.0; rp_loss_scalar = 0.0
                    if self.aux_task_manager and current_features is not None and \
                       (self.aux_task_manager.sr_weight > 0 or self.aux_task_manager.rp_weight > 0):
                         try:
                             aux_losses = self.aux_task_manager.compute_losses_for_batch(batch_states, batch_rewards, current_features)
                             sr_loss = aux_losses.get("sr_loss", torch.tensor(0.0, device=self.device))
                             rp_loss = aux_losses.get("rp_loss", torch.tensor(0.0, device=self.device))
                             sr_loss_scalar = aux_losses.get("sr_loss_scalar", 0.0)
                             rp_loss_scalar = aux_losses.get("rp_loss_scalar", 0.0)
                         except Exception as e:
                             if self.debug: print(f"[DEBUG PPO Aux] Error computing aux losses: {e}")

                    # --- Total Loss ---
                    total_loss = actor_loss + (self.critic_coef * critic_loss) + entropy_loss + sr_loss + rp_loss

                # --- Optimization Step ---
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0:
                     all_params = []; # Collect all trainable params
                     if self.shared_model: all_params.extend(p for p in self.actor.parameters() if p.requires_grad)
                     else: all_params.extend(p for p in self.actor.parameters() if p.requires_grad); all_params.extend(p for p in self.critic.parameters() if p.requires_grad)
                     if self.aux_task_manager:
                          if hasattr(self.aux_task_manager, 'get_parameters'): all_params.extend(p for p in self.aux_task_manager.get_parameters() if p.requires_grad)
                     if all_params: nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # --- Post-Optimization Steps ---
                self._project_weights() # For SimbaV2 OrthogonalLinear
                current_weight_clip_fraction = self.clip_weights() # Likely disabled
                metrics_accum['weight_clip_fraction'] += current_weight_clip_fraction

                # --- Accumulate Batch Metrics ---
                metrics_accum['actor_loss'] += actor_loss.item()
                metrics_accum['critic_loss'] += critic_loss.item()
                metrics_accum['entropy_loss'] += entropy_loss.item()
                metrics_accum['total_loss'] += total_loss.item()
                metrics_accum['sr_loss_scalar'] += sr_loss_scalar
                metrics_accum['rp_loss_scalar'] += rp_loss_scalar
                metrics_accum['kl_divergence'] += critic_loss.item() # Critic loss is KL div
                with torch.no_grad():
                    metrics_accum['clip_fraction'] += ((ratio - 1.0).abs() > epsilon_t).float().mean().item()
                # Accumulate confidence weight stats (mean is handled inside _compute_confidence_weight)
                # Min/Max were updated inside the helper
                # Accumulate entropy/variance used for weighting/clipping
                with torch.no_grad():
                     if self.confidence_weight_type == "entropy" or self.use_adaptive_epsilon:
                         eps=1e-8; log_p=torch.log(predicted_probs+eps); ent = -(predicted_probs*log_p).sum(dim=1)
                         metrics_accum['mean_critic_entropy'] += ent.mean().item()
                     # Variance was already calculated if needed for adaptive epsilon or confidence


        # --- Finalize Metrics ---
        final_metrics = {}
        if num_batches_processed > 0:
            for key, total_val in metrics_accum.items():
                if key in ['min_confidence_weight', 'max_confidence_weight', 'min_adaptive_epsilon', 'max_adaptive_epsilon', 'explained_variance']:
                    final_metrics[key] = total_val # Keep min/max/pre-calculated as is
                else: final_metrics[key] = total_val / num_batches_processed # Average others
        else: # Handle no batches processed
             if self.debug: print("[DEBUG PPO _update_policy] No batches processed.")
             for key in self.metrics:
                 if key not in ['mean_return', 'current_kappa']: final_metrics[key] = 0.0
             final_metrics['min_confidence_weight']=1.0; final_metrics['max_confidence_weight']=1.0
             final_metrics['min_adaptive_epsilon']=self.epsilon_base; final_metrics['max_adaptive_epsilon']=self.epsilon_base
             final_metrics['mean_adaptive_epsilon']=self.epsilon_base; final_metrics['mean_confidence_weight']=1.0

        final_metrics['current_kappa'] = self.weight_clip_kappa # Deprecated

        return final_metrics


    # --- Helper methods _project_weights, normalize_reward, get_state_dict, load_state_dict ---
    # (Keep these largely the same as previous version, ensure state dict saves/loads NEW params)
    def _project_weights(self):
        """Project weights of OrthogonalLinear layers (SimbaV2)."""
        # ... (Keep implementation from previous Gaussian version) ...
        try: from model_architectures.utils import OrthogonalLinear
        except ImportError:
            if self.debug and self._update_counter % 100 == 0: print("[DEBUG _project_weights] Could not import OrthogonalLinear.")
            return
        models_to_project = [self.actor]
        if not self.shared_model: models_to_project.append(self.critic)
        if self.aux_task_manager:
            if hasattr(self.aux_task_manager, 'get_parameters'): pass # Can't easily project if manager opaque
            else:
                if hasattr(self.aux_task_manager, 'sr_task') and self.aux_task_manager.sr_task: models_to_project.append(self.aux_task_manager.sr_task)
                if hasattr(self.aux_task_manager, 'rp_task') and self.aux_task_manager.rp_task: models_to_project.append(self.aux_task_manager.rp_task)
        with torch.no_grad():
            for model in models_to_project:
                if model is None: continue
                for module in model.modules():
                    if isinstance(module, OrthogonalLinear):
                        if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                           module.weight.copy_(l2_norm(module.weight))

    def normalize_reward(self, reward, done):
        """SimbaV2 reward normalization."""
        # ... (Keep implementation from previous Gaussian version) ...
        gamma = self.gamma; eps = self.reward_norm_epsilon; G_max = self.reward_norm_G_max
        is_tensor = isinstance(reward, torch.Tensor)
        device = reward.device if is_tensor else self.device
        reward_np = reward.detach().cpu().numpy() if is_tensor else np.array(reward)
        done_np = done.detach().cpu().numpy() if is_tensor else np.array(done)
        was_scalar = reward_np.ndim == 0
        if was_scalar: reward_np, done_np = reward_np.reshape(1), done_np.reshape(1)
        normed_rewards = np.empty_like(reward_np)
        for i in range(len(reward_np)):
            r, d = reward_np[i], done_np[i]
            if d: self.running_G = 0.0; self.running_G_max = 0.0
            self.running_G = gamma * self.running_G + r
            self.running_G_count += 1
            delta = self.running_G - self.running_G_mean
            self.running_G_mean += delta / self.running_G_count
            delta2 = self.running_G - self.running_G_mean
            self.running_G_var += delta * delta2
            self.running_G_max = max(self.running_G_max, abs(self.running_G))
            current_variance = self.running_G_var / self.running_G_count if self.running_G_count > 1 else 1.0
            std = np.sqrt(max(0.0, current_variance) + eps)
            denom_max_term = self.running_G_max / G_max if G_max > 0 else 0.0
            denom = max(std, denom_max_term, eps)
            normed_rewards[i] = r / denom
        if was_scalar: normed_rewards = normed_rewards.item()
        return torch.tensor(normed_rewards, dtype=torch.float32, device=device) if is_tensor else normed_rewards


    def get_state_dict(self):
        """Get state dict for saving algorithm state"""
        state = super().get_state_dict()
        state.update({
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'episode_returns': list(self.episode_returns),
            'current_episode_rewards': self.current_episode_rewards,
            '_update_counter': self._update_counter,
            # Reward normalization
            'running_G': self.running_G, 'running_G_mean': self.running_G_mean,
            'running_G_var': self.running_G_var, 'running_G_max': self.running_G_max,
            'running_G_count': self.running_G_count, 'reward_norm_G_max': self.reward_norm_G_max,
            'reward_norm_epsilon': self.reward_norm_epsilon,
            # Categorical Critic
            'v_min': self.v_min, 'v_max': self.v_max, 'num_atoms': self.num_atoms,
            # Adaptive Clipping
            'epsilon_base': self.epsilon_base, 'use_adaptive_epsilon': self.use_adaptive_epsilon,
            'adaptive_epsilon_beta': self.adaptive_epsilon_beta, 'epsilon_min': self.epsilon_min,
            'epsilon_max': self.epsilon_max,
            # Confidence Weighting
            'use_confidence_weighting': self.use_confidence_weighting,
            'confidence_weight_type': self.confidence_weight_type,
            'confidence_weight_delta': self.confidence_weight_delta,
            'normalize_confidence_weights': self.normalize_confidence_weights,
            # Deprecated
            'use_weight_clipping': self.use_weight_clipping, 'weight_clip_kappa': self.weight_clip_kappa,
        })
        if self.aux_task_manager and hasattr(self.aux_task_manager, 'get_state_dict'):
             state['aux_task_manager'] = self.aux_task_manager.get_state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load state dict for resuming algorithm state"""
        super().load_state_dict(state_dict)
        if 'optimizer' in state_dict:
            try: self.optimizer.load_state_dict(state_dict['optimizer'])
            except ValueError as e: print(f"Warning: Could not load optimizer state: {e}")
        if 'scaler' in state_dict: self.scaler.load_state_dict(state_dict['scaler'])
        if 'episode_returns' in state_dict: self.episode_returns = deque(state_dict['episode_returns'], maxlen=self.episode_returns.maxlen)
        if 'current_episode_rewards' in state_dict: self.current_episode_rewards = state_dict['current_episode_rewards']
        if '_update_counter' in state_dict: self._update_counter = state_dict['_update_counter']
        # Load reward normalization
        if 'running_G' in state_dict: self.running_G = state_dict['running_G']
        if 'running_G_mean' in state_dict: self.running_G_mean = state_dict['running_G_mean']
        if 'running_G_var' in state_dict: self.running_G_var = state_dict['running_G_var']
        if 'running_G_max' in state_dict: self.running_G_max = state_dict['running_G_max']
        if 'running_G_count' in state_dict: self.running_G_count = state_dict['running_G_count']
        if 'reward_norm_G_max' in state_dict: self.reward_norm_G_max = state_dict['reward_norm_G_max']
        if 'reward_norm_epsilon' in state_dict: self.reward_norm_epsilon = state_dict['reward_norm_epsilon']
        # Load Categorical Critic (recompute support based on loaded values)
        if 'v_min' in state_dict: self.v_min = state_dict['v_min']
        if 'v_max' in state_dict: self.v_max = state_dict['v_max']
        if 'num_atoms' in state_dict: self.num_atoms = state_dict['num_atoms']
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        # Load Adaptive Clipping
        if 'epsilon_base' in state_dict: self.epsilon_base = state_dict['epsilon_base']
        if 'use_adaptive_epsilon' in state_dict: self.use_adaptive_epsilon = state_dict['use_adaptive_epsilon']
        if 'adaptive_epsilon_beta' in state_dict: self.adaptive_epsilon_beta = state_dict['adaptive_epsilon_beta']
        if 'epsilon_min' in state_dict: self.epsilon_min = state_dict['epsilon_min']
        if 'epsilon_max' in state_dict: self.epsilon_max = state_dict['epsilon_max']
        # Load Confidence Weighting
        if 'use_confidence_weighting' in state_dict: self.use_confidence_weighting = state_dict['use_confidence_weighting']
        if 'confidence_weight_type' in state_dict: self.confidence_weight_type = state_dict['confidence_weight_type']
        if 'confidence_weight_delta' in state_dict: self.confidence_weight_delta = state_dict['confidence_weight_delta']
        if 'normalize_confidence_weights' in state_dict: self.normalize_confidence_weights = state_dict['normalize_confidence_weights']
        # Load Deprecated
        if 'use_weight_clipping' in state_dict: self.use_weight_clipping = state_dict['use_weight_clipping']
        if 'weight_clip_kappa' in state_dict: self.weight_clip_kappa = state_dict['weight_clip_kappa']

        if 'aux_task_manager' in state_dict and self.aux_task_manager and hasattr(self.aux_task_manager, 'load_state_dict'):
             self.aux_task_manager.load_state_dict(state_dict['aux_task_manager'])

    # --- Weight Clipping methods (kept for compatibility, likely unused) ---
    def init_weight_ranges(self): pass
    def clip_weights(self): return 0.0
