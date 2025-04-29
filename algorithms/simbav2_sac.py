import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Categorical
from typing import Dict, Tuple, List, Optional, Union, Any
from collections import deque

# Assuming base.py and sac.py are in the same directory or accessible
from .base import BaseAlgorithm
from .sac import ReplayBuffer, SACAlgorithm # Reuse ReplayBuffer, maybe inherit SACAlgorithm

# Helper function for L2 normalization (as used in SimbaV2 architecture)
def l2_normalize(x, dim=-1, eps=1e-8):
    return x * torch.rsqrt(torch.sum(x * x, dim=dim, keepdim=True) + eps)

# Placeholder for RunningMeanStd (needed for reward scaling variance)
class RunningMeanStd:
    # Simple implementation for demonstration
    def __init__(self, shape=(), epsilon=1e-4, device='cpu'):
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = epsilon # Use epsilon for initial count stability

    def update(self, x):
        # Welford's online algorithm
        batch_mean = torch.mean(x, dim=0, dtype=torch.float64)
        batch_var = torch.var(x, dim=0) # Removed invalid dtype argument
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        M2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return torch.sqrt(self.var)

    def normalize(self, x, update=False):
         if update:
             self.update(x)
         # Normalize using running stats
         x = x.to(self.mean.device) # Ensure device match
         normalized_x = (x - self.mean.float()) / (self.std.float() + 1e-8)
         return normalized_x

# --- SimbaV2 SAC Algorithm ---
class SimbaV2Algorithm(SACAlgorithm): # Inherit from corrected SAC for structure reuse
    """
    SimbaV2 Soft Actor-Critic (SAC) variant implementation outline.

    Based on the paper: "Hyperspherical Normalization for Scalable Deep Reinforcement Learning"
    Key differences from standard SAC:
    1. Uses specific SimbaV2 network architecture (Hyperspherical Norm, Scalers, LERP).
       - NOTE: Actor/Critic modules passed to this class MUST implement this architecture.
    2. Employs a Distributional Critic with KL-divergence loss.
    3. Implements Reward Bounding and Scaling.
    """

    def __init__(
        self,
        actor: nn.Module, # Must be SimbaV2 architecture
        critic1: nn.Module, # Must be SimbaV2 architecture (Distributional)
        critic2: nn.Module, # Must be SimbaV2 architecture (Distributional)
        action_space_type: str = "continuous",
        action_dim: int = None,
        observation_shape: tuple = None,
        action_bounds: Optional[Tuple[float, float]] = (-1.0, 1.0),
        device: str = "cpu",
        lr_actor: float = 1e-4, # SimbaV2 paper used 1e-4 -> 3e-5 decay
        lr_critic: float = 1e-4,
        gamma: float = 0.99, # SimbaV2 paper uses heuristic based on env max steps
        tau: float = 5e-3, # SimbaV2 paper used 5e-3
        alpha: float = 1e-2, # Initial temperature, SimbaV2 uses 1e-2
        auto_alpha_tuning: bool = True,
        target_entropy: Optional[float] = None, # SimbaV2 uses -|A|/2
        buffer_size: int = 1000000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        update_freq: int = 1,
        updates_per_step: int = 1, # UTD ratio (SimbaV2 paper explores 1, 2, 4, 8)
        max_grad_norm: float = 0.0, # SimbaV2 doesn't explicitly mention grad clipping
        use_amp: bool = False,
        use_wandb: bool = False,
        debug: bool = False,
        # --- SimbaV2 Specific Params ---
        critic_num_atoms: int = 101, # n_atoms
        critic_return_min: float = -5.0, # G_min
        critic_return_max: float = 5.0, # G_max
        reward_scale_epsilon: float = 1e-8,
        # Architecture params (should ideally be part of network definition)
        # shift_const: float = 3.0,
        # num_blocks_actor: int = 1,
        # num_blocks_critic: int = 2,
        # hidden_dim_actor: int = 128,
        # hidden_dim_critic: int = 512,
        **kwargs
    ):
        print("[INFO] Initializing SimbaV2 Algorithm variant.")
        # --- CRITICAL NOTE ---
        # The provided actor and critic modules MUST implement the SimbaV2 architecture
        # (ℓ2-norm, Scalers, LERP, specific initializations/updates).
        # This class assumes those modules handle their internal architecture.
        # The critic modules must output distributions (e.g., logits over atoms).

        # Initialize metrics dict first (before super call)
        self.metrics = {
            'actor_loss': 0.0,
            'critic1_loss': 0.0,
            'critic2_loss': 0.0,
            'alpha_loss': 0.0,
            'alpha': alpha,  # Current temperature value
            'mean_q_value': 0.0,
            'mean_target_q': 0.0,
            'mean_entropy': 0.0,
            'mean_return': 0.0,
            'buffer_size': 0,
            'steps_per_second': 0.0,
        }

        super().__init__(
            actor=actor, critic1=critic1, critic2=critic2,
            action_space_type=action_space_type, action_dim=action_dim,
            observation_shape=observation_shape, action_bounds=action_bounds,
            device=device, lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma,
            tau=tau, alpha=alpha, auto_alpha_tuning=auto_alpha_tuning,
            target_entropy=target_entropy if target_entropy is not None else -action_dim/2.0, # SimbaV2 default
            buffer_size=buffer_size, batch_size=batch_size,
            warmup_steps=warmup_steps, update_freq=update_freq,
            updates_per_step=updates_per_step, max_grad_norm=max_grad_norm,
            use_amp=use_amp, debug=debug, **kwargs  # Remove use_wandb since we'll use metrics dict
        )

        # Distributional Critic Parameters
        self.num_atoms = critic_num_atoms
        self.g_min = critic_return_min
        self.g_max = critic_return_max
        self.delta_z = (self.g_max - self.g_min) / (self.num_atoms - 1)
        # Create the support atoms (fixed)
        self.support = torch.linspace(self.g_min, self.g_max, self.num_atoms, device=self.device)

        # Reward Scaling Parameters (Eq 17-19)
        self.reward_scale_epsilon = reward_scale_epsilon
        self.current_episode_G = 0.0 # Running discounted return within episode (Eq 17)
        # Need running variance of G and running max of G across all steps
        # Use torch implementation for better handling
        self.G_rms = RunningMeanStd(shape=(), device=self.device) # For variance of G
        self.G_max_running = torch.tensor(-torch.inf, device=self.device) # Track max G encountered


        # --- Override Optimizers if SimbaV2 uses specific ones ---
        # SimbaV2 paper mentions Adam without weight decay. The weight updates
        # (projection onto sphere) are handled within the network modules or a custom optimizer step.
        # Assuming standard Adam here, weight projection happens elsewhere.
        self._init_optimizers() # Re-init optimizers (standard Adam as per paper)

        # Initialize metrics dict with all stats we want to track
        self.metrics = {
            'actor_loss': 0.0,
            'critic1_loss': 0.0,
            'critic2_loss': 0.0,
            'critic_kl_loss': 0.0,  # Using KL loss instead of MSE
            'alpha_loss': 0.0,
            'alpha': alpha,  # Current temperature value
            'mean_q_value': 0.0,
            'mean_entropy': 0.0,
            'mean_return': 0.0,
            'mean_reward_scale': 1.0,
            'raw_episode_return': 0.0,
            'mean_raw_episode_return': 0.0,
            'G_variance': 0.0,
            'G_max_running': 0.0,
            'buffer_size': 0,
            'steps': 0,
            'train_steps': 0,
            'steps_per_second': 0.0,
        }
        self.current_episode_raw_rewards = []
        self.raw_episode_returns = deque(maxlen=100)
        # Removed action buffer initialization


    def get_action_fast(self, obs, deterministic=False, return_features=False):
        """Fast inference path for action selection with minimal overhead"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == len(self.memory.observations.shape) - 1:  # Check if batch dim is missing
            obs = obs.unsqueeze(0)  # Add batch dimension

        self.actor.eval()  # Set actor to evaluation mode
        features = None

        with torch.no_grad():
            # Get features and action distribution
            if return_features:
                if hasattr(self.actor, 'extract_features'):
                    action_output, features = self.actor(obs, return_features=True)
                else:
                    action_output = self.actor(obs)
            else:
                action_output = self.actor(obs)

            action_output_tensor = None # Store raw output
            dist_obj = None # Store distribution object

            if self.action_space_type == "discrete":
                action_probs = F.softmax(action_output, dim=-1)
                action_output_tensor = action_output # Store raw logits
                dist_obj = Categorical(probs=action_probs) # Create dist obj
                if deterministic:
                    action_idx = torch.argmax(action_probs, dim=-1, keepdim=True)
                else:
                    action_idx = dist_obj.sample().unsqueeze(-1)
                action = action_idx # Keep as tensor [batch, 1] for consistency
            else:  # Continuous
                action_output_tensor = action_output # Store raw mean, log_std
                mean, log_std = action_output.chunk(2, dim=-1)
                log_std = torch.clamp(log_std, min=-20, max=2)
                std = torch.exp(log_std)
                dist_obj = Normal(mean, std) # Create dist obj
                if deterministic:
                    action_raw = mean
                else:
                    action_raw = dist_obj.rsample()  # Use reparameterization

                # Apply tanh squashing and scale
                action = torch.tanh(action_raw)
                low, high = self.action_bounds
                action = action * (high - low) / 2.0 + (high + low) / 2.0
                # action shape is now [batch, action_dim]

        # Return action, action_output, dist, and features
        if return_features:
            # Ensure action has batch dim removed if it was added
            # action = action.squeeze(0) if action.shape[0] == 1 and obs.shape[0] == 1 else action
            return action, action_output_tensor, dist_obj, features
        # action = action.squeeze(0) if action.shape[0] == 1 and obs.shape[0] == 1 else action
        return action, action_output_tensor, dist_obj

    # Removed _store_action_data method

    def get_action(self, obs, deterministic=False, return_features=False):
        """
        Standard get_action that computes additional metrics. Uses get_action_fast internally.

        Args:
            obs: Observation tensor/array
            deterministic: If True, return the mean action
            return_features: If True, return intermediate features for auxiliary tasks

        Returns:
            Tuple containing (action, log_prob, dummy_value, features) if return_features=True
            or (action, log_prob, dummy_value) otherwise
        """
        # Call the optimized get_action_fast which now returns intermediates
        if return_features:
            # Unpack all return values including features
            action, action_output_tensor, dist_obj, features = self.get_action_fast(obs, deterministic, return_features=True)
        else:
            # Unpack only the needed values when features are not requested
            action, action_output_tensor, dist_obj = self.get_action_fast(obs, deterministic)
            features = None # Explicitly set features to None

        # --- TEMPORARY DIAGNOSTIC ---
        # Skip correct log_prob calculation, just create dummy tensors
        # Determine expected shape: Need sum over action dim if continuous
        if action.dim() > 1 and self.action_space_type != "discrete":
             log_prob_shape_ref = action.sum(dim=-1, keepdim=True)
        elif action.dim() == 1 and self.action_space_type != "discrete": # Handle case where batch dim might be missing temporarily
             log_prob_shape_ref = action.sum(dim=-1, keepdim=True) # Still sum over action dim
        else: # Discrete or already shape [batch, 1]
             log_prob_shape_ref = action
        log_prob = torch.zeros_like(log_prob_shape_ref) # Create zero tensor with expected shape
        dummy_value = torch.zeros_like(log_prob)
        # --- END DIAGNOSTIC ---

        # --- ORIGINAL LOG PROB CALC (COMMENTED OUT FOR DIAGNOSTIC) ---
        # with torch.no_grad():
        #     if self.action_space_type == "discrete":
        #         # Action from get_action_fast should be [batch, 1]
        #         action_idx = action.squeeze(-1) # Remove last dim for log_prob
        #         log_prob = dist_obj.log_prob(action_idx).unsqueeze(-1) # Calculate log_prob and add back dim
        #     else: # Continuous
        #         # We need the *raw* action before tanh for log_prob calculation
        #         # Re-calculate raw action from the scaled/squashed action
        #         # Clamp the action before inverse tanh to avoid issues at boundaries
        #         action_clamped = torch.clamp(action, self.action_bounds[0] + 1e-6, self.action_bounds[1] - 1e-6)
        #         action_scaled = (action_clamped - (self.action_bounds[0] + self.action_bounds[1]) / 2.0) / \
        #                       ((self.action_bounds[1] - self.action_bounds[0]) / 2.0)
        #         # Clamp scaled action before atanh
        #         action_raw = torch.atanh(torch.clamp(action_scaled, -0.999999, 0.999999))
        #
        #         log_prob = dist_obj.log_prob(action_raw).sum(dim=-1, keepdim=True)
        #         # Adjust for tanh squashing using the clamped scaled action
        #         log_prob = log_prob - torch.log(1 - action_scaled.pow(2) + 1e-7).sum(dim=-1, keepdim=True)

        # # Create dummy value
        # dummy_value = torch.zeros_like(log_prob)
        # --- END ORIGINAL ---

        # Final return, ensuring correct shape for action
        # Squeeze the batch dimension ONLY if the original input obs was unbatched
        # Need to check the original obs shape passed into THIS function
        original_obs_is_batched = True # Assume batched by default
        if isinstance(obs, torch.Tensor) and obs.dim() == 1: # Check if original obs was likely unbatched
            original_obs_is_batched = False
        elif isinstance(obs, np.ndarray) and obs.ndim == 1:
            original_obs_is_batched = False

        # Squeeze action batch dim only if original input was not batched
        action_squeezed = action if original_obs_is_batched else action.squeeze(0)

        if return_features:
            return action_squeezed, log_prob, dummy_value, features
        else:
            return action_squeezed, log_prob, dummy_value

    def store_experience(self, obs, action, reward, next_obs, done, info=None):
        """Store experience, applying SimbaV2 reward scaling."""

        # --- Optimized Tensor Handling ---
        # Convert obs to tensor if needed, ensuring it's on the correct device
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self.device)
        else:
            # Use np.asarray for robustness with lists/tuples before tensor conversion
            obs_tensor = torch.tensor(np.asarray(obs), dtype=torch.float32, device=self.device)

        # Convert next_obs to tensor if needed, ensuring it's on the correct device
        if isinstance(next_obs, torch.Tensor):
            next_obs_tensor = next_obs.to(self.device)
        else:
            next_obs_tensor = torch.tensor(np.asarray(next_obs), dtype=torch.float32, device=self.device)

        # Ensure tensors have the correct shape for the buffer
        # Assuming self.memory and self.memory.observations are initialized
        if not hasattr(self.memory, 'observations') or self.memory.observations is None:
             print("[ERROR] Replay buffer observations not initialized in store_experience!")
             # Handle error appropriately - maybe initialize buffer or raise exception
             # For now, let's try to proceed cautiously or return
             # Setting expected_shape to None might cause issues later, maybe return?
             # Let's assume buffer is initialized for now, but this check is good.
             expected_shape = obs_tensor.shape # Use current shape as fallback (might be wrong)

        else:
            expected_shape = self.memory.observations.shape[1:] # Get shape without batch dim


        # Reshape tensors if necessary, handle errors gracefully
        try:
            if obs_tensor.shape != expected_shape:
                if self.debug:
                    print(f"[DEBUG SimbaV2] Attempting to reshape obs_tensor from {obs_tensor.shape} to {expected_shape}")
                obs_tensor = obs_tensor.reshape(expected_shape)
            if next_obs_tensor.shape != expected_shape:
                if self.debug:
                    print(f"[DEBUG SimbaV2] Attempting to reshape next_obs_tensor from {next_obs_tensor.shape} to {expected_shape}")
                next_obs_tensor = next_obs_tensor.reshape(expected_shape)
        except RuntimeError as e:
            print(f"[ERROR] Cannot reshape tensors. Obs: {obs_tensor.shape}, NextObs: {next_obs_tensor.shape}, Expected: {expected_shape}. Error: {e}. Skipping add.")
            # Update prev_obs with the processed next_obs_tensor before returning
            self.prev_obs = next_obs_tensor if not done else None
            return
        # --- End Optimized Tensor Handling ---

        # Extract raw reward (ensure it's a float)
        raw_reward = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
        self.current_episode_raw_rewards.append(raw_reward)

        # --- SimbaV2 Reward Scaling ---
        current_G_var = self.G_rms.var.item()
        current_G_max = self.G_max_running.item()
        safe_G_max_term = max(current_G_max / self.g_max, 1e-6) if self.g_max > 0 else 1e-6
        # Ensure variance isn't negative before sqrt
        denominator = max(np.sqrt(max(0.0, current_G_var) + self.reward_scale_epsilon), safe_G_max_term)
        # Avoid division by zero in reward_scale calculation
        reward_scale = 1.0 / max(denominator, 1e-8)
        scaled_reward = raw_reward * reward_scale # This is a float

        # --- Standard Experience Storage ---
        if self.prev_obs is not None:
            # prev_obs should already be a tensor on the correct device from the previous step
            prev_obs_tensor = self.prev_obs
            # Format action for buffer (could be numpy or tensor)
            action_for_buffer = self._format_action_for_buffer(action)

            # Add transition using processed tensors
            # Assumption: self.memory.add handles the types passed correctly
            try:
                # Double-check shapes just before adding to buffer as a safeguard
                if prev_obs_tensor.shape != expected_shape:
                     print(f"[WARNING] Shape mismatch for prev_obs before add: {prev_obs_tensor.shape} != {expected_shape}. Attempting reshape.")
                     prev_obs_tensor = prev_obs_tensor.reshape(expected_shape)
                # Note: next_obs_tensor was already reshaped and checked above

                self.memory.add(prev_obs_tensor, action_for_buffer, scaled_reward, next_obs_tensor, done)

                if self.debug and random.random() < 0.01: # Log occasionally
                    print(f"[DEBUG SAC] Added transition: {prev_obs_tensor.shape} -> {next_obs_tensor.shape}")
            except Exception as e: # Catch potential errors during buffer add
                if self.debug:
                    print(f"[ERROR] Error adding to buffer: {e}")
                    # Safely print shapes if they exist
                    prev_shape = getattr(prev_obs_tensor, 'shape', 'N/A')
                    next_shape = getattr(next_obs_tensor, 'shape', 'N/A')
                    print(f"[DEBUG SAC] prev_obs shape: {prev_shape}, next_obs shape: {next_shape}")
                    import traceback
                    traceback.print_exc()

                # Ensure prev_obs is updated correctly even on error before returning
                self.prev_obs = next_obs_tensor if not done else None
                return # Exit if add failed

        # Update prev_obs for the *next* call, storing the processed tensor
        self.prev_obs = next_obs_tensor if not done else None

        # Update step counts and metrics
        self.steps += 1
        self.metrics['steps'] = self.steps
        self.metrics['mean_reward_scale'] = reward_scale

        # --- Update Trigger ---
        # Logic to trigger the training update based on steps, buffer size etc.
        if (self.steps >= self.warmup_steps and
            len(self.memory) >= self.batch_size and
            self.steps % self.update_freq == 0):
            for _ in range(self.updates_per_step):
                # Call the main update method of the algorithm
                # Entropy calculation is now handled within the update cycle
                update_metrics = self.update()

        # --- End of Episode Handling ---
        if done:
            # Calculate episode return from raw rewards
            raw_episode_return = sum(self.current_episode_raw_rewards)
            self.raw_episode_returns.append(raw_episode_return)
            # Calculate mean over recent episodes
            mean_raw_return = np.mean(self.raw_episode_returns) if self.raw_episode_returns else 0.0

            # Update G variance and max stats for reward scaling
            try:
                 # Ensure tensors are created on the correct device
                 return_tensor = torch.tensor([[raw_episode_return]], dtype=torch.float64, device=self.device)
                 self.G_rms.update(return_tensor)
                 # Ensure tensor for max is on the correct device
                 self.G_max_running = torch.max(self.G_max_running, torch.tensor(raw_episode_return, device=self.device))
            except Exception as e:
                 if self.debug:
                      print(f"[ERROR] Error updating G_rms/G_max at episode end: {e}")

            # Update metrics dictionary
            # Use .item() safely with checks
            g_variance = self.G_rms.var.item() if hasattr(self.G_rms, 'var') else 0.0
            g_max_running = self.G_max_running.item() if hasattr(self.G_max_running, 'item') else 0.0

            self.metrics.update({
                'raw_episode_return': raw_episode_return,
                'mean_raw_episode_return': mean_raw_return,
                'G_variance': g_variance,
                'G_max_running': g_max_running,
                # Log the reward scale used in the *last step* of the episode
                'mean_reward_scale': reward_scale,
                'steps': self.steps,
                # Ensure train_steps exists or default to 0
                'train_steps': getattr(self, 'train_steps', 0)
            })

            if self.debug:
                 print(f"Step: {self.steps}, Episode Done. Raw Return: {raw_episode_return:.2f}, Mean Raw: {mean_raw_return:.2f}, "
                       f"G_var: {g_variance:.2f}, G_max: {g_max_running:.2f}, Last RewardScale: {reward_scale:.3f}")

            # Reset trackers for the next episode
            self.current_episode_raw_rewards = []

    def _format_action_for_buffer(self, action):
        """Ensure action format is consistent for buffer storage (e.g., one-hot for discrete)."""
        if self.action_space_type == "discrete":
            if isinstance(action, (int, np.integer)):
                action_idx = int(action)
            elif isinstance(action, torch.Tensor) and action.numel() == 1:
                 action_idx = int(action.item())
            elif isinstance(action, np.ndarray) and action.size == 1:
                 action_idx = int(action.item())
            elif isinstance(action, (np.ndarray, torch.Tensor)) and action.shape[-1] == self.action_dim:
                return action # Assume already one-hot
            else:
                raise TypeError(f"Unsupported discrete action type for buffer: {type(action)}, value: {action}")

            action_one_hot = np.zeros(self.action_dim)
            action_one_hot[action_idx] = 1.0
            return action_one_hot
        else: # Continuous
             return action # Store continuous action directly


    def _update_critics(self, obs, actions, rewards, next_obs, dones, alpha):
        """Update the distributional critic networks using KL divergence loss."""
         # --- CRITICAL: Distributional Critic Update ---
        # This requires projecting the Bellman target onto the fixed support `self.support`.
        # The loss is typically the KL divergence between the predicted distribution
        # (softmax over critic logits) and the projected target distribution.

        # Set models to train mode
        self.critic1.train()
        self.critic2.train()
        self.actor.eval() # Actor used for target policy

        with torch.no_grad():
            # Get next action distribution and log_probs from actor policy
            _, _, next_actions_scaled_or_onehot, next_log_probs = self._get_action_distribution(next_obs)

            # Get next state value distribution from target critics
            # Output shape: [batch_size, num_actions, num_atoms] for discrete if critic handles actions internally
            # Output shape: [batch_size, num_atoms] for continuous (critic takes obs, action)
            next_dist_logits1 = self.critic1_target(next_obs, next_actions_scaled_or_onehot)
            next_dist_logits2 = self.critic2_target(next_obs, next_actions_scaled_or_onehot)
            next_dist_probs1 = F.softmax(next_dist_logits1, dim=-1)
            next_dist_probs2 = F.softmax(next_dist_logits2, dim=-1)

            # Use probs from the critic with lower expected value (or simply average/min?)
            # Standard Double Q uses min(Q1, Q2). For distributional, often use dist whose E[V] is lower.
            # Let's use the minimum expected value critic's distribution.
            next_q_expected1 = torch.sum(next_dist_probs1 * self.support, dim=-1, keepdim=True)
            next_q_expected2 = torch.sum(next_dist_probs2 * self.support, dim=-1, keepdim=True)
            use_critic1_target = (next_q_expected1 <= next_q_expected2).float()
            next_dist_probs = use_critic1_target * next_dist_probs1 + (1 - use_critic1_target) * next_dist_probs2

            # Add entropy term to the expected value part of the target
            # Target atom values: Tz = r + (1-d) * gamma * z
            # Need to incorporate entropy: Tz = r + (1-d) * gamma * (z - alpha * E[log_pi(a'|s')])
            # Projecting (z - alpha * log_prob) is complex.
            # Simpler: Project (r + (1-d)*gamma*z) and calculate loss against target dist.
            # Let's project the standard Bellman target first.

            # Compute the target atoms: Tz = r + (1-d) * gamma * z
            target_support = rewards + (1.0 - dones) * self.gamma * self.support.unsqueeze(0)

            # Subtract entropy term (approximation: subtract from expected value)
            # This assumes alpha*log_prob affects the mean, not the distribution shape directly.
            target_support = target_support - alpha * (1.0 - dones) * self.gamma * next_log_probs


            # Project target support onto the fixed support atoms
            target_support = target_support.clamp(self.g_min, self.g_max)
            b = (target_support - self.g_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            # Handle cases where l==u or atoms are out of bounds
            l = l.clamp(0, self.num_atoms - 1)
            u = u.clamp(0, self.num_atoms - 1)

            # Distribute probability mass based on projection
            m = torch.zeros_like(next_dist_probs) # Shape [batch_size, num_atoms]
            # Add probability mass p(s',a') to atoms l and u
            # m_l = p(s',a') * (u - b)
            # m_u = p(s',a') * (b - l)
            # Need to iterate or use scatter_add_
            # Get p(s', a') which is next_dist_probs [batch_size, num_atoms]
            offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size, device=self.device).long().unsqueeze(1)
            offset = offset.expand(self.batch_size, self.num_atoms)

            m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist_probs * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist_probs * (b - l.float())).view(-1))
            # m is now the target distribution [batch_size, num_atoms]


        # --- Calculate Current Log Probs ---
        # Critic outputs logits: [batch_size, num_actions, num_atoms] or [batch_size, num_atoms]
        if self.action_space_type == "discrete":
            logits1_all = self.critic1(obs) # Shape [batch, num_actions, num_atoms]
            logits2_all = self.critic2(obs)
            # Gather the logits for the action taken
            action_indices = actions.argmax(dim=1, keepdim=True).unsqueeze(-1) # Shape [batch, 1, 1]
            action_indices = action_indices.expand(-1, -1, self.num_atoms) # Shape [batch, 1, num_atoms]
            logits1 = logits1_all.gather(1, action_indices).squeeze(1) # Shape [batch, num_atoms]
            logits2 = logits2_all.gather(1, action_indices).squeeze(1)
        else: # Continuous
            logits1 = self.critic1(obs, actions) # Shape [batch, num_atoms]
            logits2 = self.critic2(obs, actions)

        log_probs1 = F.log_softmax(logits1, dim=-1)
        log_probs2 = F.log_softmax(logits2, dim=-1)

        # --- Calculate KL Divergence Loss ---
        # KL(target || current) = sum(target_probs * (log(target_probs) - log(current_probs)))
        # Target probs are `m`, current log_probs are `log_probs1/2`
        # Avoid log(0) in target: add small epsilon to m before log
        kl_loss1 = (m * (torch.log(m + 1e-8) - log_probs1)).sum(dim=-1).mean()
        kl_loss2 = (m * (torch.log(m + 1e-8) - log_probs2)).sum(dim=-1).mean()

        # --- Optimize Critics ---
        self.critic1_optimizer.zero_grad()
        kl_loss1.backward(retain_graph=True) # Retain graph for critic2 and actor
        if self.max_grad_norm > 0: nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        kl_loss2.backward()
        if self.max_grad_norm > 0: nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()

        # Calculate batch entropy for logging (based on actor's next action distribution)
        with torch.no_grad():
             _, _, _, next_log_probs_for_entropy = self._get_action_distribution(next_obs)
             batch_entropy = -next_log_probs_for_entropy.mean().item()


        self.metrics['critic_kl_loss'] = (kl_loss1.item() + kl_loss2.item()) / 2.0

        # NOTE: Returning KL loss values and batch entropy (calculated based on next actions)
        # The original SAC returned MSE losses. Here we return the KL losses.
        return kl_loss1.item(), kl_loss2.item(), batch_entropy


    def _get_target_critic_values(self, next_obs, next_actions):
        """Helper to get distributional output from target critics."""
        # SimbaV2 critics output logits over atoms
        logits1 = self.critic1_target(next_obs, next_actions)
        logits2 = self.critic2_target(next_obs, next_actions)
        return logits1, logits2 # Return logits directly

    def _get_current_critic_values(self, obs, actions):
        """Helper to get distributional output from current critics."""
         # SimbaV2 critics output logits over atoms
        if self.action_space_type == "discrete":
            # Critic handles action internally, output shape [batch, num_actions, num_atoms]
            logits1_all = self.critic1(obs)
            logits2_all = self.critic2(obs)
            # Gather the logits for the action provided
            action_indices = actions.argmax(dim=1, keepdim=True).unsqueeze(-1) # Shape [batch, 1, 1]
            action_indices = action_indices.expand(-1, -1, self.num_atoms) # Shape [batch, 1, num_atoms]
            logits1 = logits1_all.gather(1, action_indices).squeeze(1) # Shape [batch, num_atoms]
            logits2 = logits2_all.gather(1, action_indices).squeeze(1)
        else: # Continuous
            logits1 = self.critic1(obs, actions) # Shape [batch, num_atoms]
            logits2 = self.critic2(obs, actions)
        return logits1, logits2 # Return logits directly


    def _calculate_and_log_entropy(self, obs_batch):
        """Calculate entropy for a batch of observations and log it."""
        if obs_batch is None or len(obs_batch) == 0:
            return

        with torch.no_grad():
            # Recompute actor output for the batch
            action_outputs_batch = self.actor(obs_batch)

            # Calculate entropy in batch
            if self.action_space_type == "discrete":
                # Ensure output is probabilities for Categorical
                action_probs = F.softmax(action_outputs_batch, dim=-1)
                # Add epsilon for stability if needed
                action_probs = torch.clamp(action_probs, min=1e-8)
                dist = Categorical(probs=action_probs)
                entropy = dist.entropy().mean()
            else:
                mean, log_std = action_outputs_batch.chunk(2, dim=-1)
                log_std = torch.clamp(log_std, min=-20, max=2)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                entropy = dist.entropy().mean() # entropy() is sum of entropy across dims

            # Update metrics
            self.metrics['mean_entropy'] = entropy.item()

            if self.debug:
                print(f"[DEBUG SimbaV2] Calculated entropy for batch size {len(obs_batch)}: {entropy.item():.4f}")

    def _update_actor_and_alpha(self, obs, current_alpha):
         """Update the actor network and alpha parameter using distributional critics."""
         self.actor.train()
         self.critic1.eval() # Keep critics in eval mode
         self.critic2.eval()

         alpha_loss_val = 0.0

         # Calculate and log entropy for this batch (recomputing actor output)
         self._calculate_and_log_entropy(obs)

         # Get actions and log_probs from the current policy
         dist, _, actions_scaled_or_onehot, log_probs = self._get_action_distribution(obs)

         # --- Calculate Expected Q-values for the sampled actions ---
         # Get output distributions (logits) from critics
         logits1_pi, logits2_pi = self._get_current_critic_values(obs, actions_scaled_or_onehot)
         probs1_pi = F.softmax(logits1_pi, dim=-1)
         probs2_pi = F.softmax(logits2_pi, dim=-1)

         # Calculate expected Q values
         q1_expected = torch.sum(probs1_pi * self.support, dim=-1, keepdim=True)
         q2_expected = torch.sum(probs2_pi * self.support, dim=-1, keepdim=True)
         q_expected = torch.min(q1_expected, q2_expected)

         # --- Calculate Actor Loss ---
         # Minimize E[alpha * log_prob - Q_expected]
         actor_loss = (current_alpha * log_probs - q_expected).mean()

         # --- Optimize Actor ---
         # NOTE: SimbaV2 applies hyperspherical projection *after* the optimizer step.
         # This needs to be handled either by a custom optimizer or manually after step().
         self.actor_optimizer.zero_grad()
         actor_loss.backward()
         if self.max_grad_norm > 0:
             nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
         self.actor_optimizer.step()

         # --- Apply SimbaV2 Weight Projection (Conceptual) ---
         # self._project_actor_weights() # Placeholder for hyperspherical projection

         # --- Update Alpha ---
         if self.auto_alpha_tuning:
             alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
             self.alpha_optimizer.zero_grad()
             alpha_loss.backward()
             self.alpha_optimizer.step()
             alpha_loss_val = alpha_loss.item()
             self.alpha = self.log_alpha.exp().item()
         else:
              alpha_loss_val = 0.0

         return actor_loss.item(), alpha_loss_val

    def _project_actor_weights(self):
         """Placeholder: Apply hyperspherical projection to actor weights after optimizer step."""
         # This logic needs to be implemented based on how weights are structured
         # in the SimbaV2 actor network (e.g., iterating through Linear layers).
         # Example for a single weight matrix W:
         # with torch.no_grad():
         #     W.data = l2_normalize(W.data, dim=0) # Normalize along input dim? Or output dim? Check paper/code.
         # print("[Warning] _project_actor_weights not implemented.")
         pass

    # Override save/load if distributional critic or reward scaling stats need saving
    def save(self, path):
        """Save the SimbaV2 algorithm's state."""
        base_state = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_alpha_tuning else None,
            'steps': self.steps,
            'train_steps': self.train_steps,
             # SimbaV2 specific state
            'G_rms_mean': self.G_rms.mean,
            'G_rms_var': self.G_rms.var,
            'G_rms_count': self.G_rms.count,
            'G_max_running': self.G_max_running,
            # Buffer state (optional, consider size)
            # 'buffer_ptr': self.memory.ptr,
            # 'buffer_size': self.memory.size,
        }
        torch.save(base_state, path)
        if self.debug: print(f"SimbaV2 model saved to {path}")


    def load(self, path):
        """Load the SimbaV2 algorithm's state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

        if self.auto_alpha_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp().item()

        self.steps = checkpoint['steps']
        self.train_steps = checkpoint['train_steps']

        # Load SimbaV2 specific state
        self.G_rms.mean = checkpoint['G_rms_mean'].to(self.device)
        self.G_rms.var = checkpoint['G_rms_var'].to(self.device)
        self.G_rms.count = checkpoint['G_rms_count']
        self.G_max_running = checkpoint['G_max_running'].to(self.device)


        # Buffer state (optional)
        # self.memory.ptr = checkpoint.get('buffer_ptr', 0)
        # self.memory.size = checkpoint.get('buffer_size', 0)

        # Ensure models are in eval mode
        self.actor.to(self.device).eval()
        self.critic1.to(self.device).eval()
        self.critic2.to(self.device).eval()
        self.critic1_target.to(self.device).eval()
        self.critic2_target.to(self.device).eval()

        if self.debug: print(f"SimbaV2 model loaded from {path}")
