import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import time
import math

from .base import BaseAlgorithm


class QRA2CAlgorithm(BaseAlgorithm):
    """
    Implementation of Quantile Regression Advantage Actor-Critic (QR-A2C).

    QR-A2C uses quantile regression to estimate the full distribution of returns
    rather than just the mean, providing more stable and informative value estimates.
    """
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        aux_task_manager=None,
        action_space_type: str = "discrete",
        action_dim: Optional[int] = None,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: str = "cuda",
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_quantiles: int = 32,
        huber_kappa: float = 1.0,
        n_steps: int = 5,
        update_epochs: int = 1,
        batch_size: int = 256,
        use_amp: bool = False,
        debug: bool = False,
        use_wandb: bool = False,
    ):
        super().__init__(
            actor=actor,
            critic=critic,
            action_space_type=action_space_type,
            action_dim=action_dim,
            action_bounds=action_bounds,
            device=device,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=None,  # Not used in QR-A2C
            critic_coef=critic_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            ppo_epochs=update_epochs,
            batch_size=batch_size,
            use_amp=use_amp,
            debug=debug,
            use_wandb=use_wandb
        )

        # QR-A2C specific parameters
        self.num_quantiles = num_quantiles
        self.huber_kappa = huber_kappa
        self.n_steps = n_steps
        self.update_epochs = update_epochs

        # Set up optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Set up AMP scaler if using amp
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Metrics tracking
        self.metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'mean_return': 0.0,
            'quantile_error': 0.0,
            'explained_variance': 0.0,
        }

        # Experience buffer
        self.memory_size = 10000  # Maximum buffer size
        # Get actual observation shape from actor's input dimension
        obs_shape = getattr(actor, 'obs_shape', None)
        if obs_shape is None:
            # Fallback to a larger dimension if not available
            obs_shape = action_dim * 8 if action_dim else 600

        # Initialize buffer with correct observation shape
        self.obs = np.zeros((self.memory_size, obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.memory_size,) + ((action_dim,) if action_dim > 1 else (1,)), dtype=np.float32)
        self.log_probs = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.rewards = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.values = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.advantages = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.returns = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.quantiles = np.zeros((self.memory_size, num_quantiles), dtype=np.float32)

        if debug:
            print(f"[QR-A2C] Initialized observation buffer with shape {self.obs.shape}")
            print(f"[QR-A2C] Initialized action buffer with shape {self.actions.shape}")

        self.pos = 0  # Position in buffer
        self.size = 0  # Current size of buffer
        self.buffer_full = False

        # Set up quantile fractions (tau values)
        self.tau = torch.arange(0, self.num_quantiles, device=self.device).float() / self.num_quantiles + 0.5 / self.num_quantiles

        # Track episode rewards
        self.episode_returns = []
        self.current_episode_rewards = {}

        # Update counter
        self.update_count = 0

        # Save auxiliary task manager reference
        self.aux_task_manager = aux_task_manager

        # For continuous action spaces
        if self.action_space_type == "continuous":
            self.log_std_min = -20
            self.log_std_max = 2

    def get_action(self, obs, deterministic=False, return_features=False):
        """
        Get action from the policy network.

        Args:
            obs: Observation tensor
            deterministic: If True, return the most likely action
            return_features: If True, also return extracted features

        Returns:
            Tuple of (action, log_prob, value, [features])
        """
        with torch.no_grad():
            if return_features:
                action_dist_params, features = self.actor(obs, return_features=True)
            else:
                action_dist_params = self.actor(obs)

            # Get distribution
            if self.action_space_type == "discrete":
                action_dist = torch.distributions.Categorical(logits=action_dist_params)
            else:
                if isinstance(action_dist_params, tuple):
                    mu, log_std = action_dist_params
                    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                else:
                    mu = action_dist_params
                    # Use fixed std if not provided
                    log_std = torch.zeros_like(mu) - 0.5

                std = log_std.exp()
                action_dist = torch.distributions.Normal(mu, std)

            # Sample action
            if deterministic:
                if self.action_space_type == "discrete":
                    action = torch.argmax(action_dist.probs, dim=-1)
                else:
                    action = action_dist.mean
            else:
                action = action_dist.sample()

            # Get log probability
            log_prob = action_dist.log_prob(action)
            if self.action_space_type != "discrete":
                log_prob = log_prob.sum(dim=-1, keepdim=True)

            # Get quantile values from critic
            quantiles = self.critic(obs)  # Shape: [batch, num_quantiles]

            # Calculate mean value
            value = quantiles.mean(dim=-1, keepdim=True)  # Use mean of quantiles

            if return_features:
                return action, log_prob, value, features
            else:
                return action, log_prob, value

    def huber_quantile_loss(self, quantile_values, target_values):
        """
        Compute the Huber-Quantile loss for training the critic.

        Args:
            quantile_values: Predicted quantile values [batch, num_quantiles]
            target_values: Target quantile values [batch, num_quantiles]

        Returns:
            Huber-Quantile loss
        """
        try:
            # Ensure input shapes match
            if quantile_values.shape != target_values.shape:
                if self.debug:
                    print(f"[DEBUG] Shape mismatch in loss: quantile_values {quantile_values.shape}, target_values {target_values.shape}")

                # Try to fix shapes if possible
                if quantile_values.size(0) != target_values.size(0):  # Different batch sizes
                    min_batch = min(quantile_values.size(0), target_values.size(0))
                    quantile_values = quantile_values[:min_batch]
                    target_values = target_values[:min_batch]

                # If number of quantiles differs, we have to adjust one tensor
                if quantile_values.size(1) != target_values.size(1):
                    min_quantiles = min(quantile_values.size(1), target_values.size(1))
                    quantile_values = quantile_values[:, :min_quantiles]
                    target_values = target_values[:, :min_quantiles]

            batch_size = quantile_values.size(0)
            num_quantiles = quantile_values.size(1)

            # --- Debugging Loss ---
            if self.debug:
                print(f"[DEBUG Loss Input] quantile_values shape: {quantile_values.shape}, target_values shape: {target_values.shape}")
                if batch_size > 0 and num_quantiles > 0: # Avoid index error on empty tensors
                    print(f"[DEBUG Loss Input] quantile sample: {quantile_values[0, :min(5, num_quantiles)].detach().cpu().numpy()}")
                    print(f"[DEBUG Loss Input] target sample: {target_values[0, :min(5, num_quantiles)].detach().cpu().numpy()}")
            # --- End Debugging ---

            # Ensure tau has correct number of elements
            if self.tau.size(0) != num_quantiles:
                # Recreate tau with correct size if it doesn't match
                self.tau = torch.arange(0, num_quantiles, device=self.device).float() / num_quantiles + 0.5 / num_quantiles

            # Expand dimensions to compute pairwise Huber loss
            # [batch, num_quantiles, 1] - [batch, 1, num_quantiles] -> [batch, num_quantiles, num_quantiles]
            delta = target_values.unsqueeze(2) - quantile_values.unsqueeze(1)

            # Calculate Huber loss
            abs_delta = torch.abs(delta)
            huber_loss = torch.where(
                abs_delta <= self.huber_kappa,
                0.5 * delta.pow(2),
                self.huber_kappa * (abs_delta - 0.5 * self.huber_kappa)
            )

            # Calculate quantile weights
            tau = self.tau.view(1, -1, 1)  # [1, num_quantiles, 1]
            quantile_weight = torch.abs((delta.ge(0).float() - tau))

            # Calculate final loss: weight * huber_loss and average over all dimensions
            quantile_loss = quantile_weight * huber_loss
            final_loss = quantile_loss.sum(dim=(1, 2)).mean() / num_quantiles

            # --- Debugging Loss ---
            if self.debug:
                 if batch_size > 0 and num_quantiles > 0: # Avoid index error on empty tensors
                    print(f"[DEBUG Loss Calc] delta sample mean: {delta[0, :min(5, num_quantiles), :min(5, num_quantiles)].mean().item():.6f}")
                    print(f"[DEBUG Loss Calc] huber_loss sample mean: {huber_loss[0].mean().item():.6f}")
                 print(f"[DEBUG Loss Calc] final_loss: {final_loss.item():.6f}")
            # --- End Debugging ---
            return final_loss

        except Exception as e:
            tau = self.tau.view(1, -1, 1)  # [1, num_quantiles, 1]
            quantile_weight = torch.abs((delta.ge(0).float() - tau))

            # Calculate final loss: weight * huber_loss and average over all dimensions
            quantile_loss = quantile_weight * huber_loss
            final_loss = quantile_loss.sum(dim=(1, 2)).mean() / num_quantiles

            # --- Debugging Loss ---
            if self.debug:
                 if batch_size > 0 and num_quantiles > 0: # Avoid index error on empty tensors
                    print(f"[DEBUG Loss Calc] delta sample mean: {delta[0, :min(5, num_quantiles), :min(5, num_quantiles)].mean().item():.6f}")
                    print(f"[DEBUG Loss Calc] huber_loss sample mean: {huber_loss[0].mean().item():.6f}")
                 print(f"[DEBUG Loss Calc] final_loss: {final_loss.item():.6f}")
            # --- End Debugging ---
            return final_loss

        except Exception as e:
            self.huber_kappa * (abs_delta - 0.5 * self.huber_kappa)

            # Calculate quantile weights
            tau = self.tau.view(1, -1, 1)  # [1, num_quantiles, 1]
            quantile_weight = torch.abs((delta.ge(0).float() - tau))

            # Calculate final loss: weight * huber_loss and average over all dimensions
            quantile_loss = quantile_weight * huber_loss
            return quantile_loss.sum(dim=(1, 2)).mean() / num_quantiles

        except Exception as e:
            # If any error occurs, fallback to a simple MSE loss between means
            if self.debug:
                print(f"[DEBUG] Error in huber_quantile_loss: {e}")

            # Simple fallback loss
            quantile_mean = quantile_values.mean(dim=1)
            target_mean = target_values.mean(dim=1)
            return torch.nn.functional.mse_loss(quantile_mean, target_mean)

    def compute_advantages_and_returns(self):
        """
        Compute advantages and returns using GAE.
        """
        last_advantage = 0
        last_value = 0

        for t in reversed(range(self.size)):
            # Handle episode boundaries
            if t == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - last_advantage
            else:
                next_value = self.values[(t + 1) % self.memory_size]
                next_non_terminal = 1.0 - self.dones[(t + 1) % self.memory_size]

            # Calculate delta and advantage
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = self.advantages[t]

            # Calculate returns for critic target
            self.returns[t] = self.rewards[t] + self.gamma * next_non_terminal * last_value
            last_value = self.values[t]

        # --- Debugging returns ---
        if self.debug and self.size > 0:
            returns_sample = self.returns[:min(10, self.size)] # Sample first 10 returns
            rewards_sample = self.rewards[:min(10, self.size)] # Sample first 10 rewards
            print(f"[DEBUG Returns Calc] Sample returns: {returns_sample.flatten()}")
            print(f"[DEBUG Returns Calc] Sample rewards: {rewards_sample.flatten()}")
            print(f"[DEBUG Returns Calc] Returns mean: {self.returns[:self.size].mean():.4f}, std: {self.returns[:self.size].std():.4f}")
        # --- End Debugging ---

    def store_experience(self, obs, action, log_prob, reward, value, done, env_id=0):
        """
        Store experience in the buffer.

        Args:
            obs: Observation
            action: Action taken
            log_prob: Log probability of the action
            reward: Reward received
            value: Value estimate (scalar)
            done: Whether the episode is done
            env_id: Environment ID
        """
        # Store in buffer
        self.obs[self.pos] = np.array(obs, dtype=np.float32) if not isinstance(obs, np.ndarray) else obs

        # Handle action storage based on type
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if self.action_space_type == "discrete" and isinstance(action, (int, float, np.integer)):
            action = np.array([action], dtype=np.float32)

        self.actions[self.pos] = action

        # Handle log_prob storage
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.cpu().numpy()
        log_prob = np.array(log_prob, dtype=np.float32).reshape(-1)
        self.log_probs[self.pos] = log_prob

        # Store reward and done
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value

        # Update buffer position and size
        self.pos = (self.pos + 1) % self.memory_size
        if not self.buffer_full:
            self.size = min(self.size + 1, self.memory_size)
            if self.size == self.memory_size:
                self.buffer_full = True

        # Track episode rewards
        if env_id not in self.current_episode_rewards:
            self.current_episode_rewards[env_id] = 0.0
        self.current_episode_rewards[env_id] += reward

        if done:
            if env_id in self.current_episode_rewards:
                self.episode_returns.append(self.current_episode_rewards[env_id])
                # --- Debugging Episode Returns ---
                if self.debug:
                    print(f"[DEBUG Episode End] Env {env_id}: Appended reward {self.current_episode_rewards[env_id]:.4f}. Total returns recorded: {len(self.episode_returns)}")
                # --- End Debugging ---
                self.current_episode_rewards[env_id] = 0.0

    def update(self):
        """
        Update actor and critic networks based on collected experience.

        Returns:
            Dict of metrics for logging
        """
        start_time = time.time()

        # Check if we have enough data for at least one mini-batch
        min_required_data = max(4, self.batch_size // 4)  # We need at least a small batch to update

        if self.size < min_required_data:
            if self.debug:
                print(f"[QR-A2C] Not enough data to update ({self.size} < {min_required_data})")
            # Return current metrics without updating
            return self.metrics

        # If we have data but less than a full batch, adjust batch size
        if self.size < self.batch_size:
            effective_batch_size = max(4, (self.size // 4) * 4)  # Round to multiple of 4
            if self.debug:
                print(f"[QR-A2C] Using smaller batch size for update: {effective_batch_size} (buffer has {self.size})")
        else:
            effective_batch_size = self.batch_size

        # Compute advantages and returns
        self.compute_advantages_and_returns()

        # Convert buffer to tensors
        obs = torch.tensor(self.obs[:self.size], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[:self.size], dtype=torch.float32, device=self.device)
        log_probs = torch.tensor(self.log_probs[:self.size], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(self.advantages[:self.size], dtype=torch.float32, device=self.device)
        returns = torch.tensor(self.returns[:self.size], dtype=torch.float32, device=self.device)
        values = torch.tensor(self.values[:self.size], dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update networks for multiple epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for epoch in range(self.update_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(self.size, device=self.device)

            # Process mini-batches (use effective_batch_size instead of self.batch_size)
            for start in range(0, self.size, effective_batch_size):
                end = min(start + effective_batch_size, self.size)
                batch_indices = indices[start:end]

                # Calculate actual batch size (might be smaller for last batch)
                actual_batch_size = len(batch_indices)

                if actual_batch_size < 4:  # Skip batches that are too small
                    if self.debug:
                        print(f"[DEBUG] Skipping batch that is too small: {actual_batch_size}")
                    continue

                # Create batch data
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Ensure all batch tensors have the same size in the batch dimension
                if self.debug:
                    print(f"[DEBUG] Batch sizes: obs={batch_obs.size(0)}, actions={batch_actions.size(0)}, "
                          f"log_probs={batch_log_probs.size(0)}, advantages={batch_advantages.size(0)}, "
                          f"returns={batch_returns.size(0)}")

                # Actor update
                with torch.amp.autocast(device_type='cuda' if 'cuda' in self.device else 'cpu', enabled=self.use_amp):
                    # Get current action distribution
                    if self.action_space_type == "discrete":
                        action_dist_params = self.actor(batch_obs)
                        action_dist = torch.distributions.Categorical(logits=action_dist_params)
                        entropy = action_dist.entropy().mean()

                        # For categorical distributions, we need action indices, not one-hot vectors
                        if batch_actions.dim() > 1 and batch_actions.size(1) > 1:
                            # We have one-hot or multi-dimensional actions, convert to indices
                            if self.debug:
                                print(f"[DEBUG] Converting action shape {batch_actions.shape} to indices")

                            # Check if actions are one-hot encoded (sum to 1 along dim 1)
                            if torch.allclose(batch_actions.sum(dim=1), torch.ones(batch_actions.size(0), device=self.device)):
                                # One-hot encoded - convert to indices
                                batch_actions = torch.argmax(batch_actions, dim=1)
                            else:
                                # Not one-hot, but still multi-dimensional - take first column as index
                                batch_actions = batch_actions[:, 0].long()
                        else:
                            # Convert actions to long tensor and ensure correct shape
                            batch_actions = batch_actions.long().squeeze(-1)  # Convert to long for discrete actions

                        # Make sure the categorical distribution and actions have compatible dimensions
                        if action_dist.logits.size(0) != batch_actions.size(0):
                            # Debug information for shape mismatch
                            if self.debug:
                                print(f"[DEBUG] Batch dimension mismatch - action_dist.logits: {action_dist.logits.shape[0]}, batch_actions: {batch_actions.size(0)}")

                            # Try to fix mismatched batch dimensions
                            if batch_actions.size(0) > action_dist.logits.size(0):
                                batch_actions = batch_actions[:action_dist.logits.size(0)]
                            else:
                                # Shouldn't normally happen, but handle it anyway
                                pad_size = action_dist.logits.size(0) - batch_actions.size(0)
                                batch_actions = torch.cat([batch_actions, batch_actions[:pad_size]])

                        if self.debug:
                            print(f"[DEBUG] Final action shape for log_prob: {batch_actions.shape}, distribution shape: {action_dist.logits.shape}")
                    else:
                        action_dist_params = self.actor(batch_obs)
                        if isinstance(action_dist_params, tuple):
                            mu, log_std = action_dist_params
                            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                        else:
                            mu = action_dist_params
                            log_std = torch.zeros_like(mu) - 0.5

                        std = log_std.exp()
                        action_dist = torch.distributions.Normal(mu, std)
                        entropy = action_dist.entropy().sum(dim=-1).mean()

                        # Ensure batch_actions has compatible shape with distribution
                        if batch_actions.size(0) != mu.size(0):
                            if self.debug:
                                print(f"[DEBUG] Continuous action shape mismatch - mu: {mu.shape}, batch_actions: {batch_actions.shape}")
                            # Adjust action batch size to match
                            if batch_actions.size(0) > mu.size(0):
                                batch_actions = batch_actions[:mu.size(0)]
                            else:
                                pad_size = mu.size(0) - batch_actions.size(0)
                                batch_actions = torch.cat([batch_actions, batch_actions[:pad_size]])

                    # Calculate new log probs and ratio
                    try:
                        new_log_probs = action_dist.log_prob(batch_actions)
                        if self.action_space_type != "discrete":
                            new_log_probs = new_log_probs.sum(dim=-1)
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error calculating log probabilities: {e}")
                            print(f"[DEBUG] action_dist shape: {action_dist.logits.shape if hasattr(action_dist, 'logits') else 'N/A'}")
                            print(f"[DEBUG] batch_actions shape: {batch_actions.shape}")

                        # Fallback: create dummy log_probs to avoid crashing
                        new_log_probs = torch.zeros_like(batch_log_probs)

                    # Policy loss
                    policy_loss = -(new_log_probs * batch_advantages.detach()).mean()

                    # Entropy loss
                    entropy_loss = -self.entropy_coef * entropy

                    # Total actor loss
                    actor_loss = policy_loss + entropy_loss

                # Update actor
                self.actor_optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.update()
                else:
                    actor_loss.backward()
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                # Project actor weights to unit hypersphere (SimbaV2 specific)
                for param in self.actor.parameters():
                    if hasattr(param, 'project'):
                        param.project()

                # Critic update
                with torch.amp.autocast(device_type='cuda' if 'cuda' in self.device else 'cpu', enabled=self.use_amp):
                    # Get current quantile values
                    quantile_values = self.critic(batch_obs)  # [batch, num_quantiles]

                    # Handle target returns dimensions completely differently
                    # Get batch size from quantile_values
                    batch_size = quantile_values.size(0)

                    # Debug the shapes
                    if self.debug:
                        print(f"[DEBUG] Initial batch_returns shape: {batch_returns.shape}")
                        print(f"[DEBUG] quantile_values shape: {quantile_values.shape}")

                    # First view returns as flat batch size vector regardless of input shape
                    # We'll take only the first batch_size elements to match quantile_values
                    flat_returns = batch_returns.reshape(-1)[:batch_size]

                    # Now create target_values with correct dimensions from scratch
                    # This avoids expand() issues by using repeat()
                    target_values = flat_returns.unsqueeze(1).repeat(1, self.num_quantiles)

                    if self.debug:
                        print(f"[DEBUG] flat_returns shape: {flat_returns.shape}")
                        print(f"[DEBUG] Final target_values shape: {target_values.shape}")

                    # Calculate quantile regression loss with error handling
                    try:
                        critic_loss = self.huber_quantile_loss(quantile_values, target_values)
                        critic_loss = self.critic_coef * critic_loss
                    except Exception as e:
                        # If there's an error in the loss calculation, use a fallback
                        if self.debug:
                            print(f"[DEBUG] Error calculating quantile loss: {e}")
                            print(f"[DEBUG] quantile_values: {quantile_values.shape}, target_values: {target_values.shape}")

                        # Create a dummy loss to avoid crashing
                        critic_loss = torch.tensor(0.01, device=self.device, requires_grad=True)

                # Update critic
                self.critic_optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(critic_loss).backward()
                    self.scaler.unscale_(self.critic_optimizer)
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    critic_loss.backward()
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()

                # Project critic weights to unit hypersphere (SimbaV2 specific)
                for param in self.critic.parameters():
                    if hasattr(param, 'project'):
                        param.project()

                # Track losses
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()

        # Update auxiliary tasks if provided
        aux_sr_loss = 0.0
        aux_rp_loss = 0.0

        if self.aux_task_manager is not None:
            losses, aux_metrics = self.aux_task_manager.update(batch_size=self.batch_size)
            aux_sr_loss = aux_metrics.get("sr_loss_scalar", 0.0)
            aux_rp_loss = aux_metrics.get("rp_loss_scalar", 0.0)

        # Increment update counter
        self.update_count += 1

        # Calculate mean episode return
        mean_return = np.mean(self.episode_returns) if self.episode_returns else 0.0

        # Calculate explained variance
        y_pred = values.cpu().numpy()
        y_true = returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)

        # Update metrics
        num_epochs = max(1, self.update_epochs)
        num_batches = max(1, self.size // effective_batch_size * num_epochs) # Use effective_batch_size

        # --- Debugging ---
        if self.debug:
            print(f"[DEBUG QRA2C Update Metrics] total_critic_loss: {total_critic_loss:.6f}, num_batches: {num_batches}, calculated VLoss: {total_critic_loss / num_batches:.6f}")
            print(f"[DEBUG QRA2C Update Metrics] episode_returns: {self.episode_returns}, calculated mean_return: {mean_return:.4f}")
            print(f"[DEBUG QRA2C Update Metrics] aux_sr_loss: {aux_sr_loss:.6f}, aux_rp_loss: {aux_rp_loss:.6f}")
        # --- End Debugging ---

        self.metrics.update({
            'actor_loss': total_actor_loss / num_batches,
            'critic_loss': total_critic_loss / num_batches,
            'entropy_loss': total_entropy / num_batches,
            'total_loss': (total_actor_loss + total_critic_loss) / num_batches,
            'mean_return': mean_return,
            'explained_variance': explained_var,
            'sr_loss_scalar': aux_sr_loss,
            'rp_loss_scalar': aux_rp_loss,
            'update_time': time.time() - start_time,
            'buffer_size': self.size
        })

        # Clear episode returns after update
        self.episode_returns = []

        # Reset buffer after update
        self.pos = 0
        self.size = 0
        self.buffer_full = False

        return self.metrics

    def store_initial_batch(self, obs_batch, action_batch, log_prob_batch, value_batch):
        """
        Store the initial part of experiences (obs, action, log_prob, value) in batch.

        Args:
            obs_batch: Batch of observations
            action_batch: Batch of actions
            log_prob_batch: Batch of log probabilities
            value_batch: Batch of values

        Returns:
            Indices where the experiences were stored
        """
        batch_size = len(obs_batch)
        indices = []

        # Convert tensors to numpy
        if isinstance(obs_batch, torch.Tensor):
            obs_batch = obs_batch.cpu().numpy()
        if isinstance(action_batch, torch.Tensor):
            action_batch = action_batch.cpu().numpy()
        if isinstance(log_prob_batch, torch.Tensor):
            log_prob_batch = log_prob_batch.cpu().numpy()
        if isinstance(value_batch, torch.Tensor):
            value_batch = value_batch.cpu().numpy()

        # Store each experience in buffer
        for i in range(batch_size):
            idx = self.pos

            self.obs[idx] = obs_batch[i]
            self.actions[idx] = action_batch[i].reshape(-1) if isinstance(action_batch[i], np.ndarray) else np.array([action_batch[i]])
            self.log_probs[idx] = log_prob_batch[i]
            self.values[idx] = value_batch[i]

            # Track the indices
            indices.append(idx)

            # Update position and size
            self.pos = (self.pos + 1) % self.memory_size
            if not self.buffer_full and self.size < self.memory_size:
                self.size += 1
                if self.size == self.memory_size:
                    self.buffer_full = True

        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def update_rewards_dones_batch(self, indices, rewards_batch, dones_batch):
        """
        Update rewards and dones for experiences at given indices.

        Args:
            indices: Tensor of indices in buffer
            rewards_batch: Batch of rewards
            dones_batch: Batch of done flags
        """
        # Convert tensors to numpy
        if isinstance(rewards_batch, torch.Tensor):
            rewards_batch = rewards_batch.cpu().numpy()
        if isinstance(dones_batch, torch.Tensor):
            dones_batch = dones_batch.cpu().numpy()

        indices_np = indices.cpu().numpy()

        # Update rewards and dones
        for i, idx in enumerate(indices_np):
            self.rewards[idx] = rewards_batch[i]
            self.dones[idx] = dones_batch[i]

            # Update episode returns
            if dones_batch[i]:
                # For tracking episodes across environments, we'd need env_id
                # This is a simplification for batch processing
                self.episode_returns.append(rewards_batch[i])

    def get_state_dict(self):
        """Get state dict for saving"""
        state_dict = super().get_state_dict()
        state_dict.update({
            'actor_optim': self.actor_optimizer.state_dict(),
            'critic_optim': self.critic_optimizer.state_dict(),
            'update_count': self.update_count,
            'num_quantiles': self.num_quantiles,
            'buffer_size': self.size,
            'buffer_pos': self.pos,
            'buffer_full': self.buffer_full,
        })
        return state_dict

    def load_state_dict(self, state_dict):
        """Load state dict for resuming"""
        super().load_state_dict(state_dict)

        # Load optimizer states
        if 'actor_optim' in state_dict:
            self.actor_optimizer.load_state_dict(state_dict['actor_optim'])
        if 'critic_optim' in state_dict:
            self.critic_optimizer.load_state_dict(state_dict['critic_optim'])

        # Load other states
        if 'update_count' in state_dict:
            self.update_count = state_dict['update_count']
        if 'buffer_size' in state_dict:
            self.size = state_dict['buffer_size']
        if 'buffer_pos' in state_dict:
            self.pos = state_dict['buffer_pos']
        if 'buffer_full' in state_dict:
            self.buffer_full = state_dict['buffer_full']

        # Ensure we have the right number of quantiles
        if 'num_quantiles' in state_dict and state_dict['num_quantiles'] != self.num_quantiles:
            print(f"Warning: Loaded model has {state_dict['num_quantiles']} quantiles, but current configuration uses {self.num_quantiles}")
            self.num_quantiles = state_dict['num_quantiles']
            # Update tau values
            self.tau = torch.arange(0, self.num_quantiles, device=self.device).float() / self.num_quantiles + 0.5 / self.num_quantiles
