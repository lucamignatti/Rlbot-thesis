import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Categorical
from typing import Dict, Tuple, List, Optional, Union, Any
from collections import deque
from .base import BaseAlgorithm

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity, observation_shape, action_shape, device="cpu"):
        """Initialize a replay buffer for SAC.

        Args:
            capacity: Maximum number of transitions to store
            observation_shape: Shape of the observation space
            action_shape: Shape of the action space
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.observations = torch.zeros((capacity, *observation_shape), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((capacity, *observation_shape), dtype=torch.float32, device=device)
        # Store actions based on their original shape (continuous or discrete index/one-hot)
        # For discrete, we might store indices or one-hot depending on consistency needs later.
        # Let's assume continuous shape for now and handle discrete in sampling/update if needed.
        self.actions = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """Add a new transition to the buffer."""
        # Convert to tensors and ensure proper shapes in one step
        def process_tensor(x, expected_dim, target_dtype=torch.float32):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=target_dtype, device=self.device)
            # Ensure tensor has batch dimension if expected_dim > 1
            if expected_dim > 1 and x.dim() < expected_dim :
                 x = x.unsqueeze(0)
            elif expected_dim == 1 and x.dim() == 0: # Handle scalars for 1D expected
                 x = x.unsqueeze(0)

            # Ensure correct dtype
            if x.dtype != target_dtype:
                x = x.to(target_dtype)
            return x

        obs = process_tensor(obs, len(self.observations.shape) -1 ) # Match observation dim
        # Handle action type conversion (store raw action, convert if needed during sample)
        # For discrete, action might be int, np.int, tensor(int), or one-hot np/tensor
        # Standardize action storage if needed, or handle during sampling.
        # Let's assume process_tensor handles basic conversion for now.
        action = process_tensor(action, len(self.actions.shape) - 1)
        reward = process_tensor(reward, len(self.rewards.shape) - 1)
        next_obs = process_tensor(next_obs, len(self.next_observations.shape) - 1)
        done = process_tensor(done, len(self.dones.shape) - 1)


        # Check shape compatibility before assignment
        if obs.shape[1:] != self.observations.shape[1:]:
             raise ValueError(f"Obs shape mismatch: {obs.shape[1:]} vs {self.observations.shape[1:]}")
        if action.shape[1:] != self.actions.shape[1:]:
             raise ValueError(f"Action shape mismatch: {action.shape[1:]} vs {self.actions.shape[1:]}")
        if reward.shape[1:] != self.rewards.shape[1:]:
            # Handle reward shape automatically if it's scalar vs [1]
            if reward.numel() == 1 and self.rewards.shape[1] == 1:
                reward = reward.view(1, 1)
            else:
                raise ValueError(f"Reward shape mismatch: {reward.shape[1:]} vs {self.rewards.shape[1:]}")
        if next_obs.shape[1:] != self.next_observations.shape[1:]:
             raise ValueError(f"Next obs shape mismatch: {next_obs.shape[1:]} vs {self.next_observations.shape[1:]}")
        if done.shape[1:] != self.dones.shape[1:]:
            if done.numel() == 1 and self.dones.shape[1] == 1:
                done = done.view(1, 1)
            else:
                raise ValueError(f"Done shape mismatch: {done.shape[1:]} vs {self.dones.shape[1:]}")


        # Store the transition
        self.observations[self.ptr] = obs.squeeze(0) # Remove batch dim if added by process_tensor
        self.actions[self.ptr] = action.squeeze(0)
        self.rewards[self.ptr] = reward.squeeze(0)
        self.next_observations[self.ptr] = next_obs.squeeze(0)
        self.dones[self.ptr] = done.squeeze(0)

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = torch.randint(0, self.size, (batch_size,), device='cpu') # Sample indices on CPU for wider compatibility

        # Retrieve data and move to target device
        obs_batch = self.observations[indices].to(self.device)
        actions_batch = self.actions[indices].to(self.device)
        rewards_batch = self.rewards[indices].to(self.device)
        next_obs_batch = self.next_observations[indices].to(self.device)
        dones_batch = self.dones[indices].to(self.device)

        return (
            obs_batch,
            actions_batch,
            rewards_batch,
            next_obs_batch,
            dones_batch
        )

    def __len__(self):
        return self.size


class SACAlgorithm(BaseAlgorithm):
    """
    Soft Actor-Critic (SAC) implementation (Corrected).

    SAC is an off-policy actor-critic algorithm that:
    1. Uses entropy regularization for exploration
    2. Employs twin Q-networks to mitigate overestimation bias
    3. Features automatic entropy tuning (optional)
    4. Leverages experience replay for sample efficiency
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        action_space_type: str = "continuous",
        action_dim: int = None, # Required for discrete, inferred or passed for continuous
        observation_shape: tuple = None, # Required if cannot be inferred
        action_bounds: Optional[Tuple[float, float]] = (-1.0, 1.0), # Only for continuous
        device: str = "cpu",
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,  # Target network update rate
        alpha: float = 0.2,  # Temperature parameter for entropy
        auto_alpha_tuning: bool = True,  # Automatically tune alpha
        target_entropy: Optional[float] = None,  # Target entropy when auto-tuning
        buffer_size: int = 1000000,  # Replay buffer size
        batch_size: int = 256,
        warmup_steps: int = 1000,  # Number of steps before starting to train
        update_freq: int = 1,  # How often to update networks
        updates_per_step: int = 1,  # Number of gradient updates per step
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
        use_wandb: bool = False,
        debug: bool = False,
        **kwargs
    ):
        # Input validation
        if action_space_type not in ["continuous", "discrete"]:
            raise ValueError("action_space_type must be 'continuous' or 'discrete'")
        if action_space_type == "discrete" and action_dim is None:
            raise ValueError("action_dim must be provided for discrete action spaces")
        if action_space_type == "continuous" and action_bounds is None:
            raise ValueError("action_bounds must be provided for continuous action spaces")

        # Set critic references before calling super().__init__
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.actor = actor.to(device) # Ensure actor is also on device

        # Initialize with a dummy critic and critic_coef for BaseAlgorithm
        # We will override the critic functionality in this class
        super().__init__(
            actor=self.actor,
            critic=self.critic1, # Base class needs one critic reference
            action_space_type=action_space_type,
            action_dim=action_dim,
            device=device,
            lr_actor=lr_actor,
            lr_critic=lr_critic, # Will be used for both critic optimizers
            gamma=gamma,
            gae_lambda=0.0,  # Not used in SAC
            clip_epsilon=0.0,  # Not used in SAC
            critic_coef=1.0,   # Not used in SAC
            entropy_coef=0.0,  # SAC uses alpha instead
            max_grad_norm=max_grad_norm,
            batch_size=batch_size,
            use_amp=use_amp,
            use_wandb=use_wandb,
            debug=debug,
            **kwargs
        )

        # SAC-specific attributes
        self.action_bounds = action_bounds
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha_tuning = auto_alpha_tuning
        self.buffer_size = buffer_size
        self.warmup_steps = warmup_steps
        self.update_freq = update_freq
        self.updates_per_step = updates_per_step

        # Determine action dimension if continuous and not provided
        if action_space_type == "continuous" and self.action_dim is None:
             # Try to infer from actor output shape or bounds
             try:
                 # Assuming actor outputs mean and log_std concatenated
                 dummy_obs = torch.zeros(1, *self._get_observation_shape(), device=self.device)
                 actor_output_dim = self.actor(dummy_obs).shape[-1]
                 if actor_output_dim % 2 != 0:
                     raise ValueError("Cannot infer continuous action_dim: actor output size is odd.")
                 self.action_dim = actor_output_dim // 2
                 if self.debug: print(f"[DEBUG] Inferred continuous action_dim: {self.action_dim}")
             except Exception as e:
                 raise ValueError(f"Could not infer continuous action_dim. Provide it explicitly. Error: {e}")

        # Create target networks
        self.critic1_target = self._create_target_network(self.critic1)
        self.critic2_target = self._create_target_network(self.critic2)

        # Set target entropy if not provided
        if target_entropy is None:
            if self.action_space_type == "discrete":
                # Ensure action_dim is set for discrete
                if self.action_dim is None: raise ValueError("action_dim required for discrete target entropy")
                self.target_entropy = -0.98 * np.log(1.0 / self.action_dim)
            else:
                 # Ensure action_dim is set for continuous
                if self.action_dim is None: raise ValueError("action_dim required for continuous target entropy")
                self.target_entropy = -float(self.action_dim) # Convention: negative action dim
        else:
            self.target_entropy = target_entropy

        # Initialize log_alpha if using auto-tuning
        if self.auto_alpha_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor) # Use actor lr for alpha

        # Create replay buffer with correct shapes
        obs_shape = self._get_observation_shape(observation_shape) # Pass user-provided shape
        action_shape = self._get_action_shape()
        self.memory = ReplayBuffer(buffer_size, obs_shape, action_shape, device)

        # Override optimizers
        self._init_optimizers()

        # Step counter and training step counter
        self.steps = 0
        self.train_steps = 0 # Counter for gradient updates

        # Metrics tracking
        self.metrics.update({
            'train/actor_loss': 0.0,
            'train/critic1_loss': 0.0,
            'train/critic2_loss': 0.0,
            'train/alpha_loss': 0.0,
            'train/alpha': self.alpha if not self.auto_alpha_tuning else self.log_alpha.exp().item(),
            'rollout/mean_q_value': 0.0, # Q-value estimate during rollout
            'train/mean_target_q': 0.0, # Target Q during training
            'train/mean_current_q': 0.0, # Current Q during training
            'rollout/mean_entropy': 0.0, # Entropy during rollout
            'train/batch_entropy': 0.0, # Entropy during training batch
            'rollout/episode_return': 0.0,
            'rollout/mean_episode_return': 0.0, # Smoothed over episodes
            'env_step': 0,
            'train_step': 0
        })

        # Episode return tracking
        self.current_episode_rewards = []
        self.episode_returns = deque(maxlen=100)
        self.prev_obs = None # Store previous observation for adding to buffer

        if self.debug:
            print(f"[DEBUG] Initialized SAC algorithm: device={self.device}, action_space={self.action_space_type}, "
                  f"action_dim={self.action_dim}, obs_shape={obs_shape}, action_shape={action_shape}, "
                  f"auto_alpha={auto_alpha_tuning}, target_entropy={self.target_entropy}")

    def _create_target_network(self, network):
        """Create a target network as a deep copy of the original network."""
        # Standard deepcopy works for most PyTorch modules
        import copy
        target_net = copy.deepcopy(network).to(self.device)
        target_net.load_state_dict(network.state_dict())
        for param in target_net.parameters():
            param.requires_grad = False
        target_net.eval() # Set target to evaluation mode
        return target_net

    def _get_observation_shape(self, provided_shape=None):
        """Get the shape of the observation space."""
        if provided_shape:
            return provided_shape
        # Try to infer from the actor network if possible (less reliable)
        if hasattr(self.actor, 'input_shape'):
            return self.actor.input_shape[1:] # Assuming (batch, *shape)
        elif hasattr(self.actor, 'obs_shape'):
            return self.actor.obs_shape # Assuming already tuple
        else:
            raise ValueError("Cannot determine observation shape. Please provide `observation_shape` during init.")


    def _get_action_shape(self):
        """Get the shape of the action space for the replay buffer."""
        # NOTE: For discrete actions, SAC critics typically expect the action *index*
        # or a one-hot vector. The actor outputs logits. The replay buffer needs
        # to store the action taken. Storing the action *index* might be more
        # space-efficient, but requires conversion later if critics need one-hot.
        # Storing one-hot is simpler for critic input if critic expects that.
        # Let's assume the buffer stores the format consistent with action_dim:
        # - Continuous: (action_dim,)
        # - Discrete: (action_dim,) assuming one-hot storage for simplicity here.
        # If storing index for discrete, shape would be (1,).
        if self.action_space_type == "discrete":
            return (self.action_dim,) # Assuming one-hot stored in buffer
        else: # Continuous
            if isinstance(self.action_dim, tuple):
                return self.action_dim
            else:
                return (self.action_dim,)

    def _init_optimizers(self):
        """Initialize SAC-specific optimizers."""
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        # Combine critic parameters for a single optimizer if desired, or use two separate ones
        # Using two separate optimizers is common and clear:
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic)
        # If using AMP
        # self.actor_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        # self.critic_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        # self.alpha_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)


    def _soft_update_target_networks(self):
        """Perform soft update of target network parameters."""
        with torch.no_grad():
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _get_action_distribution(self, obs):
        """Helper to get action distribution and log_probs from observation."""
        if self.action_space_type == "discrete":
            action_logits = self.actor(obs)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(probs=action_probs)
            # Sample action index
            actions_idx = dist.sample()
             # Calculate log prob of the sampled action index
            log_probs = dist.log_prob(actions_idx).unsqueeze(-1) # Ensure shape [batch, 1]
             # Convert action index to one-hot for potential critic use / storage
            actions_one_hot = F.one_hot(actions_idx, num_classes=self.action_dim).float()
            return dist, actions_idx, actions_one_hot, log_probs
        else: # Continuous
            mean_logstd = self.actor(obs)
            mean, log_std = mean_logstd.chunk(2, dim=-1)
            # Clamp log_std for stability (common practice)
            log_std = torch.clamp(log_std, min=-20, max=2)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            # Use rsample for reparameterization trick
            actions_raw = dist.rsample()
            actions_squashed = torch.tanh(actions_raw)
            # Calculate log prob with squashing correction
            # log_prob(a|s) = log_prob_raw(a_raw|s) - sum(log(1 - tanh(a_raw)^2))
            log_probs = dist.log_prob(actions_raw).sum(axis=-1, keepdim=True)
            log_probs -= torch.log(1 - actions_squashed.pow(2) + 1e-6).sum(axis=-1, keepdim=True)

            # Scale to action bounds
            low, high = self.action_bounds
            actions_scaled = actions_squashed * (high - low) / 2.0 + (high + low) / 2.0

            return dist, actions_raw, actions_scaled, log_probs


    def get_action(self, obs, deterministic=False):
        """Get an action for a given observation."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == len(self.memory.observations.shape) - 1: # Check if batch dim is missing
             obs = obs.unsqueeze(0) # Add batch dimension

        self.actor.eval() # Set actor to evaluation mode

        with torch.no_grad():
            if self.action_space_type == "discrete":
                action_logits = self.actor(obs)
                action_probs = F.softmax(action_logits, dim=-1)
                if deterministic:
                    action_idx = torch.argmax(action_probs, dim=-1, keepdim=True)
                else:
                    dist = Categorical(probs=action_probs)
                    action_idx = dist.sample().unsqueeze(-1) # Keep dim for consistency

                # Return the action index (common practice) or one-hot
                # Returning index is usually more convenient for env.step()
                action_to_return = action_idx.squeeze(0).cpu().item() # Return scalar index
                # action_to_return = F.one_hot(action_idx.squeeze(0), num_classes=self.action_dim).float().cpu().numpy() # Return one-hot numpy

                # For rollout metrics (optional, can be expensive)
                # dist = Categorical(probs=action_probs)
                # log_prob = dist.log_prob(action_idx.squeeze(-1)).unsqueeze(-1)
                # entropy = dist.entropy().mean().item()
                # q1 = self.critic1(obs) # Assuming critic(obs) outputs all action Qs
                # q2 = self.critic2(obs)
                # q1_selected = q1.gather(1, action_idx)
                # q2_selected = q2.gather(1, action_idx)
                # value = torch.min(q1_selected, q2_selected).mean().item()
                # self.metrics['rollout/mean_q_value'] = value
                # self.metrics['rollout/mean_entropy'] = entropy

            else: # Continuous
                mean_logstd = self.actor(obs)
                mean, log_std = mean_logstd.chunk(2, dim=-1)
                if deterministic:
                    action_raw = mean
                else:
                     # Clamp log_std for stability
                    log_std = torch.clamp(log_std, min=-20, max=2)
                    std = torch.exp(log_std)
                    dist = Normal(mean, std)
                    action_raw = dist.sample() # Use sample (no grad needed)

                action_squashed = torch.tanh(action_raw)
                # Scale to action bounds
                low, high = self.action_bounds
                action_scaled = action_squashed * (high - low) / 2.0 + (high + low) / 2.0
                action_to_return = action_scaled.squeeze(0).cpu().numpy()

                # For rollout metrics (optional)
                # log_std = torch.clamp(log_std, min=-20, max=2) # Recompute std if needed
                # std = torch.exp(log_std)
                # dist = Normal(mean, std)
                # log_prob_raw = dist.log_prob(action_raw).sum(axis=-1, keepdim=True)
                # log_prob_squashed = log_prob_raw - torch.log(1 - action_squashed.pow(2) + 1e-6).sum(axis=-1, keepdim=True)
                # entropy = -log_prob_squashed.mean().item() # Entropy approximation
                # q1 = self.critic1(obs, action_scaled) # Critic needs obs and *scaled* action
                # q2 = self.critic2(obs, action_scaled)
                # value = torch.min(q1, q2).mean().item()
                # self.metrics['rollout/mean_q_value'] = value
                # self.metrics['rollout/mean_entropy'] = entropy

        return action_to_return


    # Simplified store_experience - assumes BaseAlgorithm doesn't need log_prob, value
    def store_experience(self, obs, action, reward, next_obs, done, info=None):
        """Store experience in the replay buffer."""
        # Track episode rewards
        if isinstance(reward, torch.Tensor): reward_item = reward.item()
        else: reward_item = reward
        self.current_episode_rewards.append(reward_item)

        # Add to buffer (handle prev_obs logic internally now)
        if self.prev_obs is not None:
            # Ensure action format is consistent with buffer's expected shape
            # If buffer expects one-hot for discrete, convert action index here
            if self.action_space_type == "discrete":
                if isinstance(action, (int, np.integer)):
                    action_idx = int(action)
                    action_for_buffer = np.zeros(self.action_dim)
                    action_for_buffer[action_idx] = 1.0
                elif isinstance(action, torch.Tensor) and action.numel() == 1:
                     action_idx = int(action.item())
                     action_for_buffer = np.zeros(self.action_dim)
                     action_for_buffer[action_idx] = 1.0
                elif isinstance(action, np.ndarray) and action.size == 1:
                     action_idx = int(action.item())
                     action_for_buffer = np.zeros(self.action_dim)
                     action_for_buffer[action_idx] = 1.0
                elif isinstance(action, (np.ndarray, torch.Tensor)) and action.shape[-1] == self.action_dim:
                    action_for_buffer = action # Assume already one-hot
                else:
                    raise TypeError(f"Unsupported discrete action type for buffer: {type(action)}, value: {action}")
            else: # Continuous
                 action_for_buffer = action # Store continuous action directly


            self.memory.add(self.prev_obs, action_for_buffer, reward, next_obs, done)

        # Store current observation for the *next* step's transition
        self.prev_obs = next_obs if not done else None # Reset prev_obs if done

        # Update model if conditions met
        self.steps += 1
        self.metrics['env_step'] = self.steps

        if (self.steps >= self.warmup_steps and
            len(self.memory) >= self.batch_size and
            self.steps % self.update_freq == 0):
            for _ in range(self.updates_per_step):
                update_metrics = self.update()
                # Optionally log update_metrics immediately if needed

        # If episode ended, log return and reset
        if done:
            episode_return = sum(self.current_episode_rewards)
            self.episode_returns.append(episode_return)
            mean_return = np.mean(self.episode_returns) if self.episode_returns else 0.0

            self.metrics['rollout/episode_return'] = episode_return
            self.metrics['rollout/mean_episode_return'] = mean_return

            if self.use_wandb:
                import wandb
                wandb.log({
                    'rollout/episode_return': episode_return,
                    'rollout/mean_episode_return': mean_return,
                    'env_step': self.steps,
                    'train_step': self.train_steps # Log train steps as well
                })
            if self.debug:
                print(f"Step: {self.steps}, Train Step: {self.train_steps}, Episode Return: {episode_return:.2f}, Mean Return: {mean_return:.2f}")

            self.current_episode_rewards = []
            # self.prev_obs is already reset above

    def update(self):
        """Update the SAC networks using a batch from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return {} # Return empty dict if not enough samples

        self.train_steps += 1
        self.metrics['train_step'] = self.train_steps

        # Sample batch
        obs, actions, rewards, next_obs, dones = self.memory.sample(self.batch_size)

        # --- Critic Update ---
        alpha = self.log_alpha.exp().item() if self.auto_alpha_tuning else self.alpha
        critic1_loss, critic2_loss, batch_entropy = self._update_critics(obs, actions, rewards, next_obs, dones, alpha)

        # --- Actor and Alpha Update ---
        # Actor update usually happens less frequently than critic in some implementations,
        # but standard SAC often updates both together. Assuming update together here.
        actor_loss, alpha_loss = self._update_actor_and_alpha(obs, alpha)

        # --- Target Network Update ---
        self._soft_update_target_networks()

        # Update metrics dictionary
        self.metrics.update({
            'train/actor_loss': actor_loss,
            'train/critic1_loss': critic1_loss,
            'train/critic2_loss': critic2_loss,
            'train/alpha_loss': alpha_loss,
            'train/alpha': alpha,
            'train/batch_entropy': batch_entropy, # From critic update calculation
            # 'train/mean_target_q' is implicitly logged in _update_critics if desired
            # 'train/mean_current_q' is implicitly logged in _update_critics if desired
        })

        # Log to wandb if enabled
        if self.use_wandb and self.train_steps % 100 == 0: # Log every 100 train steps
             import wandb
             # Filter metrics to only include training-related ones for this log call
             train_metrics = {k: v for k, v in self.metrics.items() if k.startswith('train/')}
             train_metrics['env_step'] = self.steps # Include env_step for context
             train_metrics['train_step'] = self.train_steps
             wandb.log(train_metrics)


        return self.metrics # Return the updated metrics

    def _update_critics(self, obs, actions, rewards, next_obs, dones, alpha):
        """Update the critic networks."""
        batch_entropy = 0.0 # Placeholder for entropy

        # Set models to train mode
        self.critic1.train()
        self.critic2.train()
        self.actor.eval() # Actor is used for target calculation, keep in eval mode

        with torch.no_grad():
            # Get next action, log_prob from current policy for next state
            # Use the helper function
            _, _, next_actions_scaled, next_log_probs = self._get_action_distribution(next_obs)
            # For discrete, next_actions_scaled is one-hot, next_log_probs is for the sampled index

            # Calculate target Q-values
            q1_target_next, q2_target_next = self._get_target_critic_values(next_obs, next_actions_scaled)
            q_target_next = torch.min(q1_target_next, q2_target_next)

            # Add entropy term
            q_target_next = q_target_next - alpha * next_log_probs

            # Calculate the final target Q
            target_q = rewards + (1.0 - dones) * self.gamma * q_target_next

            # Calculate batch entropy for logging (average entropy of next actions)
            batch_entropy = -next_log_probs.mean().item()


        # --- Calculate Current Q-Values ---
        # Pass both obs and actions from the buffer to the critics
        if self.action_space_type == "discrete":
             # Critic expects observation, outputs Q-values for all actions [batch, num_actions]
             current_q1_all = self.critic1(obs)
             current_q2_all = self.critic2(obs)
             # Gather the Q-value for the action actually taken (stored in buffer)
             # Assume `actions` from buffer is one-hot [batch, num_actions]
             action_indices = actions.argmax(dim=1, keepdim=True) # Get indices [batch, 1]
             current_q1 = current_q1_all.gather(1, action_indices)
             current_q2 = current_q2_all.gather(1, action_indices)
        else: # Continuous
             # Critic expects observation and action [batch, *obs_shape], [batch, *action_shape]
             current_q1 = self.critic1(obs, actions)
             current_q2 = self.critic2(obs, actions)

        # --- Calculate Critic Losses ---
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        total_critic_loss = critic1_loss + critic2_loss # Often optimized together or separately

        # --- Optimize Critic 1 ---
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True) # Retain graph if total loss used later or for actor
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()

        # --- Optimize Critic 2 ---
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()

        # Log Q values for debugging/monitoring
        self.metrics['train/mean_target_q'] = target_q.mean().item()
        self.metrics['train/mean_current_q'] = current_q1.mean().item() # Log one of the current Qs

        return critic1_loss.item(), critic2_loss.item(), batch_entropy


    def _get_target_critic_values(self, next_obs, next_actions):
        """Helper to get Q values from target critics."""
        if self.action_space_type == "discrete":
            # Target critics output Qs for all actions, gather the one for the sampled next_action
            # next_actions should be one-hot here if needed, or index if critic handles index
            next_action_indices = next_actions.argmax(dim=1, keepdim=True) # Assuming one-hot
            q1_target_next_all = self.critic1_target(next_obs)
            q2_target_next_all = self.critic2_target(next_obs)
            q1_target = q1_target_next_all.gather(1, next_action_indices)
            q2_target = q2_target_next_all.gather(1, next_action_indices)
        else: # Continuous
            q1_target = self.critic1_target(next_obs, next_actions)
            q2_target = self.critic2_target(next_obs, next_actions)
        return q1_target, q2_target

    def _get_current_critic_values(self, obs, actions):
        """Helper to get Q values from current critics."""
        if self.action_space_type == "discrete":
             # Critic expects observation, outputs Q-values for all actions [batch, num_actions]
             q1_all = self.critic1(obs)
             q2_all = self.critic2(obs)
             # Gather the Q-value for the action provided
             action_indices = actions.argmax(dim=1, keepdim=True) # Assuming one-hot actions
             q1 = q1_all.gather(1, action_indices)
             q2 = q2_all.gather(1, action_indices)
        else: # Continuous
            q1 = self.critic1(obs, actions)
            q2 = self.critic2(obs, actions)
        return q1, q2


    def _update_actor_and_alpha(self, obs, current_alpha):
        """Update the actor network and alpha parameter."""
        self.actor.train() # Set actor to training mode
        self.critic1.eval() # Critics are used for Q-value estimation, keep in eval
        self.critic2.eval()

        alpha_loss_val = 0.0  # Default value if not auto-tuning

        # Get actions and log_probs from the current policy for the observation batch
        dist, actions_raw_or_idx, actions_scaled_or_onehot, log_probs = self._get_action_distribution(obs)
        # actions_scaled_or_onehot is the action format needed by the critic

        # --- Calculate Q-values for the sampled actions ---
        q1_pi, q2_pi = self._get_current_critic_values(obs, actions_scaled_or_onehot)
        q_pi = torch.min(q1_pi, q2_pi)

        # --- Calculate Actor Loss ---
        # Original SAC loss: E_{s~D, a~pi}[alpha*log pi(a|s) - Q(s,a)]
        # Maximize E[Q - alpha * log_prob] => Minimize E[alpha * log_prob - Q]
        actor_loss = (current_alpha * log_probs - q_pi).mean()

        # --- Optimize Actor ---
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # --- Update Alpha (Temperature) ---
        if self.auto_alpha_tuning:
            # Detach log_probs to avoid backpropagating through actor again
            # Minimize E[-log_alpha * (log_prob + target_entropy)]
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step() # Update log_alpha

            alpha_loss_val = alpha_loss.item()
            # Update self.alpha based on the new log_alpha for the next iteration's critic update
            self.alpha = self.log_alpha.exp().item()
        else:
             alpha_loss_val = 0.0 # No loss if alpha is fixed

        return actor_loss.item(), alpha_loss_val


    def reset(self):
        """Reset episode-specific state."""
        self.current_episode_rewards = []
        self.prev_obs = None
        # Reset any other episode-specific variables if needed

    def get_metrics(self):
        """Get current metrics."""
        # Ensure latest alpha value is reflected if auto-tuning
        if self.auto_alpha_tuning:
            self.metrics['train/alpha'] = self.log_alpha.exp().item()
        return self.metrics

    # Removed _batched_critic_forward as specific logic is now in _get_target/current_critic_values

    def save(self, path):
        """Save the algorithm's state."""
        torch.save({
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
            'buffer_ptr': self.memory.ptr,
            'buffer_size': self.memory.size,
            # Note: Saving the entire buffer is usually too large.
            # Consider saving buffer state separately or reconstructing.
        }, path)
        if self.debug: print(f"SAC model saved to {path}")

    def load(self, path):
        """Load the algorithm's state."""
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

        # Note: Buffer state (ptr, size) is loaded, but content isn't.
        # Buffer needs to be filled again or loaded separately.
        self.memory.ptr = checkpoint.get('buffer_ptr', 0)
        self.memory.size = checkpoint.get('buffer_size', 0)

        # Ensure models are in correct mode after loading
        self.actor.to(self.device).eval()
        self.critic1.to(self.device).eval()
        self.critic2.to(self.device).eval()
        self.critic1_target.to(self.device).eval()
        self.critic2_target.to(self.device).eval()

        if self.debug: print(f"SAC model loaded from {path}")
