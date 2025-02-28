import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical, Normal
from typing import Union, Tuple
import time
import wandb
from queue import Empty

class PPOMemory:
    """
    Process-safe buffer for storing trajectories experienced by a PPO agent.
    """
    def __init__(self, batch_size, buffer_size=10000):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Initialize arrays with default shapes
        self.states = np.zeros((buffer_size, 1), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.float32)
        self.state_initialized = False
        self.action_initialized = False

        # Pre-allocate scalar arrays
        self.logprobs = np.zeros((buffer_size,), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=bool)

        # Keep track of current position and size
        self.pos = 0
        self.size = 0

    def store(self, state, action, logprob, reward, value, done):
        """
        Process-safe store of one timestep of agent-environment interaction in the buffer.
        Uses pre-allocated arrays for efficient storage.
        """
        # Initialize arrays if needed
        if not self.state_initialized and state is not None:
            state_array = np.asarray(state, dtype=np.float32)
            self.states = np.zeros((self.buffer_size,) + state_array.shape, dtype=np.float32)
            self.state_initialized = True

        if not self.action_initialized and action is not None:
            action_array = np.asarray(action)
            if action_array.size == 1:  # Discrete action
                self.actions = np.zeros((self.buffer_size,), dtype=np.int64)
            else:  # Continuous action
                self.actions = np.zeros((self.buffer_size,) + action_array.shape, dtype=np.float32)
            self.action_initialized = True

        # Store at current position with None handling
        if state is not None:
            self.states[self.pos] = np.asarray(state, dtype=np.float32)
        else:
            self.states[self.pos] = np.zeros_like(self.states[0])

        if action is not None:
            action_array = np.asarray(action)
            if action_array.size == 1:  # Discrete action
                self.actions[self.pos] = int(action_array.item())
            else:  # Continuous action
                self.actions[self.pos] = action_array
        else:
            self.actions[self.pos] = np.zeros_like(self.actions[0])

        self.logprobs[self.pos] = float(logprob if logprob is not None else 0)
        self.rewards[self.pos] = float(reward if reward is not None else 0)
        self.values[self.pos] = float(value if value is not None else 0)
        self.dones[self.pos] = bool(done if done is not None else False)

        # Update position and size
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def clear(self):
        """
        Process-safe reset buffer after an update.
        Resets position and size counters without deallocating memory.
        """
        # Reset position and size without clearing arrays
        self.pos = 0
        self.size = 0

    def get(self):
        """
        Process-safe get all data from buffer.
        Returns valid data from the circular buffer.
        """
        if self.size == 0 or not self.state_initialized:
            return [], [], [], [], [], []

        # Get valid slice of data ensuring proper numpy arrays
        valid_states = np.array(self.states[:self.size], dtype=np.float32)
        valid_actions = np.array(self.actions[:self.size], dtype=np.float32)
        valid_logprobs = np.array(self.logprobs[:self.size], dtype=np.float32)
        valid_rewards = np.array(self.rewards[:self.size], dtype=np.float32)
        valid_values = np.array(self.values[:self.size], dtype=np.float32)
        valid_dones = np.array(self.dones[:self.size], dtype=bool)

        # Return only the valid data
        return (
            valid_states,
            valid_actions,
            valid_logprobs,
            valid_rewards,
            valid_values,
            valid_dones
        )

    def generate_batches(self):
        if self.size == 0:
            return []

        # Generate shuffled indices
        indices = np.arange(self.size, dtype=np.int32)
        np.random.shuffle(indices)

        # Create batches of equal size
        batches = []

        # Calculate how many complete batches we can make
        num_complete_batches = self.size // self.batch_size

        # First add complete batches
        for i in range(num_complete_batches):
            start_idx = i * self.batch_size
            batches.append(indices[start_idx:start_idx + self.batch_size])

        # Handle remaining data (if any)
        if self.size % self.batch_size != 0:
            # Get remaining indices
            remaining = indices[num_complete_batches * self.batch_size:]
            # Create a batch with the remaining indices, possibly with some repeated
            if len(remaining) > 0:
                # Repeat indices to fill the batch
                needed = self.batch_size - len(remaining)
                # Use indices from the beginning to pad
                padding = indices[:needed]
                batches.append(np.concatenate([remaining, padding]))

        return batches

class PPOTrainer:
    def __init__(
        self,
        actor,
        critic,
        action_space_type: str = "discrete",
        action_dim: Union[int, Tuple[int]] = None,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        device: str = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        use_wandb: bool = False,
        debug: bool = False,
        use_compile: bool = True
    ):

        # Logging
        self.use_wandb = use_wandb
        self.debug = debug

        # Determine device priority: CUDA -> MPS -> CPU
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.action_space_type = action_space_type
        self.action_dim = action_dim
        self.action_bounds = action_bounds

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Compile models if requested and available
        self.use_compile = use_compile and hasattr(torch, 'compile')
        if self.use_compile:
            try:
                self.debug_print("Compiling actor and critic models with torch.compile...")

                # Choose compile settings based on device
                if self.device == "mps":
                    # Apple Silicon - simplified settings without extra kwargs
                    self.actor = torch.compile(self.actor, backend="aot_eager")
                    self.critic = torch.compile(self.critic, backend="aot_eager")
                elif self.device == "cuda":
                    # CUDA with full optimization
                    self.actor = torch.compile(self.actor, mode="max-autotune")
                    self.critic = torch.compile(self.critic, mode="max-autotune")
                else:
                    # CPU with balanced settings
                    self.actor = torch.compile(self.actor, mode="reduce-overhead")
                    self.critic = torch.compile(self.critic, mode="reduce-overhead")

                self.debug_print(f"Models successfully compiled for {self.device}")
            except Exception as e:
                self.debug_print(f"Model compilation failed, falling back to standard models: {str(e)}")
                self.use_compile = False

        # Enable train mode
        self.actor.train()
        self.critic.train()


        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Memory and experience queue
        self.memory = PPOMemory(batch_size, buffer_size=10000)

        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.total_losses = []


        if self.debug:
            print(f"[DEBUG] PPOTrainer initialized with target device: {self.device}")

        if self.use_wandb:
            wandb.log({"device": self.device})

    def debug_print(self, message):
        """Print debug messages only when debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}")

    def get_device_for_inference(self):
        """Get the appropriate device for inference operations"""
        return self.device

    def store_experience(self, state, action, log_prob, reward, value, done):
        """
        Store experience directly in memory buffer.
        """
        self.memory.store(state, action, log_prob, reward, value, done)

    def get_action(self, state, evaluate=False):
        """
        Get action and log probability from state using the actor network.
        Optimized for both single and batched inputs.
        """
        # Handle different input formats
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            # Only add batch dimension if this isn't already a batch
            if state.dim() == 1 or (len(state.shape) == 2 and state.shape[0] == 1 and state.shape[1] > 1):
                state = state.unsqueeze(0)

        # Move state to target device for inference
        state_device = state.to(self.device)

        with torch.no_grad():
            # Get value estimate from critic
            value = self.critic(state_device)

            # Get action distribution from actor
            if self.action_space_type == "discrete":
                logits = self.actor(state_device)
                action_probs = F.softmax(logits, dim=1)
                dist = Categorical(action_probs)

                if evaluate:
                    action = torch.argmax(action_probs, dim=1)
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)

            else:
                mu, sigma = self.actor(state_device)
                sigma = F.softplus(sigma) + 1e-5
                dist = Normal(mu, sigma)

                if evaluate:
                    action = mu
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)

                # Sum log probs across action dimensions but preserve batch dimension
                if len(log_prob.shape) > 1:
                    log_prob = log_prob.sum(dim=-1)  # Sum over last dimension (action dimension)

                action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])

        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0

        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if not isinstance(dones, np.ndarray):
            dones = np.array(dones)

        all_values = np.append(values, next_value)

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * all_values[t+1] - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae

            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self):
        """Update policy and value networks using PPO algorithm."""
        self.debug_print("Starting policy update...")
        update_start = time.time()

        # Ensure correct devices
        self.critic.to(self.device)
        self.actor.to(self.device)


        states, actions, old_log_probs, rewards, values, dones = self.memory.get()
        self.debug_print(f"Memory retrieved: {len(states)} timesteps")

        # Calculate episode metrics before processing
        episodes_ended = np.sum(dones) if len(dones) > 0 else 0
        if episodes_ended > 0:
            # Calculate average reward per episode
            episode_rewards = []
            episode_reward = 0
            episode_lengths = []
            episode_length = 0

            for i, (reward, done) in enumerate(zip(rewards, dones)):
                episode_reward += reward
                episode_length += 1

                if done or i == len(rewards) - 1:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    episode_reward = 0
                    episode_length = 0

            avg_episode_reward = np.mean(episode_rewards)
            avg_episode_length = np.mean(episode_lengths)
        else:
            avg_episode_reward = np.mean(rewards) if len(rewards) > 0 else 0
            avg_episode_length = len(rewards)

        if len(states) > 0 and not dones[-1]:
            self.debug_print("Computing next value...")
            try:
                self.debug_print("Preparing next state...")
                with torch.no_grad():
                    next_state = torch.FloatTensor(states[-1]).to(self.device)
                    if next_state.dim() == 1:
                        next_state = next_state.unsqueeze(0)

                    self.debug_print("Running critic forward pass...")
                    next_value = self.critic(next_state)

                    self.debug_print("Moving result to CPU...")
                    next_value = next_value.cpu().numpy()[0]

                self.debug_print("Next value computation complete")
            except Exception as e:
                print(f"Error during next value computation: {str(e)}")
                raise
        else:
            next_value = 0

        self.debug_print("Converting data to numpy arrays...")
        array_start = time.time()
        states = np.asarray(states)
        actions = np.asarray(actions)
        old_log_probs = np.asarray(old_log_probs)
        rewards = np.asarray(rewards)
        values = np.asarray(values)
        dones = np.asarray(dones)
        self.debug_print(f"Array conversion took {time.time() - array_start:.4f}s")

        self.debug_print("Computing advantages...")
        compute_start = time.time()
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.debug_print(f"Advantage computation took {time.time() - compute_start:.2f}s")

        self.debug_print("Moving data to device...")
        device_start = time.time()
        try:
            self.debug_print("Converting and moving tensors...")
            # Convert and move data in smaller chunks
            states_tensor = torch.FloatTensor(states).to(self.device)
            if self.action_space_type == "discrete":
                actions_tensor = torch.LongTensor(actions).to(self.device)
            else:
                actions_tensor = torch.FloatTensor(actions).to(self.device)

            old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)

            self.debug_print(f"Data movement took {time.time() - device_start:.2f}s")
        except Exception as e:
            print(f"Error during data movement: {str(e)}")
            raise

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        total_loss = 0

        # Variables for tracking policy changes
        old_policy_kl = 0
        explained_var = 0

        # Default values for potentially unbound variables
        old_probs = None
        old_mu = None
        old_sigma = None
        batches = []
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        epoch_time = 0

        self.debug_print(f"Starting {self.ppo_epochs} PPO epochs...")
        for epoch in range(self.ppo_epochs):
            self.debug_print(f"Epoch {epoch+1}/{self.ppo_epochs}, generating batches...")
            batches = self.memory.generate_batches()
            self.debug_print(f"Epoch {epoch+1}/{self.ppo_epochs}, processing {len(batches)} batches...")
            epoch_batch_start = time.time()

            batch_times = []
            # Save old policy outputs for KL calculation
            if epoch == 0 and self.action_space_type == "discrete":
                with torch.no_grad():
                    old_logits = self.actor(states_tensor)
                    old_probs = F.softmax(old_logits, dim=1)
            elif epoch == 0:
                with torch.no_grad():
                    old_mu, old_sigma_raw = self.actor(states_tensor)
                    old_sigma = F.softplus(old_sigma_raw) + 1e-5

            for batch_idx, batch_indices in enumerate(batches):
                batch_loop_start = time.time()

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                try:
                    self.debug_print(f"Processing batch {batch_idx+1}/{len(batches)}")
                    self.debug_print(f"Batch states shape: {batch_states.shape}")
                    self.debug_print(f"Batch actions shape: {batch_actions.shape}")

                    # Critic forward pass
                    self.debug_print("Running critic forward pass...")
                    values = self.critic(batch_states).squeeze()
                    self.debug_print("Critic forward pass complete")

                    # Actor forward pass
                    self.debug_print("Running actor forward pass...")
                    if self.action_space_type == "discrete":
                        logits = self.actor(batch_states)

                        action_probs = F.softmax(logits, dim=1)
                        dist = Categorical(action_probs)
                        new_log_probs = dist.log_prob(batch_actions)
                        entropy = dist.entropy().mean()
                    else:
                        mu, sigma_raw = self.actor(batch_states)

                        sigma = F.softplus(sigma_raw) + 1e-5
                        dist = Normal(mu, sigma)
                        new_log_probs = dist.log_prob(batch_actions)
                        if len(new_log_probs.shape) > 1:
                            new_log_probs = new_log_probs.sum(dim=1)
                        entropy = dist.entropy().mean()


                    batch_time = time.time() - batch_loop_start
                    batch_times.append(batch_time)

                    if batch_idx % 10 == 0:
                        self.debug_print(f"Batch {batch_idx+1}/{len(batches)} took {batch_time:.3f}s")

                    self.debug_print("Computing policy ratio...")
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                except Exception as e:
                    # Keep error messages as regular prints
                    print(f"Error in batch {batch_idx+1}: {str(e)}")
                    raise

                self.debug_print("Computing PPO objectives...")
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -self.entropy_coef * entropy
                loss = actor_loss + self.critic_coef * critic_loss + entropy_loss

                self.debug_print("Starting backward pass...")
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                self.debug_print("Clipping gradients...")
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.debug_print("Applying optimizer steps...")
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                self.debug_print("Accumulating losses...")
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                self.debug_print("Batch complete\n")

            # End of epoch statistics
            epoch_time = time.time() - epoch_batch_start
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

            if self.debug:
                print(f"[DEBUG] Epoch {epoch+1} summary:")
                print(f"[DEBUG] Total epoch time: {epoch_time:.2f}s")
                print(f"[DEBUG] Average batch time: {avg_batch_time:.3f}s")
                print(f"[DEBUG] Min/Max batch time: {min(batch_times):.3f}s / {max(batch_times):.3f}s") if batch_times else None
                print(f"[DEBUG] Current losses: actor={actor_loss.item():.4f}, critic={critic_loss.item():.4f}, entropy={entropy_loss.item():.4f}")

        # Calculate final metrics
        num_updates = self.ppo_epochs * len(batches) if len(batches) > 0 else 1
        avg_actor_loss = total_actor_loss / num_updates
        avg_critic_loss = total_critic_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_loss = total_loss / num_updates

        # Calculate KL divergence after updates
        kl_div = 0
        if self.action_space_type == "discrete" and len(states) > 0 and old_probs is not None:
            with torch.no_grad():
                new_logits = self.actor(states_tensor)
                new_probs = F.softmax(new_logits, dim=1)
                kl_div = (old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10))).sum(1).mean().item()
        elif len(states) > 0 and old_mu is not None and old_sigma is not None:
            with torch.no_grad():
                new_mu, new_sigma_raw = self.actor(states_tensor)
                new_sigma = F.softplus(new_sigma_raw) + 1e-5
                kl_div = (torch.log(new_sigma/old_sigma) + (old_sigma**2 + (old_mu - new_mu)**2)/(2*new_sigma**2) - 0.5).mean().item()

        # Calculate explained variance safely
        if len(states) > 0:
            with torch.no_grad():
                # Get all values for all states
                all_values = self.critic(states_tensor).cpu().detach().numpy().flatten()
                all_returns = returns_tensor.cpu().detach().numpy().flatten()

                # Ensure same length
                min_length = min(len(all_values), len(all_returns))
                if min_length > 0:
                    all_values = all_values[:min_length]
                    all_returns = all_returns[:min_length]
                    explained_var = 1 - np.var(all_returns - all_values) / (np.var(all_returns) + 1e-8)
                else:
                    explained_var = 0
        else:
            explained_var = 0

        self.actor_losses.append(avg_actor_loss)
        self.critic_losses.append(avg_critic_loss)
        self.entropy_losses.append(avg_entropy_loss)
        self.total_losses.append(avg_loss)

        # Always log metrics to wandb if enabled and available
        if self.use_wandb:
            try:
                import sys
                if 'wandb' in sys.modules:
                    metrics = {
                        # Learning metrics
                        "actor_loss": avg_actor_loss,
                        "critic_loss": avg_critic_loss,
                        "entropy_loss": avg_entropy_loss,
                        "total_loss": avg_loss,

                        # Performance metrics
                        "mean_episode_reward": avg_episode_reward,
                        "mean_episode_length": avg_episode_length,
                        "mean_advantage": advantages_tensor.mean().item() if isinstance(advantages_tensor, torch.Tensor) and advantages_tensor.numel() > 0 else 0,
                        "mean_return": returns_tensor.mean().item() if isinstance(returns_tensor, torch.Tensor) and returns_tensor.numel() > 0 else 0,

                        # Training statistics
                        "policy_kl_divergence": kl_div,
                        "value_explained_variance": explained_var,

                        # Training efficiency
                        "training_epoch_time": epoch_time
                    }

                    import wandb
                    wandb.log(metrics)
                    self.debug_print("Logged metrics to wandb")
            except (ImportError, AttributeError) as e:
                if self.debug:
                    print(f"[DEBUG] Failed to log to wandb: {e}")

        # Clean up
        self.memory.clear()

        total_time = time.time() - update_start
        if self.debug:
            print(f"[DEBUG] Update completed in {total_time:.2f}s")

        # Return the most important metrics
        return {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy_loss": avg_entropy_loss,
            "total_loss": avg_loss,
            "mean_episode_reward": avg_episode_reward,
            "value_explained_variance": explained_var,
            "kl_divergence": kl_div
        }

    def save_models(self, actor_path, critic_path):
        """Save actor and critic models to disk."""
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_models(self, actor_path, critic_path):
        """Load actor and critic models from disk."""
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
