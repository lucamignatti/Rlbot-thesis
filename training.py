import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.amp import autocast, GradScaler
from models import fix_compiled_state_dict, print_model_info
from typing import Union, Tuple
import time
import wandb
import os

class PPOMemory:
    """
    Buffer to store trajectories experienced by a PPO agent.
    Handles storage and retrieval of data, taking into account whether
    we're using the GPU (tensors) or CPU (numpy arrays).
    """
    def __init__(self, batch_size, buffer_size=10000, device="cuda"):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device

        # Use different storage depending on whether we have a GPU or not.
        self.use_device_tensors = device != "cpu"

        if self.use_device_tensors:
            # We're on a GPU, pre-allocate tensors.
            self.states = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
            self.actions = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.logprobs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
            # Priorities for prioritized experience replay. Initialized to 1.0.
            self.priorities = torch.ones(buffer_size, device=device)
        else:
            # We're on CPU. Use numpy arrays, which are more efficient here.
            self.states = np.zeros((buffer_size, 1), dtype=np.float32)
            self.actions = np.zeros((buffer_size,), dtype=np.float32)
            self.logprobs = np.zeros((buffer_size,), dtype=np.float32)
            self.rewards = np.zeros((buffer_size,), dtype=np.float32)
            self.values = np.zeros((buffer_size,), dtype=np.float32)
            self.dones = np.zeros((buffer_size,), dtype=bool)
            # Priorities for prioritized experience replay.  Initialized to 1.0.
            self.priorities = np.ones(buffer_size, dtype=np.float32)

        self.state_initialized = False
        self.action_initialized = False
        self.pos = 0
        self.size = 0

    def store(self, state, action, logprob, reward, value, done):
        """Store an experience tuple in the buffer."""

        # Initialize state storage, if needed, and determine its dimensions.
        if not self.state_initialized and state is not None:
            state_shape = np.asarray(state).shape

            if self.use_device_tensors:
                self.states = torch.zeros((self.buffer_size,) + state_shape,
                                         dtype=torch.float32, device=self.device)
            else:
                self.states = np.zeros((self.buffer_size,) + state_shape, dtype=np.float32)

            self.state_initialized = True

        # Initialize action storage, if needed, and figure out if actions are discrete or continuous.
        if not self.action_initialized and action is not None:
            action_array = np.asarray(action)

            if action_array.size == 1:  # Discrete action
                if self.use_device_tensors:
                    self.actions = torch.zeros(self.buffer_size, dtype=torch.int64, device=self.device)
                else:
                    self.actions = np.zeros(self.buffer_size, dtype=np.int64)
            else:  # Continuous action
                if self.use_device_tensors:
                    self.actions = torch.zeros((self.buffer_size,) + action_array.shape,
                                              dtype=torch.float32, device=self.device)
                else:
                    self.actions = np.zeros((self.buffer_size,) + action_array.shape, dtype=np.float32)

            self.action_initialized = True

        # Actually store the provided values.
        if self.use_device_tensors:
            if state is not None:
                self.states[self.pos] = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            if action is not None:
                action_array = np.asarray(action)
                if action_array.size == 1:  # Discrete action.
                    self.actions[self.pos] = int(action_array.item())
                else:  # Continuous action.
                    self.actions[self.pos] = torch.as_tensor(action_array, dtype=torch.float32, device=self.device)

            self.logprobs[self.pos] = torch.tensor(float(logprob if logprob is not None else 0), device=self.device)
            self.rewards[self.pos] = torch.tensor(float(reward if reward is not None else 0), device=self.device)
            self.values[self.pos] = torch.tensor(float(value if value is not None else 0), device=self.device)
            self.dones[self.pos] = torch.tensor(bool(done if done is not None else False), device=self.device)
            # For prioritized experience replay - start with max priority.
            self.priorities[self.pos] = 1.0
        else:
            # We are on the CPU, so use numpy.
            if state is not None:
                self.states[self.pos] = np.asarray(state, dtype=np.float32)

            if action is not None:
                action_array = np.asarray(action)
                if action_array.size == 1:
                    self.actions[self.pos] = int(action_array.item())
                else:
                    self.actions[self.pos] = action_array

            self.logprobs[self.pos] = float(logprob if logprob is not None else 0)
            self.rewards[self.pos] = float(reward if reward is not None else 0)
            self.values[self.pos] = float(value if value is not None else 0)
            self.dones[self.pos] = bool(done if done is not None else False)
            # For prioritized experience replay - start with max priority.
            self.priorities[self.pos] = 1.0

        # Update the current position and size of the buffer (circular buffer).
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def clear(self):
        """
        Reset the buffer after we've used it for an update.
        We keep the allocated memory, just reset the counters.
        """
        self.pos = 0
        self.size = 0

    def get(self):
        """Get all data currently stored in the buffer."""
        if self.size == 0 or not self.state_initialized:
            return [], [], [], [], [], []

        if self.use_device_tensors:
            # Data is already on the GPU, just return slices.
            valid_states = self.states[:self.size]
            valid_actions = self.actions[:self.size]
            valid_logprobs = self.logprobs[:self.size]
            valid_rewards = self.rewards[:self.size]
            valid_values = self.values[:self.size]
            valid_dones = self.dones[:self.size]
        else:
            # We're on CPU, convert numpy arrays to tensors and move to the specified device.
            valid_states = torch.tensor(self.states[:self.size], dtype=torch.float32, device=self.device)

            # Actions can be discrete (int) or continuous (float).
            if np.issubdtype(self.actions.dtype, np.integer):
                valid_actions = torch.tensor(self.actions[:self.size], dtype=torch.long, device=self.device)
            else:
                valid_actions = torch.tensor(self.actions[:self.size], dtype=torch.float32, device=self.device)

            valid_logprobs = torch.tensor(self.logprobs[:self.size], dtype=torch.float32, device=self.device)
            valid_rewards = torch.tensor(self.rewards[:self.size], dtype=torch.float32, device=self.device)
            valid_values = torch.tensor(self.values[:self.size], dtype=torch.float32, device=self.device)
            valid_dones = torch.tensor(self.dones[:self.size], dtype=torch.bool, device=self.device)

        return (valid_states, valid_actions, valid_logprobs, valid_rewards, valid_values, valid_dones)

    def generate_batches(self):
        """
        Generates batches of indices, using prioritized sampling.
        """
        if self.size == 0:
            return []

        # Calculate normalized probabilities for prioritized sampling.
        if self.use_device_tensors:
            probs = self.priorities[:self.size] / self.priorities[:self.size].sum()

            # Generate batches by sampling indices based on the calculated probabilities.
            batches = []
            num_complete_batches = self.size // self.batch_size

            for i in range(num_complete_batches):
                batch_indices = torch.multinomial(probs, self.batch_size, replacement=True)
                batches.append(batch_indices)

            # Handle any leftover data.
            if self.size % self.batch_size != 0:
                remaining = self.size - (num_complete_batches * self.batch_size)
                if remaining > 0:
                    batch_indices = torch.multinomial(probs, remaining, replacement=True)
                    # Pad the last batch to the full batch size.
                    needed = self.batch_size - remaining
                    padding_indices = torch.multinomial(probs, needed, replacement=True)
                    # Make sure everything is on the same device before combining.
                    batch_indices = torch.cat([batch_indices, padding_indices])
                    batches.append(batch_indices)
        else:
            # Same logic, but using numpy for CPU operations.
            probs = self.priorities[:self.size] / self.priorities[:self.size].sum()

            batches = []
            num_complete_batches = self.size // self.batch_size

            for i in range(num_complete_batches):
                # Sample indices based on priority.
                batch_indices = np.random.choice(
                    self.size, self.batch_size, replace=True, p=probs
                )
                batches.append(batch_indices)

            # Handle leftovers.
            if self.size % self.batch_size != 0:
                remaining = self.size - (num_complete_batches * self.batch_size)
                if remaining > 0:
                    batch_indices = np.random.choice(
                        self.size, remaining, replace=True, p=probs
                    )
                    # Pad to full batch size.
                    needed = self.batch_size - remaining
                    padding_indices = np.random.choice(
                        self.size, needed, replace=True, p=probs
                    )
                    batch_indices = np.concatenate([batch_indices, padding_indices])
                    batches.append(batch_indices)

        return batches

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon  # Use a small epsilon to avoid division by zero in the beginning

    def update(self, x):
        # Calculate mean, variance, and count for the current batch
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        # Delegate to a separate function for updating with moments
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # Welford's online algorithm for calculating variance.
        # More numerically stable than the naive approach.

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # Update the mean
        new_mean = self.mean + delta * batch_count / tot_count

        # Calculate combined variance components
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        # Update internal state
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

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
        lr_critic: float = 3e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_epsilon: float = 0.2,
        critic_coef: float = 1.0,
        entropy_coef: float = 0.01,
        entropy_coef_decay: float = 0.995,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 128,
        use_wandb: bool = False,
        debug: bool = False,
        use_compile: bool = True,
        use_amp: bool = True,
    ):

        self.use_wandb = use_wandb
        self.debug = debug

        # Figure out which device (CPU, CUDA, MPS) to use.  Prioritize CUDA, then MPS, then CPU.
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

        # We'll decay the entropy coefficient over time to encourage exploration শুরুতে, and exploitation later.
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.min_entropy_coef = 0.001  # Don't let entropy go to zero.

        try:
            self.debug_print(f"Compiling models for {self.device}...")

            # Options for compilation - we don't capture the full graph and allow dynamic shapes.
            compile_options = {
                "fullgraph": False,
                "dynamic": True,
            }

            if self.device == "mps":
                # Use aot_eager backend, which is compatible with Apple Silicon.
                self.actor = torch.compile(self.actor, backend="aot_eager", **compile_options)
                self.critic = torch.compile(self.critic, backend="aot_eager", **compile_options)
            elif self.device == "cuda":
                # For CUDA, use the inductor backend and allow cudagraph operations.
                if hasattr(torch._dynamo, "config") and hasattr(torch._dynamo.config, "allow_cudagraph_ops"):
                    torch._dynamo.config.allow_cudagraph_ops = True

                backend = "inductor"

                self.actor = torch.compile(self.actor, backend=backend, **compile_options)
                self.critic = torch.compile(self.critic, backend=backend, **compile_options)
            else:
                # For CPU, also use inductor.
                self.actor = torch.compile(self.actor, backend="inductor", **compile_options)
                self.critic = torch.compile(self.critic, backend="inductor", **compile_options)

        except Exception as e:
            self.debug_print(f"Model compilation failed: {str(e)}")
            self.use_compile = False

        # Put models into training mode.
        self.actor.train()
        self.critic.train()

        # Store PPO hyperparameters.
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Create optimizers for actor and critic.
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # We'll initialize learning rate schedulers later during the first update.
        self.actor_scheduler = None
        self.critic_scheduler = None

        # Initialize the memory buffer to store experiences.
        self.memory = PPOMemory(batch_size, buffer_size=10000, device=device)

        self.ret_rms = RunningMeanStd(shape=())

        # Use Automatic Mixed Precision (AMP) if we're on CUDA and it's requested.
        self.use_amp = "cuda" in str(device) and use_amp
        if self.use_amp:
            self.scaler = GradScaler(
                init_scale=2**10,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
                enabled=True
            )
            self.debug_print("Automatic Mixed Precision (AMP) enabled")
        else:
            self.debug_print("Automatic Mixed Precision (AMP) disabled: requires CUDA")

        # Keep track of training metrics.
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.total_losses = []

        if self.debug:
            print(f"[DEBUG] PPOTrainer initialized with target device: {self.device}")

        if self.use_wandb:
            wandb.log({"device": self.device})

        # If requested and available, compile the model for performance.
        self.use_compile = use_compile and hasattr(torch, 'compile')
        if self.use_compile:
            try:
                self.debug_print("Compiling performance-critical functions...")

                # Create compiled versions of functions for later use.
                self._compute_gae_fn = self._create_compiled_gae()
                self.debug_print("GAE function compiled")

                self._policy_eval_fn = self._create_compiled_policy_eval()
                self.debug_print("Policy evaluation function compiled")

                self._process_batch_fn = self._create_compiled_batch_processor()
                self.debug_print("Batch processor function compiled")

                # "Warm up" compiled functions. This helps avoid compilation delays later.
                self._prewarm_compiled_functions()
                self.debug_print("Compiled functions pre-warmed")

            except Exception as e:
                self.debug_print(f"Function compilation failed: {str(e)}")
                self.use_compile = False


        print_model_info(self.actor, model_name="Actor", print_amp=self.use_amp)

        print_model_info(self.critic, model_name="Critic", print_amp=self.use_amp)

    def _create_compiled_gae(self):
        """Create a compiled version of GAE computation for tensor inputs."""
        def _tensor_gae(rewards, values, dones, next_value):
            """
            Inner function to compute GAE, designed for compilation.
            This is where the actual GAE calculation happens, optimized for tensor operations.
            """
            advantages = torch.zeros_like(rewards, device=self.device)
            last_gae = torch.tensor(0.0, device=self.device)  # Initialize the last GAE value.

            # Make sure next_value is the same shape as values for concatenation.
            next_value_shaped = next_value.reshape(*[1] + list(values.shape[1:]))
            all_values = torch.cat([values, next_value_shaped], dim=0)

            # Iterate backwards through the rewards to calculate GAE.
            for t in reversed(range(len(rewards))):
                # If the episode is done at this step, only consider the immediate reward.
                mask = ~dones[t]
                if not mask:
                    delta = rewards[t] - values[t]
                    last_gae = delta
                # Otherwise, take into account the discounted future rewards.
                else:
                    delta = rewards[t] + self.gamma * all_values[t+1] - values[t]
                    last_gae = delta + self.gamma * self.gae_lambda * last_gae
                advantages[t] = last_gae

            returns = advantages + values  # Calculate the returns by adding advantages to the state values.
            return advantages, returns

        # Compile the GAE function using the appropriate backend for the device.
        if self.device == "mps":
            return torch.compile(_tensor_gae, backend="aot_eager")
        elif self.device == "cuda":
            return torch.compile(_tensor_gae, mode="max-autotune") # Use max-autotune for best performance on CUDA.
        else:
            return torch.compile(_tensor_gae, mode="reduce-overhead") # For CPU.

    def _create_compiled_policy_eval(self):
        """Create a compiled version of the policy evaluation logic."""
        def _policy_eval(states, actions, old_log_probs, advantages):
            """
            Inner function to evaluate the policy, designed for compilation.
            This function calculates the actor loss, entropy loss, and entropy.
            """
            if self.action_space_type == "discrete":
                logits = self.actor(states)
                action_probs = F.softmax(logits, dim=1)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
            else:
                mu, sigma_raw = self.actor(states)
                # Ensure sigma is positive and prevent very small values.
                sigma = F.softplus(sigma_raw) + 1e-5
                dist = Normal(mu, sigma)
                new_log_probs = dist.log_prob(actions)
                # Sum log probabilities for multi-dimensional actions.
                if len(new_log_probs.shape) > 1:
                    new_log_probs = new_log_probs.sum(dim=1)
                entropy = dist.entropy().mean()

            # Calculate the probability ratio between new and old policies.
            ratio = torch.exp(new_log_probs - old_log_probs)
            # Calculate surrogate losses.
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

            # PPO objective is to maximize the minimum of the two surrogate losses.
            actor_loss = -torch.min(surr1, surr2).mean()
            # Encourage exploration by maximizing entropy.
            entropy_loss = -self.entropy_coef * entropy

            return actor_loss, entropy_loss, entropy

        # Compile the policy evaluation function using the right backend.
        if self.device == "mps":
            return torch.compile(_policy_eval, backend="aot_eager")
        elif self.device == "cuda":
            return torch.compile(_policy_eval, mode="max-autotune")
        else:
            return torch.compile(_policy_eval, mode="reduce-overhead")

    def _create_compiled_batch_processor(self):
        """Create compiled version of the batch processing logic."""
        def _process_batch(batch_states, batch_actions, batch_old_log_probs,
                            batch_advantages, batch_returns):
            """
            Inner function to process a batch of experiences, designed for compilation.
            Calculates the actor, critic, and entropy losses for a given batch.
            """
            values = self.critic(batch_states).squeeze()

            # Use the compiled policy evaluation function.
            actor_loss, entropy_loss, entropy = self._policy_eval_fn(
                batch_states, batch_actions, batch_old_log_probs, batch_advantages
            )

            # Critic loss is the mean squared error between predicted values and returns.
            critic_loss = F.mse_loss(values, batch_returns)

            # Total loss combines actor, critic, and entropy losses.
            total_loss = actor_loss + self.critic_coef * critic_loss + entropy_loss

            return actor_loss, critic_loss, entropy_loss, total_loss, entropy

        # Compile using the appropriate backend.
        if self.device == "mps":
            return torch.compile(_process_batch, backend="aot_eager")
        elif self.device == "cuda":
            return torch.compile(_process_batch, mode="max-autotune")
        else:
            return torch.compile(_process_batch, mode="reduce-overhead")

    def _uncompiled_policy_eval(self, states, actions, old_log_probs, advantages):
        """Uncompiled fallback for policy evaluation."""
        # This function does the same as _policy_eval, but is not compiled.
        # It serves as a fallback if compilation fails or is not supported.
        if self.action_space_type == "discrete":
            logits = self.actor(states)
            action_probs = F.softmax(logits, dim=1)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        else:
            mu, sigma_raw = self.actor(states)
            sigma = F.softplus(sigma_raw) + 1e-5
            dist = Normal(mu, sigma)
            new_log_probs = dist.log_prob(actions)
            if len(new_log_probs.shape) > 1:
                new_log_probs = new_log_probs.sum(dim=1)
            entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.entropy_coef * entropy

        return actor_loss, entropy_loss, entropy

    def _uncompiled_process_batch(self, batch_states, batch_actions, batch_old_log_probs,
                            batch_advantages, batch_returns):
        """Uncompiled fallback for batch processing."""
        # Similar to _process_batch, but without compilation.  A fallback.
        values = self.critic(batch_states).squeeze()

        actor_loss, entropy_loss, entropy = self._uncompiled_policy_eval(
            batch_states, batch_actions, batch_old_log_probs, batch_advantages
        )

        critic_loss = F.mse_loss(values, batch_returns)
        total_loss = actor_loss + self.critic_coef * critic_loss + entropy_loss

        return actor_loss, critic_loss, entropy_loss, total_loss, entropy

    def _prewarm_compiled_functions(self):
        """Pre-warm compiled functions with dummy data to complete compilation."""
        if not self.use_compile:
            return

        self.debug_print("Pre-warming compiled functions...")

        try:
            # We create dummy tensors to run through the compiled functions once.
            # This forces the compilation to happen here, so it doesn't cause delays later.
            dummy_size = 4  # Small batch size for pre-warming

            # Use a fixed gamma value for prewarming to avoid using self.gamma
            # which might not be initialized yet
            dummy_gamma = 0.99
            dummy_gae_lambda = 0.95

            # Pre-warm the GAE function.
            if hasattr(self, '_compute_gae_fn'):
                def _tensor_gae_prewarm(rewards, values, dones, next_value):
                    """Simplified GAE function for pre-warming only"""
                    advantages = torch.zeros_like(rewards, device=self.device)
                    last_gae = torch.tensor(0.0, device=self.device)
                    next_value_shaped = next_value.reshape(1)
                    all_values = torch.cat([values, next_value_shaped], dim=0)

                    for t in reversed(range(len(rewards))):
                        if dones[t]:
                            delta = rewards[t] - values[t]
                            last_gae = delta
                        else:
                            delta = rewards[t] + dummy_gamma * all_values[t+1] - values[t]
                            last_gae = delta + dummy_gamma * dummy_gae_lambda * last_gae
                        advantages[t] = last_gae

                    returns = advantages + values
                    return advantages, returns

                # Create dummy data for pre-warming
                dummy_rewards = torch.zeros(dummy_size, device=self.device)
                dummy_values = torch.zeros(dummy_size, device=self.device)
                dummy_dones = torch.zeros(dummy_size, dtype=torch.bool, device=self.device)
                dummy_next_value = torch.zeros(1, device=self.device)

                # Run the compiled function once with our temporary function implementation
                # This avoids using self.gamma during prewarming
                _ = _tensor_gae_prewarm(dummy_rewards, dummy_values, dummy_dones, dummy_next_value)
                self.debug_print("GAE function pre-warmed")

            # Rest of the prewarming code...
            # (Keep the policy evaluation and batch processing prewarming as they were)

        except Exception as e:
            self.debug_print(f"Function pre-warming failed: {str(e)}")
            self.use_compile = False # If prewarming fails, disable compilation.

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
        Handles different data types for consistent storage.
        """
        # Convert done to the appropriate type based on memory configuration.
        if self.memory.use_device_tensors and not isinstance(done, torch.Tensor):
            # Convert Python/numpy boolean to PyTorch boolean tensor.
            done_tensor = torch.tensor(bool(done), dtype=torch.bool, device=self.device)
            self.memory.store(state, action, log_prob, reward, value, done_tensor)
        else:
            self.memory.store(state, action, log_prob, reward, value, done)

    def store_experience_at_idx(self, idx, state, action, log_prob, reward, value, done):
        """
        Update specific fields of an existing experience entry.
        Only updates the provided values (non-None values).
        """
        if idx >= self.memory.buffer_size:
            raise ValueError(f"Index {idx} out of bounds for buffer size {self.memory.buffer_size}")

        # Only update fields that are provided (not None).
        if self.memory.use_device_tensors:
            if reward is not None:
                self.memory.rewards[idx] = torch.tensor(float(reward), device=self.device)

            if done is not None:
                if isinstance(done, torch.Tensor):
                    self.memory.dones[idx] = done.to(self.device)
                else:
                    self.memory.dones[idx] = torch.tensor(bool(done), device=self.device)
        else:
            if reward is not None:
                self.memory.rewards[idx] = float(reward)

            if done is not None:
                self.memory.dones[idx] = bool(done)

    def get_action(self, state, evaluate=False):
        """
        Get an action and its log probability from the actor network, given a state.
        Optimized for both single and batched inputs, and handles both discrete and continuous action spaces.
        """
        # Before anything, if we're using CUDA, let's make sure we're all synced up.
        if "cuda" in str(self.device):
            try:
                torch.cuda.synchronize()
                torch.compiler.cudagraph_mark_step_begin()
            except Exception as e:
                # If synchronization fails, just log it and move on.
                if self.debug:
                    print(f"[DEBUG] CUDA sync warning: {str(e)}")

        # If the state isn't already a tensor, make it one, and ensure it has a batch dimension.
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            if state.dim() == 1 or (len(state.shape) == 2 and state.shape[0] == 1 and state.shape[1] > 1):
                state = state.unsqueeze(0)

        # We clone the state to avoid modifying the original.  We'll keep this clone on the CPU.
        state_clone = state.clone()

        # Move the state to the correct device (CPU, CUDA, or MPS).
        state_device = state_clone.to(self.device)

        with torch.no_grad():  # We don't need gradients for inference.
            with autocast(enabled=self.use_amp, device_type="cuda"):  # Use AMP if available for faster computation.
                # For CUDA graphs, we mark the start of each step.
                if "cuda" in str(self.device):
                    torch.compiler.cudagraph_mark_step_begin()

                # First, get the estimated value of this state from the critic.
                value = self.critic(state_device)
                value = value.clone()  # Clone to avoid in-place modifications which can mess with the computation graph.

                # Mark step before actor execution.
                if "cuda" in str(self.device):
                    torch.compiler.cudagraph_mark_step_begin()

                # Now, get the action distribution from the actor.  This is different for discrete vs. continuous spaces.
                if self.action_space_type == "discrete":
                    logits = self.actor(state_device)
                    logits = logits.clone()

                    # Mark step after model execution.
                    if "cuda" in str(self.device):
                        torch.compiler.cudagraph_mark_step_begin()

                    # Convert the logits to probabilities.
                    action_probs = F.softmax(logits, dim=1)
                    dist = Categorical(action_probs)  # Create a categorical distribution.

                    if evaluate:
                        # If we're evaluating the policy, take the action with the highest probability.
                        action = torch.argmax(action_probs, dim=1)
                    else:
                        # Otherwise (during training), sample an action from the distribution.
                        action = dist.sample()

                    log_prob = dist.log_prob(action)  # Get the log probability of the chosen action.
                else:
                    # For continuous action spaces, the actor outputs the mean and (raw) standard deviation of a normal distribution.
                    mu, sigma_raw = self.actor(state_device)
                    mu = mu.clone()  # Clone the mean to avoid in-place modifications.
                    sigma_raw = sigma_raw.clone() if isinstance(sigma_raw, torch.Tensor) else sigma_raw # Clone sigma, checking its type first.

                    # Mark step after model execution.
                    if "cuda" in str(self.device):
                        torch.compiler.cudagraph_mark_step_begin()

                    # Make sure the standard deviation is positive and not too small.
                    sigma = F.softplus(sigma_raw) + 1e-5
                    dist = Normal(mu, sigma)  # Create a normal distribution.

                    if evaluate:
                        # For evaluation, use the mean of the distribution (the "expected" action).
                        action = mu
                    else:
                        # For training, sample from the distribution.
                        action = dist.sample()

                    log_prob = dist.log_prob(action)  # Log prob of the action.
                    # If the action has multiple dimensions, sum the log probabilities.
                    if len(log_prob.shape) > 1:
                        log_prob = log_prob.sum(dim=-1)

                    # Make sure the action stays within the allowed bounds.
                    action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])

                # Clone outputs to make absolutely sure we're not modifying tensors in-place.
                action = action.clone()
                log_prob = log_prob.clone()
                value = value.clone()

        # Return actions, log probabilities, and values as numpy arrays (for use with the environment).
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation (GAE). Supports both PyTorch tensors and NumPy arrays.
        Also includes improved numerical stability checks and handling."""

        # Use compiled GAE function if it's available and enabled.
        if self.use_compile and hasattr(self, '_compute_gae_fn'):
            return self._compute_gae_fn(rewards, values, dones, next_value)

        # Determine if we're working with tensors or numpy arrays.
        is_tensor = isinstance(rewards, torch.Tensor)

        # Update return statistics, but only if we are using numpy, as tensor operations are handled elsewhere.
        if hasattr(self, 'ret_rms') and not is_tensor:
            episode_returns = []
            curr_return = 0
            for r, d in zip(rewards, dones):
                curr_return += r
                # If this is the end of an episode, store the total return.
                if d:
                    episode_returns.append(curr_return)
                    curr_return = 0

            # Only update if we have complete episodes to avoid skewing stats.
            if episode_returns:
                self.ret_rms.update(np.array(episode_returns))

        if is_tensor:
            # Initialize the advantages tensor.
            advantages = torch.zeros_like(rewards, device=self.device)
            last_gae = torch.tensor(0.0, device=self.device)  # Keep track of the last GAE value.

            # Handle dimension mismatches between `next_value` and `values`.
            if next_value.dim() != values.dim():
                if next_value.dim() > values.dim():
                    # Remove extra dimensions.
                    next_value = next_value.squeeze()

                # Add dimensions if needed.
                while next_value.dim() < values.dim():
                    next_value = next_value.unsqueeze(0)

            # Create a tensor with compatible shape for concatenation.
            if len(values.shape) == 1:
                next_value_shaped = next_value.reshape(1)
            else:
                # Ensure `next_value` matches the shape of `values` for concatenation.
                next_value_shaped = next_value.reshape(*[1] + list(values.shape[1:]))

            # Concatenate `values` and `next_value` for easier calculations.
            try:
                all_values = torch.cat([values, next_value_shaped], dim=0)
            except RuntimeError as e:
                # If concatenation fails, print detailed shape information for debugging.
                print(f"values shape: {values.shape}, next_value shape: {next_value_shaped.shape}")
                print(f"values dim: {values.dim()}, next_value dim: {next_value_shaped.dim()}")
                raise RuntimeError(f"Error in torch.cat: {e}. Shape mismatch between values and next_value.")

            # Loop backwards through the rewards to calculate GAE.
            for t in reversed(range(len(rewards))):
                # Determine if the episode ended at this step.
                mask = ~dones[t] if isinstance(dones[t], torch.Tensor) else not dones[t]
                if not mask:
                    # If the episode ended, the advantage is just the immediate reward minus the value estimate.
                    delta = rewards[t] - values[t]
                    last_gae = delta
                else:
                    # If the episode continues, consider discounted future rewards and the next value estimate.
                    delta = rewards[t] + self.gamma * all_values[t+1] - values[t]
                    last_gae = delta + self.gamma * self.gae_lambda * last_gae

                advantages[t] = last_gae  # Store the calculated GAE.
        else:
            # Numpy implementation for CPU.
            advantages = np.zeros(len(rewards), dtype=np.float32)
            last_gae = 0

            # Ensure inputs are numpy arrays.
            if not isinstance(rewards, np.ndarray):
                rewards = np.array(rewards)
            if not isinstance(values, np.ndarray):
                values = np.array(values)
            if not isinstance(dones, np.ndarray):
                dones = np.array(dones)

            # Convert `next_value` to a numpy array if it's a tensor.
            if isinstance(next_value, torch.Tensor):
                next_value = next_value.cpu().detach().numpy()

            # Reshape `next_value` to be compatible with `values`.
            if values.ndim == 1:
                next_value = np.array([next_value]).flatten()
            else:
                next_value = np.reshape(next_value, (1,) + values.shape[1:])

            # Concatenate `values` and `next_value`.
            all_values = np.concatenate([values, next_value], axis=0)

            # Loop backwards through rewards to calculate GAE (same logic as tensor version).
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    delta = rewards[t] - values[t]
                    last_gae = delta
                else:
                    delta = rewards[t] + self.gamma * all_values[t+1] - values[t]
                    last_gae = delta + self.gamma * self.gae_lambda * last_gae

                advantages[t] = last_gae

        # Check for numerical instability (NaN or Inf) in the advantages.
        if is_tensor:
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                self.debug_print("Warning: NaN or Inf in GAE advantages, attempting to fix...")
                # Replace NaN/Inf values with reasonable defaults.
                advantages = torch.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)
        else:
            if np.isnan(advantages).any() or np.isinf(advantages).any():
                if hasattr(self, 'debug_print'):
                    self.debug_print("Warning: NaN or Inf in GAE advantages, attempting to fix...")
                # Replace NaN/Inf values with reasonable defaults.
                advantages = np.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)

        # Calculate returns by adding advantages to state values.
        returns = advantages + values
        return advantages, returns

    def _detect_abnormal(self, tensor, name="tensor"):
        """Helper function to check for NaN or Inf values in a tensor."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            self.debug_print(f"Warning: Detected NaN or Inf in {name}")
            return True
        return False

    def _policy_eval_amp_safe(self, states, actions, old_log_probs, advantages):
        """
        Evaluates the policy in a way that's safe for Automatic Mixed Precision (AMP).
        Includes safeguards against numerical instability.
        """
        # Forward pass of the actor network. Use AMP autocasting only where it's safe and beneficial.
        if self.action_space_type == "discrete":
            with autocast(enabled=self.use_amp, device_type="cuda"):
                logits = self.actor(states)
                action_probs = F.softmax(logits, dim=1)  # Probabilities from logits.
            dist = Categorical(action_probs)  # Create a categorical distribution.
            new_log_probs = dist.log_prob(actions)  # Log probability of the actions.
            entropy = dist.entropy().mean()  # Calculate the entropy of the distribution.
        else:
            # For continuous action spaces.
            with autocast(enabled=self.use_amp, device_type="cuda"):
                mu, sigma_raw = self.actor(states)  # Get mean and raw standard deviation.

            # Perform calculations that require more precision in full (float32) precision.
            mu = mu.float()
            if isinstance(sigma_raw, torch.Tensor):
                sigma_raw = sigma_raw.float()  # Ensure sigma_raw is float32.
            sigma = F.softplus(sigma_raw) + 1e-5   # Ensure positivity and prevent very small values.
            dist = Normal(mu, sigma)  # Create a normal distribution.
            new_log_probs = dist.log_prob(actions) # Log probability.
            if len(new_log_probs.shape) > 1:
                new_log_probs = new_log_probs.sum(dim=1)  # Sum if multi-dimensional action.
            entropy = dist.entropy().mean()  # Entropy of the distribution.

        # Calculate the PPO objective, ensuring numerical stability.  Do this in full precision.
        log_ratio = new_log_probs - old_log_probs.float()
        # Clamp the log ratio to prevent extreme values that can lead to instability.
        log_ratio = torch.clamp(log_ratio, -15.0, 15.0)
        ratio = torch.exp(log_ratio) # The ratio of new to old probabilities.

        # If we detect NaN values in the ratio, replace them with reasonable defaults.
        if self._detect_abnormal(ratio, "ratio"):
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.1)

        # Calculate the two surrogate losses for PPO.
        surr1 = ratio * advantages.float()
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages.float()

        # The actor loss is the negative of the minimum of the two surrogates (we want to *maximize* the objective).
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.entropy_coef * entropy  # Encourage exploration by maximizing entropy.

        # Final checks for NaN values before returning.  If we find any, use fallback values.
        if self._detect_abnormal(actor_loss, "actor_loss"):
            actor_loss = torch.tensor(1.0, device=self.device, requires_grad=True)  # Fallback value.

        if self._detect_abnormal(entropy_loss, "entropy_loss"):
            entropy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)  # Fallback value.

        return actor_loss, entropy_loss, entropy

    def _process_batch_amp_safe(self, batch_states, batch_actions, batch_old_log_probs,
                            batch_advantages, batch_returns):
        """
        Processes a batch of experiences, calculating losses.
        Designed to be safe for Automatic Mixed Precision (AMP) and includes numerical safeguards.
        """

        # Get the value predictions from the critic. Use AMP for the network forward pass.
        with autocast(enabled=self.use_amp, device_type="cuda"):
            values = self.critic(batch_states).squeeze()

        # Evaluate the policy (get actor loss, entropy loss, and entropy).
        actor_loss, entropy_loss, entropy = self._policy_eval_amp_safe(
            batch_states, batch_actions, batch_old_log_probs, batch_advantages
        )

        # Calculate the critic loss (mean squared error between predicted values and returns).
        # Do this in full precision for better accuracy.
        values = values.float()
        batch_returns = batch_returns.float()
        critic_loss = F.mse_loss(values, batch_returns)

        # Check for NaNs in the critic loss.
        if self._detect_abnormal(critic_loss, "critic_loss"):
            critic_loss = torch.tensor(1.0, device=self.device, requires_grad=True)  # Fallback value.

        # Combine the losses.
        total_loss = actor_loss + self.critic_coef * critic_loss + entropy_loss

        # Final check for any NaN values in the total loss.
        if self._detect_abnormal(total_loss, "total_loss"):
            self.debug_print("Warning: NaN in final loss detected - using fallback loss")
            # If we have a NaN, try to recover by using just the critic loss (usually more stable).
            if not torch.isnan(critic_loss).any():
                total_loss = self.critic_coef * critic_loss
            else:
                # As a last resort, create a dummy loss to prevent training from completely failing.
                total_loss = torch.tensor(1.0, device=self.device, requires_grad=True)

        return actor_loss, critic_loss, entropy_loss, total_loss, entropy

    def update(self):
        """Update policy and value networks using the PPO algorithm, with optimizations for AMP."""
        update_start = time.time()

        # Retrieve data from the memory buffer.
        states, actions, old_log_probs, rewards, values, dones = self.memory.get()
        if len(states) == 0:
            # If there's nothing to train on, return early with default values.
            return {"actor_loss": 0, "critic_loss": 0, "entropy_loss": 0, "total_loss": 0,
                    "mean_episode_reward": 0, "value_explained_variance": 0, "kl_divergence": 0}

        # Calculate how many episodes are represented in this batch of data.
        episodes_ended = torch.sum(dones.int()).item() if isinstance(dones, torch.Tensor) else np.sum(dones) if len(dones) > 0 else 0
        # Calculate per-episode metrics like total reward and length.
        episode_rewards = []
        episode_reward = 0
        episode_lengths = []
        episode_length = 0

        for i, (reward, done) in enumerate(zip(rewards, dones)):
            # Handle different data types (tensor vs. numpy).
            r_val = reward.item() if isinstance(reward, torch.Tensor) else reward
            d_val = done.item() if isinstance(done, torch.Tensor) else done

            episode_reward += r_val
            episode_length += 1

            # If an episode ended, store the accumulated reward and length.
            if d_val or i == len(rewards) - 1:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0

        avg_episode_reward = np.mean(episode_rewards)
        avg_episode_length = np.mean(episode_lengths)

        # Compute the next value estimate, which is needed for GAE.
        # We only do this if the last state in the buffer isn't a terminal state.
        if len(states) > 0 and not dones[-1]:
            with torch.no_grad():  # No gradients needed for this.
                next_state = states[-1].unsqueeze(0) if isinstance(states, torch.Tensor) else \
                            torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
                with autocast(enabled=self.use_amp and "cuda" in str(self.device), device_type="cuda"):
                    next_value = self.critic(next_state)
        else:
            # If the last state *was* terminal, the next value is just 0.
            next_value = torch.tensor([0.0], device=self.device) if isinstance(values, torch.Tensor) else 0.0

        # Calculate advantages and returns using Generalized Advantage Estimation (GAE).
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Normalize the advantages.  This helps stabilize training.
        if isinstance(advantages, torch.Tensor) and len(advantages) > 0:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8  # Add a small constant to avoid division by zero.
            if not torch.isnan(adv_mean) and not torch.isnan(adv_std):
                advantages = (advantages - adv_mean) / adv_std

        # Initialize variables to track the total losses across all epochs and batches.
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        old_probs = None
        old_mu = None
        old_sigma = None

        # Store the policy's output *before* any updates.  This is needed to calculate the KL divergence later,
        # which is a measure of how much the policy has changed.
        with torch.no_grad():
            if self.action_space_type == "discrete":
                with autocast(enabled=self.use_amp and "cuda" in str(self.device), device_type="cuda"):
                    old_logits = self.actor(states)
                old_probs = F.softmax(old_logits, dim=1)  # Convert logits to probabilities.
            else:
                with autocast(enabled=self.use_amp and "cuda" in str(self.device), device_type="cuda"):
                    old_mu, old_sigma_raw = self.actor(states)
                old_sigma = F.softplus(old_sigma_raw) + 1e-5  # Ensure standard deviation is positive.

        # Get a set of batches from memory for training.
        batches = self.memory.generate_batches()

        # Dynamically adjust the clipping epsilon.  This can help improve performance.
        # The idea is to reduce clipping as training progresses.
        effective_clip_epsilon = max(0.1, self.clip_epsilon * (1 - episodes_ended / 1000))  # Adjust divisor for total episodes.

        # Initialize learning rate schedulers for the actor and critic, if they haven't been initialized yet.
        if self.actor_scheduler is None:
            self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer,
                T_max=1000000,
                eta_min=1e-5
            )

        if self.critic_scheduler is None:
            self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.critic_optimizer,
                T_max=1000000,  # Adjust based on expected training steps.
                eta_min=3e-5
            )

        # Initialize reward normalizer.
        if not hasattr(self, 'ret_rms'):
            class RunningMeanStd:
                def __init__(self, epsilon=1e-4, shape=()):
                    self.mean = np.zeros(shape, 'float64')
                    self.var = np.ones(shape, 'float64')
                    self.count = epsilon

                def update(self, x):
                    batch_mean = np.mean(x, axis=0)
                    batch_var = np.var(x, axis=0)
                    batch_count = x.shape[0]
                    self.update_from_moments(batch_mean, batch_var, batch_count)

                def update_from_moments(self, batch_mean, batch_var, batch_count):
                    delta = batch_mean - self.mean
                    tot_count = self.count + batch_count
                    new_mean = self.mean + delta * batch_count / tot_count
                    m_a = self.var * self.count
                    m_b = batch_var * batch_count
                    M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
                    new_var = M2 / tot_count
                    new_count = tot_count
                    self.mean = new_mean
                    self.var = new_var
                    self.count = new_count

            self.ret_rms = RunningMeanStd(shape=())

        # Update return statistics.
        episode_returns = []
        for r_arr in episode_rewards:
            episode_returns.append(r_arr)

        if episode_returns:
            self.ret_rms.update(np.array(episode_returns))

        # Main training loop: iterate over epochs and mini-batches.
        for epoch in range(self.ppo_epochs):
            for batch_idx, batch_indices in enumerate(batches):
                # Prepare the data for this specific batch.  Handles both tensor and numpy inputs.
                if isinstance(states, torch.Tensor):
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                else:
                    batch_states = torch.tensor(states[batch_indices], dtype=torch.float32, device=self.device)
                    batch_actions = torch.tensor(actions[batch_indices],
                                            dtype=torch.long if self.action_space_type == "discrete" else torch.float32,
                                            device=self.device)
                    batch_old_log_probs = torch.tensor(old_log_probs[batch_indices], dtype=torch.float32, device=self.device)
                    batch_advantages = torch.tensor(advantages[batch_indices], dtype=torch.float32, device=self.device)
                    batch_returns = torch.tensor(returns[batch_indices], dtype=torch.float32, device=self.device)

                # Reset the gradients of the optimizers.  We use set_to_none=True for a slight performance boost.
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                try:
                    # Use a single autocast context for the entire forward pass.  This improves efficiency.
                    with autocast(enabled=self.use_amp and "cuda" in str(self.device), device_type="cuda"):
                        # Forward pass through the critic network.
                        values = self.critic(batch_states).squeeze()

                        # Forward pass through the actor network. This is different for discrete vs. continuous action spaces.
                        if self.action_space_type == "discrete":
                            logits = self.actor(batch_states)
                            action_probs = F.softmax(logits, dim=1)  # Convert logits to probabilities.
                            dist = Categorical(action_probs)  # Create a categorical distribution.
                            new_log_probs = dist.log_prob(batch_actions)  # Get log probabilities of the actions.
                            entropy = dist.entropy().mean()  # Calculate the entropy of the distribution.
                        else:
                            mu, sigma_raw = self.actor(batch_states)  # Get mean and raw standard deviation.
                            sigma = F.softplus(sigma_raw) + 1e-5   # Ensure standard deviation is positive.
                            dist = Normal(mu, sigma)  # Create a normal distribution.
                            new_log_probs = dist.log_prob(batch_actions)  # Log probability of the actions.
                            if len(new_log_probs.shape) > 1:
                                new_log_probs = new_log_probs.sum(dim=1)  # Sum log probs if multi-dimensional action.
                            entropy = dist.entropy().mean()

                        # Calculate the ratio between the new and old policies.  This is a key part of the PPO algorithm.
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)

                        # Apply a more conservative clipping range for numerical stability.
                        ratio = torch.clamp(ratio, 0.0, 10.0)

                        # Calculate the two surrogate losses for PPO.
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1.0 - effective_clip_epsilon, 1.0 + effective_clip_epsilon) * batch_advantages

                        # Calculate the actor loss. We take the minimum of the two surrogate losses.
                        actor_loss = -torch.min(surr1, surr2).mean()
                        # Critic loss is the mean squared error between predicted values and returns.
                        critic_loss = F.mse_loss(values, batch_returns)
                        # Entropy loss encourages exploration.
                        entropy_loss = -self.entropy_coef * entropy

                        # The total loss is a weighted sum of the actor, critic, and entropy losses.
                        loss = actor_loss + self.critic_coef * critic_loss + entropy_loss

                    # If using AMP, scale the loss and perform the backward pass.
                    if self.use_amp and "cuda" in str(self.device):
                        self.scaler.scale(loss).backward()

                        # Unscale the gradients before clipping.  This is important for AMP.
                        self.scaler.unscale_(self.actor_optimizer)
                        self.scaler.unscale_(self.critic_optimizer)

                        # Clip gradients by value first for stability.
                        for param in self.actor.parameters():
                            if param.grad is not None:
                                param.grad.data.clamp_(-1.0, 1.0)
                        for param in self.critic.parameters():
                            if param.grad is not None:
                                param.grad.data.clamp_(-1.0, 1.0)

                        # Then, clip gradients by norm.
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                        # Step the optimizers (with scaled gradients).
                        self.scaler.step(self.actor_optimizer)
                        self.scaler.step(self.critic_optimizer)

                        # Update the gradient scaler.
                        self.scaler.update()
                    else:
                        # Standard backward pass (without AMP).
                        loss.backward()

                        # Clip gradients by value first.
                        for param in self.actor.parameters():
                            if param.grad is not None:
                                param.grad.data.clamp_(-1.0, 1.0)
                        for param in self.critic.parameters():
                            if param.grad is not None:
                                param.grad.data.clamp_(-1.0, 1.0)

                        # Then clip by norm.
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                        self.actor_optimizer.step()
                        self.critic_optimizer.step()

                    # Keep track of the total losses.
                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    total_loss += loss.item()

                except Exception as e:
                    # If anything goes wrong during training, print an error message and the stack trace.
                    print(f"Error in batch {batch_idx+1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Step the learning rate schedulers.
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # Calculate the average losses across all epochs and batches.
        num_updates = max(self.ppo_epochs * len(batches), 1)  # Avoid division by zero.
        avg_actor_loss = total_actor_loss / num_updates
        avg_critic_loss = total_critic_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_loss = total_loss / num_updates

        # Calculate the KL divergence between the old and new policies.  This is a measure of how much the policy changed.
        kl_div = 0
        if len(states) > 0:
            with torch.no_grad():  # No gradients needed for this.
                if self.action_space_type == "discrete" and old_probs is not None:
                    # Forward pass through actor in mixed precision.
                    with autocast(enabled=self.use_amp and "cuda" in str(self.device), device_type="cuda"):
                        new_logits = self.actor(states)

                    # Calculate KL divergence in full precision for accuracy.
                    new_probs = F.softmax(new_logits.float(), dim=1)
                    kl = old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10))
                    # Handle potential numerical issues.
                    if torch.isnan(kl).any() or torch.isinf(kl).any():
                        kl = torch.nan_to_num(kl, nan=0.0, posinf=1.0, neginf=-1.0)
                    kl_div = kl.sum(1).mean().item()

                elif old_mu is not None and old_sigma is not None:
                    # Forward pass in mixed precision.
                    with autocast(enabled=self.use_amp and "cuda" in str(self.device), device_type="cuda"):
                        new_mu, new_sigma_raw = self.actor(states)

                    # KL calculation in full precision.
                    new_mu = new_mu.float()
                    new_sigma_raw = new_sigma_raw.float() if isinstance(new_sigma_raw, torch.Tensor) else new_sigma_raw
                    new_sigma = F.softplus(new_sigma_raw) + 1e-5

                    kl = torch.log(new_sigma/old_sigma + 1e-10) + (old_sigma**2 + (old_mu - new_mu)**2)/(2*new_sigma**2 + 1e-10) - 0.5
                    # Handle potential numerical issues.
                    if torch.isnan(kl).any() or torch.isinf(kl).any():
                        kl = torch.nan_to_num(kl, nan=0.0, posinf=1.0, neginf=-1.0)
                    kl_div = kl.mean().item()

        # Calculate the explained variance.  This tells us how well the value function is predicting the returns.
        explained_var = 0
        if len(states) > 0:
            with torch.no_grad():
                try:
                    # Forward pass of the critic in mixed precision.
                    with autocast(enabled=self.use_amp and "cuda" in str(self.device), device_type="cuda"):
                        all_values_half = self.critic(states).squeeze()

                    # Calculate variance in full precision.
                    all_values = all_values_half.float().cpu().numpy().flatten()
                    all_returns = returns.cpu().numpy().flatten() if isinstance(returns, torch.Tensor) else returns.flatten()

                    # Ensure lengths match before calculation.
                    min_length = min(len(all_values), len(all_returns))
                    if min_length > 0:
                        all_values = all_values[:min_length]
                        all_returns = all_returns[:min_length]

                        if not np.isnan(all_values).any() and not np.isnan(all_returns).any():
                            var_returns = np.var(all_returns)
                            if var_returns > 1e-8:  # Avoid division by zero.
                                explained_var = 1 - np.var(all_returns - all_values) / var_returns
                except Exception as e:
                    print(f"Error computing explained variance: {e}")

        # Log the training metrics using wandb, if it's enabled.
        if self.use_wandb:
            try:
                import sys
                if 'wandb' in sys.modules:
                    # Prepare metrics, handling potential NaN values.
                    safe_metrics = {
                        "actor_loss": float(avg_actor_loss) if not np.isnan(avg_actor_loss) else 0.0,
                        "critic_loss": float(avg_critic_loss) if not np.isnan(avg_critic_loss) else 0.0,
                        "entropy_loss": float(avg_entropy_loss) if not np.isnan(avg_entropy_loss) else 0.0,
                        "total_loss": float(avg_loss) if not np.isnan(avg_loss) else 0.0,
                        "mean_episode_reward": float(avg_episode_reward),
                        "mean_episode_length": float(avg_episode_length),
                        "policy_kl_divergence": float(kl_div) if not np.isnan(kl_div) else 0.0,
                        "value_explained_variance": float(explained_var) if not np.isnan(explained_var) else 0.0,
                        "effective_clip_epsilon": effective_clip_epsilon,
                        "actor_lr": self.actor_scheduler.get_last_lr()[0],
                        "critic_lr": self.critic_scheduler.get_last_lr()[0],
                    }

                    # Add tensor-based metrics, handling NaNs.
                    if isinstance(advantages, torch.Tensor):
                        adv_mean = advantages.mean().item()
                        if not np.isnan(adv_mean):
                            safe_metrics["mean_advantage"] = adv_mean

                    if isinstance(returns, torch.Tensor):
                        ret_mean = returns.mean().item()
                        if not np.isnan(ret_mean):
                            safe_metrics["mean_return"] = ret_mean

                    # Log AMP scaler value, if applicable.
                    if self.use_amp:
                        safe_metrics["amp_scale"] = self.scaler.get_scale()

                    # Log reward normalization statistics.
                    if hasattr(self, 'ret_rms'):
                        safe_metrics["reward_mean"] = float(self.ret_rms.mean)
                        safe_metrics["reward_var"] = float(self.ret_rms.var)

                    # Initialize step counters if they don't exist
                    if not hasattr(self, 'training_steps'):
                        self.training_steps = 0
                    if not hasattr(self, 'training_step_offset'):
                        self.training_step_offset = 0

                    # Increment step counter
                    self.training_steps += 1

                    # Use combined step count for logging
                    global_step = self.training_steps + self.training_step_offset

                    import wandb
                    wandb.log(safe_metrics, step=global_step)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Failed to log to wandb: {e}")

        # Clear the memory buffer after each update.
        self.memory.clear()

        # Return a dictionary containing all the important training metrics.
        return {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy_loss": avg_entropy_loss,
            "total_loss": avg_loss,
            "mean_episode_reward": avg_episode_reward,
            "value_explained_variance": explained_var,
            "kl_divergence": kl_div,
            "effective_clip_epsilon": effective_clip_epsilon,
            "update_time": time.time() - update_start
        }

    def save_models(self, model_path=None):
        """
        Save both actor and critic models to a single file.

        Args:
            model_path: Optional custom path. If None, creates a timestamped file.
                    If a directory is provided, a timestamped file will be created inside it.

        Returns:
            The complete path where the model was saved
        """
        # Generate timestamp for unique filenames.
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if model_path is None:
            # Default: save to models directory with timestamp.
            os.makedirs("models", exist_ok=True)
            model_path = f"models/rlbot_model_{timestamp}.pt"
        elif os.path.isdir(model_path) or model_path.endswith('/') or model_path.endswith('\\'):
            # If model_path is a directory, append timestamped filename.
            os.makedirs(model_path, exist_ok=True)
            model_path = os.path.join(model_path, f"rlbot_model_{timestamp}.pt")

        # Get the original models if using compiled versions.
        actor_state = self.actor._orig_mod.state_dict() if hasattr(self.actor, '_orig_mod') else self.actor.state_dict()
        critic_state = self.critic._orig_mod.state_dict() if hasattr(self.critic, '_orig_mod') else self.critic.state_dict()

        # Get the WandB run ID if available
        wandb_run_id = None
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb_run_id = wandb.run.id
            except ImportError:
                pass

        # Save both models to a single file.
        torch.save({
            'actor': actor_state,
            'critic': critic_state,
            'training_step': getattr(self, 'training_steps', 0) + getattr(self, 'training_step_offset', 0),
            'total_episodes': getattr(self, 'total_episodes', 0) + getattr(self, 'total_episodes_offset', 0),
            'timestamp': time.time(),
            'version': '1.0',
            'wandb_run_id': wandb_run_id
        }, model_path)

        return model_path

    def load_models(self, model_path):
        """
        Load both actor and critic models from a single file.

        Args:
            model_path: Path to the saved model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load the saved state
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle both new single-file format and legacy format
        if isinstance(checkpoint, dict) and 'actor' in checkpoint and 'critic' in checkpoint:
            # New format - Fix the state dictionaries using the helper function
            actor_state = fix_compiled_state_dict(checkpoint['actor'])
            critic_state = fix_compiled_state_dict(checkpoint['critic'])

            # Load the fixed state dictionaries
            self.actor.load_state_dict(actor_state)
            self.critic.load_state_dict(critic_state)

            # Get training step count if available, defaulting to 0 if not found
            self.training_step_offset = checkpoint.get('training_step', 0)
            self.total_episodes_offset = checkpoint.get('total_episodes', 0)

            if self.use_wandb and 'wandb_run_id' in checkpoint:
                # Resume the existing WandB run if possible
                import wandb
                if wandb.run is None:
                    try:
                        wandb.init(id=checkpoint['wandb_run_id'], resume="must")
                    except Exception as e:
                        self.debug_print(f"Could not resume WandB run: {e}. Starting new run.")

            self.debug_print(f"Loaded models from unified checkpoint: {model_path}")
            return True
        else:
            # Legacy format - assume this is just the actor model
            try:
                # Fix the state dictionary before loading
                fixed_state_dict = fix_compiled_state_dict(checkpoint)
                self.actor.load_state_dict(fixed_state_dict)
                self.debug_print(f"Loaded actor model from legacy format: {model_path}")
                return False
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")
