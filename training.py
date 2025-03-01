import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.amp import autocast, GradScaler
from typing import Union, Tuple
import time
import wandb

class PPOMemory:
    """
    Process-safe buffer for storing trajectories experienced by a PPO agent.
    """
    def __init__(self, batch_size, buffer_size=10000, device="cuda"):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device

        # Determine if we should keep data on device or as numpy arrays
        # For CPU, numpy arrays are often more efficient
        self.use_device_tensors = device != "cpu"

        if self.use_device_tensors:
            # Pre-allocate tensors on device (GPU/MPS)
            self.states = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
            self.actions = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.logprobs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
            self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
        else:
            # Use numpy arrays for CPU (more efficient)
            self.states = np.zeros((buffer_size, 1), dtype=np.float32)
            self.actions = np.zeros((buffer_size,), dtype=np.float32)
            self.logprobs = np.zeros((buffer_size,), dtype=np.float32)
            self.rewards = np.zeros((buffer_size,), dtype=np.float32)
            self.values = np.zeros((buffer_size,), dtype=np.float32)
            self.dones = np.zeros((buffer_size,), dtype=bool)

        self.state_initialized = False
        self.action_initialized = False
        self.pos = 0
        self.size = 0

    def store(self, state, action, logprob, reward, value, done):
        """Store experience appropriately based on device strategy"""
        # Initialize arrays/tensors if needed
        if not self.state_initialized and state is not None:
            state_shape = np.asarray(state).shape

            if self.use_device_tensors:
                self.states = torch.zeros((self.buffer_size,) + state_shape,
                                         dtype=torch.float32, device=self.device)
            else:
                self.states = np.zeros((self.buffer_size,) + state_shape, dtype=np.float32)

            self.state_initialized = True

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

        # Store values appropriately
        if self.use_device_tensors:
            if state is not None:
                self.states[self.pos] = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            if action is not None:
                action_array = np.asarray(action)
                if action_array.size == 1:  # Discrete action
                    self.actions[self.pos] = int(action_array.item())
                else:  # Continuous action
                    self.actions[self.pos] = torch.as_tensor(action_array, dtype=torch.float32, device=self.device)

            self.logprobs[self.pos] = torch.tensor(float(logprob if logprob is not None else 0), device=self.device)
            self.rewards[self.pos] = torch.tensor(float(reward if reward is not None else 0), device=self.device)
            self.values[self.pos] = torch.tensor(float(value if value is not None else 0), device=self.device)
            self.dones[self.pos] = torch.tensor(bool(done if done is not None else False), device=self.device)
        else:
            # Traditional numpy storage for CPU
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
        """Get data in appropriate format for the device"""
        if self.size == 0 or not self.state_initialized:
            return [], [], [], [], [], []

        if self.use_device_tensors:
            # For GPU/MPS, data is already on device
            valid_states = self.states[:self.size]
            valid_actions = self.actions[:self.size]
            valid_logprobs = self.logprobs[:self.size]
            valid_rewards = self.rewards[:self.size]
            valid_values = self.values[:self.size]
            valid_dones = self.dones[:self.size]
        else:
            # For CPU, convert numpy arrays to tensors and move to device
            valid_states = torch.tensor(self.states[:self.size], dtype=torch.float32, device=self.device)

            # Handle different action types
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
        if self.size == 0:
            return []

        # Generate shuffled indices
        if self.use_device_tensors:
            indices = torch.randperm(self.size, device=self.device)
        else:
            indices = np.random.permutation(self.size)

        # Create batches
        batches = []
        num_complete_batches = self.size // self.batch_size

        # Handle batch creation based on device strategy
        if self.use_device_tensors:
            # GPU/MPS tensor-based batches
            for i in range(num_complete_batches):
                start_idx = i * self.batch_size
                batches.append(indices[start_idx:start_idx + self.batch_size])

            # Handle remaining data
            if self.size % self.batch_size != 0:
                remaining = indices[num_complete_batches * self.batch_size:]
                if len(remaining) > 0:
                    needed = self.batch_size - len(remaining)
                    padding = indices[:needed]
                    batches.append(torch.cat([remaining, padding]))
        else:
            # CPU numpy-based batches
            for i in range(num_complete_batches):
                start_idx = i * self.batch_size
                batches.append(indices[start_idx:start_idx + self.batch_size])

            # Handle remaining data
            if self.size % self.batch_size != 0:
                remaining = indices[num_complete_batches * self.batch_size:]
                if len(remaining) > 0:
                    needed = self.batch_size - len(remaining)
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
        use_compile: bool = True,
        use_amp: bool = True,
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
                self.debug_print("Compiling performance-critical functions...")

                # Compile GAE computation
                self._compute_gae_fn = self._create_compiled_gae()
                self.debug_print("GAE function compiled")

                # Compile policy evaluation
                self._policy_eval_fn = self._create_compiled_policy_eval()
                self.debug_print("Policy evaluation function compiled")

                # Compile batch processor
                self._process_batch_fn = self._create_compiled_batch_processor()
                self.debug_print("Batch processor function compiled")

                # Pre-warm compiled functions
                self._prewarm_compiled_functions()
                self.debug_print("Compiled functions pre-warmed")

            except Exception as e:
                self.debug_print(f"Function compilation failed: {str(e)}")
                self.use_compile = False


        try:
            self.debug_print(f"Compiling models for {self.device}...")

            # Configure common options for torch.compile
            compile_options = {
                "fullgraph": False,  # Don't try to capture the full graph
                "dynamic": True,     # Support for dynamic shapes
            }

            if self.device == "mps":
                # Apple Silicon - use compatible settings
                self.actor = torch.compile(self.actor, backend="aot_eager", **compile_options)
                self.critic = torch.compile(self.critic, backend="aot_eager", **compile_options)
            elif self.device == "cuda":
                # Configure PyTorch Dynamo for safer CUDA graph usage
                if hasattr(torch._dynamo.config, "allow_cudagraph_ops"):
                    torch._dynamo.config.allow_cudagraph_ops = True

                # Use safer backend mode for CUDA
                backend = "inductor"

                # CUDA with safer settings
                self.actor = torch.compile(self.actor, backend=backend, **compile_options)
                self.critic = torch.compile(self.critic, backend=backend, **compile_options)
            else:
                # CPU with balanced settings
                self.actor = torch.compile(self.actor, backend="inductor", **compile_options)
                self.critic = torch.compile(self.critic, backend="inductor", **compile_options)

        except Exception as e:
            self.debug_print(f"Model compilation failed: {str(e)}")
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
        self.memory = PPOMemory(batch_size, buffer_size=10000, device=device)

        # AMP
        self.use_amp = "cuda" in str(device) and use_amp
        if self.use_amp:
            self.scaler = GradScaler()
            self.debug_print("Automatic Mixed Precision (AMP) enabled")
        else:
            self.debug_print("Automatic Mixed Precision (AMP) disabled: requires CUDA")

        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.total_losses = []


        if self.debug:
            print(f"[DEBUG] PPOTrainer initialized with target device: {self.device}")

        if self.use_wandb:
            wandb.log({"device": self.device})

    def _create_compiled_gae(self):
        """Create a compiled version of GAE computation for tensor inputs"""
        def _tensor_gae(rewards, values, dones, next_value):
            advantages = torch.zeros_like(rewards, device=self.device)
            last_gae = torch.tensor(0.0, device=self.device)

            # Match dimensions for concatenation
            next_value_shaped = next_value.reshape(*[1] + list(values.shape[1:]))
            all_values = torch.cat([values, next_value_shaped], dim=0)

            for t in reversed(range(len(rewards))):
                mask = ~dones[t]
                if not mask:
                    delta = rewards[t] - values[t]
                    last_gae = delta
                else:
                    delta = rewards[t] + self.gamma * all_values[t+1] - values[t]
                    last_gae = delta + self.gamma * self.gae_lambda * last_gae
                advantages[t] = last_gae

            returns = advantages + values
            return advantages, returns

        # Compile the function for the target device
        if self.device == "mps":
            return torch.compile(_tensor_gae, backend="aot_eager")
        elif self.device == "cuda":
            return torch.compile(_tensor_gae, mode="max-autotune")
        else:
            return torch.compile(_tensor_gae, mode="reduce-overhead")

    def _create_compiled_policy_eval(self):
        """Create a compiled version of policy evaluation logic"""
        def _policy_eval(states, actions, old_log_probs, advantages):
            # Actor forward pass
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

            # PPO objectives
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.entropy_coef * entropy

            return actor_loss, entropy_loss, entropy

        # Compile function based on device
        if self.device == "mps":
            return torch.compile(_policy_eval, backend="aot_eager")
        elif self.device == "cuda":
            return torch.compile(_policy_eval, mode="max-autotune")
        else:
            return torch.compile(_policy_eval, mode="reduce-overhead")

    def _create_compiled_batch_processor(self):
        """Create compiled version of the batch processing logic"""
        def _process_batch(batch_states, batch_actions, batch_old_log_probs,
                            batch_advantages, batch_returns):
            # Critic forward pass
            values = self.critic(batch_states).squeeze()

            # Actor evaluation (using our compiled function)
            actor_loss, entropy_loss, entropy = self._policy_eval_fn(
                batch_states, batch_actions, batch_old_log_probs, batch_advantages
            )

            # Critic loss
            critic_loss = F.mse_loss(values, batch_returns)

            # Total loss
            total_loss = actor_loss + self.critic_coef * critic_loss + entropy_loss

            return actor_loss, critic_loss, entropy_loss, total_loss, entropy

        # Compile with appropriate backend
        if self.device == "mps":
            return torch.compile(_process_batch, backend="aot_eager")
        elif self.device == "cuda":
            return torch.compile(_process_batch, mode="max-autotune")
        else:
            return torch.compile(_process_batch, mode="reduce-overhead")

    def _uncompiled_policy_eval(self, states, actions, old_log_probs, advantages):
        """Uncompiled fallback for policy evaluation"""
        # Same logic as _policy_eval without compilation
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
        """Uncompiled fallback for batch processing"""
        # Same logic as _process_batch without compilation
        values = self.critic(batch_states).squeeze()

        actor_loss, entropy_loss, entropy = self._uncompiled_policy_eval(
            batch_states, batch_actions, batch_old_log_probs, batch_advantages
        )

        critic_loss = F.mse_loss(values, batch_returns)
        total_loss = actor_loss + self.critic_coef * critic_loss + entropy_loss

        return actor_loss, critic_loss, entropy_loss, total_loss, entropy

    def _prewarm_compiled_functions(self):
        """Pre-warm compiled functions with dummy data to complete compilation"""
        if not self.use_compile:
            return

        self.debug_print("Pre-warming compiled functions...")

        try:
            # Create small dummy tensors based on expected shapes
            dummy_size = 4  # Small batch size for pre-warming

            # For GAE
            if hasattr(self, '_compute_gae_fn'):
                dummy_rewards = torch.zeros(dummy_size, device=self.device)
                dummy_values = torch.zeros(dummy_size, device=self.device)
                dummy_dones = torch.zeros(dummy_size, dtype=torch.bool, device=self.device)
                dummy_next_value = torch.zeros(1, device=self.device)

                # Run once to trigger compilation
                _ = self._compute_gae_fn(dummy_rewards, dummy_values, dummy_dones, dummy_next_value)
                self.debug_print("GAE function pre-warmed")

            # For policy evaluation
            if hasattr(self, '_policy_eval_fn'):
                action_size = 1 if self.action_space_type == "discrete" else self.action_dim
                dummy_states = torch.zeros((dummy_size, self.actor.obs_shape), device=self.device)
                dummy_actions = torch.zeros((dummy_size, action_size), device=self.device) if self.action_space_type != "discrete" else torch.zeros(dummy_size, dtype=torch.long, device=self.device)
                dummy_log_probs = torch.zeros(dummy_size, device=self.device)
                dummy_advantages = torch.zeros(dummy_size, device=self.device)

                # Run once to trigger compilation
                _ = self._policy_eval_fn(dummy_states, dummy_actions, dummy_log_probs, dummy_advantages)
                self.debug_print("Policy evaluation function pre-warmed")

            # For batch processor
            if hasattr(self, '_process_batch_fn'):
                dummy_returns = torch.zeros(dummy_size, device=self.device)

                # Run once to trigger compilation
                _ = self._process_batch_fn(dummy_states, dummy_actions, dummy_log_probs, dummy_advantages, dummy_returns)
                self.debug_print("Batch processor function pre-warmed")

        except Exception as e:
            self.debug_print(f"Function pre-warming failed: {str(e)}")
            self.use_compile = False

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
        # Convert done to the appropriate type based on memory configuration
        if self.memory.use_device_tensors and not isinstance(done, torch.Tensor):
            # Convert Python/numpy boolean to PyTorch boolean tensor
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

        # Only update fields that are provided (not None)
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
        Get action and log probability from state using the actor network.
        Optimized for both single and batched inputs.
        """
        # Before any processing, make sure we're in a clean CUDA state
        if "cuda" in str(self.device):
            try:
                # Force synchronization at the beginning of the call
                torch.cuda.synchronize()
                torch.compiler.cudagraph_mark_step_begin()
            except Exception as e:
                # Just log and continue if this fails
                if self.debug:
                    print(f"[DEBUG] CUDA sync warning: {str(e)}")

        # Convert input to tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            if state.dim() == 1 or (len(state.shape) == 2 and state.shape[0] == 1 and state.shape[1] > 1):
                state = state.unsqueeze(0)

        # Clone input tensor to avoid CUDA graph dependencies, but keep on CPU
        state_clone = state.clone()

        # Move to device
        state_device = state_clone.to(self.device)

        with torch.no_grad():  # No grad for inference actions
            with autocast(enabled=self.use_amp, device_type="cuda"):
                # Mark step before each model execution
                if "cuda" in str(self.device):
                    torch.compiler.cudagraph_mark_step_begin()

                # Get value estimate from critic
                value = self.critic(state_device)
                value = value.clone()  # Clone but keep on device

                # Mark step before actor execution
                if "cuda" in str(self.device):
                    torch.compiler.cudagraph_mark_step_begin()

                # Get action distribution from actor
                if self.action_space_type == "discrete":
                    logits = self.actor(state_device)
                    logits = logits.clone()  # Clone but keep on device

                    # Mark step after model execution
                    if "cuda" in str(self.device):
                        torch.compiler.cudagraph_mark_step_begin()

                    action_probs = F.softmax(logits, dim=1)
                    dist = Categorical(action_probs)

                    if evaluate:
                        action = torch.argmax(action_probs, dim=1)
                    else:
                        action = dist.sample()

                    log_prob = dist.log_prob(action)
                else:
                    # Continuous actions
                    mu, sigma_raw = self.actor(state_device)
                    mu = mu.clone()  # Clone but keep on device
                    sigma_raw = sigma_raw.clone() if isinstance(sigma_raw, torch.Tensor) else sigma_raw

                    # Mark step after model execution
                    if "cuda" in str(self.device):
                        torch.compiler.cudagraph_mark_step_begin()

                    sigma = F.softplus(sigma_raw) + 1e-5
                    dist = Normal(mu, sigma)

                    if evaluate:
                        action = mu
                    else:
                        action = dist.sample()

                    log_prob = dist.log_prob(action)
                    if len(log_prob.shape) > 1:
                        log_prob = log_prob.sum(dim=-1)

                    action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])

                # Make sure all outputs are cloned
                action = action.clone()
                log_prob = log_prob.clone()
                value = value.clone()

        # Return detached numpy arrays for use in the environment
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute GAE with support for both tensor and numpy inputs"""
        # Check if inputs are tensors or numpy arrays
        is_tensor = isinstance(rewards, torch.Tensor)

        if self.use_compile:
            if hasattr(self, '_compute_gae_fn'):
                return self._compute_gae_fn(rewards, values, dones, next_value)

        if is_tensor:
            advantages = torch.zeros_like(rewards, device=self.device)
            last_gae = torch.tensor(0.0, device=self.device)

            # Handle dimension mismatch more carefully
            if next_value.dim() != values.dim():
                if next_value.dim() > values.dim():
                    # Squeeze extra dimensions
                    next_value = next_value.squeeze()

                # Add necessary dimensions if still needed
                while next_value.dim() < values.dim():
                    next_value = next_value.unsqueeze(0)

            # Create compatible tensor for concatenation
            if len(values.shape) == 1:
                next_value_shaped = next_value.reshape(1)
            else:
                # Match the last dimensions
                next_value_shaped = next_value.reshape(*[1] + list(values.shape[1:]))

            # Now concatenate
            try:
                all_values = torch.cat([values, next_value_shaped], dim=0)
            except RuntimeError as e:
                # Debug shapes
                print(f"values shape: {values.shape}, next_value shape: {next_value_shaped.shape}")
                print(f"values dim: {values.dim()}, next_value dim: {next_value_shaped.dim()}")
                raise RuntimeError(f"Error in torch.cat: {e}. Shape mismatch between values and next_value.")

            for t in reversed(range(len(rewards))):
                mask = ~dones[t] if isinstance(dones[t], torch.Tensor) else not dones[t]
                if not mask:
                    delta = rewards[t] - values[t]
                    last_gae = delta
                else:
                    delta = rewards[t] + self.gamma * all_values[t+1] - values[t]
                    last_gae = delta + self.gamma * self.gae_lambda * last_gae

                advantages[t] = last_gae
        else:
            # Original numpy implementation for CPU
            advantages = np.zeros(len(rewards), dtype=np.float32)
            last_gae = 0

            if not isinstance(rewards, np.ndarray):
                rewards = np.array(rewards)
            if not isinstance(values, np.ndarray):
                values = np.array(values)
            if not isinstance(dones, np.ndarray):
                dones = np.array(dones)

            # Handle numpy arrays and torch tensors
            if isinstance(next_value, torch.Tensor):
                next_value = next_value.cpu().detach().numpy()

            # Reshape to match values dimensions
            if values.ndim == 1:
                next_value = np.array([next_value]).flatten()
            else:
                next_value = np.reshape(next_value, (1,) + values.shape[1:])

            # Concatenate
            all_values = np.concatenate([values, next_value], axis=0)

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
        # self.critic.to(self.device)
        # self.actor.to(self.device)

        states, actions, old_log_probs, rewards, values, dones = self.memory.get()
        self.debug_print(f"Memory retrieved: {len(states)} timesteps")

        # Calculate episode metrics before processing
        episodes_ended = torch.sum(dones.int()).item() if isinstance(dones, torch.Tensor) else np.sum(dones) if len(dones) > 0 else 0
        if episodes_ended > 0:
            # Calculate average reward per episode
            episode_rewards = []
            episode_reward = 0
            episode_lengths = []
            episode_length = 0

            for i, (reward, done) in enumerate(zip(rewards, dones)):
                # Handle tensor or numpy value
                r_val = reward.item() if isinstance(reward, torch.Tensor) else reward
                d_val = done.item() if isinstance(done, torch.Tensor) else done

                episode_reward += r_val
                episode_length += 1

                if d_val or i == len(rewards) - 1:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    episode_reward = 0
                    episode_length = 0

            avg_episode_reward = np.mean(episode_rewards)
            avg_episode_length = np.mean(episode_lengths)
        else:
            # Handle rewards properly based on type
            if isinstance(rewards, torch.Tensor):
                avg_episode_reward = rewards.mean().item() if rewards.numel() > 0 else 0
            else:
                avg_episode_reward = np.mean(rewards) if len(rewards) > 0 else 0
        avg_episode_length = len(rewards)

        if len(states) > 0 and not dones[-1]:
            self.debug_print("Computing next value...")
            try:
                self.debug_print("Preparing next state...")
                with torch.no_grad():
                    next_state = states[-1].unsqueeze(0) if isinstance(states, torch.Tensor) else torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
                    self.debug_print("Running critic forward pass...")
                    next_value = self.critic(next_state)
                    self.debug_print("Next value computation complete")
            except Exception as e:
                print(f"Error during next value computation: {str(e)}")
                raise
        else:
            next_value = torch.tensor([0.0], device=self.device) if isinstance(values, torch.Tensor) else 0.0

        self.debug_print("Computing advantages...")
        compute_start = time.time()
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Normalize advantages
        if isinstance(advantages, torch.Tensor):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        self.debug_print(f"Advantage computation took {time.time() - compute_start:.2f}s")

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        total_loss = 0

        # Variables for tracking policy changes
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
            if epoch == 0 and self.action_space_type == "discrete" and isinstance(states, torch.Tensor):
                with torch.no_grad():
                    old_logits = self.actor(states)
                    old_probs = F.softmax(old_logits, dim=1)
            elif epoch == 0 and isinstance(states, torch.Tensor):
                with torch.no_grad():
                    old_mu, old_sigma_raw = self.actor(states)
                    old_sigma = F.softplus(old_sigma_raw) + 1e-5

            for batch_idx, batch_indices in enumerate(batches):
                batch_loop_start = time.time()

                if isinstance(states, torch.Tensor):
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                else:
                    batch_states = torch.tensor(states[batch_indices], dtype=torch.float32, device=self.device)
                    batch_actions = torch.tensor(actions[batch_indices], dtype=torch.long if self.action_space_type == "discrete" else torch.float32, device=self.device)
                    batch_old_log_probs = torch.tensor(old_log_probs[batch_indices], dtype=torch.float32, device=self.device)
                    batch_advantages = torch.tensor(advantages[batch_indices], dtype=torch.float32, device=self.device)
                    batch_returns = torch.tensor(returns[batch_indices], dtype=torch.float32, device=self.device)

                try:
                    self.debug_print(f"Processing batch {batch_idx+1}/{len(batches)}")
                    self.debug_print(f"Batch states shape: {batch_states.shape}")
                    self.debug_print(f"Batch actions shape: {batch_actions.shape}")

                    # Reset gradients
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    # Use AMP autocast for forward pass
                    with autocast(enabled=self.use_amp, device_type="cuda"):
                        # Make sure tensors will have proper gradients
                        if not batch_states.requires_grad:
                            batch_states = batch_states.clone()

                        # Use compiled functions if available
                        if self.use_compile and hasattr(self, '_process_batch_fn'):
                            actor_loss, critic_loss, entropy_loss, loss, entropy = self._process_batch_fn(
                                batch_states, batch_actions, batch_old_log_probs,
                                batch_advantages, batch_returns
                            )
                        else:
                            # Mark CUDA graph step if on CUDA
                            if "cuda" in str(self.device):
                                torch.compiler.cudagraph_mark_step_begin()

                            # Critic forward pass
                            self.debug_print("Running critic forward pass...")
                            values = self.critic(batch_states).squeeze()
                            self.debug_print("Critic forward pass complete")

                            # Mark CUDA graph step before actor
                            if "cuda" in str(self.device):
                                torch.compiler.cudagraph_mark_step_begin()

                            # Actor forward pass
                            self.debug_print("Running actor forward pass...")
                            if self.action_space_type == "discrete":
                                logits = self.actor(batch_states)
                                logits = logits.clone() # Clone to prevent CUDA graph issues

                                action_probs = F.softmax(logits, dim=1)
                                dist = Categorical(action_probs)
                                new_log_probs = dist.log_prob(batch_actions)
                                entropy = dist.entropy().mean()
                            else:
                                mu, sigma_raw = self.actor(batch_states)
                                mu = mu.clone() # Clone to prevent CUDA graph issues
                                sigma_raw = sigma_raw.clone() if isinstance(sigma_raw, torch.Tensor) else sigma_raw

                                sigma = F.softplus(sigma_raw) + 1e-5
                                dist = Normal(mu, sigma)
                                new_log_probs = dist.log_prob(batch_actions)
                                if len(new_log_probs.shape) > 1:
                                    new_log_probs = new_log_probs.sum(dim=1)
                                entropy = dist.entropy().mean()

                            # Mark CUDA graph step before loss computation
                            if "cuda" in str(self.device):
                                torch.compiler.cudagraph_mark_step_begin()

                            ratio = torch.exp(new_log_probs - batch_old_log_probs)
                            surr1 = ratio * batch_advantages
                            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages

                            actor_loss = -torch.min(surr1, surr2).mean()
                            critic_loss = F.mse_loss(values, batch_returns)
                            entropy_loss = -self.entropy_coef * entropy
                            loss = actor_loss + self.critic_coef * critic_loss + entropy_loss

                            # Verify loss has gradients
                            if not loss.requires_grad:
                                self.debug_print("Warning: Loss doesn't have gradients. Creating new computation graph.")
                                # Create a fresh computation with explicit gradient tracking
                                with torch.enable_grad():
                                    if hasattr(self.actor, '_orig_mod'):
                                        actor = self.actor._orig_mod
                                    else:
                                        actor = self.actor

                                    if hasattr(self.critic, '_orig_mod'):
                                        critic = self.critic._orig_mod
                                    else:
                                        critic = self.critic

                                    # Recompute with gradient tracking explicitly enabled
                                    values = critic(batch_states).squeeze()

                                    if self.action_space_type == "discrete":
                                        logits = actor(batch_states)
                                        action_probs = F.softmax(logits, dim=1)
                                        dist = Categorical(action_probs)
                                        new_log_probs = dist.log_prob(batch_actions)
                                        entropy = dist.entropy().mean()
                                    else:
                                        mu, sigma_raw = actor(batch_states)
                                        sigma = F.softplus(sigma_raw) + 1e-5
                                        dist = Normal(mu, sigma)
                                        new_log_probs = dist.log_prob(batch_actions)
                                        if len(new_log_probs.shape) > 1:
                                            new_log_probs = new_log_probs.sum(dim=1)
                                        entropy = dist.entropy().mean()

                                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                                    surr1 = ratio * batch_advantages
                                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages

                                    actor_loss = -torch.min(surr1, surr2).mean()
                                    critic_loss = F.mse_loss(values, batch_returns)
                                    entropy_loss = -self.entropy_coef * entropy
                                    loss = actor_loss + self.critic_coef * critic_loss + entropy_loss

                    batch_time = time.time() - batch_loop_start
                    batch_times.append(batch_time)

                    if batch_idx % 10 == 0:
                        self.debug_print(f"Batch {batch_idx+1}/{len(batches)} took {batch_time:.3f}s")

                except Exception as e:
                    # Keep error messages as regular prints
                    print(f"Error in batch {batch_idx+1}: {str(e)}")
                    raise

                self.debug_print("Starting backward pass...")

                # Use scaler for backward pass if AMP is enabled
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.debug_print("Clipping gradients...")
                    self.scaler.unscale_(self.actor_optimizer)
                    self.scaler.unscale_(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.debug_print("Clipping gradients...")
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
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
        if self.action_space_type == "discrete" and len(states) > 0 and old_probs is not None and isinstance(states, torch.Tensor):
            with torch.no_grad():
                new_logits = self.actor(states)
                new_probs = F.softmax(new_logits, dim=1)
                kl_div = (old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10))).sum(1).mean().item()
        elif len(states) > 0 and old_mu is not None and old_sigma is not None and isinstance(states, torch.Tensor):
            with torch.no_grad():
                new_mu, new_sigma_raw = self.actor(states)
                new_sigma = F.softplus(new_sigma_raw) + 1e-5
                kl_div = (torch.log(new_sigma/old_sigma) + (old_sigma**2 + (old_mu - new_mu)**2)/(2*new_sigma**2) - 0.5).mean().item()

        # Calculate explained variance safely
        if len(states) > 0:
            with torch.no_grad():
                # Get all values for all states
                all_values = self.critic(states).cpu().detach().numpy().flatten() if isinstance(states, torch.Tensor) else self.critic(torch.tensor(states, dtype=torch.float32, device=self.device)).cpu().detach().numpy().flatten()
                all_returns = returns.cpu().detach().numpy().flatten() if isinstance(returns, torch.Tensor) else returns

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
                        "mean_advantage": advantages.mean().item() if isinstance(advantages, torch.Tensor) else np.mean(advantages),
                        "mean_return": returns.mean().item() if isinstance(returns, torch.Tensor) else np.mean(returns),

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
