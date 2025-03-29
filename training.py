from torch.distributions import Categorical, Normal
from torch.amp import autocast, GradScaler
from models import fix_compiled_state_dict, print_model_info
from typing import Union, Tuple, Optional, Dict, Any
from auxiliary import AuxiliaryTaskManager
from intrinsic_rewards import create_intrinsic_reward_generator, IntrinsicRewardEnsemble
from algorithms import BaseAlgorithm, PPOAlgorithm, StreamACAlgorithm
import time
import wandb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

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
            self.dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        else:
            # We're on a CPU, use numpy arrays which are more efficient on CPU.
            self.states = None  # Will be initialized when we get our first state
            self.actions = None  # Will be initialized when we get our first action
            self.logprobs = np.zeros((buffer_size,), dtype=np.float32)
            self.rewards = np.zeros((buffer_size,), dtype=np.float32)
            self.values = np.zeros((buffer_size,), dtype=np.float32)
            self.dones = np.zeros((buffer_size,), dtype=np.float32)

        self.state_initialized = False
        self.action_initialized = False
        self.pos = 0
        self.size = 0

    def store(self, state, action, logprob, reward, value, done):
        """Store an experience tuple in the buffer."""

        # Initialize state storage, if needed, and determine its dimensions.
        if not self.state_initialized and state is not None:
            if self.use_device_tensors:
                # If it's a tensor, create a tensor of the right shape.
                if isinstance(state, torch.Tensor):
                    state_shape = state.shape
                    # Check dimensions to handle both batched (2D) and unbatched (1D) inputs
                    if len(state_shape) == 1:
                        # Unbatched input, create a 2D states tensor [buffer_size, state_dim]
                        self.states = torch.zeros((self.buffer_size, state_shape[0]), 
                                                  device=self.device, dtype=torch.float32)
                    else:
                        # Batched input, match the dimensions [buffer_size, batch_dim, state_dim]
                        self.states = torch.zeros((self.buffer_size, *state_shape[1:]), 
                                                  device=self.device, dtype=torch.float32)
                else:
                    # If it's a numpy array or list, determine shape and create tensor.
                    state_array = np.array(state)
                    state_shape = state_array.shape
                    # Check dimensions
                    if len(state_shape) == 1:
                        # Unbatched input
                        self.states = torch.zeros((self.buffer_size, state_shape[0]),
                                                  device=self.device, dtype=torch.float32)
                    else:
                        # Batched input
                        self.states = torch.zeros((self.buffer_size, *state_shape[1:]),
                                                  device=self.device, dtype=torch.float32)
            else:
                # We're on CPU, use numpy arrays.
                if isinstance(state, torch.Tensor):
                    state_array = state.cpu().numpy()
                else:
                    state_array = np.array(state)
                    
                state_shape = state_array.shape
                # Check dimensions
                if len(state_shape) == 1:
                    # Unbatched input
                    self.states = np.zeros((self.buffer_size, state_shape[0]), dtype=np.float32)
                else:
                    # Batched input
                    self.states = np.zeros((self.buffer_size, *state_shape[1:]), dtype=np.float32)
            
            self.state_initialized = True

        # Initialize action storage, if needed, and figure out if actions are discrete or continuous.
        if not self.action_initialized and action is not None:
            if self.use_device_tensors:
                # If it's a tensor, create a tensor of the right shape.
                if isinstance(action, torch.Tensor):
                    action_shape = action.shape
                    # Check if we have scalar actions (discrete) or vector actions (continuous)
                    if action_shape == () or (len(action_shape) == 1 and action_shape[0] == 1):
                        # Scalar/discrete actions
                        self.actions = torch.zeros((self.buffer_size,), 
                                                   device=self.device, dtype=torch.long)
                    elif len(action_shape) == 1:
                        # Vector actions with a single dimension (e.g., [batch])
                        self.actions = torch.zeros((self.buffer_size, 1), 
                                                   device=self.device, dtype=torch.float32)
                    else:
                        # Vector actions with multiple dimensions (e.g., [batch, action_dim])
                        self.actions = torch.zeros((self.buffer_size, *action_shape[1:]), 
                                                   device=self.device, dtype=torch.float32)
                else:
                    # If it's a numpy array, list, or scalar, determine shape and create tensor.
                    if isinstance(action, (int, float)):
                        # Scalar action (discrete)
                        self.actions = torch.zeros((self.buffer_size,), 
                                                   device=self.device, dtype=torch.long)
                    else:
                        # Vector action (continuous)
                        action_array = np.array(action)
                        action_shape = action_array.shape
                        if not action_shape:  # Scalar numpy array
                            self.actions = torch.zeros((self.buffer_size,), 
                                                       device=self.device, dtype=torch.long)
                        elif len(action_shape) == 1:
                            # Vector with a single dimension
                            self.actions = torch.zeros((self.buffer_size, action_shape[0]),
                                                       device=self.device, dtype=torch.float32)
                        else:
                            # Vector with multiple dimensions
                            self.actions = torch.zeros((self.buffer_size, *action_shape[1:]),
                                                       device=self.device, dtype=torch.float32)
            else:
                # We're on CPU, use numpy arrays.
                if isinstance(action, torch.Tensor):
                    action_array = action.cpu().numpy()
                elif isinstance(action, (int, float)):
                    # Scalar action (discrete)
                    self.actions = np.zeros((self.buffer_size,), dtype=np.int64)
                    action_array = np.array(action, dtype=np.int64)
                else:
                    # Vector action (continuous)
                    action_array = np.array(action)
                    
                action_shape = action_array.shape
                if not action_shape:  # Scalar numpy array
                    self.actions = np.zeros((self.buffer_size,), dtype=np.int64)
                elif len(action_shape) == 1:
                    # Vector with a single dimension
                    self.actions = np.zeros((self.buffer_size, action_shape[0]), dtype=np.float32)
                else:
                    # Vector with multiple dimensions
                    self.actions = np.zeros((self.buffer_size, *action_shape[1:]), dtype=np.float32)
            
            self.action_initialized = True

        # Actually store the provided values.
        if self.use_device_tensors:
            # Convert all inputs to tensors if they're not already, and ensure on correct device.
            if state is not None and self.state_initialized:
                if isinstance(state, torch.Tensor):
                    self.states[self.pos] = state.to(self.device)
                else:
                    self.states[self.pos] = torch.tensor(state, device=self.device, dtype=torch.float32)
                    
            if action is not None and self.action_initialized:
                if isinstance(action, torch.Tensor):
                    self.actions[self.pos] = action.to(self.device)
                else:
                    # Handle both discrete and continuous cases
                    if isinstance(self.actions, torch.Tensor) and self.actions.dtype == torch.long:
                        # Discrete action
                        self.actions[self.pos] = torch.tensor(action, device=self.device, dtype=torch.long)
                    else:
                        # Continuous action
                        self.actions[self.pos] = torch.tensor(action, device=self.device, dtype=torch.float32)
                
            if logprob is not None:
                if isinstance(logprob, torch.Tensor):
                    self.logprobs[self.pos] = logprob.to(self.device)
                else:
                    self.logprobs[self.pos] = torch.tensor(logprob, device=self.device, dtype=torch.float32)
                    
            if reward is not None:
                if isinstance(reward, torch.Tensor):
                    self.rewards[self.pos] = reward.to(self.device)
                else:
                    self.rewards[self.pos] = torch.tensor(reward, device=self.device, dtype=torch.float32)
                    
            if value is not None:
                if isinstance(value, torch.Tensor):
                    self.values[self.pos] = value.to(self.device)
                else:
                    self.values[self.pos] = torch.tensor(value, device=self.device, dtype=torch.float32)
                    
            if done is not None:
                if isinstance(done, torch.Tensor):
                    self.dones[self.pos] = done.to(self.device)
                else:
                    self.dones[self.pos] = torch.tensor(float(done), dtype=torch.float32, device=self.device)
        else:
            # Store using numpy arrays for CPU.
            if state is not None and self.state_initialized:
                if isinstance(state, torch.Tensor):
                    self.states[self.pos] = state.cpu().numpy()
                else:
                    self.states[self.pos] = np.array(state, dtype=np.float32)
                    
            if action is not None and self.action_initialized:
                if isinstance(action, torch.Tensor):
                    # Convert tensor to numpy, handling both discrete and continuous
                    if isinstance(self.actions, np.ndarray) and self.actions.dtype == np.int64:
                        # Discrete action
                        self.actions[self.pos] = action.cpu().numpy().astype(np.int64)
                    else:
                        # Continuous action
                        self.actions[self.pos] = action.cpu().numpy().astype(np.float32)
                else:
                    # Handle both discrete and continuous cases
                    if isinstance(self.actions, np.ndarray) and self.actions.dtype == np.int64:
                        # Discrete action
                        self.actions[self.pos] = np.array(action, dtype=np.int64)
                    else:
                        # Continuous action
                        self.actions[self.pos] = np.array(action, dtype=np.float32)
                        
            if logprob is not None:
                if isinstance(logprob, torch.Tensor):
                    self.logprobs[self.pos] = logprob.cpu().numpy()
                else:
                    self.logprobs[self.pos] = np.array(logprob, dtype=np.float32)
                    
            if reward is not None:
                if isinstance(reward, torch.Tensor):
                    self.rewards[self.pos] = reward.cpu().numpy()
                else:
                    self.rewards[self.pos] = np.array(reward, dtype=np.float32)
                    
            if value is not None:
                if isinstance(value, torch.Tensor):
                    self.values[self.pos] = value.cpu().numpy()
                else:
                    self.values[self.pos] = np.array(value, dtype=np.float32)
                    
            if done is not None:
                if isinstance(done, torch.Tensor):
                    self.dones[self.pos] = done.cpu().numpy()
                else:
                    self.dones[self.pos] = np.array(float(done), dtype=np.float32)

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
        
        # Force CUDA memory cleanup if using device tensors on CUDA
        if self.use_device_tensors and self.device.startswith('cuda'):
            torch.cuda.empty_cache()

    def get(self):
        """Get all data currently stored in the buffer."""
        if self.size == 0 or not self.state_initialized:
            return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

        if self.use_device_tensors:
            # Get the used portion of each tensor, handling circular buffer wraparound.
            if self.pos < self.size:  # Buffer hasn't wrapped around
                valid_states = self.states[:self.size]
                valid_actions = self.actions[:self.size]
                valid_logprobs = self.logprobs[:self.size]
                valid_rewards = self.rewards[:self.size]
                valid_values = self.values[:self.size]
                valid_dones = self.dones[:self.size]
            else:  # Buffer has wrapped around
                # Concatenate the end of the buffer with the beginning.
                valid_states = torch.cat([self.states[-(self.size - self.pos):], self.states[:self.pos]])
                valid_actions = torch.cat([self.actions[-(self.size - self.pos):], self.actions[:self.pos]])
                valid_logprobs = torch.cat([self.logprobs[-(self.size - self.pos):], self.logprobs[:self.pos]])
                valid_rewards = torch.cat([self.rewards[-(self.size - self.pos):], self.rewards[:self.pos]])
                valid_values = torch.cat([self.values[-(self.size - self.pos):], self.values[:self.pos]])
                valid_dones = torch.cat([self.dones[-(self.size - self.pos):], self.dones[:self.pos]])
        else:
            # Same logic but with numpy arrays.
            if self.pos < self.size:  # Buffer hasn't wrapped around
                valid_states = self.states[:self.size]
                valid_actions = self.actions[:self.size]
                valid_logprobs = self.logprobs[:self.size]
                valid_rewards = self.rewards[:self.size]
                valid_values = self.values[:self.size]
                valid_dones = self.dones[:self.size]
            else:  # Buffer has wrapped around
                # Concatenate the end of the buffer with the beginning.
                valid_states = np.concatenate([self.states[-(self.size - self.pos):], self.states[:self.pos]])
                valid_actions = np.concatenate([self.actions[-(self.size - self.pos):], self.actions[:self.pos]])
                valid_logprobs = np.concatenate([self.logprobs[-(self.size - self.pos):], self.logprobs[:self.pos]])
                valid_rewards = np.concatenate([self.rewards[-(self.size - self.pos):], self.rewards[:self.pos]])
                valid_values = np.concatenate([self.values[-(self.size - self.pos):], self.values[:self.pos]])
                valid_dones = np.concatenate([self.dones[-(self.size - self.pos):], self.dones[:self.pos]])

        return (valid_states, valid_actions, valid_logprobs, valid_rewards, valid_values, valid_dones)

    def generate_batches(self):
        """
        Generates batches of indices, using prioritized sampling.
        """
        if self.size == 0:
            return []

        # Calculate normalized probabilities for prioritized sampling.
        if self.use_device_tensors:
            indices = torch.randperm(self.size).to(self.device)
            batches = [indices[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]
        else:
            indices = np.random.permutation(self.size)
            batches = [indices[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]

        return batches
    
    def store_experience_at_idx(self, idx, state=None, action=None, log_prob=None, reward=None, value=None, done=None):
        """Update only specific values at an index, rather than a complete experience."""
        if idx >= self.buffer_size:
            return  # Index out of range

        # Update only the provided values (non-None values).
        if self.use_device_tensors:
            if state is not None and self.state_initialized:
                if isinstance(state, torch.Tensor):
                    self.states[idx] = state.to(self.device)
                else:
                    self.states[idx] = torch.tensor(state, device=self.device, dtype=torch.float32)
                    
            if action is not None and self.action_initialized:
                if isinstance(action, torch.Tensor):
                    self.actions[idx] = action.to(self.device)
                else:
                    # Handle both discrete and continuous cases
                    if isinstance(self.actions, torch.Tensor) and self.actions.dtype == torch.long:
                        # Discrete action
                        self.actions[idx] = torch.tensor(action, device=self.device, dtype=torch.long)
                    else:
                        # Continuous action
                        self.actions[idx] = torch.tensor(action, device=self.device, dtype=torch.float32)
                        
            if log_prob is not None:
                if isinstance(log_prob, torch.Tensor):
                    self.logprobs[idx] = log_prob.to(self.device)
                else:
                    self.logprobs[idx] = torch.tensor(log_prob, device=self.device, dtype=torch.float32)
                    
            if reward is not None:
                if isinstance(reward, torch.Tensor):
                    self.rewards[idx] = reward.to(self.device)
                else:
                    self.rewards[idx] = torch.tensor(reward, device=self.device, dtype=torch.float32)
                    
            if value is not None:
                if isinstance(value, torch.Tensor):
                    self.values[idx] = value.to(self.device)
                else:
                    self.values[idx] = torch.tensor(value, device=self.device, dtype=torch.float32)
                    
            if done is not None:
                if isinstance(done, torch.Tensor):
                    self.dones[idx] = done.to(self.device)
                else:
                    self.dones[idx] = torch.tensor(float(done), dtype=torch.float32, device=self.device)
        else:
            # Do the same with numpy arrays for CPU.
            if state is not None and self.state_initialized:
                if isinstance(state, torch.Tensor):
                    self.states[idx] = state.cpu().numpy()
                else:
                    self.states[idx] = np.array(state, dtype=np.float32)
                    
            if action is not None and self.action_initialized:
                if isinstance(action, torch.Tensor):
                    # Convert tensor to numpy, handling both discrete and continuous
                    if isinstance(self.actions, np.ndarray) and self.actions.dtype == np.int64:
                        # Discrete action
                        self.actions[idx] = action.cpu().numpy().astype(np.int64)
                    else:
                        # Continuous action
                        self.actions[idx] = action.cpu().numpy().astype(np.float32)
                else:
                    # Handle both discrete and continuous cases
                    if isinstance(self.actions, np.ndarray) and self.actions.dtype == np.int64:
                        # Discrete action
                        self.actions[idx] = np.array(action, dtype=np.int64)
                    else:
                        # Continuous action
                        self.actions[idx] = np.array(action, dtype=np.float32)
                        
            if log_prob is not None:
                if isinstance(log_prob, torch.Tensor):
                    self.logprobs[idx] = log_prob.cpu().numpy()
                else:
                    self.logprobs[idx] = np.array(log_prob, dtype=np.float32)
                    
            if reward is not None:
                if isinstance(reward, torch.Tensor):
                    self.rewards[idx] = reward.cpu().numpy()
                else:
                    self.rewards[idx] = np.array(reward, dtype=np.float32)
                    
            if value is not None:
                if isinstance(value, torch.Tensor):
                    self.values[idx] = value.cpu().numpy()
                else:
                    self.values[idx] = np.array(value, dtype=np.float32)
                    
            if done is not None:
                if isinstance(done, torch.Tensor):
                    self.dones[idx] = done.cpu().numpy()
                else:
                    self.dones[idx] = np.array(float(done), dtype=np.float32)


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


class Trainer:
    """
    Base trainer class that manages the training of reinforcement learning algorithms.
    Supports both PPO and StreamAC algorithms.
    """
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        algorithm_type: str = "ppo",
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
        use_amp: bool = False,  
        use_auxiliary_tasks: bool = True,
        sr_weight: float = 1.0,
        rp_weight: float = 1.0,
        aux_amp: bool = False,
        use_pretraining: bool = False,
        pretraining_fraction: float = 0.1,
        pretraining_sr_weight: float = 10.0,
        pretraining_rp_weight: float = 5.0,
        pretraining_transition_steps: int = 1000,
        total_episode_target: int = None,
        training_step_offset: int = 0,
        # Intrinsic reward parameters
        use_intrinsic_rewards: bool = True,
        intrinsic_reward_scale: float = 1.0,
        curiosity_weight: float = 0.5,
        rnd_weight: float = 0.5,
        # StreamAC specific parameters
        adaptive_learning_rate: bool = True,
        target_step_size: float = 0.025,
        backtracking_patience: int = 10,
        backtracking_zeta: float = 0.85,
        min_lr_factor: float = 0.1,
        max_lr_factor: float = 10.0,
        use_obgd: bool = True,
        stream_buffer_size: int = 32,
        use_sparse_init: bool = True,
        update_freq: int = 1,
    ):
        self.use_wandb = use_wandb
        self.debug = debug
        self.test_mode = False  # Initialize test_mode attribute to False by default

        # Figure out which device (CPU, CUDA, MPS) to use
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.action_space_type = action_space_type
        self.action_dim = action_dim
        self.action_bounds = action_bounds

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # We'll decay the entropy coefficient over time to encourage exploration, then exploitation
        self.entropy_coef = entropy_coef if entropy_coef else 0.01
        self.entropy_coef_decay = entropy_coef_decay if entropy_coef_decay else 0.999
        self.min_entropy_coef = 0.005
        
        # Pretraining entropy management
        self.base_entropy_coef = entropy_coef
        self.pretraining_entropy_scale = 10.0  # Higher entropy during pretraining
        self.min_entropy_coef = 0.005

        # Set to training mode
        self.actor.train()
        self.critic.train()

        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Learning rates
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # Create the learning algorithm based on type
        self.algorithm_type = algorithm_type.lower()
        
        if self.algorithm_type == "ppo":
            self.algorithm = PPOAlgorithm(
                actor=self.actor,
                critic=self.critic,
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
                use_wandb=use_wandb
            )
            # For compatibility with existing code, keep a reference to the memory
            self.memory = self.algorithm.memory
        elif self.algorithm_type == "streamac":
            self.algorithm = StreamACAlgorithm(
                actor=self.actor,
                critic=self.critic,
                action_space_type=action_space_type,
                action_dim=action_dim,
                action_bounds=action_bounds,
                device=device,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                critic_coef=critic_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                use_amp=use_amp,
                debug=debug,
                use_wandb=use_wandb,
                # StreamAC specific parameters
                adaptive_learning_rate=adaptive_learning_rate,
                target_step_size=target_step_size,
                backtracking_patience=backtracking_patience,
                backtracking_zeta=backtracking_zeta,
                min_lr_factor=min_lr_factor,
                max_lr_factor=max_lr_factor,
                use_obgd=use_obgd,
                buffer_size=stream_buffer_size,
                use_sparse_init=use_sparse_init,
                update_freq=update_freq
            )
            # For compatibility, create an empty PPOMemory
            self.memory = PPOMemory(batch_size, buffer_size=1, device=device)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}. Use 'ppo' or 'streamac'.")

        # Use Automatic Mixed Precision (AMP) if requested and on CUDA
        self.use_amp = "cuda" in str(device) and use_amp
        if self.use_amp:
            self.scaler = GradScaler()

        # Track metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.total_losses = []

        # Initialize auxiliary tasks if enabled
        self.use_auxiliary_tasks = use_auxiliary_tasks
        if self.use_auxiliary_tasks:
            # Get observation dimension from model if available or derive from action_dim
            obs_dim = getattr(actor, 'obs_shape', action_dim * 2)  # Use action_dim * 2 as fallback
            
            self.aux_task_manager = AuxiliaryTaskManager(
                actor=self.actor,
                obs_dim=obs_dim,
                sr_weight=sr_weight,
                rp_weight=rp_weight,
                device=self.device,
                use_amp=aux_amp,
                update_frequency=1,
                learning_mode="stream" if algorithm_type == "streamac" else "batch"  # Explicitly set mode
            )
            
            # Enable debug mode for auxiliary tasks
            self.aux_task_manager.debug = self.debug

        if self.debug:
            print(f"[DEBUG] Initialized {self.algorithm_type.upper()} algorithm on {self.device}")
            print(f"[DEBUG] Actor: {actor.__class__.__name__}, Critic: {critic.__class__.__name__}")

        if self.use_wandb:
            wandb.config.update({
                "algorithm": self.algorithm_type,
                "action_space_type": action_space_type,
                "lr_actor": lr_actor,
                "lr_critic": lr_critic,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_epsilon": clip_epsilon,
                "critic_coef": critic_coef,
                "entropy_coef": entropy_coef,
                "max_grad_norm": max_grad_norm,
                "use_auxiliary_tasks": use_auxiliary_tasks,
                "sr_weight": sr_weight,
                "rp_weight": rp_weight,
                "use_pretraining": use_pretraining,
                "use_amp": use_amp
            }, allow_val_change=True)  # Add allow_val_change=True

        # If requested and available, compile the models for performance
        self.use_compile = use_compile and hasattr(torch, 'compile')
        if self.use_compile:
            try:
                if self.debug:
                    print("[DEBUG] Using torch.compile for models...")
                self.actor = torch.compile(self.actor)
                self.critic = torch.compile(self.critic)
                if self.debug:
                    print("[DEBUG] Models compiled successfully")
            except Exception as e:
                print(f"Error compiling models: {e}")
                print("Continuing without compilation")
                self.use_compile = False

        # Print model info
        print_model_info(self.actor, model_name="Actor", print_amp=self.use_amp, debug=self.debug)
        print_model_info(self.critic, model_name="Critic", print_amp=self.use_amp, debug=self.debug)

        # Pre-training parameters
        self.use_pretraining = use_pretraining
        self.pretraining_fraction = pretraining_fraction
        self.pretraining_sr_weight = pretraining_sr_weight
        self.pretraining_rp_weight = pretraining_rp_weight
        self.pretraining_transition_steps = pretraining_transition_steps

        self.training_steps = 0
        self.training_step_offset = training_step_offset
        self.total_episodes = 0
        self.total_episodes_offset = 0
        self.pretraining_completed = False
        self.in_transition_phase = False
        self.base_sr_weight = sr_weight
        self.base_rp_weight = rp_weight

        # Store total episode target for pretraining calculations
        self.total_episode_target = total_episode_target

        # If total episodes target is set, log the pretraining plan
        if self.use_pretraining and self.total_episode_target:
            pretraining_end = int(self.total_episode_target * self.pretraining_fraction)
            if self.debug:
                print(f"[DEBUG] Pretraining will run for {pretraining_end} episodes")
                print(f"[DEBUG] Transition phase will last {self.pretraining_transition_steps} steps")

        # Initialize intrinsic reward generator for pre-training if enabled
        self.use_intrinsic_rewards = use_intrinsic_rewards and use_pretraining
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.intrinsic_reward_generator = None
        
        if self.use_intrinsic_rewards:
            obs_dim = getattr(actor, 'obs_shape', action_dim * 2)  # Use action_dim * 2 as fallback
            self.intrinsic_reward_generator = create_intrinsic_reward_generator(
                input_dim=obs_dim,
                action_dim=action_dim,
                device=device,
                curiosity_weight=curiosity_weight,
                rnd_weight=rnd_weight
            )
            if self.debug:
                print(f"[DEBUG] Initialized intrinsic reward generator with scale {intrinsic_reward_scale}")

    def _true_training_steps(self):
        """Get the true training step count (including offset)"""
        return self.training_steps + self.training_step_offset
    
    def _log_to_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Centralized wandb logging with step validation and metric organization"""
        if not self.use_wandb or wandb.run is None:
            return
        
        # Get current step if not explicitly provided
        current_step = step if step is not None else self._true_training_steps()
        
        # Skip logging if we've already logged this step
        if hasattr(self, '_last_wandb_step') and current_step <= self._last_wandb_step:
            if self.debug:
                print(f"[STEP DEBUG] Skipping wandb log for step {current_step} (≤ {self._last_wandb_step})")
            return
        
        # Organize metrics into logical groups
        grouped_metrics = {
            'algorithm': {},
            'curriculum': {},
            'auxiliary': {},
            'system': {}
        }
        
        # --- Algorithm metrics ---
        algorithm_metrics = grouped_metrics['algorithm']
        
        # Add algorithm type to group as prefix
        alg_prefix = self.algorithm_type.upper()
        
        # Define algorithm-specific metrics to include
        ppo_metrics = {'actor_loss', 'critic_loss', 'entropy_loss', 'total_loss', 
                      'clip_fraction', 'explained_variance', 'kl_divergence', 
                      'mean_advantage', 'mean_return'}
        
        stream_metrics = {'actor_loss', 'critic_loss', 'entropy_loss', 'total_loss',
                         'effective_step_size', 'backtracking_count', 'mean_return',
                         'td_error_mean', 'td_error_max', 'td_error_min'}
        
        # Select appropriate metric set based on algorithm type
        valid_metrics = ppo_metrics if self.algorithm_type == "ppo" else stream_metrics
        
        # Add algorithm metrics based on selected set
        algo_metrics = self.algorithm.get_metrics() if hasattr(self.algorithm, 'get_metrics') else metrics
        filtered_algo_metrics = {k: v for k, v in algo_metrics.items() if k in valid_metrics}
        
        # Add filtered metrics with algorithm prefix
        for key, value in filtered_algo_metrics.items():
            algorithm_metrics[f"{alg_prefix}/{key}"] = value
        
        # Add common metrics for both algorithms
        algorithm_metrics[f"{alg_prefix}/entropy_coefficient"] = self.entropy_coef
        algorithm_metrics[f"{alg_prefix}/actor_learning_rate"] = getattr(self.algorithm, "lr_actor", self.lr_actor)
        algorithm_metrics[f"{alg_prefix}/critic_learning_rate"] = getattr(self.algorithm, "lr_critic", self.lr_critic)
        
        # --- Auxiliary metrics ---
        if self.use_auxiliary_tasks:
            auxiliary_metrics = grouped_metrics['auxiliary']
            
            # Always include auxiliary metrics in the log, even if they're 0
            # This ensures consistent logging and makes debugging easier
            
            # Check if we have the metrics coming from trainer.update() update or direct aux manager
            sr_loss = metrics.get("sr_loss", 0)
            rp_loss = metrics.get("rp_loss", 0)
            aux_loss = metrics.get("aux_loss", 0)
            
            # Debug info for auxiliary metrics
            if self.debug and (sr_loss > 0 or rp_loss > 0):
                print(f"[WANDB DEBUG] Logging auxiliary metrics - SR: {sr_loss:.6f}, RP: {rp_loss:.6f}")
                
            # If aux_loss is not provided but sr_loss and rp_loss are, calculate it
            if aux_loss == 0 and (sr_loss > 0 or rp_loss > 0):
                aux_loss = sr_loss + rp_loss
            
            # Get weights directly from aux_task_manager if available
            sr_weight = getattr(self.aux_task_manager, "sr_weight", 0) if hasattr(self, 'aux_task_manager') else 0
            rp_weight = getattr(self.aux_task_manager, "rp_weight", 0) if hasattr(self.aux_task_manager) else 0
            
            # Log the unweighted losses (more meaningful values for trends)
            auxiliary_metrics["AUX/state_representation_loss"] = sr_loss
            auxiliary_metrics["AUX/reward_prediction_loss"] = rp_loss
            auxiliary_metrics["AUX/total_loss"] = aux_loss
            
            # Also log the weights
            auxiliary_metrics["AUX/sr_weight"] = sr_weight
            auxiliary_metrics["AUX/rp_weight"] = rp_weight
            
            # Log actual last values from auxiliary manager if available
            if hasattr(self.aux_task_manager):
                if hasattr(self.aux_task_manager, 'last_sr_loss') and self.aux_task_manager.last_sr_loss > 0:
                    sr_value = self.aux_task_manager.last_sr_loss
                    auxiliary_metrics["AUX/state_representation_loss"] = sr_value
                    
                if hasattr(self.aux_task_manager, 'last_rp_loss') and self.aux_task_manager.last_rp_loss > 0:
                    rp_value = self.aux_task_manager.last_rp_loss
                    auxiliary_metrics["AUX/reward_prediction_loss"] = rp_value
                    
                # Recalculate total if we got updated values
                if auxiliary_metrics["AUX/state_representation_loss"] > 0 or auxiliary_metrics["AUX/reward_prediction_loss"] > 0:
                    auxiliary_metrics["AUX/total_loss"] = auxiliary_metrics["AUX/state_representation_loss"] + auxiliary_metrics["AUX/reward_prediction_loss"]
        
        # --- Curriculum metrics ---
        if hasattr(self, 'curriculum_manager') and self.curriculum_manager is not None:
            curriculum_metrics = grouped_metrics['curriculum']
            curriculum_stats = self.curriculum_manager.get_curriculum_stats()
            
            curriculum_metrics["CURR/difficulty"] = curriculum_stats.get("difficulty_level", 0)
            curriculum_metrics["CURR/stage"] = curriculum_stats.get("current_stage_index", 0)
            curriculum_metrics["CURR/stage_name"] = curriculum_stats.get("current_stage_name", "")
            curriculum_metrics["CURR/success_rate"] = curriculum_stats.get("success_rate", 0)
            
            # Add detailed stage-specific metrics if available
            for key, value in curriculum_stats.get("current_stage_stats", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    curriculum_metrics[f"CURR/stage_stats/{key}"] = value
        
        # --- System metrics ---
        system_metrics = grouped_metrics['system']
        system_metrics["SYS/training_step"] = current_step
        system_metrics["SYS/total_episodes"] = self.total_episodes + self.total_episodes_offset
        system_metrics["SYS/update_time"] = metrics.get("update_time", 0)
        
        # Pre-training indicators
        if self.use_pretraining:
            system_metrics["SYS/pretraining_active"] = 1 if not self.pretraining_completed else 0
            system_metrics["SYS/transition_phase"] = 1 if self.in_transition_phase else 0
            
            # If in transition phase, log progress
            if self.in_transition_phase:
                progress = min(1.0, (self.training_steps - self.transition_start_step) / 
                              self.pretraining_transition_steps)
                system_metrics["SYS/transition_progress"] = progress
        
        # --- TD Error specific metrics ---
        if self.algorithm_type == "streamac":
            td_metrics = {
                "TD/mean": metrics.get("td_error_mean", 0.0),
                "TD/max": metrics.get("td_error_max", 0.0),
                "TD/min": metrics.get("td_error_min", 0.0)
            }
            grouped_metrics["td_errors"] = td_metrics
            
            # Also create a histogram of TD errors if available
            if hasattr(self.algorithm, 'td_error_buffer') and len(self.algorithm.td_error_buffer) > 0:
                # Convert deque to list to avoid serialization issues
                td_errors = list(self.algorithm.td_error_buffer)
                # Only create histogram if we have enough data points
                if len(td_errors) >= 10:
                    grouped_metrics["td_errors"]["TD/distribution"] = wandb.Histogram(np.array(td_errors))
        
        # Flatten metrics for logging
        flat_metrics = {}
        for group, group_metrics in grouped_metrics.items():
            for key, value in group_metrics.items():
                flat_metrics[key] = value
        
        # Add extra debugging information
        if self.debug:
            flat_metrics["_DEBUG/step_source"] = "explicit" if step is not None else "calculated"
            flat_metrics["_DEBUG/logging_timestamp"] = time.time()
        
        if self.debug:
            print(f"[STEP DEBUG] About to log to wandb with step={current_step}, algorithm={self.algorithm_type}")
            
        try:
            # Log to wandb
            wandb.log(flat_metrics, step=current_step)
            
            if self.debug:
                print(f"[STEP DEBUG] Successfully logged to wandb at step {current_step}")
                
            # Remember this step for next time
            self._last_wandb_step = current_step
            
            if self.debug:
                print(f"[STEP DEBUG] Updated _last_wandb_step to {self._last_wandb_step}")
                
        except Exception as e:
            if self.debug:
                print(f"[STEP DEBUG] Error logging to wandb: {e}")
                import traceback
                traceback.print_exc()

    def store_experience(self, state, action, log_prob, reward, value, done, env_id=0):
        """Store experience in the buffer with environment ID."""
        # Get whether we're in test mode
        test_mode = getattr(self, 'test_mode', False)
        
        # Initialize extrinsic reward normalizer if it doesn't exist
        if not hasattr(self, 'extrinsic_reward_normalizer'):
            from intrinsic_rewards import ExtrinsicRewardNormalizer
            self.extrinsic_reward_normalizer = ExtrinsicRewardNormalizer()
            if self.debug:
                print("[DEBUG] Initialized extrinsic reward normalizer for adaptive intrinsic scaling")
        
        # Add intrinsic rewards if enabled (using adaptive scaling throughout training)
        if self.use_intrinsic_rewards:
            # Convert inputs to tensor format for intrinsic reward computation
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(state).to(self.device)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
            else:
                state_tensor = state.to(self.device)
                
            if not isinstance(action, torch.Tensor):
                if self.action_space_type == "discrete":
                    action_tensor = torch.LongTensor([action]).to(self.device)
                else:
                    action_tensor = torch.FloatTensor(action).to(self.device)
                    if action_tensor.dim() == 1:
                        action_tensor = action_tensor.unsqueeze(0)
            else:
                action_tensor = action.to(self.device)
                
            # Get next state if available in buffer
            next_state = None
            if hasattr(self.algorithm, 'experience_buffer') and len(self.algorithm.experience_buffer) > 0:
                next_state = self.algorithm.experience_buffer[-1][0]
                next_state = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
                
                # Compute intrinsic reward if we have a next state
                if next_state is not None:
                    intrinsic_reward = self.intrinsic_reward_generator.compute_reward(
                        state_tensor, action_tensor, next_state
                    )
                    
                    # Extract extrinsic reward value for normalization
                    extrinsic_reward = reward.item() if hasattr(reward, 'item') else reward
                    
                    # Normalize extrinsic reward using running statistics
                    normalized_reward = self.extrinsic_reward_normalizer.normalize(extrinsic_reward)
                    
                    # Calculate adaptive scale using sigmoid function:
                    # - Higher intrinsic reward when extrinsic reward is low
                    # - Lower intrinsic reward when extrinsic reward is high
                    sigmoid_factor = 1.0 / (1.0 + np.exp(2.0 * normalized_reward))
                    adaptive_scale = self.intrinsic_reward_scale * sigmoid_factor
                    
                    # Ensure minimum scale to maintain some exploration
                    adaptive_scale = max(0.1 * self.intrinsic_reward_scale, adaptive_scale)
                    
                    # Add scale to metrics for debugging if in debug mode
                    if self.debug and hasattr(self, 'metrics'):
                        self.metrics['intrinsic_scale'] = adaptive_scale
                        self.metrics['normalized_extrinsic_reward'] = normalized_reward
                    
                    # Add scaled intrinsic reward to extrinsic reward
                    if isinstance(reward, torch.Tensor):
                        reward = reward + adaptive_scale * intrinsic_reward.to(reward.device)
                    else:
                        reward = reward + adaptive_scale * intrinsic_reward.cpu().numpy()

        # Store the experience using the algorithm - only update if not in test mode
        if not test_mode:
            if self.algorithm_type == "streamac":
                # For StreamAC, check if an update was performed and pass env_id
                if self.debug:
                    print(f"[STEP DEBUG] Trainer.store_experience calling StreamAC.store_experience, current step: {self._true_training_steps()}")
                    
                metrics, did_update = self.algorithm.store_experience(state, action, log_prob, reward, value, done, env_id=env_id)
                
                # If StreamAC performed an update, log to wandb immediately
                if did_update and self.use_wandb:
                    try:
                        # Increment training steps counter BEFORE logging
                        self.training_steps += 1
                        
                        # Get a unique step value for this update (avoid duplicate steps)
                        unique_step = self._true_training_steps()
                        
                        if self.debug:
                            print(f"[STEP DEBUG] StreamAC performed update, incrementing step to {self._true_training_steps()}")
                            print(f"[STEP DEBUG] About to log to wandb with unique_step={unique_step}")
                        
                        # Log metrics using our centralized logging function
                        self._log_to_wandb(metrics, step=unique_step)
                    except Exception as e:
                        if self.debug:
                            print(f"[STEP DEBUG] Error logging to wandb: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                # For other algorithms like PPO, just store normally
                self.algorithm.store_experience(state, action, log_prob, reward, value, done)
        else:
            # In test mode, just store without updating (for StreamAC)
            if hasattr(self.algorithm, 'experience_buffer'):
                exp = {
                    'obs': state,
                    'action': action,
                    'log_prob': log_prob,
                    'reward': reward,
                    'value': value,
                    'done': done,
                    'env_id': env_id  # Add env_id to experience
                }
                self.algorithm.experience_buffer.append(exp)
                
                # Also store in environment-specific buffer if it exists
                if hasattr(self.algorithm, 'experience_buffers'):
                    if env_id not in self.algorithm.experience_buffers:
                        self.algorithm.experience_buffers[env_id] = []
                    self.algorithm.experience_buffers[env_id].append(exp)
    
        # Update auxiliary task models if enabled
        if self.use_auxiliary_tasks and hasattr(self, 'aux_task_manager'):
            # Extract features if the actor supports it
            features = None
            if hasattr(self.actor, 'extract_features'):
                with torch.no_grad():
                    features = self.actor.extract_features(state)
                    
            # Update auxiliary tasks and get metrics
            aux_metrics = self.aux_task_manager.update(
                observations=state,
                rewards=reward,
                features=features
            )
            
            # For StreamAC, log metrics immediately
            if self.algorithm_type == "streamac" and self.use_wandb and aux_metrics.get("sr_loss", 0) > 0:
                try:
                    # Set this to prevent spamming wandb with too many auxiliary updates
                    # Use the current training step to ensure it's a valid integer
                    current_step = self._true_training_steps()
                    
                    # Only log auxiliary metrics if we have new information to log
                    if getattr(self, '_last_aux_log_step', 0) != current_step:
                        if self.debug:
                            print(f"[STEP DEBUG] Logging auxiliary metrics at step {current_step}")
                        self._last_aux_log_step = current_step
                        self._log_to_wandb(aux_metrics, step=current_step)
                except Exception as e:
                    if self.debug:
                        print(f"[STEP DEBUG] Error logging auxiliary metrics: {e}")
                        import traceback
                        traceback.print_exc()

    # Add a new helper method to get a unique wandb step
    def _get_unique_wandb_step(self):
        """Get a unique step value that hasn't been used for wandb logging yet"""
        base_step = self._true_training_steps()
        
        # If we've already used this step, increment by 1 to make it unique
        if hasattr(self, '_last_wandb_step') and base_step <= self._last_wandb_step:
            return self._last_wandb_step + 1
        
        return base_step

    def store_experience_at_idx(self, idx, state, action, log_prob, reward, value, done):
        """Forward to algorithm's store_experience_at_idx method if it exists"""
        # Delegate to algorithm's method if available, otherwise use our own
        if hasattr(self.algorithm, 'store_experience_at_idx'):
            self.algorithm.store_experience_at_idx(idx, obs=state, action=action, log_prob=log_prob, reward=reward, value=value, done=done)
        else:
            # Only PPO uses this method via memory, so use memory directly
            self.memory.store_experience_at_idx(idx, state=state, action=action, log_prob=log_prob, reward=reward, value=value, done=done)

    def get_action(self, state, evaluate=False, return_features=False):
        """Forward to algorithm's get_action method"""
        return self.algorithm.get_action(state, evaluate, return_features)

    def update(self, completed_episode_rewards=None):
        """Update policy based on collected experiences. 
        Different implementations for PPO vs StreamAC."""
        metrics = {}
        
        if not self.algorithm:
            return metrics
        if self.debug:
            print(f"[DEBUG] Updating policy, algorithm_type={self.algorithm_type}")
        # Track update time
        update_start_time = time.time()
        # Check and update pretraining state before policy update
        if self.use_pretraining:
            self._update_pretraining_state()
            self._update_auxiliary_weights() # Also update weights based on state
        # Forward to specific algorithm implementation
        metrics = self.algorithm.update()
        
        # Record update time
        update_time = time.time() - update_start_time
        metrics['update_time'] = update_time
        
        # Ensure explained_variance is correctly calculated and reported for PPO
        if self.algorithm_type == "ppo" and metrics.get('explained_variance', 0.0) == 0.0:
            # If we need to recalculate it, we can look at the metrics data
            if hasattr(self.algorithm, 'memory') and hasattr(self.algorithm.memory, 'values') and hasattr(self.algorithm.memory, 'rewards'):
                try:
                    values = self.algorithm.memory.values[:self.algorithm.memory.pos].detach().cpu().numpy()
                    returns = self.algorithm.memory.rewards[:self.algorithm.memory.pos].detach().cpu().numpy()
                    
                    if len(values) > 1 and len(returns) > 1:
                        # Calculate explained variance
                        var_y = np.var(returns)
                        explained_var = 1 - np.var(returns - values) / (var_y + 1e-8)
                        metrics['explained_variance'] = float(explained_var)
                        
                        if self.debug:
                            print(f"[DEBUG] Recalculated explained_variance = {explained_var:.4f}")
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Error recalculating explained_variance: {e}")
        
        # Update training step counter
        if self.algorithm_type != "streamac":  # For StreamAC, steps are tracked in store_experience
            self.training_steps += 1
            
        # Use completed episode rewards from main.py if provided (for PPO)
        if completed_episode_rewards is not None and len(completed_episode_rewards) > 0:
            metrics['mean_episode_reward'] = np.mean(completed_episode_rewards)
            if self.debug:
                print(f"[DEBUG] Using {len(completed_episode_rewards)} completed episode rewards, mean: {metrics['mean_episode_reward']:.4f}")
        # Otherwise check internal episode rewards (fallback)
        elif hasattr(self, 'episode_rewards') and len(self.episode_rewards) > 0:
            episode_rewards = list(self.episode_rewards)
            # Calculate mean episode reward
            if episode_rewards:
                metrics['mean_episode_reward'] = np.mean(episode_rewards)
            # Clear episode rewards
            self.episode_rewards = []

        # Update auxiliary tasks if enabled
        if self.use_auxiliary_tasks and hasattr(self, 'aux_task_manager') and not self.test_mode:
            # For Stream mode, auxiliary tasks are already updated during store_experience
            # For PPO batch mode, we update here with data from memory
            if self.algorithm_type == "ppo":
                try:
                    # Sample batch of observations and features for auxiliary task update
                    if hasattr(self.algorithm, 'memory') and self.algorithm.memory.size() > 0:
                        # Get indices from memory
                        batch_size = min(64, self.algorithm.memory.size())
                        indices = torch.randperm(self.algorithm.memory.size())[:batch_size]
                        
                        # Extract observations
                        if hasattr(self.algorithm.memory, 'obs'):
                            obs_batch = self.algorithm.memory.obs[indices].to(self.device)
                            rewards_batch = self.algorithm.memory.rewards[indices].to(self.device)
                            
                            # Extract features if actor supports it
                            features = None
                            with torch.no_grad():
                                if hasattr(self.actor, 'get_features'):
                                    features = self.actor.get_features(obs_batch)
                                elif hasattr(self.actor, 'extract_features'):
                                    features = self.actor.extract_features(obs_batch)
                                else:
                                    # Use forward pass with feature extraction if possible
                                    try:
                                        if hasattr(self.actor, 'forward') and 'return_features' in self.actor.forward.__code__.co_varnames:
                                            _, features = self.actor(obs_batch, return_features=True)
                                        else:
                                            # Default approach
                                            features = self.actor(obs_batch)
                                            if isinstance(features, tuple):
                                                features = features[0]
                                    except Exception as e:
                                        if self.debug:
                                            print(f"[DEBUG] Error extracting features: {e}")
                                        # Use observations as features if extraction fails
                                        features = obs_batch
                            
                            # Update auxiliary tasks
                            aux_metrics = self.aux_task_manager.update(
                                observations=obs_batch, 
                                rewards=rewards_batch,
                                features=features
                            )
                            
                            # Always include auxiliary metrics in the main metrics dictionary
                            metrics.update(aux_metrics)
                            metrics['aux_loss'] = aux_metrics.get('sr_loss', 0) + aux_metrics.get('rp_loss', 0)
                            if self.debug:
                                print(f"[DEBUG] Auxiliary task update: SR={aux_metrics.get('sr_loss', 0):.6f}, RP={aux_metrics.get('rp_loss', 0):.6f}")
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Error updating auxiliary tasks: {e}")
                        import traceback
                        traceback.print_exc()

        # Log metrics
        if self.use_wandb:
            try:
                self._log_to_wandb(metrics)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error logging to wandb: {e}")
        
        return metrics

    def reset_auxiliary_tasks(self):
        """Reset auxiliary task history when episodes end"""
        if self.use_auxiliary_tasks:
            self.aux_task_manager.reset()

    def save_models(self, model_path=None, metadata=None):
        """
        Save both actor and critic models to a single file.

        Args:
            model_path: Optional custom path. If None, creates a timestamped file.
                    If a directory is provided, a timestamped file will be created inside it.
            metadata: Optional dictionary containing additional metadata to save with the model

        Returns:
            The complete path where the model was saved
        """
        # Generate timestamp for unique filenames.
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if model_path is None:
            model_path = f"checkpoints/model_{timestamp}.pt"
        elif os.path.isdir(model_path) or model_path.endswith('/') or model_path.endswith('\\'):
            model_path = os.path.join(model_path, f"model_{timestamp}.pt")
        
        # Make sure parent directory exists
        parent_dir = os.path.dirname(model_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Get the original models if using compiled versions
        actor_state = self.actor._orig_mod.state_dict() if hasattr(self.actor, '_orig_mod') else self.actor.state_dict()
        critic_state = self.critic._orig_mod.state_dict() if hasattr(self.critic, '_orig_mod') else self.critic.state_dict()

        # Get auxiliary task models if enabled
        aux_state = {}
        if self.use_auxiliary_tasks:
            # Save SR model
            aux_state['sr_task'] = self.aux_task_manager.sr_task.state_dict()
            # Save RP model
            aux_state['rp_task'] = self.aux_task_manager.rp_task.state_dict()

        # Get the WandB run ID if available
        wandb_run_id = None
        if self.use_wandb:
            try:
                wandb_run_id = wandb.run.id
            except:
                pass

        # Save all models to a single file
        try:
            # Prepare metadata
            if metadata is None:
                metadata = {}
                
            # Add training info to metadata
            metadata.update({
                'algorithm': self.algorithm_type,
                'training_step': self.training_steps + self.training_step_offset,
                'total_episodes': self.total_episodes + self.total_episodes_offset,
                'wandb_run_id': wandb_run_id,
                'timestamp': timestamp
            })
            
            # Save everything to a single file
            torch.save({
                'actor': actor_state,
                'critic': critic_state,
                'aux': aux_state,
                'metadata': metadata
            }, model_path)
            
            if self.debug:
                print(f"[DEBUG] Model saved to {model_path}")
            return model_path
        except Exception as e:
            print(f"Error saving models: {e}")
            return None

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
            # Load actor and critic models
            # Fix compiled state dicts if necessary
            actor_state = fix_compiled_state_dict(checkpoint['actor'])
            critic_state = fix_compiled_state_dict(checkpoint['critic'])
            
            # Load models
            self.actor.load_state_dict(actor_state)
            self.critic.load_state_dict(critic_state)
            
            # Load auxiliary tasks if available and enabled
            if self.use_auxiliary_tasks and 'aux' in checkpoint and checkpoint['aux']:
                aux_state = checkpoint['aux']
                if 'sr_head' in aux_state:
                    self.aux_task_manager.sr_head.load_state_dict(aux_state['sr_head'])
                if 'rp_lstm' in aux_state and 'rp_head' in aux_state:
                    self.aux_task_manager.rp_lstm.load_state_dict(aux_state['rp_lstm'])
                    self.aux_task_manager.rp_head.load_state_dict(aux_state['rp_head'])
                    
            # Extract metadata if available
            if 'metadata' in checkpoint:
                metadata = checkpoint['metadata']
                if 'training_step' in metadata:
                    self.training_step_offset = metadata['training_step']
                if 'total_episodes' in metadata:
                    self.total_episodes_offset = metadata['total_episodes']
                if self.debug:
                    print(f"[DEBUG] Loaded metadata: {metadata}")
                    
            if self.debug:
                print(f"[DEBUG] Successfully loaded model from {model_path}")
        else:
            # Handle legacy format - just load actor
            self.actor.load_state_dict(checkpoint)
            if self.debug:
                print(f"[DEBUG] Loaded legacy model format from {model_path}")

    def register_curriculum_manager(self, curriculum_manager):
        """Register a curriculum manager to synchronize wandb logging"""
        self.curriculum_manager = curriculum_manager
        if curriculum_manager is not None and self.debug:
            print(f"[DEBUG] Registered curriculum manager with {len(curriculum_manager.stages)} stages")

    def _get_pretraining_end_step(self):
        """Calculate the step at which pretraining should end based on available targets"""
        # Check if total_episode_target attribute exists before accessing
        if hasattr(self, 'total_episode_target') and self.total_episode_target:
            return int(self.total_episode_target * self.pretraining_fraction)
            
        # Check if training_time_target attribute exists before accessing
        if hasattr(self, 'training_time_target') and self.training_time_target:
            return int(self.training_time_target * self.pretraining_fraction)
            
        # Default fallback - use a fixed number of steps
        return int(5000 * self.pretraining_fraction)

    def _update_pretraining_state(self):
        """Update pretraining state based on current step"""
        # Check if we should exit pretraining
        pretraining_end_step = self._get_pretraining_end_step()
        current_step = self.total_episodes + self.total_episodes_offset
        
        # If we've reached the end of pretraining and not in transition phase yet
        if current_step >= pretraining_end_step and not self.pretraining_completed and not self.in_transition_phase:
            if self.debug:
                print(f"[DEBUG] Starting pretraining transition phase at episode {current_step}/{pretraining_end_step}")
            self.in_transition_phase = True
            self.transition_start_step = current_step
            
        # If we're in the transition phase and have completed it
        elif self.in_transition_phase:
            transition_progress = current_step - self.transition_start_step
            if transition_progress >= self.pretraining_transition_steps:
                if self.debug:
                    print(f"[DEBUG] Completed pretraining transition phase at episode {current_step}")
                self.pretraining_completed = True
                self.in_transition_phase = False

    def _update_auxiliary_weights(self):
        """Update auxiliary task weights based on pretraining phase"""
        if not self.use_auxiliary_tasks or not self.use_pretraining:
            return
            
        current_sr_weight = self.base_sr_weight
        current_rp_weight = self.base_rp_weight
        
        if not self.pretraining_completed:
            # In pretraining phase, use higher weights
            current_sr_weight = self.pretraining_sr_weight
            current_rp_weight = self.pretraining_rp_weight
        elif self.in_transition_phase:
            # In transition phase, gradually reduce weights
            progress = min(1.0, (self.training_steps - self.transition_start_step) / 
                          self.pretraining_transition_steps)
            current_sr_weight = self.pretraining_sr_weight + progress * (self.base_sr_weight - self.pretraining_sr_weight)
            current_rp_weight = self.pretraining_rp_weight + progress * (self.base_rp_weight - self.pretraining_rp_weight)
            
        # Update the weights in the manager
        self.aux_task_manager.sr_weight = current_sr_weight
        self.aux_task_manager.rp_weight = current_rp_weight

        # Also update entropy coefficient based on pretraining status
        if not self.pretraining_completed:
            # Higher entropy during pretraining to encourage exploration
            self.entropy_coef = self.base_entropy_coef * self.pretraining_entropy_scale
        elif self.in_transition_phase:
            progress = min(1.0, (self.training_steps - self.transition_start_step) / 
                          self.pretraining_transition_steps)
            self.entropy_coef = self.base_entropy_coef * self.pretraining_entropy_scale + \
                               progress * (self.base_entropy_coef - self.base_entropy_coef * self.pretraining_entropy_scale)
        else:
            # Use base entropy with normal decay during regular training
            self.entropy_coef = max(self.min_entropy_coef, 
                                   self.base_entropy_coef * (self.entropy_coef_decay ** self.training_steps))

        # Log to wandb
        if self.use_wandb and self.training_steps % 10 == 0:
            wandb.log({
                "sr_weight": current_sr_weight,
                "rp_weight": current_rp_weight,
                "entropy_coef": self.entropy_coef
            }, step=self._get_wandb_step())

    def _get_wandb_step(self):
        """Get the correct step for wandb logging"""
        return self.training_steps + self.training_step_offset

    def set_total_episode_target(self, total_episodes):
        """Set the target for total training episodes, used for pretraining calculations."""
        self.total_episode_target = total_episodes
        
        if self.use_pretraining and self.total_episode_target and self.debug:
            pretraining_end = self._get_pretraining_end_step()
            print(f"[DEBUG] Set total episode target to {total_episodes}")
            print(f"[DEBUG] Pretraining will end at episode {pretraining_end}")
        
        return self._get_pretraining_end_step()

    def update_intrinsic_models(self, state, action, next_state, done=False):
        """Update intrinsic reward models with new experience"""
        if not self.use_intrinsic_rewards or self.intrinsic_reward_generator is None:
            return {}
            
        # Convert inputs to tensor format for intrinsic reward update
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).to(self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
        else:
            state_tensor = state.to(self.device)
            
        if not isinstance(action, torch.Tensor):
            if self.action_space_type == "discrete":
                action_tensor = torch.LongTensor([action]).to(self.device)
            else:
                action_tensor = torch.FloatTensor(action).to(self.device)
                if action_tensor.dim() == 1:
                    action_tensor = action_tensor.unsqueeze(0)
        else:
            action_tensor = action.to(self.device)
            
        if not isinstance(next_state, torch.Tensor):
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            if next_state_tensor.dim() == 1:
                next_state_tensor = next_state_tensor.unsqueeze(0)
        else:
            next_state_tensor = next_state.to(self.device)
            
        # Update intrinsic reward models
        return self.intrinsic_reward_generator.update(state_tensor, action_tensor, next_state_tensor, done)