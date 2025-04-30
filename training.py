from collections import deque
from torch.distributions import Categorical, Normal
# Import GradScaler and autocast
from torch.amp import autocast, GradScaler
from model_architectures import (
    fix_compiled_state_dict,
    load_partial_state_dict,
    print_model_info,
    fix_rsnorm_cuda_graphs
)
from typing import Union, Tuple, Optional, Dict, Any, Deque
from auxiliary import AuxiliaryTaskManager
from intrinsic_rewards import create_intrinsic_reward_generator, IntrinsicRewardEnsemble
from algorithms import BaseAlgorithm, PPOAlgorithm, StreamACAlgorithm
from algorithms.sac import SACAlgorithm
import time
import os
import random
import collections
import copy
import wandb
import math
import inspect
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


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


class SkillRatingTracker:
    """
    Tracks agent skill rating using an Elo-like system.
    Allows comparison against baselines or historical performance.
    """
    def __init__(self, initial_rating=1500, rating_inc=32, modes=None):
        """
        Initialize a skill rating tracker with Elo-like rating system
        
        Args:
            initial_rating: Starting rating value (default: 1500)
            rating_inc: Rating change factor (K-factor in Elo) (default: 32)
            modes: Dict of game modes to track separately (default: {'default': initial_rating})
        """
        self.rating_inc = rating_inc
        self.modes = modes or {'default': initial_rating}
        self.history = {mode: [rating] for mode, rating in self.modes.items()}
        self.update_count = {mode: 0 for mode in self.modes}
        
    def update_rating(self, opponent_rating, result, mode='default'):
        """
        Update the rating based on the result against an opponent.
        
        Args:
            opponent_rating: The rating of the opponent (or baseline)
            result: 1 for win, 0 for loss, 0.5 for draw
            mode: The game mode to update
        
        Returns:
            The updated rating
        """
        if mode not in self.modes:
            self.modes[mode] = self.modes['default']
            self.history[mode] = [self.modes[mode]]
            self.update_count[mode] = 0
            
        # Calculate the expected score (probability of winning)
        exp_delta = (opponent_rating - self.modes[mode]) / 400
        expected = 1 / (pow(10, exp_delta) + 1)
        
        # Update rating
        self.modes[mode] += self.rating_inc * (result - expected)
        
        # Store in history
        self.history[mode].append(self.modes[mode])
        self.update_count[mode] += 1
        
        return self.modes[mode]

    def get_current_rating(self, mode='default'):
        """Get the current rating for a specific mode"""
        return self.modes.get(mode, self.modes['default'])
    
    def get_rating_history(self, mode='default'):
        """Get the history of ratings for a specific mode"""
        return self.history.get(mode, self.history['default'])


class Trainer:
    """
    Base trainer class that manages the training of reinforcement learning algorithms.
    Supports PPO, StreamAC, and SAC algorithms.
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
        use_weight_clipping: bool = True,
        weight_clip_kappa: float = 1.0,
        adaptive_kappa: bool = False,
        kappa_update_freq: int = 10,
        kappa_update_rate: float = 0.01,
        target_clip_fraction: float = 0.05,
        min_kappa: float = 0.1,
        max_kappa: float = 10.0,
        batch_size: int = 128,
        use_wandb: bool = False,
        debug: bool = False,
        use_compile: bool = True,
        use_amp: bool = False,
        use_auxiliary_tasks: bool = True,
        sr_weight: float = 1.0,
        rp_weight: float = 1.0,
        aux_amp: bool = False, # Keep aux_amp for specific control over aux tasks
        use_pretraining: bool = False,
        pretraining_fraction: float = 0.1,
        pretraining_sr_weight: float = 10.0,
        pretraining_rp_weight: float = 5.0,
        # Removed pretraining_transition_steps
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
        # SAC specific parameters
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha_tuning: bool = True,
        target_entropy: float = -1.0,
        buffer_size: int = 1000000,
        warmup_steps: int = 1000,
        updates_per_step: int = 1,
        # SimbaV2 Reward Scaling parameters (REMOVED - Handled in Algorithm)
        # use_reward_scaling: bool = True,
        # reward_scaling_G_max: float = 10.0,
        # reward_scaling_eps: float = 1e-5,
        # Gaussian critic parameters (for SimbaV2)
        variance_loss_coefficient: float = 0.01,
        use_uncertainty_weight: bool = True,
        uncertainty_weight_type: str = "variance",
        uncertainty_weight_temp: float = 1.0,
    ):
        self.use_wandb = use_wandb
        self.debug = debug
        self.test_mode = False  # Initialize test_mode attribute to False by default

        # IMPORTANT: Set use_intrinsic_rewards early to avoid attribute access errors
        self.use_intrinsic_rewards = use_intrinsic_rewards # Keep base flag
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.intrinsic_reward_generator = None

        # Figure out which device (CPU, CUDA, MPS) to use
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Use Automatic Mixed Precision (AMP) if requested and on CUDA
        self.use_amp = "cuda" in str(device) and use_amp
        # Use separate flag for auxiliary task AMP
        self.aux_amp = "cuda" in str(device) and aux_amp
        # GradScaler is now initialized within the algorithm (e.g., PPOAlgorithm)

        self.action_space_type = action_space_type
        self.action_dim = action_dim
        self.action_bounds = action_bounds

        self.actor = actor.to(self.device)
        # Critic might be the same instance as actor
        self.critic = critic.to(self.device)
        self.shared_model = (self.actor is self.critic)

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
        if not self.shared_model:
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

        # Initialize auxiliary tasks if enabled BEFORE creating the algorithm
        self.use_auxiliary_tasks = use_auxiliary_tasks
        self.aux_task_manager = None # Initialize as None
        if self.use_auxiliary_tasks:
            # Get observation dimension from model if available or derive from action_dim
            obs_dim = getattr(actor, 'obs_shape', action_dim * 2)  # Use action_dim * 2 as fallback

            self.aux_task_manager = AuxiliaryTaskManager(
                actor=self.actor, # Pass the actor model (or shared model)
                obs_dim=obs_dim,
                sr_weight=sr_weight,
                rp_weight=rp_weight,
                device=self.device,
                use_amp=self.aux_amp, # Use specific aux_amp setting
                update_frequency=1, # Frequency handled within PPO/StreamAC logic now
                learning_mode="batch" if algorithm_type == "ppo" else "stream", # Set mode
                debug=debug,
                batch_size=batch_size,
                internal_aux_batch_size=batch_size
            )
            # Enable debug mode for auxiliary tasks
            self.aux_task_manager.debug = self.debug
            if self.debug:
                print("[DEBUG Trainer] Auxiliary Task Manager initialized.")


        # Create the learning algorithm based on type
        self.algorithm_type = algorithm_type.lower()

        if self.algorithm_type == "ppo":
            self.algorithm = PPOAlgorithm(
                actor=self.actor,
                critic=self.critic, # Pass potentially shared instance
                # Pass the aux_task_manager to PPO
                aux_task_manager=self.aux_task_manager if self.use_auxiliary_tasks else None,
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
                use_amp=self.use_amp, # Pass the trainer's use_amp flag
                debug=debug,
                use_wandb=use_wandb,
                use_weight_clipping=use_weight_clipping,
                weight_clip_kappa=weight_clip_kappa,
                adaptive_kappa=adaptive_kappa,
                kappa_update_freq=kappa_update_freq,
                kappa_update_rate=kappa_update_rate,
                target_clip_fraction=target_clip_fraction,
                min_kappa=min_kappa,
                max_kappa=max_kappa,
            )
            # For compatibility with existing code, keep a reference to the memory
            # self.memory = self.algorithm.memory # No longer needed, access via self.algorithm.memory
        elif self.algorithm_type == "streamac":
            # Create StreamAC algorithm
            algorithm = StreamACAlgorithm(
                actor=self.actor,
                critic=self.critic, # Pass potentially shared instance
                action_space_type=action_space_type,
                action_dim=action_dim,
                action_bounds=action_bounds,
                device=self.device,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                critic_coef=critic_coef,
                entropy_coef=self.entropy_coef,
                max_grad_norm=max_grad_norm,
                adaptive_learning_rate=adaptive_learning_rate,
                target_step_size=target_step_size,
                backtracking_patience=backtracking_patience,
                backtracking_zeta=backtracking_zeta,
                min_lr_factor=min_lr_factor,
                max_lr_factor=max_lr_factor,
                use_obgd=use_obgd,
                buffer_size=stream_buffer_size,
                use_sparse_init=use_sparse_init,
                update_freq=update_freq,
                use_amp=self.use_amp, # Pass AMP flag
                debug=self.debug
            )
            # Set reference to trainer so StreamAC can update auxiliary tasks
            algorithm.trainer = self
            self.algorithm = algorithm
        elif self.algorithm_type == "sac":
            # Create a second critic network for SAC's twin Q-function
            # If using shared model, SAC needs modification or cannot use shared.
            # For now, assume SAC requires separate critics.
            if self.shared_model:
                raise ValueError("SAC algorithm currently does not support shared actor-critic models.")
            critic2 = copy.deepcopy(critic)

            # Create SAC algorithm
            algorithm = SACAlgorithm(
                actor=self.actor,
                critic1=self.critic,  # First Q-network
                critic2=critic2,      # Second Q-network
                action_space_type=action_space_type,
                action_dim=action_dim,
                action_bounds=action_bounds,
                device=self.device,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                alpha=alpha,
                auto_alpha_tuning=auto_alpha_tuning,
                target_entropy=target_entropy,
                buffer_size=buffer_size,
                batch_size=batch_size,
                warmup_steps=warmup_steps,
                update_freq=update_freq,
                updates_per_step=updates_per_step,
                max_grad_norm=max_grad_norm,
                use_amp=self.use_amp, # Pass AMP flag
                use_wandb=use_wandb,
                debug=self.debug
            )
            self.algorithm = algorithm
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}. Use 'ppo', 'streamac', or 'sac'.")

        # Track metrics using deques with a max length (e.g., 1000)
        history_len = 1000
        self.actor_losses = collections.deque(maxlen=history_len)
        self.critic_losses = collections.deque(maxlen=history_len)
        self.entropy_losses = collections.deque(maxlen=history_len)
        self.total_losses = collections.deque(maxlen=history_len)
        self.aux_sr_losses = collections.deque(maxlen=history_len)
        self.aux_rp_losses = collections.deque(maxlen=history_len)
        self.episode_rewards = collections.deque(maxlen=history_len)  # Track episode rewards
        
        # Environment statistics tracking
        # Environment statistics tracking - accumulate values between updates
        self.env_stats_buffer = {}  # Dictionary to accumulate environment statistics between updates
        self.env_stats_counts = {}  # Counter for each statistic to calculate averages
        self.env_stats_metrics = {}  # Averaged metrics to report in WandB
        
        # Initialize skill rating tracker
        self.skill_tracker = SkillRatingTracker(initial_rating=1500, rating_inc=32)
        self.baseline_rating = 1500  # Static baseline rating
        self.skill_update_frequency = 5  # Update skill rating every N episodes
        self.skill_rating_window = 100  # Number of episodes to consider for performance baseline

        if self.debug:
            print(f"[DEBUG] Initialized {self.algorithm_type.upper()} algorithm on {self.device}")
            print(f"[DEBUG] Actor: {actor.__class__.__name__}, Critic: {critic.__class__.__name__} (Shared: {self.shared_model})")

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
                "use_amp": use_amp,
                "aux_amp": aux_amp # Log aux_amp setting
            }, allow_val_change=True)

        # If requested and available, compile the models for performance
        self.use_compile = use_compile and hasattr(torch, 'compile')
        if self.use_compile:
            try:
                # Apply RSNorm fix before compiling if necessary
                try:
                    from model_architectures.utils import RSNorm
                    # Check actor (covers shared case)
                    if any(isinstance(m, RSNorm) for m in self.actor.modules()):
                        print("Applying RSNorm CUDA graphs fix before compiling...")
                        self.actor = fix_rsnorm_cuda_graphs(self.actor)
                        # If not shared, fix critic separately
                        if not self.shared_model and any(isinstance(m, RSNorm) for m in self.critic.modules()):
                            self.critic = fix_rsnorm_cuda_graphs(self.critic)
                        # If shared, critic is already fixed via actor
                        elif self.shared_model:
                            self.critic = self.actor
                except ImportError:
                    print("Warning: RSNorm not found, skipping CUDA graphs fix.")


                print("Compiling models with torch.compile...")
                # Compile actor (covers shared case)
                self.actor = torch.compile(self.actor)
                # If not shared, compile critic separately
                if not self.shared_model:
                    self.critic = torch.compile(self.critic)
                # If shared, critic is already compiled via actor
                elif self.shared_model:
                    self.critic = self.actor

                # Compile auxiliary models if they exist
                if self.aux_task_manager:
                    if hasattr(self.aux_task_manager, 'sr_task'):
                        self.aux_task_manager.sr_task = torch.compile(self.aux_task_manager.sr_task)
                    if hasattr(self.aux_task_manager, 'rp_task'):
                        self.aux_task_manager.rp_task = torch.compile(self.aux_task_manager.rp_task)

                print("Models compiled successfully.")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Proceeding without compilation.")
                self.use_compile = False # Disable compile if it fails

        # Print model info
        print_model_info(self.actor, model_name="Actor/Shared Model", print_amp=self.use_amp, debug=self.debug)
        if not self.shared_model:
            print_model_info(self.critic, model_name="Critic", print_amp=self.use_amp, debug=self.debug)
        if self.aux_task_manager:
            if hasattr(self.aux_task_manager, 'sr_task'):
                print_model_info(self.aux_task_manager.sr_task, model_name="SR Task", print_amp=self.aux_amp, debug=self.debug) # Use aux_amp
            if hasattr(self.aux_task_manager, 'rp_task'):
                print_model_info(self.aux_task_manager.rp_task, model_name="RP Task", print_amp=self.aux_amp, debug=self.debug) # Use aux_amp


        # Pre-training parameters
        self.use_pretraining = use_pretraining
        self.pretraining_fraction = pretraining_fraction
        self.pretraining_sr_weight = pretraining_sr_weight
        self.pretraining_rp_weight = pretraining_rp_weight
        # Removed pretraining_transition_steps

        self.training_steps = 0
        self.training_step_offset = training_step_offset
        self.total_episodes = 0
        self.total_episodes_offset = 0
        self.total_env_steps = 0 # Initialize total environment steps counter
        self.pretraining_completed = False
        # Removed in_transition_phase and transition_start_step
        self.base_sr_weight = sr_weight
        self.base_rp_weight = rp_weight

        # Store total episode target for pretraining calculations
        self.total_episode_target = total_episode_target

        # If total episodes target is set, log the pretraining plan
        if self.use_pretraining and self.total_episode_target:
            pretraining_end = int(self.total_episode_target * self.pretraining_fraction)
            if self.debug:
                print(f"[DEBUG] Pretraining will run for {pretraining_end} episodes")
                # Removed transition phase log

        # Initialize intrinsic reward generator if enabled
        # This depends on the flag from args, not the pretraining state itself
        if self.use_intrinsic_rewards:
            obs_dim = getattr(actor, 'obs_shape', action_dim * 2)  # Use action_dim * 2 as fallback
            self.intrinsic_reward_generator = create_intrinsic_reward_generator(
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=device,
                curiosity_weight=curiosity_weight,
                rnd_weight=rnd_weight,
                use_amp=self.use_amp # Pass main use_amp flag
            )
            if self.debug:
                print(f"[DEBUG] Initialized intrinsic reward generator with scale {intrinsic_reward_scale}")

        # SimbaV2 Reward Scaling Initialization (REMOVED - Handled in Algorithm)
        # self.use_reward_scaling = use_reward_scaling
        # self.reward_scaling_G_max = reward_scaling_G_max
        # self.reward_scaling_eps = reward_scaling_eps
        # Use dictionaries to store per-environment stats for vectorized envs
        # self.running_G: Dict[int, float] = {} # Tracks G_t per env (Eq 17)
        # self.running_G_var: Dict[int, float] = {} # Tracks variance of G_t per env
        # self.running_G_count: Dict[int, float] = {} # Tracks count for variance calculation per env
        # self.running_G_max: Dict[int, float] = {} # Tracks max(G_t) per env (Eq 18)

    def _true_training_steps(self):
        """Get the true training step count (including offset)"""
        return self.training_steps + self.training_step_offset

    def _get_pretraining_end_step(self) -> int:
        """Calculate the step/episode number when pretraining should end."""
        if not self.use_pretraining:
            return 0

        if hasattr(self, 'total_episode_target') and self.total_episode_target:
            # Calculate based on fraction of total episodes
            pretraining_episodes = int(self.total_episode_target * self.pretraining_fraction)
            # Convert episodes to approximate steps (assuming each episode contributes ~200 steps)
            return pretraining_episodes * 200
        else:
            # Default to a fixed number of steps if no episode target is set
            return 100000  # Default to 100k steps for pretraining

    def _update_auxiliary_weights(self):
        """Update auxiliary task weights based on pretraining state."""
        if not self.use_auxiliary_tasks or not hasattr(self, 'aux_task_manager'):
            return

        # Removed transition phase logic
        if not self.use_pretraining or self.pretraining_completed:
            # Use base weights if pretraining is disabled or completed
            self.aux_task_manager.sr_weight = self.base_sr_weight
            self.aux_task_manager.rp_weight = self.base_rp_weight
            self.entropy_coef = self.base_entropy_coef # Also reset entropy
        else: # Pretraining is active
            # Use pretraining weights if pretraining is active
            self.aux_task_manager.sr_weight = self.pretraining_sr_weight
            self.aux_task_manager.rp_weight = self.pretraining_rp_weight
            # Use higher entropy during active pretraining
            self.entropy_coef = max(self.min_entropy_coef, self.base_entropy_coef * self.pretraining_entropy_scale)

        # Decay entropy coefficient over time (independent of pretraining state after completion)
        if self.pretraining_completed:
             self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_coef_decay)

        # Log weight changes if debugging
        if self.debug:
            print(f"[DEBUG Aux Weights] SR: {self.aux_task_manager.sr_weight:.4f}, RP: {self.aux_task_manager.rp_weight:.4f}, Entropy: {self.entropy_coef:.6f}")

    def _log_to_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None, total_env_steps: Optional[int] = None) -> None:
        """Centralized wandb logging with step validation and metric organization"""
        if not self.use_wandb or wandb.run is None:
            return

        # Get current training step if not explicitly provided
        current_step = step if step is not None else self._true_training_steps()
        
        # Use environment steps as our primary x-axis for logging
        current_env_steps = total_env_steps if total_env_steps is not None else self.total_env_steps
        
        # Skip logging if we've already logged this environment step
        if hasattr(self, '_last_wandb_env_step') and current_env_steps <= self._last_wandb_env_step:
            if self.debug:
                print(f"[STEP DEBUG] Skipping wandb log for env_step {current_env_steps} (≤ {self._last_wandb_env_step})")
            return

        # Organize metrics into logical groups
        grouped_metrics = {
            'algorithm': {},
            'curriculum': {},
            'auxiliary': {},
            'system': {},
            'environment': {},  # Group for environment statistics
            'rewards': {},      # Group specifically for reward metrics
            'skill': {}         # Group for skill rating metrics
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

        sac_metrics = {'actor_loss', 'critic1_loss', 'critic2_loss', 'alpha_loss', 'alpha',
                       'mean_return', 'mean_q_value', 'mean_entropy'}


        # Select appropriate metric set based on algorithm type
        if self.algorithm_type == "ppo":
            valid_metrics = ppo_metrics
        elif self.algorithm_type == "streamac":
            valid_metrics = stream_metrics
        elif self.algorithm_type == "sac":
            valid_metrics = sac_metrics
        else:
            valid_metrics = set() # Empty set for unknown algorithms


        # Add algorithm metrics based on selected set
        # Use metrics passed to this function, which should contain the latest from algorithm.update()
        # algo_metrics = self.algorithm.get_metrics() if hasattr(self.algorithm, 'get_metrics') else metrics
        algo_metrics = metrics # Use the provided metrics dict
        filtered_algo_metrics = {k: v for k, v in algo_metrics.items() if k in valid_metrics}

        # Add filtered metrics with algorithm prefix
        for key, value in filtered_algo_metrics.items():
            algorithm_metrics[f"{alg_prefix}/{key}"] = value

        # Add common metrics for both algorithms
        algorithm_metrics[f"{alg_prefix}/entropy_coefficient"] = self.entropy_coef
        algorithm_metrics[f"{alg_prefix}/actor_learning_rate"] = getattr(self.algorithm, "lr_actor", self.lr_actor)
        algorithm_metrics[f"{alg_prefix}/critic_learning_rate"] = getattr(self.algorithm, "lr_critic", self.lr_critic)

        # Add buffer size for SAC
        if self.algorithm_type == "sac":
            algorithm_metrics[f"{alg_prefix}/buffer_size"] = len(self.algorithm.memory) if hasattr(self.algorithm, 'memory') else 0


        # --- Environment metrics (AVERAGED) ---
        environment_metrics = grouped_metrics['environment']
        env_stat_count = 0

        # Extract environment statistics that were passed via metrics
        # These should now be averaged values from _compute_avg_env_stats()
        for key, value in metrics.items():
            # Identify environment metrics by their prefix
            if key.startswith('env_'):
                # Strip 'env_' prefix for the display name to keep it clean
                display_key = key[4:] if key.startswith('env_') else key
                # Add to environment metrics group with ENV/ prefix for WandB organization
                environment_metrics[f"ENV/{display_key}"] = value
                env_stat_count += 1

        if self.debug and env_stat_count > 0:
            print(f"[WANDB DEBUG] Adding {env_stat_count} averaged environment metrics to WandB")
            
        # --- Skill metrics ---
        skill_metrics = grouped_metrics['skill']
        skill_stat_count = 0
        
        # Extract skill metrics that were passed via metrics
        for key, value in metrics.items():
            # Process metrics with SKILL/ prefix
            if key.startswith('SKILL/'):
                skill_metrics[key] = value
                skill_stat_count += 1
            
        # Always include current skill rating, even if not updated this step
        if skill_stat_count == 0 and hasattr(self, 'skill_tracker'):
            current_rating = self.skill_tracker.get_current_rating()
            skill_metrics["SKILL/rating"] = current_rating
            skill_stat_count += 1
            
        if self.debug and skill_stat_count > 0:
            print(f"[WANDB DEBUG] Adding {skill_stat_count} skill metrics to WandB")


        # --- Auxiliary metrics ---
        if self.use_auxiliary_tasks:
            auxiliary_metrics = grouped_metrics['auxiliary']

            # Always include auxiliary metrics in the log, even if they're 0
            # Use the SCALAR versions for logging (these should be in the metrics dict from PPO update)
            sr_loss_scalar = metrics.get("sr_loss_scalar", 0)
            rp_loss_scalar = metrics.get("rp_loss_scalar", 0)
            # Calculate total aux loss from scalars
            aux_loss_scalar = sr_loss_scalar + rp_loss_scalar

            # Debug info for auxiliary metrics
            if self.debug and (sr_loss_scalar > 0 or rp_loss_scalar > 0):
                print(f"[WANDB DEBUG] Logging auxiliary metrics - SR: {sr_loss_scalar:.6f}, RP: {rp_loss_scalar:.6f}")

            # Get weights directly from aux_task_manager if available
            sr_weight = getattr(self.aux_task_manager, "sr_weight", 0) if hasattr(self, 'aux_task_manager') else 0
            rp_weight = getattr(self.aux_task_manager, "rp_weight", 0) if hasattr(self, 'aux_task_manager') and hasattr(self.aux_task_manager, 'rp_weight') else 0

            # Log the unweighted SCALAR losses
            auxiliary_metrics["AUX/state_representation_loss"] = sr_loss_scalar
            auxiliary_metrics["AUX/reward_prediction_loss"] = rp_loss_scalar
            auxiliary_metrics["AUX/total_loss"] = aux_loss_scalar

            # Also log the weights
            auxiliary_metrics["AUX/sr_weight"] = sr_weight
            auxiliary_metrics["AUX/rp_weight"] = rp_weight

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

        # --- Reward metrics ---
        reward_metrics = grouped_metrics['rewards']

        # Add per-update reward statistics
        if 'mean_episode_reward' in metrics:
            reward_metrics["REWARD/mean"] = metrics['mean_episode_reward']
        if 'min_episode_reward' in metrics:
            reward_metrics["REWARD/min"] = metrics['min_episode_reward']
        if 'max_episode_reward' in metrics:
            reward_metrics["REWARD/max"] = metrics['max_episode_reward']
        if 'std_episode_reward' in metrics:
            reward_metrics["REWARD/std"] = metrics['std_episode_reward']

        # Calculate and add historical reward statistics if we have enough data
        if hasattr(self, 'episode_rewards') and len(self.episode_rewards) > 0:
            reward_array = np.array(list(self.episode_rewards))
            reward_metrics["REWARD/historical_mean"] = np.mean(reward_array)

            # Add percentiles for better understanding of reward distribution
            if len(reward_array) >= 5:  # Only calculate percentiles if we have enough data
                reward_metrics["REWARD/median"] = np.median(reward_array)
                reward_metrics["REWARD/percentile_25"] = np.percentile(reward_array, 25)
                reward_metrics["REWARD/percentile_75"] = np.percentile(reward_array, 75)

                # Calculate rolling statistics if we have enough data
                if len(reward_array) >= 10:
                    recent_rewards = reward_array[-10:]  # Last 10 episodes
                    reward_metrics["REWARD/recent_mean"] = np.mean(recent_rewards)
                    reward_metrics["REWARD/recent_median"] = np.median(recent_rewards)

                    # Calculate improvement rate (comparing recent to historical)
                    if len(reward_array) > 20:
                        previous_rewards = reward_array[-20:-10]  # 10 episodes before the most recent 10
                        recent_mean = np.mean(recent_rewards)
                        previous_mean = np.mean(previous_rewards)
                        if previous_mean != 0:  # Avoid division by zero
                            improvement = (recent_mean - previous_mean) / abs(previous_mean) * 100
                            reward_metrics["REWARD/improvement_percent"] = improvement

            # Add reward histogram (optional but useful)
            if len(reward_array) >= 20:
                reward_metrics["REWARD/distribution"] = wandb.Histogram(reward_array)

        # --- System metrics ---
        system_metrics = grouped_metrics['system']
        system_metrics["SYS/training_step"] = current_step
        system_metrics["SYS/total_episodes"] = self.total_episodes + self.total_episodes_offset
        system_metrics["SYS/update_time"] = metrics.get("update_time", 0)

        # Use environment steps as primary counter
        log_total_env_steps = current_env_steps  # Use the already determined env steps
        system_metrics["SYS/total_env_steps"] = log_total_env_steps

        # Add steps per second if provided in metrics
        if 'steps_per_second' in metrics:
            system_metrics["SYS/steps_per_second"] = metrics.get('steps_per_second', 0)

        # Pre-training indicators
        if self.use_pretraining:
            system_metrics["SYS/pretraining_active"] = 1 if not self.pretraining_completed else 0
            # Removed transition phase logging

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
            if env_stat_count > 0:
                flat_metrics["_DEBUG/env_stat_count"] = env_stat_count

        if self.debug:
            print(f"[STEP DEBUG] About to log to wandb with step={current_step}, algorithm={self.algorithm_type}")

        try:
            # Log to wandb using environment steps as the x-axis
            wandb.log(flat_metrics, step=current_env_steps)

            if self.debug:
                print(f"[STEP DEBUG] Successfully logged to wandb at env_step {current_env_steps} (training step {current_step})")

            # Remember both steps for next time
            self._last_wandb_step = current_step
            self._last_wandb_env_step = current_env_steps

            if self.debug:
                print(f"[STEP DEBUG] Updated tracking: _last_wandb_env_step={self._last_wandb_env_step}, _last_wandb_step={self._last_wandb_step}")

        except Exception as e:
            if self.debug:
                print(f"[STEP DEBUG] Error logging to wandb: {e}")
                import traceback
                traceback.print_exc()

    def store_initial_batch(self, obs_batch, action_batch, log_prob_batch, value_batch, env_stats_batch=None):
        """
        Store the initial part of experiences (obs, action, log_prob, value) in batch.
        Delegates to the algorithm's implementation.
        Returns the indices where the experiences were stored.
        
        Args:
            obs_batch: Batch of observations
            action_batch: Batch of actions
            log_prob_batch: Batch of log probabilities
            value_batch: Batch of values
            env_stats_batch: Optional batch of environment statistics
        """
        # Accumulate environment statistics if provided
        if env_stats_batch:
            for env_stats in env_stats_batch:
                if env_stats:
                    self.accumulate_env_stats(env_stats)
        
        if hasattr(self.algorithm, 'store_initial_batch'):
            return self.algorithm.store_initial_batch(obs_batch, action_batch, log_prob_batch, value_batch)
        else:
            # Fallback or error for algorithms not supporting batch storage
            if self.debug:
                print(f"[DEBUG] Algorithm {self.algorithm_type} does not support store_initial_batch.")
            # Simulate storing one by one and return indices (inefficient fallback)
            indices = []
            start_pos = self.algorithm.memory.pos if hasattr(self.algorithm, 'memory') else 0
            buffer_size = self.algorithm.memory.buffer_size if hasattr(self.algorithm, 'memory') else 0
            for i in range(len(obs_batch)):
                # Get env_stats for this item if available
                env_stats = env_stats_batch[i] if env_stats_batch and i < len(env_stats_batch) else None
                self.store_experience(obs_batch[i], action_batch[i], log_prob_batch[i], 0, value_batch[i], False, env_stats=env_stats)
                # Calculate index (handle wrap around)
                idx = (start_pos + i) % buffer_size
                indices.append(idx)
            return torch.tensor(indices, dtype=torch.long, device=self.device)


    def update_rewards_dones_batch(self, indices, rewards_batch, dones_batch):
        """
        Update rewards and dones for experiences at given indices in batch.
        Delegates to the algorithm's implementation.
        """
        if hasattr(self.algorithm, 'update_rewards_dones_batch'):
            self.algorithm.update_rewards_dones_batch(indices, rewards_batch, dones_batch)
        else:
            # Fallback or error for algorithms not supporting batch update
            if self.debug:
                print(f"[DEBUG] Algorithm {self.algorithm_type} does not support update_rewards_dones_batch.")
            # Simulate updating one by one (inefficient fallback)
            for i, idx in enumerate(indices):
                self.store_experience_at_idx(idx, reward=rewards_batch[i], done=dones_batch[i])


    def store_experience(self, state, action, log_prob, reward, value, done, env_id=0, env_stats=None):
        """Store experience in the buffer with environment ID."""
        # Get whether we're in test mode
        test_mode = getattr(self, 'test_mode', False)
        
        # Accumulate environment statistics if provided
        if env_stats and not test_mode:
            self.accumulate_env_stats(env_stats)

        # Reward Scaling (SimbaV2) (REMOVED - Handled in Algorithm)
        original_reward = reward # Keep original for potential intrinsic calculation later
        # if self.use_reward_scaling and not test_mode:
             # ... scaling logic removed ...
             # reward = self._scale_reward(env_id, reward_float, variance_G, max_G)
             # ... debug print removed ...

        # Skip intrinsic reward calculation when reward is 0 (placeholder value)
        # This is to avoid calculating intrinsic rewards before we have the actual reward from the environment
        # Instead, we'll use the update_experience_with_intrinsic_reward method after we have the real reward
        # --- Add pretraining check here ---
        add_intrinsic = self.use_intrinsic_rewards and not self.pretraining_completed and self.intrinsic_reward_generator is not None
        is_placeholder_reward = isinstance(original_reward, (int, float)) and original_reward == 0 or \
                              hasattr(original_reward, 'item') and original_reward.item() == 0

        intrinsic_reward_value = 0.0 # Default value

        if add_intrinsic and not is_placeholder_reward: # add_intrinsic now includes the pretraining check
            # Convert inputs to tensor format for intrinsic reward computation
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(state).to(self.device)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
            else:
                state_tensor = state.to(self.device)

            if not isinstance(action, torch.Tensor):
                if self.action_space_type == "discrete":
                    if isinstance(action, np.ndarray):
                        # For numpy arrays
                        if action.size == 1:
                            # Single value in array
                            action = int(action.item())
                        else:
                            # Convert first element if multiple values
                            action = int(action[0])
                    elif isinstance(action, (float, np.floating)):
                        # Direct float types
                        action = int(action)
                    elif hasattr(action, 'item'):
                        # Tensor-like objects
                        action = int(action.item())
                    action_tensor = torch.LongTensor([action]).to(self.device)
                else:
                    action_tensor = torch.FloatTensor(action).to(self.device)
                    if action_tensor.dim() == 1:
                        action_tensor = action_tensor.unsqueeze(0)
            else:
                action_tensor = action.to(self.device)

            # Get next state based on algorithm type and buffer structure
            next_state = None
            if self.algorithm_type == "streamac" and hasattr(self.algorithm, 'experience_buffers') and env_id in self.algorithm.experience_buffers:
                # For StreamAC, use environment-specific buffer
                if len(self.algorithm.experience_buffers[env_id]) > 0:
                    next_state = self.algorithm.experience_buffers[env_id][-1]['obs']
            elif self.algorithm_type == "ppo":
                # For PPO, use the memory buffer
                if hasattr(self.algorithm, 'memory') and hasattr(self.algorithm.memory, 'obs') and self.algorithm.memory.size > 0:
                    # Get the most recent observation (accounting for circular buffer)
                    idx = (self.algorithm.memory.pos - 1) % self.algorithm.memory.buffer_size
                    next_state = self.algorithm.memory.obs[idx]
            elif hasattr(self.algorithm, 'experience_buffer') and len(self.algorithm.experience_buffer) > 0:
                # For other algorithms with a flat buffer
                next_state = self.algorithm.experience_buffer[-1][0]

            # If we found a next state, compute and apply intrinsic reward
            if next_state is not None:
                # Convert next_state to tensor if needed
                if not isinstance(next_state, torch.Tensor):
                    next_state = torch.FloatTensor(next_state).to(self.device)
                # Ensure batch dimension
                if next_state.dim() == 1:
                    next_state = next_state.unsqueeze(0)

                # Compute intrinsic reward using the correct method name
                intrinsic_reward_value = self.intrinsic_reward_generator.compute_intrinsic_reward(
                    state_tensor, action_tensor, next_state
                )

                # Removed adaptive scaling logic
                # Scale the intrinsic reward using the fixed scale
                intrinsic_reward_value = intrinsic_reward_value * self.intrinsic_reward_scale

                # Log that we're applying intrinsic rewards in debug mode
                if self.debug:
                    extrinsic_reward_val = reward.item() if hasattr(reward, 'item') else reward
                    print(f"[DEBUG] Adding intrinsic reward: {intrinsic_reward_value} to scaled extrinsic: {extrinsic_reward_val}")


        # Get the ORIGINAL extrinsic reward (scaling is handled by the algorithm now)
        # Convert reward to float for calculations
        if isinstance(original_reward, torch.Tensor): extrinsic_reward = original_reward.item()
        elif isinstance(original_reward, np.ndarray): extrinsic_reward = original_reward.item() if original_reward.size == 1 else float(original_reward[0])
        else: extrinsic_reward = float(original_reward)

        # Calculate total reward (extrinsic + scaled intrinsic)
        total_reward = extrinsic_reward + intrinsic_reward_value

        # Store the experience using the algorithm - only update if not in test mode
        # Pass the TOTAL reward (extrinsic + scaled intrinsic) to the algorithm storage
        if not test_mode:
            if self.algorithm_type == "streamac":
                # For StreamAC, check if an update was performed and pass env_id
                if self.debug:
                    print(f"[STEP DEBUG] Trainer.store_experience calling StreamAC.store_experience, current step: {self._true_training_steps()}")

                metrics, did_update = self.algorithm.store_experience(state, action, log_prob, total_reward, value, done, env_id=env_id) # Use total_reward

                # If StreamAC performed an update, log to wandb immediately
                if did_update and self.use_wandb:
                    try:
                        # Increment training steps counter BEFORE logging
                        self.training_steps += 1

                        # Get a unique environment step value for this update
                        unique_env_step = self._get_unique_wandb_step()

                        if self.debug:
                            print(f"[STEP DEBUG] StreamAC performed update, incrementing step to {self._true_training_steps()}")
                            print(f"[STEP DEBUG] About to log to wandb with env_step={unique_env_step}")

                        # Log metrics using our centralized logging function - no need to pass step parameter
                        # as it will use total_env_steps (which we pass) by default
                        self._log_to_wandb(metrics, total_env_steps=unique_env_step)
                    except Exception as e:
                        if self.debug:
                            print(f"[STEP DEBUG] Error logging to wandb: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                # For other algorithms like PPO, just store normally
                # This method is now less used for PPO due to batching
                self.algorithm.store_experience(state, action, log_prob, total_reward, value, done) # Use total_reward
        else:
            # In test mode, just store without updating (for StreamAC)
            if hasattr(self.algorithm, 'experience_buffer'):
                exp = {
                    'obs': state,
                    'action': action,
                    'log_prob': log_prob,
                    'reward': original_reward, # Store original reward in test mode
                    'value': value,
                    'done': done,
                    'env_id': env_id
                }
                self.algorithm.experience_buffer.append(exp)

                # Also store in environment-specific buffer if it exists
                if hasattr(self.algorithm, 'experience_buffers'):
                    if env_id not in self.algorithm.experience_buffers:
                        self.algorithm.experience_buffers[env_id] = []
                    self.algorithm.experience_buffers[env_id].append(exp)

        # Update auxiliary task HISTORY BUFFERS if enabled
        # Loss calculation is now handled within PPO update or separately for StreamAC
        if self.use_auxiliary_tasks and hasattr(self, 'aux_task_manager'):
            # Pass the ORIGINAL observation and the SCALED reward to the history
            self.aux_task_manager.update(
                observations=state,
                rewards=total_reward # Use the total reward (scaled extrinsic + scaled intrinsic)
            )


    # Helper method to get a unique wandb step based on environment steps
    def _get_unique_wandb_step(self):
        """Get a unique environment step value that hasn't been used for wandb logging yet"""
        base_env_step = self.total_env_steps

        # If we've already used this environment step, increment by 1 to make it unique
        if hasattr(self, '_last_wandb_env_step') and base_env_step <= self._last_wandb_env_step:
            return self._last_wandb_env_step + 1

        return base_env_step

    def store_experience_at_idx(self, idx, state=None, action=None, log_prob=None, reward=None, value=None, done=None):
        """Forward to algorithm's store_experience_at_idx method if it exists"""
        # Delegate to algorithm's method if available, otherwise use our own
        if hasattr(self.algorithm, 'store_experience_at_idx'):
            self.algorithm.store_experience_at_idx(idx, obs=state, action=action, log_prob=log_prob, reward=reward, value=value, done=done)
        elif self.algorithm_type == "ppo":
            # Only PPO uses this method via memory, so use memory directly
            self.algorithm.memory.store_experience_at_idx(idx, state=state, action=action, log_prob=log_prob, reward=reward, value=value, done=done)
        else:
             if self.debug:
                 print(f"[DEBUG] store_experience_at_idx called but not supported by {self.algorithm_type}")


    def get_action(self, obs, deterministic=False, return_features=False):
        """
        Get an action and log probability for a given observation.
        Value is no longer calculated here for PPO.

        Args:
            obs: Observation tensor or array
            deterministic: If True, return the most likely action without sampling
            return_features: If True, also return extracted features (for auxiliary tasks)

        Returns:
            Tuple containing (action, log_prob, value[, features])
            For PPO, value is a dummy tensor of zeros.
        """
        # Convert observation to tensor if it's not already
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(self.device)

        # Ensure observation has batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Forward to algorithm's get_action method
        result = self.algorithm.get_action(obs, deterministic, return_features)

        # PPO's get_action now returns (action, log_prob, dummy_value[, features]) directly
        return result


    def update(self, completed_episode_rewards=None, total_env_steps: Optional[int] = None,
              steps_per_second: Optional[float] = None, env_stats: Optional[Dict[str, float]] = None):
        """Update policy based on collected experiences.
        Different implementations for PPO vs StreamAC.

        Args:
            completed_episode_rewards: List of episode rewards for completed episodes since last update
            total_env_steps: Total environment steps (across all environments)
            steps_per_second: Environment steps per second (for performance tracking)
            env_stats: Dictionary of environment statistics collected from observations
        """
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
            self._update_auxiliary_weights()

        # Store total_env_steps if provided (mainly for PPO)
        if total_env_steps is not None:
            self.total_env_steps = total_env_steps

        # Store steps_per_second if provided
        if steps_per_second is not None:
            metrics['steps_per_second'] = steps_per_second

        # Accumulate environment stats if provided instead of directly logging them
        if env_stats:
            if self.debug and len(env_stats) > 0:
                print(f"[DEBUG] Received {len(env_stats)} environment stats for accumulation")
            # Accumulate stats for averaging at log time
            self.accumulate_env_stats(env_stats)
            
        # Compute averages from accumulated environment stats 
        # and add them to metrics dictionary
        avg_env_stats = self._compute_avg_env_stats()
        if avg_env_stats:
            if self.debug: 
                print(f"[DEBUG] Adding {len(avg_env_stats)} averaged environment stats to metrics")
            
            # Add the averaged stats to the metrics dictionary with 'env_' prefix
            for key, value in avg_env_stats.items():
                # Ensure all env stats have the 'env_' prefix
                if not key.startswith('env_'):
                    metrics[f'env_{key}'] = value
                else:
                    metrics[key] = value

        # --- Algorithm Update ---
        # Forward to specific algorithm implementation
        # PPO's update now handles auxiliary loss internally and returns all metrics
        algorithm_metrics = self.algorithm.update()

        # Merge algorithm metrics with our metrics
        metrics.update(algorithm_metrics)

        # Record update time
        update_time = time.time() - update_start_time
        metrics['update_time'] = update_time

        # Ensure explained_variance is correctly calculated and reported for PPO
        if self.algorithm_type == "ppo" and 'explained_variance' not in metrics:
            # PPOAlgorithm._update_policy now calculates explained_variance directly
            # If it's missing, it means the update likely failed or was skipped.
            if self.debug:
                print("[DEBUG] Explained variance missing from PPO update metrics.")
            metrics['explained_variance'] = 0.0 # Default to 0 if missing


        # Update training step counter
        if self.algorithm_type != "streamac":  # For StreamAC, steps are tracked in store_experience
            self.training_steps += 1

        # Use completed episode rewards from main.py if provided (for PPO)
        if completed_episode_rewards is not None and len(completed_episode_rewards) > 0:
            # Add all completed episode rewards to our tracking deque
            for reward in completed_episode_rewards:
                self.episode_rewards.append(reward)

            # Calculate reward statistics
            metrics['mean_episode_reward'] = np.mean(completed_episode_rewards)
            metrics['min_episode_reward'] = np.min(completed_episode_rewards)
            metrics['max_episode_reward'] = np.max(completed_episode_rewards)
            metrics['std_episode_reward'] = np.std(completed_episode_rewards)
            
            # Update skill rating if we have enough history
            if len(self.episode_rewards) >= self.skill_update_frequency:
                # Get current batch performance
                current_reward = metrics['mean_episode_reward']
                
                # Only update skill rating every N episodes
                if self.total_episodes % self.skill_update_frequency == 0:
                    # Calculate performance relative to historical baseline
                    if len(self.episode_rewards) > self.skill_rating_window:
                        # Use recent history excluding current batch for comparison
                        historical_rewards = list(self.episode_rewards)[:-len(completed_episode_rewards)]
                        recent_rewards = historical_rewards[-self.skill_rating_window:]
                        
                        historical_mean = np.mean(recent_rewards) if recent_rewards else current_reward
                        historical_std = max(0.1, np.std(recent_rewards)) if len(recent_rewards) > 1 else 1.0
                        
                        # Calculate z-score (how many standard deviations from mean)
                        z_score = (current_reward - historical_mean) / historical_std
                        
                        # Determine win/loss/draw based on performance with more sensitive thresholds
                        if z_score > 0.2:  # Better than historical performance
                            result = 1.0  # Win
                        elif z_score < -0.2:  # Worse than historical performance
                            result = 0.0  # Loss
                        else:  # Similar to historical performance
                            result = 0.5  # Draw
                            
                        # Update skill rating
                        new_rating = self.skill_tracker.update_rating(self.baseline_rating, result)
                        
                        # Add skill metrics to wandb logging
                        metrics['SKILL/rating'] = new_rating
                        metrics['SKILL/z_score'] = z_score
                        metrics['SKILL/result'] = result
                        
                        if self.debug:
                            print(f"[DEBUG] Updated skill rating: {new_rating:.1f} (z-score: {z_score:.2f}, result: {result})")
                    else:
                        # Not enough history yet, just log current rating
                        metrics['SKILL/rating'] = self.skill_tracker.get_current_rating()

            if self.debug:
                print(f"[DEBUG] Using {len(completed_episode_rewards)} completed episode rewards, mean: {metrics['mean_episode_reward']:.4f}")
        # Check if mean_return was provided directly in metrics (new preferred method)
        elif 'mean_return' in metrics and metrics['mean_return'] != 0:
            # Use mean_return directly from algorithm metrics (works for any algorithm)
            metrics['mean_episode_reward'] = metrics['mean_return']

            if self.debug:
                print(f"[DEBUG] Using algorithm-provided mean_return: {metrics['mean_return']:.4f}")

            # Add to our episode_rewards tracking for historical stats
            self.episode_rewards.append(metrics['mean_return'])

        # Otherwise check internal episode returns (fallback, mainly for PPO)
        elif self.algorithm_type == "ppo" and hasattr(self.algorithm, 'episode_returns') and len(self.algorithm.episode_returns) > 0:
            episode_returns = list(self.algorithm.episode_returns)
            if episode_returns:
                # Add all completed episode returns to our tracking deque
                for reward in episode_returns:
                    self.episode_rewards.append(reward)

                # Calculate reward statistics
                metrics['mean_episode_reward'] = np.mean(episode_returns)
                metrics['min_episode_reward'] = np.min(episode_returns)
                metrics['max_episode_reward'] = np.max(episode_returns)
                metrics['std_episode_reward'] = np.std(episode_returns)

                if self.debug:
                    print(f"[DEBUG] Using internal PPO episode returns, mean: {metrics['mean_episode_reward']:.4f}")
        # For StreamAC, fallback to standard mean_return in metrics
        elif self.algorithm_type == "streamac" and 'mean_return' in metrics:
             metrics['mean_episode_reward'] = metrics['mean_return']

             # If there's a return value but no completed episodes, we still want to track it
             if 'mean_return' in metrics:
                 self.episode_rewards.append(metrics['mean_return'])


        # Log metrics (metrics dict should now contain aux losses from PPO update)
        if self.use_wandb:
            try:
                # Pass total_env_steps to the logging function
                self._log_to_wandb(metrics, total_env_steps=self.total_env_steps)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error logging to wandb: {str(e)}")
                    import traceback
                    traceback.print_exc()

        # Append losses to deques (use SCALARS for tracking)
        if 'actor_loss' in metrics:
            self.actor_losses.append(metrics['actor_loss'])
        if 'critic_loss' in metrics:
            self.critic_losses.append(metrics['critic_loss'])
        if 'entropy_loss' in metrics:
            self.entropy_losses.append(metrics['entropy_loss'])
        if 'total_loss' in metrics:
            self.total_losses.append(metrics['total_loss'])
        if 'sr_loss_scalar' in metrics: # Add aux loss tracking using scalars
            self.aux_sr_losses.append(metrics['sr_loss_scalar'])
        if 'rp_loss_scalar' in metrics: # Add aux loss tracking using scalars
            self.aux_rp_losses.append(metrics['rp_loss_scalar'])

        # Clean up tensors to reduce memory usage
        self._cleanup_tensors()

        return metrics

    def reset_auxiliary_tasks(self):
        """Reset auxiliary task history when episodes end"""
        if self.use_auxiliary_tasks and hasattr(self, 'aux_task_manager'):
            self.aux_task_manager.reset()

    def save_models(self, model_path=None, metadata=None):
        """Save actor and critic models, including curriculum state if available."""
        if model_path is None:
            model_path = f"models/model_{int(time.time())}.pt"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        current_timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Initialize metadata if None
        if metadata is None:
            metadata = {}

        # Update metadata with training info
        metadata.update({
            'algorithm': self.algorithm_type,
            'training_step': self.training_steps + self.training_step_offset,
            'total_episodes': self.total_episodes + self.total_episodes_offset,
            'total_env_steps': self.total_env_steps, # Add total env steps to metadata
            'timestamp': current_timestamp,
            'version': '2.1',  # Checkpoint format version (updated for shared model)
            'shared_model': self.shared_model # Indicate if the model was shared
        })

        # Save pretraining state if enabled
        if self.use_pretraining:
            metadata['pretraining'] = {
                'completed': self.pretraining_completed,
                # Removed transition-related state
                'pretraining_fraction': self.pretraining_fraction,
                'base_sr_weight': self.base_sr_weight,
                'base_rp_weight': self.base_rp_weight,
                'pretraining_sr_weight': self.pretraining_sr_weight,
                'pretraining_rp_weight': self.pretraining_rp_weight
            }

        # Prepare the main checkpoint dictionary
        checkpoint = {
            'metadata': metadata # Include updated metadata
        }

        # Save model state dict(s)
        if self.shared_model:
            checkpoint['shared_model_state'] = self.actor.state_dict()
            if self.debug: print("[DEBUG Trainer Save] Saved shared model state.")
        else:
            checkpoint['actor_state'] = self.actor.state_dict()
            checkpoint['critic_state'] = self.critic.state_dict()
            if self.debug: print("[DEBUG Trainer Save] Saved separate actor and critic states.")


        # Include algorithm state if available
        if hasattr(self, 'algorithm') and hasattr(self.algorithm, 'get_state_dict'):
            checkpoint['algorithm_state'] = self.algorithm.get_state_dict()
            if self.debug:
                print(f"[DEBUG Trainer Save] Included algorithm state in checkpoint.")

        # Include curriculum state if manager is registered
        if hasattr(self, 'curriculum_manager') and self.curriculum_manager is not None:
             if hasattr(self.curriculum_manager, 'get_state'):
                 checkpoint['curriculum'] = self.curriculum_manager.get_state()
                 if self.debug:
                     print(f"[DEBUG Trainer Save] Included curriculum state in checkpoint.")

        # Include auxiliary task state if manager exists
        if hasattr(self, 'aux_task_manager') and self.aux_task_manager is not None:
             if hasattr(self.aux_task_manager, 'get_state_dict'):
                 checkpoint['aux'] = self.aux_task_manager.get_state_dict()
                 if self.debug:
                     print(f"[DEBUG Trainer Save] Included auxiliary task state in checkpoint.")

        # Include intrinsic reward state if generator exists
        if hasattr(self, 'intrinsic_reward_generator') and self.intrinsic_reward_generator is not None:
             if hasattr(self.intrinsic_reward_generator, 'get_state_dict'):
                 checkpoint['intrinsic'] = self.intrinsic_reward_generator.get_state_dict()
                 if self.debug:
                     print(f"[DEBUG Trainer Save] Included intrinsic reward state in checkpoint.")
                     
        # Include skill tracker state
        if hasattr(self, 'skill_tracker'):
            checkpoint['skill_tracker'] = {
                'modes': self.skill_tracker.modes,
                'history': self.skill_tracker.history,
                'update_count': self.skill_tracker.update_count,
                'rating_inc': self.skill_tracker.rating_inc,
                'baseline_rating': self.baseline_rating,
                'skill_update_frequency': self.skill_update_frequency,
                'skill_rating_window': self.skill_rating_window
            }
            if self.debug:
                print(f"[DEBUG Trainer Save] Included skill tracker state in checkpoint. Current rating: {self.skill_tracker.get_current_rating()}")

        # Include random states for reproducibility
        checkpoint['random_states'] = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate()
        }

        # Save episode rewards history
        if hasattr(self, 'episode_rewards') and len(self.episode_rewards) > 0:
            checkpoint['episode_rewards'] = list(self.episode_rewards)
            if self.debug:
                print(f"[DEBUG Trainer Save] Saved episode rewards history ({len(self.episode_rewards)} entries)")

        # Include entropy settings
        checkpoint['entropy'] = {
            'entropy_coef': self.entropy_coef,
            'entropy_coef_decay': self.entropy_coef_decay,
            'min_entropy_coef': self.min_entropy_coef,
            'base_entropy_coef': self.base_entropy_coef
        }

        # Include wandb info if enabled
        if self.use_wandb and wandb.run:
            checkpoint['wandb'] = {
                'run_id': wandb.run.id,
                'project': wandb.run.project,
                'entity': wandb.run.entity,
                'name': wandb.run.name
            }

        # Save to a single file with all components
        torch.save(checkpoint, model_path)

        if self.debug:
            print(f"[DEBUG Trainer Save] Saved checkpoint to {model_path}")

        return model_path

    def load_models(self, model_path):
        """
        Load a comprehensive checkpoint to resume training from the exact state.
        Handles both shared and separate model architectures based on checkpoint metadata.

        Args:
            model_path: Path to the saved checkpoint file

        Returns:
            bool: Whether loading was successful
        """
        if not os.path.exists(model_path):
            print(f"Error: Checkpoint file not found: {model_path}")
            return False

        try:
            # Load the saved state with CPU mapping to allow loading on any device
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Check for checkpoint version and shared model status
            metadata = checkpoint.get('metadata', {})
            version = metadata.get('version', '1.0')
            checkpoint_shared_model = metadata.get('shared_model', False) # Check if checkpoint used shared model

            # Log loading info
            if self.debug:
                print(f"[DEBUG] Loading checkpoint version {version} from {model_path}")
                print(f"[DEBUG] Checkpoint shared_model: {checkpoint_shared_model}, Trainer shared_model: {self.shared_model}")

            # Check for architecture mismatch (shared vs separate)
            if checkpoint_shared_model != self.shared_model:
                print(f"Warning: Architecture mismatch! Checkpoint shared_model={checkpoint_shared_model}, but current Trainer shared_model={self.shared_model}.")
                # Fail loading if architectures mismatch to prevent unexpected errors
                print("Loading aborted due to architecture mismatch.")
                return False
                # print("Attempting to load parameters anyway, but this might lead to errors or unexpected behavior.")
                # Decide how to handle mismatch - e.g., prioritize current setup or fail?
                # For now, we'll attempt to load what we can.

            # ===== Restore Model Parameters =====
            if checkpoint_shared_model:
                if 'shared_model_state' in checkpoint:
                    # Should already match self.shared_model due to check above
                    model_state = fix_compiled_state_dict(checkpoint['shared_model_state'])
                    mismatches = load_partial_state_dict(self.actor, model_state)
                    # Critic is the same instance, already loaded.
                    if self.debug:
                        print(f"[DEBUG] Loaded shared model state ({mismatches} mismatches)")
                else:
                    print("Warning: Checkpoint indicates shared model, but 'shared_model_state' key not found.")
                    return False
            else: # Checkpoint has separate actor/critic states
                if 'actor_state' in checkpoint and 'critic_state' in checkpoint:
                     # Should already match self.shared_model == False due to check above
                    actor_state = fix_compiled_state_dict(checkpoint['actor_state'])
                    critic_state = fix_compiled_state_dict(checkpoint['critic_state'])
                    actor_mismatches = load_partial_state_dict(self.actor, actor_state)
                    critic_mismatches = load_partial_state_dict(self.critic, critic_state)
                    if self.debug:
                        print(f"[DEBUG] Loaded separate actor ({actor_mismatches} mismatches) and critic ({critic_mismatches} mismatches) states")
                else:
                    print("Warning: Checkpoint indicates separate models, but 'actor_state' or 'critic_state' key not found.")
                    return False


            # ===== Restore Algorithm State =====
            if 'algorithm_state' in checkpoint and checkpoint['algorithm_state']:
                algorithm_state = checkpoint['algorithm_state']

                # Restore the algorithm's internal state using its load_state_dict method
                if hasattr(self.algorithm, 'load_state_dict'):
                    try:
                        self.algorithm.load_state_dict(algorithm_state)
                        if self.debug:
                            print(f"[DEBUG] Restored algorithm state using load_state_dict.")
                    except Exception as e:
                        print(f"Warning: Could not load algorithm state using load_state_dict: {e}")
                        if self.debug: import traceback; traceback.print_exc()
                else:
                    # Legacy loading if load_state_dict doesn't exist
                    if self.algorithm_type == "ppo":
                        # PPO optimizer is now handled by algorithm's load_state_dict
                        if self.debug: print("[DEBUG] Restored algorithm state using legacy PPO method (potential issues).")
                    elif self.algorithm_type == "streamac":
                        # ... legacy StreamAC state loading ...
                        pass
                    if self.debug: print("[DEBUG] Restored algorithm state using legacy method.")


            # ===== Restore Auxiliary Task State =====
            if self.use_auxiliary_tasks and 'aux' in checkpoint and checkpoint['aux']:
                aux_state = checkpoint['aux']

                # Only try to restore if we have an auxiliary task manager
                if hasattr(self, 'aux_task_manager') and self.aux_task_manager is not None:
                    try:
                        # Restore model parameters and optimizer states using its method
                        self.aux_task_manager.load_state_dict(aux_state)

                        if self.debug:
                            print(f"[DEBUG] Restored auxiliary task state with weights - SR: {self.aux_task_manager.sr_weight}, RP: {self.aux_task_manager.rp_weight}")
                    except Exception as e:
                        print(f"Warning: Could not restore auxiliary task state: {e}")
                        if self.debug:
                            import traceback
                            traceback.print_exc()

            # ===== Restore Intrinsic Reward State =====
            if self.use_intrinsic_rewards and 'intrinsic' in checkpoint and checkpoint['intrinsic']:
                intrinsic_state = checkpoint['intrinsic']

                # Only try to restore if we have an intrinsic reward generator
                if self.intrinsic_reward_generator is not None:
                    # Load model parameters
                    if hasattr(self.intrinsic_reward_generator, 'load_state_dict'):
                        try:
                            self.intrinsic_reward_generator.load_state_dict(intrinsic_state) # Pass the whole dict
                        except Exception as e:
                            print(f"Warning: Could not load intrinsic reward generator state: {e}")

                    # Restore configuration values (these might be part of the generator's state dict now)
                    # Removed extrinsic normalizer loading

                    if self.debug:
                        print(f"[DEBUG] Restored intrinsic reward generator with scale {self.intrinsic_reward_scale}")

            # ===== Restore Curriculum State =====
            if hasattr(self, 'curriculum_manager') and self.curriculum_manager is not None and 'curriculum' in checkpoint:
                curriculum_state = checkpoint['curriculum']

                try:
                    # Load the entire curriculum state
                    self.curriculum_manager.load_state(curriculum_state)

                    # Critical fix: Re-register the trainer with the curriculum
                    # This ensures the curriculum stages can access trainer attributes like pretraining_completed
                    self.curriculum_manager.register_trainer(self)

                    if self.debug:
                        print(f"[DEBUG] Restored curriculum state and re-registered trainer")
                        print(f"[DEBUG] Current stage: {self.curriculum_manager.current_stage.name}, index: {self.curriculum_manager.current_stage_index}")
                except Exception as e:
                    print(f"Warning: Could not fully restore curriculum state: {e}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()

            # ===== Restore Random States =====
            if 'random_states' in checkpoint:
                try:
                    random_states = checkpoint['random_states']
                    # Restore torch random state
                    if 'torch' in random_states:
                        torch.set_rng_state(random_states['torch'])
                    # Restore numpy random state
                    if 'numpy' in random_states:
                        np.random.set_state(random_states['numpy'])
                    # Restore Python random state
                    if 'random' in random_states:
                        random.setstate(random_states['random'])

                    if self.debug:
                        print("[DEBUG] Restored random states for reproducibility")
                except Exception as e:
                    print(f"Warning: Could not restore random states: {e}")

            # ===== Restore Entropy Settings =====
            if 'entropy' in checkpoint:
                entropy_settings = checkpoint['entropy']
                self.entropy_coef = entropy_settings.get('entropy_coef', self.entropy_coef)
                self.entropy_coef_decay = entropy_settings.get('entropy_coef_decay', self.entropy_coef_decay)
                self.min_entropy_coef = entropy_settings.get('min_entropy_coef', self.min_entropy_coef)
                self.base_entropy_coef = entropy_settings.get('base_entropy_coef', self.base_entropy_coef)

                if self.debug:
                    print(f"[DEBUG] Restored entropy settings: coef={self.entropy_coef:.6f}, decay={self.entropy_coef_decay:.6f}")

            # ===== Restore Training Counters =====
            if 'metadata' in checkpoint:
                metadata = checkpoint['metadata']
                if 'training_step' in metadata:
                    self.training_step_offset = metadata['training_step']
                if 'total_episodes' in metadata:
                    self.total_episodes_offset = metadata['total_episodes']
                # Restore total environment steps
                if 'total_env_steps' in metadata:
                    self.total_env_steps = metadata['total_env_steps']

                if self.debug:
                    print(f"[DEBUG] Restored training counters: steps={self.training_step_offset}, episodes={self.total_episodes_offset}, env_steps={self.total_env_steps}")

            # ===== Restore Episode Rewards History =====
            if 'episode_rewards' in checkpoint:
                rewards_history = checkpoint['episode_rewards']
                # Ensure episode_rewards deque exists with correct maxlen
                if not hasattr(self, 'episode_rewards') or self.episode_rewards is None:
                    # Need history_len, assume 1000 as default from init if not easily accessible
                    # A more robust way would be to store history_len in the checkpoint too
                    history_len = getattr(self, 'history_len', 1000)
                    self.episode_rewards = collections.deque(maxlen=history_len)

                # Clear existing deque before loading
                self.episode_rewards.clear()
                # Add loaded rewards
                for reward in rewards_history:
                    self.episode_rewards.append(reward)

                if self.debug:
                    print(f"[DEBUG] Restored episode rewards history ({len(self.episode_rewards)} entries)")
            else:
                # If not in checkpoint, ensure the deque exists but is empty
                if not hasattr(self, 'episode_rewards') or self.episode_rewards is None:
                    history_len = getattr(self, 'history_len', 1000)
                    self.episode_rewards = collections.deque(maxlen=history_len)
                else:
                    self.episode_rewards.clear() # Ensure it's empty if not loaded
                if self.debug:
                    print("[DEBUG] No episode rewards history found in checkpoint. Initializing empty deque.")
                    
            # ===== Restore Skill Tracker State =====
            if 'skill_tracker' in checkpoint:
                skill_data = checkpoint['skill_tracker']
                
                # Make sure we have a skill tracker
                if not hasattr(self, 'skill_tracker'):
                    self.skill_tracker = SkillRatingTracker()
                
                # Restore skill tracker state
                self.skill_tracker.modes = skill_data.get('modes', {'default': 1500})
                self.skill_tracker.history = skill_data.get('history', {'default': [1500]})
                self.skill_tracker.update_count = skill_data.get('update_count', {'default': 0})
                self.skill_tracker.rating_inc = skill_data.get('rating_inc', 32)
                
                # Restore skill tracking configuration
                self.baseline_rating = skill_data.get('baseline_rating', 1500)
                self.skill_update_frequency = skill_data.get('skill_update_frequency', 5)
                self.skill_rating_window = skill_data.get('skill_rating_window', 100)
                
                if self.debug:
                    current_rating = self.skill_tracker.get_current_rating()
                    update_count = self.skill_tracker.update_count.get('default', 0)
                    print(f"[DEBUG] Restored skill tracker state. Current rating: {current_rating}, Updates: {update_count}")
            else:
                # If not in checkpoint, make sure we have default values
                if not hasattr(self, 'skill_tracker'):
                    self.skill_tracker = SkillRatingTracker()
                    self.baseline_rating = 1500
                    self.skill_update_frequency = 5
                    self.skill_rating_window = 100
                if self.debug:
                    print("[DEBUG] No skill tracker state found in checkpoint. Using default values.")

            # ===== Auto-detect models trained without pretraining =====
            if 'pretraining_completed' not in checkpoint:
                if 'metadata' in checkpoint:
                    metadata = checkpoint['metadata']
                    # Check if this model was created with --no-pretraining flag
                    if 'use_pretraining' in metadata and metadata['use_pretraining'] is False:
                        print(f"Detected model trained with --no-pretraining - setting pretraining_completed = True")
                        self.pretraining_completed = True
                    elif 'pretraining' not in metadata:
                        print(f"No pretraining configuration found - assuming pretraining was disabled")
                        self.pretraining_completed = True
                else:
                    # Older model format without metadata - assume pretraining was completed
                    print(f"Legacy model format detected - assuming pretraining was completed")
                    self.pretraining_completed = True

            # ===== Restore Pretraining State =====
            if 'metadata' in checkpoint and 'pretraining' in metadata: # Check metadata first
                pretraining_state = metadata['pretraining']
                self.pretraining_completed = pretraining_state.get('completed', self.pretraining_completed)
                # Removed loading of transition-related state

                # Restore weight configurations
                self.pretraining_fraction = pretraining_state.get('pretraining_fraction', self.pretraining_fraction)
                # Removed loading of pretraining_transition_steps
                self.base_sr_weight = pretraining_state.get('base_sr_weight', self.base_sr_weight)
                self.base_rp_weight = pretraining_state.get('base_rp_weight', self.base_rp_weight)
                self.pretraining_sr_weight = pretraining_state.get('pretraining_sr_weight', self.pretraining_sr_weight)
                self.pretraining_rp_weight = pretraining_state.get('pretraining_rp_weight', self.pretraining_rp_weight)

                # Explicitly check and print the pretraining_completed flag
                if self.debug:
                    pretraining_status = "completed" if self.pretraining_completed else "active"
                    print(f"[DEBUG] Restored pretraining state from metadata: {pretraining_status}")
                    print(f"[DEBUG] pretraining_completed = {self.pretraining_completed}")
            # If we explicitly find the 'pretraining_completed' at the top level (older format), use it
            elif 'pretraining_completed' in checkpoint:
                self.pretraining_completed = checkpoint['pretraining_completed']
                if self.debug:
                    print(f"[DEBUG] Found top-level pretraining_completed = {self.pretraining_completed}")


            # ===== Resume WandB Run =====
            # Check if wandb is intended to be used for this session (self.use_wandb)
            # and if the checkpoint contains wandb info
            if self.use_wandb and 'wandb' in checkpoint and checkpoint['wandb']:
                wandb_info = checkpoint['wandb']
                run_id_to_resume = wandb_info.get('run_id')

                if run_id_to_resume:
                    project = wandb_info.get('project', 'rlbot-training') # Use project from checkpoint or default
                    entity = wandb_info.get('entity') # Entity might be None
                    name = wandb_info.get('name') # Name might be None

                    # Check if a run is already active and if its ID matches
                    if wandb.run is not None and wandb.run.id != run_id_to_resume:
                        print(f"Warning: Active wandb run ({wandb.run.id}) differs from checkpoint run ({run_id_to_resume}).")
                        print("Finishing current run and attempting to resume the correct one.")
                        wandb.finish() # Stop the currently active run

                    # Now, attempt to resume the run from the checkpoint
                    # Check if wandb.run is None *after* potential finish() call
                    if wandb.run is None:
                        try:
                            print(f"Attempting to resume wandb run: id={run_id_to_resume}, project={project}, entity={entity}")
                            wandb.init(
                                project=project,
                                entity=entity,
                                id=run_id_to_resume,
                                resume="must", # Force resume or fail
                                name=name # Resume with original name if available
                            )
                            if self.debug:
                                print(f"[DEBUG] Successfully resumed wandb run: {run_id_to_resume}")
                            # Sync config from checkpoint if needed (optional)
                            # wandb.config.update(checkpoint.get('config', {}), allow_val_change=True)

                        except Exception as e:
                            print(f"Error resuming wandb run {run_id_to_resume}: {e}")
                            print("Proceeding without resuming the specific wandb run.")
                            # Optionally, start a new run here if resume fails,
                            # or let the initial main.py wandb.init handle it if it hasn't run yet.
                            # For simplicity, we'll let it proceed without a specific run if resume fails.
                            # Ensure wandb.run is None if resumption failed badly
                            if not wandb.run or wandb.run.id != run_id_to_resume:
                                print("Wandb run could not be resumed. Logging might go to a new run or be disabled.")

                    elif wandb.run.id == run_id_to_resume:
                         if self.debug:
                             print(f"[DEBUG] Wandb run {run_id_to_resume} was already active and matches checkpoint.")
                    else:
                         # This case should ideally not be reached after the finish() call
                         print(f"Warning: Could not resume wandb run {run_id_to_resume}. Active run is {wandb.run.id}.")

            print(f"Successfully loaded checkpoint from {model_path}")
            print(f"Resumed training at step {self.training_steps + self.training_step_offset}")
            return True

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False

    def accumulate_env_stats(self, env_stats):
        """
        Accumulate environment statistics between updates for averaging.
        
        Args:
            env_stats (dict): Dictionary of environment statistics from observations
        """
        if not env_stats:
            return
            
        for key, value in env_stats.items():
            # Only process numerical values
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                if key not in self.env_stats_buffer:
                    self.env_stats_buffer[key] = 0.0
                    self.env_stats_counts[key] = 0
                
                self.env_stats_buffer[key] += float(value)
                self.env_stats_counts[key] += 1
                
        if self.debug and len(env_stats) > 0 and self._true_training_steps() % 500 == 0:
            print(f"[DEBUG] Accumulated env stats for {len(env_stats)} metrics. Total accumulated metrics: {len(self.env_stats_buffer)}")

    def _compute_avg_env_stats(self):
        """
        Compute average environment statistics from accumulated values.
        Returns a dictionary of averaged statistics.
        """
        avg_stats = {}
        for key, total_value in self.env_stats_buffer.items():
            count = max(1, self.env_stats_counts.get(key, 1))  # Avoid division by zero
            avg_stats[key] = total_value / count
            
        # Clear the buffers for next accumulation period
        self.env_stats_buffer = {}
        self.env_stats_counts = {}
        
        # Store the computed averages
        self.env_stats_metrics = avg_stats
        
        return avg_stats

    def _update_pretraining_state(self):
        """Update pretraining state and manage transition to regular training."""
        # Skip if pretraining is not enabled or already completed
        if not self.use_pretraining or self.pretraining_completed:
            return

        # Calculate when pretraining should end based on episodes or steps
        pretraining_end_step = self._get_pretraining_end_step()
        current_step = self._true_training_steps()

        # Check if we've reached the end of the pretraining phase
        if current_step >= pretraining_end_step:
            self.pretraining_completed = True
            print(f"Pretraining phase completed at step {current_step}. Switching to regular training.")

            # Reset auxiliary task weights to regular values immediately
            if self.use_auxiliary_tasks and self.aux_task_manager:
                self.aux_task_manager.sr_weight = self.base_sr_weight
                self.aux_task_manager.rp_weight = self.base_rp_weight

            # Reset entropy coefficient to base value immediately
            self.entropy_coef = self.base_entropy_coef

            if self.use_wandb:
                wandb.log({'training/pretraining_completed': True}, step=current_step)
        # Removed transition phase logic

    # Reward Scaling Methods (REMOVED - Handled in Algorithm)
    # def _initialize_reward_scaling_stats(self, env_id):
    # ... (removed method body)
    # def _update_reward_scaling_stats(self, env_id, reward_t, done):
    # ... (removed method body)
    # def _scale_reward(self, env_id, reward_t, variance_G, max_G):
    # ... (removed method body)

    def update_experience_with_intrinsic_reward(self, state=None, action=None, next_state=None, done=None, reward=None, store_idx=None, env_id=None):
        """
        Calculate intrinsic reward and update stored experience.
        Uses the fixed intrinsic reward scale.
        Combines intrinsic reward with the SCALED extrinsic reward.

        Args:
            state: Current observation
            action: Action taken
            next_state: Next observation
            done: Done flag
            reward: ORIGINAL extrinsic reward (unscaled)
            store_idx: Index in memory to update (for PPO)
            env_id: Environment ID

        Returns:
            Total reward (scaled extrinsic + scaled intrinsic)
        """
        # If intrinsic rewards are disabled, pretraining is completed, or generator is not initialized, return original reward
        # --- Add pretraining check here ---
        if not self.use_intrinsic_rewards or self.pretraining_completed or self.intrinsic_reward_generator is None:
            # Return the ORIGINAL reward
            if isinstance(reward, torch.Tensor): return reward.item()
            elif isinstance(reward, np.ndarray): return reward.item() if reward.size == 1 else float(reward[0])
            else: return float(reward)

        intrinsic_reward_value = 0.0
        try:
            # Convert inputs to tensors if needed
            if not isinstance(state, torch.Tensor): obs_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            else: obs_tensor = state.to(self.device)
            if not isinstance(action, torch.Tensor):
                 if self.action_space_type == "discrete":
                     if isinstance(action, np.ndarray) and action.shape == (self.action_dim,): action_idx = np.argmax(action)
                     elif isinstance(action, (int, float, np.number)): action_idx = int(action)
                     else: action_idx = 0
                     action_tensor = torch.tensor([action_idx], dtype=torch.long, device=self.device)
                 else:
                     action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
            else: action_tensor = action.to(self.device)

            if not isinstance(next_state, torch.Tensor): next_obs_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            else: next_obs_tensor = next_state.to(self.device)

            # Ensure batch dimension
            if obs_tensor.dim() == 1: obs_tensor = obs_tensor.unsqueeze(0)
            if action_tensor.dim() == 0: action_tensor = action_tensor.unsqueeze(0)
            elif action_tensor.dim() == 1 and self.action_space_type != "discrete": action_tensor = action_tensor.unsqueeze(0)
            if next_obs_tensor.dim() == 1: next_obs_tensor = next_obs_tensor.unsqueeze(0)


            # Compute intrinsic reward
            intrinsic_reward_value = self.intrinsic_reward_generator.compute_intrinsic_reward(
                obs_tensor, action_tensor, next_obs_tensor
            )

            # Update intrinsic reward model
            if done is not None:
                self.intrinsic_reward_generator.update(obs_tensor, action_tensor, next_obs_tensor, done)

            # Convert to scalar if tensor
            if isinstance(intrinsic_reward_value, torch.Tensor):
                intrinsic_reward_value = intrinsic_reward_value.item()

            # --- Apply FIXED intrinsic scale ---
            scaled_intrinsic_reward = intrinsic_reward_value * self.intrinsic_reward_scale
            # --- End Fixed Scaling ---

        except Exception as e:
            if self.debug:
                print(f"Error computing intrinsic reward: {e}")
                import traceback
                traceback.print_exc() # Print stack trace for debugging
            scaled_intrinsic_reward = 0.0

        # Get the ORIGINAL extrinsic reward
        if isinstance(reward, torch.Tensor): original_extrinsic_reward = reward.item()
        elif isinstance(reward, np.ndarray): original_extrinsic_reward = reward.item() if reward.size == 1 else float(reward[0])
        else: original_extrinsic_reward = float(reward)

        # Calculate total reward (original extrinsic + scaled intrinsic)
        total_reward = original_extrinsic_reward + scaled_intrinsic_reward

        # Update stored experience if index is provided (for PPO)
        if store_idx is not None and self.algorithm_type == "ppo":
            # Update the experience in memory using the dedicated method
            # Ensure the total_reward is used here
            self.store_experience_at_idx(store_idx, reward=total_reward, done=done)

        return total_reward

    def calculate_intrinsic_rewards_batch(self, states, actions, next_states, env_ids):
        """
        Calculates intrinsic rewards in batch for multiple agents/environments.
        Applies the fixed intrinsic reward scale.

        Args:
            states: Batch of states/observations (numpy array or tensor)
            actions: Batch of actions (numpy array or tensor) - Can be indices or one-hot for discrete
            next_states: Batch of next states/observations (numpy array or tensor)
            env_ids: List of environment IDs for each sample (not used for fixed scaling)

        Returns:
            Batch of scaled intrinsic rewards (numpy array)
        """
        # --- Add pretraining check here ---
        if not self.use_intrinsic_rewards or self.pretraining_completed or self.intrinsic_reward_generator is None:
            # Return zeros with the right shape
            return np.zeros(len(states))

        # Convert inputs to tensors
        if not isinstance(states, torch.Tensor):
            states_tensor = torch.FloatTensor(states).to(self.device)
        else:
            states_tensor = states.to(self.device)

        # Handle actions based on action space type
        if not isinstance(actions, torch.Tensor):
            if self.action_space_type == "discrete":
                try:
                    if isinstance(actions, np.ndarray) and actions.ndim == 2 and actions.shape[1] == self.action_dim:
                        actions_indices = np.argmax(actions, axis=1)
                    else:
                        actions_indices = np.array(actions, dtype=int)
                    actions_tensor = torch.LongTensor(actions_indices).to(self.device)
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Failed to convert actions to LongTensor indices: {e}. Using FloatTensor.")
                    actions_tensor = torch.FloatTensor(actions).to(self.device)
            else: # Continuous
                actions_tensor = torch.FloatTensor(actions).to(self.device)
        else:
            actions_tensor = actions.to(self.device)
            if self.action_space_type == "discrete" and actions_tensor.dtype != torch.long:
                 if actions_tensor.ndim == 2 and actions_tensor.shape[1] == self.action_dim:
                     actions_tensor = torch.argmax(actions_tensor, dim=1).long()
                 else:
                     actions_tensor = actions_tensor.long()


        if not isinstance(next_states, torch.Tensor):
            next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        else:
            next_states_tensor = next_states.to(self.device)

        # Ensure batch dimension
        if states_tensor.dim() == 1: states_tensor = states_tensor.unsqueeze(0)
        if next_states_tensor.dim() == 1: next_states_tensor = next_states_tensor.unsqueeze(0)

        batch_size = states_tensor.size(0)

        # Properly handle actions tensor based on action space type
        if self.action_space_type == "discrete":
            if actions_tensor.dim() == 0:
                actions_tensor = actions_tensor.unsqueeze(0)
            elif actions_tensor.dim() == 1:
                if actions_tensor.size(0) != batch_size:
                    if self.debug:
                        print(f"[DEBUG] Reshaping actions_tensor from shape {actions_tensor.shape} to match batch size {batch_size}")
                    if actions_tensor.size(0) == 1:
                        actions_tensor = actions_tensor.repeat(batch_size)
        else:  # Continuous action space
            if actions_tensor.dim() == 1:
                if actions_tensor.size(0) == self.action_dim:
                    actions_tensor = actions_tensor.unsqueeze(0).repeat(batch_size, 1)
                else:
                    actions_tensor = actions_tensor.unsqueeze(1) if actions_tensor.size(0) == batch_size else actions_tensor.unsqueeze(0)
            elif actions_tensor.dim() == 2 and actions_tensor.size(0) != batch_size:
                if actions_tensor.size(0) == 1:
                    actions_tensor = actions_tensor.repeat(batch_size, 1)

        if self.debug:
            print(f"[DEBUG] Tensor shapes before intrinsic reward calculation:")
            print(f"  - states_tensor: {states_tensor.shape}")
            print(f"  - actions_tensor: {actions_tensor.shape}")
            print(f"  - next_states_tensor: {next_states_tensor.shape}")

        # Compute batch intrinsic rewards
        try:
            intrinsic_rewards = self.intrinsic_reward_generator.compute_intrinsic_reward(
                states_tensor, actions_tensor, next_states_tensor
            )
        except RuntimeError as e:
            if self.debug:
                print(f"[DEBUG] Error in compute_intrinsic_reward: {e}")
                print(f"Falling back to single item processing")

            intrinsic_rewards_list = []
            for i in range(batch_size):
                single_reward = self.intrinsic_reward_generator.compute_intrinsic_reward(
                    states_tensor[i:i+1],
                    actions_tensor[i:i+1] if actions_tensor.dim() > 0 and actions_tensor.size(0) >= i+1 else actions_tensor,
                    next_states_tensor[i:i+1]
                )
                if isinstance(single_reward, torch.Tensor):
                    intrinsic_rewards_list.append(single_reward.item())
                else:
                    intrinsic_rewards_list.append(single_reward)
            intrinsic_rewards = np.array(intrinsic_rewards_list)

        # Handle different types of intrinsic rewards output
        if isinstance(intrinsic_rewards, torch.Tensor):
            intrinsic_rewards = intrinsic_rewards.detach().cpu().numpy()
        elif isinstance(intrinsic_rewards, (float, int)):
            intrinsic_rewards = np.array([float(intrinsic_rewards)] * batch_size)
        elif isinstance(intrinsic_rewards, (list, tuple)):
            intrinsic_rewards = np.array(intrinsic_rewards)
        elif not isinstance(intrinsic_rewards, np.ndarray):
            if self.debug:
                print(f"[DEBUG] Unexpected intrinsic rewards type: {type(intrinsic_rewards)}")
            intrinsic_rewards = np.zeros(batch_size)

        # Ensure we have a 1D array of the correct size
        if isinstance(intrinsic_rewards, np.ndarray):
            if intrinsic_rewards.ndim > 1:
                intrinsic_rewards = intrinsic_rewards.flatten()
            if len(intrinsic_rewards) != batch_size:
                if len(intrinsic_rewards) == 1:
                    intrinsic_rewards = np.repeat(intrinsic_rewards, batch_size)
                else:
                    if self.debug:
                        print(f"[DEBUG] Intrinsic rewards length {len(intrinsic_rewards)} doesn't match batch size {batch_size}")
                    if len(intrinsic_rewards) > batch_size:
                        intrinsic_rewards = intrinsic_rewards[:batch_size]
                    else:
                        intrinsic_rewards = np.pad(intrinsic_rewards,
                                                 (0, batch_size - len(intrinsic_rewards)),
                                                 'constant', constant_values=0)

        # Apply the fixed intrinsic reward scale
        scaled_intrinsic_rewards = intrinsic_rewards * self.intrinsic_reward_scale

        return scaled_intrinsic_rewards

    def _cleanup_tensors(self):
        """Clean up any cached tensors to reduce memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
