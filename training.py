from collections import deque
from torch.distributions import Categorical, Normal
from torch.amp import autocast, GradScaler
from models import fix_compiled_state_dict, load_partial_state_dict, print_model_info
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
import random
import collections


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
        use_weight_clipping: bool = True,
        weight_clip_kappa: float = 1.0,
        adaptive_kappa: bool = True,
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
        
        # IMPORTANT: Set use_intrinsic_rewards early to avoid attribute access errors
        self.use_intrinsic_rewards = use_intrinsic_rewards and use_pretraining
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
            self.memory = self.algorithm.memory
        elif self.algorithm_type == "streamac":
            # Create StreamAC algorithm
            algorithm = StreamACAlgorithm(
                actor=self.actor, 
                critic=self.critic,
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
                debug=self.debug
            )
            # Set reference to trainer so StreamAC can update auxiliary tasks
            algorithm.trainer = self
            self.algorithm = algorithm
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}. Use 'ppo' or 'streamac'.")

        # Use Automatic Mixed Precision (AMP) if requested and on CUDA
        self.use_amp = "cuda" in str(device) and use_amp
        if self.use_amp:
            self.scaler = GradScaler()

        # Track metrics using deques with a max length (e.g., 1000)
        history_len = 1000
        self.actor_losses = collections.deque(maxlen=history_len)
        self.critic_losses = collections.deque(maxlen=history_len)
        self.entropy_losses = collections.deque(maxlen=history_len)
        self.total_losses = collections.deque(maxlen=history_len)
        self.aux_sr_losses = collections.deque(maxlen=history_len) # Add deque for aux losses
        self.aux_rp_losses = collections.deque(maxlen=history_len) # Add deque for aux losses

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
                
                # Also compile auxiliary task models if they exist
                if self.use_auxiliary_tasks and hasattr(self, 'aux_task_manager'):
                    if hasattr(self.aux_task_manager, 'sr_model') and self.aux_task_manager.sr_model is not None:
                        self.aux_task_manager.sr_model = torch.compile(self.aux_task_manager.sr_model)
                        if self.debug:
                            print("[DEBUG] Compiled SR auxiliary model")
                    
                    if hasattr(self.aux_task_manager, 'rp_model') and self.aux_task_manager.rp_model is not None:
                        self.aux_task_manager.rp_model = torch.compile(self.aux_task_manager.rp_model)
                        if self.debug:
                            print("[DEBUG] Compiled RP auxiliary model")
                
                # Also compile intrinsic reward models if they exist
                if self.use_intrinsic_rewards and self.intrinsic_reward_generator is not None:
                    # For ensemble, compile each component
                    if hasattr(self.intrinsic_reward_generator, 'components'):
                        for i, component in enumerate(self.intrinsic_reward_generator.components):
                            if hasattr(component, 'forward_model') and component.forward_model is not None:
                                component.forward_model = torch.compile(component.forward_model)
                            if hasattr(component, 'inverse_model') and component.inverse_model is not None:
                                component.inverse_model = torch.compile(component.inverse_model)
                            if hasattr(component, 'target_network') and component.target_network is not None:
                                component.target_network = torch.compile(component.target_network)
                            if hasattr(component, 'predictor_network') and component.predictor_network is not None:
                                component.predictor_network = torch.compile(component.predictor_network)
                        if self.debug:
                            print("[DEBUG] Compiled intrinsic reward ensemble components")
                    # For individual models
                    else:
                        if hasattr(self.intrinsic_reward_generator, 'forward_model') and self.intrinsic_reward_generator.forward_model is not None:
                            self.intrinsic_reward_generator.forward_model = torch.compile(self.intrinsic_reward_generator.forward_model)
                        if hasattr(self.intrinsic_reward_generator, 'inverse_model') and self.intrinsic_reward_generator.inverse_model is not None:
                            self.intrinsic_reward_generator.inverse_model = torch.compile(self.intrinsic_reward_generator.inverse_model)
                        if hasattr(self.intrinsic_reward_generator, 'target_network') and self.intrinsic_reward_generator.target_network is not None:
                            self.intrinsic_reward_generator.target_network = torch.compile(self.intrinsic_reward_generator.target_network)
                        if hasattr(self.intrinsic_reward_generator, 'predictor_network') and self.intrinsic_reward_generator.predictor_network is not None:
                            self.intrinsic_reward_generator.predictor_network = torch.compile(self.intrinsic_reward_generator.predictor_network)
                        if self.debug:
                            print("[DEBUG] Compiled intrinsic reward models")
                
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
        self.transition_start_step = 0  # Initialize transition_start_step to avoid the attribute error
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
                obs_dim=obs_dim,
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

        if not self.use_pretraining or self.pretraining_completed:
            # Use base weights if pretraining is disabled or completed
            self.aux_task_manager.sr_weight = self.base_sr_weight
            self.aux_task_manager.rp_weight = self.base_rp_weight
            self.entropy_coef = self.base_entropy_coef # Also reset entropy
        elif self.in_transition_phase:
            # Calculate transition progress (0 to 1)
            current_step = self._true_training_steps()
            if self.pretraining_transition_steps > 0:
                 transition_progress = min(1.0, (current_step - self.transition_start_step) / self.pretraining_transition_steps)
            else:
                 transition_progress = 1.0 # Avoid division by zero

            # Interpolate weights during transition
            sr_weight = self.pretraining_sr_weight + transition_progress * (self.base_sr_weight - self.pretraining_sr_weight)
            rp_weight = self.pretraining_rp_weight + transition_progress * (self.base_rp_weight - self.pretraining_rp_weight)
            self.aux_task_manager.sr_weight = sr_weight
            self.aux_task_manager.rp_weight = rp_weight

            # Interpolate entropy coefficient
            pretraining_entropy = self.base_entropy_coef * self.pretraining_entropy_scale
            entropy_coef = pretraining_entropy + transition_progress * (self.base_entropy_coef - pretraining_entropy)
            self.entropy_coef = max(self.min_entropy_coef, entropy_coef)
        else:
            # Use pretraining weights if pretraining is active but not transitioning
            self.aux_task_manager.sr_weight = self.pretraining_sr_weight
            self.aux_task_manager.rp_weight = self.pretraining_rp_weight
            # Use higher entropy during active pretraining
            self.entropy_coef = max(self.min_entropy_coef, self.base_entropy_coef * self.pretraining_entropy_scale)

        # Decay entropy coefficient over time (independent of pretraining state after transition)
        if self.pretraining_completed:
             self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_coef_decay)

        # Log weight changes if debugging
        if self.debug:
            print(f"[DEBUG Aux Weights] SR: {self.aux_task_manager.sr_weight:.4f}, RP: {self.aux_task_manager.rp_weight:.4f}, Entropy: {self.entropy_coef:.6f}")
    
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
            rp_weight = getattr(self.aux_task_manager, "rp_weight", 0) if hasattr(self.aux_task_manager, 'rp_weight') else 0
            
            # Log the unweighted losses (more meaningful values for trends)
            auxiliary_metrics["AUX/state_representation_loss"] = sr_loss
            auxiliary_metrics["AUX/reward_prediction_loss"] = rp_loss
            auxiliary_metrics["AUX/total_loss"] = aux_loss
            
            # Also log the weights
            auxiliary_metrics["AUX/sr_weight"] = sr_weight
            auxiliary_metrics["AUX/rp_weight"] = rp_weight
            
            # Log actual last values from auxiliary manager if available
            if hasattr(self, 'aux_task_manager'): # Check existence on self
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
        
        # Skip intrinsic reward calculation when reward is 0 (placeholder value)
        # This is to avoid calculating intrinsic rewards before we have the actual reward from the environment
        # Instead, we'll use the update_experience_with_intrinsic_reward method after we have the real reward
        add_intrinsic = self.use_intrinsic_rewards and self.intrinsic_reward_generator is not None
        is_placeholder_reward = isinstance(reward, (int, float)) and reward == 0 or \
                              hasattr(reward, 'item') and reward.item() == 0
        
        if add_intrinsic and not is_placeholder_reward:
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
                intrinsic_reward = self.intrinsic_reward_generator.compute_intrinsic_reward(
                    state_tensor, action_tensor, next_state
                )

                if self.debug:
                    print(f"[DEBUG] Processing reward: {reward}")
                    
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
                    
                # Log that we're applying intrinsic rewards in debug mode
                if self.debug:
                    print(f"[DEBUG] Adding intrinsic reward: {adaptive_scale * intrinsic_reward[0]} to extrinsic: {extrinsic_reward}")
                
                # Add scaled intrinsic reward to extrinsic reward
                if isinstance(reward, torch.Tensor):
                    reward = reward + adaptive_scale * intrinsic_reward[0].to(reward.device)
                else:
                    reward = reward + adaptive_scale * intrinsic_reward[0].cpu().numpy()

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

    def get_action(self, obs, deterministic=False, return_features=False):
        """
        Get an action, log probability, and value for a given observation.
        
        Args:
            obs: Observation tensor or array
            deterministic: If True, return the most likely action without sampling
            return_features: If True, also return extracted features (for auxiliary tasks)
            
        Returns:
            Tuple containing (action, log_prob, value) and features if requested
        """
        # Convert observation to tensor if it's not already
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(self.device)
        
        # Ensure observation has batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        # Forward to algorithm's get_action method
        return self.algorithm.get_action(obs, deterministic, return_features)

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
        
        # --- Algorithm Update ---
        # Forward to specific algorithm implementation
        metrics = self.algorithm.update()
        
        # Record update time
        update_time = time.time() - update_start_time
        metrics['update_time'] = update_time
        
        # Ensure explained_variance is correctly calculated and reported for PPO
        if self.algorithm_type == "ppo" and metrics.get('explained_variance', 0.0) == 0.0:
            # If we need to recalculate it, we can look at the metrics data
            if hasattr(self.algorithm, 'memory') and \
               hasattr(self.algorithm.memory, 'values') and \
               hasattr(self.algorithm.memory, 'rewards') and \
               self.algorithm.memory.values is not None and \
               self.algorithm.memory.rewards is not None and \
               self.algorithm.memory.size > 1: 
                try:
                    # Access tensors only after validation
                    values = self.algorithm.memory.values[:self.algorithm.memory.size].detach().cpu().numpy()
                    returns = self.algorithm.memory.rewards[:self.algorithm.memory.size].detach().cpu().numpy()
                    
                    # Calculate explained variance: 1 - (var(y - pred) / var(y))
                    var_y = np.var(returns)
                    # Add small epsilon to denominator to prevent division by zero
                    explained_var = 1 - np.var(returns - values) / (var_y + 1e-8) 
                    metrics['explained_variance'] = explained_var
                    
                    if self.debug:
                        print(f"[DEBUG] Recalculated explained_variance: {explained_var:.4f}")
                            
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Error recalculating explained_variance: {str(e)}")
                        import traceback
                        traceback.print_exc()
            elif self.debug:
                 # Log why calculation was skipped
                 if not hasattr(self.algorithm, 'memory'):
                     print("[DEBUG] Skipping explained_variance: algorithm has no memory attribute.")
                 elif self.algorithm.memory.values is None:
                     print("[DEBUG] Skipping explained_variance: memory.values is None.")
                 elif self.algorithm.memory.rewards is None:
                     print("[DEBUG] Skipping explained_variance: memory.rewards is None.")
                 elif self.algorithm.memory.size <= 1:
                     print(f"[DEBUG] Skipping explained_variance: memory size ({self.algorithm.memory.size}) <= 1.")

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

        # --- Auxiliary Task Update ---
        if self.use_auxiliary_tasks and hasattr(self, 'aux_task_manager') and not self.test_mode:
            # For Stream mode, auxiliary tasks are already updated during store_experience
            # For PPO batch mode, we compute losses here using the accumulated history
            if self.algorithm_type == "ppo":
                try:
                    # Compute losses by sampling from the history buffers
                    aux_metrics = self.aux_task_manager.compute_losses()
                    
                    # Always include auxiliary metrics in the main metrics dictionary
                    metrics.update(aux_metrics)
                    # Calculate total aux loss if individual losses are present
                    metrics['aux_loss'] = aux_metrics.get('sr_loss', 0) + aux_metrics.get('rp_loss', 0)
                    if self.debug:
                        print(f"[DEBUG] Auxiliary task update (PPO): SR={aux_metrics.get('sr_loss', 0):.6f}, RP={aux_metrics.get('rp_loss', 0):.6f}")
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Error computing auxiliary losses (PPO): {e}")
                        import traceback
                        traceback.print_exc()

        # Log metrics
        if self.use_wandb:
            try:
                self._log_to_wandb(metrics)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error logging to wandb: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # Append losses to deques
        if 'actor_loss' in metrics:
            self.actor_losses.append(metrics['actor_loss'])
        if 'critic_loss' in metrics:
            self.critic_losses.append(metrics['critic_loss'])
        if 'entropy_loss' in metrics:
            self.entropy_losses.append(metrics['entropy_loss'])
        if 'total_loss' in metrics:
            self.total_losses.append(metrics['total_loss'])
        if 'sr_loss' in metrics: # Add aux loss tracking
            self.aux_sr_losses.append(metrics['sr_loss'])
        if 'rp_loss' in metrics: # Add aux loss tracking
            self.aux_rp_losses.append(metrics['rp_loss'])

        # Clean up tensors to reduce memory usage
        self._cleanup_tensors()

        return metrics

    def reset_auxiliary_tasks(self):
        """Reset auxiliary task history when episodes end"""
        if self.use_auxiliary_tasks:
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
            'timestamp': current_timestamp,
            'version': '2.0'  # Checkpoint format version
        })
        
        # Save pretraining state if enabled
        if self.use_pretraining:
            metadata['pretraining'] = {
                'completed': self.pretraining_completed,
                'in_transition': self.in_transition_phase,
                'transition_start_step': getattr(self, 'transition_start_step', 0),
                'pretraining_fraction': self.pretraining_fraction,
                'pretraining_transition_steps': self.pretraining_transition_steps,
                'base_sr_weight': self.base_sr_weight,
                'base_rp_weight': self.base_rp_weight,
                'pretraining_sr_weight': self.pretraining_sr_weight,
                'pretraining_rp_weight': self.pretraining_rp_weight
            }
            
        # Prepare the main checkpoint dictionary
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'metadata': metadata # Include updated metadata
        }
        
        # Include algorithm state if available
        if hasattr(self, 'algorithm') and hasattr(self.algorithm, 'get_state_dict'):
            checkpoint['algorithm_state'] = self.algorithm.get_state_dict()
            if self.debug:
                print(f"[DEBUG Trainer Save] Included algorithm state in checkpoint.")
        
        # Include curriculum state if manager is registered
        if hasattr(self, 'curriculum_manager') and self.curriculum_manager is not None:
             if hasattr(self.curriculum_manager, 'get_state'):
                 checkpoint['curriculum'] = self.curriculum_manager.get_state() # Use 'curriculum' key
                 if self.debug:
                     print(f"[DEBUG Trainer Save] Included curriculum state in checkpoint.")

        # Include auxiliary task state if manager exists
        if hasattr(self, 'aux_task_manager') and self.aux_task_manager is not None:
             if hasattr(self.aux_task_manager, 'get_state_dict'):
                 checkpoint['aux'] = self.aux_task_manager.get_state_dict() # Use 'aux' key
                 if self.debug:
                     print(f"[DEBUG Trainer Save] Included auxiliary task state in checkpoint.")

        # Include intrinsic reward state if generator exists
        if hasattr(self, 'intrinsic_reward_generator') and self.intrinsic_reward_generator is not None:
             if hasattr(self.intrinsic_reward_generator, 'get_state_dict'):
                 checkpoint['intrinsic'] = self.intrinsic_reward_generator.get_state_dict() # Use 'intrinsic' key
                 if self.debug:
                     print(f"[DEBUG Trainer Save] Included intrinsic reward state in checkpoint.")
                     
        # Include random states for reproducibility
        checkpoint['random_states'] = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate()
        }
        
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
            
        return model_path # Return the path where it was saved

    def load_models(self, model_path):
        """
        Load a comprehensive checkpoint to resume training from the exact state.
        
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
            
            # Check for checkpoint version to handle different formats
            metadata = checkpoint.get('metadata', {})
            version = metadata.get('version', '1.0')
            
            # Log loading info
            if self.debug:
                print(f"[DEBUG] Loading checkpoint version {version} from {model_path}")
            
            # ===== Restore Model Parameters =====
            if 'actor' in checkpoint and 'critic' in checkpoint:
                # Fix state dicts for compiled models
                actor_state = fix_compiled_state_dict(checkpoint['actor'])
                critic_state = fix_compiled_state_dict(checkpoint['critic'])
                
                # Load models with error handling for architecture mismatches
                actor_mismatches = load_partial_state_dict(self.actor, actor_state)
                critic_mismatches = load_partial_state_dict(self.critic, critic_state)
                
                if self.debug:
                    print(f"[DEBUG] Loaded actor and critic models")
                    if actor_mismatches > 0:
                        print(f"[DEBUG] {actor_mismatches} actor parameters could not be loaded")
                    if critic_mismatches > 0:
                        print(f"[DEBUG] {critic_mismatches} critic parameters could not be loaded")
            else:
                print(f"Warning: Checkpoint does not contain expected model parameters")
                return False
            
            # ===== Restore Algorithm State =====
            if 'algorithm_state' in checkpoint and checkpoint['algorithm_state']:
                algorithm_state = checkpoint['algorithm_state']
                
                # Restore the algorithm's internal state with type-specific handling
                if self.algorithm_type == "ppo":
                    # Restore PPO optimizers
                    if 'actor_optimizer' in algorithm_state:
                        try:
                            self.algorithm.actor_optimizer.load_state_dict(algorithm_state['actor_optimizer'])
                        except Exception as e:
                            print(f"Warning: Could not load actor optimizer state: {e}")
                            
                    if 'critic_optimizer' in algorithm_state:
                        try:
                            self.algorithm.critic_optimizer.load_state_dict(algorithm_state['critic_optimizer'])
                        except Exception as e:
                            print(f"Warning: Could not load critic optimizer state: {e}")
                    
                    # Restore PPO memory state if possible
                    if hasattr(self.algorithm, 'memory'):
                        self.algorithm.memory.pos = algorithm_state.get('memory_pos', 0)
                        self.algorithm.memory.size = algorithm_state.get('memory_size', 0)
                        self.algorithm.memory.full = algorithm_state.get('memory_full', False)
                        
                        if self.debug:
                            print(f"[DEBUG] Restored PPO memory state: pos={self.algorithm.memory.pos}, size={self.algorithm.memory.size}")
                    
                    # Restore episode returns tracking
                    if hasattr(self.algorithm, 'episode_returns') and 'episode_returns' in algorithm_state:
                        self.algorithm.episode_returns = deque(algorithm_state['episode_returns'], maxlen=self.algorithm.episode_returns.maxlen)
                        
                    if hasattr(self.algorithm, 'current_episode_rewards') and 'current_episode_rewards' in algorithm_state:
                        self.algorithm.current_episode_rewards = algorithm_state['current_episode_rewards']
                        
                elif self.algorithm_type == "streamac":
                    # Restore StreamAC optimizers
                    if 'actor_optimizer' in algorithm_state:
                        try:
                            self.algorithm.actor_optimizer.load_state_dict(algorithm_state['actor_optimizer'])
                        except Exception as e:
                            print(f"Warning: Could not load StreamAC actor optimizer state: {e}")
                            
                    if 'critic_optimizer' in algorithm_state:
                        try:
                            self.algorithm.critic_optimizer.load_state_dict(algorithm_state['critic_optimizer'])
                        except Exception as e:
                            print(f"Warning: Could not load StreamAC critic optimizer state: {e}")
                    
                    # Restore learning rates
                    if 'learning_rates' in algorithm_state:
                        lr_dict = algorithm_state['learning_rates']
                        self.algorithm.lr_actor = lr_dict.get('actor', self.lr_actor)
                        self.algorithm.lr_critic = lr_dict.get('critic', self.lr_critic)
                    
                    # Restore TD error buffer if it exists
                    if hasattr(self.algorithm, 'td_error_buffer') and 'td_error_buffer' in algorithm_state:
                        self.algorithm.td_error_buffer = deque(algorithm_state['td_error_buffer'], 
                                                            maxlen=self.algorithm.td_error_buffer.maxlen)
                    
                    # Restore update counters
                    if hasattr(self.algorithm, 'backtracking_count'):
                        self.algorithm.backtracking_count = algorithm_state.get('backtracking_count', 0)
                    if hasattr(self.algorithm, 'update_count'):
                        self.algorithm.update_count = algorithm_state.get('update_count', 0)
                        
                    if self.debug:
                        print(f"[DEBUG] Restored StreamAC state with learning rates - actor: {self.algorithm.lr_actor}, critic: {self.algorithm.lr_critic}")
            
            # ===== Restore Auxiliary Task State =====
            if self.use_auxiliary_tasks and 'aux' in checkpoint and checkpoint['aux']:
                aux_state = checkpoint['aux']
                
                # Only try to restore if we have an auxiliary task manager
                if hasattr(self, 'aux_task_manager'):
                    try:
                        # Restore model parameters and optimizer states
                        self.aux_task_manager.load_state_dict(aux_state)
                        
                        # Restore configuration values
                        if 'sr_weight' in aux_state:
                            self.aux_task_manager.sr_weight = aux_state['sr_weight']
                        if 'rp_weight' in aux_state:
                            self.aux_task_manager.rp_weight = aux_state['rp_weight']
                        if 'update_frequency' in aux_state:
                            self.aux_task_manager.update_frequency = aux_state['update_frequency']
                        if 'history_size' in aux_state:
                            self.aux_task_manager.history_size = aux_state['history_size']
                        if 'update_count' in aux_state:
                            self.aux_task_manager.update_count = aux_state['update_count']
                        if 'last_sr_loss' in aux_state:
                            self.aux_task_manager.last_sr_loss = aux_state['last_sr_loss']
                        if 'last_rp_loss' in aux_state:
                            self.aux_task_manager.last_rp_loss = aux_state['last_rp_loss']
                        if 'learning_mode' in aux_state:
                            self.aux_task_manager.learning_mode = aux_state['learning_mode']
                            
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
                            # Filter out configuration items and only pass the model state
                            model_state = {k: v for k, v in intrinsic_state.items() 
                                        if k not in ['intrinsic_reward_scale', 'curiosity_weight', 'rnd_weight', 'extrinsic_normalizer']}
                            if model_state:
                                self.intrinsic_reward_generator.load_state_dict(model_state)
                        except Exception as e:
                            print(f"Warning: Could not load intrinsic reward generator state: {e}")
                    
                    # Restore configuration values
                    if 'intrinsic_reward_scale' in intrinsic_state:
                        self.intrinsic_reward_scale = intrinsic_state['intrinsic_reward_scale']
                    
                    # Set component weights if the generator has those attributes
                    if hasattr(self.intrinsic_reward_generator, 'curiosity_weight') and 'curiosity_weight' in intrinsic_state:
                        self.intrinsic_reward_generator.curiosity_weight = intrinsic_state['curiosity_weight']
                    if hasattr(self.intrinsic_reward_generator, 'rnd_weight') and 'rnd_weight' in intrinsic_state:
                        self.intrinsic_reward_generator.rnd_weight = intrinsic_state['rnd_weight']
                    
                    # Restore extrinsic reward normalizer if it exists in the checkpoint
                    if 'extrinsic_normalizer' in intrinsic_state:
                        normalizer_state = intrinsic_state['extrinsic_normalizer']
                        if not hasattr(self, 'extrinsic_reward_normalizer'):
                            from intrinsic_rewards import ExtrinsicRewardNormalizer
                            self.extrinsic_reward_normalizer = ExtrinsicRewardNormalizer()
                        
                        try:
                            if 'mean' in normalizer_state:
                                self.extrinsic_reward_normalizer.mean = np.array(normalizer_state['mean'])
                            if 'var' in normalizer_state:
                                self.extrinsic_reward_normalizer.var = np.array(normalizer_state['var'])
                            if 'count' in normalizer_state:
                                self.extrinsic_reward_normalizer.count = normalizer_state['count']
                        except Exception as e:
                            print(f"Warning: Could not restore extrinsic reward normalizer: {e}")
                    
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
                    
                if self.debug:
                    print(f"[DEBUG] Restored training counters: steps={self.training_step_offset}, episodes={self.total_episodes_offset}")
            
            
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
            if 'metadata' in checkpoint and 'pretraining' in metadata.get('pretraining', {}):
                pretraining_state = metadata['pretraining']
                self.pretraining_completed = pretraining_state.get('completed', self.pretraining_completed)
                self.in_transition_phase = pretraining_state.get('in_transition', self.in_transition_phase)
                
                if 'transition_start_step' in pretraining_state:
                    self.transition_start_step = pretraining_state['transition_start_step']
                
                # Restore weight configurations
                self.pretraining_fraction = pretraining_state.get('pretraining_fraction', self.pretraining_fraction)
                self.pretraining_transition_steps = pretraining_state.get('pretraining_transition_steps', self.pretraining_transition_steps)
                self.base_sr_weight = pretraining_state.get('base_sr_weight', self.base_sr_weight)
                self.base_rp_weight = pretraining_state.get('base_rp_weight', self.base_rp_weight)
                self.pretraining_sr_weight = pretraining_state.get('pretraining_sr_weight', self.pretraining_sr_weight)
                self.pretraining_rp_weight = pretraining_state.get('pretraining_rp_weight', self.pretraining_rp_weight)
                
                # Explicitly check and print the pretraining_completed flag
                if self.debug:
                    pretraining_status = "completed" if self.pretraining_completed else ("transitioning" if self.in_transition_phase else "active")
                    print(f"[DEBUG] Restored pretraining state: {pretraining_status}")
                    print(f"[DEBUG] pretraining_completed = {self.pretraining_completed}")
            # If we explicitly find the 'pretraining_completed' at the top level, use it
            elif 'pretraining_completed' in checkpoint:
                self.pretraining_completed = checkpoint['pretraining_completed']
                if self.debug:
                    print(f"[DEBUG] Found top-level pretraining_completed = {self.pretraining_completed}")
            
            # ===== Resume WandB Run =====
            if self.use_wandb and 'wandb' in checkpoint and checkpoint['wandb']:
                wandb_info = checkpoint['wandb']
                
                try:
                    # Check if we need to resume a wandb run
                    if 'run_id' in wandb_info and wandb.run is None or (wandb.run and wandb.run.id != wandb_info['run_id']):
                        run_id = wandb_info['run_id']
                        project = wandb_info.get('project', 'rlbot-training')
                        entity = wandb_info.get('entity', None)
                        name = wandb_info.get('name', None)
                        
                        print(f"Resuming wandb run: {run_id}")
                        wandb.init(
                            project=project,
                            entity=entity,
                            id=run_id,
                            resume="must",
                            name=name
                        )
                        
                        if self.debug:
                            print(f"[DEBUG] Successfully resumed wandb run: {run_id}")
                except Exception as e:
                    print(f"Warning: Could not resume wandb run: {e}")
            
            print(f"Successfully loaded checkpoint from {model_path}")
            print(f"Resumed training at step {self.training_steps + self.training_step_offset}")
            return True
        
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False

    def _update_pretraining_state(self):
        """Update pretraining state and manage transition to regular training."""
        # Skip if pretraining is not enabled
        if not self.use_pretraining:
            return
        
        # If pretraining is already completed, nothing to do
        if self.pretraining_completed and not self.in_transition_phase:
            return
            
        # Calculate when pretraining should end based on episodes or steps
        pretraining_end_step = self._get_pretraining_end_step()
        current_step = self._true_training_steps()
        
        # Determine if we're in regular pretraining mode, transition phase, or fully completed
        if not self.pretraining_completed:
            # Check if we've reached the end of the pretraining phase
            if current_step >= pretraining_end_step:
                self.pretraining_completed = True
                self.in_transition_phase = True
                self.transition_start_step = current_step
                
                # Only print transition message when we just entered transition phase
                print(f"Transitioning from pretraining to regular training over next {self.pretraining_transition_steps} steps.")
                
                if self.use_wandb:
                    wandb.log({
                        'training/pretraining_completed': True,
                        'training/transition_started': True
                    }, step=current_step)

        # Handle transition phase
        elif self.in_transition_phase:
            # Check if transition phase is complete
            transition_progress = min(1.0, (current_step - self.transition_start_step) / self.pretraining_transition_steps)
            
            if transition_progress >= 1.0:
                # Mark pretraining as fully completed
                self.pretraining_completed = True
                self.in_transition_phase = False
                print("Pretraining phase completed. Switching to regular training.")
                
                # Reset auxiliary task weights to regular values
                if self.use_auxiliary_tasks:
                    self.aux_task_manager.sr_weight = self.base_sr_weight
                    self.aux_task_manager.rp_weight = self.base_rp_weight
                    
                # Reset entropy coefficient to base value
                self.entropy_coef = self.base_entropy_coef
                
            else:
                # Gradually transition auxiliary task weights
                if self.use_auxiliary_tasks:
                    sr_weight = self.pretraining_sr_weight + transition_progress * (self.base_sr_weight - self.pretraining_sr_weight)
                    rp_weight = self.pretraining_rp_weight + transition_progress * (self.base_rp_weight - self.pretraining_rp_weight)
                    self.aux_task_manager.sr_weight = sr_weight
                    self.aux_task_manager.rp_weight = rp_weight
                    
                    # Gradually transition entropy coefficient
                    pretraining_entropy = self.base_entropy_coef * self.pretraining_entropy_scale
                    entropy_coef = pretraining_entropy + transition_progress * (self.base_entropy_coef - pretraining_entropy)
                    self.entropy_coef = max(self.min_entropy_coef, entropy_coef)

                # During early pretraining, use higher values for auxiliary tasks and entropy
                elif not self.in_transition_phase and not self.pretraining_completed:
                    if self.use_auxiliary_tasks:
                        self.aux_task_manager.sr_weight = self.pretraining_sr_weight
                        self.aux_task_manager.rp_weight = self.pretraining_rp_weight
                
                    # Higher entropy during pretraining
                    self.entropy_coef = max(self.min_entropy_coef, self.base_entropy_coef * self.pretraining_entropy_scale)

    def update_experience_with_intrinsic_reward(self, state=None, action=None, next_state=None, done=None, reward=None, store_idx=None, env_id=None):
        """
        Calculate intrinsic reward and update stored experience
        
        Args:
            state: Current observation (renamed from obs to match main.py calls)
            action: Action taken
            next_state: Next observation (renamed from next_obs to match main.py calls)
            done: Done flag (now optional with default None)
            reward: Original extrinsic reward
            store_idx: Index in memory to update (for PPO)
            env_id: Environment ID for vectorized environments
        
        Returns:
            Total reward (extrinsic + intrinsic)
        """
        # If intrinsic rewards are disabled or generator is not initialized, return original reward
        if not self.use_intrinsic_rewards or self.intrinsic_reward_generator is None:
            return reward
        
        # Calculate intrinsic reward
        intrinsic_reward = 0.0
        
        try:
            # Convert inputs to tensors if needed
            if not isinstance(state, torch.Tensor):
                obs_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            else:
                obs_tensor = state
            
            if not isinstance(action, torch.Tensor):
                action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
            else:
                action_tensor = action
            
            if not isinstance(next_state, torch.Tensor):
                next_obs_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            else:
                next_obs_tensor = next_state
    
            # Add batch dimension if needed
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)
            if next_obs_tensor.dim() == 1:
                next_obs_tensor = next_obs_tensor.unsqueeze(0)
            
            # Compute intrinsic reward
            intrinsic_reward = self.intrinsic_reward_generator.compute_intrinsic_reward(
                obs_tensor, action_tensor, next_obs_tensor
            )

            # Update intrinsic reward model
            if done is not None:  # Only update if done flag is provided
                self.intrinsic_reward_generator.update(obs_tensor, action_tensor, next_obs_tensor, done)
        
            # Convert to scalar if tensor
            if isinstance(intrinsic_reward, torch.Tensor):
                intrinsic_reward = intrinsic_reward.item()
                
            # Scale intrinsic reward
            intrinsic_reward *= self.intrinsic_reward_scale
            
        except Exception as e:
            if self.debug:
                print(f"Error computing intrinsic reward: {e}")
            intrinsic_reward = 0.0
            
        # Calculate total reward
        total_reward = reward + intrinsic_reward
        
        # Update stored experience if index is provided (for PPO)
        if store_idx is not None and self.algorithm_type == "ppo":
            # Convert to tensor if needed
            if not isinstance(total_reward, torch.Tensor):
                total_reward_tensor = torch.tensor(total_reward, dtype=torch.float32, device=self.device)
            else:
                total_reward_tensor = total_reward
        
            # Update the experience in memory
            self.algorithm.store_experience_at_idx(store_idx, reward=total_reward_tensor)
    
        return total_reward

    def _cleanup_tensors(self):
        """Clean up any cached tensors to reduce memory usage"""
        # Clear PyTorch cache occasionally
        if torch.cuda.is_available():
            torch.cuda.empty_cache()