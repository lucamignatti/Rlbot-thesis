import sys
import gc
import psutil
import heapq
import os
import time
import traceback
import argparse
import signal
import torch
import wandb
import multiprocessing as mp
from queue import Empty as QueueEmpty # Import Empty exception
import numpy as np
from collections import Counter
from collections.abc import Sized, Iterable
# Removed unused rlgym imports
from model_architectures import (
    BasicModel, SimBa, SimbaV2, SimbaV2Shared # Add SimbaV2Shared
    # Removed unused functions: fix_compiled_state_dict, extract_model_dimensions, load_partial_state_dict
)
from observation import ActionStacker
from training import Trainer
# Removed unused algorithm imports
# Removed unused concurrent.futures
from tqdm import tqdm # Keep tqdm for the manager process
from typing import Optional # Keep only Optional
from envs.factory import get_env
from envs.vectorized import VectorizedEnv
from curriculum import create_curriculum
from curriculum.manual import ManualCurriculumManager

# --- TQDM Manager Process ---
class TqdmManager:
    """Manages a tqdm progress bar in a separate process."""
    def __init__(self, queue: mp.Queue, total: Optional[int], desc: str, bar_format: str, dynamic_ncols: bool):
        self.queue = queue
        self.total = total
        self.desc = desc
        self.bar_format = bar_format
        self.dynamic_ncols = dynamic_ncols
        self.process: Optional[mp.Process] = None
        self._initial_postfix = {} # Store initial postfix if sent early

    def _run(self):
        """The target function for the tqdm process."""
        progress_bar = None
        try:
            # Wait for initial postfix if not already received
            while not self._initial_postfix:
                 try:
                     msg = self.queue.get(timeout=0.1)
                     if isinstance(msg, dict):
                         self._initial_postfix = msg
                     elif msg is None: # Check for early termination
                         return
                 except QueueEmpty:
                     pass # Keep waiting for initial postfix

            # Initialize tqdm instance
            progress_bar = tqdm(
                total=self.total,
                desc=self.desc,
                bar_format=self.bar_format,
                dynamic_ncols=self.dynamic_ncols,
                postfix=self._initial_postfix # Set initial postfix
            )

            while True:
                try:
                    # Wait for messages from the main process
                    msg = self.queue.get() # Blocking get

                    if msg is None: # Termination signal
                        break
                    elif isinstance(msg, int): # Update progress bar count
                        progress_bar.update(msg)
                    elif isinstance(msg, float): # Update progress bar total (for time-based)
                        progress_bar.n = min(int(msg), progress_bar.total if progress_bar.total else 0)
                        progress_bar.refresh()
                    elif isinstance(msg, dict): # Update postfix dictionary
                        progress_bar.set_postfix(msg)
                    elif isinstance(msg, tuple) and msg[0] == "set_total": # Command to update total
                        progress_bar.total = msg[1]
                        progress_bar.refresh()


                except EOFError:
                    print("[TQDM Manager] Queue closed unexpectedly.")
                    break
                except Exception as e:
                    print(f"[TQDM Manager] Error: {e}")
                    traceback.print_exc()
                    # Continue processing if possible
        finally:
            if progress_bar:
                progress_bar.close()

    def start(self):
        """Starts the tqdm manager process."""
        if self.process is None or not self.process.is_alive():
            self.process = mp.Process(target=self._run, daemon=True)
            self.process.start()
    def stop(self):
        """Signals the tqdm manager process to stop and waits for it."""
        if self.process and self.process.is_alive():
            try:
                self.queue.put(None)
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
            except Exception as e:
                 print(f"[TQDM Manager] Error during stop: {e}")
        self.process = None

# --- End TQDM Manager Process ---


# Global queue for TQDM communication
tqdm_queue: Optional[mp.Queue] = None
tqdm_manager_instance: Optional[TqdmManager] = None


def log_memory_usage(step=0, location=""):
    """Log memory usage at a specific point in the code"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"[MEMORY] Step {step}, {location}: {memory_info.rss / 1024 / 1024:.2f} MB")

    # Get counts of common types
    counts = Counter(type(obj).__name__ for obj in gc.get_objects())
    top_types = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"[MEMORY] Top object types: {top_types}")

# Added memory inspection function
def print_memory_objects(top_n=5):
    """
    Collects all objects tracked by the GC, finds the largest ones using heapq,
    and prints details for them.
    """
    print("\n--- Memory Object Report (End of Training) ---")
    # Force garbage collection to get a cleaner state
    gc.collect()

    # Use heapq.nlargest to find the top_n largest objects directly
    # This avoids sorting the entire list of potentially millions of objects
    try:
        largest_objects = heapq.nlargest(top_n, (
            (sys.getsizeof(obj), obj) for obj in gc.get_objects()
        ), key=lambda x: x[0])
    except Exception as e:
        print(f"Error collecting or finding largest objects: {e}")
        largest_objects = [] # Ensure largest_objects is defined

    print(f"\n--- Top {top_n} Largest Objects (Found via heapq) ---")

    for i, (size, obj) in enumerate(largest_objects):
        obj_type = type(obj)
        # --- Added try-except block for repr() ---
        try:
            obj_repr = repr(obj)[:100] # Limit representation length
        except Exception as e:
            obj_repr = f"<repr error: {e}>"
        # --- End added block ---
        obj_len = len(obj) if isinstance(obj, Sized) else 'N/A'
        obj_iterable = isinstance(obj, Iterable)

        print(f"\n#{i+1}: Size={size} bytes")
        print(f"  Type: {obj_type}")
        print(f"  Iterable: {obj_iterable}")
        print(f"  Length: {obj_len}")
        print(f"  Repr: {obj_repr}...")

    print("--- End Memory Object Report ---\n")


def run_training(
    actor,
    critic,
    device,
    num_envs: int,
    model_path_to_load: Optional[str] = None,
    total_episodes: Optional[int] = None,
    training_time: Optional[float] = None,
    render: bool = False,
    update_interval: int = 1000,
    use_wandb: bool = False,
    debug: bool = False,
    use_compile: bool = True,
    use_amp: bool = False,
    save_interval: int = 200,
    output_path: Optional[str] = None,
    use_curriculum: bool = False,
    stage: Optional[int] = None,
    # Hyperparameters
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    critic_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 10,
    use_weight_clipping: bool = False,
    weight_clip_kappa: float = 1.0,
    batch_size: int = 64,
    aux_amp: bool = False,
    aux_scale: float = 0.005,
    algorithm: str = "ppo",
    auxiliary: bool = True,
    sr_weight: float = 1.0,
    rp_weight: float = 1.0,
    test: bool = False,
    # Pre-training parameters
    use_pretraining: bool = False,
    pretraining_fraction: float = 0.1,
    pretraining_sr_weight: float = 10.0,
    pretraining_rp_weight: float = 5.0,
    pretraining_transition_steps: int = 1000,
    # Intrinsic rewards parameters
    use_intrinsic_rewards: bool = True,
    intrinsic_reward_scale: float = 0.1,
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
    # Add reward scaling parameters
    use_reward_scaling: bool = True,
    reward_scaling_G_max: float = 10.0,
    # Add TQDM queue
    tqdm_q: Optional[mp.Queue] = None
):
    """
    Main training loop.
    """
    # Performance optimizations
    os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 4))
    os.environ['KMP_BLOCKTIME'] = '0'
    os.environ['KMP_SETTINGS'] = '0'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'

    # Allow indefinite training when no episode count or time limit is provided
    if total_episodes is None and training_time is None:
        print("No episodes or time limit provided - training will continue indefinitely until interrupted")

    actor.to(device)
    # If critic is a separate instance, move it too
    if actor is not critic:
        critic.to(device)

    # Initialize action stacker for keeping track of previous actions
    action_stacker = ActionStacker(stack_size=5, action_size=actor.action_shape)

    # Initialize the Trainer
    trainer = Trainer(
        actor,
        critic,
        # Pass training_step_offset=0 initially, load_models will update it
        training_step_offset=0,
        algorithm_type=algorithm,  # Use the specified algorithm
        action_space_type="discrete",  # For RLBot, we use discrete actions
        action_dim=actor.action_shape,
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
        use_weight_clipping=use_weight_clipping,
        weight_clip_kappa=weight_clip_kappa,
        use_wandb=use_wandb,
        debug=debug,
        use_compile=use_compile,
        use_amp=use_amp,
        use_auxiliary_tasks=auxiliary,
        sr_weight=sr_weight * aux_scale,
        rp_weight=rp_weight * aux_scale,
        aux_amp=aux_amp,
        use_pretraining=use_pretraining,
        pretraining_fraction=pretraining_fraction,
        pretraining_sr_weight=pretraining_sr_weight,
        pretraining_rp_weight=pretraining_rp_weight,
        pretraining_transition_steps=pretraining_transition_steps,
        # New intrinsic reward parameters
        use_intrinsic_rewards=use_intrinsic_rewards,
        intrinsic_reward_scale=intrinsic_reward_scale,
        curiosity_weight=curiosity_weight,
        rnd_weight=rnd_weight,
        # StreamAC specific parameters
        adaptive_learning_rate=adaptive_learning_rate,
        target_step_size=target_step_size,
        backtracking_patience=backtracking_patience,
        backtracking_zeta=backtracking_zeta,
        min_lr_factor=min_lr_factor,
        max_lr_factor=max_lr_factor,
        use_obgd=use_obgd,
        stream_buffer_size=stream_buffer_size,
        use_sparse_init=use_sparse_init,
        update_freq=update_freq,
        # Pass reward scaling parameters to Trainer
        use_reward_scaling=use_reward_scaling,
        reward_scaling_G_max=reward_scaling_G_max,
    )

    # --- MODEL LOADING LOGIC ---
    if model_path_to_load:
        print(f"Attempting to load model from: {model_path_to_load}")
        if trainer.load_models(model_path_to_load):
             print(f"Successfully loaded model and state from {model_path_to_load}")
             # Add a debug print to confirm the trainer's state AFTER loading
             if debug:
                 print(f"[DEBUG] Trainer state after load: pretraining_completed={trainer.pretraining_completed}")
             print(f"Continuing from training step {trainer.training_step_offset}")
        else:
             print(f"Failed to load model from {model_path_to_load}. Starting fresh.")
             trainer.training_step_offset = 0
    # --- END MODEL LOADING LOGIC ---

    # Set total_episodes and training_time attributes for pretraining calculation
    if total_episodes is not None:
        trainer.total_episode_target = total_episodes
    if training_time is not None:
        trainer.training_time_target = training_time

    #  Use train mode for training and eval for testing
    if test:
        trainer.actor.eval()
        # Only set critic to eval if it's a separate instance
        if trainer.actor is not trainer.critic:
            trainer.critic.eval()
    else:
        trainer.actor.train()
        if trainer.actor is not trainer.critic:
            trainer.critic.train()

    # Add test mode flag to trainer
    if test:
        trainer.test_mode = True

    # Initialize curriculum manager AFTER trainer is created and potentially loaded
    curriculum_manager = None
    # Use the trainer's potentially updated pretraining status
    effective_use_pretraining = trainer.use_pretraining and not trainer.pretraining_completed
    if use_curriculum: # Check the original arg flag first
        try:
            # If a specific stage is requested, use the manual curriculum manager
            if stage is not None:
                # Get all stages by temporarily creating the full curriculum
                temp_curriculum = create_curriculum(
                    debug=debug,
                    use_wandb=use_wandb,
                    lr_actor=trainer.lr_actor,
                    lr_critic=trainer.lr_critic,
                    use_pretraining=effective_use_pretraining
                )

                # Create a manual curriculum with just the selected stage
                curriculum_manager = ManualCurriculumManager(
                    stages=temp_curriculum.stages,
                    stage_index=stage,
                    debug=debug,
                    use_wandb=use_wandb
                )

                if debug:
                    print(f"[DEBUG] Using manual curriculum with stage {stage}: '{curriculum_manager.current_stage.name}'")
            else:
                # Use the automatic curriculum system
                curriculum_manager = create_curriculum(
                    debug=debug,
                    use_wandb=use_wandb,
                    lr_actor=trainer.lr_actor, # Use potentially loaded LR
                    lr_critic=trainer.lr_critic, # Use potentially loaded LR
                    # IMPORTANT: Use the trainer's state to decide if pretraining is active
                    use_pretraining=effective_use_pretraining
                )

            # Register trainer with curriculum manager
            curriculum_manager.register_trainer(trainer)
            # Assign curriculum manager back to trainer
            trainer.curriculum_manager = curriculum_manager

            # If loading occurred, ensure curriculum state matches loaded state
            if model_path_to_load and hasattr(trainer, '_loaded_curriculum_state'):
                 # If load_models stored the state, re-apply it AFTER manager creation
                 print("[DEBUG] Re-applying loaded curriculum state to manager...")
                 curriculum_manager.load_state(trainer._loaded_curriculum_state)
                 # Re-register again just in case load_state overwrites something
                 curriculum_manager.register_trainer(trainer)
                 del trainer._loaded_curriculum_state # Clean up temporary state

            if debug:
                print(f"[DEBUG] Initialized curriculum with {len(curriculum_manager.stages)} stages.")
                print(f"[DEBUG] Effective pretraining for curriculum: {effective_use_pretraining}")
                print(f"[DEBUG] Starting at stage: {curriculum_manager.current_stage.name}")

        except Exception as e:
            print(f"Failed to initialize curriculum: {e}")
            traceback.print_exc()
            print("Continuing without curriculum")
            use_curriculum = False
            trainer.curriculum_manager = None # Ensure trainer doesn't have stale reference
    else:
         trainer.curriculum_manager = None # Ensure it's None if not used

    # Use vectorized environment for parallel data collection
    env_class = VectorizedEnv

    # Create environment constructor arguments based on the class
    env_args = {
        'num_envs': num_envs,
        'render': render,
        'action_stacker': action_stacker,
        'curriculum_manager': curriculum_manager,
        'debug': debug
    }

    vec_env = env_class(**env_args)

    # For time-based training, we'll need to know when we started.
    start_time = time.time()

    # Set up the progress bar parameters for the TqdmManager
    tqdm_params = {
        "total": None,
        "desc": "Training",
        "bar_format": '{desc}: [{elapsed}] |{bar:30}| {postfix}'
    }

    if training_time is not None:
        tqdm_params["total"] = int(training_time)
        tqdm_params["desc"] = "Time"
        tqdm_params["bar_format"] = '{desc}: {percentage:3.0f}% [{elapsed}<{remaining}] |{bar}| {postfix}'
    elif total_episodes is not None:
        tqdm_params["total"] = total_episodes
        tqdm_params["desc"] = "Episodes"
        tqdm_params["bar_format"] = '{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%|{bar}| {postfix}'
    # else: use defaults for indefinite training

    # This dictionary holds statistics we'll display in the progress bar.
    stats_dict = {
        "Device": device,
        "Envs": num_envs,
        "Episodes": 0,  # Total episodes completed
        "Algorithm": algorithm  # Display which algorithm is being used
    }

    # Add algorithm-specific progress metrics
    if algorithm == "ppo":
        stats_dict["Exp"] = f"0/{update_interval}"  # Experience counter for PPO
    else:  # StreamAC or SAC
        stats_dict["Steps"] = 0  # Step counter
        stats_dict["Updates"] = 0  # Update counter

    # Add common metrics
    stats_dict.update({
        "Reward": "0.0",  # Average reward per episode
        "PLoss": "0.0",  # Actor loss
        "VLoss": "0.0",  # Critic loss
        "Entropy": "0.0",  # Entropy loss
        "Steps/s": "0.0"  # Steps per second
    })

    # Add auxiliary task metrics if enabled
    if auxiliary:
        stats_dict.update({
            "SR_Loss": "0.0",  # State reconstruction loss
            "RP_Loss": "0.0",  # Reward prediction loss
        })

    # Add pretraining info if enabled
    if use_pretraining:
        pretraining_end_step = int(total_episodes * pretraining_fraction) if total_episodes else int(5000 * pretraining_fraction)
        stats_dict.update({
            "Mode": "PreTraining",
            "PT_Progress": f"0/{pretraining_end_step}"
        })

    # Add curriculum info if enabled
    if curriculum_manager:
        curriculum_stats = curriculum_manager.get_curriculum_stats()
        stats_dict.update({
            "Stage": curriculum_stats["current_stage_name"],
            "Diff": f"{curriculum_stats['difficulty_level']:.2f}"
        })

    # Add StreamAC specific info if using that algorithm
    if algorithm == "streamac":
        stats_dict.update({
            "StepSize": "0.0",  # Effective step size
            "ActorLR": f"{lr_actor:.6f}",  # Display actor learning rate (will change with adaptive LR)
            "CriticLR": f"{lr_critic:.6f}"  # Display critic learning rate
        })
    # Add SAC specific info
    elif algorithm == "sac":
         stats_dict.update({
             "Alpha": f"{trainer.algorithm.alpha:.4f}" if hasattr(trainer.algorithm, 'alpha') else "N/A",
             "Buffer": "0"
         })

    # Send initial postfix to TQDM manager
    if tqdm_q:
        tqdm_q.put(stats_dict.copy())

    # Initialize variables to track training progress.
    collected_experiences = 0
    total_episodes_so_far = 0
    total_env_steps = 0 # Initialize total environment steps counter
    last_update_time = time.time()
    last_save_episode = 0
    last_intrinsic_reset_episode = 0 # Track last episode where intrinsic models were reset

    # Variables to track steps per second
    steps_per_second = 0
    last_steps_time = time.time()
    last_steps_count = 0

    # Initialize episode rewards lazily within the loop
    episode_rewards = {}

    # Add a list to track rewards of completed episodes between PPO updates
    completed_episode_rewards_since_last_update = []

    last_progress_update = start_time  # For updating time-based progress bar
    should_continue = True  # Initialize the control variable

    try:
        # Let's keep training until it's time to stop
        while should_continue:
            current_time = time.time()
            elapsed = current_time - start_time

            # Figure out if we should keep going based on time or episode count, or indefinitely
            if training_time is not None:
                # Using a time-based training schedule - update progress bar once per second
                if current_time - last_progress_update >= 1.0:
                    if tqdm_q:
                        tqdm_q.put(float(elapsed)) # Send elapsed time as float for 'n' update
                    last_progress_update = current_time

                should_continue = elapsed < training_time
            elif total_episodes is not None:
                # Using an episode-based schedule
                should_continue = total_episodes_so_far < total_episodes
            else:
                # Neither time nor episodes specified - run indefinitely
                should_continue = True
                # Update progress indicators periodically
                if current_time - last_progress_update >= 1.0:
                    last_progress_update = current_time
                    if tqdm_q:
                        tqdm_q.put(stats_dict.copy())

            if not should_continue:
                break

            # Batch up observations from all environments for efficiency
            all_obs_list = []
            all_env_indices = []
            all_agent_ids = []
            # Organize observations into lists for batch processing.
            for env_idx, obs_dict in enumerate(vec_env.obs_dicts):
                 # Lazily initialize episode rewards dictionary for this env if needed
                 if env_idx not in episode_rewards:
                     episode_rewards[env_idx] = {}

                 for agent_id, obs in obs_dict.items():
                    all_obs_list.append(obs)
                    all_env_indices.append(env_idx)
                    all_agent_ids.append(agent_id)
                    # Lazily initialize agent reward tracking
                    if agent_id not in episode_rewards[env_idx]:
                         episode_rewards[env_idx][agent_id] = 0.0
                         if debug: print(f"[DEBUG] Initialized episode_rewards for {agent_id} in env {env_idx} (during obs collection)")


            # Only proceed if we have observations. (It's possible all environments ended at the same time.)
            actions_dict_list = [{} for _ in range(num_envs)]  # Initialize outside the block to avoid unbound issues
            store_indices = None # Indices where experiences are stored (for PPO)

            if len(all_obs_list) > 0:
                obs_batch = torch.FloatTensor(np.stack(all_obs_list)).to(device)
                with torch.no_grad():
                    # Get actions from the networks - now returns dummy values for PPO
                    # For shared model, get_action in algorithm handles requesting only actor output
                    action_batch, log_prob_batch, value_batch, features_batch = trainer.get_action(obs_batch, return_features=True)

                # --- Batch Store Initial Experience (PPO) ---
                if algorithm == "ppo" and not test:
                    # Store obs, action, log_prob, value (placeholder) in batch
                    store_indices = trainer.store_initial_batch(
                        obs_batch, action_batch, log_prob_batch, value_batch
                    )
                    collected_experiences += len(obs_batch) # Increment collected experiences count
                    if debug and collected_experiences % 100 == 0:
                         print(f"[DEBUG PPO] Collected {collected_experiences} experiences (batch size {len(obs_batch)})")

                # --- Prepare actions for environment step ---
                # Convert actions to CPU numpy arrays for the environment step
                # Handle both tensor and non-tensor actions returned by get_action
                if isinstance(action_batch, torch.Tensor):
                    action_batch_np = action_batch.cpu().numpy()
                else:
                    # If not a tensor, assume it's a list/array already compatible
                    action_batch_np = np.array(action_batch)


                # Organize actions into a list of dictionaries, one for each environment.
                for i, action_np in enumerate(action_batch_np):
                    env_idx = all_env_indices[i]
                    agent_id = all_agent_ids[i]
                    actions_dict_list[env_idx][agent_id] = action_np # Use CPU numpy action

                    # --- Store Experience (Non-PPO or Test Mode) ---
                    # If not PPO or in test mode, use the old single-store method
                    if (algorithm != "ppo" or test) and not test: # Only store if not testing
                        # Pass original tensors (potentially GPU) to store_experience
                        trainer.store_experience(
                            all_obs_list[i],
                            action_batch[i], # Original action tensor/value
                            log_prob_batch[i],
                            0,  # Placeholder reward, updated after environment step
                            value_batch[i],
                            False,  # Placeholder done flag, updated after environment step
                            env_id=env_idx
                        )
                        collected_experiences += 1 # Increment for non-PPO algorithms too


            # Step all environments forward in parallel - optimized implementation
            results, dones, episode_counts = vec_env.step(actions_dict_list)

            # Increment total environment steps counter
            current_batch_env_steps = len(all_obs_list) # Number of agent steps in this batch
            total_env_steps += current_batch_env_steps

            # Update trainer's total_env_steps attribute for logging
            trainer.total_env_steps = total_env_steps

            # Calculate steps per second
            current_time = time.time()
            elapsed_since_last_calc = current_time - last_steps_time
            if elapsed_since_last_calc >= 1.0:  # Update once per second
                steps_this_period = total_env_steps - last_steps_count
                steps_per_second = steps_this_period / elapsed_since_last_calc
                last_steps_time = current_time
                last_steps_count = total_env_steps
                stats_dict["Steps/s"] = f"{steps_per_second:.1f}"
                # Send updated stats_dict to TQDM manager (includes Steps/s)
                if tqdm_q:
                    tqdm_q.put(stats_dict.copy())


            # --- Process Results and Update Rewards/Dones (PPO Batch) ---
            if algorithm == "ppo" and not test and store_indices is not None:
                batch_rewards = []
                batch_dones = []
                batch_next_obs = []
                batch_env_ids_for_intrinsic = []
                batch_store_indices_for_update = [] # Indices corresponding to rewards/dones

                exp_idx_map = {i: idx for i, idx in enumerate(store_indices.tolist())} # Map batch index to buffer index

                current_exp_idx = 0
                for env_idx, result in enumerate(results):
                    # Handle return formats
                    if isinstance(result, tuple):
                        if len(result) == 5: _, next_obs_dict, reward_dict, terminated_dict, truncated_dict = result
                        else: next_obs_dict, reward_dict, terminated_dict, truncated_dict = result
                    else: continue

                    # Ensure episode_rewards dict exists for this env_idx
                    if env_idx not in episode_rewards: episode_rewards[env_idx] = {}

                    for agent_id in reward_dict.keys():
                        if current_exp_idx < len(all_obs_list): # Ensure we don't go out of bounds
                            extrinsic_reward = reward_dict[agent_id]
                            done = terminated_dict[agent_id] or truncated_dict[agent_id]
                            next_obs = next_obs_dict.get(agent_id, None)

                            batch_rewards.append(extrinsic_reward)
                            batch_dones.append(done)
                            if next_obs is not None: batch_next_obs.append(next_obs)
                            batch_env_ids_for_intrinsic.append(env_idx)
                            # Get the correct buffer index for this agent step
                            buffer_idx = exp_idx_map.get(current_exp_idx)
                            if buffer_idx is not None:
                                batch_store_indices_for_update.append(buffer_idx)

                            # Accumulate rewards for episode tracking (using extrinsic for now)
                            # Defensive check/initialization before accumulating
                            if agent_id not in episode_rewards[env_idx]:
                                episode_rewards[env_idx][agent_id] = 0.0
                                if debug: print(f"[DEBUG] Initialized episode_rewards for {agent_id} in env {env_idx} (during PPO result processing)")
                            episode_rewards[env_idx][agent_id] += extrinsic_reward

                            if done:
                                action_stacker.reset_agent(agent_id)
                                trainer.reset_auxiliary_tasks()

                            current_exp_idx += 1
                        else:
                             if debug: print(f"[DEBUG] Warning: current_exp_idx ({current_exp_idx}) exceeded all_obs_list length ({len(all_obs_list)})")


                # Ensure all lists have the same length before proceeding
                min_len = min(len(batch_rewards), len(batch_dones), len(batch_store_indices_for_update))
                if len(batch_rewards) != min_len: batch_rewards = batch_rewards[:min_len]
                if len(batch_dones) != min_len: batch_dones = batch_dones[:min_len]
                if len(batch_store_indices_for_update) != min_len: batch_store_indices_for_update = batch_store_indices_for_update[:min_len]
                if len(batch_next_obs) != min_len: batch_next_obs = batch_next_obs[:min_len] # Need next_obs for intrinsic
                if len(batch_env_ids_for_intrinsic) != min_len: batch_env_ids_for_intrinsic = batch_env_ids_for_intrinsic[:min_len]


                if min_len > 0:
                    batch_rewards_np = np.array(batch_rewards, dtype=np.float32)
                    batch_dones_np = np.array(batch_dones, dtype=bool)
                    batch_store_indices_tensor = torch.tensor(batch_store_indices_for_update, dtype=torch.long, device=device)

                    final_rewards = batch_rewards_np

                    # --- Batch Intrinsic Reward Calculation (PPO) ---
                    if trainer.use_intrinsic_rewards and len(batch_next_obs) == min_len:
                        if debug: print(f"[DEBUG PPO] Calculating intrinsic rewards for batch size {min_len}")
                        try:
                            # Prepare inputs for batch intrinsic calculation
                            intrinsic_states = np.stack(all_obs_list[:min_len]) # Use the initial states for this batch
                            intrinsic_actions = action_batch_np[:min_len] # Use the actions taken
                            intrinsic_next_states = np.stack(batch_next_obs)
                            intrinsic_env_ids = batch_env_ids_for_intrinsic[:min_len]

                            intrinsic_rewards = trainer.calculate_intrinsic_rewards_batch(
                                states=intrinsic_states,
                                actions=intrinsic_actions,
                                next_states=intrinsic_next_states,
                                env_ids=intrinsic_env_ids
                            )
                            # Add intrinsic rewards to extrinsic rewards
                            final_rewards = batch_rewards_np + intrinsic_rewards
                            if debug and min_len > 0:
                                print(f"[DEBUG PPO] Intrinsic reward example: Extrinsic={batch_rewards_np[0]:.4f}, Intrinsic={intrinsic_rewards[0]:.4f}, Total={final_rewards[0]:.4f}")

                        except Exception as e:
                            if debug:
                                print(f"[DEBUG PPO] Error calculating batch intrinsic rewards: {e}")
                                traceback.print_exc()
                            # Use only extrinsic rewards if intrinsic calculation fails
                            final_rewards = batch_rewards_np

                    # --- Update Rewards and Dones in Buffer (PPO) ---
                    trainer.update_rewards_dones_batch(
                        batch_store_indices_tensor,
                        torch.tensor(final_rewards, dtype=torch.float32, device=device),
                        torch.tensor(batch_dones_np, dtype=torch.bool, device=device)
                    )
                elif debug:
                     print("[DEBUG PPO] No valid experiences to update rewards/dones for.")


            # --- Process Results (Non-PPO or Test Mode) ---
            elif algorithm != "ppo" or test:
                exp_idx = 0
                for env_idx, result in enumerate(results):
                    # Handle return formats
                    if isinstance(result, tuple):
                        if len(result) == 5: _, next_obs_dict, reward_dict, terminated_dict, truncated_dict = result
                        else: next_obs_dict, reward_dict, terminated_dict, truncated_dict = result
                    else: continue

                    # Ensure episode_rewards dict exists for this env_idx
                    if env_idx not in episode_rewards: episode_rewards[env_idx] = {}

                    for agent_id in reward_dict.keys():
                        if exp_idx < len(all_obs_list): # Check bounds
                            extrinsic_reward = reward_dict[agent_id]
                            done = terminated_dict[agent_id] or truncated_dict[agent_id]
                            next_obs = next_obs_dict.get(agent_id, None)

                            # For StreamAC/SAC, update the stored experience with actual reward/done
                            # This part remains similar, but uses the single store_experience_at_idx
                            # or relies on StreamAC's internal buffer update mechanism.
                            if not test: # Only update if not testing
                                if algorithm == "streamac":
                                     # StreamAC updates happen within store_experience, called earlier
                                     # We might need to update the *last* stored reward/done if store_experience
                                     # doesn't handle the next_state logic correctly.
                                     # Let's assume StreamAC's store_experience handles this for now.
                                     # We still need to track episode rewards.
                                     # Calculate total reward (extrinsic + intrinsic) if needed for tracking
                                     total_reward_for_tracking = extrinsic_reward
                                     if next_obs is not None and trainer.use_intrinsic_rewards:
                                         # Use the single update method to get the combined reward
                                         total_reward_for_tracking = trainer.update_experience_with_intrinsic_reward(
                                             state=all_obs_list[exp_idx],
                                             action=action_batch[exp_idx], # Original action
                                             next_state=next_obs,
                                             reward=extrinsic_reward, # Original extrinsic
                                             env_id=env_idx,
                                             done=done # Pass done flag
                                             # store_idx is None for StreamAC here
                                         )
                                     # Defensive check/initialization before accumulating reward
                                     if agent_id not in episode_rewards[env_idx]:
                                         episode_rewards[env_idx][agent_id] = 0.0
                                         if debug: print(f"[DEBUG] Initialized episode_rewards for {agent_id} in env {env_idx} (during {algorithm} result processing)")
                                     episode_rewards[env_idx][agent_id] += total_reward_for_tracking

                                elif algorithm == "sac":
                                     # SAC uses a replay buffer, update the last stored experience
                                     # This assumes SAC's store_experience adds placeholders first
                                     # We need a way to find the index or update the last item.
                                     # Let's use store_experience_at_idx if SAC supports it,
                                     # otherwise, this needs refinement based on SAC's buffer.
                                     # Assuming SAC's store_experience handles it for now.
                                     # Defensive check/initialization before accumulating reward
                                     if agent_id not in episode_rewards[env_idx]:
                                         episode_rewards[env_idx][agent_id] = 0.0
                                         if debug: print(f"[DEBUG] Initialized episode_rewards for {agent_id} in env {env_idx} (during {algorithm} result processing)")
                                     episode_rewards[env_idx][agent_id] += extrinsic_reward # Track extrinsic for now

                            if done:
                                action_stacker.reset_agent(agent_id)
                                trainer.reset_auxiliary_tasks()

                            exp_idx += 1


            # Check if any episodes have completed.
            newly_completed_episodes = sum(dones)
            if newly_completed_episodes > 0:
                # Update progress bar for episodes-based training
                if training_time is None and tqdm_q:
                    tqdm_q.put(int(newly_completed_episodes)) # Send integer update count

                total_episodes_so_far += newly_completed_episodes
                stats_dict["Episodes"] = total_episodes_so_far
                # Send updated stats_dict to TQDM manager
                if tqdm_q:
                    tqdm_q.put(stats_dict.copy())

                # Check if we should save the model.
                if save_interval > 0 and (total_episodes_so_far - last_save_episode) >= save_interval:
                    checkpoint_dir = "checkpoints"
                    if output_path:
                        if os.path.isdir(output_path):
                            checkpoint_dir = os.path.join(output_path, "checkpoints")
                        else:
                            # If output path is a file, use its directory
                            output_dir = os.path.dirname(output_path)
                            checkpoint_dir = os.path.join(output_dir if output_dir else ".", "checkpoints")

                    os.makedirs(checkpoint_dir, exist_ok=True)

                    # save checkpoint
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_{total_episodes_so_far}.pt")
                    trainer.save_models(checkpoint_path)

                    # Also save as "latest" for easy loading (keep this for compatibility)
                    latest_path = os.path.join(checkpoint_dir, "model_latest.pt")
                    trainer.save_models(latest_path)

                    if debug:
                        print(f"[DEBUG] Saved checkpoint at episode {total_episodes_so_far} to {checkpoint_path}")

                    last_save_episode = total_episodes_so_far

                # Reset episode rewards for the environments that finished an episode.
                for env_idx, done_flag in enumerate(dones): # Use dones from vec_env.step
                    if done_flag:
                        # Only calculate average reward if the dict is not empty
                        if env_idx in episode_rewards and episode_rewards[env_idx]:
                            avg_reward = sum(episode_rewards[env_idx].values()) / len(episode_rewards[env_idx])
                            if debug:
                                print(f"Episode {episode_counts[env_idx]} in env {env_idx} completed with avg reward: {avg_reward:.2f}")

                            # Track average episode reward for PPO updates
                            if algorithm == "ppo":
                                completed_episode_rewards_since_last_update.append(avg_reward)
                        else:
                            # Handle case where there are no rewards (e.g., env finished immediately)
                            if debug:
                                print(f"Warning: Episode {episode_counts[env_idx]} in env {env_idx} completed with no rewards recorded or dict missing.")

                        # Reset rewards dictionary for next episode
                        # Use the agent IDs from the *next* observation after reset, if available
                        if env_idx in vec_env.obs_dicts and vec_env.obs_dicts[env_idx]:
                             episode_rewards[env_idx] = {agent_id: 0.0 for agent_id in vec_env.obs_dicts[env_idx]}
                        else:
                             episode_rewards[env_idx] = {} # Reset to empty if env obs are missing


            # Determine if it's time to update the policy
            should_update = False

            if algorithm == "ppo":
                # For PPO, check if we've collected enough experiences based on update_interval
                # The buffer size check is handled inside trainer.update() now
                should_update = collected_experiences >= update_interval

                if debug and should_update:
                    buffer_size = trainer.algorithm.memory.size if hasattr(trainer.algorithm, 'memory') else 0
                    buffer_capacity = trainer.algorithm.memory.buffer_size if hasattr(trainer.algorithm, 'memory') else 0
                    print(f"[DEBUG] PPO update triggered - collected: {collected_experiences}/{update_interval}, " +
                          f"buffer: {buffer_size}/{buffer_capacity}")
            else:
                # For StreamAC/SAC, updates happen online or based on their own logic.
                # We still update the progress bar periodically based on collected_experiences.
                # Use a smaller interval for UI updates for online algorithms.
                ui_update_interval = update_interval // 10 if update_interval > 100 else 50
                should_update_ui = collected_experiences > 0 and (collected_experiences % ui_update_interval == 0)

                # Handle indefinite training specially for online algorithms
                if total_episodes is None and training_time is None:
                    # For indefinite training, update more frequently
                    ui_update_interval = min(ui_update_interval, 20)
                    should_update_ui = collected_experiences > 0 and (collected_experiences % ui_update_interval == 0)

                if should_update_ui:
                    # Update step count and other metrics for UI
                    stats_dict["Steps"] = collected_experiences
                    if hasattr(trainer.algorithm, "update_count"): # StreamAC/SAC might have this
                        stats_dict["Updates"] = getattr(trainer.algorithm, "update_count", 0)

                    # Get latest metrics from algorithm for UI update
                    metrics = trainer.algorithm.get_metrics() if hasattr(trainer.algorithm, 'get_metrics') else {}
                    if metrics:
                        stats_dict.update({
                            "Reward": f"{metrics.get('mean_return', 0):.2f}",
                            "PLoss": f"{metrics.get('actor_loss', 0):.4f}",
                            "VLoss": f"{metrics.get('critic_loss', 0):.4f}", # Or critic1/2 for SAC
                            "Entropy": f"{metrics.get('entropy_loss', 0):.4f}"
                        })
                        if algorithm == "streamac":
                             stats_dict.update({
                                 "StepSize": f"{metrics.get('effective_step_size', 0):.4f}",
                                 "ActorLR": f"{getattr(trainer.algorithm, 'lr_actor', lr_actor):.6f}",
                                 "CriticLR": f"{getattr(trainer.algorithm, 'lr_critic', lr_critic):.6f}"
                             })
                        elif algorithm == "sac":
                             stats_dict.update({
                                 "VLoss": f"{(metrics.get('critic1_loss', 0) + metrics.get('critic2_loss', 0))/2:.4f}",
                                 "Alpha": f"{metrics.get('alpha', 0):.4f}",
                                 "Buffer": f"{len(trainer.algorithm.memory)}" if hasattr(trainer.algorithm, 'memory') else "N/A"
                             })

                        # Use SCALAR versions for progress bar aux losses
                        if 'sr_loss_scalar' in metrics or 'rp_loss_scalar' in metrics:
                            stats_dict.update({
                                "SR_Loss": f"{metrics.get('sr_loss_scalar', 0):.4f}",
                                "RP_Loss": f"{metrics.get('rp_loss_scalar', 0):.4f}"
                            })

                    # Send updated stats_dict to TQDM manager
                    if tqdm_q:
                        tqdm_q.put(stats_dict.copy())


            # Check if intrinsic reward models should be reset
            if use_pretraining and not trainer.pretraining_completed and \
               total_episodes_so_far > 0 and \
               total_episodes_so_far % 50 == 0 and \
               total_episodes_so_far > last_intrinsic_reset_episode and \
               hasattr(trainer, 'intrinsic_reward_generator') and \
               trainer.intrinsic_reward_generator is not None:

                trainer.intrinsic_reward_generator.reset_models()
                last_intrinsic_reset_episode = total_episodes_so_far # Update the tracker
                if debug:
                    print(f"[DEBUG] Reset intrinsic reward models at episode {total_episodes_so_far}")


            # Update policy only if conditions met and not in test mode
            if should_update and not test:
                if debug:
                    print(f"[DEBUG] Updating policy ({algorithm.upper()}) with {collected_experiences} experiences after {time.time() - last_update_time:.2f}s")

                # Perform the policy update.
                if algorithm == "ppo":
                    # For PPO, do a normal batch update and pass the completed episode rewards and total env steps
                    stats = trainer.update(
                        completed_episode_rewards=completed_episode_rewards_since_last_update,
                        total_env_steps=total_env_steps, # Pass total env steps
                        steps_per_second=steps_per_second # Pass steps per second
                    )
                    # Reset the list of completed episode rewards
                    completed_episode_rewards_since_last_update = []
                    collected_experiences = 0 # Reset PPO experience counter after update
                else:
                    # For StreamAC/SAC, update might happen internally or via trainer.update()
                    # If trainer.update() is called, it should just fetch metrics for online algos
                    stats = trainer.update(total_env_steps=total_env_steps, steps_per_second=steps_per_second)
                    # Don't reset collected_experiences for online algos here

                # Update the statistics displayed in the progress bar.
                # Use SCALAR versions for display
                stats_dict.update({
                    "Device": device,
                    "Envs": num_envs,
                    "Episodes": total_episodes_so_far,
                    "Reward": f"{stats.get('mean_episode_reward', stats.get('mean_return', 0)):.2f}", # Use mean_return as fallback
                    "PLoss": f"{stats.get('actor_loss', 0):.4f}",
                    "VLoss": f"{stats.get('critic_loss', 0):.4f}", # Default, SAC overrides below
                    "Entropy": f"{stats.get('entropy_loss', 0):.4f}",
                })
                # Only include Aux stats if enabled
                if auxiliary:
                    stats_dict.update({
                        "SR_Loss": f"{stats.get('sr_loss_scalar', 0):.4f}",
                        "RP_Loss": f"{stats.get('rp_loss_scalar', 0):.4f}"
                    })


                # Update algorithm-specific metrics if using that algorithm
                if algorithm == "ppo":
                     stats_dict["Exp"] = f"{collected_experiences}/{update_interval}" # Show 0 after update
                elif algorithm == "streamac":
                    stats_dict.update({
                        "StepSize": f"{stats.get('effective_step_size', 0)::.4f}",
                        "ActorLR": f"{getattr(trainer.algorithm, 'lr_actor', lr_actor):.6f}",
                        "CriticLR": f"{getattr(trainer.algorithm, 'lr_critic', lr_critic):.6f}",
                        "Updates": getattr(trainer.algorithm, "update_count", 0) # Use update_count
                    })
                elif algorithm == "sac":
                    stats_dict.update({
                        "VLoss": f"{(stats.get('critic1_loss', 0) + stats.get('critic2_loss', 0))/2:.4f}",
                        "Alpha": f"{stats.get('alpha', 0):.4f}",
                        "Buffer": f"{len(trainer.algorithm.memory)}" if hasattr(trainer.algorithm, 'memory') else "N/A"
                    })


                # Update curriculum stats if enabled
                if curriculum_manager:
                    curriculum_stats = curriculum_manager.get_curriculum_stats()
                    stats_dict.update({
                        "Stage": curriculum_stats["current_stage_name"],
                        "Diff": f"{curriculum_stats['difficulty_level']:.2f}"
                    })

                    # WandB logging for curriculum is handled within trainer.update -> _log_to_wandb

                last_update_time = time.time()

                # Send updated stats_dict to TQDM manager after policy update
                if tqdm_q:
                    tqdm_q.put(stats_dict.copy())

                # Garbage collection after updates
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Update the experience count in the progress bar - ONLY for PPO
            if algorithm == "ppo":
                # Update the 'Exp' field in stats_dict
                new_exp_str = f"{collected_experiences}/{update_interval}"
                if stats_dict.get("Exp") != new_exp_str:
                    stats_dict["Exp"] = new_exp_str
                    # Send updated stats_dict only if 'Exp' changed
                    if tqdm_q:
                        tqdm_q.put(stats_dict.copy())


            # Update pretraining progress if enabled
            if use_pretraining:
                pretraining_end_step = trainer._get_pretraining_end_step()
                in_pretraining = not trainer.pretraining_completed
                in_transition = trainer.in_transition_phase
                needs_postfix_update = False # Flag if pretraining status changed

                # Check pretraining completion status based on current step count
                current_step = trainer._true_training_steps() # Use trainer's step counter
                if current_step >= pretraining_end_step and not trainer.pretraining_completed and not trainer.in_transition_phase:
                    print(f"Triggering pretraining transition at step {current_step}/{pretraining_end_step}")
                    trainer.in_transition_phase = True
                    trainer.transition_start_step = current_step # Use trainer step
                    needs_postfix_update = True

                if in_pretraining:
                    new_mode = "PreTraining"
                    new_progress = f"{current_step}/{pretraining_end_step}"
                    if stats_dict.get("Mode") != new_mode or stats_dict.get("PT_Progress") != new_progress:
                        stats_dict["Mode"] = new_mode
                        stats_dict["PT_Progress"] = new_progress
                        needs_postfix_update = True
                elif in_transition:
                    new_mode = "PT_Transition"
                    transition_progress = min(100, int(100 * (current_step - trainer.transition_start_step) /
                                              max(1, trainer.pretraining_transition_steps))) # Avoid div by zero
                    new_progress = f"{transition_progress}%"
                    if stats_dict.get("Mode") != new_mode or stats_dict.get("PT_Progress") != new_progress:
                        stats_dict["Mode"] = new_mode
                        stats_dict["PT_Progress"] = new_progress
                        needs_postfix_update = True
                else:
                    # Remove pretraining info from stats once completed
                    if "Mode" in stats_dict:
                        stats_dict.pop("Mode", None)
                        needs_postfix_update = True
                    if "PT_Progress" in stats_dict:
                        stats_dict.pop("PT_Progress", None)
                        needs_postfix_update = True

                # Send updated stats_dict if pretraining status changed
                if needs_postfix_update and tqdm_q:
                    tqdm_q.put(stats_dict.copy())


            # No need to call set_postfix here, handled by sending messages to the queue

    except KeyboardInterrupt:
        print("\nTraining interrupted by main process. Cleaning up...")
        # Signal handler should take care of stopping TQDM manager
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
    finally:
        # Always clean up the environments.
        vec_env.close()

        # TQDM manager is stopped by the main script's finally block or signal handler

        # Perform a final policy update if there are any remaining experiences (PPO).
        if algorithm == "ppo" and collected_experiences > 0 and not test:
            if debug:
                print(f"[DEBUG] Final PPO update with {collected_experiences} experiences")
            try:
                # Ensure buffer has enough data for at least one batch before final update
                if trainer.algorithm.memory.size >= trainer.batch_size:
                     trainer.update(
                         completed_episode_rewards=completed_episode_rewards_since_last_update,
                         total_env_steps=total_env_steps,
                         steps_per_second=steps_per_second
                     )
                elif debug:
                     print(f"[DEBUG] Skipping final PPO update, buffer size ({trainer.algorithm.memory.size}) < batch size ({trainer.batch_size})")
            except Exception as e:
                print(f"Error during final update: {str(e)}")
                traceback.print_exc()

        # Return the trainer instance
        return trainer

def parse_time(time_str):
    """
    Parses a string representing a time duration (e.g., '5m', '2h', '1d') and converts it to seconds.
    """
    if not time_str:
        return None

    # Get the number and the unit (minutes, hours, or days).
    time_str = time_str.lower().strip()
    if len(time_str) < 2:
        raise ValueError(f"Invalid time format: {time_str}. Use format like '5m', '2h', '1d'")

    try:
        value = float(time_str[:-1])
        unit = time_str[-1]

        # Convert to seconds.
        if unit == 'm':  # minutes
            return value * 60
        elif unit == 'h':  # hours
            return value * 3600
        elif unit == 'd':  # days
            return value * 86400
        else:
            raise ValueError(f"Unknown time unit: {unit}. Use 'm' for minutes, 'h' for hours, 'd' for days")
    except ValueError as e:
        if "Unknown time unit" in str(e):
            raise
        raise ValueError(f"Invalid time format: {time_str}. Use format like '5m', '2h', '1d'")


def signal_handler(sig, frame):
    """Handles Ctrl+C gracefully, so the program exits cleanly."""
    global tqdm_manager_instance # Need global to access the instance
    print("\nInterrupted by user. Cleaning up...")
    if tqdm_manager_instance:
        print("Attempting to stop TQDM manager from signal handler...")
        tqdm_manager_instance.stop() # Use the manager's stop method
    sys.exit(0)

if __name__ == "__main__":
    # Set start method
    if sys.platform == 'darwin':
        mp.set_start_method('spawn', force=True)
    elif sys.platform == 'linux':
        mp.set_start_method('fork', force=True)


    # Set up Ctrl+C handler to exit gracefully.
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='RLBot Training Script')
    parser.add_argument('--render', action='store_true', help='Enable rendering of the game environment')

    # Allow user to specify training duration either by episode count OR by time.
    training_duration = parser.add_mutually_exclusive_group()
    training_duration.add_argument('-e', '--episodes', type=int, default=None, help='Number of episodes to run')
    training_duration.add_argument('-t', '--time', type=str, default=None,
                                  help='Training duration in format: 5m (minutes), 5h (hours), 5d (days)')

    parser.add_argument('-n', '--num_envs', type=int, default=300,
                        help='Number of parallel environments to run for faster data collection')
    parser.add_argument('--update_interval', type=int, default=1048576,
                        help='Number of experiences to collect before updating the policy (PPO)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training (cuda/mps/cpu).  Autodetects if not specified.')
    parser.add_argument('--wandb', action='store_true', help='Enable logging to Weights & Biases')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging')

    parser.add_argument('--render-delay', type=float, default=0.025,
                    help='Delay between rendered frames in seconds (higher values = slower visualization)')

    parser.add_argument('--curriculum', action='store_true',
                    help='Enable curriculum learning')
    parser.add_argument('--no-curriculum', action='store_false', dest='curriculum',
                    help='Disable curriculum learning')
    parser.set_defaults(curriculum=True)

    # Add single-stage mode argument
    parser.add_argument('--stage', type=int, default=None,
                    help='Run a specific curriculum stage indefinitely (0-based index)')

    # Algorithm choice
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'streamac', 'sac'],
                       help='Learning algorithm to use: ppo (default) or streamac')

    # Learning rates
    parser.add_argument('--lra', type=float, default=1e-4, help='Learning rate for actor network')
    parser.add_argument('--lrc', type=float, default=3e-4, help='Learning rate for critic network')

    # Discount factors
    parser.add_argument('--gamma', type=float, default=0.997, help='Discount factor for future rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='Lambda parameter for Generalized Advantage Estimation')

    # PPO parameters
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clipping parameter')
    parser.add_argument('--critic_coef', type=float, default=0.5, help='Weight of the critic loss')
    parser.add_argument('--entropy_coef', type=float, default=0.005, help='Weight of the entropy bonus (encourages exploration)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm for clipping')

    # Training loop parameters
    parser.add_argument('--ppo_epochs', type=int, default=10, help='Number of PPO epochs per update')
    parser.add_argument('--batch_size', type=int, default=24576, help='Batch size for PPO updates')

    parser.add_argument('--weight_clip_kappa', type=float, default=1.0, help='Weight clipping factor for PPO')
    parser.add_argument('--weight_clipping', type=bool, default=False, help='Enable weight clipping for PPO')

    # Adaptive weight clipping parameters
    parser.add_argument('--adaptive_kappa', action='store_false', help='Enable adaptive weight clipping kappa')
    parser.add_argument('--no-adaptive_kappa', action='store_true', dest='adaptive_kappa', help='Disable adaptive weight clipping kappa')
    parser.set_defaults(adaptive_kappa=False)  # Disable by default
    parser.add_argument('--kappa_update_freq', type=int, default=10, help='Update frequency for adaptive kappa')
    parser.add_argument('--kappa_update_rate', type=float, default=0.01, help='Update rate for adaptive kappa (1% by default)')
    parser.add_argument('--target_clip_fraction', type=float, default=0.05, help='Target fraction of weights to clip (5% by default)')
    parser.add_argument('--min_kappa', type=float, default=0.1, help='Minimum value for adaptive kappa')
    parser.add_argument('--max_kappa', type=float, default=10.0, help='Maximum value for adaptive kappa')

    parser.add_argument('--compile', action='store_true', help='Use torch.compile for model optimization (if available)')
    parser.add_argument('--no-compile', action='store_false', dest='compile', help='Disable torch.compile')
    parser.set_defaults(compile=True)

    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision for faster training (requires CUDA)')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='Disable automatic mixed precision')
    parser.set_defaults(amp=False)

    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Path to a pre-trained model file to load')

    parser.add_argument('-o', '--out', type=str, default=None,
                    help='Path where the trained model will be saved')

    parser.add_argument('--test', action='store_true',
                        help='Enable test mode (enables rendering and limits to 1 environment)')


    parser.add_argument('--save_interval', type=int, default=10000,
                       help='Save the model every N episodes')

    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension for the network')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of residual blocks in the network')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate for regularization')

    # Action stacking parameters
    parser.add_argument('--stack_size', type=int, default=5, help='Number of previous actions to stack')

    # Auxiliary learning parameters - Corrected Logic
    # By default, auxiliary tasks are enabled (default=True).
    # Using --no-auxiliary will set args.auxiliary to False.
    parser.add_argument('--no-auxiliary', action='store_false', dest='auxiliary',
                        help='Disable auxiliary task learning (SR and RP tasks)')
    parser.set_defaults(auxiliary=True) # Keep default as True


    parser.add_argument('--sr_weight', type=float, default=1.0,
                        help='Weight for the State Representation auxiliary task')
    parser.add_argument('--rp_weight', type=float, default=1.0,
                        help='Weight for the Reward Prediction auxiliary task')

    # Add option to control AMP for auxiliary tasks specifically
    parser.add_argument('--aux-amp', action='store_true',
                        help='Enable AMP for auxiliary tasks (can be disabled separately from main training)')
    parser.add_argument('--no-aux-amp', action='store_false', dest='aux_amp',
                        help='Disable AMP for auxiliary tasks even if main training uses AMP')
    parser.set_defaults(aux_amp=False)

    parser.add_argument('--aux_freq', type=int, default=8,
                        help='Auxiliary task update frequency (higher = less frequent updates)')

    parser.add_argument('--aux_scale', type=float, default=0.1,
                        help='Scaling factor for auxiliary task losses')

    # Pre-training parameters
    parser.add_argument('--pretraining', action='store_true', help='Enable unsupervised pre-training phase at the start')
    parser.add_argument('--no-pretraining', action='store_false', dest='pretraining', help='Disable unsupervised pre-training')
    parser.set_defaults(pretraining=True)  # Changed default to True so --no-pretraining has an effect
    parser.add_argument('--pretraining-fraction', type=float, default=0.1, help='Fraction of total training time/episodes to use for pre-training (default: 0.1)')
    parser.add_argument('--pretraining-sr-weight', type=float, default=10.0, help='Weight for State Representation task during pre-training (default: 10.0)')
    parser.add_argument('--pretraining-rp-weight', type=float, default=5.0, help='Weight for Reward Prediction task during pre-training (default: 5.0)')
    parser.add_argument('--pretraining-transition-steps', type=int, default=5000, help='Number of steps for smooth transition from pre-training to RL training (default: 1000)')

    # Intrinsic reward parameters
    parser.add_argument('--use-intrinsic', action='store_true', help='Use intrinsic rewards during pre-training')
    parser.add_argument('--no-intrinsic', action='store_false', dest='use_intrinsic', help='Disable intrinsic rewards during pre-training')
    parser.set_defaults(use_intrinsic=True)

    parser.add_argument('--intrinsic-scale', type=float, default=0.7,
                       help='Scaling factor for intrinsic rewards (default: 0.7)')

    parser.add_argument('--curiosity-weight', type=float, default=0.5,
                       help='Weight for curiosity-based intrinsic rewards (default: 0.5)')

    parser.add_argument('--rnd-weight', type=float, default=0.5,
                       help='Weight for Random Network Distillation rewards (default: 0.5)')

    # StreamAC specific parameters
    streamac_group = parser.add_argument_group('StreamAC parameters')
    streamac_group.add_argument('--adaptive-lr', action='store_true', dest='adaptive_learning_rate',
                              help='Enable adaptive learning rate for StreamAC (default: True)')
    streamac_group.add_argument('--no-adaptive-lr', action='store_false', dest='adaptive_learning_rate',
                               help='Disable adaptive learning rate for StreamAC')
    parser.set_defaults(adaptive_learning_rate=True)

    streamac_group.add_argument('--target-step-size', type=float, default=0.025,
                               help='Target effective step size for StreamAC (default: 0.025)')

    streamac_group.add_argument('--backtracking-patience', type=int, default=10,
                                help='Number of steps before backtracking parameters (default: 10)')

    streamac_group.add_argument('--backtracking-zeta', type=float, default=0.85,
                               help='Scaling factor for learning rate during backtracking (default: 0.85)')

    streamac_group.add_argument('--min-lr-factor', type=float, default=0.1,
                               help='Minimum learning rate factor relative to initial (default: 0.1)')

    streamac_group.add_argument('--max-lr-factor', type=float, default=10.0,
                               help='Maximum learning rate factor relative to initial (default: 10.0)')

    streamac_group.add_argument('--obgd', action='store_true', dest='use_obgd',
                               help='Enable Online Backpropagation with Decoupled Gradient (default: True)')
    streamac_group.add_argument('--no-obgd', action='store_false', dest='use_obgd',
                                help='Disable Online Backpropagation with Decoupled Gradient')
    parser.set_defaults(use_obgd=True)

    streamac_group.add_argument('--stream-buffer-size', type=int, default=32,
                               help='Size of experience buffer for StreamAC (default: 32)')

    streamac_group.add_argument('--sparse-init', action='store_true', dest='use_sparse_init',
                               help='Use SparseInit for network initialization (default: True)')
    streamac_group.add_argument('--no-sparse-init', action='store_false', dest='use_sparse_init',
                                help='Use default PyTorch initialization instead of SparseInit')
    parser.set_defaults(use_sparse_init=True)

    streamac_group.add_argument('--update-freq', type=int, default=4,
                               help='Frequency of updates for StreamAC (default: 4, update every 4th step)')

    # SAC-specific parameters
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Target network update rate for SAC")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Temperature parameter for SAC entropy")
    parser.add_argument("--auto_alpha_tuning", action="store_true",
                        help="Enable automatic entropy tuning for SAC")
    parser.add_argument("--target_entropy", type=float, default=None,
                        help="Target entropy when auto-tuning (default: -action_dim)")
    parser.add_argument("--buffer_size", type=int, default=1000000,
                        help="Replay buffer size for SAC")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of steps before starting to train SAC")
    parser.add_argument("--updates_per_step", type=int, default=1,
                        help="Number of gradient updates per step for SAC")

    # Backwards compatibility.
    parser.add_argument('-p', '--processes', type=int, default=None,
                        help='Legacy parameter; use --num_envs instead')

    # Model architecture argument
    parser.add_argument('--model-arch', type=str, default='simbav2', choices=['basic', 'simba', 'simbav2', 'simbav2-shared'], # Add simbav2-shared
                       help='Model architecture to use (basic, simba, simbav2, simbav2-shared)')

    # Add reward scaling args here, before parse_args()
    parser.add_argument('--use-reward-scaling', action='store_true', dest='use_reward_scaling', help='Enable SimbaV2 reward scaling')
    parser.add_argument('--no-reward-scaling', action='store_false', dest='use_reward_scaling', help='Disable SimbaV2 reward scaling')
    parser.set_defaults(use_reward_scaling=True)
    parser.add_argument('--reward-scaling-gmax', type=float, default=10.0, help='G_max hyperparameter for reward scaling')

    args = parser.parse_args()

    if args.model is not None:
        args.use_sparse_init = False

    # If a training time is provided, parse it into seconds.
    training_time_seconds = None
    if args.time is not None:
        try:
            training_time_seconds = parse_time(args.time)
            if args.debug:
                print(f"[DEBUG] Training time set to {args.time} ({training_time_seconds} seconds)")
        except ValueError as e:
            print(str(e))
            sys.exit(1)

    if args.test:
        args.render = True
        args.num_envs = 1
        args.auxiliary = False
        args.curriculum = False
        print("Test mode enabled: Rendering ON, using 1 environment")

    # Handle legacy --processes argument.
    if args.processes is not None and args.num_envs == 4:  # 4 is the default for num_envs
        args.num_envs = args.processes
        if args.debug:
            print(f"[DEBUG] Using --processes value ({args.processes}) for number of environments")

    # Create action stacker
    action_stacker = ActionStacker(stack_size=args.stack_size, action_size=8)  # RLGym uses 8 actions

    # Get the dimensions of the observation and action spaces.
    env = get_env(action_stacker=action_stacker)
    env.reset()
    obs_space = env.observation_space(env.agents[0])
    obs_space_dims = obs_space[0]
    action_space_dims = env.action_space(env.agents[0])[1]
    env.close()

    # Use the best available device (CUDA if available, then MPS, then CPU).
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and False: # Disable because mps is slow
            device = "mps"
        else:
            device = "cpu"

    # --- Model Instantiation ---
    actor = None
    critic = None
    model_kwargs = {
        'hidden_dim': args.hidden_dim,
        'num_blocks': args.num_blocks,
        'device': device
    }

    if args.model_arch == 'basic':
        ModelClass = BasicModel
        model_kwargs['dropout_rate'] = args.dropout
        actor = ModelClass(obs_shape=obs_space_dims, action_shape=action_space_dims, **model_kwargs)
        critic = ModelClass(obs_shape=obs_space_dims, action_shape=1, **model_kwargs)
    elif args.model_arch == 'simba':
        ModelClass = SimBa
        model_kwargs['dropout_rate'] = args.dropout
        actor = ModelClass(obs_shape=obs_space_dims, action_shape=action_space_dims, **model_kwargs)
        critic = ModelClass(obs_shape=obs_space_dims, action_shape=1, **model_kwargs)
    elif args.model_arch == 'simbav2':
        ModelClass = SimbaV2
        # SimbaV2 does not use dropout_rate
        actor = ModelClass(obs_shape=obs_space_dims, action_shape=action_space_dims, **model_kwargs)
        critic = ModelClass(obs_shape=obs_space_dims, action_shape=1, **model_kwargs)
    elif args.model_arch == 'simbav2-shared':
        ModelClass = SimbaV2Shared
        # SimbaV2Shared does not use dropout_rate
        # Instantiate ONE model and use it for both actor and critic
        shared_model = ModelClass(obs_shape=obs_space_dims, action_shape=action_space_dims, **model_kwargs)
        actor = shared_model
        critic = shared_model # Assign the same instance
        if args.debug:
            print("[DEBUG] Using SimbaV2Shared architecture (actor and critic share the same instance)")
    else:
        raise ValueError(f"Unknown model architecture: {args.model_arch}")
    # --- End Model Instantiation ---


    torch.set_printoptions(precision=10)

    if "cuda" in str(device):
        # CUDA-specific optimizations
        torch.set_float32_matmul_precision('high')  # Use Tensor Cores
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere and newer
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

        # Configure BLAS operations
        torch.backends.cuda.preferred_linalg_library('cusolver')  # Prefer cuSOLVER for stability

        # Try to set bailout depth for CUDA graphs
        try:
            torch._C._jit_set_bailout_depth(20)
        except AttributeError:
            if args.debug:
                print("[DEBUG] _jit_set_bailout_depth not available in this PyTorch version")

        torch.cuda.set_device(torch.cuda.current_device())

        # Improve CUDA graph memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Configure Dynamo for safer CUDA graphs
        if hasattr(torch, '_dynamo'):
            try:
                torch._dynamo.config.cache_size_limit = 16  # Limit cache size
                torch._dynamo.config.suppress_errors = True
                if args.debug:
                    torch._dynamo.config.verbose = True
            except AttributeError:
                if args.debug:
                    print("[DEBUG] torch._dynamo.config not available in this PyTorch version")

    # Set up Weights & Biases for experiment tracking, if enabled.
    if args.wandb:
        # Initialize wandb with proper step counting setup
        run = wandb.init(
            project="rlbot-training",
            resume="allow",  # Allow resuming previous runs
            config={
                # Algorithm
                "algorithm": args.algorithm,

                # Hyperparameters
                "learning_rate_actor": args.lra,
                "learning_rate_critic": args.lrc,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_epsilon": args.clip_epsilon,
                "critic_coef": args.critic_coef,
                "entropy_coef": args.entropy_coef,
                "max_grad_norm": args.max_grad_norm,
                "ppo_epochs": args.ppo_epochs,
                "batch_size": args.batch_size,

                # Model Details
                "model_arch": args.model_arch, # Log model architecture
                "hidden_dim": args.hidden_dim,
                "num_blocks": args.num_blocks,
                "dropout": args.dropout if 'dropout_rate' in model_kwargs else None, # Log dropout if applicable
                "action_stack_size": args.stack_size,
                "auxiliary_tasks": args.auxiliary, # Log the effective value
                "sr_weight": args.sr_weight,
                "rp_weight": args.rp_weight,

                # Environment details
                "action_repeat": 8,
                "num_agents": 4,  # 2v2

                # System configuration
                "episodes": args.episodes if args.episodes is not None else "indefinite" if args.time is None else None,
                "training_time": args.time if args.time is not None else "indefinite" if args.episodes is None else None,
                "num_envs": args.num_envs,
                "update_interval": args.update_interval,
                "device": device,
            },
            name=f"{args.algorithm.upper()}_{args.model_arch}_{time.strftime('%Y%m%d-%H%M%S')}", # Include model arch in name
            monitor_gym=False,  # Don't use wandb's default gym monitoring
        )

        # Add StreamAC specific config if using that algorithm
        if args.algorithm == "streamac":
            wandb.config.update({
                "adaptive_learning_rate": args.adaptive_learning_rate,
                "target_step_size": args.target_step_size,
                "backtracking_patience": args.backtracking_patience,
                "backtracking_zeta": args.backtracking_zeta,
                "use_obgd": args.use_obgd,
                "stream_buffer_size": args.stream_buffer_size,
                "use_sparse_init": args.use_sparse_init,
                "update_freq": args.update_freq
            })

    if args.debug:
        print(f"[DEBUG] Starting training with {args.num_envs} environments on {device}")
        print(f"[DEBUG] Using {args.algorithm.upper()} algorithm")
        print(f"[DEBUG] Using model architecture: {args.model_arch}")
        print(f"[DEBUG] Auxiliary tasks enabled: {args.auxiliary}") # Print effective value
        print(f"[DEBUG] Actor model: {actor}")
        # Only print critic separately if it's a different instance
        if actor is not critic:
            print(f"[DEBUG] Critic model: {critic}")
        print(f"[DEBUG] Action stacking size: {args.stack_size}")
        if args.time:
            print(f"[DEBUG] Training for {args.time} ({training_time_seconds} seconds)")
        elif args.episodes:
            print(f"[DEBUG] Training for {args.episodes} episodes")
        else:
            print(f"[DEBUG] Training indefinitely - press Ctrl+C to stop")


    # Get the training step offset (initialize to 0)
    # This will be updated inside run_training if a model is loaded
    trainer_offset = 0

    # --- Initialize TQDM Manager ---
    # Note: tqdm_queue and tqdm_manager_instance are already defined at module level
    tqdm_queue = mp.Queue() # Assign to the module-level variable

    # Determine tqdm parameters based on training mode
    tqdm_params = {
        "total": None,
        "desc": "Training",
        "bar_format": '{desc}: [{elapsed}] |{bar:30}| {postfix}'
    }
    dynamic_ncols = True

    if training_time_seconds is not None:
        tqdm_params["total"] = int(training_time_seconds)
        tqdm_params["desc"] = "Time"
        tqdm_params["bar_format"] = '{desc}: {percentage:3.0f}% [{elapsed}<{remaining}] |{bar}| {postfix}'
    elif args.episodes is not None:
        tqdm_params["total"] = args.episodes
        tqdm_params["desc"] = "Episodes"
        tqdm_params["bar_format"] = '{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%|{bar}| {postfix}'
    # else: use defaults for indefinite training

    # Assign to the module-level variable so signal handler can access it
    tqdm_manager_instance = TqdmManager(
        queue=tqdm_queue,
        total=tqdm_params["total"],
        desc=tqdm_params["desc"],
        bar_format=tqdm_params["bar_format"],
        dynamic_ncols=dynamic_ncols
    )
    tqdm_manager_instance.start()
    # --- End TQDM Manager Initialization ---

    trainer = None # Initialize trainer to None
    saved_path = None
    try:
        # Start the main training process
        # Pass args.model path to run_training
        trainer = run_training(
            actor=actor,
            critic=critic, # Pass the potentially shared instance
            # Pass reward scaling args
            use_reward_scaling=args.use_reward_scaling,
            reward_scaling_G_max=args.reward_scaling_gmax,
            # training_step_offset is now handled inside run_training via load_models
            device=device,
            num_envs=args.num_envs,
            total_episodes=args.episodes if args.time is None else None,
            training_time=training_time_seconds,
            render=args.render,
            update_interval=args.update_interval,
            use_wandb=args.wandb,
            debug=args.debug,
            use_compile=args.compile,
            use_amp=args.amp,
            save_interval=args.save_interval,
            output_path=args.out,
            use_curriculum=args.curriculum,
            stage=args.stage,
            # Pass model path to run_training
            model_path_to_load=args.model,
            algorithm=args.algorithm,
            lr_actor=args.lra,
            lr_critic=args.lrc,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            critic_coef=args.critic_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            ppo_epochs=args.ppo_epochs,
            weight_clip_kappa=args.weight_clip_kappa,
            use_weight_clipping=args.weight_clipping,
            batch_size=args.batch_size,
            aux_amp=args.aux_amp if args.aux_amp is not None else args.amp,
            aux_scale=args.aux_scale,
            auxiliary=args.auxiliary, # Pass the corrected flag value
            sr_weight=args.sr_weight,
            rp_weight=args.rp_weight,
            test=args.test,
            use_pretraining=args.pretraining,
            pretraining_fraction=args.pretraining_fraction,
            pretraining_sr_weight=args.pretraining_sr_weight,
            pretraining_rp_weight=args.pretraining_rp_weight,
            pretraining_transition_steps=args.pretraining_transition_steps,
            use_intrinsic_rewards=args.use_intrinsic,
            intrinsic_reward_scale=args.intrinsic_scale,
            curiosity_weight=args.curiosity_weight,
            rnd_weight=args.rnd_weight,
            adaptive_learning_rate=args.adaptive_learning_rate,
            target_step_size=args.target_step_size,
            backtracking_patience=args.backtracking_patience,
            backtracking_zeta=args.backtracking_zeta,
            min_lr_factor=args.min_lr_factor,
            max_lr_factor=args.max_lr_factor,
            use_obgd=args.use_obgd,
            stream_buffer_size=args.stream_buffer_size,
            use_sparse_init=args.use_sparse_init,
            update_freq=args.update_freq,
            tqdm_q=tqdm_queue # Pass the queue to the training function
        )

        # Save the final trained models, unless we're in test mode or training failed.
        if trainer is not None:  # Only check if trainer exists, not test mode
            # Always save when trainer exists and small number of episodes were run - these are likely evaluation runs
            output_path = args.out if args.out else None
            # Save model with step count and wandb info
            metadata = {
                'training_step': getattr(trainer, 'training_steps', 0) + getattr(trainer, 'training_step_offset', 0),
                'total_env_steps': getattr(trainer, 'total_env_steps', 0), # Add total env steps to metadata
                'wandb_run_id': wandb.run.id if args.wandb and wandb.run else None,
                'algorithm': args.algorithm,  # Save which algorithm was used
                'model_arch': args.model_arch # Save model architecture used
            }
            saved_path = trainer.save_models(output_path, metadata)  # Capture the returned path
            print(f"Training complete - Model saved to {saved_path} at step {metadata['training_step']} (Total Env Steps: {metadata['total_env_steps']})")
        else:
            print("Training failed - no model saved.")

    except Exception as main_exception:
         print(f"An error occurred in the main script: {main_exception}")
         traceback.print_exc()
    finally:
        # --- Stop TQDM Manager ---
        if tqdm_manager_instance:
            tqdm_manager_instance.stop()
        # --- End Stop TQDM Manager ---

        # Upload the saved model to WandB as an artifact
        if args.wandb and saved_path and os.path.exists(saved_path):
            try:
                # Handle WandB run ID safely for artifact naming
                run_id = getattr(wandb.run, 'id', 'unknown') if wandb.run else 'unknown'
                artifact_name = f"{args.algorithm}_{args.model_arch}_{run_id}" # Include model arch
                if args.time is not None:
                    time_seconds = 0
                    if args.time:
                        try:
                            time_seconds = int(parse_time(args.time))
                        except Exception:
                            time_seconds = 0
                    artifact_name += f"_t{time_seconds}s"
                elif args.episodes is not None:
                    artifact_name += f"_ep{args.episodes}"
                else:
                    artifact_name += "_indefinite"

                # Log the model as an artifact with metadata
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model",
                    description=f"RL model trained with {args.algorithm.upper()} ({args.model_arch}) for {args.episodes if args.time is None else args.time}"
                )

                # Add the model file
                artifact.add_file(saved_path)

                # Add metadata
                metadata = {
                    "algorithm": args.algorithm,
                    "model_arch": args.model_arch, # Add model arch to artifact metadata
                    "episodes": args.episodes if args.episodes is not None else "indefinite" if args.time is None else None,
                    "training_time": args.time if args.time is not None else "indefinite" if args.episodes is None else None,
                    "device": str(device),
                    "lr_actor": args.lra,
                    "lr_critic": args.lrc,
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "clip_epsilon": args.clip_epsilon,
                    "critic_coef": args.critic_coef,
                    "entropy_coef": args.entropy_coef,
                    "model_type": type(actor).__name__,
                    "num_envs": args.num_envs,
                    "update_interval": args.update_interval,
                    "saved_path": saved_path,
                    'model_config': {
                        'hidden_dim': args.hidden_dim,
                        'num_blocks': args.num_blocks,
                        'dropout': args.dropout if 'dropout_rate' in model_kwargs else None # Log dropout if applicable
                    },
                    # Add total env steps to artifact metadata
                    'total_env_steps': getattr(trainer, 'total_env_steps', 0) if trainer else 0
                }

                # Add StreamAC specific metadata if relevant
                if args.algorithm == "streamac":
                    metadata.update({
                        "adaptive_learning_rate": args.adaptive_learning_rate,
                        "target_step_size": args.target_step_size,
                        "backtracking_patience": args.backtracking_patience,
                        "backtracking_zeta": args.backtracking_zeta,
                        "use_obgd": args.use_obgd,
                        "stream_buffer_size": args.stream_buffer_size,
                        "use_sparse_init": args.use_sparse_init
                    })

                # Add metadata to the artifact
                for key, value in metadata.items():
                    if value is not None:  # Only add non-None values
                        artifact.metadata[key] = value

                # Log the artifact to wandb
                wandb.log_artifact(artifact)

                if args.debug:
                    print(f"[DEBUG] Uploaded model to WandB as artifact '{artifact_name}'")
                else:
                    print(f"Uploaded model to WandB as artifact '{artifact_name}'")
            except ImportError:
                print("WandB not available, skipping artifact upload")
            except Exception as e:
                print(f"Error uploading model to WandB: {str(e)}")
                if args.debug:
                    traceback.print_exc()

        # Ensure wandb run finishes if it exists
        if args.wandb and wandb.run:
            wandb.finish()
