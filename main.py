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
# Import VizTracer conditionally to avoid dependency issues
try:
    from viztracer import VizTracer
    VIZTRACER_AVAILABLE = True
except ImportError:
    VIZTRACER_AVAILABLE = False
# Removed unused rlgym imports
from model_architectures import (
    BasicModel, SimBa, SimbaV2, SimbaV2Shared, MLPModel
    # Removed unused functions: fix_compiled_state_dict, extract_model_dimensions, load_partial_state_dict
)
from observation import ActionStacker
from training import Trainer
# Removed unused algorithm imports
# Removed unused concurrent.futures
from tqdm import tqdm # Keep tqdm for the manager process
from typing import Optional, Dict, Any, List, Tuple # Expanded typing imports
from envs.factory import get_env
from envs.vectorized import VectorizedEnv
from curriculum import create_curriculum
from curriculum.manual import ManualCurriculumManager

# --- Environment Stats Calculation ---

def _parse_observation(obs_array: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Parse a single agent's observation array into structured data.
    Assumes DefaultObs observation structure with potential action stacking.

    DefaultObs structure (assumed):
    - [0:3] - Ball position (x,y,z)
    - [3:6] - Ball linear velocity (x,y,z)
    - [6:9] - Ball angular velocity (x,y,z)
    - [9:13] - Car orientation quaternion or similar (4 values)
    - [13:16] - Car position (x,y,z)
    - [16:19] - Car linear velocity (x,y,z)
    - [19:22] - Car angular velocity (x,y,z)
    - [22:25] - Car forward direction (x,y,z)
    - [25:28] - Car up direction (x,y,z)
    - [28:31] - Car right direction (x,y,z)
    - [31] - Car boost amount (0-1)
    - [32] - Car on ground (1) or in air (0)
    - [33] - Car has jump (1) or not (0)
    - [34] - Car has flip (1) or not (0)
    - Additional values may include:
      - Relative positions to other cars, goals, or boost pads
      - Previous actions for stacked observations

    Returns a dictionary of extracted features.
    """
    # Safety check - if observation is not an array or is empty, return empty dict
    if not isinstance(obs_array, np.ndarray) or obs_array.size == 0:
        return {}

    # Basic stats dictionary - adjust indices if observation structure differs
    stats = {}

    try:
        # ------ Basic extraction of raw values ------
        # Ball stats
        stats['ball_pos'] = obs_array[0:3] if len(obs_array) > 3 else np.zeros(3)
        stats['ball_lin_vel'] = obs_array[3:6] if len(obs_array) > 6 else np.zeros(3)
        stats['ball_ang_vel'] = obs_array[6:9] if len(obs_array) > 9 else np.zeros(3)

        # Car stats
        stats['car_pos'] = obs_array[13:16] if len(obs_array) > 16 else np.zeros(3)
        stats['car_lin_vel'] = obs_array[16:19] if len(obs_array) > 19 else np.zeros(3)
        stats['car_ang_vel'] = obs_array[19:22] if len(obs_array) > 22 else np.zeros(3)

        # Car orientation (normalized vectors)
        stats['car_forward'] = obs_array[22:25] if len(obs_array) > 25 else np.array([1, 0, 0])
        stats['car_up'] = obs_array[25:28] if len(obs_array) > 28 else np.array([0, 0, 1])

        # Car state features - indices might vary, so check array length
        if len(obs_array) > 31:
            stats['car_boost'] = float(obs_array[31])  # Typically 0-1 boost amount
        else:
            stats['car_boost'] = 0.0

        if len(obs_array) > 32:
            stats['car_on_ground'] = bool(obs_array[32])  # 1 if on ground, 0 if in air
        else:
            stats['car_on_ground'] = True  # Default to on ground

        # ------ Derived ball metrics ------
        # Ball physics
        stats['ball_speed'] = np.linalg.norm(stats['ball_lin_vel'])
        stats['ball_height'] = stats['ball_pos'][2]  # z-coordinate is height
        stats['ball_ang_speed'] = np.linalg.norm(stats['ball_ang_vel'])  # Ball spin rate

        # Ball position relative to field
        field_center = np.array([0, 0, 0])  # Assume center of field is origin
        stats['ball_dist_to_center'] = np.linalg.norm(stats['ball_pos'] - field_center)

        # Approximate goal positions - this will need adjustment based on actual game
        # Standard Rocket League field dimensions: ~100 units length
        blue_goal = np.array([0, -50, 0])   # Back of blue goal (y-negative)
        orange_goal = np.array([0, 50, 0])  # Back of orange goal (y-positive)

        stats['ball_dist_to_blue_goal'] = np.linalg.norm(stats['ball_pos'] - blue_goal)
        stats['ball_dist_to_orange_goal'] = np.linalg.norm(stats['ball_pos'] - orange_goal)

        # Field positioning - which half of field is ball in?
        stats['ball_in_orange_half'] = stats['ball_pos'][1] > 0  # y > 0 means orange half
        stats['ball_in_blue_half'] = stats['ball_pos'][1] < 0    # y < 0 means blue half

        # Is ball in the air? (above standard car height)
        stats['ball_in_air'] = stats['ball_height'] > 20  # ~20 units is roughly car height

        # ------ Derived car metrics ------
        stats['car_speed'] = np.linalg.norm(stats['car_lin_vel'])
        stats['car_height'] = stats['car_pos'][2]  # z-coordinate is height
        stats['car_ang_speed'] = np.linalg.norm(stats['car_ang_vel'])  # Car rotation rate

        # Car in air info - either from direct observation or derived from height
        if 'car_on_ground' in stats:
            stats['car_in_air'] = not stats['car_on_ground']
        else:
            stats['car_in_air'] = stats['car_height'] > 20  # Approximate

        # Car boost state
        stats['car_has_boost'] = stats['car_boost'] > 0.1  # More than 10% boost

        # Car supersonic state (~95% of max speed)
        max_car_speed = 2300  # Supersonic threshold
        stats['car_supersonic'] = stats['car_speed'] >= (0.95 * max_car_speed)

        # ------ Positional relationships ------
        # Car-ball relationships
        ball_car_vector = stats['ball_pos'] - stats['car_pos']
        stats['ball_car_distance'] = np.linalg.norm(ball_car_vector)
        stats['ball_car_distance_xy'] = np.linalg.norm(ball_car_vector[:2])  # Horizontal distance

        # Is car close enough for potential touch? (~2 car lengths)
        stats['potential_touch'] = stats['ball_car_distance'] < 200

        # Car height vs ball height (positive if car is higher)
        stats['height_diff_car_ball'] = stats['car_height'] - stats['ball_height']

        # Car position relative to goals
        stats['car_dist_to_blue_goal'] = np.linalg.norm(stats['car_pos'] - blue_goal)
        stats['car_dist_to_orange_goal'] = np.linalg.norm(stats['car_pos'] - orange_goal)

        # Field positioning - which half is car in?
        stats['car_in_orange_half'] = stats['car_pos'][1] > 0
        stats['car_in_blue_half'] = stats['car_pos'][1] < 0

        # ------ Aerial play metrics ------
        # Is car in position for aerial hit? (car in air + ball in air + proximity)
        stats['aerial_hit_potential'] = (
            stats['car_in_air'] and
            stats['ball_in_air'] and
            stats['ball_car_distance'] < 300
        )

        # Look direction relative to ball (dot product between car_forward and ball_car_vector)
        # Higher value means car is facing more directly toward ball
        if stats['ball_car_distance'] > 0:  # Avoid division by zero
            normalized_ball_car = ball_car_vector / stats['ball_car_distance']
            stats['facing_ball'] = np.dot(stats['car_forward'], normalized_ball_car)
        else:
            stats['facing_ball'] = 0.0

    except Exception as e:
        # Gracefully handle parsing errors to avoid disrupting training
        if len(obs_array) > 0:  # Only log if obs isn't empty
            print(f"Error parsing observation (len={len(obs_array)}): {str(e)}")
        return {}  # Return empty dict on error

    return stats

def _calculate_stats_from_obs(obs_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Calculate streamlined environment statistics from a single environment's observation dict.
    Focuses on key metrics that indicate play speed, aerial mechanics, and positioning.
    """
    if not obs_dict:
        return {}

    env_stats = {}
    agent_stats = []

    # Process each agent's observation
    for agent_id, obs_array in obs_dict.items():
        if not isinstance(obs_array, np.ndarray) or obs_array.size == 0:
            continue

        # Single agent stats dictionary
        stats = {}
        try:
            # Extract basic positional and velocity information
            stats['ball_pos'] = obs_array[0:3] if len(obs_array) > 3 else np.zeros(3)
            stats['ball_lin_vel'] = obs_array[3:6] if len(obs_array) > 6 else np.zeros(3)
            stats['car_pos'] = obs_array[13:16] if len(obs_array) > 16 else np.zeros(3)
            stats['car_lin_vel'] = obs_array[16:19] if len(obs_array) > 19 else np.zeros(3)

            # Car boost amount (typically index 31)
            stats['car_boost'] = float(obs_array[31]) if len(obs_array) > 31 else 0.0

            # Car ground state (typically index 32)
            stats['car_on_ground'] = bool(obs_array[32]) if len(obs_array) > 32 else True

            # Calculate derived metrics
            stats['ball_height'] = stats['ball_pos'][2]  # z-coordinate
            stats['ball_speed'] = np.linalg.norm(stats['ball_lin_vel'])
            stats['car_height'] = stats['car_pos'][2]  # z-coordinate
            stats['car_speed'] = np.linalg.norm(stats['car_lin_vel'])

            # Positional metrics
            blue_goal = np.array([0, -50, 0])
            orange_goal = np.array([0, 50, 0])
            stats['ball_blue_goal_dist'] = np.linalg.norm(stats['ball_pos'] - blue_goal)
            stats['ball_orange_goal_dist'] = np.linalg.norm(stats['ball_pos'] - orange_goal)
            stats['car_blue_goal_dist'] = np.linalg.norm(stats['car_pos'] - blue_goal)
            stats['car_orange_goal_dist'] = np.linalg.norm(stats['car_pos'] - orange_goal)

            # Distance between car and ball
            ball_car_vector = stats['ball_pos'] - stats['car_pos']
            stats['ball_car_distance'] = np.linalg.norm(ball_car_vector)

            # Binary state calculations
            stats['car_in_air'] = not stats['car_on_ground']
            stats['ball_in_air'] = stats['ball_height'] > 20  # ~20 units is car height
            stats['car_supersonic'] = stats['car_speed'] >= (0.95 * 2300)  # 95% of max speed
            stats['close_to_ball'] = stats['ball_car_distance'] < 200  # Within touch range

            agent_stats.append(stats)
        except Exception as e:
            if len(obs_array) > 0:
                print(f"Error parsing observation (len={len(obs_array)}): {str(e)}")
            continue

    if not agent_stats:
        return {}

    # Calculate environment-wide averages
    try:
        # Core metrics about ball
        env_stats['ball_height_mean'] = float(np.mean([s['ball_height'] for s in agent_stats]))
        env_stats['ball_speed_mean'] = float(np.mean([s['ball_speed'] for s in agent_stats]))

        # Core metrics about cars
        env_stats['car_height_mean'] = float(np.mean([s['car_height'] for s in agent_stats]))
        env_stats['car_speed_mean'] = float(np.mean([s['car_speed'] for s in agent_stats]))
        env_stats['boost_amount_mean'] = float(np.mean([s['car_boost'] for s in agent_stats]))

        # Distance metrics
        env_stats['ball_car_dist_mean'] = float(np.mean([s['ball_car_distance'] for s in agent_stats]))
        env_stats['ball_blue_goal_dist_mean'] = float(np.mean([s['ball_blue_goal_dist'] for s in agent_stats]))
        env_stats['ball_orange_goal_dist_mean'] = float(np.mean([s['ball_orange_goal_dist'] for s in agent_stats]))
        env_stats['car_blue_goal_dist_mean'] = float(np.mean([s['car_blue_goal_dist'] for s in agent_stats]))
        env_stats['car_orange_goal_dist_mean'] = float(np.mean([s['car_orange_goal_dist'] for s in agent_stats]))

        # Percentage metrics (already averages)
        env_stats['supersonic_pct_mean'] = float(np.mean([float(s['car_supersonic']) for s in agent_stats]))
        env_stats['car_in_air_pct_mean'] = float(np.mean([float(s['car_in_air']) for s in agent_stats]))
        env_stats['ball_in_air_pct_mean'] = float(np.mean([float(s['ball_in_air']) for s in agent_stats]))
        env_stats['close_to_ball_pct_mean'] = float(np.mean([float(s['close_to_ball']) for s in agent_stats]))

    except Exception as e:
        print(f"Error calculating environment averages: {str(e)}")
        return {}

    return env_stats

def _aggregate_env_stats(env_stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate statistics across multiple environments, focusing only on means.
    """
    if not env_stats_list:
        return {}

    # Filter out empty dictionaries
    valid_env_stats = [stats for stats in env_stats_list if stats]

    if not valid_env_stats:
        return {}

    aggregated_stats = {}

    # List of metrics we want to track
    tracked_metrics = [
        'ball_height_mean',
        'ball_speed_mean',
        'car_height_mean',
        'car_speed_mean',
        'boost_amount_mean',
        'ball_car_dist_mean',
        'supersonic_pct_mean',
        'car_in_air_pct_mean',
        'ball_in_air_pct_mean',
        'close_to_ball_pct_mean',
        'ball_blue_goal_dist_mean',
        'ball_orange_goal_dist_mean',
        'car_blue_goal_dist_mean',
        'car_orange_goal_dist_mean'
    ]

    # Calculate the mean across environments for each metric
    for metric in tracked_metrics:
        values = [
            env_stats[metric] for env_stats in valid_env_stats
            if metric in env_stats and not (
                np.isnan(env_stats[metric]) or
                np.isinf(env_stats[metric])
            )
        ]

        if values:
            # Add env_ prefix for WandB logging
            aggregated_stats[f"env_{metric}"] = float(np.mean(values))

    # Add count of environments that contributed to stats
    aggregated_stats["env_count"] = len(valid_env_stats)

    return aggregated_stats

# --- TQDM Manager Process ---
class TqdmManager:
    """Manages a tqdm progress bar in a separate process."""
    def __init__(self, queue: mp.Queue, total: Optional[int], desc: str, bar_format: str, dynamic_ncols: bool, use_viztracer: bool = False, debug: bool = False):
        self.queue = queue
        self.total = total
        self.desc = desc
        self.bar_format = bar_format
        self.dynamic_ncols = dynamic_ncols
        self.use_viztracer = use_viztracer
        self.debug = debug
        self.process: Optional[mp.Process] = None
        self._initial_postfix = {} # Store initial postfix if sent early

    def _run(self):
        """The target function for the tqdm process."""
        tracer = None
        if self.use_viztracer and VIZTRACER_AVAILABLE:
            try:
                tracer = VizTracer(
                    output_file=f"viztracer_tqdm_{mp.current_process().pid}.json",
                    log_async=True,
                    tracer_entries=50000000, # Increase buffer size
                    verbose=0
                )
                tracer.start()
                if self.debug: print(f"[DEBUG] TQDM Manager {mp.current_process().pid} started VizTracer.")
            except Exception as e:
                print(f"TQDM WARNING: Failed to start VizTracer in TQDM Manager {mp.current_process().pid}: {e}")
                tracer = None

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
            # Stop VizTracer if it was started
            if tracer:
                try:
                    tracer.stop()
                    tracer.save()
                    if self.debug: print(f"[DEBUG] TQDM Manager {mp.current_process().pid} stopped and saved VizTracer.")
                except Exception as e:
                    print(f"TQDM WARNING: Failed to stop/save VizTracer in TQDM Manager {mp.current_process().pid}: {e}")

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
    batch_size: int = 128,
    buffer_size: int = 262144,
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
    # Removed pretraining_transition_steps
    # Learning rate decay parameters
    use_lr_decay: bool = False,
    lr_decay_rate: float = 0.7,
    lr_decay_steps: int = 1000000,
    min_lr: float = 3e-5,
    # Intrinsic rewards parameters
    use_intrinsic_rewards: bool = False,  # Changed default to match CLI default (False)
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
    # Add reward scaling parameters (REMOVED)
    # use_reward_scaling: bool = True,
    # reward_scaling_G_max: float = 10.0,
    # Add TQDM queue
    tqdm_q: Optional[mp.Queue] = None,
    # Add viztracer flag
    use_viztracer: bool = False
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

    # Validate buffer_size for PPO
    if algorithm == "ppo" and buffer_size < update_interval:
        print(f"WARNING: PPO buffer_size ({buffer_size}) is smaller than update_interval ({update_interval})")
        print(f"This will cause PPO stats to appear as 0 because the buffer will be overwritten before updates.")
        print(f"Automatically increasing buffer_size to {update_interval}")
        buffer_size = update_interval

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
        buffer_size=buffer_size,

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
        # Removed pretraining_transition_steps
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
        # Pass reward scaling parameters to Trainer (REMOVED)
        # use_reward_scaling=use_reward_scaling,
        # reward_scaling_G_max=reward_scaling_G_max,
        skill_zscore_threshold=args.skill_zscore_threshold,
    )

    # --- MODEL LOADING LOGIC ---
    loaded_successfully = False  # Flag to track loading status
    if model_path_to_load:
        print(f"Attempting to load model from: {model_path_to_load}")
        if trainer.load_models(model_path_to_load):
             print(f"Successfully loaded model and state from {model_path_to_load}")
             loaded_successfully = True  # Set flag
             # Add a debug print to confirm the trainer's state AFTER loading
             if debug:
                 print(f"[DEBUG] Trainer state after load: pretraining_completed={trainer.pretraining_completed}, "
                       f"step_offset={trainer.training_step_offset}, total_env_steps={trainer.total_env_steps}")
             print(f"Resuming training from step offset {trainer.training_step_offset}, total env steps {trainer.total_env_steps}")
        else:
             print(f"Failed to load model from {model_path_to_load}. Starting fresh.")
             trainer.training_step_offset = 0  # Ensure offsets are 0 if loading fails
             trainer.total_env_steps = 0
             trainer.total_episodes_offset = 0
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

    vec_env = env_class(**env_args, use_viztracer=use_viztracer) # Pass viztracer flag

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
        pretraining_end_step = trainer._get_pretraining_end_step() # Use trainer's method
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
    # Initialize total_episodes_so_far using the loaded offset
    total_episodes_so_far = trainer.total_episodes_offset
    # Initialize total_env_steps using the loaded value from trainer
    total_env_steps = trainer.total_env_steps
    last_update_time = time.time()
    # Initialize last_save_episode using the loaded offset
    last_save_episode = trainer.total_episodes_offset
    last_intrinsic_reset_episode = 0 # Track last episode where intrinsic models were reset

    # Variables to track steps per second
    steps_per_second = 0
    last_steps_time = time.time()
    last_steps_count = 0

    # Initialize episode rewards lazily within the loop
    episode_rewards = {}

    # Add a list to track rewards of completed episodes between PPO updates
    completed_episode_rewards_since_last_update = []

    # Update initial stats_dict with loaded values
    stats_dict["Episodes"] = total_episodes_so_far  # Use loaded episode count
    if algorithm == "ppo":
        # Experience counter should reset between updates, so start at 0
        stats_dict["Exp"] = f"0/{update_interval}"
    else:
        # Use loaded step count for StreamAC/SAC progress
        stats_dict["Steps"] = total_env_steps  # Initialize steps display with loaded value
        # Updates count might be restored via algorithm state loading, check if possible
        stats_dict["Updates"] = getattr(trainer.algorithm, "update_count", 0) if hasattr(trainer.algorithm, 'update_count') else 0

    # Send potentially updated initial postfix to TQDM manager
    if tqdm_q:
        tqdm_q.put(stats_dict.copy())

    last_progress_update = start_time  # For updating time-based progress bar
    should_continue = True  # Initialize the control variable

    try:
        # Define pause file path for the pause/resume functionality
        pause_file_path = "pause.flag"  # Flag file to check for pausing

        # Let's keep training until it's time to stop
        while should_continue:
            # --- PAUSE CHECK ---
            if os.path.exists(pause_file_path):
                if not stats_dict.get("Status") == "[PAUSED]":  # Avoid spamming logs/queue
                    print("\nPause signal file found. Pausing training...")
                    stats_dict["Status"] = "[PAUSED]"
                    if tqdm_q:
                        tqdm_q.put(stats_dict.copy())  # Update TQDM postfix

                # Loop while the pause file exists
                while os.path.exists(pause_file_path):
                    time.sleep(1)  # Check every second

                # Pause file has been removed
                print("Resume signal detected. Resuming training...")
                if "Status" in stats_dict:  # Remove pause status
                    stats_dict.pop("Status")
                if tqdm_q:
                    tqdm_q.put(stats_dict.copy())  # Update TQDM postfix
            # --- END PAUSE CHECK ---

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

                # --- Prepare actions for environment step ---
                if isinstance(action_batch, torch.Tensor):
                    action_batch_np = action_batch.cpu().numpy()
                else:
                    action_batch_np = np.array(action_batch)

                for i, action_np_val in enumerate(action_batch_np): # Renamed to avoid conflict
                    env_idx = all_env_indices[i]
                    agent_id = all_agent_ids[i]
                    actions_dict_list[env_idx][agent_id] = action_np_val

            # Step all environments forward in parallel
            results, dones, episode_counts = vec_env.step(actions_dict_list)
            current_batch_env_steps = len(all_obs_list)
            total_env_steps += current_batch_env_steps
            trainer.total_env_steps = total_env_steps

            # Calculate steps per second
            current_time_calc = time.time() # Renamed to avoid conflict
            elapsed_since_last_calc = current_time_calc - last_steps_time
            if elapsed_since_last_calc >= 1.0:
                steps_this_period = total_env_steps - last_steps_count
                steps_per_second = steps_this_period / elapsed_since_last_calc
                last_steps_time = current_time_calc
                last_steps_count = total_env_steps
                stats_dict["Steps/s"] = f"{steps_per_second:.1f}"
                if tqdm_q:
                    tqdm_q.put(stats_dict.copy())

            # --- Process Results and Store Experiences ---
            if not test:
                # Prepare data for batch storage
                batch_obs_store = []
                batch_actions_store = []
                batch_log_probs_store = []
                batch_rewards_store = []
                batch_values_store = []
                batch_dones_store = []
                batch_env_ids_store = [] # For intrinsic rewards if calculated individually

                current_exp_idx = 0
                for env_idx, result_item in enumerate(results): # Renamed to avoid conflict
                    if isinstance(result_item, tuple):
                        if len(result_item) == 5: _, next_obs_dict, reward_dict, terminated_dict, truncated_dict = result_item
                        else: next_obs_dict, reward_dict, terminated_dict, truncated_dict = result_item
                    else: continue

                    if env_idx not in episode_rewards: episode_rewards[env_idx] = {}

                    for agent_id in reward_dict.keys():
                        if current_exp_idx < len(all_obs_list):
                            # Original data from before env step
                            obs_to_store = all_obs_list[current_exp_idx]
                            action_to_store = action_batch[current_exp_idx] # Original tensor/value
                            log_prob_to_store = log_prob_batch[current_exp_idx]
                            value_to_store = value_batch[current_exp_idx]

                            # Data from after env step
                            extrinsic_reward = reward_dict[agent_id]
                            done_val = terminated_dict[agent_id] or truncated_dict[agent_id] # Renamed
                            next_obs = next_obs_dict.get(agent_id, None)

                            batch_obs_store.append(obs_to_store)
                            batch_actions_store.append(action_to_store)
                            batch_log_probs_store.append(log_prob_to_store)
                            batch_rewards_store.append(extrinsic_reward) # Store original extrinsic for now
                            batch_values_store.append(value_to_store)
                            batch_dones_store.append(done_val)
                            batch_env_ids_store.append(env_idx)


                            if agent_id not in episode_rewards[env_idx]:
                                episode_rewards[env_idx][agent_id] = 0.0
                            episode_rewards[env_idx][agent_id] += extrinsic_reward

                            if done_val:
                                action_stacker.reset_agent(agent_id)
                                trainer.reset_auxiliary_tasks()
                            current_exp_idx += 1
                        else:
                            if debug: print(f"[DEBUG] Warning: current_exp_idx ({current_exp_idx}) exceeded all_obs_list length ({len(all_obs_list)})")

                # Call batch store experience
                if len(batch_obs_store) > 0:
                    # Pass next_obs_batch if available and needed for batch intrinsic rewards
                    # For now, intrinsic rewards in store_experience_batch might be individual
                    final_rewards_batch = trainer.store_experience_batch(
                        states=batch_obs_store,
                        actions=batch_actions_store,
                        log_probs=batch_log_probs_store,
                        rewards=batch_rewards_store, # Original extrinsic rewards
                        values=batch_values_store,
                        dones=batch_dones_store,
                        env_ids=batch_env_ids_store
                        # next_obs_batch=batch_next_obs_store # If you collect next_obs in a batch
                    )
                    collected_experiences += len(batch_obs_store)
                    if debug and algorithm == "ppo" and collected_experiences % 100 == 0:
                        print(f"[DEBUG PPO] Batch stored {len(batch_obs_store)} experiences. Total collected: {collected_experiences}")


            # --- Process Results (Test Mode Only - No Storage) ---
            elif test: # Only process for episode tracking if in test mode
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
                            extrinsic_reward = reward_dict[agent_id] # ORIGINAL extrinsic reward
                            done = terminated_dict[agent_id] or truncated_dict[agent_id]
                            next_obs = next_obs_dict.get(agent_id, None)

                            # For StreamAC/SAC, update the stored experience with actual reward/done
                            if not test: # Only update if not testing
                                if algorithm == "streamac":
                                     # StreamAC updates happen within store_experience, called earlier
                                     # We need to update the *last* stored reward/done if store_experience
                                     # doesn't handle the next_state logic correctly.
                                     # update_experience_with_intrinsic_reward handles scaling and intrinsic addition
                                     total_reward_for_tracking = extrinsic_reward # Default to original
                                     if next_obs is not None:
                                         # Only call intrinsic reward calculation if enabled
                                         if trainer.use_intrinsic_rewards:
                                             # Use the single update method to get the combined reward
                                             total_reward_for_tracking = trainer.update_experience_with_intrinsic_reward(
                                                 state=all_obs_list[exp_idx],
                                                 action=action_batch[exp_idx], # Original action
                                                 next_state=next_obs,
                                                 reward=extrinsic_reward, # Pass ORIGINAL extrinsic
                                                 env_id=env_idx,
                                                 done=done # Pass done flag
                                                 # store_idx is None for StreamAC here
                                             )
                                         # Otherwise just use extrinsic reward - no need to call the intrinsic method
                                     # Defensive check/initialization before accumulating reward
                                     if agent_id not in episode_rewards[env_idx]:
                                         episode_rewards[env_idx][agent_id] = 0.0
                                         if debug: print(f"[DEBUG] Initialized episode_rewards for {agent_id} in env {env_idx} (during {algorithm} result processing)")
                                     # Accumulate the ORIGINAL extrinsic reward for episode tracking
                                     episode_rewards[env_idx][agent_id] += extrinsic_reward

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
                                     # Accumulate the ORIGINAL extrinsic reward for episode tracking
                                     episode_rewards[env_idx][agent_id] += extrinsic_reward

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
                            # Use the accumulated ORIGINAL extrinsic rewards for average calculation
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
            if trainer.use_intrinsic_rewards and use_pretraining and not trainer.pretraining_completed and \
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

                # --- Calculate Environment Statistics ---
                # Only calculate stats before policy updates to minimize overhead
                aggregated_env_stats = {}
                try:
                    # Calculate environment stats from the current observations
                    if vec_env.obs_dicts:
                        # Process all valid environments
                        env_stats_list = []
                        for env_idx, obs_dict in enumerate(vec_env.obs_dicts):
                            if obs_dict:  # Skip empty observation dicts
                                env_stats = _calculate_stats_from_obs(obs_dict)
                                if env_stats:  # Only include if stats were calculated successfully
                                    env_stats_list.append(env_stats)

                        # Aggregate across environments
                        if env_stats_list:
                            aggregated_env_stats = _aggregate_env_stats(env_stats_list)

                            if debug and aggregated_env_stats:
                                print(f"[DEBUG] Calculated stats from {aggregated_env_stats.get('env_count', 0)} environments")
                except Exception as e:
                    # Ensure stats calculation never interrupts training
                    if debug:
                        print(f"[DEBUG] Error calculating environment stats: {str(e)}")
                        traceback.print_exc()
                    aggregated_env_stats = {}
                # --- End Environment Statistics Calculation ---

                # Perform the policy update.
                if algorithm == "ppo":
                    # For PPO, do a normal batch update and pass the completed episode rewards and total env steps
                    stats = trainer.update(
                        completed_episode_rewards=completed_episode_rewards_since_last_update,
                        total_env_steps=total_env_steps, # Pass total env steps
                        steps_per_second=steps_per_second, # Pass steps per second
                        env_stats=aggregated_env_stats # Pass environment stats
                    )
                    # Reset the list of completed episode rewards
                    completed_episode_rewards_since_last_update = []
                    collected_experiences = 0 # Reset PPO experience counter after update
                else:
                    # For StreamAC/SAC, update might happen internally or via trainer.update()
                    # If trainer.update() is called, it should just fetch metrics for online algos
                    stats = trainer.update(
                        total_env_steps=total_env_steps,
                        steps_per_second=steps_per_second,
                        env_stats=aggregated_env_stats # Pass environment stats
                    )
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
                needs_postfix_update = False # Flag if pretraining status changed

                # Check pretraining completion status based on current step count
                current_step = trainer._true_training_steps() # Use trainer's step counter
                # Transition is now immediate, no separate phase
                if current_step >= pretraining_end_step and not trainer.pretraining_completed:
                    print(f"Pretraining completed at step {current_step}/{pretraining_end_step}")
                    # trainer.pretraining_completed is set inside trainer.update() -> _update_pretraining_state()
                    needs_postfix_update = True # Trigger postfix update

                if in_pretraining:
                    new_mode = "PreTraining"
                    new_progress = f"{current_step}/{pretraining_end_step}"
                    if stats_dict.get("Mode") != new_mode or stats_dict.get("PT_Progress") != new_progress:
                        stats_dict["Mode"] = new_mode
                        stats_dict["PT_Progress"] = new_progress
                        needs_postfix_update = True
                # Removed transition phase logic
                else: # Pretraining completed
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
        # Clean up the pause file if it exists on exit
        if os.path.exists(pause_file_path):
            try:
                os.remove(pause_file_path)
                print(f"Removed '{pause_file_path}' on exit.")
            except OSError as e:
                print(f"Error removing '{pause_file_path}' on exit: {e}")

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
    # --- Early parse for viztracer to set start_method ---
    # Need to know if viztracer is enabled before setting the start method,
    # especially to choose between fork/spawn on Linux.
    temp_parser = argparse.ArgumentParser(add_help=False) # Avoid conflict with main help
    temp_parser.add_argument('--viztracer', action='store_true')
    known_args, _ = temp_parser.parse_known_args()
    use_viztracer_early = known_args.viztracer
    # --- End early parse ---

    # Set start method based on OS and viztracer flag
    if sys.platform == 'darwin':
        mp.set_start_method('spawn', force=True)
        print("[INFO] Using 'spawn' start method (macOS default)")
    elif sys.platform == 'linux':
        if use_viztracer_early:
            mp.set_start_method('spawn', force=True)
            print("[INFO] Using 'spawn' start method (VizTracer enabled on Linux)")
        else:
            # Check if fork is available and safe (optional, but good practice)
            if hasattr(os, 'fork') and hasattr(mp, 'get_context'):
                 try:
                     mp.get_context('fork') # Check if 'fork' context is supported
                     mp.set_start_method('fork', force=True)
                     print("[INFO] Using 'fork' start method (default for Linux, VizTracer disabled)")
                 except ValueError:
                      mp.set_start_method('spawn', force=True) # Fallback if fork isn't available/safe
                      print("[INFO] Using 'spawn' start method (fork not available/safe on Linux)")
            else:
                 mp.set_start_method('spawn', force=True) # Fallback if checks aren't possible
                 print("[INFO] Using 'spawn' start method (fork check failed on Linux)")

    # Set up Ctrl+C handler to exit gracefully.
    signal.signal(signal.SIGINT, signal_handler)

    # --- Main Argument Parsing (remains here) ---
    parser = argparse.ArgumentParser(description='RLBot Training Script')
    parser.add_argument('--render', action='store_true', help='Enable rendering of the game environment')

    # Add viztracer argument
    parser.add_argument('--viztracer', action='store_true', default=False, help='Enable VizTracer profiling (default: disabled)')

    # Allow user to specify training duration either by episode count OR by time.
    training_duration = parser.add_mutually_exclusive_group()
    training_duration.add_argument('-e', '--episodes', type=int, default=None, help='Number of episodes to run')
    training_duration.add_argument('-t', '--time', type=str, default=None,
                                  help='Training duration in format: 5m (minutes), 5h (hours), 5d (days)')

    parser.add_argument('-n', '--num_envs', type=int, default=300,
                        help='Number of parallel environments to run for faster data collection')
    parser.add_argument('--update_interval', type=int, default=204800,
                        help='Number of experiences to collect before updating the policy (PPO)')
    parser.add_argument('--ppo_buffer_size', type=int, default=262144,
                        help='Size of the PPO experience buffer (should be >= update_interval)')
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
    parser.add_argument('--lra', type=float, default=3e-4, help='Learning rate for actor network')
    parser.add_argument('--lrc', type=float, default=5e-4, help='Learning rate for critic network')

    # Learning rate decay
    parser.add_argument('--lr-decay', action='store_true', default=True, help='Enable learning rate decay')
    parser.add_argument('--lr-decay-rate', type=float, default=0.005, help='Learning rate decay factor (e.g., 0.7 means decay to 70%% over decay steps)')
    parser.add_argument('--lr-decay-steps', type=int, default=1000000, help='Number of steps over which to decay the learning rate')
    parser.add_argument('--min-lr', type=float, default=3e-5, help='Minimum learning rate after decay')

    # Discount factors
    parser.add_argument('--gamma', type=float, default=0.997, help='Discount factor for future rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.98, help='Lambda parameter for Generalized Advantage Estimation')

    # PPO parameters
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clipping parameter')
    parser.add_argument('--critic_coef', type=float, default=1.5, help='Weight of the critic loss')
    parser.add_argument('--entropy_coef', type=float, default=0.001, help='Weight of the entropy bonus (encourages exploration)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm for clipping')

    # Skill rating z-score threshold
    parser.add_argument('--skill-zscore-threshold', type=float, default=0.3,
                        help='Z-score threshold for skill rating win/loss/draw (default: 0.3)')

    # Distributional critic args removed - no longer used
    # ------------------------------------

    # Training loop parameters
    parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of PPO epochs per update')
    parser.add_argument('--batch_size', type=int, default=24576, help='Batch size for PPO updates')

    # Weight clipping parameters removed - no longer used

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


    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Save the model every N episodes')

    # Actor/Shared Network Config (Defaults for MLP: 1024/4)
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension for the ACTOR network (or shared network if using simbav2-shared/mlp)')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of layers/blocks in the ACTOR network (or shared network if using simbav2-shared/mlp)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for regularization (used by basic/simba/mlp architectures)')

    # Critic Network Config (Defaults for MLP: 1024/4)
    parser.add_argument('--hidden_dim_critic', type=int, default=1024, help='Hidden dimension for the CRITIC network (only used if model-arch is not shared)')
    parser.add_argument('--num_blocks_critic', type=int, default=4, help='Number of layers/blocks in the CRITIC network (only used if model-arch is not shared)')

    # Action stacking parameters
    parser.add_argument('--stack_size', type=int, default=5, help='Number of previous actions to stack')

    # Auxiliary learning parameters - Corrected Logic
    # By default, auxiliary tasks are enabled (default=True).
    # Using --no-auxiliary will set args.auxiliary to False.
    parser.add_argument('--no-auxiliary', action='store_false', dest='auxiliary',
                        help='Disable auxiliary task learning (SR and RP tasks)')
    parser.set_defaults(auxiliary=False)


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
    parser.set_defaults(pretraining=False)  # Changed default to True so --no-pretraining has an effect
    parser.add_argument('--pretraining-fraction', type=float, default=0.1, help='Fraction of total training time/episodes to use for pre-training (default: 0.1)')
    parser.add_argument('--pretraining-sr-weight', type=float, default=10.0, help='Weight for State Representation task during pre-training (default: 10.0)')
    parser.add_argument('--pretraining-rp-weight', type=float, default=5.0, help='Weight for Reward Prediction task during pre-training (default: 5.0)')

    # Intrinsic reward parameters
    parser.add_argument('--use-intrinsic', action='store_true', help='Use intrinsic rewards during pre-training')
    parser.add_argument('--no-intrinsic', action='store_false', dest='use_intrinsic', help='Disable intrinsic rewards during pre-training')
    parser.set_defaults(use_intrinsic=False)

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
    parser.add_argument('--model-arch', type=str, default='mlp', choices=['mlp', 'basic', 'simba', 'simbav2', 'simbav2-shared'],
                       help='Model architecture to use (mlp, basic, simba, simbav2, simbav2-shared)')

    # Add reward scaling args here, before parse_args() (REMOVED - Handled in Algorithm)
    # parser.add_argument('--use-reward-scaling', action='store_true', dest='use_reward_scaling', help='Enable SimbaV2 reward scaling')
    # parser.add_argument('--no-reward-scaling', action='store_false', dest='use_reward_scaling', help='Disable SimbaV2 reward scaling')
    # parser.set_defaults(use_reward_scaling=True)
    # parser.add_argument('--reward-scaling-gmax', type=float, default=3.0, help='G_max hyperparameter for reward scaling')

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
    # Common args for all models
    common_kwargs = {
        'device': device
    }
    # Actor/Shared args (use actor defaults)
    actor_kwargs = {
        'hidden_dim': args.hidden_dim,
        'num_blocks': args.num_blocks,
        **common_kwargs
    }
    # Critic args (use critic defaults, only relevant for separate architectures)
    critic_kwargs = {
        'hidden_dim': args.hidden_dim_critic,
        'num_blocks': args.num_blocks_critic,
        **common_kwargs
    }


    if args.model_arch == 'mlp':
        ModelClass = MLPModel
        actor_kwargs['dropout_rate'] = args.dropout
        critic_kwargs['dropout_rate'] = args.dropout
        actor = ModelClass(obs_shape=obs_space_dims, action_shape=action_space_dims, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, dropout_rate=args.dropout)
        critic = ModelClass(obs_shape=obs_space_dims, action_shape=1, hidden_dim=args.hidden_dim_critic, num_blocks=args.num_blocks_critic, dropout_rate=args.dropout)
    elif args.model_arch == 'basic':
        ModelClass = BasicModel
        actor_kwargs['dropout_rate'] = args.dropout
        critic_kwargs['dropout_rate'] = args.dropout # Use same dropout for basic critic
        actor = ModelClass(obs_shape=obs_space_dims, action_shape=action_space_dims, **actor_kwargs)
        critic = ModelClass(obs_shape=obs_space_dims, action_shape=1, **critic_kwargs)
    elif args.model_arch == 'simba':
        ModelClass = SimBa
        actor_kwargs['dropout_rate'] = args.dropout
        critic_kwargs['dropout_rate'] = args.dropout # Use same dropout for simba critic
        actor = ModelClass(obs_shape=obs_space_dims, action_shape=action_space_dims, **actor_kwargs)
        critic = ModelClass(obs_shape=obs_space_dims, action_shape=1, **critic_kwargs)
    elif args.model_arch == 'simbav2':
        ModelClass = SimbaV2
        # SimbaV2 does not use dropout_rate
        actor = ModelClass(obs_shape=obs_space_dims, action_shape=action_space_dims, **actor_kwargs)
        critic = ModelClass(obs_shape=obs_space_dims, action_shape=args.num_atoms, **critic_kwargs)
    elif args.model_arch == 'simbav2-shared':
        ModelClass = SimbaV2Shared
        # SimbaV2Shared does not use dropout_rate
        # Instantiate ONE model and use it for both actor and critic
        # Use actor_kwargs for shared model config
        shared_model = ModelClass(obs_shape=obs_space_dims, action_shape=action_space_dims, **actor_kwargs)
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

                # Learning rate decay
                "use_lr_decay": args.lr_decay,
                "lr_decay_rate": args.lr_decay_rate,
                "lr_decay_steps": args.lr_decay_steps,
                "min_lr": args.min_lr,

                # Model Details
                "model_arch": args.model_arch, # Log model architecture
                # Actor/Shared config
                "hidden_dim_actor": args.hidden_dim,
                "num_blocks_actor": args.num_blocks,
                # Critic config (only relevant if not shared)
                "hidden_dim_critic": args.hidden_dim_critic if args.model_arch != 'simbav2-shared' else None,
                "num_blocks_critic": args.num_blocks_critic if args.model_arch != 'simbav2-shared' else None,
                # Dropout (only relevant for basic/simba)
                "dropout": args.dropout if args.model_arch in ['basic', 'simba'] else None,
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
        print(f"[DEBUG] Intrinsic rewards enabled: {args.use_intrinsic}") # Print intrinsic rewards state
        if args.model_arch == 'simbav2-shared':
            print(f"[DEBUG] Shared Model Config: hidden_dim={args.hidden_dim}, num_blocks={args.num_blocks}")
            print(f"[DEBUG] Shared model instance: {actor}")
        else:
            print(f"[DEBUG] Actor Model Config: hidden_dim={args.hidden_dim}, num_blocks={args.num_blocks}")
            print(f"[DEBUG] Actor model: {actor}")
            print(f"[DEBUG] Critic Model Config: hidden_dim={args.hidden_dim_critic}, num_blocks={args.num_blocks_critic}")
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
        dynamic_ncols=dynamic_ncols,
        use_viztracer=args.viztracer, # Pass flag
        debug=args.debug # Pass debug flag too
    )
    tqdm_manager_instance.start()

    # --- End TQDM Manager Initialization ---

    # --- Start Main Process VizTracer ---
    main_tracer = None
    if args.viztracer and VIZTRACER_AVAILABLE:
        print("Starting VizTracer for main process...")
        try:
            main_tracer = VizTracer(
                output_file="viztracer_main.json",
                log_async=True,
                tracer_entries=50000000, # Increase buffer size
                # Use pid_suffix=True to avoid overwriting if multiple main processes run somehow
                pid_suffix=True,
                verbose=1 # Show some output for main process
            )
            main_tracer.start()
        except Exception as e:
            print(f"MAIN WARNING: Failed to start VizTracer for main process: {e}")
            main_tracer = None
    # --- End Main Process VizTracer ---

    trainer = None # Initialize trainer to None
    saved_path = None
    try:
        # Start the main training process
        # Pass args.model path to run_training
        trainer = run_training(
            actor=actor,
            critic=critic, # Pass the potentially shared instance
            # Pass reward scaling args (REMOVED)
            # training_step_offset is now handled inside run_training via load_models
            device=device,
            num_envs=args.num_envs,
            total_episodes=args.episodes if args.time is None else None,
            training_time=training_time_seconds,
            render=args.render,
            update_interval=args.update_interval,
            buffer_size=args.ppo_buffer_size,
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
            # Learning rate decay parameters
            use_lr_decay=args.lr_decay,
            lr_decay_rate=args.lr_decay_rate,
            lr_decay_steps=args.lr_decay_steps,
            min_lr=args.min_lr,
            # Removed pretraining_transition_steps argument
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
            tqdm_q=tqdm_queue, # Pass the queue to the training function
            use_viztracer=args.viztracer # Pass flag here
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

        # --- Stop Main Process VizTracer ---
        if 'main_tracer' in locals() and main_tracer: # Check if tracer was initialized and started
            print("Stopping and saving VizTracer for main process...")
            try:
                main_tracer.stop()
                main_tracer.save()
                print("VizTracer data saved.") # Simplified message
            except Exception as e:
                print(f"MAIN WARNING: Failed to stop/save VizTracer for main process: {e}")
        # --- End Stop Main Process VizTracer ---

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
                        'hidden_dim_actor': args.hidden_dim,
                        'num_blocks_actor': args.num_blocks,
                        'hidden_dim_critic': args.hidden_dim_critic if args.model_arch != 'simbav2-shared' else None,
                        'num_blocks_critic': args.num_blocks_critic if args.model_arch != 'simbav2-shared' else None,
                        'dropout': args.dropout if args.model_arch in ['basic', 'simba'] else None
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
