import os

import numpy as np
import torch
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rewards import BallProximityReward, BallNetProximityReward
from models import BasicModel, SimBa, fix_compiled_state_dict, extract_model_dimensions
from training import PPOTrainer
import concurrent.futures
import time
import argparse
from tqdm import tqdm
import signal
import sys

def get_env(renderer=None):
    """Creates a standardized RL environment for Rocket League.  This environment uses
    a combination of mutators, observations, actions, rewards, and termination conditions
    to define the learning task."""
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),  # Forces 2v2 matches
            KickoffMutator()  # Start each episode with a kickoff
        ),
        obs_builder=DefaultObs(zero_padding=2),  # Use a basic observation builder, padding for a maximum of 4 players
        action_parser=RepeatAction(LookupTableAction(), repeats=8),  # Use a discrete action space, repeating the chosen action 8 times
        reward_fn=CombinedReward(  # Combine several reward functions to create a more comprehensive reward signal
            (GoalReward(), 12.),  # Large reward for scoring goals
            (TouchReward(), 6.),  # Moderate reward for touching the ball
            (BallNetProximityReward(), 3.),  # Smaller reward for getting the ball closer to the opponent's net
            (BallProximityReward(), 1.),  # Small reward for getting closer to the ball
        ),
        termination_cond=GoalCondition(),  # End the episode when a goal is scored
        truncation_cond=AnyCondition(  # Truncate the episode if it goes on for too long or if the ball isn't touched
            TimeoutCondition(300.),  # Max episode length of 300 steps (300/8 = 37.5 seconds)
            NoTouchTimeoutCondition(30.)  # No touch for 30 steps
        ),
        transition_engine=RocketSimEngine(),  # Use the default physics engine
        renderer=renderer  # Use provided renderer if available (or None)
    )


def run_training(
    actor,
    critic,
    device,
    num_envs: int,
    total_episodes: int = None,
    training_time: float = None,
    render: bool = False,
    update_interval: int = 1000,
    use_wandb: bool = False,
    debug: bool = False,
    use_compile: bool = True,
    use_amp: bool = True,
    save_interval: int = 200,
    output_path: str = None,
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
    batch_size: int = 64
):
    """
    Runs the main training loop using a vectorized environment approach.
    This function orchestrates the data collection, policy updates, and model saving.
    Training can be limited by episodes (total_episodes) or time (training_time in seconds).
    """
    # Verify that at least one termination condition is provided
    if total_episodes is None and training_time is None:
        raise ValueError("Either total_episodes or training_time must be provided")

    actor.to(device)
    critic.to(device)

    # Initialize the PPO trainer with the specified hyperparameters.
    trainer = PPOTrainer(
        actor,
        critic,
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
        use_wandb=use_wandb,
        debug=debug,
        use_compile=use_compile,
        use_amp=use_amp
    )

    # Set up the vectorized environment to run multiple instances in parallel.
    vec_env = VectorizedEnv(num_envs=num_envs, render=render)

    # Record the start time for time-based training
    start_time = time.time()

    # Create the appropriate progress bar based on the training mode
    if training_time is not None:
        # For time-based training, use the time in seconds as total with a different format
        progress_bar = tqdm(
            total=int(training_time),
            desc="Time",
            bar_format='{desc}: {percentage:3.0f}% [{elapsed}<{remaining}] |{bar}| {postfix}',
            dynamic_ncols=True
        )
    else:
        # For episode-based training, use the traditional format
        progress_bar = tqdm(
            total=total_episodes,
            desc="Episodes",
            bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%|{bar}| {postfix}',
            dynamic_ncols=True
        )

    # Initialize a dictionary to hold various statistics for display.
    stats_dict = {
        "Device": device,
        "Envs": num_envs,
        "Exp": "0/0",
        "Episodes": 0,
        "Reward": "0.0",
        "PLoss": "0.0",
        "VLoss": "0.0",
        "Entropy": "0.0"
    }
    progress_bar.set_postfix(stats_dict)

    # Initialize variables for tracking training progress.
    collected_experiences = 0
    total_episodes_so_far = 0
    last_update_time = time.time()
    last_save_episode = 0
    episode_rewards = {i: {agent_id: 0 for agent_id in vec_env.obs_dicts[i]} for i in range(num_envs)}
    last_progress_update = start_time

    try:
        # Main training loop, continues until the target number of episodes is reached or time is up.
        should_continue = True
        while should_continue:
            current_time = time.time()
            elapsed = current_time - start_time

            # Check termination conditions
            if training_time is not None:
                # Update progress bar for time-based training (update once per second to avoid overhead)
                if current_time - last_progress_update >= 1.0:
                    progress_bar.n = min(int(elapsed), int(training_time))
                    progress_bar.refresh()
                    last_progress_update = current_time

                should_continue = elapsed < training_time
            else:
                should_continue = total_episodes_so_far < total_episodes

            if not should_continue:
                break

            # Collect observations from all environments for batched inference
            all_obs = []
            all_env_indices = []
            all_agent_ids = []

            # Organize observations for more efficient processing
            for env_idx, obs_dict in enumerate(vec_env.obs_dicts):
                for agent_id, obs in obs_dict.items():
                    all_obs.append(obs)
                    all_env_indices.append(env_idx)
                    all_agent_ids.append(agent_id)

            # Only proceed if there are observations.  The length could be 0
            # if all environments terminated at the same time
            if len(all_obs) > 0:
                obs_batch = torch.FloatTensor(np.stack(all_obs)).to(device)
                with torch.no_grad():
                    # Get actions, log probabilities, and values for all observations at once.
                    action_batch, log_prob_batch, value_batch = trainer.get_action(obs_batch)

                # Organize actions back into per-environment dictionaries for the step function.
                actions_dict_list = [{} for _ in range(num_envs)]
                for i, (action, log_prob, value) in enumerate(zip(action_batch, log_prob_batch, value_batch)):
                    env_idx = all_env_indices[i]
                    agent_id = all_agent_ids[i]
                    actions_dict_list[env_idx][agent_id] = action

                    # Store experience. Reward and done are placeholders, updated after the env step.
                    trainer.store_experience(
                        all_obs[i],
                        action,
                        log_prob,
                        0,
                        value,
                        False
                    )

                    collected_experiences += 1

                # Step all environments in parallel.
                results, dones, episode_counts = vec_env.step(actions_dict_list)

                # Process results from each environment.
                exp_idx = 0
                for env_idx, (next_obs_dict, reward_dict, terminated_dict, truncated_dict) in enumerate(results):
                    for agent_id in reward_dict.keys():  # Iterate using reward_dict to ensure correct agent IDs
                        # Retrieve reward and done status for the agent.
                        reward = reward_dict[agent_id]
                        done = terminated_dict[agent_id] or truncated_dict[agent_id]

                        # Correctly update the stored experience with the actual reward and done status.
                        mem_idx = trainer.memory.pos - len(all_obs) + exp_idx
                        if mem_idx < 0:  # Wrap around circular buffer
                            mem_idx += trainer.memory.buffer_size

                        trainer.store_experience_at_idx(mem_idx, None, None, None, reward, None, done)

                        # Accumulate rewards for this episode.
                        episode_rewards[env_idx][agent_id] += reward
                        exp_idx += 1

                # Check for completed episodes.
                newly_completed_episodes = sum(dones)
                if newly_completed_episodes > 0:
                    # For episode-based training, update the progress bar
                    if training_time is None:
                        progress_bar.update(newly_completed_episodes)

                    total_episodes_so_far += newly_completed_episodes
                    stats_dict["Episodes"] = total_episodes_so_far

                    # Check if it's time to save the model.
                    if save_interval > 0 and (total_episodes_so_far - last_save_episode) >= save_interval:
                        # Use a "checkpoints" subdirectory under the output path
                        checkpoint_dir = "checkpoints"
                        if output_path:
                            if os.path.isdir(output_path):
                                checkpoint_dir = os.path.join(output_path, "checkpoints")
                            else:
                                # If output path is a file, use its directory
                                output_dir = os.path.dirname(output_path)
                                checkpoint_dir = os.path.join(output_dir if output_dir else ".", "checkpoints")

                        os.makedirs(checkpoint_dir, exist_ok=True)

                        # Save with episode number
                        checkpoint_path = os.path.join(checkpoint_dir, f"model_ep{total_episodes_so_far}.pt")
                        trainer.save_models(checkpoint_path)

                        # Save "latest" model for easy loading
                        latest_path = os.path.join(checkpoint_dir, "model_latest.pt")
                        trainer.save_models(latest_path)

                        if debug:
                            print(f"[DEBUG] Saved checkpoint at episode {total_episodes_so_far} to {checkpoint_path}")

                        last_save_episode = total_episodes_so_far

                    # Reset episode rewards for completed episodes, for tracking the average reward.
                    for env_idx, done in enumerate(dones):
                        if done:
                            avg_reward = sum(episode_rewards[env_idx].values()) / len(episode_rewards[env_idx])
                            if debug:
                                print(f"Episode {episode_counts[env_idx]} in env {env_idx} completed with avg reward: {avg_reward:.2f}")
                            episode_rewards[env_idx] = {agent_id: 0 for agent_id in vec_env.obs_dicts[env_idx]}

                # Determine if it's time to update the policy.
                time_since_update = time.time() - last_update_time
                enough_experiences = collected_experiences >= update_interval

                # We update if enough experiences have been collected OR if enough time has passed
                if enough_experiences or (collected_experiences > 100 and time_since_update > 30):
                    if debug:
                        print(f"[DEBUG] Updating policy with {collected_experiences} experiences after {time_since_update:.2f}s")

                    # Perform the policy update.
                    stats = trainer.update()

                    # Update the stats dictionary for the progress bar.
                    stats_dict.update({
                        "Device": device,
                        "Envs": num_envs,
                        "Exp": f"0/{update_interval}",  # Reset experience count
                        "Episodes": total_episodes_so_far,
                        "Reward": f"{stats.get('mean_episode_reward', 0):.2f}",
                        "PLoss": f"{stats.get('actor_loss', 0):.4f}",
                        "VLoss": f"{stats.get('critic_loss', 0):.4f}",
                        "Entropy": f"{stats.get('entropy_loss', 0):.4f}"
                    })

                    progress_bar.set_postfix(stats_dict)

                    collected_experiences = 0
                    last_update_time = time.time()

                # Update the "Exp" (experiences) count in the progress bar.
                stats_dict["Exp"] = f"{collected_experiences}/{update_interval}"
                progress_bar.set_postfix(stats_dict)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Cleaning up...")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Always clean up environments and progress bar.
        vec_env.close()
        progress_bar.close()

        # Perform a final policy update if any experiences remain.
        if collected_experiences > 0:
            if debug:
                print(f"[DEBUG] Final update with {collected_experiences} experiences")
            try:
                trainer.update()  # Final update
            except Exception as e:
                print(f"Error during final update: {str(e)}")
                import traceback
                traceback.print_exc()

    return trainer

def parse_time(time_str):
    """
    Parse a time string in format like '5m', '2h', '1d' to seconds
    """
    if not time_str:
        return None

    # Extract number and unit
    time_str = time_str.lower().strip()
    if len(time_str) < 2:
        raise ValueError(f"Invalid time format: {time_str}. Use format like '5m', '2h', '1d'")

    try:
        value = float(time_str[:-1])
        unit = time_str[-1]

        # Convert to seconds based on unit
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

class VectorizedEnv:
    """
    Manages multiple independent environments for vectorized execution.
    This allows us to collect experiences from multiple environments in parallel,
    while still performing a single forward pass for inference, significantly
    speeding up the training process.
    """
    def __init__(self, num_envs, render=False):
        self.render = render
        # Create a single shared renderer if rendering is enabled
        self.renderer = RLViserRenderer() if render else None

        # Create environments, passing the renderer to the one that will be rendered
        self.envs = []
        for i in range(num_envs):
            # Only pass renderer to the first environment when rendering is enabled
            env_renderer = self.renderer if (render and i == 0) else None
            self.envs.append(get_env(renderer=env_renderer))

        self.num_envs = num_envs
        self.render_env_idx = 0  # Only render the first environment

        self.obs_dicts = [env.reset() for env in self.envs]  # Reset all environments to get initial observations

        # Explicitly trigger rendering after reset if enabled
        if self.render:
            self.envs[self.render_env_idx].render()

        self.dones = [False] * num_envs  # Track which environments have completed an episode
        self.episode_counts = [0] * num_envs
        self.is_discrete = True  # RLGYM Discrete actions

        # Use a thread pool to parallelize environment steps
        self.max_workers = min(32, num_envs)  # Limit the number of threads to avoid excessive overhead
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

    def _step_env(self, args):
        """
        Handles a single environment step. Designed to be run in parallel.
        """
        i, env, actions_dict = args

        # Format actions for the RLGym API.
        formatted_actions = {}
        for agent_id, action in actions_dict.items():
            if isinstance(action, np.ndarray):
                formatted_actions[agent_id] = action
            elif isinstance(action, int):
                formatted_actions[agent_id] = np.array([action])
            else:
                formatted_actions[agent_id] = np.array([int(action)])

        # Only explicitly render if this is the environment we want to render
        if self.render and i == self.render_env_idx:
            env.render()  # Make sure to call render explicitly
            time.sleep(6/120)  # Slow down rendering to make it more visible

        next_obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(formatted_actions)

        return i, (next_obs_dict, reward_dict, terminated_dict, truncated_dict)

    def step(self, actions_dict_list):
        """
        Steps all environments forward with their corresponding actions.
        """
        futures = []
        for i, (env, actions_dict) in enumerate(zip(self.envs, actions_dict_list)):
            future = self.executor.submit(self._step_env, (i, env, actions_dict))
            futures.append(future)

        # Wait for all steps to complete and collect results in original order
        results = [None] * self.num_envs
        for future in concurrent.futures.as_completed(futures):
            i, result = future.result()
            results[i] = result

            next_obs_dict, reward_dict, terminated_dict, truncated_dict = result

            # If an episode is done, we need to reset the environment.
            self.dones[i] = any(terminated_dict.values()) or any(truncated_dict.values())

            if self.dones[i]:
                self.episode_counts[i] += 1
                self.obs_dicts[i] = self.envs[i].reset()
                # Re-trigger the render after reset for the rendering environment
                if self.render and i == self.render_env_idx:
                    self.envs[i].render()
            else:
                self.obs_dicts[i] = next_obs_dict

        return results, self.dones.copy(), self.episode_counts.copy()

    def close(self):
        """
        Cleanly shuts down the environments and the thread pool.
        """
        self.executor.shutdown()
        for env in self.envs:
            env.close()

        # Explicitly close the renderer if it exists
        if self.renderer:
            self.renderer.close()




def signal_handler(sig, frame):
    print("\nInterrupted by user. Cleaning up...")
    sys.exit(0)

if __name__ == "__main__":

    # Register a signal handler for SIGINT (Ctrl+C) to allow graceful termination.
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='RLBot Training Script')
    parser.add_argument('--render', action='store_true', help='Enable rendering')

    # Create a mutually exclusive group for episode count vs time-based training
    training_duration = parser.add_mutually_exclusive_group()
    training_duration.add_argument('-e', '--episodes', type=int, default=5000, help='Number of episodes to run')
    training_duration.add_argument('-t', '--time', type=str, default=None,
                                  help='Training duration in format: 5m (minutes), 5h (hours), 5d (days)')

    parser.add_argument('-n', '--num_envs', type=int, default=os.cpu_count(),
                        help='Number of parallel environments')
    parser.add_argument('--update_interval', type=int, default=min(1000 * (os.cpu_count() or 4), 6000),
                        help='Experiences before policy update')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu). If not specified, will use best available.')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging')


    # Hyperparameter arguments
    parser.add_argument('--lra', type=float, default=3e-4, help='Learning rate for actor network')
    parser.add_argument('--lrc', type=float, default=3e-3, help='Learning rate for critic network')
    parser.add_argument('--gamma', type=float, default=0.997, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.97, help='GAE lambda parameter')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--critic_coef', type=float, default=0.7, help='Critic loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm for clipping')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='Number of PPO epochs per update')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for PPO updates')

    parser.add_argument('--compile', action='store_true', help='Use torch.compile for model optimization')
    parser.add_argument('--no-compile', action='store_false', dest='compile', help='Disable torch.compile')
    parser.set_defaults(compile=True)

    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='Disable automatic mixed precision')
    parser.set_defaults(amp=False)

    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Path to the model file to load (contains both actor and critic)')

    parser.add_argument('-o', '--out', type=str, default=None,
                    help='Output path where to save the trained model')

    parser.add_argument('--test', action='store_true',
                        help='Enable test mode: enables rendering and limits to 1 environment')


    parser.add_argument('--save_interval', type=int, default=200,
                       help='Save models every N episodes')


    # For backwards compatibility with older versions
    parser.add_argument('-p', '--processes', type=int, default=None,
                        help='Legacy parameter; use --num_envs instead')

    args = parser.parse_args()

    # Parse the time argument if provided
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
        print("Test mode enabled: Rendering ON, using 1 environment")

    # Handle legacy --processes argument
    if args.processes is not None and args.num_envs == 4:  # 4 is the default for num_envs
        args.num_envs = args.processes
        if args.debug:
            print(f"[DEBUG] Using --processes value ({args.processes}) for number of environments")

    # Initialize environment to get observation and action space dimensions.
    env = get_env()
    env.reset()
    obs_space_dims = env.observation_space(env.agents[0])[1]
    action_space_dims = env.action_space(env.agents[0])[1]
    env.close()

    # Initialize actor and critic networks.
    actor = SimBa(obs_shape=obs_space_dims, action_shape=action_space_dims)
    critic = SimBa(obs_shape=obs_space_dims, action_shape=1)

    # Determine the best available device.
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"


    torch.set_printoptions(precision=10)

    if "cuda" in str(device):
        # Optimize for CUDA performance.
        torch.set_float32_matmul_precision('high')
        # Enable TF32 for faster computations on Ampere GPUs and above.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Settings to improve safety with CUDA graphs.
        try:
            torch._C._jit_set_bailout_depth(20)
        except AttributeError:
            if args.debug:
                print("[DEBUG] _jit_set_bailout_depth not available in this PyTorch version")

        # Use current CUDA device.
        torch.cuda.set_device(torch.cuda.current_device())

        # Improve tensor allocation for CUDA graphs.
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Dynamo settings for safer CUDA graphs, if available.
        if hasattr(torch, '_dynamo'):
            try:
                torch._dynamo.config.cache_size_limit = 16  # Reduce cache size
                torch._dynamo.config.suppress_errors = True
            except AttributeError:
                if args.debug:
                    print("[DEBUG] torch._dynamo.config not available in this PyTorch version")

    # Initialize Weights & Biases for experiment tracking, if enabled.
    if args.wandb:
        import wandb
        wandb.init(
            project="rlbot-training",
            config={
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

                # Environment details
                "action_repeat": 8,
                "num_agents": 4,  # 2 per team

                # System configuration
                "episodes": args.episodes,
                "training_time": args.time,
                "num_envs": args.num_envs,
                "update_interval": args.update_interval,
                "device": device,
            },
            name=f"PPO_{time.strftime('%Y%m%d-%H%M%S')}",
            monitor_gym=False,  # Disable default gym monitoring
        )

    if args.debug:
        print(f"[DEBUG] Starting training with {args.num_envs} environments on {device}")
        print(f"[DEBUG] Actor model: {actor}")
        print(f"[DEBUG] Critic model: {critic}")
        if args.time:
            print(f"[DEBUG] Training for {args.time} ({training_time_seconds} seconds)")
        else:
            print(f"[DEBUG] Training for {args.episodes} episodes")


    if args.model:
        try:
            # Load checkpoint to inspect its structure
            checkpoint = torch.load(args.model, map_location=device)

            if args.debug:
                print(f"[DEBUG] Loaded checkpoint from {args.model}")

            # Extract model parameters if it's a combined checkpoint
            if isinstance(checkpoint, dict) and 'actor' in checkpoint and 'critic' in checkpoint:
                # Get dimensions from the actor model
                actor_obs_shape, actor_hidden_dim, actor_action_shape, actor_num_blocks = extract_model_dimensions(checkpoint['actor'])
                # Get dimensions from the critic model
                critic_obs_shape, critic_hidden_dim, critic_action_shape, critic_num_blocks = extract_model_dimensions(checkpoint['critic'])

                if args.debug:
                    print("[DEBUG] Extracted model dimensions from checkpoint:")
                    print(f"[DEBUG] Actor: obs_shape={actor_obs_shape}, hidden_dim={actor_hidden_dim}, action_shape={actor_action_shape}, num_blocks={actor_num_blocks}")
                    print(f"[DEBUG] Critic: obs_shape={critic_obs_shape}, hidden_dim={critic_hidden_dim}, action_shape={critic_action_shape}, num_blocks={critic_num_blocks}")

                # Recreate models with the correct dimensions
                actor = SimBa(
                    obs_shape=actor_obs_shape,
                    action_shape=actor_action_shape,
                    hidden_dim=int(actor_hidden_dim) if actor_hidden_dim is not None else 1024,
                    num_blocks=actor_num_blocks if actor_num_blocks is not None else 4
                )

                critic = SimBa(
                    obs_shape=critic_obs_shape,
                    action_shape=critic_action_shape,
                    hidden_dim=int(critic_hidden_dim) if critic_hidden_dim is not None else 1024,
                    num_blocks=critic_num_blocks if critic_num_blocks is not None else 4
                )

                # Load a pre-trained model with the correct dimensions now
                temp_trainer = PPOTrainer(
                    actor=actor,
                    critic=critic,
                    device=device,
                    action_dim=int(actor_action_shape) if actor_action_shape is not None else action_space_dims,
                    use_compile=False  # Don't compile before loading
                )
                temp_trainer.load_models(args.model)
                print(f"Successfully loaded model from {args.model}")

                # After loading, compile if needed
                if args.compile:
                    if hasattr(torch, 'compile'):
                        try:
                            actor = torch.compile(actor)
                            critic = torch.compile(critic)
                            print("Models compiled after loading")
                        except Exception as e:
                            print(f"Warning: Failed to compile models after loading: {e}")
            else:
                print(f"Error: Unsupported model format in {args.model}")
        except Exception as e:
            print(f"Error loading model: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

    # Start the training process.
    trainer = run_training(
        actor=actor,
        critic=critic,
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
        # Hyperparameters
        lr_actor=args.lra,
        lr_critic=args.lrc,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        critic_coef=args.critic_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size
    )



    # Save final models
    if not args.test and trainer is not None:
        output_path = args.out if args.out else None
        saved_path = trainer.save_models(output_path)
        print(f"Training complete - Model saved to {saved_path}")
    else:
        print("Training failed or in test mode - no model saved.")
