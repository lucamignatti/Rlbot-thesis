import os

# Configure environment variables and settings for MPS backend compatibility
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
from rewards import ProximityReward
from models import BasicModel, SimBa
from training import PPOTrainer
import time
import argparse
from tqdm import tqdm
import signal
import sys

def get_env():
    """Creates a standardized RL environment for Rocket League."""
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            KickoffMutator()
        ),
        obs_builder=DefaultObs(zero_padding=2),
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=CombinedReward(
            (GoalReward(), 12.),
            (TouchReward(), 3.),
            (ProximityReward(), 1.),
        ),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            TimeoutCondition(300.),
            NoTouchTimeoutCondition(30.)
        ),
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer()
    )

class VectorizedEnv:
    """
    Manages multiple independent environments for vectorized execution.
    This allows us to collect experiences from multiple environments in parallel,
    while still performing a single forward pass for inference.
    """
    def __init__(self, num_envs, render=False):
        # Create multiple independent environments
        self.envs = [get_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.render = render

        # Only render the first environment if rendering is enabled
        self.render_env_idx = 0

        # Reset all environments to get initial observations
        self.obs_dicts = [env.reset() for env in self.envs]

        # Track which environments need resetting
        self.dones = [False] * num_envs

        # Track episodes per environment
        self.episode_counts = [0] * num_envs

        # Get action space information for proper formatting
        self.is_discrete = True  # LookupTableAction uses discrete actions

    def step(self, actions_dict_list):
        """Step all environments with their corresponding actions."""
        results = []

        for i, (env, actions_dict) in enumerate(zip(self.envs, actions_dict_list)):
            # Make sure actions are in the correct format for LookupTableAction
            # LookupTableAction expects integers for discrete actions
            formatted_actions = {}
            for agent_id, action in actions_dict.items():
                # For LookupTableAction, we need a single integer value
                if isinstance(action, np.ndarray):
                    formatted_actions[agent_id] = action
                elif isinstance(action, int):
                    formatted_actions[agent_id] = np.array([action])
                else:
                    formatted_actions[agent_id] = np.array([int(action)])

            # Render only the first environment when rendering is enabled
            if self.render and i == self.render_env_idx:
                env.render()
                time.sleep(6/120)  # ~120 FPS limit

            # Step the environment
            next_obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(formatted_actions)

            # Store the results
            results.append((next_obs_dict, reward_dict, terminated_dict, truncated_dict))

            # Check if episode is done for this environment
            self.dones[i] = any(terminated_dict.values()) or any(truncated_dict.values())

            # Update episode count and reset if needed
            if self.dones[i]:
                self.episode_counts[i] += 1
                self.obs_dicts[i] = env.reset()
            else:
                self.obs_dicts[i] = next_obs_dict

        return results, self.dones.copy(), self.episode_counts.copy()

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

def run_training(
    actor,
    critic,
    device,
    num_envs: int,
    total_episodes: int,
    render: bool = False,
    update_interval: int = 1000,
    use_wandb: bool = False,
    debug: bool = False,
    use_compile: bool = True,
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
    Single-process vectorized training loop that maximizes GPU efficiency.
    Uses multiple environments and batched inference for optimal performance.
    """
    # Move models to the target device (GPU) for faster inference and training
    actor.to(device)
    critic.to(device)

    # Create trainer instance
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
        use_compile=use_compile
    )

    # Create vectorized environment
    vec_env = VectorizedEnv(num_envs=num_envs, render=render)

    # Set up progress tracking
    progress_bar = tqdm(
        total=total_episodes,
        desc="Episodes",
        bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%|{bar}| {postfix}',
        dynamic_ncols=True
    )

    # Initialize stats dictionary for progress bar
    stats_dict = {
        "Device": device,
        "Envs": num_envs,
        "Exp": 0,
        "Reward": 0.0,
        "PLoss": 0.0,
        "VLoss": 0.0,
        "Entropy": 0.0
    }
    progress_bar.set_postfix(stats_dict)

    # Tracking variables
    collected_experiences = 0
    total_episodes_so_far = 0
    last_update_time = time.time()
    episode_rewards = {i: {agent_id: 0 for agent_id in vec_env.obs_dicts[i]} for i in range(num_envs)}

    try:
        # Main training loop
        while total_episodes_so_far < total_episodes:
            # Group observations from all environments for batched inference
            # This maximizes GPU throughput by processing all observations at once
            all_obs = []
            all_env_indices = []
            all_agent_ids = []

            # Organize observations for batched processing
            for env_idx, obs_dict in enumerate(vec_env.obs_dicts):
                for agent_id, obs in obs_dict.items():
                    all_obs.append(obs)
                    all_env_indices.append(env_idx)
                    all_agent_ids.append(agent_id)

            # Run batched inference for all observations
            # This is a major performance optimization as we run a single forward pass
            # through the network for all observations across all environments
            if len(all_obs) > 0:  # Make sure we have observations
                obs_batch = torch.FloatTensor(np.stack(all_obs)).to(device)
                with torch.no_grad():
                    # Single batched forward pass for both actor and critic
                    action_batch, log_prob_batch, value_batch = trainer.get_action(obs_batch)

                # Organize actions back into per-environment dictionaries
                actions_dict_list = [{} for _ in range(num_envs)]
                for i, (action, log_prob, value) in enumerate(zip(action_batch, log_prob_batch, value_batch)):
                    env_idx = all_env_indices[i]
                    agent_id = all_agent_ids[i]
                    actions_dict_list[env_idx][agent_id] = action

                    # Store experience in trainer memory
                    trainer.store_experience(
                        all_obs[i],
                        action,
                        log_prob,
                        0,  # Placeholder for reward, will be updated after environment step
                        value,
                        False  # Placeholder for done, will be updated after environment step
                    )

                    collected_experiences += 1

                # Step all environments with their corresponding actions
                results, dones, episode_counts = vec_env.step(actions_dict_list)

                # Process results and update experiences with rewards and dones
                exp_idx = 0
                for env_idx, (next_obs_dict, reward_dict, terminated_dict, truncated_dict) in enumerate(results):
                    for agent_id in reward_dict.keys():  # Use reward_dict to get valid agent_ids
                        # Update rewards in trainer's memory
                        reward = reward_dict[agent_id]
                        done = terminated_dict[agent_id] or truncated_dict[agent_id]

                        # Update the last stored experience with actual reward and done status
                        # Need to handle cases where the order might be different
                        mem_idx = trainer.memory.pos - len(all_obs) + exp_idx
                        if mem_idx < 0:  # Handle wraparound in circular buffer
                            mem_idx += trainer.memory.buffer_size

                        trainer.memory.rewards[mem_idx] = reward
                        trainer.memory.dones[mem_idx] = done

                        # Track episode rewards
                        episode_rewards[env_idx][agent_id] += reward
                        exp_idx += 1

                # Count completed episodes
                newly_completed_episodes = sum(dones)
                if newly_completed_episodes > 0:
                    # Update progress bar for completed episodes
                    progress_bar.update(newly_completed_episodes)
                    total_episodes_so_far += newly_completed_episodes

                    # Reset episode rewards for completed episodes
                    for env_idx, done in enumerate(dones):
                        if done:
                            avg_reward = sum(episode_rewards[env_idx].values()) / len(episode_rewards[env_idx])
                            if debug:
                                print(f"Episode {episode_counts[env_idx]} in env {env_idx} completed with avg reward: {avg_reward:.2f}")
                            episode_rewards[env_idx] = {agent_id: 0 for agent_id in vec_env.obs_dicts[env_idx]}

                # Update policy when enough experiences are collected or enough time has passed
                time_since_update = time.time() - last_update_time
                enough_experiences = collected_experiences >= update_interval

                if enough_experiences or (collected_experiences > 100 and time_since_update > 30):
                    if debug:
                        print(f"[DEBUG] Updating policy with {collected_experiences} experiences after {time_since_update:.2f}s")

                    # Update policy using collected experiences
                    stats = trainer.update()

                    # Update stats dictionary
                    stats_dict.update({
                        "Device": device,
                        "Envs": num_envs,
                        "Exp": f"0/{update_interval}",  # Reset after update
                        "Reward": f"{stats.get('mean_episode_reward', 0):.2f}",
                        "PLoss": f"{stats.get('actor_loss', 0):.4f}",
                        "VLoss": f"{stats.get('critic_loss', 0):.4f}",
                        "Entropy": f"{stats.get('entropy_loss', 0):.4f}"
                    })

                    # Update progress bar with all stats
                    progress_bar.set_postfix(stats_dict)

                    # Reset counters
                    collected_experiences = 0
                    last_update_time = time.time()

                # Update experience count in progress bar
                stats_dict["Exp"] = f"{collected_experiences}/{update_interval}"
                progress_bar.set_postfix(stats_dict)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Cleaning up...")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up resources
        vec_env.close()
        progress_bar.close()

        # Final update with any remaining experiences
        if collected_experiences > 0:
            if debug:
                print(f"[DEBUG] Final update with {collected_experiences} experiences")
            try:
                trainer.update()
            except Exception as e:
                print(f"Error during final update: {str(e)}")
                import traceback
                traceback.print_exc()

    return trainer

def signal_handler(sig, frame):
    print("\nInterrupted by user. Cleaning up...")
    sys.exit(0)

if __name__ == "__main__":


    # Register signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='RLBot Training Script')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('-e', '--episodes', type=int, default=200, help='Number of episodes to run')
    parser.add_argument('-n', '--num_envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--update_interval', type=int, default=1000,
                        help='Experiences before policy update')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu). If not specified, will use best available.')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging')

    parser.add_argument('--lra', type=float, default=3e-4, help='Learning rate for actor network')
    parser.add_argument('--lrc', type=float, default=1e-3, help='Learning rate for critic network')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--critic_coef', type=float, default=0.5, help='Critic loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm for clipping')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='Number of PPO epochs per update')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for PPO updates')

    parser.add_argument('--compile', action='store_true', help='Use torch.compile for model optimization')
    parser.add_argument('--no-compile', action='store_false', dest='compile', help='Disable torch.compile')
    parser.set_defaults(compile=True)


    # For backwards compatibility
    parser.add_argument('-p', '--processes', type=int, default=None,
                        help='Legacy parameter; use --num_envs instead')

    args = parser.parse_args()

    # If processes is specified but num_envs isn't, use processes value
    if args.processes is not None and args.num_envs == 4:  # 4 is the default for num_envs
        args.num_envs = args.processes
        if args.debug:
            print(f"[DEBUG] Using --processes value ({args.processes}) for number of environments")

    # Initialize environment to get dimensions
    env = get_env()
    env.reset()
    obs_space_dims = env.observation_space(env.agents[0])[1]
    action_space_dims = env.action_space(env.agents[0])[1]
    env.close()

    # Initialize models
    actor = SimBa(obs_shape=obs_space_dims, action_shape=action_space_dims)
    critic = SimBa(obs_shape=obs_space_dims, action_shape=1)

    # Determine device
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Initialize wandb if requested
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

                # Environment
                "action_repeat": 8,
                "num_agents": 4,  # 2 per team

                # System
                "episodes": args.episodes,
                "num_envs": args.num_envs,
                "update_interval": args.update_interval,
                "device": device,
            },
            name=f"PPO_{time.strftime('%Y%m%d-%H%M%S')}",
            monitor_gym=False,
        )

    if args.debug:
        print(f"[DEBUG] Starting training with {args.num_envs} environments on {device}")
        print(f"[DEBUG] Actor model: {actor}")
        print(f"[DEBUG] Critic model: {critic}")

    # Start training with single-process vectorized approach
    trainer = run_training(
        actor=actor,
        critic=critic,
        device=device,
        num_envs=args.num_envs,
        total_episodes=args.episodes,
        render=args.render,
        update_interval=args.update_interval,
        use_wandb=args.wandb,
        debug=args.debug,
        use_compile=args.compile,
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

    # Create models directory and save models
    os.makedirs("models", exist_ok=True)
    if trainer is not None:
        trainer.save_models("models/actor.pth", "models/critic.pth")
        print("Training complete - Models saved")
    else:
        print("Training failed.")
