import numpy as np
import torch
import torch.multiprocessing as mp
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
import os
import signal
import sys
from queue import Empty

def get_env():
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

def worker(
    rank,
    shared_actor,
    shared_critic,
    experience_queue,
    episode_counter,
    update_event,
    terminate_event,
    total_episodes,
    render,
    debug=False
):
    """Worker process that runs environment with local inference."""
    try:
        if debug:
            print(f"[DEBUG] Worker {rank} starting")

        # Create local copies of models for inference
        local_actor = SimBa(
            obs_shape=shared_actor.obs_shape,
            action_shape=shared_actor.action_shape,
        )
        local_critic = SimBa(
            obs_shape=shared_critic.obs_shape,
            action_shape=shared_critic.action_shape,
        )

        # Initialize with shared model weights
        local_actor.load_state_dict(shared_actor.state_dict())
        local_critic.load_state_dict(shared_critic.state_dict())

        # Create local trainer for action selection only
        local_trainer = PPOTrainer(
            local_actor,
            local_critic,
            action_dim=local_actor.action_shape,
            device="cpu",
            debug=False
        )

        # Set up environment
        env = get_env()
        obs_dict = env.reset()

        # Process variables
        episode_steps = 0
        sync_interval = 10  # Sync models every N steps

        if debug:
            print(f"[DEBUG] Worker {rank} initialized")

        # Main worker loop
        while not terminate_event.is_set():
            # Check if we've reached total episodes
            with episode_counter.get_lock():
                if episode_counter.value >= total_episodes:
                    break

            # Sync with shared models periodically or when update is ready
            if update_event.is_set() or episode_steps % sync_interval == 0:
                local_actor.load_state_dict(shared_actor.state_dict())
                local_critic.load_state_dict(shared_critic.state_dict())

            # Process each agent's observation and compute actions locally
            actions = {}
            values = {}
            log_probs = {}

            for agent_id, obs in obs_dict.items():
                # Get action using local model
                action, log_prob, value = local_trainer.get_action(obs)
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
                values[agent_id] = value

            # Render if enabled and this is the designated rendering worker
            if render and rank == 0:
                env.render()
                time.sleep(6/120)

            # Step the environment with the actions
            next_obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)

            # Send experiences to main process
            for agent_id in obs_dict.keys():
                obs = obs_dict[agent_id]
                action = actions[agent_id]
                log_prob = log_probs[agent_id]
                value = values[agent_id]
                reward = reward_dict[agent_id]

                # Check if episode ended for this agent
                terminated = terminated_dict[agent_id]
                truncated = truncated_dict[agent_id]
                done = terminated or truncated

                # Send experience tuple to queue
                experience = (obs, action, log_prob, reward, value, done)
                experience_queue.put(experience)

            # Update for next iteration
            obs_dict = next_obs_dict
            episode_steps += 1

            # Check if episode is done
            terminated = any(terminated_dict.values())
            truncated = any(truncated_dict.values())

            if terminated or truncated:
                if debug:
                    print(f"[DEBUG] Worker {rank} finished episode after {episode_steps} steps")

                # Reset environment
                obs_dict = env.reset()
                episode_steps = 0

                # Update episode counter
                with episode_counter.get_lock():
                    episode_counter.value += 1
                    current_episodes = episode_counter.value

                if debug:
                    print(f"[DEBUG] Worker {rank} completed episode. Total: {current_episodes}/{total_episodes}")

    except Exception as e:
        print(f"Error in worker process {rank}: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()

    finally:
        # Clean up environment
        if 'env' in locals():
            env.close()
        if debug:
            print(f"[DEBUG] Worker {rank} shut down")

def run_parallel_training(
    actor,
    critic,
    device,
    num_processes: int,
    total_episodes: int,
    render: bool = False,
    update_interval: int = 1000,
    use_wandb: bool = False,
    debug: bool = False,
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

    # Initialize variables that might be accessed in finally block
    collected_experiences = 0
    processes = []
    trainer = None  # Initialize trainer to None for safety

    try:
        # IMPORTANT: Move models to CPU first for sharing
        actor_cpu = actor.cpu()
        critic_cpu = critic.cpu()

        # Make CPU models shareable across processes
        shared_actor = actor_cpu.share_memory()
        shared_critic = critic_cpu.share_memory()

        # Create experience queue for collecting transitions
        experience_queue = mp.Queue(maxsize=update_interval*2)  # Allow buffer room

        # Create synchronization primitives
        update_event = mp.Event()     # Signal that model has been updated
        terminate_event = mp.Event()  # Signal workers to terminate
        episode_counter = mp.Value('i', 0)  # Shared episode counter

        # Setup improved progress tracking with more details
        progress_bar = tqdm(
            total=total_episodes, 
            desc="Episodes", 
            bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%|{bar}| {postfix}',
            dynamic_ncols=True  # Better adapts to terminal resizing
        )
        
        # Initialize stats dict to avoid KeyErrors when updating
        stats_dict = {
            "Device": device, 
            "Workers": num_processes,
            "Exp": 0,
            "Reward": 0.0,
            "PLoss": 0.0,
            "VLoss": 0.0,
            "Entropy": 0.0
        }
        progress_bar.set_postfix(stats_dict)

        # Start worker processes
        processes = []
        for rank in range(num_processes):
            p = mp.Process(
                target=worker,
                args=(
                    rank,
                    shared_actor,
                    shared_critic,
                    experience_queue,
                    episode_counter,
                    update_event,
                    terminate_event,
                    total_episodes,
                    render,
                    debug
                )
            )
            p.daemon = True
            p.start()
            processes.append(p)

        if debug:
            print(f"[DEBUG] Started {num_processes} worker processes")

        # Create separate models for the trainer on the target device
        # Don't share the CUDA models with worker processes
        if device != "cpu":
            trainer_actor = SimBa(
                obs_shape=actor.obs_shape,
                action_shape=actor.action_shape,
            ).to(device)

            trainer_critic = SimBa(
                obs_shape=critic.obs_shape,
                action_shape=critic.action_shape,
            ).to(device)

            # Copy the weights from CPU models
            trainer_actor.load_state_dict(actor_cpu.state_dict())
            trainer_critic.load_state_dict(critic_cpu.state_dict())
        else:
            # For CPU, we can use the shared models directly
            trainer_actor = shared_actor
            trainer_critic = shared_critic

        # Create trainer with the device-specific models
        trainer = PPOTrainer(
            trainer_actor,
            trainer_critic,
            action_dim=trainer_actor.action_shape,
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
            debug=debug
        )

        # Main training loop variables
        collected_experiences = 0
        last_episode_count = 0
        last_update_time = time.time()

        # Main loop: collect experiences and update policy
        while episode_counter.value < total_episodes:
            # Update progress bar based on completed episodes
            current_episodes = episode_counter.value
            if current_episodes > last_episode_count:
                progress_increment = current_episodes - last_episode_count
                progress_bar.update(progress_increment)
                last_episode_count = current_episodes

            # Check if processes are still alive
            alive_count = sum(1 for p in processes if p.is_alive())
            if alive_count < num_processes:
                print(f"WARNING: Only {alive_count}/{num_processes} processes alive")

            # Collect experiences from queue (non-blocking)
            try:
                # Try to get experiences without using qsize()
                experience_batch_start = time.time()
                experiences_collected_this_loop = 0

                # Set a limit for how many experiences to process in one loop iteration
                max_experiences_per_loop = 100

                # Keep collecting until either queue is empty or we've hit the update threshold
                while experiences_collected_this_loop < max_experiences_per_loop and collected_experiences < update_interval:
                    try:
                        # Use get_nowait() to avoid blocking
                        experience = experience_queue.get_nowait()
                        trainer.store_experience(*experience)
                        collected_experiences += 1
                        experiences_collected_this_loop += 1
                    except Empty:
                        # Queue is empty, break the inner loop
                        break

                if experiences_collected_this_loop > 0 and debug:
                    print(f"[DEBUG] Processed {experiences_collected_this_loop} experiences in {time.time() - experience_batch_start:.3f}s")

                # Update progress bar with current experience count
                stats_dict["Exp"] = f"{collected_experiences}/{update_interval}"
                progress_bar.set_postfix(stats_dict)

            except Empty:
                # No experiences available right now, short sleep
                time.sleep(0.001)

            except Exception as e:
                print(f"Error collecting experiences: {str(e)}")
                import traceback
                traceback.print_exc()

            # Update policy when enough experiences are collected or enough time has passed
            time_since_update = time.time() - last_update_time
            enough_experiences = collected_experiences >= update_interval

            if enough_experiences or (collected_experiences > 100 and time_since_update > 30):
                if debug:
                    print(f"[DEBUG] Updating policy with {collected_experiences} experiences")

                if collected_experiences > 0:
                    stats = trainer.update()
                    
                    # Update stats dictionary with all possible values
                    stats_dict.update({
                        "Device": device,
                        "Workers": num_processes,
                        "Exp": f"0/{update_interval}",  # Reset after update
                        "Reward": f"{stats.get('mean_episode_reward', 0):.2f}",
                        "PLoss": f"{stats.get('actor_loss', 0):.4f}",
                        "VLoss": f"{stats.get('critic_loss', 0):.4f}",
                        "Entropy": f"{stats.get('entropy_loss', 0):.4f}"
                    })
                    
                    # Update progress bar with all stats
                    progress_bar.set_postfix(stats_dict)

                    # After policy update, sync the weights back to the shared CPU models
                    if device != "cpu":
                        # Copy weights from trainer models back to the shared CPU models
                        with torch.no_grad():
                            for param_cpu, param_device in zip(shared_actor.parameters(), trainer_actor.parameters()):
                                param_cpu.copy_(param_device.cpu())
                            for param_cpu, param_device in zip(shared_critic.parameters(), trainer_critic.parameters()):
                                param_cpu.copy_(param_device.cpu())

                        # Signal workers to update their models
                        update_event.set()
                        # Clear the event after a short delay to ensure workers have time to see it
                        time.sleep(0.01)
                        update_event.clear()

                    collected_experiences = 0

                last_update_time = time.time()

    except KeyboardInterrupt:
        print("\nTraining interrupted. Cleaning up...")

    finally:
        # Signal all processes to terminate
        terminate_event.set()

        # Wait for processes to finish
        for i, p in enumerate(processes):
            if debug:
                print(f"[DEBUG] Waiting for worker {i} to terminate...")
            p.join(timeout=3.0)
            if p.is_alive():
                if debug:
                    print(f"[DEBUG] Forcing termination of worker {i}")
                p.terminate()

        progress_bar.close()

        # Final update with any remaining experiences
        if collected_experiences > 0 and trainer is not None:
            if debug:
                print(f"[DEBUG] Final update with {collected_experiences} experiences")
            trainer.update()

        return trainer

def signal_handler(sig, frame):
    print("\nInterrupted by user. Cleaning up...")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)

    # Enable PyTorch multiprocessing support with spawn method
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='RLBot Training Script')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('-e', '--episodes', type=int, default=200, help='Number of episodes to run')
    parser.add_argument('-p', '--processes', type=int, default=os.cpu_count(),
                        help='Number of parallel processes')
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


    args = parser.parse_args()

    # Ensure at least one process
    args.processes = max(1, args.processes)

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

    # Move models to the device
    actor.to(device)
    critic.to(device)

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
                "processes": args.processes,
                "update_interval": args.update_interval,
                "device": device,
            },
            name=f"PPO_{time.strftime('%Y%m%d-%H%M%S')}",
            monitor_gym=False,
        )

    # Start training with descriptive progress bar
    trainer = run_parallel_training(
        actor=actor,
        critic=critic,
        device=device,
        num_processes=args.processes,
        total_episodes=args.episodes,
        render=args.render,
        update_interval=args.update_interval,
        use_wandb=args.wandb,
        debug=args.debug,
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
    trainer.save_models("models/actor.pth", "models/critic.pth")
    print("Training complete - Models saved")
