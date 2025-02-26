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
from models import BasicModel
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

def worker(rank,
           obs_batch_queue,
           action_batch_queue,
           experience_queue,
           progress_queue,
           num_episodes,
           render,
           debug=False):
    """Worker function that runs an environment and waits for actions from main process."""
    try:
        if debug:
            print(f"[DEBUG] Worker {rank} starting")

        # Only render in first process if enabled
        should_render = render and rank == 0

        # Set up environment
        env = get_env()
        obs_dict = env.reset()

        # Process variables
        episode_count = 0
        episode_steps = 0

        if debug:
            print(f"[DEBUG] Worker {rank} initialized. Running {num_episodes} episodes.")

        # Main worker loop
        while episode_count < num_episodes:
            # Send current observations to main process for batch processing
            batch_item = {
                'rank': rank,
                'obs_dict': obs_dict
            }
            if debug:
                print(f"[DEBUG] Worker {rank} sending observations")
            obs_batch_queue.put(batch_item)

            # Wait for the main process to compute actions for the observations
            if debug:
                print(f"[DEBUG] Worker {rank} waiting for actions")
            action_batch = action_batch_queue.get()
            if debug:
                print(f"[DEBUG] Worker {rank} received actions")

            # Check if this action batch is for our process
            if action_batch['rank'] != rank:
                print(f"Worker {rank} received wrong actions (for rank {action_batch['rank']})")
                continue

            # Get actions for the current observations
            actions = action_batch['actions']
            values = action_batch['values']
            log_probs = action_batch['log_probs']

            # Apply actions to the environment
            if should_render:
                env.render()
                time.sleep(6/120)

            # Step the environment with the received actions
            next_obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)

            # Send experience to main process
            experience = {
                'rank': rank,
                'obs_dict': obs_dict,
                'actions': actions,
                'log_probs': log_probs,
                'values': values,
                'rewards': reward_dict,
                'next_obs_dict': next_obs_dict,
                'terminated_dict': terminated_dict,
                'truncated_dict': truncated_dict,
                'episode_steps': episode_steps
            }
            if debug:
                print(f"[DEBUG] Worker {rank} sending experience")
            experience_queue.put(experience)

            # Update for next iteration
            obs_dict = next_obs_dict
            episode_steps += 1

            # Check if episode is done
            terminated = any(terminated_dict.values())
            truncated = any(truncated_dict.values())

            if terminated or truncated:
                if debug:
                    print(f"[DEBUG] Worker {rank} finished episode {episode_count+1} after {episode_steps} steps")

                # Reset environment
                obs_dict = env.reset()
                episode_count += 1
                episode_steps = 0

                # Signal episode completion to the progress tracker
                progress_queue.put(1)

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

def process_experiences(trainer, experience_batch, debug=False):
    """Process a batch of experiences and store in trainer's memory."""
    if debug:
        print(f"[DEBUG] Processing experience batch from rank {experience_batch['rank']}")

    # Process each agent's experience in the batch
    for agent_id in experience_batch['obs_dict'].keys():
        # Get the observations, actions, etc. for this agent
        obs = experience_batch['obs_dict'][agent_id]
        action = experience_batch['actions'][agent_id]
        log_prob = experience_batch['log_probs'][agent_id]
        value = experience_batch['values'][agent_id]
        reward = experience_batch['rewards'][agent_id]

        # Check if episode ended for this agent
        terminated = experience_batch['terminated_dict'][agent_id]
        truncated = experience_batch['truncated_dict'][agent_id]
        done = terminated or truncated

        # Store in trainer's memory
        trainer.store_experience(obs, action, log_prob, reward, value, done)

def run_parallel_training(
    actor,
    critic,
    device,
    num_processes: int,
    total_episodes: int,
    render: bool = False,
    update_interval: int = 1000,  # Number of experiences before policy update
    use_wandb: bool = False,
    debug: bool = False
):
    # Initialize trainer in the main process
    trainer = PPOTrainer(actor, critic, device=device, use_wandb=use_wandb, debug=debug)

    # Setup progress tracking
    episodes_per_process = max(1, total_episodes // num_processes)
    progress_bar = tqdm(total=total_episodes)

    # Create shared queues for communication
    obs_batch_queue = mp.Queue()
    action_batch_queue = mp.Queue()
    experience_queue = mp.Queue()
    progress_queue = mp.Queue()

    # Start worker processes
    processes = []
    try:
        for rank in range(num_processes):
            # Handle uneven distribution of episodes
            process_episodes = episodes_per_process
            if rank < total_episodes % num_processes:
                process_episodes += 1

            p = mp.Process(
                target=worker,
                args=(
                    rank,
                    obs_batch_queue,
                    action_batch_queue,
                    experience_queue,
                    progress_queue,
                    process_episodes,
                    render,
                    debug
                )
            )
            p.daemon = True  # Make sure processes exit when main exits
            p.start()
            processes.append(p)

        if debug:
            print(f"[DEBUG] Started {num_processes} worker processes")

        # Main process: collect experiences and update policy
        collected_experiences = 0
        completed_episodes = 0

        # Set a timeout for checking process aliveness
        check_alive_interval = 5.0
        last_alive_check = time.time()
        worker_activity = [time.time()] * num_processes

        while completed_episodes < total_episodes and any(p.is_alive() for p in processes):
            current_time = time.time()

            # Periodically check if processes are still alive and making progress
            if current_time - last_alive_check > check_alive_interval:
                alive_count = sum(1 for p in processes if p.is_alive())
                if debug:
                    print(f"[DEBUG] {alive_count}/{num_processes} processes alive")
                last_alive_check = current_time

                # Check for processes that haven't made progress
                for i, last_active in enumerate(worker_activity):
                    if current_time - last_active > 60:  # 1 minute timeout
                        print(f"WARNING: Worker {i} may be stuck (no activity for {current_time - last_active:.1f}s)")

            # Update progress bar
            try:
                while True:
                    try:
                        _ = progress_queue.get_nowait()
                        completed_episodes += 1
                        progress_bar.update(1)

                        # Log to wandb if enabled
                        if use_wandb:
                            wandb.log({
                                "episodes_completed": completed_episodes,
                                "completion_percentage": completed_episodes / total_episodes * 100
                            })
                    except Empty:
                        break
            except Exception as e:
                print(f"Error updating progress: {e}")

            # Step 1: Get observations from workers (non-blocking)
            try:
                obs_batch = obs_batch_queue.get(timeout=0.1)
                rank = obs_batch['rank']
                worker_activity[rank] = time.time()  # Update worker activity timestamp

                if debug:
                    print(f"[DEBUG] Got observation batch from worker {rank}")

                # Step 2: Compute actions for the observations
                obs_dict = obs_batch['obs_dict']

                # Process each agent's observation and compute actions
                actions = {}
                values = {}
                log_probs = {}

                for agent_id, obs in obs_dict.items():
                    # Get action from the model
                    action, log_prob, value = trainer.get_action(obs)
                    actions[agent_id] = action
                    log_probs[agent_id] = log_prob
                    values[agent_id] = value

                # Send actions back to the worker
                action_batch = {
                    'rank': rank,
                    'actions': actions,
                    'values': values,
                    'log_probs': log_probs
                }
                action_batch_queue.put(action_batch)
                if debug:
                    print(f"[DEBUG] Sent actions to worker {rank}")

            except Empty:
                # No observations to process, continue to check for experiences
                pass
            except Exception as e:
                print(f"Error processing observations: {e}")
                import traceback
                traceback.print_exc()

            # Step 3: Process experiences from workers (non-blocking)
            try:
                while True:
                    try:
                        experience_batch = experience_queue.get_nowait()
                        rank = experience_batch['rank']
                        worker_activity[rank] = time.time()  # Update worker activity timestamp

                        process_experiences(trainer, experience_batch, debug)
                        collected_experiences += len(experience_batch['obs_dict'])

                        if debug:
                            print(f"[DEBUG] Processed experiences from worker {rank}, total: {collected_experiences}")
                    except Empty:
                        break
            except Exception as e:
                print(f"Error processing experiences: {e}")
                import traceback
                traceback.print_exc()

            # Update progress bar with collection info
            if collected_experiences > 0:
                progress_bar.set_description(f"Collecting: {collected_experiences}/{update_interval}")

            # Step 4: Update policy when enough experiences are collected
            if collected_experiences >= update_interval:
                if debug:
                    print(f"[DEBUG] Updating policy after {collected_experiences} experiences")
                progress_bar.set_description("Updating policy...")
                stats = trainer.update()
                collected_experiences = 0
                progress_bar.set_description(f"Loss: {stats['total_loss']:.4f}")

            # Short sleep to prevent CPU hogging
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Cleaning up...")

    finally:
        # Clean up processes
        for p in processes:
            if p.is_alive():
                print(f"Terminating process {processes.index(p)}")
                p.terminate()
                p.join(timeout=1.0)

        progress_bar.close()

        # Final update with any remaining experiences
        if collected_experiences > 0:
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
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes to run')
    parser.add_argument('--processes', type=int, default=min(os.cpu_count()-1, 4),
                        help='Number of parallel processes')
    parser.add_argument('--update_interval', type=int, default=1000,
                        help='Experiences before policy update')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu). If not specified, will use best available.')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging')

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
    actor = BasicModel(input_size=obs_space_dims, output_size=action_space_dims, hidden_size=obs_space_dims//2)
    critic = BasicModel(input_size=obs_space_dims, output_size=1, hidden_size=obs_space_dims//2)

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
        wandb.init(project="rlbot-training", config={
            "episodes": args.episodes,
            "processes": args.processes,
            "update_interval": args.update_interval,
            "device": device,
            "obs_space_dims": obs_space_dims,
            "action_space_dims": action_space_dims
        })

    # Initialize progress tracking description
    progress_desc = f"Training ({device}, {args.processes} proc)"

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
        debug=args.debug
    )

    # Create models directory and save models
    os.makedirs("models", exist_ok=True)
    trainer.save_models("models/actor.pth", "models/critic.pth")
    print("Training complete - Models saved")
