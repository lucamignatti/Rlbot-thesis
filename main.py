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

def worker(rank, actor, critic, device, render, num_episodes, progress_queue, experience_queue, debug=False):
    """Worker function for collecting experiences."""
    try:
        # Only render in first process if enabled
        should_render = render and rank == 0
        
        # Set up environment
        env = get_env()
        env.reset()

        # Create a local copy of the trainer (models are already shared in memory)
        trainer = PPOTrainer(actor, critic, device=device, debug=debug)
        if debug:
            print(f"[DEBUG] Worker {rank} starting with {num_episodes} episodes to run")
        trainer.experience_queue = experience_queue
        # Execute episodes
        for episode in range(num_episodes):
            if debug:
                print(f"[DEBUG] Worker {rank} starting episode {episode+1}/{num_episodes}")
            obs_dict = env.reset()
            terminated = False
            truncated = False
            episode_steps = 0
            
            while not (terminated or truncated):
                if should_render:
                    env.render()
                    time.sleep(6/120)
                
                current_actions = {}
                episode_steps += 1
                current_actions = {}
                for agent_id, obs in obs_dict.items():
                    # Get action from the model (will use the shared parameters)
                    action, log_prob, value = trainer.get_action(obs)
                    current_actions[agent_id] = action
                    
                    # Store experience in shared queue
                    experience_queue.put((
                        obs,
                        action,
                        log_prob,
                        0,  # Initial reward (will be updated after step)
                        value,
                        False  # Initial done flag
                    ))
                
                # Execute step in environment
                obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(current_actions)
                
                # Update rewards and done flags for stored experiences
                for agent_id, reward in reward_dict.items():
                    done = terminated_dict[agent_id] or truncated_dict[agent_id]
                    experience_queue.put((
                        obs_dict[agent_id],
                        None,  # Not needed for reward update
                        None,  # Not needed for reward update
                        reward,
                        None,  # Not needed for reward update
                        done
                    ))
                    
                truncated = any(truncated_dict.values())
                terminated = any(terminated_dict.values())
    
                if terminated or truncated:
                    if debug:
                        print(f"[DEBUG] Worker {rank} completed episode {episode+1} after {episode_steps} steps")
                    # Signal episode completion before resetting
                    progress_queue.put(1)
                    # No need to break here as the outer while loop will exit
            
            # Signal episode completion to progress bar
            progress_queue.put(1)
    
    except Exception as e:
        print(f"Error in worker process {rank}: {str(e)}", file=sys.stderr)
    finally:
        # Clean up environment
        if 'env' in locals():
            env.close()

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
    # Prepare models to be shared
    actor.to("cpu")  # Ensure models are on CPU for sharing
    critic.to("cpu")
    actor.share_memory()
    critic.share_memory()
    
    # Initialize trainer in the main process
    trainer = PPOTrainer(actor, critic, device=device, use_wandb=use_wandb, debug=debug)
    
    # Setup progress tracking
    episodes_per_process = max(1, total_episodes // num_processes)
    progress_bar = tqdm(total=total_episodes)
    progress_queue = mp.Queue()
    experience_queue = mp.Queue()
    trainer.experience_queue = experience_queue
    
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
                    actor,
                    critic,
                    device,
                    render,
                    process_episodes,
                    progress_queue,
                    experience_queue,
                    debug
                )
            )
            p.daemon = True  # Make sure processes exit when main exits
            p.start()
            processes.append(p)
        
        # Main process: collect experiences and update policy
        collected_experiences = 0
        completed_episodes = 0
        
        while completed_episodes < total_episodes and any(p.is_alive() for p in processes):
            # Update progress bar
            while not progress_queue.empty():
                completed_episodes += progress_queue.get()
                progress_bar.update(1)
                
                # Log to wandb if enabled
                if use_wandb:
                    wandb.log({
                        "episodes_completed": completed_episodes,
                        "completion_percentage": completed_episodes / total_episodes * 100
                    })
            
            # Collect experiences from queue
            experiences_this_iteration = 0
            while True:
                if trainer.collect_experiences(timeout=0.01):
                    collected_experiences += 1
                    experiences_this_iteration += 1
                else:
                    break  # No more experiences in queue

            # Update progress bar with collection info
            if collected_experiences > 0:
                progress_bar.set_description(f"Collecting: {collected_experiences}/{update_interval}")
            
            # Update policy when enough experiences are collected
            if collected_experiences >= update_interval:
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
            p.terminate()
            p.join(timeout=1.0)
        
        progress_bar.close()
        
        # Final update with any remaining experiences
        if collected_experiences > 0:
            # Collect any remaining experiences
            while True:
                if not trainer.collect_experiences(timeout=0.01):
                    break
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
    progress_bar.set_description("Training complete - Models saved")
