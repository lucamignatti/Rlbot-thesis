#!/usr/bin/env python3
"""
Curriculum Validation Script

This script validates a curriculum to ensure all components work correctly
before starting a long training session.

Usage:
  python verify.py --stage <stage_index> [--episodes <num_episodes>] [--no-render] [--debug]
  python verify.py --stage <stage_index> --skill <skill_index> [--episodes <num_episodes>] [--no-render] [--debug]
"""

import argparse
import importlib
import traceback
import sys
import numpy as np
import os
from typing import Any, Optional, Tuple, Dict, List

# Add parent directory to path to allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import project modules using absolute imports
from observation import ActionStacker
from envs.factory import get_env
from rlgym.rocket_league.rlviser import RLViserRenderer

# Import from curriculum using absolute imports
import curriculum.base
from curriculum.base import CurriculumManager
from curriculum.curriculum import create_curriculum
from curriculum.skills import SkillBasedCurriculumStage, SkillModule

import signal
import time

# Local implementation of DiscreteAction
class DiscreteAction:
    """Local implementation of discrete actions without external dependencies."""
    def __init__(self):
        # Instead of returning a single integer, we need to return a numpy array
        # for the lookup_table_action parser
        pass

    def get_action(self) -> np.ndarray:
        # Return a numpy array with a single element (0)
        # This format is expected by lookup_table_action parser
        return np.array([8], dtype=np.int32)


def import_module_function(module_path: str, function_name: str) -> Tuple[Any, Optional[Exception]]:
    """
    Import a function from a module path.
    
    Args:
        module_path: The path to the module (e.g. 'curriculum.curriculum')
        function_name: The name of the function (e.g. 'create_lucy_skg_curriculum')
    
    Returns:
        Tuple containing:
        - The imported function, or None if import failed
        - Any exception that occurred during import, or None on success
    """
    try:
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        return function, None
    except ImportError as e:
        return None, e
    except AttributeError as e:
        return None, e
    except Exception as e:
        return None, e

def validate_curriculum(curriculum_module: str = "curriculum.curriculum", 
                       curriculum_func: str = "create_lucy_skg_curriculum",
                       debug: bool = True):
    """
    Validate a curriculum by creating it and running the validation function.
    
    Args:
        curriculum_module: Module path containing the curriculum creation function
        curriculum_func: Name of the function that creates the curriculum
        debug: Whether to run with debug output enabled
    
    Returns:
        bool: True if validation passed, False otherwise
    """
    print(f"\n=== CURRICULUM VALIDATION ===")
    print(f"Loading curriculum from {curriculum_module}.{curriculum_func}...")
    
    # Import the curriculum creation function
    creator_func, import_error = import_module_function(curriculum_module, curriculum_func)
    if import_error:
        print(f"ERROR: Failed to import curriculum from {curriculum_module}.{curriculum_func}")
        print(f"       {import_error}")
        return False
    
    # Create the curriculum
    try:
        print(f"Creating curriculum...")
        curriculum_manager = creator_func(debug=debug, use_wandb=False)
        if not isinstance(curriculum_manager, CurriculumManager):
            print(f"ERROR: The function did not return a CurriculumManager object")
            print(f"       Got: {type(curriculum_manager)}")
            return False
    except Exception as e:
        print(f"ERROR: Failed to create curriculum")
        print(f"       {e}")
        traceback.print_exc(file=sys.stdout)
        return False
    
    # Run validation on the curriculum
    try:
        print(f"Running validation on curriculum with {len(curriculum_manager.stages)} stages...\n")
        validation_result = curriculum_manager.validate_all_stages()
        return validation_result
    except Exception as e:
        print(f"ERROR: Exception during curriculum validation")
        print(f"       {e}")
        traceback.print_exc(file=sys.stdout)
        return False

# Add timeout handler to prevent infinite waiting
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def run_skill_module(stage_index: int, skill_index: int, render: bool = True, num_episodes: int = 5, debug: bool = False, render_delay: float = 0.05):
    """
    Run a specific skill module for testing.
    
    Args:
        stage_index: Index of the curriculum stage containing the skill module
        skill_index: Index of the skill module within the stage
        render: Whether to render the environment
        num_episodes: Number of episodes to run
        debug: Whether to run with debug output enabled
        render_delay: Delay between rendered frames (seconds)
    """
    print(f"\nTesting skill module at stage {stage_index}, skill {skill_index}")
    
    # Create curriculum manager and get stage
    curriculum_manager = create_curriculum(debug=debug)
    
    if stage_index >= len(curriculum_manager.stages):
        print(f"Error: Stage index {stage_index} is out of range. Max stage is {len(curriculum_manager.stages) - 1}")
        return
    
    stage = curriculum_manager.stages[stage_index]
    print(f"Stage name: {stage.name}")
    
    # Check if stage is skill-based
    if not isinstance(stage, SkillBasedCurriculumStage):
        print(f"Error: Stage at index {stage_index} is not a SkillBasedCurriculumStage.")
        print(f"Stage type: {type(stage).__name__}")
        print(f"Cannot test individual skill modules on this stage type.")
        return
        
    # Check if skill index is valid
    if not stage.skill_modules or skill_index >= len(stage.skill_modules):
        print(f"Error: Skill index {skill_index} is out of range.")
        print(f"Stage '{stage.name}' has {len(stage.skill_modules) if stage.skill_modules else 0} skill modules.")
        if stage.skill_modules:
            print(f"Available skill modules:")
            for i, skill in enumerate(stage.skill_modules):
                print(f"  {i}: {skill.name}")
        return
    
    # Get the skill module
    skill = stage.skill_modules[skill_index]
    print(f"Skill module name: {skill.name}")
    
    # Create action stacker with default parameters
    action_stacker = ActionStacker(stack_size=5, action_size=8)
    
    # Create a single renderer instance to be reused
    renderer = None
    if render:
        try:
            # Force enable RLViser before importing
            os.environ['RLVISER_ENABLED'] = '1'
            renderer = RLViserRenderer()
            print("RLViser renderer initialized successfully")
            # Give renderer time to initialize properly
            time.sleep(1.0)
        except Exception as e:
            print(f"Warning: Failed to initialize RLViser renderer: {e}")
            render = False
    
    # Apply default difficulty level (0.5)
    difficulty_level = 0.5
    
    # Get the skill configuration
    skill_config = skill.get_config(difficulty_level)
    print(f"Using difficulty level: {difficulty_level}")
    if debug:
        print(f"Skill difficulty parameters:")
        for param, value in skill_config.items():
            print(f"  {param}: {value}")
    
    # Create curriculum configuration for just this skill module
    curriculum_config = {
        'stage_name': f"{stage.name} - {skill.name}",
        'state_mutator': skill.state_mutator,
        'reward_function': skill.reward_function,
        'termination_condition': skill.termination_condition,
        'truncation_condition': skill.truncation_condition,
        'difficulty_params': skill_config
    }

    if debug:
        print(f"\nCreating environment with configuration:")
        print(f"- State mutator: {curriculum_config['state_mutator'].__class__.__name__}")
        print(f"- Reward function: {curriculum_config['reward_function'].__class__.__name__}")
        print(f"- Termination condition: {curriculum_config['termination_condition'].__class__.__name__}")
        print(f"- Truncation condition: {curriculum_config['truncation_condition'].__class__.__name__}")

    print("Initializing environment...")
    env = get_env(
        renderer=renderer,
        action_stacker=action_stacker,
        curriculum_config=curriculum_config,
        debug=debug
    )
    print("Environment initialized successfully")

    discrete_action = DiscreteAction()
    zero_action = discrete_action.get_action()
    
    # Track rewards across episodes
    episode_rewards = []
    episode_steps = []
    success_count = 0
    MAX_STEPS_PER_EPISODE = 500  # Safety limit

    print(f"\nRunning {num_episodes} episodes...")
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print("Resetting environment...")
        obs = env.reset()
        print("Environment reset complete")
        
        episode_reward = 0
        steps = 0
        done = False
        done_dict = {}
        
        print("Starting episode loop...")
        while True:
            # Check for episode completion
            if isinstance(done_dict, dict):
                if done_dict.get('__all__', False):
                    print(f"Episode complete: __all__ flag set")
                    break
                
                # Check if all agents are done
                if done_dict and all(done_dict.values()):
                    print(f"Episode complete: all agents done")
                    break
                    
                # Check if any agent is done (some environments end when any agent is done)
                if done_dict and any(done_dict.values()):
                    agent_dones = [agent for agent, is_done in done_dict.items() if is_done]
                    if len(agent_dones) > 0:
                        print(f"Agents {agent_dones} are done")
            
            # Simple done flag (backwards compatibility)
            if steps > 0 and (done is True):
                print(f"Episode complete: done flag is True")
                break
            
            # Create action dictionary based on observation
            if isinstance(obs, dict):
                action_dict = {agent_id: zero_action for agent_id in obs.keys()}
            else:
                action_dict = {'blue-0': zero_action}  # Default agent ID if not specified
            
            # Add timeout for env.step to catch hangs
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 second timeout
            
            try:
                # Only print detailed logs every few steps to reduce output
                verbose_log = steps % 50 == 0 or steps < 5
                
                if verbose_log:
                    print(f"Step {steps}...")
                    
                step_start = time.time()
                obs, reward, done_dict, info = env.step(action_dict)
                step_end = time.time()
                
                if verbose_log:
                    print(f"  Step completed in {step_end - step_start:.2f}s")
                    print(f"  Reward: {reward}")
                    if isinstance(reward, dict) and len(reward) > 0:
                        print(f"  Agent rewards: {', '.join([f'{a}: {r:.4f}' for a, r in reward.items()])}")
                
                signal.alarm(0)  # Cancel the alarm
                
                # Track rewards for this episode
                if isinstance(reward, dict):
                    step_reward = sum(reward.values())
                else:
                    step_reward = reward
                    
                episode_reward += step_reward
                
                # Store done flag for next iteration
                done = done_dict if isinstance(done_dict, bool) else any(done_dict.values())
                
                steps += 1
                
                # Only render and delay for the rendered environment
                if render and hasattr(env, 'render'):
                    env.render()
                    # Add delay to slow down simulation and give renderer time to display
                    time.sleep(render_delay)
                
            except TimeoutError:
                print("ERROR: env.step() timed out after 10 seconds!")
                print("This suggests a deadlock in the environment's step function")
                # Try to gracefully exit
                try:
                    env.close()
                except:
                    pass
                return
            
            # Safety exit - prevent infinite episodes
            if steps >= MAX_STEPS_PER_EPISODE:
                print(f"Reached maximum steps ({MAX_STEPS_PER_EPISODE}) - forcing episode end for safety")
                break
        
        # Store episode stats
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        
        # Evaluate episode success based on skill's success criteria
        # Here we use a simple heuristic - check if the reward is above 50% of max observed so far
        # This is just for visualization - the actual skill module would have proper success metrics
        is_success = done and episode_reward > 0
        if is_success:
            success_count += 1
            
        print(f"\nEpisode {episode+1} summary:")
        print(f"- Steps: {steps}")
        print(f"- Total reward: {episode_reward:.4f}")
        print(f"- Success: {'Yes' if is_success else 'No'}")
        
        # Reset action stacker between episodes if needed
        if action_stacker is not None and isinstance(obs, dict):
            for agent_id in obs.keys():
                action_stacker.reset_agent(agent_id)
    
    # Print summary statistics
    if episode_rewards:
        print("\n=== Skill Module Statistics ===")
        print(f"Skill: {skill.name}")
        print(f"Episodes run: {len(episode_rewards)}")
        print(f"Success rate: {success_count/len(episode_rewards):.2%}")
        print(f"Average steps per episode: {np.mean(episode_steps):.1f}")
        print(f"Average reward per episode: {np.mean(episode_rewards):.4f}")
        print(f"Min reward: {np.min(episode_rewards):.4f}")
        print(f"Max reward: {np.max(episode_rewards):.4f}")
        print(f"Std deviation: {np.std(episode_rewards):.4f}")
    
    print("\nClosing environment...")
    env.close()
    print("Environment closed")

def run_stage(stage_index: int, render: bool = True, num_episodes: int = 5, debug: bool = False, render_delay: float = 0.05):
    print(f"\nTesting curriculum stage {stage_index}")
    
    # Create curriculum manager and get stage
    curriculum_manager = create_curriculum(debug=debug)
    if stage_index >= len(curriculum_manager.stages):
        print(f"Error: Stage index {stage_index} is out of range. Max stage is {len(curriculum_manager.stages) - 1}")
        return
        
    stage = curriculum_manager.stages[stage_index]
    print(f"Stage name: {stage.name}")
    
    # List skill modules if this is a SkillBasedCurriculumStage
    if isinstance(stage, SkillBasedCurriculumStage) and stage.skill_modules:
        print(f"Stage contains {len(stage.skill_modules)} skill modules:")
        for i, skill in enumerate(stage.skill_modules):
            print(f"  {i}: {skill.name}")
        print(f"Use --skill <index> to test a specific skill module")
    
    # Create action stacker with default parameters
    action_stacker = ActionStacker(stack_size=5, action_size=8)
    
    # Create a single renderer instance to be reused
    renderer = None
    if render:
        try:
            # Force enable RLViser before importing
            os.environ['RLVISER_ENABLED'] = '1'
            renderer = RLViserRenderer()
            print("RLViser renderer initialized successfully")
            # Give renderer time to initialize properly
            time.sleep(1.0)
        except Exception as e:
            print(f"Warning: Failed to initialize RLViser renderer: {e}")
            render = False
    
    # Get the curriculum configuration for this stage
    curriculum_config = {
        'stage_name': stage.name,
        'state_mutator': stage.state_mutator,
        'reward_function': stage.reward_function,
        'termination_condition': stage.termination_condition,
        'truncation_condition': stage.truncation_condition
    }
    if debug:
        print(f"\nCreating environment with configuration:")
        print(f"- State mutator: {curriculum_config['state_mutator'].__class__.__name__}")
        print(f"- Reward function: {curriculum_config['reward_function'].__class__.__name__}")
        print(f"- Termination condition: {curriculum_config['termination_condition'].__class__.__name__}")
        print(f"- Truncation condition: {curriculum_config['truncation_condition'].__class__.__name__}")
    print("Initializing environment...")
    env = get_env(
        renderer=renderer,
        action_stacker=action_stacker,
        curriculum_config=curriculum_config,
        debug=debug
    )
    print("Environment initialized successfully")
    discrete_action = DiscreteAction()
    zero_action = discrete_action.get_action()
    
    # Track rewards across episodes
    episode_rewards = []
    episode_steps = []
    MAX_STEPS_PER_EPISODE = 500  # Safety limit
    print(f"\nRunning {num_episodes} episodes...")
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print("Resetting environment...")
        obs = env.reset()
        print("Environment reset complete")
        
        episode_reward = 0
        steps = 0
        done = False
        done_dict = {}
        
        print("Starting episode loop...")
        while True:
            # Check for episode completion
            if isinstance(done_dict, dict):
                if done_dict.get('__all__', False):
                    print(f"Episode complete: __all__ flag set")
                    break
                
                # Check if all agents are done
                if done_dict and all(done_dict.values()):
                    print(f"Episode complete: all agents done")
                    break
                    
                # Check if any agent is done (some environments end when any agent is done)
                if done_dict and any(done_dict.values()):
                    agent_dones = [agent for agent, is_done in done_dict.items() if is_done]
                    if len(agent_dones) > 0:
                        print(f"Agents {agent_dones} are done")
            
            # Simple done flag (backwards compatibility)
            if steps > 0 and (done is True):
                print(f"Episode complete: done flag is True")
                break
            
            # Create action dictionary based on observation
            if isinstance(obs, dict):
                action_dict = {agent_id: zero_action for agent_id in obs.keys()}
            else:
                action_dict = {'blue-0': zero_action}  # Default agent ID if not specified
            
            # Add timeout for env.step to catch hangs
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 second timeout
            
            try:
                # Only print detailed logs every few steps to reduce output
                verbose_log = steps % 50 == 0 or steps < 5
                
                if verbose_log:
                    print(f"Step {steps}...")
                    
                step_start = time.time()
                obs, reward, done_dict, info = env.step(action_dict)
                step_end = time.time()
                
                if verbose_log:
                    print(f"  Step completed in {step_end - step_start:.2f}s")
                    print(f"  Reward: {reward}")
                    if isinstance(reward, dict) and len(reward) > 0:
                        print(f"  Agent rewards: {', '.join([f'{a}: {r:.4f}' for a, r in reward.items()])}")
                
                signal.alarm(0)  # Cancel the alarm
                
                # Track rewards for this episode
                if isinstance(reward, dict):
                    step_reward = sum(reward.values())
                else:
                    step_reward = reward
                    
                episode_reward += step_reward
                
                # Store done flag for next iteration
                done = done_dict if isinstance(done_dict, bool) else any(done_dict.values())
                
                steps += 1
                
                # Only render and delay for the rendered environment
                if render and hasattr(env, 'render'):
                    env.render()
                    # Add delay to slow down simulation and give renderer time to display
                    time.sleep(render_delay)
                
            except TimeoutError:
                print("ERROR: env.step() timed out after 10 seconds!")
                print("This suggests a deadlock in the environment's step function")
                # Try to gracefully exit
                try:
                    env.close()
                except:
                    pass
                return
            
            # Safety exit - prevent infinite episodes
            if steps >= MAX_STEPS_PER_EPISODE:
                print(f"Reached maximum steps ({MAX_STEPS_PER_EPISODE}) - forcing episode end for safety")
                break
        
        # Store episode stats
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        print(f"\nEpisode {episode+1} summary:")
        print(f"- Steps: {steps}")
        print(f"- Total reward: {episode_reward:.4f}")
        
        # Reset action stacker between episodes if needed
        if action_stacker is not None and isinstance(obs, dict):
            for agent_id in obs.keys():
                action_stacker.reset_agent(agent_id)
    
    # Print summary statistics
    if episode_rewards:
        print("\n=== Episode Statistics ===")
        print(f"Episodes run: {len(episode_rewards)}")
        print(f"Average steps per episode: {np.mean(episode_steps):.1f}")
        print(f"Average reward per episode: {np.mean(episode_rewards):.4f}")
        print(f"Min reward: {np.min(episode_rewards):.4f}")
        print(f"Max reward: {np.max(episode_rewards):.4f}")
        print(f"Std deviation: {np.std(episode_rewards):.4f}")
    
    print("\nClosing environment...")
    env.close()
    print("Environment closed")

def main():
    parser = argparse.ArgumentParser(description="Run and render curriculum stages or skill modules for testing.")
    parser.add_argument("--stage", type=int, help="Index of the curriculum stage to run.")
    parser.add_argument("--skill", type=int, help="Index of the skill module to run (only for SkillBasedCurriculumStage).")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    parser.add_argument("--render-delay", type=float, default=0.05, help="Delay between frames in seconds (default: 0.05)")
    parser.add_argument("--list", action="store_true", help="List available stages and their skill modules.")
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        curriculum_manager = create_curriculum(debug=args.debug)
        print("\n=== Available Curriculum Stages ===")
        for i, stage in enumerate(curriculum_manager.stages):
            if isinstance(stage, SkillBasedCurriculumStage) and stage.skill_modules:
                print(f"{i}: {stage.name} ({len(stage.skill_modules)} skill modules)")
                for j, skill in enumerate(stage.skill_modules):
                    print(f"   - Skill {j}: {skill.name}")
            else:
                print(f"{i}: {stage.name}")
        return
    
    # Check if stage is provided for other operations
    if args.stage is None:
        parser.error("the --stage argument is required when not using --list")
    
    # Run either a skill module or a stage
    if args.skill is not None:
        run_skill_module(
            stage_index=args.stage,
            skill_index=args.skill,
            render=not args.no_render,
            num_episodes=args.episodes,
            debug=args.debug,
            render_delay=args.render_delay
        )
    else:
        run_stage(
            stage_index=args.stage,
            render=not args.no_render,
            num_episodes=args.episodes,
            debug=args.debug,
            render_delay=args.render_delay
        )

if __name__ == "__main__":
    main()