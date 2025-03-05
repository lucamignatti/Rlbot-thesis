import os
import sys
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import multiprocessing as mp
from multiprocessing import Process, Pipe
import concurrent.futures
from collections import defaultdict

from rlbot_registry import RLBotPackRegistry
from rlgym.rocket_league.api import GameState

class RLBotVectorizedEnv:
    """Vectorized environment that supports RLBotPack opponents"""
    def __init__(self, num_envs: int, render: bool = False, action_stacker=None,
                 curriculum_manager=None, rlbotpack_path: Optional[str] = None):
        """
        Initialize vectorized environment with RLBot support.
        
        Args:
            num_envs: Number of parallel environments
            render: Whether to render the environment
            action_stacker: For tracking action history
            curriculum_manager: For curriculum learning
            rlbotpack_path: Path to RLBotPack repository
        """
        self.num_envs = num_envs
        self.render = render
        self.action_stacker = action_stacker
        self.curriculum_manager = curriculum_manager
        self.render_delay = 0.025  # 25ms delay between rendered frames
        
        # Initialize RLBotPack if path provided
        self.bot_registry = None
        if rlbotpack_path:
            self.bot_registry = RLBotPackRegistry(rlbotpack_path)
        
        # For tracking episode metrics for curriculum
        self.episode_rewards = [{} for _ in range(num_envs)]
        self.episode_successes = [False] * num_envs
        self.episode_timeouts = [False] * num_envs
        
        # Get curriculum configurations if available
        self.curriculum_configs = []
        for env_idx in range(num_envs):
            if self.curriculum_manager:
                # Get potentially different configs for each environment (for rehearsal)
                self.curriculum_configs.append(self.curriculum_manager.get_environment_config())
            else:
                self.curriculum_configs.append(None)
        
        # Track active bots in each environment
        self.bot_agents = defaultdict(dict)  # (env_idx, agent_id) -> RLBotAdapter
        
        # Use threading when rendering is enabled
        if render:
            self.mode = "thread"
            
            # Only render the first environment
            self._setup_threaded_environments()
            
        else:
            # Use multiprocessing for maximum performance when not rendering
            self.mode = "multiprocess"
            self._setup_multiprocess_environments()
        
        # Common initialization
        self.dones = [False] * num_envs
        self.episode_counts = [0] * num_envs
    
    def _setup_threaded_environments(self):
        """Set up thread-based environments for rendering"""
        from rlgym.api import RLGym
        from rlgym.rocket_league.rlviser import RLViserRenderer
        
        # Create environments directly
        self.envs = []
        for i in range(self.num_envs):
            # Only create renderer for the first environment
            env_renderer = RLViserRenderer() if (i == 0) else None
            env = RLGym(renderer=env_renderer, curriculum_config=self.curriculum_configs[i])
            self.envs.append(env)
        
        # Reset all environments
        self.obs_dicts = [env.reset() for env in self.envs]
        
        # Explicitly render the first environment
        if self.num_envs > 0:
            self.envs[0].render()
        
        # Set up thread pool for parallel execution
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, self.num_envs),
            thread_name_prefix='EnvWorker'
        )
    
    def _setup_multiprocess_environments(self):
        """Set up process-based environments for parallel execution"""
        # Set the multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Create communication pipes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        
        # Create and start worker processes
        self.processes = []
        for idx, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            process = Process(
                target=self._env_worker,
                args=(work_remote, self.curriculum_configs[idx]),
                daemon=True
            )
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        # Get initial observations
        for remote in self.remotes:
            remote.send(('reset', None))
        self.obs_dicts = [remote.recv() for remote in self.remotes]
    
    def _env_worker(self, remote, curriculum_config):
        """Worker process that runs a single environment"""
        from rlgym.api import RLGym
        
        # Create environment with curriculum config if provided
        env = RLGym(curriculum_config=curriculum_config)
        
        while True:
            try:
                cmd, data = remote.recv()
                
                if cmd == 'step':
                    actions_dict = data
                    # Step the environment
                    next_obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions_dict)
                    remote.send((next_obs_dict, reward_dict, terminated_dict, truncated_dict))
                
                elif cmd == 'reset':
                    obs = env.reset()
                    remote.send(obs)
                
                elif cmd == 'get_state':
                    state = env.get_state()
                    remote.send(state)
                
                elif cmd == 'set_curriculum':
                    # Update environment with new curriculum configuration
                    env.close()
                    env = RLGym(curriculum_config=data)
                    remote.send(True)  # Acknowledge the update
                
                elif cmd == 'close':
                    env.close()
                    remote.close()
                    break
                
            except EOFError:
                break
    
    def register_opponent_bot(self, env_idx: int, agent_id: int, bot_id: Optional[str] = None,
                            skill_range: Optional[Tuple[float, float]] = None):
        """
        Register an opponent bot for a specific agent in an environment.
        
        Args:
            env_idx: Environment index
            agent_id: Agent ID to replace with bot
            bot_id: Specific bot ID to use (or None for random selection)
            skill_range: Tuple of (min_skill, max_skill) for random selection
        """
        if not self.bot_registry:
            raise RuntimeError("RLBotPack registry not initialized")
        
        # Stop any existing bot for this agent
        if (env_idx, agent_id) in self.bot_agents:
            self.bot_agents[env_idx, agent_id].stop()
            del self.bot_agents[env_idx, agent_id]
        
        # Select a bot if not specified
        if bot_id is None:
            min_skill = 0.3
            max_skill = 0.7
            
            if skill_range:
                min_skill, max_skill = skill_range
            
            # Get random bot in skill range
            bot_info = self.bot_registry.get_random_bot(min_skill, max_skill)
            bot_id = bot_info['id']
        
        # Create and start the adapter
        adapter = self.bot_registry.create_bot_adapter(bot_id, team=self._get_agent_team(agent_id))
        try:
            adapter.start()
            self.bot_agents[env_idx, agent_id] = adapter
        except Exception as e:
            print(f"Failed to start bot {bot_id} for agent {agent_id} in env {env_idx}: {str(e)}")
    
    def _get_agent_team(self, agent_id: int) -> int:
        """Determine team for an agent ID based on environment configuration"""
        # Team 0 (blue) is agent_id % 2 == 0
        # Team 1 (orange) is agent_id % 2 == 1
        return agent_id % 2
    
    def _step_env(self, args):
        """Step a single threaded environment forward"""
        env_idx, env, actions_dict = args
        
        try:
            # Get bot actions first
            full_actions = actions_dict.copy()
            
            # Get the current game state if we have bot opponents
            if (env_idx, 1) in self.bot_agents:  # Check if we have any bots in this env
                state = env.get_state()
                
                # Get actions from each bot
                for agent_id, bot in self.bot_agents.items():
                    if agent_id[0] == env_idx:  # Match environment index
                        try:
                            bot_action = bot.get_action(state)
                            full_actions[agent_id[1]] = bot_action
                        except Exception as e:
                            print(f"Error getting bot action: {e}")
                            full_actions[agent_id[1]] = np.zeros(8)  # Fallback to no-op
            
            # Step the environment with all actions
            next_obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(full_actions)
            
            # Add rendering delay if this is the rendered environment
            if self.render and env_idx == 0:
                env.render()
                time.sleep(self.render_delay)
            
            # Track rewards for curriculum
            if self.curriculum_manager is not None:
                for agent_id, reward in reward_dict.items():
                    if agent_id not in self.episode_rewards[env_idx]:
                        self.episode_rewards[env_idx][agent_id] = 0
                    self.episode_rewards[env_idx][agent_id] += reward
            
            return env_idx, next_obs_dict, reward_dict, terminated_dict, truncated_dict
            
        except Exception as e:
            print(f"Error in environment {env_idx}: {e}")
            return env_idx, {}, {}, {}, {}
    
    def step(self, actions_dict_list: List[Dict[int, np.ndarray]]):
        """
        Step all environments forward.
        
        Args:
            actions_dict_list: List of action dictionaries for each environment
            
        Returns:
            Tuple of (results, dones, episode_counts)
        """
        if self.mode == "thread":
            # Use thread pool for parallel execution
            futures = [
                self.executor.submit(self._step_env, (i, env, actions))
                for i, (env, actions) in enumerate(zip(self.envs, actions_dict_list))
            ]
            
            # Wait for all steps to complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            
            # Sort results by environment index
            results.sort(key=lambda x: x[0])
            
            # Process results and track episode completion
            processed_results = []
            for env_idx, next_obs_dict, reward_dict, terminated_dict, truncated_dict in results:
                # Check if episode is done
                self.dones[env_idx] = any(terminated_dict.values()) or any(truncated_dict.values())
                
                # Track success/timeout for curriculum
                if self.dones[env_idx]:
                    self.episode_successes[env_idx] = any(terminated_dict.values())
                    self.episode_timeouts[env_idx] = any(truncated_dict.values())
                    
                    # Update curriculum with episode results
                    if self.curriculum_manager:
                        # Prepare episode metrics
                        bot_ids = {
                            agent_id[1]: bot.bot_id
                            for agent_id, bot in self.bot_agents.items()
                            if agent_id[0] == env_idx
                        }
                        
                        self.curriculum_manager.update_progression_stats({
                            'success': self.episode_successes[env_idx],
                            'timeout': self.episode_timeouts[env_idx],
                            'episode_reward': sum(self.episode_rewards[env_idx].values()),
                            'env_idx': env_idx,
                            'opponent_bot_ids': bot_ids
                        })
                    
                    # Reset episode tracking
                    if self.dones[env_idx]:
                        self.episode_counts[env_idx] += 1
                        self.episode_rewards[env_idx] = {}
                        self.episode_successes[env_idx] = False
                        self.episode_timeouts[env_idx] = False
                        
                        # Reset the environment
                        self.obs_dicts[env_idx] = self.envs[env_idx].reset()
                        
                        # Reset action stacker if used
                        if self.action_stacker:
                            for agent_id in next_obs_dict:
                                self.action_stacker.reset_agent(agent_id)
                    
                processed_results.append((next_obs_dict, reward_dict, terminated_dict, truncated_dict))
            
        else:  # multiprocess mode
            # Send step commands to all workers
            for remote, actions_dict in zip(self.remotes, actions_dict_list):
                remote.send(('step', actions_dict))
            
            # Collect results from all workers
            results = []
            for i, remote in enumerate(self.remotes):
                next_obs_dict, reward_dict, terminated_dict, truncated_dict = remote.recv()
                
                # Track rewards for curriculum
                if self.curriculum_manager is not None:
                    for agent_id, reward in reward_dict.items():
                        if agent_id not in self.episode_rewards[i]:
                            self.episode_rewards[i][agent_id] = 0
                        self.episode_rewards[i][agent_id] += reward
                
                # Check if episode is done
                self.dones[i] = any(terminated_dict.values()) or any(truncated_dict.values())
                
                if self.dones[i]:
                    # Update curriculum with episode results
                    if self.curriculum_manager:
                        bot_ids = {
                            agent_id[1]: bot.bot_id
                            for agent_id, bot in self.bot_agents.items()
                            if agent_id[0] == i
                        }
                        
                        self.curriculum_manager.update_progression_stats({
                            'success': any(terminated_dict.values()),
                            'timeout': any(truncated_dict.values()),
                            'episode_reward': sum(self.episode_rewards[i].values()),
                            'env_idx': i,
                            'opponent_bot_ids': bot_ids
                        })
                    
                    # Reset environment and tracking variables
                    remote.send(('reset', None))
                    self.obs_dicts[i] = remote.recv()
                    self.episode_counts[i] += 1
                    self.episode_rewards[i] = {}
                    
                    # Reset action stacker if used
                    if self.action_stacker:
                        for agent_id in next_obs_dict:
                            self.action_stacker.reset_agent(agent_id)
                
                results.append((next_obs_dict, reward_dict, terminated_dict, truncated_dict))
            
            processed_results = results
        
        return processed_results, self.dones.copy(), self.episode_counts.copy()
    
    def close(self):
        """Clean up resources properly"""
        # Stop all bot processes first
        for adapter in self.bot_agents.values():
            try:
                adapter.stop()
            except:
                pass
        self.bot_agents.clear()
        
        # Clean up environment resources
        if self.mode == "thread":
            if hasattr(self, 'executor'):
                self.executor.shutdown()
            
            if hasattr(self, 'envs'):
                for env in self.envs:
                    env.close()
        
        else:  # multiprocess mode
            if hasattr(self, 'remotes'):
                for remote in self.remotes:
                    try:
                        remote.send(('close', None))
                    except (BrokenPipeError, EOFError):
                        pass
            
            if hasattr(self, 'processes'):
                for process in self.processes:
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.terminate()