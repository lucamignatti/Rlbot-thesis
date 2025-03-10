from .vectorized import VectorizedEnv
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
import os
from rlbot.registry import RLBotPackRegistry
from .factory import get_env
import select


class RLBotVectorizedEnv(VectorizedEnv):
    """
    Subclass of VectorizedEnv that integrates RLBot opponents.
    
    This class extends the VectorizedEnv to allow the use of pre-built RLBot opponents
    from the RLBotPack in training environments.
    """
    def __init__(
        self,
        num_envs: int,
        render: bool = False,
        action_stacker=None,
        curriculum_manager=None,
        rlbotpack_path: Optional[str] = None,
        debug: bool = False
    ):
        # Initialize bot registry in main process only
        self.rlbotpack_path = rlbotpack_path
        if rlbotpack_path and not hasattr(self, 'bot_registry'):
            from rlbot.registry import RLBotPackRegistry
            self.bot_registry = RLBotPackRegistry(rlbotpack_path)
            if debug:
                print(f"[DEBUG] Bot registry initialized with path: {rlbotpack_path}")
                print(f"[DEBUG] Found {len(self.bot_registry.available_bots)} available bots")
        
        # Initialize parent class
        super().__init__(
            num_envs=num_envs,
            render=render,
            action_stacker=action_stacker,
            curriculum_manager=curriculum_manager,
            debug=debug
        )
        
        # Dictionary to store bot agents
        self.bot_agents = defaultdict(dict)
        
    def _setup_bot_registry_in_workers(self):
        """Initialize bot registry in all worker processes"""
        if not self.rlbotpack_path:
            return
            
        for remote in self.remotes:
            try:
                remote.send(('set_bot_registry_path', self.rlbotpack_path))
                if select.select([remote], [], [], 10.0)[0]:  # 10 second timeout
                    response = remote.recv()
                    if self.debug and response:
                        print(f"[DEBUG] Bot registry path sent to worker")
                else:
                    if self.debug:
                        print(f"[DEBUG] Timeout waiting for bot registry setup")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error setting up bot registry in worker: {e}")

    def register_opponent_bot(self, env_idx: int, agent_id: int, bot_id: str):
        """
        Register an opponent bot to a specific environment and agent ID.
        
        Args:
            env_idx: Index of the environment where the bot should be added
            agent_id: Agent ID for the bot within the environment
            bot_id: Identifier of the bot in the RLBotPack
            
        Returns:
            bool: Whether the registration was successful
        """
        if self.debug:
            print(f"[DEBUG] Registering bot {bot_id} in env {env_idx} as agent {agent_id}")
            
        if env_idx >= self.num_envs:
            raise ValueError(f"Environment index {env_idx} out of range (0-{self.num_envs-1})")
        
        # Handle registration differently based on execution mode
        if self.mode == "multiprocess":
            # In multiprocessing mode, send command to worker process
            self.remotes[env_idx].send(('register_bot', (env_idx, agent_id, bot_id)))
            success = self.remotes[env_idx].recv()  # Wait for confirmation
            
            # Store bot ID for reference (actual adapter lives in worker process)
            if success:
                self.bot_agents[(env_idx, agent_id)] = {"bot_id": bot_id}
            
            return success
        else:
            # In thread mode, create bot adapter directly
            if self.bot_registry is None:
                raise ValueError("Bot registry not initialized. Provide rlbotpack_path in constructor.")
                
            try:
                # Create bot adapter - Fixed: use create_bot_adapter instead of get_bot
                bot_adapter = self.bot_registry.create_bot_adapter(bot_id, team=1)  # Default to team 1
                if bot_adapter is None:
                    if self.debug:
                        print(f"[DEBUG] Failed to create adapter for bot {bot_id}")
                    return False
                    
                # Store the adapter
                self.bot_agents[(env_idx, agent_id)] = bot_adapter
                
                if self.debug:
                    print(f"[DEBUG] Bot {bot_id} registered successfully")
                return True
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error registering bot: {str(e)}")
                return False
    
    def unregister_opponent_bot(self, env_idx: int, agent_id: int):
        """Remove a bot from a specific environment and agent ID"""
        key = (env_idx, agent_id)
        
        if key not in self.bot_agents:
            return False
            
        if self.mode == "multiprocess":
            # Send command to worker process
            self.remotes[env_idx].send(('unregister_bot', (env_idx, agent_id)))
            success = self.remotes[env_idx].recv()  # Wait for confirmation
            
            # Remove from local tracking if successful
            if success:
                del self.bot_agents[key]
                
            return success
        else:
            # In thread mode, stop and remove the bot adapter
            try:
                if hasattr(self.bot_agents[key], 'stop'):
                    self.bot_agents[key].stop()
                del self.bot_agents[key]
                return True
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error unregistering bot: {str(e)}")
                return False
    
    def step(self, actions_dict_list):
        """Step all environments forward with bot actions efficiently batched in main process"""
        # Process bot actions in the main process first if we're in multiprocessing mode
        if self.mode == "multiprocess" and self.bot_agents:
            # Collect states from workers first
            states_by_env = {}
            for env_idx, remote in enumerate(self.remotes):
                # Non-blocking check if there are bots for this env
                bot_exists = False
                for (e_idx, _), _ in self.bot_agents.items():
                    if e_idx == env_idx:
                        bot_exists = True
                        break
                
                # Only request state if there are bots for this env
                if bot_exists:
                    try:
                        remote.send(('get_state', None))
                        if hasattr(select, 'select'):
                            if select.select([remote], [], [], 5.0)[0]:
                                states_by_env[env_idx] = remote.recv()
                            else:
                                if self.debug:
                                    print(f"[DEBUG] Timeout getting state from env {env_idx}")
                        else:
                            # Fallback if select not available
                            states_by_env[env_idx] = remote.recv()
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error getting state from env {env_idx}: {e}")
            
            # Process bot actions in batch in main process
            bot_actions_by_env = {}
            for (e_idx, agent_id), bot_info in self.bot_agents.items():
                if e_idx in states_by_env:  # Only process if we have a state
                    state = states_by_env[e_idx]
                    try:
                        # Get or create bot adapter if needed (lazy initialization)
                        if not hasattr(self, '_bot_adapters'):
                            self._bot_adapters = {}
                        
                        # Use bot_id as adapter key if available
                        adapter_key = bot_info.get("bot_id", None) if isinstance(bot_info, dict) else str(agent_id)
                        
                        if adapter_key and adapter_key not in self._bot_adapters:
                            if isinstance(bot_info, dict) and "bot_id" in bot_info:
                                self._bot_adapters[adapter_key] = self.bot_registry.create_bot_adapter(bot_info["bot_id"])
                        
                        # Calculate bot action only if we have a valid adapter
                        if adapter_key in self._bot_adapters:
                            if e_idx not in bot_actions_by_env:
                                bot_actions_by_env[e_idx] = {}
                                
                            # Get the bot action and store it
                            bot_action = self._bot_adapters[adapter_key].get_action(state)
                            bot_actions_by_env[e_idx][agent_id] = bot_action
                            
                            if self.debug and np.random.random() < 0.001:  # Occasional debug print
                                print(f"[DEBUG] Bot {agent_id} in env {e_idx} action computed")
                                
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error computing bot action for env {e_idx}, agent {agent_id}: {e}")
            
            # Send bot actions to workers
            for env_idx, bot_actions in bot_actions_by_env.items():
                if bot_actions:  # Only send if we have actions
                    try:
                        # Use non-blocking send to avoid deadlock
                        self.remotes[env_idx].send(('prepare_bot_actions', bot_actions))
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error sending bot actions to env {env_idx}: {e}")

        # Let parent class handle the stepping with our modified actions
        results, dones, episode_counts = super().step(actions_dict_list)
        return results, dones, episode_counts
    
    def reset(self):
        """Extended reset that maintains bot assignments"""
        # Let parent class handle the basic reset
        obs = super().reset()
        
        # Re-register bots after reset if we're in thread mode
        # (In multiprocessing mode, workers handle this internally)
        if self.mode == "thread":
            # Dictionary to track which bots need to be re-registered
            bots_to_register = {}
            
            # Create copies of bot registrations to avoid modification during iteration
            for (env_idx, agent_id), bot in list(self.bot_agents.items()):
                if env_idx < self.num_envs:  # In case env count changed
                    if isinstance(bot, dict) and "bot_id" in bot:
                        bots_to_register[(env_idx, agent_id)] = bot["bot_id"]
                    elif hasattr(bot, 'bot_id'):
                        bots_to_register[(env_idx, agent_id)] = bot.bot_id
            
            # Clear bot agents dict first
            self.bot_agents.clear()
            
            # Re-register all bots
            for (env_idx, agent_id), bot_id in bots_to_register.items():
                self.register_opponent_bot(env_idx, agent_id, bot_id)
        
        return obs
    
    def close(self):
        """Extended cleanup that stops bot processes"""
        # Clean up bots first
        if self.mode == "thread":
            # In thread mode, we need to explicitly stop each bot
            for key, bot in list(self.bot_agents.items()):
                try:
                    if hasattr(bot, 'stop'):
                        bot.stop()
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Error stopping bot: {str(e)}")
        else:
            # In multiprocessing mode, worker processes handle bot cleanup
            # We just need to send unregister commands for all bots
            for (env_idx, agent_id) in list(self.bot_agents.keys()):
                if env_idx < len(self.remotes):
                    try:
                        self.remotes[env_idx].send(('unregister_bot', (env_idx, agent_id)))
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error sending unregister_bot command: {str(e)}")
        
        # Clear the bot agents dictionary
        self.bot_agents.clear()
        
        # Let parent class handle environment cleanup
        super().close()


def worker(remote, env_fn, render, action_stacker=None, curriculum_config=None, debug=False, bot_registry_path=None):
    """Enhanced worker process that supports RLBot integration"""
    try:
        # Create environment first with proper error handling
        if debug:
            print(f"[DEBUG] Worker initializing with stage: {curriculum_config['stage_name'] if curriculum_config else 'Default'}")
        
        env = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                env = env_fn(renderer=None, action_stacker=action_stacker, curriculum_config=curriculum_config, debug=debug)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    if debug:
                        print(f"[DEBUG] Attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(1)
                else:
                    raise
        
        if env is None:
            raise RuntimeError("Failed to create environment after max retries")

        # Initialize renderer if needed
        renderer = None
        if render:
            from rlgym.rocket_league.rlviser import RLViserRenderer
            renderer = RLViserRenderer()
            env.renderer = renderer
        
        # Initialize bot registry only if path is provided
        bot_registry = None
        bot_adapters = {}
        bot_actions = {}  # Store bot actions received from main process
        
        while True:
            try:
                # Use non-blocking receive with timeout
                cmd, data = remote.recv()
                
                if cmd == 'set_bot_registry_path':
                    try:
                        from rlbot.registry import RLBotPackRegistry
                        bot_registry_path = data
                        bot_registry = RLBotPackRegistry(bot_registry_path)
                        remote.send(True)
                        if debug:
                            print(f"[DEBUG] Bot registry initialized with path: {bot_registry_path}")
                    except Exception as e:
                        if debug:
                            print(f"[DEBUG] Error initializing bot registry: {e}")
                        remote.send(False)
                
                elif cmd == 'register_bot':
                    # Extract data
                    env_idx, agent_id, bot_id = data
                    
                    # Create bot adapter
                    success = False
                    if bot_registry:
                        try:
                            bot = bot_registry.create_bot_adapter(bot_id, team=1)  # Default to team 1
                            if bot:
                                bot_adapters[agent_id] = bot
                                success = True
                        except Exception as e:
                            if debug:
                                print(f"[DEBUG] Worker error registering bot: {str(e)}")
                    
                    remote.send(success)
                    
                elif cmd == 'unregister_bot':
                    # Extract data
                    env_idx, agent_id = data
                    
                    # Remove bot
                    success = False
                    if agent_id in bot_adapters:
                        try:
                            if hasattr(bot_adapters[agent_id], 'stop'):
                                bot_adapters[agent_id].stop()
                            del bot_adapters[agent_id]
                            success = True
                        except Exception as e:
                            if debug:
                                print(f"[DEBUG] Worker error unregistering bot: {str(e)}")
                    
                    remote.send(success)
                    
                elif cmd == 'step':
                    actions = data
                    # Inject bot actions before stepping
                    actions.update(bot_actions)
                    # Get bot actions first if we have any bots
                    try:
                        state = env.transition_engine.get_state()
                        for agent_id, bot in bot_adapters.items():
                            if agent_id not in actions:  # Don't override provided actions
                                actions[agent_id] = bot.get_action(state)
                    except Exception as e:
                        if debug:
                            print(f"[DEBUG] Error getting bot actions: {str(e)}")
                    
                    # Format actions for RLGym API
                    formatted_actions = {}
                    for agent_id, action in actions.items():
                        if isinstance(action, np.ndarray):
                            formatted_actions[agent_id] = action
                        else:
                            formatted_actions[agent_id] = np.array([action if isinstance(action, int) else int(action)])
                        
                        # Add action to stacker history
                        if action_stacker is not None:
                            action_stacker.add_action(agent_id, formatted_actions[agent_id])
                    
                    # Step the environment
                    next_obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(formatted_actions)
                    remote.send((next_obs_dict, reward_dict, terminated_dict, truncated_dict))
                    
                elif cmd == 'get_state_for_bots':
                    # Get current state and send back to main process
                    state = env.transition_engine.get_state()
                    remote.send(state)
                    
                elif cmd == 'inject_bot_actions':
                    # Save bot actions for next step
                    bot_actions = data
                    
                elif cmd == 'reset':
                    if debug:
                        print(f"[DEBUG] Worker resetting with stage: {curriculum_config['stage_name'] if curriculum_config else 'Default'}")
                    
                    reset_success = False
                    for reset_attempt in range(3):
                        try:
                            obs = env.reset()
                            reset_success = True
                            remote.send(obs)
                            if debug:
                                print(f"[DEBUG] Reset successful for {curriculum_config['stage_name'] if curriculum_config else 'Default'}")
                            break
                        except Exception as e:
                            if debug:
                                print(f"[DEBUG] Reset attempt {reset_attempt + 1} failed: {e}")
                            if reset_attempt == 2:
                                raise
                            time.sleep(0.1)
                    
                    if not reset_success:
                        remote.send({})
                
                elif cmd == 'close':
                    # Clean up bots first
                    for bot in bot_adapters.values():
                        try:
                            if hasattr(bot, 'stop'):
                                bot.stop()
                        except Exception:
                            pass
                    
                    # Clean up environment and renderer
                    if renderer:
                        renderer.close()
                    env.close()
                    remote.close()
                    break
                    
                elif cmd == 'reset_action_stacker':
                    agent_id = data
                    if action_stacker is not None:
                        action_stacker.reset_agent(agent_id)
                    remote.send(True)
                
                elif cmd == 'get_state':
                    # Get current state and send back to main process
                    try:
                        state = env.transition_engine.get_state()
                        remote.send(state)
                    except Exception as e:
                        if debug:
                            print(f"[DEBUG] Error getting state: {str(e)}")
                        remote.send(None)  # Send None on error
                
                elif cmd == 'prepare_bot_actions':
                    # Store bot actions for next step
                    try:
                        bot_actions = data
                    except Exception as e:
                        if debug:
                            print(f"[DEBUG] Error storing bot actions: {str(e)}")
                        bot_actions = {}
                    
            except EOFError:
                break
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Error in worker process: {str(e)}")
                break
                
    except Exception as e:
        if debug:
            print(f"[DEBUG] Fatal error in worker initialization: {str(e)}")
        raise