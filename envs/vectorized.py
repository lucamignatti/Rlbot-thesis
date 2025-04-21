import concurrent.futures
import time
import multiprocessing as mp
from multiprocessing import Process, Pipe
import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple, List # Added List
from rlgym.rocket_league.rlviser import RLViserRenderer
from .factory import get_env
from rlgym.rocket_league.done_conditions import TimeoutCondition
import select
import traceback # Import traceback for better error logging

# Sentinel object for signaling worker errors
class WorkerError:
    def __init__(self, message, tb):
        self.message = message
        self.traceback = tb
    def __str__(self):
        return f"WorkerError: {self.message}\nTraceback:\n{self.traceback}"


def worker(remote, env_fn, render: bool, action_stacker=None, initial_curriculum_config=None, debug=False):
    """
    Worker process managing two environments (active and standby) for zero-latency resets.
    """
    active_env = None
    standby_env = None
    active_renderer = None # Only used if render=True, attached to active_env
    standby_obs = None
    current_config = initial_curriculum_config
    action_space_cache: Dict[Any, np.ndarray] = {} # Cache action spaces per agent_id

    def get_dummy_action(agent_id, env_instance):
        """Returns a default zero action based on the agent's action space."""
        # Use env_instance to get action space, as it might differ between active/standby briefly during config changes
        if agent_id not in action_space_cache:
            try:
                # Check if env_instance and action_space exist and are dict-like
                if env_instance and hasattr(env_instance, 'action_space') and isinstance(env_instance.action_space, dict):
                    space = env_instance.action_space.get(agent_id) # Use .get for safety
                    if space:
                        action_space_cache[agent_id] = np.zeros(space.shape, dtype=space.dtype)
                    else:
                         if debug: print(f"[DEBUG] Worker: Agent ID {agent_id} not found in action space. Using default [0].")
                         action_space_cache[agent_id] = np.array([0])
                else:
                     if debug: print(f"[DEBUG] Worker: Invalid env_instance or action_space. Using default [0] for agent {agent_id}.")
                     action_space_cache[agent_id] = np.array([0])
            except Exception as e:
                 if debug:
                     print(f"[DEBUG] Worker: Could not get action space for agent {agent_id} ({e}). Using default [0].")
                 action_space_cache[agent_id] = np.array([0]) # Default fallback
        # Return a copy to prevent modification issues if stacker modifies actions
        return np.copy(action_space_cache.get(agent_id, np.array([0])))


    def _create_env(config, use_renderer=False):
        """Helper to create a single environment instance."""
        env = None
        created_renderer = None
        max_retries = 3
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Create renderer *first* if needed, pass it during env creation
                if use_renderer:
                    created_renderer = RLViserRenderer()

                env = env_fn(renderer=created_renderer, action_stacker=action_stacker, curriculum_config=config, debug=debug)

                # Initialize tick count
                if hasattr(env, 'transition_engine'):
                    env.transition_engine._tick_count = 0
                # Initialize TimeoutCondition
                if hasattr(env, 'truncation_cond'):
                    def set_initial_tick(cond):
                        if isinstance(cond, TimeoutCondition): cond.initial_tick = 0
                        elif hasattr(cond, 'conditions'):
                            for sub_cond in cond.conditions: set_initial_tick(sub_cond)
                    set_initial_tick(env.truncation_cond)

                return env, created_renderer # Return env and potentially created renderer
            except Exception as e:
                last_exception = e
                if created_renderer: # Clean up renderer if env creation failed
                    try: created_renderer.close()
                    except: pass
                    created_renderer = None
                if attempt < max_retries - 1:
                    if debug: print(f"[DEBUG] Worker env creation attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(1)
                else:
                    if debug: print(f"[DEBUG] Worker env creation failed after {max_retries} attempts.")
                    # Don't raise here, return None, None to signal failure
                    return None, None
                    # raise RuntimeError(f"Failed to create environment after {max_retries} retries: {last_exception}") from last_exception

    def _safe_reset(env_instance) -> Optional[Dict[Any, Any]]:
        """Helper to reset an environment with retries. Returns None on failure."""
        max_retries = 3
        last_exception = None
        for attempt in range(max_retries):
            try:
                if not env_instance: return None # Skip if env is None
                obs = env_instance.reset()
                action_space_cache.clear() # Clear cache as agents might change
                return obs
            except Exception as e:
                last_exception = e
                if debug: print(f"[DEBUG] Worker reset attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print(f"WORKER ERROR: Failed to reset environment after {max_retries} retries: {last_exception}")
                    return None # Return None on failure
                time.sleep(0.1)
        return None # Should not be reached

    def _safe_close(env_instance, renderer_instance=None):
        """Helper to close env and its renderer."""
        # Close renderer first if it's separate
        if renderer_instance and (not env_instance or renderer_instance != getattr(env_instance, 'renderer', None)):
             try:
                 if debug: print("[DEBUG] Worker closing separate renderer instance.")
                 renderer_instance.close()
             except Exception as e:
                 if debug: print(f"[DEBUG] Worker error closing separate renderer: {e}")

        # Close environment (which might close its attached renderer)
        if env_instance:
            try:
                if debug: print("[DEBUG] Worker closing environment instance.")
                env_instance.close()
            except Exception as e:
                if debug: print(f"[DEBUG] Worker error closing environment instance: {e}")


    try:
        # --- Initial Setup ---
        if debug: print(f"[DEBUG] Worker initializing with config: {current_config.get('stage_name', 'Default') if current_config else 'Default'}")

        # Create active env (potentially with renderer)
        active_env, active_renderer = _create_env(current_config, use_renderer=render)
        if active_env is None: raise RuntimeError("Failed to create initial active_env")
        initial_obs_active = _safe_reset(active_env)
        if initial_obs_active is None: raise RuntimeError("Failed to reset initial active_env")


        # Create standby env (no renderer)
        standby_env, _ = _create_env(current_config, use_renderer=False)
        if standby_env is None: raise RuntimeError("Failed to create initial standby_env")
        standby_obs = _safe_reset(standby_env) # Reset standby and store its obs
        if standby_obs is None: raise RuntimeError("Failed to reset initial standby_env")


        # Perform dummy step on active env
        if initial_obs_active:
            dummy_actions = {agent_id: get_dummy_action(agent_id, active_env) for agent_id in initial_obs_active.keys()}
            if debug: print(f"[DEBUG] Worker performing initial dummy step on active_env.")
            _ = active_env.step(dummy_actions)
        else:
             # This case should ideally not happen if reset succeeded
             if debug: print("[DEBUG] Worker WARNING: initial active_env reset returned empty/None obs, skipping dummy step.")


        # Send the initial observation from the *active* env back to main process
        remote.send(initial_obs_active)
        if debug: print("[DEBUG] Worker initialization complete, sent initial active obs.")

        # --- Main Command Loop ---
        while True:
            try:
                cmd, data = remote.recv()

                if cmd == 'step':
                    if active_env is None: # Check if env is usable
                         raise RuntimeError("Worker received 'step' command but active_env is None.")

                    actions_dict = data
                    formatted_actions = {}
                    for agent_id, action in actions_dict.items():
                        if isinstance(action, torch.Tensor):
                            action_np = action.cpu().detach().numpy()
                            if len(action_np.shape) == 1 and action_np.shape[0] > 1:
                                formatted_actions[agent_id] = np.array([np.argmax(action_np)])
                            else:
                                formatted_actions[agent_id] = action_np
                        elif isinstance(action, np.ndarray):
                            formatted_actions[agent_id] = action
                        elif isinstance(action, (int, float)):
                             formatted_actions[agent_id] = np.array([action])
                        else:
                            try:
                                formatted_actions[agent_id] = np.array([action])
                            except:
                                if debug: print(f"[DEBUG] Worker step: Unable to process action type {type(action)}, using default.")
                                formatted_actions[agent_id] = get_dummy_action(agent_id, active_env)

                        if action_stacker is not None:
                            # Ensure stacker handles potential numpy arrays correctly
                            action_to_stack = formatted_actions[agent_id]
                            action_stacker.add_action(agent_id, action_to_stack)


                    # Step the active environment
                    next_obs_dict, reward_dict, terminated_dict, truncated_dict = active_env.step(formatted_actions)

                    # Render if needed (only active_env has renderer)
                    if render and active_renderer:
                        try:
                            # Ensure renderer is still attached (might be None if creation failed)
                            if hasattr(active_env, 'renderer') and active_env.renderer:
                                active_env.render()
                        except Exception as e:
                             if debug: print(f"[DEBUG] Worker error during render: {e}")


                    remote.send((next_obs_dict, reward_dict, terminated_dict, truncated_dict))

                elif cmd == 'reset':
                    if debug: print(f"[DEBUG] Worker received reset command.")

                    # Check if standby is ready
                    if standby_env is None or standby_obs is None:
                         # This is bad, means previous standby reset failed or env creation failed.
                         # Signal error back to main process.
                         raise RuntimeError("Worker received 'reset' but standby_env/standby_obs is not ready.")

                    # 1. Send the observation from the standby environment immediately
                    remote.send(standby_obs)
                    if debug: print(f"[DEBUG] Worker sent standby obs: {list(standby_obs.keys()) if standby_obs else 'None'}")

                    # 2. Swap roles
                    active_env, standby_env = standby_env, active_env
                    # Swap renderers if rendering is enabled
                    if render:
                        # Detach renderer from new standby (old active)
                        if hasattr(standby_env, 'renderer') and standby_env.renderer:
                            standby_env.renderer = None
                        # Attach renderer to new active (old standby)
                        if active_renderer: # Check if renderer exists
                             active_env.renderer = active_renderer

                    if debug: print(f"[DEBUG] Worker swapped active/standby roles.")

                    # 3. Perform dummy step on the *new* active environment
                    # Use the already available standby_obs to know agents
                    if standby_obs:
                        try:
                            dummy_actions = {agent_id: get_dummy_action(agent_id, active_env) for agent_id in standby_obs.keys()}
                            if debug: print(f"[DEBUG] Worker performing dummy step on new active_env.")
                            _ = active_env.step(dummy_actions)
                        except Exception as e:
                             # If dummy step fails, the env might be broken. The next 'step' will likely fail.
                             # Log error, but proceed with resetting standby.
                             tb_str = traceback.format_exc()
                             print(f"WORKER ERROR during dummy step on new active_env: {e}\n{tb_str}")
                    else:
                         # This case means the standby_obs was empty/None, which shouldn't happen if check passed.
                         if debug: print("[DEBUG] Worker WARNING: skipping dummy step on new active_env as standby_obs was empty/None.")


                    # 4. Reset the *new* standby environment (old active) and store its observation
                    # Set standby_obs to None *before* reset attempt
                    standby_obs = None
                    try:
                        if debug: print(f"[DEBUG] Worker resetting new standby_env.")
                        standby_obs = _safe_reset(standby_env) # Returns None on failure
                        if standby_obs is not None:
                             if debug: print(f"[DEBUG] Worker new standby_env reset complete. Stored obs keys: {list(standby_obs.keys()) if standby_obs else 'None'}")
                        else:
                             # Reset failed, standby_obs remains None. Error already printed by _safe_reset.
                             # The next 'reset' command will raise an error.
                             pass
                    except Exception as e:
                        # Catch any unexpected error from _safe_reset itself (shouldn't happen)
                        standby_obs = None
                        tb_str = traceback.format_exc()
                        print(f"CRITICAL WORKER ERROR: Unexpected exception during standby reset call: {e}\n{tb_str}")


                elif cmd == 'set_curriculum':
                    new_config = data
                    old_stage = current_config.get('stage_name', 'Unknown') if current_config else 'Default'
                    new_stage = new_config.get('stage_name', 'Unknown') if new_config else 'Default'
                    if debug: print(f"[DEBUG] Worker received set_curriculum. From {old_stage} to {new_stage}")

                    current_config = new_config # Store new config

                    # Close existing environments and renderer
                    _safe_close(standby_env)
                    standby_env = None
                    standby_obs = None # Reset standby obs
                    _safe_close(active_env, active_renderer)
                    active_env = None
                    active_renderer = None
                    action_space_cache.clear() # Clear action space cache

                    # Recreate both environments with the new config
                    ack = False
                    try:
                        # Create active env (potentially with renderer)
                        active_env, active_renderer = _create_env(current_config, use_renderer=render)
                        if active_env is None: raise RuntimeError("Failed to recreate active_env after curriculum change")
                        initial_obs_active = _safe_reset(active_env)
                        if initial_obs_active is None: raise RuntimeError("Failed to reset recreated active_env")

                        # Create standby env (no renderer)
                        standby_env, _ = _create_env(current_config, use_renderer=False)
                        if standby_env is None: raise RuntimeError("Failed to recreate standby_env after curriculum change")
                        standby_obs = _safe_reset(standby_env) # Reset standby and store its obs
                        if standby_obs is None: raise RuntimeError("Failed to reset recreated standby_env")

                        # Perform dummy step on active env
                        if initial_obs_active:
                            dummy_actions = {agent_id: get_dummy_action(agent_id, active_env) for agent_id in initial_obs_active.keys()}
                            if debug: print(f"[DEBUG] Worker performing dummy step on active_env after curriculum change.")
                            _ = active_env.step(dummy_actions)

                        ack = True # Signal success
                        if debug: print(f"[DEBUG] Worker curriculum change complete.")

                    except Exception as e:
                         tb_str = traceback.format_exc()
                         print(f"CRITICAL WORKER ERROR during set_curriculum recreation: {e}\n{tb_str}")
                         # Ensure envs are None if creation failed
                         _safe_close(standby_env) # Close potentially partially created envs
                         _safe_close(active_env, active_renderer)
                         active_env = None
                         standby_env = None
                         active_renderer = None
                         standby_obs = None
                         ack = False # Signal failure
                    finally:
                         # Send acknowledgment (True/False) back to main process
                         try:
                             remote.send(ack)
                         except (BrokenPipeError, EOFError):
                              if debug: print("[DEBUG] Worker pipe broken sending curriculum ack.")
                         except Exception as e_ack:
                              if debug: print(f"[DEBUG] Worker error sending curriculum ack: {e_ack}")


                elif cmd == 'close':
                    if debug: print("[DEBUG] Worker received close command.")
                    break # Exit loop, finally block will handle cleanup

                elif cmd == 'reset_action_stacker':
                    agent_id = data
                    if action_stacker is not None:
                        action_stacker.reset_agent(agent_id)
                    # No need to send ack for this simple command unless required by main loop
                    # remote.send(True)

            except EOFError:
                if debug: print("[DEBUG] Worker received EOFError, exiting.")
                break
            except (BrokenPipeError, ConnectionResetError) as e:
                 if debug: print(f"[DEBUG] Worker pipe broken: {e}, exiting.")
                 break
            except Exception as e:
                # Log unexpected errors in command processing
                tb_str = traceback.format_exc()
                print(f"Error in worker command loop (cmd={cmd}): {e}\n{tb_str}")
                # Attempt to send error back to main process
                try:
                    # Ensure remote is valid before sending
                    if remote and not remote.closed:
                         remote.send(WorkerError(f"Unhandled exception in worker loop: {e}", tb_str))
                except:
                    pass # Ignore if pipe is broken during error reporting
                break # Exit loop on unhandled exception

    except Exception as e:
        # Catch initialization errors or major issues
        tb_str = traceback.format_exc()
        print(f"Fatal error in worker setup: {e}\n{tb_str}")
        try:
            # Try sending error signal back if remote exists
            if remote and not remote.closed:
                remote.send(WorkerError(f"Worker failed initialization: {e}", tb_str))
        except:
            pass # Ignore pipe errors during error reporting
    finally:
        # Guaranteed cleanup
        if debug: print("[DEBUG] Worker entering finally block for cleanup.")
        _safe_close(standby_env)
        # Pass active_renderer explicitly as it might be detached from active_env
        _safe_close(active_env, active_renderer)
        try:
            if remote and not remote.closed:
                if debug: print("[DEBUG] Worker closing remote connection.")
                remote.close()
        except Exception as e:
            if debug: print(f"[DEBUG] Worker error closing remote: {e}")
        if debug: print("[DEBUG] Worker cleanup finished.")


class VectorizedEnv:
    """
    Runs multiple RLGym environments in parallel using worker processes
    that manage active/standby environments for zero-latency resets.
    """
    def __init__(self, num_envs, render=False, action_stacker=None, curriculum_manager=None, debug=False):
        self.num_envs = num_envs
        # Render mode uses multiprocessing but only one worker creates a renderer instance.
        self.render = render
        self.action_stacker = action_stacker
        self.curriculum_manager = curriculum_manager
        self.debug = debug
        self.mode = "multiprocess" # Always use multiprocessing now

        # For tracking episode metrics for curriculum
        self.episode_rewards = [{} for _ in range(num_envs)]
        self.episode_successes = [False] * num_envs
        self.episode_timeouts = [False] * num_envs
        # Store stats for the *completed* episode before they are reset
        self._last_episode_rewards = [{} for _ in range(num_envs)]
        self._last_episode_success = [False] * num_envs
        self._last_episode_timeout = [False] * num_envs


        # Get initial curriculum configurations
        self.curriculum_configs = []
        for env_idx in range(num_envs):
            config = None
            if self.curriculum_manager:
                config = self.curriculum_manager.get_environment_config()
                if self.debug:
                    print(f"[DEBUG] Env {env_idx} initializing with stage: {config.get('stage_name', 'Default')}")
                config = self._make_config_picklable(config)
            self.curriculum_configs.append(config)

        # Create communication pipes
        pipes = [Pipe(duplex=True) for _ in range(num_envs)] # Ensure duplex=True
        self.remotes = [remote for remote, _ in pipes]
        self.work_remotes = [work_remote for _, work_remote in pipes]

        # Create and start worker processes
        self.processes = []
        for idx, work_remote in enumerate(self.work_remotes):
            # Worker 0 handles rendering if self.render is True
            worker_render = self.render and (idx == 0)
            process = Process(
                target=worker,
                args=(work_remote, get_env, worker_render, action_stacker, self.curriculum_configs[idx], self.debug),
                daemon=True
            )
            process.start()
            self.processes.append(process)
            work_remote.close() # Close worker end in main process

        # Initialize dones list *before* checking for failed workers
        self.dones = [False] * self.num_envs

        # Initialize: Wait for the initial observation from each worker's active_env
        self.obs_dicts = [{}] * num_envs
        failed_workers = []
        initialization_timeout = 60.0 # Increased timeout for creating/resetting two envs

        remotes_waiting_init = list(enumerate(self.remotes))
        start_time = time.time()

        while remotes_waiting_init and time.time() - start_time < initialization_timeout:
            ready_remotes_list = []
            try:
                # Filter out potentially closed remotes before select
                valid_remotes_to_check = [r for _, r in remotes_waiting_init if r and not r.closed]
                if not valid_remotes_to_check: break # Exit if no valid remotes left

                if hasattr(select, 'select'):
                    ready_remotes_list, _, _ = select.select(valid_remotes_to_check, [], [], 1.0)
                else:
                    ready_remotes_list = [r for r in valid_remotes_to_check if r.poll()]
                    if not ready_remotes_list: time.sleep(0.1)
            except ValueError as e:
                 # Can happen if a pipe is closed between filtering and select
                 if self.debug: print(f"[DEBUG] Select error during init (likely closed pipe): {e}")
                 # Rebuild remotes_waiting_init by checking individually
                 current_waiting = []
                 for idx, r in remotes_waiting_init:
                     try:
                         if r and not r.closed:
                             current_waiting.append((idx, r))
                         elif idx not in failed_workers:
                              if self.debug: print(f"[DEBUG] Worker {idx} remote closed during init wait.")
                              failed_workers.append(idx)
                     except Exception:
                          if idx not in failed_workers:
                              if self.debug: print(f"[DEBUG] Worker {idx} remote error during init wait check.")
                              failed_workers.append(idx)
                 remotes_waiting_init = current_waiting
                 continue # Retry select/poll


            ready_remotes_set = set(ready_remotes_list) # Faster lookup
            next_remotes_waiting = []

            for env_idx, remote in remotes_waiting_init:
                 # Double check remote validity before using
                 try:
                     if not remote or remote.closed:
                         if env_idx not in failed_workers:
                             if self.debug: print(f"[DEBUG] Worker {env_idx} remote closed before processing init response.")
                             failed_workers.append(env_idx)
                         continue # Skip this remote
                 except Exception:
                      if env_idx not in failed_workers:
                          if self.debug: print(f"[DEBUG] Worker {env_idx} remote error checking validity.")
                          failed_workers.append(env_idx)
                      continue


                 if remote in ready_remotes_set:
                    try:
                        initial_obs = remote.recv()
                        if isinstance(initial_obs, WorkerError):
                            print(f"Worker {env_idx} failed initialization:\n{initial_obs}")
                            failed_workers.append(env_idx)
                        elif initial_obs is None: # Should not happen, but check
                             print(f"Worker {env_idx} initialization returned None observation.")
                             failed_workers.append(env_idx)
                        else:
                            self.obs_dicts[env_idx] = initial_obs
                            if self.debug: print(f"[DEBUG] Received initial obs from worker {env_idx}.")
                    except (BrokenPipeError, EOFError, ConnectionResetError) as e:
                        if self.debug: print(f"[DEBUG] Worker {env_idx} pipe broken during initial obs recv: {e}")
                        if env_idx not in failed_workers: failed_workers.append(env_idx)
                    except Exception as e:
                        if self.debug: print(f"[DEBUG] Error receiving initial obs from worker {env_idx}: {e}")
                        if env_idx not in failed_workers: failed_workers.append(env_idx)
                 else:
                    next_remotes_waiting.append((env_idx, remote)) # Keep waiting

            remotes_waiting_init = next_remotes_waiting

        # Handle workers that timed out
        for env_idx, remote in remotes_waiting_init:
            if env_idx not in failed_workers:
                if self.debug: print(f"[DEBUG] Worker {env_idx} timed out during initialization.")
                failed_workers.append(env_idx)

        # Terminate failed workers and update dones list
        unique_failed_workers = sorted(list(set(failed_workers)))
        for env_idx in unique_failed_workers:
             try:
                 # Terminate process
                 if env_idx < len(self.processes) and self.processes[env_idx] and self.processes[env_idx].is_alive():
                     if self.debug: print(f"[DEBUG] Terminating failed worker {env_idx}.")
                     self.processes[env_idx].terminate()
                     self.processes[env_idx].join(timeout=1.0)
                 # Close remote connection
                 if env_idx < len(self.remotes) and self.remotes[env_idx] and not self.remotes[env_idx].closed:
                     self.remotes[env_idx].close()
             except Exception as e:
                 if self.debug: print(f"[DEBUG] Error terminating failed worker {env_idx}: {e}")
             # Mark obs as empty for failed workers
             self.obs_dicts[env_idx] = {}
             # Mark as done so they don't receive step commands
             self.dones[env_idx] = True


        # Note: Recreation logic removed for simplicity. If a worker fails init, it stays failed.

        # Common initialization
        self.episode_counts = [0] * num_envs

        if unique_failed_workers:
             print(f"WARNING: {len(unique_failed_workers)} workers failed to initialize or errored during init: {unique_failed_workers}")


    def _make_config_picklable(self, config):
        """Ensure config is picklable. Returns a new dict."""
        if config is None:
            return {"required_agents": 1}

        processed = {}
        for key, value in config.items():
            try:
                # Basic check: try pickling/unpickling individually? Too slow.
                # Assume most basic types are fine. Focus on known problematic types.
                if key in ["state_mutator", "reward_function", "terminal_conditions", "obs_builder"]:
                     # If it has 'mutators', try the MutatorSequence fix
                     if key == "state_mutator" and hasattr(value, "mutators"):
                         try:
                             from rlgym.rocket_league.state_mutators import MutatorSequence
                             if not isinstance(value, MutatorSequence):
                                 mutators = value.mutators
                                 # Ensure mutators is a list or tuple before unpacking
                                 if isinstance(mutators, (list, tuple)):
                                     processed[key] = MutatorSequence(*mutators)
                                     if self.debug: print(f"[DEBUG] Pickling fix: Recreated MutatorSequence for {key}")
                                 else:
                                      if self.debug: print(f"[DEBUG] Pickling fix: 'mutators' attribute for {key} is not list/tuple. Keeping original.")
                                      processed[key] = value
                             else:
                                 processed[key] = value # Already a sequence, assume ok
                         except Exception as e:
                             if self.debug: print(f"[DEBUG] Pickling fix: Failed to recreate MutatorSequence for {key}: {e}. Keeping original.")
                             processed[key] = value # Keep original if fix fails
                     else:
                         # For other complex objects, just pass them through.
                         processed[key] = value
                else:
                    processed[key] = value
            except Exception as e:
                 if self.debug: print(f"[DEBUG] Error processing config key '{key}' for pickling: {e}")
                 processed[key] = None # Or skip the key

        # Force required_agents field
        if "required_agents" not in processed:
            required_agents = 1 # Default
            state_mutator = processed.get("state_mutator")
            if state_mutator:
                try:
                    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator
                    if hasattr(state_mutator, "mutators") and isinstance(state_mutator.mutators, (list, tuple)):
                         team_mutators = [m for m in state_mutator.mutators if isinstance(m, FixedTeamSizeMutator)]
                         if team_mutators:
                             required_agents = team_mutators[0].blue_size + team_mutators[0].orange_size
                except Exception as e:
                     if self.debug: print(f"[DEBUG] Error determining required_agents from state_mutator: {e}")
            processed["required_agents"] = required_agents
            # if self.debug and required_agents == 1 and state_mutator:
            #      print(f"[DEBUG] Setting required_agents=1 for config stage {processed.get('stage_name', 'Unknown')}")


        return processed


    def step(self, actions_dict_list: List[Dict[Any, Any]]) -> Tuple[List[Tuple], List[bool], List[int]]:
        """
        Step the environments. Handles communication with workers and zero-latency resets.
        """
        if len(actions_dict_list) != self.num_envs:
             raise ValueError(f"actions_dict_list length ({len(actions_dict_list)}) must match num_envs ({self.num_envs})")

        step_results = [None] * self.num_envs # Stores (next_obs, rew, term, trunc) or WorkerError
        reset_obs_received = [None] * self.num_envs # Stores obs received from 'reset' command

        # --- Send Commands ---
        active_remotes_step = {} # remote -> index
        active_remotes_reset = {} # remote -> index
        active_remotes_curriculum = {} # remote -> index (waiting for curriculum ack)

        for i in range(self.num_envs):
            remote = self.remotes[i]
            # Check if process is alive and remote is valid
            process_alive = i < len(self.processes) and self.processes[i] and self.processes[i].is_alive()
            remote_valid = remote and not remote.closed

            if not process_alive or not remote_valid:
                if not self.dones[i]: # If it wasn't already marked done, mark it now
                     if self.debug: print(f"[DEBUG] Env {i} process/remote is dead/closed before command send. Marking as done.")
                     self.dones[i] = True
                     step_results[i] = ({}, {}, {True: True}, {True: True}) # Dummy step result
                continue # Skip dead/closed workers

            if self.dones[i]:
                # --- Handle Done Environments: Curriculum Update? -> Reset ---
                curriculum_update_sent = False
                try:
                    # 1. Check/Update Curriculum Manager (using stats from the episode that just finished)
                    if self.curriculum_manager:
                        # Use the stored stats from the *last* completed episode
                        avg_reward = sum(self._last_episode_rewards[i].values()) / max(len(self._last_episode_rewards[i]), 1) if self._last_episode_rewards[i] else 0.0
                        metrics = { "success": self._last_episode_success[i], "timeout": self._last_episode_timeout[i], "episode_reward": avg_reward }

                        if type(self.curriculum_manager).__name__ == "ManualCurriculumManager":
                             self.curriculum_manager.update_progression_stats(episode_rewards=avg_reward, success=metrics["success"], timeout=metrics["timeout"], env_id=i)
                        else:
                             self.curriculum_manager.update_progression_stats(metrics)

                        new_config = self.curriculum_manager.get_environment_config()
                        new_config = self._make_config_picklable(new_config)

                        # Check if config actually changed before sending update
                        if new_config != self.curriculum_configs[i]:
                            if self.debug: print(f"[DEBUG] Env {i} curriculum changed. Sending update to worker.")
                            self.curriculum_configs[i] = new_config
                            remote.send(('set_curriculum', new_config))
                            active_remotes_curriculum[remote] = i # Expect an ack
                            curriculum_update_sent = True
                        # else: curriculum didn't change, proceed to reset

                    # 2. Send Reset Command (if no curriculum update was sent, or after ack is received)
                    if not curriculum_update_sent:
                        remote.send(('reset', None))
                        active_remotes_reset[remote] = i
                        if self.debug: print(f"[DEBUG] Sent reset to worker {i}")

                except (BrokenPipeError, EOFError) as e:
                    if self.debug: print(f"[DEBUG] Worker {i} pipe broken sending curriculum/reset: {e}. Keeping done flag.")
                    self.dones[i] = True # Keep done
                    step_results[i] = ({}, {}, {True: True}, {True: True})
                except Exception as e:
                    if self.debug: print(f"[DEBUG] Error sending curriculum/reset to worker {i}: {e}. Keeping done flag.")
                    self.dones[i] = True # Keep done
                    step_results[i] = ({}, {}, {True: True}, {True: True})

            else:
                # --- Handle Active Environments: Send Step ---
                actions = actions_dict_list[i]
                if not actions:
                     # Treat empty actions as a signal to end the episode? Or error?
                     # For now, let's send a reset command instead of step.
                     if self.debug: print(f"[DEBUG] Env {i} received empty actions dict. Sending reset.")
                     try:
                         # Store current stats before reset
                         self._last_episode_rewards[i] = self.episode_rewards[i].copy()
                         self._last_episode_success[i] = False # Assume not success if actions stopped
                         self._last_episode_timeout[i] = True # Assume timeout if actions stopped
                         # Now check curriculum and send reset (similar to done case)
                         # (Duplication, consider refactoring later)
                         curriculum_update_sent = False
                         if self.curriculum_manager:
                             avg_reward = sum(self._last_episode_rewards[i].values()) / max(len(self._last_episode_rewards[i]), 1) if self._last_episode_rewards[i] else 0.0
                             metrics = { "success": self._last_episode_success[i], "timeout": self._last_episode_timeout[i], "episode_reward": avg_reward }
                             # Update manager (simplified)
                             if type(self.curriculum_manager).__name__ == "ManualCurriculumManager":
                                 self.curriculum_manager.update_progression_stats(episode_rewards=avg_reward, success=metrics["success"], timeout=metrics["timeout"], env_id=i)
                             else:
                                 self.curriculum_manager.update_progression_stats(metrics)

                             new_config = self.curriculum_manager.get_environment_config()
                             new_config = self._make_config_picklable(new_config)
                             if new_config != self.curriculum_configs[i]:
                                 self.curriculum_configs[i] = new_config
                                 remote.send(('set_curriculum', new_config))
                                 active_remotes_curriculum[remote] = i
                                 curriculum_update_sent = True

                         if not curriculum_update_sent:
                             remote.send(('reset', None))
                             active_remotes_reset[remote] = i
                             if self.debug: print(f"[DEBUG] Sent reset to worker {i} (due to empty actions)")

                     except Exception as e:
                         if self.debug: print(f"[DEBUG] Error sending reset (due to empty actions) to worker {i}: {e}. Marking done.")
                         self.dones[i] = True
                         step_results[i] = ({}, {}, {True: True}, {True: True})
                else:
                    # Send Step Command
                    try:
                        remote.send(('step', actions))
                        active_remotes_step[remote] = i
                    except (BrokenPipeError, EOFError) as e:
                        if self.debug: print(f"[DEBUG] Worker {i} pipe broken sending step: {e}. Marking done.")
                        self.dones[i] = True
                        step_results[i] = ({}, {}, {True: True}, {True: True})
                    except Exception as e:
                        if self.debug: print(f"[DEBUG] Error sending step to worker {i}: {e}. Marking done.")
                        self.dones[i] = True
                        step_results[i] = ({}, {}, {True: True}, {True: True})

        # --- Receive Results ---
        expected_responses = len(active_remotes_step) + len(active_remotes_reset) + len(active_remotes_curriculum)
        received_responses = 0
        receive_timeout = 30.0 # Timeout for receiving responses
        start_time = time.time()

        # Combine all remotes we expect a response from
        all_waiting_remotes = list(active_remotes_step.keys()) + list(active_remotes_reset.keys()) + list(active_remotes_curriculum.keys())
        # Use a set for faster removal checks
        waiting_remotes_set = set(all_waiting_remotes)

        while received_responses < expected_responses and time.time() - start_time < receive_timeout:
            ready_remotes_list = []
            if not waiting_remotes_set: break # Exit if no more responses expected

            try:
                 # Ensure we only select valid remotes
                 valid_remotes_to_select = [r for r in waiting_remotes_set if r and not r.closed]
                 if not valid_remotes_to_select:
                      # If no valid remotes left, check if any were expected
                      if waiting_remotes_set:
                           if self.debug: print("[DEBUG] All remaining waiting remotes are closed/invalid.")
                           # Mark corresponding envs as done/error
                           for r in list(waiting_remotes_set): # Iterate copy
                                env_idx = active_remotes_step.get(r, active_remotes_reset.get(r, active_remotes_curriculum.get(r, -1)))
                                if env_idx != -1 and not self.dones[env_idx]:
                                     self.dones[env_idx] = True
                                     step_results[env_idx] = ({}, {}, {True: True}, {True: True})
                                waiting_remotes_set.remove(r)
                                received_responses +=1 # Count as processed (error)
                      break # Exit loop

                 if hasattr(select, 'select'):
                     ready_remotes_list, _, _ = select.select(valid_remotes_to_select, [], [], 0.1) # Short wait
                 else:
                     ready_remotes_list = [r for r in valid_remotes_to_select if r.poll()]
                     if not ready_remotes_list: time.sleep(0.01)

            except ValueError as e:
                 if self.debug: print(f"[DEBUG] Select error during receive (likely closed pipe): {e}")
                 # Re-validate waiting_remotes_set
                 current_waiting = set()
                 for r in list(waiting_remotes_set): # Iterate copy
                     try:
                         if r and not r.closed:
                             current_waiting.add(r)
                         else:
                              env_idx = active_remotes_step.get(r, active_remotes_reset.get(r, active_remotes_curriculum.get(r, -1)))
                              if env_idx != -1 and not self.dones[env_idx]:
                                   self.dones[env_idx] = True
                                   step_results[env_idx] = ({}, {}, {True: True}, {True: True})
                              received_responses += 1 # Count as processed (error)
                     except Exception:
                          env_idx = active_remotes_step.get(r, active_remotes_reset.get(r, active_remotes_curriculum.get(r, -1)))
                          if env_idx != -1 and not self.dones[env_idx]:
                               self.dones[env_idx] = True
                               step_results[env_idx] = ({}, {}, {True: True}, {True: True})
                          received_responses += 1 # Count as processed (error)
                 waiting_remotes_set = current_waiting
                 continue # Retry select/poll

            if not ready_remotes_list:
                continue # No remotes ready, continue waiting

            for remote in ready_remotes_list:
                # Check if remote is still in the set (might have been removed by error handling)
                if remote not in waiting_remotes_set:
                    continue

                try:
                    result = remote.recv()
                    received_responses += 1
                    waiting_remotes_set.remove(remote) # Response received

                    # --- Process Received Result ---
                    if remote in active_remotes_curriculum:
                        env_idx = active_remotes_curriculum[remote]
                        ack = result
                        if ack is True:
                            if self.debug: print(f"[DEBUG] Worker {env_idx} acknowledged curriculum update. Sending reset.")
                            # Now send the reset command
                            try:
                                remote.send(('reset', None))
                                active_remotes_reset[remote] = env_idx # Now expect reset obs
                                waiting_remotes_set.add(remote) # Add back to wait for reset obs
                                expected_responses += 1 # Expect one more response
                            except (BrokenPipeError, EOFError) as e_reset:
                                if self.debug: print(f"[DEBUG] Worker {env_idx} pipe broken sending reset after curriculum ack: {e_reset}. Marking done.")
                                self.dones[env_idx] = True
                                step_results[env_idx] = ({}, {}, {True: True}, {True: True})
                            except Exception as e_reset:
                                if self.debug: print(f"[DEBUG] Error sending reset after curriculum ack to worker {env_idx}: {e_reset}. Marking done.")
                                self.dones[env_idx] = True
                                step_results[env_idx] = ({}, {}, {True: True}, {True: True})
                        else:
                             # Curriculum update failed at worker
                             if self.debug: print(f"[DEBUG] Worker {env_idx} failed curriculum update (ack={ack}). Marking done.")
                             self.dones[env_idx] = True # Keep marked done
                             step_results[env_idx] = ({}, {}, {True: True}, {True: True}) # Dummy result

                    elif remote in active_remotes_step:
                        env_idx = active_remotes_step[remote]
                        if isinstance(result, WorkerError):
                             print(f"Worker {env_idx} reported error during step:\n{result}")
                             self.dones[env_idx] = True
                             step_results[env_idx] = ({}, {}, {True: True}, {True: True})
                        elif isinstance(result, tuple) and len(result) == 4: # Check if it's a valid step tuple
                            next_obs_dict, reward_dict, terminated_dict, truncated_dict = result
                            step_results[env_idx] = result # Store the tuple
                            self.obs_dicts[env_idx] = next_obs_dict # Update current obs

                            # Track rewards
                            if self.curriculum_manager is not None:
                                if not isinstance(self.episode_rewards[env_idx], dict): self.episode_rewards[env_idx] = {}
                                for agent_id, reward in reward_dict.items():
                                    self.episode_rewards[env_idx][agent_id] = self.episode_rewards[env_idx].get(agent_id, 0) + reward

                            # Check for done state
                            is_done = any(terminated_dict.values()) or any(truncated_dict.values())
                            if is_done:
                                self.dones[env_idx] = True # Mark for reset on *next* step
                                self.episode_successes[env_idx] = any(terminated_dict.values())
                                self.episode_timeouts[env_idx] = any(truncated_dict.values()) and not self.episode_successes[env_idx]
                                self.episode_counts[env_idx] += 1
                                # Store stats for curriculum update on next step
                                self._last_episode_rewards[env_idx] = self.episode_rewards[env_idx].copy()
                                self._last_episode_success[env_idx] = self.episode_successes[env_idx]
                                self._last_episode_timeout[env_idx] = self.episode_timeouts[env_idx]
                                if self.debug: print(f"[DEBUG] Env {env_idx} finished episode. Marked done for next step.")
                        else:
                             # Received unexpected data type instead of step tuple
                             print(f"WORKER ERROR: Worker {env_idx} sent unexpected data type during step: {type(result)}. Marking done.")
                             self.dones[env_idx] = True
                             step_results[env_idx] = ({}, {}, {True: True}, {True: True})


                    elif remote in active_remotes_reset:
                        env_idx = active_remotes_reset[remote]
                        if isinstance(result, WorkerError):
                             print(f"Worker {env_idx} reported error during reset:\n{result}")
                             self.dones[env_idx] = True # Keep marked done
                             self.obs_dicts[env_idx] = {}
                             step_results[env_idx] = ({}, {}, {True: True}, {True: True}) # Dummy step result
                        elif isinstance(result, dict) or result is None: # Expecting obs dict (or None on failure)
                            # This is the observation from the worker's standby env
                            reset_obs_received[env_idx] = result if result is not None else {}
                            self.obs_dicts[env_idx] = reset_obs_received[env_idx] # Update main process obs immediately
                            self.dones[env_idx] = False # Mark as ready for next step
                            if self.debug: print(f"[DEBUG] Env {env_idx} received reset obs. Marked ready.")

                            # Reset episode trackers (already done when episode finished)
                            self.episode_rewards[env_idx] = {}
                            self.episode_successes[env_idx] = False
                            self.episode_timeouts[env_idx] = False

                            # Reset action stacker (send command, no ack expected)
                            if self.action_stacker is not None and self.obs_dicts[env_idx]:
                                for agent_id in self.obs_dicts[env_idx].keys():
                                    try:
                                        # Check remote is still valid before sending
                                        if remote and not remote.closed:
                                             remote.send(('reset_action_stacker', agent_id))
                                        else: break # Stop if remote closed
                                    except (BrokenPipeError, EOFError): break # Stop if pipe broken
                                    except Exception: pass # Ignore other errors here
                        else:
                             # Received unexpected data type instead of reset obs dict
                             print(f"WORKER ERROR: Worker {env_idx} sent unexpected data type during reset: {type(result)}. Marking done.")
                             self.dones[env_idx] = True
                             self.obs_dicts[env_idx] = {}
                             step_results[env_idx] = ({}, {}, {True: True}, {True: True})


                except (BrokenPipeError, EOFError, ConnectionResetError) as e:
                    # Handle pipe errors during recv
                    env_idx = active_remotes_step.get(remote, active_remotes_reset.get(remote, active_remotes_curriculum.get(remote, -1)))
                    if env_idx != -1:
                        if self.debug: print(f"[DEBUG] Worker {env_idx} pipe broken during recv: {e}. Marking done.")
                        if not self.dones[env_idx]:
                             self.dones[env_idx] = True
                             step_results[env_idx] = ({}, {}, {True: True}, {True: True}) # Dummy result
                    # Remove from waiting set even on error
                    if remote in waiting_remotes_set:
                         waiting_remotes_set.remove(remote)
                         received_responses += 1 # Count as processed (error)

                except Exception as e:
                    # Handle other unexpected errors during recv
                    env_idx = active_remotes_step.get(remote, active_remotes_reset.get(remote, active_remotes_curriculum.get(remote, -1)))
                    if env_idx != -1:
                        tb_str = traceback.format_exc()
                        print(f"Error receiving result from worker {env_idx}: {e}\n{tb_str}")
                        if not self.dones[env_idx]:
                             self.dones[env_idx] = True
                             step_results[env_idx] = ({}, {}, {True: True}, {True: True})
                    # Remove from waiting set even on error
                    if remote in waiting_remotes_set:
                         waiting_remotes_set.remove(remote)
                         received_responses += 1 # Count as processed (error)


        # --- Handle Timeouts ---
        if time.time() - start_time >= receive_timeout:
            if waiting_remotes_set: # If any remotes didn't respond
                 for remote in list(waiting_remotes_set): # Iterate copy
                    env_idx = active_remotes_step.get(remote, active_remotes_reset.get(remote, active_remotes_curriculum.get(remote, -1)))
                    if env_idx != -1:
                        if self.debug: print(f"[DEBUG] Worker {env_idx} timed out waiting for response. Marking done.")
                        if not self.dones[env_idx]:
                             self.dones[env_idx] = True
                             step_results[env_idx] = ({}, {}, {True: True}, {True: True}) # Dummy result


        # --- Finalize Results ---
        # Ensure step_results has valid entries for all envs
        final_results = []
        for i in range(self.num_envs):
             if step_results[i] is None:
                 # If env was reset, step_results is None, return dummy step data
                 if reset_obs_received[i] is not None:
                     final_results.append(({}, {}, {True: True}, {True: True}))
                 # If env was already done and reset failed/timed out, return dummy
                 elif self.dones[i]:
                     final_results.append(({}, {}, {True: True}, {True: True}))
                 else:
                     # Should not happen if logic is correct (active env timeout handled above)
                     if self.debug: print(f"[WARN] Env {i} has no step result but not marked done/reset. Returning dummy data.")
                     final_results.append(({}, {}, {True: True}, {True: True}))
             elif isinstance(step_results[i], tuple) and len(step_results[i]) == 4:
                 final_results.append(step_results[i]) # Use actual step result
             else: # Should be dummy data already if error occurred
                  final_results.append(({}, {}, {True: True}, {True: True}))


        # Return observations *after* potential resets, step results, dones, counts
        return final_results, self.dones.copy(), self.episode_counts.copy()


    def force_env_reset(self, env_idx):
        """Force reset by marking as done. Recovery handled in step()."""
        if self.debug:
            print(f"[DEBUG] force_env_reset called for env {env_idx}. Marking as done for auto-recovery.")
        if 0 <= env_idx < self.num_envs:
            if not self.dones[env_idx]: # Only store stats if it wasn't already done
                 # Store current stats before marking done
                 self._last_episode_rewards[env_idx] = self.episode_rewards[env_idx].copy()
                 self._last_episode_success[env_idx] = False # Assume failure on force reset
                 self._last_episode_timeout[env_idx] = True # Assume timeout on force reset

            self.dones[env_idx] = True
            self.obs_dicts[env_idx] = {} # Clear current obs


    def close(self):
        """Clean up resources properly"""
        if self.debug: print("[DEBUG] Closing VectorizedEnv...")

        # Send close command to all workers
        if hasattr(self, 'remotes'):
            for i, remote in enumerate(self.remotes):
                try:
                    # Check if process is alive and remote is valid
                    process_alive = i < len(self.processes) and self.processes[i] and self.processes[i].is_alive()
                    remote_valid = remote and hasattr(remote, 'closed') and not remote.closed
                    if process_alive and remote_valid:
                        remote.send(('close', None))
                        if self.debug: print(f"[DEBUG] Sent close command to worker {i}")
                except (BrokenPipeError, EOFError):
                    if self.debug: print(f"[DEBUG] Worker {i} pipe already closed when sending close command.")
                except Exception as e:
                     if self.debug: print(f"[DEBUG] Unexpected error sending close to worker {i}: {e}")

        # Wait briefly for workers to process
        time.sleep(1.0)

        # Join/Terminate processes
        if hasattr(self, 'processes'):
            active_processes = []
            for i, process in enumerate(self.processes):
                 # Check if process object exists before calling is_alive()
                 if process and process.is_alive():
                     active_processes.append((i, process))

            start_join = time.time()
            join_timeout = 10.0
            while active_processes and time.time() - start_join < join_timeout:
                remaining_processes = []
                for i, process in active_processes:
                    process.join(timeout=0.1)
                    if process.is_alive():
                        remaining_processes.append((i, process))
                    elif self.debug: print(f"[DEBUG] Worker {i} exited gracefully.")
                active_processes = remaining_processes
                if not active_processes: break

            for i, process in active_processes:
                 if process.is_alive():
                     try:
                         if self.debug: print(f"[DEBUG] Worker {i} did not exit gracefully. Terminating...")
                         process.terminate()
                         process.join(timeout=1.0)
                         if process.is_alive(): process.kill(); process.join(timeout=1.0)
                     except Exception as e:
                         if self.debug: print(f"[DEBUG] Error terminating/killing process {i}: {e}")
            self.processes = []
            if self.debug: print("[DEBUG] Worker processes joined/terminated.")

        # Close main process side of pipes
        if hasattr(self, 'remotes'):
            for i, remote in enumerate(self.remotes):
                try:
                    if remote and hasattr(remote, 'close'): remote.close()
                except Exception as e:
                    if self.debug: print(f"[DEBUG] Error closing remote pipe for worker {i}: {e}")
            self.remotes = []
            if self.debug: print("[DEBUG] Remote pipes closed.")

        self.work_remotes = []

        # GC
        try:
            import gc
            gc.collect()
            if self.debug: print("[DEBUG] Garbage collection triggered.")
        except ImportError: pass
        if self.debug: print("[DEBUG] VectorizedEnv closed.")
