import concurrent.futures
import time
import multiprocessing as mp
from multiprocessing import Process, Pipe
import numpy as np
import torch  # Add missing torch import for tensor handling
from typing import Optional  # Keep only what we use
from rlgym.rocket_league.rlviser import RLViserRenderer
from .factory import get_env
from rlgym.rocket_league.done_conditions import TimeoutCondition
import select

def worker(remote, env_fn, render: bool, action_stacker=None, curriculum_config=None, debug=False):
    """Worker process that runs a single environment"""
    env = None
    renderer = None
    try:
        # Create environment first
        if debug:
            print(f"[DEBUG] Worker process creating environment with config: {curriculum_config['stage_name'] if curriculum_config else 'Default'}")

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
                    time.sleep(1)  # Small delay before retry
                else:
                    raise

        if env is None:
            raise RuntimeError("Failed to create environment after max retries")

        # Then create renderer if needed
        renderer = None
        if render:
            try:
                from rlgym.rocket_league.rlviser import RLViserRenderer
                renderer = RLViserRenderer()
                env.renderer = renderer
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Error creating renderer: {e}")

        # Initialize tick count at creation time
        if hasattr(env, 'transition_engine'):
            env.transition_engine._tick_count = 0

        # Initialize initial_tick for TimeoutCondition if it exists
        if hasattr(env, 'truncation_cond'):
            def set_initial_tick(cond):
                if isinstance(cond, TimeoutCondition):
                    cond.initial_tick = 0
                elif hasattr(cond, 'conditions'):
                    for sub_cond in cond.conditions:
                        set_initial_tick(sub_cond)
            set_initial_tick(env.truncation_cond)

        curr_config = curriculum_config  # Keep track of current curriculum config

        try:  # Add a try block around the command loop
            while True:
                try:
                    # Don't use select here - it can cause issues with some platforms
                    cmd, data = remote.recv()

                    if cmd == 'step':
                        actions_dict = data
                        # Format actions for RLGym API
                        formatted_actions = {}
                        for agent_id, action in actions_dict.items():
                            # Handle tensor conversion properly
                            if isinstance(action, torch.Tensor):
                                # Properly convert tensor to numpy, regardless of dimensions
                                action_np = action.cpu().detach().numpy()

                                # If action is a one-hot vector, convert it to an index
                                if len(action_np.shape) == 1 and action_np.shape[0] > 1:
                                    # Get the index of the maximum value (one-hot to index)
                                    action_idx = np.argmax(action_np)
                                    formatted_actions[agent_id] = np.array([action_idx])
                                else:
                                    # Use the numpy array directly
                                    formatted_actions[agent_id] = action_np
                            elif isinstance(action, np.ndarray):
                                formatted_actions[agent_id] = action
                            elif isinstance(action, int):
                                formatted_actions[agent_id] = np.array([action])
                            else:
                                # For any other type, try to convert to numpy array
                                try:
                                    formatted_actions[agent_id] = np.array([action])
                                except:
                                    if debug:
                                        print(f"[DEBUG] Unable to process action of type {type(action)}, using default action")
                                    formatted_actions[agent_id] = np.array([0])  # Default action

                            # Add action to stacker history
                            if action_stacker is not None:
                                action_stacker.add_action(agent_id, formatted_actions[agent_id])
                        # Step the environment
                        next_obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(formatted_actions)
                        remote.send((next_obs_dict, reward_dict, terminated_dict, truncated_dict))

                    elif cmd == 'reset':
                        if debug:
                            print(f"[DEBUG] Worker resetting with stage: {curr_config.get('stage_name', 'Unknown') if curr_config else 'Default'}")

                        # Add retry logic for resets
                        reset_success = False
                        for reset_attempt in range(3):  # Try up to 3 times
                            try:
                                obs = env.reset()
                                reset_success = True
                                remote.send(obs)  # Send response immediately after successful reset
                                break
                            except Exception as e:
                                if debug:
                                    print(f"[DEBUG] Reset attempt {reset_attempt + 1} failed: {e}")
                                if reset_attempt == 2:  # Last attempt
                                    raise
                                time.sleep(0.1)  # Small delay between attempts

                        if not reset_success:
                            # If all reset attempts failed, send empty dict
                            remote.send({})

                    elif cmd == 'set_curriculum':
                        # Update environment with new curriculum configuration
                        old_config = curr_config
                        curr_config = data

                        if debug:
                            old_stage = old_config.get('stage_name', 'Unknown') if old_config else 'Default'
                            new_stage = curr_config.get('stage_name', 'Unknown') if curr_config else 'Default'
                            print(f"[DEBUG] Changing stage from {old_stage} to {new_stage}")

                        # Safely close and recreate environment
                        try:
                            temp_renderer = None
                            if renderer:
                                temp_renderer = env.renderer
                                env.renderer = None  # Prevent renderer from being closed
                            env.close()
                            env = env_fn(renderer=renderer, action_stacker=action_stacker, curriculum_config=curr_config, debug=debug)
                            if temp_renderer:
                                env.renderer = temp_renderer
                            remote.send(True)  # Acknowledge the update
                        except Exception as e:
                            if debug:
                                print(f"[DEBUG] Error recreating environment: {e}")
                            raise

                    elif cmd == 'close':
                        if debug:
                            print("[DEBUG] Worker received close command, exiting gracefully.")
                        break

                    elif cmd == 'reset_action_stacker':
                        agent_id = data
                        if action_stacker is not None:
                            action_stacker.reset_agent(agent_id)
                        remote.send(True)  # Acknowledge

                except EOFError:
                    # Parent closed the connection, expected during shutdown
                    if debug:
                        print("[DEBUG] Worker received EOFError, exiting.")
                    break
                except Exception as e:
                    # Log unexpected errors
                    import traceback
                    print(f"Error in worker loop: {str(e)}")
                    print(traceback.format_exc())
                    break
        except Exception as e:
            # Catch any errors in the command loop
            import traceback
            print(f"Error in worker main loop: {str(e)}")
            print(traceback.format_exc())

    except Exception as e:
        # Catch initialization errors
        import traceback
        print(f"Fatal error in worker initialization: {str(e)}")
        print(traceback.format_exc())
    finally:
        # Guaranteed cleanup block
        if debug:
            print("[DEBUG] Worker entering finally block for cleanup.")

        try:
            if renderer:
                if debug:
                    print("[DEBUG] Closing renderer.")
                renderer.close()
        except Exception as e:
            if debug:
                print(f"[DEBUG] Error closing renderer in worker: {e}")

        try:
            if env:  # Check if env was successfully created
                if debug:
                    print("[DEBUG] Closing environment.")
                env.close()
        except Exception as e:
            if debug:
                print(f"[DEBUG] Error closing environment in worker: {e}")

        try:
            if remote and not remote.closed:
                if debug:
                    print("[DEBUG] Closing remote connection.")
                remote.close()
        except Exception as e:
            if debug:
                print(f"[DEBUG] Error closing remote in worker: {e}")

        if debug:
            print("[DEBUG] Worker cleanup finished.")

class VectorizedEnv:
    """
    Runs multiple RLGym environments in parallel.
    Uses thread-based execution for rendered environments and
    multiprocessing for non-rendered environments.
    Now supports curriculum learning.
    """
    def __init__(self, num_envs, render=False, action_stacker=None, curriculum_manager=None, debug=False):
        self.num_envs = num_envs
        self.render = render
        self.action_stacker = action_stacker
        self.curriculum_manager = curriculum_manager
        self.render_delay = 0.0025
        self.debug = debug

        # For tracking episode metrics for curriculum
        self.episode_rewards = [{} for _ in range(num_envs)]
        self.episode_successes = [False] * num_envs
        self.episode_timeouts = [False] * num_envs

        # Get curriculum configurations if available
        self.curriculum_configs = []
        for env_idx in range(num_envs):
            if self.curriculum_manager:
                # Get potentially different configs for each environment (for rehearsal)
                config = self.curriculum_manager.get_environment_config()

                if self.debug:
                    print(f"[DEBUG] Env {env_idx} initialized with stage: {config['stage_name']}")
                    # Check if config has car position mutator
                    state_mutator = config['state_mutator']
                    has_car_pos = False
                    if hasattr(state_mutator, 'mutators'):
                        for i, mutator in enumerate(state_mutator.mutators):
                            if 'CarPositionMutator' in mutator.__class__.__name__:
                                has_car_pos = True
                                print(f"[DEBUG] Env {env_idx}, stage {config['stage_name']} has CarPositionMutator at index {i}")
                                break
                    if not has_car_pos:
                        print(f"[DEBUG] WARNING: Env {env_idx}, stage {config['stage_name']} has NO CarPositionMutator!")

                # Process config to make it picklable for multiprocessing
                if not render:  # Only needed for multiprocessing mode
                    config = self._make_config_picklable(config)
                self.curriculum_configs.append(config)
            else:
                self.curriculum_configs.append(None)

        # Decide whether to use threading for rendering
        if render:
            # Use thread-based approach for all environments when rendering is enabled
            self.mode = "thread"
            # Create environments directly
            self.envs = []
            for i in range(num_envs):
                # Only create renderer for the first environment
                env_renderer = RLViserRenderer() if (i == 0) else None
                env = get_env(renderer=env_renderer, action_stacker=action_stacker,
                             curriculum_config=self.curriculum_configs[i], debug=self.debug)
                self.envs.append(env)
            # Reset all environments
            self.obs_dicts = [env.reset() for env in self.envs]
            # Explicitly render the first environment
            if num_envs > 0:
                self.envs[0].render()
            # Set up thread pool for parallel execution
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(32, num_envs),
                thread_name_prefix='EnvWorker'
            )
        else:
            # Use multiprocessing for maximum performance when not rendering
            self.mode = "multiprocess"
            # Create communication pipes
            pipes = [Pipe() for _ in range(num_envs)]
            self.remotes = [remote for remote, _ in pipes]  # Store as list instead of tuple
            self.work_remotes = [work_remote for _, work_remote in pipes]  # Store as list

            # Create and start worker processes
            self.processes = []
            for idx, work_remote in enumerate(self.work_remotes):
                process = Process(
                    target=worker,
                    args=(work_remote, get_env, False, action_stacker, self.curriculum_configs[idx], self.debug),
                    daemon=True
                )
                process.start()
                self.processes.append(process)
                work_remote.close()

            # Initialize all environments with proper timeout handling
            self.obs_dicts = []
            failed_workers = []

            for env_idx, remote in enumerate(self.remotes):
                try:
                    remote.send(('reset', None))
                    if hasattr(remote, 'poll') and hasattr(select, 'select'):
                        if select.select([remote], [], [], 10.0)[0]:  # 10 second timeout per worker
                            obs = remote.recv()
                            self.obs_dicts.append(obs)
                            if self.debug:
                                print(f"[DEBUG] Worker {env_idx} initialized successfully")
                            continue
                    else:
                        # Fallback without select
                        obs = remote.recv()
                        self.obs_dicts.append(obs)
                        continue

                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Worker {env_idx} failed to initialize: {e}")

                # If we reach here, the worker failed to initialize properly
                failed_workers.append(env_idx)
                self.obs_dicts.append({})

                # Try to terminate the failed worker
                try:
                    if env_idx < len(self.processes):
                        self.processes[env_idx].terminate()
                        self.processes[env_idx].join(timeout=1.0)
                except:
                    pass

            # Recreate failed workers if any
            if failed_workers and self.debug:
                print(f"[DEBUG] Recreating {len(failed_workers)} failed workers")

            for env_idx in failed_workers:
                try:
                    # Create new pipe
                    new_remote, new_work_remote = Pipe()
                    self.remotes[env_idx] = new_remote

                    # Create new process
                    process = Process(
                        target=worker,
                        args=(new_work_remote, get_env, False, action_stacker,
                              self.curriculum_configs[env_idx], self.debug),
                        daemon=True
                    )
                    process.start()
                    self.processes[env_idx] = process
                    new_work_remote.close()

                    # Initialize the new worker
                    new_remote.send(('reset', None))
                    if select.select([new_remote], [], [], 10.0)[0]:
                        self.obs_dicts[env_idx] = new_remote.recv()
                        if self.debug:
                            print(f"[DEBUG] Worker {env_idx} reinitialized successfully")
                    else:
                        if self.debug:
                            print(f"[DEBUG] Worker {env_idx} failed to respond to reset command")
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Failed to recreate worker {env_idx}: {e}")

        # Common initialization
        self.dones = [False] * num_envs
        self.episode_counts = [0] * num_envs

    def _make_config_picklable(self, config):
        """Ensure valid team size configuration"""
        processed = dict(config)

        # Make sure state_mutator is properly constructed if it's a MutatorSequence
        if "state_mutator" in processed and hasattr(processed["state_mutator"], "mutators"):
            # Ensure that the MutatorSequence is properly constructed with individual mutator arguments
            # rather than a list of mutators
            old_mutator_sequence = processed["state_mutator"]
            if hasattr(old_mutator_sequence, "mutators"):
                # Create a new MutatorSequence with the individual mutators unpacked
                from rlgym.rocket_league.state_mutators import MutatorSequence
                mutators = old_mutator_sequence.mutators
                processed["state_mutator"] = MutatorSequence(*mutators)

        # Force required_agents field
        if "required_agents" not in processed:
            from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator
            team_mutators = []

            # Check if state_mutator is a MutatorSequence with mutators attribute
            if hasattr(processed["state_mutator"], "mutators"):
                team_mutators = [m for m in processed["state_mutator"].mutators
                               if isinstance(m, FixedTeamSizeMutator)]

            if team_mutators:
                processed["required_agents"] = (
                    team_mutators[0].blue_size
                    + team_mutators[0].orange_size
                )
            else:
                processed["required_agents"] = 1

        return processed

    def _step_env(self, args):
        env_idx, env, actions_dict = args

        # Format actions for RLGym API
        formatted_actions = {}
        for agent_id, action in actions_dict.items():
            # Handle tensor conversion properly
            if isinstance(action, torch.Tensor):
                # Properly convert tensor to numpy, regardless of dimensions
                action_np = action.cpu().detach().numpy()

                # If action is a one-hot vector, convert it to an index
                if len(action_np.shape) == 1 and action_np.shape[0] > 1:
                    # Get the index of the maximum value (one-hot to index)
                    action_idx = np.argmax(action_np)
                    formatted_actions[agent_id] = np.array([action_idx])
                else:
                    # Use the numpy array directly
                    formatted_actions[agent_id] = action_np
            elif isinstance(action, np.ndarray):
                formatted_actions[agent_id] = action
            elif isinstance(action, int):
                formatted_actions[agent_id] = np.array([action])
            else:
                # For any other type, try to convert to numpy array
                try:
                    formatted_actions[agent_id] = np.array([action])
                except:
                    if self.debug:
                        print(f"[DEBUG] Unable to process action of type {type(action)}, using default action")
                    formatted_actions[agent_id] = np.array([0])  # Default action

            # Add action to the stacker history
            if self.action_stacker is not None:
                self.action_stacker.add_action(agent_id, formatted_actions[agent_id])

        # Step the environment
        next_obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(formatted_actions)

        # Add rendering and delay if this is the rendered environment
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

    def step(self, actions_dict_list):
        """Step all environments forward using appropriate method based on mode"""
        stats_dict = {}
        if self.mode == "thread":
            # Use thread pool for parallel execution
            futures = [
                self.executor.submit(self._step_env, (i, env, actions))
                for i, (env, actions) in enumerate(zip(self.envs, actions_dict_list))
                if actions  # Only submit if actions exist
            ]

            # Wait for all steps to complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

            # Sort results by environment index
            results.sort(key=lambda x: x[0])

            processed_results = []
            for env_idx, next_obs_dict, reward_dict, terminated_dict, truncated_dict in results:
                # Check if episode is done
                self.dones[env_idx] = any(terminated_dict.values()) or any(truncated_dict.values())

                # Validate agents match expected
                config = self.curriculum_configs[env_idx]
                if config is None:
                    required = len(next_obs_dict)
                else:
                    required = config.get("required_agents", len(next_obs_dict))

                # Track success/timeout for curriculum
                if self.dones[env_idx]:
                    self.episode_successes[env_idx] = any(terminated_dict.values())
                    self.episode_timeouts[env_idx] = any(truncated_dict.values()) and not self.episode_successes[env_idx]

                    # Update curriculum manager with episode results
                    if self.curriculum_manager:
                        # Calculate average reward across all agents
                        avg_reward = sum(self.episode_rewards[env_idx].values()) / max(len(self.episode_rewards[env_idx]), 1)

                        # Submit episode metrics based on curriculum manager type
                        # ManualCurriculumManager expects positional args, CurriculumManager expects a dict
                        if type(self.curriculum_manager).__name__ == "ManualCurriculumManager":
                            self.curriculum_manager.update_progression_stats(
                                episode_rewards=avg_reward,
                                success=self.episode_successes[env_idx],
                                timeout=self.episode_timeouts[env_idx],
                                env_id=env_idx
                            )
                        else: # Assume base CurriculumManager or similar
                            metrics = {
                                "success": self.episode_successes[env_idx],
                                "timeout": self.episode_timeouts[env_idx],
                                "episode_reward": avg_reward
                            }
                            self.curriculum_manager.update_progression_stats(metrics)


                        # Get new curriculum configuration for next episode
                        new_config = self.curriculum_manager.get_environment_config()
                        self.curriculum_configs[env_idx] = new_config

                        # In threaded mode, need to recreate the environment
                        if self.dones[env_idx]:
                            # 1) Detach the renderer before closing
                            existing_renderer = None
                            if env_idx == 0 and self.render:
                                existing_renderer = self.envs[env_idx].renderer
                                self.envs[env_idx].renderer = None  # Prevent renderer.close() in env.close()

                            self.envs[env_idx].close()

                            # 2) Reuse that same renderer (fall back to a new one if it was somehow None)
                            if env_idx == 0 and self.render:
                                if existing_renderer is None:
                                    existing_renderer = RLViserRenderer()
                                # Pass the old renderer back in
                                env_renderer = existing_renderer
                            else:
                                env_renderer = None

                            self.envs[env_idx] = get_env(
                                renderer=env_renderer,
                                action_stacker=self.action_stacker,
                                curriculum_config=self.curriculum_configs[env_idx],
                                debug=self.debug
                            )
                            self.obs_dicts[env_idx] = self.envs[env_idx].reset()

                    if self.dones[env_idx]:
                        # If done, reset the environment
                        self.episode_counts[env_idx] += 1

                        max_reset_attempts = 3
                        for attempt in range(max_reset_attempts):
                            try:
                                obs = self.envs[env_idx].reset()
                                if len(obs) == required:
                                    self.obs_dicts[env_idx] = obs
                                    break
                            except Exception as e:
                                print(f"Reset attempt {attempt + 1} failed: {e}")
                                if attempt == max_reset_attempts - 1:
                                    print("Max reset attempts reached, recreating environment")
                                    self.envs[env_idx].close()
                                    self.envs[env_idx] = get_env(
                                        renderer=RLViserRenderer() if (env_idx == 0 and self.render) else None,
                                        action_stacker=self.action_stacker,
                                        curriculum_config=self.curriculum_configs[env_idx],
                                        debug=self.debug
                                    )
                                    self.obs_dicts[env_idx] = self.envs[env_idx].reset()

                        # Reset action history for all agents
                        if self.action_stacker is not None:
                            for agent_id in next_obs_dict.keys():
                                self.action_stacker.reset_agent(agent_id)

                        # Reset episode tracking variables
                        self.episode_rewards[env_idx] = {}
                        self.episode_successes[env_idx] = False
                        self.episode_timeouts[env_idx] = False

                        # Render again after reset if this is the rendered environment
                        if self.render and env_idx == 0:
                            self.envs[env_idx].render()
                            time.sleep(self.render_delay)
                    else:
                        # Otherwise just update observations
                        self.obs_dicts[env_idx] = next_obs_dict

                    processed_results.append((next_obs_dict, reward_dict, terminated_dict, truncated_dict))

        else:  # multiprocess mode
            # Send step command to all workers with error handling
            active_remotes = []
            for i, (remote, actions_dict) in enumerate(zip(self.remotes, actions_dict_list)):
                try:
                    remote.send(('step', actions_dict))
                    active_remotes.append((i, remote))
                except BrokenPipeError:
                    if self.debug:
                        print(f"[DEBUG] Worker {i} has a broken pipe. Attempting recovery...")
                    self.dones[i] = True  # Mark as done to trigger reset

                    # Try to recreate this worker in the background
                    try:
                        # Close the broken connection if possible
                        try:
                            remote.close()
                        except:
                            pass

                        # Create a new connection
                        new_remote, new_work_remote = Pipe()

                        # Create and start a new worker process
                        process = Process(
                            target=worker,
                            args=(new_work_remote, get_env, False, self.action_stacker,
                                self.curriculum_configs[i], self.debug),
                            daemon=True
                        )
                        process.start()
                        new_work_remote.close()

                        # Properly terminate the old process before replacing it
                        old_process = self.processes[i]
                        try:
                            old_process.terminate()
                            old_process.join(timeout=2.0)  # Wait for termination
                            if old_process.is_alive():
                                if self.debug:
                                    print(f"[DEBUG] Process {i} still alive after terminate, killing...")
                                old_process.kill()  # Force kill if still alive
                                old_process.join(timeout=1.0)
                        except Exception as e:
                            if self.debug:
                                print(f"[DEBUG] Error cleaning up old process {i}: {e}")

                        # Replace the old remote with the new one
                        self.remotes[i] = new_remote
                        self.processes[i] = process

                        # Reset this environment
                        new_remote.send(('reset', None))
                        if hasattr(select, 'select') and select.select([new_remote], [], [], 5.0)[0]:
                            self.obs_dicts[i] = new_remote.recv()
                        else:
                            self.obs_dicts[i] = {}  # Empty dict as fallback

                        if self.debug:
                            print(f"[DEBUG] Worker {i} successfully recreated")
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Failed to recreate worker {i}: {e}")

            # Collect results from all workers
            results = []
            for i, remote in enumerate(self.remotes):
                if (i, remote) in active_remotes:
                    try:
                        if hasattr(select, 'select'):
                            if select.select([remote], [], [], 10.0)[0]:  # 10-second timeout
                                next_obs_dict, reward_dict, terminated_dict, truncated_dict = remote.recv()
                            else:
                                if self.debug:
                                    print(f"[DEBUG] Worker {i} timed out during step")
                                # Use empty results as fallback
                                next_obs_dict, reward_dict, terminated_dict, truncated_dict = {}, {}, {True: True}, {True: True}
                                # Mark this environment as done to trigger reset
                                self.dones[i] = True
                        else:
                            # Fallback without select
                            next_obs_dict, reward_dict, terminated_dict, truncated_dict = remote.recv()
                    except (BrokenPipeError, EOFError, ConnectionResetError) as e:
                        if self.debug:
                            print(f"[DEBUG] Error receiving data from worker {i}: {e}")
                        # Use empty results
                        next_obs_dict, reward_dict, terminated_dict, truncated_dict = {}, {}, {True: True}, {True: True}
                        # Mark this environment as done to trigger reset
                        self.dones[i] = True
                else:
                    # For inactive workers, use empty results
                    next_obs_dict, reward_dict, terminated_dict, truncated_dict = {}, {}, {True: True}, {True: True}
                    self.dones[i] = True

                # Track rewards for curriculum
                if self.curriculum_manager is not None:
                    # Ensure episode rewards dict is initialized
                    if not isinstance(self.episode_rewards[i], dict):
                        self.episode_rewards[i] = {}

                    for agent_id, reward in reward_dict.items():
                        if agent_id not in self.episode_rewards[i]:
                            self.episode_rewards[i][agent_id] = 0
                        self.episode_rewards[i][agent_id] += reward

                # Check if episode is done
                self.dones[i] = any(terminated_dict.values()) or any(truncated_dict.values())

                # Validate agents match expected
                config = self.curriculum_configs[i]
                if config is None:
                    required = len(next_obs_dict) if next_obs_dict else 0
                else:
                    required = config.get("required_agents", len(next_obs_dict) if next_obs_dict else 0)

                # Track success/timeout for curriculum
                if self.dones[i]:
                    self.episode_successes[i] = any(terminated_dict.values())
                    self.episode_timeouts[i] = any(truncated_dict.values()) and not self.episode_successes[i]

                    # Update curriculum manager with episode results
                    if self.curriculum_manager:
                        # Calculate average reward across all agents
                        if len(self.episode_rewards[i]) > 0:
                            avg_reward = sum(self.episode_rewards[i].values()) / len(self.episode_rewards[i])
                        else:
                            avg_reward = 0.0

                        # Submit episode metrics based on curriculum manager type
                        if type(self.curriculum_manager).__name__ == "ManualCurriculumManager":
                            self.curriculum_manager.update_progression_stats(
                                episode_rewards=avg_reward,
                                success=self.episode_successes[i],
                                timeout=self.episode_timeouts[i],
                                env_id=i
                            )
                        else: # Assume base CurriculumManager or similar
                            metrics = {
                                "success": self.episode_successes[i],
                                "timeout": self.episode_timeouts[i],
                                "episode_reward": avg_reward
                            }
                            self.curriculum_manager.update_progression_stats(metrics)


                        # Get new curriculum configuration for next episode
                        new_config = self.curriculum_manager.get_environment_config()
                        # Process config to make it picklable
                        new_config = self._make_config_picklable(new_config)
                        self.curriculum_configs[i] = new_config

                        # Send the new curriculum configuration to the worker if it's still active
                        try:
                            remote.send(('set_curriculum', new_config))
                            if select.select([remote], [], [], 5.0)[0]:
                                remote.recv()  # Wait for acknowledgment
                        except (BrokenPipeError, EOFError):
                            if self.debug:
                                print(f"[DEBUG] Worker {i} failed to update curriculum - will recreate on reset")

                    # If done, reset the environment with retry logic
                    self.episode_counts[i] += 1
                    max_reset_attempts = 3
                    reset_success = False

                    for attempt in range(max_reset_attempts):
                        try:
                            remote.send(('reset', None))
                            if hasattr(select, 'select') and select.select([remote], [], [], 5.0)[0]:
                                obs = remote.recv()
                                if len(obs) == required:
                                    self.obs_dicts[i] = obs
                                    reset_success = True
                                    break
                            if self.debug:
                                print(f"[DEBUG] Reset attempt {attempt + 1} failed or timed out")
                        except (BrokenPipeError, EOFError) as e:
                            if self.debug:
                                print(f"[DEBUG] Reset attempt {attempt + 1} failed: {e}")
                            if attempt == max_reset_attempts - 1:
                                # Try to recreate the worker
                                try:
                                    # Close connections
                                    try:
                                        remote.close()
                                    except:
                                        pass

                                    # Create new connection
                                    new_remote, new_work_remote = Pipe()

                                    # Create and start new worker
                                    process = Process(
                                        target=worker,
                                        args=(new_work_remote, get_env, False, self.action_stacker,
                                            self.curriculum_configs[i], self.debug),
                                        daemon=True
                                    )
                                    process.start()
                                    new_work_remote.close()

                                    # Replace old remote with new one
                                    self.remotes[i] = new_remote
                                    self.processes[i].terminate()
                                    self.processes[i] = process

                                    # Reset new environment
                                    new_remote.send(('reset', None))
                                    if hasattr(select, 'select') and select.select([new_remote], [], [], 5.0)[0]:
                                        self.obs_dicts[i] = new_remote.recv()
                                        reset_success = True

                                    if self.debug:
                                        print(f"[DEBUG] Worker {i} successfully recreated")
                                except Exception as e:
                                    if self.debug:
                                        print(f"[DEBUG] Failed to recreate worker {i}: {e}")

                    # Reset action stacker for all agents if there's valid observation data
                    if self.action_stacker is not None and next_obs_dict and reset_success:
                        for agent_id in next_obs_dict.keys():
                            try:
                                remote.send(('reset_action_stacker', agent_id))
                                if select.select([remote], [], [], 5.0)[0]:
                                    remote.recv()  # Wait for confirmation
                            except (BrokenPipeError, EOFError):
                                if self.debug:
                                    print(f"[DEBUG] Failed to reset action stacker for agent {agent_id}")

                    # Reset episode tracking variables
                    self.episode_rewards[i] = {}
                    self.episode_successes[i] = False
                    self.episode_timeouts[i] = False
                else:
                    # Otherwise just update observations
                    self.obs_dicts[i] = next_obs_dict

                results.append((next_obs_dict, reward_dict, terminated_dict, truncated_dict))

        return results, self.dones.copy(), self.episode_counts.copy()

    def force_env_reset(self, env_idx):
        """Force reset a problematic environment (thread mode)"""
        if self.mode == "thread":
            existing_renderer = None
            if (env_idx == 0 and self.render and hasattr(self.envs[env_idx], 'renderer')):
                existing_renderer = self.envs[env_idx].renderer
            # Detach renderer before closing so it isn’t closed
            if existing_renderer is not None:
                self.envs[env_idx].renderer = None
            self.envs[env_idx].close()
            self.envs[env_idx] = get_env(
                renderer=existing_renderer,
                action_stacker=self.action_stacker,
                curriculum_config=self.curriculum_configs[env_idx],
                debug=self.debug
            )
            self.obs_dicts[env_idx] = self.envs[env_idx].reset()

    def close(self):
        """Clean up resources properly based on the mode"""
        if self.mode == "thread":
            # Close the thread pool
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)  # Wait for all tasks to complete

            # Close all environments
            if hasattr(self, 'envs'):
                for env in self.envs:
                    try:
                        env.close()
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error closing environment: {e}")

        else:  # multiprocess mode
            # First try sending close command to all workers
            if hasattr(self, 'remotes'):
                closed_remotes = set() # Keep track of remotes we sent close to
                for i, remote in enumerate(self.remotes):
                    try:
                        if remote and not getattr(remote, 'closed', False):  # Check if remote exists and is not already closed
                            remote.send(('close', None))
                            closed_remotes.add(i)
                            if self.debug:
                                print(f"[DEBUG] Sent close command to worker {i}")
                    except (BrokenPipeError, EOFError) as e:
                        if self.debug:
                            print(f"[DEBUG] Error sending close to worker {i} (already closed?): {e}")
                    except Exception as e:
                         if self.debug:
                            print(f"[DEBUG] Unexpected error sending close to worker {i}: {e}")

            # Wait longer for workers to process the close command - viztracer needs time
            time.sleep(1.0)

            # Wait for processes to finish gracefully without terminating them
            # This allows viztracer to properly finish writing its trace files
            if hasattr(self, 'processes'):
                for i, process in enumerate(self.processes):
                    try:
                        if process.is_alive():
                            if self.debug:
                                print(f"[DEBUG] Waiting for worker {i} to exit gracefully...")
                            process.join(timeout=30.0)  # Increased timeout to 30 seconds for viztracer
                            if process.is_alive():
                                # If it's still alive after generous timeout, log it but DO NOT kill it
                                # This is critical for viztracer to work correctly
                                if self.debug:
                                    print(f"[DEBUG] Worker {i} still running after 30s. Waiting for natural exit.")
                            else:
                                 if self.debug:
                                     print(f"[DEBUG] Worker {i} exited successfully.")
                        else:
                             if self.debug:
                                 print(f"[DEBUG] Worker {i} was already finished.")
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error joining process {i}: {e}")

            # Finally close all pipe connections on the main process side
            if hasattr(self, 'remotes'):
                for i, remote in enumerate(self.remotes):
                    try:
                        if remote and not getattr(remote, 'closed', False):
                            remote.close()
                            if self.debug:
                                print(f"[DEBUG] Closed remote {i} connection")
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error closing pipe for worker {i}: {e}")

            if hasattr(self, 'work_remotes'):
                for i, work_remote in enumerate(self.work_remotes):
                    try:
                        if work_remote is not None and not getattr(work_remote, 'closed', False):
                            work_remote.close()
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error closing work remote {i}: {e}")

        # Force garbage collection to clean up any remaining references
        try:
            import gc
            gc.collect()
        except ImportError:
            pass
            pass
