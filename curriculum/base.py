"""Core curriculum learning functionality."""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from rlgym.api import StateMutator, RewardFunction, DoneCondition
from dataclasses import dataclass
import wandb

@dataclass
class ProgressionRequirements:
    min_success_rate: float
    min_avg_reward: float
    min_episodes: int
    max_std_dev: float
    required_consecutive_successes: int = 3

    def __post_init__(self):
        """Validate progression requirements"""
        if not 0 <= self.min_success_rate <= 1:
            raise ValueError("min_success_rate must be between 0 and 1")
        if self.min_episodes < 1:
            raise ValueError("min_episodes must be positive")
        if self.max_std_dev < 0:
            raise ValueError("max_std_dev cannot be negative")
        if self.min_avg_reward <= -2.0:
            raise ValueError("Average reward threshold too low")
        if self.required_consecutive_successes < 1:
            raise ValueError("required_consecutive_successes must be positive")

class CurriculumStage:
    """
    Defines a single stage in the curriculum learning process.
    Each stage has its own environment configuration, reward functions, and completion criteria.
    """
    def __init__(
        self,
        name: str,
        state_mutator: StateMutator,
        reward_function: RewardFunction,
        termination_condition: DoneCondition,
        truncation_condition: DoneCondition,
        progress_metrics: List[str] = None,
        difficulty_params: Dict[str, Tuple[float, float]] = None,
        hyperparameter_adjustments: Dict[str, float] = None,
        progression_requirements: ProgressionRequirements = None
    ):
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        if not isinstance(state_mutator, StateMutator):
            raise ValueError("state_mutator must be an instance of StateMutator")
        if not isinstance(reward_function, RewardFunction):
            raise ValueError("reward_function must be an instance of RewardFunction")
        if not isinstance(termination_condition, DoneCondition):
            raise ValueError("termination_condition must be an instance of DoneCondition")
        if not isinstance(truncation_condition, DoneCondition):
            raise ValueError("truncation_condition must be an instance of DoneCondition")

        self.name = name
        self.state_mutator = state_mutator
        self.reward_function = reward_function
        self.termination_condition = termination_condition
        self.truncation_condition = truncation_condition
        self.progress_metrics = progress_metrics or ["episode_reward", "success_rate"]
        self.difficulty_params = difficulty_params or {}
        self.hyperparameter_adjustments = hyperparameter_adjustments or {}
        self.progression_requirements = progression_requirements

        # Initialize tracking variables
        self.episode_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.rewards_history = []
        self.moving_success_rate = 0.0
        self.moving_avg_reward = 0.0

    def validate_progression(self) -> bool:
        """Check if stage progression requirements are met."""
        if not self.progression_requirements or self.episode_count < self.progression_requirements.min_episodes:
            return False

        recent_rewards = self.rewards_history[-100:]
        if not recent_rewards:
            return False

        recent_avg_reward = np.mean(recent_rewards)
        recent_std_dev = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
        consecutive_successes = self.get_consecutive_successes()

        return (self.moving_success_rate >= self.progression_requirements.min_success_rate and
                recent_avg_reward >= self.progression_requirements.min_avg_reward and
                recent_std_dev <= self.progression_requirements.max_std_dev and
                consecutive_successes >= self.progression_requirements.required_consecutive_successes)

    def get_consecutive_successes(self) -> int:
        """Count current streak of consecutive successful episodes"""
        count = 0
        for success in reversed(self.rewards_history):
            if success > 0:
                count += 1
            else:
                break
        return count

    def get_environment_config(self, difficulty_level: float) -> Dict[str, Any]:
        """Returns the stage configuration with parameters adjusted for difficulty level."""
        difficulty = max(0.0, min(1.0, difficulty_level))
        difficulty_params = {}

        for param_name, (min_val, max_val) in self.difficulty_params.items():
            difficulty_params[param_name] = min_val + difficulty * (max_val - min_val)

        return {
            "stage_name": self.name,
            "state_mutator": self.state_mutator,
            "reward_function": self.reward_function,
            "termination_condition": self.termination_condition,
            "truncation_condition": self.truncation_condition,
            "difficulty_level": difficulty,
            "difficulty_params": difficulty_params
        }

    def update_statistics(self, episode_data):
        """Update the stage's statistics with new episode data"""
        # Validate input data
        if not isinstance(episode_data, dict):
            raise ValueError("episode_data must be a dictionary")

        # Check for required fields
        if "episode_reward" not in episode_data:
            raise ValueError("episode_data must contain 'episode_reward'")

        # Initialize rewards_history as empty list if it doesn't exist
        if not hasattr(self, 'rewards_history') or self.rewards_history is None:
            self.rewards_history = []

        # Initialize counters if they don't exist
        if not hasattr(self, 'episode_count') or self.episode_count is None:
            self.episode_count = 0

        if not hasattr(self, 'success_count') or self.success_count is None:
            self.success_count = 0

        if not hasattr(self, 'failure_count') or self.failure_count is None:
            self.failure_count = 0

        if "success" in episode_data:
            if not isinstance(episode_data["success"], bool):
                raise ValueError("episode_data['success'] must be a boolean")

            if episode_data["success"]:
                self.success_count += 1
            else:
                self.failure_count += 1

        # Get the episode reward

        reward = episode_data["episode_reward"]

        # Validate reward is numeric
        if not isinstance(reward, (int, float)):
            raise ValueError("episode_data['episode_reward'] must be numeric")

        # Update episode count and rewards history
        self.episode_count += 1
        self.rewards_history.append(reward)

        # Calculate moving success rate
        if hasattr(self, 'success_count') and hasattr(self, 'episode_count') and self.episode_count > 0:
            # Make sure it's a valid ratio (between 0 and 1)
            self.moving_success_rate = min(1.0, max(0.0, self.success_count / self.episode_count))
        else:
            self.moving_success_rate = 0.0

        # Calculate moving average reward
        # Handle case where rewards_history might contain invalid values
        try:
            if hasattr(self, 'rewards_history') and len(self.rewards_history) > 0:
                # Make sure all values are numeric before calculating mean
                valid_rewards = [r for r in self.rewards_history if isinstance(r, (int, float))]
                self.moving_avg_reward = float(np.mean(valid_rewards)) if valid_rewards else 0.0
            else:
                self.moving_avg_reward = 0.0
        except Exception:
            # Fallback in case of any calculation errors
            self.moving_avg_reward = 0.0

    def reset_statistics(self) -> None:
        """Reset all progress tracking statistics."""
        self.episode_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.rewards_history = []
        self.moving_success_rate = 0.0
        self.moving_avg_reward = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get all progress statistics for this stage."""
        # Update derived statistics
        if self.episode_count > 0:
            self.moving_success_rate = self.success_count / self.episode_count
        if self.rewards_history:
            self.moving_avg_reward = np.mean(self.rewards_history)

        return {
            "name": self.name,
            "episodes": self.episode_count,
            "successes": self.success_count,
            "failures": self.failure_count,
            "success_rate": self.moving_success_rate,
            "avg_reward": self.moving_avg_reward,
            "consecutive_successes": self.get_consecutive_successes()
        }

    def get_config_with_difficulty(self, difficulty_level: float) -> Dict[str, Any]:
        """Alias for get_environment_config for compatibility."""
        return self.get_environment_config(difficulty_level)


class CurriculumManager:
    """
    Manages progression through curriculum stages based on agent performance.
    Handles stage transitions and hyperparameter adjustments.
    """
    def __init__(
        self,
        stages: List[CurriculumStage],
        progress_thresholds: Dict[str, float] = None,
        max_rehearsal_stages: int = 3,
        rehearsal_decay_factor: float = 0.5,
        evaluation_window: int = 100,
        debug: bool = False,
        testing: bool = False,
        use_wandb: bool = True
    ):
        if not stages:
            raise ValueError("At least one curriculum stage must be provided")

        # Add back validation for rehearsal parameters
        if max_rehearsal_stages < 0:
            raise ValueError("max_rehearsal_stages must be non-negative")

        if rehearsal_decay_factor <= 0 or rehearsal_decay_factor > 1:
            raise ValueError("rehearsal_decay_factor must be in range (0, 1]")

        self.stages = stages
        self.current_stage_index = 0
        self.current_stage = stages[0]
        self.progress_thresholds = progress_thresholds or {}
        self.max_rehearsal_stages = max_rehearsal_stages
        self.rehearsal_decay_factor = rehearsal_decay_factor
        self.evaluation_window = evaluation_window
        self.debug = debug
        self.testing = testing
        self.use_wandb = use_wandb

        # Tracking variables
        self.completed_stages = []
        self.current_difficulty = 0.0  # Initialize difficulty level to 0
        self.current_rehearsal = None  # Track current rehearsal stage
        self.difficulty_threshold = 0.95  # Threshold to progress to next stage
        self.difficulty_increase_rate = 0.1  # Rate of difficulty increase
        self.difficulty_level = 0.0
        self.episodes_in_current_stage = 0
        self.success_count = 0
        self.failure_count = 0
        self.consecutive_successes = 0
        self.trainer = None  # Will be set via register_trainer
        self._last_wandb_step = 0
        self.total_episodes = 0  # Total episodes across all stages

        if debug:
            print(f"[DEBUG] Initialized curriculum with {len(stages)} stages")
            print(f"[DEBUG] Starting at stage: {self.current_stage.name}")
        
        # Initialize any stage-specific variables
        if hasattr(self.current_stage, 'initialize') and callable(self.current_stage.initialize):
            self.current_stage.initialize()

    def register_trainer(self, trainer):
        """Register a trainer object for hyperparameter adjustments"""
        # Only register if not already registered to prevent infinite recursion
        if self.trainer != trainer:
            self.trainer = trainer
            # Only register back with trainer if not already registered
            if hasattr(trainer, 'register_curriculum_manager'):
                if not hasattr(trainer, '_curriculum_manager') or trainer._curriculum_manager != self:
                    trainer.register_curriculum_manager(self)
            
            # After registering, check if we have a pretraining stage and connect it
            if len(self.stages) > 0:
                first_stage = self.stages[0]
                if hasattr(first_stage, 'is_pretraining') and first_stage.is_pretraining:
                    # Register trainer with the pretraining stage
                    first_stage.register_trainer(trainer)
                    
                    if self.debug:
                        print(f"[DEBUG] Registered trainer with pre-training stage: {first_stage.name}")

    def _get_current_step(self) -> Optional[int]:
        """Get current training step for wandb logging"""
        if self.debug:
            print(f"[STEP DEBUG] _get_current_step called in curriculum")
            
        if self.trainer is None:
            # If no trainer is registered, use internal counter
            self._last_wandb_step += 1
            if self.debug:
                print(f"[STEP DEBUG] No trainer registered, using internal counter: {self._last_wandb_step}")
            return self._last_wandb_step
            
        # Use trainer's step counter as source of truth when available
        if hasattr(self.trainer, '_true_training_steps'):
            # Use the true training steps counter from the trainer (most accurate)
            step_value = self.trainer._true_training_steps()  # Call the method instead of accessing as property
            if self.debug:
                print(f"[STEP DEBUG] Using trainer._true_training_steps(): {step_value}")
            return step_value
        elif hasattr(self.trainer, 'training_steps') and hasattr(self.trainer, 'training_step_offset'):
            # Fall back to calculated training steps if true counter isn't available
            step_value = self.trainer.training_steps + self.trainer.training_step_offset
            if self.debug:
                print(f"[STEP DEBUG] Using trainer.training_steps + offset: {step_value}")
            return step_value
        else:
            # Last resort: increment internal counter if no trainer counters are available
            self._last_wandb_step += 1
            if self.debug:
                print(f"[STEP DEBUG] Trainer lacks step counters, using internal: {self._last_wandb_step}")
            return self._last_wandb_step

    def _log_to_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Centralized wandb logging with step validation"""
        if not self.use_wandb or wandb.run is None:
            return
            
        # If step is explicitly provided, use it
        # Otherwise get synchronized step from trainer
        current_step = step if step is not None else self._get_current_step()
        
        if current_step is None:
            # Skip logging if we can't get a valid step
            if self.debug:
                print("Skipping wandb logging - couldn't get valid step")
            return
            
        # Never log to a step that's less than our last step
        if not self.testing and hasattr(self, '_last_wandb_step') and current_step <= self._last_wandb_step:
            if self.debug:
                print(f"Skipping wandb log for step {current_step} (≤ {self._last_wandb_step})")
            return
            
        # Use trainer's centralized logging if available
        if self.trainer is not None and hasattr(self.trainer, '_log_to_wandb'):
            # Create curriculum-specific metrics with CURR prefix
            curriculum_metrics = {}
            for key, value in metrics.items():
                # Add CURR/ prefix to match our trainer's structure
                curriculum_metrics[f"CURR/{key}"] = value
            
            # Let the trainer handle logging with proper step syncing
            self.trainer._log_to_wandb(curriculum_metrics, current_step)
        else:
            # Fall back to direct wandb logging if trainer unavailable
            wandb.log(metrics, step=current_step)
        
        # Remember this step for next time
        self._last_wandb_step = current_step

    def get_environment_config(self) -> Dict[str, Any]:
        """Get the current environment configuration based on the active stage and difficulty level."""
        # Check if we're in pretraining mode
        current_stage = self.stages[self.current_stage_index]
        is_pretraining = (hasattr(current_stage, 'is_pretraining') and current_stage.is_pretraining and 
                         hasattr(self, 'trainer') and not getattr(self.trainer, 'pretraining_completed', True))
        
        # Determine if we should use a rehearsal stage
        if self.current_stage_index > 0 and self.max_rehearsal_stages > 0:
            if np.random.random() < self._get_rehearsal_probability():
                # Select a previous stage for rehearsal
                rehearsal_index = self._select_rehearsal_stage()
                stage = self.stages[rehearsal_index]
                config = stage.get_config_with_difficulty(0.0)  # Use difficulty 0.0 for rehearsal
                config["is_rehearsal"] = True
                
                if self.debug:
                    print(f"[DEBUG] Rehearsing stage {stage.name} at difficulty 0.0")
                    
                return config
        
        # Normal mode, use the current stage
        config = current_stage.get_config_with_difficulty(self.current_difficulty)
        config["is_rehearsal"] = False
        
        # Add pretraining flag if applicable
        if is_pretraining:
            config["is_pretraining"] = True
            
        return config

    def update_progression_stats(self, episode_metrics: Dict[str, Any]) -> None:
        """Update statistics based on completed episode metrics."""
        current_stage = self.stages[self.current_stage_index]
        current_stage.update_statistics(episode_metrics)
        self.episodes_in_current_stage += 1

        if self.use_wandb:
            self._log_to_wandb({
                "curriculum/current_stage": self.current_stage_index,
                "curriculum/stage_name": current_stage.name,
                "curriculum/current_difficulty": self.current_difficulty,
                "curriculum/success_rate": current_stage.moving_success_rate,
                "curriculum/avg_reward": current_stage.moving_avg_reward,
                "curriculum/total_episodes": self.episodes_in_current_stage,
                "curriculum/stage_progress": self.get_stage_progress(),
                "curriculum/overall_progress": self.get_overall_progress()["total_progress"],
                "curriculum/episodes_in_stage": current_stage.episode_count,
                "curriculum/consecutive_successes": current_stage.get_consecutive_successes()
            })

        if self.episodes_in_current_stage % self.evaluation_window == 0:
            self._evaluate_progression()

    def _evaluate_progression(self) -> bool:
        """Evaluate if the agent should progress to the next curriculum stage."""
        if self.current_stage_index >= len(self.stages) - 1:
            return False

        current_stage = self.stages[self.current_stage_index]
        progression_requirements = current_stage.progression_requirements
        stats = current_stage.get_statistics()

        # Don't proceed if we haven't met minimum episode requirement
        if stats["episodes"] < progression_requirements.min_episodes:
            return False

        # Update difficulty based on performance
        if self.current_difficulty < 1.0:
            if (stats["success_rate"] >= progression_requirements.min_success_rate and
                stats["avg_reward"] >= progression_requirements.min_avg_reward):
                old_difficulty = self.current_difficulty
                self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_increase_rate)

                if self.use_wandb and wandb.run is not None:
                    self._log_to_wandb({
                        "curriculum/difficulty_increase": self.current_difficulty - old_difficulty,
                        "curriculum/current_difficulty": self.current_difficulty,
                        "curriculum/stage_name": current_stage.name
                    })

        # Check for stage progression
        if (self.current_difficulty >= 0.95 and
            stats["success_rate"] >= progression_requirements.min_success_rate and
            stats["avg_reward"] >= progression_requirements.min_avg_reward and
            stats["consecutive_successes"] >= progression_requirements.required_consecutive_successes and
            stats["episodes"] >= progression_requirements.min_episodes):

            # Calculate standard deviation of recent rewards
            recent_rewards = current_stage.rewards_history[-100:]
            if len(recent_rewards) >= 2:  # Need at least 2 samples for std
                std_dev = np.std(recent_rewards)
                if std_dev <= progression_requirements.max_std_dev:
                    self._progress_to_next_stage()
                    return True

        return False

    def _evaluate_progression(self):
        """
        Evaluate if the current stage should progress to the next stage.
        Returns True if progression occurred, False otherwise.
        """
        current_stage = self.stages[self.current_stage_index]
        
        # Special handling for pretraining stage
        if hasattr(current_stage, 'is_pretraining') and current_stage.is_pretraining:
            if hasattr(self, 'trainer') and getattr(self.trainer, 'pretraining_completed', False):
                # Pretraining is complete, proceed to the next stage
                self._progress_to_next_stage()
                return True
            # Pretraining is still ongoing
            return False
            
        # Regular stage progression logic
        if current_stage.validate_progression():
            if self.current_difficulty >= self.difficulty_threshold or self.current_stage_index == len(self.stages) - 1:
                self._progress_to_next_stage()
                return True
            else:
                # Increase difficulty but stay on the same stage
                self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_increase_rate)
                if self.debug:
                    print(f"[DEBUG] Increased difficulty to {self.current_difficulty:.2f} for stage {current_stage.name}")
                return False
        else:
            # Optionally decrease difficulty if performance is consistently poor
            if hasattr(current_stage, 'moving_success_rate') and current_stage.moving_success_rate < 0.3 and self.current_difficulty > 0.2:
                self.current_difficulty = max(0.0, self.current_difficulty - self.difficulty_increase_rate * 0.5)
                if self.debug:
                    print(f"[DEBUG] Decreased difficulty to {self.current_difficulty:.2f} due to poor performance")
            return False

    def _get_rehearsal_probability(self) -> float:
        """Get the probability of using a rehearsal stage instead of the current stage."""
        # Base probability depends on how many stages we've progressed through
        base_prob = 0.3  # 30% chance of rehearsal when possible

        # Increase probability based on poor recent performance
        current_stage = self.stages[self.current_stage_index]
        if current_stage.moving_success_rate < 0.3:  # If success rate is very low
            base_prob = 0.5  # Increase rehearsal chance to 50%
        elif current_stage.moving_avg_reward < 0.2:  # If rewards are very low
            base_prob = 0.4  # Increase rehearsal chance to 40%

        # Scale probability based on stage progress
        progress_factor = min(1.0, self.current_stage_index / max(len(self.stages) - 1, 1))
        prob = base_prob * progress_factor

        # Return increased probability during regression
        return prob if current_stage.moving_success_rate >= 0.4 else 0.5

    def _select_rehearsal_stage(self) -> int:
        """
        Select a previous stage for rehearsal using decay-based probability
        """
        # Determine available stages for rehearsal
        available_stages = min(self.current_stage_index, self.max_rehearsal_stages)

        if available_stages <= 0:
            return 0  # No rehearsal possible

        # Create array of indices for previous stages
        stage_indices = np.arange(self.current_stage_index - available_stages, self.current_stage_index)

        # Create decay probabilities favoring more recent stages
        probs = np.array([self.rehearsal_decay_factor ** (available_stages - i - 1)
                        for i in range(available_stages)])

        # Normalize probabilities
        probs = probs / np.sum(probs)

        # Select stage based on probabilities
        selected_idx = np.random.choice(stage_indices, p=probs)

        return selected_idx

    def _progress_to_next_stage(self) -> None:
        """Progress to the next curriculum stage."""
        if self.current_stage_index >= len(self.stages) - 1:
            if self.debug:
                print("Already at final stage")
            return

        old_stage = self.stages[self.current_stage_index]
        old_stage_name = old_stage.name
        old_stats = old_stage.get_statistics()

        # Progress to next stage
        self.current_stage_index += 1
        new_stage = self.stages[self.current_stage_index]

        # Reset difficulty for new stage
        self.current_difficulty = 0.0

        # Record transition
        self.completed_stages.append({
            "episode": self.episodes_in_current_stage,
            "from_stage": old_stage_name,
            "to_stage": new_stage.name,
            "timestamp": np.datetime64('now'),
            "final_stats": old_stats
        })

        # Apply hyperparameter adjustments
        self._adjust_hyperparameters()

        # Log transition with validated step
        if self.use_wandb:
            self._log_to_wandb({
                "curriculum/stage_transition": 1.0,
                "curriculum/from_stage": old_stage_name,
                "curriculum/to_stage": new_stage.name,
                "curriculum/from_stage_index": self.current_stage_index - 1,
                "curriculum/to_stage_index": self.current_stage_index,
                "curriculum/completed_stage/success_rate": old_stats["success_rate"],
                "curriculum/completed_stage/avg_reward": old_stats["avg_reward"],
                "curriculum/completed_stage/episodes": old_stats["episodes"]
            })

    def _adjust_hyperparameters(self) -> None:
        """Adjust training hyperparameters based on the current stage."""
        if self.trainer is None:
            return

        current_stage = self.stages[self.current_stage_index]
        adjustments = current_stage.hyperparameter_adjustments

        if not adjustments:
            return

        for param, value in adjustments.items():
            if param == "lr_actor" and hasattr(self.trainer, "actor_optimizer"):
                for param_group in self.trainer.actor_optimizer.param_groups:
                    param_group["lr"] = value
            elif param == "lr_critic" and hasattr(self.trainer, "critic_optimizer"):
                for param_group in self.trainer.critic_optimizer.param_groups:
                    param_group["lr"] = value
            elif param == "entropy_coef" and hasattr(self.trainer, "entropy_coef"):
                self.trainer.entropy_coef = value

        if self.use_wandb:
            self._log_to_wandb({
                f"curriculum/hyperparams/{param}": value
                for param, value in adjustments.items()
            })

    def requires_bots(self) -> bool:
        """Check if any stage requires RLBot opponents."""
        return any(stage.__class__.__name__ == "RLBotSkillStage" for stage in self.stages)

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the curriculum manager as a dictionary."""
        # Ensure stage data is serializable (e.g., convert numpy arrays if needed)
        stages_data = []
        for stage in self.stages:
            stage_state = stage.get_statistics() # Use existing method to get stats
            # Convert numpy types if necessary for broader compatibility
            for key, value in stage_state.items():
                if isinstance(value, np.ndarray):
                    stage_state[key] = value.tolist()
                elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                     stage_state[key] = int(value)
                elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                     stage_state[key] = float(value)
                elif isinstance(value, list):
                     # Ensure lists don't contain numpy types
                     stage_state[key] = [float(v) if isinstance(v, (np.float_, np.float16, np.float32, np.float64)) else v for v in value]

            # Add rewards history separately if needed (can be large)
            # stage_state['rewards_history'] = stage.rewards_history # Consider if this is too large
            stages_data.append(stage_state)

        return {
            'current_stage_index': self.current_stage_index,
            'current_difficulty': self.current_difficulty,
            'total_episodes': self.total_episodes,
            'completed_stages': self.completed_stages,
            'stages_data': stages_data # Store stats, not full objects
        }

    def load_state(self, state_dict: Dict[str, Any]):
        """Load curriculum state from a dictionary."""
        if not isinstance(state_dict, dict):
            raise ValueError("State must be a dictionary")
            
        # Load basic properties
        if 'current_stage_index' in state_dict:
            self.current_stage_index = state_dict['current_stage_index']
            if 0 <= self.current_stage_index < len(self.stages):
                self.current_stage = self.stages[self.current_stage_index]
            else:
                print(f"Warning: Loaded invalid stage index {self.current_stage_index}, resetting to 0.")
                self.current_stage_index = 0
                self.current_stage = self.stages[0]
            
        if 'current_difficulty' in state_dict:
            self.current_difficulty = state_dict['current_difficulty']
            
        if 'completed_stages' in state_dict:
            self.completed_stages = state_dict['completed_stages']
            
        if 'total_episodes' in state_dict:
            self.total_episodes = state_dict['total_episodes']
            
        # Load stage-specific states if available
        if 'stages_data' in state_dict and isinstance(state_dict['stages_data'], list):
            stages_data = state_dict['stages_data']
            for i, stage_data in enumerate(stages_data):
                if i < len(self.stages):
                    stage = self.stages[i]
                    # Reset and update stage statistics
                    if isinstance(stage_data, dict):
                        # Basic attributes we need to restore
                        if 'episodes' in stage_data:
                            stage.episode_count = stage_data['episodes']
                        elif 'episode_count' in stage_data:
                            stage.episode_count = stage_data['episode_count']
                            
                        if 'successes' in stage_data:
                            stage.success_count = stage_data['successes']
                        elif 'success_count' in stage_data:
                            stage.success_count = stage_data['success_count']
                            
                        if 'failures' in stage_data:
                            stage.failure_count = stage_data['failures']
                        elif 'failure_count' in stage_data:
                            stage.failure_count = stage_data['failure_count']
                            
                        if 'success_rate' in stage_data:
                            stage.moving_success_rate = stage_data['success_rate']
                        elif 'moving_success_rate' in stage_data:
                            stage.moving_success_rate = stage_data['moving_success_rate']
                            
                        if 'avg_reward' in stage_data:
                            stage.moving_avg_reward = stage_data['avg_reward']
                        elif 'moving_avg_reward' in stage_data:
                            stage.moving_avg_reward = stage_data['moving_avg_reward']
                            
                        # Conditionally restore rewards history (may be large) if available
                        if 'rewards_history' in stage_data and isinstance(stage_data['rewards_history'], list):
                            rewards = stage_data['rewards_history']
                            # Convert numpy types to native Python types if needed
                            stage.rewards_history = []
                            for reward in rewards:
                                if hasattr(reward, 'item'):  # Handle numpy scalars
                                    stage.rewards_history.append(reward.item())
                                else:
                                    stage.rewards_history.append(float(reward))
                                    
        # Check if pretraining flag exists in the state, and handle it appropriately
        # First, check if we have a pretraining stage
        is_pretraining_stage_exists = False
        for stage in self.stages:
            if hasattr(stage, 'is_pretraining') and stage.is_pretraining:
                is_pretraining_stage_exists = True
                break
                
        # If we have a pretraining stage, check if we need to synchronize with trainer
        if is_pretraining_stage_exists and 'pretraining_completed' in state_dict:
            # If trainer is available, synchronize pretraining state
            if self.trainer is not None:
                # Update trainer's pretraining state
                if hasattr(self.trainer, 'pretraining_completed'):
                    self.trainer.pretraining_completed = state_dict['pretraining_completed']
                    if self.debug:
                        print(f"[DEBUG] Updated trainer.pretraining_completed to {state_dict['pretraining_completed']}")
                        
        # Ensure each stage that has the is_pretraining flag reflects the correct state
        for stage in self.stages:
            if hasattr(stage, 'is_pretraining') and stage.is_pretraining:
                # Make sure trainer is connected to this stage
                if hasattr(stage, 'register_trainer') and self.trainer is not None:
                    stage.register_trainer(self.trainer)
                    if self.debug:
                        print(f"[DEBUG] Re-registered trainer with pretraining stage: {stage.name}")
                        
                # If this stage is marked as a pretraining stage, make sure it has the right state based on
                # pretraining_completed in the state_dict
                if 'pretraining_completed' in state_dict:
                    if self.debug:
                        print(f"[DEBUG] Setting pretraining state for stage {stage.name}: {state_dict['pretraining_completed']}")
                        
        # Debug info
        if self.debug:
            print(f"[DEBUG] Loaded curriculum state: Stage {self.current_stage_index} ({self.current_stage.name}), "
                  f"Difficulty {self.current_difficulty:.2f}")
            if is_pretraining_stage_exists:
                print(f"[DEBUG] Pretraining stage exists, pretraining completed: "
                      f"{getattr(self.trainer, 'pretraining_completed', None) if self.trainer else 'No trainer'}")

    def save_curriculum(self, path: str):
        """Save curriculum state to a file using get_state."""
        state = self.get_state()
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_curriculum(self, path: str):
        """Load curriculum state from a file using load_state."""
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        self.load_state(state_dict)

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """
        Get the current statistics of the curriculum.
        Used by the training loop to display progress.
        """
        # This method is being removed in favor of the more comprehensive implementation below
        return self.get_curriculum_stats()
        
    def get_curriculum_stats(self):
        """Get current curriculum statistics for display and logging."""
        current_stage = self.stages[self.current_stage_index]
        
        # Handle pretraining special case
        is_pretraining = hasattr(current_stage, 'is_pretraining') and current_stage.is_pretraining
        if is_pretraining and hasattr(self, 'trainer'):
            pretraining_completed = getattr(self.trainer, 'pretraining_completed', False)
            pretraining_transition = getattr(self.trainer, 'in_transition_phase', False)
            
            # Get progress info from trainer if available
            pretraining_step = getattr(self.trainer, 'training_steps', 0) 
            pretraining_end_step = getattr(self.trainer, '_get_pretraining_end_step', lambda: 0)()
            
            if pretraining_end_step > 0:
                pretraining_progress = min(1.0, pretraining_step / pretraining_end_step)
            else:
                pretraining_progress = 0
                
        else:
            pretraining_completed = True
            pretraining_transition = False
            pretraining_progress = 1.0
            pretraining_step = 0
            pretraining_end_step = 0
        
        stats = {
            "current_stage": self.current_stage_index,  # Adding this for backward compatibility
            "current_stage_index": self.current_stage_index,
            "current_stage_name": current_stage.name,
            "total_stages": len(self.stages),  # Adding this for backward compatibility
            "difficulty_level": self.current_difficulty,
            "total_episodes": self.total_episodes,
            "in_rehearsal": self.current_rehearsal is not None,
            "is_pretraining": is_pretraining and not pretraining_completed,
            "pretraining_completed": pretraining_completed if is_pretraining else None,
            "pretraining_transition": pretraining_transition if is_pretraining else None,
            "pretraining_progress": pretraining_progress if is_pretraining else None,
            "pretraining_step": pretraining_step if is_pretraining else None,
            "pretraining_end_step": pretraining_end_step if is_pretraining else None
        }
        
        # Add current stage statistics
        try:
            stage_stats = current_stage.get_statistics()
            stats["current_stage_stats"] = stage_stats
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Error getting stage statistics: {str(e)}")
            stats["current_stage_stats"] = {}
            
        return stats

    def get_stage_progress(self) -> float:
        """Get progress through current stage (0-1)"""
        if not self.stages:
            return 0.0
        current = self.current_stage_index
        return current / (len(self.stages) - 1) if len(self.stages) > 1 else 1.0

    def get_overall_progress(self) -> Dict[str, float]:
        """Get detailed progress metrics"""
        return {
            'stage_progress': self.get_stage_progress(),
            'difficulty_progress': self.current_difficulty,
            'total_progress': (self.get_stage_progress() + self.current_difficulty) / 2,
            'episodes_completed': self.total_episodes
        }

    def validate_all_stages(self) -> bool:
        """Validate all stages in the curriculum."""
        if not self.stages:
            print("ERROR: No stages in curriculum")
            return False
            
        print("Validating stages:")
        for i, stage in enumerate(self.stages):
            print(f"\nStage {i}: {stage.name}")
            
            # Check required components
            if not stage.state_mutator:
                print(f"ERROR: Stage {stage.name} missing state_mutator")
                return False
            if not stage.reward_function:
                print(f"ERROR: Stage {stage.name} missing reward_function")
                return False
            if not stage.termination_condition:
                print(f"ERROR: Stage {stage.name} missing termination_condition")
                return False
            if not stage.truncation_condition:
                print(f"ERROR: Stage {stage.name} missing truncation_condition")
                return False
                
            # Check progression requirements if not final stage
            if i < len(self.stages) - 1 and not stage.progression_requirements:
                print(f"WARNING: Non-final stage {stage.name} has no progression requirements")
                
            if self.debug:
                print(f"- State mutator: {stage.state_mutator.__class__.__name__}")
                print(f"- Reward function: {stage.reward_function.__class__.__name__}")
                print(f"- Termination condition: {stage.termination_condition.__class__.__name__}")
                print(f"- Truncation condition: {stage.truncation_condition.__class__.__name__}")
                if stage.progression_requirements:
                    print("- Has progression requirements")
                    
        print("\nAll stages validated successfully!")
        return True
