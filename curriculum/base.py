"""Core curriculum learning functionality."""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rlgym.api import StateMutator, RewardFunction, DoneCondition
from dataclasses import dataclass
import wandb
import pickle

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
        testing = False,
        use_wandb: bool = True
    ):
        if not stages:
            raise ValueError("At least one curriculum stage must be provided")

        self.debug = debug
        self.use_wandb = use_wandb
        self.stages = stages
        self.current_stage_index = 0
        self.evaluation_window = evaluation_window
        self.progress_thresholds = progress_thresholds or {
            "success_rate": 0.7,
            "avg_reward": 0.8
        }

        # Initialize tracking variables
        self.trainer = None
        self.stage_transitions = []
        self.total_episodes = 0
        self.current_difficulty = 0.0
        self.difficulty_increase_rate = 0.01
        # Initialize step counter
        self._last_wandb_step = 0

        if max_rehearsal_stages < 0:
            raise ValueError("max_rehearsal_stages must be non-negative")

        if rehearsal_decay_factor <= 0 or rehearsal_decay_factor > 1:
            raise ValueError("rehearsal_decay_factor must be in range (0, 1]")

        self.max_rehearsal_stages = max_rehearsal_stages
        self.rehearsal_decay_factor = rehearsal_decay_factor
        self._testing = testing

    def register_trainer(self, trainer) -> None:
        """Register the PPO trainer for hyperparameter adjustment."""
        self.trainer = trainer

    def _get_current_step(self) -> Optional[int]:
        """Get current training step for wandb logging"""
        if self.trainer is None:
            # If no trainer is registered, use internal counter
            self._last_wandb_step += 1
            return self._last_wandb_step
            
        # Use trainer's step counter as source of truth when available
        if hasattr(self.trainer, '_true_training_steps'):
            # Use the true training steps counter from the trainer (most accurate)
            return self.trainer._true_training_steps
        elif hasattr(self.trainer, 'training_steps') and hasattr(self.trainer, 'training_step_offset'):
            # Fall back to calculated training steps if true counter isn't available
            return self.trainer.training_steps + self.trainer.training_step_offset
        else:
            # Last resort: increment internal counter if no trainer counters are available
            self._last_wandb_step += 1
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
        if not self._testing and current_step <= self._last_wandb_step:
            if self.debug:
                print(f"Skipping wandb log for step {current_step} (≤ {self._last_wandb_step})")
            return
            
        # Log with the valid step
        wandb.log(metrics, step=current_step)
        
        # Remember this step for next time
        self._last_wandb_step = current_step

    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get the environment configuration for the current training step,
        potentially including rehearsal of earlier stages.
        """
        # Determine if we should use a rehearsal stage instead
        if self.current_stage_index > 0 and self.max_rehearsal_stages > 0:
            if np.random.random() < self._get_rehearsal_probability():
                # Select a previous stage for rehearsal
                rehearsal_index = self._select_rehearsal_stage()
                selected_stage = self.stages[rehearsal_index]
                is_rehearsal = True
            else:
                # Use current stage
                selected_stage = self.stages[self.current_stage_index]
                is_rehearsal = False
        else:
            # Always use current stage if it's the first one
            selected_stage = self.stages[self.current_stage_index]
            is_rehearsal = False

        # Get difficulty-adjusted configuration
        difficulty = 0.0 if is_rehearsal else self.current_difficulty
        difficulty_config = selected_stage.get_config_with_difficulty(difficulty)

        # Build complete configuration
        config = {
            "stage_name": selected_stage.name,
            "state_mutator": selected_stage.state_mutator,
            "reward_function": selected_stage.reward_function,
            "termination_condition": selected_stage.termination_condition,
            "truncation_condition": selected_stage.truncation_condition,
            "is_rehearsal": is_rehearsal,
            "difficulty_level": difficulty,
            "difficulty_params": difficulty_config
        }

        return config

    def update_progression_stats(self, episode_metrics: Dict[str, Any]) -> None:
        """Update statistics based on completed episode metrics."""
        current_stage = self.stages[self.current_stage_index]
        current_stage.update_statistics(episode_metrics)
        self.total_episodes += 1

        if self.use_wandb:
            self._log_to_wandb({
                "curriculum/current_stage": self.current_stage_index,
                "curriculum/stage_name": current_stage.name,
                "curriculum/current_difficulty": self.current_difficulty,
                "curriculum/success_rate": current_stage.moving_success_rate,
                "curriculum/avg_reward": current_stage.moving_avg_reward,
                "curriculum/total_episodes": self.total_episodes,
                "curriculum/stage_progress": self.get_stage_progress(),
                "curriculum/overall_progress": self.get_overall_progress()["total_progress"],
                "curriculum/episodes_in_stage": current_stage.episode_count,
                "curriculum/consecutive_successes": current_stage.get_consecutive_successes()
            })

        if self.total_episodes % self.evaluation_window == 0:
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
        self.stage_transitions.append({
            "episode": self.total_episodes,
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

    def save_curriculum(self, path: str):
        """Save curriculum state"""
        save_data = {
            'current_stage_index': self.current_stage_index,
            'current_difficulty': self.current_difficulty,
            'total_episodes': self.total_episodes,
            'stage_transitions': self.stage_transitions,
            'stages_data': [{
                'name': stage.name,
                'episode_count': stage.episode_count,
                'success_count': stage.success_count,
                'failure_count': stage.failure_count,
                'rewards_history': stage.rewards_history,
                'moving_success_rate': stage.moving_success_rate,
                'moving_avg_reward': stage.moving_avg_reward
            } for stage in self.stages]
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    def load_curriculum(self, path: str):
        """Load curriculum state"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        self.current_stage_index = save_data['current_stage_index']
        self.current_difficulty = save_data['current_difficulty']
        self.total_episodes = save_data['total_episodes']
        self.stage_transitions = save_data['stage_transitions']

        # Restore stage data
        for stage, stage_data in zip(self.stages, save_data['stages_data']):
            stage.episode_count = stage_data['episode_count']
            stage.success_count = stage_data['success_count']
            stage.failure_count = stage_data['failure_count']
            stage.rewards_history = stage_data['rewards_history']
            stage.moving_success_rate = stage_data['moving_success_rate']
            stage.moving_avg_reward = stage_data['moving_avg_reward']

        # Log the loaded curriculum state
        if self.use_wandb:
            current_stage = self.stages[self.current_stage_index]
            self._log_to_wandb({
                "curriculum/loaded_state/current_stage": self.current_stage_index,
                "curriculum/loaded_state/stage_name": current_stage.name,
                "curriculum/loaded_state/difficulty": self.current_difficulty,
                "curriculum/loaded_state/total_episodes": self.total_episodes,
                "curriculum/loaded_state/success_rate": current_stage.moving_success_rate
            })

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """
        Get the current statistics of the curriculum.
        Used by the training loop to display progress.
        """
        current_stage = self.stages[self.current_stage_index]
        current_stage_stats = current_stage.get_statistics()

        return {
            "current_stage": self.current_stage_index,
            "total_stages": len(self.stages),
            "current_stage_name": current_stage.name,
            "difficulty_level": self.current_difficulty,
            "total_episodes": self.total_episodes,
            "current_stage_stats": current_stage_stats,
            "stage_progress": self.get_stage_progress(),
            "overall_progress": self.get_overall_progress()
        }

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
