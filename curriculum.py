import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from rlgym.api import StateMutator, RewardFunction, DoneCondition, RLGym
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.state_mutators import MutatorSequence
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.obs_builders import DefaultObs


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
        hyperparameter_adjustments: Dict[str, float] = None
    ):
        """
        Initialize a curriculum stage.

        Args:
            name: Descriptive name of this stage
            state_mutator: Configures the environment for this stage
            reward_function: Stage-specific reward function
            termination_condition: When to end episodes (success)
            truncation_condition: When to cut off episodes (failure/timeout)
            progress_metrics: List of metric names to track for progression
            difficulty_params: Dict of params that can vary with difficulty level, with (min, max) values
            hyperparameter_adjustments: Changes to make to training hyperparameters during this stage
        """
        self.name = name
        self.state_mutator = state_mutator
        self.reward_function = reward_function
        self.termination_condition = termination_condition
        self.truncation_condition = truncation_condition

        # Set default values if not provided
        self.progress_metrics = progress_metrics or ["episode_reward", "success_rate"]
        self.difficulty_params = difficulty_params or {}
        self.hyperparameter_adjustments = hyperparameter_adjustments or {}

        # Internal tracking
        self.episode_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.rewards_history = []
        self.moving_success_rate = 0.0
        self.moving_avg_reward = 0.0

    def get_config_with_difficulty(self, difficulty_level: float) -> Dict[str, Any]:
        """
        Returns the stage configuration with parameters adjusted for difficulty level.

        Args:
            difficulty_level: Value between 0.0 (easiest) and 1.0 (hardest)

        Returns:
            Dictionary with configuration parameters adjusted for difficulty
        """
        # Clamp difficulty between 0 and 1
        difficulty = max(0.0, min(1.0, difficulty_level))

        # Interpolate each difficulty parameter based on the current level
        params = {}
        for param_name, (min_val, max_val) in self.difficulty_params.items():
            params[param_name] = min_val + difficulty * (max_val - min_val)

        return params

    def update_statistics(self, episode_metrics: Dict[str, Any]) -> None:
        """
        Update stage progress statistics based on episode results.

        Args:
            episode_metrics: Dictionary of metrics from the completed episode
        """
        self.episode_count += 1

        # Track success/failure
        if episode_metrics.get("success", False):
            self.success_count += 1
        elif episode_metrics.get("timeout", False):
            self.failure_count += 1

        # Track rewards
        if "episode_reward" in episode_metrics:
            self.rewards_history.append(episode_metrics["episode_reward"])
            # Keep only the last 100 rewards
            if len(self.rewards_history) > 100:
                self.rewards_history.pop(0)

        # Update moving averages
        if self.episode_count > 0:
            self.moving_success_rate = self.success_count / self.episode_count

        if self.rewards_history:
            self.moving_avg_reward = np.mean(self.rewards_history)

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
        return {
            "name": self.name,
            "episodes": self.episode_count,
            "successes": self.success_count,
            "failures": self.failure_count,
            "success_rate": self.moving_success_rate,
            "avg_reward": self.moving_avg_reward
        }


class CurriculumManager:
    """
    Manages progression through curriculum stages based on agent performance.
    Handles stage transitions, rehearsal of previous stages, and hyperparameter adjustments.
    """
    def __init__(
        self,
        stages: List[CurriculumStage],
        progress_thresholds: Dict[str, float] = None,
        max_rehearsal_stages: int = 3,
        rehearsal_decay_factor: float = 0.5,
        evaluation_window: int = 100,
        debug = False
    ):
        """
        Initialize the curriculum manager.

        Args:
            stages: List of curriculum stages in order of increasing difficulty
            progress_thresholds: Dictionary mapping metric names to threshold values for progression
            max_rehearsal_stages: Maximum number of previous stages to include in rehearsal
            rehearsal_decay_factor: How quickly to reduce probability of older stages
            evaluation_window: Number of episodes to consider when evaluating progression
        """
        if not stages:
            raise ValueError("At least one curriculum stage must be provided")

        self.debug = debug

        self.stages = stages
        self.current_stage_index = 0
        self.max_rehearsal_stages = max_rehearsal_stages
        self.rehearsal_decay_factor = rehearsal_decay_factor
        self.evaluation_window = evaluation_window

        # Default progress thresholds if not specified
        self.progress_thresholds = progress_thresholds or {
            "success_rate": 0.7,
            "avg_reward": 0.8  # Normalized expected reward
        }

        # Trainer reference (set later via register_trainer)
        self.trainer = None

        # Stage transition history
        self.stage_transitions = []

        # Track total episodes completed
        self.total_episodes = 0

        # Difficulty progression within a stage (0.0 to 1.0)
        self.current_difficulty = 0.0
        self.difficulty_increase_rate = 0.01  # Per evaluation

        # Validate all stages during initialization
        self.validate_all_stages()

    def debug_print(self, message: str):
        if self.debug:
            print(message)

    def validate_all_stages(self) -> None:
        """Check that all stages have the required components and can function properly."""
        for i, stage in enumerate(self.stages):
            stage_num = i + 1
            if self.debug:
                self.debug_print(f"Validating stage {stage_num}/{len(self.stages)}: {stage.name}")
            try:
                self._validate_stage(stage)
                if self.debug:
                    self.debug_print(f"✓ Stage {stage.name} passed validation")
            except Exception as e:
                raise RuntimeError(
                    f"Stage {stage.name} (#{stage_num}) failed validation: {str(e)}"
                ) from e

    def _validate_stage(self, stage: CurriculumStage):
        """
        Validate a curriculum stage by creating a minimal test environment and running one step.
        This ensures the stage components work together as expected.
        """
        try:
            self.debug_print(f"  - Creating minimal test environment for {stage.name}")

            # Create a minimal curriculum configuration with this stage's components
            curriculum_config = {
                "state_mutator": stage.state_mutator,
                "reward_function": stage.reward_function,
                "termination_condition": stage.termination_condition,
                "truncation_condition": stage.truncation_condition,
            }

            # Create a minimal test environment
            env = RLGym(
                state_mutator=curriculum_config["state_mutator"],
                obs_builder=DefaultObs(),
                action_parser=RepeatAction(LookupTableAction(), repeats=8),
                reward_fn=curriculum_config["reward_function"],
                termination_cond=curriculum_config["termination_condition"],
                truncation_cond=curriculum_config["truncation_condition"],
                transition_engine=RocketSimEngine()
            )

            self.debug_print(f"  - Running minimal test sequence for {stage.name}")

            # Reset the environment
            self.debug_print(f"    - Resetting environment")
            obs_dict = env.reset()

            # Verify we have at least one agent
            if not obs_dict:
                raise ValueError(f"No agents found in environment after reset")

            # Take a simple action with each agent
            actions_dict = {}
            for agent_id in obs_dict.keys():
                actions_dict[agent_id] = np.array([0])  # Index 0 in lookup table

            # Step the environment with correct return value unpacking
            self.debug_print(f"    - Stepping environment")
            try:
                result = env.step(actions_dict)

                # Handle different RLGym versions (4 or 5 return values)
                if len(result) == 5:
                    next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = result
                    self.debug_print(f"    - Step successful (5-value return format)")
                else:
                    next_obs_dict, reward_dict, terminated_dict, truncated_dict = result
                    self.debug_print(f"    - Step successful (4-value return format)")
            except Exception as e:
                self.debug_print(f"    - Step failed: {e.__class__.__name__}: {str(e)}")
                raise

            # Clean up
            env.close()

            self.debug_print(f"  - ✓ Stage {stage.name} validation successful")

        except Exception as e:
            self.debug_print(f"  - ✗ Failed during validation: {e.__class__.__name__}: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())
            raise ValueError(f"Stage failed validation: {str(e)}")

    def register_trainer(self, trainer) -> None:
        """Register the PPO trainer for hyperparameter adjustment."""
        self.trainer = trainer

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
        """
        Update statistics based on completed episode metrics.

        Args:
            episode_metrics: Dictionary of metrics from the completed episode
        """
        # Update current stage statistics
        current_stage = self.stages[self.current_stage_index]
        current_stage.update_statistics(episode_metrics)

        # Track overall progress
        self.total_episodes += 1

        # Every N episodes, check if we should progress to the next stage
        if self.total_episodes % self.evaluation_window == 0:
            self._evaluate_progression()

    def _evaluate_progression(self) -> bool:
        """
        Evaluate if the agent should progress to the next curriculum stage.

        Returns:
            Boolean indicating whether progression occurred
        """
        if self.current_stage_index >= len(self.stages) - 1:
            # Already at the final stage
            return False

        current_stage = self.stages[self.current_stage_index]
        stats = current_stage.get_statistics()

        # First, try to increase difficulty within the current stage
        if self.current_difficulty < 1.0:
            # Check if we're meeting the thresholds at the current difficulty
            meets_thresholds = True
            for metric_name, threshold in self.progress_thresholds.items():
                if metric_name == "success_rate" and stats["success_rate"] < threshold:
                    meets_thresholds = False
                    break
                elif metric_name == "avg_reward" and stats["avg_reward"] < threshold:
                    meets_thresholds = False
                    break

            # If meeting thresholds, increase difficulty
            if meets_thresholds:
                self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_increase_rate)
                self.debug_print(f"Increasing difficulty in stage {current_stage.name} to {self.current_difficulty:.2f}")
                return False  # No stage progression yet

        # Only progress to next stage if current one is at max difficulty and thresholds are met
        if self.current_difficulty >= 0.95:  # Close enough to max
            # Check if we're meeting all thresholds
            ready_for_next_stage = True
            for metric_name, threshold in self.progress_thresholds.items():
                if metric_name == "success_rate" and stats["success_rate"] < threshold:
                    ready_for_next_stage = False
                    break
                elif metric_name == "avg_reward" and stats["avg_reward"] < threshold:
                    ready_for_next_stage = False
                    break

            if ready_for_next_stage:
                self._progress_to_next_stage()
                return True

        return False

    def _progress_to_next_stage(self) -> None:
        """Progress to the next curriculum stage."""
        old_stage = self.stages[self.current_stage_index].name
        self.current_stage_index += 1
        new_stage = self.stages[self.current_stage_index].name

        # Reset difficulty for the new stage
        self.current_difficulty = 0.0

        # Record the transition
        self.stage_transitions.append({
            "episode": self.total_episodes,
            "from_stage": old_stage,
            "to_stage": new_stage,
            "timestamp": np.datetime64('now')
        })

        self.debug_print(f"Curriculum progressed from '{old_stage}' to '{new_stage}' at episode {self.total_episodes}")

        # Apply hyperparameter adjustments for the new stage
        self._adjust_hyperparameters()

    def _get_rehearsal_probability(self) -> float:
        """Get the probability of using a rehearsal stage instead of the current stage."""
        # Base probability depends on how many stages we've progressed through
        base_prob = 0.3  # 30% chance of rehearsal when possible

        # Scale down if we're still early in the curriculum
        progress_factor = min(1.0, self.current_stage_index / max(len(self.stages) - 1, 1))
        return base_prob * progress_factor

    def _select_rehearsal_stage(self) -> int:
        """
        Select a previous stage for rehearsal using a decay-based probability.

        Returns:
            Index of the selected previous stage
        """
        # Determine how many stages are available for rehearsal
        available_stages = min(self.current_stage_index, self.max_rehearsal_stages)

        if available_stages <= 0:
            return 0  # No rehearsal possible

        # Create decay probabilities favoring more recent stages
        probs = np.array([self.rehearsal_decay_factor ** (available_stages - i - 1)
                          for i in range(available_stages)])
        probs = probs / np.sum(probs)  # Normalize

        # Select a stage based on these probabilities
        selected_idx = np.random.choice(available_stages, p=probs)

        # Convert to actual stage index (counting backward from current)
        return self.current_stage_index - available_stages + selected_idx

    def _adjust_hyperparameters(self) -> None:
        """Adjust training hyperparameters based on the current stage."""
        if self.trainer is None:
            return

        current_stage = self.stages[self.current_stage_index]
        adjustments = current_stage.hyperparameter_adjustments

        if not adjustments:
            return

        # Apply hyperparameter adjustments
        if "lr_actor" in adjustments and hasattr(self.trainer, "actor_optimizer"):
            for param_group in self.trainer.actor_optimizer.param_groups:
                param_group["lr"] = adjustments["lr_actor"]
            self.debug_print(f"Adjusted actor learning rate to {adjustments['lr_actor']}")

        if "lr_critic" in adjustments and hasattr(self.trainer, "critic_optimizer"):
            for param_group in self.trainer.critic_optimizer.param_groups:
                param_group["lr"] = adjustments["lr_critic"]
            self.debug_print(f"Adjusted critic learning rate to {adjustments['lr_critic']}")

        if "entropy_coef" in adjustments and hasattr(self.trainer, "entropy_coef"):
            self.trainer.entropy_coef = adjustments["entropy_coef"]
            self.debug_print(f"Adjusted entropy coefficient to {adjustments['entropy_coef']}")

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get statistics about the curriculum progress."""
        return {
            "current_stage": self.current_stage_index,
            "current_stage_name": self.stages[self.current_stage_index].name,
            "total_stages": len(self.stages),
            "difficulty_level": self.current_difficulty,
            "total_episodes": self.total_episodes,
            "stage_transitions": self.stage_transitions,
            "current_stage_stats": self.stages[self.current_stage_index].get_statistics()
        }

    def reset_current_stage_stats(self) -> None:
        """Reset statistics for the current stage."""
        self.stages[self.current_stage_index].reset_statistics()
