"""Skill-based curriculum implementation."""
import numpy as np
from typing import Dict, Tuple, List, Optional
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.api import StateMutator, RewardFunction, DoneCondition
from typing import List, Dict, Any, Tuple, Union, Optional, Callable
from .base import CurriculumStage, ProgressionRequirements
from .mutators import (
    BallTowardGoalSpawnMutator, BallPositionMutator, BallVelocityMutator,
    CarPositionMutator, CarBoostMutator
)
import random
import collections

class SkillModule:
    """A module representing a specific skill to be learned with configurable difficulty."""

    def __init__(
        self,
        name: str,
        state_mutator: StateMutator,
        reward_function: RewardFunction,
        termination_condition: DoneCondition,
        truncation_condition: DoneCondition,
        difficulty_params: Dict[str, Tuple[float, float]],
        success_threshold: float = 0.7
    ):
        self.name = name
        self.state_mutator = state_mutator
        self.reward_function = reward_function
        self.termination_condition = termination_condition
        self.truncation_condition = truncation_condition
        self.difficulty_params = difficulty_params
        self.success_threshold = success_threshold

        # Statistics tracking
        self.episode_count = 0
        self.success_count = 0
        self.rewards_history = collections.deque(maxlen=500)
        self.success_rate = 0.0
        self.success_history = collections.deque(maxlen=500)  # Track success/failure sequence

    def update_statistics(self, metrics: Dict[str, Any]):
        """Update skill statistics with new episode results"""
        self.episode_count += 1
        success = metrics.get('success', False)
        if success:
            self.success_count += 1

        self.success_history.append(success)
        self.rewards_history.append(metrics.get('episode_reward', 0.0))
        self.success_rate = self.success_count / max(1, self.episode_count)

    def get_config(self, difficulty: float) -> Dict[str, Any]:
        """Get difficulty-adjusted configuration"""
        config = {}
        difficulty = max(0.0, min(1.0, difficulty))  # Clamp between 0 and 1

        for param_name, (min_val, max_val) in self.difficulty_params.items():
            config[param_name] = min_val + difficulty * (max_val - min_val)

        return config

    def meets_mastery_criteria(self) -> bool:
        """Check if the skill meets mastery criteria"""
        min_episodes = 20  # Minimum episodes needed for reliable assessment
        if self.episode_count < min_episodes:
            return False

        # Check success rate meets threshold
        if self.success_rate < self.success_threshold:
            return False

        # Check for stability in recent performance
        recent_rewards = self.rewards_history[-10:]  # Last 10 episodes
        if len(recent_rewards) >= 10:
            std_dev = np.std(recent_rewards)
            if std_dev > 0.3:  # High variance indicates unstable performance
                return False

        return True

    def get_consecutive_successes(self) -> int:
        """Count current streak of consecutive successful episodes"""
        count = 0
        for success in reversed(self.success_history):
            if success:
                count += 1
            else:
                break
        return count


class SkillBasedCurriculumStage(CurriculumStage):
    """A curriculum stage that combines a base task with specific skill modules."""

    def __init__(
        self,
        name: str,
        base_task_state_mutator: StateMutator,
        base_task_reward_function: RewardFunction,
        base_task_termination_condition: DoneCondition,
        base_task_truncation_condition: DoneCondition,
        skill_modules: List[SkillModule] = None,
        base_task_prob: float = 0.7,
        debug: bool = False,
        progression_requirements: Optional[ProgressionRequirements] = None,
        difficulty_params: Optional[Dict[str, Tuple[float, float]]] = None,
        hyperparameter_adjustments: Optional[Dict[str, float]] = None
    ):
        # Initialize with base task configuration
        super().__init__(
            name=name,
            state_mutator=base_task_state_mutator,
            reward_function=base_task_reward_function,
            termination_condition=base_task_termination_condition,
            truncation_condition=base_task_truncation_condition,
            progression_requirements=progression_requirements,
            difficulty_params=difficulty_params,
            hyperparameter_adjustments=hyperparameter_adjustments
        )

        self.skill_modules = skill_modules or []
        self.base_task_prob = base_task_prob
        self.selected_skill = None
        self.debug = debug

        # Additional tracking for base task vs skills
        self.base_task_episodes = 0
        self.base_task_successes = 0
        self.skill_episodes = {skill.name: 0 for skill in self.skill_modules} if self.skill_modules else {}
        self.skill_successes = {skill.name: 0 for skill in self.skill_modules} if self.skill_modules else {}

        # Debug validation of car position mutators - ensure at least one exists in the base_task
        if self.debug:
            has_car_position_mutator = False
            if isinstance(base_task_state_mutator, MutatorSequence):
                for mutator in base_task_state_mutator.mutators:
                    if isinstance(mutator, CarPositionMutator):
                        has_car_position_mutator = True
                        break
            else:
                has_car_position_mutator = isinstance(base_task_state_mutator, CarPositionMutator)

            if not has_car_position_mutator:
                print(f"[DEBUG] WARNING: Stage '{name}' has no CarPositionMutator in base task!")
                print(f"[DEBUG] Mutator sequence: {base_task_state_mutator}")

    def get_environment_config(self, difficulty_level: float) -> Dict[str, Any]:
        """Returns the stage configuration with parameters adjusted for difficulty level."""
        # Determine if we should use a base task or a specialized skill module
        is_base_task, selected_skill = self._select_task_for_training()
        self.selected_skill = selected_skill

        if is_base_task:
            # Use base task
            config = super().get_environment_config(difficulty_level)
            config["task_type"] = "base"  # Add this to identify task type

            # Debug car position in base task
            if self.debug:
                print(f"[DEBUG] Stage: {self.name} - Using base task with difficulty {difficulty_level}")

                # Check for car position mutator
                mutators = []
                if isinstance(self.state_mutator, MutatorSequence):
                    mutators = self.state_mutator.mutators
                else:
                    mutators = [self.state_mutator]

                has_car_pos = False
                for mutator in mutators:
                    if isinstance(mutator, CarPositionMutator):
                        has_car_pos = True
                        print(f"[DEBUG] Base task has CarPositionMutator: {mutator}")

                if not has_car_pos:
                    print(f"[DEBUG] WARNING: Base task for '{self.name}' does not have a CarPositionMutator!")

            return config
        else:
            # Use the selected skill module
            skill_config = self.selected_skill.get_config(difficulty_level)

            if self.debug:
                print(f"[DEBUG] Stage: {self.name} - Using skill module: {self.selected_skill.name}")
                print(f"[DEBUG] Skill mutator: {self.selected_skill.state_mutator}")
                print(f"[DEBUG] Skill config: {skill_config}")

            return {
                "stage_name": f"{self.name} - {self.selected_skill.name}",
                "state_mutator": self.selected_skill.state_mutator,
                "reward_function": self.selected_skill.reward_function,
                "termination_condition": self.selected_skill.termination_condition,
                "truncation_condition": self.selected_skill.truncation_condition,
                "difficulty_level": difficulty_level,
                "difficulty_params": skill_config,
                "task_type": "skill",
                "skill_name": self.selected_skill.name
            }

    def update_statistics(self, episode_data):
        """Update the stage's statistics with new episode data"""
        super().update_statistics(episode_data)

        # Track whether this was a base task or skill module episode
        is_base_task = episode_data.get('is_base_task', True)

        if is_base_task:
            self.base_task_episodes += 1
            if episode_data.get('success', False):
                self.base_task_successes += 1
        else:
            # Update for the specific skill
            skill_name = episode_data.get('skill_name')
            if skill_name:
                if skill_name not in self.skill_episodes:
                    self.skill_episodes[skill_name] = 0
                    self.skill_successes[skill_name] = 0

                self.skill_episodes[skill_name] += 1
                if episode_data.get('success', False):
                    self.skill_successes[skill_name] += 1

        # Also update the skill module's statistics if one was selected
        if 'skill_name' in episode_data and not is_base_task:
            skill_name = episode_data['skill_name']
            for skill in self.skill_modules:
                if skill.name == skill_name:
                    skill.update_statistics(episode_data)
                    break

    def _select_task_for_training(self):
        """Select task (base task or skill) based on progression stats."""
        # If no skill modules, always use base task
        if not self.skill_modules:
            return True, None

        # Base probability to use base task
        if random.random() < self.base_task_prob:
            return True, None

        # Weight skills by their inverse success rate
        skill_weights = []
        for skill in self.skill_modules:
            if skill.episode_count == 0:
                weight = 2.0  # Higher weight for unexplored skills
            else:
                # Higher weight for skills with lower success rates
                success_rate = skill.success_rate
                weight = 1.0 - min(success_rate, 0.8)  # Cap at 0.8
            skill_weights.append(weight)

        # Normalize weights
        sum_weights = sum(skill_weights)
        if sum_weights > 0:
            skill_weights = [w/sum_weights for w in skill_weights]
        else:
            skill_weights = [1.0/len(self.skill_modules) for _ in self.skill_modules]

        # Choose a skill based on weights
        selected_skill = np.random.choice(self.skill_modules, p=skill_weights)
        return False, selected_skill

    def select_task(self):
        """Public method to select a task - for test compatibility"""
        return self._select_task_for_training()

    def meets_progression_requirements(self) -> bool:
        """Check if this stage meets the requirements to progress to the next stage."""
        # First check base task requirements from parent class
        if not super().validate_progression():
            if self.debug:
                print("[DEBUG] Base task requirements not met")
            return False

        # Then check if we have enough skill episodes and mastery
        if self.skill_modules:
            min_success_rate = self.progression_requirements.min_success_rate
            min_episodes = self.progression_requirements.min_episodes
            for skill in self.skill_modules:
                if skill.episode_count < min_episodes:
                    if self.debug:
                        print(f"[DEBUG] Not enough episodes for skill '{skill.name}': {skill.episode_count} < {min_episodes}")
                    return False

                # Check skill success rate against same criteria as base task
                if skill.success_rate < min_success_rate:
                    if self.debug:
                        print(f"[DEBUG] Skill '{skill.name}' success rate too low: {skill.success_rate:.2f} < {min_success_rate}")
                    return False

                # Check consecutive successes for skills
                skill_consecutive = skill.get_consecutive_successes()
                if skill_consecutive < self.progression_requirements.required_consecutive_successes:
                    if self.debug:
                        print(f"[DEBUG] Skill '{skill.name}' needs more consecutive successes: {skill_consecutive} < {self.progression_requirements.required_consecutive_successes}")
                    return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed stage statistics"""
        stats = super().get_statistics()

        # Add task-specific statistics
        stats["base_task"] = {
            "episodes": self.base_task_episodes,
            "successes": self.base_task_successes,
            "success_rate": self.base_task_successes / max(1, self.base_task_episodes)
        }

        # Add skill-specific statistics
        stats["skills"] = {}
        for skill in self.skill_modules:
            stats["skills"][skill.name] = {
                "episodes": skill.episode_count,
                "successes": skill.success_count,
                "success_rate": skill.success_rate,
                "mastered": skill.meets_mastery_criteria()
            }

        return stats
        
    def cleanup(self) -> None:
        """
        Clean up resources to prevent memory leaks.
        This should be called when the stage is completed or no longer needed.
        """
        # Clear dictionaries that track skill statistics
        self.skill_episodes.clear()
        self.skill_successes.clear()
        
        # Clean up each skill module's large data structures
        for skill in self.skill_modules:
            if hasattr(skill, 'rewards_history'):
                skill.rewards_history.clear()
            if hasattr(skill, 'success_history'):
                skill.success_history.clear()
                
        # Break any potential circular references
        self.selected_skill = None
        
        # Force garbage collection
        import gc
        gc.collect()
