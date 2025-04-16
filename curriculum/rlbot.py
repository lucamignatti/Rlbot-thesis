from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import random
import os

from curriculum.base import CurriculumStage, ProgressionRequirements
from .skills import SkillBasedCurriculumStage, SkillModule
from rlbot.registry import RLBotPackRegistry
from rlgym.rocket_league.state_mutators import MutatorSequence
from .mutators import CarPositionMutator
from rlgym.api import StateMutator, RewardFunction, DoneCondition, AgentID
import wandb

class RLBotSkillStage(CurriculumStage):
    """
    A curriculum stage for RLBot that manages opponent selection based on skill level.
    """
    def __init__(self,
                 name: str,
                 base_task_state_mutator: StateMutator,
                 base_task_reward_function: RewardFunction,
                 base_task_termination_condition: DoneCondition,
                 base_task_truncation_condition: DoneCondition,
                 progression_requirements: Optional[ProgressionRequirements] = None,
                 difficulty_params: Dict[str, Tuple[float, float]] = None,
                 bot_skill_ranges: Dict[Tuple[float, float], float] = None,
                 bot_tags: List[str] = None,
                 allowed_bots: List[str] = None,
                 hyperparameter_adjustments: Dict[str, Any] = None,
                 is_pretraining: bool = False,
                 max_bots_to_track: int = 50):
        """
        Initialize a new RLBotSkillStage.

        Args:
            name: Name of the stage
            base_task_state_mutator: StateMutator for the base task
            base_task_reward_function: RewardFunction for the base task
            base_task_termination_condition: DoneCondition for base task termination
            base_task_truncation_condition: DoneCondition for base task truncation
            progression_requirements: Requirements for progressing to the next stage
            difficulty_params: Dictionary of (min, max) tuples for scalar difficulty parameters
            bot_skill_ranges: Dictionary of (min_skill, max_skill) tuples mapped to selection probabilities
            bot_tags: List of bot tags to prefer (e.g., "defensive", "aerial", etc.)
            allowed_bots: List of specific bot names to use. If None, all compatible bots may be selected.
            hyperparameter_adjustments: Dictionary of hyperparameters to adjust when this stage is reached
            is_pretraining: Whether this stage represents the pre-training phase
            max_bots_to_track: Maximum number of bots to track in performance stats
        """
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

        # Bot selection parameters
        self.bot_skill_ranges = bot_skill_ranges or {}
        self.bot_tags = set(bot_tags) if bot_tags else set()
        self.allowed_bots = set(allowed_bots) if allowed_bots else None

        # Track performance against specific bots with size limit
        self.bot_performance = {}  # Maps bot name -> stats dict
        self.max_bots_to_track = max_bots_to_track
        self.bot_last_used = {}  # Maps bot name -> last time used (episode counter)
        self.total_episodes = 0   # Counter for tracking recency

        # Pre-training flag
        self.is_pretraining = is_pretraining

    def select_opponent_skill_range(self, difficulty: float) -> Tuple[float, float]:
        """
        Select an appropriate bot skill range based on the current difficulty.

        Args:
            difficulty: Current difficulty level (0.0 to 1.0)

        Returns:
            Tuple of (min_skill, max_skill) for bot selection
        """
        if not self.bot_skill_ranges:
            # Default to linear skill progression if no ranges specified
            return (0.0, difficulty)

        # For low difficulty (< 0.7), prefer lower skill ranges
        if difficulty < 0.7:
            # Find ranges with max <= 0.7
            low_skill_ranges = [r for r in self.bot_skill_ranges.keys() if r[1] <= 0.7]
            if low_skill_ranges:
                # Find the range that contains the current difficulty
                for skill_range in low_skill_ranges:
                    min_skill, max_skill = skill_range
                    if min_skill <= difficulty <= max_skill:
                        return skill_range
                # If no exact match, pick the closest low skill range
                return min(low_skill_ranges, key=lambda r: abs((r[0] + r[1])/2 - difficulty))

        # For high difficulty (>= 0.7), prefer higher skill ranges
        if difficulty >= 0.7:
            # Find ranges with min >= 0.7
            high_skill_ranges = [r for r in self.bot_skill_ranges.keys() if r[0] >= 0.7]
            if high_skill_ranges:
                # Pick any high skill range - taking the one with highest weight if multiple exist
                return max(high_skill_ranges, key=lambda r: self.bot_skill_ranges[r])

        # Use weighted random selection based on specified probabilities
        total_weight = sum(self.bot_skill_ranges.values())
        if total_weight <= 0:
            # Fall back to default range if weights are invalid
            return (0.0, difficulty)

        rand_val = random.random() * total_weight
        cumulative = 0

        for skill_range, weight in self.bot_skill_ranges.items():
            cumulative += weight
            if rand_val <= cumulative:
                return skill_range

        # Default fallback - should not reach here
        return list(self.bot_skill_ranges.keys())[0]

    def update_bot_performance(self, bot_name: str, success: bool, reward: float, difficulty: float) -> None:
        """
        Update performance statistics against a specific bot.

        Args:
            bot_name: Name of the opponent bot
            success: Whether the episode was successful
            reward: Episode reward value
            difficulty: Current difficulty level
        """
        # Increment episode counter for recency tracking
        self.total_episodes += 1

        # Update the last used timestamp for this bot
        self.bot_last_used[bot_name] = self.total_episodes

        # Check if we need to remove least recently used bots
        if len(self.bot_performance) >= self.max_bots_to_track and bot_name not in self.bot_performance:
            # Remove the least recently used bot
            lru_bot = min(self.bot_last_used.items(), key=lambda x: x[1])[0]
            if lru_bot != bot_name:  # Don't remove the bot we're about to update
                del self.bot_performance[lru_bot]
                # Keep the timestamp in case we see this bot again

        # Create stats dictionary if this is a new bot (or was removed due to LRU)
        if bot_name not in self.bot_performance:
            self.bot_performance[bot_name] = {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "rewards": [],
                "difficulties": [],
                "win_rate": 0.0,
                "avg_reward": 0.0
            }

        stats = self.bot_performance[bot_name]
        stats["games_played"] += 1
        if success:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        stats["rewards"].append(reward)
        stats["difficulties"].append(difficulty)

        # Keep only the most recent 100 games for each bot to prevent unbounded memory growth
        if len(stats["rewards"]) > 100:
            stats["rewards"] = stats["rewards"][-100:]
            stats["difficulties"] = stats["difficulties"][-100:]

        # Update derived statistics
        stats["win_rate"] = stats["wins"] / stats["games_played"] if stats["games_played"] > 0 else 0.0
        stats["avg_reward"] = sum(stats["rewards"]) / len(stats["rewards"]) if stats["rewards"] else 0.0

    def get_challenging_bots(self, min_games: int = 5, max_win_rate: float = 0.4) -> List[str]:
        """
        Get a list of the most challenging bots based on win rate.

        Args:
            min_games: Minimum number of games required to consider a bot
            max_win_rate: Maximum win rate to consider a bot challenging

        Returns:
            List of challenging bot names, sorted by win rate (lowest first)
        """
        challenging = []

        for bot_name, stats in self.bot_performance.items():
            if stats["games_played"] >= min_games and stats["win_rate"] <= max_win_rate:
                challenging.append((bot_name, stats["win_rate"]))

        # Sort by win rate (ascending)
        challenging.sort(key=lambda x: x[1])

        return [bot[0] for bot in challenging]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated stage statistics including bot performance.

        Returns:
            Dictionary of statistics
        """
        stats = super().get_statistics()

        # Add bot-specific statistics
        stats["bot_performance"] = self.bot_performance

        # Add overall bot statistics
        total_games = sum(bot["games_played"] for bot in self.bot_performance.values())
        total_wins = sum(bot["wins"] for bot in self.bot_performance.values())

        stats["total_bot_games"] = total_games
        stats["overall_win_rate"] = total_wins / total_games if total_games > 0 else 0.0

        return stats

    def meets_progression_requirements(self) -> bool:
        """
        Check if the stage is ready to progress, which requires:
        1. Meeting base progression requirements
        2. Adequate performance against a variety of bots (if applicable)

        Returns:
            True if ready to progress, False otherwise
        """
        if self.is_pretraining:
            # Special handling for pretraining - progress only if PPOTrainer indicates pretraining is done
            # Don't check any regular progression requirements
            if hasattr(self, '_trainer'):
                return getattr(self._trainer, "pretraining_completed", False)
            return False

        # Start with base progression check
        if not self.validate_progression():
            return False

        # If we have no bot performance metrics, we're done
        if not self.bot_performance:
            return True

        # Additional check: Must have a minimum win rate against a variety of bots
        min_win_rate = 0.4
        min_bot_count = min(3, len(self.bot_performance))
        min_games_per_bot = 5

        bots_with_sufficient_games = [
            bot_name for bot_name, stats in self.bot_performance.items()
            if stats["games_played"] >= min_games_per_bot
        ]

        # Not enough bots have been played against
        if len(bots_with_sufficient_games) < min_bot_count:
            return False

        # Must have adequate win rate against most bots
        adequate_bot_count = 0
        for bot_name in bots_with_sufficient_games:
            if self.bot_performance[bot_name]["win_rate"] >= min_win_rate:
                adequate_bot_count += 1

        # At least 2/3 of bots must have adequate win rate
        return adequate_bot_count >= (min_bot_count * 2 // 3)

    def validate_progression(self) -> bool:
        """Special version of validation for pretraining stage"""
        # If this is a pretraining stage, check if we should progress to the next stage
        if self.is_pretraining:
            # Check if pretraining flag is active in PPOTrainer
            if hasattr(self, '_trainer') and self._trainer is not None:
                # If pretraining is completed, we should progress
                return getattr(self._trainer, "pretraining_completed", False)
            return False

        # For regular stages, use the normal validation logic
        return super().validate_progression()

    def register_trainer(self, trainer) -> None:
        """Register a trainer object to enable pretraining stage communication"""
        # Use weakref to avoid circular reference memory leaks
        import weakref
        self._trainer = weakref.proxy(trainer) if trainer is not None else None

    def cleanup(self) -> None:
        """
        Clean up resources to prevent memory leaks.
        This should be called when the stage is completed or no longer needed.
        """
        # Clear large data structures
        self.bot_performance.clear()
        self.bot_last_used.clear()
