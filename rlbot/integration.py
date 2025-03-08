"""RLBot integration for RLGym environments."""
import os
import sys
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import multiprocessing as mp
from multiprocessing import Process, Pipe
import concurrent.futures
from collections import defaultdict
from collections import deque
import random
from curriculum import CurriculumStage, ProgressionRequirements
from .registry import RLBotPackRegistry
from rlgym.rocket_league.api import GameState
from rlgym.api import RLGym, StateMutator, RewardFunction, DoneCondition
from envs.vectorized import VectorizedEnv
from envs.rlbot_vectorized import RLBotVectorizedEnv

# Helper functions to manage bot compatibility
def load_bot_skills() -> Dict[str, float]:
    """Load skill mapping for validated bots"""
    skills = {}
    try:
        with open("bot_skills.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    bot, skill = line.split("=")
                    skills[bot.strip()] = float(skill)
    except FileNotFoundError:
        print("Warning: bot_skills.txt not found")
    return skills

def load_disabled_bots() -> set:
    """Load list of disabled/incompatible bots"""
    disabled = set()
    try:
        with open("disabled_bots.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith(":"):
                    disabled.add(line.split("(")[0].strip())  # Remove parenthetical notes
    except FileNotFoundError:
        print("Warning: disabled_bots.txt not found")
    return disabled

def is_bot_compatible(bot_name: str) -> bool:
    """Check if a bot is compatible with the curriculum"""
    disabled_bots = load_disabled_bots()
    skills = load_bot_skills()

    # Check if bot is explicitly disabled
    if bot_name in disabled_bots:
        return False

    # Check if bot is validated and has skill mapping
    if bot_name not in skills:
        return False

    return True

def get_bot_skill(bot_name: str) -> Optional[float]:
    """Get skill level for a bot if available"""
    skills = load_bot_skills()
    return skills.get(bot_name)

def get_compatible_bots(min_skill: float = 0.0, max_skill: float = 1.0) -> Dict[str, float]:
    """Get compatible bots within skill range"""
    skills = load_bot_skills()
    disabled = load_disabled_bots()

    compatible = {}
    for bot, skill in skills.items():
        if bot not in disabled and min_skill <= skill <= max_skill:
            compatible[bot] = skill

    return compatible

class RLBotStage(CurriculumStage):
    """Extended curriculum stage with RLBot integration"""
    def __init__(
        self,
        name: str,
        state_mutator: StateMutator,
        reward_function: RewardFunction,
        termination_condition: DoneCondition,
        truncation_condition: DoneCondition,
        bot_skill_ranges: Optional[Dict[Tuple[float, float], float]] = None,
        bot_tags: Optional[List[str]] = None,
        progression_requirements: Optional[ProgressionRequirements] = None,
        difficulty_params: Optional[Dict[str, Tuple[float, float]]] = None,
        hyperparameter_adjustments: Optional[Dict[str, float]] = None
    ):
        # Type check inputs for better error messages
        if not isinstance(reward_function, RewardFunction):
            raise TypeError(f"reward_function must be a RewardFunction instance, got {type(reward_function)}")

        super().__init__(
            name=name,
            state_mutator=state_mutator,
            reward_function=reward_function,
            termination_condition=termination_condition,
            truncation_condition=truncation_condition,
            progression_requirements=progression_requirements,
            difficulty_params=difficulty_params,
            hyperparameter_adjustments=hyperparameter_adjustments
        )

        # RLBot-specific configuration
        self.bot_skill_ranges = bot_skill_ranges or {(0.3, 0.7): 1.0}
        self.bot_tags = set(bot_tags or [])
        self.bot_performance = {}
        self.recent_bot_win_rate = {}

    def select_opponent(self, difficulty_level: float) -> Optional[str]:
        """Select an opponent based on the current difficulty level"""
        skill_min, skill_max = self.select_opponent_skill_range(difficulty_level)

        # Find compatible bots within this skill range
        compatible_bots = get_compatible_bots(skill_min, skill_max, self.bot_tags)

        if not compatible_bots:
            return None

        # Select randomly from compatible bots
        return random.choice(compatible_bots)

    def select_opponent_skill_range(self, difficulty_level: float) -> Tuple[float, float]:
        """Select a skill range based on the current difficulty level"""
        # Find the range closest to the current difficulty
        best_range = None
        best_match = -1

        for skill_range, probability in self.bot_skill_ranges.items():
            min_skill, max_skill = skill_range
            mid_point = (min_skill + max_skill) / 2

            # Calculate how well this range matches the difficulty
            match_quality = 1 - abs(difficulty_level - mid_point)

            if match_quality > best_match:
                best_match = match_quality
                best_range = skill_range

        # Return the best matching range, or a default range if none found
        return best_range if best_range else (0.3, 0.7)

    def update_bot_performance(self, bot_id: str, win: bool, reward: float, difficulty: float) -> None:
        """Update performance statistics for a specific bot"""
        if bot_id not in self.bot_performance:
            self.bot_performance[bot_id] = {
                'wins': 0,
                'losses': 0,
                'rewards': [],
                'difficulties': []
            }
            self.recent_bot_win_rate[bot_id] = deque(maxlen=10)  # Track recent 10 matches

        stats = self.bot_performance[bot_id]

        if win:
            stats['wins'] += 1
            self.recent_bot_win_rate[bot_id].append(1)
        else:
            stats['losses'] += 1
            self.recent_bot_win_rate[bot_id].append(0)

        stats['rewards'].append(reward)
        stats['difficulties'].append(difficulty)

    def get_bot_performance(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a bot"""
        if bot_id not in self.bot_performance:
            return None

        stats = self.bot_performance[bot_id]
        total_games = stats['wins'] + stats['losses']

        if total_games == 0:
            return None

        recent_results = self.recent_bot_win_rate[bot_id]

        return {
            'wins': stats['wins'],
            'losses': stats['losses'],
            'total_games': total_games,
            'win_rate': stats['wins'] / total_games,
            'recent_win_rate': sum(recent_results) / len(recent_results) if recent_results else 0,
            'avg_reward': sum(stats['rewards']) / len(stats['rewards']) if stats['rewards'] else 0,
            'avg_difficulty': sum(stats['difficulties']) / len(stats['difficulties']) if stats['difficulties'] else 0
        }

    def get_challenging_bots(self, threshold: float = 0.4) -> List[str]:
        """Find bots where win rate is below threshold"""
        challenging_bots = []

        for bot_id, stats in self.bot_performance.items():
            perf = self.get_bot_performance(bot_id)
            if perf and perf['win_rate'] <= threshold:
                challenging_bots.append(bot_id)

        return challenging_bots
