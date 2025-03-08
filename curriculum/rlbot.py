from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import random

from curriculum.base import CurriculumStage, ProgressionRequirements
from .skills import SkillBasedCurriculumStage, SkillModule

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import random
import os

from curriculum.base import CurriculumStage, ProgressionRequirements
from .skills import SkillBasedCurriculumStage, SkillModule
from rlbot.registry import RLBotPackRegistry
# Fix the import - MutatorSequence should be imported from rlgym, not curriculum.mutators
from rlgym.rocket_league.state_mutators import MutatorSequence
from .mutators import CarPositionMutator

class RLBotSkillStage(SkillBasedCurriculumStage):
    """Curriculum stage with both RLBot opponent selection and skill module capabilities."""
    
    def __init__(
        self,
        name: str,
        base_task_state_mutator,
        base_task_reward_function,
        base_task_termination_condition,
        base_task_truncation_condition,
        skill_modules: List[SkillModule] = None,
        base_task_prob: float = 0.6,
        # RLBot-specific parameters
        bot_skill_ranges: Optional[Dict[Tuple[float, float], float]] = None,
        bot_tags: Optional[List[str]] = None,
        allowed_bots: Optional[List[str]] = None,
        progression_requirements: Optional[ProgressionRequirements] = None,
        difficulty_params: Optional[Dict[str, Tuple[float, float]]] = None,
        hyperparameter_adjustments: Optional[Dict[str, float]] = None,
        rlbot_registry: Optional[RLBotPackRegistry] = None,
        bot_performance_threshold: float = 0.4,
        debug: bool = False,
        use_wandb: bool = True
    ):
        # Initialize the parent class
        super().__init__(
            name=name,
            base_task_state_mutator=base_task_state_mutator,
            base_task_reward_function=base_task_reward_function,
            base_task_termination_condition=base_task_termination_condition,
            base_task_truncation_condition=base_task_truncation_condition,
            skill_modules=skill_modules,
            base_task_prob=base_task_prob,
            debug=debug,
            progression_requirements=progression_requirements,
            difficulty_params=difficulty_params,
            hyperparameter_adjustments=hyperparameter_adjustments
        )
        
        # RLBot-specific attributes
        self.bot_skill_ranges = bot_skill_ranges or {}
        self.bot_tags = bot_tags or []
        self.allowed_bots = allowed_bots
        self.rlbot_registry = rlbot_registry
        self.bot_performance_threshold = bot_performance_threshold
        self.use_wandb = use_wandb
        
        # Bot performance tracking
        self.bot_ratings = defaultdict(lambda: 0.0)
        self.bot_wins = defaultdict(int)
        self.bot_losses = defaultdict(int)
        self.bot_encounters = defaultdict(int)
        self.recent_bot_results = defaultdict(lambda: deque(maxlen=5))
        
        # Initialize bot performance tracking
        self.bot_performance = {}
        
        # Debug validation of car position mutators in the constructor
        if debug:
            self._debug_validate_car_position()
            
    def _debug_validate_car_position(self):
        """Debug helper to validate car position mutators exist in the task"""
        print(f"[DEBUG] Validating car position for stage '{self.name}'")
        
        # Check for car position mutator in base task
        has_car_position_mutator = False
        if isinstance(self.state_mutator, MutatorSequence):
            for mutator in self.state_mutator.mutators:
                if isinstance(mutator, CarPositionMutator):
                    has_car_position_mutator = True
                    print(f"[DEBUG] Found CarPositionMutator in base task: {mutator.__class__.__name__}")
                    break
        elif isinstance(self.state_mutator, CarPositionMutator):
            has_car_position_mutator = True
            print(f"[DEBUG] Base task is a CarPositionMutator")
                
        if not has_car_position_mutator:
            print(f"[DEBUG] WARNING: Stage '{self.name}' has no CarPositionMutator in base task!")
            print(f"[DEBUG] This may cause 'NoneType' errors during environment reset")
            
            # Print the mutator sequence to help diagnose
            if isinstance(self.state_mutator, MutatorSequence):
                print(f"[DEBUG] Mutator sequence contains:")
                for i, mutator in enumerate(self.state_mutator.mutators):
                    print(f"[DEBUG]   {i}: {mutator.__class__.__name__}")
            else:
                print(f"[DEBUG] Base mutator is: {self.state_mutator.__class__.__name__}")
                
    def get_environment_config(self, difficulty_level: float) -> Dict[str, Any]:
        """Get environment configuration with proper car positions"""
        config = super().get_environment_config(difficulty_level)
        
        if self.debug:
            print(f"[DEBUG] RLBotSkillStage '{self.name}' getting env config with difficulty={difficulty_level}")
            
            # Validate that state_mutator includes car position setting
            mutators = []
            if isinstance(config["state_mutator"], MutatorSequence):
                mutators = config["state_mutator"].mutators
            else:
                mutators = [config["state_mutator"]]
                
            has_car_pos = False
            for mutator in mutators:
                if isinstance(mutator, CarPositionMutator):
                    has_car_pos = True
                    print(f"[DEBUG] Found CarPositionMutator in config: {mutator.__class__.__name__}")
                    break
                
            if not has_car_pos:
                print(f"[DEBUG] WARNING: Config for '{self.name}' does not have a CarPositionMutator!")
                print(f"[DEBUG] This will likely cause errors with car physics position being None")
        
        return config
        
    def select_opponent_bot(self, difficulty_level: float) -> str:
        """Select an opponent bot based on the current difficulty level."""
        if not self.rlbot_registry:
            return None
            
        # Find compatible bots based on tags and allowed lists
        compatible_bots = self.rlbot_registry.get_compatible_bots(
            tags=self.bot_tags, 
            allowed_list=self.allowed_bots
        )
        
        if not compatible_bots:
            if self.debug:
                print(f"[DEBUG] No compatible bots found for difficulty {difficulty_level}")
            return None
            
        # Filter by skill level appropriate for the current difficulty
        suitable_bots = []
        for bot_name, bot_info in compatible_bots.items():
            skill_level = self.rlbot_registry.get_bot_skill_level(bot_name)
            
            # Check if this bot's skill is appropriate for the current difficulty
            for (min_skill, max_skill), prob in self.bot_skill_ranges.items():
                if min_skill <= skill_level <= max_skill and random.random() < prob:
                    suitable_bots.append(bot_name)
                    
        if not suitable_bots:
            # Fallback - just use any compatible bot
            suitable_bots = list(compatible_bots.keys())
            
        # Choose a bot, considering past performance
        chosen_bot = random.choice(suitable_bots)
        
        if self.debug:
            print(f"[DEBUG] Selected opponent bot '{chosen_bot}' (skill: {self.rlbot_registry.get_bot_skill_level(chosen_bot)}) for difficulty {difficulty_level}")
            
        return chosen_bot
    
    # Adding this method to align with the tests
    def select_opponent_skill_range(self, difficulty_level: float) -> Tuple[float, float]:
        """Select an opponent skill range based on the current difficulty level."""
        # Default to full range if none specified
        if not self.bot_skill_ranges:
            return (0.0, 1.0)
            
        # Find appropriate skill range based on difficulty
        # For simplicity, we'll choose the range with the highest probability
        # that includes our difficulty level
        best_prob = 0
        best_range = (0.0, 1.0)  # Default
        
        for (min_skill, max_skill), prob in self.bot_skill_ranges.items():
            if prob > best_prob:
                best_prob = prob
                best_range = (min_skill, max_skill)
                
        # Choose between low and high range based on current difficulty
        all_ranges = list(self.bot_skill_ranges.keys())
        all_ranges.sort(key=lambda x: x[0])  # Sort by min skill
        
        # For simpler logic, divide ranges into "low" and "high"
        if len(all_ranges) >= 2:
            low_ranges = all_ranges[:len(all_ranges)//2]
            high_ranges = all_ranges[len(all_ranges)//2:]
            
            if difficulty_level < 0.5 and low_ranges:
                return low_ranges[0]
            elif high_ranges:
                return high_ranges[0]
                
        return best_range
        
    def update_bot_performance(self, bot_name: str, agent_won: bool, score_diff: float, difficulty_level: float = 0.5):
        """Update performance metrics for a bot
        
        Args:
            bot_name: Name of the bot
            agent_won: Whether the agent won against this bot
            score_diff: Score difference (agent - bot)
            difficulty_level: The difficulty level used for the bot (added to match test expectations)
        """
        if bot_name not in self.bot_performance:
            self.bot_performance[bot_name] = {
                'total_games': 0,
                'wins': 0,
                'total_reward': 0.0,
                'win_rate': 0.0,
                'avg_reward': 0.0
            }
            
        perf = self.bot_performance[bot_name]
        perf['total_games'] += 1
        perf['wins'] += 1 if agent_won else 0
        perf['total_reward'] += score_diff
        perf['win_rate'] = perf['wins'] / perf['total_games']
        perf['avg_reward'] = perf['total_reward'] / perf['total_games']

        self.bot_encounters[bot_name] += 1
        
        if agent_won:
            self.bot_wins[bot_name] += 1
            self.recent_bot_results[bot_name].append(1)
        else:
            self.bot_losses[bot_name] += 1
            self.recent_bot_results[bot_name].append(0)
            
        # Update bot rating (simple win rate with score difference factor)
        win_rate = self.bot_wins[bot_name] / max(1, self.bot_encounters[bot_name])
        recent_win_rate = sum(self.recent_bot_results[bot_name]) / max(1, len(self.recent_bot_results[bot_name]))
        
        # Score difference influences the rating (closer games matter more)
        score_factor = 1.0
        if score_diff != 0:
            score_factor = min(1.0, 1.0 / (abs(score_diff) / 2.0))
            
        # Combined rating: 70% from recent games, 30% from overall record
        self.bot_ratings[bot_name] = (0.7 * recent_win_rate + 0.3 * win_rate) * score_factor
        
        if self.debug:
            print(f"[DEBUG] Bot '{bot_name}' updated: WL={self.bot_wins[bot_name]}/{self.bot_losses[bot_name]}, " +
                  f"Recent={recent_win_rate:.2f}, Rating={self.bot_ratings[bot_name]:.2f}")
                  
    def get_bot_performance(self, bot_name: str) -> Optional[Dict[str, Any]]:
        return self.bot_performance.get(bot_name)

    def get_matched_bot_difficulty(self, player_skill: float) -> float:
        """Get appropriate bot difficulty based on player skill"""
        # Scale player skill (0-1) to a bot difficulty level
        # This logic can be adjusted based on desired difficulty curve
        base_difficulty = player_skill * 0.8  # Max 0.8 so there's room to grow
        
        # Add some randomness for variety
        variance = 0.1
        difficulty = max(0.0, min(1.0, base_difficulty + random.uniform(-variance, variance)))
        
        if self.debug:
            print(f"[DEBUG] Player skill {player_skill:.2f} mapped to bot difficulty {difficulty:.2f}")
            
        return difficulty
        
    def get_challenging_bots(self) -> List[str]:
        """Return a list of bots that are particularly challenging"""
        challenging_bots = []
        
        # Find bots with low win rates and high encounter count
        for bot_name, encounters in self.bot_encounters.items():
            if encounters >= 5:  # Minimum sample size
                win_rate = self.bot_wins[bot_name] / max(1, encounters)
                if win_rate < self.bot_performance_threshold:
                    challenging_bots.append(bot_name)
                    
        return challenging_bots
        
    # Add for compatibility with tests
    def select_task(self):
        """Select task (base task or skill) based on progression stats."""
        return super()._select_task_for_training()
        
    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        bot_stats = {}
        
        for bot_id in self.bot_performance:
            perf = self.get_bot_performance(bot_id)
            if perf:
                bot_stats[bot_id] = {
                    'games_played': perf['total_games'],
                    'win_rate': perf['win_rate'],
                    'avg_reward': perf['avg_reward']
                }
        
        stats['bot_performance'] = bot_stats
        return stats

    def meets_progression_requirements(self) -> bool:
        # First check base requirements from parent class
        if not super().meets_progression_requirements():
            return False

        # Then check bot-specific requirements if we have any bot performance data
        if self.bot_performance:
            for bot_id, stats in self.bot_performance.items():
                # Only consider bots with enough games played
                if stats['total_games'] >= 5:
                    # Require at least 40% win rate against each bot
                    if stats['win_rate'] < 0.4:
                        return False

        return True