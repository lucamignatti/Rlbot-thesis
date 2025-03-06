"""RLBot curriculum implementation."""
from typing import Dict, List, Any, Tuple, Optional
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from curriculum import CurriculumManager, CurriculumStage, ProgressionRequirements
from rewards import (
    BallProximityReward, BallToGoalDistanceReward, BallVelocityToGoalReward,
    TouchBallReward, TouchBallToGoalAccelerationReward, AlignBallToGoalReward,
    PlayerVelocityTowardBallReward, KRCReward
)
from collections import defaultdict, deque
import os

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
    """Curriculum stage with RLBot opponent selection."""
    
    def __init__(self, name, state_mutator, reward_function, termination_condition, truncation_condition,
                 bot_skill_ranges=None, bot_tags=None, allowed_bots=None, progression_requirements=None,
                 difficulty_params=None, hyperparameter_adjustments=None):
        super().__init__(name, state_mutator, reward_function, termination_condition, truncation_condition,
                        bot_skill_ranges=bot_skill_ranges, bot_tags=bot_tags, allowed_bots=allowed_bots,
                        progression_requirements=progression_requirements, difficulty_params=difficulty_params,
                        hyperparameter_adjustments=hyperparameter_adjustments)
        
        # Additional tracking specific to bot opponents
        self.bot_stats = {}  # Use dict instead of defaultdict for easier serialization
    
    def update_bot_stats(self, bot_id: str, outcome: str):
        """Update performance tracking for a specific bot"""
        if bot_id not in self.bot_stats:
            self.bot_stats[bot_id] = {
                'wins': 0, 
                'losses': 0, 
                'draws': 0,
                'total_reward': 0.0,
                'episodes': 0,
                'recent_outcomes': []  # Last 20 outcomes
            }
        
        stats = self.bot_stats[bot_id]
        if outcome == 'win':
            stats['wins'] += 1
            stats['recent_outcomes'].append(1)
        elif outcome == 'loss':
            stats['losses'] += 1
            stats['recent_outcomes'].append(-1)
        else:  # draw
            stats['draws'] += 1
            stats['recent_outcomes'].append(0)
            
        # Keep only last 20 outcomes
        if len(stats['recent_outcomes']) > 20:
            stats['recent_outcomes'] = stats['recent_outcomes'][-20:]
    
    def get_bot_performance(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific bot"""
        if bot_id not in self.bot_stats:
            return None
        
        stats = self.bot_stats[bot_id]
        total_games = stats['wins'] + stats['losses'] + stats['draws']
        
        if total_games == 0:
            return None
        
        # Calculate recent win rate from last 20 games
        recent = stats['recent_outcomes']
        recent_wins = sum(1 for x in recent if x == 1)
        recent_win_rate = recent_wins / len(recent) if recent else 0
        
        return {
            'win_rate': stats['wins'] / total_games,
            'recent_win_rate': recent_win_rate,
            'total_games': total_games,
            'wins': stats['wins'],
            'losses': stats['losses'],
            'draws': stats['draws'],
            'avg_reward': stats['total_reward'] / total_games if total_games > 0 else 0.0
        }
    
    def update_bot_performance(self, bot_id: str, win: bool, reward: float):
        """Track performance against specific bots."""
        # Update base class stats
        super().update_bot_performance(
            bot_id=bot_id,
            win=win,
            reward=reward,
            difficulty=get_bot_skill(bot_id) or 0.5
        )
        
        # Update our detailed stats
        if bot_id not in self.bot_stats:
            self.bot_stats[bot_id] = {
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'total_reward': 0.0,
                'episodes': 0,
                'recent_outcomes': []
            }
        
        stats = self.bot_stats[bot_id]
        stats['episodes'] += 1
        stats['total_reward'] += reward
        
        if win:
            self.update_bot_stats(bot_id, 'win')
        else:
            self.update_bot_stats(bot_id, 'loss')
    
    def get_challenging_opponents(self, count: int = 3) -> List[str]:
        """Get list of challenging opponents to focus on."""
        challenging = []
        
        for bot_id in self.bot_stats:
            stats = self.get_bot_performance(bot_id)
            if stats and stats['total_games'] >= 10:
                # Consider a bot challenging if recent performance is poor
                if stats['recent_win_rate'] < 0.5 or stats['win_rate'] < 0.4:
                    challenging.append((bot_id, stats['recent_win_rate']))
        
        # Sort by recent win rate ascending (hardest bots first)
        challenging.sort(key=lambda x: x[1])
        return [bot_id for bot_id, _ in challenging[:count]]
    
    def select_opponent(self, difficulty: float) -> Optional[str]:
        """Select an appropriate opponent based on current difficulty."""
        min_skill, max_skill = self.select_opponent_skill_range(difficulty)
        
        # Get compatible bots in skill range
        available_bots = get_compatible_bots(min_skill, max_skill)
        if not available_bots:
            if hasattr(self, 'debug') and self.debug:
                print(f"Warning: No compatible bots found in skill range {min_skill}-{max_skill}")
            return None
            
        # Sort by skill to prefer bots closer to target difficulty
        target_skill = (min_skill + max_skill) / 2
        sorted_bots = sorted(
            available_bots.items(),
            key=lambda x: abs(x[1] - target_skill)
        )
        
        return sorted_bots[0][0] if sorted_bots else None  # Return name of best matching bot
    
    def validate_opponent(self, bot_id: str) -> bool:
        """Verify that a bot is valid and compatible."""
        return is_bot_compatible(bot_id)


def create_rlbot_curriculum(debug=False):
    """Create a curriculum for training against RLBotPack bots."""
    
    # Stage 1: Basic Play against Easy Bots
    stage1 = RLBotStage(
        name="Basic Play",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator()
        ),
        reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 0.5),
            (BallToGoalDistanceReward(), 0.3)
        ),
        termination_condition=GoalCondition(),
        truncation_condition=TimeoutCondition(300),
        bot_skill_ranges={(0.0, 0.3): 0.7, (0.3, 0.5): 0.3},  # 70% easy bots, 30% medium
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.4,
            min_episodes=50,
            max_std_dev=0.3,
            required_consecutive_successes=3
        )
    )
    
    # Stage 2: Intermediate Play against Medium Bots
    stage2 = RLBotStage(
        name="Intermediate Play",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator()
        ),
        reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 0.3),
            (BallToGoalDistanceReward(), 0.4),
            (BallVelocityToGoalReward(), 0.3)
        ),
        termination_condition=GoalCondition(),
        truncation_condition=TimeoutCondition(300),
        bot_skill_ranges={(0.3, 0.5): 0.5, (0.5, 0.7): 0.5},  # Equal mix of medium/harder bots
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.45,
            min_episodes=100,
            max_std_dev=0.35,
            required_consecutive_successes=3
        )
    )
    
    # Stage 3: Advanced Play against Hard Bots
    stage3 = RLBotStage(
        name="Advanced Play",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator()
        ),
        reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 0.2),
            (BallToGoalDistanceReward(), 0.3),
            (BallVelocityToGoalReward(), 0.3),
            (TouchBallToGoalAccelerationReward(), 0.2)
        ),
        termination_condition=GoalCondition(),
        truncation_condition=TimeoutCondition(300),
        bot_skill_ranges={(0.5, 0.7): 0.3, (0.7, 0.9): 0.7},  # Mostly hard bots
        progression_requirements=None  # Final stage
    )
    
    # Create the curriculum manager
    manager = CurriculumManager(
        stages=[stage1, stage2, stage3],
        evaluation_window=50,
        debug=debug
    )
    
    return manager