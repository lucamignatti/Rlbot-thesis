"""RLBot curriculum implementation."""
from typing import Dict, List, Any, Tuple, Optional, Set
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.api import StateMutator, RewardFunction, DoneCondition
from curriculum import CurriculumManager, CurriculumStage, ProgressionRequirements
from rewards import (
    BallProximityReward, BallToGoalDistanceReward, BallVelocityToGoalReward,
    TouchBallReward, TouchBallToGoalAccelerationReward, AlignBallToGoalReward,
    PlayerVelocityTowardBallReward, KRCReward
)
from collections import defaultdict, deque
import os
import random
import numpy as np

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

# Global bot skill cache
_bot_skills = {}
_bot_tags = {}
_compatible_bots = None

def _load_bot_skills() -> None:
    """Load bot skills from file"""
    global _bot_skills
    if _bot_skills:  # Already loaded
        return
        
    try:
        with open("bot_skills.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=")
                    if len(parts) == 2:
                        bot_name, skill = parts
                        try:
                            _bot_skills[bot_name.strip()] = float(skill)
                        except ValueError:
                            pass
    except FileNotFoundError:
        print("Warning: Could not find bot_skills.txt")
        
def _load_bot_tags() -> None:
    """Load bot tags from file"""
    global _bot_tags
    if _bot_tags:  # Already loaded
        return
        
    try:
        with open("bot_metadata.txt", "r") as f:
            bot_id = None
            tags = []
            
            for line in f:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    # Save previous bot if any
                    if bot_id:
                        _bot_tags[bot_id] = set(tags)
                        
                    # Start new bot
                    bot_id = line[1:-1]
                    tags = []
                elif ":" in line:
                    key, value = line.split(":", 1)
                    if key.strip() == "tags":
                        tags = [t.strip() for t in value.split(",")]
                        
            # Save final bot
            if bot_id:
                _bot_tags[bot_id] = set(tags)
    except FileNotFoundError:
        print("Warning: Could not find bot_metadata.txt")

def _load_compatible_bots() -> None:
    """Load list of compatible bots"""
    global _compatible_bots
    if _compatible_bots is not None:  # Already loaded
        return
        
    _compatible_bots = set()
    try:
        if os.path.exists("validated_bots.txt"):
            with open("validated_bots.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        _compatible_bots.add(line)
        elif os.path.exists("runnable_bots.txt"):
            with open("runnable_bots.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        _compatible_bots.add(line)
        else:
            # Use all bots with skills as fallback
            _load_bot_skills()
            _compatible_bots = set(_bot_skills.keys())
            
        # Filter out disabled bots
        if os.path.exists("disabled_bots.txt"):
            disabled = set()
            with open("disabled_bots.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        disabled.add(line)
            _compatible_bots -= disabled
    except Exception as e:
        print(f"Warning: Error loading compatible bots: {e}")
        _compatible_bots = set()

def get_bot_skill(bot_id: str) -> Optional[float]:
    """Get skill rating for bot"""
    _load_bot_skills()
    return _bot_skills.get(bot_id)
    
def get_bot_tags(bot_id: str) -> Set[str]:
    """Get tags for bot"""
    _load_bot_tags()
    return _bot_tags.get(bot_id, set())
    
def is_bot_compatible(bot_id: str) -> bool:
    """Check if bot is compatible with the system"""
    _load_compatible_bots()
    return bot_id in _compatible_bots
    
def get_compatible_bots(min_skill: float = 0.0, max_skill: float = 1.0, 
                       required_tags: Optional[List[str]] = None) -> List[str]:
    """Get bots within skill range with required tags"""
    _load_bot_skills()
    _load_compatible_bots()
    _load_bot_tags()
    
    required_tag_set = set(required_tags or [])
    result = []
    
    for bot_id in _compatible_bots:
        # Check skill range
        skill = _bot_skills.get(bot_id)
        if skill is None or not (min_skill <= skill <= max_skill):
            continue
            
        # Check tags if specified
        if required_tag_set:
            bot_tag_set = _bot_tags.get(bot_id, set())
            if not required_tag_set.issubset(bot_tag_set):
                continue
                
        result.append(bot_id)
        
    return result

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