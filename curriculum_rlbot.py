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

class RLBotStage(CurriculumStage):
    """Curriculum stage specialized for RLBot opponents"""
    def __init__(self, name, state_mutator, reward_function, termination_condition, truncation_condition,
                 bot_skill_ranges=None, bot_tags=None, allowed_bots=None, progression_requirements=None,
                 difficulty_params=None, hyperparameter_adjustments=None):
        super().__init__(name, state_mutator, reward_function, termination_condition, truncation_condition,
                        bot_skill_ranges=bot_skill_ranges, bot_tags=bot_tags, allowed_bots=allowed_bots,
                        progression_requirements=progression_requirements, difficulty_params=difficulty_params,
                        hyperparameter_adjustments=hyperparameter_adjustments)
        
        # Additional tracking specific to bot opponents
        self.bot_win_rates = {}
        self.recent_bot_performance = defaultdict(lambda: deque(maxlen=20))  # Last 20 games per bot
        self.bot_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
    
    def update_bot_stats(self, bot_id, outcome):
        """Update performance tracking for a specific bot"""
        if outcome == 'win':
            self.bot_stats[bot_id]['wins'] += 1
            self.recent_bot_performance[bot_id].append(1)
        elif outcome == 'loss':
            self.bot_stats[bot_id]['losses'] += 1
            self.recent_bot_performance[bot_id].append(-1)
        else:  # draw
            self.bot_stats[bot_id]['draws'] += 1
            self.recent_bot_performance[bot_id].append(0)
        
        # Update win rate
        stats = self.bot_stats[bot_id]
        total_games = stats['wins'] + stats['losses'] + stats['draws']
        if total_games > 0:
            self.bot_win_rates[bot_id] = stats['wins'] / total_games
    
    def get_bot_performance(self, bot_id):
        """Get performance metrics for a specific bot"""
        if bot_id not in self.bot_stats:
            return None
        
        stats = self.bot_stats[bot_id]
        total_games = stats['wins'] + stats['losses'] + stats['draws']
        
        if total_games == 0:
            return None
        
        # Calculate recent win rate
        recent = list(self.recent_bot_performance[bot_id])
        recent_win_rate = sum(1 for x in recent if x == 1) / len(recent) if recent else 0
        
        return {
            'win_rate': stats['wins'] / total_games,
            'recent_win_rate': recent_win_rate,
            'total_games': total_games,
            'wins': stats['wins'],
            'losses': stats['losses'],
            'draws': stats['draws']
        }
    
    def get_challenging_bots(self, min_games=10):
        """Find bots that are currently challenging for the agent"""
        challenging = []
        
        for bot_id in self.bot_stats:
            perf = self.get_bot_performance(bot_id)
            if perf and perf['total_games'] >= min_games:
                if perf['recent_win_rate'] < 0.5 or perf['win_rate'] < 0.4:
                    challenging.append({
                        'bot_id': bot_id,
                        'win_rate': perf['win_rate'],
                        'recent_win_rate': perf['recent_win_rate'],
                        'total_games': perf['total_games']
                    })
        
        return sorted(challenging, key=lambda x: x['recent_win_rate'])

def create_rlbot_curriculum(debug=False):
    """Create a curriculum for training against RLBotPack bots"""
    
    # Stage 1: Basic Ball Control against Easy Bots
    stage1 = RLBotStage(
        name="Basic Ball Control",
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
        bot_skill_ranges={(0.1, 0.3): 0.7, (0.3, 0.5): 0.3},  # 70% easy bots, 30% medium
        bot_tags=['beginner', 'easy'],  # Prefer bots tagged as beginner/easy
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
        bot_skill_ranges={(0.3, 0.5): 0.5, (0.5, 0.7): 0.5},  # Equal mix of medium and harder bots
        bot_tags=['intermediate'],  # Prefer intermediate bots
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.45,
            min_episodes=100,
            max_std_dev=0.35,
            required_consecutive_successes=3
        )
    )
    
    # Stage 3: Advanced Play against Challenging Bots
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
        bot_tags=['advanced', 'expert', 'pro'],  # Prefer challenging bots
        progression_requirements=None  # Final stage
    )
    
    # Create the curriculum manager with the stages
    manager = CurriculumManager(
        stages=[stage1, stage2, stage3],
        progress_thresholds={
            'success_rate': 0.55,
            'avg_reward': 0.45
        },
        max_rehearsal_stages=2,
        rehearsal_decay_factor=0.6,
        evaluation_window=50,
        debug=debug
    )
    
    return manager