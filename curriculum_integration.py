from rlgym.api import StateMutator
from rlgym.rocket_league.state_mutators import (
    MutatorSequence, FixedTeamSizeMutator, KickoffMutator
)
from rlgym.rocket_league.done_conditions import (
    GoalCondition, AnyCondition, TimeoutCondition, NoTouchTimeoutCondition
)
from rlgym.rocket_league.reward_functions import (
    CombinedReward, GoalReward, TouchReward
)
from rewards import (
    BallProximityReward, BallToGoalDistanceReward, BallVelocityToGoalReward,
    TouchBallReward, TouchBallToGoalAccelerationReward, AlignBallToGoalReward,
    PlayerVelocityTowardBallReward, KRCReward
)
from curriculum import CurriculumManager, CurriculumStage, ProgressionRequirements
import numpy as np

class BallVariationMutator(StateMutator):
    def __init__(self, velocity_variance=0.1, position_variance=0.0):
        self.velocity_variance = velocity_variance
        self.position_variance = position_variance

    def apply(self, state, shared_info):
        if self.velocity_variance > 0:
            # Access linear_velocity without underscore
            random_velocity = (np.random.random(3) * 2 - 1) * self.velocity_variance
            state.ball.linear_velocity = random_velocity

        if self.position_variance > 0:
            # Access position without underscore
            random_position = np.array([
                (np.random.random() * 2 - 1) * self.position_variance,
                (np.random.random() * 2 - 1) * self.position_variance,
                100  # Fixed reasonable height
            ])
            state.ball.position = random_position

class CarBoostMutator(StateMutator):
    def __init__(self, boost_amount=100):
        self.boost_amount = boost_amount

    def apply(self, state, shared_info):
        # Access boost_amount without underscore
        for car in state.cars.values():
            car.boost_amount = self.boost_amount

def create_basic_curriculum(debug=False):
    """
    Creates a basic curriculum that starts with simple tasks and gradually
    increases difficulty. Each stage introduces new challenges while maintaining
    core skills.

    The curriculum has three main stages:
    1. Ball Control - Focus on basic ball interaction and movement
    2. Tactical Play - Develop positioning and strategic thinking
    3. Advanced Play - Master high-speed play and complex maneuvers

    Args:
        debug: Whether to print debug information during stage transitions
    """
    # Common conditions used across stages
    timeout_condition = TimeoutCondition(300)  # 5 minutes max per episode
    no_touch_timeout = NoTouchTimeoutCondition(30)  # Reset if ball not touched for 30s
    common_truncation = AnyCondition(timeout_condition, no_touch_timeout)

    # Stage 1: Ball Control
    ball_control_stage = CurriculumStage(
        name="Ball Control",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),  # Start with 1v1
            KickoffMutator()
        ),
        reward_function=KRCReward([
            (TouchBallReward(), 1.0),  # Encourage any ball contact
            (BallProximityReward(dispersion=1.2, density=1.0), 0.8),  # Stay close to ball
            (PlayerVelocityTowardBallReward(), 0.6)  # Move efficiently
        ], team_spirit=0.1),  # Low team spirit - focus on individual skill
        termination_condition=GoalCondition(),
        truncation_condition=common_truncation,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,  # 60% goal success
            min_avg_reward=0.4,
            min_episodes=100,
            max_std_dev=0.3,
            required_consecutive_successes=3
        ),
        difficulty_params={
            "ball_velocity": (0.7, 1.0),  # Start with slower ball
            "car_velocity": (0.8, 1.0)     # Slightly reduced car speed
        }
    )

    # Stage 2: Tactical Play
    tactical_stage = CurriculumStage(
        name="Tactical Play",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),  # Progress to 2v2
            KickoffMutator()
        ),
        reward_function=KRCReward([
            (TouchBallToGoalAccelerationReward(), 1.0),  # Reward purposeful hits
            (AlignBallToGoalReward(dispersion=1.1, density=1.0), 0.8),  # Position strategically
            (BallToGoalDistanceReward(), 0.6)  # Control field position
        ], team_spirit=0.3),  # Increased team coordination
        termination_condition=GoalCondition(),
        truncation_condition=common_truncation,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.65,  # Higher success needed
            min_avg_reward=0.5,
            min_episodes=150,
            max_std_dev=0.25,
            required_consecutive_successes=4
        ),
        difficulty_params={
            "boost_strength": (0.8, 1.0),  # Gradually increase boost power
            "car_velocity": (0.9, 1.0)     # Faster car movement
        }
    )

    # Stage 3: Advanced Play
    advanced_stage = CurriculumStage(
        name="Advanced Play",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            KickoffMutator()
        ),
        reward_function=KRCReward([
            (TouchBallToGoalAccelerationReward(), 1.0),
            (BallVelocityToGoalReward(), 0.8),
            (AlignBallToGoalReward(dispersion=1.0, density=1.2), 0.6),
            (BallToGoalDistanceReward(), 0.4)
        ], team_spirit=0.4),  # High team coordination
        termination_condition=GoalCondition(),
        truncation_condition=common_truncation,
        progression_requirements=None,  # Final stage has no progression
        difficulty_params={
            "boost_strength": (1.0, 1.0),  # Full boost power
            "car_velocity": (1.0, 1.0),    # Full car speed
            "ball_velocity": (1.0, 1.0)    # Full ball physics
        }
    )

    # Create and return curriculum manager
    return CurriculumManager(
        stages=[ball_control_stage, tactical_stage, advanced_stage],
        evaluation_window=50,  # Look at last 50 episodes for progression
        debug=debug
    )
