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
    """Create curriculum with improved stage definitions and progression"""
    stages = []

    # Stage 1: Basic Ball Control
    stage1 = CurriculumStage(
        name="Ball Touch Training",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            KickoffMutator(),
            BallVariationMutator(velocity_variance=0.1),
            CarBoostMutator(boost_amount=100)
        ),
        reward_function=CombinedReward(
            (TouchBallReward(), 5.0),
            (BallProximityReward(max_distance=5000), 1.0),
            (PlayerVelocityTowardBallReward(), 0.5)
        ),
        termination_condition=GoalCondition(),
        truncation_condition=AnyCondition(
            TimeoutCondition(120.),
            NoTouchTimeoutCondition(15.)
        ),
        difficulty_params={
            "velocity_variance": (0.1, 0.7),
            "boost_amount": (100, 50)
        },
        hyperparameter_adjustments={
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "entropy_coef": 0.02
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.65,
            min_avg_reward=0.7,
            min_episodes=50,
            max_std_dev=0.3
        )
    )
    stages.append(stage1)

    # Stage 2: Directional Control
    stage2 = CurriculumStage(
        name="Directional Control",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            KickoffMutator(),
            BallVariationMutator(
                velocity_variance=0.7,
                position_variance=1000
            ),
            CarBoostMutator(boost_amount=75)
        ),
        reward_function=CombinedReward(
            (TouchBallToGoalAccelerationReward(), 4.0),
            (BallVelocityToGoalReward(), 2.0),
            (AlignBallToGoalReward(), 1.0),
            (TouchBallReward(), 1.0)
        ),
        termination_condition=GoalCondition(),
        truncation_condition=AnyCondition(
            TimeoutCondition(180.),
            NoTouchTimeoutCondition(20.)
        ),
        difficulty_params={
            "velocity_variance": (0.7, 1.5),
            "position_variance": (1000, 2000),
            "boost_amount": (75, 50)
        },
        hyperparameter_adjustments={
            "lr_actor": 2e-4,
            "lr_critic": 8e-4,
            "entropy_coef": 0.015
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.7,
            min_avg_reward=0.8,
            min_episodes=75,
            max_std_dev=0.25,
            required_consecutive_successes=5
        )
    )
    stages.append(stage2)

    # Stage 3: Solo Scoring
    stage3 = CurriculumStage(
        name="Solo Scoring",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            KickoffMutator(),
            BallVariationMutator(
                velocity_variance=1.5,
                position_variance=2000
            ),
            CarBoostMutator(boost_amount=50)
        ),
        reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (BallVelocityToGoalReward(), 3.0),
            (BallToGoalDistanceReward(), 2.0),
            (TouchBallToGoalAccelerationReward(), 2.0)
        ),
        termination_condition=GoalCondition(),
        truncation_condition=AnyCondition(
            TimeoutCondition(240.),
            NoTouchTimeoutCondition(25.)
        ),
        difficulty_params={
            "velocity_variance": (1.5, 3.0),
            "position_variance": (2000, 3000),
            "boost_amount": (50, 33)
        },
        hyperparameter_adjustments={
            "lr_actor": 1e-4,
            "lr_critic": 5e-4,
            "entropy_coef": 0.01
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.75,
            min_avg_reward=1.0,
            min_episodes=100,
            max_std_dev=0.2,
            required_consecutive_successes=7
        )
    )
    stages.append(stage3)

    # Stage 4: 1v1 Training
    stage4 = CurriculumStage(
        name="1v1 Training",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator(),
            CarBoostMutator(boost_amount=100)  # Full boost for competitive play
        ),
        reward_function=CombinedReward(
            (GoalReward(), 15.0),
            (KRCReward([
                (AlignBallToGoalReward(dispersion=1.1, density=1.0), 1.0),
                (BallProximityReward(dispersion=0.8, density=1.2), 0.8),
                (PlayerVelocityTowardBallReward(), 0.6)
            ], team_spirit=0.0), 8.0),
            (KRCReward([
                (TouchBallToGoalAccelerationReward(), 1.0),
                (TouchBallReward(), 0.8),
                (BallVelocityToGoalReward(), 0.6)
            ], team_spirit=0.0), 6.0)
        ),
        termination_condition=GoalCondition(),
        truncation_condition=AnyCondition(
            TimeoutCondition(300.),
            NoTouchTimeoutCondition(30.)
        ),
        difficulty_params={},  # Difficulty handled by opponent skill
        hyperparameter_adjustments={
            "lr_actor": 5e-5,
            "lr_critic": 2e-4,
            "entropy_coef": 0.005
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,  # Lower for 1v1
            min_avg_reward=0.5,  # Lower for 1v1
            min_episodes=150,
            max_std_dev=0.4,  # Higher std dev is expected
            required_consecutive_successes=3
        )
    )
    stages.append(stage4)

    # Stage 5: 2v2 Training
    stage5 = CurriculumStage(
        name="2v2 Training",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            KickoffMutator(),
            CarBoostMutator(boost_amount=100)
        ),
        reward_function=CombinedReward(
            (GoalReward(), 15.0),
            (KRCReward([
                (AlignBallToGoalReward(dispersion=1.1, density=1.0), 1.0),
                (BallProximityReward(dispersion=0.8, density=1.2), 0.8),
                (PlayerVelocityTowardBallReward(), 0.6)
            ], team_spirit=0.3), 8.0),
            (KRCReward([
                (TouchBallToGoalAccelerationReward(), 1.0),
                (TouchBallReward(), 0.8),
                (BallVelocityToGoalReward(), 0.6)
            ], team_spirit=0.3), 6.0),
            (KRCReward([
                (AlignBallToGoalReward(dispersion=1.1, density=1.0), 1.0),
                (BallProximityReward(dispersion=0.8, density=1.2), 0.8)
            ], team_spirit=0.3), 4.0),
            (KRCReward([
                (BallToGoalDistanceReward(
                    offensive_dispersion=0.6,
                    defensive_dispersion=0.4,
                    offensive_density=1.0,
                    defensive_density=1.0
                ), 1.0),
                (BallProximityReward(dispersion=0.7, density=1.0), 0.4)
            ], team_spirit=0.3), 2.0)
        ),
        termination_condition=GoalCondition(),
        truncation_condition=AnyCondition(
            TimeoutCondition(300.),
            NoTouchTimeoutCondition(30.)
        ),
        difficulty_params={},
        hyperparameter_adjustments={
            "lr_actor": 3e-5,
            "lr_critic": 1e-4,
            "entropy_coef": 0.003
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5,
            min_avg_reward=0.4,
            min_episodes=200,
            max_std_dev=0.5,
            required_consecutive_successes=3
        )
    )
    stages.append(stage5)

    return CurriculumManager(
        stages=stages,
        max_rehearsal_stages=3,
        rehearsal_decay_factor=0.5,
        debug=debug
    )
