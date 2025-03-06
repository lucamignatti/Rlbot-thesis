import os
from typing import Dict, Set, Optional, List, Any, Tuple, Union
import numpy as np
import torch
import wandb
from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from env_base import EnvBase
from curriculum_config import DEFAULT_CURRICULUM_CONFIG, BOT_SKILL_RANGES, STAGE_REQUIREMENTS
from curriculum import CurriculumManager, CurriculumStage, ProgressionRequirements
from rewards import (
    BallProximityReward, BallToGoalDistanceReward, BallVelocityToGoalReward,
    TouchBallReward, TouchBallToGoalAccelerationReward, AlignBallToGoalReward,
    PlayerVelocityTowardBallReward, KRCReward
)
from curriculum_mutators import (
    BallTowardGoalSpawnMutator, BallPositionMutator, BallVelocityMutator,
    CarPositionMutator, CarBoostMutator, BallTouchedCondition,
    KickoffPerformanceReward, DefensivePositioningReward, OffensiveClearReward
)

class BallVariationMutator(StateMutator):
    """Mutator that adds variation to ball position and velocity"""
    def __init__(self,
                 position_range: Tuple[float, float] = (-100, 100),
                 height_range: Tuple[float, float] = (100, 300),
                 velocity_range: Tuple[float, float] = (-300, 300),
                 angular_velocity_range: Tuple[float, float] = (-1, 1)):
        self.position_range = position_range
        self.height_range = height_range
        self.velocity_range = velocity_range
        self.angular_velocity_range = angular_velocity_range

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        # Add slight randomization to ball position
        if hasattr(state, 'ball') and state.ball is not None:
            # Get current position and add variation
            curr_pos = np.array(state.ball.position)
            variation = np.array([
                np.random.uniform(*self.position_range),
                np.random.uniform(*self.position_range),
                np.random.uniform(*self.height_range)
            ])
            state.ball.position = curr_pos + variation

            # Add velocity variation
            curr_vel = np.array(state.ball.linear_velocity)
            velocity_variation = np.array([
                np.random.uniform(*self.velocity_range),
                np.random.uniform(*self.velocity_range),
                np.random.uniform(*self.velocity_range)
            ])
            state.ball.linear_velocity = curr_vel + velocity_variation

            # Add angular velocity variation
            curr_ang = np.array(state.ball.angular_velocity)
            angular_variation = np.array([
                np.random.uniform(*self.angular_velocity_range),
                np.random.uniform(*self.angular_velocity_range),
                np.random.uniform(*self.angular_velocity_range)
            ])
            state.ball.angular_velocity = curr_ang + angular_variation

def create_basic_curriculum(debug=False):
    """
    Creates a basic curriculum focused on fundamental game mechanics.

    Args:
        debug: Whether to print debug information

    Returns:
        CurriculumManager configured with basic stages
    """
    # Create the stages list
    stages = []

    # Common termination/truncation conditions
    termination = GoalCondition()
    truncation = AnyCondition(
        TimeoutCondition(300.),
        NoTouchTimeoutCondition(30.)
    )

    # Stage 1: Basic Ball Control
    basic_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    )
    basic_reward = CombinedReward(
        (GoalReward(), 10.0),
        (TouchBallReward(), 0.3),
        (BallProximityReward(), 0.3),
        (PlayerVelocityTowardBallReward(), 0.2)
    )
    stages.append(CurriculumStage(
        name="Basic Ball Control",
        state_mutator=basic_mutator,
        reward_function=basic_reward,
        termination_condition=termination,
        truncation_condition=truncation,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.5,
            min_episodes=50,
            max_std_dev=1.0
        )
    ))

    # Stage 2: Speed Control
    speed_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    )
    speed_reward = CombinedReward(
        (GoalReward(), 12.0),
        (TouchBallToGoalAccelerationReward(), 0.4),
        (BallVelocityToGoalReward(), 0.3),
        (PlayerVelocityTowardBallReward(), 0.3)
    )
    stages.append(CurriculumStage(
        name="Speed Control",
        state_mutator=speed_mutator,
        reward_function=speed_reward,
        termination_condition=termination,
        truncation_condition=truncation,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.6,
            min_episodes=75,
            max_std_dev=1.2
        )
    ))

    # Stage 3: Advanced Mechanics
    advanced_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    )
    advanced_reward = CombinedReward(
        (GoalReward(), 15.0),
        (TouchBallToGoalAccelerationReward(), 0.5),
        (AlignBallToGoalReward(), 0.3),
        (BallToGoalDistanceReward(), 0.2)
    )
    stages.append(CurriculumStage(
        name="Advanced Mechanics",
        state_mutator=advanced_mutator,
        reward_function=advanced_reward,
        termination_condition=termination,
        truncation_condition=truncation,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5,
            min_avg_reward=0.7,
            min_episodes=100,
            max_std_dev=1.5
        )
    ))

    # Stage 4: Team Play
    team_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=2, orange_size=2),
        KickoffMutator()
    )
    team_reward = CombinedReward(
        (GoalReward(), 15.0),
        (KRCReward([
            (AlignBallToGoalReward(), 1.0),
            (BallToGoalDistanceReward(), 0.8),
            (PlayerVelocityTowardBallReward(), 0.6)
        ], team_spirit=0.3), 8.0)
    )
    stages.append(CurriculumStage(
        name="Team Play",
        state_mutator=team_mutator,
        reward_function=team_reward,
        termination_condition=termination,
        truncation_condition=truncation,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.45,
            min_avg_reward=0.8,
            min_episodes=150,
            max_std_dev=2.0
        )
    ))

    # Stage 5: Mastery
    mastery_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=2, orange_size=2),
        KickoffMutator()
    )
    mastery_reward = CombinedReward(
        (GoalReward(), 20.0),
        (KRCReward([
            (AlignBallToGoalReward(), 1.0),
            (TouchBallToGoalAccelerationReward(), 0.8),
            (BallToGoalDistanceReward(), 0.6)
        ], team_spirit=0.4), 10.0)
    )
    stages.append(CurriculumStage(
        name="Mastery",
        state_mutator=mastery_mutator,
        reward_function=mastery_reward,
        termination_condition=termination,
        truncation_condition=truncation,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.4,
            min_avg_reward=1.0,
            min_episodes=200,
            max_std_dev=2.5
        )
    ))

    # Create and return the curriculum manager with the stages
    curriculum = CurriculumManager(stages=stages, debug=debug)

    if debug:
        print("[DEBUG] Created basic curriculum with 5 stages")
        for stage in curriculum.stages:
            print(f"[DEBUG] Stage: {stage.name}")

    return curriculum

def load_curriculum(curriculum_type="skill_based", debug=False, use_wandb=True):
    """
    Load the specified curriculum.

    Args:
        curriculum_type: Type of curriculum to load ('skill_based', 'rlbot', or 'none')
        debug: Whether to print debug information
        use_wandb: Whether to log curriculum data to wandb

    Returns:
        CurriculumManager or None
    """
    if curriculum_type == "none":
        return None

    if curriculum_type == "skill_based":
        print("Loading skill-based 60/40 curriculum...")
        from curriculum_implementation import create_skill_based_curriculum
        return create_skill_based_curriculum(debug=debug, use_wandb=use_wandb)

    if curriculum_type == "rlbot":
        print("Loading standard RLBot curriculum...")
        from curriculum_rlbot import create_rlbot_curriculum
        return create_rlbot_curriculum(debug=debug, use_wandb=use_wandb)

    print(f"Unknown curriculum type: {curriculum_type}")
    return None
