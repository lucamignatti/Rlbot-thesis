from .base import CurriculumManager, ProgressionRequirements
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator
from .mutators import (
    BallTowardGoalSpawnMutator, BallPositionMutator,
    CarBoostMutator, BallVelocityMutator, CarPositionMutator
)
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
import numpy as np
import random
from curriculum.rlbot import RLBotSkillStage

# Define position functions as regular functions instead of lambdas
def aerial_ball_position():
    """Get random aerial ball position"""
    x = np.random.uniform(-2000, 2000)
    y = np.random.uniform(-2000, 2000)
    z = np.random.uniform(300, 1200)
    return np.array([x, y, z])

def get_aerial_ball_position():
    """Get random aerial ball position"""
    return aerial_ball_position()

def get_wall_play_ball_position():
    """Get random wall play ball position"""
    # Randomly select a side wall
    if np.random.random() < 0.5:
        # Left wall
        x = -4096 + np.random.uniform(100, 300)  # Slightly off the wall
        y = np.random.uniform(-3000, 3000)
    else:
        # Right wall
        x = 4096 - np.random.uniform(100, 300)  # Slightly off the wall
        y = np.random.uniform(-3000, 3000)

    # Set height based on difficulty
    z = np.random.uniform(500, 1500)

    return np.array([x, y, z])

def get_fast_aerial_ball_position():
    """Get random fast aerial ball position"""
    x = np.random.uniform(-2000, 2000)
    y = np.random.uniform(-2000, 2000)
    z = np.random.uniform(800, 1800)
    return np.array([x, y, z])

def get_ground_dribbling_ball_position():
    """Get ground dribbling ball position"""
    x = np.random.uniform(-2000, 2000)
    y = np.random.uniform(-3000, 0)  # Start in blue half for dribbling toward orange
    z = 93  # Ball radius + small offset
    return np.array([x, y, z])

def get_ground_dribbling_ball_velocity():
    """Get slight forward momentum for dribbling"""
    speed = np.random.uniform(0, 500)
    return np.array([0, speed, 0])

def get_car_position(x=0, y=-3000, z=17):
    """Get car position with default near blue goal"""
    return np.array([x, y, z])

def get_car_position_aerial(x=0, y=-4000, z=17):
    """Get car position for aerial training (further back)"""
    return np.array([x, y, z])

def get_car_position_wall(x=0, y=-4000, z=17):
    """Get car position for wall training (center field)"""
    return np.array([x, y, z])

def create_skill_based_curriculum(debug=False, use_wandb=True):
    """Create a curriculum with progressive skill development stages"""

    # Define some constants and common components
    ORANGE_GOAL_LOCATION = np.array([0, 5120, 642.775])
    BLUE_GOAL_LOCATION = np.array([0, -5120, 642.775])
    BASE_TIMEOUT = 300  # 5 minutes
    SKILL_TIMEOUT = 120  # 2 minutes for skill training

    # Common conditions
    goal_condition = GoalCondition()
    timeout_condition = TimeoutCondition(BASE_TIMEOUT)
    skill_timeout = TimeoutCondition(SKILL_TIMEOUT)
    no_touch_timeout = NoTouchTimeoutCondition(timeout_seconds=30)

    # Stage 1: Basic Shooting
    basic_shooting = RLBotSkillStage(
        name="Basic Shooting",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallTowardGoalSpawnMutator(offensive_team=0, distance_from_goal=0.7),
            CarPositionMutator(car_id=0, position_function=get_car_position)
        ),
        base_task_reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 0.5)
        ),
        base_task_termination_condition=GoalCondition(),
        base_task_truncation_condition=TimeoutCondition(10.0),
        difficulty_params={
            "ball_distance": (0.7, 0.3),  # Start close, get farther
            "ball_speed": (0, 500)        # Start slow, get faster
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.7,
            min_avg_reward=0.5,
            min_episodes=50,
            max_std_dev=2.0,
            required_consecutive_successes=3
        )
    )

    # Stage 2: Aerial Control
    aerial_control = RLBotSkillStage(
        name="Aerial Control",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(get_aerial_ball_position),
            CarBoostMutator(boost_amount=100),
            CarPositionMutator(car_id=0, position_function=get_car_position_aerial)  # Added car position
        ),
        base_task_reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 1.0)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=TimeoutCondition(15.0),
        difficulty_params={
            "ball_height": (300, 1200),
            "car_boost": (50, 100)
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.4,
            min_episodes=100,
            max_std_dev=2.5,
            required_consecutive_successes=3
        )
    )

    # Stage 3: Wall Play
    wall_play = RLBotSkillStage(
        name="Wall Play",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(get_wall_play_ball_position),
            CarPositionMutator(car_id=0, position_function=get_car_position_wall)  # Added car position
        ),
        base_task_reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 1.0)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=TimeoutCondition(15.0),
        difficulty_params={
            "ball_height": (500, 1500),
            "wall_distance": (0.2, 0.8)
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5,
            min_avg_reward=0.3,
            min_episodes=150,
            max_std_dev=3.0,
            required_consecutive_successes=2
        )
    )

    # Stage 4: Fast Aerial
    fast_aerial = RLBotSkillStage(
        name="Fast Aerial",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(get_fast_aerial_ball_position),
            CarBoostMutator(boost_amount=100),
            CarPositionMutator(car_id=0, position_function=get_car_position_aerial)  # Added car position
        ),
        base_task_reward_function=CombinedReward(
            (GoalReward(), 15.0),
            (TouchReward(), 2.0)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=TimeoutCondition(12.0),
        difficulty_params={
            "ball_height": (800, 1800),
            "time_limit": (12, 8)
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.4,
            min_avg_reward=0.2,
            min_episodes=200,
            max_std_dev=3.5,
            required_consecutive_successes=2
        )
    )

    # Stage 5: Ground Dribbling
    ground_dribbling = RLBotSkillStage(
        name="Ground Dribbling",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(get_ground_dribbling_ball_position),
            BallVelocityMutator(get_ground_dribbling_ball_velocity),  # Slight forward momentum
            CarPositionMutator(car_id=0, position_function=get_car_position)  # Added car position
        ),
        base_task_reward_function=CombinedReward(
            (GoalReward(), 20.0),
            (TouchReward(), 0.2)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=TimeoutCondition(20.0),
        difficulty_params={
            "starting_speed": (0, 500),
            "distance": (0.3, 0.8)
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.3,
            min_avg_reward=0.15,
            min_episodes=250,
            max_std_dev=4.0,
            required_consecutive_successes=2
        )
    )

    # Create stages list in order of progression
    stages = [
        basic_shooting,
        aerial_control,
        wall_play,
        fast_aerial,
        ground_dribbling
    ]

    # Create curriculum manager
    curriculum_manager = CurriculumManager(
        stages=stages,
        progress_thresholds={
            "success_rate": 0.7,
            "avg_reward": 0.6
        },
        evaluation_window=50,
        debug=debug,
        use_wandb=use_wandb
    )

    return curriculum_manager
