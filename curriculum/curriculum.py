from .base import CurriculumManager, ProgressionRequirements
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator
from .mutators import (
    BallTowardGoalSpawnMutator, BallPositionMutator, CarBallRelativePositionMutator,
    CarBoostMutator, BallVelocityMutator, CarPositionMutator
)
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
import numpy as np
import random
from curriculum.rlbot import RLBotSkillStage
from rlgym.rocket_league.state_mutators import KickoffMutator
from rewards import (
    TouchBallReward, PlayerVelocityTowardBallReward, create_offensive_potential_reward,
    BallVelocityToGoalReward, TouchBallToGoalAccelerationReward, SaveBoostReward,
    create_distance_weighted_alignment_reward, BallToGoalDistanceReward, create_lucy_skg_reward, 
    BallProximityReward, KRCRewardFunction
)
from functools import partial
from typing import Tuple, List, Dict, Any, Optional

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
    x_offset = np.random.uniform(-100, 100)  # Small x variation
    y_offset = 300  # Place ball ~300 units in front of car
    z = 93  # Ball radius + small offset
    return np.array([x_offset, y_offset, z])  # Relative to car position

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

def create_position(x, y, z):
    return np.array([x, y, z])

def get_directional_shooting_car_position():
    return np.array([
        np.random.uniform(-500, 500),
        np.random.uniform(-3000, -2000),
        17
    ])

def get_aerial_ball_position():
    return np.array([
        np.random.uniform(-800, 800),
        np.random.uniform(-2000, -1000),
        np.random.uniform(300, 800)
    ])

def get_aerial_car_position():
    return np.array([
        np.random.uniform(-200, 200),
        np.random.uniform(-3500, -3000),
        17
    ])

def get_strategic_ball_position():
    return np.array([
        np.random.uniform(-2000, 2000),
        np.random.uniform(-2000, 2000),
        93
    ])

def get_strategic_car_position():
    return np.array([
        np.random.uniform(-4000, 4000),
        np.random.uniform(-4700, 4700),
        17
    ])

def get_advanced_aerial_ball_position():
    return np.array([
        np.random.uniform(-1500, 1500),
        np.random.uniform(-2000, 0),
        np.random.uniform(800, 1500)
    ])

def get_advanced_aerial_ball_velocity():
    return np.array([
        np.random.uniform(-300, 300),
        np.random.uniform(-300, 300),
        np.random.uniform(-100, 100)
    ])

def get_advanced_aerial_car_position():
    return np.array([
        np.random.uniform(-500, 500),
        np.random.uniform(-4000, -3000),
        17
    ])

def get_ground_ball_position():
    return np.array([600 + np.random.uniform(-50, 50), np.random.uniform(-100, 100), 93])

def get_car_position_near_goal(team: int = 0) -> np.ndarray:
    """Get initial car position near the goal
    Args:
        team: 0 for blue team (negative y), 1 for orange team (positive y)
    Returns:
        np.ndarray: [x, y, z] position
    """
    y_pos = -3000 if team == 0 else 3000
    return np.array([0, y_pos, 17])

def get_car_defensive_position(team: int = 0) -> np.ndarray:
    """Get defensive car position
    Args:
        team: 0 for blue team (negative y), 1 for orange team (positive y)
    Returns:
        np.ndarray: [x, y, z] position
    """
    y_sign = -1 if team == 0 else 1
    return np.array([
        np.random.uniform(-200, 200),
        np.random.uniform(4000, 4500) * y_sign,
        17
    ])

def get_car_offensive_position(team: int = 0) -> np.ndarray:
    """Get offensive car position
    Args:
        team: 0 for blue team (negative y), 1 for orange team (positive y)
    Returns:
        np.ndarray: [x, y, z] position
    """
    y_sign = -1 if team == 0 else 1
    return np.array([
        np.random.uniform(-1000, 1000),
        np.random.uniform(2000, 3000) * y_sign,
        17
    ])

def get_car_wall_position(side: str = 'left') -> np.ndarray:
    """Get car position on wall
    Args:
        side: 'left' or 'right' wall
    Returns:
        np.ndarray: [x, y, z] position
    """
    x_pos = -4096 + 17 if side == 'left' else 4096 - 17
    return np.array([x_pos, np.random.uniform(-3000, 3000), 250])

def get_ball_ground_position(neutral: bool = False) -> np.ndarray:
    """Get ball position on ground
    Args:
        neutral: If True, position will be in neutral field position
    Returns:
        np.ndarray: [x, y, z] position
    """
    if neutral:
        x = np.random.uniform(-2000, 2000)
        y = np.random.uniform(-2000, 2000)
    else:
        x = np.random.uniform(-3000, 3000)
        y = np.random.uniform(-4000, 4000)
    return np.array([x, y, 93])

def get_ball_aerial_position(difficulty: str = 'basic') -> np.ndarray:
    """Get aerial ball position
    Args:
        difficulty: 'basic', 'intermediate', or 'advanced'
    Returns:
        np.ndarray: [x, y, z] position
    """
    height_ranges = {
        'basic': (300, 800),
        'intermediate': (800, 1200),
        'advanced': (1200, 1800)
    }
    min_height, max_height = height_ranges.get(difficulty, (300, 800))
    
    return np.array([
        np.random.uniform(-2000, 2000),
        np.random.uniform(-2000, 2000),
        np.random.uniform(min_height, max_height)
    ])

def get_ball_wall_position(side: str = 'left') -> np.ndarray:
    """Get ball position near wall
    Args:
        side: 'left' or 'right' wall
    Returns:
        np.ndarray: [x, y, z] position
    """
    x_pos = -4096 + 100 if side == 'left' else 4096 - 100
    return np.array([
        x_pos,
        np.random.uniform(-3000, 3000),
        np.random.uniform(500, 1000)
    ])

def get_ball_dribble_position(car_pos: np.ndarray) -> np.ndarray:
    """Get ball position for dribbling, relative to car position
    Args:
        car_pos: Current car position
    Returns:
        np.ndarray: [x, y, z] position
    """
    return np.array([
        car_pos[0] + np.random.uniform(-50, 50),
        car_pos[1] + 300,  # Ball slightly in front of car
        93  # Ball radius
    ])

def get_ball_shot_position(team: int = 0) -> np.ndarray:
    """Get ball position for shooting training
    Args:
        team: 0 for blue team (shooting positive y), 1 for orange (shooting negative y)
    Returns:
        np.ndarray: [x, y, z] position
    """
    y_sign = 1 if team == 0 else -1
    return np.array([
        np.random.uniform(-1000, 1000),
        np.random.uniform(2000, 3000) * y_sign,
        93
    ])

def get_ball_passing_position(team: int = 0) -> np.ndarray:
    """Get ball position for passing plays
    Args:
        team: 0 for blue team, 1 for orange team
    Returns:
        np.ndarray: [x, y, z] position
    """
    y_sign = 1 if team == 0 else -1
    return np.array([
        np.random.uniform(-2000, 2000),
        np.random.uniform(1000, 2000) * y_sign,
        93
    ])

def get_strategic_position(role: str = 'defense', team: int = 0) -> np.ndarray:
    """Get strategic position based on role
    Args:
        role: 'defense', 'midfield', or 'offense'
        team: 0 for blue team, 1 for orange team
    Returns:
        np.ndarray: [x, y, z] position
    """
    y_ranges = {
        'defense': (-4500, -3500) if team == 0 else (3500, 4500),
        'midfield': (-2000, -1000) if team == 0 else (1000, 2000),
        'offense': (-1000, 0) if team == 0 else (0, 1000)
    }
    y_min, y_max = y_ranges.get(role, (-4500, -3500) if team == 0 else (3500, 4500))
    
    return np.array([
        np.random.uniform(-1000, 1000),
        np.random.uniform(y_min, y_max),
        17
    ])

# Ball velocity functions
def get_ball_rolling_velocity(speed_range: Tuple[float, float] = (500, 1000)) -> np.ndarray:
    """Get velocity for rolling ball
    Args:
        speed_range: (min_speed, max_speed) in uu/s
    Returns:
        np.ndarray: [vx, vy, vz] velocity
    """
    speed = np.random.uniform(*speed_range)
    angle = np.random.uniform(-np.pi, np.pi)
    return np.array([
        speed * np.cos(angle),
        speed * np.sin(angle),
        0
    ])

def get_ball_aerial_velocity(speed_range: Tuple[float, float] = (100, 500)) -> np.ndarray:
    """Get velocity for aerial ball
    Args:
        speed_range: (min_speed, max_speed) in uu/s
    Returns:
        np.ndarray: [vx, vy, vz] velocity
    """
    speed = np.random.uniform(*speed_range)
    horizontal_angle = np.random.uniform(-np.pi, np.pi)
    vertical_angle = np.random.uniform(0, np.pi/3)  # Up to 60 degrees up
    
    return np.array([
        speed * np.cos(vertical_angle) * np.cos(horizontal_angle),
        speed * np.cos(vertical_angle) * np.sin(horizontal_angle),
        speed * np.sin(vertical_angle)
    ])

def get_defensive_ball_position(team: int = 0) -> np.ndarray:
    """Get ball position for defensive training
    Args:
        team: 0 for blue team (negative y), 1 for orange team (positive y)
    Returns:
        np.ndarray: [x, y, z] position
    """
    y_coord = -3000 if team == 0 else 3000  # Ball in defensive third
    return np.array([
        np.random.uniform(-2000, 2000),
        y_coord,
        np.random.uniform(100, 200)
    ])

def get_defensive_save_ball_position() -> np.ndarray:
    """Get ball position for practicing goal-line saves
    Returns:
        np.ndarray: [x, y, z] position - positioned toward blue goal
    """
    return np.array([
        np.random.uniform(-800, 800),
        -3500,  # Ball positioned toward blue goal
        np.random.uniform(100, 300)
    ])

def get_defensive_save_ball_velocity() -> np.ndarray:
    """Get ball velocity for practicing goal-line saves
    Returns:
        np.ndarray: [vx, vy, vz] velocity - moving toward blue goal
    """
    return np.array([
        np.random.uniform(-300, 300),
        -2000,  # Ball moving toward blue goal
        np.random.uniform(-100, 100)
    ])

def get_defensive_save_car_position() -> np.ndarray:
    """Get car position for practicing goal-line saves
    Returns:
        np.ndarray: [x, y, z] position - car near blue goal
    """
    return np.array([
        np.random.uniform(-500, 500),
        -4500,  # Blue car positioned near goal
        17
    ])

def get_offensive_ball_position() -> np.ndarray:
    """Get ball position for offensive coordination training
    Returns:
        np.ndarray: [x, y, z] position - ball in orange defensive third
    """
    return np.array([
        np.random.uniform(-2000, 2000),
        3000,  # Ball in orange defensive third
        np.random.uniform(100, 200)
    ])

def get_varied_ground_ball_position():
    """Get varied ball positions across the field"""
    x = np.random.uniform(-2000, 2000)  # Wide x-range across field
    y = np.random.uniform(-2000, 2000)  # Full field y-range
    return np.array([x, y, 93])  # Ball radius height

def get_varied_approach_car_position(ball_pos=None):
    """Get car position with varied approach angles to ball"""
    if ball_pos is None:
        # Default positioning if no ball reference
        return np.array([
            np.random.uniform(-3000, 3000),
            np.random.uniform(-4000, 0),  # Blue half of field
            17
        ])
    
    # Position car at varied distances and angles from ball
    distance = np.random.uniform(1000, 2500)
    angle = np.random.uniform(-np.pi, np.pi)  # Full 360° approach angles
    
    return np.array([
        ball_pos[0] - distance * np.cos(angle),
        ball_pos[1] - distance * np.sin(angle),
        17
    ])

def get_blue_defender_position():
    """Get position for blue team defensive player"""
    return get_strategic_position(role='defense', team=0)

def get_blue_attacker_position():
    """Get position for blue team offensive player"""
    return get_strategic_position(role='offense', team=0)


def get_blue_primary_defender_position():
    """Get position for blue team's primary defender (closest to goal)"""
    return np.array([
        np.random.uniform(-700, 700),
        np.random.uniform(-5000, -4300),
        17
    ])

def get_blue_secondary_defender_position():
    """Get position for blue team's secondary defender (slightly upfield)"""
    return np.array([
        np.random.uniform(-1200, 1200),  # Wider position range
        np.random.uniform(-4200, -3500),  # Further upfield than primary defender
        17
    ])

def get_orange_attacker_position():
    """Get position for orange team attacker"""
    return np.array([
        np.random.uniform(-1000, 1000),
        np.random.uniform(-3000, -2000),  # In blue's defensive third
        17
    ])

def get_orange_support_position():
    """Get position for orange team support player"""
    return np.array([
        np.random.uniform(-2000, 2000),  # Wider positioning
        np.random.uniform(-2500, -1500),  # Behind the main attacker
        17
    ])


def get_blue_attacker_offensive_position():
    """Get position for blue team primary attacker deep in orange territory"""
    return np.array([
        np.random.uniform(-700, 700),
        np.random.uniform(2000, 3500),  # Orange half of field
        17
    ])

def get_blue_support_offensive_position():
    """Get position for blue team support player in orange territory"""
    return np.array([
        np.random.uniform(-1500, 1500),  # Wider positioning
        np.random.uniform(1000, 2000),   # Behind the main attacker
        17
    ])

def get_orange_primary_defender_position():
    """Get position for orange team's primary defender (closest to goal)"""
    return np.array([
        np.random.uniform(-700, 700),
        np.random.uniform(4300, 5000),  # Near orange goal
        17
    ])

def get_orange_secondary_defender_position():
    """Get position for orange team's secondary defender (slightly downfield)"""
    return np.array([
        np.random.uniform(-1200, 1200),  # Wider position range
        np.random.uniform(3500, 4200),   # Further downfield than primary defender
        17
    ])

def safe_ball_position():
    """Get a safe default ball position - always valid"""
    return np.array([0.0, 0.0, 93.0])

# Replace the safe_function_wrapper with a class that can be pickled
class SafePositionWrapper:
    """Wrapper class for position functions to ensure they always return valid coordinates"""
    def __init__(self, func):
        self.func = func
        self.default_car_position = np.array([0.0, -2000.0, 17.0], dtype=np.float32)  # Safe default car position
        self.default_ball_position = np.array([0.0, 0.0, 93.0], dtype=np.float32)  # Safe default ball position
    
    def __call__(self, *args, **kwargs):
        """Call the wrapped function and ensure it returns valid coordinates"""
        try:
            # Call the wrapped function to get position
            position = self.func(*args, **kwargs)
            
            # Ensure position is a numpy array
            if not isinstance(position, np.ndarray):
                position = np.array(position, dtype=np.float32)
                
            # Check if position contains NaN or infinite values
            if np.isnan(position).any() or np.isinf(position).any():
                # Determine appropriate default based on context clues in function name
                if 'car' in self.func.__name__.lower():
                    return self.default_car_position
                else:
                    return self.default_ball_position
                    
            # If position seems valid, return it
            return position
            
        except Exception as e:
            # If any exception occurs, fall back to default position
            if 'car' in self.func.__name__.lower():
                return self.default_car_position
            else:
                return self.default_ball_position
    
    def __str__(self):
        """Return a string representation of the position"""
        result = self()
        return f"Position: {result}"

def create_curriculum(debug=False):
    # Basic constants
    SHORT_TIMEOUT = 8  # seconds for short skill training
    MED_TIMEOUT = 15   # medium timeout for more complex skills
    LONG_TIMEOUT = 20  # longer timeout for team coordination
    MATCH_TIMEOUT = 300  # full match time
    
    # Common conditions
    goal_condition = GoalCondition()
    short_timeout = TimeoutCondition(SHORT_TIMEOUT)
    med_timeout = TimeoutCondition(MED_TIMEOUT)
    long_timeout = TimeoutCondition(LONG_TIMEOUT)
    match_timeout = TimeoutCondition(MATCH_TIMEOUT)
    no_touch_timeout = NoTouchTimeoutCondition(4)  # Time without touching ball
    
    # Create the base task mutators list
    base_task_mutators = MutatorSequence(
        FixedTeamSizeMutator(1),
        BallPositionMutator(position_function=safe_ball_position),
        BallVelocityMutator(lambda: np.zeros(3))
    )
    
    base_task_reward_function = KRCRewardFunction([
        BallProximityReward(),
        BallToGoalDistanceReward(team_goal_y=5120),
        TouchBallReward(),
        BallVelocityToGoalReward(team_goal_y=5120),
        create_distance_weighted_alignment_reward(5120)
    ])

    # ======== STAGE 0: PRE-TRAINING ========
    # This is a special stage focused only on unsupervised learning
    pretraining = RLBotSkillStage(
        name="Unsupervised Pre-training",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(get_strategic_car_position)),
            BallPositionMutator(position_function=SafePositionWrapper(safe_ball_position))
        ),
        base_task_reward_function=CombinedReward(
            (BallProximityReward(dispersion=1.0), 1.0),
            (PlayerVelocityTowardBallReward(), 0.8),
            (SaveBoostReward(weight=0.5), 0.3)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=short_timeout,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.0,  # No success requirement for pre-training
            min_avg_reward=0.0,    # No reward requirement for pre-training
            min_episodes=100,      # Just need enough data to do useful pre-training
            max_std_dev=10.0,      # Very permissive 
            required_consecutive_successes=1  # Set to minimum valid value (1) instead of 0
        ),
        is_pretraining=True  # Mark this as a special pre-training stage
    )
    
    # ======== STAGE 1: MOVEMENT FOUNDATIONS ========
    movement_foundations = RLBotSkillStage(
        name="Movement Foundations",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(get_strategic_car_position)),
            BallPositionMutator(position_function=SafePositionWrapper(safe_ball_position))
        ),
        base_task_reward_function=CombinedReward(
            (BallProximityReward(dispersion=1.0), 1.0),
            (PlayerVelocityTowardBallReward(), 0.8),
            (SaveBoostReward(weight=0.5), 0.3)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=short_timeout,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.85,
            min_avg_reward=0.5,
            min_episodes=50,
            max_std_dev=1.0,
            required_consecutive_successes=3
        )
    )
    
    # ======== STAGE 2: BALL ENGAGEMENT ========
    ball_engagement = RLBotSkillStage(
        name="Ball Engagement",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            # First set ball position
            BallPositionMutator(position_function=SafePositionWrapper(get_varied_ground_ball_position)),
            # Then position the car relative to the ball with varied angles and distances
            CarBallRelativePositionMutator(car_id="blue-0", position_function=SafePositionWrapper(get_varied_approach_car_position))
        ),
        base_task_reward_function=KRCRewardFunction([
            (BallProximityReward(dispersion=0.8), 0.8),
            (TouchBallReward(weight=0.3), 0.3),
            (TouchBallToGoalAccelerationReward(team_goal_y=5120, weight=0.5), 0.5)
        ], team_spirit=0.0),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=short_timeout,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.8,
            min_avg_reward=0.6,
            min_episodes=75,
            max_std_dev=1.5,
            required_consecutive_successes=3
        )
    )

    # ======== STAGE 3: BALL CONTROL & DIRECTION ========
    ball_control = RLBotSkillStage(
        name="Ball Control & Direction",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(position_function=get_ground_ball_position),
            CarPositionMutator(car_id="blue-0", position_function=partial(create_position, 0, -2500, 17))
        ),
        base_task_reward_function=KRCRewardFunction([
            (TouchBallReward(weight=0.2), 0.2),
            (BallVelocityToGoalReward(team_goal_y=5120, weight=0.6), 0.6),
            (TouchBallToGoalAccelerationReward(team_goal_y=5120, weight=0.3), 0.3)
        ], team_spirit=0.0),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=TimeoutCondition(10),
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.65,
            min_avg_reward=0.4,
            min_episodes=100,
            max_std_dev=1.8,
            required_consecutive_successes=2
        )
    )
    
    # ======== STAGE 4: SHOOTING FUNDAMENTALS ========
    shooting_mutators = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=0),
        BallTowardGoalSpawnMutator(offensive_team=0, distance_from_goal=0.75),
        CarPositionMutator(car_id="blue-0", position_function=get_directional_shooting_car_position)
    )
    
    shooting_fundamentals = RLBotSkillStage(
        name="Shooting Fundamentals",
        base_task_state_mutator=shooting_mutators,
        base_task_reward_function=KRCRewardFunction([
            (BallVelocityToGoalReward(team_goal_y=5120, weight=0.7), 0.7),
            (TouchBallToGoalAccelerationReward(team_goal_y=5120, weight=0.3), 0.3),
            (create_distance_weighted_alignment_reward(5120), 1.0)
        ], team_spirit=0.0),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=short_timeout,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.5,
            min_episodes=125,
            max_std_dev=1.8,
            required_consecutive_successes=2
        )
    )
    
    # ======== STAGE 5: WALL & AIR MECHANICS ========
    wall_air_mechanics = RLBotSkillStage(
        name="Wall & Air Mechanics",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(position_function=get_aerial_ball_position),
            CarBoostMutator(boost_amount=50),
            CarPositionMutator(car_id="blue-0", position_function=get_aerial_car_position)
        ),
        base_task_reward_function=KRCRewardFunction([
            (TouchBallReward(weight=0.2), 0.2),
            (SaveBoostReward(weight=0.3), 0.3),
            (BallVelocityToGoalReward(team_goal_y=5120, weight=0.5), 0.5)
        ], team_spirit=0.0),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=short_timeout,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.4,
            min_episodes=150,
            max_std_dev=2.0,
            required_consecutive_successes=2
        )
    )
    
    # ======== STAGE 6: BEGINNING TEAM PLAY (2v0) ========
    beginning_team_play = RLBotSkillStage(
        name="Beginning Team Play",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=0),
            BallPositionMutator(position_function=get_strategic_ball_position),
            # Position first car in defensive role
            CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(get_blue_defender_position)),
            # Position second car in offensive role
            CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(get_blue_attacker_position))

        ),
        base_task_reward_function=KRCRewardFunction([
            (create_offensive_potential_reward(5120), 0.7),
            (TouchBallReward(weight=0.5), 0.5),
            (BallVelocityToGoalReward(team_goal_y=5120, weight=0.5), 0.5)
        ], team_spirit=0.7),  # Higher team spirit for team play
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=med_timeout,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.4,
            min_episodes=125,
            max_std_dev=2.0,
            required_consecutive_successes=2
        )
    )
    
    # ======== STAGE 7: DEFENSE & GOAL-LINE SAVES ========
    defense = RLBotSkillStage(
        name="Defense & Goal-line Saves",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            BallPositionMutator(position_function=get_defensive_save_ball_position),
            BallVelocityMutator(velocity_function=get_defensive_save_ball_velocity),
            CarPositionMutator(car_id="blue-0", position_function=get_defensive_save_car_position)
        ),
        base_task_reward_function=KRCRewardFunction([
            (SaveBoostReward(weight=0.5), 0.5),
            (TouchBallReward(weight=0.7), 0.7),
            (BallVelocityToGoalReward(team_goal_y=-5120, weight=0.4), 0.4)
        ], team_spirit=0.0),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=TimeoutCondition(10),
        bot_skill_ranges={(0.2, 0.4): 1.0},
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.4,
            min_episodes=150,
            max_std_dev=2.0,
            required_consecutive_successes=2
        )
    )
    
    # ======== STAGE 8: INTERMEDIATE BALL CONTROL ========
    # Focus on close ball control and dribbling setup
    intermediate_ball_control = RLBotSkillStage(
        name="Intermediate Ball Control",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(position_function=get_ground_dribbling_ball_position),
            BallVelocityMutator(velocity_function=get_ground_dribbling_ball_velocity),
            CarPositionMutator(car_id="blue-0", position_function=partial(create_position, 0, -2800, 17)),
            CarBoostMutator(boost_amount=50)
        ),
        base_task_reward_function=KRCRewardFunction([
            (BallProximityReward(dispersion=0.5), 0.6),  # Keep ball close
            (PlayerVelocityTowardBallReward(), 0.4),
            (BallVelocityToGoalReward(team_goal_y=5120, weight=0.4), 0.4)
        ], team_spirit=0.0),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=TimeoutCondition(12),
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5,
            min_avg_reward=0.4,
            min_episodes=175,
            max_std_dev=2.0,
            required_consecutive_successes=2
        )
    )
    
    # ======== STAGE 9: 2v2 DEFENSIVE ROTATION ========
    # Focus on team defense and rotation
    defensive_rotation = RLBotSkillStage(
        name="2v2 Defensive Rotation",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            BallPositionMutator(position_function=get_defensive_ball_position),
            # Blue team defensive positions
            CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(get_blue_primary_defender_position)),
            CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(get_blue_secondary_defender_position)),
            # Orange team attacking positions
            CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(get_orange_attacker_position)),
            CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(get_orange_support_position))

        ),
        base_task_reward_function=KRCRewardFunction([
            (BallVelocityToGoalReward(team_goal_y=-5120, weight=0.6), 0.6),
            (TouchBallReward(weight=0.5), 0.5),
            (create_offensive_potential_reward(5120), 0.6)
        ], team_spirit=0.7),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=long_timeout,
        bot_skill_ranges={(0.3, 0.5): 1.0},
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.4,
            min_episodes=175,
            max_std_dev=2.0,
            required_consecutive_successes=2
        )
    )
    
    # ======== STAGE 10: 2v2 OFFENSIVE COORDINATION ========
    # Focus on team offense and passing
    offensive_coordination = RLBotSkillStage(
        name="2v2 Offensive Coordination", 
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            BallPositionMutator(position_function=get_offensive_ball_position),
            # Blue team attacking positions
            CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(get_blue_attacker_offensive_position)),
            CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(get_blue_support_offensive_position)),
            # Orange team defensive positions
            CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(get_orange_primary_defender_position)),
            CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(get_orange_secondary_defender_position))

        ),
        base_task_reward_function=KRCRewardFunction([
            (BallVelocityToGoalReward(team_goal_y=5120, weight=0.7), 0.7),
            (TouchBallReward(weight=0.5), 0.5),
            (create_offensive_potential_reward(5120), 0.6),
            (TouchBallToGoalAccelerationReward(team_goal_y=5120, weight=0.6), 0.6)
        ], team_spirit=0.7),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=long_timeout,
        bot_skill_ranges={(0.3, 0.5): 1.0},
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5,
            min_avg_reward=0.4,
            min_episodes=200,
            max_std_dev=2.0,
            required_consecutive_successes=2
        )
    )
    
    # ======== STAGE 11: INTERMEDIATE AERIALS & WALL PLAY ========
    # Focus on more complex aerial mechanics and wall plays
    intermediate_aerials = RLBotSkillStage(
        name="Intermediate Aerials & Wall Play",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(position_function=get_advanced_aerial_ball_position),
            BallVelocityMutator(velocity_function=get_advanced_aerial_ball_velocity),
            CarPositionMutator(car_id="blue-0", position_function=get_advanced_aerial_car_position),
            CarBoostMutator(boost_amount=75)
        ),
        base_task_reward_function=KRCRewardFunction([
            (TouchBallReward(weight=0.2), 0.2),
            (BallVelocityToGoalReward(team_goal_y=5120, weight=0.4), 0.4),
            (SaveBoostReward(weight=0.4), 0.4),
            (TouchBallToGoalAccelerationReward(team_goal_y=5120, weight=0.6), 0.6)
        ], team_spirit=0.0),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=TimeoutCondition(10),
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.45,
            min_avg_reward=0.3,
            min_episodes=200,
            max_std_dev=2.5,
            required_consecutive_successes=2
        )
    )
    
    # ======== STAGE 12: FULL 2v2 INTEGRATION ========
    # Focus on complete 2v2 match play
    full_2v2_integration = RLBotSkillStage(
        name="Full 2v2 Integration",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            KickoffMutator()
        ),
        base_task_reward_function=create_lucy_skg_reward(5120),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=match_timeout,
        bot_skill_ranges={(0.4, 0.6): 0.7, (0.6, 0.7): 0.3},
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.3,
            min_episodes=250,
            max_std_dev=2.5,
            required_consecutive_successes=2
        )
    )
    
    # Create the full curriculum
    stages = [
        pretraining,       # Start with pre-training stage
        movement_foundations,
        ball_engagement,
        ball_control,
        shooting_fundamentals,
        wall_air_mechanics,
        beginning_team_play,
        defense,
        intermediate_ball_control,
        defensive_rotation,
        offensive_coordination,
        intermediate_aerials,
        full_2v2_integration
    ]
    
    curriculum_manager = CurriculumManager(
        stages=stages,
        progress_thresholds={"success_rate": 0.6, "avg_reward": 0.4},
        max_rehearsal_stages=3,  # Revisit up to 3 previous stages to prevent skill degradation
        rehearsal_decay_factor=0.7,  # Higher weight for more recent stages
        evaluation_window=50,
        debug=debug,
        use_wandb=True
    )
    
    return curriculum_manager
