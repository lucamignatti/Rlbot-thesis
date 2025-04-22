from .base import CurriculumManager, CurriculumStage, ProgressionRequirements
from .mutators import (
    BallTowardGoalSpawnMutator, BallPositionMutator, CarBallRelativePositionMutator,
    CarBoostMutator, BallVelocityMutator, CarPositionMutator, TouchBallCondition
    # AugmentMutator, RandomPhysicsMutator removed as they don't exist
)
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from .rewards import (
    BallProximityReward, BallToGoalDistanceReward, TouchBallReward,
    BallVelocityToGoalReward, AlignBallToGoalReward, SaveBoostReward,
    KRCReward, PlayerVelocityTowardBallReward, TouchBallToGoalAccelerationReward,
    PassCompletionReward, ScoringOpportunityCreationReward, AerialControlReward,
    AerialDirectionalTouchReward, BlockSuccessReward, DefensivePositioningReward,
    BallClearanceReward, TeamSpacingReward, TeamPossessionReward,
    create_distance_weighted_alignment_reward, create_offensive_potential_reward, create_lucy_skg_reward,
    AerialDistanceReward, FlipResetReward, WavedashReward, FirstTouchSpeedReward, SpeedflipReward
)
from curriculum.skills import SkillModule, SkillBasedCurriculumStage
from functools import partial
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import random

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

def get_basic_aerial_ball_position():
    return np.array([
        np.random.uniform(-800, 800),
        np.random.uniform(-2000, -1000),
        np.random.uniform(300, 800) # Lower height for basic aerials
    ])

def get_basic_aerial_car_position():
    return np.array([
        np.random.uniform(-200, 200),
        np.random.uniform(-3500, -3000),
        17
    ])

def get_aerial_ball_position_v2():
    """Get aerial ball position with different range parameters than the first version"""
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
    """Get a ball velocity aimed toward the goal for defensive save training"""
    # Calculate velocity aimed approximately at the goal
    direction = np.array([np.random.uniform(-0.4, 0.4), -1.0, np.random.uniform(-0.2, 0.5)])
    direction = direction / np.linalg.norm(direction)  # Normalize
    speed = np.random.uniform(1000, 2000)  # Fast ball
    return direction * speed

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

# --- Orientation Functions ---

def get_random_yaw_orientation(*args) -> np.ndarray: # Added *args
    """Returns a random yaw orientation (rotation around Z-axis)."""
    yaw = np.random.uniform(-np.pi, np.pi)
    return np.array([0, yaw, 0]) # Pitch, Yaw, Roll

def get_face_opp_goal_orientation(*args) -> np.ndarray: # Added *args
    """Returns orientation facing the opponent's goal (default blue team)."""
    return np.array([0, 0, 0])

def get_face_own_goal_orientation(*args) -> np.ndarray: # Added *args
    """Returns orientation facing own goal (default blue team)."""
    return np.array([0, np.pi, 0])

# Replace the safe_function_wrapper with a class that can be pickled
class SafePositionWrapper:
    """Wrapper class for position functions to ensure they always return valid coordinates"""
    def __init__(self, func):
        self.func = func
        self.default_car_position = np.array([0.0, -2000.0, 17.0], dtype=np.float32)  # Safe default car position
        self.default_ball_position = np.array([0.0, 0.0, 93.0], dtype=np.float32)  # Safe default ball position

        # Flag to identify if this is a relative position function
        # Handle both regular functions and functools.partial objects
        if hasattr(func, '__name__'):
            func_name = func.__name__.lower()
        elif hasattr(func, 'func') and hasattr(func.func, '__name__'):  # For functools.partial
            func_name = func.func.__name__.lower()
        else:
            func_name = str(func).lower()  # Fallback to string representation

        self.is_relative = 'dribbling' in func_name or 'relative' in func_name

    def __call__(self, *args, **kwargs):
        """Call the wrapped function and ensure it returns valid coordinates"""
        try:
            # Call the wrapped function to get position
            position = self.func(*args, **kwargs)

            # Debug output for tracking issues
            if position is None:
                # Get function name for better error messages
                if hasattr(self.func, '__name__'):
                    func_name = self.func.__name__
                elif hasattr(self.func, 'func') and hasattr(self.func.func, '__name__'):
                    func_name = f"{self.func.func.__name__}[partial]"
                else:
                    func_name = str(self.func)

                print(f"WARNING: Position function {func_name} returned None")

                # For relative position functions, provide a default relative offset
                if self.is_relative:
                    print(f"Using default relative position for {func_name}")
                    # Default position in front of car for dribbling functions
                    return np.array([0.0, 200.0, 93.0], dtype=np.float32)

                # For non-relative functions, use appropriate default
                if 'car' in str(self.func).lower():
                    return self.default_car_position.copy()
                else:
                    return self.default_ball_position.copy()

            # Ensure position is a numpy array
            if not isinstance(position, np.ndarray):
                position = np.array(position, dtype=np.float32)

            # Check if position contains NaN or infinite values
            if np.isnan(position).any() or np.isinf(position).any():
                # Get function name for better error messages
                if hasattr(self.func, '__name__'):
                    func_name = self.func.__name__
                elif hasattr(self.func, 'func') and hasattr(self.func.func, '__name__'):
                    func_name = f"{self.func.func.__name__}[partial]"
                else:
                    func_name = str(self.func)

                print(f"WARNING: Position function {func_name} returned invalid values: {position}")

                # Determine appropriate default based on context clues in function name
                if 'car' in str(self.func).lower():
                    return self.default_car_position.copy()
                else:
                    return self.default_ball_position.copy()

            # If position seems valid, return it
            return position

        except Exception as e:
            # Get function name for better error messages
            if hasattr(self.func, '__name__'):
                func_name = self.func.__name__
            elif hasattr(self.func, 'func') and hasattr(self.func.func, '__name__'):
                func_name = f"{self.func.func.__name__}[partial]"
            else:
                func_name = str(self.func)

            # If any exception occurs, fall back to default position
            print(f"ERROR in position function {func_name}: {str(e)}")
            if 'car' in str(self.func).lower():
                return self.default_car_position.copy()
            else:
                return self.default_ball_position.copy()

    def __str__(self):
        """Return a string representation of the position"""
        try:
            result = self()
            # Get function name for better display
            if hasattr(self.func, '__name__'):
                func_name = self.func.__name__
            elif hasattr(self.func, 'func') and hasattr(self.func.func, '__name__'):
                func_name = f"{self.func.func.__name__}[partial]"
            else:
                func_name = str(self.func)

            return f"Position from {func_name}: {result}"
        except:
            return f"Position function (error evaluating)"

def create_car_position_mutator(car_id, position_function, orientation_function=None, debug=False):
    """Helper function to create CarPositionMutator instances with debug flag"""
    # Removed debug=debug as it's not a valid parameter for CarPositionMutator
    return CarPositionMutator(
        car_id=car_id,
        position_function=position_function,
        orientation_function=orientation_function
    )

def create_curriculum(debug=False, use_wandb=True, lr_actor=None, lr_critic=None, use_pretraining=True):
    """
    Create a curriculum for training.

    Args:
        debug: Whether to print debug information
        use_wandb: Whether to use Weights & Biases for logging
        lr_actor: Learning rate for the actor network (overrides default values)
        lr_critic: Learning rate for the critic network (overrides default values)
        use_pretraining: Whether to include the pretraining stage

    Returns:
        CurriculumManager: The created curriculum manager
    """
    # Store default learning rates to use if not provided
    default_lr_actor = 0.0002  # Default actor learning rate
    default_lr_critic = 0.0005  # Default critic learning rate

    # Use provided learning rates or fall back to defaults
    lr_actor = lr_actor if lr_actor is not None else default_lr_actor
    lr_critic = lr_critic if lr_critic is not None else default_lr_critic

    if debug:
        print(f"[DEBUG] Creating curriculum with learning rates - Actor: {lr_actor}, Critic: {lr_critic}")
        print(f"[DEBUG] Pretraining enabled: {use_pretraining}")

    # Basic constants
    SHORT_TIMEOUT = 8
    MED_TIMEOUT = 15
    LONG_TIMEOUT = 20
    MATCH_TIMEOUT = 300

    # Common conditions
    goal_condition = GoalCondition()
    short_timeout = TimeoutCondition(SHORT_TIMEOUT)
    med_timeout = TimeoutCondition(MED_TIMEOUT)
    long_timeout = TimeoutCondition(LONG_TIMEOUT)
    match_timeout = TimeoutCondition(MATCH_TIMEOUT)
    no_touch_timeout_short = NoTouchTimeoutCondition(4)
    no_touch_timeout_med = NoTouchTimeoutCondition(6)

    # Common Rewards
    touch_ball_reward = TouchBallReward()
    goal_velocity_reward = BallVelocityToGoalReward(team_goal_y=5120)
    goal_accel_reward = TouchBallToGoalAccelerationReward(team_goal_y=5120)
    save_boost_reward = SaveBoostReward()
    ball_proximity_reward = BallProximityReward()
    velocity_to_ball_reward = PlayerVelocityTowardBallReward()
    alignment_reward = create_distance_weighted_alignment_reward(5120)
    offensive_potential_reward = create_offensive_potential_reward(5120)
    lucy_reward = create_lucy_skg_reward(5120) # Full team reward

    # Default Progression Requirements (can be overridden per stage)
    default_progression = ProgressionRequirements(
        min_success_rate=0.6,
        min_avg_reward=0.4,
        min_episodes=100,
        max_std_dev=2.0,
        required_consecutive_successes=3
    )

    # ======== STAGE 0: PRE-TRAINING (Keep existing) ========
    pretraining = SkillBasedCurriculumStage(
        name="Unsupervised Pre-training",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            # Add random yaw orientation
            CarPositionMutator(car_id="blue-0",
                               position_function=SafePositionWrapper(get_strategic_car_position),
                               orientation_function=get_random_yaw_orientation),
            CarPositionMutator(car_id="blue-1",
                               position_function=SafePositionWrapper(get_blue_attacker_position),
                               orientation_function=get_random_yaw_orientation),
            CarPositionMutator(car_id="orange-0",
                               position_function=SafePositionWrapper(get_orange_primary_defender_position),
                               orientation_function=get_random_yaw_orientation),
            CarPositionMutator(car_id="orange-1",
                               position_function=SafePositionWrapper(get_orange_attacker_position),
                               orientation_function=get_random_yaw_orientation),
            BallPositionMutator(position_function=SafePositionWrapper(safe_ball_position))
        ),
        base_task_reward_function=CombinedReward(
            (touch_ball_reward, 0.5),                         # Increased reward for touching ball
            (goal_velocity_reward, 0.4),                      # Reward for hitting ball toward goal
            (velocity_to_ball_reward, 0.3),                   # Reward for moving toward ball
            (goal_accel_reward, 0.3),                         # Reward for accelerating ball toward goal
            (ball_proximity_reward, 0.2),                     # Reduced proximity reward
            (alignment_reward, 0.2),                          # Reward for aligning car with ball-to-goal
            (save_boost_reward, 0.1),                         # Keep boost conservation
            (BallToGoalDistanceReward(team_goal_y=5120), 0.1) # Keep ball to goal reward
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=short_timeout,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.0, min_avg_reward=0.0, min_episodes=100,
            max_std_dev=10.0, required_consecutive_successes=1
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "entropy_coef": 0.01
        },
        skill_modules=[],  # Empty list since this is pretraining
        base_task_prob=1.0, # Added for consistency with SkillBasedCurriculumStage
        debug=debug
    )

    # ======== STAGE 1: BALL TOUCHES (1v0) ========
    ball_touches_stage = SkillBasedCurriculumStage(
        name="Ball Touches",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(position_function=SafePositionWrapper(get_varied_ground_ball_position)),
            CarPositionMutator(car_id="blue-0",
                               position_function=SafePositionWrapper(get_varied_approach_car_position),
                               orientation_function=get_random_yaw_orientation)
        ),
        base_task_reward_function=CombinedReward(
            (touch_ball_reward, 1.0),
            (ball_proximity_reward, 0.7),
            (velocity_to_ball_reward, 0.3)
        ),
        base_task_termination_condition=TouchBallCondition(),
        base_task_truncation_condition=short_timeout,
        skill_modules=[
            SkillModule(
                name="Moving Ball",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=1, orange_size=0),
                    BallPositionMutator(position_function=SafePositionWrapper(get_varied_ground_ball_position)),
                    BallVelocityMutator(velocity_function=SafePositionWrapper(
                        partial(get_ball_rolling_velocity, speed_range=(300, 700))
                    )),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_varied_approach_car_position),
                                      orientation_function=get_random_yaw_orientation)
                ),
                reward_function=CombinedReward(
                    (ball_proximity_reward, 0.7),
                    (touch_ball_reward, 1.0),
                    (velocity_to_ball_reward, 0.3)
                ),
                termination_condition=TouchBallCondition(),
                truncation_condition=short_timeout,
                difficulty_params={},
                success_threshold=0.7
            ),
            SkillModule(
                name="Bouncing Ball",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=1, orange_size=0),
                    BallPositionMutator(position_function=SafePositionWrapper(
                        lambda: np.array([np.random.uniform(-2000, 2000), np.random.uniform(-2000, 2000), 600])
                    )),
                    BallVelocityMutator(velocity_function=SafePositionWrapper(
                        lambda: np.array([0, 0, -500]) # Falling ball
                    )),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_varied_approach_car_position),
                                      orientation_function=get_random_yaw_orientation)
                ),
                reward_function=CombinedReward(
                    (ball_proximity_reward, 0.7),
                    (touch_ball_reward, 1.0),
                    (velocity_to_ball_reward, 0.3)
                ),
                termination_condition=TouchBallCondition(),
                truncation_condition=short_timeout,
                difficulty_params={},
                success_threshold=0.6
            )
        ],
        base_task_prob=0.7,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.8, min_avg_reward=0.5, min_episodes=100,
            max_std_dev=1.0, required_consecutive_successes=3
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 1.2,
            "lr_critic": lr_critic * 1.2,
            "entropy_coef": 0.01
        },
        debug=debug
    )

    # ======== STAGE 2: SCORING (1v0) ========
    scoring_stage = SkillBasedCurriculumStage(
        name="Scoring",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(position_function=SafePositionWrapper(
                lambda: np.array([np.random.uniform(-2000, 2000), np.random.uniform(0, 3000), 93])
            )),
            CarPositionMutator(car_id="blue-0",
                              position_function=SafePositionWrapper(get_directional_shooting_car_position),
                              orientation_function=get_face_opp_goal_orientation)
        ),
        base_task_reward_function=CombinedReward(
            (GoalReward(), 2.0),
            (goal_velocity_reward, 0.5),
            (touch_ball_reward, 0.3),
            (ball_proximity_reward, 0.2)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=med_timeout,
        skill_modules=[
            SkillModule(
                name="Moving Ball Scoring",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=1, orange_size=0),
                    BallPositionMutator(position_function=SafePositionWrapper(
                        lambda: np.array([np.random.uniform(-2000, 2000), np.random.uniform(0, 2500), 93])
                    )),
                    BallVelocityMutator(velocity_function=SafePositionWrapper(
                        partial(get_ball_rolling_velocity, speed_range=(300, 600))
                    )),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_directional_shooting_car_position),
                                      orientation_function=get_face_opp_goal_orientation)
                ),
                reward_function=CombinedReward(
                    (GoalReward(), 2.0),
                    (goal_velocity_reward, 0.5),
                    (touch_ball_reward, 0.3)
                ),
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.6
            ),
            SkillModule(
                name="Bouncing Ball Scoring",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=1, orange_size=0),
                    BallPositionMutator(position_function=SafePositionWrapper(
                        lambda: np.array([np.random.uniform(-2000, 2000), np.random.uniform(0, 2500), 600])
                    )),
                    BallVelocityMutator(velocity_function=SafePositionWrapper(
                        lambda: np.array([0, 0, np.random.uniform(-300, -500)]) # Falling ball
                    )),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_directional_shooting_car_position),
                                      orientation_function=get_face_opp_goal_orientation)
                ),
                reward_function=CombinedReward(
                    (GoalReward(), 2.0),
                    (goal_velocity_reward, 0.5),
                    (touch_ball_reward, 0.3)
                ),
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.55
            )
        ],
        base_task_prob=0.7,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.65, min_avg_reward=0.4, min_episodes=125,
            max_std_dev=1.5, required_consecutive_successes=3
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "entropy_coef": 0.008
        },
        debug=debug
    )

    # ======== STAGE 3: BASIC 1v1 ========
    basic_1v1_stage = SkillBasedCurriculumStage(
        name="Basic 1v1",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator()
        ),
        base_task_reward_function=lucy_reward,
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=match_timeout,
        skill_modules=[
            SkillModule(
                name="Offensive Spawns",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=1, orange_size=1),
                    BallPositionMutator(position_function=SafePositionWrapper(
                        lambda: np.array([np.random.uniform(-2000, 2000), 2000, 93]) # Ball in orange half
                    )),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_directional_shooting_car_position),
                                      orientation_function=get_face_opp_goal_orientation),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(get_orange_primary_defender_position),
                                      orientation_function=get_face_own_goal_orientation)
                ),
                reward_function=lucy_reward,
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.6
            ),
            SkillModule(
                name="Defensive Spawns",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=1, orange_size=1),
                    BallPositionMutator(position_function=SafePositionWrapper(
                        lambda: np.array([np.random.uniform(-2000, 2000), -2000, 93]) # Ball in blue half
                    )),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_blue_primary_defender_position),
                                      orientation_function=get_face_own_goal_orientation),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(get_orange_attacker_position),
                                      orientation_function=get_face_opp_goal_orientation)
                ),
                reward_function=lucy_reward,
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.6
            ),
            SkillModule(
                name="Random Spawns",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=1, orange_size=1),
                    BallPositionMutator(position_function=SafePositionWrapper(
                        lambda: np.array([np.random.uniform(-3000, 3000), np.random.uniform(-3000, 3000), 93])
                    )),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_strategic_car_position),
                                      orientation_function=get_random_yaw_orientation),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(
                                          lambda: np.array([np.random.uniform(-3000, 3000), np.random.uniform(0, 4000), 17])
                                      ),
                                      orientation_function=get_random_yaw_orientation)
                ),
                reward_function=lucy_reward,
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.55
            )
        ],
        base_task_prob=0.7,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55, min_avg_reward=0.3, min_episodes=150,
            max_std_dev=2.0, required_consecutive_successes=2
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.9,
            "lr_critic": lr_critic * 0.9,
            "entropy_coef": 0.007
        },
        debug=debug
    )

    # ======== STAGE 4: ADVANCED MECHANICS & KICKOFFS (1v1) ========
    # Create the custom advanced kickoff reward
    advanced_kickoff_reward_1v1 = CombinedReward(
        (FirstTouchSpeedReward(weight=2.0, team_spirit=0.0), 1.0),  # Individual first touch reward
        (SpeedflipReward(weight=1.0), 0.7),                         # Speedflip mechanic reward
        (lucy_reward, 0.6),                                         # Base game reward
        (FlipResetReward(obtain_flip_weight=0.5, hit_ball_weight=1.0), 0.4),
        (WavedashReward(scale_by_acceleration=True), 0.4),
        (AerialDistanceReward(), 0.3),
        (AerialDirectionalTouchReward(), 0.3)
    )

    advanced_kickoffs_1v1_stage = SkillBasedCurriculumStage(
        name="Advanced Mechanics & Kickoffs (1v1)",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator()
        ),
        base_task_reward_function=advanced_kickoff_reward_1v1,
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=match_timeout,
        skill_modules=[
            SkillModule(
                name="Aerial Setups",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=1, orange_size=1),
                    BallPositionMutator(position_function=SafePositionWrapper(get_ball_aerial_position)),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_aerial_car_position),
                                      orientation_function=get_face_opp_goal_orientation),
                    CarBoostMutator(boost_amount=80, car_id="blue-0"),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(get_orange_primary_defender_position),
                                      orientation_function=get_face_own_goal_orientation)
                ),
                reward_function=CombinedReward(
                    (lucy_reward, 0.6),
                    (AerialControlReward(), 0.5),
                    (AerialDistanceReward(), 0.5),
                    (AerialDirectionalTouchReward(), 0.4),
                    (touch_ball_reward, 0.2)
                ),
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.5
            ),
            SkillModule(
                name="Wall Play Setups",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=1, orange_size=1),
                    BallPositionMutator(position_function=SafePositionWrapper(get_ball_wall_position)),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_car_wall_position),
                                      orientation_function=get_random_yaw_orientation),
                    CarBoostMutator(boost_amount=60, car_id="blue-0"),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(get_orange_primary_defender_position),
                                      orientation_function=get_face_own_goal_orientation)
                ),
                reward_function=CombinedReward(
                    (lucy_reward, 0.6),
                    (touch_ball_reward, 0.4),
                    (goal_velocity_reward, 0.3),
                    (AerialDirectionalTouchReward(), 0.2)
                ),
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.5
            ),
            SkillModule(
                name="Flip Reset Setups",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=1, orange_size=1),
                    BallPositionMutator(position_function=SafePositionWrapper(
                        lambda: np.array([np.random.uniform(-1000, 1000), 2500, np.random.uniform(900, 1200)])
                    )),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(
                                          lambda: np.array([np.random.uniform(-500, 500), 1500, 17])
                                      ),
                                      orientation_function=get_face_opp_goal_orientation),
                    CarBoostMutator(boost_amount=100, car_id="blue-0"),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(get_orange_primary_defender_position),
                                      orientation_function=get_face_own_goal_orientation)
                ),
                reward_function=CombinedReward(
                    (lucy_reward, 0.6),
                    (FlipResetReward(obtain_flip_weight=0.6, hit_ball_weight=1.0), 0.8),
                    (AerialControlReward(), 0.3),
                    (AerialDirectionalTouchReward(), 0.3)
                ),
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.4
            )
        ],
        base_task_prob=0.6,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5, min_avg_reward=0.3, min_episodes=200,
            max_std_dev=2.5, required_consecutive_successes=2
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.8,
            "lr_critic": lr_critic * 0.8,
            "entropy_coef": 0.006
        },
        debug=debug
    )

    # ======== STAGE 5: BASIC 2v2 ========
    basic_2v2_stage = SkillBasedCurriculumStage(
        name="Basic 2v2",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            KickoffMutator()
        ),
        base_task_reward_function=CombinedReward(
            (lucy_reward, 1.0),
            (TeamSpacingReward(), 0.3),
            (TeamPossessionReward(), 0.3)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=match_timeout,
        skill_modules=[
            SkillModule(
                name="Teammate Passing",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=2, orange_size=2),
                    BallPositionMutator(position_function=SafePositionWrapper(get_ball_passing_position)),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_blue_attacker_position),
                                      orientation_function=get_face_opp_goal_orientation),
                    CarPositionMutator(car_id="blue-1",
                                      position_function=SafePositionWrapper(get_blue_support_offensive_position),
                                      orientation_function=get_face_opp_goal_orientation),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(get_orange_primary_defender_position),
                                      orientation_function=get_face_own_goal_orientation),
                    CarPositionMutator(car_id="orange-1",
                                      position_function=SafePositionWrapper(get_orange_secondary_defender_position),
                                      orientation_function=get_face_own_goal_orientation)
                ),
                reward_function=CombinedReward(
                    (lucy_reward, 0.7),
                    (TeamSpacingReward(), 0.3),
                    (TeamPossessionReward(), 0.3),
                    (PassCompletionReward(), 0.5),
                    (ScoringOpportunityCreationReward(), 0.4)
                ),
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.5
            ),
            SkillModule(
                name="Team Defense",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=2, orange_size=2),
                    BallPositionMutator(position_function=SafePositionWrapper(get_defensive_ball_position)),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_blue_primary_defender_position),
                                      orientation_function=get_face_own_goal_orientation),
                    CarPositionMutator(car_id="blue-1",
                                      position_function=SafePositionWrapper(get_blue_secondary_defender_position),
                                      orientation_function=get_face_own_goal_orientation),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(get_orange_attacker_position),
                                      orientation_function=get_face_opp_goal_orientation),
                    CarPositionMutator(car_id="orange-1",
                                      position_function=SafePositionWrapper(get_orange_support_position),
                                      orientation_function=get_face_opp_goal_orientation)
                ),
                reward_function=CombinedReward(
                    (lucy_reward, 0.7),
                    (BlockSuccessReward(), 0.5),
                    (DefensivePositioningReward(), 0.4),
                    (BallClearanceReward(), 0.4),
                    (TeamSpacingReward(), 0.2)
                ),
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.5
            )
        ],
        base_task_prob=0.7,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55, min_avg_reward=0.3, min_episodes=200,
            max_std_dev=2.0, required_consecutive_successes=2
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.7,
            "lr_critic": lr_critic * 0.7,
            "entropy_coef": 0.005
        },
        debug=debug
    )

    # ======== STAGE 6: ADVANCED TEAM PLAY & KICKOFFS (2v2) ========
    # Create custom advanced team kickoff reward
    advanced_kickoff_reward_2v2 = CombinedReward(
        (FirstTouchSpeedReward(weight=2.0, team_spirit=1.0), 1.0),  # Team-based first touch reward
        (SpeedflipReward(weight=1.0), 0.5),                         # Speedflip mechanic reward
        (lucy_reward, 0.7),                                         # Base game reward
        (TeamSpacingReward(), 0.4),                                # Team spacing
        (PassCompletionReward(), 0.4),                              # Passing
        (ScoringOpportunityCreationReward(), 0.4),                  # Creating scoring opportunities
        (FlipResetReward(obtain_flip_weight=0.4, hit_ball_weight=0.8), 0.3),
        (WavedashReward(scale_by_acceleration=True), 0.3),
        (AerialDistanceReward(), 0.2),
        (AerialDirectionalTouchReward(), 0.2)
    )

    advanced_kickoffs_2v2_stage = SkillBasedCurriculumStage(
        name="Advanced Team Play & Kickoffs (2v2)",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            KickoffMutator()
        ),
        base_task_reward_function=advanced_kickoff_reward_2v2,
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=match_timeout,
        skill_modules=[
            SkillModule(
                name="Aerial Passing Setups",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=2, orange_size=2),
                    BallPositionMutator(position_function=SafePositionWrapper(
                        lambda: np.array([np.random.uniform(-1500, 1500), np.random.uniform(1500, 2500), np.random.uniform(600, 1000)])
                    )),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_blue_attacker_position),
                                      orientation_function=get_face_opp_goal_orientation),
                    CarBoostMutator(boost_amount=80, car_id="blue-0"),
                    CarPositionMutator(car_id="blue-1",
                                      position_function=SafePositionWrapper(get_blue_support_offensive_position),
                                      orientation_function=get_face_opp_goal_orientation),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(get_orange_primary_defender_position),
                                      orientation_function=get_face_own_goal_orientation),
                    CarPositionMutator(car_id="orange-1",
                                      position_function=SafePositionWrapper(get_orange_secondary_defender_position),
                                      orientation_function=get_face_own_goal_orientation)
                ),
                reward_function=CombinedReward(
                    (lucy_reward, 0.6),
                    (PassCompletionReward(), 0.6),
                    (ScoringOpportunityCreationReward(), 0.5),
                    (AerialControlReward(), 0.4),
                    (AerialDirectionalTouchReward(), 0.3),
                    (TeamSpacingReward(), 0.3)
                ),
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.5
            ),
            SkillModule(
                name="Coordinated Defense Setups",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=2, orange_size=2),
                    BallPositionMutator(position_function=SafePositionWrapper(get_defensive_save_ball_position)),
                    BallVelocityMutator(velocity_function=SafePositionWrapper(get_defensive_save_ball_velocity)),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_blue_primary_defender_position),
                                      orientation_function=get_face_own_goal_orientation),
                    CarPositionMutator(car_id="blue-1",
                                      position_function=SafePositionWrapper(get_blue_secondary_defender_position),
                                      orientation_function=get_face_own_goal_orientation),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(get_orange_attacker_position),
                                      orientation_function=get_face_opp_goal_orientation),
                    CarPositionMutator(car_id="orange-1",
                                      position_function=SafePositionWrapper(get_orange_support_position),
                                      orientation_function=get_face_opp_goal_orientation)
                ),
                reward_function=CombinedReward(
                    (lucy_reward, 0.6),
                    (BlockSuccessReward(), 0.6),
                    (DefensivePositioningReward(), 0.5),
                    (BallClearanceReward(), 0.5),
                    (TeamSpacingReward(), 0.4)
                ),
                termination_condition=goal_condition,
                truncation_condition=med_timeout,
                difficulty_params={},
                success_threshold=0.5
            ),
            SkillModule(
                name="Complex Random Spawns",
                state_mutator=MutatorSequence(
                    FixedTeamSizeMutator(blue_size=2, orange_size=2),
                    BallPositionMutator(position_function=SafePositionWrapper(
                        lambda: np.array([np.random.uniform(-3000, 3000), np.random.uniform(-3000, 3000), np.random.uniform(93, 800)])
                    )),
                    BallVelocityMutator(velocity_function=SafePositionWrapper(
                        lambda: np.array([np.random.uniform(-500, 500), np.random.uniform(-500, 500), np.random.uniform(-100, 300)])
                    )),
                    CarPositionMutator(car_id="blue-0",
                                      position_function=SafePositionWrapper(get_strategic_car_position),
                                      orientation_function=get_random_yaw_orientation),
                    CarPositionMutator(car_id="blue-1",
                                      position_function=SafePositionWrapper(get_strategic_car_position),
                                      orientation_function=get_random_yaw_orientation),
                    CarPositionMutator(car_id="orange-0",
                                      position_function=SafePositionWrapper(
                                          lambda: np.array([np.random.uniform(-3000, 3000), np.random.uniform(0, 4000), 17])
                                      ),
                                      orientation_function=get_random_yaw_orientation),
                    CarPositionMutator(car_id="orange-1",
                                      position_function=SafePositionWrapper(
                                          lambda: np.array([np.random.uniform(-3000, 3000), np.random.uniform(0, 4000), 17])
                                      ),
                                      orientation_function=get_random_yaw_orientation)
                ),
                reward_function=CombinedReward(
                    (lucy_reward, 0.7),
                    (PassCompletionReward(), 0.4),
                    (TeamSpacingReward(), 0.4),
                    (AerialControlReward(), 0.2),
                    (FlipResetReward(obtain_flip_weight=0.3, hit_ball_weight=0.7), 0.2),
                    (DefensivePositioningReward(), 0.2)
                ),
                termination_condition=goal_condition,
                truncation_condition=long_timeout,
                difficulty_params={},
                success_threshold=0.45
            )
        ],
        base_task_prob=0.6,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5, min_avg_reward=0.3, min_episodes=250,
            max_std_dev=2.5, required_consecutive_successes=2
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.6,
            "lr_critic": lr_critic * 0.6,
            "entropy_coef": 0.004
        },
        debug=debug
    )

    # Create the full curriculum list
    stages = []

    # Only include pretraining stage if use_pretraining is True
    if use_pretraining:
        stages.append(pretraining)

    # Add all stages based on revised progression path
    stages.extend([
        ball_touches_stage,              # Stage 1: Ball Touches (1v0)
        scoring_stage,                   # Stage 2: Scoring (1v0)
        basic_1v1_stage,                 # Stage 3: Basic 1v1
        advanced_kickoffs_1v1_stage,     # Stage 4: Advanced Mechanics & Kickoffs (1v1)
        basic_2v2_stage,                 # Stage 5: Basic 2v2
        advanced_kickoffs_2v2_stage      # Stage 6: Advanced Team Play & Kickoffs (2v2)
    ])

    curriculum_manager = CurriculumManager(
        stages=stages,
        max_rehearsal_stages=3,
        rehearsal_decay_factor=0.7,
        evaluation_window=50,
        debug=debug,
        use_wandb=use_wandb
    )

    return curriculum_manager
