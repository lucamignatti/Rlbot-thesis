# Rlbot-thesis/curriculum/curriculum.py
from .base import CurriculumManager, CurriculumStage, ProgressionRequirements
from .mutators import (
    BallTowardGoalSpawnMutator, BallPositionMutator, CarBallRelativePositionMutator,
    CarBoostMutator, BallVelocityMutator, CarPositionMutator, TouchBallCondition
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
from functools import partial
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import random

# --- Position/Orientation Helper Functions (Keep existing ones) ---

def get_random_yaw_orientation(*args) -> np.ndarray:
    """Returns a random yaw orientation (rotation around Z-axis)."""
    yaw = np.random.uniform(-np.pi, np.pi)
    return np.array([0, yaw, 0]) # Pitch, Yaw, Roll

def get_face_opp_goal_orientation(*args) -> np.ndarray:
    """Returns orientation facing the opponent's goal (default blue team)."""
    return np.array([0, 0, 0])

def get_face_own_goal_orientation(*args) -> np.ndarray:
    """Returns orientation facing own goal (default blue team)."""
    return np.array([0, np.pi, 0])

class SafePositionWrapper:
    """Wrapper class for position functions to ensure they always return valid coordinates"""
    def __init__(self, func):
        self.func = func
        self.default_car_position = np.array([0.0, -2000.0, 17.0], dtype=np.float32)  # Safe default car position
        self.default_ball_position = np.array([0.0, 0.0, 93.0], dtype=np.float32)  # Safe default ball position

        # Flag to identify if this is a relative position function
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
            position = self.func(*args, **kwargs)
            if position is None:
                func_name = getattr(self.func, '__name__', str(self.func))
                print(f"WARNING: Position function {func_name} returned None")
                if self.is_relative:
                    return np.array([0.0, 200.0, 93.0], dtype=np.float32)
                if 'car' in str(self.func).lower():
                    return self.default_car_position.copy()
                else:
                    return self.default_ball_position.copy()
            if not isinstance(position, np.ndarray):
                position = np.array(position, dtype=np.float32)
            if np.isnan(position).any() or np.isinf(position).any():
                func_name = getattr(self.func, '__name__', str(self.func))
                print(f"WARNING: Position function {func_name} returned invalid values: {position}")
                if 'car' in str(self.func).lower():
                    return self.default_car_position.copy()
                else:
                    return self.default_ball_position.copy()
            return position
        except Exception as e:
            func_name = getattr(self.func, '__name__', str(self.func))
            print(f"ERROR in position function {func_name}: {str(e)}")
            if 'car' in str(self.func).lower():
                return self.default_car_position.copy()
            else:
                return self.default_ball_position.copy()

    def __str__(self):
        try:
            result = self()
            func_name = getattr(self.func, '__name__', str(self.func))
            return f"Position from {func_name}: {result}"
        except:
            return f"Position function (error evaluating)"

def create_curriculum(debug=False, use_wandb=True, lr_actor=None, lr_critic=None, use_pretraining=True):
    """
    Create a 2v2 focused curriculum based on the RLBot guide.

    Args:
        debug: Whether to print debug information
        use_wandb: Whether to use Weights & Biases for logging
        lr_actor: Learning rate for the actor network (overrides default values)
        lr_critic: Learning rate for the critic network (overrides default values)
        use_pretraining: Flag indicating if pretraining was requested (ignored in this impl)

    Returns:
        CurriculumManager: The created curriculum manager
    """
    # Use provided learning rates or fall back to defaults
    default_lr_actor = 0.0002
    default_lr_critic = 0.0003 # Slightly higher default critic LR
    lr_actor = lr_actor if lr_actor is not None else default_lr_actor
    lr_critic = lr_critic if lr_critic is not None else default_lr_critic

    if debug:
        print(f"[DEBUG Curriculum] Creating 2v2 curriculum. LR Actor: {lr_actor}, LR Critic: {lr_critic}")
        if use_pretraining:
            print("[DEBUG Curriculum] Note: use_pretraining=True flag ignored, starting directly with 2v2 stages.")

    # --- Constants and Common Components ---
    VERY_SHORT_TIMEOUT = 5
    SHORT_TIMEOUT = 10
    MED_TIMEOUT = 20
    LONG_TIMEOUT = 45
    MATCH_TIMEOUT = 300

    goal_condition = GoalCondition()
    timeout_vs = TimeoutCondition(VERY_SHORT_TIMEOUT)
    timeout_s = TimeoutCondition(SHORT_TIMEOUT)
    timeout_m = TimeoutCondition(MED_TIMEOUT)
    timeout_l = TimeoutCondition(LONG_TIMEOUT)
    timeout_match = TimeoutCondition(MATCH_TIMEOUT)
    touch_condition = TouchBallCondition() # Assuming this exists and works

    # --- Reward Components (Instantiate once) ---
    touch_ball_reward = TouchBallReward()
    velocity_to_ball_reward = PlayerVelocityTowardBallReward()
    ball_proximity_reward = BallProximityReward(negative_slope=True) # Reward closer proximity more
    save_boost_reward = SaveBoostReward()
    velocity_ball_to_goal_reward = BallVelocityToGoalReward(team_goal_y=5120)
    goal_reward = GoalReward() # Default goal reward/penalty
    touch_accel_reward = TouchBallToGoalAccelerationReward(team_goal_y=5120)
    offensive_potential_krc = create_offensive_potential_reward(team_goal_y=5120)
    block_success_reward = BlockSuccessReward(goal_y=5120)
    defensive_positioning_reward = DefensivePositioningReward(goal_y=5120)
    ball_clearance_reward = BallClearanceReward(goal_y=5120)
    team_spacing_reward = TeamSpacingReward()
    team_possession_reward = TeamPossessionReward()
    aerial_control_reward = AerialControlReward()
    aerial_touch_reward = AerialDirectionalTouchReward(goal_y=5120)
    lucy_skg_reward = create_lucy_skg_reward(team_goal_y=5120) # Base full game reward
    pass_completion_reward = PassCompletionReward()
    opportunity_creation_reward = ScoringOpportunityCreationReward(goal_y=5120)
    # Optional advanced mechanics
    # flip_reset_reward = FlipResetReward()
    # wavedash_reward = WavedashReward()
    # speedflip_reward = SpeedflipReward()

    # --- Mutators ---
    kickoff_mutator = KickoffMutator()
    fixed_2v2_mutator = FixedTeamSizeMutator(blue_size=2, orange_size=2)

    # Basic random ground spawn
    basic_ground_spawn = MutatorSequence(
        fixed_2v2_mutator,
        BallPositionMutator(position_function=SafePositionWrapper(lambda: np.array([random.uniform(-3000, 3000), random.uniform(-4000, 4000), 93]))),
        CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(-4500, -1000), 17])), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(-4500, -1000), 17])), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(1000, 4500), 17])), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(1000, 4500), 17])), orientation_function=get_random_yaw_orientation),
    )

    # Offensive scenario spawn
    offensive_spawn = MutatorSequence(
        fixed_2v2_mutator,
        BallPositionMutator(position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2500, 2500), random.uniform(1000, 4000), 93]))), # Ball in opponent half
        CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(-1000, 2000), 17])), orientation_function=get_face_opp_goal_orientation),
        CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(-1000, 2000), 17])), orientation_function=get_face_opp_goal_orientation),
        # Opponents further back
        CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-1500, 1500), random.uniform(3500, 4800), 17])), orientation_function=get_face_own_goal_orientation),
        CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-1500, 1500), random.uniform(3500, 4800), 17])), orientation_function=get_face_own_goal_orientation),
    )

    # Defensive scenario spawn
    defensive_spawn = MutatorSequence(
        fixed_2v2_mutator,
        BallPositionMutator(position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2500, 2500), random.uniform(-4000, -1000), 93]))), # Ball in own half
        BallVelocityMutator(velocity_function=lambda: np.array([random.uniform(-500, 500), random.uniform(-1500, -500), 0])), # Ball moving towards own goal
        CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-1500, 1500), random.uniform(-4800, -3500), 17])), orientation_function=get_face_own_goal_orientation),
        CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-1500, 1500), random.uniform(-4800, -3500), 17])), orientation_function=get_face_own_goal_orientation),
        # Opponents further up
        CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(-2000, 1000), 17])), orientation_function=get_face_opp_goal_orientation),
        CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(-2000, 1000), 17])), orientation_function=get_face_opp_goal_orientation),
    )

    # Aerial scenario spawn
    aerial_spawn = MutatorSequence(
        fixed_2v2_mutator,
        BallPositionMutator(position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(-3000, 3000), random.uniform(300, 1200)]))), # Ball in air
        BallVelocityMutator(velocity_function=lambda: np.array([random.uniform(-500, 500), random.uniform(-500, 500), random.uniform(-200, 200)])), # Slow random velocity
        # Cars start grounded
        CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(-4500, -1000), 17])), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(-4500, -1000), 17])), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(1000, 4500), 17])), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(lambda: np.array([random.uniform(-2000, 2000), random.uniform(1000, 4500), 17])), orientation_function=get_random_yaw_orientation),
    )

    # --- Stage Definitions ---

    # STAGE 1: 2v2 Basic Interaction & Movement
    stage1 = CurriculumStage(
        name="2v2 Basic Interaction",
        state_mutator=MutatorSequence(fixed_2v2_mutator, kickoff_mutator),
        reward_function=CombinedReward(
            (touch_ball_reward, 1.5),          # High weight for touching
            (velocity_to_ball_reward, 0.5),
            (ball_proximity_reward, 0.3),
            (save_boost_reward, 0.1)
        ),
        termination_condition=timeout_s, # End quickly after kickoff phase
        truncation_condition=timeout_s,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.0, # Not applicable here
            min_avg_reward=0.1,   # Just need some positive interaction signal
            min_episodes=100,
            max_std_dev=5.0,
            required_consecutive_successes=1 # Not applicable
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor,       # Start with base LR
            "lr_critic": lr_critic,
            "entropy_coef": 0.01      # Higher entropy for exploration
        }
    )

    # STAGE 2: 2v2 Directional Ball Control
    stage2 = CurriculumStage(
        name="2v2 Directional Control",
        state_mutator=MutatorSequence(fixed_2v2_mutator, kickoff_mutator),
        reward_function=CombinedReward(
            (velocity_ball_to_goal_reward, 0.8), # Start encouraging direction
            (touch_ball_reward, 0.5),          # Lower touch reward
            (velocity_to_ball_reward, 0.2),
            (ball_proximity_reward, 0.1),
            (save_boost_reward, 0.3)           # Increase boost saving importance
        ),
        termination_condition=goal_condition, # End on goal or timeout
        truncation_condition=timeout_m,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.0,
            min_avg_reward=0.2,   # Expect slightly higher reward with directional push
            min_episodes=150,
            max_std_dev=4.0,
            required_consecutive_successes=2
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.9,
            "lr_critic": lr_critic * 0.9,
            "entropy_coef": 0.008
        }
    )

    # STAGE 3: 2v2 Basic Offense & Scoring
    stage3 = CurriculumStage(
        name="2v2 Basic Offense",
        state_mutator=offensive_spawn, # Use offensive scenarios
        reward_function=CombinedReward(
            (goal_reward, 5.0),                  # Moderate goal reward (as per guide)
            (velocity_ball_to_goal_reward, 1.0), # Stronger push to goal
            (touch_accel_reward, 0.5),           # Reward accelerating ball to goal
            (offensive_potential_krc, 0.3),      # Basic offensive positioning
            (save_boost_reward, 0.2)
        ),
        termination_condition=goal_condition,
        truncation_condition=timeout_m,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.4, # Expect some goals now
            min_avg_reward=0.3,
            min_episodes=200,
            max_std_dev=3.0,
            required_consecutive_successes=3
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.8,
            "lr_critic": lr_critic * 0.8,
            "entropy_coef": 0.007
        }
    )

    # STAGE 4: 2v2 Basic Defense & Saves
    stage4 = CurriculumStage(
        name="2v2 Basic Defense",
        state_mutator=defensive_spawn, # Use defensive scenarios
        reward_function=CombinedReward(
            (block_success_reward, 1.5),         # Reward successful blocks
            (defensive_positioning_reward, 1.0), # Reward staying between ball and goal
            (ball_clearance_reward, 0.7),        # Reward clearing the ball
            (save_boost_reward, 0.3)
            # Goal reward is implicitly negative via termination
        ),
        termination_condition=goal_condition, # Opponent scoring ends episode
        truncation_condition=timeout_l,       # Longer timeout for defensive practice
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5, # High success = preventing goals
            min_avg_reward=0.25,  # Lower avg reward expected in defense
            min_episodes=200,
            max_std_dev=2.5,
            required_consecutive_successes=3
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.7,
            "lr_critic": lr_critic * 0.7,
            "entropy_coef": 0.006
        }
    )

    # STAGE 5: 2v2 Team Positioning & Spacing
    stage5 = CurriculumStage(
        name="2v2 Team Positioning",
        state_mutator=MutatorSequence(fixed_2v2_mutator, kickoff_mutator),
        reward_function=CombinedReward(
            (lucy_skg_reward, 0.6),           # Base reward for general play
            (team_spacing_reward, 0.8),       # Strongly reward good spacing
            (team_possession_reward, 0.5)     # Reward keeping ball near team
        ),
        termination_condition=goal_condition,
        truncation_condition=timeout_match,   # Full match length
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5, # Based on goals (proxy for effectiveness)
            min_avg_reward=0.3,
            min_episodes=250,
            max_std_dev=2.0,
            required_consecutive_successes=2
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.6,
            "lr_critic": lr_critic * 0.6,
            "entropy_coef": 0.005
        }
    )

    # STAGE 6: 2v2 Aerial Foundations
    stage6 = CurriculumStage(
        name="2v2 Aerial Foundations",
        state_mutator=aerial_spawn, # Use aerial scenarios
        reward_function=CombinedReward(
            (lucy_skg_reward, 0.5),              # Base reward
            (aerial_touch_reward, 1.0),          # Reward touching ball in air towards goal
            (aerial_control_reward, 0.7),        # Reward stable aerial control
            # Optional: Add AerialDistanceReward if complex aerials are desired later
        ),
        termination_condition=goal_condition,
        truncation_condition=timeout_l,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.45, # Aerials are harder
            min_avg_reward=0.25,
            min_episodes=300,
            max_std_dev=2.5,
            required_consecutive_successes=2
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.5,
            "lr_critic": lr_critic * 0.5,
            "entropy_coef": 0.004
        }
    )

    # STAGE 7: 2v2 Full Game & Advanced Play
    stage7 = CurriculumStage(
        name="2v2 Full Game",
        state_mutator=MutatorSequence(fixed_2v2_mutator, kickoff_mutator),
        reward_function=CombinedReward(
             (lucy_skg_reward, 1.0),              # Primary reward function
             (pass_completion_reward, 0.3),       # Add small pass incentive
             (opportunity_creation_reward, 0.2)   # Add small opportunity incentive
             # Optional: Add advanced mechanics rewards here if needed
             # (speedflip_reward, 0.1),
             # (wavedash_reward, 0.1),
        ),
        termination_condition=goal_condition,
        truncation_condition=timeout_match,
        progression_requirements=ProgressionRequirements( # Final stage requirements are less critical
            min_success_rate=0.5,
            min_avg_reward=0.3,
            min_episodes=500, # Train longer in final stage
            max_std_dev=2.0,
            required_consecutive_successes=2
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.4, # Lowest LR
            "lr_critic": lr_critic * 0.4,
            "entropy_coef": 0.003      # Lowest entropy
        }
    )

    # --- Assemble Curriculum ---
    stages = [
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6,
        stage7
    ]

    curriculum_manager = CurriculumManager(
        stages=stages,
        max_rehearsal_stages=2,      # Allow rehearsing previous 2 stages
        rehearsal_decay_factor=0.6,
        evaluation_window=100,       # Check progression every 100 episodes
        debug=debug,
        use_wandb=use_wandb
    )

    # Validate the created curriculum
    curriculum_manager.validate_all_stages()

    return curriculum_manager
