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
    BaseRewardFunction, ZeroSumRewardWrapper, BallProximityReward, BallToGoalDistanceReward, TouchBallReward,
    BallVelocityToGoalReward, AlignBallToGoalReward, SaveBoostReward,
    KRCReward, PlayerVelocityTowardBallReward, TouchBallToGoalAccelerationReward,
    PassCompletionReward, ScoringOpportunityCreationReward, AerialControlReward,
    AerialDirectionalTouchReward, BlockSuccessReward, DefensivePositioningReward,
    BallClearanceReward, TeamSpacingReward, TeamPossessionReward, AirReward,
    create_distance_weighted_alignment_reward, create_offensive_potential_reward, create_lucy_skg_reward,
    AerialDistanceReward, FlipResetReward, WavedashReward, FirstTouchSpeedReward, SpeedflipReward, LucySKGReward,
    DistanceWeightedAlignmentKRC, OffensivePotentialKRC
)

# Advanced touch reward that scales with touch strength as per guide recommendation
class TouchBallVelocityReward(TouchBallReward):
    """A more advanced touch reward that scales with the strength of the touch.
    Following the guide recommendation for Middle Stages.
    """
    def __init__(self, min_vel_change=100, max_vel_change=4000):
        super().__init__()
        self.min_vel_change = min_vel_change  # Minimum velocity change to count
        self.max_vel_change = max_vel_change  # Maximum velocity change for scaling
        self.prev_ball_vel = None

    def reset(self, initial_state):
        super().reset(initial_state)
        self.prev_ball_vel = None

    def get_reward(self, player, state, previous_action):
        # Check if player touched the ball
        if player.ball_touched:
            # Get current ball velocity
            curr_ball_vel = state.ball.linear_velocity

            # If we have previous velocity, calculate the change
            if self.prev_ball_vel is not None:
                # Calculate velocity change magnitude
                vel_change = np.linalg.norm(curr_ball_vel - self.prev_ball_vel)

                # Scale the reward based on velocity change
                if vel_change > self.min_vel_change:
                    # Clamp to max and normalize to 0-1
                    normalized_change = min(vel_change, self.max_vel_change) / self.max_vel_change
                    return normalized_change

            # Default touch reward if we can't calculate velocity change
            return 0.2

        # Store current velocity for next step
        self.prev_ball_vel = state.ball.linear_velocity.copy()
        return 0
from functools import partial
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import random

# Enhanced aerial touch reward that combines height and air time as recommended by the guide
class EnhancedAerialTouchReward(AerialDirectionalTouchReward):
    """
    An improved aerial touch reward that scales with both ball height and player air time.
    Based on the guide recommendation for combining these factors to encourage proper aerials.
    """
    def __init__(self, goal_y=5120, max_time_in_air=1.75):
        super().__init__(goal_y=goal_y)
        self.max_time_in_air = max_time_in_air
        self.player_air_time = {}  # Track air time for each player

    def reset(self, initial_state):
        super().reset(initial_state)
        self.player_air_time = {}

    def get_reward(self, player, state, previous_action):
        # Base reward from parent class (directional touch)
        directional_touch_reward = super().get_reward(player, state, previous_action)

        # Update air time tracking
        player_id = player.car_id
        if player_id not in self.player_air_time:
            self.player_air_time[player_id] = 0

        if not player.on_ground:
            self.player_air_time[player_id] += state.dt
        else:
            self.player_air_time[player_id] = 0

        # Calculate height fraction
        ceiling_height = 2044  # Ceiling height in Rocket League
        height_frac = state.ball.position[2] / ceiling_height

        # Calculate air time fraction
        air_time_frac = min(self.player_air_time[player_id], self.max_time_in_air) / self.max_time_in_air

        # Combine the rewards - only add the air/height component if we have a touch
        if directional_touch_reward > 0:
            # Use minimum to require both height and air time
            height_air_factor = min(air_time_frac, height_frac)
            return directional_touch_reward * (1 + height_air_factor)

        return directional_touch_reward

# Import standard Python libraries
from typing import Any, Dict, List, Tuple
import numpy as np
import random
import inspect

# Import the RLGym reward functions
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

# Try to import reward functions - these may fail if libraries aren't installed
try:
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    RLGYM_REWARDS_AVAILABLE = True
except ImportError:
    print("Warning: rlgym reward functions not available. CustomReward will be used instead.")
    RLGYM_REWARDS_AVAILABLE = False

# Try to import rlgym-tools reward functions
try:
    from rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward
    from rlgym_tools.rocket_league.reward_functions.ball_travel_reward import BallTravelReward
    from rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward
    from rlgym_tools.rocket_league.reward_functions.boost_change_reward import BoostChangeReward
    from rlgym_tools.rocket_league.reward_functions.goal_prob_reward import GoalViewReward
    from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import AerialDistanceReward
    from rlgym_tools.rocket_league.reward_functions.team_spirit_reward_wrapper import TeamSpiritRewardWrapper
    from rlgym_tools.rocket_league.reward_functions.flip_reset_reward import FlipResetReward
    from rlgym_tools.rocket_league.reward_functions.wavedash_reward import WavedashReward
    from rlgym_tools.rocket_league.reward_functions.demo_reward import DemoReward
    RLGYM_TOOLS_AVAILABLE = True
except ImportError:
    print("Warning: rlgym-tools reward functions not available. CustomReward will be used instead.")
    RLGYM_TOOLS_AVAILABLE = False



# Adapter class is no longer needed; using ZeroSumRewardWrapper from .rewards
    def __init__(self, reward_function):
        super().__init__()
        self.reward_function = reward_function
        self.agents = {}  # Map car_id to AgentID
        self.current_state = None
        self.cached_rewards = {}

    def reset(self, initial_state):
        """
        Reset method for curriculum interface
        """
        # Create a simplified GameState from initial_state for the reset method
        game_state = self._convert_to_game_state(initial_state)

        # Initialize agent_ids map
        self.agents = {}
        for player in initial_state.players:
            self.agents[player.car_id] = player.car_id  # Use car_id as the AgentID

        # Call the wrapped function's reset
        agent_ids = list(self.agents.values())
        shared_info = {}
        # Use rlgym_reset if available, otherwise fall back to reset
        if hasattr(self.reward_function, 'rlgym_reset'):
            self.reward_function.rlgym_reset(agent_ids, game_state, shared_info)
        elif hasattr(self.reward_function, 'reset'):
            self.reward_function.reset(agent_ids, game_state, shared_info)
        self.cached_rewards = {}

    # Method for curriculum interface
    def get_reward(self, player, state, previous_action):
        # Convert state to GameState expected by rlgym
        game_state = self._convert_to_game_state(state)
        self.current_state = game_state

        # Make sure this agent is in our map
        if player.car_id not in self.agents:
            self.agents[player.car_id] = player.car_id

        # We only need the reward for one agent
        agent_id = self.agents[player.car_id]
        is_terminated = {agent_id: False}
        is_truncated = {agent_id: False}
        shared_info = {}

        # Call the wrapped function and extract just the reward we need
        rewards = self.reward_function.get_rewards([agent_id], game_state, is_terminated, is_truncated, shared_info)
        # Cache the reward for potential future calls to get_rewards
        self.cached_rewards[agent_id] = rewards[agent_id]
        return rewards[agent_id]

    def rlgym_reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        """Reset implementation for rlgym API - won't be called in current curriculum"""
        self.agents = {agent: agent for agent in agents}
        # Use rlgym_reset if available, otherwise fall back to reset
        if hasattr(self.reward_function, 'rlgym_reset'):
            self.reward_function.rlgym_reset(agents, initial_state, shared_info)
        elif hasattr(self.reward_function, 'reset') and len(inspect.signature(self.reward_function.reset).parameters) >= 3:
            self.reward_function.reset(agents, initial_state, shared_info)
        self.cached_rewards = {}

    def get_rewards(self, agents: List[AgentID], state: GameState,
                   is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool],
                   shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        """Get rewards implementation for rlgym API"""
        self.current_state = state
        return self.reward_function.get_rewards(agents, state, is_terminated, is_truncated, shared_info)

    def _convert_to_game_state(self, state):
        """Create a simplified GameState compatible with rlgym from our state"""
        # This is a simplified adapter - you may need to add more state conversions
        class SimpleGameState:
            def __init__(self, state):
                self.ball = state.ball
                self.cars = {player.car_id: player for player in state.players}
                self.goal_scored = hasattr(state, 'goal_scored') and state.goal_scored
                self.scoring_team = getattr(state, 'scoring_team', -1)
                self.players = state.players
                # Add any other required fields for rlgym compatibility

        return SimpleGameState(state)

# --- Configure RLGym-Based Reward Functions ---
def create_phase1_reward():
    """
    Phase 1: Basic Ball Interaction
    Rewards for touching the ball and moving towards it
    Based on the RLGym-PPO guide recommendation for early training
    """
    # Main components with appropriate weights
    velocity_reward = VelocityPlayerToBallReward(include_negative_values=True)
    touch_reward = AdvancedTouchReward(touch_reward=10.0, acceleration_reward=5.0)
    ball_travel_reward = BallTravelReward(consecutive_weight=2.0, goal_weight=3.0)

    # Combine rewards
    combined = CombinedReward(
        (velocity_reward, 5.0),
        (touch_reward, 50.0),
        (ball_travel_reward, 1.0)
    )

    # Wrap in zero-sum reward wrapper
    return combined

def create_phase2_reward():
    """
    Phase 2: Goal Scoring Focus
    Rewards for moving the ball toward the goal and scoring
    """
    # Main components with appropriate weights
    velocity_reward = VelocityPlayerToBallReward(include_negative_values=True)
    touch_reward = AdvancedTouchReward(touch_reward=5.0, acceleration_reward=7.0)
    ball_travel_reward = BallTravelReward(consecutive_weight=1.0, goal_weight=5.0)
    goal_reward = GoalReward()
    goal_prob_reward = GoalViewReward(gamma=0.99)
    boost_reward = BoostChangeReward(gain_weight=0.5, lose_weight=0.3)

    # Combine rewards
    combined = CombinedReward(
        (velocity_reward, 5.0),
        (touch_reward, 25.0),
        (ball_travel_reward, 10.0),
        (goal_reward, 200.0),
        (goal_prob_reward, 10.0),
        (boost_reward, 10.0)
    )

    # Wrap in zero-sum reward wrapper
    return ZeroSumRewardWrapper(combined, team_spirit=0.3)

def create_phase3_reward():
    """
    Phase 3: Advanced Play
    Full game rewards including aerial play and team coordination
    """
    # Basic components
    velocity_reward = VelocityPlayerToBallReward(include_negative_values=True)
    touch_reward = AdvancedTouchReward(touch_reward=3.0, acceleration_reward=5.0)
    ball_travel_reward = BallTravelReward(consecutive_weight=1.0, goal_weight=5.0,
                                          pass_weight=2.0, receive_weight=2.0)
    goal_reward = GoalReward()
    goal_prob_reward = GoalViewReward(gamma=0.99)
    boost_reward = BoostChangeReward(gain_weight=0.3, lose_weight=0.2)

    # Advanced mechanics components
    aerial_reward = AerialDistanceReward(touch_height_weight=1.0, ball_distance_weight=1.0)
    flip_reset_reward = FlipResetReward()
    demo_reward = DemoReward()
    wavedash_reward = WavedashReward()

    # Combine all rewards with appropriate weights
    combined = CombinedReward(
        (velocity_reward, 10.0),
        (touch_reward, 25.0),
        (ball_travel_reward, 5.0),
        (goal_reward, 200.0),
        (goal_prob_reward, 25.0),
        (boost_reward, 15.0),
        (aerial_reward, 50.0),
        (flip_reset_reward, 15.0),
        (demo_reward, 20.0),
        (wavedash_reward, 10.0)
    )

    # Wrap in zero-sum reward wrapper
    return ZeroSumRewardWrapper(combined, team_spirit=0.5)

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

# Define position and velocity functions outside of create_curriculum for pickling compatibility
def random_ground_ball_position():
    return np.array([random.uniform(-3000, 3000), random.uniform(-4000, 4000), 93])

def random_blue_car_position():
    return np.array([random.uniform(-2000, 2000), random.uniform(-4500, -1000), 17])

def random_orange_car_position():
    return np.array([random.uniform(-2000, 2000), random.uniform(1000, 4500), 17])

def center_ball_position():
    return np.array([0, 0, 93.15])

def blue_left_kickoff_position():
    return np.array([-2048, -2560, 17.01])

def blue_right_kickoff_position():
    return np.array([2048, -2560, 17.01])
    
def orange_left_kickoff_position():
    return np.array([-2048, 2560, 17.01])
    
def orange_right_kickoff_position():
    return np.array([2048, 2560, 17.01])

def offensive_ball_position():
    return np.array([random.uniform(-2500, 2500), random.uniform(1000, 4000), 93])

def offensive_blue_car_position():
    return np.array([random.uniform(-2000, 2000), random.uniform(-1000, 2000), 17])

def defensive_orange_car_position():
    return np.array([random.uniform(-1500, 1500), random.uniform(3500, 4800), 17])

def defensive_ball_position():
    return np.array([random.uniform(-2500, 2500), random.uniform(-4000, -1000), 93])

def defensive_ball_velocity():
    return np.array([random.uniform(-500, 500), random.uniform(-1500, -500), 0])

def defensive_blue_car_position():
    return np.array([random.uniform(-1500, 1500), random.uniform(-4800, -3500), 17])

def offensive_orange_car_position():
    return np.array([random.uniform(-2000, 2000), random.uniform(-2000, 1000), 17])

def aerial_ball_position():
    return np.array([random.uniform(-2000, 2000), random.uniform(-3000, 3000), random.uniform(300, 1200)])

def aerial_ball_velocity():
    return np.array([random.uniform(-500, 500), random.uniform(-500, 500), random.uniform(-200, 200)])

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
    default_lr_actor = 0.0003 # Adjusted default
    default_lr_critic = 0.0003 # Adjusted default
    lr_actor = lr_actor if lr_actor is not None else default_lr_actor
    lr_critic = lr_critic if lr_critic is not None else default_lr_critic

    if debug:
        print(f"[DEBUG Curriculum] Creating 3-stage curriculum. Initial LR Actor: {lr_actor}, Initial LR Critic: {lr_critic}")
        if use_pretraining: # This flag is noted as ignored but kept for compatibility
            print("[DEBUG Curriculum] Note: use_pretraining=True flag is present but this curriculum starts directly with 2v2 stages.")

    # --- Constants and Common Components ---
    VERY_SHORT_TIMEOUT = 10 # Increased slightly
    MED_TIMEOUT = 30
    MATCH_TIMEOUT = 300 # Standard match time

    goal_condition = GoalCondition()
    # Timeout conditions, instantiated per stage for clarity or if values differ significantly
    # For simplicity, we can define them here if they are reused with these exact values.
    timeout_touch_stage = TimeoutCondition(VERY_SHORT_TIMEOUT)
    timeout_score_stage = TimeoutCondition(MED_TIMEOUT)
    timeout_full_game_stage = TimeoutCondition(MATCH_TIMEOUT)

    # TouchBallCondition might be useful for the first stage's termination
    touch_ball_condition = TouchBallCondition()

    # --- Keeping some of the existing mutators and position functions ---

    # Use imported reward functions instead of custom LucySKGReward

    # --- Mutators ---
    kickoff_mutator = KickoffMutator()
    fixed_2v2_mutator = FixedTeamSizeMutator(blue_size=2, orange_size=2)

    # Basic random ground spawn
    basic_ground_spawn = MutatorSequence(
        fixed_2v2_mutator,
        BallPositionMutator(position_function=SafePositionWrapper(random_ground_ball_position)),
        CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(random_blue_car_position), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(random_blue_car_position), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(random_orange_car_position), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(random_orange_car_position), orientation_function=get_random_yaw_orientation),
    )

    # Simple spawn for touch stage (ball at center, cars in kickoff pos)
    simple_touch_spawn = MutatorSequence(
        fixed_2v2_mutator,
        BallPositionMutator(position_function=SafePositionWrapper(center_ball_position)), # Standard ball spawn height
        CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(blue_right_kickoff_position), orientation_function=get_face_opp_goal_orientation),
        CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(blue_left_kickoff_position), orientation_function=get_face_opp_goal_orientation),
        CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(orange_right_kickoff_position), orientation_function=get_face_own_goal_orientation),
        CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(orange_left_kickoff_position), orientation_function=get_face_own_goal_orientation),
        CarBoostMutator(boost_amount=33) # Start with some boost
    )

    # Offensive scenario spawn
    offensive_spawn = MutatorSequence(
        fixed_2v2_mutator,
        BallPositionMutator(position_function=SafePositionWrapper(offensive_ball_position)), # Ball in opponent half
        CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(offensive_blue_car_position), orientation_function=get_face_opp_goal_orientation),
        CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(offensive_blue_car_position), orientation_function=get_face_opp_goal_orientation),
        # Opponents further back
        CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(defensive_orange_car_position), orientation_function=get_face_own_goal_orientation),
        CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(defensive_orange_car_position), orientation_function=get_face_own_goal_orientation),
    )

    # Defensive scenario spawn
    defensive_spawn = MutatorSequence(
        fixed_2v2_mutator,
        BallPositionMutator(position_function=SafePositionWrapper(defensive_ball_position)), # Ball in own half
        BallVelocityMutator(velocity_function=defensive_ball_velocity), # Ball moving towards own goal
        CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(defensive_blue_car_position), orientation_function=get_face_own_goal_orientation),
        CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(defensive_blue_car_position), orientation_function=get_face_own_goal_orientation),
        # Opponents further up
        CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(offensive_orange_car_position), orientation_function=get_face_opp_goal_orientation),
        CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(offensive_orange_car_position), orientation_function=get_face_opp_goal_orientation),
    )

    # Aerial scenario spawn
    aerial_spawn = MutatorSequence(
        fixed_2v2_mutator,
        BallPositionMutator(position_function=SafePositionWrapper(aerial_ball_position)), # Ball in air
        BallVelocityMutator(velocity_function=aerial_ball_velocity), # Slow random velocity
        # Cars start grounded
        CarPositionMutator(car_id="blue-0", position_function=SafePositionWrapper(random_blue_car_position), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="blue-1", position_function=SafePositionWrapper(random_blue_car_position), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="orange-0", position_function=SafePositionWrapper(random_orange_car_position), orientation_function=get_random_yaw_orientation),
        CarPositionMutator(car_id="orange-1", position_function=SafePositionWrapper(random_orange_car_position), orientation_function=get_random_yaw_orientation),
    )

    # --- Stage Definitions ---

    # STAGE 1: Touch the Ball and Learn Basic Control
    # Goal: Encourage the agent to make contact with the ball and move toward it effectively.
    # Reward: Combined reward focused on ball touches and velocity toward the ball.
    touch_reward_fn = create_phase1_reward()
    stage_touch = CurriculumStage(
        name="2v2 Basic Ball Control",
        state_mutator=simple_touch_spawn, # Simple spawn, or kickoff_mutator
        reward_function=touch_reward_fn,
        termination_condition=goal_condition, # End episode on touch
        truncation_condition=timeout_touch_stage, # Short episodes
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5, # Success = positive reward from touching
            min_avg_reward=2.0,   # Agent should be getting significant touch rewards
            min_episodes=150,     # More episodes to ensure learning this basic skill
            max_std_dev=15.0,     # Higher variance expected initially
            required_consecutive_successes=5 # Ensure consistent touches
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor,       # Use initial LR
            "lr_critic": lr_critic,
            "entropy_coef": 0.02      # Higher entropy for exploration
        }
    )

    # STAGE 2: Score Goals and Offensive Play
    # Goal: Teach the agent to score goals and manage boost.
    # Reward: Combined reward focused on goal scoring and ball positioning.
    score_reward_fn = create_phase2_reward()
    stage_score = CurriculumStage(
        name="2v2 Goal Scoring",
        state_mutator=offensive_spawn, # Spawn in offensive positions
        reward_function=score_reward_fn,
        termination_condition=goal_condition,
        truncation_condition=timeout_score_stage, # Medium length episodes
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.2, # Success = scoring a goal (can be challenging)
            min_avg_reward=3.0,   # Higher reward due to goal focus
            min_episodes=300,
            max_std_dev=10.0,
            required_consecutive_successes=3
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.7, # Slightly reduce LR
            "lr_critic": lr_critic * 0.7,
            "entropy_coef": 0.01
        }
    )

    # STAGE 3: Full Game with Advanced Mechanics
    # Goal: Play a full game with aerial skills, demos, and team coordination.
    # Reward: Combined reward with full feature set.
    full_game_reward_fn = create_phase3_reward()
    stage_full_game = CurriculumStage(
        name="2v2 Advanced Play",
        state_mutator=MutatorSequence(fixed_2v2_mutator, kickoff_mutator),
        reward_function=full_game_reward_fn,
        termination_condition=goal_condition,
        truncation_condition=timeout_full_game_stage, # Full match length
        progression_requirements=ProgressionRequirements( # Final stage, less critical for auto-progression
            min_success_rate=0.4, # Proxy for win rate / effectiveness
            min_avg_reward=0.5,   # Expect balanced reward from full SKG
            min_episodes=500,     # Train longer on the full task
            max_std_dev=5.0,
            required_consecutive_successes=2 # Less strict consecutive successes
        ),
        hyperparameter_adjustments={
            "lr_actor": lr_actor * 0.5, # Further reduce LR for fine-tuning
            "lr_critic": lr_critic * 0.5,
            "entropy_coef": 0.005      # Lower entropy
        }
    )

    # --- Assemble Curriculum ---
    stages = [
        stage_touch,
        stage_score,
        stage_full_game
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
