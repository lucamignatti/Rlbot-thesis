import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from rlgym.api import StateMutator, AgentID, RewardFunction, DoneCondition
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
from rlgym.rocket_league.done_conditions import (
    GoalCondition, TimeoutCondition, NoTouchTimeoutCondition
)
from rlgym.rocket_league.reward_functions import (
    CombinedReward, GoalReward, TouchReward
)
from rlgym.rocket_league.state_mutators import (
    MutatorSequence, FixedTeamSizeMutator, KickoffMutator
)
from curriculum import CurriculumManager, CurriculumStage, ProgressionRequirements

# Custom reward class to replace EventReward
class SimpleEventReward(RewardFunction):
    """Simple reward for specific events"""

    def __init__(self, goal=0.0, team_goal=0.0, concede=-0.0, touch=0.0, shot=0.0, save=0.0):
        self.goal_weight = goal
        self.team_goal_weight = team_goal
        self.concede_weight = concede
        self.touch_weight = touch
        self.shot_weight = shot
        self.save_weight = save
        self.last_registered_values = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        """Reset tracked values for all agents"""
        self.last_registered_values = {}
        for agent in agents:
            self.last_registered_values[agent] = {
                "goals": 0,
                "team_goals": 0,
                "touches": 0,
                "shots": 0,
                "saves": 0,
                "last_touch": False
            }

    def get_reward(self, agent: AgentID, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate reward based on tracked events"""
        if agent not in self.last_registered_values:
            # Initialize if this is a new agent
            self.last_registered_values[agent] = {
                "goals": 0,
                "team_goals": 0,
                "touches": 0,
                "shots": 0,
                "saves": 0,
                "last_touch": False
            }

        reward = 0.0

        # Check for ball touches - this is the most reliable event we can track
        ball_touched = False
        if hasattr(state.cars[agent], 'ball_touched'):
            try:
                ball_touched = state.cars[agent].ball_touched
            except AttributeError:
                # Fallback if attribute doesn't exist or is inaccessible
                ball_touched = False

        # Reward for new ball touches
        if ball_touched and not self.last_registered_values[agent].get("last_touch", False):
            reward += self.touch_weight
            self.last_registered_values[agent]["last_touch"] = True
        elif not ball_touched:
            self.last_registered_values[agent]["last_touch"] = False

        return reward

    def get_rewards(self, agents: List[AgentID], state: GameState,
                   is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool],
                   shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            # Pass an empty array as previous_action since we don't use it
            rewards[agent] = self.get_reward(agent, state, np.array([]))
        return rewards

# Custom ball touched condition
class BallTouchedCondition(DoneCondition):
    """Terminates episode when ball is touched with a minimum speed"""

    def __init__(self, min_speed: float = 2000):
        """
        Args:
            min_speed: Minimum speed required for the ball to trigger termination
        """
        self.min_speed = min_speed
        self.last_touch_counts = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        # Reset touch counts
        self.last_touch_counts = {}
        for agent in agents:
            self.last_touch_counts[agent] = 0

    def is_done(self, agents: List[AgentID], state: GameState,
                shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        ball_speed = np.linalg.norm(state.ball.linear_velocity)
        result = {agent: False for agent in agents}

        # Simple implementation that just checks ball speed
        if ball_speed >= self.min_speed:
            for agent in agents:
                result[agent] = True

        return result

# Custom rewards for specific tasks
class KickoffPerformanceReward(RewardFunction):
    """Rewards quick touches on kickoff"""

    def __init__(self, max_time: float = 3.0, touch_reward: float = 10.0, time_penalty_factor: float = 3.0):
        self.max_time = max_time
        self.touch_reward = touch_reward
        self.time_penalty_factor = time_penalty_factor
        self.start_times = {}
        self.last_touch_status = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.start_times = {agent: 0.0 for agent in agents}
        self.last_touch_status = {agent: False for agent in agents}

    def get_reward(self, agent: AgentID, state: GameState, previous_action: np.ndarray) -> float:
        # Check for ball touches
        ball_touched = False
        if hasattr(state.cars[agent], 'ball_touched'):
            try:
                ball_touched = state.cars[agent].ball_touched
            except AttributeError:
                ball_touched = False

        # If ball is newly touched, give a time-based reward
        if ball_touched and not self.last_touch_status.get(agent, False):
            # Calculate time elapsed (using tick count as a proxy if seconds_elapsed isn't available)
            time_elapsed = 0
            if hasattr(state, 'tick_count'):
                # Approximate time based on 120 ticks per second
                time_elapsed = state.tick_count / 120.0

            # Scale reward by time taken (faster is better)
            time_factor = max(0, (self.max_time - time_elapsed) / self.max_time)
            reward = self.touch_reward * time_factor
            self.last_touch_status[agent] = True
            return reward

        # Small penalty for time passing without touching
        if not ball_touched:
            self.last_touch_status[agent] = False
            return -self.time_penalty_factor * 0.008  # Assume ~8ms per tick

        return 0.0

    def get_rewards(self, agents: List[AgentID], state: GameState,
                   is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool],
                   shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            # Pass an empty array as previous_action since we don't use it
            rewards[agent] = self.get_reward(agent, state, np.array([]))
        return rewards

class DefensivePositioningReward(RewardFunction):
    """Rewards positioning between the ball and defensive goal"""

    def __init__(self, team_num: int, min_ball_distance: float = 300):
        self.team_num = team_num
        self.min_ball_distance = min_ball_distance

    def get_reward(self, agent: AgentID, state: GameState, previous_action: np.ndarray) -> float:
        # Check team
        car_team = -1
        try:
            car_team = getattr(state.cars[agent], 'team_num', -1)
        except AttributeError:
            return 0.0

        if car_team != self.team_num:
            return 0.0

        # Get positions
        try:
            ball_pos = state.ball.position
            car_pos = state.cars[agent].position
        except AttributeError:
            return 0.0

        # Get goal position (defensive goal)
        goal_x = -4096 if self.team_num == 0 else 4096
        goal_pos = np.array([goal_x, 0, 0])

        # Calculate distances and vectors
        car_to_ball = ball_pos - car_pos
        car_to_ball_dist = np.linalg.norm(car_to_ball)

        goal_to_ball = ball_pos - goal_pos
        goal_to_ball_dist = np.linalg.norm(goal_to_ball)

        goal_to_car = car_pos - goal_pos
        goal_to_car_dist = np.linalg.norm(goal_to_car)

        # If car is too close to ball, no positioning reward
        if car_to_ball_dist < self.min_ball_distance:
            return 0.0

        # Check if car is between ball and goal
        if goal_to_car_dist < goal_to_ball_dist:
            # Calculate dot product to determine alignment
            if goal_to_ball_dist > 0 and goal_to_car_dist > 0:
                goal_to_ball_dir = goal_to_ball / goal_to_ball_dist
                goal_to_car_dir = goal_to_car / goal_to_car_dist
                alignment = np.dot(goal_to_ball_dir, goal_to_car_dir)

                # Only reward positive alignment
                if alignment > 0:
                    # Scale by distance - better positioning closer to ball but not too close
                    distance_factor = 1.0 - (goal_to_car_dist / max(goal_to_ball_dist, 1.0))
                    return float(alignment * distance_factor)

        return 0.0

    def get_rewards(self, agents: List[AgentID], state: GameState,
                   is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool],
                   shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            # Pass an empty array as previous_action since we don't use it
            rewards[agent] = self.get_reward(agent, state, np.array([]))
        return rewards

class OffensiveClearReward(RewardFunction):
    """Rewards hitting the ball away from defensive goal with high velocity"""

    def __init__(self, team_num: int, min_vel_magnitude: float = 1000):
        self.team_num = team_num
        self.min_vel_magnitude = min_vel_magnitude
        self.last_touch_status = {}
        self.prev_ball_vel = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_touch_status = {agent: False for agent in agents}
        self.prev_ball_vel = {agent: np.zeros(3) for agent in agents}

    def get_reward(self, agent: AgentID, state: GameState, previous_action: np.ndarray) -> float:
        # Check team
        car_team = -1
        try:
            car_team = getattr(state.cars[agent], 'team_num', -1)
        except AttributeError:
            return 0.0

        if car_team != self.team_num:
            return 0.0

        reward = 0.0
        ball_touched = False

        # Check for ball touches
        if hasattr(state.cars[agent], 'ball_touched'):
            try:
                ball_touched = state.cars[agent].ball_touched
            except AttributeError:
                ball_touched = False

        # If ball is newly touched
        if ball_touched and not self.last_touch_status.get(agent, False):
            self.last_touch_status[agent] = True

            # Get ball velocity and check magnitude
            ball_vel = state.ball.linear_velocity
            vel_magnitude = np.linalg.norm(ball_vel)

            if vel_magnitude > self.min_vel_magnitude:
                # Get defensive goal position
                goal_x = -4096 if self.team_num == 0 else 4096
                goal_pos = np.array([goal_x, 0, 0])

                # Check if ball is moving away from goal
                goal_to_ball = state.ball.position - goal_pos
                if np.linalg.norm(goal_to_ball) > 0:
                    goal_to_ball_dir = goal_to_ball / np.linalg.norm(goal_to_ball)
                    ball_vel_dir = ball_vel / vel_magnitude

                    # Measure alignment between ball velocity and away-from-goal direction
                    alignment = np.dot(ball_vel_dir, goal_to_ball_dir)

                    # Reward velocity away from goal
                    if alignment > 0.7:  # Ball moving away from defense
                        reward = float(alignment * (vel_magnitude / 4000.0) * 8.0)

        # Reset touch detection
        elif not ball_touched:
            self.last_touch_status[agent] = False

        # Store current ball velocity for next step
        self.prev_ball_vel[agent] = state.ball.linear_velocity

        return reward

    def get_rewards(self, agents: List[AgentID], state: GameState,
                   is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool],
                   shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            # Pass an empty array as previous_action since we don't use it
            rewards[agent] = self.get_reward(agent, state, np.array([]))
        return rewards

def create_basic_curriculum(debug=False):
    """
    Creates a comprehensive curriculum for training a Rocket League agent with
    progressive skill development from basic shooting to team play.

    Args:
        debug: Whether to print debug information during stage transitions

    Returns:
        CurriculumManager: Configured curriculum manager with all stages
    """
    # Create stages with increasing complexity
    stages = []

    # Stage 1: Basic Shooting
    basic_shooting = CurriculumStage(
        name="Basic Shooting",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallTowardGoalSpawnMutator(offensive_team=0, distance_from_goal=0.7)
        ),
        reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 0.5),
            (SimpleEventReward(goal=5.0, touch=0.2), 1.0)
        ),
        termination_condition=AnyCondition(
            GoalCondition(),
            TimeoutCondition(timeout_seconds=8.0)
        ),
        truncation_condition=TimeoutCondition(timeout_seconds=8.0),
        difficulty_params={
            "ball_distance": (0.7, 0.3),  # Start close, get farther
            "ball_speed": (0, 500)       # Start slow, get faster
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.7,
            min_avg_reward=0.5,
            min_episodes=50,
            max_std_dev=2.0,
            required_consecutive_successes=3
        )
    )
    stages.append(basic_shooting)

    # Stage 2: Shooting with Angles
    angled_shooting = CurriculumStage(
        name="Angled Shooting",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(position_function=lambda: np.array([
                np.random.uniform(-2000, 2000),
                np.random.uniform(-3000, 3000),
                100
            ]))
        ),
        reward_function=CombinedReward(
            (GoalReward(), 8.0),
            (TouchReward(), 0.3),
            (SimpleEventReward(goal=5.0, touch=0.2), 1.0)
        ),
        termination_condition=AnyCondition(
            GoalCondition(),
            TimeoutCondition(timeout_seconds=10.0)
        ),
        truncation_condition=TimeoutCondition(timeout_seconds=10.0),
        difficulty_params={
            "ball_distance": (0.5, 0.8),
            "angle_range": (10, 45)
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.4,
            min_episodes=100,
            max_std_dev=2.5,
            required_consecutive_successes=3
        )
    )
    stages.append(angled_shooting)

    # Stage 3: Aerial Training
    aerial_training = CurriculumStage(
        name="Aerial Training",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(position_function=lambda: np.array([
                np.random.uniform(-1000, 1000),
                np.random.uniform(-1000, 1000),
                np.random.uniform(300, 800)  # Ball in the air
            ])),
            CarBoostMutator(boost_amount=100)  # Full boost
        ),
        reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 1.0),
            (SimpleEventReward(goal=5.0, touch=0.5), 1.0)
        ),
        termination_condition=AnyCondition(
            GoalCondition(),
            TimeoutCondition(timeout_seconds=12.0)
        ),
        truncation_condition=TimeoutCondition(timeout_seconds=12.0),
        difficulty_params={
            "ball_height": (300, 1200),
            "car_boost": (50, 100)
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5,
            min_avg_reward=0.3,
            min_episodes=150,
            max_std_dev=3.0,
            required_consecutive_successes=2
        )
    )
    stages.append(aerial_training)

    # Stage 4: Defense Training
    defense_training = CurriculumStage(
        name="Defense Training",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(position_function=lambda: np.array([
                -3000,  # Near blue goal
                np.random.uniform(-2000, 2000),
                np.random.uniform(100, 300)
            ])),
            BallVelocityMutator(velocity_function=lambda: np.array([
                np.random.uniform(500, 1500),  # Ball coming toward goal
                np.random.uniform(-200, 200),
                np.random.uniform(-100, 100)
            ]))
        ),
        reward_function=CombinedReward(
            (GoalReward(), -10.0),  # Penalty for conceding
            (TouchReward(), 1.5),
            (DefensivePositioningReward(team_num=0, min_ball_distance=200), 1.0),
            (OffensiveClearReward(team_num=0, min_vel_magnitude=800), 1.0)
        ),
        termination_condition=AnyCondition(
            GoalCondition(),
            TimeoutCondition(timeout_seconds=10.0)
        ),
        truncation_condition=TimeoutCondition(timeout_seconds=10.0),
        difficulty_params={
            "ball_speed": (500, 2000),
            "ball_angle": (5, 30)
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.4,
            min_episodes=200,
            max_std_dev=2.5,
            required_consecutive_successes=3
        )
    )
    stages.append(defense_training)

    # Stage 5: Kickoff Training
    kickoff_training = CurriculumStage(
        name="Kickoff Training",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            KickoffMutator()
        ),
        reward_function=CombinedReward(
            (KickoffPerformanceReward(max_time=3.0, touch_reward=5.0), 1.0),
            (GoalReward(), 10.0),
            (TouchReward(), 0.2)
        ),
        termination_condition=AnyCondition(
            BallTouchedCondition(min_speed=1000),
            TimeoutCondition(timeout_seconds=5.0)
        ),
        truncation_condition=TimeoutCondition(timeout_seconds=5.0),
        difficulty_params={},  # Kickoffs are standardized
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.8,
            min_avg_reward=0.6,
            min_episodes=100,
            max_std_dev=2.0,
            required_consecutive_successes=5
        )
    )
    stages.append(kickoff_training)

    # Stage 6: 1v1 Training
    one_v_one_training = CurriculumStage(
        name="1v1 Training",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator()
        ),
        reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (GoalReward(), -10.0),  # Penalty for conceding
            (TouchReward(), 0.1),
            (SimpleEventReward(goal=5.0, concede=-5.0, touch=0.1), 1.0)
        ),
        termination_condition=AnyCondition(
            GoalCondition(),
            TimeoutCondition(timeout_seconds=60.0)  # Longer matches
        ),
        truncation_condition=TimeoutCondition(timeout_seconds=60.0),
        difficulty_params={
            "opponent_difficulty": (0.1, 0.7)  # Increase opponent difficulty
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.3,
            min_episodes=50,
            max_std_dev=4.0,
            required_consecutive_successes=2
        )
    )
    stages.append(one_v_one_training)

    # Stage 7: Team Play
    team_play = CurriculumStage(
        name="Team Play",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            KickoffMutator()
        ),
        reward_function=CombinedReward(
            (GoalReward(), 5.0),
            (GoalReward(), -5.0),
            (TouchReward(), 0.1),
            (SimpleEventReward(goal=3.0, team_goal=1.0, concede=-3.0, touch=0.1), 1.0)
        ),
        termination_condition=AnyCondition(
            GoalCondition(),
            TimeoutCondition(timeout_seconds=120.0)  # Full matches
        ),
        truncation_condition=TimeoutCondition(timeout_seconds=120.0),
        difficulty_params={
            "opponent_difficulty": (0.3, 0.8)
        },
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.52,
            min_avg_reward=0.2,
            min_episodes=30,
            max_std_dev=5.0,
            required_consecutive_successes=1
        )
    )
    stages.append(team_play)

    # Create and return curriculum manager
    curriculum_manager = CurriculumManager(
        stages=stages,
        progress_thresholds={
            "success_rate": 0.7,
            "avg_reward": 0.6
        },
        max_rehearsal_stages=2,
        rehearsal_decay_factor=0.6,
        evaluation_window=50,
        debug=debug,
        use_wandb=True  # Enable wandb logging
    )

    if debug:
        print(f"Curriculum created with {len(stages)} stages:")
        for i, stage in enumerate(stages):
            print(f"  {i+1}. {stage.name}")

    return curriculum_manager

# Additional utility functions for the curriculum

def random_ball_position(min_height=100, max_height=200):
    """Generate a random ball position on the field"""
    return np.array([
        np.random.uniform(-3000, 3000),
        np.random.uniform(-4000, 4000),
        np.random.uniform(min_height, max_height)
    ])

def random_ball_velocity(max_speed=1000):
    """Generate a random ball velocity vector"""
    direction = np.random.rand(3) * 2 - 1  # Random direction
    direction = direction / np.linalg.norm(direction)  # Normalize
    speed = np.random.uniform(0, max_speed)
    return direction * speed

def random_car_position(team_num=0):
    """Generate a random car position appropriate for the given team"""
    if team_num == 0:  # Blue team
        x_range = (-4000, -2000)
    else:  # Orange team
        x_range = (2000, 4000)

    return np.array([
        np.random.uniform(x_range[0], x_range[1]),
        np.random.uniform(-3000, 3000),
        17  # Car height
    ])

def create_custom_mutator_sequence(
    blue_size=1,
    orange_size=0,
    ball_pos_func=None,
    ball_vel_func=None,
    blue_car_pos_func=None,
    orange_car_pos_func=None,
    boost_amount=100
):
    """Create a custom mutator sequence based on given parameters"""
    # Create an initial fixed team size mutator
    base_mutator = FixedTeamSizeMutator(blue_size=blue_size, orange_size=orange_size)
    mutators = [base_mutator]

    # Add each additional mutator individually, not as a list
    if ball_pos_func:
        mutators.append(BallPositionMutator(position_function=ball_pos_func))

    if ball_vel_func:
        mutators.append(BallVelocityMutator(velocity_function=ball_vel_func))

    if boost_amount is not None:
        mutators.append(CarBoostMutator(boost_amount=boost_amount))

    # Add car position mutators for all players if functions are provided
    if blue_car_pos_func and blue_size > 0:
        for i in range(blue_size):
            mutators.append(CarPositionMutator(car_id=i, position_function=blue_car_pos_func))

    if orange_car_pos_func and orange_size > 0:
        for i in range(orange_size):
            # Orange team car IDs start after all blue team car IDs
            car_id = blue_size + i
            mutators.append(CarPositionMutator(car_id=car_id, position_function=orange_car_pos_func))

    # Return a MutatorSequence with unpacked individual mutators
    return MutatorSequence(*mutators)
