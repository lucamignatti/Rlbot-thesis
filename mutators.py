from typing import List, Dict, Any, Tuple, Union, Optional, Callable
from rlgym.api import StateMutator, AgentID, RewardFunction, DoneCondition
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np

class BallTowardGoalSpawnMutator(StateMutator):
    """Spawns the ball in a position that lines up with the goal"""
    def __init__(self, offensive_team=0, distance_from_goal=0.7, random_offset=0.2):
        self.offensive_team = offensive_team
        self.distance_from_goal = distance_from_goal
        self.random_offset = random_offset

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        # Set goal direction based on offensive team
        goal_direction = 1 if self.offensive_team == 0 else -1

        # Calculate ball position
        x_pos = goal_direction * (1 - self.distance_from_goal) * 4096  # Field is roughly -4096 to 4096 in x
        y_pos = (np.random.random() * 2 - 1) * self.random_offset * 5120  # Field is roughly -5120 to 5120 in y

        # Set ball position (slightly elevated to prevent ground friction initially)
        state.ball.position = np.array([x_pos, y_pos, 100])

        # Give ball slight velocity toward goal
        velocity_magnitude = np.random.random() * 500  # Random initial speed
        state.ball.linear_velocity = np.array([goal_direction * velocity_magnitude, 0, 0])

class BallPositionMutator(StateMutator):
    """Sets the ball's position using a callback function"""
    def __init__(self, position_function: Optional[Callable[[], np.ndarray]] = None):
        """
        Args:
            position_function: Function that returns a 3D position array
        """
        self.position_function = position_function or (lambda: np.array([0, 0, 100]))

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        position = self.position_function()
        state.ball.position = position
        state.ball.linear_velocity = np.zeros(3)
        state.ball.angular_velocity = np.zeros(3)

class CarPositionMutator(StateMutator):
    """Sets a car's position using a callback function"""
    def __init__(self, car_id: int, position_function: Optional[Callable[[], np.ndarray]] = None):
        """
        Args:
            car_id: ID of the car to reposition
            position_function: Function that returns a 3D position array
        """
        self.car_id = car_id
        self.position_function = position_function or (lambda: np.array([0, 0, 17]))

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        if str(self.car_id) not in state.cars:
            return  # Car not found

        position = self.position_function()
        try:
            # Try to set the position directly
            state.cars[str(self.car_id)].position = position
        except (AttributeError, TypeError):
            # If direct setting fails, we might need to access physics object
            if hasattr(state.cars[str(self.car_id)], 'physics'):
                state.cars[str(self.car_id)].physics.position = position

        # Reset rotation - handle both direct and physics attributes
        try:
            state.cars[str(self.car_id)].euler_angles = np.array([0, 0, 0])
        except (AttributeError, TypeError):
            if hasattr(state.cars[str(self.car_id)], 'physics'):
                state.cars[str(self.car_id)].physics.euler_angles = np.array([0, 0, 0])

class CarVelocityMutator(StateMutator):
    """Sets a car's velocity using a callback function"""
    def __init__(self, car_id: int, velocity_function: Optional[Callable[[], np.ndarray]] = None):
        """
        Args:
            car_id: ID of the car to modify
            velocity_function: Function that returns a 3D velocity array
        """
        self.car_id = car_id
        self.velocity_function = velocity_function or (lambda: np.zeros(3))

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        if str(self.car_id) not in state.cars:
            return  # Car not found

        velocity = self.velocity_function()
        try:
            # Try to set the velocity directly
            state.cars[str(self.car_id)].linear_velocity = velocity
            state.cars[str(self.car_id)].angular_velocity = np.zeros(3)
        except (AttributeError, TypeError):
            # If direct setting fails, we might need to access physics object
            if hasattr(state.cars[str(self.car_id)], 'physics'):
                state.cars[str(self.car_id)].physics.linear_velocity = velocity
                state.cars[str(self.car_id)].physics.angular_velocity = np.zeros(3)

class BallVelocityMutator(StateMutator):
    """Sets the ball's velocity using a callback function"""
    def __init__(self, velocity_function: Optional[Callable[[], np.ndarray]] = None):
        """
        Args:
            velocity_function: Function that returns a 3D velocity array
        """
        self.velocity_function = velocity_function or (lambda: np.zeros(3))

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        velocity = self.velocity_function()
        state.ball.linear_velocity = velocity
        state.ball.angular_velocity = np.zeros(3)

class CarBoostMutator(StateMutator):
    """Sets boost amount for all cars or for specific cars"""
    def __init__(self, boost_amount: Union[float, Callable[[], float]] = 100, car_ids: Optional[List[int]] = None):
        """
        Args:
            boost_amount: Amount of boost (0-100) or a function that returns boost amount
            car_ids: List of car IDs to modify boost for. If None, affects all cars.
        """
        self.boost_amount = boost_amount
        self.car_ids = car_ids

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        # Determine boost amount (fixed value or from callback)
        if callable(self.boost_amount):
            amount = self.boost_amount()
        else:
            amount = self.boost_amount

        # Clamp to valid range
        amount = max(0, min(100, amount))

        # Apply to either specified cars or all cars
        if self.car_ids is not None:
            for car_id in self.car_ids:
                if str(car_id) in state.cars:
                    try:
                        state.cars[str(car_id)].boost_amount = amount
                    except (AttributeError, TypeError):
                        # Try alternative attribute names
                        if hasattr(state.cars[str(car_id)], 'boost'):
                            state.cars[str(car_id)].boost = amount
        else:
            for car_id, car in state.cars.items():
                try:
                    car.boost_amount = amount
                except (AttributeError, TypeError):
                    # Try alternative attribute names
                    if hasattr(car, 'boost'):
                        car.boost = amount

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

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        # Check ball speed
        ball_speed = np.linalg.norm(state.ball.linear_velocity)

        # Create result dictionary
        result = {agent: False for agent in agents}

        # If ball speed exceeds threshold, terminate for all agents
        if ball_speed >= self.min_speed:
            for agent in agents:
                result[agent] = True

        return result

class KickoffPerformanceReward(RewardFunction):
    """Rewards quick touches on kickoff"""
    def __init__(self, max_time: float = 3.0, touch_reward: float = 10.0, time_penalty_factor: float = 3.0):
        """
        Args:
            max_time: Maximum time in seconds for a "good" kickoff
            touch_reward: Maximum reward for touching the ball immediately
            time_penalty_factor: Factor to penalize time taken
        """
        self.max_time = max_time
        self.touch_reward = touch_reward
        self.time_penalty_factor = time_penalty_factor
        self.last_touch_counts = {}
        self.kickoff_started = {}
        self.kickoff_times = {}
        self.last_touch_status = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_touch_counts = {}
        self.kickoff_started = {}
        self.kickoff_times = {}
        self.last_touch_status = {}

        # Initialize for each agent
        for agent in agents:
            self.last_touch_counts[agent] = 0
            self.kickoff_started[agent] = True
            self.last_touch_status[agent] = False

            # Try to get the current game time
            if hasattr(initial_state, 'game_info') and hasattr(initial_state.game_info, 'seconds_elapsed'):
                self.kickoff_times[agent] = initial_state.game_info.seconds_elapsed
            elif hasattr(initial_state, 'tick_count'):
                # Approximate time based on 120 ticks per second
                self.kickoff_times[agent] = initial_state.tick_count / 120.0
            else:
                self.kickoff_times[agent] = 0

    def get_reward(self, agent: AgentID, state: GameState, previous_action: np.ndarray) -> float:
        if agent not in self.kickoff_started or not self.kickoff_started[agent]:
            return 0.0

        # Check for ball touches
        ball_touched = False

        # Try to detect ball touches
        try:
            if hasattr(state.cars[agent], 'ball_touched'):
                ball_touched = state.cars[agent].ball_touched
            elif hasattr(state, 'ball_touched') and isinstance(state.ball_touched, dict):
                ball_touched = state.ball_touched.get(agent, False)
        except (AttributeError, KeyError):
            # If we can't detect touches, use a basic heuristic
            # Check if car is close to ball
            try:
                car_pos = state.cars[agent].position
                ball_pos = state.ball.position
                distance = np.linalg.norm(car_pos - ball_pos)
                # Rough estimate: car + ball radius ≈ 200
                ball_touched = distance < 200
            except (AttributeError, TypeError):
                ball_touched = False

        # If ball is newly touched
        if ball_touched and not self.last_touch_status.get(agent, False):
            # Calculate time since kickoff began
            time_taken = 0
            if hasattr(state, 'game_info') and hasattr(state.game_info, 'seconds_elapsed'):
                time_taken = state.game_info.seconds_elapsed - self.kickoff_times.get(agent, 0)
            elif hasattr(state, 'tick_count'):
                # Approximate time based on 120 ticks per second
                time_taken = state.tick_count / 120.0 - self.kickoff_times.get(agent, 0)

            # Calculate reward based on time (quicker = better)
            time_factor = max(0, (self.max_time - time_taken) / self.max_time)
            reward = self.touch_reward * time_factor

            # End kickoff state for this agent
            self.kickoff_started[agent] = False
            self.last_touch_status[agent] = True
            return reward

        # Update touch status
        self.last_touch_status[agent] = ball_touched

        # Small penalty for time passing
        return -self.time_penalty_factor * 0.008  # Assuming ~8ms per tick

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
        """
        Args:
            team_num: Team number (0 for blue, 1 for orange)
            min_ball_distance: Minimum distance to ball required to get positioning reward
        """
        self.team_num = team_num
        self.min_ball_distance = min_ball_distance

    def get_reward(self, agent: AgentID, state: GameState, previous_action: np.ndarray) -> float:
        # Check if agent is on the right team
        car_team = -1
        try:
            if hasattr(state.cars[agent], 'team_num'):
                car_team = state.cars[agent].team_num
            # Alternative attribute names
            elif hasattr(state.cars[agent], 'team'):
                car_team = state.cars[agent].team
        except (AttributeError, KeyError):
            return 0.0

        if car_team != self.team_num:
            return 0.0

        # Get positions
        try:
            ball_pos = np.array(state.ball.position)
            car_pos = np.array(state.cars[agent].position)
        except (AttributeError, TypeError):
            return 0.0

        # Get goal position (defensive goal)
        goal_x = -4096 if self.team_num == 0 else 4096
        goal_pos = np.array([goal_x, 0, 0])

        # Vector from goal to ball
        goal_to_ball = ball_pos - goal_pos
        goal_to_ball_dist = np.linalg.norm(goal_to_ball)

        # Vector from goal to car
        goal_to_car = car_pos - goal_pos
        goal_to_car_dist = np.linalg.norm(goal_to_car)

        # Calculate distance between car and ball
        car_to_ball_dist = np.linalg.norm(car_pos - ball_pos)

        # If car is too close to ball, reduce reward (we want positioning, not ball chasing)
        if car_to_ball_dist < self.min_ball_distance:
            return 0.0

        # Reward is higher when car is between ball and goal
        # And distance from goal to car is less than distance from goal to ball
        if goal_to_car_dist < goal_to_ball_dist:
            # Calculate angle between vectors
            if goal_to_car_dist > 0 and goal_to_ball_dist > 0:
                goal_to_ball_dir = goal_to_ball / goal_to_ball_dist
                goal_to_car_dir = goal_to_car / goal_to_car_dist
                cos_angle = np.dot(goal_to_ball_dir, goal_to_car_dir)
                angle_reward = max(0, cos_angle)  # Only reward when in right direction

                # Scale by distance - better positioning closer to ball but not too close
                distance_factor = 1.0 - (goal_to_car_dist / max(goal_to_ball_dist, 1.0))
                return float(angle_reward * distance_factor)

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
        """
        Args:
            team_num: Team number (0 for blue, 1 for orange)
            min_vel_magnitude: Minimum velocity magnitude required for rewards
        """
        self.team_num = team_num
        self.min_vel_magnitude = min_vel_magnitude
        self.prev_ball_pos = {}
        self.ball_touched = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ball_pos = {}
        self.ball_touched = {}
        for agent in agents:
            self.prev_ball_pos[agent] = initial_state.ball.position
            self.ball_touched[agent] = False

    def get_reward(self, agent: AgentID, state: GameState, previous_action: np.ndarray) -> float:
        # Check if agent is on the right team
        car_team = -1
        try:
            if hasattr(state.cars[agent], 'team_num'):
                car_team = state.cars[agent].team_num
            # Alternative attribute names
            elif hasattr(state.cars[agent], 'team'):
                car_team = state.cars[agent].team
        except (AttributeError, KeyError):
            return 0.0

        if car_team != self.team_num:
            return 0.0

        # Initialize if this is a new agent
        if agent not in self.prev_ball_pos:
            self.prev_ball_pos[agent] = state.ball.position
            self.ball_touched[agent] = False

        reward = 0.0

        # Check if agent just touched the ball
        ball_touched = False

        # Try to detect ball touches
        try:
            if hasattr(state.cars[agent], 'ball_touched'):
                ball_touched = state.cars[agent].ball_touched
            elif hasattr(state, 'ball_touched') and isinstance(state.ball_touched, dict):
                ball_touched = state.ball_touched.get(agent, False)
        except (AttributeError, KeyError):
            # If we can't detect touches, estimate based on proximity and velocity changes
            try:
                car_pos = state.cars[agent].position
                ball_pos = state.ball.position
                prev_ball_pos = self.prev_ball_pos.get(agent, ball_pos)

                # Calculate ball velocity change
                ball_velocity = state.ball.linear_velocity
                ball_displacement = ball_pos - prev_ball_pos

                # Calculate car-ball distance
                car_ball_dist = np.linalg.norm(car_pos - ball_pos)

                # Use heuristic: car is close to ball and ball changed velocity significantly
                if car_ball_dist < 200 and np.linalg.norm(ball_displacement) > 100:
                    ball_touched = True
            except (AttributeError, TypeError):
                ball_touched = False

        if ball_touched and not self.ball_touched.get(agent, False):
            self.ball_touched[agent] = True

            # Get defensive goal position
            goal_x = -4096 if self.team_num == 0 else 4096
            goal_pos = np.array([goal_x, 0, 0])

            # Calculate ball velocity direction relative to goal
            ball_vel = np.array(state.ball.linear_velocity)
            vel_magnitude = np.linalg.norm(ball_vel)

            if vel_magnitude > self.min_vel_magnitude:
                # Determine if ball is moving away from defense
                goal_to_ball = np.array(state.ball.position) - goal_pos
                goal_to_ball_dist = np.linalg.norm(goal_to_ball)

                if goal_to_ball_dist > 0:
                    goal_to_ball_dir = goal_to_ball / goal_to_ball_dist
                    ball_vel_dir = ball_vel / vel_magnitude

                    # Dot product to determine how aligned velocity is with away direction
                    vel_alignment = np.dot(ball_vel_dir, goal_to_ball_dir)

                    if vel_alignment > 0.7:  # Ball moving away from defense
                        # Reward based on velocity magnitude and alignment
                        reward = float(vel_alignment * (vel_magnitude / 4000) * 8)  # Scale reward

        # Reset touch detection if no longer touching
        if not ball_touched:
            self.ball_touched[agent] = False

        # Update previous ball position
        self.prev_ball_pos[agent] = state.ball.position

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
