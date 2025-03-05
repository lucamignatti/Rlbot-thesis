from typing import List, Dict, Any, Tuple, Union, Optional
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np


class NormalizedReward(RewardFunction[AgentID, GameState, float]):
    """Base class for all normalized reward functions that return values in [0,1] range."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_reward_range(self) -> Tuple[float, float]:
        """Returns the theoretical min and max values this reward can return."""
        return 0.0, 1.0


class BallProximityReward(NormalizedReward):
    """
    Rewards players for being close to the ball, with intelligent scaling
    based on how close other players are.
    """

    def __init__(self, max_distance: float = 7000, dispersion: float = 1.0, density: float = 1.0):
        """
        Args:
            max_distance: Distance (in unreal units) at which the reward becomes zero
            dispersion: Controls reward spread - higher values extend rewards further
            density: Controls reward curve shape - higher values make rewards more concentrated
        """
        self.max_distance = max_distance
        self.dispersion = dispersion
        self.density = density

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car_pos = state.cars[agent].physics.position
        ball_pos = state.ball.position

        # Calculate distance between car and ball
        distance = np.linalg.norm(np.array(car_pos) - np.array(ball_pos))

        # Apply parameterized distance formula from Lucy-SKG paper
        normalized_reward = float(np.exp(-0.5 * (distance / (self.max_distance * self.dispersion)) ** (1/self.density)))

        return normalized_reward


class BallToGoalDistanceReward(NormalizedReward):
    """
    Rewards moving the ball closer to the opponent's goal while keeping it
    away from your own goal.
    """

    def __init__(self, offensive_dispersion: float = 0.6, defensive_dispersion: float = 0.4,
                 offensive_density: float = 1.0, defensive_density: float = 1.0):
        """
        Args:
            offensive_dispersion: Dispersion factor for offensive movement (toward opponent goal)
            defensive_dispersion: Dispersion factor for defensive movement (away from own goal)
            offensive_density: Density factor for offensive movement
            defensive_density: Density factor for defensive movement
        """
        # Field length from one goal to the other
        self.field_length = abs(common_values.BLUE_GOAL_CENTER[0] - common_values.ORANGE_GOAL_CENTER[0])
        self.goal_depth = 880  # Standard goal depth in Rocket League (depth into the wall)

        # Lucy-SKG parameters
        self.offensive_dispersion = offensive_dispersion
        self.defensive_dispersion = defensive_dispersion
        self.offensive_density = offensive_density
        self.defensive_density = defensive_density

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        ball_pos = np.array(state.ball.position)

        # Determine team
        try:
            agent_id = int(agent) if isinstance(agent, str) else agent
            is_blue_team = agent_id % 2 == 0
        except (ValueError, TypeError):
            # If we can't determine team from agent ID, use a fallback
            is_blue_team = state.cars[agent].team_num == 0

        if is_blue_team:
            # Blue team wants ball close to orange goal
            own_goal_pos = np.array(common_values.BLUE_GOAL_CENTER)
            opponent_goal_pos = np.array(common_values.ORANGE_GOAL_CENTER)
        else:
            # Orange team wants ball close to blue goal
            own_goal_pos = np.array(common_values.ORANGE_GOAL_CENTER)
            opponent_goal_pos = np.array(common_values.BLUE_GOAL_CENTER)

        # Calculate ball distances to both goals (adjusted for goal depth)
        ball_to_own_goal = np.linalg.norm(ball_pos - own_goal_pos) - self.goal_depth
        ball_to_opponent_goal = np.linalg.norm(ball_pos - opponent_goal_pos) - self.goal_depth

        # Normalize to maximum field distance (2 * field length)
        normalizing_constant = 6000  # Approx. distance between goals in RL

        # Use Lucy-SKG's asymmetric dispersion/density - with different parameters for offense vs defense
        # Offensive component (ball close to opponent goal)
        offensive_component = np.exp(-0.5 * (ball_to_opponent_goal /
                                           (normalizing_constant * self.offensive_dispersion)) **
                                   (1/self.offensive_density))

        # Defensive component (ball far from own goal) - we take the inverse of proximity
        defensive_component = 1 - np.exp(-0.5 * (ball_to_own_goal /
                                               (normalizing_constant * self.defensive_dispersion)) **
                                       (1/self.defensive_density))

        # Combine components (balanced between offense and defense)
        # We use 2.0 for offensive weight and 1.0 for defensive weight as in the paper
        normalized_reward = (2.0 * offensive_component + 1.0 * defensive_component) / 3.0

        return float(normalized_reward)


class BallVelocityToGoalReward(NormalizedReward):
    """
    Rewards when the ball's velocity is directed toward the opponent's goal.
    Encourages strategic hits that create scoring opportunities.
    """

    def __init__(self, max_ball_speed: float = 6000):
        """
        Args:
            max_ball_speed: Maximum ball speed in unreal units to normalize against
        """
        self.max_ball_speed = max_ball_speed

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        ball_vel = np.array(state.ball.linear_velocity)

        # If ball is not moving, return neutral reward
        ball_speed = np.linalg.norm(ball_vel)
        if ball_speed < 1:  # Small threshold to avoid division by zero
            return 0.5

        # Determine team
        try:
            agent_id = int(agent) if isinstance(agent, str) else agent
            is_blue_team = agent_id % 2 == 0
        except (ValueError, TypeError):
            # Fallback to car team info
            is_blue_team = state.cars[agent].team_num == 0

        if is_blue_team:
            # Blue team wants ball moving toward orange goal (positive X)
            goal_direction = np.array([1, 0, 0])
        else:
            # Orange team wants ball moving toward blue goal (negative X)
            goal_direction = np.array([-1, 0, 0])

        # Normalize ball velocity
        ball_vel_normalized = ball_vel / ball_speed

        # Calculate dot product to find component of velocity in goal direction
        direction_alignment = np.dot(ball_vel_normalized, goal_direction)

        # Scale by speed as a fraction of max speed, capped at 1.0
        speed_factor = min(ball_speed / self.max_ball_speed, 1.0)

        # Combine direction and speed:
        # -1 to 1 for direction, scaled by speed, then normalized to 0-1 range
        normalized_reward = (direction_alignment * speed_factor + 1) / 2

        return float(normalized_reward)


class TouchBallReward(NormalizedReward):
    """Simple reward for touching the ball"""

    def __init__(self):
        self.last_touch_counts = {}  # Track ball touches count for each agent

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        # Initialize with current touch counts
        self.last_touch_counts = {}
        for agent in agents:
            # Make sure the attribute exists before accessing it
            if hasattr(initial_state.cars[agent], 'ball_touches'):
                self.last_touch_counts[agent] = initial_state.cars[agent].ball_touches
            else:
                # If the attribute doesn't exist, start with 0
                self.last_touch_counts[agent] = 0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}

        for agent in agents:
            # Make sure the attribute exists before accessing it
            if hasattr(state.cars[agent], 'ball_touches'):
                current_touches = state.cars[agent].ball_touches
            else:
                # If the attribute doesn't exist, assume no touches
                current_touches = 0

            last_touches = self.last_touch_counts.get(agent, 0)

            # Reward only for new touches
            rewards[agent] = 1.0 if current_touches > last_touches else 0.0

            # Update last touches count
            self.last_touch_counts[agent] = current_touches

        return rewards


class TouchBallToGoalAccelerationReward(NormalizedReward):
    """
    Rewards touching the ball in a way that accelerates it toward the opponent's goal.
    This encourages strategic hits rather than just any contact.
    """

    def __init__(self, max_acceleration: float = 4000):
        """
        Args:
            max_acceleration: Maximum expected ball acceleration when hit
        """
        self.max_acceleration = max_acceleration
        self.last_vel = None  # Ball velocity from previous step
        self.last_touch_counts = {}  # Track ball touches count for each agent

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_vel = np.array(initial_state.ball.linear_velocity)

        # Initialize touch counts
        self.last_touch_counts = {}
        for agent in agents:
            # Make sure the attribute exists before accessing it
            if hasattr(initial_state.cars[agent], 'ball_touches'):
                self.last_touch_counts[agent] = initial_state.cars[agent].ball_touches
            else:
                # If the attribute doesn't exist, start with 0
                self.last_touch_counts[agent] = 0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        current_vel = np.array(state.ball.linear_velocity)

        # If last_vel wasn't initialized properly
        if self.last_vel is None:
            self.last_vel = current_vel
            return {agent: 0.0 for agent in agents}

        # Calculate acceleration vector
        acceleration = current_vel - self.last_vel
        acceleration_magnitude = np.linalg.norm(acceleration)

        rewards = {}

        for agent in agents:
            # Make sure the attribute exists before accessing it
            if hasattr(state.cars[agent], 'ball_touches'):
                current_touches = state.cars[agent].ball_touches
            else:
                # If the attribute doesn't exist, assume no touches
                current_touches = 0

            last_touches = self.last_touch_counts.get(agent, 0)

            if current_touches > last_touches and acceleration_magnitude > 100:  # New touch with significant acceleration
                # Determine team
                try:
                    agent_id = int(agent) if isinstance(agent, str) else agent
                    is_blue_team = agent_id % 2 == 0
                except (ValueError, TypeError):
                    # Fallback to car team info
                    is_blue_team = state.cars[agent].team_num == 0

                if is_blue_team:
                    # Blue team wants ball moving toward orange goal (positive X)
                    goal_direction = np.array([1, 0, 0])
                else:
                    # Orange team wants ball moving toward blue goal (negative X)
                    goal_direction = np.array([-1, 0, 0])

                # Normalize acceleration vector
                if acceleration_magnitude > 0:
                    acceleration_norm = acceleration / acceleration_magnitude

                    # Calculate direction component toward goal
                    direction_component = np.dot(acceleration_norm, goal_direction)

                    # Calculate magnitude component (capped at max_acceleration)
                    magnitude_component = min(acceleration_magnitude / self.max_acceleration, 1.0)

                    # Combine direction and magnitude as in Lucy-SKG paper
                    # Map from [-1, 1] to [0, 1] with stronger penalties for bad hits
                    reward = (direction_component * magnitude_component + 1) / 2
                else:
                    reward = 0.5  # Neutral if no acceleration

                rewards[agent] = float(reward)
            else:
                rewards[agent] = 0.0  # No reward if no touch

            # Update last touches count
            self.last_touch_counts[agent] = current_touches

        # Update velocity for next time
        self.last_vel = current_vel

        return rewards


class AlignBallToGoalReward(NormalizedReward):
    """
    Rewards positioning that puts the ball between the player and the opponent's goal.
    This encourages tactical positioning for potential shots.
    """

    def __init__(self, dispersion: float = 1.1, density: float = 1.0):
        """
        Args:
            dispersion: Controls how quickly alignment reward falls off with distance
            density: Controls the concentration of the reward distribution
        """
        self.dispersion = dispersion
        self.density = density

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car_pos = np.array(state.cars[agent].physics.position)
        ball_pos = np.array(state.ball.position)

        # Determine team
        try:
            agent_id = int(agent) if isinstance(agent, str) else agent
            is_blue_team = agent_id % 2 == 0
        except (ValueError, TypeError):
            is_blue_team = state.cars[agent].team_num == 0

        if is_blue_team:
            goal_pos = np.array(common_values.ORANGE_GOAL_CENTER)
        else:
            goal_pos = np.array(common_values.BLUE_GOAL_CENTER)

        # Calculate vectors
        car_to_ball = ball_pos - car_pos
        ball_to_goal = goal_pos - ball_pos

        # Calculate distances
        car_to_ball_dist = np.linalg.norm(car_to_ball)
        ball_to_goal_dist = np.linalg.norm(ball_to_goal)

        # Avoid division by zero
        if car_to_ball_dist < 1e-5 or ball_to_goal_dist < 1e-5:
            return 0.0

        # Calculate angle between vectors
        cos_angle = np.dot(car_to_ball, ball_to_goal) / (car_to_ball_dist * ball_to_goal_dist)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert cosine to angle in radians
        angle = np.arccos(cos_angle)

        # Calculate base alignment reward
        # pi/2 radians (90 degrees) is the threshold for 0 reward
        alignment_reward = max(0.0, 1.0 - angle / (np.pi/2))

        # Apply distance weighting using Lucy-SKG parameterized formula
        distance_weight = np.exp(-0.5 * (car_to_ball_dist/(2000 * self.dispersion))**(1/self.density))

        # Combine alignment and distance components
        reward = alignment_reward * distance_weight

        return float(reward)


class PlayerVelocityTowardBallReward(NormalizedReward):
    """
    Rewards moving toward the ball efficiently.
    This encourages active play and ball chase prevention through smart positioning.
    """

    def __init__(self, max_car_speed: float = 2300):
        """
        Args:
            max_car_speed: Maximum car speed in unreal units to normalize against
        """
        self.max_car_speed = max_car_speed

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car_pos = np.array(state.cars[agent].physics.position)
        ball_pos = np.array(state.ball.position)
        car_vel = np.array(state.cars[agent].physics.linear_velocity)

        # Calculate direction vector from car to ball
        car_to_ball = ball_pos - car_pos
        car_to_ball_dist = np.linalg.norm(car_to_ball)

        # If car is very close to the ball, return maximum reward
        if car_to_ball_dist < 100:
            return 1.0

        # Normalize direction vector
        car_to_ball_normalized = car_to_ball / car_to_ball_dist

        # Get car velocity magnitude
        car_speed = np.linalg.norm(car_vel)

        # If car isn't moving, return neutral reward
        if car_speed < 1:
            return 0.5

        # Normalize car velocity
        car_vel_normalized = car_vel / car_speed

        # Calculate dot product to find velocity component toward ball
        direction_alignment = np.dot(car_vel_normalized, car_to_ball_normalized)

        # Scale by speed as a fraction of max speed, capped at 1.0
        speed_factor = min(car_speed / self.max_car_speed, 1.0)

        # Combine direction and speed:
        # -1 to 1 for direction, scaled by speed, then normalized to 0-1 range
        normalized_reward = (direction_alignment * speed_factor + 1) / 2

        return float(normalized_reward)


class KRCReward(RewardFunction[AgentID, GameState, float]):
    """
    Knowledge Representation and Coordination (KRC) Reward.
    Combines multiple reward functions with team coordination.
    
    The KRC reward helps agents learn both individual skills and team coordination
    by blending personal and team-based rewards. This encourages cooperative play
    while still maintaining individual skill development.
    """

    def __init__(self, reward_functions_and_weights: List[Tuple[RewardFunction, float]],
                 team_spirit: float = 0.3):
        """
        Args:
            reward_functions_and_weights: List of (reward_function, weight) tuples
            team_spirit: Factor τ for distributing rewards among teammates (0-1)
        """
        self.reward_functions = [r for r, _ in reward_functions_and_weights]
        self.weights = [w for _, w in reward_functions_and_weights]
        self.team_spirit = team_spirit

        # Validate all rewards are normalized
        for reward_fn in self.reward_functions:
            if not isinstance(reward_fn, NormalizedReward):
                raise ValueError(
                    f"KRCReward requires all reward functions to be instances of NormalizedReward. "
                    f"Got {type(reward_fn)}"
                )

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        for reward_fn in self.reward_functions:
            reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        # Get rewards from all component functions
        component_rewards = []
        for reward_fn, weight in zip(self.reward_functions, self.weights):
            rewards = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            # Apply weights to the rewards
            weighted_rewards = {agent: r * weight for agent, r in rewards.items()}
            component_rewards.append(weighted_rewards)

        # Calculate KRC reward for each agent
        krc_rewards = {}
        for agent in agents:
            # Get this agent's rewards from all components
            agent_rewards = []
            signs = []
            for rewards in component_rewards:
                r = rewards[agent]
                if abs(r) < 1e-10:
                    agent_rewards.append(1e-10)
                else:
                    agent_rewards.append(abs(r))
                signs.append(np.sign(r))

            try:
                # Calculate geometric mean using log-sum-exp for numerical stability
                n = len(agent_rewards)
                log_rewards = [np.log(max(r, 1e-10)) for r in agent_rewards]
                geometric_mean = np.exp(sum(log_rewards) / n)

                # Determine sign based on majority
                sign = -1 if sum(s < 0 for s in signs) > n/2 else 1

                base_reward = float(sign * geometric_mean)

                # Apply team spirit factor
                if self.team_spirit > 0:
                    # Get team rewards
                    team_rewards = []
                    for other_agent in agents:
                        if self._is_same_team(agent, other_agent, state):
                            team_rewards.append(base_reward)

                    # Calculate team average
                    team_avg = sum(team_rewards) / len(team_rewards) if team_rewards else base_reward

                    # Mix individual and team rewards
                    final_reward = (1 - self.team_spirit) * base_reward + self.team_spirit * team_avg
                else:
                    final_reward = base_reward

                krc_rewards[agent] = final_reward

            except (ValueError, OverflowError) as e:
                print(f"Warning: Numerical error in KRC calculation: {e}")
                krc_rewards[agent] = 0.0

        return krc_rewards

    def _is_same_team(self, agent1: AgentID, agent2: AgentID, state: GameState) -> bool:
        """Helper function to determine if two agents are on the same team."""
        try:
            agent1_id = int(agent1) if isinstance(agent1, str) else agent1
            agent2_id = int(agent2) if isinstance(agent2, str) else agent2
            return (agent1_id % 2) == (agent2_id % 2)
        except (ValueError, TypeError):
            return state.cars[agent1].team_num == state.cars[agent2].team_num
