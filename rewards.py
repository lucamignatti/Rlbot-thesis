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


def clamp_reward(reward: float) -> float:
    """Ensure reward stays within [-1, 1] range"""
    return max(-1.0, min(1.0, reward))


# Base class that implements common RLGym API methods for all reward classes
class BaseRewardFunction(RewardFunction):
    """Base class that implements common methods required by RLGym API"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        """Reset any state in the reward function"""
        pass
        
    def get_reward(self, agent: AgentID, state: GameState, previous_action: np.ndarray) -> float:
        """
        Default implementation that calls calculate, which all subclasses must implement.
        This adapts our calculate method to the RLGym API.
        """
        return self.calculate(agent, state, None)
        
    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        """
        Calculate rewards for all agents - required by RLGym API.
        Default implementation calls get_reward for each agent.
        """
        return {agent: self.get_reward(agent, state, np.array([])) for agent in agents}
        
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        """All subclasses must implement this method"""
        raise NotImplementedError
        
    def _get_car_data_from_state(self, agent_id, state: GameState):
        """Helper method to get car data for an agent from the state"""
        if isinstance(agent_id, str):
            car_id = agent_id
        else:
            # If agent_id is not a string, assume it has car_id attribute
            try:
                car_id = str(agent_id.car_id)
            except AttributeError:
                # Just use the agent_id as car_id if all else fails
                car_id = str(agent_id)
                
        if car_id in state.cars:
            return state.cars[car_id]
        return None


class BallProximityReward(BaseRewardFunction):
    """Reward based on proximity to ball"""
    def __init__(self, negative_slope=False, dispersion=1.0, density=1.0):
        super().__init__()
        self.negative_slope = negative_slope  # If true, reward decreases with distance
        self.dispersion = dispersion  # Controls how spread out the reward is
        self.density = density  # Controls reward intensity
        
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        if not state or not hasattr(state, 'ball') or not state.ball:
            return 0.0
        
        # Get car data for this agent
        car_data = self._get_car_data_from_state(agent_id, state)
        if not car_data or not hasattr(car_data, 'position'):
            return 0.0
            
        ball_pos = np.array(state.ball.position)
        car_pos = np.array(car_data.position)
        
        # Calculate distance
        distance = np.linalg.norm(ball_pos - car_pos)
        
        # Base reward calculation
        if self.negative_slope:
            # Linear penalty based on distance
            max_distance = 12000  # Approximate max distance on field
            reward = 1.0 - (distance / max_distance)
        else:
            # Traditional proximity reward (higher when closer)
            reward = 1.0 / (1.0 + distance)
        
        # Apply dispersion modifier - higher values spread the reward more evenly
        if self.dispersion != 1.0:
            if reward > 0:
                reward = reward ** (1.0 / self.dispersion)
            else:
                reward = -((-reward) ** (1.0 / self.dispersion))
                
        # Apply density modifier - higher values increase reward intensity
        if self.density != 1.0:
            reward = reward * self.density
            
        return clamp_reward(reward)


class BallToGoalDistanceReward(BaseRewardFunction):
    """Reward based on ball's distance to goal"""
    def __init__(self, team_goal_y=5120, offensive_dispersion=1.0, defensive_dispersion=1.0, 
                 offensive_density=1.0, defensive_density=1.0):
        super().__init__()
        self.team_goal_y = team_goal_y
        self.offensive_dispersion = offensive_dispersion
        self.defensive_dispersion = defensive_dispersion
        self.offensive_density = offensive_density
        self.defensive_density = defensive_density
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        if not state or not hasattr(state, 'ball') or not state.ball:
            return 0.0
            
        ball_pos = np.array(state.ball.position)
        goal_pos = np.array([0, self.team_goal_y, 100])  # Basic goal position
        
        # Calculate distance
        distance = np.linalg.norm(ball_pos - goal_pos)
        max_distance = 12000  # Approximate max possible distance
        reward = 1.0 - (distance / max_distance)
        
        # Determine if we're in offensive or defensive half
        is_offensive = (self.team_goal_y > 0 and ball_pos[1] > 0) or (self.team_goal_y < 0 and ball_pos[1] < 0)
        
        # Apply appropriate modifiers based on field position
        if is_offensive:
            # Apply offensive modifiers
            if self.offensive_dispersion != 1.0:
                if reward > 0:
                    reward = reward ** (1.0 / self.offensive_dispersion)
                else:
                    reward = -((-reward) ** (1.0 / self.offensive_dispersion))
            
            if self.offensive_density != 1.0:
                reward = reward * self.offensive_density
        else:
            # Apply defensive modifiers
            if self.defensive_dispersion != 1.0:
                if reward > 0:
                    reward = reward ** (1.0 / self.defensive_dispersion)
                else:
                    reward = -((-reward) ** (1.0 / self.defensive_dispersion))
            
            if self.defensive_density != 1.0:
                reward = reward * self.defensive_density
        
        return clamp_reward(reward)


class BallVelocityToGoalReward(BaseRewardFunction):
    """Reward based on ball velocity towards goal"""
    def __init__(self, team_goal_y=5120):
        super().__init__()
        self.team_goal_y = team_goal_y
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        if (not state or not hasattr(state, 'ball') or not state.ball or 
            not hasattr(state.ball, 'linear_velocity')):
            return 0.0
            
        ball_vel = np.array(state.ball.linear_velocity)
        ball_pos = np.array(state.ball.position)
        goal_pos = np.array([0, self.team_goal_y, 100])
        
        # Get direction to goal
        to_goal = goal_pos - ball_pos
        to_goal_dist = np.linalg.norm(to_goal)
        if to_goal_dist == 0:
            return 0.0
            
        to_goal = to_goal / to_goal_dist
        
        # Project velocity onto goal direction
        vel_to_goal = np.dot(ball_vel, to_goal)
        max_vel = 6000  # Approximate max ball velocity
        
        reward = vel_to_goal / max_vel
        return clamp_reward(reward)


class TouchBallReward(BaseRewardFunction):
    """Reward for touching the ball"""
    def __init__(self):
        super().__init__()
        self.last_touched = {}
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        """Reset last touched status for all agents"""
        self.last_touched = {str(agent): False for agent in agents}
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        if not state or not hasattr(state, 'last_touch') or not state.last_touch:
            return 0.0
        
        # Get the car id as a string
        if isinstance(agent_id, str):
            car_id = agent_id
        else:
            try:
                car_id = str(agent_id.car_id)
            except AttributeError:
                car_id = str(agent_id)
        
        # Initialize for this player if not already present
        if car_id not in self.last_touched:
            self.last_touched[car_id] = False
            
        # Check if this player touched the ball
        player_touched = state.last_touch.player_index == car_id
        if player_touched and not self.last_touched[car_id]:
            self.last_touched[car_id] = True
            return 1.0
            
        # Reset touch status if not currently touching
        if not player_touched:
            self.last_touched[car_id] = False
            
        return 0.0


class TouchBallToGoalAccelerationReward(BaseRewardFunction):
    """Reward for touching the ball in a way that accelerates it towards the goal"""
    def __init__(self, team_goal_y=5120):
        super().__init__()
        self.team_goal_y = team_goal_y
        self.last_velocity = {}
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        """Reset last velocity for all agents"""
        self.last_velocity = {str(agent): None for agent in agents}
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        if (not state or not hasattr(state, 'ball') or not state.ball or 
            not hasattr(state.ball, 'linear_velocity')):
            return 0.0
        
        # Get the car id as a string
        if isinstance(agent_id, str):
            car_id = agent_id
        else:
            try:
                car_id = str(agent_id.car_id)
            except AttributeError:
                car_id = str(agent_id)
            
        # Initialize for this player if not already present
        if car_id not in self.last_velocity:
            self.last_velocity[car_id] = None
            
        current_vel = np.array(state.ball.linear_velocity)
        
        # If no previous velocity or no touch, update and return 0
        player_touched = hasattr(state, 'last_touch') and state.last_touch and state.last_touch.player_index == car_id
        if self.last_velocity[car_id] is None or not player_touched:
            self.last_velocity[car_id] = current_vel
            return 0.0
            
        # Calculate acceleration towards goal
        ball_pos = np.array(state.ball.position)
        goal_pos = np.array([0, self.team_goal_y, 100])
        to_goal = goal_pos - ball_pos
        to_goal_dist = np.linalg.norm(to_goal)
        if to_goal_dist == 0:
            return 0.0
            
        to_goal = to_goal / to_goal_dist
        
        # Get change in velocity towards goal
        prev_vel_to_goal = np.dot(self.last_velocity[car_id], to_goal)
        curr_vel_to_goal = np.dot(current_vel, to_goal)
        acceleration = curr_vel_to_goal - prev_vel_to_goal
        
        self.last_velocity[car_id] = current_vel
        
        # Normalize by max possible acceleration
        max_acceleration = 6000  # Approximate max acceleration
        reward = acceleration / max_acceleration
        
        return clamp_reward(reward)


class AlignBallToGoalReward(BaseRewardFunction):
    """Reward for being between the ball and goal"""
    def __init__(self, team_goal_y=5120, dispersion=1.0, density=1.0):
        super().__init__()
        self.team_goal_y = team_goal_y
        self.dispersion = dispersion  # Controls how spread out the reward is
        self.density = density  # Controls reward intensity
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        if not state or not hasattr(state, 'ball') or not state.ball:
            return 0.0
        
        # Get car data for this agent
        car_data = self._get_car_data_from_state(agent_id, state)
        if not car_data or not hasattr(car_data, 'position'):
            return 0.0
            
        ball_pos = np.array(state.ball.position)
        car_pos = np.array(car_data.position)
        goal_pos = np.array([0, self.team_goal_y, 100])
        
        # Vector from ball to goal
        ball_to_goal = goal_pos - ball_pos
        ball_to_goal_dist = np.linalg.norm(ball_to_goal)
        if ball_to_goal_dist == 0:
            return 0.0
            
        ball_to_goal = ball_to_goal / ball_to_goal_dist
        
        # Vector from ball to car
        ball_to_car = car_pos - ball_pos
        ball_to_car_dist = np.linalg.norm(ball_to_car)
        if ball_to_car_dist == 0:
            return 1.0
            
        ball_to_car = ball_to_car / ball_to_car_dist
        
        # Reward is dot product (cosine of angle) modified by dispersion and density
        alignment = np.dot(ball_to_goal, ball_to_car)
        
        # Apply dispersion and density modifiers if enabled
        if self.dispersion != 1.0:
            # Higher dispersion makes the reward more spread out
            alignment = alignment ** (1.0 / self.dispersion)
        
        if self.density != 1.0:
            # Higher density increases reward intensity in high-alignment areas
            alignment = alignment * self.density
            
        return clamp_reward(alignment)


class PlayerVelocityTowardBallReward(BaseRewardFunction):
    """Reward for moving toward the ball"""
    def __init__(self):
        super().__init__()
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        if not state or not hasattr(state, 'ball') or not state.ball:
            return 0.0
            
        # Get car data for this agent
        car_data = self._get_car_data_from_state(agent_id, state)
        if not car_data or not hasattr(car_data, 'position') or not hasattr(car_data, 'linear_velocity'):
            return 0.0
            
        ball_pos = np.array(state.ball.position)
        car_pos = np.array(car_data.position)
        car_vel = np.array(car_data.linear_velocity)
        
        # Get direction to ball
        to_ball = ball_pos - car_pos
        distance = np.linalg.norm(to_ball)
        if distance == 0:
            return 1.0
            
        to_ball = to_ball / distance
        
        # Project velocity onto direction to ball
        vel_to_ball = np.dot(car_vel, to_ball)
        max_vel = 2300  # Max car velocity
        
        reward = vel_to_ball / max_vel
        return clamp_reward(reward)


class KRCReward(BaseRewardFunction):
    """Key Rocket Concepts reward - combines multiple reward components"""
    def __init__(self, reward_functions=None, team_spirit=0.0, team_goal_y=5120):
        """
        Initialize the KRC reward with customizable reward components
        
        Args:
            reward_functions: List of (reward_function, weight) tuples
            team_spirit: Weight for team cooperation (0.0 = individual, 1.0 = team)
            team_goal_y: Y coordinate of the team's goal
        """
        super().__init__()
        # Use provided reward functions or default ones
        self.reward_functions = reward_functions if reward_functions is not None else [
            (BallProximityReward(), 0.3),
            (BallToGoalDistanceReward(team_goal_y), 0.2),
            (TouchBallReward(), 0.2),
            (BallVelocityToGoalReward(team_goal_y), 0.15),
            (AlignBallToGoalReward(team_goal_y), 0.15)
        ]
        self.team_spirit = team_spirit
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        """Reset all component reward functions"""
        for reward_fn, _ in self.reward_functions:
            reward_fn.reset(agents, initial_state, shared_info)
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        reward = 0.0
        for reward_fn, weight in self.reward_functions:
            reward += weight * reward_fn.calculate(agent_id, state, previous_state)
        
        # Team cooperation logic could be added here using self.team_spirit
        
        return clamp_reward(reward)
