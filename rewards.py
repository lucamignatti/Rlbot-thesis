from typing import List, Dict, Any, Tuple, Union, Optional
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
import numpy as np


class DummyReward(RewardFunction[AgentID, GameState, float]):
    """A reward function that always returns zero.
    
    This is used for unsupervised pre-training where we want intrinsic rewards only,
    with no domain-specific extrinsic rewards.
    """
    def __init__(self):
        super().__init__()
        
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        """Reset any state in the reward function"""
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState,
                   is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool],
                   shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        """Return zero rewards for all agents"""
        return {agent: 0.0 for agent in agents}


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
        
    def _get_player_team(self, agent_id, state: GameState):
        """Helper method to get player team from the state"""
        car_data = self._get_car_data_from_state(agent_id, state)
        if car_data and hasattr(car_data, 'team_num'):
            return car_data.team_num
        return 0  # Default to blue team if team cannot be determined


class ParameterizedReward(BaseRewardFunction):
    """
    Base class for parameterized rewards as described in Lucy-SKG paper.
    Implements Equation 3: Rdist = exp(-0.5 * d(i,j)/(cd*wdis))^(1/wden)
    """
    def __init__(self, dispersion=1.0, density=1.0):
        """
        Args:
            dispersion: Controls distance reward spread (wdis in paper)
            density: Controls concavity/value intensity (wden in paper)
        """
        super().__init__()
        self.dispersion = dispersion  # wdis in the paper
        self.density = density  # wden in the paper

    def apply_parameterization(self, raw_value: float, normalize_constant: float = 1.0) -> float:
        """
        Apply parameterization to distance values as per paper Equation 3:
        Rdist = exp(-0.5 * d(i,j)/(cd*wdis))^(1/wden)
        
        Args:
            raw_value: Raw distance or scalar value to parameterize
            normalize_constant: Normalizing constant (cd in paper)
        """
        # Handle sign preservation
        sign = np.sign(raw_value)
        abs_value = abs(raw_value)
        
        # Apply parameterization (paper's Equation 3)
        parameterized = np.exp(-0.5 * abs_value / (normalize_constant * self.dispersion)) ** (1.0 / self.density)
        
        return sign * parameterized


class BallProximityReward(ParameterizedReward):
    """
    Reward based on proximity to ball.
    Implements a parameterized distance reward function.
    """
    def __init__(self, negative_slope=False, dispersion=1.0, density=1.0):
        super().__init__(dispersion, density)
        self.negative_slope = negative_slope  # If true, reward decreases with distance

    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:        
        if not state:
            return 0.0
            
        if not hasattr(state, 'ball'):
            return 0.0
            
        if not state.ball:
            return 0.0
        
        # Get car data for this agent
        car_data = self._get_car_data_from_state(agent_id, state)
        
        if not car_data:
            return 0.0
        
        
        # Try different common position attributes
        car_pos = None
        if hasattr(car_data, 'position'):
            car_pos = np.array(car_data.position)
        elif hasattr(car_data, 'physics'):
            physics = car_data.physics
            if hasattr(physics, 'location'):
                car_pos = np.array(physics.location)
            elif hasattr(physics, 'position'):
                car_pos = np.array(physics.position)
            elif hasattr(physics, 'pos'):
                car_pos = np.array(physics.pos)
            elif hasattr(physics, 'x') and hasattr(physics, 'y') and hasattr(physics, 'z'):
                car_pos = np.array([physics.x, physics.y, physics.z])
        elif hasattr(car_data, 'location'):
            car_pos = np.array(car_data.location)
        elif hasattr(car_data, 'pos'):
            car_pos = np.array(car_data.pos)
        
        if car_pos is None:
            return 0.0
                
        ball_pos = np.array(state.ball.position)
        
        # Calculate distance
        distance = np.linalg.norm(ball_pos - car_pos)
        
        # Normalize distance (cd in paper)
        normalize_constant = 2300  # Approximate car length for normalization
        normalized_distance = distance / normalize_constant
        
        if self.negative_slope:
            # For negative slope, invert the value
            normalized_distance = 1.0 - normalized_distance
        
        # Apply parameterization as defined in the paper (Equation 3)
        reward = self.apply_parameterization(normalized_distance, 1.0)
        
        return clamp_reward(reward)

class BallToGoalDistanceReward(ParameterizedReward):
    """
    Reward based on ball's distance to goal.
    Implements a parameterized distance reward function with separate params for offense/defense.
    """
    def __init__(self, team_goal_y=5120, 
                 offensive_dispersion=0.6, defensive_dispersion=0.4,
                 offensive_density=1.0, defensive_density=1.0):
        """
        Initialize with separate parameters for offensive and defensive situations
        as described in the Lucy-SKG paper.
        """
        super().__init__(1.0, 1.0)  # Base class init with dummy values
        self.team_goal_y = team_goal_y
        self.offensive_dispersion = offensive_dispersion  # wdisoff in the paper
        self.defensive_dispersion = defensive_dispersion  # wdisdef in the paper
        self.offensive_density = offensive_density  # wdenoff in the paper
        self.defensive_density = defensive_density  # wdendef in the paper

    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        if not state or not hasattr(state, 'ball') or not state.ball:
            return 0.0
        
        ball_pos = np.array(state.ball.position)
        goal_pos = np.array([0, self.team_goal_y, 100])  # Basic goal position
        team_goal_pos = np.array([0, -self.team_goal_y, 100])  # Own team's goal
        
        # Calculate ball-to-opponent-goal distance
        distance_to_goal = np.linalg.norm(ball_pos - goal_pos)
        # Calculate ball-to-own-goal distance
        distance_to_own_goal = np.linalg.norm(ball_pos - team_goal_pos)
        
        # Normalize distances (cd in paper)
        normalize_constant = 6000  # Approximate distance constant 
        goal_depth = 880  # Approximate goal depth
        
        # Formula from the paper's "Ball-to-Goal Distance Difference" function:
        # Φddb2g = woff * exp(-0.5 * ||d_ball,target|| - ℓgoal / 6000 * wdisoff)^(1/wdenoff)
        #         - wdef * exp(-0.5 * ||d_ball,blue_target|| - ℓgoal / 6000 * wdisdef)^(1/wdendef)
        
        # Offensive component (ball close to opponent goal)
        offensive_normalized = (distance_to_goal - goal_depth) / normalize_constant
        # Use the paper's parameterization
        offensive_reward = np.exp(-0.5 * offensive_normalized / self.offensive_dispersion) ** (1.0 / self.offensive_density)
        
        # Defensive component (ball far from own goal)
        defensive_normalized = (distance_to_own_goal - goal_depth) / normalize_constant
        # Use the paper's parameterization
        defensive_reward = np.exp(-0.5 * defensive_normalized / self.defensive_dispersion) ** (1.0 / self.defensive_density)
        
        # Combine offensive and defensive components
        # The paper uses weights woff and wdef, defaulting to 2.0 for offensive weight
        offensive_weight = 2.0
        defensive_weight = 1.0
        reward = offensive_weight * offensive_reward - defensive_weight * defensive_reward
        
        return clamp_reward(reward)


class BallVelocityToGoalReward(BaseRewardFunction):
    """Reward based on ball velocity towards goal"""
    def __init__(self, team_goal_y=5120, weight=0.8):
        super().__init__()
        self.team_goal_y = team_goal_y
        self.weight = weight

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
        
        # Formula from the paper's "Ball-to-Goal Velocity" function:
        # Φub2g = d_ball,target / ||d_ball,target|| * u_ball / 6000
        reward = vel_to_goal / max_vel
        return clamp_reward(reward * self.weight)


class TouchBallReward(BaseRewardFunction):
    """Reward for touching the ball - event reward"""
    def __init__(self, weight=0.05):
        super().__init__()
        self.last_touched = {}
        self.weight = weight

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
            # Weight as per paper's "Touch" function (weight = 0.05)
            return self.weight
        
        # Reset touch status if not currently touching
        if not player_touched:
            self.last_touched[car_id] = False
        
        return 0.0


class TouchBallToGoalAccelerationReward(BaseRewardFunction):
    """
    Reward for touching the ball in a way that accelerates it towards the goal.
    Updated from 'Touch Ball Acceleration' to 'Touch Ball-to-Goal Acceleration' as in paper.
    """
    def __init__(self, team_goal_y=5120, weight=0.25):
        super().__init__()
        self.team_goal_y = team_goal_y
        self.last_velocity = {}
        self.weight = weight

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
        
        # Per paper "Touch Ball-to-Goal Acceleration" with weight 0.25
        # This doubles function range by introducing direction and penalizes for hitting the ball toward team goal
        reward = acceleration / 6000  # Normalized by max possible acceleration
        
        return clamp_reward(reward * self.weight)


class AlignBallToGoalReward(ParameterizedReward):
    """
    Reward for being between the ball and goal.
    Implements a parameterized version to match Lucy-SKG paper.
    """
    def __init__(self, team_goal_y=5120, dispersion=1.0, density=1.0):
        super().__init__(dispersion, density)
        self.team_goal_y = team_goal_y

    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        if not state or not hasattr(state, 'ball') or not state.ball:
            return 0.0
        
        # Get car data for this agent
        car_data = self._get_car_data_from_state(agent_id, state)
        if not car_data:
            return 0.0
            
        # Try different common position attributes
        car_pos = None
        if hasattr(car_data, 'position'):
            car_pos = np.array(car_data.position)
        elif hasattr(car_data, 'physics'):
            physics = car_data.physics
            if hasattr(physics, 'location'):
                car_pos = np.array(physics.location)
            elif hasattr(physics, 'position'):
                car_pos = np.array(physics.position)
            elif hasattr(physics, 'pos'):
                car_pos = np.array(physics.pos)
            elif hasattr(physics, 'x') and hasattr(physics, 'y') and hasattr(physics, 'z'):
                car_pos = np.array([physics.x, physics.y, physics.z])
        elif hasattr(car_data, 'location'):
            car_pos = np.array(car_data.location)
        elif hasattr(car_data, 'pos'):
            car_pos = np.array(car_data.pos)
            
        if car_pos is None:
            return 0.0
        
        ball_pos = np.array(state.ball.position)
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
        
        # Alignment is dot product (cosine of angle)
        alignment = np.dot(ball_to_goal, ball_to_car)
        
        # Apply parameterization as described in the paper
        # We scale the alignment from [-1,1] to [0,1] for parameterization
        scaled_alignment = (alignment + 1) / 2
        parameterized = self.apply_parameterization(scaled_alignment)
        # Scale back to [-1,1]
        reward = parameterized * 2 - 1
        
        return clamp_reward(reward)


class PlayerVelocityTowardBallReward(BaseRewardFunction):
    """
    Reward for moving toward the ball.
    Used in Lucy-SKG for the 'Offensive Potential' KRC.
    """
    def __init__(self):
        super().__init__()

    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        if not state or not hasattr(state, 'ball') or not state.ball:
            return 0.0
        
        # Get car data for this agent
        car_data = self._get_car_data_from_state(agent_id, state)
        if not car_data:
            return 0.0
            
        # Try different common position attributes
        car_pos = None
        if hasattr(car_data, 'position'):
            car_pos = np.array(car_data.position)
        elif hasattr(car_data, 'physics'):
            physics = car_data.physics
            if hasattr(physics, 'location'):
                car_pos = np.array(physics.location)
            elif hasattr(physics, 'position'):
                car_pos = np.array(physics.position)
            elif hasattr(physics, 'pos'):
                car_pos = np.array(physics.pos)
            elif hasattr(physics, 'x') and hasattr(physics, 'y') and hasattr(physics, 'z'):
                car_pos = np.array([physics.x, physics.y, physics.z])
        elif hasattr(car_data, 'location'):
            car_pos = np.array(car_data.location)
        elif hasattr(car_data, 'pos'):
            car_pos = np.array(car_data.pos)
            
        # Try different common velocity attributes
        car_vel = None
        if hasattr(car_data, 'linear_velocity'):
            car_vel = np.array(car_data.linear_velocity)
        elif hasattr(car_data, 'physics'):
            physics = car_data.physics
            if hasattr(physics, 'velocity'):
                car_vel = np.array(physics.velocity)
            elif hasattr(physics, 'linear_velocity'):
                car_vel = np.array(physics.linear_velocity)
            elif hasattr(physics, 'vel'):
                car_vel = np.array(physics.vel)
        elif hasattr(car_data, 'velocity'):
            car_vel = np.array(car_data.velocity)
        elif hasattr(car_data, 'vel'):
            car_vel = np.array(car_data.vel)
            
        if car_pos is None or car_vel is None:
            return 0.0
        
        ball_pos = np.array(state.ball.position)
        
        # Get direction to ball
        to_ball = ball_pos - car_pos
        distance = np.linalg.norm(to_ball)
        if distance == 0:
            return 1.0
        
        to_ball = to_ball / distance
        
        # Project velocity onto direction to ball
        vel_to_ball = np.dot(car_vel, to_ball)
        max_vel = 2300  # Max car velocity
        
        # Formula from paper's "Player-to-Ball Velocity" component:
        # ϕup2b = d_car,ball / ||d_car,ball|| · u_car / 2300
        reward = vel_to_ball / max_vel
        return clamp_reward(reward)


class SaveBoostReward(BaseRewardFunction):
    """
    Reward for saving boost.
    Implements the 'Save boost' utility from Lucy-SKG paper.
    """
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight

    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        # Get car data for this agent
        car_data = self._get_car_data_from_state(agent_id, state)
        if not car_data or not hasattr(car_data, 'boost_amount'):
            return 0.0
        
        # Formula from paper: Φboost = √boost/100
        boost_amount = car_data.boost_amount
        reward = np.sqrt(boost_amount / 100.0)
        
        return clamp_reward(reward * self.weight)


class KRCRewardFunction(BaseRewardFunction):
    """
    Kinesthetic Reward Combination (KRC) as described in the Lucy-SKG paper.
    Combines reward components using geometric mean approach rather than linear combinations.
    
    Formula from paper: Rc = sgn(r) * n√(∏|Ri|)
    Where sgn(r) is 1 if all rewards are positive, -1 otherwise.
    """
    def __init__(self, components, team_spirit=0.3):
        """
        Args:
            components: List of reward functions to combine
            team_spirit: Team spirit factor τ (0 = individual rewards only, 1 = team rewards only)
        """
        super().__init__()
        self.components = components
        self.team_spirit = team_spirit
        self.team_rewards = {0: [], 1: []}  # Blue team is 0, Orange team is 1
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        # Reset all component reward functions
        for component in self.components:
            if hasattr(component, 'reset'):
                component.reset(agents, initial_state, shared_info)
        
        # Reset team rewards tracking
        self.team_rewards = {0: [], 1: []}
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        # Get rewards from all components
        rewards = []
        for component in self.components:
            # Check if component is a tuple (reward_fn, params...) or direct reward function
            if isinstance(component, tuple):
                reward_fn = component[0]  # First element should be the reward function
                reward = reward_fn.calculate(agent_id, state, previous_state)
            else:
                # Direct reward function
                reward = component.calculate(agent_id, state, previous_state)
            rewards.append(reward)
        
        # Apply KRC formula: sgn(r) * n√(∏|Ri|)
        if 0.0 in rewards:
            # If any component is zero, result is zero
            krc_reward = 0.0
        else:
            # Take absolute values for product
            abs_rewards = [abs(r) for r in rewards]
            # Calculate geometric mean (n√∏|Ri|)
            geometric_mean = np.prod(abs_rewards) ** (1.0 / len(rewards))
            # Sign is determined by all components - positive only when all are positive
            # Sign is determined by all components - positive only when all are positive
            sign = 1.0 if all(r >= 0 for r in rewards) else -1.0
            krc_reward = sign * geometric_mean
        
        # Track rewards for team spirit calculation
        player_team = self._get_player_team(agent_id, state)
        self.team_rewards[player_team].append(krc_reward)
        
        # Apply team spirit formula: R'i = (1-τ)*R'i + τ*(R'team - R'opponent)
        if self.team_spirit > 0:
            opponent_team = 1 if player_team == 0 else 0
            team_avg = np.mean(self.team_rewards[player_team]) if self.team_rewards[player_team] else 0
            opponent_avg = np.mean(self.team_rewards[opponent_team]) if self.team_rewards[opponent_team] else 0
            team_component = team_avg - opponent_avg
            krc_reward = (1.0 - self.team_spirit) * krc_reward + self.team_spirit * team_component
        
        return clamp_reward(krc_reward)


class DistanceWeightedAlignmentKRC(KRCRewardFunction):
    """
    Distance-weighted Alignment KRC from Lucy-SKG paper.
    Combines 'Align Ball-to-Goal' and 'Player-to-Ball Distance'.
    """
    def __init__(self, team_goal_y=5120, dispersion=1.1, weight=0.6):
        self.align_ball_goal = AlignBallToGoalReward(team_goal_y=team_goal_y, dispersion=dispersion)
        self.player_ball_dist = BallProximityReward(dispersion=dispersion)
        
        super().__init__(
            components=[self.align_ball_goal, self.player_ball_dist],
            team_spirit=0.3
        )
        self.weight = weight
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        # Get the KRC reward and apply weight
        krc_reward = super().calculate(agent_id, state, previous_state)
        return krc_reward * self.weight


class OffensivePotentialKRC(KRCRewardFunction):
    """
    Offensive Potential KRC reward from Lucy-SKG paper.
    Combines ball alignment to goal, player-ball distance, and player-ball velocity.
    """
    def __init__(self, team_goal_y=5120, dispersion=1.1, density=1.1, weight=1.0):
        self.align_ball_goal = AlignBallToGoalReward(team_goal_y=team_goal_y, dispersion=dispersion)
        self.player_ball_dist = BallProximityReward(dispersion=dispersion, density=density)
        self.player_ball_vel = PlayerVelocityTowardBallReward()
        
        super().__init__(
            components=[self.align_ball_goal, self.player_ball_dist, self.player_ball_vel],
            team_spirit=0.3
        )
        self.weight = weight
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        # Get the KRC reward and apply weight
        krc_reward = super().calculate(agent_id, state, previous_state)
        return krc_reward * self.weight


class LucySKGReward(BaseRewardFunction):
    """
    Main reward function for Lucy-SKG as described in the paper.
    Combines all reward components with appropriate weights.
    """
    def __init__(self, team_goal_y=5120, team_spirit=0.3):
        super().__init__()
        # Reward shaping functions (utilities)
        self.ball_to_goal_distance = BallToGoalDistanceReward(
            team_goal_y=team_goal_y,
            offensive_dispersion=0.6,
            defensive_dispersion=0.4,
            offensive_density=1.0,
            defensive_density=1.0
        )
        self.ball_to_goal_velocity = BallVelocityToGoalReward(team_goal_y=team_goal_y, weight=0.8)
        self.save_boost = SaveBoostReward(weight=0.5)
        self.distance_weighted_alignment = DistanceWeightedAlignmentKRC(team_goal_y=team_goal_y, dispersion=1.1, weight=0.6)
        self.offensive_potential = OffensivePotentialKRC(team_goal_y=team_goal_y, dispersion=1.1, density=1.1, weight=1.0)
        
        # Event reward functions
        self.goal = None  # Goal event reward - weight 10.0
        self.concede = None  # Concede event reward - weight -3.0
        self.shot = None  # Shot event reward - weight 1.5
        self.touch_ball_to_goal_acceleration = TouchBallToGoalAccelerationReward(team_goal_y=team_goal_y, weight=0.25)
        self.touch_ball = TouchBallReward(weight=0.05)
        self.demolish = None  # Demolish event reward - weight 2.0
        self.demolished = None  # Demolished event reward - weight -2.0
        
        # Weights from paper
        self.weights = {
            'ball_to_goal_distance': 2.0,
            'ball_to_goal_velocity': 0.8,
            'save_boost': 0.5,
            'distance_weighted_alignment': 0.6,
            'offensive_potential': 1.0,
            'goal': 10.0,
            'concede': -3.0,
            'shot': 1.5,
            'touch_ball_to_goal_acceleration': 0.25,
            'touch_ball': 0.05,
            'demolish': 2.0,
            'demolished': -2.0
        }
        
        self.team_spirit = team_spirit
        self.team_rewards = {0: [], 1: []}  # Blue team is 0, Orange team is 1
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        # Reset all component reward functions
        for component in [
            self.ball_to_goal_distance, self.ball_to_goal_velocity, 
            self.save_boost, self.distance_weighted_alignment,
            self.offensive_potential, self.touch_ball_to_goal_acceleration, 
            self.touch_ball
        ]:
            if hasattr(component, 'reset'):
                component.reset(agents, initial_state, shared_info)
        
        # Reset team rewards tracking
        self.team_rewards = {0: [], 1: []}
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        reward = 0.0
        
        # Add utility rewards
        reward += self.weights['ball_to_goal_distance'] * self.ball_to_goal_distance.calculate(agent_id, state, previous_state)
        reward += self.weights['ball_to_goal_velocity'] * self.ball_to_goal_velocity.calculate(agent_id, state, previous_state)
        reward += self.weights['save_boost'] * self.save_boost.calculate(agent_id, state, previous_state)
        reward += self.distance_weighted_alignment.calculate(agent_id, state, previous_state)
        reward += self.offensive_potential.calculate(agent_id, state, previous_state)
        

        # Add event rewards
        # Note: Goal, Concede, Shot, Demolish and Demolished are not implemented here
        # as they're typically handled by the game environment directly
        reward += self.touch_ball_to_goal_acceleration.calculate(agent_id, state, previous_state)
        reward += self.touch_ball.calculate(agent_id, state, previous_state)
        
        # Track rewards for team spirit calculation
        player_team = self._get_player_team(agent_id, state)
        self.team_rewards[player_team].append(reward)
        
        # Apply team spirit formula: R'i = (1-τ)*R'i + τ*(R'team - R'opponent)
        if self.team_spirit > 0:
            opponent_team = 1 if player_team == 0 else 0
            team_avg = np.mean(self.team_rewards[player_team]) if self.team_rewards[player_team] else 0
            opponent_avg = np.mean(self.team_rewards[opponent_team]) if self.team_rewards[opponent_team] else 0
            team_component = team_avg - opponent_avg
            reward = (1.0 - self.team_spirit) * reward + self.team_spirit * team_component
        
        return clamp_reward(reward)


class KRCReward(BaseRewardFunction):
    """
    Kinesthetic Reward Combination (KRC) as described in the Lucy-SKG paper.
    Combines reward components using geometric mean approach rather than linear combinations.
    
    Formula from paper: Rc = sgn(r) * n√(∏|Ri|)
    Where sgn(r) is 1 if all rewards are positive, -1 otherwise.
    """
    def __init__(self, reward_functions=None, team_spirit=0.3, team_goal_y=5120):
        """
        Initialize the KRC reward with customizable reward components

        Args:
            reward_functions: List of (reward_function, dispersion, density) tuples
                              or List of (reward_function, weight) tuples
                              If dispersion/density not needed, use None
            team_spirit: Weight for team cooperation (0.0 = individual, 1.0 = team)
            team_goal_y: Y coordinate of the team's goal
        """
        super().__init__()
        
        # Process reward functions to ensure they all have 3 elements
        processed_reward_functions = []
        if reward_functions is not None:
            for item in reward_functions:
                if len(item) == 2:
                    # If only 2 elements (reward_fn, weight), add None for dispersion and density
                    reward_fn, weight = item
                    processed_reward_functions.append((reward_fn, None, None))
                elif len(item) == 3:
                    # If already 3 elements, use as is
                    processed_reward_functions.append(item)
                else:
                    raise ValueError(f"Invalid reward function format: {item}")
        else:
            # Default reward functions as before
            processed_reward_functions = [
                (BallProximityReward(), None, None),
                (BallToGoalDistanceReward(team_goal_y), None, None),
                (TouchBallReward(), None, None),
                (BallVelocityToGoalReward(team_goal_y), None, None),
                (AlignBallToGoalReward(team_goal_y), None, None)
            ]
            
        self.reward_functions = processed_reward_functions
        self.team_spirit = team_spirit
        self.team_rewards = {0: [], 1: []}  # Blue team is 0, Orange team is 1
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        """Reset all component reward functions"""
        for reward_fn, _, _ in self.reward_functions:
            if hasattr(reward_fn, 'reset'):
                reward_fn.reset(agents, initial_state, shared_info)
        
        # Reset team rewards tracking
        self.team_rewards = {0: [], 1: []}
    
    def calculate(self, agent_id, state: GameState, previous_state: Optional[GameState] = None) -> float:
        # Get rewards from all components
        rewards = []
        for reward_fn, dispersion, density in self.reward_functions:
            raw_reward = reward_fn.calculate(agent_id, state, previous_state)
            
            # Apply parameterization if provided and if the reward function supports it
            if dispersion is not None and density is not None and hasattr(reward_fn, 'apply_parameterization'):
                reward = reward_fn.apply_parameterization(raw_reward, dispersion, density)
            else:
                reward = raw_reward
                
            rewards.append(reward)
        
        # Apply KRC formula: sgn(r) * n√(∏|Ri|)
        if 0.0 in rewards:
            # If any component is zero, result is zero
            krc_reward = 0.0
        else:
            # Take absolute values for product
            abs_rewards = [abs(r) for r in rewards]
            # Calculate geometric mean (n√∏|Ri|)
            geometric_mean = np.prod(abs_rewards) ** (1.0 / len(rewards))
            # Sign is determined by all components - positive only when all are positive
            sign = 1.0 if all(r >= 0 for r in rewards) else -1.0
            krc_reward = sign * geometric_mean
        
        # Track rewards for team spirit calculation
        player_team = self._get_player_team(agent_id, state)
        self.team_rewards[player_team].append(krc_reward)
        
        # Apply team spirit formula: R'i = (1-τ)*R'i + τ*(R'team - R'opponent)
        if self.team_spirit > 0:
            opponent_team = 1 if player_team == 0 else 0
            team_avg = np.mean(self.team_rewards[player_team]) if self.team_rewards[player_team] else 0
            opponent_avg = np.mean(self.team_rewards[opponent_team]) if self.team_rewards[opponent_team] else 0
            team_component = team_avg - opponent_avg
            krc_reward = (1.0 - self.team_spirit) * krc_reward + self.team_spirit * team_component
        
        return clamp_reward(krc_reward)

# Helper functions for creating Lucy-SKG reward components

def create_offensive_potential_reward(team_goal_y=5120):
    """
    Creates the 'Offensive Potential' KRC from the Lucy-SKG paper,
    which combines Ball-to-Goal Alignment, Player-to-Ball Distance,
    and Player-to-Ball Velocity.
    """
    return OffensivePotentialKRC(
        team_goal_y=team_goal_y,
        dispersion=1.1, 
        density=1.1, 
        weight=1.0
    )

def create_distance_weighted_alignment_reward(team_goal_y=5120):
    """
    Creates the 'Distance-weighted Alignment' KRC from the Lucy-SKG paper,
    which combines Ball-to-Goal Alignment and Player-to-Ball Distance.
    """
    return DistanceWeightedAlignmentKRC(
        team_goal_y=team_goal_y,
        dispersion=1.1,
        weight=0.6
    )


def create_lucy_skg_reward(team_goal_y=5120):
    """
    Creates the complete Lucy-SKG reward function as described in the paper.
    Includes all utility and event rewards with appropriate weights.
    """
    # Create the reward components
    ball_to_goal_distance = BallToGoalDistanceReward(
        team_goal_y=team_goal_y,
        offensive_dispersion=0.6,
        defensive_dispersion=0.4,
        offensive_density=1.0,
        defensive_density=1.0
    )
    ball_to_goal_velocity = BallVelocityToGoalReward(team_goal_y=team_goal_y, weight=0.8)
    save_boost = SaveBoostReward(weight=0.5)
    distance_weighted_alignment = create_distance_weighted_alignment_reward(team_goal_y)
    offensive_potential = create_offensive_potential_reward(team_goal_y)
    touch_ball_to_goal_acceleration = TouchBallToGoalAccelerationReward(team_goal_y=team_goal_y, weight=0.25)
    touch_ball = TouchBallReward(weight=0.05)
    
    # Combine into a single reward function
    return LucySKGReward(team_goal_y=team_goal_y, team_spirit=0.3)
