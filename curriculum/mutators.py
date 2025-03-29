"""Game state mutators for curriculum learning."""
from typing import Optional, Callable, Dict, Any, List
import numpy as np
from rlgym.api import StateMutator, DoneCondition, AgentID
from rlgym.rocket_league.api import GameState, PhysicsObject # Import PhysicsObject

# --- Helper Functions for Orientation ---
def euler_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Converts Euler angles (pitch, yaw, roll) to a quaternion (x, y, z, w)."""
    pitch, yaw, roll = rotation
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qx, qy, qz, qw])

def euler_to_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Converts Euler angles (pitch, yaw, roll) to a 3x3 rotation matrix."""
    pitch, yaw, roll = rotation
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combined rotation: ZYX order is common for aerospace/vehicles
    rotation_mtx = np.dot(R_z, np.dot(R_y, R_x))
    return rotation_mtx
# --- End Helper Functions ---

class BallTowardGoalSpawnMutator(StateMutator):
    """Spawns the ball in a position that lines up with the goal"""
    def __init__(self, offensive_team=0, distance_from_goal=0.7, random_offset=0.2):
        self.offensive_team = offensive_team
        self.distance_from_goal = distance_from_goal
        self.random_offset = random_offset

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        goal_direction = 1 if self.offensive_team == 0 else -1
        x_pos = goal_direction * (1 - self.distance_from_goal) * 4096
        y_pos = (np.random.random() * 2 - 1) * self.random_offset * 5120
        
        state.ball.position = np.array([x_pos, y_pos, 100])
        velocity_magnitude = np.random.random() * 500
        state.ball.linear_velocity = np.array([goal_direction * velocity_magnitude, 0, 0])
        state.ball.angular_velocity = np.zeros(3) 

class BallPositionMutator(StateMutator):
    """Sets the ball's position using a callback function"""
    def __init__(self, position_function: Optional[Callable[[], np.ndarray]] = None):
        self.position_function = position_function or (lambda: np.array([0, 0, 100]))

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        position = self.position_function()
        state.ball.position = position
        state.ball.linear_velocity = np.zeros(3)
        state.ball.angular_velocity = np.zeros(3)

class BallVelocityMutator(StateMutator):
    """Sets the ball's velocity using a callback function"""
    def __init__(self, velocity_function: Optional[Callable[[], np.ndarray]] = None):
        self.velocity_function = velocity_function or (lambda: np.zeros(3))

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        velocity = self.velocity_function()
        state.ball.linear_velocity = velocity
        state.ball.angular_velocity = np.zeros(3)

class CarPositionMutator(StateMutator):
    """Sets a car's position and optionally orientation using callback functions"""
    def __init__(self, car_id: str, 
                 position_function: Optional[Callable[[], np.ndarray]] = None,
                 orientation_function: Optional[Callable[[], np.ndarray]] = None,
                 debug: bool = False):
        self.car_id = car_id
        self.position_function = position_function
        self.orientation_function = orientation_function
        self.debug = debug
    
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        car_id_str = str(self.car_id)
        if car_id_str not in state.cars:
            return
        
        car = state.cars[car_id_str]
        
        # --- Position Setting (Added Debugging) ---
        position = None # Initialize position
        try:
            position = self.position_function()
            if self.debug:
                print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Generated Position: {position}")
            if position is None or not isinstance(position, (np.ndarray, list, tuple)):
                raise ValueError("Position function returned invalid position")
        except Exception as e:
            if self.debug:
                print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Error in position_function: {e}, using default position")
            position = np.array([0.0, -2000.0, 17.0]) # Default position
        
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        
        # Ensure physics object exists using the correct type
        if not hasattr(car, 'physics') or car.physics is None:
             car.physics = PhysicsObject() # Use the imported PhysicsObject
             if self.debug:
                 print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Created PhysicsObject instance")
             # Initialize rotation attribute if it doesn't exist after creation
             if not hasattr(car.physics, 'rotation'):
                 car.physics.rotation = np.zeros(3)
                 if self.debug:
                     print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Initialized physics.rotation")
        # Apply position
        try:
            car.physics.position = position
            if self.debug:
                print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Applied position: {car.physics.position}")
        except Exception as e:
             if self.debug:
                 print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Error applying position: {e}")
        # Ensure velocities exist (often needed by physics engine)
        if not hasattr(car.physics, 'linear_velocity') or car.physics.linear_velocity is None:
            car.physics.linear_velocity = np.zeros(3)
        if not hasattr(car.physics, 'angular_velocity') or car.physics.angular_velocity is None:
            car.physics.angular_velocity = np.zeros(3)
        # --- End Position Setting ---
        # --- Orientation Setting (Attach Euler Angles to Car Object) ---
        if self.orientation_function:
            try:
                rotation_euler = self.orientation_function()
                if self.debug:
                    print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Generated Euler Rotation: {rotation_euler}")
                if rotation_euler is None or not isinstance(rotation_euler, (np.ndarray, list, tuple)):
                     raise ValueError("Orientation function returned invalid rotation")
                if not isinstance(rotation_euler, np.ndarray):
                    rotation_euler = np.array(rotation_euler)
                
                # Attach Euler angles directly to the car object being modified
                try:
                    car._temp_euler_rotation = rotation_euler
                    if self.debug:
                        print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Attached _temp_euler_rotation = {car._temp_euler_rotation}")
                except Exception as e_set_temp:
                     if self.debug:
                         print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Error attaching _temp_euler_rotation: {repr(e_set_temp)}")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Error processing orientation function: {repr(e)}")
                # Attempt to set default neutral orientation if error occurs
                try:
                    car._temp_euler_rotation = np.zeros(3)
                    if self.debug:
                        print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Attached default Euler rotation due to error.")
                except Exception as fallback_e:
                     if self.debug:
                         print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Error attaching default Euler rotation: {repr(fallback_e)}")
        else:
             if self.debug:
                 print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, No orientation_function provided.")
             # Ensure default rotation exists if no function provided
             try:
                 if not hasattr(car, '_temp_euler_rotation'):
                      car._temp_euler_rotation = np.zeros(3)
             except:
                 pass # Ignore if we can't attach
        # --- End Orientation Setting ---

class CarBoostMutator(StateMutator):
    """Sets boost amount for specific cars"""
    def __init__(self, boost_amount: float = 100, car_id: Optional[int] = None):
        self.boost_amount = min(max(boost_amount, 0), 100)
        self.car_id = car_id

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        if self.car_id is not None:
            if str(self.car_id) in state.cars:
                state.cars[str(self.car_id)].boost_amount = self.boost_amount
        else:
            for car in state.cars.values():
                car.boost_amount = self.boost_amount

class CarBallRelativePositionMutator(StateMutator):
    """Sets a car's position relative to the ball's position"""
    def __init__(self, car_id: str, position_function: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Args:
            car_id: Identifier for the car to position
            position_function: Function that takes ball position and returns car position
                               If None, positions the car 1000 units behind the ball on y-axis
        """
        self.car_id = car_id
        self.position_function = position_function or (lambda ball_pos: np.array([ball_pos[0], ball_pos[1]-1000, 17]))
        
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        car_id_str = str(self.car_id)

        # Check if the car exists in the state
        if car_id_str not in state.cars:
            return
            
        # Get the car from the state
        car = state.cars[car_id_str]
        
        # Get the ball's current position
        ball_position = state.ball.position
        
        # Generate the car position relative to the ball
        try:
            position = self.position_function(ball_position)
            if position is None or not isinstance(position, (np.ndarray, list, tuple)):
                raise ValueError("Position function returned invalid position")
        except Exception as e:
            # If there's an error, use a safe default position (1000 units behind ball)
            print(f"Error in position_function: {e}, using default relative position")
            position = np.array([ball_position[0], ball_position[1]-1000, 17])
        
        # Ensure position is always a numpy array
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        
        # Set the car position
        if hasattr(car, 'physics'):
            if car.physics is None:
                car.physics = type('Physics', (), {})()
            car.physics.position = position
            
            # Initialize velocity as zeros if it doesn't exist
            if not hasattr(car.physics, 'linear_velocity') or car.physics.linear_velocity is None:
                car.physics.linear_velocity = np.zeros(3)
                
            # Initialize angular velocity as zeros if it doesn't exist
            if not hasattr(car.physics, 'angular_velocity') or car.physics.angular_velocity is None:
                car.physics.angular_velocity = np.zeros(3)
        else:
            car.position = position
        
        # Reset rotation to point car toward ball
        try:
            # Calculate direction vector from car to ball
            direction = ball_position - position
            
            # Only set rotation if we're not too close to the ball (avoid division by zero)
            if np.linalg.norm(direction) > 1.0:
                # Normalize direction
                direction = direction / np.linalg.norm(direction)
                
                # Simple rotation to face the ball (only yaw)
                yaw = np.arctan2(direction[1], direction[0])
                
                if hasattr(car, 'physics') and car.physics is not None:
                    car.physics.rotation = np.array([0, 0, yaw])
                else:
                    car.rotation = np.array([0, 0, yaw])
            else:
                # Default rotation if too close to ball
                if hasattr(car, 'physics') and car.physics is not None:
                    car.physics.rotation = np.zeros(3)
                else:
                    car.rotation = np.zeros(3)
        except (AttributeError, TypeError):
            # Fallback for rotation
            try:
                if hasattr(car, 'rotation_mtx'):
                    car.rotation_mtx = np.eye(3)
            except:
                pass

# --- Add a new TouchBallCondition class ---
class TouchBallCondition(DoneCondition[AgentID, GameState]):
    """Termination condition that ends an episode when a player touches the ball."""

    def __init__(self):
        """Initialize the TouchBallCondition."""
        super().__init__()
        self.last_touch_time = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        """Reset the touch tracking for all agents, safely handling missing last_touch."""
        # Safely check if initial_state has last_touch and if it's not None
        last_touch_time = 0.0
        if hasattr(initial_state, 'last_touch') and initial_state.last_touch:
            if hasattr(initial_state.last_touch, 'time_seconds'):
                 last_touch_time = initial_state.last_touch.time_seconds

        self.last_touch_time = {agent: last_touch_time for agent in agents}


    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        """Return whether each agent has touched the ball since the last reset."""
        is_done = {}

        # Safely check if state has last_touch and if it's not None
        current_touch_time = 0.0
        touching_player_index = None
        if hasattr(state, 'last_touch') and state.last_touch:
             if hasattr(state.last_touch, 'time_seconds'):
                 current_touch_time = state.last_touch.time_seconds
             if hasattr(state.last_touch, 'player_index'):
                 touching_player_index = state.last_touch.player_index

        for agent in agents:
            # Initialize to the reset time if this agent hasn't been seen before
            if agent not in self.last_touch_time:
                self.last_touch_time[agent] = self.last_touch_time.get(agent, 0.0) # Use the initial reset time

            # Check if there is a touch and it's more recent than what we've stored
            if current_touch_time > self.last_touch_time[agent]:
                # Check if this agent was the one who touched the ball
                if touching_player_index == agent:
                    is_done[agent] = True
                    # Update last touch time for this agent upon detecting their touch
                    self.last_touch_time[agent] = current_touch_time
                    continue

            # No new touch detected for this agent
            is_done[agent] = False

        return is_done