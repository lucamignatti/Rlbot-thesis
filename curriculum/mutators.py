"""Game state mutators for curriculum learning."""
from typing import Optional, Callable, Dict, Any, List
import numpy as np
from rlgym.api import StateMutator, DoneCondition, AgentID
from rlgym.rocket_league.api import GameState, PhysicsObject # Import PhysicsObject

# --- Helper Functions for Orientation ---
def euler_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles (pitch, yaw, roll) to a quaternion (w, x, y, z).
    Uses the ZYX convention: first roll around X, then pitch around Y, then yaw around Z.
    """
    pitch, yaw, roll = rotation
    
    # Half angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Quaternion computation
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

def euler_to_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles (pitch, yaw, roll) to a 3x3 rotation matrix.
    Uses the ZYX convention: first roll around X, then pitch around Y, then yaw around Z.
    """
    pitch, yaw, roll = rotation
    
    # Roll (X-axis rotation)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (Y-axis rotation)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (Z-axis rotation)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R

def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to Euler angles (pitch, yaw, roll).
    Uses the ZYX convention.
    """
    w, x, y, z = quaternion
    
    # Roll (X-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (Y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    
    # Yaw (Z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([pitch, yaw, roll])

def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
    """
    w, x, y, z = quaternion
    
    # Precompute common terms
    x2, y2, z2 = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    # Build rotation matrix
    R = np.array([
        [1 - 2*(y2 + z2), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(x2 + z2), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(x2 + y2)]
    ])
    
    return R

def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length."""
    norm = np.linalg.norm(quaternion)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Default to identity quaternion
    return quaternion / norm

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).
    Uses Sarabandi & Thomas algorithm for robust conversion.
    """
    # Ensure proper 3x3 rotation matrix
    if R.shape != (3, 3):
        raise ValueError("Expected 3x3 rotation matrix")
        
    # Get trace
    trace = np.trace(R)
    
    if trace > 0:
        S = 2 * np.sqrt(trace + 1.0)
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = 2 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = 2 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = 2 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
            
    return normalize_quaternion(np.array([w, x, y, z]))

def get_forward_vector(rotation: np.ndarray) -> np.ndarray:
    """Get forward vector from Euler angles (pitch, yaw, roll)."""
    R = euler_to_rotation_matrix(rotation)
    return R[:, 0]  # First column is forward vector

def get_up_vector(rotation: np.ndarray) -> np.ndarray:
    """Get up vector from Euler angles (pitch, yaw, roll)."""
    R = euler_to_rotation_matrix(rotation)
    return R[:, 2]  # Third column is up vector

def get_right_vector(rotation: np.ndarray) -> np.ndarray:
    """Get right vector from Euler angles (pitch, yaw, roll)."""
    R = euler_to_rotation_matrix(rotation)
    return R[:, 1]  # Second column is right vector

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
    """Sets a car's position and orientation using callback functions"""
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
            if self.debug:
                print(f"[DEBUG CarPositionMutator] Car {car_id_str} not found in state")
            return
        
        car = state.cars[car_id_str]
        
        # --- Position Setting ---
        try:
            position = self.position_function() if self.position_function else np.array([0.0, -2000.0, 17.0])
            if self.debug:
                print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Generated Position: {position}")
            
            if not isinstance(position, np.ndarray):
                position = np.array(position)
            
            # Ensure physics object exists with proper type
            if not hasattr(car, 'physics') or car.physics is None:
                car.physics = PhysicsObject()
                
            # Apply position and initialize velocities
            car.physics.position = position
            car.physics.linear_velocity = np.zeros(3)
            car.physics.angular_velocity = np.zeros(3)
            
        except Exception as e:
            if self.debug:
                print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Position error: {str(e)}")
            # Use safe default position
            car.physics.position = np.array([0.0, -2000.0, 17.0])
        
        # --- Orientation Setting ---
        try:
            if self.orientation_function:
                euler_rotation = self.orientation_function()
                if self.debug:
                    print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Euler rotation: {euler_rotation}")
                
                if not isinstance(euler_rotation, np.ndarray):
                    euler_rotation = np.array(euler_rotation)
                
                # Convert Euler angles to quaternion
                quaternion = euler_to_quaternion(euler_rotation)
                if self.debug:
                    print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Quaternion: {quaternion}")
                
                # Convert quaternion to rotation matrix
                rotation_matrix = quaternion_to_rotation_matrix(quaternion)
                if self.debug:
                    print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Rotation matrix: {rotation_matrix}")
                
                # Apply rotation matrix to car physics (using rotation_mtx instead of rotation)
                car.physics.rotation_mtx = rotation_matrix  # Apply rotation matrix
                # Store quaternion if the physics object supports it
                if hasattr(car.physics, 'quaternion'):
                    car.physics.quaternion = quaternion
                
                # Verify rotation was applied
                if self.debug:
                    if hasattr(car.physics, 'rotation_mtx'):
                        print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Applied rotation matrix: {car.physics.rotation_mtx}")
                    else:
                        print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Warning: rotation_mtx not set")
            else:
                # Default to neutral orientation
                car.physics.rotation_mtx = np.eye(3)
                if hasattr(car.physics, 'quaternion'):
                    car.physics.quaternion = np.array([1., 0., 0., 0.])
                
        except Exception as e:
            if self.debug:
                print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Orientation error: {str(e)}")
            # Use safe default orientation
            car.physics.rotation_mtx = np.eye(3)
            if hasattr(car.physics, 'quaternion'):
                car.physics.quaternion = np.array([1., 0., 0., 0.])

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