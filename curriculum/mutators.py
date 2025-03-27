"""Game state mutators for curriculum learning."""
import numpy as np
from typing import Optional, Callable, Dict, Any
from rlgym.api import StateMutator
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
                 orientation_function: Optional[Callable[[], np.ndarray]] = None):
        self.car_id = car_id
        self.position_function = position_function
        self.orientation_function = orientation_function

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        car_id_str = str(self.car_id)
        if car_id_str not in state.cars:
            return
        
        car = state.cars[car_id_str]
        
        # --- Position Setting (Added Debugging) ---
        position = None # Initialize position
        try:
            position = self.position_function()
            print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Generated Position: {position}") # Debug print
            if position is None or not isinstance(position, (np.ndarray, list, tuple)):
                raise ValueError("Position function returned invalid position")
        except Exception as e:
            print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Error in position_function: {e}, using default position")
            position = np.array([0.0, -2000.0, 17.0]) # Default position
        
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        
        # Ensure physics object exists using the correct type
        if not hasattr(car, 'physics') or car.physics is None:
             car.physics = PhysicsObject() # Use the imported PhysicsObject
             print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Created PhysicsObject instance")
             # Initialize rotation attribute if it doesn't exist after creation
             if not hasattr(car.physics, 'rotation'):
                 car.physics.rotation = np.zeros(3)
                 print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Initialized physics.rotation")

        # Apply position
        try:
            car.physics.position = position
            print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Applied position: {car.physics.position}") # Debug print
        except Exception as e:
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
                print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Generated Euler Rotation: {rotation_euler}")
                if rotation_euler is None or not isinstance(rotation_euler, (np.ndarray, list, tuple)):
                     raise ValueError("Orientation function returned invalid rotation")
                if not isinstance(rotation_euler, np.ndarray):
                    rotation_euler = np.array(rotation_euler)
                
                # Attach Euler angles directly to the car object being modified
                try:
                    car._temp_euler_rotation = rotation_euler
                    print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Attached _temp_euler_rotation = {car._temp_euler_rotation}")
                except Exception as e_set_temp:
                     print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Error attaching _temp_euler_rotation: {repr(e_set_temp)}")

            except Exception as e:
                print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Error processing orientation function: {repr(e)}")
                # Attempt to set default neutral orientation if error occurs
                try:
                    car._temp_euler_rotation = np.zeros(3)
                    print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Attached default Euler rotation due to error.")
                except Exception as fallback_e:
                     print(f"[DEBUG CarPositionMutator] Car: {self.car_id}, Error attaching default Euler rotation: {repr(fallback_e)}")
        else:
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