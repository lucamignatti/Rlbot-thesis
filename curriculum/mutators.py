"""Game state mutators for curriculum learning."""
import numpy as np
from typing import Optional, Callable, Dict, Any
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState

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
    """Sets a car's position using a callback function"""
    def __init__(self, car_id: str, position_function: Optional[Callable[[], np.ndarray]] = None):
        self.car_id = car_id
        self.position_function = position_function

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        car_id_str = str(self.car_id)

        # Check if the car exists in the state
        if car_id_str not in state.cars:
            # Car doesn't exist yet, there might be nothing we can do at this point
            # We could log this for debugging but it's not necessarily an error
            return
        
        # Get the car from the state
        car = state.cars[car_id_str]
        
        # Generate the position using the callback function
        try:
            position = self.position_function()
            if position is None or not isinstance(position, (np.ndarray, list, tuple)):
                # raise an error if position is None or not iterable
                raise
        except Exception as e:
            # If the position function throws an error, use the default position
            print(f"Error in position_function: {e}, using default position")
        
        # Ensure position is always a numpy array
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        
        # Set the car position - handle both physics and direct position attributes
        if hasattr(car, 'physics'):
            # Handle RLGym's physics attribute structure
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
            # Direct position setting (for some implementations)
            car.position = position
        
        # Reset rotation to default
        try:
            if hasattr(car, 'physics') and car.physics is not None:
                car.physics.rotation = np.zeros(3)
            else:
                car.rotation = np.zeros(3)
        except (AttributeError, TypeError):
            # If rotation setting fails, try to get it working with different attributes
            try:
                if hasattr(car, 'rotation_mtx'):
                    # Some implementations use rotation matrix
                    car.rotation_mtx = np.eye(3)
            except:
                pass

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