from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
import RocketSim as rsim
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rewards import BallProximityReward, BallToGoalDistanceReward, BallVelocityToGoalReward, TouchBallReward, TouchBallToGoalAccelerationReward, AlignBallToGoalReward, PlayerVelocityTowardBallReward, KRCReward
from observation import StackedActionsObs
import numpy as np

# --- Helper Function for Orientation ---
def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Converts a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    x, y, z, w = quat
    
    # Precompute squares
    x2, y2, z2 = x*x, y*y, z*z
    
    # Precompute products
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # Rotation matrix elements
    m00 = 1.0 - 2.0 * (y2 + z2)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (x2 + z2)
    m12 = 2.0 * (yz - wx)
    
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (x2 + y2)
    
    # Return as 3x3 numpy array (transposed for RocketSim's expected format if needed)
    # RocketSim expects RotMat(m00, m01, m02, m10, m11, m12, m20, m21, m22)
    # which corresponds to row-major flattened matrix. Numpy creates row-major by default.
    return np.array([
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22]
    ])

def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """Converts Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix."""
    roll, pitch, yaw = euler
    
    # Precompute cosines and sines of Euler angles
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # Rotation matrix elements
    m00 = cy * cp
    m01 = cy * sp * sr - sy * cr
    m02 = cy * sp * cr + sy * sr
    
    m10 = sy * cp
    m11 = sy * sp * sr + cy * cr
    m12 = sy * sp * cr - cy * sr
    
    m20 = -sp
    m21 = cp * sr
    m22 = cp * cr
    
    return np.array([
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22]
    ])
# --- End Helper Function ---

# Patched RocketSim engine to handle None car positions and prioritize quat
class PatchedRocketSimEngine(RocketSimEngine):
    """A patched version of RocketSimEngine that ensures car positions are never None"""
    
    def __init__(self, debug=False):
        # The parent class no longer accepts rlbot_delay parameter
        super().__init__()
        self.debug = debug
    
    def _set_car_state(self, car: rsim.Car, desired_car: any):
        """
        Sets the state of a RocketSim car to match a given car state.
        Overrides the original method to ensure position is never None and prioritizes quaternion.
        """
        try:
            # --- Ensure basic physics attributes exist and are valid ---
            if not hasattr(desired_car, 'physics') or desired_car.physics is None:
                from rlgym.rocket_league.api import PhysicsObject
                desired_car.physics = PhysicsObject()
            
            if not hasattr(desired_car.physics, 'position') or desired_car.physics.position is None:
                desired_car.physics.position = np.array([0, 0, 17])
            if not hasattr(desired_car.physics, 'linear_velocity') or desired_car.physics.linear_velocity is None:
                desired_car.physics.linear_velocity = np.zeros(3)
            if not hasattr(desired_car.physics, 'angular_velocity') or desired_car.physics.angular_velocity is None:
                desired_car.physics.angular_velocity = np.zeros(3)
            # --- End basic physics validation ---
            # Create a new car state
            car_state = rsim.CarState()
            
            # --- Set Position (Added Debugging) ---
            position_to_set = np.array([0, 0, 17]) # Default
            if hasattr(desired_car.physics, 'position') and desired_car.physics.position is not None:
                try:
                    # Directly use the position if it's already a numpy array
                    if isinstance(desired_car.physics.position, np.ndarray):
                        pos_read = desired_car.physics.position
                    else:
                        # Convert if it's a list/tuple
                        pos_read = np.array(desired_car.physics.position)
                    
                    if pos_read.shape == (3,) and not np.isnan(pos_read).any():
                        position_to_set = pos_read
                        if self.debug:
                            print(f"[DEBUG _set_car_state] Read Position: {position_to_set}")
                    else:
                        if self.debug:
                            print(f"[DEBUG _set_car_state] Invalid position read: {pos_read}, using default.")
                except Exception as e_pos:
                    if self.debug:
                        print(f"[DEBUG _set_car_state] Error reading position: {e_pos}, using default.")
            else:
                if self.debug:
                    print(f"[DEBUG _set_car_state] No position attribute found, using default.")
            
            car_state.pos = rsim.Vec(*position_to_set)
            # --- End Position Setting ---

            # Set core physics vectors (velocity)
            vel_to_set = np.zeros(3)
            if hasattr(desired_car.physics, 'linear_velocity') and desired_car.physics.linear_velocity is not None:
                 vel_to_set = np.asarray(desired_car.physics.linear_velocity)
            car_state.vel = rsim.Vec(*vel_to_set)
            
            ang_vel_to_set = np.zeros(3)
            if hasattr(desired_car.physics, 'angular_velocity') and desired_car.physics.angular_velocity is not None:
                 ang_vel_to_set = np.asarray(desired_car.physics.angular_velocity)
            car_state.ang_vel = rsim.Vec(*ang_vel_to_set)
            
            # --- Set Rotation (Read Euler from desired_car._temp_euler_rotation) ---
            rotation_matrix_to_set = None
            read_euler = None # Debug variable
            try:
                # Check if temporary Euler rotation exists and is valid
                if hasattr(desired_car, '_temp_euler_rotation') and desired_car._temp_euler_rotation is not None:
                    # Read the temporary Euler angles
                    euler_angles = desired_car._temp_euler_rotation 
                    read_euler = euler_angles # Store for debugging
                    if isinstance(euler_angles, (np.ndarray, list, tuple)) and len(euler_angles) == 3:
                        euler_angles_np = np.asarray(euler_angles)
                        if not np.isnan(euler_angles_np).any():
                            # Convert valid Euler angles to rotation matrix
                            rotation_matrix_to_set = euler_to_rotation_matrix(euler_angles_np)
                            if self.debug:
                                print(f"[DEBUG _set_car_state] Used _temp_euler_rotation {euler_angles_np} -> rot_mat")
                        else:
                            if self.debug:
                                print(f"[DEBUG _set_car_state] Invalid _temp_euler_rotation read (contains NaN): {euler_angles_np}")
                    else:
                        if self.debug:
                            print(f"[DEBUG _set_car_state] Invalid _temp_euler_rotation read (wrong type/shape): {euler_angles}")
            except Exception as e:
                # Print the actual exception
                if self.debug:
                    print(f"[DEBUG _set_car_state] Error processing _temp_euler_rotation: {repr(e)}") 
                rotation_matrix_to_set = None # Fallback if conversion fails

            # If still no valid rotation, use identity matrix
            if rotation_matrix_to_set is None:
                rotation_matrix_to_set = np.eye(3)
                if self.debug:
                    print("[DEBUG _set_car_state] Used identity matrix") # Debug print

            # --- More Debugging ---
            if self.debug:
                print(f"[DEBUG _set_car_state] Read Euler value (from _temp_euler_rotation): {read_euler}")
                print(f"[DEBUG _set_car_state] Final rotation_matrix_to_set:\n{rotation_matrix_to_set}")
            # --- End More Debugging ---

            # Set the rotation matrix in RocketSim format (flattened row-major)
            rot_elements = rotation_matrix_to_set.flatten()
            if self.debug:
                print(f"[DEBUG _set_car_state] Flattened rot_elements: {rot_elements}") # Debug print
            car_state.rot_mat = rsim.RotMat(*rot_elements)
            # --- End Rotation Setting ---
            
            # --- Set Other Car State Attributes (Simplified) ---
            # Use getattr with defaults to avoid errors if attributes are missing
            car_state.boost = float(getattr(desired_car, 'boost_amount', 33.3))
            car_state.is_demoed = bool(getattr(desired_car, 'is_demoed', False))
            car_state.demo_respawn_timer = float(getattr(desired_car, 'demo_respawn_timer', 0.0))
            
            # Jump/Flip related attributes (ensure boolean/float types)
            car_state.has_jumped = bool(getattr(desired_car, 'has_jumped', False))
            car_state.has_double_jumped = bool(getattr(desired_car, 'has_double_jumped', False))
            car_state.has_flipped = bool(getattr(desired_car, 'has_flipped', False))
            car_state.is_jumping = bool(getattr(desired_car, 'is_jumping', False))
            car_state.is_flipping = bool(getattr(desired_car, 'is_flipping', False))
            car_state.air_time_since_jump = float(getattr(desired_car, 'air_time_since_jump', 0.0))
            car_state.jump_time = float(getattr(desired_car, 'jump_time', 0.0))
            car_state.flip_time = float(getattr(desired_car, 'flip_time', 0.0))
            
            # Flip torque (needs to be Vec)
            flip_torque_val = getattr(desired_car, 'flip_torque', np.zeros(3))
            if flip_torque_val is None: flip_torque_val = np.zeros(3)
            car_state.flip_rel_torque = rsim.Vec(*np.asarray(flip_torque_val))
            
            # Supersonic state
            car_state.is_supersonic = bool(getattr(desired_car, 'is_supersonic', False))
            car_state.supersonic_time = float(getattr(desired_car, 'supersonic_time', 0.0))
            
            # Handbrake
            car_state.handbrake_val = float(getattr(desired_car, 'handbrake', 0.0)) # RocketSim uses float for handbrake value

            # Last controls (optional, create default if missing)
            if hasattr(car_state, 'last_controls'):
                controls = rsim.CarControls()
                controls.throttle = float(getattr(desired_car, 'throttle', 0.0))
                controls.steer = float(getattr(desired_car, 'steer', 0.0))
                controls.pitch = float(getattr(desired_car, 'pitch', 0.0))
                controls.yaw = float(getattr(desired_car, 'yaw', 0.0))
                controls.roll = float(getattr(desired_car, 'roll', 0.0))
                controls.jump = bool(getattr(desired_car, 'jump_active', False))
                controls.boost = bool(getattr(desired_car, 'boost_active', False))
                controls.handbrake = bool(getattr(desired_car, 'handbrake_active', False))
                car_state.last_controls = controls
            # --- End Other Attributes ---

            # Set the final state
            car.set_state(car_state)
            if self.debug:
                print(f"[DEBUG _set_car_state] car.set_state called successfully") # Debug print

        except Exception as e:
            print(f"Fatal error in patched _set_car_state: {str(e)}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            # As a last resort, try setting a minimal default state
            try:
                minimal_state = rsim.CarState()
                minimal_state.pos = rsim.Vec(0, 0, 17)
                minimal_state.vel = rsim.Vec(0, 0, 0)
                minimal_state.ang_vel = rsim.Vec(0, 0, 0)
                minimal_state.rot_mat = rsim.RotMat(1, 0, 0, 0, 1, 0, 0, 0, 1)
                minimal_state.boost = 33.0
                car.set_state(minimal_state)
            except Exception as final_e:
                 print(f"Failed to set even minimal state: {final_e}")
                 raise e # Re-raise original error if fallback fails

    def get_state(self):
        """
        Public wrapper for _get_state to maintain API compatibility
        """
        if hasattr(self, '_get_state'):
            return self._get_state()
        else:
            # Fallback if parent class doesn't have _get_state
            return None


def get_env(renderer=None, action_stacker=None, curriculum_config=None, debug=False):
    """
    Sets up the Rocket League environment with curriculum support.
    """
    # Use curriculum configuration if provided
    if curriculum_config is not None:
        if debug:
            print(f"[DEBUG] Creating environment with curriculum config: {curriculum_config['stage_name']}")
            
            # Debug the state mutator to check for car position initialization
            state_mutator = curriculum_config["state_mutator"]
            if hasattr(state_mutator, 'mutators'):
                print(f"[DEBUG] State mutator is a MutatorSequence with {len(state_mutator.mutators)} mutators:")
                for i, mutator in enumerate(state_mutator.mutators):
                    print(f"[DEBUG]   Mutator {i}: {mutator.__class__.__name__}")
                    if 'CarPositionMutator' in mutator.__class__.__name__:
                        print(f"[DEBUG]     Found CarPositionMutator in position {i}")
            else:
                print(f"[DEBUG] State mutator is a single mutator: {state_mutator.__class__.__name__}")
        
        # Create the environment with the provided configuration
        env = RLGym(
            state_mutator=curriculum_config["state_mutator"],
            obs_builder=StackedActionsObs(action_stacker, zero_padding=2),
            action_parser=RepeatAction(LookupTableAction(), repeats=8),
            reward_fn=curriculum_config["reward_function"],
            termination_cond=curriculum_config["termination_condition"],
            truncation_cond=curriculum_config["truncation_condition"],
            # Use the patched engine to handle car position issues
            transition_engine=PatchedRocketSimEngine(debug=debug),
            renderer=renderer
        )
        
        # Add debugging during environment reset/step if needed
        if debug:
            orig_reset = env.reset
            def debug_reset():
                try:
                    print(f"[DEBUG] Resetting environment with stage: {curriculum_config['stage_name']}")
                    result = orig_reset()
                    print(f"[DEBUG] Reset successful for {curriculum_config['stage_name']}")
                    return result
                except Exception as e:
                    print(f"[DEBUG] Reset failed for {curriculum_config['stage_name']}: {str(e)}")
                    if "Vec() argument after * must be an iterable, not NoneType" in str(e):
                        # Inspect the car physics problem
                        print("[DEBUG] Car position is None! This is likely from missing CarPositionMutator")
                    raise
            
            env.reset = debug_reset
        
        return env

    # Otherwise use the default configuration
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            KickoffMutator()
        ),
        obs_builder=StackedActionsObs(action_stacker, zero_padding=2),
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=CombinedReward(
            # Primary objective rewards - slightly increased to emphasize scoring
            (GoalReward(), 15.0),

            # Offensive Potential KRC group
            (KRCReward([
                (AlignBallToGoalReward(dispersion=1.1, density=1.0), 1.0),
                (BallProximityReward(dispersion=0.8, density=1.2), 0.8),
                (PlayerVelocityTowardBallReward(), 0.6)
            ], team_spirit=0.3), 8.0),

            # Ball Control KRC group
            (KRCReward([
                (TouchBallToGoalAccelerationReward(), 1.0),
                (TouchBallReward(), 0.8),
                (BallVelocityToGoalReward(), 0.6)
            ], team_spirit=0.3), 6.0),

            # Distance-weighted Alignment KRC group
            (KRCReward([
                (AlignBallToGoalReward(dispersion=1.1, density=1.0), 1.0),
                (BallProximityReward(dispersion=0.8, density=1.2), 0.8)
            ], team_spirit=0.3), 4.0),

            # Strategic Ball Positioning
            (KRCReward([
                (BallToGoalDistanceReward(
                    offensive_dispersion=0.6,
                    defensive_dispersion=0.4,
                    offensive_density=1.0,
                    defensive_density=1.0
                ), 1.0),
                (BallProximityReward(dispersion=0.7, density=1.0), 0.4)
            ], team_spirit=0.3), 2.0),
        ),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            TimeoutCondition(300.),
            NoTouchTimeoutCondition(30.)
        ),
        # Use the patched engine for default environment too
        transition_engine=PatchedRocketSimEngine(debug=debug),
        renderer=renderer
    )
