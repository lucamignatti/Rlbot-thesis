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

# Patched RocketSim engine to handle None car positions
class PatchedRocketSimEngine(RocketSimEngine):
    """A patched version of RocketSimEngine that ensures car positions are never None"""
    
    def __init__(self, rlbot_delay=True):
        super().__init__(rlbot_delay=rlbot_delay)
    
    def _set_car_state(self, car: rsim.Car, desired_car: any):
        """
        Sets the state of a RocketSim car to match a given car state.
        Overrides the original method to ensure position is never None.
        """
        try:
            # Simplified approach - use the original implementation but catch and handle exceptions
            if hasattr(desired_car, 'physics') and desired_car.physics is not None:
                if not hasattr(desired_car.physics, 'position') or desired_car.physics.position is None:
                    desired_car.physics.position = np.array([0, 0, 17])
                
                if not hasattr(desired_car.physics, 'linear_velocity') or desired_car.physics.linear_velocity is None:
                    desired_car.physics.linear_velocity = np.zeros(3)
                
                if not hasattr(desired_car.physics, 'angular_velocity') or desired_car.physics.angular_velocity is None:
                    desired_car.physics.angular_velocity = np.zeros(3)
                
                # Try to handle rotation_mtx, but it might throw
                try:
                    if not hasattr(desired_car.physics, 'rotation_mtx') or desired_car.physics.rotation_mtx is None:
                        # Create a 3x3 identity matrix
                        desired_car.physics.rotation_mtx = np.eye(3)
                except (ValueError, AttributeError, TypeError):
                    pass # Will be handled by fallback
            else:
                # Create minimal physics
                from rlgym.rocket_league.api import Car, PhysicsObject
                if isinstance(desired_car, Car):
                    desired_car.physics = PhysicsObject()
                    desired_car.physics.position = np.array([0, 0, 17])
                    desired_car.physics.linear_velocity = np.zeros(3)
                    desired_car.physics.angular_velocity = np.zeros(3)
                    desired_car.physics.rotation_mtx = np.eye(3)
            
            # Now safely create a car state with proper types
            try:
                # Let's avoid using the super implementation directly and instead
                # create a completely new car state with proper sequence types
                
                # Get a brand new car state
                car_state = rsim.CarState()
                
                # Core physics with vectors (these are expected to be Vec objects)
                car_state.pos = rsim.Vec(*desired_car.physics.position)
                car_state.vel = rsim.Vec(*desired_car.physics.linear_velocity)
                car_state.ang_vel = rsim.Vec(*desired_car.physics.angular_velocity)
                
                # Identity matrix as fallback for rotation
                identity_rot = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
                
                try:
                    if hasattr(desired_car.physics, 'rotation_mtx') and desired_car.physics.rotation_mtx is not None:
                        rot = desired_car.physics.rotation_mtx.transpose().flatten()
                        car_state.rot_mat = rsim.RotMat(*rot)
                    else:
                        car_state.rot_mat = rsim.RotMat(*identity_rot)
                except (ValueError, AttributeError, TypeError):
                    car_state.rot_mat = rsim.RotMat(*identity_rot)
                
                # Use sequence types for attributes that require them
                # For example wheels_with_contact might expect a list/sequence
                try:
                    # For attributes expecting sequences, convert to list/array
                    wheels_contact = getattr(desired_car, 'wheels_with_contact', [0])
                    
                    # Ensure it's a sequence type
                    if not isinstance(wheels_contact, (list, tuple, np.ndarray)):
                        wheels_contact = [wheels_contact]
                        
                    # Some attributes in newer RocketSim might need to be set as arrays
                    setattr(car_state, 'wheels_with_contact', wheels_contact)
                    
                    # The following are scalar properties - convert if needed
                    car_state.boost = float(getattr(desired_car, 'boost_amount', 33))
                    car_state.demo_respawn_timer = float(getattr(desired_car, 'demo_respawn_timer', 0))
                    car_state.is_demoed = bool(getattr(desired_car, 'is_demoed', False))
                    car_state.supersonic_time = float(getattr(desired_car, 'supersonic_time', 0))
                    car_state.time_spent_boosting = float(getattr(desired_car, 'boost_active_time', 0))
                    car_state.handbrake_val = bool(getattr(desired_car, 'handbrake', False))
                    
                    # Set jump state
                    car_state.has_jumped = bool(getattr(desired_car, 'has_jumped', False))
                    car_state.is_jumping = bool(getattr(desired_car, 'is_jumping', False))
                    car_state.jump_time = float(getattr(desired_car, 'jump_time', 0))
                    
                    # Set flip state
                    car_state.has_flipped = bool(getattr(desired_car, 'has_flipped', False))
                    car_state.is_flipping = bool(getattr(desired_car, 'is_flipping', False))
                    car_state.has_double_jumped = bool(getattr(desired_car, 'has_double_jumped', False))
                    car_state.air_time_since_jump = float(getattr(desired_car, 'air_time_since_jump', 0))
                    car_state.flip_time = float(getattr(desired_car, 'flip_time', 0))
                    
                    # Try to set flip_rel_torque
                    if hasattr(desired_car, 'flip_torque') and desired_car.flip_torque is not None:
                        car_state.flip_rel_torque = rsim.Vec(*desired_car.flip_torque)
                    else:
                        car_state.flip_rel_torque = rsim.Vec(0, 0, 0)
                    
                    # Set auto-flip state
                    car_state.is_auto_flipping = bool(getattr(desired_car, 'is_autoflipping', False))
                    car_state.auto_flip_timer = float(getattr(desired_car, 'autoflip_timer', 0))
                    car_state.auto_flip_torque_scale = float(getattr(desired_car, 'autoflip_direction', 0))
                    
                    # Try to set last controls
                    if hasattr(car_state, 'last_controls'):
                        controls = rsim.CarControls()
                        controls.throttle = 0
                        controls.steer = 0
                        controls.pitch = 0
                        controls.yaw = 0
                        controls.roll = 0
                        controls.jump = bool(getattr(desired_car, 'is_holding_jump', False))
                        controls.boost = False
                        controls.handbrake = False
                        car_state.last_controls = controls
                    
                    # Set the car state
                    car.set_state(car_state)
                    return
                    
                except TypeError as e:
                    # This is likely due to a type mismatch - possibly a sequence vs scalar issue
                    pass
                    # We'll fall through to a different approach
                
            except Exception as e:
                pass
                # Continue to next fallback
            
            # Try the simplest possible car state - only the essential physics
            try:
                minimal_state = rsim.CarState()
                minimal_state.pos = rsim.Vec(*desired_car.physics.position)
                minimal_state.vel = rsim.Vec(*desired_car.physics.linear_velocity)
                minimal_state.ang_vel = rsim.Vec(*desired_car.physics.angular_velocity)
                minimal_state.rot_mat = rsim.RotMat(*[1, 0, 0, 0, 1, 0, 0, 0, 1])
                minimal_state.boost = 33.0
                
                # Apply only the essential state
                car.set_state(minimal_state)
                return
            except Exception as e:
                pass
            
            # If all else fails, recreate the car from scratch
            try:
                # We reached here because all other approaches failed
                # Last resort: try to remove and recreate the car
                team_num = car.team
                car_id = car.id
                
                # Try to recreate the car with proper position
                config = rsim.CarConfig()
                car_new = self._arena.add_car(team_num, config)
                
                # Set just position and rotation, nothing else
                minimal_state = rsim.CarState()
                minimal_state.pos = rsim.Vec(*desired_car.physics.position)
                minimal_state.rot_mat = rsim.RotMat(*[1, 0, 0, 0, 1, 0, 0, 0, 1])
                car_new.set_state(minimal_state)
                
                return
            except Exception as e:
                # At this point we're out of options
                raise
                
        except Exception as e:
            print(f"Fatal error in patched _set_car_state: {str(e)}")
            raise
    
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
            transition_engine=PatchedRocketSimEngine(),
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
        transition_engine=PatchedRocketSimEngine(),
        renderer=renderer
    )
