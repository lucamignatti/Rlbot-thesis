from .base import CurriculumManager, ProgressionRequirements
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator
from .mutators import (
    BallTowardGoalSpawnMutator, BallPositionMutator,
    CarBoostMutator, BallVelocityMutator, CarPositionMutator
)
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
import numpy as np
import random
from curriculum.rlbot import RLBotSkillStage
from rlgym.rocket_league.state_mutators import KickoffMutator
from rewards import (
    TouchBallReward, PlayerVelocityTowardBallReward, create_offensive_potential_reward,
    BallVelocityToGoalReward, TouchBallToGoalAccelerationReward, SaveBoostReward,
    create_distance_weighted_alignment_reward, BallToGoalDistanceReward, create_lucy_skg_reward
)
from functools import partial

# Define position functions as regular functions instead of lambdas
def aerial_ball_position():
    """Get random aerial ball position"""
    x = np.random.uniform(-2000, 2000)
    y = np.random.uniform(-2000, 2000)
    z = np.random.uniform(300, 1200)
    return np.array([x, y, z])

def get_aerial_ball_position():
    """Get random aerial ball position"""
    return aerial_ball_position()

def get_wall_play_ball_position():
    """Get random wall play ball position"""
    # Randomly select a side wall
    if np.random.random() < 0.5:
        # Left wall
        x = -4096 + np.random.uniform(100, 300)  # Slightly off the wall
        y = np.random.uniform(-3000, 3000)
    else:
        # Right wall
        x = 4096 - np.random.uniform(100, 300)  # Slightly off the wall
        y = np.random.uniform(-3000, 3000)

    # Set height based on difficulty
    z = np.random.uniform(500, 1500)

    return np.array([x, y, z])

def get_fast_aerial_ball_position():
    """Get random fast aerial ball position"""
    x = np.random.uniform(-2000, 2000)
    y = np.random.uniform(-2000, 2000)
    z = np.random.uniform(800, 1800)
    return np.array([x, y, z])

def get_ground_dribbling_ball_position():
    """Get ground dribbling ball position"""
    x = np.random.uniform(-2000, 2000)
    y = np.random.uniform(-3000, 0)  # Start in blue half for dribbling toward orange
    z = 93  # Ball radius + small offset
    return np.array([x, y, z])

def get_ground_dribbling_ball_velocity():
    """Get slight forward momentum for dribbling"""
    speed = np.random.uniform(0, 500)
    return np.array([0, speed, 0])

def get_car_position(x=0, y=-3000, z=17):
    """Get car position with default near blue goal"""
    return np.array([x, y, z])

def get_car_position_aerial(x=0, y=-4000, z=17):
    """Get car position for aerial training (further back)"""
    return np.array([x, y, z])

def get_car_position_wall(x=0, y=-4000, z=17):
    """Get car position for wall training (center field)"""
    return np.array([x, y, z])

def create_position(x, y, z):
    return np.array([x, y, z])

def get_directional_shooting_car_position():
    return np.array([
        np.random.uniform(-500, 500),
        np.random.uniform(-3000, -2000),
        17
    ])

def get_aerial_ball_position():
    return np.array([
        np.random.uniform(-800, 800),
        np.random.uniform(-2000, -1000),
        np.random.uniform(300, 800)
    ])

def get_aerial_car_position():
    return np.array([
        np.random.uniform(-200, 200),
        np.random.uniform(-3500, -3000),
        17
    ])

def get_strategic_ball_position():
    return np.array([
        np.random.uniform(-2000, 2000),
        np.random.uniform(-2000, 2000),
        93
    ])

def get_strategic_car_position():
    return np.array([
        np.random.uniform(-1000, 1000),
        np.random.uniform(-4000, -3000),
        17
    ])

def get_advanced_aerial_ball_position():
    return np.array([
        np.random.uniform(-1500, 1500),
        np.random.uniform(-2000, 0),
        np.random.uniform(800, 1500)
    ])

def get_advanced_aerial_ball_velocity():
    return np.array([
        np.random.uniform(-300, 300),
        np.random.uniform(-300, 300),
        np.random.uniform(-100, 100)
    ])

def get_advanced_aerial_car_position():
    return np.array([
        np.random.uniform(-500, 500),
        np.random.uniform(-4000, -3000),
        17
    ])

def get_ground_ball_position():
    return np.array([0, 0, 93])

def create_lucy_skg_curriculum(team_goal_y=5120, debug=False, use_wandb=True):
    """Create a progressive curriculum inspired by the Lucy-SKG approach"""

    # Basic constants
    SKILL_TIMEOUT = 20  # seconds for skill training
    BASE_TIMEOUT = 300  # full match time
    
    # Common conditions
    goal_condition = GoalCondition()
    timeout_condition = TimeoutCondition(BASE_TIMEOUT)
    skill_timeout = TimeoutCondition(SKILL_TIMEOUT)
    no_touch_timeout = NoTouchTimeoutCondition(10)
    
    
    # ======== STAGE 1: BALL CONTROL FUNDAMENTALS ========
    # Focus on basic touch and control with KRC rewards
    ball_control = RLBotSkillStage(
        name="Ball Control Fundamentals",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(get_ground_ball_position),  # Ball on ground
            CarPositionMutator(car_id=0, position_function=partial(create_position, 100, -2000, 17))
        ),
        base_task_reward_function=CombinedReward(
            (TouchBallReward(weight=0.05), 1.0),
            (PlayerVelocityTowardBallReward(), 0.8),
            (create_offensive_potential_reward(team_goal_y), 1.0)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=skill_timeout,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.7,
            min_avg_reward=0.4,
            min_episodes=100,
            max_std_dev=1.5,
            required_consecutive_successes=3
        )
    )

    # ======== STAGE 2: DIRECTIONAL SHOOTING ========
    # Learn to hit the ball toward the goal using Distance-weighted Alignment KRC
    directional_shooting = RLBotSkillStage(
        name="Directional Shooting",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallTowardGoalSpawnMutator(offensive_team=0, distance_from_goal=0.6),
            CarPositionMutator(car_id=0, position_function=get_directional_shooting_car_position)
        ),
        base_task_reward_function=CombinedReward(
            (TouchBallReward(weight=0.05), 0.5),
            (BallVelocityToGoalReward(team_goal_y=team_goal_y, weight=0.8), 1.0),
            (TouchBallToGoalAccelerationReward(team_goal_y=team_goal_y, weight=0.25), 1.5),
            (create_distance_weighted_alignment_reward(team_goal_y), 2.0)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=skill_timeout,
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.5,
            min_episodes=150,
            max_std_dev=1.8,
            required_consecutive_successes=3
        )
    )

    # ======== STAGE 3: AERIAL FUNDAMENTALS ========
    # Basic aerial control using both offensive potential and alignment
    aerial_fundamentals = RLBotSkillStage(
        name="Aerial Fundamentals",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(get_aerial_ball_position),
            CarBoostMutator(boost_amount=100),
            CarPositionMutator(car_id=0, position_function=get_aerial_car_position)
        ),
        base_task_reward_function=CombinedReward(
            (TouchBallReward(weight=0.05), 1.0),
            (BallVelocityToGoalReward(team_goal_y=team_goal_y, weight=0.8), 1.0),
            (SaveBoostReward(weight=0.5), 0.5),
            (create_offensive_potential_reward(team_goal_y), 1.5)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=TimeoutCondition(15),
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5,
            min_avg_reward=0.4,
            min_episodes=200,
            max_std_dev=2.0,
            required_consecutive_successes=2
        )
    )

    # ======== STAGE 4: STRATEGIC POSITIONING ========
    # Focus on ball-to-goal distance and positioning with KRCs
    strategic_positioning = RLBotSkillStage(
        name="Strategic Positioning",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),  # Add opponent
            BallPositionMutator(get_strategic_ball_position),
            CarPositionMutator(car_id=0, position_function=get_strategic_car_position)
        ),
        base_task_reward_function=CombinedReward(
            (BallToGoalDistanceReward(
                team_goal_y=team_goal_y,
                offensive_dispersion=0.6, 
                defensive_dispersion=0.4,
                offensive_density=1.0,
                defensive_density=1.0
            ), 2.0),
            (SaveBoostReward(weight=0.5), 0.5),
            (create_distance_weighted_alignment_reward(team_goal_y), 1.0),
            (create_offensive_potential_reward(team_goal_y), 1.0)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=timeout_condition,
        bot_skill_ranges={(0.0, 0.3): 0.7, (0.3, 0.6): 0.3},  # Easy opponents
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.5,
            min_episodes=200,
            max_std_dev=2.0,
            required_consecutive_successes=3
        )
    )

    # ======== STAGE 5: ADVANCED AERIALS ========
    # Complex aerial control with full reward system
    advanced_aerials = RLBotSkillStage(
        name="Advanced Aerials",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallPositionMutator(get_advanced_aerial_ball_position),
            BallVelocityMutator(get_advanced_aerial_ball_velocity),
            CarBoostMutator(boost_amount=100),
            CarPositionMutator(car_id=0, position_function=get_advanced_aerial_car_position)
        ),
        base_task_reward_function=CombinedReward(
            (TouchBallReward(weight=0.05), 0.5),
            (BallVelocityToGoalReward(team_goal_y=team_goal_y, weight=0.8), 1.5),
            (TouchBallToGoalAccelerationReward(team_goal_y=team_goal_y, weight=0.25), 2.0),
            (SaveBoostReward(weight=0.5), 0.3),
            (create_offensive_potential_reward(team_goal_y), 1.0)
        ),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=TimeoutCondition(18),
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.4,
            min_avg_reward=0.3,
            min_episodes=250, 
            max_std_dev=2.5,
            required_consecutive_successes=2
        )
    )

    # ======== STAGE 6: COMPETITIVE PLAY ========
    # Full game with opponents using the complete Lucy-SKG reward structure
    competitive_play = RLBotSkillStage(
        name="Competitive Play",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator()  # Use standard kickoffs for proper matches
        ),
        base_task_reward_function=create_lucy_skg_reward(team_goal_y),
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=timeout_condition,
        bot_skill_ranges={(0.3, 0.7): 0.6, (0.7, 1.0): 0.4},  # Moderate to challenging opponents
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.4,
            min_episodes=300,
            max_std_dev=2.0,
            required_consecutive_successes=3
        )
    )

    # ======== STAGE 7: TEAM PLAY ========
    # 2v2 play with team spirit and full Lucy-SKG rewards
    team_play = RLBotSkillStage(
        name="Team Play",
        base_task_state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),  # 2v2 format
            KickoffMutator()
        ),
        base_task_reward_function=create_lucy_skg_reward(team_goal_y),  # Full Lucy-SKG reward
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=timeout_condition,
        bot_skill_ranges={(0.5, 1.0): 1.0},  # Challenging opponents only
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.5,
            min_avg_reward=0.3,
            min_episodes=300,
            max_std_dev=2.5,
            required_consecutive_successes=2
        )
    )

    # Create the full curriculum
    stages = [
        ball_control,
        directional_shooting,
        aerial_fundamentals,
        strategic_positioning,
        advanced_aerials,
        competitive_play,
        team_play
    ]

    curriculum_manager = CurriculumManager(
        stages=stages,
        progress_thresholds={"success_rate": 0.6, "avg_reward": 0.5},
        max_rehearsal_stages=2,  # Revisit up to 2 previous stages
        rehearsal_decay_factor=0.6,  # Higher weight for more recent stages
        evaluation_window=50,
        debug=debug,
        use_wandb=use_wandb
    )

    return curriculum_manager
