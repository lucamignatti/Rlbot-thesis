import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import random
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.state_mutators import (
    MutatorSequence, FixedTeamSizeMutator, KickoffMutator, 
    BoostMutator, BallVelocityMutator, CarSpawnLocationMutator,
    BallSpawnLocationMutator, BoostSpawnLocationMutator
)
from curriculum import CurriculumManager, CurriculumStage, ProgressionRequirements
from curriculum_rlbot import RLBotStage
from curriculum_skills import SkillBasedCurriculumStage, SkillModule
from rewards import (
    BallProximityReward, BallToGoalDistanceReward, BallVelocityToGoalReward,
    TouchBallReward, TouchBallToGoalAccelerationReward, AlignBallToGoalReward,
    PlayerVelocityTowardBallReward, KRCReward
)

def create_skill_based_curriculum(debug=False, use_wandb=True):
    """Create a curriculum with a 60/40 split between base tasks and specialized skills"""
    
    # Define some constants and common components that will be reused
    ORANGE_GOAL_LOCATION = np.array([0, 5120, 642.775])
    BLUE_GOAL_LOCATION = np.array([0, -5120, 642.775])
    BASE_TIMEOUT = 300  # 5 minutes (300 seconds)
    SKILL_TIMEOUT = 120  # 2 minutes for skill training
    
    # Common conditions
    goal_condition = GoalCondition()
    timeout_condition = TimeoutCondition(BASE_TIMEOUT)
    skill_timeout = TimeoutCondition(SKILL_TIMEOUT)
    no_touch_timeout = NoTouchTimeoutCondition(timeout=30)
    
    # Common base reward components
    goal_reward = (GoalReward(), 10.0)
    touch_reward = (TouchReward(), 0.5)
    ball_to_goal_reward = (BallToGoalDistanceReward(negative_slope=False), 0.3)
    
    # Stage 1: Solo Ball Control & Basic Skills
    # ----------------------------------------
    
    # Define Stage 1 base task
    stage1_base_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=0),  # Solo play
        KickoffMutator()
    )
    
    stage1_base_reward = CombinedReward(
        goal_reward,
        touch_reward,
        ball_to_goal_reward,
        (PlayerVelocityTowardBallReward(), 0.2)  # Encourage moving toward ball
    )
    
    # Define Stage 1 skill modules
    
    # Skill 1: Ball Touch Accuracy - spawn the ball in random positions and touch it
    ball_touch_skill = SkillModule(
        name="Ball Touch Accuracy",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallSpawnLocationMutator(
                min_x=-2000, max_x=2000,
                min_y=-3000, max_y=3000,
                min_z=100, max_z=300
            )
        ),
        reward_function=CombinedReward(
            (TouchReward(), 5.0),
            (BallProximityReward(negative_slope=True), 0.5)  # High reward for getting close
        ),
        termination_condition=goal_condition,
        truncation_condition=no_touch_timeout,
        difficulty_params={
            "ball_distance": (1000, 3000),  # Min and max distance as difficulty increases
            "ball_height": (100, 800)  # Min and max height as difficulty increases
        }
    )
    
    # Skill 2: Basic Dribbling - keep the ball close while moving
    dribbling_skill = SkillModule(
        name="Basic Dribbling",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallSpawnLocationMutator(min_x=-1000, max_x=1000, min_y=-1000, max_y=1000, min_z=100, max_z=100),
            CarSpawnLocationMutator(min_x=-1000, max_x=1000, min_y=-1000, max_y=1000)
        ),
        reward_function=CombinedReward(
            (TouchReward(), 0.5),  # Small reward for touches
            (BallProximityReward(negative_slope=True), 0.7),  # Higher reward for keeping ball close
            (BallToGoalDistanceReward(), 0.3)  # Move toward goal
        ),
        termination_condition=goal_condition,
        truncation_condition=skill_timeout,
        difficulty_params={
            "min_touch_interval": (0.5, 2.0),  # Maximum seconds between touches
            "max_ball_distance": (300, 150)  # Gets more strict with difficulty
        }
    )
    
    # Skill 3: Basic Shooting - shooting from different angles
    shooting_skill = SkillModule(
        name="Basic Shooting",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallSpawnLocationMutator(
                min_x=-3000, max_x=3000,
                min_y=-3000, max_y=3000,
                min_z=100, max_z=100
            )
        ),
        reward_function=CombinedReward(
            (GoalReward(), 10.0),
            (BallToGoalDistanceReward(negative_slope=False), 0.5),
            (BallVelocityToGoalReward(), 0.5)  # Reward shooting ball toward goal
        ),
        termination_condition=goal_condition,
        truncation_condition=skill_timeout,
        difficulty_params={
            "shot_angle": (0, 60),  # Shot angle from straight-on to angled (degrees)
            "shot_distance": (1000, 4000)  # Distance from goal
        }
    )
    
    # Create Stage 1
    stage1 = SkillBasedCurriculumStage(
        name="Solo Ball Control",
        base_task_state_mutator=stage1_base_mutator,
        base_task_reward_function=stage1_base_reward,
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=timeout_condition,
        skill_modules=[ball_touch_skill, dribbling_skill, shooting_skill],
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.4,
            min_episodes=100,
            max_std_dev=0.3,
            required_consecutive_successes=3
        ),
        allowed_bots=[],  # No bots for Stage 1
        debug=debug,
        use_wandb=use_wandb
    )
    
    # Stage 2: Simple 1v1 Play
    # -----------------------
    
    # Define Stage 2 base task - 1v1 against beginner bots
    stage2_base_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    )
    
    stage2_base_reward = CombinedReward(
        goal_reward,
        (TouchReward(), 0.3),  # Lower touch reward
        ball_to_goal_reward,
        (BallVelocityToGoalReward(), 0.3)  # Encourage shooting
    )
    
    # Skill 1: Defensive Positioning
    defensive_positioning_skill = SkillModule(
        name="Defensive Positioning",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            BallSpawnLocationMutator(
                min_x=-2000, max_x=2000,
                min_y=-1000, max_y=4000,  # Ball toward our goal
                min_z=100, max_z=100
            ),
            CarSpawnLocationMutator(
                min_x=-1000, max_x=1000,
                min_y=-4000, max_y=-3000,  # We spawn near our goal
                min_z=17, max_z=17
            )
        ),
        reward_function=CombinedReward(
            (GoalReward(), 5.0),  # Lower goal reward since defense is primary
            (TouchReward(), 0.3),
            (BallProximityReward(negative_slope=True), 0.5),  # Get to the ball
            # Positioning between ball and our goal gets extra reward
            # TODO: Add a proper defensive positioning reward
        ),
        termination_condition=goal_condition,
        truncation_condition=skill_timeout,
        difficulty_params={
            "ball_speed": (500, 1500)  # Ball initial velocity increases with difficulty
        }
    )
    
    # Skill 2: Ball Interception
    interception_skill = SkillModule(
        name="Ball Interception",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),  # No opponent
            BallSpawnLocationMutator(
                min_x=-3000, max_x=3000,
                min_y=-3000, max_y=3000,
                min_z=100, max_z=800
            ),
            BallVelocityMutator(
                min_x_vel=-1000, max_x_vel=1000,
                min_y_vel=-1000, max_y_vel=1000,
                min_z_vel=-100, max_z_vel=300
            )
        ),
        reward_function=CombinedReward(
            (TouchReward(), 10.0),  # High reward for touching moving ball
            (BallProximityReward(negative_slope=True), 0.5)
        ),
        termination_condition=goal_condition,
        truncation_condition=no_touch_timeout,
        difficulty_params={
            "ball_speed": (500, 2000),  # Ball speed increases with difficulty
            "prediction_time": (2.0, 0.5)  # Time to intercept decreases with difficulty
        }
    )
    
    # Create Stage 2
    stage2 = SkillBasedCurriculumStage(
        name="Simple 1v1 Play",
        base_task_state_mutator=stage2_base_mutator,
        base_task_reward_function=stage2_base_reward,
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=timeout_condition,
        skill_modules=[defensive_positioning_skill, interception_skill],
        bot_skill_ranges={(0.1, 0.3): 0.7, (0.3, 0.5): 0.3},  # Mostly easy bots
        bot_tags=['beginner', 'easy'],  # Prefer beginner/easy bots
        allowed_bots=['Noob_Black', 'RedBot', 'MirrorBot', 'Bumblebee'],  # Simple bots
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.6,
            min_avg_reward=0.45,
            min_episodes=200,
            max_std_dev=0.3,
            required_consecutive_successes=3
        ),
        debug=debug,
        use_wandb=use_wandb
    )
    
    # Stage 3: Advanced 1v1 & Basic Team Play
    # --------------------------------------
    
    # Define Stage 3 base task - 1v1 against intermediate bots
    stage3_base_reward = CombinedReward(
        goal_reward,
        (TouchReward(), 0.2),  # Lower touch reward
        ball_to_goal_reward,
        (BallVelocityToGoalReward(), 0.3),  # Encourage shooting
        (TouchBallToGoalAccelerationReward(), 0.2)  # Reward powerful shots toward goal
    )
    
    # Skill 1: Basic Aerial Hits
    aerial_skill = SkillModule(
        name="Basic Aerial Hits",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallSpawnLocationMutator(
                min_x=-2000, max_x=2000,
                min_y=-2000, max_y=2000,
                min_z=500, max_z=1200  # Ball in the air
            )
        ),
        reward_function=CombinedReward(
            (TouchReward(), 8.0),  # High reward for aerial touches
            (BallToGoalDistanceReward(), 0.3),
            (BallVelocityToGoalReward(), 0.5)  # Reward hitting toward goal
        ),
        termination_condition=goal_condition,
        truncation_condition=skill_timeout,
        difficulty_params={
            "ball_height": (500, 1500),  # Height increases with difficulty
            "car_boost": (100, 50)  # Less boost with higher difficulty
        }
    )
    
    # Skill 2: Wall Play
    wall_play_skill = SkillModule(
        name="Wall Play",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallSpawnLocationMutator(
                min_x=3000, max_x=4000,  # Near side wall
                min_y=-3000, max_y=3000,
                min_z=100, max_z=800
            )
        ),
        reward_function=CombinedReward(
            (TouchReward(), 3.0),
            (BallToGoalDistanceReward(), 0.5),
            (BallVelocityToGoalReward(), 0.5)  # Reward hitting toward goal from wall
        ),
        termination_condition=goal_condition,
        truncation_condition=skill_timeout,
        difficulty_params={
            "wall_distance": (4000, 3000),  # Distance from center to wall (smaller = harder)
            "ball_height": (100, 1000)  # Ball height increases with difficulty
        }
    )
    
    # Create Stage 3
    stage3 = SkillBasedCurriculumStage(
        name="Advanced 1v1 Play",
        base_task_state_mutator=stage2_base_mutator,  # Reuse the 1v1 mutator
        base_task_reward_function=stage3_base_reward,
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=timeout_condition,
        skill_modules=[aerial_skill, wall_play_skill],
        bot_skill_ranges={(0.3, 0.5): 0.5, (0.5, 0.7): 0.5},  # Mix of medium and harder bots
        bot_tags=['intermediate'],  # Prefer intermediate bots
        allowed_bots=['ReliefBot', 'Slime', 'Kamael', 'AtlasBot', 'DiabloBOT'],  # More advanced bots
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.5,
            min_episodes=300,
            max_std_dev=0.35,
            required_consecutive_successes=3
        ),
        debug=debug,
        use_wandb=use_wandb
    )
    
    # Stage 4: Basic 2v2 Play
    # ----------------------
    
    # Define Stage 4 base task - 2v2 with consistent ally
    stage4_base_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=2, orange_size=2),
        KickoffMutator()
    )
    
    stage4_base_reward = CombinedReward(
        goal_reward,
        (TouchReward(), 0.2),
        ball_to_goal_reward,
        (BallVelocityToGoalReward(), 0.3),
        # Add team play specific rewards
        # TODO: Add passing and positioning rewards
    )
    
    # Skill 1: Team Positioning
    team_positioning_skill = SkillModule(
        name="Team Positioning",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            BallSpawnLocationMutator(
                min_x=-2000, max_x=2000,
                min_y=-2000, max_y=2000,
                min_z=100, max_z=100
            )
        ),
        reward_function=CombinedReward(
            (TouchReward(), 0.5),
            # TODO: Add proper team positioning reward
        ),
        termination_condition=goal_condition,
        truncation_condition=skill_timeout,
        difficulty_params={
            "positioning_strictness": (0.2, 0.8)  # More strict positioning requirements with difficulty
        }
    )
    
    # Create Stage 4
    stage4 = SkillBasedCurriculumStage(
        name="Basic 2v2 Play",
        base_task_state_mutator=stage4_base_mutator,
        base_task_reward_function=stage4_base_reward,
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=timeout_condition,
        skill_modules=[team_positioning_skill],
        bot_skill_ranges={(0.3, 0.6): 1.0},  # Medium difficulty bots
        bot_tags=['intermediate', '2v2'],
        allowed_bots=['ReliefBot', 'Kamael', 'AtlasBot', 'Botimus', 'DiabloBOT'],
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.5,
            min_episodes=300,
            max_std_dev=0.35,
            required_consecutive_successes=3
        ),
        debug=debug,
        use_wandb=use_wandb
    )
    
    # Stage 5: Advanced 2v2 Play
    # -------------------------
    
    # Define more advanced 2v2 reward
    stage5_base_reward = CombinedReward(
        goal_reward,
        (TouchReward(), 0.1),  # Lower basic touch reward
        ball_to_goal_reward,
        (BallVelocityToGoalReward(), 0.3),
        # Add advanced team play rewards
        # TODO: Add passing, positioning, and team coordination rewards
    )
    
    # Skill: Advanced Aerials
    advanced_aerial_skill = SkillModule(
        name="Advanced Aerials",
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            BallSpawnLocationMutator(
                min_x=-2000, max_x=2000,
                min_y=-2000, max_y=2000,
                min_z=800, max_z=1500  # Higher in the air
            ),
            BallVelocityMutator(  # Moving ball
                min_x_vel=-300, max_x_vel=300,
                min_y_vel=-300, max_y_vel=300,
                min_z_vel=-100, max_z_vel=100
            )
        ),
        reward_function=CombinedReward(
            (TouchReward(), 8.0),  # High reward for aerial touches
            (BallToGoalDistanceReward(), 0.3),
            (BallVelocityToGoalReward(), 0.5)  # Reward hitting toward goal
        ),
        termination_condition=goal_condition,
        truncation_condition=skill_timeout,
        difficulty_params={
            "ball_height": (800, 1800),  # Height increases with difficulty
            "ball_velocity": (100, 500)  # Ball moves faster with difficulty
        }
    )
    
    # Create Stage 5
    stage5 = SkillBasedCurriculumStage(
        name="Advanced 2v2 Play",
        base_task_state_mutator=stage4_base_mutator,  # Reuse 2v2 mutator
        base_task_reward_function=stage5_base_reward,
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=timeout_condition,
        skill_modules=[advanced_aerial_skill],
        bot_skill_ranges={(0.5, 0.8): 1.0},  # Harder bots
        bot_tags=['advanced', '2v2', 'expert'],
        allowed_bots=['Necto', 'Nexto', 'ReliefBot', 'KStyle', 'Botimus Prime'],
        progression_requirements=ProgressionRequirements(
            min_success_rate=0.55,
            min_avg_reward=0.5,
            min_episodes=400,
            max_std_dev=0.4,
            required_consecutive_successes=3
        ),
        debug=debug,
        use_wandb=use_wandb
    )
    
    # Stage 6: Self-play 2v2
    # ---------------------
    
    # Define self-play stage - our bot competes against itself
    stage6_base_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=2, orange_size=2),
        KickoffMutator()
    )
    
    # Self-play has a simplified reward function
    stage6_base_reward = CombinedReward(
        (GoalReward(), 10.0),  # Focus on winning
        (TouchReward(), 0.1),
        (BallVelocityToGoalReward(), 0.2)
    )
    
    # No skill modules in final stage, pure self-play
    stage6 = SkillBasedCurriculumStage(
        name="Self-play 2v2",
        base_task_state_mutator=stage6_base_mutator,
        base_task_reward_function=stage6_base_reward,
        base_task_termination_condition=goal_condition,
        base_task_truncation_condition=timeout_condition,
        skill_modules=[],  # No special skills, just self-play
        # Self-play uses our own bot, not RLBotPack bots
        progression_requirements=None,  # Final stage, no progression requirements
        debug=debug,
        use_wandb=use_wandb,
        base_task_prob=1.0  # Always use base task
    )
    
    # Create the curriculum manager with all stages
    manager = CurriculumManager(
        stages=[stage1, stage2, stage3, stage4, stage5, stage6],
        progress_thresholds={
            'success_rate': 0.6,
            'avg_reward': 0.5
        },
        max_rehearsal_stages=2,
        rehearsal_decay_factor=0.6,
        evaluation_window=50,
        debug=debug,
        use_wandb=use_wandb
    )
    
    return manager

# Helper function to get a list of available RLBots from RLBotPack
def get_available_rlbots(rlbotpack_path=None):
    """Scan the RLBotPack directory to find available bots"""
    if rlbotpack_path is None:
        # Try to find RLBotPack in the current directory
        if os.path.exists('RLBotPack'):
            rlbotpack_path = 'RLBotPack/RLBotPack'
        else:
            return []
    
    if not os.path.exists(rlbotpack_path):
        return []
    
    # List directories in RLBotPack, each should be a bot
    bots = []
    for item in os.listdir(rlbotpack_path):
        bot_dir = os.path.join(rlbotpack_path, item)
        if os.path.isdir(bot_dir):
            # Check if this looks like a bot (has .cfg file)
            cfg_files = [f for f in os.listdir(bot_dir) if f.endswith('.cfg')]
            if cfg_files:
                bots.append(item)
    
    return bots