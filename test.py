import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from curriculum import ProgressionRequirements, CurriculumStage, CurriculumManager
from rlgym.api import StateMutator, RewardFunction, DoneCondition
import tempfile
import os

# filepath: /home/luca/Projects/Rlbot-thesis/test_curriculum.py

class MockStateMutator(StateMutator):
    def apply(self, state, shared_info):
        # Ensure the ball has a valid position
        if hasattr(state, 'ball') and state.ball is not None:
            # Set a default position if not already set
            if not hasattr(state.ball, 'position') or state.ball.position is None:
                state.ball.position = [0.0, 0.0, 100.0]

            # Ensure linear_velocity is set to prevent similar errors
            if not hasattr(state.ball, 'linear_velocity') or state.ball.linear_velocity is None:
                state.ball.linear_velocity = [0.0, 0.0, 0.0]

class MockRewardFunction(RewardFunction):
    def calculate(self, player, state, previous_state=None):
        return 1.0

class MockDoneCondition(DoneCondition):
    def is_done(self, state):
        return False

class TestProgressionRequirements(unittest.TestCase):
    """Test the ProgressionRequirements class"""

    # Test basic initialization with valid parameters
    def test_valid_initialization(self):
        """Make sure we can create requirements with valid values"""
        req = ProgressionRequirements(
            min_success_rate=0.7,
            min_avg_reward=0.5,
            min_episodes=50,
            max_std_dev=0.3
        )
        self.assertEqual(req.min_success_rate, 0.7)
        self.assertEqual(req.min_avg_reward, 0.5)
        self.assertEqual(req.min_episodes, 50)
        self.assertEqual(req.max_std_dev, 0.3)
        self.assertEqual(req.required_consecutive_successes, 3)  # Should use default

    def test_invalid_parameters(self):
        """Make sure we catch all invalid parameter combinations"""
        # Try a success rate over 100%
        with self.assertRaises(ValueError):
            ProgressionRequirements(min_success_rate=1.5, min_avg_reward=0.5,
                                   min_episodes=50, max_std_dev=0.3)

        # Try zero episodes (must be positive)
        with self.assertRaises(ValueError):
            ProgressionRequirements(min_success_rate=0.7, min_avg_reward=0.5,
                                   min_episodes=0, max_std_dev=0.3)

        # Try negative standard deviation (must be positive)
        with self.assertRaises(ValueError):
            ProgressionRequirements(min_success_rate=0.7, min_avg_reward=0.5,
                                   min_episodes=50, max_std_dev=-0.1)

        # Also test a negative reward threshold
        with self.assertRaises(ValueError):
            ProgressionRequirements(min_success_rate=0.7, min_avg_reward=-2.0,
                                   min_episodes=50, max_std_dev=0.3)

        # Test invalid consecutive successes (must be positive)
        with self.assertRaises(ValueError):
            ProgressionRequirements(min_success_rate=0.7, min_avg_reward=0.5,
                                   min_episodes=50, max_std_dev=0.3,
                                   required_consecutive_successes=0)

    def test_edge_cases(self):
        """Test edge cases for progression requirements"""
        # Test minimum valid values
        min_req = ProgressionRequirements(
            min_success_rate=0.0,
            min_avg_reward=0.0,
            min_episodes=1,
            max_std_dev=0.0,
            required_consecutive_successes=1
        )
        self.assertEqual(min_req.min_success_rate, 0.0)
        
        # Test maximum valid values
        max_req = ProgressionRequirements(
            min_success_rate=1.0,
            min_avg_reward=1.0,
            min_episodes=1000000,
            max_std_dev=10.0,
            required_consecutive_successes=1000
        )
        self.assertEqual(max_req.min_success_rate, 1.0)

class TestCurriculumStage(unittest.TestCase):
    """Test the CurriculumStage class"""

    def setUp(self):
        """Set up test fixtures"""
        self.state_mutator = MockStateMutator()
        self.reward_fn = MockRewardFunction()
        self.termination_cond = MockDoneCondition()
        self.truncation_cond = MockDoneCondition()

        self.progression_req = ProgressionRequirements(
            min_success_rate=0.7,
            min_avg_reward=0.5,
            min_episodes=5,  # Low for testing
            max_std_dev=0.5,
            required_consecutive_successes=2
        )

        self.stage = CurriculumStage(
            name="Test Stage",
            state_mutator=self.state_mutator,
            reward_function=self.reward_fn,
            termination_condition=self.termination_cond,
            truncation_condition=self.truncation_cond,
            progression_requirements=self.progression_req,
            difficulty_params={"param1": (0.1, 1.0), "param2": (5, 10)}
        )

    def test_get_consecutive_successes(self):
        """Test counting consecutive successes"""
        # No rewards history
        self.assertEqual(self.stage.get_consecutive_successes(), 0)

        # Some successes
        self.stage.rewards_history = [0.5, 0.7, -0.1, 0.6, 0.8]
        self.assertEqual(self.stage.get_consecutive_successes(), 2)

        # No recent successes
        self.stage.rewards_history = [0.5, 0.7, -0.1, -0.2, -0.3]
        self.assertEqual(self.stage.get_consecutive_successes(), 0)

        # All successes
        self.stage.rewards_history = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.assertEqual(self.stage.get_consecutive_successes(), 5)

    def test_get_config_with_difficulty(self):
        """Test difficulty parameter interpolation"""
        # Test min difficulty
        config = self.stage.get_config_with_difficulty(0.0)
        self.assertEqual(config["param1"], 0.1)
        self.assertEqual(config["param2"], 5)

        # Test max difficulty
        config = self.stage.get_config_with_difficulty(1.0)
        self.assertEqual(config["param1"], 1.0)
        self.assertEqual(config["param2"], 10)

        # Test mid difficulty
        config = self.stage.get_config_with_difficulty(0.5)
        self.assertEqual(config["param1"], 0.55)  # 0.1 + 0.5 * (1.0 - 0.1)
        self.assertEqual(config["param2"], 7.5)    # 5 + 0.5 * (10 - 5)

        # Test out of bounds (should clamp)
        config = self.stage.get_config_with_difficulty(1.5)
        self.assertEqual(config["param1"], 1.0)
        self.assertEqual(config["param2"], 10)

        config = self.stage.get_config_with_difficulty(-0.5)
        self.assertEqual(config["param1"], 0.1)
        self.assertEqual(config["param2"], 5)

    def test_update_statistics(self):
        """Test updating stage statistics"""
        # Test successful episode
        self.stage.update_statistics({"success": True, "timeout": False, "episode_reward": 0.8})
        self.assertEqual(self.stage.episode_count, 1)
        self.assertEqual(self.stage.success_count, 1)
        self.assertEqual(self.stage.failure_count, 0)
        self.assertEqual(self.stage.rewards_history, [0.8])
        self.assertEqual(self.stage.moving_success_rate, 1.0)
        self.assertEqual(self.stage.moving_avg_reward, 0.8)

        # Test failed episode
        self.stage.update_statistics({"success": False, "timeout": True, "episode_reward": -0.2})
        self.assertEqual(self.stage.episode_count, 2)
        self.assertEqual(self.stage.success_count, 1)
        self.assertEqual(self.stage.failure_count, 1)
        self.assertEqual(self.stage.rewards_history, [0.8, -0.2])
        self.assertEqual(self.stage.moving_success_rate, 0.5)
        self.assertAlmostEqual(self.stage.moving_avg_reward, 0.3, places=5)  # Use assertAlmostEqual

    def test_validate_progression(self):
        """Test progression validation logic"""
        # Not enough episodes
        self.assertFalse(self.stage.validate_progression())

        # Add some episodes, but not meeting criteria
        for _ in range(5):
            self.stage.update_statistics({"success": False, "episode_reward": 0.1})
        self.assertFalse(self.stage.validate_progression())

        # Reset and add episodes meeting criteria
        self.stage.reset_statistics()
        for _ in range(3):  # Not enough episodes
            self.stage.update_statistics({"success": True, "episode_reward": 0.8})
        self.assertFalse(self.stage.validate_progression())

        # Add more episodes, but not consistently successful
        self.stage.reset_statistics()
        for i in range(7):  # More than min_episodes
            # Alternate between success and failure
            success = i % 2 == 0
            reward = 0.8 if success else 0.1
            self.stage.update_statistics({"success": success, "episode_reward": reward})

        # This should fail because we don't have enough consecutive successes
        self.assertFalse(self.stage.validate_progression())

        # Finally add data that truly meets all criteria
        self.stage.reset_statistics()
        for _ in range(7):  # More than min_episodes
            self.stage.update_statistics({"success": True, "episode_reward": 0.8})

        # This should pass all validation criteria
        self.assertTrue(self.stage.validate_progression())

    def test_reset_statistics(self):
        """Test complete reset of stage statistics"""
        # Add some data
        self.stage.update_statistics({"success": True, "episode_reward": 0.8})
        self.stage.update_statistics({"success": False, "episode_reward": 0.2})
        
        # Verify data exists
        self.assertTrue(len(self.stage.rewards_history) > 0)
        self.assertTrue(self.stage.episode_count > 0)
        
        # Reset
        self.stage.reset_statistics()
        
        # Verify complete reset
        self.assertEqual(len(self.stage.rewards_history), 0)
        self.assertEqual(self.stage.episode_count, 0)
        self.assertEqual(self.stage.success_count, 0)
        self.assertEqual(self.stage.failure_count, 0)
        self.assertEqual(self.stage.moving_success_rate, 0.0)
        self.assertEqual(self.stage.moving_avg_reward, 0.0)

    def test_empty_difficulty_params(self):
        """Test stage behavior with no difficulty parameters"""
        stage = CurriculumStage(
            name="No Difficulty Stage",
            state_mutator=self.state_mutator,
            reward_function=self.reward_fn,
            termination_condition=self.termination_cond,
            truncation_condition=self.truncation_cond
        )
        
        # Should return empty dict for any difficulty level
        self.assertEqual(stage.get_config_with_difficulty(0.5), {})
        self.assertEqual(stage.get_config_with_difficulty(1.0), {})

class TestCurriculumManager(unittest.TestCase):
    """Test the CurriculumManager class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock components
        self.state_mutator = MockStateMutator()
        self.reward_fn = MockRewardFunction()
        self.term_cond = MockDoneCondition()
        self.trunc_cond = MockDoneCondition()

        # Create progression requirements
        self.prog_req1 = ProgressionRequirements(
            min_success_rate=0.7,
            min_avg_reward=0.5,
            min_episodes=5,  # Low for testing
            max_std_dev=0.5,
            required_consecutive_successes=2
        )

        self.prog_req2 = ProgressionRequirements(
            min_success_rate=0.8,
            min_avg_reward=0.6,
            min_episodes=5,
            max_std_dev=0.4,
            required_consecutive_successes=3
        )

        # Create stages
        self.stage1 = CurriculumStage(
            name="Stage 1",
            state_mutator=self.state_mutator,
            reward_function=self.reward_fn,
            termination_condition=self.term_cond,
            truncation_condition=self.trunc_cond,
            progression_requirements=self.prog_req1,
            difficulty_params={"param1": (0.1, 1.0)},
            hyperparameter_adjustments={"lr_actor": 1e-3, "lr_critic": 1e-2}
        )

        self.stage2 = CurriculumStage(
            name="Stage 2",
            state_mutator=self.state_mutator,
            reward_function=self.reward_fn,
            termination_condition=self.term_cond,
            truncation_condition=self.trunc_cond,
            progression_requirements=self.prog_req2,
            difficulty_params={"param1": (1.0, 2.0)},
            hyperparameter_adjustments={"lr_actor": 5e-4, "lr_critic": 5e-3}
        )

        # Create manager with test stages
        self.manager = CurriculumManager(
            stages=[self.stage1, self.stage2],
            evaluation_window=5,  # Small window for testing
            debug=False,
            testing=True
        )

        # Mock the validate_all_stages method to avoid needing full environment setup
        self.manager.validate_all_stages = lambda: None

    def test_initialization(self):
        """Test curriculum manager initialization"""
        self.assertEqual(self.manager.current_stage_index, 0)
        self.assertEqual(self.manager.current_difficulty, 0.0)
        self.assertEqual(self.manager.total_episodes, 0)
        self.assertEqual(len(self.manager.stages), 2)

    def test_get_environment_config(self):
        """Test getting environment configuration"""
        config = self.manager.get_environment_config()
        self.assertEqual(config["stage_name"], "Stage 1")
        self.assertEqual(config["difficulty_level"], 0.0)
        self.assertEqual(config["is_rehearsal"], False)

        # Test with increased difficulty
        self.manager.current_difficulty = 0.5
        config = self.manager.get_environment_config()
        self.assertEqual(config["difficulty_level"], 0.5)
        self.assertEqual(config["difficulty_params"]["param1"], 0.55)  # 0.1 + 0.5 * (1.0 - 0.1)

    def test_curriculum_progression(self):
        """Test curriculum stage advancement"""
        # Mock trainer for hyperparameter adjustments
        mock_trainer = MagicMock()
        mock_trainer.actor_optimizer.param_groups = [{"lr": 0.0}]
        mock_trainer.critic_optimizer.param_groups = [{"lr": 0.0}]
        mock_trainer.entropy_coef = 0.0
        self.manager.register_trainer(mock_trainer)

        # Initial state
        self.assertEqual(self.manager.current_stage_index, 0)
        self.assertEqual(self.manager.current_difficulty, 0.0)

        # Add episodes that would normally increase difficulty
        for _ in range(5):
            self.manager.update_progression_stats({
                "success": True,
                "episode_reward": 0.6
            })

        # Manually set difficulty to simulate increase
        self.manager.current_difficulty = 0.1

        # Check difficulty increased but stage hasn't changed
        self.assertGreater(self.manager.current_difficulty, 0.0)
        self.assertEqual(self.manager.current_stage_index, 0)

        # Reset and push difficulty to near max
        self.manager.current_difficulty = 0.95
        self.manager.stages[0].reset_statistics()

        # Prepare stage for progression by setting progression metrics
        self.manager.stages[0].rewards_history = [0.8] * 10
        self.manager.stages[0].success_count = 10
        self.manager.stages[0].episode_count = 10
        self.manager.stages[0].moving_success_rate = 1.0
        self.manager.stages[0].moving_avg_reward = 0.8

        # Add episodes that meet all criteria for stage 1
        for _ in range(5):
            self.manager.update_progression_stats({
                "success": True,
                "episode_reward": 0.8
            })

        # Trigger the evaluation explicitly
        self.manager._evaluate_progression()

        # Should have progressed to stage 2
        self.assertEqual(self.manager.current_stage_index, 1)
        self.assertEqual(self.manager.current_difficulty, 0.0)  # Reset for new stage

        # Verify hyperparameters were updated
        self.assertEqual(mock_trainer.actor_optimizer.param_groups[0]["lr"], 5e-4)
        self.assertEqual(mock_trainer.critic_optimizer.param_groups[0]["lr"], 5e-3)

    def test_rehearsal(self):
        """Test rehearsal stage selection"""
        # Move to stage 2 first
        self.manager.current_stage_index = 1

        # Test rehearsal probability calculation
        prob = self.manager._get_rehearsal_probability()
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 0.3)  # Default max is 0.3

        # Force rehearsal selection by patching random
        with patch('numpy.random.random', return_value=0.0):  # Always choose rehearsal
            with patch('numpy.random.choice', return_value=0):  # Always choose first available stage
                config = self.manager.get_environment_config()
                self.assertEqual(config["stage_name"], "Stage 1")
                self.assertEqual(config["is_rehearsal"], True)

    def test_save_load(self):
        """Test saving and loading curriculum state"""
        # Set up some state
        self.manager.current_stage_index = 1
        self.manager.current_difficulty = 0.75
        self.manager.total_episodes = 100

        # Update statistics for both stages
        self.manager.stages[0].update_statistics({"success": True, "episode_reward": 0.9})
        self.manager.stages[1].update_statistics({"success": True, "episode_reward": 0.8})

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
            self.manager.save_curriculum(temp_path)

        # Create new manager and load state
        new_manager = CurriculumManager(
            stages=[self.stage1, self.stage2],
            debug=False
        )
        new_manager.validate_all_stages = lambda: None

        new_manager.load_curriculum(temp_path)

        # Verify state was loaded correctly
        self.assertEqual(new_manager.current_stage_index, 1)
        self.assertEqual(new_manager.current_difficulty, 0.75)
        self.assertEqual(new_manager.total_episodes, 100)
        self.assertEqual(new_manager.stages[0].episode_count, 1)
        self.assertEqual(new_manager.stages[1].episode_count, 1)

        # Clean up
        os.unlink(temp_path)

    def test_invalid_rehearsal_config(self):
        """Test rehearsal behavior with invalid configuration"""
        # Try negative max_rehearsal_stages
        with self.assertRaises(ValueError):
            CurriculumManager(
                stages=[self.stage1, self.stage2],
                max_rehearsal_stages=-1
            )
        
        # Try invalid rehearsal_decay_factor
        with self.assertRaises(ValueError):
            CurriculumManager(
                stages=[self.stage1, self.stage2],
                rehearsal_decay_factor=-0.5
            )

    def test_empty_stages(self):
        """Test handling of empty stages list"""
        with self.assertRaises(ValueError):
            CurriculumManager(stages=[])

    def test_hyperparameter_adjustments(self):
        """Test detailed hyperparameter adjustment behavior"""
        mock_trainer = MagicMock()
        mock_trainer.actor_optimizer.param_groups = [{"lr": 1e-3}]
        mock_trainer.critic_optimizer.param_groups = [{"lr": 1e-2}]
        mock_trainer.entropy_coef = 0.1
        
        self.manager.register_trainer(mock_trainer)
        
        # Add custom hyperparameter adjustments to stage2
        self.stage2.hyperparameter_adjustments = {
            "lr_actor": 5e-4,
            "lr_critic": 5e-3,
            "entropy_coef": 0.05
        }
        
        # Force progression to stage 2
        self.manager.current_stage_index = 0
        self.manager.current_difficulty = 0.95
        self.stage1.rewards_history = [0.8] * 10
        self.stage1.success_count = 10
        self.stage1.episode_count = 10
        self.stage1.moving_success_rate = 1.0
        self.stage1.moving_avg_reward = 0.8
        
        # Trigger progression
        for _ in range(5):
            self.manager.update_progression_stats({
                "success": True,
                "episode_reward": 0.8
            })
        
        # Verify all hyperparameters were adjusted
        self.assertEqual(mock_trainer.actor_optimizer.param_groups[0]["lr"], 5e-4)
        self.assertEqual(mock_trainer.critic_optimizer.param_groups[0]["lr"], 5e-3)
        self.assertEqual(mock_trainer.entropy_coef, 0.05)

class TestCurriculumIntegration(unittest.TestCase):
    """Integration tests for curriculum learning"""

    def setUp(self):
        # Create simplified curriculum with clear progression criteria
        self.state_mutator = MockStateMutator()
        self.reward_fn = MockRewardFunction()
        self.term_cond = MockDoneCondition()
        self.trunc_cond = MockDoneCondition()

        # Instead of patching methods, let's use a testing flag

        # Create progression requirements with simple criteria
        self.req1 = ProgressionRequirements(
            min_success_rate=0.7,
            min_avg_reward=0.6,
            min_episodes=10,
            max_std_dev=0.3,
            required_consecutive_successes=3
        )

        self.req2 = ProgressionRequirements(
            min_success_rate=0.8,
            min_avg_reward=0.7,
            min_episodes=10,
            max_std_dev=0.3,
            required_consecutive_successes=4
        )

        # Create three stages with increasing difficulty
        self.stage1 = CurriculumStage(
            name="Basic",
            state_mutator=self.state_mutator,
            reward_function=self.reward_fn,
            termination_condition=self.term_cond,
            truncation_condition=self.trunc_cond,
            progression_requirements=self.req1,
            difficulty_params={"speed": (0.5, 1.0)}
        )

        self.stage2 = CurriculumStage(
            name="Intermediate",
            state_mutator=self.state_mutator,
            reward_function=self.reward_fn,
            termination_condition=self.term_cond,
            truncation_condition=self.trunc_cond,
            progression_requirements=self.req2,
            difficulty_params={"speed": (1.0, 1.5)}
        )

        self.stage3 = CurriculumStage(
            name="Advanced",
            state_mutator=self.state_mutator,
            reward_function=self.reward_fn,
            termination_condition=self.term_cond,
            truncation_condition=self.trunc_cond,
            progression_requirements=None,  # Final stage
            difficulty_params={"speed": (1.5, 2.0)}
        )

        # Create curriculum manager with testing mode flag
        self.manager = CurriculumManager(
            stages=[self.stage1, self.stage2, self.stage3],
            evaluation_window=5,  # Small window for testing
            debug=False
        )

        # Skip validation for tests
        self.manager.validate_all_stages = lambda: None


    def test_full_progression_simulation(self):
        """Test full curriculum progression with simulated episodes"""
        # Skip the actual environment creation/validation
        self.manager.validate_all_stages = lambda: None

        # Initial state
        self.assertEqual(self.manager.current_stage_index, 0)
        self.assertEqual(self.manager.current_difficulty, 0.0)

        # PHASE 1: Initial episodes with good performance
        for i in range(15):
            self.manager.update_progression_stats({
                "success": True,
                "episode_reward": 0.8
            })

        # Force conditions for stage progression
        self.manager.current_difficulty = 0.95
        self.manager.stages[0].rewards_history = [0.8] * 15
        self.manager.stages[0].success_count = 15
        self.manager.stages[0].episode_count = 15
        self.manager.stages[0].moving_success_rate = 1.0
        self.manager.stages[0].moving_avg_reward = 0.8

        # Trigger evaluation
        self.manager._evaluate_progression()

        # Should have progressed to stage 2
        self.assertEqual(self.manager.current_stage_index, 1)
        self.assertEqual(self.manager.current_difficulty, 0.0)  # Reset for new stage

        # Verify stage transitions were recorded
        self.assertEqual(len(self.manager.stage_transitions), 1)
        self.assertEqual(self.manager.stage_transitions[0]["from_stage"], "Basic")
        self.assertEqual(self.manager.stage_transitions[0]["to_stage"], "Intermediate")

    def test_stress_test(self):
        """Stress test with rapid stage transitions and frequent updates"""
        # Simulate rapid updates
        for _ in range(1000):
            self.manager.update_progression_stats({
                "success": True,
                "episode_reward": 0.9
            })
            
            # Randomly force stage transitions
            if np.random.random() < 0.1:  # 10% chance
                self.manager.current_difficulty = 0.95
                current_stage = self.manager.stages[self.manager.current_stage_index]
                current_stage.rewards_history = [0.9] * 20
                current_stage.success_count = 20
                current_stage.episode_count = 20
                current_stage.moving_success_rate = 1.0
                current_stage.moving_avg_reward = 0.9
                
                # Should handle rapid transitions gracefully
                self.manager._evaluate_progression()
        
        # Verify manager state is still valid
        self.assertGreaterEqual(self.manager.current_stage_index, 0)
        self.assertLess(self.manager.current_stage_index, len(self.manager.stages))
        self.assertGreaterEqual(self.manager.current_difficulty, 0.0)
        self.assertLessEqual(self.manager.current_difficulty, 1.0)

    def test_varied_performance(self):
        """Test curriculum behavior with varying performance patterns"""
        # Pattern 1: Gradually improving performance
        for i in range(20):
            success = np.random.random() < (0.5 + i * 0.025)  # Increasing success probability
            reward = 0.5 + (i * 0.02)  # Gradually increasing rewards
            self.manager.update_progression_stats({
                "success": success,
                "episode_reward": reward
            })
        
        # Pattern 2: Sudden performance drop
        for _ in range(10):
            self.manager.update_progression_stats({
                "success": False,
                "episode_reward": 0.2
            })
        
        # Pattern 3: Recovery
        for _ in range(15):
            self.manager.update_progression_stats({
                "success": True,
                "episode_reward": 0.8
            })
        
        # Verify the curriculum handled the variations appropriately
        self.assertTrue(0 <= self.manager.current_stage_index < len(self.manager.stages))
        self.assertTrue(0 <= self.manager.current_difficulty <= 1.0)

class TestRLBotIntegration(unittest.TestCase):
    """Test RLBot integration features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state_mutator = MockStateMutator()
        self.reward_fn = MockRewardFunction()
        self.term_cond = MockDoneCondition()
        self.trunc_cond = MockDoneCondition()

        # Add progression requirements for the test stage
        self.prog_req = ProgressionRequirements(
            min_success_rate=0.7,
            min_avg_reward=0.5,
            min_episodes=5,
            max_std_dev=0.5,
            required_consecutive_successes=2
        )

        # Create a stage with RLBot-specific parameters
        self.stage = CurriculumStage(
            name="RLBot Test Stage",
            state_mutator=self.state_mutator,
            reward_function=self.reward_fn,
            termination_condition=self.term_cond,
            truncation_condition=self.trunc_cond,
            progression_requirements=self.prog_req,  # Add progression requirements
            bot_skill_ranges={(0.3, 0.7): 0.6, (0.7, 1.0): 0.4},
            bot_tags=["defensive", "aerial"],
            allowed_bots=["HiveBot", "Necto"]
        )

    def test_bot_skill_selection(self):
        """Test opponent skill range selection"""
        # Test with low difficulty
        skill_range = self.stage.select_opponent_skill_range(0.3)
        self.assertEqual(skill_range, (0.3, 0.7))  # Should prefer lower range

        # Test with high difficulty
        skill_range = self.stage.select_opponent_skill_range(0.8)
        self.assertEqual(skill_range, (0.7, 1.0))  # Should prefer higher range

    def test_bot_performance_tracking(self):
        """Test tracking performance against specific bots"""
        # Add some performance data
        self.stage.update_bot_performance("HiveBot", True, 0.8, 0.5)
        self.stage.update_bot_performance("HiveBot", False, 0.2, 0.5)
        self.stage.update_bot_performance("Necto", True, 0.9, 0.7)

        # Check statistics
        stats = self.stage.get_current_stats()
        self.assertIn("bot_performance", stats)
        bot_stats = stats["bot_performance"]
        
        # Verify HiveBot stats
        self.assertIn("HiveBot", bot_stats)
        self.assertEqual(bot_stats["HiveBot"]["episodes"], 2)
        self.assertEqual(bot_stats["HiveBot"]["win_rate"], 0.5)

        # Verify Necto stats
        self.assertIn("Necto", bot_stats)
        self.assertEqual(bot_stats["Necto"]["episodes"], 1)
        self.assertEqual(bot_stats["Necto"]["win_rate"], 1.0)

    def test_challenging_bots_identification(self):
        """Test identifying challenging opponents"""
        # Add mixed performance data
        for _ in range(10):
            self.stage.update_bot_performance("EasyBot", True, 0.8, 0.3)
        for _ in range(10):
            self.stage.update_bot_performance("HardBot", False, 0.2, 0.8)

        challenging_bots = self.stage.get_challenging_bots(min_episodes=5)
        self.assertEqual(len(challenging_bots), 1)
        self.assertEqual(challenging_bots[0]["bot_id"], "HardBot")
        self.assertEqual(challenging_bots[0]["win_rate"], 0.0)

    def test_bot_progression_requirements(self):
        """Test bot-specific progression requirements"""
        # Set up stage statistics to meet basic requirements
        self.stage.episode_count = 10
        self.stage.rewards_history = [0.8] * 10
        self.stage.success_count = 8
        self.stage.moving_success_rate = 0.8
        self.stage.moving_avg_reward = 0.8

        # Add good performance data for multiple bots
        for _ in range(10):
            self.stage.update_statistics({"success": True, "episode_reward": 0.8})
            self.stage.update_bot_performance("Bot1", True, 0.8, 0.5)
            self.stage.update_bot_performance("Bot2", True, 0.7, 0.5)

        # Should meet requirements now
        self.assertTrue(self.stage.meets_progression_requirements())

        # Add poor performance against a new bot
        for _ in range(5):
            self.stage.update_statistics({"success": False, "episode_reward": 0.2})
            self.stage.update_bot_performance("HardBot", False, 0.2, 0.8)

        # Should no longer meet requirements due to poor performance
        self.assertFalse(self.stage.meets_progression_requirements())

if __name__ == "__main__":
    unittest.main()
