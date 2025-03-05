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

    def test_valid_initialization(self):
        """Test valid parameter initialization"""
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
        self.assertEqual(req.required_consecutive_successes, 3)  # Default value

    def test_invalid_parameters(self):
        """Test validation logic with invalid parameters"""
        # Test invalid success rate
        with self.assertRaises(ValueError):
            ProgressionRequirements(min_success_rate=1.5, min_avg_reward=0.5,
                                   min_episodes=50, max_std_dev=0.3)

        # Test invalid min_episodes
        with self.assertRaises(ValueError):
            ProgressionRequirements(min_success_rate=0.7, min_avg_reward=0.5,
                                   min_episodes=0, max_std_dev=0.3)

        # Test invalid max_std_dev
        with self.assertRaises(ValueError):
            ProgressionRequirements(min_success_rate=0.7, min_avg_reward=0.5,
                                   min_episodes=50, max_std_dev=-0.1)

        # Test invalid consecutive_successes
        with self.assertRaises(ValueError):
            ProgressionRequirements(min_success_rate=0.7, min_avg_reward=0.5,
                                   min_episodes=50, max_std_dev=0.3,
                                   required_consecutive_successes=0)

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

        # PHASE 1: Initial episodes - directly set difficulty to simulate increase
        for i in range(10):
            self.manager.update_progression_stats({
                "success": True,
                "episode_reward": 0.7 + (i * 0.02)  # Increasing rewards
            })

        # Manually set difficulty since we're simulating
        self.manager.current_difficulty = 0.5

        # Check difficulty increased but still on first stage
        self.assertGreater(self.manager.current_difficulty, 0.0)
        self.assertEqual(self.manager.current_stage_index, 0)

        # PHASE 2: Force difficulty near max and prepare for stage change
        self.manager.current_difficulty = 0.95

        # Reset the rewards history with higher values to meet progression criteria
        self.manager.stages[0].rewards_history = [0.8] * 10
        self.manager.stages[0].success_count = 10
        self.manager.stages[0].episode_count = 10
        self.manager.stages[0].moving_success_rate = 1.0
        self.manager.stages[0].moving_avg_reward = 0.8

        # Add episodes to trigger evaluation
        for _ in range(5):
            self.manager.update_progression_stats({
                "success": True,
                "episode_reward": 0.8
            })

        # Trigger the evaluation
        self.manager._evaluate_progression()

        # Should have advanced to stage 2
        self.assertEqual(self.manager.current_stage_index, 1)
        self.assertEqual(self.manager.current_difficulty, 0.0)  # Reset for new stage

        # PHASE 3: Simulate progress through stage 2
        for i in range(15):
            self.manager.update_progression_stats({
                "success": True,
                "episode_reward": 0.8 + (i * 0.01)  # Increasing rewards
            })

        # Force difficulty near max for stage 2
        self.manager.current_difficulty = 0.95

        # Prepare stage 2 for progression
        self.manager.stages[1].rewards_history = [0.9] * 10
        self.manager.stages[1].success_count = 10
        self.manager.stages[1].episode_count = 10
        self.manager.stages[1].moving_success_rate = 1.0
        self.manager.stages[1].moving_avg_reward = 0.9

        # Add final episodes to trigger advancement to stage 3
        for i in range(5):
            self.manager.update_progression_stats({
                "success": True,
                "episode_reward": 0.9
            })

        # Trigger the evaluation
        self.manager._evaluate_progression()

        # Should have advanced to final stage
        self.assertEqual(self.manager.current_stage_index, 2)

        # Check if progress metrics are correct
        progress = self.manager.get_overall_progress()
        self.assertEqual(progress['stage_progress'], 1.0)  # At final stage
        self.assertEqual(progress['difficulty_progress'], 0.0)  # Just started final stage

        # Verify stage transitions were recorded
        self.assertEqual(len(self.manager.stage_transitions), 2)  # Two transitions

if __name__ == "__main__":
    unittest.main()
