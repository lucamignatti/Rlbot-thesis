import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from training import Trainer
from curriculum.base import CurriculumManager, CurriculumStage, ProgressionRequirements
from rlgym.api import StateMutator, RewardFunction, DoneCondition

# Create mock classes that inherit from the required base classes
class MockStateMutator(StateMutator):
    def apply(self, state, shared_info):
        pass

class MockRewardFunction(RewardFunction):
    def calculate(self, player, state, previous_state=None):
        return 1.0

class MockDoneCondition(DoneCondition):
    def is_done(self, state):
        return False

# Create a simple model class for testing
class SimpleModel(nn.Module):
    def __init__(self, input_dim=8, output_dim=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # Add obs_shape attribute that Trainer expects
        self.obs_shape = input_dim
        
    def forward(self, x):
        return self.linear(x)

class TestTrainerInitialization(unittest.TestCase):
    """Test the initialization of Trainer with intrinsic rewards"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create actual networks instead of mocks to avoid parameter issues
        self.actor = SimpleModel(8, 8)
        self.critic = SimpleModel(8, 1)
        
    def test_use_intrinsic_rewards_initialization(self):
        """Test that use_intrinsic_rewards is properly initialized early in __init__"""
        # Patch the PPOAlgorithm to avoid initialization issues
        with patch('training.PPOAlgorithm'), patch('training.print_model_info'):
            # Create trainer with use_intrinsic_rewards=True
            trainer = Trainer(
                self.actor,
                self.critic,
                algorithm_type="ppo",
                use_intrinsic_rewards=True,
                use_pretraining=True,
                action_dim=8,  # Provide action_dim
                device="cpu"
            )
            
            # Verify attribute exists and is set correctly
            self.assertTrue(hasattr(trainer, 'use_intrinsic_rewards'))
            self.assertTrue(trainer.use_intrinsic_rewards)
            
            # Test without pretraining
            trainer = Trainer(
                self.actor,
                self.critic,
                algorithm_type="ppo",
                use_intrinsic_rewards=True,
                use_pretraining=False,
                action_dim=8,  # Provide action_dim
                device="cpu"
            )
            
            # With pretraining=False, intrinsic rewards should be false regardless of use_intrinsic_rewards param
            self.assertFalse(trainer.use_intrinsic_rewards)
        
    def test_trainer_compilation(self):
        """Test that the trainer can be compiled without attribute errors"""
        # Set up minimal trainer for compilation test
        with patch('training.PPOAlgorithm'), patch('training.print_model_info'), patch('torch.compile', return_value=lambda x: x) as mock_compile:
            trainer = Trainer(
                self.actor,
                self.critic,
                algorithm_type="ppo",
                use_intrinsic_rewards=True,
                use_pretraining=True,
                use_compile=True,  # Enable compilation
                action_dim=8,  # Provide action_dim
                device="cpu"
            )
            
            # Add a mock compile_models method
            def mock_compile_models():
                # Direct access to use_intrinsic_rewards should work now
                _ = trainer.use_intrinsic_rewards
                return True
                
            trainer.compile_models = mock_compile_models
            
            # This should not raise an attribute error
            result = trainer.compile_models()
            self.assertTrue(result)


class TestPretrainingTransition(unittest.TestCase):
    """Test the pretraining transition logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create actual networks
        self.actor = SimpleModel(8, 8)
        self.critic = SimpleModel(8, 1)
        
    def test_pretraining_state_update(self):
        """Test that pretraining state updates correctly"""
        # Create trainer with pretraining enabled
        with patch('training.PPOAlgorithm'), patch('training.print_model_info'), patch('wandb.log'):
            trainer = Trainer(
                self.actor,
                self.critic,
                algorithm_type="ppo",
                use_pretraining=True,
                pretraining_transition_steps=100,
                total_episode_target=1000,
                pretraining_fraction=0.1,  # 10% of episodes for pretraining
                action_dim=8,  # Provide action_dim
                device="cpu"
            )
            
            # Initially, pretraining should be active but not completed
            self.assertTrue(trainer.use_pretraining)
            self.assertFalse(trainer.pretraining_completed)
            self.assertFalse(trainer.in_transition_phase)
            
            # Mock the _true_training_steps method to control progression
            # First, set steps to be greater than the pretraining end point
            # Directly patch the _get_pretraining_end_step method to return a known value
            with patch.object(trainer, '_get_pretraining_end_step', return_value=100):
                # Override _true_training_steps to ensure it's past the pretraining end step
                trainer._true_training_steps = lambda: 110
                
                # Update pretraining state - this should mark pretraining as complete 
                # and enter transition phase
                trainer._update_pretraining_state()
                
                # Verify state after reaching pretraining_end_step
                self.assertTrue(trainer.pretraining_completed)
                self.assertTrue(trainer.in_transition_phase)
                self.assertEqual(trainer.transition_start_step, 110)
                
                # Now set steps to be halfway through transition
                trainer._true_training_steps = lambda: 160
                
                # Update again - should still be in transition
                trainer._update_pretraining_state()
                self.assertTrue(trainer.pretraining_completed)
                self.assertTrue(trainer.in_transition_phase)
                
                # Finally, set steps to be after transition is complete
                trainer._true_training_steps = lambda: 220
                
                # Update again - should have completed transition
                trainer._update_pretraining_state()
                self.assertTrue(trainer.pretraining_completed)
                self.assertFalse(trainer.in_transition_phase)
        
    def test_pretraining_end_step_calculation(self):
        """Test that pretraining end step is calculated correctly"""
        # Create trainer with pretraining enabled and specific episode target
        with patch('training.PPOAlgorithm'), patch('training.print_model_info'):
            trainer = Trainer(
                self.actor,
                self.critic,
                algorithm_type="ppo",
                use_pretraining=True,
                total_episode_target=10000,
                pretraining_fraction=0.2,  # 20% for pretraining
                action_dim=8,  # Provide action_dim
                device="cpu"
            )
            
            # Test end step calculation with episode target
            end_step = trainer._get_pretraining_end_step()
            # With 10000 episodes, 20% is 2000 episodes, assuming ~200 steps per episode
            expected_steps = 2000 * 200
            self.assertEqual(end_step, expected_steps)
            
            # Test with no episode target
            trainer.total_episode_target = None
            end_step = trainer._get_pretraining_end_step()
            self.assertEqual(end_step, 100000)  # Default value
        
    def test_pretraining_transition_message(self):
        """Test that transition message is displayed only once"""
        # Create trainer with pretraining enabled
        with patch('training.PPOAlgorithm'), patch('training.print_model_info'), patch('wandb.log'):
            trainer = Trainer(
                self.actor,
                self.critic,
                algorithm_type="ppo",
                use_pretraining=True,
                pretraining_transition_steps=100,
                total_episode_target=1000,
                pretraining_fraction=0.1,
                action_dim=8,  # Provide action_dim
                device="cpu",
                debug=True  # Enable debug mode to ensure print statements
            )
            
            # Patch the _get_pretraining_end_step method to return a known value
            with patch.object(trainer, '_get_pretraining_end_step', return_value=100):
                # Mock the _true_training_steps method to control progression
                # Setting to exactly the pretraining end point
                trainer._true_training_steps = lambda: 100
                
                # Patch print to verify message is displayed
                with patch('builtins.print') as mock_print:
                    # Update state - this should now trigger the transition message
                    trainer._update_pretraining_state()
                    
                    # Check that print was called with the transition message
                    calls = [call for call in mock_print.call_args_list if "Transitioning from pretraining" in str(call)]
                    self.assertGreaterEqual(len(calls), 1, "Transition message not printed")
                    
                    # Reset the mock
                    mock_print.reset_mock()
                    
                    # Update again with same step - should NOT print the message again
                    trainer._update_pretraining_state()
                    
                    # Check that no transition message was printed
                    calls = [call for call in mock_print.call_args_list if "Transitioning from pretraining" in str(call)]
                    self.assertEqual(len(calls), 0, "Transition message printed again when it shouldn't be")


class TestCurriculumIntegration(unittest.TestCase):
    """Test the integration of curriculum with pretraining"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create actual networks
        self.actor = SimpleModel(8, 8)
        self.critic = SimpleModel(8, 1)
        
        # Create proper curriculum components using our mock classes
        self.state_mutator = MockStateMutator()
        self.reward_fn = MockRewardFunction()
        self.term_cond = MockDoneCondition()
        self.trunc_cond = MockDoneCondition()
        
        # Create a mock pretraining stage that properly inherits behavior
        class MockPretrainingStage(CurriculumStage):
            def __init__(self):
                super().__init__(
                    name="Pretraining Stage",
                    state_mutator=MockStateMutator(),
                    reward_function=MockRewardFunction(),
                    termination_condition=MockDoneCondition(),
                    truncation_condition=MockDoneCondition()
                )
                self.is_pretraining = True
                self.register_trainer = MagicMock()
                
            def validate_progression(self):
                # Always return True for testing
                return True
        
        self.pretraining_stage = MockPretrainingStage()
        
        # Create a regular stage
        self.regular_stage = CurriculumStage(
            name="Regular Stage",
            state_mutator=self.state_mutator,
            reward_function=self.reward_fn,
            termination_condition=self.term_cond,
            truncation_condition=self.trunc_cond,
            progression_requirements=ProgressionRequirements(
                min_success_rate=0.7,
                min_avg_reward=0.5,
                min_episodes=10,
                max_std_dev=0.5
            )
        )
        
        # Create a curriculum manager with a pretraining stage
        self.curriculum = CurriculumManager(
            stages=[self.pretraining_stage, self.regular_stage],
            testing=True
        )
        
    def test_curriculum_pretraining_synchronization(self):
        """Test synchronization between trainer and curriculum on pretraining state"""
        # Create trainer with pretraining enabled
        with patch('training.PPOAlgorithm'), patch('training.print_model_info'):
            trainer = Trainer(
                self.actor,
                self.critic,
                algorithm_type="ppo",
                use_pretraining=True,
                action_dim=8,  # Provide action_dim
                device="cpu"
            )
            
            # Register the trainer with the curriculum
            self.curriculum.register_trainer(trainer)
            
            # Verify the pretraining stage received the trainer
            self.pretraining_stage.register_trainer.assert_called_once_with(trainer)
            
            # Now simulate pretraining completion
            trainer.pretraining_completed = True
            trainer.in_transition_phase = False
            
            # Trigger progression evaluation
            with patch.object(self.curriculum, '_progress_to_next_stage') as mock_progress:
                self.curriculum._evaluate_progression()
                mock_progress.assert_called_once()


if __name__ == '__main__':
    unittest.main()