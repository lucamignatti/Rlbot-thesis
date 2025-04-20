"""Manual curriculum stage selection for RLBot."""

from typing import List, Dict, Any, Optional
import collections
from .base import CurriculumStage

class ManualCurriculumManager:
    """
    A simplified curriculum manager that runs a single specified stage without automatic progression.
    This replaces the automatic curriculum system with a manual approach while maintaining stage
    configuration and bot support.
    """
    def __init__(
        self,
        stages: List[CurriculumStage],
        stage_index: int = 0,
        debug: bool = False,
        use_wandb: bool = True
    ):
        """
        Initialize the manual curriculum manager.
        
        Args:
            stages: List of all available curriculum stages
            stage_index: Index of the stage to run (0-based)
            debug: Whether to print debug information
            use_wandb: Whether to use Weights & Biases for logging
        """
        if not stages:
            raise ValueError("At least one curriculum stage must be provided")
            
        self.stages = stages
        self.debug = debug
        self.use_wandb = use_wandb
        self.trainer = None
        
        # Set the current stage based on the provided index
        if stage_index >= len(stages):
            print(f"WARNING: Stage index {stage_index} out of range (max: {len(stages)-1}). Using last stage.")
            self.current_stage_index = len(stages) - 1
        else:
            self.current_stage_index = stage_index
            
        self.current_stage = stages[self.current_stage_index]
        
        # Initialize tracking variables (minimally required for compatibility)
        self.total_episodes = 0
        self.current_difficulty = 0.0
        self.difficulty_level = 0.0
        
        # Additional variables needed for compatibility
        self.completed_stages = collections.deque(maxlen=50)
        self.episodes_in_current_stage = 0
        self._last_wandb_step = 0
        
        print(f"Manual curriculum mode: Running stage {self.current_stage_index} '{self.current_stage.name}'")
        
        # No initialization needed - CurriculumStage doesn't have an initialize method
    
    def register_trainer(self, trainer):
        """Register a trainer object for hyperparameter adjustments"""
        if self.trainer != trainer:
            # Use weakref to avoid circular reference memory leaks
            import weakref
            self.trainer = weakref.proxy(trainer) if trainer is not None else None
            
            # Only register back with trainer if not already registered
            if hasattr(trainer, 'register_curriculum_manager'):
                if not hasattr(trainer, '_curriculum_manager') or trainer._curriculum_manager != self:
                    trainer.register_curriculum_manager(self)
    
    def get_environment_config(self):
        """Get the environment configuration for the current stage."""
        # Add difficulty_level=0.0 parameter required by the API
        return self.current_stage.get_environment_config(difficulty_level=self.difficulty_level)
    
    def update_progression_stats(self, episode_rewards, success, timeout=False, env_id=0):
        """
        Update progression statistics (track episodes but don't progress).
        
        Args:
            episode_rewards: The rewards obtained in the episode
            success: Whether the episode was successful
            timeout: Whether the episode timed out
            env_id: The environment ID
        """
        self.total_episodes += 1
        self.episodes_in_current_stage += 1
        
        # Update stage statistics (but don't trigger progression)
        episode_data = {
            "episode_reward": episode_rewards,
            "success": success,
            "timeout": timeout
        }
        self.current_stage.update_statistics(episode_data)
        
        # Log to WandB if enabled
        if self.use_wandb and self.total_episodes % 10 == 0:
            self._log_to_wandb()
    
    def _log_to_wandb(self):
        """Log curriculum statistics to WandB."""
        if not self.use_wandb:
            return
            
        try:
            import wandb
            if not wandb.run:
                return
                
            # Get the current training step if possible
            step = self._get_current_step()
            if step is None:
                return
                
            # Log current stage info
            stage_stats = self.current_stage.get_statistics()
            for key, value in stage_stats.items():
                wandb.log({f"curriculum/{key}": value}, step=step)
                
            # Log general curriculum info
            wandb.log({
                "curriculum/current_stage": self.current_stage_index,
                "curriculum/current_stage_name": self.current_stage.name,
                "curriculum/total_episodes": self.total_episodes,
            }, step=step)
            
            self._last_wandb_step = step
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Error logging to wandb: {e}")
    
    def _get_current_step(self) -> Optional[int]:
        """Get current training step for wandb logging."""
        if self.trainer is None:
            return self.total_episodes
            
        try:
            if hasattr(self.trainer, 'total_env_steps'):
                return self.trainer.total_env_steps
            else:
                return self.total_episodes
        except:
            return self.total_episodes
    
    def requires_bots(self):
        """Check if the current stage requires bots."""
        # Check if current stage is RLBotSkillStage (we can't import it directly due to circular imports)
        return self.current_stage.__class__.__name__ in ['RLBotSkillStage']
    
    def get_curriculum_stats(self):
        """Get curriculum statistics."""
        return {
            "current_stage_index": self.current_stage_index,
            "current_stage_name": self.current_stage.name,
            "total_stages": len(self.stages),
            "difficulty_level": self.difficulty_level,
            "progress": 0.0  # Manual mode has no progression
        }
    
    # Additional methods needed for compatibility
    def get_state(self):
        """Get the current curriculum state for saving."""
        return {
            "current_stage_index": self.current_stage_index,
            "total_episodes": self.total_episodes,
            "current_stage_stats": self.current_stage.get_statistics(),
        }
    
    def load_state(self, state):
        """Load a saved curriculum state."""
        if state and "current_stage_index" in state:
            # Only loading statistics, not changing stages in manual mode
            self.total_episodes = state.get("total_episodes", 0)
            
            if self.debug:
                print(f"[DEBUG] Manual curriculum loaded state: episodes={self.total_episodes}")
                
    def _adjust_hyperparameters(self):
        """Dummy method for compatibility."""
        pass
        
    def _evaluate_progression(self):
        """Dummy method for compatibility - always returns False in manual mode."""
        return False
        
    def get_stage_progress(self):
        """Get progress within the current stage (dummy implementation)."""
        return 0.0
        
    def get_overall_progress(self):
        """Get overall curriculum progress (dummy implementation)."""
        return self.current_stage_index / max(1, len(self.stages))
        
    def save_curriculum(self, path):
        """Compatibility method for saving curriculum state."""
        pass
        
    def load_curriculum(self, path):
        """Compatibility method for loading curriculum state."""
        pass