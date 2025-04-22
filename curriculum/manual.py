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
        use_wandb: bool = True,
        memory_limit_gb: float = 32.0,  # Default memory limit in GB
        aggressive_gc: bool = True      # Enable aggressive garbage collection
    ):
        """
        Initialize the manual curriculum manager.

        Args:
            stages: List of all available curriculum stages
            stage_index: Index of the stage to run (0-based)
            debug: Whether to print debug information
            use_wandb: Whether to use Weights & Biases for logging
            memory_limit_gb: Memory limit in GB before forcing garbage collection
            aggressive_gc: Whether to enable aggressive garbage collection
        """
        if not stages:
            raise ValueError("At least one curriculum stage must be provided")

        self.stages = stages
        self.debug = debug
        self.use_wandb = use_wandb
        self.trainer = None
        self.memory_limit_gb = memory_limit_gb
        self.aggressive_gc = aggressive_gc
        self.last_memory_check = 0

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
        print(f"Memory management: limit={memory_limit_gb:.1f}GB, aggressive_gc={aggressive_gc}")

        # Initial memory cleanup
        import gc
        gc.collect()
        
        # Configure garbage collector for more aggressive collection if enabled
        if aggressive_gc:
            gc.set_threshold(700, 10, 5)  # More aggressive than default (700, 10, 10)

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
        
        # Monitor memory usage and perform cleanup when needed
        self._monitor_memory()
        
        # Periodically trigger cleanup to prevent memory leaks
        # Run cleanup every 50 episodes (increased frequency)
        if self.total_episodes % 50 == 0:
            self._run_cleanup()

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
        
    def _monitor_memory(self):
        """Monitor memory usage and force cleanup if memory usage exceeds threshold."""
        # Only check memory every 10 episodes to avoid overhead
        if self.total_episodes - self.last_memory_check < 10:
            return
            
        self.last_memory_check = self.total_episodes
        
        try:
            # Try to use psutil for accurate memory measurements
            import psutil
            process = psutil.Process()
            memory_usage_gb = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert bytes to GB
            
            if self.debug and self.total_episodes % 100 == 0:
                print(f"[MEMORY] Current usage: {memory_usage_gb:.2f} GB (limit: {self.memory_limit_gb:.1f} GB)")
                
            # If memory exceeds threshold, force aggressive cleanup
            if memory_usage_gb > self.memory_limit_gb * 0.9:  # 90% of limit as warning threshold
                print(f"[MEMORY WARNING] High memory usage detected: {memory_usage_gb:.2f} GB")
                self._run_cleanup(aggressive=True)
                
                # If still above limit after cleanup, take more drastic measures
                memory_usage_gb = process.memory_info().rss / (1024 * 1024 * 1024)
                if memory_usage_gb > self.memory_limit_gb:
                    print(f"[MEMORY CRITICAL] Still using {memory_usage_gb:.2f} GB after cleanup!")
                    self._emergency_cleanup()
                else:
                    print(f"[MEMORY] Reduced to {memory_usage_gb:.2f} GB after cleanup")
                    
        except ImportError:
            # Fallback using gc module only
            import gc
            # Run collection if episodes are a multiple of 100
            if self.total_episodes % 100 == 0:
                gc.collect()
    
    def _emergency_cleanup(self):
        """Perform aggressive memory cleanup in critical situations."""
        print("[MEMORY EMERGENCY] Performing emergency memory reduction")
        
        import gc
        import sys
        
        # Clear all caches we can find
        gc.collect(2)  # Full collection with generation 2
        
        # If Python 3.9+, we can manually reduce memory usage more aggressively
        if hasattr(gc, 'collect') and callable(getattr(gc, 'collect', None)):
            for i in range(3):  # Run multiple collections
                gc.collect()
                
        # Reset rewards history in all stages
        for stage in self.stages:
            if hasattr(stage, 'rewards_history'):
                stage.rewards_history.clear()
            if hasattr(stage, 'cleanup'):
                stage.cleanup()
                
        # Clear any other large data structures
        if hasattr(self.current_stage, 'skill_modules'):
            for skill in self.current_stage.skill_modules:
                if hasattr(skill, 'rewards_history'):
                    skill.rewards_history.clear()
                if hasattr(skill, 'success_history'):
                    skill.success_history.clear()
                    
        print("[MEMORY EMERGENCY] Emergency cleanup complete")
        
    def _run_cleanup(self, aggressive=False):
        """Perform memory cleanup to prevent leaks.
        This is particularly important for SkillBasedCurriculumStage instances
        that maintain history of rewards and episode outcomes.
        
        Args:
            aggressive: If True, perform more aggressive memory cleanup
        """
        if self.debug:
            print(f"[CLEANUP] Running {'aggressive ' if aggressive else ''}cleanup for stage {self.current_stage.name}")
            
        # Call cleanup method if it exists
        if hasattr(self.current_stage, 'cleanup') and callable(self.current_stage.cleanup):
            self.current_stage.cleanup()
            
        # Force garbage collection to clean up any remaining references
        import gc
        
        # Clear module caches if in aggressive mode
        if aggressive:
            # Prune dictionaries more thoroughly
            i = 0
            for stage in self.stages:
                if hasattr(stage, 'rewards_history') and len(stage.rewards_history) > 100:
                    # Keep only the last 100 entries
                    temp = list(stage.rewards_history)[-100:]
                    stage.rewards_history.clear()
                    for item in temp:
                        stage.rewards_history.append(item)
                    i += 1
            
            if self.debug and i > 0:
                print(f"[CLEANUP] Pruned rewards history in {i} stages")
                
        # Run collection
        gc.collect()
