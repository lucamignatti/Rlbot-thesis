import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import random
from rlgym.api import StateMutator, RewardFunction, DoneCondition
from curriculum import ProgressionRequirements

class SkillModule:
    """A module representing a specific skill to be learned with configurable difficulty."""
    
    def __init__(
        self,
        name: str,
        state_mutator: StateMutator,
        reward_function: RewardFunction,
        termination_condition: DoneCondition,
        truncation_condition: DoneCondition,
        difficulty_params: Dict[str, Tuple[float, float]],
        success_threshold: float = 0.7
    ):
        self.name = name
        self.state_mutator = state_mutator
        self.reward_function = reward_function
        self.termination_condition = termination_condition
        self.truncation_condition = truncation_condition
        self.difficulty_params = difficulty_params
        self.success_threshold = success_threshold
        
        # Statistics tracking
        self.episode_count = 0
        self.success_count = 0
        self.rewards_history = []
        self.success_rate = 0.0
        
    def update_statistics(self, metrics: Dict[str, Any]):
        """Update skill statistics with new episode results"""
        self.episode_count += 1
        if metrics.get('success', False):
            self.success_count += 1
            
        self.rewards_history.append(metrics.get('episode_reward', 0.0))
        self.success_rate = self.success_count / max(1, self.episode_count)
        
    def get_config(self, difficulty: float) -> Dict[str, Any]:
        """Get difficulty-adjusted configuration"""
        config = {}
        difficulty = max(0.0, min(1.0, difficulty))  # Clamp between 0 and 1
        
        for param_name, (min_val, max_val) in self.difficulty_params.items():
            config[param_name] = min_val + difficulty * (max_val - min_val)
        
        return config

    def meets_mastery_criteria(self) -> bool:
        """Check if the skill meets mastery criteria"""
        min_episodes = 20  # Minimum episodes needed for reliable assessment
        if self.episode_count < min_episodes:
            return False
            
        # Check success rate meets threshold
        if self.success_rate < self.success_threshold:
            return False
            
        # Check for stability in recent performance
        recent_rewards = self.rewards_history[-10:]  # Last 10 episodes
        if len(recent_rewards) >= 10:
            std_dev = np.std(recent_rewards)
            if std_dev > 0.3:  # High variance indicates unstable performance
                return False
                
        return True


class SkillBasedCurriculumStage:
    """A curriculum stage that combines a base task with specific skill modules."""
    
    def __init__(
        self,
        name: str,
        base_task_state_mutator: StateMutator,
        base_task_reward_function: RewardFunction,
        base_task_termination_condition: DoneCondition,
        base_task_truncation_condition: DoneCondition,
        skill_modules: List[SkillModule],
        progression_requirements: ProgressionRequirements,
        base_task_prob: float = 0.6  # Probability of selecting base task vs skill
    ):
        self.name = name
        self.base_task_state_mutator = base_task_state_mutator
        self.base_task_reward_function = base_task_reward_function
        self.base_task_termination_condition = base_task_termination_condition
        self.base_task_truncation_condition = base_task_truncation_condition
        self.skill_modules = skill_modules
        self.progression_requirements = progression_requirements
        self.base_task_prob = base_task_prob
        
        # Statistics tracking for base task
        self.base_task_episodes = 0
        self.base_task_successes = 0
        self.base_task_rewards = []
        
    def select_task(self) -> Tuple[bool, Optional[SkillModule]]:
        """Select whether to do base task or which skill to practice.
        
        Returns:
            Tuple of (is_base_task, selected_skill)
            where selected_skill is None if is_base_task is True
        """
        if random.random() < self.base_task_prob:
            return True, None
            
        # Select a skill, weighting towards ones with lower success rates
        if not self.skill_modules:
            return True, None
            
        success_rates = np.array([1 - skill.success_rate for skill in self.skill_modules])
        weights = success_rates / success_rates.sum() if success_rates.sum() > 0 else None
        selected_skill = np.random.choice(self.skill_modules, p=weights)
        return False, selected_skill
        
    def get_environment_config(self, difficulty: float) -> Dict[str, Any]:
        """Get environment configuration based on selected task"""
        is_base_task, selected_skill = self.select_task()
        
        if is_base_task:
            return {
                "task_type": "base",
                "stage_name": self.name,
                "state_mutator": self.base_task_state_mutator,
                "reward_function": self.base_task_reward_function,
                "termination_condition": self.base_task_termination_condition,
                "truncation_condition": self.base_task_truncation_condition,
                "difficulty_level": difficulty
            }
        else:
            config = selected_skill.get_config(difficulty)
            return {
                "task_type": "skill",
                "stage_name": self.name,
                "skill_name": selected_skill.name,
                "state_mutator": selected_skill.state_mutator,
                "reward_function": selected_skill.reward_function,
                "termination_condition": selected_skill.termination_condition,
                "truncation_condition": selected_skill.truncation_condition,
                "difficulty_level": difficulty,
                "difficulty_params": config
            }
            
    def update_statistics(self, metrics: Dict[str, Any]) -> None:
        """Update statistics for either base task or skill module"""
        if metrics.get("is_base_task", True):
            self.base_task_episodes += 1
            if metrics.get("success", False):
                self.base_task_successes += 1
            self.base_task_rewards.append(metrics.get("episode_reward", 0.0))
        else:
            # Update specific skill module
            skill_name = metrics.get("skill_name")
            for skill in self.skill_modules:
                if skill.name == skill_name:
                    skill.update_statistics(metrics)
                    break
                    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for both base task and skills"""
        stats = {
            "base_task": {
                "episodes": self.base_task_episodes,
                "success_rate": self.base_task_successes / max(1, self.base_task_episodes),
                "avg_reward": np.mean(self.base_task_rewards) if self.base_task_rewards else 0.0
            },
            "skills": {skill.name: {
                "episodes": skill.episode_count,
                "success_rate": skill.success_rate,
                "avg_reward": np.mean(skill.rewards_history) if skill.rewards_history else 0.0
            } for skill in self.skill_modules}
        }
        return stats
        
    def meets_progression_requirements(self) -> bool:
        """Check if both base task and all skills meet progression requirements"""
        # Check base task requirements
        if self.base_task_episodes < self.progression_requirements.min_episodes:
            return False
            
        base_success_rate = self.base_task_successes / max(1, self.base_task_episodes)
        if base_success_rate < self.progression_requirements.min_success_rate:
            return False
            
        # Calculate base task average reward
        base_avg_reward = np.mean(self.base_task_rewards) if self.base_task_rewards else 0.0
        if base_avg_reward < self.progression_requirements.min_avg_reward:
            return False
            
        # Check each skill's mastery
        for skill in self.skill_modules:
            if skill.episode_count < self.progression_requirements.min_episodes:
                return False
            if skill.success_rate < self.progression_requirements.min_success_rate:
                return False
                
        return True