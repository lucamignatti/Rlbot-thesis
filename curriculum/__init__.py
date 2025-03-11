"""RLBot curriculum learning package."""
from .base import CurriculumManager, CurriculumStage, ProgressionRequirements
from .curriculum import create_curriculum
from .rlbot import RLBotSkillStage

__all__ = [
    'CurriculumManager',
    'CurriculumStage', 
    'ProgressionRequirements',
    'create_skill_based_curriculum'
]