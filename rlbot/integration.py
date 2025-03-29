"""RLBot integration for RLGym environments."""
import os
import sys
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import multiprocessing as mp
from multiprocessing import Process, Pipe
import concurrent.futures
from collections import defaultdict
from collections import deque
import random
from curriculum import CurriculumStage, ProgressionRequirements
from curriculum.rlbot import RLBotSkillStage 
from .registry import RLBotPackRegistry
from rlgym.rocket_league.api import GameState
from rlgym.api import RLGym, StateMutator, RewardFunction, DoneCondition
from envs.vectorized import VectorizedEnv
from envs.rlbot_vectorized import RLBotVectorizedEnv

# Helper functions to manage bot compatibility
def load_bot_skills() -> Dict[str, float]:
    """Load skill mapping for validated bots"""
    skills = {}
    try:
        with open("bot_skills.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    bot, skill = line.split("=")
                    skills[bot.strip()] = float(skill)
    except FileNotFoundError:
        print("Warning: bot_skills.txt not found")
    return skills

def load_disabled_bots() -> set:
    """Load list of disabled/incompatible bots"""
    disabled = set()
    try:
        with open("disabled_bots.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith(":"):
                    disabled.add(line.split("(")[0].strip())  # Remove parenthetical notes
    except FileNotFoundError:
        print("Warning: disabled_bots.txt not found")
    return disabled

def is_bot_compatible(bot_name: str) -> bool:
    """Check if a bot is compatible with the curriculum"""
    disabled_bots = load_disabled_bots()
    skills = load_bot_skills()

    # Check if bot is explicitly disabled
    if bot_name in disabled_bots:
        return False

    # Check if bot is validated and has skill mapping
    if bot_name not in skills:
        return False

    return True

def get_bot_skill(bot_name: str) -> Optional[float]:
    """Get skill level for a bot if available"""
    skills = load_bot_skills()
    return skills.get(bot_name)

def get_compatible_bots(min_skill: float = 0.0, max_skill: float = 1.0) -> Dict[str, float]:
    """Get compatible bots within skill range"""
    skills = load_bot_skills()
    disabled = load_disabled_bots()

    compatible = {}
    for bot, skill in skills.items():
        if bot not in disabled and min_skill <= skill <= max_skill:
            compatible[bot] = skill

    return compatible
