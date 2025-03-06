from typing import Dict, Any

DEFAULT_CURRICULUM_CONFIG = {
    "evaluation_window": 50,
    "max_rehearsal_stages": 2,
    "rehearsal_decay_factor": 0.6,
    "progress_thresholds": {
        "success_rate": 0.7,
        "avg_reward": 0.6
    }
}

BOT_SKILL_RANGES = {
    "beginner": (0.0, 0.3),
    "intermediate": (0.3, 0.6),
    "advanced": (0.6, 0.8),
    "expert": (0.8, 1.0)
}

# Progression requirements for each stage
STAGE_REQUIREMENTS = {
    "beginner": {
        "min_success_rate": 0.6,
        "min_avg_reward": 0.5,
        "min_episodes": 100,
        "max_std_dev": 0.3,
        "required_consecutive_successes": 3
    },
    "intermediate": {
        "min_success_rate": 0.65,
        "min_avg_reward": 0.6,
        "min_episodes": 150,
        "max_std_dev": 0.25,
        "required_consecutive_successes": 4
    },
    "advanced": {
        "min_success_rate": 0.7,
        "min_avg_reward": 0.7,
        "min_episodes": 200,
        "max_std_dev": 0.2,
        "required_consecutive_successes": 5
    }
}