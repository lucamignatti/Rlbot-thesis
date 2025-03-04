from typing import List, Dict, Any, Tuple
from rlgym.api import AgentID, StateType, ObsType
from rlgym.rocket_league.obs_builders import DefaultObs
import numpy as np


class ActionStacker:
    """Optimized ActionStacker with efficient operations"""
    def __init__(self, stack_size=5, action_size=8):
        self.stack_size = stack_size
        self.action_size = action_size
        self.agent_action_history = {}

    def reset_agent(self, agent_id):
        """Reset action history for an agent"""
        self.agent_action_history[agent_id] = np.zeros((self.stack_size, self.action_size), dtype=np.float32)

    def add_action(self, agent_id, action):
        """Add new action to agent's history - optimized version"""
        if agent_id not in self.agent_action_history:
            self.reset_agent(agent_id)

        # Use direct array operations instead of np.roll
        history = self.agent_action_history[agent_id]
        history[:-1] = history[1:]
        history[-1] = action

    def get_stacked_actions(self, agent_id):
        """Get stacked history for an agent"""
        if agent_id not in self.agent_action_history:
            self.reset_agent(agent_id)
        # Using ravel() is faster than flatten() since it avoids copy when possible
        return self.agent_action_history[agent_id].ravel()


class StackedActionsObs(DefaultObs):
    """Extends DefaultObs to include stacked previous actions"""
    def __init__(self, action_stacker, zero_padding=2):
        super().__init__(zero_padding=zero_padding)
        self.action_stacker = action_stacker

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        """Reset method required by RLGym API"""
        super().reset(agents, initial_state, shared_info)
        # Reset action history for all agents
        for agent in agents:
            self.action_stacker.reset_agent(agent)

    def get_obs_space(self, agent: AgentID) -> Tuple[int]:
        """Get the observation space including stacked actions"""
        # Get base observation space from parent
        base_obs_space = super().get_obs_space(agent)

        # Print type and value for debugging
        print(f"Base obs_space: {base_obs_space} (type: {type(base_obs_space)})")

        # Calculate total observation size
        stacked_actions_size = self.action_stacker.stack_size * self.action_stacker.action_size
        print(f"Stacked actions size: {stacked_actions_size}")

        # Handle different types of base_obs_space
        if isinstance(base_obs_space, tuple):
            if len(base_obs_space) > 0:
                base_size = base_obs_space[0]
                print(f"Base size from tuple: {base_size} (type: {type(base_size)})")
            else:
                base_size = 542  # Default fallback for DefaultObs
                print(f"Empty tuple, using default: {base_size}")
        else:
            # Try to convert to int, or use default if that fails
            try:
                base_size = int(base_obs_space)
                print(f"Converted to int: {base_size}")
            except (ValueError, TypeError):
                print(f"Conversion failed, using default")
                base_size = 542  # Default fallback for DefaultObs

        # Ensure base_size is an integer
        if not isinstance(base_size, int):
            print(f"Warning: base_size is not an integer: {base_size} (type: {type(base_size)})")
            base_size = 542  # Default fallback

        total_size = base_size + stacked_actions_size

        return (total_size,)

    def build_obs(self, agents: List[AgentID], state: StateType, shared_info: Dict[str, Any]) -> Dict[AgentID, ObsType]:
        """Build observations for all agents"""
        # Get standard observations from parent class
        observations = super().build_obs(agents, state, shared_info)

        # Add stacked actions to each agent's observation
        augmented_observations = {}
        for agent_id in agents:
            obs = observations[agent_id]
            stacked_actions = self.action_stacker.get_stacked_actions(agent_id)
            augmented_observations[agent_id] = np.concatenate([obs, stacked_actions])

        return augmented_observations
