from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np


class BallProximityReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        carpos = state.cars[agent].physics.position
        ballpos = state.ball.position
        distance = np.linalg.norm([carpos[0] - ballpos[0], carpos[1] - ballpos[1], carpos[2] - ballpos[2]])
        reward = 1/(distance+1)
        return float(reward)

class BallNetProximityReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        # Get ball position
        ball_pos = state.ball.position

        # Determine which team the agent is on
        # First, try to convert agent to int if it's a string
        try:
            agent_id = int(agent) if isinstance(agent, str) else agent
            is_blue_team = agent_id % 2 == 0
        except (ValueError, TypeError):
            # If conversion fails, use a different approach - assume first half of agents are blue
            agent_index = list(state.cars.keys()).index(agent)
            is_blue_team = agent_index < len(state.cars) / 2

        if is_blue_team:
            # If agent is blue team, opponent's net is orange (positive x)
            opponent_net_pos = np.array([common_values.ORANGE_GOAL_CENTER[0], 0, common_values.GOAL_HEIGHT/2])
        else:
            # If agent is orange team, opponent's net is blue (negative x)
            opponent_net_pos = np.array([common_values.BLUE_GOAL_CENTER[0], 0, common_values.GOAL_HEIGHT/2])

        # Calculate distance between ball and opponent's net
        distance = np.linalg.norm(np.array(ball_pos) - opponent_net_pos)

        # Reward is higher when ball is closer to opponent's net
        # Using a decreasing function of distance
        reward = 1 / (distance + 1)

        return float(reward)
