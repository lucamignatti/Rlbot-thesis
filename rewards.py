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
        # Calculate reward for each agent
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        # Get the car and ball positions
        carpos = state.cars[agent].physics.position
        ballpos = state.ball.position
        # Calculate the distance between the car and the ball
        distance = np.linalg.norm([carpos[0] - ballpos[0], carpos[1] - ballpos[1], carpos[2] - ballpos[2]])
        # The reward is inversely proportional to the distance, so closer cars get higher rewards
        reward = 1/(distance+1)
        return float(reward)

class BallNetProximityReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        # Calculate reward for each agent
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        ball_pos = state.ball.position

        # We need to figure out which team the agent is on to know which net is the opponent's net.
        # We attempt to use a remainder-based approach on the assumption that agent IDs are integers.
        try:
            agent_id = int(agent) if isinstance(agent, str) else agent
            is_blue_team = agent_id % 2 == 0
        except (ValueError, TypeError):
            # If the remainder approach fails (e.g., agent IDs aren't integers),
            # we use the order of agents in the dictionary.
            agent_index = list(state.cars.keys()).index(agent)
            is_blue_team = agent_index < len(state.cars) / 2

        if is_blue_team:
            # The opponent's net is the orange goal.
            opponent_net_pos = np.array([common_values.ORANGE_GOAL_CENTER[0], 0, common_values.GOAL_HEIGHT/2])
        else:
            # The opponent's net is the blue goal.
            opponent_net_pos = np.array([common_values.BLUE_GOAL_CENTER[0], 0, common_values.GOAL_HEIGHT/2])

        # Calculate distance between ball and opponent's net.
        distance = np.linalg.norm(np.array(ball_pos) - opponent_net_pos)

        # We want to give a higher reward when the ball is closer to the opponent's net.
        # We use a reciprocal function to achieve this.
        reward = 1 / (distance + 1)

        return float(reward)
