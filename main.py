from rlgym.api import RLGym
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
import numpy as np
from rewards import ProximityReward
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor



#Parse args later

render = True

episodes = 100

processes = 4




def getenv():
    return RLGym(
            state_mutator=MutatorSequence(
                FixedTeamSizeMutator(blue_size=2, orange_size=2),
                KickoffMutator()
            ),
            obs_builder=DefaultObs(zero_padding=2),
            action_parser=RepeatAction(LookupTableAction(), repeats=8),
            reward_fn=CombinedReward(
                (GoalReward(), 12.),
                (TouchReward(), 3.),
                (ProximityReward(), 1.),
            ),
            termination_cond=GoalCondition(),
            truncation_cond=AnyCondition(
                TimeoutCondition(300.),
                NoTouchTimeoutCondition(30.)
            ),
            transition_engine=RocketSimEngine(),
            renderer=RLViserRenderer()
        )



def episode(env, render=False):
    obs_dict = env.reset()

    obs_space_dims = env.observation_space(env.agents[0])[1]

    action_space_dims = env.action_space(env.agents[0])[1]

    print(obs_space_dims)
    print(action_space_dims)

    terminated = False
    truncated = False

    states = {agent: [] for agent in env.agents}
    actions = {agent: [] for agent in env.agents}
    rewards = {agent: [] for agent in env.agents}
    next_states = {agent: [] for agent in env.agents}
    dones = {agent: [] for agent in env.agents}
    log_probs = {agent: [] for agent in env.agents}


    while not terminated or truncated:
        if render:
            env.render()
            time.sleep(6/120)


        actions = {}

        for agent_id, action_space in env.action_spaces.items():
            action = np.random.randint(action_space_dims, size=1)
            actions[agent_id] = action


        obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)

        for agent in reward_dict.keys():
            rewards[agent].append(reward_dict[agent])
            dones[agent].append(terminated_dict[agent] or truncated_dict[agent])
            next_states[agent].append(obs_dict[agent])


        truncated = True in list(truncated_dict.values())
        terminated = True in list(terminated_dict.values())

    return states, actions, rewards, next_states, dones, log_probs



def run_parallel_episodes(num_processes, envs):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:

        # Create tasks
        futures = []
        for i in range(num_processes):
            futures.append(executor.submit(episode, envs[i], True if i==0 and render else False))

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"An error occurred: {e}")

    return results

if __name__ == "__main__":
    envs = [getenv() for _ in range(processes)]

    results = run_parallel_episodes(processes, envs)

    print(results)
