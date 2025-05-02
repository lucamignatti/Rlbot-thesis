#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import time
import os
from functools import partial
import inspect

# Import your custom implementations
from algorithms.dppo import DPPOAlgorithm
from model_architectures.simba_v2 import SimbaV2

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configuration
ENV_ID = "LunarLander-v3"  # A good benchmark environment with discrete actions
NUM_ENVS = 12  # Number of parallel environments
TOTAL_TIMESTEPS = 500_000  # Total timesteps for training
EVAL_FREQ = 10_000  # Frequency of evaluation during training
NUM_EVAL_EPISODES = 10  # Number of episodes for evaluation
LOG_DIR = "./ppo_comparison_results"
os.makedirs(LOG_DIR, exist_ok=True)

# Common hyperparameters
COMMON_PARAMS = {
    "learning_rate": 3e-4,  # Initial learning rate (will be decayed)
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "n_epochs": 10,
    "batch_size": 64,
}
# C51 distributional RL parameters
NUM_ATOMS = 101
V_MIN = -5.0  # Minimum possible value
V_MAX = 5.0   # Maximum possible value

# DPPO specific parameters - map to what DPPOAlgorithm expects
DPPO_PARAMS = {
    # Correctly map to DPPOAlgorithm parameters
    "lr_actor": COMMON_PARAMS["learning_rate"],
    "lr_critic": COMMON_PARAMS["learning_rate"],
    "gamma": COMMON_PARAMS["gamma"],
    "gae_lambda": COMMON_PARAMS["gae_lambda"],
    "epsilon_base": COMMON_PARAMS["clip_range"],  # Base clipping value
    "entropy_coef": COMMON_PARAMS["ent_coef"],
    "ppo_epochs": COMMON_PARAMS["n_epochs"],
    "batch_size": COMMON_PARAMS["batch_size"],
    "v_min": V_MIN,
    "v_max": V_MAX,
    "num_atoms": NUM_ATOMS,
    # Additional required parameters with default values
    "action_space_type": "discrete",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# SimbaV2 parameters
SIMBA_PARAMS = {
    "hidden_dim": 256,
    "num_blocks": 4,
    "shift_constant": 3.0,
}


# Create the support for distributional value function
SUPPORTS = torch.linspace(V_MIN, V_MAX, NUM_ATOMS)

# Flag to skip testing SimbaV2+SB3 if there are integration issues
SKIP_SIMBA_SB3 = True  # Set to False to enable the test

# Make sure the script handles both old and new gym API
# Add workaround for gym vs gymnasium API differences
def handle_env_step(env_step_result):
    """Handle differences between gym and gymnasium step returns"""
    if len(env_step_result) == 5:  # gymnasium
        next_obs, reward, terminated, truncated, info = env_step_result
        done = terminated or truncated
        # print(f"Gymnasium API: done type = {type(done)}")
        return next_obs, reward, done, info
    else:  # gym
        next_obs, reward, done, info = env_step_result
        # print(f"Gym API: done type = {type(done)}")
        return next_obs, reward, done, info

def handle_env_reset(env_reset_result):
    """Handle differences between gym and gymnasium reset returns"""
    if isinstance(env_reset_result, tuple) and len(env_reset_result) == 2:  # gymnasium
        obs, info = env_reset_result
        return obs
    else:  # gym
        return env_reset_result  # obs

# Create custom Sequential wrapper compatible with DPPO
class DPPOCompatibleSequential(nn.Module):
    def __init__(self, sequential_model):
        super().__init__()
        self.model = sequential_model
        # The last layer contains the output features
        self.feature_extraction = nn.Sequential(*list(sequential_model.children())[:-1])
        self.output_head = list(sequential_model.children())[-1]

    def forward(self, x, return_features=False, return_actor=True, return_critic=True, **kwargs):
        """Forward pass with optional feature return to be compatible with DPPO"""
        # Process through all but the last layer to get features
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]  # Take first element if list/tuple passed

        if return_features:
            # Extract features from all but last layer
            features = self.feature_extraction(x)
            # Get final output
            output = self.output_head(features)
            return output, features
        else:
            # Regular forward pass
            return self.model(x)
# ======= Helpers and Custom Implementations ======
# DPPOCompatibleSequential is retained as it's used elsewhere

# ======= Custom policies for SB3 ======

# ======= Distributional Critic Policy for SB3 ======
class DistributionalActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        **kwargs
    ):
        # Initialize the standard SB3 policy
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )

        # Replace the value head with a distributional one (C51 approach)
        features_dim = self.mlp_extractor.latent_dim_vf
        self.value_net = nn.Linear(features_dim, NUM_ATOMS)

        # Register the supports as a buffer (so it's saved/loaded with the model)
        self.register_buffer("supports", SUPPORTS)

    def forward(self, obs, deterministic=False):
        """
        Forward pass in the neural network.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Actor (policy) network
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

        # Value network (distributional C51)
        value_logits = self.value_net(latent_vf)
        # Convert to probabilities
        value_probs = F.softmax(value_logits, dim=1)
        # Compute expected value
        expected_value = torch.sum(value_probs * self.supports.view(1, -1), dim=1, keepdim=True)

        return actions, value_probs, log_probs, expected_value

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy, given the observations.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # Distributional value (C51)
        value_logits = self.value_net(latent_vf)
        # Convert to probabilities
        value_probs = F.softmax(value_logits, dim=1)
        # Compute expected value
        expected_value = torch.sum(value_probs * self.supports.view(1, -1), dim=1, keepdim=True)

        return value_probs, log_probs, entropy, expected_value

# ======= SimbaV2 Integration with SB3 ======
class SimbaV2DistributionalPolicy(ActorCriticPolicy):
    """
    Policy that uses SimbaV2 architecture with distributional critic for SB3.
    Inherits from ActorCriticPolicy to be compatible with SB3.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        # Don't add any custom parameters that SB3 doesn't expect
        *args,
        **kwargs
    ):
        # We need to simplify the network for SB3 compatibility
        # Let SB3 handle the basics, but we'll override methods as needed
        # No need for special net_arch as we'll handle that ourselves
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )

        # Create our SimbaV2 networks after parent init
        obs_shape = observation_space.shape[0]

        # Create SimbaV2 actor network
        if isinstance(action_space, gym.spaces.Discrete):
            action_shape = action_space.n
        else:
            action_shape = action_space.shape[0]

        # Initialize SimbaV2 network for actor
        self.simba_actor = SimbaV2(
            obs_shape=obs_shape,
            action_shape=action_shape,
            is_critic=False,
            num_atoms=None,
            **SIMBA_PARAMS
        )

        # Initialize SimbaV2 network for critic with distributional output
        self.simba_critic = SimbaV2(
            obs_shape=obs_shape,
            action_shape=1,
            is_critic=True,
            num_atoms=NUM_ATOMS,
            **SIMBA_PARAMS
        )

        # Register support for distributional calculations
        self.register_buffer("value_supports", SUPPORTS)

    def _predict(self, observation, deterministic=False):
        """
        Get action distribution from SimbaV2 actor.
        """
        action_logits = self.simba_actor(observation)

        # Use SB3's action distribution system
        # Handle different distribution types correctly
        if isinstance(self.action_dist, torch.distributions.categorical.Categorical) or hasattr(self.action_dist, 'categorical_distribution'):
            # Discrete actions - just pass the logits
            distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        else:
            # Continuous actions - need to interpret as mean and log_std
            # Extract mean and log_std
            action_dim = self.action_space.shape[0]

            # Check if action_logits is the right shape for splitting
            if action_logits.shape[-1] == action_dim * 2:
                mu, log_std = torch.chunk(action_logits, 2, dim=-1)
                distribution = self.action_dist.proba_distribution(mu, log_std)
            else:
                # If not splittable, treat as mean only
                distribution = self.action_dist.proba_distribution(action_logits)

        return distribution

    def forward(self, obs, deterministic=False):
        """
        Forward pass in the neural network - reimplemented to use SimbaV2.
        SB3 expects actions, values, log_probs as return values.
        """
        try:
            # Get distribution from our predict method
            distribution = self._predict(obs)

            # Get actions according to the distribution
            actions = distribution.get_actions(deterministic=deterministic)
            log_probs = distribution.log_prob(actions)

            # Get value distribution from critic network
            value_logits = self.simba_critic(obs)
            # Convert to probabilities
            value_probs = F.softmax(value_logits, dim=1)
            # Calculate expected value (SB3 expects scalar values)
            expected_value = torch.sum(value_probs * self.value_supports.view(1, -1), dim=1, keepdim=True)

            return actions, expected_value, log_probs

        except Exception as e:
            # Fallback to parent method if our custom implementation fails
            print(f"SimbaV2 policy error: {e}, falling back to parent implementation")
            return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy - reimplemented to use SimbaV2.
        SB3 expects values, log_probs, entropy as return values.
        """
        try:
            # Get distribution from our predict method
            distribution = self._predict(obs)
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy()

            # Get value distribution from critic network
            value_logits = self.simba_critic(obs)
            # Convert to probabilities
            value_probs = F.softmax(value_logits, dim=1)
            # Calculate expected value (SB3 expects scalar values)
            expected_value = torch.sum(value_probs * self.value_supports.view(1, -1), dim=1, keepdim=True)

            return expected_value, log_probs, entropy

        except Exception as e:
            # Fallback to parent method if our custom implementation fails
            print(f"SimbaV2 policy evaluation error: {e}, falling back to parent implementation")
            return super().evaluate_actions(obs, actions)

# ======= Custom DPPO Wrapper ======
class CustomDPPOWrapper:
    def __init__(self, env, model_name, common_params):
        self.env = env
        self.model_name = model_name
        self.common_params = common_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rewards_history = []
        # Store initial learning rates for scheduler
        self.initial_lr_actor = DPPO_PARAMS["lr_actor"]
        self.initial_lr_critic = DPPO_PARAMS["lr_critic"]
        self.lr_scheduler = None

    def setup(self):
        """Configure environment and algorithm"""
        # Determine observation and action space
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        # Create actor and critic networks
        obs_shape = obs_space.shape[0]

        if isinstance(act_space, gym.spaces.Discrete):
            action_shape = act_space.n
            action_space_type = "discrete"
        else:
            action_shape = act_space.shape[0]
            action_space_type = "continuous"

        # Create actor (policy) network
        actor_network = SimbaV2(
            obs_shape=obs_shape,
            action_shape=action_shape,
            is_critic=False,
            num_atoms=None,
            **SIMBA_PARAMS,
            device=self.device
        ).to(self.device)

        # Create critic (value) network with distributional output
        critic_network = SimbaV2(
            obs_shape=obs_shape,
            action_shape=1,  # Value function has single output
            is_critic=True,
            num_atoms=NUM_ATOMS,  # For distributional value function (C51)
            **SIMBA_PARAMS,
            device=self.device
        ).to(self.device)

        # Initialize DPPO algorithm with the correct parameters
        self.alg = DPPOAlgorithm(
            actor=actor_network,
            critic=critic_network,
            action_space_type=action_space_type,
            action_dim=action_shape if action_space_type == "continuous" else None,
            device=self.device,
            lr_actor=self.initial_lr_actor,
            lr_critic=self.initial_lr_critic,
            gamma=DPPO_PARAMS["gamma"],
            gae_lambda=DPPO_PARAMS["gae_lambda"],
            epsilon_base=DPPO_PARAMS["epsilon_base"],
            entropy_coef=DPPO_PARAMS["entropy_coef"],
            ppo_epochs=DPPO_PARAMS["ppo_epochs"],
            batch_size=DPPO_PARAMS["batch_size"],
            v_min=DPPO_PARAMS["v_min"],
            v_max=DPPO_PARAMS["v_max"],
            num_atoms=DPPO_PARAMS["num_atoms"]
        )

        # Set up learning rate scheduler using linear decay
        # For DPPO we'll use a lambda function scheduler that linearly decays from initial LR to min_lr
        self.min_lr_factor = 0.1  # Final learning rate will be 10% of initial rate
        # We'll update the scheduler in the train loop

    def train(self, total_timesteps, eval_freq, num_eval_episodes):
        """Train the algorithm and evaluate periodically"""
        timesteps_so_far = 0
        episode_rewards = []
        current_episode_reward = 0

        obs = handle_env_reset(self.env.reset())

        while timesteps_so_far < total_timesteps:
            # Calculate current learning rate based on progress
            progress = min(1.0, timesteps_so_far / total_timesteps)
            # Linear decay from initial_lr to initial_lr * min_lr_factor
            current_lr_actor = self.initial_lr_actor * (1 - progress * (1 - self.min_lr_factor))
            current_lr_critic = self.initial_lr_critic * (1 - progress * (1 - self.min_lr_factor))

            # Update learning rates in optimizer
            for param_group in self.alg.optimizer.param_groups:
                param_group['lr'] = current_lr_actor  # Use actor LR for all parameters since DPPO uses a single optimizer

            # Get action and other values from policy
            action_result, log_prob_result, value_result = self.alg.get_action(
                torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            )

            # Step environment
            env_step_result = self.env.step(action_result.cpu().numpy())
            next_obs, reward, done, _ = handle_env_step(env_step_result)

            # Store experience with correct method signature
            indices = self.alg.store_initial_batch(
                torch.FloatTensor(obs).to(self.device).unsqueeze(0),
                action_result.unsqueeze(0) if action_result.dim() == 0 else action_result,
                log_prob_result.unsqueeze(0) if log_prob_result.dim() == 0 else log_prob_result,
                value_result.unsqueeze(0) if value_result.dim() == 0 else value_result
            )

            # Update rewards and dones with indices
            self.alg.update_rewards_dones_batch(
                indices,
                torch.tensor([reward], dtype=torch.float32).to(self.device),
                torch.tensor([bool(done)], dtype=torch.bool).to(self.device)  # Convert to bool explicitly
            )

            current_episode_reward += reward

            # Update observation
            obs = next_obs

            # Handle episode termination
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                obs = handle_env_reset(self.env.reset())

            # Update policy after collecting enough data
            if timesteps_so_far > 0 and timesteps_so_far % self.common_params.get("batch_size", 64) == 0:
                self.alg.update()

            timesteps_so_far += 1

            # Evaluate policy
            if timesteps_so_far % eval_freq == 0:
                mean_reward = self._evaluate(num_eval_episodes)
                self.rewards_history.append((timesteps_so_far, mean_reward))
                print(f"{self.model_name} - Step: {timesteps_so_far}, Mean Reward: {mean_reward:.2f}")

        # Final evaluation
        mean_reward = self._evaluate(num_eval_episodes)
        self.rewards_history.append((total_timesteps, mean_reward))
        print(f"{self.model_name} - Final Mean Reward: {mean_reward:.2f}")

        return self.rewards_history

    def _evaluate(self, num_episodes):
        """Evaluate current policy"""
        eval_env = gym.make(ENV_ID)
        episode_rewards = []

        for _ in range(num_episodes):
            obs = handle_env_reset(eval_env.reset())
            done = False
            episode_reward = 0

            while not done:
                with torch.no_grad():
                    action, _, _ = self.alg.get_action(
                        torch.FloatTensor(obs).to(self.device).unsqueeze(0),
                        deterministic=True
                    )

                env_step_result = eval_env.step(action.cpu().numpy())
                obs, reward, done, _ = handle_env_step(env_step_result)
                episode_reward += reward

            episode_rewards.append(episode_reward)

        eval_env.close()
        return np.mean(episode_rewards)

    def save(self, path):
        """Save model state"""
        if hasattr(self.alg, "get_state_dict"):
            torch.save(self.alg.get_state_dict(), path)


# ======= SB3 PPO Wrapper ======
class SB3Wrapper:
    def __init__(self, env_id, num_envs, common_params, model_name, policy_cls=None, policy_kwargs=None):
        self.env_id = env_id
        self.num_envs = num_envs
        self.common_params = common_params
        self.model_name = model_name
        self.rewards_history = []
        self.policy_cls = policy_cls if policy_cls else "MlpPolicy"
        self.policy_kwargs = policy_kwargs if policy_kwargs else {}

    def setup(self):
        """Configure environment and algorithm"""
        # Create vectorized environment
        self.vec_env = make_vec_env(self.env_id, n_envs=self.num_envs, seed=SEED)

        # Define a linear learning rate schedule function
        initial_lr = self.common_params["learning_rate"]
        min_lr_factor = 0.1  # Final LR will be 10% of initial

        def linear_schedule(progress_remaining):
            # progress_remaining goes from 1 (beginning) to 0 (end)
            # We want to go from initial_lr to initial_lr * min_lr_factor
            return initial_lr * (min_lr_factor + progress_remaining * (1.0 - min_lr_factor))

        # Convert hyperparameters to SB3 format with LR schedule
        sb3_params = {
            "learning_rate": linear_schedule,  # Use our custom schedule function
            "gamma": self.common_params["gamma"],
            "gae_lambda": self.common_params["gae_lambda"],
            "clip_range": self.common_params["clip_range"],
            "ent_coef": self.common_params["ent_coef"],
            "n_epochs": self.common_params["n_epochs"],
            "batch_size": self.common_params["batch_size"],
        }

        # Create SB3 PPO model
        self.model = SB3_PPO(
            policy=self.policy_cls,
            env=self.vec_env,
            verbose=1,
            seed=SEED,
            policy_kwargs=self.policy_kwargs,
            **sb3_params
        )

    def train(self, total_timesteps, eval_freq, num_eval_episodes):
        """Train the algorithm and evaluate periodically"""
        eval_env = gym.make(self.env_id)

        # Create eval callback
        class CustomEvalCallback:
            def __init__(self, eval_env, eval_freq, num_eval_episodes, reward_history, model_name):
                self.eval_env = eval_env
                self.eval_freq = eval_freq
                self.num_eval_episodes = num_eval_episodes
                self.reward_history = reward_history
                self.model_name = model_name

            def __call__(self, locals_, globals_):
                # Extract current model
                model = locals_["self"]
                timesteps_so_far = model.num_timesteps

                if timesteps_so_far % self.eval_freq == 0:
                    mean_reward, _ = evaluate_policy(
                        model,
                        self.eval_env,
                        n_eval_episodes=self.num_eval_episodes
                    )
                    self.reward_history.append((timesteps_so_far, mean_reward))
                    print(f"{self.model_name} - Step: {timesteps_so_far}, Mean Reward: {mean_reward:.2f}")

                return True

        # Train with callback
        callback = CustomEvalCallback(
            eval_env,
            eval_freq,
            num_eval_episodes,
            self.rewards_history,
            self.model_name
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        # Final evaluation
        mean_reward, _ = evaluate_policy(self.model, eval_env, n_eval_episodes=num_eval_episodes)
        if not any(step == total_timesteps for step, _ in self.rewards_history):
            self.rewards_history.append((total_timesteps, mean_reward))
        print(f"{self.model_name} - Final Mean Reward: {mean_reward:.2f}")

        eval_env.close()
        return self.rewards_history

    def save(self, path):
        """Save model"""
        self.model.save(path)


def create_generic_dppo_for_sb3(env_id, num_envs, model_name):
    """Create a version of DPPO that works with standard network architecture"""
    vec_env = make_vec_env(env_id, n_envs=num_envs, seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get environment spaces
    obs_space = vec_env.observation_space
    act_space = vec_env.action_space

    # Create actor and critic networks
    obs_shape = obs_space.shape[0]

    if isinstance(act_space, gym.spaces.Discrete):
        action_shape = act_space.n
        action_space_type = "discrete"
    else:
        action_shape = act_space.shape[0]
        action_space_type = "continuous"

    # Create MLP actor network with Sequential
    actor_sequential = nn.Sequential(
        nn.Linear(obs_shape, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_shape)
    ).to(device)

    # Create critic network with distributional output
    critic_sequential = nn.Sequential(
        nn.Linear(obs_shape, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, NUM_ATOMS)
    ).to(device)

    # Wrap sequential models to be compatible with DPPO
    actor_network = DPPOCompatibleSequential(actor_sequential).to(device)
    critic_network = DPPOCompatibleSequential(critic_sequential).to(device)

    # Initialize DPPO algorithm with properly mapped parameters
    dppo = DPPOAlgorithm(
        actor=actor_network,
        critic=critic_network,
        action_space_type=action_space_type,
        action_dim=action_shape if action_space_type == "continuous" else None,
        device=device,
        lr_actor=DPPO_PARAMS["lr_actor"],
        lr_critic=DPPO_PARAMS["lr_critic"],
        gamma=DPPO_PARAMS["gamma"],
        gae_lambda=DPPO_PARAMS["gae_lambda"],
        epsilon_base=DPPO_PARAMS["epsilon_base"],
        entropy_coef=DPPO_PARAMS["entropy_coef"],
        ppo_epochs=DPPO_PARAMS["ppo_epochs"],
        batch_size=DPPO_PARAMS["batch_size"],
        v_min=DPPO_PARAMS["v_min"],
        v_max=DPPO_PARAMS["v_max"],
        num_atoms=DPPO_PARAMS["num_atoms"]
    )

    return dppo, vec_env, device


def plot_comparison(reward_histories, legends):
    plt.figure(figsize=(12, 8))

    for history, legend in zip(reward_histories, legends):
        steps, scores = zip(*history)
        plt.plot(steps, scores, label=legend)

    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title(f'Performance Comparison on {ENV_ID}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f"{LOG_DIR}/comparison_plot.png")
    plt.close()

    # Print final scores
    print("\nFinal Mean Rewards:")
    for history, legend in zip(reward_histories, legends):
        print(f"{legend}: {history[-1][1]:.2f}")


def run_comparison():
    start_time = time.time()
    all_rewards = []
    all_legends = []

    # ===== 1. Standard SB3 PPO =====
    print("\n1. Training Standard SB3 PPO...")
    sb3_standard_wrapper = SB3Wrapper(
        env_id=ENV_ID,
        num_envs=NUM_ENVS,
        common_params=COMMON_PARAMS,
        model_name="SB3 Standard PPO"
    )
    sb3_standard_wrapper.setup()
    sb3_standard_rewards = sb3_standard_wrapper.train(TOTAL_TIMESTEPS, EVAL_FREQ, NUM_EVAL_EPISODES)
    sb3_standard_wrapper.save(f"{LOG_DIR}/sb3_standard_ppo_model")
    all_rewards.append(sb3_standard_rewards)
    all_legends.append("1. SB3 Standard PPO")

    # ===== 2. Your SimbaV2 with DPPO =====
    print("\n2. Training SimbaV2 with Your DPPO...")
    env = gym.make(ENV_ID)
    dppo_wrapper = CustomDPPOWrapper(
        env=env,
        model_name="SimbaV2 with Your DPPO",
        common_params=COMMON_PARAMS
    )
    dppo_wrapper.setup()
    dppo_rewards = dppo_wrapper.train(TOTAL_TIMESTEPS, EVAL_FREQ, NUM_EVAL_EPISODES)
    dppo_wrapper.save(f"{LOG_DIR}/simba_your_dppo_model.pt")
    env.close()
    all_rewards.append(dppo_rewards)
    all_legends.append("2. SimbaV2 with Your DPPO")

    elapsed_time = time.time() - start_time
    print(f"\nTotal training time: {elapsed_time:.2f} seconds")

    # Plot all results
    plot_comparison(all_rewards, all_legends)


if __name__ == "__main__":
    run_comparison()
