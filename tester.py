import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Type, Union

# Import SimBa model
from model_architectures.simba import SimBa

# Define a features extractor based on SimBa
class SimBaFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=256, num_blocks=4, dropout_rate=0.1):
        # Define the output dimension of the features extractor
        features_dim = hidden_dim
        super(SimBaFeaturesExtractor, self).__init__(observation_space, features_dim)

        obs_shape = observation_space.shape[0]

        # Create a SimBa model without the final output layer
        self.simba_core = SimBa(
            obs_shape=obs_shape,
            action_shape=features_dim,  # Temporary, we won't use this output
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def forward(self, observations):
        # Use the SimBa model but return features only
        output, features = self.simba_core(observations, return_features=True)
        return features

# Custom callback to track and save learning progress
class TrainingProgressCallback(BaseCallback):
    def __init__(self, verbose=0, model_name="model"):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.model_name = model_name
        self.rewards = []
        self.timesteps = []

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            self.rewards.append(mean_reward)
            self.timesteps.append(self.num_timesteps)
        return True

    def on_training_end(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.timesteps, self.rewards)
        plt.title(f"{self.model_name} Learning Curve")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Reward")
        plt.savefig(f"{self.model_name}_learning_curve.png")

# Main script to compare SimBa with MLP
def main():
    # Create folders for logs and models
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Create environment
    env_id = "LunarLander-v3"  # Good standard continuous control environment

    # Create vectorized environments for training
    def make_env():
        env = gym.make(env_id)
        env = Monitor(env)  # Monitor to record stats
        return env

    # Create vectorized environments
    env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel envs
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env for _ in range(1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    # Define evaluation callbacks
    eval_callback_simba = EvalCallback(
        eval_env,
        best_model_save_path="./models/simba_best_model",
        log_path="./logs/simba_results",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    eval_callback_mlp = EvalCallback(
        eval_env,
        best_model_save_path="./models/mlp_best_model",
        log_path="./logs/mlp_results",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    progress_callback_simba = TrainingProgressCallback(model_name="simba")
    progress_callback_mlp = TrainingProgressCallback(model_name="mlp")

    # Define models
    # 1. SimBa-based PPO
    simba_policy_kwargs = {
        "features_extractor_class": SimBaFeaturesExtractor,
        "features_extractor_kwargs": {
            "hidden_dim": 256,  # Smaller for faster training
            "num_blocks": 4,
            "dropout_rate": 0.1
        },
        "net_arch": [dict(pi=[64], vf=[64])]  # Small heads for actor-critic
    }

    model_simba = PPO(
        policy="MlpPolicy",  # We use MlpPolicy but with custom feature extractor
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=simba_policy_kwargs
    )

    # 2. MLP-based PPO (standard implementation)
    mlp_policy_kwargs = {
        "net_arch": dict(pi=[256, 256, 64], vf=[256, 256, 64]) # Shared layers [256, 256], separate heads [64]
    }

    model_mlp = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=mlp_policy_kwargs
    )

    # Define total timesteps
    total_timesteps = 500000  # Adjust based on environment complexity

    # Train models
    print("Training SimBa model...")
    model_simba.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback_simba, progress_callback_simba]
    )

    print("Training MLP model...")
    model_mlp.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback_mlp, progress_callback_mlp]
    )

    # Evaluate final models with more episodes for better statistical significance
    mean_reward_simba, std_reward_simba = evaluate_policy(
        model_simba, eval_env, n_eval_episodes=30, deterministic=True
    )

    mean_reward_mlp, std_reward_mlp = evaluate_policy(
        model_mlp, eval_env, n_eval_episodes=30, deterministic=True
    )

    print("\n==== FINAL EVALUATION ====")
    print(f"SimBa - Mean reward: {mean_reward_simba:.2f} +/- {std_reward_simba:.2f}")
    print(f"MLP - Mean reward: {mean_reward_mlp:.2f} +/- {std_reward_mlp:.2f}")

    # Compare the models statistically
    if mean_reward_simba > mean_reward_mlp:
        improvement = ((mean_reward_simba - mean_reward_mlp) / abs(mean_reward_mlp)) * 100
        print(f"SimBa outperforms MLP by {improvement:.2f}%")
    else:
        difference = ((mean_reward_mlp - mean_reward_simba) / abs(mean_reward_simba)) * 100
        print(f"MLP outperforms SimBa by {difference:.2f}%")

    # Save models
    model_simba.save("models/simba_ppo_final")
    model_mlp.save("models/mlp_ppo_final")

    # Plot comparison of learning curves
    plt.figure(figsize=(12, 8))
    plt.plot(progress_callback_simba.timesteps, progress_callback_simba.rewards, label='SimBa')
    plt.plot(progress_callback_mlp.timesteps, progress_callback_mlp.rewards, label='MLP')
    plt.title("Learning Curves: SimBa vs MLP")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.legend()
    plt.savefig("simba_vs_mlp_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
