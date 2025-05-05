import os
import time
import numpy as np
import torch
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.evaluation import evaluate_policy
from algorithms.dppo import DPPOAlgorithm
from model_architectures.simba_v2 import SimbaV2

# Function to count model parameters
def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Configuration - smaller scale for quick testing
ENV_ID = "BipedalWalker-v3"  # Simple environment
SEED = 42
TIMESTEPS = 1_000_000     # Reduced timesteps
EVAL_FREQ = 10_000      # More frequent evaluation
EVAL_EPISODES = 10      # Fewer evaluation episodes
SIMBAV2_HIDDEN_DIM = 29  # Increased from 16 to better match SB3 parameter count
SIMBAV2_NUM_BLOCKS = 1   # Single block for simpler architecture
SAVE_DIR = "results"

CRITICSCALE = 2

# Learning rates
INITIAL_ACTOR_LR = 3e-4
INITIAL_CRITIC_LR = 3e-4
# LR decay parameters
LR_DECAY_FACTOR = 1.0    # No decay, learning rate stays constant
LR_DECAY_STEPS = 0       # No decay steps

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

def make_env(seed=None):
    """Create the environment with optional seed"""
    env = gym.make(ENV_ID)
    if seed is not None:
        env.action_space.seed(seed)
        env.reset(seed=seed)
    return env

def setup_sb3_ppo():
    """Initialize Stable Baselines3 PPO algorithm"""
    # Create the environment
    env = make_env(SEED)

    # Print environment details
    print(f"SB3 Environment: {ENV_ID}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # Test environment - make sure we get proper observations and rewards
    obs, _ = env.reset(seed=SEED)
    print(f"Initial Observation shape: {obs.shape}, type: {type(obs)}, value: {obs[:5]}...")
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    print(f"After step - Action: {action}, Reward: {reward}, Done: {done}")
    print(f"Next Observation shape: {next_obs.shape}, type: {type(next_obs)}, value: {next_obs[:5]}...")

    # Configure PPO with simpler network
    model = SB3_PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,  # Aligned with DPPO's update frequency (batch_size * 32 ~= 2048)
        batch_size=64,
        n_epochs=5,    # Fewer epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Slightly more exploration
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[int(64 * CRITICSCALE *1.2), int(64 * CRITICSCALE *1.2)])]  # Smaller networks
        ),
        verbose=1,
        seed=SEED
    )

    # Print parameter counts
    # SB3 PPO uses a shared network architecture, so we count the total parameters
    total_params = count_parameters(model.policy)

    # For more detailed breakdown, we can examine the policy mlp_extractor
    # which contains separate networks for policy and value function
    policy_net_params = count_parameters(model.policy.mlp_extractor.policy_net)
    value_net_params = count_parameters(model.policy.mlp_extractor.value_net)

    print(f"SB3 PPO Policy Network Parameters: {policy_net_params:,}")
    print(f"SB3 PPO Value Network Parameters: {value_net_params:,}")
    print(f"SB3 PPO Total Parameters: {total_params:,}")

    return model, env

def setup_simbav2_dppo():
    """Initialize SimbaV2 + DPPO algorithm"""
    # Create the environment
    env = make_env(SEED)

    # Get environment dimensions
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n

    # Initialize the SimbaV2 actor and critic
    actor = SimbaV2(
        obs_shape=obs_shape,
        action_shape=action_shape,
        hidden_dim=SIMBAV2_HIDDEN_DIM,
        num_blocks=SIMBAV2_NUM_BLOCKS,
        device="cpu",
        shift_constant=3.0,
        is_critic=False
    )

    critic = SimbaV2(
        obs_shape=obs_shape,
        action_shape=action_shape,  # Not used for critic but keeping API consistent
        hidden_dim=SIMBAV2_HIDDEN_DIM * CRITICSCALE,  # Use same hidden dimension as actor instead of 3x
        num_blocks=SIMBAV2_NUM_BLOCKS,
        device="cpu",
        shift_constant=3.0,
        is_critic=True,        # Mark as critic
        num_atoms=101           # Reduce from 101 to 51 atoms to decrease parameters
    )

    # Initialize DPPO algorithm with distributional critic
    dppo = DPPOAlgorithm(
        actor=actor,
        critic=critic,
        action_space_type="discrete",
        action_dim=env.action_space.n,
        device="cpu",
        lr_actor=INITIAL_ACTOR_LR,
        lr_critic=INITIAL_CRITIC_LR,
        buffer_size=2048,  # Aligned with SB3 PPO's n_steps=2048
        gamma=0.99,
        gae_lambda=0.95,
        epsilon_base=0.2,  # PPO clipping parameter
        critic_coef=0.5,
        entropy_coef=0.01,  # Encourage exploration
        max_grad_norm=0.5,
        ppo_epochs=5,      # More PPO epochs
        batch_size=64,
        # Use distributional critic since DPPO requires it
        v_min=-10.0,
        v_max=10.0,
        num_atoms=101,       # Match critic's num_atoms (previously 101)
        debug=True,          # Enable debug output
    )

    # Print parameter counts
    actor_params = count_parameters(actor)
    critic_params = count_parameters(critic)
    total_params = actor_params + critic_params

    print(f"SimbaV2 Actor Parameters: {actor_params:,}")
    print(f"SimbaV2 Critic Parameters: {critic_params:,}")
    print(f"SimbaV2 Total Parameters: {total_params:,}")

    time.sleep(2)

    return dppo, env

def evaluate(model, env, n_episodes=5):
    """Evaluate a model on an environment for n_episodes"""
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        if isinstance(model, SB3_PPO):
            # For SB3 PPO - Use their reset format and action prediction
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not (done or truncated):
                # Get action from SB3 model
                action, _ = model.predict(obs, deterministic=True)

                # Take step in environment
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
        else:
            # For custom DPPO
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not (done or truncated):
                # Convert observation to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device="cpu")
                action, _, _ = model.get_action(obs_tensor, deterministic=True)

                # Convert action to numpy if it's a tensor
                if isinstance(action, torch.Tensor):
                    # For discrete actions, need to extract the scalar value
                    if action.dim() == 0:  # Scalar tensor
                        action_np = action.item()
                    else:  # Array tensor
                        action_np = action.cpu().numpy()
                        # If we have a single-item array, extract the scalar
                        if isinstance(action_np, np.ndarray) and action_np.size == 1:
                            action_np = action_np.item()
                else:
                    action_np = action

                # Take step in environment
                obs, reward, done, truncated, _ = env.step(action_np)
                episode_reward += reward
                episode_length += 1

        # Print evaluation details
        print(f"Eval Episode {episode+1}/{n_episodes}: Reward={episode_reward}, Length={episode_length}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print(f"Evaluation results - Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.2f}")

    return mean_reward, std_reward

def train_and_evaluate():
    """Train and evaluate both algorithms"""
    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Setup algorithms
    print("Setting up SB3 PPO...")
    sb3_ppo, sb3_env = setup_sb3_ppo()

    print("Setting up SimbaV2 + DPPO...")
    dppo, dppo_env = setup_simbav2_dppo()

    # Initialize tracking variables
    sb3_rewards = []
    dppo_rewards = []
    timesteps = []

    total_timesteps = 0

    # Calculate learning rate decay steps
    lr_decay_at_timesteps = [int(i * TIMESTEPS / LR_DECAY_STEPS) for i in range(1, LR_DECAY_STEPS+1)]
    current_actor_lr = INITIAL_ACTOR_LR
    current_critic_lr = INITIAL_CRITIC_LR

    print(f"Learning rate will decay at timesteps: {lr_decay_at_timesteps}")
    print(f"Initial learning rates - Actor: {current_actor_lr}, Critic: {current_critic_lr}")

    # Train and evaluate
    print("Starting training...")
    while total_timesteps < TIMESTEPS:
        # Train SB3 PPO for EVAL_FREQ steps
        print("\nTraining SB3 PPO...")
        # Check if model parameters are changing
        before_params = []
        for param in sb3_ppo.policy.parameters():
            before_params.append(param.data.clone().mean().item())

        sb3_ppo.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=False)

        # Check if parameters actually changed
        after_params = []
        for param in sb3_ppo.policy.parameters():
            after_params.append(param.data.clone().mean().item())

        param_changes = [abs(a - b) for a, b in zip(before_params, after_params)]
        mean_change = sum(param_changes) / len(param_changes) if param_changes else 0
        print(f"SB3 PPO Parameter Change: {mean_change:.6f} (mean abs diff)")

        if mean_change < 1e-6:
            print("WARNING: SB3 PPO parameters barely changing. Check if training is working properly!")

        # Train DPPO for EVAL_FREQ steps
        total_dppo_steps = 0
        while total_dppo_steps < EVAL_FREQ:
            # Collect experience for DPPO
            obs, _ = dppo_env.reset()
            episode_steps = 0
            episode_reward = 0
            done = False
            truncated = False

            while not (done or truncated) and total_dppo_steps < EVAL_FREQ:
                # Get action from DPPO
                # Convert observation to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device="cpu")
                action, log_prob, value = dppo.get_action(obs_tensor)

                # Debug prints to understand what's happening
                if total_dppo_steps % 100 == 0:
                    # Convert tensors to scalar values for printing if needed
                    action_val = action.item() if isinstance(action, torch.Tensor) else action
                    log_prob_val = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob
                    value_val = value.item() if isinstance(value, torch.Tensor) else value

                    print(f"\nDPPO Action: {action_val}, Log Prob: {log_prob_val:.4f}, Value: {value_val:.4f}")

                    # Get raw policy output
                    with torch.no_grad():
                        raw_output = dppo.actor(obs_tensor)
                        if isinstance(raw_output, torch.Tensor):
                            probs = torch.nn.functional.softmax(raw_output, dim=-1)
                            print(f"Policy Output: {raw_output.cpu().numpy()}")
                            print(f"Action Probs: {probs.cpu().numpy()}")
                            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                            entropy_val = entropy.item() if entropy.numel() == 1 else entropy[0].item()
                            print(f"Policy Entropy: {entropy_val:.4f}")

                        # Get critic output
                        critic_output = dppo.critic(obs_tensor)
                        if isinstance(critic_output, torch.Tensor):
                            if critic_output.dim() > 1 and critic_output.size(1) > 1:
                                # Distributional critic
                                critic_probs = torch.nn.functional.softmax(critic_output, dim=1)
                                support = torch.linspace(-10.0, 10.0, 101, device="cpu")

                                # Calculate expected value from distribution
                                expected_values = (critic_probs * support.unsqueeze(0)).sum(dim=1)
                                expected_value = expected_values[0].item() if expected_values.numel() > 0 else 0.0

                                # Calculate entropy of the value distribution
                                critic_entropy = -torch.sum(critic_probs * torch.log(critic_probs + 1e-10), dim=1)
                                entropy_val = critic_entropy[0].item() if critic_entropy.numel() > 0 else 0.0

                                print(f"Critic Expected Value: {expected_value:.4f}, Entropy: {entropy_val:.4f}")
                            else:
                                # Standard critic
                                print(f"Critic Output: {critic_output.cpu().numpy()}")
                        else:
                            print(f"Critic Output: {critic_output}")

                        # Print observation for reference
                        print(f"Observation: {obs}")

                    # Print episode metrics
                    print(f"Episode Steps: {episode_steps}, Reward: {episode_reward}")

                # Convert action to numpy if it's a tensor
                if isinstance(action, torch.Tensor):
                    # For discrete actions, need to extract the scalar value
                    if action.dim() == 0:  # Scalar tensor
                        action_np = action.item()
                    else:  # Array tensor
                        action_np = action.cpu().numpy()
                        # If we have a single-item array, extract the scalar
                        if isinstance(action_np, np.ndarray) and action_np.size == 1:
                            action_np = action_np.item()
                else:
                    action_np = action

                # Step environment
                next_obs, reward, done, truncated, _ = dppo_env.step(action_np)

                # Early stopping for initial learning stability
                # If we're in the first 10000 steps and episode is too long, terminate early
                # This helps prevent getting stuck in poor policies
                if total_timesteps < 10000 and episode_steps > 100:
                    print("Early stopping episode for initial stability")
                    done = True

                # Store initial batch (obs, action, log_prob, value)
                obs_tensor = torch.tensor([obs], dtype=torch.float32, device="cpu")
                action_tensor = torch.tensor([action], dtype=torch.long, device="cpu")
                log_prob_tensor = torch.tensor([log_prob], dtype=torch.float32, device="cpu")
                value_tensor = torch.tensor([value], dtype=torch.float32, device="cpu")

                indices = dppo.store_initial_batch(
                    obs_tensor, action_tensor, log_prob_tensor, value_tensor
                )

                # Update rewards and dones
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device="cpu")
                done_tensor = torch.tensor([done or truncated], dtype=torch.bool, device="cpu")
                dppo.update_rewards_dones_batch(indices, reward_tensor, done_tensor)

                # Update for next iteration
                obs = next_obs
                episode_steps += 1
                episode_reward += reward
                total_dppo_steps += 1

            # Update DPPO after enough experience has been collected
            # Only update when buffer is full, to match SB3 PPO's behavior
            if dppo.memory.size >= dppo.buffer_size:  # Buffer is full, like SB3's rollout buffer
                print(f"Updating DPPO with buffer size {dppo.memory.size}")
                metrics = dppo.update()
                # Format metrics nicely
                actor_loss = metrics.get('actor_loss', 'N/A')
                if isinstance(actor_loss, torch.Tensor):
                    actor_loss = actor_loss.item()

                critic_loss = metrics.get('critic_loss', 'N/A')
                if isinstance(critic_loss, torch.Tensor):
                    critic_loss = critic_loss.item()

                entropy_loss = metrics.get('entropy_loss', 'N/A')
                if isinstance(entropy_loss, torch.Tensor):
                    entropy_loss = entropy_loss.item()

                # Format the values based on their types
                actor_loss_str = f"{actor_loss:.4f}" if isinstance(actor_loss, float) else str(actor_loss)
                critic_loss_str = f"{critic_loss:.4f}" if isinstance(critic_loss, float) else str(critic_loss)
                entropy_loss_str = f"{entropy_loss:.4f}" if isinstance(entropy_loss, float) else str(entropy_loss)

                print(f"Update metrics: actor_loss={actor_loss_str}, "
                      f"critic_loss={critic_loss_str}, "
                      f"entropy_loss={entropy_loss_str}")

                # Clear memory after updates, similar to SB3 PPO's rollout buffer clearing
                dppo.memory.clear()
                print("Cleared memory buffer after update, like SB3 PPO")

            # Periodically reset optimizer to escape local minima (every 10000 steps)
            if total_dppo_steps > 0 and total_dppo_steps % 10000 == 0:
                # Re-initialize optimizer with current learning rates
                combined_params = list(dppo.actor.parameters()) + list(dppo.critic.parameters())
                dppo.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, combined_params),
                    lr=current_actor_lr
                )
                print(f"\nReset optimizer at step {total_dppo_steps} with learning rate {current_actor_lr}")

        total_timesteps += EVAL_FREQ
        timesteps.append(total_timesteps)

        # Apply learning rate decay if needed
        if total_timesteps in lr_decay_at_timesteps:
            # Calculate new learning rates
            current_actor_lr *= LR_DECAY_FACTOR
            current_critic_lr *= LR_DECAY_FACTOR

            # Update optimizer learning rates
            for param_group in dppo.optimizer.param_groups:
                param_group['lr'] = current_actor_lr  # Use actor LR for all params for simplicity

            print(f"\nDecayed learning rate at timestep {total_timesteps}")
            print(f"New learning rates - Actor: {current_actor_lr}, Critic: {current_critic_lr}")

        # Evaluate SB3 PPO
        # Create fresh environments for evaluation to ensure clean state
        eval_env_sb3 = make_env(SEED)
        sb3_mean_reward, sb3_std_reward = evaluate(sb3_ppo, eval_env_sb3, EVAL_EPISODES)
        sb3_rewards.append(sb3_mean_reward)
        eval_env_sb3.close()  # Clean up environment

        # Evaluate DPPO
        eval_env_dppo = make_env(SEED)
        dppo_mean_reward, dppo_std_reward = evaluate(dppo, eval_env_dppo, EVAL_EPISODES)
        dppo_rewards.append(dppo_mean_reward)
        eval_env_dppo.close()  # Clean up environment

        print(f"Timestep: {total_timesteps}/{TIMESTEPS}")
        print(f"SB3 PPO - Mean reward: {sb3_mean_reward:.2f}, Std: {sb3_std_reward:.2f}")
        print(f"SimbaV2+DPPO - Mean reward: {dppo_mean_reward:.2f}, Std: {dppo_std_reward:.2f}")
        print("-" * 50)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, sb3_rewards, label="SB3 PPO")
    plt.plot(timesteps, dppo_rewards, label="SimbaV2+DPPO")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title(f"Performance Comparison on {ENV_ID}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    # Create sanitized ENV_ID for filename (replace / with _)
    sanitized_env_id = ENV_ID.replace('/', '_')

    # Ensure directory exists
    save_path = os.path.join(SAVE_DIR)
    os.makedirs(save_path, exist_ok=True)

    plt.savefig(os.path.join(save_path, f"comparison_{sanitized_env_id}_{int(time.time())}.png"))
    plt.show()

    # Save final results
    result_data = {
        "timesteps": timesteps,
        "sb3_rewards": sb3_rewards,
        "dppo_rewards": dppo_rewards
    }

    np.save(os.path.join(save_path, f"results_{sanitized_env_id}_{int(time.time())}.npy"), result_data)

    print("\nFinal Results:")
    print(f"SB3 PPO - Final mean reward: {sb3_rewards[-1]:.2f}")
    print(f"SimbaV2+DPPO - Final mean reward: {dppo_rewards[-1]:.2f}")

    # Close environments
    sb3_env.close()
    dppo_env.close()

    return sb3_rewards, dppo_rewards

if __name__ == "__main__":
    train_and_evaluate()
