import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Assuming these files are in the same directory or Python path
from algorithms.ppo import PPOAlgorithm
from algorithms.dppo import DPPOAlgorithm
from model_architectures.simba_v2 import SimbaV2

# --- Simple MLP Model for comparison ---
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, **kwargs):
        # Ignore any extra keyword arguments like return_features
        return self.network(x)

# --- Hyperparameters ---
ENV_NAME = "LunarLander-v3"  # More complex environment
TOTAL_TIMESTEPS = 500000    # Increased training time for more complex environment
N_STEPS = 1024  # Slightly larger steps for more stable gradient updates
BATCH_SIZE = 128  # Larger batch size for more stable updates
N_EPOCHS = 10
GAMMA = 0.995   # Higher discount factor for longer-term rewards
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
INITIAL_LR = 3e-4
FINAL_LR = 1e-5
LR_DECAY_TIMESTEPS = TOTAL_TIMESTEPS  # Decay over actual training duration
LR = INITIAL_LR  # For initialization - will be decayed during training
HIDDEN_DIM = 128  # Increased hidden dimension for more capacity
NUM_BLOCKS = 3    # More transformer blocks for more complex patterns
TARGET_KL = 0.01  # For PPO early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DPPO specific HPs
V_MIN = -300.0  # LunarLander can have much larger negative values
V_MAX = 300.0   # LunarLander can have much larger positive values
NUM_ATOMS = 101  # Number of atoms for distributional critic in DPPO

# --- Helper Function for Moving Average ---
def moving_average(data, window_size):
    if len(data) < window_size:
        return np.mean(data) if data else 0 # Return mean if not enough data yet
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def get_lr_decay(current_step, total_steps, initial_lr, final_lr):
    """
    Calculate the learning rate based on linear decay schedule
    Args:
        current_step: Current training timestep
        total_steps: Total timesteps for decay
        initial_lr: Starting learning rate (3e-4)
        final_lr: Final learning rate (1e-5)
    Returns:
        Decayed learning rate
    """
    # Ensure we don't exceed the total steps for decay calculation
    progress = min(current_step / total_steps, 1.0)
    # Linear decay from initial_lr to final_lr
    return initial_lr - progress * (initial_lr - final_lr)

# --- Training Function ---
def train_algorithm(algo_class, algo_name, total_timesteps, env_name):
    print(f"\n--- Training {algo_name} on {env_name} ---")
    start_time = time.time()

    env = gym.make(env_name)
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n # Discrete action space

    # Create Actor and Critic networks using SimbaV2
    if algo_class == PPOAlgorithm:
        # PPO uses a distributional critic that outputs logits for a categorical distribution
        actor_net = SimbaV2(obs_shape=obs_shape, action_shape=action_shape,
                          hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
                          is_critic=False, device=DEVICE).to(DEVICE)

        critic_net = SimbaV2(obs_shape=obs_shape, action_shape=1,  # Still use action_shape=1 as SimbaV2 handles num_atoms internally
                           hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
                           is_critic=True, num_atoms=NUM_ATOMS, device=DEVICE).to(DEVICE)
    else:  # DPPO
        # DPPO also uses a distributional critic
        actor_net = SimbaV2(obs_shape=obs_shape, action_shape=action_shape,
                          hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
                          is_critic=False, device=DEVICE).to(DEVICE)

        critic_net = SimbaV2(obs_shape=obs_shape, action_shape=1,  # Still use action_shape=1 as SimbaV2 handles num_atoms internally
                           hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
                           is_critic=True, num_atoms=NUM_ATOMS, device=DEVICE).to(DEVICE)

    # Create optimizers with the fixed learning rate
    current_lr = LR
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=current_lr)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=current_lr)

    # Instantiate the algorithm with the correct parameters based on the actual implementation
    if algo_class == PPOAlgorithm:
        algorithm = PPOAlgorithm(
            actor=actor_net,
            critic=critic_net,
            action_space_type="discrete",
            action_dim=action_shape,
            device=DEVICE,
            lr_actor=current_lr,
            lr_critic=current_lr,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_epsilon=CLIP_RANGE,
            critic_coef=VF_COEF,
            entropy_coef=ENT_COEF,
            ppo_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE
        )
    elif algo_class == DPPOAlgorithm:
        algorithm = DPPOAlgorithm(
            actor=actor_net,
            critic=critic_net,
            action_space_type="discrete",
            action_dim=action_shape,
            device=DEVICE,
            lr_actor=current_lr,
            lr_critic=current_lr,
            buffer_size=N_STEPS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            epsilon_base=CLIP_RANGE,
            critic_coef=VF_COEF,
            entropy_coef=ENT_COEF,
            ppo_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            v_min=V_MIN,
            v_max=V_MAX,
            num_atoms=NUM_ATOMS
        )
    else:
        raise ValueError(f"Unknown algorithm class: {algo_class}")

    episode_rewards = []
    episode_lengths = []
    all_rewards = [] # Store raw rewards per episode end

    obs, _ = env.reset()
    current_episode_reward = 0
    current_episode_length = 0
    timesteps_elapsed = 0

    while timesteps_elapsed < total_timesteps:
        # --- Rollout Phase ---
        # Temporary storage for the rollout data
        rollout_obs, rollout_actions, rollout_rewards, rollout_dones = [], [], [], []
        rollout_values, rollout_log_probs = [], []

        for step in range(N_STEPS):
            # Get action, value, and log probability
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                # Get action and log_prob
                action, value, log_prob = algorithm.get_action(obs_tensor)

                # Ensure we have scalar values for storing
                action_np = action.cpu().numpy().item() if action.numel() == 1 else action.cpu().numpy().flatten()[0]
                value_np = value.cpu().numpy().item() if value.numel() == 1 else value.cpu().numpy().mean()  # Use mean if not scalar
                log_prob_np = log_prob.cpu().numpy().item() if log_prob.numel() == 1 else log_prob.cpu().numpy().mean()

                # Debug print for troubleshooting
                if step == 0:  # Only print on first step to avoid spam
                    print(f"[{algo_name}] First step - action: {action.shape}, value: {value.shape}, log_prob: {log_prob.shape}")
                    print(f"[{algo_name}] Converted to - action_np: {type(action_np)}, value_np: {type(value_np)}")


            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Store data for this step
            rollout_obs.append(obs)
            rollout_actions.append(action_np) # Store numpy action
            rollout_rewards.append(reward)
            rollout_dones.append(done)
            rollout_values.append(value_np)
            rollout_log_probs.append(log_prob_np)

            obs = next_obs
            current_episode_reward += reward
            current_episode_length += 1
            timesteps_elapsed += 1

            if done:
                print(f"[{algo_name}] Timestep: {timesteps_elapsed}, Episode Reward: {current_episode_reward:.2f}, Length: {current_episode_length}")
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                all_rewards.append(current_episode_reward) # Store raw reward
                obs, _ = env.reset()
                current_episode_reward = 0
                current_episode_length = 0

            # Early exit if total timesteps reached during rollout
            if timesteps_elapsed >= total_timesteps:
                break

        # --- Storage Phase ---
        # Convert lists to numpy arrays
        obs_arr = np.array(rollout_obs)
        actions_arr = np.array(rollout_actions)
        rewards_arr = np.array(rollout_rewards)
        dones_arr = np.array(rollout_dones)
        values_arr = np.array(rollout_values)
        log_probs_arr = np.array(rollout_log_probs)

        # Convert numpy arrays to tensors for algorithm storage
        obs_tensor = torch.FloatTensor(obs_arr).to(DEVICE)
        actions_tensor = torch.LongTensor(actions_arr).to(DEVICE)  # Using LongTensor for discrete actions
        rewards_tensor = torch.FloatTensor(rewards_arr).to(DEVICE)
        dones_tensor = torch.BoolTensor(dones_arr).to(DEVICE)  # Fix: Use BoolTensor instead of FloatTensor
        values_tensor = torch.FloatTensor(values_arr).to(DEVICE)  # Values are scalars for both algorithms
        log_probs_tensor = torch.FloatTensor(log_probs_arr).to(DEVICE)

        # Debug prints
        print(f"[{algo_name}] Tensor shapes - obs: {obs_tensor.shape}, actions: {actions_tensor.shape}, values: {values_tensor.shape}")
        print(f"[{algo_name}] Tensor dtypes - obs: {obs_tensor.dtype}, actions: {actions_tensor.dtype}, values: {values_tensor.dtype}, dones: {dones_tensor.dtype}")

        # Make sure values_tensor is 1D for both algorithms (flattened if needed)
        if values_tensor.dim() > 1:
            print(f"[{algo_name}] Flattening values tensor from shape {values_tensor.shape}")
            values_tensor = values_tensor.flatten()
            print(f"[{algo_name}] New values tensor shape: {values_tensor.shape}")

        # Store experience based on algorithm type
        if isinstance(algorithm, PPOAlgorithm):
            # Check if PPOAlgorithm has store_initial_batch method
            if hasattr(algorithm, 'store_initial_batch'):
                # Store initial batch and get indices where data was stored
                indices = algorithm.store_initial_batch(
                    obs_batch=obs_tensor,
                    action_batch=actions_tensor,
                    value_batch=values_tensor,
                    log_prob_batch=log_probs_tensor
                )
                # Update rewards and dones with the same indices
                algorithm.update_rewards_dones_batch(
                    indices=indices,
                    rewards_batch=rewards_tensor,
                    dones_batch=dones_tensor
                )
            else:
                # Individual storage using store methods
                for i in range(len(obs_arr)):
                    algorithm.store(
                        obs=obs_tensor[i],
                        action=actions_tensor[i],
                        reward=rewards_tensor[i],
                        done=dones_tensor[i],
                        value=values_tensor[i],
                        log_prob=log_probs_tensor[i]
                    )
        elif isinstance(algorithm, DPPOAlgorithm):
            # DPPO expects a simple scalar value for its values buffer
            # We'll create a new 1D tensor with the proper dimensions
            print(f"[{algo_name}] DPPO store_initial_batch - Creating simplified value tensor...")
            # Create a simple value tensor with the same batch size
            batch_size = obs_tensor.shape[0]
            simplified_values = torch.zeros(batch_size, dtype=torch.float32, device=DEVICE)
            # Fill it with our scalar values
            for i in range(batch_size):
                simplified_values[i] = values_tensor[i] if values_tensor.numel() > i else 0.0

            print(f"[{algo_name}] DPPO simplified_values shape: {simplified_values.shape}, dtype: {simplified_values.dtype}")

            # Store with the simplified values
            indices = algorithm.store_initial_batch(
                obs_batch=obs_tensor,
                action_batch=actions_tensor,
                value_batch=simplified_values,  # Use simplified values
                log_prob_batch=log_probs_tensor
            )
            algorithm.update_rewards_dones_batch(
                indices=indices,
                rewards_batch=rewards_tensor,
                dones_batch=dones_tensor
            )
        else:
             raise TypeError("Algorithm type not recognized for storage.")

        # --- Update Phase ---
        # Update learning rate based on decay schedule
        new_lr = get_lr_decay(
            current_step=timesteps_elapsed,
            total_steps=LR_DECAY_TIMESTEPS,
            initial_lr=INITIAL_LR,
            final_lr=FINAL_LR
        )

        # Only update if LR has changed significantly to avoid unnecessary operations
        if abs(new_lr - current_lr) > 1e-7:
            current_lr = new_lr
            print(f"[{algo_name}] Updating learning rate to {current_lr:.6f} at timestep {timesteps_elapsed}/{total_timesteps} ({timesteps_elapsed/LR_DECAY_TIMESTEPS*100:.2f}% of decay schedule)")

            # Update learning rate in the algorithm's optimizer
            if hasattr(algorithm, 'optimizer'):
                for param_group in algorithm.optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                print(f"[{algo_name}] Warning: Algorithm does not have 'optimizer' attribute. LR decay might not work.")

            # Handle any additional optimizers in the algorithm
            if hasattr(algorithm, 'set_lr'):
                algorithm.set_lr(current_lr)
            elif hasattr(algorithm, 'actor_optimizer') and hasattr(algorithm, 'critic_optimizer'):
                for param_group in algorithm.actor_optimizer.param_groups:
                    param_group['lr'] = current_lr
                for param_group in algorithm.critic_optimizer.param_groups:
                    param_group['lr'] = current_lr

        print(f"[{algo_name}] Updating policy at timestep {timesteps_elapsed}...")
        algorithm.update()  # Most algorithms compute GAE internally

    env.close()
    end_time = time.time()
    print(f"--- {algo_name} Training Finished in {end_time - start_time:.2f} seconds ---")
    return all_rewards # Return raw episode rewards

# --- Main Execution ---
if __name__ == "__main__":
    ppo_rewards = train_algorithm(PPOAlgorithm, "PPO", TOTAL_TIMESTEPS, ENV_NAME)
    dppo_rewards = train_algorithm(DPPOAlgorithm, "DPPO", TOTAL_TIMESTEPS, ENV_NAME)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Calculate moving averages
    window = 10 # Moving average window size
    if len(ppo_rewards) >= window:
        ppo_smoothed = moving_average(ppo_rewards, window)
        # Generate x-axis values (episode numbers) for smoothed data
        ppo_episodes = np.arange(window -1, len(ppo_rewards))
        plt.plot(ppo_episodes, ppo_smoothed, label=f'PPO (Avg over {window} episodes)')
    else:
        plt.plot(ppo_rewards, label='PPO (Raw)', alpha=0.7)

    if len(dppo_rewards) >= window:
        dppo_smoothed = moving_average(dppo_rewards, window)
        # Generate x-axis values (episode numbers) for smoothed data
        dppo_episodes = np.arange(window -1, len(dppo_rewards))
        plt.plot(dppo_episodes, dppo_smoothed, label=f'DPPO (Avg over {window} episodes)')
    else:
        plt.plot(dppo_rewards, label='DPPO (Raw)', alpha=0.7)

    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward")
    plt.title(f"PPO vs DPPO Performance on {ENV_NAME}\n({TOTAL_TIMESTEPS} timesteps, LR decay {INITIAL_LR:.1e} → {FINAL_LR:.1e})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ppo_vs_dppo_lr_decay_{INITIAL_LR:.1e}_to_{FINAL_LR:.1e}.png")
    print(f"\nPlot saved as ppo_vs_dppo_lr_decay_{INITIAL_LR:.1e}_to_{FINAL_LR:.1e}.png")
    plt.show()
