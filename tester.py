# Rlbot-thesis/tester.py

import gymnasium as gym
import torch
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Assuming your modules are importable from the project root
from algorithms.ppo import PPOAlgorithm
from model_architectures.simba_v2 import SimbaV2
# Import AuxiliaryTaskManager if you plan to use it (optional)
# from auxiliary import AuxiliaryTaskManager

# --- Plotting Utility ---
class TrainingPlotter:
    """Utility class to handle real-time plotting of training metrics using matplotlib."""

    def __init__(self, save_dir, update_interval=10, figsize=(12, 12)): # Adjusted figsize slightly
        """
        Initialize the plotter.

        Args:
            save_dir (str): Directory to save plots
            update_interval (int): Plot update frequency (number of data points)
            figsize (tuple): Figure size for matplotlib
        """
        self.save_dir = save_dir
        self.update_interval = update_interval
        self.figsize = figsize

        self.plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

        # Initialize data storage - Added critic variance loss, mean variance
        self.data = {
            'steps': [],
            'episode_returns': [],
            'episode_lengths': [],
            'actor_loss': [],
            'critic_loss_mean': [], # Changed
            'critic_loss_variance': [], # Added
            'critic_loss_total': [], # Added
            'entropy_loss': [],
            'learning_rate': [],
            'mean_return': [], # Mean return from buffer (PPO internal)
            'mean_rewards_100': [], # Mean return from last 100 episodes
            'mean_predicted_variance': [], # Added
        }

        plt.switch_backend('Agg')

        # Setup the main metrics figure and subplots (Adjusted layout)
        self.fig, self.axes = plt.subplots(4, 2, figsize=self.figsize) # Changed to 4x2

        self.rewards_fig, self.rewards_axes = plt.subplots(figsize=(10, 6))
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3) # Increased hspace
        self.setup_axes()

        self.update_counter = 0
        self.last_100_rewards = []
        self.best_reward = -float('inf')

    def setup_axes(self):
        """Set up axes with titles and labels."""
        # Row 0: Returns and Lengths
        self.axes[0, 0].set_title('Episode Returns')
        self.axes[0, 0].set_xlabel('Steps')
        self.axes[0, 0].set_ylabel('Return')
        self.axes[0, 0].grid(True)

        self.axes[0, 1].set_title('Episode Lengths')
        self.axes[0, 1].set_xlabel('Steps')
        self.axes[0, 1].set_ylabel('Length')
        self.axes[0, 1].grid(True)

        # Row 1: Actor and Entropy Losses
        self.axes[1, 0].set_title('Actor Loss')
        self.axes[1, 0].set_xlabel('Steps')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].grid(True)

        self.axes[1, 1].set_title('Entropy Loss')
        self.axes[1, 1].set_xlabel('Steps')
        self.axes[1, 1].set_ylabel('Loss')
        self.axes[1, 1].grid(True)

        # Row 2: Critic Losses (Mean, Variance, Total)
        self.axes[2, 0].set_title('Critic Losses')
        self.axes[2, 0].set_xlabel('Steps')
        self.axes[2, 0].set_ylabel('Loss')
        self.axes[2, 0].grid(True)

        # Row 3: Learning Rate and Mean Predicted Variance
        self.axes[2, 1].set_title('Mean Predicted Variance (Critic)') # Changed
        self.axes[2, 1].set_xlabel('Steps')
        self.axes[2, 1].set_ylabel('Variance')
        self.axes[2, 1].grid(True)

        self.axes[3, 0].set_title('Learning Rate')
        self.axes[3, 0].set_xlabel('Steps')
        self.axes[3, 0].set_ylabel('Learning Rate')
        self.axes[3, 0].grid(True)

        # Row 4: Mean Return (Buffer)
        self.axes[3, 1].set_title('Mean Return (Buffer)')
        self.axes[3, 1].set_xlabel('Steps')
        self.axes[3, 1].set_ylabel('Return')
        self.axes[3, 1].grid(True)

        # Dedicated rewards plot
        self.rewards_axes.set_title('Training Rewards')
        self.rewards_axes.set_xlabel('Steps')
        self.rewards_axes.set_ylabel('Reward')
        self.rewards_axes.grid(True)

    def add_data(self, step, metrics=None, episode_info=None):
        """Add new data point."""
        self.data['steps'].append(step)

        if episode_info is not None:
            episode_return = episode_info.get('return', None)
            episode_length = episode_info.get('length', None)
            if episode_return is not None:
                if episode_return > self.best_reward: self.best_reward = episode_return
                self.last_100_rewards.append(episode_return)
                if len(self.last_100_rewards) > 100: self.last_100_rewards.pop(0)
                mean_reward = sum(self.last_100_rewards) / len(self.last_100_rewards)
                self.data['mean_rewards_100'].append((step, mean_reward))
            self.data['episode_returns'].append(episode_return)
            self.data['episode_lengths'].append(episode_length)
        else:
            # Maintain list length consistency with None values if only metrics are added
            if len(self.data['steps']) > len(self.data['episode_returns']):
                self.data['episode_returns'].append(None)
                self.data['episode_lengths'].append(None)


        # Add metrics data if provided - Updated keys
        if metrics is not None:
            self.data['actor_loss'].append(metrics.get('actor_loss', None))
            self.data['critic_loss_mean'].append(metrics.get('critic_loss_mean', None))
            self.data['critic_loss_variance'].append(metrics.get('critic_loss_variance', None))
            self.data['critic_loss_total'].append(metrics.get('critic_loss_total', None))
            self.data['entropy_loss'].append(metrics.get('entropy_loss', None))
            self.data['learning_rate'].append(metrics.get('learning_rate', None))
            self.data['mean_return'].append(metrics.get('mean_return', None))
            self.data['mean_predicted_variance'].append(metrics.get('mean_predicted_variance', None))
        else:
            # Maintain list length consistency with None values if only episode info is added
            if len(self.data['steps']) > len(self.data['actor_loss']):
                self.data['actor_loss'].append(None)
                self.data['critic_loss_mean'].append(None)
                self.data['critic_loss_variance'].append(None)
                self.data['critic_loss_total'].append(None)
                self.data['entropy_loss'].append(None)
                self.data['learning_rate'].append(None)
                self.data['mean_return'].append(None)
                self.data['mean_predicted_variance'].append(None)

        self.update_counter += 1
        if self.update_counter >= self.update_interval:
            self.update_plot()
            self.update_counter = 0

    def _plot_metric(self, ax, key, label, style='b-', steps_key='steps', alpha=0.8):
        """Helper to plot a single metric, filtering None values."""
        valid_indices = [i for i, x in enumerate(self.data[key]) if x is not None]
        if valid_indices:
            valid_steps = [self.data[steps_key][i] for i in valid_indices]
            valid_values = [self.data[key][i] for i in valid_indices]
            ax.plot(valid_steps, valid_values, style, label=label, alpha=alpha)
            return True, valid_steps # Return steps for positioning text
        return False, []

    def update_plot(self):
        """Update all plots with current data."""
        valid_steps = self.data['steps']

        # Row 0: Returns and Lengths
        self.axes[0, 0].clear()
        plotted_returns, return_steps = self._plot_metric(self.axes[0, 0], 'episode_returns', 'Episode Return', 'b-')
        self.axes[0, 0].set_title('Episode Returns')
        self.axes[0, 0].set_xlabel('Steps'); self.axes[0, 0].set_ylabel('Return'); self.axes[0, 0].grid(True)

        self.axes[0, 1].clear()
        plotted_lengths, _ = self._plot_metric(self.axes[0, 1], 'episode_lengths', 'Episode Length', 'r-')
        self.axes[0, 1].set_title('Episode Lengths')
        self.axes[0, 1].set_xlabel('Steps'); self.axes[0, 1].set_ylabel('Length'); self.axes[0, 1].grid(True)

        # Update the dedicated rewards plot
        self.rewards_axes.clear()
        plotted_rewards_dedicated, reward_steps_dedicated = self._plot_metric(self.rewards_axes, 'episode_returns', 'Episodes', 'b-', alpha=0.3)
        if plotted_rewards_dedicated and self.data['mean_rewards_100']:
            mean_steps = [x[0] for x in self.data['mean_rewards_100']]
            mean_values = [x[1] for x in self.data['mean_rewards_100']]
            self.rewards_axes.plot(mean_steps, mean_values, 'r-', label='Mean (100 eps)', linewidth=2)
            if mean_values and reward_steps_dedicated:
                current_mean = mean_values[-1]
                first_step = reward_steps_dedicated[0] # Use first valid step from plotted data
                self.rewards_axes.axhline(y=current_mean, color='r', linestyle='--', alpha=0.5)
                self.rewards_axes.text(first_step, current_mean, f' Mean: {current_mean:.2f}', color='r', verticalalignment='bottom')
            if self.best_reward > -float('inf') and reward_steps_dedicated:
                 first_step = reward_steps_dedicated[0]
                 self.rewards_axes.axhline(y=self.best_reward, color='g', linestyle='--', alpha=0.5)
                 self.rewards_axes.text(first_step, self.best_reward, f' Best: {self.best_reward:.2f}', color='g', verticalalignment='bottom')
        num_episodes = len([r for r in self.data['episode_returns'] if r is not None])
        self.rewards_axes.set_title(f'Training Rewards (Episodes: {num_episodes})')
        self.rewards_axes.set_xlabel('Steps'); self.rewards_axes.set_ylabel('Reward'); self.rewards_axes.legend(); self.rewards_axes.grid(True)


        # Row 1: Actor and Entropy Losses
        self.axes[1, 0].clear()
        self._plot_metric(self.axes[1, 0], 'actor_loss', 'Actor Loss', 'g-')
        self.axes[1, 0].set_title('Actor Loss')
        self.axes[1, 0].set_xlabel('Steps'); self.axes[1, 0].set_ylabel('Loss'); self.axes[1, 0].grid(True)

        self.axes[1, 1].clear()
        self._plot_metric(self.axes[1, 1], 'entropy_loss', 'Entropy Loss', 'c-')
        self.axes[1, 1].set_title('Entropy Loss')
        self.axes[1, 1].set_xlabel('Steps'); self.axes[1, 1].set_ylabel('Loss'); self.axes[1, 1].grid(True)

        # Row 2: Critic Losses
        self.axes[2, 0].clear()
        self._plot_metric(self.axes[2, 0], 'critic_loss_mean', 'Critic Mean Loss', 'm-')
        self._plot_metric(self.axes[2, 0], 'critic_loss_variance', 'Critic Var Loss', 'y-')
        self._plot_metric(self.axes[2, 0], 'critic_loss_total', 'Critic Total Loss', 'k-')
        self.axes[2, 0].set_title('Critic Losses')
        self.axes[2, 0].set_xlabel('Steps'); self.axes[2, 0].set_ylabel('Loss'); self.axes[2, 0].legend(); self.axes[2, 0].grid(True)

        # Row 3: Mean Predicted Variance and Learning Rate
        self.axes[2, 1].clear()
        self._plot_metric(self.axes[2, 1], 'mean_predicted_variance', 'Mean Pred Var', 'orange') # Changed
        self.axes[2, 1].set_title('Mean Predicted Variance (Critic)')
        self.axes[2, 1].set_xlabel('Steps'); self.axes[2, 1].set_ylabel('Variance'); self.axes[2, 1].grid(True); self.axes[2, 1].set_yscale('log') # Use log scale for variance

        self.axes[3, 0].clear()
        self._plot_metric(self.axes[3, 0], 'learning_rate', 'Learning Rate', 'k-')
        self.axes[3, 0].set_title('Learning Rate')
        self.axes[3, 0].set_xlabel('Steps'); self.axes[3, 0].set_ylabel('Learning Rate'); self.axes[3, 0].grid(True)

        # Row 4: Mean Return (Buffer)
        self.axes[3, 1].clear()
        self._plot_metric(self.axes[3, 1], 'mean_return', 'Mean Return (Buffer)', 'purple') # Changed color
        self.axes[3, 1].set_title('Mean Return (Buffer)')
        self.axes[3, 1].set_xlabel('Steps'); self.axes[3, 1].set_ylabel('Return'); self.axes[3, 1].grid(True)

        plt.tight_layout()
        self.save_plot()

    def save_plot(self):
        """Save the current plots to disk."""
        try:
            self.fig.savefig(os.path.join(self.plots_dir, 'training_progress.png'), dpi=300)
            self.rewards_fig.tight_layout()
            self.rewards_fig.savefig(os.path.join(self.plots_dir, 'rewards.png'), dpi=300)
        except Exception as e:
            print(f"Error saving plots: {e}")


    def get_reward_stats(self):
        """Get current reward statistics."""
        if not self.last_100_rewards:
            return {'mean_reward_last_100': 0.0, 'best_reward': float('-inf'), 'num_episodes': 0}
        mean_reward = sum(self.last_100_rewards) / len(self.last_100_rewards)
        return {'mean_reward_last_100': mean_reward, 'best_reward': self.best_reward,
                'num_episodes': len([r for r in self.data['episode_returns'] if r is not None])}

    def close(self):
        """Close all plots properly."""
        plt.close(self.fig)
        plt.close(self.rewards_fig)

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO with SimbaV2 (Gaussian Critic) on LunarLander-v3")

    # Environment Args
    parser.add_argument("--env_id", type=str, default="LunarLander-v3", help="Gymnasium environment ID")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total number of training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model Args (SimbaV2)
    parser.add_argument("--hidden_dim_actor", type=int, default=256, help="Hidden dimension for SimbaV2 actor")
    parser.add_argument("--hidden_dim_critic", type=int, default=384, help="Hidden dimension for SimbaV2 critic")
    parser.add_argument("--num_blocks_actor", type=int, default=2, help="Number of residual blocks in SimbaV2 actor")
    parser.add_argument("--num_blocks_critic", type=int, default=3, help="Number of residual blocks in SimbaV2 critic")
    parser.add_argument("--shift_constant", type=float, default=3.0, help="Shift constant for SimbaV2 input embedding")
    parser.add_argument("--shared_model", action="store_true", help="Use a shared backbone for actor and critic (Requires SimbaV2 modification)")

    # Algorithm Args (PPO)
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Learning rate for the actor (and combined optimizer)")
    parser.add_argument("--lr_critic", type=float, default=3e-4, help="Learning rate for the critic (ignored if shared_model)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Lambda for Generalized Advantage Estimation")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument("--critic_coef", type=float, default=0.5, help="Coefficient for the total critic loss")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Coefficient for the entropy bonus")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum norm for gradient clipping")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="Number of optimization epochs per PPO update")
    parser.add_argument("--batch_size", type=int, default=16384, help="Batch size for PPO updates")
    parser.add_argument("--buffer_size", type=int, default=32768, help="Size of the PPO replay buffer and update interval")

    # Gaussian Critic Args
    parser.add_argument("--variance_loss_coefficient", type=float, default=0.01, help="Coefficient for the critic's variance loss (NLL)")
    parser.add_argument("--reward_norm_G_max", type=float, default=300.0, help="Estimated max discounted return for reward normalization (adjust for env)")

    # Distributional PPO Uncertainty Weighting Args (Optional, uses Gaussian variance/entropy)
    parser.add_argument("--use_uncertainty_weight", action="store_true", help="Use uncertainty weighting in PPO objective")
    parser.add_argument("--uncertainty_weight_type", type=str, default="variance", choices=["entropy", "variance"], help="Type of uncertainty measure ('entropy' or 'variance')")
    parser.add_argument("--uncertainty_weight_temp", type=float, default=1.0, help="Temperature for uncertainty weighting sigmoid")

    # Learning Rate Decay Args
    parser.add_argument("--use_lr_decay", action="store_true", default=True, help="Use linear learning rate decay")
    parser.add_argument("--lr_decay_end_factor", type=float, default=0.1, help="End learning rate factor (final_lr = lr * factor)")

    # Training Args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="rlbot-thesis-ppo-gaussian", help="WandB project name") # Updated project name
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity (username or team)")
    parser.add_argument("--save_interval", type=int, default=100_000, help="Save model and state every N timesteps")
    parser.add_argument("--save_dir", type=str, default="./ppo_lunarlander_gaussian_checkpoints", help="Directory to save checkpoints") # Updated dir name
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for this run (used for saving and wandb)")

    return parser.parse_args()

# --- Main Training Script ---
if __name__ == "__main__":
    args = parse_args()

    # --- Setup ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    run_name = args.run_name if args.run_name else f"ppo_simbav2_gaussian_{args.env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(args.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Run Name: {run_name}")
    print(f"Save Path: {save_path}")
    print(f"Device: {args.device}")

    plotter = TrainingPlotter(save_path, update_interval=5)

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        print("Weights & Biases initialized.")

    # --- Environment ---
    env = gym.make(args.env_id)
    _, _ = env.reset(seed=args.seed)
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    action_space_type = "discrete"
    print(f"Observation shape: {obs_shape}, Action shape: {action_shape} ({action_space_type})")

    # --- Models ---
    if args.shared_model:
        print("WARNING: Shared model (--shared_model) requires modifications to SimbaV2 forward pass. Using SEPARATE models.")
        args.shared_model = False # Force separate

    print("Using SEPARATE SimbaV2 models for Actor and Critic (Gaussian).")
    actor = SimbaV2(
        obs_shape=obs_shape,
        action_shape=action_shape,
        hidden_dim=args.hidden_dim_actor,
        num_blocks=args.num_blocks_actor,
        shift_constant=args.shift_constant,
        device=args.device,
        is_critic=False,
    ).to(args.device)

    critic = SimbaV2(
        obs_shape=obs_shape,
        action_shape=action_shape,
        hidden_dim=args.hidden_dim_critic,
        num_blocks=args.num_blocks_critic,
        shift_constant=args.shift_constant,
        device=args.device,
        is_critic=True,
    ).to(args.device)
    shared_model_flag = False


    # --- Auxiliary Task Manager (Optional) ---
    aux_task_manager = None

    # --- Algorithm ---
    ppo_algorithm = PPOAlgorithm(
        actor=actor,
        critic=critic,
        aux_task_manager=aux_task_manager,
        action_space_type=action_space_type,
        action_dim=action_shape,
        device=args.device,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        critic_coef=args.critic_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        use_amp=args.use_amp,
        debug=args.debug,
        use_wandb=args.use_wandb,
        variance_loss_coefficient=args.variance_loss_coefficient,
        reward_norm_G_max=args.reward_norm_G_max,
        use_uncertainty_weight=args.use_uncertainty_weight,
        uncertainty_weight_type=args.uncertainty_weight_type,
        uncertainty_weight_temp=args.uncertainty_weight_temp,
        buffer_size=args.buffer_size,
    )

    print("PPO Algorithm (Gaussian Critic) initialized.")
    print(f"Buffer size (update interval): {ppo_algorithm.memory.buffer_size}")
    update_interval = ppo_algorithm.memory.buffer_size

    # --- Training Loop ---
    start_time = time.time()
    obs, info = env.reset()
    total_updates = 0
    timesteps_since_last_save = 0
    initial_lr_actor = args.lr_actor
    initial_lr_critic = args.lr_critic

    obs_batch_list = []
    action_batch_list = []
    log_prob_batch_list = []
    value_batch_list = [] # Placeholder values
    reward_batch_list = []
    done_batch_list = []

    print(f"Starting training for {args.total_timesteps} timesteps...")
    print(f"Learning rate decay: {'Enabled' if args.use_lr_decay else 'Disabled'}")

    for global_step in range(1, args.total_timesteps + 1):
        # --- Action Selection ---
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        action, log_prob, value_dummy = ppo_algorithm.get_action(obs_tensor, deterministic=False)
        action_np = action.cpu().numpy().item()

        # --- Environment Step ---
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        # --- Store Experience ---
        obs_batch_list.append(obs)
        action_value = action.item() if isinstance(action, torch.Tensor) and action.numel() == 1 else action.cpu().numpy()
        if hasattr(action_value, 'item'): action_value = action_value.item()
        action_batch_list.append(action_value)

        log_prob_value = log_prob.item() if isinstance(log_prob, torch.Tensor) and log_prob.numel() == 1 else log_prob.cpu().numpy()
        if hasattr(log_prob_value, 'item'): log_prob_value = log_prob_value.item()
        log_prob_batch_list.append(log_prob_value)

        value_value = value_dummy.item() if isinstance(value_dummy, torch.Tensor) and value_dummy.numel() == 1 else value_dummy.cpu().numpy()
        if hasattr(value_value, 'item'): value_value = value_value.item()
        value_batch_list.append(value_value) # Store dummy value

        reward_batch_list.append(reward)
        done_batch_list.append(done)

        obs = next_obs

        # Handle episode end
        if done:
            # Reset environment
            obs, info = env.reset()
            # Reset running G in reward normalizer (handled internally by PPO normalize_reward)

        # --- Update Policy ---\
        # Check if we have enough data to fill the buffer (or update interval)
        if global_step % update_interval == 0:
            print(f"\nGlobal Step: {global_step} | Preparing PPO Update...")

            # === DEBUG PRINT ADDED ===
            print(f"[DEBUG Update Start @ {global_step}] List lengths: obs={len(obs_batch_list)}, "
                  f"action={len(action_batch_list)}, log_prob={len(log_prob_batch_list)}, "
                  f"value={len(value_batch_list)}, reward={len(reward_batch_list)}, done={len(done_batch_list)}")
            # Print first few elements to check content
            if len(log_prob_batch_list) > 5:
                 print(f"[DEBUG Update Start @ {global_step}] First 5 log_probs: {log_prob_batch_list[:5]}")
            if len(value_batch_list) > 5:
                 print(f"[DEBUG Update Start @ {global_step}] First 5 values: {value_batch_list[:5]}")
            # === END DEBUG PRINT ===


            # Use try-except block for robust tensor conversion and NaN/Inf checks
            try:
                # Check list sizes first
                if len(obs_batch_list) != update_interval or len(log_prob_batch_list) != update_interval or \
                   len(value_batch_list) != update_interval or len(action_batch_list) != update_interval or \
                   len(reward_batch_list) != update_interval or len(done_batch_list) != update_interval:
                    print(f"[ERROR @ {global_step}] Mismatched list lengths before tensor conversion! Expected {update_interval}. Skipping update.")
                    raise ValueError("Mismatched list lengths") # Raise error to enter except block

                obs_batch_tensor = torch.tensor(np.array(obs_batch_list), dtype=torch.float32)
                action_array = np.array(action_batch_list)

                log_prob_array = np.array(log_prob_batch_list)
                value_array = np.array(value_batch_list)

                # Check for NaNs or Infs *after* converting to numpy array
                if np.isnan(log_prob_array).any() or np.isinf(log_prob_array).any():
                    print(f"[ERROR @ {global_step}] NaN or Inf detected in log_prob_array before tensor conversion. Skipping update.")
                    raise ValueError("NaN/Inf in log_prob_array")
                if np.isnan(value_array).any() or np.isinf(value_array).any():
                    print(f"[ERROR @ {global_step}] NaN or Inf detected in value_array before tensor conversion. Skipping update.")
                    raise ValueError("NaN/Inf in value_array")

            except Exception as e:
                print(f"[ERROR @ {global_step}] Failed during pre-conversion checks or numpy array creation: {e}. Skipping update.")
                # Clear lists to prevent loop in case of persistent error
                obs_batch_list.clear(); action_batch_list.clear(); log_prob_batch_list.clear(); value_batch_list.clear(); reward_batch_list.clear(); done_batch_list.clear()
                continue # Skip the rest of the update logic


            # Convert numpy arrays to tensors (should be safe now)
            action_batch_tensor = torch.tensor(action_array, dtype=torch.long if action_space_type == "discrete" else torch.float32)
            log_prob_array = log_prob_array.flatten() if log_prob_array.ndim > 1 else log_prob_array
            value_array = value_array.flatten() if value_array.ndim > 1 else value_array
            log_prob_batch_tensor = torch.tensor(log_prob_array, dtype=torch.float32)
            value_batch_tensor = torch.tensor(value_array, dtype=torch.float32) # Still dummy values here
            reward_batch_tensor = torch.tensor(np.array(reward_batch_list), dtype=torch.float32)
            done_batch_tensor = torch.tensor(np.array(done_batch_list), dtype=torch.bool)


            if args.debug:
                # This print should now show correct shapes if conversion succeeded
                print(f"Tensor shapes - obs: {obs_batch_tensor.shape}, action: {action_batch_tensor.shape}, "
                      f"log_prob: {log_prob_batch_tensor.shape}, value: {value_batch_tensor.shape}")

            # Store batch data and update rewards/dones
            # The PPO algorithm's update_rewards_dones_batch method handles tracking episode returns
            # into ppo_algorithm.episode_returns deque.
            indices = ppo_algorithm.store_initial_batch(
                obs_batch_tensor, action_batch_tensor, log_prob_batch_tensor, value_batch_tensor
            )
            if indices.numel() > 0:
                 ppo_algorithm.update_rewards_dones_batch(indices, reward_batch_tensor, done_batch_tensor)
            else:
                 # This case should ideally not happen if the previous length checks pass
                 print("[WARN] store_initial_batch returned empty indices despite lists appearing full. Skipping reward/done update.")


            # Perform PPO update
            metrics = ppo_algorithm.update()
            total_updates += 1

            # Log metrics and episode statistics
            elapsed_time = time.time() - start_time
            metrics_log = {f"losses/{k}": v for k, v in metrics.items() if 'loss' in k}
            metrics_log["global_step"] = global_step
            metrics_log["total_updates"] = total_updates
            metrics_log["time/elapsed_time_s"] = elapsed_time
            metrics_log["time/sps"] = int(global_step / elapsed_time) if elapsed_time > 0 else 0
            current_lr = ppo_algorithm.optimizer.param_groups[0]['lr']
            metrics_log["charts/learning_rate"] = current_lr
            if 'mean_return' in metrics: metrics_log["charts/avg_episodic_return_buffer"] = metrics['mean_return']
            if 'mean_predicted_variance' in metrics: metrics_log["charts/mean_predicted_variance"] = metrics['mean_predicted_variance']
            if 'mean_uncertainty_weight' in metrics: metrics_log["charts/mean_uncertainty_weight"] = metrics['mean_uncertainty_weight']

            # Add PPO's internal episode returns to plotter if they exist
            internal_episodes = []
            if len(ppo_algorithm.episode_returns) > 0:
                # Get the count of newly completed episodes since last update
                # We need to ensure this tracker is initialized in PPOAlgorithm's __init__ or load_state_dict
                if not hasattr(ppo_algorithm, '_last_reported_episode_count'):
                     ppo_algorithm._last_reported_episode_count = 0

                new_episode_count = len(ppo_algorithm.episode_returns) - ppo_algorithm._last_reported_episode_count

                # Only report newly completed episodes to the plotter
                if new_episode_count > 0:
                    # Get the newly completed episodes from the end of the deque
                    internal_episodes = list(ppo_algorithm.episode_returns)[-new_episode_count:]
                    for i, ep_return in enumerate(internal_episodes):
                        # Estimate episode duration (simplified for single env)
                        # For multi-env, length from env info is better.
                        # For now, use a placeholder or derive from total steps / episode count.
                        # Let's derive step based on buffer size and number of episodes in the buffer.
                        est_episode_length = update_interval / max(1, new_episode_count) # Very rough estimate

                        # Estimate the global step at which this episode ended within the buffer interval
                        # Assuming episodes are somewhat evenly distributed within the buffer
                        # This is an approximation, more accurate with multi-env info.
                        est_step = global_step - update_interval + int((i + 1) * (update_interval / max(1, new_episode_count)))

                        # print(f"Adding PPO internal episode return to plotter: {ep_return:.2f} at step ~{est_step}") # Removed verbose print
                        plotter.add_data(
                            step=est_step,
                            episode_info={'return': ep_return, 'length': est_episode_length} # Length is rough
                        )

                # Store current count for next update
                ppo_algorithm._last_reported_episode_count = len(ppo_algorithm.episode_returns)

            reward_stats = plotter.get_reward_stats()
            # Add latest reward stats to log
            metrics_log["charts/mean_reward_100"] = reward_stats['mean_reward_last_100']
            metrics_log["charts/best_reward"] = reward_stats['best_reward']
            metrics_log["charts/total_episodes"] = reward_stats['num_episodes']

            # Print consolidated stats
            print(f"Global Step: {global_step}, Update: {total_updates}, SPS: {metrics_log['time/sps']}")
            print(f"  Losses - Actor: {metrics.get('actor_loss', 0):.3f}, Critic: {metrics.get('critic_loss_total', 0):.3f} "
                  f"(M: {metrics.get('critic_loss_mean', 0):.3f}, V: {metrics.get('critic_loss_variance', 0):.3f}), "
                  f"Entropy: {metrics.get('entropy_loss', 0):.3f}")
            print(f"  Stats - Mean Return (Buffer): {metrics.get('mean_return', 0):.2f}, Mean Pred Var: {metrics.get('mean_predicted_variance', 0):.3f}")
            print(f"  Episodes - PPO Internal: {len(ppo_algorithm.episode_returns)} (New this update: {len(internal_episodes)})")
            print(f"  Rewards - Mean (Plotter 100): {reward_stats['mean_reward_last_100']:.2f}, Best (Plotter): {reward_stats['best_reward']:.2f}")

            if args.use_wandb:
                wandb.log(metrics_log)

            # Clear the temporary lists *AFTER* all processing for this update is done
            obs_batch_list.clear() # Use clear() for efficiency
            action_batch_list.clear()
            log_prob_batch_list.clear()
            value_batch_list.clear()
            reward_batch_list.clear()
            done_batch_list.clear()

            # --- Save Model ---
            timesteps_since_last_save += update_interval
            if timesteps_since_last_save >= args.save_interval:
                checkpoint_path = os.path.join(save_path, f"checkpoint_step_{global_step}.pth")
                print(f"\nSaving checkpoint to {checkpoint_path}...")
                save_data = {
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'ppo_state_dict': ppo_algorithm.get_state_dict(),
                    'args': vars(args),
                    'global_step': global_step,
                }
                torch.save(save_data, checkpoint_path)
                timesteps_since_last_save = 0
                print("Checkpoint saved.")


    # --- Final Save ---
    final_checkpoint_path = os.path.join(save_path, "checkpoint_final.pth")
    print(f"\nTraining finished. Saving final checkpoint to {final_checkpoint_path}...")
    save_data = {
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'ppo_state_dict': ppo_algorithm.get_state_dict(),
        'args': vars(args),
        'global_step': global_step,
    }
    torch.save(save_data, final_checkpoint_path)
    print("Final checkpoint saved.")

    # --- Cleanup ---
    env.close()
    plotter.update_plot() # Final plot update
    final_stats = plotter.get_reward_stats()
    print("\n" + "="*50 + "\nTRAINING COMPLETE\n" + "="*50)
    print(f"Total steps: {global_step}")
    print(f"Total episodes: {final_stats['num_episodes']}")
    print(f"Mean reward (last 100 episodes): {final_stats['mean_reward_last_100']:.2f}")
    print(f"Best episode reward: {final_stats['best_reward']:.2f}")
    print(f"Final plots saved to {os.path.join(save_path, 'plots')}")
    print("="*50)
    plotter.close()

    if args.use_wandb:
        wandb.finish()
    print("Training complete.")
