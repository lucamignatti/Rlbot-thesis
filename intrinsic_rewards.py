from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
# Import GradScaler and autocast
from torch.amp import GradScaler, autocast

class IntrinsicRewardGenerator:
    """Base class for intrinsic reward generators"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, device: str = "cpu", use_amp: bool = False):
        """Initialize the intrinsic reward generator

        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
            device: Device to use for computation
            use_amp: Whether to use Automatic Mixed Precision
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        # Enable AMP only if flag is true and device is CUDA
        self.use_amp = use_amp and "cuda" in str(device)

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute intrinsic reward for a given transition

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            intrinsic_reward: Intrinsic reward value
        """
        raise NotImplementedError("Subclasses must implement compute_intrinsic_reward")

    def update(self, state, action, next_state, done=False):
        """Update the intrinsic reward model based on the transition

        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            done: Whether the episode is done

        Returns:
            loss: Loss value from the update
        """
        raise NotImplementedError("Subclasses must implement update")

    def reset_models(self):
        """Reset the models used for computing intrinsic rewards"""
        raise NotImplementedError("Subclasses must implement reset_models")

    # Add get_state_dict and load_state_dict for saving/loading
    def get_state_dict(self):
        """Get state dict for saving generator state"""
        # Default implementation, subclasses should override if they have more state
        return {}

    def load_state_dict(self, state_dict):
        """Load state dict for resuming generator state"""
        # Default implementation, subclasses should override
        pass


class CuriosityReward(IntrinsicRewardGenerator): # Inherit from base class
    """
    Intrinsic curiosity module that rewards agent for exploring novel states.
    Uses a forward model to predict the next state features given the current state and action.
    The prediction error is used as the intrinsic reward.
    """

    def __init__(
        self,
        observation_shape,
        action_shape,
        feature_dim=256,
        hidden_dim=256,
        lr=1e-4,
        device="cuda",
        normalize_rewards=True,
        reward_scale=1.0,
        use_amp=False # Add use_amp
    ):
        super().__init__(observation_shape, action_shape, hidden_dim, device, use_amp) # Call base init
        self.normalize_rewards = normalize_rewards
        self.reward_scale = reward_scale

        # Create feature encoder for state representation
        self.feature_encoder = nn.Sequential(
            nn.Linear(observation_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        ).to(device)

        # Create forward model for next state prediction
        if isinstance(action_shape, int):
            action_dim = action_shape
        else:
            action_dim = np.prod(action_shape)

        # Store action_dim as an instance variable
        self.action_dim = action_dim

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        ).to(device)

        # Create optimizer
        params = list(self.feature_encoder.parameters()) + list(self.forward_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Initialize GradScaler if AMP is enabled
        self.scaler = GradScaler(enabled=self.use_amp)

        # Initialize reward normalization (for the raw prediction error)
        self.reward_normalizer = RunningMeanStd(shape=())

    def compute_features(self, state):
        """Extract feature representation from state"""
        # Use autocast for the forward pass if AMP is enabled
        with autocast("cuda", enabled=self.use_amp):
            return self.feature_encoder(state)

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute curiosity-based intrinsic reward."""
        # Switch to evaluation mode
        self.feature_encoder.eval()
        self.forward_model.eval()

        with torch.no_grad():
            # Ensure state is a torch tensor on the correct device
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            elif state.device != self.device:
                state = state.to(self.device)

            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            elif next_state.device != self.device:
                next_state = next_state.to(self.device)

            # Ensure action is a torch tensor on the correct device
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)
            elif action.device != self.device:
                action = action.to(self.device) # Explicitly move action to the correct device

            # Add batch dimension if needed
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)

            # Handle differently shaped action tensors
            if action.dim() == 0:  # Single scalar value
                action = action.unsqueeze(0).unsqueeze(0)  # Make it [1, 1]
            elif action.dim() == 1:  # Vector of indices [batch_size]
                # For discrete actions, we'll convert to one-hot later
                # For continuous actions with a single value, add a feature dimension
                if not isinstance(self.action_dim, int):  # Continuous case
                    action = action.unsqueeze(1)  # Make it [batch_size, 1]
                else:
                    # Make discrete actions [batch_size, 1] for one-hot encoding later
                    action = action.unsqueeze(1)

            # Ensure batch dimensions match
            batch_size = state.size(0)
            if action.size(0) != batch_size:
                if action.size(0) == 1:  # Single action for multiple states
                    action = action.repeat(batch_size, 1)
                else:
                    # This case should be handled by the caller, but just in case
                    raise ValueError(f"Action batch size {action.size(0)} doesn't match state batch size {batch_size}")

            # Handle discrete actions by converting to one-hot encoding
            if isinstance(self.action_dim, int):
                one_hot_action = torch.zeros((action.size(0), self.action_dim), device=self.device)
                # Ensure action is long type for scatter_
                one_hot_action.scatter_(1, action.long(), 1.0)
                action = one_hot_action

            # Compute features (already uses autocast internally)
            current_features = self.compute_features(state)
            next_features = self.compute_features(next_state)

            # Forward model prediction within autocast context
            with autocast("cuda", enabled=self.use_amp):
                # Ensure both tensors are on the same device before concatenating
                forward_input = torch.cat([current_features.to(self.device), action.to(self.device)], dim=1) # Ensure both tensors are on the correct device
                predicted_next_features = self.forward_model(forward_input)

                # Calculate prediction error (MSE)
                prediction_error = F.mse_loss(predicted_next_features, next_features, reduction='none').sum(dim=1)

            # Ensure prediction_error is a tensor before converting to numpy
            if not isinstance(prediction_error, torch.Tensor):
                prediction_error = torch.tensor(prediction_error, device=self.device)

            # Normalize reward (prediction error)
            raw_reward = prediction_error.cpu().numpy()
            if self.normalize_rewards:
                self.reward_normalizer.update(raw_reward)
                normalized_reward = raw_reward / (np.sqrt(self.reward_normalizer.var) + 1e-8)
                # Apply fixed reward scale here (no adaptive logic)
                clipped_reward = np.clip(normalized_reward, 0, 5) # * self.reward_scale (Scale applied in ensemble/trainer)
            else:
                clipped_reward = np.clip(raw_reward, 0, 5) # * self.reward_scale (Scale applied in ensemble/trainer)

        # Switch back to training mode
        self.feature_encoder.train()
        self.forward_model.train()

        # Ensure a scalar tensor is returned by taking the mean
        scalar_reward = np.mean(clipped_reward)
        return torch.tensor(scalar_reward, device=self.device, dtype=torch.float32)

    def update(self, state, action, next_state, done=False):
        """Update forward model to improve prediction accuracy"""
        # Convert inputs to tensors if they aren't already and move to device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif state.device != self.device:
            state = state.to(self.device)

        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        elif next_state.device != self.device:
            next_state = next_state.to(self.device)

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        elif action.device != self.device:
            action = action.to(self.device) # Explicitly move action to the correct device

        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Handle discrete actions by converting to one-hot encoding
        if action.shape[1] == 1 and isinstance(self.action_dim, int):
            one_hot_action = torch.zeros((action.size(0), self.action_dim), device=self.device)
            # Ensure action is long type for scatter_
            one_hot_action.scatter_(1, action.long(), 1.0)
            action = one_hot_action

        # Use autocast for forward passes and loss calculation
        with autocast("cuda", enabled=self.use_amp):
            # Extract features (uses autocast internally)
            current_features = self.compute_features(state)
            next_features = self.compute_features(next_state)

            # Predict next features
            # Ensure both tensors are on the same device before concatenating
            forward_input = torch.cat([current_features.to(self.device), action.to(self.device)], dim=1) # Ensure both tensors are on the correct device
            predicted_next_features = self.forward_model(forward_input)

            # Calculate loss
            forward_loss = F.mse_loss(predicted_next_features, next_features.detach())

        # Update models using GradScaler
        self.optimizer.zero_grad()
        self.scaler.scale(forward_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'curiosity_loss': forward_loss.item()} # Return dict

    def reset_models(self):
        """Reset the models used for computing intrinsic rewards"""
        # Reset feature encoder weights
        for layer in self.feature_encoder.modules():
            if isinstance(layer, nn.Linear):
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # Reset forward model weights
        for layer in self.forward_model.modules():
            if isinstance(layer, nn.Linear):
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # Reset optimizer
        self.optimizer = torch.optim.Adam(
            list(self.feature_encoder.parameters()) + list(self.forward_model.parameters()),
            lr=self.optimizer.param_groups[0]['lr']
        )

        # Reset GradScaler state if AMP is enabled
        if self.use_amp:
            self.scaler = GradScaler(enabled=True)

        # Reset reward normalization
        self.reward_normalizer = RunningMeanStd(shape=())

    def get_state_dict(self):
        """Get state dict for saving CuriosityReward state"""
        return {
            'feature_encoder_state': self.feature_encoder.state_dict(),
            'forward_model_state': self.forward_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict() if self.use_amp else None,
            'reward_normalizer_mean': self.reward_normalizer.mean,
            'reward_normalizer_var': self.reward_normalizer.var,
            'reward_normalizer_count': self.reward_normalizer.count,
            'reward_scale': self.reward_scale,
            'normalize_rewards': self.normalize_rewards
        }

    def load_state_dict(self, state_dict):
        """Load state dict for resuming CuriosityReward state"""
        self.feature_encoder.load_state_dict(state_dict['feature_encoder_state'])
        self.forward_model.load_state_dict(state_dict['forward_model_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        if self.use_amp and 'scaler_state' in state_dict and state_dict['scaler_state'] is not None:
            self.scaler.load_state_dict(state_dict['scaler_state'])
        self.reward_normalizer.mean = state_dict['reward_normalizer_mean']
        self.reward_normalizer.var = state_dict['reward_normalizer_var']
        self.reward_normalizer.count = state_dict['reward_normalizer_count']
        self.reward_scale = state_dict.get('reward_scale', 1.0) # Load reward scale
        self.normalize_rewards = state_dict.get('normalize_rewards', True)


class RNDReward(IntrinsicRewardGenerator):
    """Implementation of Random Network Distillation (RND)"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 learning_rate: float = 1e-3, device: str = "cpu", use_amp: bool = False,
                 normalize_rewards=True, reward_scale=1.0): # Add normalize and scale
        """Initialize the RND reward generator

        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space (not used for RND)
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for the optimizer
            device: Device to use for computation
            use_amp: Whether to use Automatic Mixed Precision
            normalize_rewards: Whether to normalize the raw prediction error
            reward_scale: Fixed scaling factor for the reward
        """
        super().__init__(obs_dim, action_dim, hidden_dim, device, use_amp) # Pass use_amp to base
        self.normalize_rewards = normalize_rewards
        self.reward_scale = reward_scale # Fixed scale

        # Random target network (fixed)
        self.target_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)

        # Predictor network (trained to match target)
        self.predictor_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)

        # Initialize target network with random weights and freeze
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Optimizer for predictor network
        self.optimizer = Adam(self.predictor_network.parameters(), lr=learning_rate)

        # Initialize GradScaler if AMP is enabled
        self.scaler = GradScaler(enabled=self.use_amp)

        # Running normalization for observations and rewards
        self.obs_normalizer = RunningMeanStd(shape=(obs_dim,))
        self.reward_normalizer = RunningMeanStd(shape=())

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute RND-based intrinsic reward

        Args:
            state: Current state (not used for RND)
            action: Action taken (not used for RND)
            next_state: Next state [B, obs_dim]

        Returns:
            intrinsic_reward: Intrinsic reward value [B]
        """
        # Switch to evaluation mode
        self.predictor_network.eval()

        with torch.no_grad():
            # Ensure next_state is a torch tensor on the correct device
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            elif next_state.device != self.device:
                next_state = next_state.to(self.device)

            # Add batch dimension if needed
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)

            # Normalize observations using running statistics
            next_state_np = next_state.cpu().numpy()
            self.obs_normalizer.update(next_state_np)
            normalized_obs = (next_state_np - self.obs_normalizer.mean) / (np.sqrt(self.obs_normalizer.var) + 1e-8)
            normalized_obs = torch.tensor(normalized_obs, dtype=torch.float32, device=self.device)

            # Compute target and prediction within autocast context
            with autocast("cuda", enabled=self.use_amp):
                target_features = self.target_network(normalized_obs)
                predicted_features = self.predictor_network(normalized_obs)

                # Compute prediction error (intrinsic reward)
                prediction_error = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=1)

            # Normalize reward if enabled
            reward_np = prediction_error.cpu().numpy()
            if self.normalize_rewards:
                self.reward_normalizer.update(reward_np)
                normalized_reward = reward_np / (np.sqrt(self.reward_normalizer.var) + 1e-8)
                # Apply fixed reward scale here (no adaptive logic)
                clipped_reward = np.clip(normalized_reward, 0, 5.0) # * self.reward_scale (Scale applied in ensemble/trainer)
            else:
                clipped_reward = np.clip(reward_np, 0, 5.0) # * self.reward_scale (Scale applied in ensemble/trainer)

        # Switch back to training mode
        self.predictor_network.train()

        # Ensure a scalar tensor is returned by taking the mean
        scalar_reward = np.mean(clipped_reward)
        return torch.tensor(scalar_reward, device=self.device, dtype=torch.float32)

    def update(self, state, action, next_state, done=False):
        """Update the RND model based on the transition

        Args:
            state: Current state (not used for RND)
            action: Action taken (not used for RND)
            next_state: Next state [B, obs_dim]
            done: Whether the episode is done (not used)

        Returns:
            loss: Loss value from the update
        """
        # Ensure next_state is a torch tensor on the correct device
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        elif next_state.device != self.device:
            next_state = next_state.to(self.device)

        # Add batch dimension if needed
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)

        # Normalize observations using running statistics
        next_state_np = next_state.cpu().numpy()
        normalized_obs = (next_state_np - self.obs_normalizer.mean) / (np.sqrt(self.obs_normalizer.var) + 1e-8)
        normalized_obs = torch.tensor(normalized_obs, dtype=torch.float32, device=self.device)

        # Use autocast for forward passes and loss calculation
        with autocast("cuda", enabled=self.use_amp):
            # Compute target and prediction
            with torch.no_grad():
                target_features = self.target_network(normalized_obs)
            predicted_features = self.predictor_network(normalized_obs)

            # Compute loss
            loss = F.mse_loss(predicted_features, target_features)

        # Backward and optimize using GradScaler
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'rnd_loss': loss.item()}

    def reset_models(self):
        """Reset the models used for computing intrinsic rewards"""
        # Reset predictor network (target network stays fixed)
        for layer in self.predictor_network.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Reset optimizer
        self.optimizer = Adam(self.predictor_network.parameters(),
                              lr=self.optimizer.param_groups[0]['lr'])

        # Reset GradScaler state if AMP is enabled
        if self.use_amp:
            self.scaler = GradScaler(enabled=True)

        # Reset normalizers
        self.obs_normalizer = RunningMeanStd(shape=(self.obs_dim,))
        self.reward_normalizer = RunningMeanStd(shape=())

    def get_state_dict(self):
        """Get state dict for saving RNDReward state"""
        return {
            'predictor_network_state': self.predictor_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict() if self.use_amp else None,
            'obs_normalizer_mean': self.obs_normalizer.mean,
            'obs_normalizer_var': self.obs_normalizer.var,
            'obs_normalizer_count': self.obs_normalizer.count,
            'reward_normalizer_mean': self.reward_normalizer.mean,
            'reward_normalizer_var': self.reward_normalizer.var,
            'reward_normalizer_count': self.reward_normalizer.count,
            'reward_scale': self.reward_scale, # Save fixed scale
            'normalize_rewards': self.normalize_rewards
        }

    def load_state_dict(self, state_dict):
        """Load state dict for resuming RNDReward state"""
        self.predictor_network.load_state_dict(state_dict['predictor_network_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        if self.use_amp and 'scaler_state' in state_dict and state_dict['scaler_state'] is not None:
            self.scaler.load_state_dict(state_dict['scaler_state'])
        self.obs_normalizer.mean = state_dict['obs_normalizer_mean']
        self.obs_normalizer.var = state_dict['obs_normalizer_var']
        self.obs_normalizer.count = state_dict['obs_normalizer_count']
        self.reward_normalizer.mean = state_dict['reward_normalizer_mean']
        self.reward_normalizer.var = state_dict['reward_normalizer_var']
        self.reward_normalizer.count = state_dict['reward_normalizer_count']
        self.reward_scale = state_dict.get('reward_scale', 1.0) # Load fixed scale
        self.normalize_rewards = state_dict.get('normalize_rewards', True)


class RunningMeanStd:
    """Tracks the mean and standard deviation of a stream of values"""

    def __init__(self, epsilon=1e-4, shape=()):
        """Initialize the running statistics tracker

        Args:
            epsilon: Small constant to avoid numerical instability
            shape: Shape of the values to track
        """
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon  # Small epsilon to avoid division by zero

    def update(self, x):
        """Update statistics with new values

        Args:
            x: New values to include in the statistics
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # Update statistics using Welford's online algorithm
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update statistics using the batch mean, variance, and count

        Args:
            batch_mean: Mean of the batch
            batch_var: Variance of the batch
            batch_count: Number of samples in the batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * batch_count / tot_count

        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        # Update count
        new_count = tot_count

        # Store updated values
        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class IntrinsicRewardEnsemble:
    """Combines multiple intrinsic reward generators"""

    def __init__(self, reward_generators, weights=None):
        """Initialize the ensemble

        Args:
            reward_generators: Dictionary of reward generators
            weights: Dictionary of weights for each generator (same keys as reward_generators)
        """
        self.reward_generators = reward_generators

        # Use equal weights if not specified
        if weights is None:
            weights = {name: 1.0 / len(reward_generators) for name in reward_generators}
        self.weights = weights

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute combined intrinsic reward

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            intrinsic_reward: Combined intrinsic reward value (scalar float)
        """
        # Ensure inputs are tensors on the correct device
        device = next(iter(self.reward_generators.values())).device # Get device from first generator
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        elif state.device != device:
            state = state.to(device)

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=device)
        elif action.device != device:
            action = action.to(device)

        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        elif next_state.device != device:
            next_state = next_state.to(device)

        # Add batch dimension if needed
        if state.dim() == 1: state = state.unsqueeze(0)
        if action.dim() == 1: action = action.unsqueeze(0)
        if next_state.dim() == 1: next_state = next_state.unsqueeze(0)

        # Compute rewards from each generator
        rewards = {}
        for name, generator in self.reward_generators.items():
            # Generator should return tensor on correct device
            rewards[name] = generator.compute_intrinsic_reward(state, action, next_state)

        # Combine rewards using weights - Use .item() to ensure scalar multiplication
        combined_reward_val = 0.0
        for name, reward in rewards.items():
            weight = self.weights.get(name, 0.0)
            try:
                reward_item = reward.item()
                term = weight * reward_item
                combined_reward_val += term
            except Exception as e:
                # Handle potential errors during item extraction or multiplication
                print(f"Error processing reward for {name}: {e}")
                # Optionally add a default value or skip if error occurs
                # combined_reward_val += 0.0

        # Return ONLY the combined scalar reward value
        return combined_reward_val

    def update(self, state, action, next_state, done=False):
        """Update all reward generators

        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            done: Whether the episode is done

        Returns:
            losses: Dictionary of losses from each generator
        """
        # Ensure inputs are tensors on the correct device
        device = next(iter(self.reward_generators.values())).device # Get device from first generator
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        elif state.device != device:
            state = state.to(device)

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=device)
        elif action.device != device:
            action = action.to(device)

        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        elif next_state.device != device:
            next_state = next_state.to(device)

        # Add batch dimension if needed
        if state.dim() == 1: state = state.unsqueeze(0)
        if action.dim() == 1: action = action.unsqueeze(0)
        if next_state.dim() == 1: next_state = next_state.unsqueeze(0)

        losses = {}
        for name, generator in self.reward_generators.items():
            generator_losses = generator.update(state, action, next_state, done)
            # Ensure generator_losses is a dictionary
            if isinstance(generator_losses, dict):
                losses.update({f"{name}_{k}": v for k, v in generator_losses.items()})
            else: # Handle case where generator might return a single loss value
                losses[f"{name}_loss"] = generator_losses


        return losses

    def reset_models(self):
        """Reset all reward generator models"""
        for generator in self.reward_generators.values():
            generator.reset_models()

    # Add get_state_dict and load_state_dict for saving/loading
    def get_state_dict(self):
        """Get state dict for saving ensemble state"""
        state = {'weights': self.weights}
        for name, generator in self.reward_generators.items():
            if hasattr(generator, 'get_state_dict'):
                 state[name] = generator.get_state_dict()
            # Removed fallback saving logic as generators should implement get_state_dict
        return state

    def load_state_dict(self, state_dict):
        """Load state dict for resuming ensemble state"""
        self.weights = state_dict.get('weights', self.weights)
        for name, generator in self.reward_generators.items():
            if name in state_dict:
                gen_state = state_dict[name]
                if hasattr(generator, 'load_state_dict'):
                    generator.load_state_dict(gen_state)
                # Removed fallback loading logic


def create_intrinsic_reward_generator(obs_dim, action_dim,
                                      use_curiosity=True, use_rnd=True,
                                      curiosity_weight=0.5, rnd_weight=0.5,
                                      hidden_dim=128, device="cpu", use_amp=False,
                                      normalize_rewards=True, reward_scale=1.0): # Add normalize and scale args
    """Create an intrinsic reward generator based on selected options

    Args:
        obs_dim: Dimension of observation space
        action_dim: Dimension of action space
        use_curiosity: Whether to use curiosity-based rewards
        use_rnd: Whether to use RND-based rewards
        curiosity_weight: Weight for curiosity rewards in the ensemble
        rnd_weight: Weight for RND rewards in the ensemble
        hidden_dim: Hidden dimension for networks
        device: Device to use for computation
        use_amp: Whether to use Automatic Mixed Precision
        normalize_rewards: Whether to normalize raw intrinsic rewards (prediction errors)
        reward_scale: Fixed scaling factor applied AFTER normalization/clipping

    Returns:
        IntrinsicRewardGenerator: A single generator or ensemble
    """
    # Create the selected generators
    generators = {}
    weights = {}

    if use_curiosity:
        generators['curiosity'] = CuriosityReward(
            obs_dim, action_dim, hidden_dim, device=device, use_amp=use_amp,
            normalize_rewards=normalize_rewards, reward_scale=reward_scale # Pass args
        )
        weights['curiosity'] = curiosity_weight

    if use_rnd:
        generators['rnd'] = RNDReward(
            obs_dim, action_dim, hidden_dim, device=device, use_amp=use_amp,
            normalize_rewards=normalize_rewards, reward_scale=reward_scale # Pass args
        )
        weights['rnd'] = rnd_weight

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0: # Avoid division by zero if no generators selected
        weights = {k: v / total_weight for k, v in weights.items()}

    # Return the appropriate generator based on selection
    if len(generators) == 0:
        # Return None if no intrinsic rewards are used
        return None
    elif len(generators) == 1:
        # Return the single generator
        return next(iter(generators.values()))
    else:
        # Return an ensemble of generators
        return IntrinsicRewardEnsemble(generators, weights)
