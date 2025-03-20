from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

class IntrinsicRewardGenerator:
    """Base class for intrinsic reward generators"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, device: str = "cpu"):
        """Initialize the intrinsic reward generator
        
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
            device: Device to use for computation
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
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


class CuriosityReward(IntrinsicRewardGenerator):
    """Implementation of Intrinsic Curiosity Module (ICM)"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, 
                 forward_coef: float = 0.2, inverse_coef: float = 0.8,
                 learning_rate: float = 1e-3, device: str = "cpu"):
        """Initialize the curiosity reward generator
        
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
            forward_coef: Coefficient for the forward model loss
            inverse_coef: Coefficient for the inverse model loss
            learning_rate: Learning rate for the optimizer
            device: Device to use for computation
        """
        super().__init__(obs_dim, action_dim, hidden_dim, device)
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        ).to(device)
        
        # Forward model predicts next state features from current state features and action
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        
        # Inverse model predicts action from current and next state features
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        
        # Loss weights
        self.forward_coef = forward_coef
        self.inverse_coef = inverse_coef
        
        # Optimizer
        self.optimizer = Adam(list(self.feature_encoder.parameters()) + 
                              list(self.forward_model.parameters()) + 
                              list(self.inverse_model.parameters()), 
                              lr=learning_rate)
        
        # Running normalization for rewards
        self.reward_normalizer = RunningMeanStd(shape=())
        
    def compute_features(self, state):
        """Compute features from state
        
        Args:
            state: State tensor [B, obs_dim]
            
        Returns:
            features: Feature tensor [B, hidden_dim]
        """
        # Ensure state is a torch tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Compute features
        features = self.feature_encoder(state)
        return features
        
    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute curiosity-based intrinsic reward
        
        Args:
            state: Current state [B, obs_dim]
            action: Action taken [B, action_dim]
            next_state: Next state [B, obs_dim]
            
        Returns:
            intrinsic_reward: Intrinsic reward value [B]
        """
        # Switch to evaluation mode
        self.feature_encoder.eval()
        self.forward_model.eval()
        
        with torch.no_grad():
            # Compute features
            current_features = self.compute_features(state)
            next_features = self.compute_features(next_state)
            
            # Prepare action
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)
                
            # Add batch dimension if needed
            if action.dim() == 1:
                action = action.unsqueeze(0)
                
            # Predict next features
            forward_input = torch.cat([current_features, action], dim=1)
            predicted_next_features = self.forward_model(forward_input)
            
            # Compute prediction error (intrinsic reward)
            prediction_error = F.mse_loss(predicted_next_features, next_features, reduction='none').mean(dim=1)
            
            # Normalize reward if we have enough history
            reward_np = prediction_error.cpu().numpy()
            self.reward_normalizer.update(reward_np)
            normalized_reward = reward_np / (np.sqrt(self.reward_normalizer.var) + 1e-8)
            
            # Clip rewards to prevent outliers
            clipped_reward = np.clip(normalized_reward, 0, 5.0)
            
        # Switch back to training mode
        self.feature_encoder.train()
        self.forward_model.train()
        
        return torch.tensor(clipped_reward, device=self.device)
    
    def update(self, state, action, next_state, done=False):
        """Update the curiosity model based on the transition
        
        Args:
            state: Current state [B, obs_dim]
            action: Action taken [B, action_dim]
            next_state: Next state [B, obs_dim]
            done: Whether the episode is done (not used)
            
        Returns:
            loss: Loss value from the update
        """
        # Ensure inputs are tensors
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
            
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute features
        current_features = self.compute_features(state)
        next_features = self.compute_features(next_state)
        
        # Forward model: predict next state features
        forward_input = torch.cat([current_features, action], dim=1)
        predicted_next_features = self.forward_model(forward_input)
        
        # Forward model loss
        forward_loss = F.mse_loss(predicted_next_features, next_features.detach())
        
        # Inverse model: predict action from state and next state
        inverse_input = torch.cat([current_features, next_features], dim=1)
        predicted_action = self.inverse_model(inverse_input)
        
        # Inverse model loss (depending on action type)
        if action.shape[1] == 1:  # Discrete action (assuming one-hot encoded)
            inverse_loss = F.cross_entropy(predicted_action, action.squeeze(1).long())
        else:  # Continuous action
            inverse_loss = F.mse_loss(predicted_action, action)
            
        # Total loss
        total_loss = self.forward_coef * forward_loss + self.inverse_coef * inverse_loss
        
        # Backward and optimize
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'total_loss': total_loss.item()
        }
        
    def reset_models(self):
        """Reset the models used for computing intrinsic rewards"""
        # Reset feature encoder
        for layer in self.feature_encoder.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
        # Reset forward model
        for layer in self.forward_model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
        # Reset inverse model
        for layer in self.inverse_model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
        # Reset optimizer
        self.optimizer = Adam(list(self.feature_encoder.parameters()) + 
                              list(self.forward_model.parameters()) + 
                              list(self.inverse_model.parameters()), 
                              lr=self.optimizer.param_groups[0]['lr'])
                              
        # Reset reward normalizer
        self.reward_normalizer = RunningMeanStd(shape=())


class RNDReward(IntrinsicRewardGenerator):
    """Implementation of Random Network Distillation (RND)"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, 
                 learning_rate: float = 1e-3, device: str = "cpu"):
        """Initialize the RND reward generator
        
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space (not used for RND)
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for the optimizer
            device: Device to use for computation
        """
        super().__init__(obs_dim, action_dim, hidden_dim, device)
        
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
            # Ensure next_state is a torch tensor
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                
            # Add batch dimension if needed
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)
                
            # Normalize observations using running statistics
            next_state_np = next_state.cpu().numpy()
            self.obs_normalizer.update(next_state_np)
            normalized_obs = (next_state_np - self.obs_normalizer.mean) / (np.sqrt(self.obs_normalizer.var) + 1e-8)
            normalized_obs = torch.tensor(normalized_obs, dtype=torch.float32, device=self.device)
            
            # Compute target and prediction
            target_features = self.target_network(normalized_obs)
            predicted_features = self.predictor_network(normalized_obs)
            
            # Compute prediction error (intrinsic reward)
            prediction_error = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=1)
            
            # Normalize reward if we have enough history
            reward_np = prediction_error.cpu().numpy()
            self.reward_normalizer.update(reward_np)
            normalized_reward = reward_np / (np.sqrt(self.reward_normalizer.var) + 1e-8)
            
            # Clip rewards to prevent outliers
            clipped_reward = np.clip(normalized_reward, 0, 5.0)
            
        # Switch back to training mode
        self.predictor_network.train()
        
        return torch.tensor(clipped_reward, device=self.device)
    
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
        # Ensure next_state is a torch tensor
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            
        # Add batch dimension if needed
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
            
        # Normalize observations using running statistics
        next_state_np = next_state.cpu().numpy()
        normalized_obs = (next_state_np - self.obs_normalizer.mean) / (np.sqrt(self.obs_normalizer.var) + 1e-8)
        normalized_obs = torch.tensor(normalized_obs, dtype=torch.float32, device=self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute target and prediction
        with torch.no_grad():
            target_features = self.target_network(normalized_obs)
        predicted_features = self.predictor_network(normalized_obs)
        
        # Compute loss
        loss = F.mse_loss(predicted_features, target_features)
        
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        
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
                              
        # Reset normalizers
        self.obs_normalizer = RunningMeanStd(shape=(self.obs_dim,))
        self.reward_normalizer = RunningMeanStd(shape=())


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
            intrinsic_reward: Combined intrinsic reward value
        """
        # Compute rewards from each generator
        rewards = {}
        for name, generator in self.reward_generators.items():
            rewards[name] = generator.compute_intrinsic_reward(state, action, next_state)
            
        # Combine rewards using weights
        combined_reward = sum(self.weights[name] * reward for name, reward in rewards.items())
        
        return combined_reward, rewards
    
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
        losses = {}
        for name, generator in self.reward_generators.items():
            generator_losses = generator.update(state, action, next_state, done)
            losses[name] = generator_losses
            
        return losses
        
    def reset_models(self):
        """Reset all reward generator models"""
        for generator in self.reward_generators.values():
            generator.reset_models()


def create_intrinsic_reward_generator(obs_dim, action_dim, 
                                      use_curiosity=True, use_rnd=True, 
                                      curiosity_weight=0.5, rnd_weight=0.5,
                                      hidden_dim=128, device="cpu"):
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
        
    Returns:
        IntrinsicRewardGenerator: A single generator or ensemble
    """
    # Create the selected generators
    generators = {}
    weights = {}
    
    if use_curiosity:
        generators['curiosity'] = CuriosityReward(obs_dim, action_dim, hidden_dim, device=device)
        weights['curiosity'] = curiosity_weight
        
    if use_rnd:
        generators['rnd'] = RNDReward(obs_dim, action_dim, hidden_dim, device=device)
        weights['rnd'] = rnd_weight
        
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Return the appropriate generator based on selection
    if len(generators) == 0:
        raise ValueError("At least one intrinsic reward type must be selected")
    elif len(generators) == 1:
        # Return the single generator
        return next(iter(generators.values()))
    else:
        # Return an ensemble of generators
        return IntrinsicRewardEnsemble(generators, weights)