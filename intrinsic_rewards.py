import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union

class IntrinsicRewardGenerator:
    """Base class for intrinsic reward generation mechanisms"""
    def __init__(self, input_dim: int, device: str = "cpu"):
        self.input_dim = input_dim
        self.device = device
    
    def compute_reward(self, state, action=None, next_state=None) -> float:
        """Compute intrinsic reward based on state or state-action-next_state tuple"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def update(self, state, action, next_state, done=False) -> Dict[str, float]:
        """Update internal models and return training metrics"""
        raise NotImplementedError("Subclasses must implement this method")


class CuriosityReward(IntrinsicRewardGenerator):
    """
    Implementation of Intrinsic Curiosity Module (ICM)
    Rewards agent for encountering states that are hard to predict
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256, 
                 eta: float = 0.01, beta: float = 0.2, device: str = "cpu"):
        super().__init__(input_dim, device)
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.eta = eta  # Scaling factor for intrinsic reward
        self.beta = beta  # Weight for forward vs inverse model loss
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(device)
        
        # Forward model: predicts next state features from current state features and action
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        
        # Inverse model: predicts action from current and next state features
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + 
                                        list(self.forward_model.parameters()) + 
                                        list(self.inverse_model.parameters()), 
                                        lr=1e-4)
        
        self.latest_reward = 0.0
        self.latest_forward_loss = 0.0
        self.latest_inverse_loss = 0.0
        
    def compute_reward(self, state, action=None, next_state=None) -> float:
        """Compute curiosity reward based on prediction error"""
        # We need all three: state, action and next_state
        if action is None or next_state is None:
            return self.latest_reward
            
        with torch.no_grad():
            # Convert to tensors if they aren't already
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if not isinstance(action, torch.Tensor):
                action = torch.FloatTensor(action).to(self.device)
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.FloatTensor(next_state).to(self.device)
                
            # Ensure proper dimensionality
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)
                
            # Get feature representations
            phi_state = self.encoder(state)
            phi_next_state = self.encoder(next_state)
            
            # Predict next state features
            if action.shape[1] != self.action_dim:
                # Handle one-hot or discrete actions
                if action.shape[1] == 1:
                    # Convert single index to one-hot
                    one_hot_action = torch.zeros(action.size(0), self.action_dim, device=self.device)
                    one_hot_action.scatter_(1, action.long(), 1)
                    action_input = one_hot_action
                else:
                    # Already in appropriate form
                    action_input = action
            else:
                action_input = action
                
            # Concatenate state features and action
            forward_input = torch.cat([phi_state, action_input], dim=1)
            
            # Predict next state
            pred_next_phi = self.forward_model(forward_input)
            
            # Prediction error as reward
            forward_loss = F.mse_loss(pred_next_phi, phi_next_state, reduction='none').sum(dim=1)
            reward = self.eta * forward_loss.item()
            
            self.latest_reward = reward
            return reward
    
    def update(self, state, action, next_state, done=False) -> Dict[str, float]:
        """Update predictive models based on experience"""
        # Convert to tensors if they aren't already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action).to(self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).to(self.device)
            
        # Ensure proper dimensionality
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
            
        # Get feature representations
        phi_state = self.encoder(state)
        phi_next_state = self.encoder(next_state)
        
        # Format action for network input
        if action.shape[1] != self.action_dim:
            # Handle one-hot or discrete actions
            if action.shape[1] == 1:
                # Convert single index to one-hot
                one_hot_action = torch.zeros(action.size(0), self.action_dim, device=self.device)
                one_hot_action.scatter_(1, action.long(), 1)
                action_input = one_hot_action
                action_target = action.squeeze(1).long()  # Target for cross entropy
            else:
                # Already in appropriate form
                action_input = action
                action_target = action  # Target for MSE
        else:
            action_input = action
            action_target = action
            
        # Forward model: predict next state features
        forward_input = torch.cat([phi_state, action_input], dim=1)
        pred_next_phi = self.forward_model(forward_input)
        forward_loss = F.mse_loss(pred_next_phi, phi_next_state)
        
        # Inverse model: predict action from states
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        pred_action = self.inverse_model(inverse_input)
        
        # Action prediction loss based on action type
        if len(action_target.shape) > 1 and action_target.shape[1] > 1:
            # For continuous or one-hot actions
            inverse_loss = F.mse_loss(pred_action, action_target)
        else:
            # For discrete actions (indexes)
            inverse_loss = F.cross_entropy(pred_action, action_target)
            
        # Combined loss with beta weighting
        total_loss = forward_loss * self.beta + inverse_loss * (1 - self.beta)
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.latest_forward_loss = forward_loss.item()
        self.latest_inverse_loss = inverse_loss.item()
        
        return {
            "forward_loss": forward_loss.item(),
            "inverse_loss": inverse_loss.item(),
            "total_loss": total_loss.item()
        }


class RNDReward(IntrinsicRewardGenerator):
    """
    Random Network Distillation implementation
    Uses a fixed random network as a target for novelty detection
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 output_dim: int = 128, scale: float = 1.0, device: str = "cpu"):
        super().__init__(input_dim, device)
        self.scale = scale
        
        # Target network - fixed random weights, never trained
        self.target_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        
        # Predictor network - tries to match target network
        self.predictor_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        
        # Fix target network weights - never train these
        for param in self.target_network.parameters():
            param.requires_grad = False
            
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=1e-4)
        self.latest_reward = 0.0
        self.latest_loss = 0.0
        
    def compute_reward(self, state, action=None, next_state=None) -> float:
        """Compute novelty reward based on prediction error to random target network"""
        with torch.no_grad():
            # Convert to tensor if it isn't already
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
                
            # Ensure proper dimensionality
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            # Get target and prediction
            target_features = self.target_network(state)
            predicted_features = self.predictor_network(state)
            
            # Calculate prediction error
            prediction_error = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=1)
            
            # Scale the reward
            reward = prediction_error.item() * self.scale
            
            self.latest_reward = reward
            return reward
    
    def update(self, state, action=None, next_state=None, done=False) -> Dict[str, float]:
        """Update predictor network based on new state"""
        # Convert to tensor if it isn't already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
            
        # Ensure proper dimensionality
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Get target features (don't compute gradients for target)
        with torch.no_grad():
            target_features = self.target_network(state)
            
        # Get predictor features
        predicted_features = self.predictor_network(state)
        
        # Compute loss
        loss = F.mse_loss(predicted_features, target_features)
        
        # Update predictor network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.latest_loss = loss.item()
        
        return {"rnd_loss": loss.item()}


class IntrinsicRewardEnsemble:
    """
    Combines multiple intrinsic reward generators
    """
    def __init__(self, generators: Dict[str, Tuple[IntrinsicRewardGenerator, float]]):
        """
        Initialize with dictionary of generators and their weights
        Args:
            generators: Dict mapping names to (generator, weight) tuples
        """
        self.generators = generators
    
    def compute_reward(self, state, action=None, next_state=None) -> float:
        """Compute weighted sum of all intrinsic rewards"""
        total_reward = 0.0
        rewards = {}
        
        for name, (generator, weight) in self.generators.items():
            reward = generator.compute_reward(state, action, next_state)
            weighted_reward = reward * weight
            total_reward += weighted_reward
            rewards[f"{name}_reward"] = reward
            
        rewards["total_intrinsic_reward"] = total_reward
        self.latest_rewards = rewards
        return total_reward
        
    def update(self, state, action, next_state, done=False) -> Dict[str, float]:
        """Update all generators and return combined metrics"""
        metrics = {}
        
        for name, (generator, _) in self.generators.items():
            generator_metrics = generator.update(state, action, next_state, done)
            for key, value in generator_metrics.items():
                metrics[f"{name}_{key}"] = value
                
        return metrics

def create_intrinsic_reward_generator(
    input_dim: int, 
    action_dim: int,
    device: str = "cpu",
    curiosity_weight: float = 0.5,
    rnd_weight: float = 0.5
) -> IntrinsicRewardEnsemble:
    """Create an ensemble of intrinsic reward generators"""
    generators = {}
    
    # Add Curiosity-driven reward
    if curiosity_weight > 0:
        curiosity = CuriosityReward(
            input_dim=input_dim,
            action_dim=action_dim,
            device=device
        )
        generators["curiosity"] = (curiosity, curiosity_weight)
    
    # Add Random Network Distillation reward
    if rnd_weight > 0:
        rnd = RNDReward(
            input_dim=input_dim,
            device=device
        )
        generators["rnd"] = (rnd, rnd_weight)
    
    return IntrinsicRewardEnsemble(generators)