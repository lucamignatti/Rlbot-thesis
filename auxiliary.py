import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math  # Add the missing math import
from collections import deque
from typing import Dict, List, Tuple, Optional, Union, Any

class StateRepresentationTask(nn.Module):
    """
    Autoencoder for State Representation (SR) auxiliary task.
    Compresses the observation to a low-dimensional space and reconstructs it.
    """
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, device="cpu"):
        super(StateRepresentationTask, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: compress input to latent space - use LayerNorm instead of BatchNorm
        # to avoid the batch size=1 issue
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),  # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_dim//2, latent_dim)
        )

        # Decoder: reconstruct from latent space - use LayerNorm instead of BatchNorm
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),  # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.to(self.device)

    def forward(self, x):
        # Encode input to latent representation
        latent = self.encoder(x)
        # Decode back to reconstruction
        reconstruction = self.decoder(latent)
        return reconstruction

    def get_loss(self, x):
        # Get reconstruction
        reconstruction = self.forward(x)
        # Calculate smooth L1 loss as mentioned in the paper
        loss = F.smooth_l1_loss(reconstruction, x)
        return loss


class RewardPredictionTask(nn.Module):
    """
    LSTM-based network for Reward Prediction (RP) auxiliary task.
    Predicts immediate rewards based on a sequence of observations.
    """
    def __init__(self, input_dim, hidden_dim=64, sequence_length=20, device="cpu"):
        super(RewardPredictionTask, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        # LSTM layer to process observation sequences
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Output layer for 3-class classification (negative, near-zero, positive reward)
        self.output_layer = nn.Linear(hidden_dim, 3)

        # Thresholds for classifying rewards
        self.pos_threshold = 0.009
        self.neg_threshold = -0.009

        self.to(self.device)

    def forward(self, x_seq):
        # x_seq shape: [batch_size, sequence_length, input_dim]
        lstm_out, _ = self.lstm(x_seq)

        # Take only the last timestep's output
        last_output = lstm_out[:, -1, :]

        # Predict reward class
        logits = self.output_layer(last_output)
        return logits

    def get_loss(self, x_seq, rewards):
        # Get reward class predictions
        logits = self.forward(x_seq)

        # Convert rewards to class labels:
        # 0: negative, 1: near-zero, 2: positive
        labels = torch.zeros_like(rewards, dtype=torch.long, device=self.device)
        labels[rewards > self.pos_threshold] = 2
        labels[rewards < self.neg_threshold] = 0
        labels[(rewards >= self.neg_threshold) & (rewards <= self.pos_threshold)] = 1

        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss


class AuxiliaryTaskManager:
    """
    Manages auxiliary tasks for both batch learning (PPO) and stream learning (StreamAC).
    Adapts to the learning algorithm being used.
    """
    def __init__(self, actor, obs_dim, sr_weight=1.0, rp_weight=1.0,
                 sr_hidden_dim=128, sr_latent_dim=32,
                 rp_hidden_dim=64, rp_sequence_length=5, 
                 device="cpu", use_amp=False, update_frequency=8,
                 learning_mode="batch"):
        """
        Initialize the auxiliary task manager
        
        Args:
            actor: The actor network (for feature extraction)
            obs_dim: Dimension of the observation space
            sr_weight: Weight for the state representation task
            rp_weight: Weight for the reward prediction task
            sr_hidden_dim: Hidden dimension for state representation
            sr_latent_dim: Latent dimension for state representation
            rp_hidden_dim: Hidden dimension for reward prediction
            rp_sequence_length: Sequence length for reward prediction
            device: Device to use for computation
            use_amp: Whether to use automatic mixed precision
            update_frequency: How often to update auxiliary tasks
            learning_mode: Either "batch" (for PPO) or "stream" (for StreamAC)
        """
        self.device = device
        self.sr_weight = sr_weight
        self.rp_weight = rp_weight
        self.rp_sequence_length = rp_sequence_length
        self.actor = actor
        self.debug = False
        self.use_amp = use_amp
        self.learning_mode = learning_mode

        # Store update frequency counter
        self.update_counter = 0
        self.update_frequency = update_frequency
        self.history_filled = 0

        # Ensure hidden_dim exists
        self.hidden_dim = getattr(actor, 'hidden_dim', 1536)

        # State representation task
        self.sr_task = StateRepresentationTask(
            input_dim=obs_dim,
            hidden_dim=sr_hidden_dim,
            latent_dim=sr_latent_dim,
            device=device
        )

        # Reward prediction task
        self.rp_task = RewardPredictionTask(
            input_dim=self.hidden_dim,
            hidden_dim=rp_hidden_dim,
            sequence_length=rp_sequence_length,
            device=device
        )

        # Initialize optimizers
        self.sr_optimizer = torch.optim.Adam(self.sr_task.parameters(), lr=3e-4)
        self.rp_optimizer = torch.optim.Adam(self.rp_task.parameters(), lr=3e-4)

        # Initialize history buffers with fixed-size tensors for better memory management
        # For batch learning (PPO), use large buffers
        if learning_mode == "batch":
            self.obs_history = deque(maxlen=10000)
            self.feature_history = deque(maxlen=10000)
            self.reward_history = deque(maxlen=10000)
        # For stream learning (StreamAC), use small buffers
        else:
            self.obs_history = deque(maxlen=rp_sequence_length * 2)
            self.feature_history = deque(maxlen=rp_sequence_length * 2)
            self.reward_history = deque(maxlen=rp_sequence_length * 2)

    def update(self, observations, rewards, features=None):
        """
        Update auxiliary task models with new experiences
        
        Args:
            observations: New observations
            rewards: New rewards
            features: Features extracted from observations (optional)
            
        Returns:
            dict: Dictionary of update metrics
        """
        # Ensure inputs are torch tensors
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            
        # Extract features if not provided
        if features is None:
            with torch.no_grad():
                if hasattr(self.actor, 'get_features'):
                    features = self.actor.get_features(observations)
                else:
                    # Use a forward pass through most of the actor network
                    try:
                        if hasattr(self.actor, 'forward') and 'return_features' in self.actor.forward.__code__.co_varnames:
                            features = self.actor(observations, return_features=True)[1]
                        else:
                            # If actor doesn't support return_features, use a default approach
                            features = self.actor(observations)
                            if isinstance(features, tuple):
                                features = features[0]  # Try to get first item if it's a tuple
                    except Exception as e:
                        if self.debug:
                            print(f"[AUX DEBUG] Error extracting features: {e}")
                        # Use observations as features if extraction fails
                        features = observations
                    
        # Ensure features is a torch tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32, device=self.device)
                
        # Store experiences in history
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        # Add experiences to history
        for i in range(observations.shape[0]):
            self.obs_history.append(observations[i].detach().cpu())
            self.feature_history.append(features[i].detach().cpu())
            self.reward_history.append(rewards[i].detach().cpu())
            
        # Increment history counter
        self.history_filled += 1
            
        # Update counter and check if we should update models
        self.update_counter += 1
        
        # For stream learning, update more frequently
        update_threshold = 1 if self.learning_mode == "stream" else self.update_frequency
        
        if self.update_counter >= update_threshold:
            self.update_counter = 0
            
            # Check if we have enough data to update
            if len(self.obs_history) >= self.rp_sequence_length:
                # Compute losses and update models
                losses = self.compute_losses()
                
                # Debug logging to diagnose zeroed out losses
                if self.debug:
                    print(f"[AUX DEBUG] SR loss: {losses['sr_loss']:.6f}, RP loss: {losses['rp_loss']:.6f}")
                    print(f"[AUX DEBUG] History lengths: Obs={len(self.obs_history)}, Features={len(self.feature_history)}, Rewards={len(self.reward_history)}")
                
                # Ensure losses are valid (not NaN or zero)
                if math.isnan(losses['sr_loss']) or math.isnan(losses['rp_loss']):
                    if self.debug:
                        print("[AUX DEBUG] NaN losses detected! Resetting auxiliary task models")
                    self.reset()
                    losses = {"sr_loss": 0.0, "rp_loss": 0.0}
                    
                return losses
            else:
                if self.debug:
                    print(f"[AUX DEBUG] Not enough history for auxiliary tasks: {len(self.obs_history)}/{self.rp_sequence_length}")
                    
        # If we don't update, still return the latest metrics but with zero values
        return {"sr_loss": 0.0, "rp_loss": 0.0}

    def compute_losses(self, features=None, observations=None, rewards_sequence=None):
        """
        Compute losses for auxiliary tasks and update models
        
        Args:
            features: Features to use for reward prediction (optional)
            observations: Observations to use for state representation (optional)
            rewards_sequence: Sequence of rewards for reward prediction (optional)
            
        Returns:
            dict: Dictionary of loss metrics
        """
        # Skip if not enough history - but this shouldn't happen with our checks in update()
        if len(self.obs_history) < self.rp_sequence_length:
            return {"sr_loss": 0.0, "rp_loss": 0.0}
            
        # For state representation, use the most recent observation if not provided
        if observations is None:
            # Ensure we have at least one observation
            observations = torch.stack([self.obs_history[-1]]).to(self.device)
            
        # For reward prediction, create sequences from history if not provided
        if rewards_sequence is None:
            # Different sampling for batch vs. stream
            if self.learning_mode == "batch":
                # For batch learning, randomly sample sequences if we have enough data
                batch_size = min(32, max(1, len(self.feature_history) - self.rp_sequence_length))
                
                # Create sequences
                feature_sequences = []
                reward_targets = []
                
                # Ensure we don't exceed available history
                max_start_idx = max(0, len(self.feature_history) - self.rp_sequence_length)
                if max_start_idx > 0:
                    # Sample random starting indices
                    indices = np.random.randint(0, max_start_idx, batch_size)
                    
                    for idx in indices:
                        # Create sequence of features
                        seq = [self.feature_history[idx + i] for i in range(self.rp_sequence_length)]
                        feature_sequences.append(torch.stack(seq))
                        
                        # Get corresponding reward (at the end of the sequence)
                        reward_targets.append(self.reward_history[idx + self.rp_sequence_length - 1])
                else:
                    # Not enough history yet, use a single sequence with the available data
                    seq = [self.feature_history[i % len(self.feature_history)] for i in range(self.rp_sequence_length)]
                    feature_sequences.append(torch.stack(seq))
                    reward_targets.append(self.reward_history[-1])
                    
                # Convert to tensors
                feature_sequences = torch.stack(feature_sequences).to(self.device)
                reward_targets = torch.tensor(reward_targets, device=self.device)
                
            else:
                # For stream learning, use the most recent sequence
                if len(self.feature_history) < self.rp_sequence_length:
                    # If not enough history, pad with repeated elements
                    seq = [self.feature_history[i % len(self.feature_history)] 
                          for i in range(self.rp_sequence_length)]
                    feature_sequences = torch.stack([torch.stack(seq)]).to(self.device)
                    reward_targets = torch.tensor([self.reward_history[-1]], device=self.device)
                else:
                    # Create sequence from the most recent experiences
                    feature_sequences = torch.stack([
                        torch.stack([self.feature_history[-i-1] for i in range(self.rp_sequence_length)][::-1])
                    ]).to(self.device)
                    
                    reward_targets = torch.tensor([self.reward_history[-1]], device=self.device)
        else:
            # Use provided sequences
            feature_sequences = features
            reward_targets = rewards_sequence
            
        # Initialize loss values
        sr_loss_value = 0.0
        rp_loss_value = 0.0
            
        # Compute SR loss
        try:
            with torch.amp.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_amp):
                sr_loss = self.sr_task.get_loss(observations)
                sr_loss = sr_loss * self.sr_weight
                sr_loss_value = sr_loss.item() / self.sr_weight
                
            # Update SR model
            self.sr_optimizer.zero_grad()
            sr_loss.backward()
            self.sr_optimizer.step()
        except Exception as e:
            if self.debug:
                print(f"[AUX DEBUG] Error computing SR loss: {e}")
            
        # Compute RP loss
        try:
            with torch.amp.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_amp):
                rp_loss = self.rp_task.get_loss(feature_sequences, reward_targets)
                rp_loss = rp_loss * self.rp_weight
                rp_loss_value = rp_loss.item() / self.rp_weight
                
            # Update RP model
            self.rp_optimizer.zero_grad()
            rp_loss.backward()
            self.rp_optimizer.step()
        except Exception as e:
            if self.debug:
                print(f"[AUX DEBUG] Error computing RP loss: {e}")
            
        return {
            "sr_loss": sr_loss_value,
            "rp_loss": rp_loss_value
        }

    def reset(self):
        """Reset history buffers when episodes end"""
        if self.learning_mode == "stream":
            # For stream learning, clear the buffers completely
            self.obs_history.clear()
            self.feature_history.clear()
            self.reward_history.clear()
        # For batch learning, we retain the history across episodes
        
    def set_learning_mode(self, mode):
        """
        Set the learning mode
        
        Args:
            mode: Either "batch" (for PPO) or "stream" (for StreamAC)
        """
        if mode not in ["batch", "stream"]:
            raise ValueError(f"Unknown learning mode: {mode}")
            
        prev_mode = self.learning_mode
        self.learning_mode = mode
        
        # If switching modes, adjust buffer sizes
        if prev_mode != mode:
            if mode == "batch":
                # Switching to batch mode, increase buffer size
                temp_obs = list(self.obs_history)
                temp_features = list(self.feature_history)
                temp_rewards = list(self.reward_history)
                
                self.obs_history = deque(maxlen=10000)
                self.feature_history = deque(maxlen=10000)
                self.reward_history = deque(maxlen=10000)
                
                # Restore data
                self.obs_history.extend(temp_obs)
                self.feature_history.extend(temp_features)
                self.reward_history.extend(temp_rewards)
            else:
                # Switching to stream mode, decrease buffer size
                temp_obs = list(self.obs_history)[-self.rp_sequence_length*2:] if self.obs_history else []
                temp_features = list(self.feature_history)[-self.rp_sequence_length*2:] if self.feature_history else []
                temp_rewards = list(self.reward_history)[-self.rp_sequence_length*2:] if self.reward_history else []
                
                self.obs_history = deque(maxlen=self.rp_sequence_length*2)
                self.feature_history = deque(maxlen=self.rp_sequence_length*2)
                self.reward_history = deque(maxlen=self.rp_sequence_length*2)
                
                # Restore data
                self.obs_history.extend(temp_obs)
                self.feature_history.extend(temp_features)
                self.reward_history.extend(temp_rewards)
                
    def get_state_dict(self):
        """Get state dict for saving models"""
        return {
            'sr_task': self.sr_task.state_dict(),
            'rp_task': self.rp_task.state_dict(),
            'sr_optimizer': self.sr_optimizer.state_dict(),
            'rp_optimizer': self.rp_optimizer.state_dict()
        }
        
    def load_state_dict(self, state_dict):
        """Load state dict for loading models"""
        self.sr_task.load_state_dict(state_dict['sr_task'])
        self.rp_task.load_state_dict(state_dict['rp_task'])
        self.sr_optimizer.load_state_dict(state_dict['sr_optimizer'])
        self.rp_optimizer.load_state_dict(state_dict['rp_optimizer'])
