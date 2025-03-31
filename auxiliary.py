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
    def __init__(self, input_dim, hidden_dim=64, sequence_length=20, device="cpu", debug=False):
        super(RewardPredictionTask, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.debug = debug
        self.input_dim = input_dim  # Store the expected input dimension

        # Add projection layer to handle varying input dimensions
        # The projection will be created or recreated in the forward pass if needed
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layer to process observation sequences
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Output layer for 3-class classification (negative, near-zero, positive reward)
        self.output_layer = nn.Linear(hidden_dim, 3)

        # Lower thresholds for classifying rewards - adjusted for typical reward scales
        self.pos_threshold = 0.001
        self.neg_threshold = -0.001
        
        # Debug counters for reward distribution
        self.debug_reward_counts = {"negative": 0, "zero": 0, "positive": 0}
        self.debug_total_rewards = 0
        self.debug_reward_history = []
        self.debug_threshold_adjust_counter = 0

        self.to(self.device)

    def forward(self, x_seq):
        # x_seq shape: [batch_size, sequence_length, input_dim]
        batch_size, seq_len, actual_dim = x_seq.shape
        
        # Check if input dimensions match the expected dimensions
        # If not, recreate the projection layer with the correct dimensions
        if actual_dim != self.input_dim:
            if self.debug:
                print(f"[RP DEBUG] Input dimension mismatch. Expected: {self.input_dim}, Got: {actual_dim}. Recreating projection layer.")
            
            # Create a new projection layer with correct input dimensions
            self.projection = nn.Linear(actual_dim, self.hidden_dim).to(self.device)
            # Update stored input dimension
            self.input_dim = actual_dim
        
        # Reshape to process all sequence elements at once
        x_reshaped = x_seq.reshape(batch_size * seq_len, -1)
        
        # Now apply the projection
        projected = self.projection(x_reshaped)
        
        # Reshape back to sequence format
        projected = projected.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Now pass through LSTM
        lstm_out, _ = self.lstm(projected)

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

        # Calculate class distributions for debugging
        self.debug_total_rewards += len(rewards)
        neg_count = (labels == 0).sum().item()
        zero_count = (labels == 1).sum().item() 
        pos_count = (labels == 2).sum().item()
        
        self.debug_reward_counts["negative"] += neg_count
        self.debug_reward_counts["zero"] += zero_count
        self.debug_reward_counts["positive"] += pos_count
        
        # Store some raw reward values for debugging
        for r in rewards.cpu().tolist()[:10]:  # Limit to first 10 to avoid memory issues
            self.debug_reward_history.append(r)
            if len(self.debug_reward_history) > 100:
                self.debug_reward_history.pop(0)
        
        # Print detailed debug info periodically only if debug mode is enabled
        if self.debug and self.debug_total_rewards % 500 == 0:
            total = self.debug_reward_counts["negative"] + self.debug_reward_counts["zero"] + self.debug_reward_counts["positive"]
            neg_pct = 100 * self.debug_reward_counts["negative"] / max(1, total)
            zero_pct = 100 * self.debug_reward_counts["zero"] / max(1, total)
            pos_pct = 100 * self.debug_reward_counts["positive"] / max(1, total)
            
            print(f"[RP DEBUG] Reward class distribution after {self.debug_total_rewards} rewards:")
            print(f"[RP DEBUG]   Negative: {self.debug_reward_counts['negative']} ({neg_pct:.1f}%)")
            print(f"[RP DEBUG]   Zero: {self.debug_reward_counts['zero']} ({zero_pct:.1f}%)")
            print(f"[RP DEBUG]   Positive: {self.debug_reward_counts['positive']} ({pos_pct:.1f}%)")
            print(f"[RP DEBUG] Current batch - Neg: {neg_count}, Zero: {zero_count}, Pos: {pos_count}")
            print(f"[RP DEBUG] Current batch rewards sample: {rewards[:5].cpu().tolist()}")
            print(f"[RP DEBUG] Current thresholds - Pos: {self.pos_threshold}, Neg: {self.neg_threshold}")
            
            # Show statistics on recent rewards
            if self.debug_reward_history:
                min_reward = min(self.debug_reward_history)
                max_reward = max(self.debug_reward_history)
                avg_reward = sum(self.debug_reward_history) / len(self.debug_reward_history)
                print(f"[RP DEBUG] Recent rewards stats - Min: {min_reward:.6f}, Max: {max_reward:.6f}, Avg: {avg_reward:.6f}")
                
            # Check if class imbalance is severe (>95% in one class)
            if neg_pct > 95 or zero_pct > 95 or pos_pct > 95:
                print(f"[RP DEBUG] WARNING: Severe class imbalance detected in reward classes!")
                
                # Suggest adjusting thresholds if needed
                if zero_pct > 95:
                    print(f"[RP DEBUG] Most rewards are classified as 'zero' - consider decreasing thresholds.")
                    suggested_pos = self.pos_threshold / 2
                    suggested_neg = self.neg_threshold / 2
                    print(f"[RP DEBUG] Suggested new thresholds - Pos: {suggested_pos:.6f}, Neg: {suggested_neg:.6f}")
                elif pos_pct > 95:
                    print(f"[RP DEBUG] Most rewards are classified as 'positive' - consider increasing positive threshold.")
                    suggested_pos = self.pos_threshold * 2
                    print(f"[RP DEBUG] Suggested new threshold - Pos: {suggested_pos:.6f}")
                elif neg_pct > 95:
                    print(f"[RP DEBUG] Most rewards are classified as 'negative' - consider decreasing negative threshold.")
                    suggested_neg = self.neg_threshold * 2
                    print(f"[RP DEBUG] Suggested new threshold - Neg: {suggested_neg:.6f}")

        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Debug prediction quality only if debug mode is enabled
        if self.debug:
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean().item()
                if self.debug_total_rewards % 500 == 0:
                    print(f"[RP DEBUG] Prediction accuracy: {accuracy:.4f}")
                    
                    # Check if all predictions are the same class
                    pred_classes = predictions.unique()
                    if len(pred_classes) == 1:
                        print(f"[RP DEBUG] WARNING: All predictions are class {pred_classes.item()}!")
                    
                    # Get confidence scores
                    softmax_probs = F.softmax(logits, dim=1)
                    avg_confidence = softmax_probs.max(dim=1)[0].mean().item()
                    print(f"[RP DEBUG] Average prediction confidence: {avg_confidence:.4f}")
                    print(f"[RP DEBUG] Logits sample: {logits[0].cpu().tolist()}")
                
        return loss

def _adjust_reward_thresholds(self, rewards):
    """Dynamically adjust reward thresholds to better balance classes"""
    # Only adjust if we have enough data
    if len(self.debug_reward_history) < 50:
        return
        
    rewards_tensor = torch.tensor(self.debug_reward_history)
    
    # Calculate percentiles
    sorted_rewards, _ = torch.sort(rewards_tensor)
    num_rewards = len(sorted_rewards)
    
    # Set thresholds to the 33rd and 67th percentiles for a balanced 3-way split
    idx_neg = max(0, int(num_rewards * 0.33) - 1)
    idx_pos = min(num_rewards - 1, int(num_rewards * 0.67))
    
    new_neg_threshold = sorted_rewards[idx_neg].item()
    new_pos_threshold = sorted_rewards[idx_pos].item()
    
    # Ensure minimum separation between thresholds
    min_separation = 0.005
    if new_pos_threshold - new_neg_threshold < min_separation:
        mid_point = (new_pos_threshold + new_neg_threshold) / 2
        new_neg_threshold = mid_point - min_separation/2
        new_pos_threshold = mid_point + min_separation/2
    
    # Update thresholds with some momentum to avoid rapid changes
    momentum = 0.9
    self.neg_threshold = momentum * self.neg_threshold + (1-momentum) * new_neg_threshold
    self.pos_threshold = momentum * self.pos_threshold + (1-momentum) * new_pos_threshold
    
    print(f"[RP DEBUG] Adjusted thresholds - Neg: {self.neg_threshold:.6f}, Pos: {self.pos_threshold:.6f}")


class AuxiliaryTaskManager:
    """
    Manages auxiliary tasks for both batch learning (PPO) and stream learning (StreamAC).
    Adapts to the learning algorithm being used.
    """
    def __init__(self, actor, obs_dim, sr_weight=1.0, rp_weight=1.0,
                 sr_hidden_dim=128, sr_latent_dim=32,
                 rp_hidden_dim=64, rp_sequence_length=5, 
                 device="cpu", use_amp=False, update_frequency=8, # Keep update_frequency for potential periodic calls
                 learning_mode="batch", debug=False, batch_size=64): # Add batch_size
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
        self.debug = debug
        self.use_amp = use_amp # Note: AMP might be complex with manual loss computation
        self.learning_mode = learning_mode
        self.update_frequency = update_frequency # How often compute_losses should be called externally
        self.update_counter = 0 # Counter for external calls
        self.batch_size = batch_size # Batch size for sampling from history

        # Ensure hidden_dim exists
        self.hidden_dim = getattr(actor, 'hidden_dim', 1536)

        if self.debug:
            print(f"[AUX INIT] Initializing AuxiliaryTaskManager with sr_weight={sr_weight}, rp_weight={rp_weight}")
            print(f"[AUX INIT] Device: {device}, Learning mode: {learning_mode}, Update frequency: {update_frequency}")
            print(f"[AUX INIT] Observation dimension: {obs_dim}, Hidden dimension: {self.hidden_dim}")
            print(f"[AUX INIT] RP Sequence Length: {rp_sequence_length}, Batch Size: {batch_size}")

        # State representation task
        self.sr_task = StateRepresentationTask(
            input_dim=obs_dim,
            hidden_dim=sr_hidden_dim,
            latent_dim=sr_latent_dim,
            device=device
        )
        # Reward prediction task
        self.rp_task = RewardPredictionTask(
            input_dim=self.hidden_dim, # RP task uses features
            hidden_dim=rp_hidden_dim,
            sequence_length=rp_sequence_length,
            device=device,
            debug=debug
        )
        # Initialize optimizers
        self.sr_optimizer = torch.optim.Adam(self.sr_task.parameters(), lr=3e-4)
        self.rp_optimizer = torch.optim.Adam(self.rp_task.parameters(), lr=3e-4)

        # History buffers - Use deques for efficient appending/popping
        # Size needs to be large enough for batch sampling + sequence length
        history_maxlen = 10000 if learning_mode == "batch" else 1000  # Much larger buffer for batch mode, reasonable for stream
        self.obs_history = deque(maxlen=history_maxlen)
        self.feature_history = deque(maxlen=history_maxlen)
        self.reward_history = deque(maxlen=history_maxlen)

        # Counters and last loss values
        self.update_count = 0
        self.last_sr_loss = 0.0
        self.last_rp_loss = 0.0
        self.history_size = 0 # Track current number of items in history

    @property
    def history_filled(self):
        """Property to track how many items are in the history (for test compatibility)"""
        return self.history_size
        
    def update(self, observations, rewards, features=None):
        """
        Add new experiences to the history buffers. Does NOT compute losses.
        Loss computation is handled by compute_losses().

        Args:
            observations: New observations (Tensor or np.array) [batch_size, obs_dim] or [obs_dim]
            rewards: New rewards (Tensor or np.array) [batch_size] or scalar
            features: Features extracted from observations (Tensor or np.array) [batch_size, feature_dim] or [feature_dim]
        """
        # Validate reward dimensions - should be scalar or 1D tensor/array
        if isinstance(rewards, torch.Tensor) and rewards.dim() > 1:
            raise ValueError(f"Rewards must be scalar or 1D tensor, got shape {rewards.shape}")
        elif isinstance(rewards, np.ndarray) and rewards.ndim > 1:
            raise ValueError(f"Rewards must be scalar or 1D array, got shape {rewards.shape}")

        # Ensure inputs are torch tensors on CPU (history stored on CPU)
        if isinstance(observations, torch.Tensor):
            observations = observations.detach().cpu()
        else:
            observations = torch.tensor(observations, dtype=torch.float32)

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu()
        else:
            rewards = torch.tensor(rewards, dtype=torch.float32)

        # Extract features if not provided
        if features is None:
            with torch.no_grad():
                obs_tensor = observations.to(self.device) # Move to device for model
                if hasattr(self.actor, 'get_features'):
                    features = self.actor.get_features(obs_tensor).detach().cpu()
                elif hasattr(self.actor, 'extract_features'):
                     features = self.actor.extract_features(obs_tensor).detach().cpu()
                else:
                    # Fallback: Use observations if feature extraction fails
                    features = observations
        elif isinstance(features, torch.Tensor):
            features = features.detach().cpu()
        else:
            features = torch.tensor(features, dtype=torch.float32)

        # Handle single vs batch inputs
        if observations.dim() == 1: # Single observation
            observations = observations.unsqueeze(0)
            rewards = rewards.unsqueeze(0) if rewards.dim() == 0 else rewards
            features = features.unsqueeze(0)

        # Add experiences to history
        num_added = 0
        for i in range(observations.shape[0]):
            self.obs_history.append(observations[i])
            self.feature_history.append(features[i])
            self.reward_history.append(rewards[i])
            num_added += 1
        
        self.history_size = len(self.obs_history) # Update history size
        
        # Update counter for test compatibility
        self.update_counter += 1
        if self.update_counter >= self.update_frequency:
            self.update_counter = 0  # Reset when we reach the update frequency

        # Return status (e.g., number of items added) - No loss computation here
        return {"items_added": num_added}


    def compute_losses(self):
        """
        Compute losses for auxiliary tasks by sampling from history buffers.
        This should be called periodically by the Trainer.

        Returns:
            dict: Dictionary containing SR and RP losses, or empty if not enough data.
        """
        sr_loss = 0.0
        rp_loss = 0.0
        
        # --- State Representation Task ---
        if self.sr_weight > 0 and self.history_size >= self.batch_size:
            # Sample a batch of observations from history
            indices = np.random.choice(self.history_size, self.batch_size, replace=False)
            obs_batch = torch.stack([self.obs_history[i] for i in indices]).to(self.device)

            self.sr_task.train()
            self.sr_optimizer.zero_grad()
            
            # Compute SR loss
            sr_loss_val = self.sr_task.get_loss(obs_batch) * self.sr_weight
            
            if not torch.isnan(sr_loss_val) and not torch.isinf(sr_loss_val):
                sr_loss_val.backward()
                self.sr_optimizer.step()
                sr_loss = sr_loss_val.item()
                self.last_sr_loss = sr_loss
            else:
                if self.debug:
                    print(f"[AUX DEBUG] Invalid SR loss detected: {sr_loss_val.item()}")
                self.last_sr_loss = 0.0
        else:
             if self.debug and self.sr_weight > 0:
                 print(f"[AUX DEBUG] SR Task skipped: Not enough history ({self.history_size}/{self.batch_size})")
             self.last_sr_loss = 0.0 # Ensure last loss is reset if skipped

        # --- Reward Prediction Task ---
        # Need at least rp_sequence_length steps in history to form one sequence
        # Need enough history to sample a batch of sequences
        min_rp_history = self.rp_sequence_length 
        if self.rp_weight > 0 and self.history_size >= min_rp_history + self.batch_size -1 :
            
            # Sample valid start indices for sequences
            # A start index `i` is valid if `i + rp_sequence_length <= history_size`
            max_start_index = self.history_size - self.rp_sequence_length
            if max_start_index < self.batch_size -1:
                 if self.debug:
                     print(f"[AUX DEBUG] RP Task skipped: Not enough valid start indices ({max_start_index+1}/{self.batch_size})")
                 self.last_rp_loss = 0.0
                 return {"sr_loss": self.last_sr_loss, "rp_loss": self.last_rp_loss}

            start_indices = np.random.choice(max_start_index + 1, self.batch_size, replace=False)

            # Prepare batches for LSTM
            feature_seq_batch = []
            reward_target_batch = []

            for start_idx in start_indices:
                # Extract feature sequence
                feature_seq = [self.feature_history[i] for i in range(start_idx, start_idx + self.rp_sequence_length)]
                feature_seq_batch.append(torch.stack(feature_seq))
                
                # Extract target reward (reward at the end of the sequence)
                target_reward_idx = start_idx + self.rp_sequence_length - 1
                reward_target_batch.append(self.reward_history[target_reward_idx])

            # Stack batches and move to device
            features_seq_tensor = torch.stack(feature_seq_batch).to(self.device)
            reward_targets_tensor = torch.stack(reward_target_batch).to(self.device)

            # Compute RP loss
            self.rp_task.train()
            self.rp_optimizer.zero_grad()
            
            rp_loss_val = self.rp_task.get_loss(features_seq_tensor, reward_targets_tensor) * self.rp_weight
            
            if not torch.isnan(rp_loss_val) and not torch.isinf(rp_loss_val):
                rp_loss_val.backward()
                self.rp_optimizer.step()
                rp_loss = rp_loss_val.item()
                self.last_rp_loss = rp_loss
            else:
                if self.debug:
                    print(f"[AUX DEBUG] Invalid RP loss detected: {rp_loss_val.item()}")
                self.last_rp_loss = 0.0
        else:
            if self.debug and self.rp_weight > 0:
                 print(f"[AUX DEBUG] RP Task skipped: Not enough history ({self.history_size}/{min_rp_history + self.batch_size -1})")
            self.last_rp_loss = 0.0 # Ensure last loss is reset if skipped

        # Log detailed results in debug mode
        if self.debug and (sr_loss > 0 or rp_loss > 0):
            print(f"[AUX DEBUG] Computed Losses - SR: {sr_loss:.6f}, RP: {rp_loss:.6f}")
            
        self.update_count += 1 # Increment internal update counter

        return {"sr_loss": self.last_sr_loss, "rp_loss": self.last_rp_loss}

    # Ensure reset() clears history buffers correctly
    def reset(self):
        """Reset history buffers"""
        # Always clear history on reset, regardless of mode, as compute_losses uses history
        self.obs_history.clear()
        self.feature_history.clear()
        self.reward_history.clear()
        self.history_size = 0 # Reset size tracker
        
        if self.debug:
            print("[AUX RESET] Reset called - cleared history buffers")

    def reset_auxiliary_tasks(self):
        """Reset auxiliary task models (weights and optimizers)"""
        # Re-initialize SR task with the same parameters
        self.sr_task = StateRepresentationTask(
            input_dim=self.sr_task.input_dim,
            hidden_dim=self.sr_task.hidden_dim,
            latent_dim=self.sr_task.latent_dim,
            device=self.device
        )
        # Re-initialize RP task with the same parameters
        self.rp_task = RewardPredictionTask(
            input_dim=self.hidden_dim,
            hidden_dim=self.rp_task.hidden_dim,
            sequence_length=self.rp_sequence_length,
            device=self.device,
            debug=self.debug  # Pass debug flag when recreating
        )
        # Re-initialize optimizers
        self.sr_optimizer = torch.optim.Adam(self.sr_task.parameters(), lr=3e-4)
        self.rp_optimizer = torch.optim.Adam(self.rp_task.parameters(), lr=3e-4)
        
        # Reset history buffers too
        self.reset()
        
        # Reset counters
        self.update_counter = 0
        self.history_size = 0
        self.update_count = 0
        self.last_sr_loss = 0.0
        self.last_rp_loss = 0.0
        
        if self.debug:
            print("[AUX RESET] Complete reset of auxiliary task models and optimizers")
        
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
                
            if self.debug:
                print(f"[AUX MODE] Changed learning mode from {prev_mode} to {mode}")
                print(f"[AUX MODE] New history sizes - Obs: {len(self.obs_history)}, Features: {len(self.feature_history)}, Rewards: {len(self.reward_history)}")
                
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
        
        if self.debug:
            print("[AUX LOAD] Loaded auxiliary task models from state dict")
