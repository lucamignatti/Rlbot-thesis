import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StateRepresentationTask(nn.Module):
    """
    Autoencoder for State Representation (SR) auxiliary task.
    Compresses the observation to a low-dimensional space and reconstructs it.
    """
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, device="cpu"):
        super(StateRepresentationTask, self).__init__()
        self.device = device

        # Encoder: compress input to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, latent_dim)
        )

        # Decoder: reconstruct from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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
    Manages the auxiliary tasks and their integration with the main PPO algorithm.
    """
    def __init__(self, actor, obs_dim, sr_weight=1.0, rp_weight=1.0,
                 sr_hidden_dim=128, sr_latent_dim=32,
                 rp_hidden_dim=64, rp_sequence_length=20, device="cpu"):
        self.device = device
        self.sr_weight = sr_weight
        self.rp_weight = rp_weight
        self.rp_sequence_length = rp_sequence_length
        self.actor = actor

        # These heads attach to the actor's representation
        self.sr_head = nn.Sequential(
            nn.Linear(actor.hidden_dim, sr_hidden_dim),
            nn.ReLU(),
            nn.Linear(sr_hidden_dim, obs_dim)
        ).to(device)

        self.rp_head = nn.Sequential(
            nn.LSTM(actor.hidden_dim, rp_hidden_dim, batch_first=True),
            nn.Linear(rp_hidden_dim, 3)
        ).to(device)

        # Initialize observation history for RP task
        self.obs_history = []
        self.rewards_history = []

        # Optimizers for auxiliary heads
        self.sr_optimizer = torch.optim.Adam(self.sr_head.parameters(), lr=3e-4)
        self.rp_optimizer = torch.optim.Adam(self.rp_head.parameters(), lr=3e-4)

    def update(self, observations, rewards):
        """
        Store observations and rewards for sequence-based tasks.
        This method only updates the history buffers, not the networks.

        Args:
            observations: Current batch of observations
            rewards: Current batch of rewards

        Returns:
            dict: Dictionary with current SR and RP loss values (for logging)
        """
        # Convert inputs to tensors if they aren't already
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Store observations and rewards for RP task
        self.obs_history.append(observations)
        self.rewards_history.append(rewards)

        # Keep only the last sequence_length observations/rewards
        if len(self.obs_history) > self.rp_sequence_length:
            self.obs_history.pop(0)
            self.rewards_history.pop(0)

        # Get features from actor network
        with torch.no_grad():
            _, features = self.actor(observations, return_features=True)

        # Calculate auxiliary losses (just for reporting, not for training)
        sr_loss, rp_loss = self.compute_losses(
            features, observations,
            torch.stack(self.rewards_history, dim=1) if len(self.obs_history) == self.rp_sequence_length else None
        )

        return {
            'sr_loss': sr_loss.item(),
            'rp_loss': rp_loss.item() if isinstance(rp_loss, torch.Tensor) and rp_loss.numel() > 0 else 0.0
        }

    def compute_losses(self, features, observations, rewards_sequence=None):
        """
        Compute losses for both SR and RP auxiliary tasks

        Args:
            features: Features from actor network's hidden layers
            observations: Original observations for SR reconstruction
            rewards_sequence: Sequence of rewards for RP prediction (or None if not enough history)

        Returns:
            sr_loss: Loss for the State Representation task
            rp_loss: Loss for the Reward Prediction task (or zero tensor if no sequence)
        """
        # SR task - reconstruct observations from features
        sr_reconstruction = self.sr_head(features)
        sr_loss = self.sr_weight * F.smooth_l1_loss(sr_reconstruction, observations)

        # RP task
        rp_loss = torch.tensor(0.0, device=self.device)
        if rewards_sequence is not None:
            # Create features sequence by passing observations through the actor
            with torch.no_grad():
                # For sequence handling, we need to ensure proper dimensions:
                # [batch_size, seq_length, feature_dim]
                seq_features = []
                for t in range(rewards_sequence.size(1)):
                    # Get features for each timestep
                    _, feat = self.actor(rewards_sequence[:, t], return_features=True)
                    seq_features.append(feat)

                # Stack along sequence dimension
                seq_features = torch.stack(seq_features, dim=1)

            # Pass through RP head to get class logits
            rp_logits = self.rp_head(seq_features)

            # Convert rewards to classification labels (last rewards in sequence)
            last_rewards = rewards_sequence[:, -1]  # Get the latest reward in each sequence
            labels = torch.zeros_like(last_rewards, dtype=torch.long, device=self.device)
            labels[last_rewards > self.pos_threshold] = 2  # Positive reward
            labels[last_rewards < self.neg_threshold] = 0  # Negative reward
            labels[(last_rewards >= self.neg_threshold) & (last_rewards <= self.pos_threshold)] = 1  # Near-zero reward

            # Calculate cross entropy loss
            rp_loss = self.rp_weight * F.cross_entropy(rp_logits, labels)

        return sr_loss, rp_loss

    def reset(self):
        """Reset observation and reward history."""
        self.obs_history = []
        self.rewards_history = []
