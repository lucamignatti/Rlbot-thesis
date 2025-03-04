import torch.nn as nn
import torch.nn.functional as F
import torch

class StateRepresentationTask(nn.Module):
    """
    State Representation (SR) auxiliary task.
    Tries to reconstruct the observation using an autoencoder.
    """
    def __init__(self, input_size, hidden_dim=128, latent_dim=64, device='cuda'):
        super(StateRepresentationTask, self).__init__()
        self.device = device

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size)
        )

    def forward(self, x):
        # Encode the input
        encoded = self.encoder(x)
        # Decode back to input space
        decoded = self.decoder(encoded)
        return decoded

    def compute_loss(self, x):
        # Reconstruct the input
        x_hat = self.forward(x)
        # Compute reconstruction loss using smooth L1 loss
        loss = F.smooth_l1_loss(x_hat, x)
        return loss


class RewardPredictionTask(nn.Module):
    """
    Reward Prediction (RP) auxiliary task.
    Tries to predict immediate rewards based on a sequence of observations.
    """
    def __init__(self, input_size, hidden_dim=128, sequence_length=20, device='cuda'):
        super(RewardPredictionTask, self).__init__()
        self.device = device
        self.sequence_length = sequence_length

        # LSTM to process sequence of states
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Output layer - classify rewards into positive, negative, or near-zero
        self.output_layer = nn.Linear(hidden_dim, 3)

    def forward(self, state_sequence):
        # Process the sequence through LSTM
        lstm_out, _ = self.lstm(state_sequence)
        # Take the last output for prediction
        last_output = lstm_out[:, -1, :]
        # Predict reward class
        logits = self.output_layer(last_output)
        return logits

    def compute_loss(self, state_sequences, reward_classes):
        # Predict reward classes
        logits = self.forward(state_sequences)
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, reward_classes)
        return loss
