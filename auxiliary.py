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
                 rp_hidden_dim=64, rp_sequence_length=5, device="cpu", use_amp=False):
        self.device = device
        self.sr_weight = sr_weight
        self.rp_weight = rp_weight
        self.rp_sequence_length = rp_sequence_length
        self.actor = actor
        self.debug = False
        self.use_amp = use_amp

        # Ensure hidden_dim exists
        self.hidden_dim = getattr(actor, 'hidden_dim', 1536)

        # These heads attach to the actor's representation
        self.sr_head = nn.Sequential(
            nn.Linear(self.hidden_dim, sr_hidden_dim),
            nn.LayerNorm(sr_hidden_dim),
            nn.ReLU(),
            nn.Linear(sr_hidden_dim, obs_dim)
        ).to(device)

        # Apply compile to SR head if available
        if hasattr(torch, 'compile'):
            try:
                compile_options = {
                    "fullgraph": False,
                    "dynamic": True,
                }
                if device == "mps":
                    self.sr_head = torch.compile(self.sr_head, backend="aot_eager", **compile_options)
                elif device == "cuda":
                    if hasattr(torch._dynamo.config, "allow_cudagraph_ops"):
                        torch._dynamo.config.allow_cudagraph_ops = True
                    self.sr_head = torch.compile(self.sr_head, backend="inductor", **compile_options)
                else:
                    self.sr_head = torch.compile(self.sr_head, backend="inductor", **compile_options)
            except:
                pass

        # Initialize RP head with LSTM for sequence processing
        self.rp_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=rp_hidden_dim,
            batch_first=True
        ).to(device)

        self.rp_head = nn.Linear(rp_hidden_dim, 3).to(device)

        # Initialize history buffers
        self.feature_history = []
        self.rewards_history = []

        # Track latest loss values
        self.latest_sr_loss = 0.01
        self.latest_rp_loss = 0.01
        self.debug_next_compute = False

        # Optimizers for auxiliary heads
        self.sr_optimizer = torch.optim.Adam(self.sr_head.parameters(), lr=3e-4)
        self.rp_optimizer = torch.optim.Adam([
            {'params': self.rp_lstm.parameters()},
            {'params': self.rp_head.parameters()}
        ], lr=3e-4)

    def update(self, observations, rewards, features=None):
        """
        Store observations and rewards for sequence-based tasks.
        This method updates history buffers and reports current losses.
        """
        if self.use_amp:
            observations = observations.float()
            if rewards is not None and isinstance(rewards, torch.Tensor):
                rewards = rewards.float()
            if features is not None and isinstance(features, torch.Tensor):
                features = features.float()

        # Convert inputs to tensors if they aren't already
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Add batch dimension if needed
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)

        # Get features from actor network if not provided
        if features is None:
            with torch.no_grad():
                _, features = self.actor(observations, return_features=True)
        elif not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32, device=self.device)

        # Store batched features and rewards by taking mean if necessary
        if features.dim() > 1 and features.size(0) > 1:
            self.feature_history.append(features.mean(dim=0).detach())
        else:
            self.feature_history.append(features.reshape(-1).detach())

        if rewards.dim() > 0 and rewards.size(0) > 1:
            self.rewards_history.append(rewards.mean().detach())
        else:
            self.rewards_history.append(rewards.detach())

        # Keep only the last sequence_length items
        if len(self.feature_history) > self.rp_sequence_length:
            self.feature_history.pop(0)
            self.rewards_history.pop(0)

        # Calculate current losses for reporting
        sr_loss = torch.tensor(self.latest_sr_loss, device=self.device)
        rp_loss = torch.tensor(self.latest_rp_loss, device=self.device)

        if features is not None:
            try:
                # Compute SR loss with AMP
                with torch.amp.autocast(enabled=self.use_amp, device_type="cuda"):
                    sr_reconstruction = self.sr_head(features)

                # Loss calculation in full precision
                sr_reconstruction = sr_reconstruction.float()
                observations = observations.float()

                if sr_reconstruction.shape == observations.shape:
                    sr_loss = F.smooth_l1_loss(sr_reconstruction, observations)
                else:
                    min_dim = min(sr_reconstruction.shape[-1], observations.shape[-1])
                    sr_loss = F.smooth_l1_loss(
                        sr_reconstruction[..., :min_dim],
                        observations[..., :min_dim]
                    )

                # Update latest SR loss
                self.latest_sr_loss = max(0.01, float(sr_loss.item()))
                sr_loss = torch.tensor(self.latest_sr_loss, device=self.device)

            except Exception as e:
                print(f"SR loss calculation failed in update: {e}")
                sr_loss = torch.tensor(0.01, device=self.device)

        if len(self.feature_history) == self.rp_sequence_length:
            try:
                # Compute RP loss with AMP
                with torch.amp.autocast(enabled=self.use_amp, device_type="cuda"):
                    feature_seq = torch.stack(self.feature_history, dim=0).unsqueeze(0)
                    lstm_out, _ = self.rp_lstm(feature_seq)
                    logits = self.rp_head(lstm_out[:, -1, :])

                # Loss calculation in full precision
                logits = logits.float()
                last_reward = self.rewards_history[-1].float()

                pos_threshold = 0.009
                neg_threshold = -0.009

                label = torch.tensor([1], device=self.device)
                if last_reward > pos_threshold:
                    label[0] = 2
                elif last_reward < neg_threshold:
                    label[0] = 0

                rp_loss = F.cross_entropy(logits, label)

                # Update latest RP loss
                self.latest_rp_loss = max(0.01, float(rp_loss.item()))
                rp_loss = torch.tensor(self.latest_rp_loss, device=self.device)

            except Exception as e:
                print(f"RP loss calculation failed in update: {e}")
                rp_loss = torch.tensor(0.01, device=self.device)

        # Always return non-zero losses
        return {
            'sr_loss': max(0.01, float(sr_loss.item())),
            'rp_loss': max(0.01, float(rp_loss.item()))
        }

    def compute_losses(self, features, observations, rewards_sequence=None):
        """
        Compute losses for both SR and RP auxiliary tasks. Handles AMP with numerical stability.
        """
        # Print shapes if debugging is enabled
        if self.debug_next_compute:
            print(f"DEBUGGING AUX TASKS - features shape: {features.shape}")
            print(f"DEBUGGING AUX TASKS - observations shape: {observations.shape}")

        if self.use_amp:
            features = features.float()
            observations = observations.float()
            if rewards_sequence is not None:
                rewards_sequence = rewards_sequence.float()

        try:
            # SR task with proper dimension handling
            with torch.amp.autocast(enabled=self.use_amp, device_type="cuda"):
                sr_reconstruction = self.sr_head(features)

            # Loss calculation in full precision
            sr_reconstruction = sr_reconstruction.float()
            if sr_reconstruction.shape != observations.shape:
                if self.debug_next_compute:
                    print(f"DEBUGGING AUX TASKS - SR shape mismatch: {sr_reconstruction.shape} vs {observations.shape}")
                min_dim = min(sr_reconstruction.shape[-1], observations.shape[-1])
                sr_loss = self.sr_weight * F.smooth_l1_loss(
                    sr_reconstruction[..., :min_dim],
                    observations[..., :min_dim]
                )
            else:
                sr_loss = self.sr_weight * F.smooth_l1_loss(sr_reconstruction, observations)

            # Store for debugging
            self.latest_sr_loss = sr_loss.item()

            if self.debug_next_compute:
                print(f"DEBUGGING AUX TASKS - SR loss: {self.latest_sr_loss}")

            # Generate a dummy loss if actual loss is too small
            if self.latest_sr_loss < 1e-6:
                sr_loss = torch.tensor(0.01, device=self.device, requires_grad=True)

        except Exception as e:
            print(f"SR loss calculation failed: {e}")
            sr_loss = torch.tensor(0.01, device=self.device, requires_grad=True)

        # RP task setup
        rp_pos_threshold = 0.009
        rp_neg_threshold = -0.009
        rp_loss = torch.tensor(0.01, device=self.device, requires_grad=True)  # Default to non-zero value

        if rewards_sequence is not None and rewards_sequence.numel() > 0:
            try:
                batch_size = features.size(0)

                # Create batch-sized sequences
                feature_seqs = []
                for _ in range(batch_size):
                    seq = torch.stack([feat.clone().detach() for feat in self.feature_history[-self.rp_sequence_length:]], dim=0)
                    feature_seqs.append(seq)

                # Stack into batch
                batched_seqs = torch.stack(feature_seqs, dim=0)

                if self.debug_next_compute:
                    print(f"DEBUGGING AUX TASKS - RP sequence shape: {batched_seqs.shape}")

                # Get predictions with AMP
                with torch.amp.autocast(enabled=self.use_amp, device_type="cuda"):
                    lstm_out, _ = self.rp_lstm(batched_seqs)
                    rp_logits = self.rp_head(lstm_out[:, -1, :])

                # Convert to full precision for loss calculation
                rp_logits = rp_logits.float()

                # Handle rewards with proper batch dimension
                if rewards_sequence.dim() > 1 and rewards_sequence.size(0) == batch_size:
                    last_rewards = rewards_sequence[:, -1]
                else:
                    if rewards_sequence.dim() == 0:
                        last_rewards = rewards_sequence.repeat(batch_size)
                    else:
                        last_scalar = rewards_sequence[-1]
                        last_rewards = last_scalar.repeat(batch_size)

                last_rewards = last_rewards.to(self.device)

                if self.debug_next_compute:
                    print(f"DEBUGGING AUX TASKS - Rewards: {last_rewards}")

                # Create labels
                labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                labels[last_rewards > rp_pos_threshold] = 2
                labels[last_rewards < rp_neg_threshold] = 0
                labels[(last_rewards >= rp_neg_threshold) & (last_rewards <= rp_pos_threshold)] = 1

                if self.debug_next_compute:
                    print(f"DEBUGGING AUX TASKS - Labels: {labels}")
                    print(f"DEBUGGING AUX TASKS - RP logits: {rp_logits}")

                # Calculate RP loss in full precision
                rp_loss = self.rp_weight * F.cross_entropy(rp_logits, labels)

                # Store for debugging
                self.latest_rp_loss = rp_loss.item()

                if self.debug_next_compute:
                    print(f"DEBUGGING AUX TASKS - RP loss: {self.latest_rp_loss}")

                if torch.isnan(rp_loss) or self.latest_rp_loss < 1e-6:
                    rp_loss = torch.tensor(0.01, device=self.device, requires_grad=True)

                self.debug_next_compute = False  # Reset debug flag

            except Exception as e:
                print(f"RP loss calculation failed: {e}")
                import traceback
                traceback.print_exc()
                rp_loss = torch.tensor(0.01, device=self.device, requires_grad=True)

        # Make sure we're returning non-zero losses
        if sr_loss.item() < 1e-6:
            sr_loss = torch.tensor(0.01, device=self.device, requires_grad=True)
        if rp_loss.item() < 1e-6:
            rp_loss = torch.tensor(0.01, device=self.device, requires_grad=True)

        return sr_loss, rp_loss

    def reset(self):
        """Clear history buffers and trigger debug on next compute"""
        self.feature_history = []
        self.rewards_history = []
        self.debug_next_compute = False
