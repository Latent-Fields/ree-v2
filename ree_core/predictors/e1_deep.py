"""
E1 Deep Predictor Implementation

E1 predicts longer-horizon latent trajectories and context, operating at
slower timescales than E2. It maintains and updates the deep world model
that E2's fast predictions are conditioned on.

Key responsibilities:
- Long-horizon prediction and context maintenance
- World model learning from accumulated experience
- Providing contextual priors to E2
- Supporting offline integration (sleep-like consolidation)

E1 operates primarily on the theta (θ) and delta (δ) depth levels
of the latent stack.

Episode-boundary semantics (V2 note):
E1 maintains self._hidden_state across episode steps. This is reset at
episode start via reset_hidden_state(). The offline prediction-loss
computation (compute_prediction_loss in REEAgent) uses a save/restore
pattern so that training replays do not disturb the inference hidden state.
Calling reset_hidden_state() clears the LSTM state to None; the next call
to predict_long_horizon re-initialises it from zeros. Do not call
reset_hidden_state() mid-episode outside of episode boundaries.
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import E1Config
from ree_core.latent.stack import LatentState


class ContextMemory(nn.Module):
    """
    Context memory for E1's long-horizon predictions.

    Maintains a compressed representation of past experience
    that provides context for predictions.
    """

    def __init__(
        self,
        latent_dim: int,
        memory_dim: int = 128,
        num_slots: int = 16
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_dim = memory_dim
        self.num_slots = num_slots

        # Memory slots
        self.memory = nn.Parameter(torch.randn(num_slots, memory_dim) * 0.01)

        # Query, key, value projections for attention
        self.query_proj = nn.Linear(latent_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)

        # Output projection
        self.output_proj = nn.Linear(memory_dim, latent_dim)

        # Write gate for memory updates
        self.write_gate = nn.Sequential(
            nn.Linear(latent_dim, memory_dim),
            nn.Sigmoid()
        )

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Read from context memory using attention.

        Args:
            query: Query vector [batch, latent_dim]

        Returns:
            Retrieved context [batch, latent_dim]
        """
        batch_size = query.shape[0]

        # Expand memory for batch
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute attention
        q = self.query_proj(query).unsqueeze(1)  # [batch, 1, memory_dim]
        k = self.key_proj(memory)                  # [batch, num_slots, memory_dim]
        v = self.value_proj(memory)                # [batch, num_slots, memory_dim]

        # Scaled dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.memory_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)

        # Read from memory
        context = torch.bmm(weights, v).squeeze(1)  # [batch, memory_dim]

        return self.output_proj(context)

    def write(self, state: torch.Tensor) -> None:
        """
        Write to context memory.

        Uses a gated write to selectively update memory slots.

        Args:
            state: State to write [batch, latent_dim]
        """
        # Compute write signal
        write_signal = self.write_gate(state)  # [batch, memory_dim]

        # Update memory with running average (detached to avoid backprop issues)
        with torch.no_grad():
            # Find slot with lowest activation for the write
            query = self.query_proj(state)
            scores = torch.mm(query, self.memory.t())  # [batch, num_slots]
            min_idx = scores.mean(0).argmin()

            # Soft update
            self.memory.data[min_idx] = 0.9 * self.memory.data[min_idx] + 0.1 * write_signal.mean(0)


class E1DeepPredictor(nn.Module):
    """
    E1 Deep Predictor for long-horizon world modelling.

    E1 operates at slower timescales than E2, maintaining:
    - Long-horizon predictions of latent trajectories
    - Contextual priors that condition E2's fast predictions
    - A deep world model learned from experience

    Architecture components:
    - Context memory for long-term dependencies
    - Transition model for long-horizon prediction
    - Prior generator for conditioning E2
    """

    def __init__(self, config: Optional[E1Config] = None):
        super().__init__()
        self.config = config or E1Config()

        # Context memory
        self.context_memory = ContextMemory(
            latent_dim=self.config.latent_dim,
            memory_dim=self.config.hidden_dim,
            num_slots=16
        )

        # Deep transition model for long-horizon prediction
        # Uses LSTM for sequence modelling
        self.transition_rnn = nn.LSTM(
            input_size=self.config.latent_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=0.1 if self.config.num_layers > 1 else 0
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        )

        # Prior generator: produces priors that condition E2
        self.prior_generator = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        )

        # Hidden state for temporal continuity across episode steps.
        # Reset at episode boundaries via reset_hidden_state().
        # See module docstring for episode-boundary semantics.
        self._hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden_state(self) -> None:
        """Reset the hidden state for a new episode."""
        self._hidden_state = None

    def predict_long_horizon(
        self,
        current_state: torch.Tensor,
        horizon: Optional[int] = None
    ) -> torch.Tensor:
        """
        Predict latent states over a long horizon.

        Args:
            current_state: Current latent state [batch, latent_dim]
            horizon: Number of steps to predict

        Returns:
            Predicted states [batch, horizon, latent_dim]
        """
        horizon = horizon or self.config.prediction_horizon
        batch_size = current_state.shape[0]
        device = current_state.device

        # Retrieve context from memory
        context = self.context_memory.read(current_state)

        # Initial input combines state and context
        combined = torch.cat([current_state, context], dim=-1)
        prior = self.prior_generator(combined)

        # Roll out predictions
        predictions = []
        input_state = prior.unsqueeze(1)  # [batch, 1, latent_dim]

        # Initialize hidden state if needed or if batch size changed
        if self._hidden_state is None or self._hidden_state[0].shape[1] != batch_size:
            h0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)
            c0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)
            self._hidden_state = (h0, c0)

        # Unroll predictions
        hidden = self._hidden_state
        for t in range(horizon):
            output, hidden = self.transition_rnn(input_state, hidden)
            predicted = self.output_proj(output.squeeze(1))
            predictions.append(predicted)
            input_state = predicted.unsqueeze(1)

        # Update hidden state
        self._hidden_state = (hidden[0].detach(), hidden[1].detach())

        return torch.stack(predictions, dim=1)

    def generate_prior(
        self,
        current_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate a prior for E2 conditioning.

        The prior encodes E1's long-horizon expectations
        that should inform E2's short-horizon predictions.

        Args:
            current_state: Current latent state [batch, latent_dim]

        Returns:
            Prior vector [batch, latent_dim]
        """
        # Read context from memory
        context = self.context_memory.read(current_state)

        # Generate prior
        combined = torch.cat([current_state, context], dim=-1)
        prior = self.prior_generator(combined)

        return prior

    def update_from_observation(
        self,
        observation_state: torch.Tensor,
        prediction_error: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Update E1 based on observed outcomes.

        This is called after actions are taken and outcomes observed.
        Systematic prediction errors update the deep world model.

        Args:
            observation_state: Observed latent state
            prediction_error: Error from E2 predictions

        Returns:
            Dictionary of update metrics
        """
        # Write to context memory
        self.context_memory.write(observation_state)

        # Compute update magnitude
        error_magnitude = prediction_error.pow(2).mean()

        return {
            "e1_error_magnitude": error_magnitude,
            "context_updated": torch.tensor(1.0)
        }

    def integrate_experience(
        self,
        experience_buffer: List[torch.Tensor],
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Offline integration of experience (sleep-like consolidation).

        Replays past experience to improve the world model
        without interacting with the environment.

        Args:
            experience_buffer: List of past states to replay
            num_iterations: Number of integration iterations

        Returns:
            Dictionary of integration metrics
        """
        if len(experience_buffer) < 2:
            return {"integration_loss": 0.0}

        total_loss = 0.0

        for _ in range(num_iterations):
            # Reset hidden state for each iteration to handle varying batch sizes
            self.reset_hidden_state()

            # Sample a contiguous sequence
            start_idx = torch.randint(0, len(experience_buffer) - 1, (1,)).item()
            end_idx = min(start_idx + self.config.prediction_horizon, len(experience_buffer))

            # Get sequence
            sequence = torch.stack(experience_buffer[start_idx:end_idx])
            if sequence.dim() == 2:
                sequence = sequence.unsqueeze(0)  # Add batch dim

            # Predict from first state
            initial_state = sequence[:, 0, :]
            predictions = self.predict_long_horizon(initial_state, horizon=sequence.shape[1] - 1)

            # Compute prediction loss
            targets = sequence[:, 1:, :]
            loss = F.mse_loss(predictions[:, :targets.shape[1], :], targets)
            total_loss += loss.item()

        return {"integration_loss": total_loss / num_iterations}

    def forward(
        self,
        current_state: torch.Tensor,
        horizon: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict long horizon and generate prior for E2.

        Args:
            current_state: Current latent state

        Returns:
            predictions: Long-horizon predictions [batch, horizon, latent_dim]
            prior: Prior for E2 conditioning [batch, latent_dim]
        """
        predictions = self.predict_long_horizon(current_state, horizon)
        prior = self.generate_prior(current_state)
        return predictions, prior
