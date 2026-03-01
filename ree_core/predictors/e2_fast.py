"""
E2 Fast Predictor Implementation

E2 predicts immediate observations and short-horizon state. It is a *pure*
fast transition model: f(z_t, a_t) → z_{t+1}.

V2 changes vs V1:
- generate_candidates_cem() REMOVED — CEM-style iterative refinement is
  hippocampal work and has moved to HippocampalModule (SD-001 resolution).
- generate_candidates() now only supports method="random".
- forward_counterfactual() ADDED — pure E2 query for SD-003 self-attribution
  experiments. Allows the agent to ask "what would have happened if I had
  taken a different action?"

E2's role in self-attribution (SD-003):
The transition model f(z_t, a_t) → z_{t+1} is the substrate for
counterfactual reasoning. By querying E2 with the actual action taken
and a counterfactual action, the agent can isolate its own causal
contribution to observed outcomes:

    z_actual_next   = e2.predict_next_state(z, a_actual)
    z_cf_next       = e2.forward_counterfactual(z, a_cf)
    causal_delta    = z_actual_next - z_cf_next  # agent's causal signature

This is only possible when E2 is a pure, independently-callable transition
model. The SD-001 conflation (E2 doing hippocampal work) prevented this.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import E2Config
from ree_core.latent.stack import LatentState


@dataclass
class Trajectory:
    """A candidate trajectory through latent space.

    Attributes:
        states: List of latent states along the trajectory
        actions: Action sequence that generated this trajectory
        predicted_observations: Predicted observations at each step
        harm_predictions: Predicted harm at each step [batch, horizon]
        total_length: Number of steps in the trajectory
    """
    states: List[torch.Tensor]      # List of [batch, latent_dim] tensors
    actions: torch.Tensor           # [batch, horizon, action_dim]
    predicted_observations: Optional[List[torch.Tensor]] = None
    harm_predictions: Optional[torch.Tensor] = None  # [batch, horizon]

    @property
    def total_length(self) -> int:
        return len(self.states)

    def get_final_state(self) -> torch.Tensor:
        """Get the final state of the trajectory."""
        return self.states[-1]

    def get_state_sequence(self) -> torch.Tensor:
        """Stack all states into a tensor [batch, horizon, latent_dim]."""
        return torch.stack(self.states, dim=1)


class E2FastPredictor(nn.Module):
    """
    E2 Fast Predictor — pure transition model f(z_t, a_t) → z_{t+1}.

    V2: E2 is strictly a transition model. It does NOT perform trajectory
    search or CEM refinement; that is HippocampalModule's responsibility.
    E2 provides:
      - predict_next_state(): single-step transition
      - rollout(): multi-step rollout given an action sequence
      - generate_candidates_random(): random-shooting for use by HippocampalModule
      - forward_counterfactual(): SD-003 substrate for self-attribution

    Architecture:
    - Transition model: z_{t+1} = z_t + delta(z_t, a_t)  [residual connection]
    - Observation model: o_t = g(z_t)
    - Harm model: h_t = h(z_t, a_t)
    """

    def __init__(self, config: Optional[E2Config] = None):
        super().__init__()
        self.config = config or E2Config()

        # Transition model: z_{t+1} = z_t + delta(concat(z_t, a_t))
        # Residual connection keeps predictions close to current state
        self.transition = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        )

        # Observation predictor: o_t = g(z_t)
        self.observation_predictor = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        )

        # Harm predictor: h_t = h(z_t, a_t)
        self.harm_predictor = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()  # Harm in [0, 1]
        )

        # Action encoder
        self.action_encoder = nn.Linear(self.config.action_dim, self.config.action_dim)

    def predict_next_state(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the next latent state given current state and action.

        z_{t+1} = z_t + delta(z_t, a_t)   [residual connection]

        Args:
            current_state: Current latent state [batch, latent_dim]
            action: Action to take [batch, action_dim]

        Returns:
            Predicted next latent state [batch, latent_dim]
        """
        action_embed = self.action_encoder(action)
        state_action = torch.cat([current_state, action_embed], dim=-1)
        delta = self.transition(state_action)
        next_state = current_state + delta
        return next_state

    def predict_observation(self, state: torch.Tensor) -> torch.Tensor:
        """Predict observation from latent state."""
        return self.observation_predictor(state)

    def predict_harm(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict potential harm from state-action pair.

        Args:
            state: Latent state [batch, latent_dim]
            action: Action [batch, action_dim]

        Returns:
            Predicted harm level [batch, 1] in [0, 1]
        """
        state_action = torch.cat([state, action], dim=-1)
        return self.harm_predictor(state_action)

    def forward_counterfactual(
        self,
        current_state: torch.Tensor,
        counterfactual_action: torch.Tensor
    ) -> torch.Tensor:
        """
        Counterfactual E2 query: what would have happened under a different action?

        This is the substrate for SD-003 self-attribution experiments. By
        comparing the actual transition to the counterfactual, the agent can
        isolate its own causal contribution to observed outcomes:

            z_actual_next = e2.predict_next_state(z, a_actual)
            z_cf_next     = e2.forward_counterfactual(z, a_cf)
            causal_delta  = z_actual_next - z_cf_next  # agent's causal signature

        The difference between z_actual_next and z_cf_next is attributable
        to the agent's action choice. Transitions that look the same under
        any action are environment-caused; differences are agent-caused.

        This method is callable independently — it does not modify E2's
        internal state. E2 must be a pure transition model for this to be
        meaningful (SD-001 resolution requirement).

        Args:
            current_state: Current latent state [batch, latent_dim]
            counterfactual_action: The action NOT taken [batch, action_dim]

        Returns:
            Predicted next state under the counterfactual action [batch, latent_dim]
        """
        return self.predict_next_state(current_state, counterfactual_action)

    def rollout(
        self,
        initial_state: torch.Tensor,
        action_sequence: torch.Tensor
    ) -> Trajectory:
        """
        Roll out a trajectory given an action sequence.

        Args:
            initial_state: Starting latent state [batch, latent_dim]
            action_sequence: Sequence of actions [batch, horizon, action_dim]

        Returns:
            Trajectory containing states, actions, and predictions
        """
        horizon = action_sequence.shape[1]

        states = [initial_state]
        observations = []
        harm_predictions = []

        current_state = initial_state

        for t in range(horizon):
            action = action_sequence[:, t, :]

            harm = self.predict_harm(current_state, action)
            harm_predictions.append(harm)

            obs = self.predict_observation(current_state)
            observations.append(obs)

            next_state = self.predict_next_state(current_state, action)
            states.append(next_state)
            current_state = next_state

        return Trajectory(
            states=states,
            actions=action_sequence,
            predicted_observations=observations,
            harm_predictions=torch.cat(harm_predictions, dim=-1)  # [batch, horizon]
        )

    def generate_random_actions(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate random action sequences for exploration."""
        return torch.randn(batch_size, horizon, self.config.action_dim, device=device)

    def generate_candidates_random(
        self,
        initial_state: torch.Tensor,
        num_candidates: Optional[int] = None,
        horizon: Optional[int] = None
    ) -> List[Trajectory]:
        """
        Generate candidate trajectories using random shooting.

        This is used by HippocampalModule during CEM iterations.
        E2 provides the transitions; HippocampalModule orchestrates the search.

        Args:
            initial_state: Starting latent state [batch, latent_dim]
            num_candidates: Number of candidates to generate
            horizon: Rollout horizon

        Returns:
            List of Trajectory objects
        """
        num_candidates = num_candidates or self.config.num_candidates
        horizon = horizon or self.config.rollout_horizon
        device = initial_state.device
        batch_size = initial_state.shape[0]

        candidates = []
        for _ in range(num_candidates):
            actions = self.generate_random_actions(batch_size, horizon, device)
            trajectory = self.rollout(initial_state, actions)
            candidates.append(trajectory)

        return candidates

    def generate_candidates(
        self,
        initial_state: torch.Tensor,
        method: str = "random",
        **kwargs
    ) -> List[Trajectory]:
        """
        Generate candidate trajectories.

        V2: only "random" method supported. CEM refinement is
        HippocampalModule's responsibility (SD-001 resolution).

        Args:
            initial_state: Starting latent state
            method: Generation method (only "random" supported in V2)
            **kwargs: Additional arguments for the generation method

        Returns:
            List of Trajectory objects
        """
        if method == "random":
            return self.generate_candidates_random(initial_state, **kwargs)
        else:
            raise ValueError(
                f"Unknown generation method: {method!r}. "
                "V2 E2 only supports 'random'; use HippocampalModule for "
                "CEM-style iterative refinement."
            )

    def forward(
        self,
        initial_state: torch.Tensor,
        num_candidates: Optional[int] = None
    ) -> List[Trajectory]:
        """
        Forward pass: generate random candidate trajectories from initial state.
        """
        return self.generate_candidates_random(initial_state, num_candidates=num_candidates)
