"""
E3 Trajectory Selector Implementation

E3 evaluates candidate sensorimotor futures and commits to one trajectory.
It is not a perceptual system and does not overwrite the shared sensory latent.

Key responsibilities:
- Select a coherent sensorimotor future from affordance space
- Raise precision on the selected plan (commitment)
- Convert hypothetical rollouts into learning-relevant prediction errors

Current trajectory scoring equation (working hypothesis — see ARCHITECTURE NOTE):
    J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)

Where:
- F(ζ): Reality constraint (predictive coherence and viability)
- M(ζ): Ethical cost (predicted degradation of self/others)
- Φ_R(ζ): Residue field (persistent ethical cost, path dependent)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import E3Config
from ree_core.predictors.e2_fast import Trajectory
from ree_core.residue.field import ResidueField


@dataclass
class SelectionResult:
    """Result of trajectory selection.

    Attributes:
        selected_trajectory: The chosen trajectory
        selected_index: Index of the chosen trajectory
        selected_action: First action from the selected trajectory
        scores: All trajectory scores
        precision: Commitment precision level
        committed: Whether commitment threshold was met
        log_prob: Log-probability of the selected trajectory (for policy gradient).
            Remains connected to the computation graph so that policy_loss.backward()
            can update reality_scorer and ethical_scorer weights.
    """
    selected_trajectory: Trajectory
    selected_index: int
    selected_action: torch.Tensor
    scores: torch.Tensor
    precision: float
    committed: bool
    log_prob: Optional[torch.Tensor] = None


class E3TrajectorySelector(nn.Module):
    """
    E3 Trajectory Selector for ethical action selection.

    Evaluates candidate trajectories from HippocampalModule (V2) or E2 (V1)
    and selects one by minimising the combined cost:
        J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)

    Commitment is implemented as precision gating:
    - Before commitment, rollouts are hypotheses
    - After commitment, the selected trajectory is treated as intended
    - Only committed outcomes make errors diagnostic of the model

    # ARCHITECTURE NOTE (V2)
    # -----------------------------------------------------------------------
    # The scoring equation J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ) is a WORKING
    # HYPOTHESIS, not a settled canonical formulation.
    #
    # The weights lambda_ethical and rho_residue are placeholder parameters.
    # The entire scoring function — including which terms are included, how
    # they are weighted, and how they interact — requires genuine experimental
    # validation and is expected to be redesigned as the three-gate basal
    # ganglia model and HippocampalModule architecture mature.
    #
    # In particular:
    # - F(ζ) (reality constraint) is currently implemented as trajectory
    #   smoothness + final-state viability score. This is a proxy; the actual
    #   reality constraint mechanism is not yet specified.
    # - M(ζ) (ethical cost) currently aggregates E2 harm predictions. As
    #   self-attribution experiments mature (Step 2.4), M(ζ) should distinguish
    #   agent-caused from environment-caused harm — the current implementation
    #   does not make this distinction.
    # - Φ_R(ζ) (residue field) is treated as a scalar cost term here. See
    #   the architectural note in residue/field.py: the residue field may
    #   eventually be an *input* to multiple modules rather than a penalty
    #   appended to E3's score.
    #
    # Do not treat the current implementation as architecturally settled.
    # The scoring weights (lambda_ethical, rho_residue) are not tuned
    # constants; calibration experiments are needed.
    # -----------------------------------------------------------------------
    """

    def __init__(
        self,
        config: Optional[E3Config] = None,
        residue_field: Optional[ResidueField] = None
    ):
        super().__init__()
        self.config = config or E3Config()

        # Residue field reference (shared with agent)
        self.residue_field = residue_field

        # Reality constraint scorer: F(ζ)
        # Evaluates predictive coherence and physical viability
        self.reality_scorer = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Ethical cost estimator: M(ζ)
        # Predicts degradation of self and mirrored other-models
        self.ethical_scorer = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Precision control
        self.current_precision = self.config.precision_init

        # Track commitment state
        self._committed_trajectory: Optional[Trajectory] = None

        # Last selection scores (stored for confidence diagnostics)
        self.last_scores: Optional[torch.Tensor] = None

    def compute_reality_cost(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Compute reality constraint cost F(ζ).

        Evaluates:
        - Predictive coherence across the trajectory
        - Physical viability / constraint satisfaction

        Note: This is currently a proxy implementation (smoothness +
        viability score). The actual reality constraint mechanism is
        not yet specified — see ARCHITECTURE NOTE.

        Args:
            trajectory: Candidate trajectory

        Returns:
            Reality cost [batch]
        """
        states = trajectory.get_state_sequence()  # [batch, horizon, latent_dim]

        # Coherence: smoothness of state transitions
        if states.shape[1] > 1:
            transitions = states[:, 1:, :] - states[:, :-1, :]
            coherence_cost = transitions.pow(2).sum(dim=-1).mean(dim=-1)
        else:
            coherence_cost = torch.zeros(states.shape[0], device=states.device)

        # Viability: score final state
        final_state = trajectory.get_final_state()
        viability_score = self.reality_scorer(final_state).squeeze(-1)

        # Combine (lower is better)
        reality_cost = coherence_cost - viability_score

        return reality_cost

    def compute_ethical_cost(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Compute ethical cost M(ζ).

        Evaluates predicted degradation of self-model and mirrored other-models.

        Note: Currently does not distinguish agent-caused from environment-caused
        harm. This distinction requires SD-003 self-attribution experiments
        (Step 2.4). See ARCHITECTURE NOTE.

        Args:
            trajectory: Candidate trajectory

        Returns:
            Ethical cost [batch]
        """
        # Use E2's harm predictions
        if trajectory.harm_predictions is not None:
            harm_cost = trajectory.harm_predictions.sum(dim=-1)
        else:
            harm_cost = torch.zeros(trajectory.states[0].shape[0], device=trajectory.states[0].device)

        # Additional scoring from final state
        final_state = trajectory.get_final_state()
        ethical_score = self.ethical_scorer(final_state).squeeze(-1)

        # Combine (higher harm + lower ethical score = higher cost)
        ethical_cost = harm_cost - ethical_score

        return ethical_cost

    def compute_residue_cost(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Compute residue field cost Φ_R(ζ).

        Evaluates the trajectory through the persistent ethical geometry.
        States near past harm have higher cost.

        Args:
            trajectory: Candidate trajectory

        Returns:
            Residue cost [batch]
        """
        if self.residue_field is None:
            return torch.zeros(trajectory.states[0].shape[0], device=trajectory.states[0].device)

        states = trajectory.get_state_sequence()
        residue_cost = self.residue_field.evaluate_trajectory(states)

        return residue_cost

    def score_trajectory(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Compute total score J(ζ) for a trajectory.

        J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)  [working hypothesis]

        Lower score is better.

        Args:
            trajectory: Candidate trajectory

        Returns:
            Total score [batch]
        """
        reality_cost = self.compute_reality_cost(trajectory)
        ethical_cost = self.compute_ethical_cost(trajectory)
        residue_cost = self.compute_residue_cost(trajectory)

        ethical_term = self.config.lambda_ethical * ethical_cost
        residue_term = self.config.rho_residue * residue_cost

        total_score = reality_cost + ethical_term + residue_term

        return total_score

    def select(
        self,
        candidates: List[Trajectory],
        temperature: float = 1.0
    ) -> SelectionResult:
        """
        Select the best trajectory from candidates.

        Selection uses softmax over negative scores (lower is better).
        Temperature controls exploration vs exploitation.

        Args:
            candidates: List of candidate trajectories
            temperature: Softmax temperature for selection

        Returns:
            SelectionResult with chosen trajectory and metadata
        """
        if not candidates:
            raise ValueError("No candidate trajectories provided")

        # Score all candidates
        scores = torch.stack([self.score_trajectory(traj) for traj in candidates])
        scores = scores.mean(dim=-1)  # Aggregate over batch
        self.last_scores = scores.detach()

        # Apply softmax selection (lower score = higher probability)
        probs = F.softmax(-scores / temperature, dim=0)

        # Sample or take argmax based on precision
        if self.current_precision > self.config.commitment_threshold:
            # High precision: commit to best
            selected_idx = scores.argmin().item()
            committed = True
        else:
            # Low precision: sample from distribution
            selected_idx = torch.multinomial(probs, 1).item()
            committed = False

        selected_trajectory = candidates[selected_idx]

        # Extract first action from selected trajectory
        selected_action = selected_trajectory.actions[:, 0, :]

        # Compute log-probability for policy gradient (REINFORCE)
        log_probs = F.log_softmax(-scores / temperature, dim=0)
        log_prob = log_probs[selected_idx]

        if committed:
            self._committed_trajectory = selected_trajectory

        return SelectionResult(
            selected_trajectory=selected_trajectory,
            selected_index=selected_idx,
            selected_action=selected_action,
            scores=scores,
            precision=self.current_precision,
            committed=committed,
            log_prob=log_prob,
        )

    def update_precision(
        self,
        prediction_error: torch.Tensor,
        increase_rate: float = 0.1,
        decrease_rate: float = 0.05
    ) -> None:
        """
        Update precision based on prediction accuracy.

        Good predictions increase precision (confidence in plans).
        Poor predictions decrease precision (more exploration needed).

        Args:
            prediction_error: Error from committed trajectory
            increase_rate: Rate of precision increase on success
            decrease_rate: Rate of precision decrease on failure
        """
        error_magnitude = prediction_error.pow(2).mean().item()
        error_threshold = 0.1

        if error_magnitude < error_threshold:
            self.current_precision = min(
                self.config.precision_max,
                self.current_precision + increase_rate
            )
        else:
            self.current_precision = max(
                self.config.precision_min,
                self.current_precision - decrease_rate
            )

    def get_commitment_state(self) -> Dict[str, any]:
        """Get current commitment state for monitoring."""
        return {
            "precision": self.current_precision,
            "is_committed": self._committed_trajectory is not None,
            "commitment_threshold": self.config.commitment_threshold
        }

    def post_action_update(
        self,
        actual_outcome: torch.Tensor,
        harm_occurred: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Update E3 after action execution.

        After executing a committed plan:
        - Prediction errors become diagnostic
        - Residue field is updated if harm occurred

        Args:
            actual_outcome: Actual observed state
            harm_occurred: Whether harm/degradation occurred

        Returns:
            Dictionary of update metrics
        """
        metrics = {}

        if self._committed_trajectory is not None:
            predicted_state = self._committed_trajectory.states[1]
            prediction_error = actual_outcome - predicted_state

            self.update_precision(prediction_error)

            metrics["prediction_error"] = prediction_error.pow(2).mean()
            metrics["precision_after"] = torch.tensor(self.current_precision)

            if harm_occurred and self.residue_field is not None:
                self.residue_field.accumulate(predicted_state, harm_magnitude=1.0)
                metrics["residue_updated"] = torch.tensor(1.0)

        self._committed_trajectory = None

        return metrics

    def forward(
        self,
        candidates: List[Trajectory],
        temperature: float = 1.0
    ) -> SelectionResult:
        """Forward pass: select best trajectory from candidates."""
        return self.select(candidates, temperature)
