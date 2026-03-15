"""
HippocampalModule — Trajectory Proposal via Terrain Navigation

Resolves SD-001: In V1, E2FastPredictor.generate_candidates_cem() was
performing hippocampal work — iterative CEM-based trajectory search guided
by harm scores. E2 is a pure fast transition model f(z_t, a_t) → z_{t+1};
trajectory search and refinement belong here.

Conceptual role:
- E2 knows HOW the world transitions (pure physics, no preference)
- HippocampalModule knows WHERE to search (terrain navigation guided by
  the harm geometry encoded in the residue field)
- HippocampalModule uses E2 as a forward model to evaluate candidate paths
- The residue field provides the terrain: past harm creates "elevation"
  that trajectory proposals learn to route around

Architecture:
1. terrain_prior network: maps (z_beta, residue_value_at_z) → initial action
   distribution mean, biasing proposals toward low-residue regions
2. CEM refinement loop: iteratively improves candidates using E2 rollouts
   scored against terrain + E2 harm predictions
3. Returns final candidates for E3 to evaluate

V3 considerations (SD-004 — see docs/architecture/e2.md §5):
- The terrain_prior is currently a simple MLP operating in z_gamma state
  space. The V3 target replaces this with navigation over ACTION OBJECTS —
  compressed latent representations of action transformations produced by
  E2's bottleneck layer. Action objects are the place-cell-like primitives
  the hippocampal map should be built over, not raw states.
  Benefits: (a) map compaction — many states share the same action objects;
  (b) E2 hidden layer can be smaller (bottleneck reduces dimensionality);
  (c) CEM rollouts can be much longer horizon in action-object space.
- The CEM scoring function here (harm + residue) is independent of E3's
  J(ζ) scoring. Whether these should be unified or kept separate is an
  open design question.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from ree_core.utils.config import HippocampalConfig
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory
from ree_core.residue.field import ResidueField


class HippocampalModule(nn.Module):
    """
    HippocampalModule: trajectory proposal by residue-terrain navigation.

    Uses E2 (pure transition model) to roll out candidate action sequences,
    and the residue field to bias search away from harm-dense regions.
    CEM-style iterative refinement is implemented here (not in E2).

    SD-001 resolution:
    This module replaces E2FastPredictor.generate_candidates_cem() entirely.
    E2 is passed in as a dependency and used only for forward rollouts.
    The search strategy and terrain weighting are hippocampal concerns.
    """

    def __init__(
        self,
        config: HippocampalConfig,
        e2: E2FastPredictor,
        residue_field: ResidueField
    ):
        super().__init__()
        self.config = config
        self.e2 = e2
        self.residue_field = residue_field

        # Terrain-aware action prior:
        # Maps (z_beta, e1_prior, scalar_residue_at_z) → initial action distribution mean
        # The residue value at the current state biases the prior toward
        # action directions that have historically avoided harm.
        # The E1 prior conditions the proposal on long-horizon associative context
        # (SD-002: E1 primes E2 via associative prior; wired here into terrain search).
        # Input: 2*latent_dim + 1 (z_beta + e1_prior + residue scalar)
        # Output: action_dim * horizon (flattened action mean)
        self.terrain_prior = nn.Sequential(
            nn.Linear(config.latent_dim * 2 + 1, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim * config.horizon)
        )

    def _get_terrain_action_mean(
        self,
        z_beta: torch.Tensor,
        e1_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute terrain-informed action distribution mean.

        Queries the residue field at the current state and uses it
        to bias the initial action proposal toward low-harm directions.
        The E1 prior (if provided) conditions the proposal on long-horizon
        associative context — implementing SD-002's E1→HippocampalModule
        wiring.

        Args:
            z_beta: Affordance latent [batch, latent_dim]
            e1_prior: E1-generated prior for E2 conditioning [batch, latent_dim].
                If None, zeros are used (unconditioned fallback).

        Returns:
            Action mean [batch, horizon, action_dim]
        """
        with torch.no_grad():
            residue_val = self.residue_field.evaluate(z_beta).unsqueeze(-1)  # [batch, 1]

        # Use E1 prior when available; fall back to zeros (unconditioned)
        if e1_prior is None:
            e1_prior = torch.zeros_like(z_beta)

        # [batch, 2*latent_dim + 1]
        combined = torch.cat([z_beta, e1_prior, residue_val], dim=-1)
        mean_flat = self.terrain_prior(combined)  # [batch, action_dim * horizon]
        return mean_flat.view(z_beta.shape[0], self.config.horizon, self.config.action_dim)

    def _score_trajectory(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Score a trajectory for CEM elite selection.

        Lower is better. Combines:
        - E2's harm predictions (direct harm estimate along the path)
        - Residue field cost along the trajectory (accumulated harm geometry)

        Returns a scalar tensor.
        """
        harm = (
            trajectory.harm_predictions.sum()
            if trajectory.harm_predictions is not None
            else torch.tensor(0.0)
        )
        states = trajectory.get_state_sequence()  # [batch, horizon, latent_dim]
        residue = self.residue_field.evaluate_trajectory(states).sum()
        return harm + 0.5 * residue

    def propose_trajectories(
        self,
        z_beta: torch.Tensor,
        num_candidates: Optional[int] = None,
        e1_prior: Optional[torch.Tensor] = None
    ) -> List[Trajectory]:
        """
        Propose candidate trajectories via terrain-guided CEM.

        Unlike E2's random shooting, this iteratively refines action
        proposals using the residue field as navigation terrain and
        E1's associative prior as long-horizon context.

        Algorithm:
        1. Initialise action distribution mean from terrain prior
           (conditioned on z_beta, E1 prior, and residue geometry)
        2. For each CEM iteration:
           a. Sample num_candidates action sequences from current distribution
           b. Roll each out through E2 (pure transition model)
           c. Score each trajectory (harm + residue)
           d. Refit distribution mean/std to elite (lowest-scoring) samples
        3. Return final set of trajectories for E3 to evaluate

        SD-001: E2 is used here *only* for rollouts — it has no role in
        the search strategy. The terrain prior and CEM loop are entirely
        this module's responsibility.

        SD-002: The E1 prior is passed in from the agent and conditions
        the terrain prior, implementing the E1→HippocampalModule wiring
        described in docs/architecture/e2.md §4.

        Args:
            z_beta: Affordance latent [batch, latent_dim]
            num_candidates: Number of candidates per CEM iteration
            e1_prior: E1-generated prior for conditioning [batch, latent_dim].
                If None, unconditioned fallback (zeros) is used.

        Returns:
            List of Trajectory objects (final CEM iteration)
        """
        num_candidates = num_candidates or self.config.num_candidates
        num_elite = max(1, int(num_candidates * self.config.elite_fraction))
        batch_size = z_beta.shape[0]
        device = z_beta.device

        # Initialise from terrain prior (bias toward low-residue regions,
        # conditioned on E1 long-horizon associative context — SD-002)
        action_mean = self._get_terrain_action_mean(z_beta, e1_prior=e1_prior)  # [batch, horizon, action_dim]
        action_std = torch.ones_like(action_mean)

        all_trajectories: List[Trajectory] = []

        for _iteration in range(self.config.num_cem_iterations):
            trajectories: List[Trajectory] = []
            scores: List[torch.Tensor] = []

            for _ in range(num_candidates):
                noise = torch.randn_like(action_mean)
                actions = action_mean + action_std * noise  # [batch, horizon, action_dim]

                trajectory = self.e2.rollout(z_beta, actions)
                trajectories.append(trajectory)
                scores.append(self._score_trajectory(trajectory))

            scores_tensor = torch.stack(scores)  # [num_candidates]

            # Select elite (lowest score = least harm)
            elite_indices = torch.argsort(scores_tensor)[:num_elite]

            # Refit distribution to elite samples
            elite_actions = torch.stack(
                [trajectories[i].actions for i in elite_indices]
            )  # [num_elite, batch, horizon, action_dim]
            action_mean = elite_actions.mean(dim=0)
            action_std = elite_actions.std(dim=0) + 1e-6  # Prevent collapse

            all_trajectories = trajectories

        return all_trajectories

    def forward(
        self,
        z_beta: torch.Tensor,
        num_candidates: Optional[int] = None,
        e1_prior: Optional[torch.Tensor] = None
    ) -> List[Trajectory]:
        """Forward pass: propose trajectories from affordance latent."""
        return self.propose_trajectories(z_beta, num_candidates, e1_prior=e1_prior)
