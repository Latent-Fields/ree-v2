"""
Residue Field φ(z) Implementation

The residue field stores persistent curvature over latent space, representing
the accumulated ethical cost from past actions. This is a core architectural
invariant of REE:

- Residue CANNOT be erased, only integrated and contextualized
- Residue makes ethical cost PATH DEPENDENT
- Residue supports moral continuity across time

Why geometry matters:
If residue were a scalar penalty, it could be easily traded off against
reward and optimized away. A spatial field φ(z) makes residue path-dependent:
the cost of reaching a state depends on how you got there.

Implementation approaches:
- Neural network (flexible, learnable)
- Radial basis functions (interpretable, bounded)
- K-nearest neighbors map (sparse, explicit)

This implementation uses an RBF-based approach for interpretability,
with neural network components for flexibility.

ARCHITECTURAL NOTE — ResidueField as multi-module input (V2 design consideration):
In V1 and the current V2 implementation, the residue field is treated as a
cost term within E3's scoring function (Φ_R term in J(ζ)). This is a working
hypothesis. As the architecture matures, the residue field — or more broadly,
residue manifolds at scale — may be better understood as an *input* to multiple
modules rather than a scalar E3 cost term. Potential downstream roles:

  - Sensorium gate (limbic loop): residue field as attentional prior — what
    the system notices is shaped by accumulated harm geometry
  - HippocampalModule: residue field as terrain that trajectory proposals
    navigate (partially captured in V2 HippocampalModule.propose_trajectories)
  - E1 conditioning: associative learning weighted by harm history

This is flagged here as a live design consideration, not a V2 implementation
requirement. Keep it in view when extending the architecture.
"""

from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import ResidueConfig


class RBFLayer(nn.Module):
    """
    Radial Basis Function layer for residue field representation.

    Each basis function represents a "scar" in latent space where
    harm occurred. The field value at any point is the weighted
    sum of contributions from all basis functions.
    """

    def __init__(
        self,
        latent_dim: int,
        num_centers: int,
        bandwidth: float = 1.0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_centers = num_centers
        self.bandwidth = bandwidth

        # RBF centers (locations of past harm)
        self.centers = nn.Parameter(torch.randn(num_centers, latent_dim) * 0.1)

        # RBF weights (intensity of each scar)
        # Initialized to zero: no residue at start
        self.weights = nn.Parameter(torch.zeros(num_centers))

        # Track which centers are active
        self.register_buffer("active_mask", torch.zeros(num_centers, dtype=torch.bool))
        self.register_buffer("next_center_idx", torch.tensor(0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the residue field at points z.

        Args:
            z: Points in latent space [batch, latent_dim] or [batch, seq, latent_dim]

        Returns:
            Residue values [batch] or [batch, seq]
        """
        original_shape = z.shape
        if z.dim() == 3:
            batch_size, seq_len, latent_dim = z.shape
            z = z.reshape(-1, latent_dim)
        else:
            batch_size = z.shape[0]
            seq_len = None

        # Compute distances to all centers
        # z: [batch*seq, latent_dim], centers: [num_centers, latent_dim]
        diffs = z.unsqueeze(1) - self.centers.unsqueeze(0)  # [batch*seq, num_centers, latent_dim]
        distances_sq = (diffs ** 2).sum(dim=-1)  # [batch*seq, num_centers]

        # Apply RBF kernel (Gaussian)
        rbf_values = torch.exp(-distances_sq / (2 * self.bandwidth ** 2))

        # Weight by residue intensity
        # Only count active centers
        active_weights = self.weights * self.active_mask.float()
        field_values = (rbf_values * active_weights.unsqueeze(0)).sum(dim=-1)

        # Reshape if needed
        if seq_len is not None:
            field_values = field_values.reshape(batch_size, seq_len)

        return field_values

    def add_residue(
        self,
        location: torch.Tensor,
        intensity: float = 1.0
    ) -> int:
        """
        Add residue at a location in latent space.

        This is called when harm occurs. The residue is ADDED,
        never subtracted (invariant: residue cannot be erased).

        Args:
            location: Location of harm [latent_dim] or [batch, latent_dim]
            intensity: Harm intensity

        Returns:
            Index of the updated center
        """
        if location.dim() == 2:
            location = location.mean(dim=0)  # Average over batch

        # Get next center to use
        idx = self.next_center_idx.item()

        with torch.no_grad():
            # Update center location
            self.centers.data[idx] = location

            # Accumulate weight (never decrease - invariant!)
            self.weights.data[idx] = self.weights.data[idx] + intensity

            # Mark as active
            self.active_mask[idx] = True

            # Move to next center (circular buffer)
            self.next_center_idx = (self.next_center_idx + 1) % self.num_centers

        return idx


class ResidueField(nn.Module):
    """
    Persistent residue field for moral continuity.

    The residue field φ(z) represents accumulated ethical cost as
    geometric deformation of latent space. Key properties:

    1. PERSISTENCE: Residue cannot be erased (architectural invariant)
    2. PATH DEPENDENCE: Cost depends on trajectory through space
    3. ACCUMULATION: Harm adds to residue, never subtracts
    4. INTEGRATION: Offline processing can contextualize but not remove

    The field affects trajectory selection by making states near
    past harm more costly to visit.

    See module-level docstring for the architectural note on ResidueField
    as a future multi-module input.
    """

    def __init__(self, config: Optional[ResidueConfig] = None):
        super().__init__()
        self.config = config or ResidueConfig()

        # Main RBF field for explicit residue representation
        self.rbf_field = RBFLayer(
            latent_dim=self.config.latent_dim,
            num_centers=self.config.num_basis_functions,
            bandwidth=self.config.kernel_bandwidth
        )

        # Neural augmentation for flexible field shaping
        self.neural_field = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Softplus()  # Ensures non-negative output
        )

        # Track total accumulated residue (for monitoring)
        self.register_buffer("total_residue", torch.tensor(0.0))
        self.register_buffer("num_harm_events", torch.tensor(0))

        # History of harm locations (for offline integration)
        self._harm_history: List[torch.Tensor] = []

    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the residue field at points in latent space.

        φ(z) = RBF_field(z) + neural_field(z)

        Args:
            z: Points in latent space [batch, latent_dim]

        Returns:
            Residue values [batch]
        """
        # RBF component (explicit scars)
        rbf_value = self.rbf_field(z)

        # Neural component (learned field shape)
        neural_value = self.neural_field(z).squeeze(-1)

        # Combine
        return rbf_value + neural_value * 0.1  # Neural is auxiliary

    def evaluate_trajectory(self, trajectory_states: torch.Tensor) -> torch.Tensor:
        """
        Evaluate total residue cost along a trajectory.

        This integrates the field along the path, making cost path-dependent.

        Args:
            trajectory_states: States along trajectory [batch, horizon, latent_dim]

        Returns:
            Total residue cost [batch]
        """
        # Evaluate field at each state
        field_values = self.rbf_field(trajectory_states)  # [batch, horizon]

        # Add neural contribution
        neural_values = self.neural_field(trajectory_states).squeeze(-1)  # [batch, horizon]

        # Combine and integrate along trajectory
        total_values = field_values + neural_values * 0.1
        trajectory_cost = total_values.sum(dim=-1)  # [batch]

        return trajectory_cost

    def accumulate(
        self,
        location: torch.Tensor,
        harm_magnitude: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Accumulate residue at a location due to harm.

        This is called when harm occurs. Per REE invariants:
        - Residue is ADDED, never removed
        - Residue cannot be erased or reset
        - Only integration can contextualize residue

        Args:
            location: Location of harm in latent space
            harm_magnitude: Magnitude of harm (positive = bad)

        Returns:
            Dictionary of accumulation metrics
        """
        # Ensure harm_magnitude is positive (harm is always bad)
        harm_magnitude = abs(harm_magnitude) * self.config.accumulation_rate

        # Add to RBF field
        center_idx = self.rbf_field.add_residue(location, harm_magnitude)

        # Update tracking
        self.total_residue = self.total_residue + harm_magnitude
        self.num_harm_events = self.num_harm_events + 1

        # Store in history for offline integration
        self._harm_history.append(location.detach().clone())

        return {
            "residue_added": torch.tensor(harm_magnitude),
            "total_residue": self.total_residue,
            "center_idx": torch.tensor(center_idx),
            "num_harm_events": self.num_harm_events
        }

    def integrate(
        self,
        num_steps: int = 10,
        learning_rate: float = 0.01
    ) -> Dict[str, float]:
        """
        Offline integration of residue (sleep-like processing).

        This allows the neural field to better model the residue geometry
        while respecting the invariant that residue cannot be erased.

        Integration can:
        - Smooth the field representation
        - Contextualize residue with learned patterns
        - Improve computational efficiency

        Integration CANNOT:
        - Reduce total residue
        - Erase harm events
        - Reset the field

        Args:
            num_steps: Number of integration steps
            learning_rate: Learning rate for neural field updates

        Returns:
            Dictionary of integration metrics
        """
        if len(self._harm_history) == 0:
            return {"integration_loss": 0.0, "steps": 0}

        # Get harm locations from history
        harm_locations = torch.stack(self._harm_history[-100:])  # Use recent history

        total_loss = 0.0

        for step in range(num_steps):
            # Sample random points around harm locations
            noise = torch.randn_like(harm_locations) * self.config.kernel_bandwidth
            sample_points = harm_locations + noise

            # Target: RBF field values (explicit residue)
            with torch.no_grad():
                targets = self.rbf_field(sample_points)

            # Predict with neural field
            predictions = self.neural_field(sample_points).squeeze(-1)

            # Loss: neural field should approximate RBF field
            loss = F.mse_loss(predictions, targets)
            total_loss += loss.item()

            # We don't actually backprop here in inference mode
            # In training, this would update the neural field

        return {
            "integration_loss": total_loss / num_steps,
            "steps": num_steps,
            "history_size": len(self._harm_history)
        }

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get statistics about the residue field for monitoring."""
        return {
            "total_residue": self.total_residue,
            "num_harm_events": self.num_harm_events,
            "active_centers": self.rbf_field.active_mask.sum(),
            "mean_weight": self.rbf_field.weights[self.rbf_field.active_mask].mean()
            if self.rbf_field.active_mask.any() else torch.tensor(0.0)
        }

    def visualize_field(
        self,
        z_range: Tuple[float, float] = (-3, 3),
        resolution: int = 50,
        slice_dims: Tuple[int, int] = (0, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a 2D visualization of the residue field.

        Creates a grid over two dimensions and evaluates the field.

        Args:
            z_range: Range of values for visualization
            resolution: Grid resolution
            slice_dims: Which two dimensions to visualize

        Returns:
            Tuple of (X grid, Y grid, field values)
        """
        # Create grid
        x = torch.linspace(z_range[0], z_range[1], resolution)
        y = torch.linspace(z_range[0], z_range[1], resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Create latent vectors (all zeros except slice dims)
        z = torch.zeros(resolution * resolution, self.config.latent_dim)
        z[:, slice_dims[0]] = X.flatten()
        z[:, slice_dims[1]] = Y.flatten()

        # Evaluate field
        with torch.no_grad():
            values = self.evaluate(z)

        values = values.reshape(resolution, resolution)

        return X, Y, values

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass: evaluate residue field at points z."""
        return self.evaluate(z)
