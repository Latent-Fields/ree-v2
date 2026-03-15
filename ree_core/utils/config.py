"""
Configuration classes for REE-v2 components.

V2 changes vs V1:
- Added HippocampalConfig for the new HippocampalModule (resolves SD-001)
- E2Config docstring updated: E2 is now a *pure* fast transition model;
  CEM refinement has moved to HippocampalModule
- E3Config docstring flags J(ζ) scoring equation as a working hypothesis
- REEConfig.from_dims() propagates hippocampal dimensions
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LatentStackConfig:
    """Configuration for the multi-timescale latent stack (L-space).

    The latent stack represents temporally displaced prediction depths:
    - gamma (γ): Shared sensory binding / feature conjunction
    - beta (β): Affordance and immediate action-set maintenance
    - theta (θ): Sequence context, temporal ordering
    - delta (δ): Regime, motivational set, long-horizon context
    """
    observation_dim: int = 64
    latent_dim: int = 64

    # Dimensions for each depth level
    gamma_dim: int = 64   # Sensory binding
    beta_dim: int = 64    # Affordance/action
    theta_dim: int = 32   # Sequence context
    delta_dim: int = 32   # Regime/motivation

    # Top-down conditioning dimensions
    topdown_dim: int = 16

    # Activation function
    activation: str = "relu"


@dataclass
class E1Config:
    """Configuration for E1 Deep Predictor.

    E1 handles long-horizon latent trajectories and context,
    operating at slower timescales than E2.

    Episode-boundary semantics (V2 note):
    E1 maintains a persistent hidden state (self._hidden_state) across
    episode steps. This state is reset at the start of each episode via
    reset_hidden_state(). The hidden state is intentionally NOT reset
    during the offline prediction-loss computation — a saved/restored
    pattern is used instead — so that inference continuity is preserved
    while training replays are still possible.
    """
    latent_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    prediction_horizon: int = 20  # Steps into future
    learning_rate: float = 1e-4


@dataclass
class E2Config:
    """Configuration for E2 Fast Predictor.

    V2: E2 is a *pure* fast transition model — f(z_t, a_t) → z_{t+1}.
    Candidate generation via CEM (generate_candidates_cem) has been removed;
    iterative trajectory refinement is now the responsibility of
    HippocampalModule. E2 exposes generate_candidates_random() for simple
    random shooting (used internally by HippocampalModule during CEM
    iterations) and forward_counterfactual() for SD-003 self-attribution
    experiments.

    Architectural note — E2 horizon vs E1 horizon:
    E2 operates exclusively on z_gamma, the unified latent space of all
    sensory modalities (where coherent objects form). It does NOT operate
    on raw sensory streams. E2's rollout_horizon should be LONGER than
    E1's prediction_horizon because:
    - E1 predicts the "perceived present" by predicting a short way ahead
      in sensory-only latent space (associative, no action conditioning).
    - E2 predicts how motor actions will transform latent sensory objects
      further into the future, supporting multi-step trajectory planning.
    E1 prediction_horizon default: 20. E2 rollout_horizon default: 30.
    """
    latent_dim: int = 64
    action_dim: int = 4
    hidden_dim: int = 128
    num_layers: int = 2
    rollout_horizon: int = 30  # Steps for trajectory rollout; must exceed E1.prediction_horizon
    num_candidates: int = 32   # Default candidate count (used by HippocampalModule)
    learning_rate: float = 3e-4


@dataclass
class E3Config:
    """Configuration for E3 Trajectory Selector.

    Working hypothesis scoring equation:
        J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)

    This formulation is a description of how trajectory selection could
    work — it is NOT a settled canonical design. The weights lambda_ethical
    and rho_residue are placeholder parameters pending proper calibration
    experiments. The scoring function as a whole is expected to be redesigned
    as the three-gate BG model and HippocampalModule architecture mature.
    See ARCHITECTURE NOTE in trajectory/e3_selector.py.
    """
    latent_dim: int = 64
    hidden_dim: int = 64

    # Scoring weights — placeholder parameters, not tuned constants
    lambda_ethical: float = 1.0   # Weight for ethical cost M
    rho_residue: float = 0.5      # Weight for residue field Φ_R

    # Precision control
    commitment_threshold: float = 0.7  # Precision threshold for commitment
    precision_init: float = 0.5
    precision_max: float = 1.0
    precision_min: float = 0.1


@dataclass
class HippocampalConfig:
    """Configuration for HippocampalModule (new in V2, resolves SD-001).

    The HippocampalModule proposes candidate trajectories by navigating
    the residue-weighted affective terrain, using E2 as a pure forward
    model for rollouts. CEM-style iterative refinement lives here.

    SD-001 resolution: In V1, E2FastPredictor.generate_candidates_cem()
    was conflating two roles — pure transition prediction (E2's job) and
    trajectory search/refinement (hippocampal function). This config
    covers the hippocampal search component only.
    """
    latent_dim: int = 64
    action_dim: int = 4
    hidden_dim: int = 128
    horizon: int = 10          # Should match E2Config.rollout_horizon
    num_candidates: int = 32   # Default candidate count per CEM iteration
    num_cem_iterations: int = 3
    elite_fraction: float = 0.2  # Fraction of top samples retained for refitting


@dataclass
class ResidueConfig:
    """Configuration for the Residue Field φ(z).

    Residue is stored as persistent curvature over latent space,
    making ethical cost path-dependent and supporting moral continuity.
    """
    latent_dim: int = 64
    hidden_dim: int = 64

    # Residue accumulation
    accumulation_rate: float = 0.1  # How fast residue accumulates
    decay_rate: float = 0.0         # 0 = no decay (invariant: residue cannot be erased)

    # Field representation
    num_basis_functions: int = 32   # For RBF representation
    kernel_bandwidth: float = 1.0

    # Integration (offline processing)
    integration_rate: float = 0.01  # Rate of contextualization


@dataclass
class EnvironmentConfig:
    """Configuration for environments."""
    size: int = 10
    num_resources: int = 5
    num_hazards: int = 3
    num_other_agents: int = 1

    # Reward/harm signals
    resource_benefit: float = 1.0
    hazard_harm: float = -1.0
    collision_harm: float = -0.5

    # Other agent properties
    other_agent_coupling: float = 0.5  # Mirror modelling weight


@dataclass
class REEConfig:
    """Master configuration for the complete REE-v2 agent.

    Bundles all component configurations with coordination settings.
    """
    # Component configs
    latent: LatentStackConfig = field(default_factory=LatentStackConfig)
    e1: E1Config = field(default_factory=E1Config)
    e2: E2Config = field(default_factory=E2Config)
    e3: E3Config = field(default_factory=E3Config)
    hippocampal: HippocampalConfig = field(default_factory=HippocampalConfig)
    residue: ResidueConfig = field(default_factory=ResidueConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # Global settings
    device: str = "cpu"
    seed: Optional[int] = None

    # Agent loop settings
    offline_integration_frequency: int = 100  # Steps between "sleep" cycles

    # MECH-057a: Action-loop completion gate
    # When True and sequence_in_progress=True, generate_trajectories() returns
    # cached candidates from the previous call rather than invoking HippocampalModule
    # for fresh proposals. Ablatable — set False for NO_GATE condition in EXQ-020.
    action_loop_gate_enabled: bool = False

    @classmethod
    def from_dims(
        cls,
        observation_dim: int,
        action_dim: int,
        latent_dim: int = 64
    ) -> "REEConfig":
        """Create config from basic dimension specifications."""
        config = cls()
        config.latent.observation_dim = observation_dim
        config.latent.latent_dim = latent_dim
        config.latent.gamma_dim = latent_dim
        config.latent.beta_dim = latent_dim
        config.e1.latent_dim = latent_dim
        config.e2.latent_dim = latent_dim
        config.e2.action_dim = action_dim
        config.e3.latent_dim = latent_dim
        config.hippocampal.latent_dim = latent_dim
        config.hippocampal.action_dim = action_dim
        config.hippocampal.horizon = config.e2.rollout_horizon
        config.residue.latent_dim = latent_dim
        return config
