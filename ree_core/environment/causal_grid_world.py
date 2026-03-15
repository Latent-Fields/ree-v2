"""
CausalGridWorld — Persistent Agent Causal Footprint Environment

V2 substrate for SD-003 self-attribution experiments.

The CausalGridWorld is a 2D grid world in which agent actions leave
persistent causal marks on the environment. This enables disambiguation
between agent-caused and environment-caused transitions — the prerequisite
for genuine self-attribution experiments.

Two distinguishable transition types
-------------------------------------
1. AGENT_CAUSED: state changes triggered by the agent's prior actions.
   - Each cell the agent visits accumulates a contamination value.
   - When contamination at a cell exceeds contamination_threshold, the cell
     becomes a "contaminated hazard" that causes harm on future visits.
   - The harm from a contaminated cell is agent-caused: it exists because
     the agent's prior trajectory created it.

2. ENV_CAUSED: state changes independent of agent actions.
   - Every env_drift_interval steps, active hazards move randomly
     (with probability env_drift_prob per hazard per drift event).
   - Resources occasionally respawn at new locations.
   - These transitions occur regardless of what the agent does.

Attribution signal in info dict
---------------------------------
Every step(), info['transition_type'] encodes what happened:
  "agent_caused_hazard"   — stepped on a contaminated cell the agent created
  "env_caused_hazard"     — stepped on a hazard that drifted there from env
  "resource"              — collected a resource
  "waypoint"              — visited next waypoint in sequence (subgoal_mode only)
  "sequence_complete"     — completed all waypoints in order (subgoal_mode only)
  "none"                  — normal step, no event

info also includes:
  "contamination_delta"   — contamination added this step at the agent's cell
  "env_drift_occurred"    — whether background drift happened this step
  "footprint_at_cell"     — agent's visit count at the cell just entered
  "sequence_in_progress"  — bool: agent is mid-committed-sequence (subgoal_mode)
  "sequence_step"         — int: which waypoint was just reached (subgoal_mode)

Observation additions vs GridWorld
------------------------------------
The observation includes two extra channels compared to V1 GridWorld:
  - Contamination local view (5×5 float): contamination level at each cell
    in the agent's 5×5 neighbourhood — reveals the agent's own footprint
  - Footprint density (1 float): normalised visit count at the agent's
    current cell — for self-model calibration

Sub-goal mode (subgoal_mode=True)
----------------------------------
Adds waypoint cells that the agent must visit in sequence to earn a
completion reward. This creates genuine multi-step committed action
sequences (required by MECH-057a / EXQ-020).

  - num_waypoints waypoint cells are placed at reset.
  - Visiting waypoint 0 starts a committed sequence (sequence_in_progress=True).
  - Visiting waypoints 1, 2, ... in order advances the sequence.
  - Completing all waypoints gives waypoint_completion_reward; waypoints respawn.
  - Visiting a waypoint out of order has no effect.
  - If sequence_commitment_timeout steps pass without hitting the next waypoint,
    the sequence resets (no completion reward; waypoints respawn).
  - Hazards remain active throughout, creating real interruption pressure.
  - Backward-compatible: subgoal_mode=False preserves all existing behaviour.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class CausalGridWorld:
    """
    2D grid world with persistent agent causal footprint (SD-003 substrate).

    Actions:
        0: Move up     (-1, 0)
        1: Move down   (+1, 0)
        2: Move left   (0, -1)
        3: Move right  (0, +1)
        4: Stay        (0,  0)

    Observation structure:
        position          : size*size floats  (one-hot)
        local_view        : 5*5*6 floats      (entity type, one-hot per cell)
        homeostatic       : 2 floats          (health, energy)
        contamination_view: 5*5 floats        (contamination level in 5×5 neighbourhood)
        footprint_density : 1 float           (normalised visit count at current cell)
    """

    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
        4: (0, 0),
    }

    ENTITY_TYPES: Dict[str, int] = {
        "empty": 0,
        "wall": 1,
        "resource": 2,
        "hazard": 3,
        "contaminated": 4,  # Agent-caused hazard
        "agent": 5,
        "waypoint": 6,      # Sub-goal waypoint (subgoal_mode only)
    }

    def __init__(
        self,
        size: int = 10,
        num_hazards: int = 3,
        num_resources: int = 5,
        contamination_spread: float = 0.5,
        contamination_threshold: float = 2.0,
        env_drift_interval: int = 5,
        env_drift_prob: float = 0.3,
        hazard_harm: float = 0.5,
        contaminated_harm: float = 0.4,
        resource_benefit: float = 0.3,
        energy_decay: float = 0.01,
        seed: Optional[int] = None,
        # --- Sub-goal mode (MECH-057a / EXQ-020) ---
        subgoal_mode: bool = False,
        num_waypoints: int = 3,
        waypoint_visit_reward: float = 0.2,
        waypoint_completion_reward: float = 0.8,
        sequence_commitment_timeout: int = 20,
    ):
        """
        Args:
            size: Grid dimensions (size × size)
            num_hazards: Number of env-caused hazards placed at reset
            num_resources: Number of resources placed at reset
            contamination_spread: Contamination increment per agent visit
            contamination_threshold: Contamination level at which a cell
                becomes a contaminated hazard
            env_drift_interval: Steps between background hazard drift events
            env_drift_prob: Per-hazard probability of moving during a drift event
            hazard_harm: Harm signal from env-caused hazard
            contaminated_harm: Harm signal from agent-caused contaminated cell
            resource_benefit: Benefit signal from collecting a resource
            energy_decay: Energy lost per step
            seed: RNG seed for reproducibility
            subgoal_mode: If True, place waypoints that must be visited in order.
                Creates genuine multi-step committed action sequences for
                MECH-057a / EXQ-020. Default False preserves original behaviour.
            num_waypoints: Number of waypoints in the sequence (subgoal_mode only)
            waypoint_visit_reward: Reward for visiting each intermediate waypoint
                in the correct order (subgoal_mode only)
            waypoint_completion_reward: Reward for completing the full sequence
                (visiting all waypoints in order) (subgoal_mode only)
            sequence_commitment_timeout: Max steps allowed between consecutive
                waypoint visits before the sequence resets (subgoal_mode only)
        """
        self.size = size
        self.num_hazards = num_hazards
        self.num_resources = num_resources
        self.contamination_spread = contamination_spread
        self.contamination_threshold = contamination_threshold
        self.env_drift_interval = env_drift_interval
        self.env_drift_prob = env_drift_prob
        self.hazard_harm = hazard_harm
        self.contaminated_harm = contaminated_harm
        self.resource_benefit = resource_benefit
        self.energy_decay = energy_decay

        # Sub-goal mode
        self.subgoal_mode = subgoal_mode
        self.num_waypoints = num_waypoints
        self.waypoint_visit_reward = waypoint_visit_reward
        self.waypoint_completion_reward = waypoint_completion_reward
        self.sequence_commitment_timeout = sequence_commitment_timeout

        self._rng = np.random.default_rng(seed)

        self.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        """Dimension of the observation vector."""
        position_dim = self.size * self.size
        local_view_dim = 5 * 5 * len(self.ENTITY_TYPES)
        homeostatic_dim = 2
        contamination_view_dim = 5 * 5
        footprint_dim = 1
        return position_dim + local_view_dim + homeostatic_dim + contamination_view_dim + footprint_dim

    @property
    def action_dim(self) -> int:
        return len(self.ACTIONS)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> torch.Tensor:
        """Reset environment to initial state."""
        # Base grid: 0 = empty, 1 = wall, etc.
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)

        # Border walls
        self.grid[0, :] = self.ENTITY_TYPES["wall"]
        self.grid[-1, :] = self.ENTITY_TYPES["wall"]
        self.grid[:, 0] = self.ENTITY_TYPES["wall"]
        self.grid[:, -1] = self.ENTITY_TYPES["wall"]

        # Persistent causal footprint layers
        self.contamination_grid = np.zeros((self.size, self.size), dtype=np.float32)
        self.footprint_grid = np.zeros((self.size, self.size), dtype=np.int32)

        # Available interior cells
        available = [
            (i, j)
            for i in range(1, self.size - 1)
            for j in range(1, self.size - 1)
        ]
        self._rng.shuffle(available)

        # Agent
        ax, ay = available.pop()
        self.agent_x = ax
        self.agent_y = ay
        self.agent_health = 1.0
        self.agent_energy = 1.0
        self.grid[ax, ay] = self.ENTITY_TYPES["agent"]

        # Env-caused hazards
        self.hazards: List[List[int]] = []
        for _ in range(min(self.num_hazards, len(available))):
            hx, hy = available.pop()
            self.grid[hx, hy] = self.ENTITY_TYPES["hazard"]
            self.hazards.append([hx, hy])

        # Resources
        self.resources: List[List[int]] = []
        for _ in range(min(self.num_resources, len(available))):
            rx, ry = available.pop()
            self.grid[rx, ry] = self.ENTITY_TYPES["resource"]
            self.resources.append([rx, ry])

        # Sub-goal waypoints (subgoal_mode only)
        self.waypoints: List[List[int]] = []
        self._next_waypoint_idx: int = 0      # Index of next expected waypoint
        self._sequence_in_progress: bool = False
        self._sequence_step: int = 0           # Last waypoint index reached
        self._steps_since_waypoint: int = 0    # Steps since last waypoint hit
        self._sequences_completed: int = 0

        if self.subgoal_mode:
            for _ in range(min(self.num_waypoints, len(available))):
                wx, wy = available.pop()
                self.grid[wx, wy] = self.ENTITY_TYPES["waypoint"]
                self.waypoints.append([wx, wy])

        self.steps = 0
        self.total_harm = 0.0
        self.total_benefit = 0.0

        return self._get_observation()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Execute one environment step.

        Returns:
            observation: New observation tensor
            harm_signal: float — negative = harm, positive = benefit
            done: bool
            info: dict with keys transition_type, contamination_delta,
                  env_drift_occurred, footprint_at_cell, health, energy,
                  steps, total_harm, total_benefit
        """
        if isinstance(action, torch.Tensor):
            action = action.argmax().item() if action.dim() > 0 else action.item()
        action = int(action) % len(self.ACTIONS)

        dx, dy = self.ACTIONS[action]
        new_x = self.agent_x + dx
        new_y = self.agent_y + dy

        harm_signal = 0.0
        transition_type = "none"
        contamination_delta = 0.0

        # --- Move agent if not wall ---
        if self.grid[new_x, new_y] != self.ENTITY_TYPES["wall"]:
            old_x, old_y = self.agent_x, self.agent_y

            # Clear old agent position
            # Determine what was under the agent before (contaminated or empty)
            if self.contamination_grid[old_x, old_y] >= self.contamination_threshold:
                self.grid[old_x, old_y] = self.ENTITY_TYPES["contaminated"]
            else:
                self.grid[old_x, old_y] = self.ENTITY_TYPES["empty"]

            target_type = self.grid[new_x, new_y]

            if target_type == self.ENTITY_TYPES["hazard"]:
                # ENV-CAUSED hazard
                harm_signal = -self.hazard_harm
                self.agent_health = max(0.0, self.agent_health - self.hazard_harm)
                transition_type = "env_caused_hazard"
                self.total_harm += self.hazard_harm

            elif target_type == self.ENTITY_TYPES["contaminated"]:
                # AGENT-CAUSED hazard (contaminated by prior visits)
                harm_signal = -self.contaminated_harm
                self.agent_health = max(0.0, self.agent_health - self.contaminated_harm)
                transition_type = "agent_caused_hazard"
                self.total_harm += self.contaminated_harm

            elif target_type == self.ENTITY_TYPES["resource"]:
                # Resource collection
                harm_signal = self.resource_benefit
                self.agent_health = min(1.0, self.agent_health + self.resource_benefit * 0.5)
                self.agent_energy = min(1.0, self.agent_energy + self.resource_benefit * 0.5)
                transition_type = "resource"
                self.total_benefit += self.resource_benefit
                # Remove this resource; it may respawn later via drift
                self.resources = [r for r in self.resources if not (r[0] == new_x and r[1] == new_y)]

            elif target_type == self.ENTITY_TYPES["waypoint"] and self.subgoal_mode:
                # Waypoint visit — check if this is the correct next waypoint
                wp_idx = next(
                    (i for i, w in enumerate(self.waypoints)
                     if w[0] == new_x and w[1] == new_y),
                    None
                )
                if wp_idx == self._next_waypoint_idx:
                    # Correct next waypoint in sequence
                    self._next_waypoint_idx += 1
                    self._sequence_step = wp_idx
                    self._steps_since_waypoint = 0

                    if wp_idx == 0:
                        # First waypoint — sequence begins
                        self._sequence_in_progress = True

                    # Remove visited waypoint from grid
                    self.grid[new_x, new_y] = self.ENTITY_TYPES["empty"]
                    self.waypoints[wp_idx] = [-1, -1]  # Mark as consumed

                    if self._next_waypoint_idx >= self.num_waypoints:
                        # All waypoints visited in order — sequence complete
                        harm_signal = self.waypoint_completion_reward
                        self.total_benefit += self.waypoint_completion_reward
                        transition_type = "sequence_complete"
                        self._sequence_in_progress = False
                        self._sequence_step = 0
                        self._next_waypoint_idx = 0
                        self._sequences_completed += 1
                        self._steps_since_waypoint = 0
                        # Respawn waypoints for next sequence
                        self._respawn_waypoints()
                    else:
                        # Intermediate waypoint — partial reward, sequence continues
                        harm_signal = self.waypoint_visit_reward
                        self.total_benefit += self.waypoint_visit_reward
                        transition_type = "waypoint"
                # else: wrong waypoint visited out of order — no effect

            # Move agent
            self.agent_x = new_x
            self.agent_y = new_y
            self.grid[new_x, new_y] = self.ENTITY_TYPES["agent"]

            # --- Update causal footprint ---
            self.footprint_grid[new_x, new_y] += 1
            self.contamination_grid[new_x, new_y] += self.contamination_spread
            contamination_delta = self.contamination_spread

            # Promote cell to contaminated type if threshold crossed
            # (will be visible on next step when agent moves off this cell)
            # The grid overlay is handled when agent departs.

        # --- Energy decay ---
        self.agent_energy = max(0.0, self.agent_energy - self.energy_decay)

        # --- Sequence timeout check (subgoal_mode) ---
        if self.subgoal_mode and self._sequence_in_progress:
            self._steps_since_waypoint += 1
            if self._steps_since_waypoint > self.sequence_commitment_timeout:
                # Sequence timed out — reset progress, respawn consumed waypoints
                self._sequence_in_progress = False
                self._sequence_step = 0
                self._next_waypoint_idx = 0
                self._steps_since_waypoint = 0
                self._respawn_waypoints()

        # --- Background env drift ---
        env_drift_occurred = False
        if self.steps > 0 and self.steps % self.env_drift_interval == 0:
            env_drift_occurred = True
            self._drift_hazards()
            self._maybe_respawn_resource()

        self.steps += 1

        done = (
            self.agent_health <= 0.0
            or self.agent_energy <= 0.0
            or self.steps >= 1000
        )

        info = {
            "transition_type": transition_type,
            "contamination_delta": contamination_delta,
            "env_drift_occurred": env_drift_occurred,
            "footprint_at_cell": int(self.footprint_grid[self.agent_x, self.agent_y]),
            "health": self.agent_health,
            "energy": self.agent_energy,
            "steps": self.steps,
            "total_harm": self.total_harm,
            "total_benefit": self.total_benefit,
            # Sub-goal sequence state (always present; False/0 when subgoal_mode=False)
            "sequence_in_progress": self._sequence_in_progress,
            "sequence_step": self._sequence_step,
            "sequences_completed": self._sequences_completed,
        }

        return self._get_observation(), harm_signal, done, info

    # ------------------------------------------------------------------
    # Sub-goal waypoint management (subgoal_mode only)
    # ------------------------------------------------------------------

    def _respawn_waypoints(self) -> None:
        """
        Respawn all waypoints at new random empty cells.

        Called after sequence completion or timeout to reset the
        waypoint sequence for the next committed-sequence cycle.
        """
        if not self.subgoal_mode:
            return

        # Clear any remaining waypoints from the grid
        for wp in self.waypoints:
            wx, wy = wp
            if (
                0 < wx < self.size - 1
                and 0 < wy < self.size - 1
                and self.grid[wx, wy] == self.ENTITY_TYPES["waypoint"]
            ):
                self.grid[wx, wy] = self.ENTITY_TYPES["empty"]

        # Find available cells
        empty_cells = [
            (i, j)
            for i in range(1, self.size - 1)
            for j in range(1, self.size - 1)
            if self.grid[i, j] == self.ENTITY_TYPES["empty"]
        ]
        self._rng.shuffle(empty_cells)

        self.waypoints = []
        for k in range(min(self.num_waypoints, len(empty_cells))):
            wx, wy = empty_cells[k]
            self.grid[wx, wy] = self.ENTITY_TYPES["waypoint"]
            self.waypoints.append([wx, wy])

    # ------------------------------------------------------------------
    # Background drift (env-caused transitions)
    # ------------------------------------------------------------------

    def _drift_hazards(self) -> None:
        """Move env-caused hazards randomly (independent of agent)."""
        for hazard in self.hazards:
            if self._rng.random() > self.env_drift_prob:
                continue

            hx, hy = hazard
            # Try up to 4 random moves
            candidates = list(self.ACTIONS.values())
            self._rng.shuffle(candidates)

            for dx, dy in candidates:
                nx, ny = hx + dx, hy + dy
                if (
                    0 < nx < self.size - 1
                    and 0 < ny < self.size - 1
                    and self.grid[nx, ny] == self.ENTITY_TYPES["empty"]
                ):
                    # Move hazard
                    self.grid[hx, hy] = self.ENTITY_TYPES["empty"]
                    self.grid[nx, ny] = self.ENTITY_TYPES["hazard"]
                    hazard[0] = nx
                    hazard[1] = ny
                    break

    def _maybe_respawn_resource(self) -> None:
        """Respawn a resource at a random empty cell (with low probability)."""
        if self._rng.random() > 0.3:
            return
        empty_cells = [
            (i, j)
            for i in range(1, self.size - 1)
            for j in range(1, self.size - 1)
            if self.grid[i, j] == self.ENTITY_TYPES["empty"]
        ]
        if empty_cells:
            rx, ry = empty_cells[int(self._rng.integers(0, len(empty_cells)))]
            self.grid[rx, ry] = self.ENTITY_TYPES["resource"]
            self.resources.append([rx, ry])

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> torch.Tensor:
        """Construct observation tensor from current state."""
        obs_parts = []

        # 1. Position one-hot
        position = torch.zeros(self.size * self.size)
        position[self.agent_x * self.size + self.agent_y] = 1.0
        obs_parts.append(position)

        # 2. Local entity-type view (5×5 one-hot)
        local_view = torch.zeros(5, 5, len(self.ENTITY_TYPES))
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni, nj = self.agent_x + di, self.agent_y + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    etype = self.grid[ni, nj]
                else:
                    etype = self.ENTITY_TYPES["wall"]
                local_view[di + 2, dj + 2, etype] = 1.0
        obs_parts.append(local_view.flatten())

        # 3. Homeostatic state
        obs_parts.append(torch.tensor([self.agent_health, self.agent_energy]))

        # 4. Contamination local view (5×5 float) — agent's causal footprint terrain
        cont_view = torch.zeros(5, 5)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni, nj = self.agent_x + di, self.agent_y + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    cont_view[di + 2, dj + 2] = float(self.contamination_grid[ni, nj])
        # Normalise by threshold so the value is in ~[0, 1+]
        obs_parts.append((cont_view / (self.contamination_threshold + 1e-6)).flatten())

        # 5. Footprint density at current cell (normalised)
        max_visits = max(1, self.footprint_grid.max())
        footprint_density = float(self.footprint_grid[self.agent_x, self.agent_y]) / max_visits
        obs_parts.append(torch.tensor([footprint_density]))

        return torch.cat(obs_parts).float()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_subgoal_state(self) -> dict:
        """Return current sub-goal sequence state (subgoal_mode only)."""
        return {
            "sequence_in_progress": self._sequence_in_progress,
            "sequence_step": self._sequence_step,
            "next_waypoint_idx": self._next_waypoint_idx,
            "sequences_completed": self._sequences_completed,
            "waypoints": [(w[0], w[1]) for w in self.waypoints if w[0] >= 0],
        }

    def get_contamination_map(self) -> np.ndarray:
        """Return the full contamination grid (agent-caused footprint)."""
        return self.contamination_grid.copy()

    def get_footprint_map(self) -> np.ndarray:
        """Return the full visit-count footprint grid."""
        return self.footprint_grid.copy()

    def get_agent_position(self) -> Tuple[int, int]:
        return (self.agent_x, self.agent_y)

    def get_hazard_positions(self) -> List[Tuple[int, int]]:
        return [(h[0], h[1]) for h in self.hazards]

    def render(self, mode: str = "text") -> Optional[str]:
        """ASCII render of the grid."""
        if mode != "text":
            return None

        symbols = {
            self.ENTITY_TYPES["empty"]: ".",
            self.ENTITY_TYPES["wall"]: "#",
            self.ENTITY_TYPES["resource"]: "R",
            self.ENTITY_TYPES["hazard"]: "X",
            self.ENTITY_TYPES["contaminated"]: "c",
            self.ENTITY_TYPES["agent"]: "A",
            self.ENTITY_TYPES["waypoint"]: "W",
        }

        lines = []
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                row += symbols.get(self.grid[i, j], "?")
            lines.append(row)

        lines.append(
            f"\nHealth: {self.agent_health:.2f} | Energy: {self.agent_energy:.2f} | "
            f"Steps: {self.steps}"
        )
        lines.append(
            f"Harm: {self.total_harm:.2f} | Benefit: {self.total_benefit:.2f} | "
            f"Max contamination: {self.contamination_grid.max():.2f}"
        )
        return "\n".join(lines)
