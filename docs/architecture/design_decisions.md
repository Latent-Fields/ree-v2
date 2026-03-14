# Design Decisions Log

Sub-design decisions (SDs) that affect the REE-v2 architecture. Each entry records the problem, resolution, and status.

---

## SD-001 — E2/HippocampalModule Separation
**Status: RESOLVED in V2**

**Problem:** In V1, `E2FastPredictor.generate_candidates_cem()` was performing hippocampal work — iterative CEM-based trajectory search guided by harm scores. This conflated two distinct responsibilities: (a) forward transition modelling and (b) terrain-guided search.

**Resolution:** E2 is now a pure transition model `f(z_t, a_t) → z_{t+1}`. All CEM-style trajectory search has moved to `HippocampalModule`, which uses E2 only for forward rollouts. E2 is never called directly from the agent loop.

**Consequence (SD-003):** Making E2 a pure, independently-callable model unlocks `forward_counterfactual()` for self-attribution experiments. The SD-001 conflation made this impossible.

---

## SD-002 — E1→HippocampalModule Mutual Constitution
**Status: RESOLVED in V2 (2026-03-06)**

**Problem:** HippocampalModule was initialising its terrain search from the residue field alone, ignoring the long-horizon associative context held by E1. The E1 prior over likely latent trajectories was unused in trajectory proposal.

**Resolution:** E1's associative prior is now passed into `HippocampalModule.propose_trajectories()` as `e1_prior`. The `terrain_prior` network takes `(z_beta, e1_prior, residue_val)` as input, conditioning the initial action distribution on both the affective terrain and E1's long-horizon context. E1 shapes the prior over *where* to search; E2 evaluates the trajectories within that region.

See `agent.py:generate_trajectories()` and `hippocampal/module.py:_get_terrain_action_mean()`.

---

## SD-003 — Self-Attribution via Counterfactual E2
**Status: OPEN — substrate ready, experiments pending**

**Problem:** The agent needs a mechanism to distinguish agent-caused harm from environment-caused harm. Without this, the residue field accumulates harm regardless of whether the agent's action was causally responsible.

**Substrate:** `E2FastPredictor.forward_counterfactual(z, a_cf)` is exposed as a pure query. Causal attribution is computed as:

```python
z_actual_next = e2.predict_next_state(z, a_actual)
z_cf_next     = e2.forward_counterfactual(z, a_cf)
causal_delta  = z_actual_next - z_cf_next  # agent's causal signature
```

Transitions where `causal_delta ≈ 0` for any `a_cf` are environment-caused. The magnitude of `causal_delta` indexes the agent's causal contribution.

**Next steps:** Design and run self-attribution experiments using `CausalGridWorld`, which tracks `transition_type` (agent-caused vs. environment-caused) as ground truth for calibrating `causal_delta`.

---

## SD-004 — Action Objects as Hippocampal Map Backbone
**Status: HELD FOR V3**

**Problem:** The hippocampal map currently navigates raw `z_gamma` state space. This limits planning horizon and requires HippocampalModule to operate over a high-dimensional representation at every CEM step.

**Insight:** E2 can produce **action objects** — compressed latent representations of the transformation `(z_t → z_{t+1})` under action `a_t` — as a bottleneck inside its transition model. Action objects capture *what this action does* in latent space, not just the outcome state. Because many distinct states afford the same abstract transformation, the action-object space is fundamentally more compact than state space.

**Target design:** E2 becomes `f(z_t, a_t) → (z_{t+1}, o_t)` where `o_t` is the action object. HippocampalModule builds and navigates its map over action-object space rather than `z_gamma`. This enables:
- Map compaction: nodes are action affordances, not individual states
- E2 hidden layer compression: the bottleneck reduces internal dimensionality
- Much longer hippocampal rollout horizons: CEM operates in compressed object space; plans are action-object sequences unfolded to actual actions at execution time

**Why held:** Requires co-redesigning `E2FastPredictor`, `HippocampalModule`, `Trajectory`, `E2Config`, and `HippocampalConfig`. Not safe to implement incrementally mid-experiment series.

**Full design spec:** `docs/architecture/e2.md §5`

---

## SD-005 — Self/World Latent Split
**Status: HELD FOR V3**

**Problem:** V2's `z_gamma` (conceptual sensorium) conflates two ontologically distinct kinds
of state change caused by action:

1. **Self-directed effects** — what happens to the agent's OWN body: proprioceptive
   (position, movement), interoceptive (energy, internal state). These are immediate and
   direct. E2 should predict these.
2. **World-directed effects** — what happens to the EXTERNAL environment: contamination
   left behind, objects displaced, hazards created. These can be delayed and have causal
   chains. E3 / Hippocampus plans over these. Residue should track these.

Currently E2 operates on a single z_gamma that mixes both. The causal_delta
`||E2(z, a_actual) - E2(z, a_cf)||` captures both self-state change and world-state change
in one number, making moral attribution ambiguous: "my energy dropped" and "I contaminated
that cell" produce indistinguishable causal deltas.

**Resolution (V3):** Split z_gamma into:
- `z_self` — proprioceptive + interoceptive. E2 operates here. Motor-sensory error =
  prediction error on `z_self_{t+1}`. Body-state model. Low moral weight.
- `z_world` — exteroceptive world model. E3 / Hippocampus plans over this. Harm/goal
  error and residue live here. High moral weight.

The causal signature for self-attribution then decomposes:
```
self_delta  = ||E2(z_self, a_actual) - E2(z_self, a_cf)||   # body state change
world_delta = ||planned_world_delta(a_actual) - planned_world_delta(a_cf)||  # world change
```
Residue accumulates on `world_delta`, not `self_delta`.

**Relation to SD-004:** Action objects (SD-004) encode `(z_world_t → z_world_{t+1})` under
action `a_t` — exactly the world-directed causal footprint. SD-005 and SD-004 co-evolve.

**Relation to MECH-069:** The self/world split clarifies why the three error signals are
incommensurable: self-state prediction error (E2's domain) and world-consequence harm error
(E3's domain) differ not just in learning rate but in ontological category.

**Why held:** Requires redesigning the observation encoder (routing sensory channels to
z_self vs z_world), splitting z_gamma, re-routing E2 to z_self, and updating all downstream
consumers (HippocampalModule, ResidueField, E3). Not safe to implement mid-experiment series.

**Full design rationale:** `docs/thoughts/2026-03-14_self_world_latent_split_sd003_limitation.md`
