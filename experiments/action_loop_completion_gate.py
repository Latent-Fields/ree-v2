"""
EXQ-020: MECH-057a Action-Loop Completion Gate (V2)

Tests whether the action-loop completion gate is load-bearing in a multi-step
committed action sequence substrate. This is the first genuine test where the
substrate limitation that caused EXQ-004, EXQ-007, and the V2 re-run to fail
has been removed: CausalGridWorld is extended with a subgoal_mode that creates
real 3-step waypoint sequences. The gate now has an "action in progress" state
to protect.

Substrate
---------
CausalGridWorld(subgoal_mode=True, num_waypoints=3):
  - 3 waypoint cells placed at reset; must be visited in order (W0 → W1 → W2).
  - Visiting W0 starts a committed sequence (sequence_in_progress=True).
  - Visiting W1 and W2 in order earns intermediate + completion rewards.
  - If num_waypoints steps elapse without hitting the next waypoint, the
    sequence resets (sequence_commitment_timeout=20).
  - Hazards remain active throughout — real interruption pressure.
  - info['sequence_in_progress'] (bool) and info['sequence_step'] (int) are
    exposed every step and piped to the agent.

Conditions (3)
--------------
  FULL             gate=True   attribution=True
  NO_GATE          gate=False  attribution=True
  NO_ATTRIBUTION   gate=True   attribution=False

Gate (action_loop_gate_enabled):
  True  — REEAgent.generate_trajectories() returns cached candidates when
          sequence_in_progress=True; HippocampalModule is suppressed
          mid-sequence. Normal proposals resume when sequence ends.
  False — HippocampalModule generates fresh candidates every step,
          including mid-sequence (may interrupt or redirect execution).

Attribution:
  True  — Policy signal G = -harm_during_sequence_in_progress.
          Residue updates only while sequence_in_progress=True (owned=True).
          The agent learns specifically from harms its committed path caused.
  False — Policy signal G = -total_harm (all harm, regardless of sequence).
          Residue updates for ALL harm events (owned=True always).
          The learning signal is undifferentiated between sequence and
          background harm.

Pass criteria (BOTH must hold for full PASS):
  1. harm[NO_GATE]        / harm[FULL] >= 1.10   (gate removes ≥10% harm)
  2. harm[NO_ATTRIBUTION] / harm[FULL] >= 1.05   (attribution removes ≥5% harm)

If only criterion 1 passes: partial PASS (gate is load-bearing, attribution
is not — informative given multi-step substrate is finally correct).
If neither passes: informative FAIL — stronger negative signal than V1 because
the substrate limitation has now been removed.

Primary metric: mean harm in the final quartile of evaluation episodes
(last 25% of total episodes per condition per seed).

Usage:
    /opt/local/bin/python3 experiments/action_loop_completion_gate.py
    /opt/local/bin/python3 experiments/action_loop_completion_gate.py \\
        --episodes 50 --seeds 7 42

Claims:
    MECH-057a: agentic_extension.action_loop_completion_gate
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_EPISODES = 250         # 200 warmup + 50 eval (last quarter = 62 episodes)
DEFAULT_MAX_STEPS = 100
DEFAULT_SEEDS = [7, 42, 99, 13, 77]   # 5 seeds as required by EXQ-020
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 4        # Same as attribution_completion_gating.py
DEFAULT_NUM_WAYPOINTS = 3
DEFAULT_WAYPOINT_VISIT_REWARD = 0.2
DEFAULT_WAYPOINT_COMPLETION_REWARD = 0.8
DEFAULT_SEQUENCE_COMMITMENT_TIMEOUT = 20

MAX_GRAD_NORM = 1.0
E1_LR = 1e-4
POLICY_LR = 1e-3
E2_LR = 1e-3

# ── Pass thresholds ───────────────────────────────────────────────────────────

# Criterion 1: gate removal must degrade harm avoidance by ≥10%
GATE_HARM_RATIO_THRESHOLD: float = 1.10
# Criterion 2: attribution removal must show measurable effect ≥5%
ATTRIBUTION_HARM_RATIO_THRESHOLD: float = 1.05


# ── Optimizers ────────────────────────────────────────────────────────────────

def make_optimizers(
    agent: REEAgent,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
    """Build E1 world-model, E3 policy, and E2 transition-model optimizers."""
    e1_params = (
        list(agent.e1.parameters())
        + list(agent.latent_stack.parameters())
        + list(agent.obs_encoder.parameters())
    )
    policy_params = list(agent.e3.parameters())
    e1_opt = torch.optim.Adam(e1_params, lr=E1_LR)
    policy_opt = torch.optim.Adam(policy_params, lr=POLICY_LR)
    e2_opt = torch.optim.Adam(list(agent.e2.parameters()), lr=E2_LR)
    return e1_opt, policy_opt, e2_opt


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    e2_opt: torch.optim.Optimizer,
    attribution_enabled: bool,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode and return per-episode metrics.

    Gate behaviour is determined by agent.config.action_loop_gate_enabled.
    Attribution behaviour is controlled by attribution_enabled:

      attribution=True  : G = -harm_during_sequence; residue only during sequence.
      attribution=False : G = -total_harm;           residue for all harm.

    E1 and E2 world-model updates are identical across all conditions.
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    sequence_harm = 0.0   # Harm that occurred while sequence_in_progress=True
    sequence_in_progress = False

    prev_latent_z: Optional[torch.Tensor] = None
    prev_action_tensor: Optional[torch.Tensor] = None
    sequences_started = 0
    sequences_completed = 0

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            # Pass sequence_in_progress for MECH-057a gate routing
            candidates = agent.generate_trajectories(
                agent._current_latent,
                sequence_in_progress=sequence_in_progress,
            )

        # Capture z_t; record E2 transition
        z_t = agent._current_latent.z_gamma.detach().clone()
        if prev_latent_z is not None and prev_action_tensor is not None:
            agent.record_transition(prev_latent_z, prev_action_tensor, z_t)

        result = agent.e3.select(candidates)
        if result.log_prob is not None:
            log_probs.append(result.log_prob)

        action_tensor = result.selected_action.detach().clone()
        action_idx = action_tensor.argmax(dim=-1).item()
        next_obs, harm, done, info = env.step(action_idx)

        # Sequence state from environment
        sequence_in_progress = info.get("sequence_in_progress", False)
        tt = info.get("transition_type", "none")
        if tt == "waypoint" or tt == "sequence_complete":
            if tt == "waypoint" and info.get("sequence_step", 0) == 0:
                sequences_started += 1
            if tt == "sequence_complete":
                sequences_completed += 1

        actual_harm = abs(harm) if harm < 0 else 0.0
        total_harm += actual_harm
        if sequence_in_progress or tt in ("sequence_complete",):
            sequence_harm += actual_harm

        # Residue update: conditioned on sequence ownership for attribution=True
        # With attribution enabled: only accumulate when sequence_in_progress
        # With attribution disabled: accumulate for all harm (owned=True always)
        if attribution_enabled:
            agent.update_residue(harm, owned=sequence_in_progress)
        else:
            agent.update_residue(harm, owned=True)

        prev_latent_z = z_t
        prev_action_tensor = action_tensor
        obs = next_obs

        if done:
            break

    # ── Condition-specific return signal ──────────────────────────────────────
    if attribution_enabled:
        G = float(-sequence_harm)     # Only sequence-attributed harm
    else:
        G = float(-total_harm)         # All harm equally

    # ── REINFORCE policy update ───────────────────────────────────────────────
    policy_loss_val = 0.0
    if log_probs:
        policy_loss = -(torch.stack(log_probs) * G).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in policy_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        policy_opt.step()
        policy_loss_val = policy_loss.item()

    # ── E1 world-model update (same across all conditions) ───────────────────
    e1_loss_val = 0.0
    e1_loss = agent.compute_prediction_loss()
    if e1_loss.requires_grad:
        e1_opt.zero_grad()
        e1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in e1_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        e1_opt.step()
        e1_loss_val = e1_loss.item()

    # ── E2 motor-sensory update (same across all conditions) ─────────────────
    e2_loss = agent.compute_e2_loss()
    if e2_loss.requires_grad:
        e2_opt.zero_grad()
        e2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in e2_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        e2_opt.step()

    return {
        "total_harm": total_harm,
        "sequence_harm": sequence_harm,
        "sequences_started": sequences_started,
        "sequences_completed": sequences_completed,
        "e1_loss": e1_loss_val,
        "policy_loss": policy_loss_val,
    }


# ── Condition runner ──────────────────────────────────────────────────────────

def run_condition(
    seed: int,
    condition: str,
    num_episodes: int,
    max_steps: int,
    grid_size: int,
    num_hazards: int,
    num_waypoints: int,
    waypoint_visit_reward: float,
    waypoint_completion_reward: float,
    sequence_commitment_timeout: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one condition (FULL / NO_GATE / NO_ATTRIBUTION) for a single seed."""
    torch.manual_seed(seed)

    gate_enabled = condition in ("FULL", "NO_ATTRIBUTION")
    attribution_enabled = condition in ("FULL", "NO_GATE")

    env = CausalGridWorld(
        size=grid_size,
        num_hazards=num_hazards,
        subgoal_mode=True,
        num_waypoints=num_waypoints,
        waypoint_visit_reward=waypoint_visit_reward,
        waypoint_completion_reward=waypoint_completion_reward,
        sequence_commitment_timeout=sequence_commitment_timeout,
    )

    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    config.action_loop_gate_enabled = gate_enabled

    agent = REEAgent(config=config)
    e1_opt, policy_opt, e2_opt = make_optimizers(agent)

    ep_total_harms: List[float] = []
    ep_sequence_harms: List[float] = []
    ep_sequences_completed: List[int] = []

    for ep in range(num_episodes):
        metrics = run_episode(
            agent, env, e1_opt, policy_opt, e2_opt,
            attribution_enabled=attribution_enabled,
            max_steps=max_steps,
        )
        ep_total_harms.append(metrics["total_harm"])
        ep_sequence_harms.append(metrics["sequence_harm"])
        ep_sequences_completed.append(metrics["sequences_completed"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_total_harms[-20:])
            recent_seq = statistics.mean(ep_sequences_completed[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  "
                f"seed={seed}  cond={condition}  "
                f"harm={recent_harm:.3f}  seq_completed={recent_seq:.2f}"
            )

    quarter = max(1, num_episodes // 4)
    return {
        "condition": condition,
        "seed": seed,
        "gate_enabled": gate_enabled,
        "attribution_enabled": attribution_enabled,
        "first_quarter_total_harm": round(statistics.mean(ep_total_harms[:quarter]), 4),
        "last_quarter_total_harm": round(statistics.mean(ep_total_harms[-quarter:]), 4),
        "first_quarter_sequence_harm": round(statistics.mean(ep_sequence_harms[:quarter]), 4),
        "last_quarter_sequence_harm": round(statistics.mean(ep_sequence_harms[-quarter:]), 4),
        "mean_sequences_completed": round(statistics.mean(ep_sequences_completed), 4),
        "last_quarter_sequences_completed": round(
            statistics.mean(ep_sequences_completed[-quarter:]), 4
        ),
        "episode_count": num_episodes,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(
    num_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seeds: Optional[List[int]] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_hazards: int = DEFAULT_NUM_HAZARDS,
    num_waypoints: int = DEFAULT_NUM_WAYPOINTS,
    waypoint_visit_reward: float = DEFAULT_WAYPOINT_VISIT_REWARD,
    waypoint_completion_reward: float = DEFAULT_WAYPOINT_COMPLETION_REWARD,
    sequence_commitment_timeout: int = DEFAULT_SEQUENCE_COMMITMENT_TIMEOUT,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print("[EXQ-020: MECH-057a Action-Loop Completion Gate] (REE-v2)")
        print(
            f"  CausalGridWorld (subgoal_mode=True): {grid_size}×{grid_size}, "
            f"{num_hazards} hazards, {num_waypoints} waypoints"
        )
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print("    FULL            gate=True   attribution=True")
        print("    NO_GATE         gate=False  attribution=True")
        print("    NO_ATTRIBUTION  gate=True   attribution=False")
        print()
        print("  Pass criteria (primary metric: last-Q total harm):")
        print(
            f"    1. harm[NO_GATE]        / harm[FULL] >= {GATE_HARM_RATIO_THRESHOLD:.2f}"
            "  (gate is load-bearing)"
        )
        print(
            f"    2. harm[NO_ATTRIBUTION] / harm[FULL] >= {ATTRIBUTION_HARM_RATIO_THRESHOLD:.2f}"
            "  (attribution is load-bearing)"
        )
        print()

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for condition in ["FULL", "NO_GATE", "NO_ATTRIBUTION"]:
            if verbose:
                print(f"  Seed {seed}  Condition {condition}")
            result = run_condition(
                seed=seed,
                condition=condition,
                num_episodes=num_episodes,
                max_steps=max_steps,
                grid_size=grid_size,
                num_hazards=num_hazards,
                num_waypoints=num_waypoints,
                waypoint_visit_reward=waypoint_visit_reward,
                waypoint_completion_reward=waypoint_completion_reward,
                sequence_commitment_timeout=sequence_commitment_timeout,
                verbose=verbose,
            )
            all_results.append(result)
            if verbose:
                print(
                    f"    harm {result['first_quarter_total_harm']:.3f} → "
                    f"{result['last_quarter_total_harm']:.3f}  "
                    f"seq_completed_last_Q={result['last_quarter_sequences_completed']:.2f}"
                )
                print()

    # ── Aggregate by condition ────────────────────────────────────────────────
    full_r = [r for r in all_results if r["condition"] == "FULL"]
    no_gate_r = [r for r in all_results if r["condition"] == "NO_GATE"]
    no_attr_r = [r for r in all_results if r["condition"] == "NO_ATTRIBUTION"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    full_harm = _agg(full_r, "last_quarter_total_harm")
    no_gate_harm = _agg(no_gate_r, "last_quarter_total_harm")
    no_attr_harm = _agg(no_attr_r, "last_quarter_total_harm")

    crit_1_ratio = round(no_gate_harm / full_harm, 4) if full_harm > 0 else 1.0
    crit_2_ratio = round(no_attr_harm / full_harm, 4) if full_harm > 0 else 1.0

    crit_1 = crit_1_ratio >= GATE_HARM_RATIO_THRESHOLD
    crit_2 = crit_2_ratio >= ATTRIBUTION_HARM_RATIO_THRESHOLD

    num_met = sum([crit_1, crit_2])
    if num_met == 2:
        verdict = "PASS"
        partial = False
    elif num_met == 1:
        verdict = "PARTIAL_PASS"
        partial = True
    else:
        verdict = "FAIL"
        partial = False

    if verbose:
        print("=" * 70)
        print("[EXQ-020 Summary]")
        print(f"  FULL            harm (last-Q): {full_harm:.4f}")
        print(f"  NO_GATE         harm (last-Q): {no_gate_harm:.4f}")
        print(f"  NO_ATTRIBUTION  harm (last-Q): {no_attr_harm:.4f}")
        print()
        print(
            f"  Crit 1  NO_GATE/FULL = {crit_1_ratio:.3f}  "
            f"(threshold {GATE_HARM_RATIO_THRESHOLD})  "
            f"{'PASS' if crit_1 else 'FAIL'}"
        )
        print(
            f"  Crit 2  NO_ATTR/FULL = {crit_2_ratio:.3f}  "
            f"(threshold {ATTRIBUTION_HARM_RATIO_THRESHOLD})  "
            f"{'PASS' if crit_2 else 'FAIL'}"
        )
        print()
        print(f"  Criteria met: {num_met}/2  →  MECH-057a verdict: {verdict}")
        if partial:
            print("  (partial PASS — gate load-bearing but attribution not)")
        print()
        if verdict in ("PASS", "PARTIAL_PASS"):
            print("  Interpretation:")
            if crit_1:
                print("    Gate is load-bearing: removing mid-sequence replanning")
                print("    increases harm by ≥10%. MECH-057a action-loop gate confirmed.")
            if crit_2:
                print("    Attribution is load-bearing: undifferentiated return signal")
                print("    degrades harm avoidance. Ownership-conditioned residue confirmed.")
        else:
            print("  Interpretation:")
            print("    Neither criterion met on multi-step substrate.")
            print("    This is a STRONGER negative signal than V1 FAILs (substrate")
            print("    limitation removed). Gate mechanism may not be load-bearing")
            print("    in CausalGridWorld even with waypoint sequences.")

    result_doc: Dict[str, Any] = {
        "experiment": "action_loop_completion_gate",
        "claim": "MECH-057a",
        "substrate": "ree-v2",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "num_waypoints": num_waypoints,
            "waypoint_visit_reward": waypoint_visit_reward,
            "waypoint_completion_reward": waypoint_completion_reward,
            "sequence_commitment_timeout": sequence_commitment_timeout,
            "environment": "CausalGridWorld(subgoal_mode=True)",
            "e1_lr": E1_LR,
            "policy_lr": POLICY_LR,
            "e2_lr": E2_LR,
            "gate_harm_ratio_threshold": GATE_HARM_RATIO_THRESHOLD,
            "attribution_harm_ratio_threshold": ATTRIBUTION_HARM_RATIO_THRESHOLD,
        },
        "verdict": verdict,
        "partial_pass": partial,
        "aggregate": {
            "full_harm_last_quarter": full_harm,
            "no_gate_harm_last_quarter": no_gate_harm,
            "no_attribution_harm_last_quarter": no_attr_harm,
            "no_gate_vs_full_ratio": crit_1_ratio,
            "no_attribution_vs_full_ratio": crit_2_ratio,
            "criterion_1_gate_met": crit_1,
            "criterion_2_attribution_met": crit_2,
            "criteria_met": num_met,
        },
        "per_run": all_results,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "action_loop_completion_gate"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"action_loop_completion_gate_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EXQ-020: MECH-057a Action-Loop Completion Gate (REE-v2 subgoal substrate)"
        )
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--num-hazards", type=int, default=DEFAULT_NUM_HAZARDS)
    parser.add_argument("--num-waypoints", type=int, default=DEFAULT_NUM_WAYPOINTS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seeds=args.seeds,
        grid_size=args.grid_size,
        num_hazards=args.num_hazards,
        num_waypoints=args.num_waypoints,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
