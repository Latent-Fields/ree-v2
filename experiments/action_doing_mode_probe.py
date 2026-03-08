"""
Action-Doing Mode Probe (MECH-025 / EVB-0025) — REE-v2

Tests MECH-025: cognitive_modes.action_doing_mode

MECH-025 asserts that the system enters a distinct 'action-doing mode' when
actively executing intentions with causal consequences.  In REE terms this
corresponds to elevated E3 precision modulation during steps where the agent's
action directly produces a causal outcome (transition_type == 'agent_caused_hazard'),
distinguishable from passive monitoring or environment-driven harm steps.

The key claim: the action-doing mode is a real, load-bearing mode — measurable
as a distinct precision signature around agent-caused events relative to
environment-caused or neutral steps.  If no such signature exists, E3 precision
is indifferent to whether the agent is "doing" vs. "monitoring", which would
falsify MECH-025.

CausalGridWorld enables a clean test: by varying num_hazards we control how
frequently agent actions produce agent_caused_hazard transitions, giving a
naturally varying density of action-doing-mode invocations across conditions.
In the HIGH_CAUSAL condition the agent frequently steps into hazard cells it
has mapped, providing dense action-doing feedback; in LOW_CAUSAL, hazard
encounters are rare and mostly environment-driven.

Conditions (2):
  HIGH_CAUSAL : num_hazards=8.  Agent actions frequently produce causal outcomes.
                Action-doing mode is invoked densely; should show a clear
                precision lift on agent_caused_hazard steps relative to
                env_caused or neutral steps.

  LOW_CAUSAL  : num_hazards=1.  Agent actions rarely produce causal outcomes.
                Action-doing mode rarely invoked; precision lift should be
                small or absent.

Key diagnostics:
  1. action_precision_lift:
       mean E3 precision on agent_caused_hazard steps
       minus mean E3 precision on non-agent-caused steps (env_caused + none).
     PASS: HIGH_CAUSAL lift > LOW_CAUSAL lift + LIFT_MARGIN
     (action-doing mode produces sharper precision modulation in high-agency environments)

  2. harm_reduction_slope: HIGH_CAUSAL > LOW_CAUSAL + SLOPE_MARGIN
     (action-doing mode enables faster harm-avoidance learning via sharp causal feedback)

Pass: >= 2 of 2 criteria met.

Usage:
    python experiments/action_doing_mode_probe.py
    python experiments/action_doing_mode_probe.py --episodes 5 --seeds 7

Claims:
    MECH-025: cognitive_modes.action_doing_mode
    EVB-0025
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

DEFAULT_EPISODES = 200
DEFAULT_MAX_STEPS = 100
DEFAULT_SEEDS = [7, 42, 99]
DEFAULT_GRID_SIZE = 10
MAX_GRAD_NORM = 1.0
E1_LR = 1e-4
POLICY_LR = 1e-3

# Condition parameters: num_hazards controls density of agent-caused events
CONDITION_HAZARDS: Dict[str, int] = {
    "HIGH_CAUSAL": 8,
    "LOW_CAUSAL": 1,
}

# ── Pass thresholds ───────────────────────────────────────────────────────────

LIFT_MARGIN: float = 0.05   # criterion 1: HIGH lift > LOW lift + margin
SLOPE_MARGIN: float = 0.05  # criterion 2: HIGH slope > LOW slope + margin


# ── Optimizer factory ─────────────────────────────────────────────────────────

def make_optimizers(
    agent: REEAgent,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    e1_params = (
        list(agent.e1.parameters())
        + list(agent.latent_stack.parameters())
        + list(agent.obs_encoder.parameters())
    )
    policy_params = list(agent.e3.parameters())
    e1_opt = torch.optim.Adam(e1_params, lr=E1_LR)
    policy_opt = torch.optim.Adam(policy_params, lr=POLICY_LR)
    return e1_opt, policy_opt


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode.

    Tracks E3 precision at each step categorised by the step's causal outcome:
      - agent_caused_hazard steps  (action-doing mode invocations)
      - all other steps            (env_caused_hazard, resource, none)

    The precision recorded is the value at action-selection time (before env.step),
    so it reflects the cognitive state active during action execution.
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    steps = 0

    # Precision tracking — sorted by causal outcome of the same step
    agent_caused_precisions: List[float] = []
    baseline_precisions: List[float] = []

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

        # Record E3 precision at action-selection time (action-doing state)
        current_prec = float(
            agent.e3.current_precision
            if hasattr(agent.e3, "current_precision")
            else 1.0
        )

        result = agent.e3.select(candidates)
        if result.log_prob is not None:
            log_probs.append(result.log_prob)

        action_idx = result.selected_action.argmax(dim=-1).item()
        next_obs, harm, done, info = env.step(action_idx)

        actual_harm = abs(harm) if harm < 0 else 0.0
        total_harm += actual_harm

        # Categorise this step's precision by causal outcome
        transition_type = info.get("transition_type", "none")
        if transition_type == "agent_caused_hazard":
            agent_caused_precisions.append(current_prec)
        else:
            baseline_precisions.append(current_prec)

        agent.update_residue(harm)

        # ── E1 update ──
        e1_loss = agent.compute_prediction_loss()
        if e1_loss.requires_grad:
            e1_opt.zero_grad()
            e1_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for grp in e1_opt.param_groups for p in grp["params"]],
                MAX_GRAD_NORM,
            )
            e1_opt.step()

        obs = next_obs
        steps += 1
        if done:
            break

    # ── REINFORCE policy update ──
    policy_loss_val = 0.0
    if log_probs:
        G = float(-total_harm)
        policy_loss = -(torch.stack(log_probs) * G).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in policy_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        policy_opt.step()
        policy_loss_val = policy_loss.item()

    # Action precision lift for this episode
    if agent_caused_precisions and baseline_precisions:
        action_precision_lift = (
            statistics.mean(agent_caused_precisions)
            - statistics.mean(baseline_precisions)
        )
    else:
        action_precision_lift = 0.0

    return {
        "total_harm": total_harm,
        "steps": steps,
        "policy_loss": policy_loss_val,
        "action_precision_lift": action_precision_lift,
        "agent_caused_steps": len(agent_caused_precisions),
        "baseline_steps": len(baseline_precisions),
    }


# ── Condition runner ──────────────────────────────────────────────────────────

def run_condition(
    seed: int,
    condition: str,
    num_episodes: int,
    max_steps: int,
    grid_size: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    num_hazards = CONDITION_HAZARDS[condition]
    env = CausalGridWorld(size=grid_size, num_hazards=num_hazards)
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)
    e1_opt, policy_opt = make_optimizers(agent)

    ep_harms: List[float] = []
    ep_lifts: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_lifts.append(metrics["action_precision_lift"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            recent_lift = statistics.mean(ep_lifts[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  seed={seed}  "
                f"cond={condition}  harm={recent_harm:.3f}  lift={recent_lift:+.3f}"
            )

    quarter = max(1, num_episodes // 4)
    first_q = statistics.mean(ep_harms[:quarter])
    last_q = statistics.mean(ep_harms[-quarter:])
    slope = (first_q - last_q) / max(1e-6, first_q)

    mean_lift = statistics.mean(ep_lifts) if ep_lifts else 0.0
    last_q_lift = statistics.mean(ep_lifts[-quarter:]) if ep_lifts else 0.0

    return {
        "condition": condition,
        "seed": seed,
        "num_hazards": num_hazards,
        "first_quarter_harm": round(first_q, 4),
        "last_quarter_harm": round(last_q, 4),
        "harm_reduction_slope": round(slope, 4),
        "mean_action_precision_lift": round(mean_lift, 4),
        "last_quarter_precision_lift": round(last_q_lift, 4),
        "episode_count": num_episodes,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(
    num_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seeds: Optional[List[int]] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print("[Action-Doing Mode Probe — MECH-025 / EVB-0025] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print(
            f"    HIGH_CAUSAL : num_hazards={CONDITION_HAZARDS['HIGH_CAUSAL']}"
            "  (dense agent-caused outcomes — action-doing mode frequent)"
        )
        print(
            f"    LOW_CAUSAL  : num_hazards={CONDITION_HAZARDS['LOW_CAUSAL']}"
            "  (sparse agent-caused outcomes — action-doing mode rare)"
        )
        print()
        print("  Diagnostics:")
        print(f"    1. HIGH precision lift > LOW lift + {LIFT_MARGIN}")
        print(f"    2. HIGH slope > LOW slope + {SLOPE_MARGIN}")
        print()

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for condition in ["HIGH_CAUSAL", "LOW_CAUSAL"]:
            if verbose:
                print(f"  Seed {seed}  Condition {condition}")
            result = run_condition(
                seed=seed,
                condition=condition,
                num_episodes=num_episodes,
                max_steps=max_steps,
                grid_size=grid_size,
                verbose=verbose,
            )
            all_results.append(result)
            if verbose:
                print(
                    f"    harm {result['first_quarter_harm']:.3f} -> "
                    f"{result['last_quarter_harm']:.3f}  "
                    f"slope={result['harm_reduction_slope']:+.3f}  "
                    f"lift={result['mean_action_precision_lift']:+.3f}"
                )
                print()

    high = [r for r in all_results if r["condition"] == "HIGH_CAUSAL"]
    low = [r for r in all_results if r["condition"] == "LOW_CAUSAL"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    high_lift = _agg(high, "mean_action_precision_lift")
    low_lift = _agg(low, "mean_action_precision_lift")
    high_slope = _agg(high, "harm_reduction_slope")
    low_slope = _agg(low, "harm_reduction_slope")

    crit_1 = high_lift > low_lift + LIFT_MARGIN
    crit_2 = high_slope > low_slope + SLOPE_MARGIN

    num_met = sum([crit_1, crit_2])
    verdict = "PASS" if num_met >= 2 else "FAIL"
    partial = num_met == 1

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  HIGH_CAUSAL  lift={high_lift:+.3f}  slope={high_slope:+.3f}")
        print(f"  LOW_CAUSAL   lift={low_lift:+.3f}  slope={low_slope:+.3f}")
        print()
        print(
            f"  Crit 1  lift  HIGH={high_lift:+.3f} > "
            f"LOW+{LIFT_MARGIN}={low_lift+LIFT_MARGIN:+.3f}  "
            f"{'MET' if crit_1 else 'MISSED'}"
        )
        print(
            f"  Crit 2  slope HIGH={high_slope:+.3f} > "
            f"LOW+{SLOPE_MARGIN}={low_slope+SLOPE_MARGIN:+.3f}  "
            f"{'MET' if crit_2 else 'MISSED'}"
        )
        print()
        print(f"  Criteria met: {num_met}/2  ->  MECH-025 verdict: {verdict}")
        if partial:
            print("  (partial -- 1 of 2 criteria met)")
        print()

    result_doc: Dict[str, Any] = {
        "experiment": "action_doing_mode_probe",
        "claim": "MECH-025",
        "evb_id": "EVB-0025",
        "substrate": "ree-v2",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "condition_hazards": CONDITION_HAZARDS,
            "environment": "CausalGridWorld",
            "e1_lr": E1_LR,
            "policy_lr": POLICY_LR,
            "lift_margin": LIFT_MARGIN,
            "slope_margin": SLOPE_MARGIN,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "high_causal_precision_lift": high_lift,
            "low_causal_precision_lift": low_lift,
            "high_causal_slope": high_slope,
            "low_causal_slope": low_slope,
            "criterion_1_lift_met": crit_1,
            "criterion_2_slope_met": crit_2,
            "criteria_met": num_met,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "action_doing_mode_probe"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"action_doing_mode_probe_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MECH-025: Action-Doing Mode Probe (REE-v2)"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seeds=args.seeds,
        grid_size=args.grid_size,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
