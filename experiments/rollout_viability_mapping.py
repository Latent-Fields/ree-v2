"""
Rollout Viability Mapping Experiment (ARC-018 / EVB-0017) — REE-v2

Tests ARC-018: hippocampus.rollout_viability_mapping

ARC-018 asserts that after action execution the system updates a viability
map of action-space using predicted vs observed outcome mismatch.  In REE
terms this corresponds to E1 learning from compute_prediction_loss() after
each committed step: the world-model absorbs prediction-reality mismatch and
builds an internal map of which trajectory sequences are stable, fragile, or
path-closing.

CausalGridWorld enables a clean test: E1's viability map should learn that
contaminated cells are reliably harmful (stable region of the harm map), while
drifting env hazards are less predictable.  An agent whose E1 actively updates
(VIABILITY_MAPPED) should progressively navigate to lower-harm regions;  an
agent whose E1 is frozen (VIABILITY_FIXED) has no viability map and must rely
on policy gradient alone.

Conditions (2):
  VIABILITY_MAPPED : E1 world-model updates every step from prediction loss.
                     Viability map grows over the episode.
  VIABILITY_FIXED  : E1 is frozen (e1_opt.step() never called).
                     Policy gradient still runs; world model does not evolve.

Key diagnostics:
  1. harm_reduction_slope: (first_quarter_harm - last_quarter_harm) / first_quarter_harm
     Positive = harm reduced over episode.
     VIABILITY_MAPPED harm_reduction_slope > VIABILITY_FIXED harm_reduction_slope + 0.05

  2. last_quarter_harm: VIABILITY_MAPPED < VIABILITY_FIXED * 0.95

Pass: >= 2 of 2 criteria met (both must hold for clean PASS; partial if 1 of 2).

Usage:
    python experiments/rollout_viability_mapping.py
    python experiments/rollout_viability_mapping.py --episodes 5 --seeds 7

Claims:
    ARC-018: hippocampus.rollout_viability_mapping
    EVB-0017
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
DEFAULT_NUM_HAZARDS = 4
MAX_GRAD_NORM = 1.0
E1_LR = 1e-4
POLICY_LR = 1e-3
E2_LR = 1e-3

# ── Pass thresholds ───────────────────────────────────────────────────────────

SLOPE_MARGIN: float = 0.05   # criterion 1: MAPPED slope > FIXED slope + margin
HARM_FACTOR: float = 0.95    # criterion 2: MAPPED last-Q harm < FIXED * factor


# ── Optimizer factory ─────────────────────────────────────────────────────────

def make_optimizers(
    agent: REEAgent,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
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
    condition: str,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode.

    VIABILITY_MAPPED: E1 world-model updates every step — builds viability map.
    VIABILITY_FIXED:  E1 frozen — policy gradient runs but world model static.
    E2 trains in both conditions.
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    e1_losses: List[float] = []
    steps = 0

    prev_latent_z: Optional[torch.Tensor] = None
    prev_action_tensor: Optional[torch.Tensor] = None

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

        # Capture z_t; record E2 transition (z_{t-1}, a_{t-1}, z_t)
        z_t = agent._current_latent.z_gamma.detach().clone()
        if prev_latent_z is not None and prev_action_tensor is not None:
            agent.record_transition(prev_latent_z, prev_action_tensor, z_t)

        result = agent.e3.select(candidates)
        if result.log_prob is not None:
            log_probs.append(result.log_prob)

        action_tensor = result.selected_action.detach().clone()
        action_idx = action_tensor.argmax(dim=-1).item()
        next_obs, harm, done, _info = env.step(action_idx)

        actual_harm = abs(harm) if harm < 0 else 0.0
        total_harm += actual_harm

        agent.update_residue(harm)

        # ── E1 update (condition-specific) ──
        e1_loss = agent.compute_prediction_loss()
        e1_loss_val = e1_loss.item() if e1_loss.requires_grad else 0.0

        if condition == "VIABILITY_MAPPED" and e1_loss.requires_grad:
            e1_opt.zero_grad()
            e1_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for grp in e1_opt.param_groups for p in grp["params"]],
                MAX_GRAD_NORM,
            )
            e1_opt.step()
        # VIABILITY_FIXED: skip e1_opt.step() — world model stays frozen

        # ── E2 update (both conditions) ──
        e2_loss = agent.compute_e2_loss()
        if e2_loss.requires_grad:
            e2_opt.zero_grad()
            e2_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for grp in e2_opt.param_groups for p in grp["params"]],
                MAX_GRAD_NORM,
            )
            e2_opt.step()

        e1_losses.append(e1_loss_val)
        prev_latent_z = z_t
        prev_action_tensor = action_tensor
        obs = next_obs
        steps += 1
        if done:
            break

    # ── REINFORCE policy update (both conditions) ──
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

    mean_e1_loss = statistics.mean(e1_losses) if e1_losses else 0.0

    return {
        "total_harm": total_harm,
        "steps": steps,
        "mean_e1_loss": mean_e1_loss,
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
    verbose: bool = True,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    env = CausalGridWorld(size=grid_size, num_hazards=num_hazards)
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)
    e1_opt, policy_opt, e2_opt = make_optimizers(agent)

    ep_harms: List[float] = []
    ep_e1_losses: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, e2_opt, condition, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_e1_losses.append(metrics["mean_e1_loss"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  seed={seed}  "
                f"cond={condition}  harm={recent_harm:.3f}"
            )

    quarter = max(1, num_episodes // 4)
    first_q = statistics.mean(ep_harms[:quarter])
    last_q = statistics.mean(ep_harms[-quarter:])
    harm_reduction_slope = (first_q - last_q) / max(1e-6, first_q)

    return {
        "condition": condition,
        "seed": seed,
        "first_quarter_harm": round(first_q, 4),
        "last_quarter_harm": round(last_q, 4),
        "harm_reduction_slope": round(harm_reduction_slope, 4),
        "mean_e1_loss": round(statistics.mean(ep_e1_losses), 6),
        "last_quarter_e1_loss": round(statistics.mean(ep_e1_losses[-quarter:]), 6),
        "episode_count": num_episodes,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(
    num_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seeds: Optional[List[int]] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_hazards: int = DEFAULT_NUM_HAZARDS,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print("[Rollout Viability Mapping — ARC-018 / EVB-0017] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print("    VIABILITY_MAPPED : E1 updates from prediction loss (viability map learns)")
        print("    VIABILITY_FIXED  : E1 frozen (no viability map; policy gradient only)")
        print()
        print("  Diagnostics:")
        print(f"    1. MAPPED slope > FIXED slope + {SLOPE_MARGIN:.2f}")
        print(f"    2. MAPPED last-Q harm < FIXED last-Q harm * {HARM_FACTOR}")
        print()

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for condition in ["VIABILITY_MAPPED", "VIABILITY_FIXED"]:
            if verbose:
                print(f"  Seed {seed}  Condition {condition}")
            result = run_condition(
                seed=seed,
                condition=condition,
                num_episodes=num_episodes,
                max_steps=max_steps,
                grid_size=grid_size,
                num_hazards=num_hazards,
                verbose=verbose,
            )
            all_results.append(result)
            if verbose:
                print(
                    f"    harm {result['first_quarter_harm']:.3f} -> "
                    f"{result['last_quarter_harm']:.3f}  "
                    f"slope={result['harm_reduction_slope']:+.3f}"
                )
                print()

    mapped = [r for r in all_results if r["condition"] == "VIABILITY_MAPPED"]
    fixed = [r for r in all_results if r["condition"] == "VIABILITY_FIXED"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    mapped_harm_last = _agg(mapped, "last_quarter_harm")
    fixed_harm_last = _agg(fixed, "last_quarter_harm")
    mapped_slope = _agg(mapped, "harm_reduction_slope")
    fixed_slope = _agg(fixed, "harm_reduction_slope")

    crit_1 = mapped_slope > fixed_slope + SLOPE_MARGIN
    crit_2 = mapped_harm_last < fixed_harm_last * HARM_FACTOR

    num_met = sum([crit_1, crit_2])
    verdict = "PASS" if num_met >= 2 else "FAIL"
    partial = num_met == 1

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  VIABILITY_MAPPED harm (last-Q): {mapped_harm_last:.3f}  slope={mapped_slope:+.3f}")
        print(f"  VIABILITY_FIXED  harm (last-Q): {fixed_harm_last:.3f}  slope={fixed_slope:+.3f}")
        print()
        print(
            f"  Crit 1  slope MAPPED={mapped_slope:+.3f} > FIXED+{SLOPE_MARGIN}={fixed_slope+SLOPE_MARGIN:+.3f}  "
            f"{'MET' if crit_1 else 'MISSED'}"
        )
        print(
            f"  Crit 2  harm  MAPPED={mapped_harm_last:.3f} < FIXED*{HARM_FACTOR}={fixed_harm_last*HARM_FACTOR:.3f}  "
            f"{'MET' if crit_2 else 'MISSED'}"
        )
        print()
        print(f"  Criteria met: {num_met}/2  ->  ARC-018 verdict: {verdict}")
        if partial:
            print("  (partial -- 1 of 2 criteria met)")
        print()

    result_doc: Dict[str, Any] = {
        "experiment": "rollout_viability_mapping",
        "claim": "ARC-018",
        "evb_id": "EVB-0017",
        "substrate": "ree-v2",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "environment": "CausalGridWorld",
            "e1_lr": E1_LR,
            "policy_lr": POLICY_LR,
            "slope_margin": SLOPE_MARGIN,
            "harm_factor": HARM_FACTOR,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "viability_mapped_harm_last_quarter": mapped_harm_last,
            "viability_fixed_harm_last_quarter": fixed_harm_last,
            "viability_mapped_harm_slope": mapped_slope,
            "viability_fixed_harm_slope": fixed_slope,
            "criterion_1_slope_met": crit_1,
            "criterion_2_harm_met": crit_2,
            "criteria_met": num_met,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "rollout_viability_mapping"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(evidence_dir / f"rollout_viability_mapping_{ts}.json")
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARC-018: Rollout Viability Mapping (REE-v2)"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--num-hazards", type=int, default=DEFAULT_NUM_HAZARDS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seeds=args.seeds,
        grid_size=args.grid_size,
        num_hazards=args.num_hazards,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
