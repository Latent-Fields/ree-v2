"""
Kernel Chaining Interface Experiment (MECH-033 / EVB-0016) — REE-v2

Tests MECH-033: hippocampus.kernel_chaining_interface

MECH-033 asserts that E2 supplies SHORT-horizon forward-prediction kernels
(local conditional transitions) and that hippocampal systems chain these
kernels into EXPLICIT multi-step rollouts, constrained by E1 priors.
The key claim: the kernel → rollout handoff is a distinct, load-bearing
interface — not just E2 doing multi-step prediction directly.

CausalGridWorld provides a natural test environment: the E2-kernel-to-rollout
pipeline (generate_trajectories + e3.select) must correctly chain local
transition predictions into viable multi-step paths while navigating
contamination terrain and drift hazards.

Conditions (2):
  WITH_CHAIN  : Full pipeline. E2 kernels chained into candidate trajectories
                via generate_trajectories(); E3 selects best candidate.
                The complete E2 → hippocampus → E3 interface is active.

  NO_CHAIN    : Kernel chaining disabled.  Agent acts randomly (uniform over
                action space) without trajectory generation or E3 selection.
                E1 world-model still updates from prediction loss.
                This is the ablation baseline: E2 kernel quality still
                improves the world model, but the kernel-chaining interface
                to hippocampus is not used for action selection.

Key diagnostics:
  1. last_quarter_harm: WITH_CHAIN < NO_CHAIN * 0.90
     (chaining E2 kernels into multi-step rollouts substantially reduces harm)

  2. harm_reduction_slope: WITH_CHAIN slope > NO_CHAIN slope + 0.05
     (kernel-chained policy improves faster over the session)

Pass: >= 2 of 2 criteria met.

Note on REINFORCE in NO_CHAIN: there is no differentiable selection in the
random condition, so no policy gradient is computed for NO_CHAIN.  E1 still
updates.  The comparison is fair: both conditions have the same E1 quality;
the difference is whether E2 kernels are chained into rollouts for selection.

Usage:
    python experiments/kernel_chaining_interface.py
    python experiments/kernel_chaining_interface.py --episodes 5 --seeds 7

Claims:
    MECH-033: hippocampus.kernel_chaining_interface
    EVB-0016
"""

from __future__ import annotations

import argparse
import json
import random as py_random
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

HARM_FACTOR: float = 0.90         # WITH_CHAIN last-Q harm < NO_CHAIN * factor
SLOPE_MARGIN: float = 0.05        # WITH_CHAIN slope > NO_CHAIN slope + margin


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

    WITH_CHAIN: full trajectory generation + E3 selection (E2 kernels chained).
    NO_CHAIN:   random action selection; no trajectory generation; E1 still updates.
    E2 trains in both conditions — it learns motor-sensory transitions regardless.
    """
    agent.reset()
    obs = env.reset()
    num_actions = env.action_dim

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
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

        # Capture z_t; record E2 transition (z_{t-1}, a_{t-1}, z_t)
        z_t = agent._current_latent.z_gamma.detach().clone()
        if prev_latent_z is not None and prev_action_tensor is not None:
            agent.record_transition(prev_latent_z, prev_action_tensor, z_t)

        if condition == "WITH_CHAIN":
            with torch.no_grad():
                candidates = agent.generate_trajectories(agent._current_latent)
            result = agent.e3.select(candidates)
            if result.log_prob is not None:
                log_probs.append(result.log_prob)
            action_tensor = result.selected_action.detach().clone()
            action_idx = action_tensor.argmax(dim=-1).item()
        else:  # NO_CHAIN
            # Random action — no kernel chaining, no hippocampal selection
            action_idx = py_random.randrange(num_actions)
            action_tensor = torch.zeros(1, num_actions)
            action_tensor[0, action_idx] = 1.0

        next_obs, harm, done, _info = env.step(action_idx)

        actual_harm = abs(harm) if harm < 0 else 0.0
        total_harm += actual_harm

        agent.update_residue(harm)

        # ── E1 update (both conditions) ──
        e1_loss = agent.compute_prediction_loss()
        if e1_loss.requires_grad:
            e1_opt.zero_grad()
            e1_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for grp in e1_opt.param_groups for p in grp["params"]],
                MAX_GRAD_NORM,
            )
            e1_opt.step()

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

        prev_latent_z = z_t
        prev_action_tensor = action_tensor
        obs = next_obs
        steps += 1
        if done:
            break

    # ── REINFORCE policy update (WITH_CHAIN only) ──
    policy_loss_val = 0.0
    if condition == "WITH_CHAIN" and log_probs:
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

    return {
        "total_harm": total_harm,
        "steps": steps,
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
    py_random.seed(seed)
    env = CausalGridWorld(size=grid_size, num_hazards=num_hazards)
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)
    e1_opt, policy_opt, e2_opt = make_optimizers(agent)

    ep_harms: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, e2_opt, condition, max_steps)
        ep_harms.append(metrics["total_harm"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  seed={seed}  "
                f"cond={condition}  harm={recent_harm:.3f}"
            )

    quarter = max(1, num_episodes // 4)
    first_q = statistics.mean(ep_harms[:quarter])
    last_q = statistics.mean(ep_harms[-quarter:])
    slope = (first_q - last_q) / max(1e-6, first_q)

    return {
        "condition": condition,
        "seed": seed,
        "first_quarter_harm": round(first_q, 4),
        "last_quarter_harm": round(last_q, 4),
        "harm_reduction_slope": round(slope, 4),
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
        print("[Kernel Chaining Interface — MECH-033 / EVB-0016] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print("    WITH_CHAIN : E2 kernels chained into rollouts; E3 selects trajectory")
        print("    NO_CHAIN   : random action; no chaining; E1 still learns")
        print()
        print("  Diagnostics:")
        print(f"    1. WITH_CHAIN last-Q harm < NO_CHAIN * {HARM_FACTOR}")
        print(f"    2. WITH_CHAIN slope > NO_CHAIN slope + {SLOPE_MARGIN}")
        print()

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for condition in ["WITH_CHAIN", "NO_CHAIN"]:
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
                    f"{result['last_quarter_harm']:.3f}  slope={result['harm_reduction_slope']:+.3f}"
                )
                print()

    with_chain = [r for r in all_results if r["condition"] == "WITH_CHAIN"]
    no_chain = [r for r in all_results if r["condition"] == "NO_CHAIN"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    wc_harm_last = _agg(with_chain, "last_quarter_harm")
    nc_harm_last = _agg(no_chain, "last_quarter_harm")
    wc_slope = _agg(with_chain, "harm_reduction_slope")
    nc_slope = _agg(no_chain, "harm_reduction_slope")

    crit_1 = wc_harm_last < nc_harm_last * HARM_FACTOR
    crit_2 = wc_slope > nc_slope + SLOPE_MARGIN

    num_met = sum([crit_1, crit_2])
    verdict = "PASS" if num_met >= 2 else "FAIL"
    partial = num_met == 1

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  WITH_CHAIN harm (last-Q): {wc_harm_last:.3f}  slope={wc_slope:+.3f}")
        print(f"  NO_CHAIN   harm (last-Q): {nc_harm_last:.3f}  slope={nc_slope:+.3f}")
        print()
        print(
            f"  Crit 1  CHAIN={wc_harm_last:.3f} < "
            f"NO_CHAIN*{HARM_FACTOR}={nc_harm_last*HARM_FACTOR:.3f}  "
            f"{'MET' if crit_1 else 'MISSED'}"
        )
        print(
            f"  Crit 2  slope CHAIN={wc_slope:+.3f} > "
            f"NO_CHAIN+{SLOPE_MARGIN}={nc_slope+SLOPE_MARGIN:+.3f}  "
            f"{'MET' if crit_2 else 'MISSED'}"
        )
        print()
        print(f"  Criteria met: {num_met}/2  ->  MECH-033 verdict: {verdict}")
        if partial:
            print("  (partial -- 1 of 2 criteria met)")

    result_doc: Dict[str, Any] = {
        "experiment": "kernel_chaining_interface",
        "claim": "MECH-033",
        "evb_id": "EVB-0016",
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
            "harm_factor": HARM_FACTOR,
            "slope_margin": SLOPE_MARGIN,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "with_chain_harm_last_quarter": wc_harm_last,
            "no_chain_harm_last_quarter": nc_harm_last,
            "with_chain_slope": wc_slope,
            "no_chain_slope": nc_slope,
            "criterion_1_harm_met": crit_1,
            "criterion_2_slope_met": crit_2,
            "criteria_met": num_met,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "kernel_chaining_interface"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(evidence_dir / f"kernel_chaining_interface_{ts}.json")
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MECH-033: Kernel Chaining Interface (REE-v2)"
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
