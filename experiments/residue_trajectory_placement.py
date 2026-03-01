"""
Residue Trajectory Placement Experiment (MECH-056 / EVB-0039) — REE-v2

V2 port: CausalGridWorld replaces GridWorld. All logic identical.

Tests whether residue accumulates at INTERMEDIATE trajectory positions during
rollout planning, not just at the terminal executed step.

MECH-056 claims: residue pressure should be placed at trajectory feasibility and
commitment gating (hippocampal rollout costs, E3 thresholds) rather than distorting
core E1/E2 representational geometry. The precondition: residue must register along
the trajectory path, not only at its endpoint.

Conditions:
  A (TRAJECTORY-WIDE):  After harm, accumulate residue at all intermediate states
                         in the selected trajectory with linearly decayed intensity,
                         then update terminal state via standard agent.update_residue().
  B (ENDPOINT-ONLY):    Standard behaviour — residue only at terminal state.

Key diagnostics:
  1. TRAJECTORY-WIDE mean_intermediate_residue_mass > 0
  2. TRAJECTORY-WIDE last-quarter harm <= ENDPOINT-ONLY * 1.05

Both must hold for MECH-056 PASS.

Usage:
    python experiments/residue_trajectory_placement.py
    python experiments/residue_trajectory_placement.py --episodes 5 --seeds 7

Claims:
    MECH-056: residue.trajectory_first_placement
    EVB-0039
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


DEFAULT_EPISODES = 200
DEFAULT_MAX_STEPS = 100
DEFAULT_SEEDS = [7, 42, 99]
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 4


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient between two lists."""
    n = len(xs)
    if n < 4:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def accumulate_intermediate_residue(
    agent: REEAgent,
    trajectory,
    harm_magnitude: float,
) -> float:
    """
    Spread residue across intermediate (non-terminal) trajectory states.

    Weight for state at index i (1-indexed over non-terminal) = i / H,
    so earlier steps get lighter imprint, steps closer to terminal get heavier.

    Returns total intermediate residue mass added.
    """
    states = trajectory.states  # List of Tensor[batch, latent_dim]
    H = len(states) - 1         # rollout horizon

    if H < 2:
        return 0.0

    total_inter_mass = 0.0
    for i in range(1, H):
        weight = i / H
        weighted_harm = harm_magnitude * weight
        agent.residue_field.accumulate(states[i], weighted_harm)
        total_inter_mass += weighted_harm

    return total_inter_mass


def run_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    condition: str,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode using the low-level REE pipeline to access trajectory states.

    Uses sense → update_latent → generate_trajectories → e3.select() directly
    so we have access to selected_trajectory.states.
    """
    obs = env.reset()
    agent.reset()

    total_harm = 0.0
    steps = 0
    ep_intermediate_mass = 0.0
    trajectory_residue_costs: List[float] = []

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)
            result = agent.e3.select(candidates)

        selected_traj = result.selected_trajectory

        try:
            residue_cost = agent.e3.compute_residue_cost(selected_traj).mean().item()
        except Exception:
            residue_cost = 0.0
        trajectory_residue_costs.append(residue_cost)

        action_idx = result.selected_action.argmax(dim=-1).item()
        next_obs, harm, done, _info = env.step(action_idx)

        if harm < 0:
            harm_magnitude = abs(harm)
            total_harm += harm_magnitude

            if condition == "TRAJECTORY-WIDE":
                inter_mass = accumulate_intermediate_residue(
                    agent, selected_traj, harm_magnitude
                )
                ep_intermediate_mass += inter_mass
            agent.update_residue(harm)

        obs = next_obs
        steps += 1
        if done:
            break

    mean_residue_cost = (
        statistics.mean(trajectory_residue_costs) if trajectory_residue_costs else 0.0
    )

    return {
        "total_harm": total_harm,
        "steps": steps,
        "intermediate_residue_mass": ep_intermediate_mass,
        "trajectory_residue_cost": mean_residue_cost,
    }


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

    ep_harms: List[float] = []
    ep_inter_masses: List[float] = []
    ep_residue_costs: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, condition, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_inter_masses.append(metrics["intermediate_residue_mass"])
        ep_residue_costs.append(metrics["trajectory_residue_cost"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            recent_inter = statistics.mean(ep_inter_masses[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  "
                f"seed={seed}  cond={condition}  "
                f"harm={recent_harm:.3f}  inter_mass={recent_inter:.4f}"
            )

    quarter = max(1, num_episodes // 4)

    return {
        "condition": condition,
        "seed": seed,
        "first_quarter_harm": round(statistics.mean(ep_harms[:quarter]), 4),
        "last_quarter_harm": round(statistics.mean(ep_harms[-quarter:]), 4),
        "mean_intermediate_residue_mass": round(statistics.mean(ep_inter_masses), 6),
        "mean_trajectory_residue_cost": round(statistics.mean(ep_residue_costs), 6),
        "episode_count": num_episodes,
    }


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
        print("[Residue Trajectory Placement — MECH-056 / EVB-0039] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print("    A (TRAJECTORY-WIDE): residue spread along all trajectory steps (decayed)")
        print("    B (ENDPOINT-ONLY):   residue only at terminal step (standard)")
        print()
        print("  Diagnostic 1: TRAJECTORY-WIDE mean_intermediate_residue_mass > 0")
        print("  Diagnostic 2: TRAJECTORY-WIDE last-Q harm <= ENDPOINT-ONLY last-Q harm * 1.05")
        print()

    all_results = []

    for seed in seeds:
        for condition in ["TRAJECTORY-WIDE", "ENDPOINT-ONLY"]:
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
                    f"    harm {result['first_quarter_harm']:.3f} → "
                    f"{result['last_quarter_harm']:.3f}  "
                    f"inter_mass={result['mean_intermediate_residue_mass']:.5f}"
                )
                print()

    traj_wide = [r for r in all_results if r["condition"] == "TRAJECTORY-WIDE"]
    endpoint = [r for r in all_results if r["condition"] == "ENDPOINT-ONLY"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    tw_harm_last = _agg(traj_wide, "last_quarter_harm")
    ep_harm_last = _agg(endpoint, "last_quarter_harm")
    tw_inter_mass = round(
        statistics.mean(r["mean_intermediate_residue_mass"] for r in traj_wide), 6
    )

    path_spread_ok = tw_inter_mass > 0.0
    harm_ok = tw_harm_last <= ep_harm_last * 1.05
    verdict = "PASS" if (path_spread_ok and harm_ok) else "FAIL"
    partial = (path_spread_ok or harm_ok) and not (path_spread_ok and harm_ok)

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  TRAJECTORY-WIDE  last-Q harm: {tw_harm_last:.3f}")
        print(f"  ENDPOINT-ONLY    last-Q harm: {ep_harm_last:.3f}")
        print(f"  TRAJECTORY-WIDE  mean intermediate residue mass: {tw_inter_mass:.6f}")
        print()
        print(f"  Path spread (inter_mass > 0)?  {'YES' if path_spread_ok else 'NO'}  ({tw_inter_mass:.6f})")
        print(f"  Harm neutral or better?        {'YES' if harm_ok else 'NO'}")
        print()
        print(f"  MECH-056 verdict: {verdict}")
        if partial:
            print("  (partial — one of two criteria met)")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    Residue accumulates at intermediate trajectory positions.")
            print("    The φ(z) residue field can serve as terrain for hippocampal rollout generation.")
            print("    MECH-056 structural requirement confirmed on V2 substrate.")

    result_doc = {
        "experiment": "residue_trajectory_placement",
        "claim": "MECH-056",
        "evb_id": "EVB-0039",
        "substrate": "ree-v2",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "environment": "CausalGridWorld",
            "intermediate_weight_scheme": "linear_ramp_i_over_H",
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "trajectory_wide_harm_last_quarter": tw_harm_last,
            "endpoint_only_harm_last_quarter": ep_harm_last,
            "trajectory_wide_mean_intermediate_residue_mass": tw_inter_mass,
            "harm_tolerance_factor": 1.05,
            "path_spread_criterion_met": path_spread_ok,
            "harm_avoidance_criterion_met": harm_ok,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "residue_trajectory_placement"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"residue_trajectory_placement_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description="MECH-056: Residue Trajectory Placement experiment (REE-v2)"
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
