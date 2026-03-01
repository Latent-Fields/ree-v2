"""
CausalGridWorld Baseline Validation (SD-003 Prerequisite / Step 2.3 Exit Criteria)

Tests that:
  1. CausalGridWorld produces structurally distinct agent_caused_hazard vs
     env_caused_hazard transition types over a baseline run.
  2. The baseline REEAgent creates a non-trivial contamination footprint
     (contamination_events > 0).
  3. Both harm types are observed: agent_caused_harm_count > 0 AND
     env_caused_harm_count > 0.
  4. Agent's causal contribution is non-trivial:
     agent_caused / (agent_caused + env_caused) > 0.1

Pass Criteria (ALL must hold):
  1. contamination_events_total > 0
  2. agent_caused_harm_count_total > 0
  3. env_caused_harm_count_total > 0
  4. agent_fraction > 0.1  (agent has genuine footprint)

Usage:
    python experiments/causal_grid_world_baseline.py
    python experiments/causal_grid_world_baseline.py --episodes 20 --seeds 42

Claims:
    SD-003 substrate prerequisite (Step 2.3 exit criterion)
    EVB-0045
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


DEFAULT_EPISODES = 50
DEFAULT_MAX_STEPS = 100
DEFAULT_SEEDS = [7, 42, 99]
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 3
DEFAULT_NUM_RESOURCES = 5
AGENT_FRACTION_THRESHOLD = 0.1
E1_LR = 1e-4
POLICY_LR = 1e-3
MAX_GRAD_NORM = 1.0


def make_optimizers(
    agent: REEAgent,
) -> tuple:
    """Build two optimizers matching REETrainer's parameter grouping."""
    e1_params = (
        list(agent.e1.parameters())
        + list(agent.latent_stack.parameters())
        + list(agent.obs_encoder.parameters())
    )
    policy_params = list(agent.e3.parameters())
    e1_opt = torch.optim.Adam(e1_params, lr=E1_LR)
    policy_opt = torch.optim.Adam(policy_params, lr=POLICY_LR)
    return e1_opt, policy_opt


def run_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode and collect CausalGridWorld attribution statistics.

    Tracks agent_caused_hazard vs env_caused_hazard transition types,
    and contamination events, to validate SD-003 substrate prerequisites.
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    agent_caused_count = 0
    env_caused_count = 0
    contamination_events = 0
    total_harm = 0.0
    steps = 0

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

        result = agent.e3.select(candidates)
        if result.log_prob is not None:
            log_probs.append(result.log_prob)

        action_idx = result.selected_action.argmax(dim=-1).item()
        next_obs, harm, done, info = env.step(action_idx)

        # Track CausalGridWorld transition types (SD-003 prerequisite)
        tt = info.get("transition_type", "none")
        if tt == "agent_caused_hazard":
            agent_caused_count += 1
        elif tt == "env_caused_hazard":
            env_caused_count += 1

        # Track contamination events (footprint existence criterion)
        if info.get("contamination_delta", 0.0) > 0:
            contamination_events += 1

        actual_harm = abs(harm) if harm < 0 else 0.0
        total_harm += actual_harm

        agent.update_residue(harm)

        obs = next_obs
        steps += 1
        if done:
            break

    # Policy update (REINFORCE with actual harm)
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

    # E1 update
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

    return {
        "total_harm": total_harm,
        "steps": steps,
        "agent_caused_harm_count": agent_caused_count,
        "env_caused_harm_count": env_caused_count,
        "contamination_events": contamination_events,
        "e1_loss": e1_loss_val,
        "policy_loss": policy_loss_val,
    }


def run_seed(
    seed: int,
    num_episodes: int,
    max_steps: int,
    grid_size: int,
    num_hazards: int,
    num_resources: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the baseline condition for a single seed."""
    torch.manual_seed(seed)

    env = CausalGridWorld(
        size=grid_size,
        num_hazards=num_hazards,
        num_resources=num_resources,
    )
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)
    e1_opt, policy_opt = make_optimizers(agent)

    ep_agent_caused: List[int] = []
    ep_env_caused: List[int] = []
    ep_contamination: List[int] = []
    ep_harms: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, max_steps)
        ep_agent_caused.append(metrics["agent_caused_harm_count"])
        ep_env_caused.append(metrics["env_caused_harm_count"])
        ep_contamination.append(metrics["contamination_events"])
        ep_harms.append(metrics["total_harm"])

        if verbose and (ep + 1) % 10 == 0:
            recent_agent = sum(ep_agent_caused[-10:])
            recent_env = sum(ep_env_caused[-10:])
            recent_cont = sum(ep_contamination[-10:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  seed={seed}  "
                f"agent_caused={recent_agent}  env_caused={recent_env}  "
                f"contamination={recent_cont}"
            )

    total_agent = sum(ep_agent_caused)
    total_env = sum(ep_env_caused)
    total_contamination = sum(ep_contamination)
    total_harm_events = total_agent + total_env

    agent_fraction = (
        total_agent / total_harm_events if total_harm_events > 0 else 0.0
    )

    return {
        "seed": seed,
        "episode_count": num_episodes,
        "agent_caused_harm_count_total": total_agent,
        "env_caused_harm_count_total": total_env,
        "contamination_events_total": total_contamination,
        "agent_fraction": round(agent_fraction, 4),
        "mean_harm_per_episode": round(statistics.mean(ep_harms) if ep_harms else 0.0, 4),
    }


def run_experiment(
    num_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seeds: Optional[List[int]] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_hazards: int = DEFAULT_NUM_HAZARDS,
    num_resources: int = DEFAULT_NUM_RESOURCES,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print("[CausalGridWorld Baseline Validation — SD-003 Prerequisite / Step 2.3]")
        print(
            f"  CausalGridWorld: {grid_size}x{grid_size}, "
            f"{num_hazards} hazards, {num_resources} resources"
        )
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Pass criteria (ALL must hold):")
        print("    1. contamination_events_total > 0")
        print("    2. agent_caused_harm_count_total > 0")
        print("    3. env_caused_harm_count_total > 0")
        print(
            f"    4. agent_fraction > {AGENT_FRACTION_THRESHOLD}"
            "  (agent has genuine footprint)"
        )
        print()

    all_results = []

    for seed in seeds:
        if verbose:
            print(f"  Seed {seed}  Condition BASELINE")
        result = run_seed(
            seed=seed,
            num_episodes=num_episodes,
            max_steps=max_steps,
            grid_size=grid_size,
            num_hazards=num_hazards,
            num_resources=num_resources,
            verbose=verbose,
        )
        all_results.append(result)
        if verbose:
            print(
                f"    agent_caused={result['agent_caused_harm_count_total']}  "
                f"env_caused={result['env_caused_harm_count_total']}  "
                f"contamination={result['contamination_events_total']}  "
                f"agent_fraction={result['agent_fraction']:.3f}"
            )
            print()

    # Aggregate across seeds
    total_agent = sum(r["agent_caused_harm_count_total"] for r in all_results)
    total_env = sum(r["env_caused_harm_count_total"] for r in all_results)
    total_contamination = sum(r["contamination_events_total"] for r in all_results)
    total_harm_events = total_agent + total_env
    agg_agent_fraction = (
        total_agent / total_harm_events if total_harm_events > 0 else 0.0
    )

    crit1 = total_contamination > 0
    crit2 = total_agent > 0
    crit3 = total_env > 0
    crit4 = agg_agent_fraction > AGENT_FRACTION_THRESHOLD

    verdict = "PASS" if (crit1 and crit2 and crit3 and crit4) else "FAIL"
    criteria_met = sum([crit1, crit2, crit3, crit4])

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  contamination_events_total:    {total_contamination}")
        print(f"  agent_caused_harm_count_total: {total_agent}")
        print(f"  env_caused_harm_count_total:   {total_env}")
        print(f"  aggregate agent_fraction:      {agg_agent_fraction:.3f}")
        print()
        print(
            f"  Criterion 1 (contamination > 0)?               "
            f"{'YES' if crit1 else 'NO'}  ({total_contamination})"
        )
        print(
            f"  Criterion 2 (agent_caused > 0)?                "
            f"{'YES' if crit2 else 'NO'}  ({total_agent})"
        )
        print(
            f"  Criterion 3 (env_caused > 0)?                  "
            f"{'YES' if crit3 else 'NO'}  ({total_env})"
        )
        print(
            f"  Criterion 4 (agent_fraction > {AGENT_FRACTION_THRESHOLD})?  "
            f"{'YES' if crit4 else 'NO'}  ({agg_agent_fraction:.3f})"
        )
        print()
        print(f"  SD-003 substrate verdict: {verdict}  ({criteria_met}/4 criteria met)")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    CausalGridWorld produces structurally distinct transition types.")
            print("    REE agent generates non-trivial contamination footprint.")
            print(
                "    Step 2.3 exit criteria confirmed — SD-003 substrate ready."
            )

    result_doc = {
        "experiment": "causal_grid_world_baseline",
        "claim": "SD-003-prereq",
        "evb_id": "EVB-0045",
        "substrate": "ree-v2",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "num_resources": num_resources,
            "environment": "CausalGridWorld",
            "agent_fraction_threshold": AGENT_FRACTION_THRESHOLD,
            "e1_lr": E1_LR,
            "policy_lr": POLICY_LR,
        },
        "verdict": verdict,
        "criteria_met": criteria_met,
        "aggregate": {
            "contamination_events_total": total_contamination,
            "agent_caused_harm_count_total": total_agent,
            "env_caused_harm_count_total": total_env,
            "agent_fraction": round(agg_agent_fraction, 4),
            "criterion_1_contamination_gt_0": crit1,
            "criterion_2_agent_caused_gt_0": crit2,
            "criterion_3_env_caused_gt_0": crit3,
            "criterion_4_agent_fraction_gt_threshold": crit4,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "causal_grid_world_baseline"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"causal_grid_world_baseline_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SD-003 Substrate Prerequisite: CausalGridWorld Baseline Validation "
            "(Step 2.3 exit criteria)"
        )
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--num-hazards", type=int, default=DEFAULT_NUM_HAZARDS)
    parser.add_argument("--num-resources", type=int, default=DEFAULT_NUM_RESOURCES)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seeds=args.seeds,
        grid_size=args.grid_size,
        num_hazards=args.num_hazards,
        num_resources=args.num_resources,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
