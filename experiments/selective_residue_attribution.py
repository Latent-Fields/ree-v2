"""
Selective Residue Attribution (SD-003 / MECH-072 / EVB-0047) — REE-v2

Tests whether gating residue accumulation on E2 harm foreseeability reduces
false attribution (residue at env-caused events) vs naive accumulation.

Background:
  Current behaviour: residue is accumulated whenever harm < 0, regardless of
  whether the harm was agent-caused or environment-caused. This conflates the
  agent's moral footprint with random environmental harm.

  SD-003 proposes using E2.predict_harm(z_t, a_actual) as a foreseeability
  signal: if E2 predicted high harm before the step (contamination visible in
  z_t), the harm was likely agent-caused. If E2 was surprised (low predicted
  harm, actual harm occurred), the harm may be environment-caused.

  This experiment tests whether foreseeable-harm gating produces:
  (a) lower false_attribution_rate, and
  (b) no regression in harm avoidance

Conditions:
  NAIVE       — accumulate residue on every harm<0 step (current behaviour)
  FORESEEABLE — accumulate only when predicted_harm > ATTRIBUTION_THRESHOLD
  ORACLE      — accumulate only on agent_caused_hazard (ground truth upper bound)

Pass criteria (BOTH must hold):
  1. false_attribution_rate[FORESEEABLE] < false_attribution_rate[NAIVE]
     (gating reduces false attribution)
  2. final_harm[FORESEEABLE] <= final_harm[NAIVE] * 1.05
     (harm avoidance does not degrade)

ORACLE condition is interpretive: shows theoretical maximum benefit of perfect attribution.

Usage:
    python experiments/selective_residue_attribution.py
    python experiments/selective_residue_attribution.py --episodes 300 --seeds 7 42 99

Claims:
    MECH-072: residue.foreseeable_harm_gating_reduces_false_attribution
    EVB-0047
"""

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


DEFAULT_EPISODES = 300
DEFAULT_MAX_STEPS = 100
DEFAULT_SEEDS = [7, 42, 99]
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 4
DEFAULT_NUM_RESOURCES = 5

# Threshold for foreseeable harm gating (E2 predicted harm in [0,1])
ATTRIBUTION_THRESHOLD = 0.3

# Harm tolerance for criterion 2 (FORESEEABLE harm <= NAIVE * this)
HARM_REGRESSION_TOLERANCE = 1.05

# How many episodes at the end to measure "final harm"
FINAL_QUARTILE_FRACTION = 0.25

E1_LR = 1e-4
E2_LR = 1e-3
POLICY_LR = 1e-3
MAX_GRAD_NORM = 1.0

CONDITIONS = ["NAIVE", "FORESEEABLE", "ORACLE"]


def make_optimizers(agent: REEAgent) -> Tuple:
    """Create E1, policy (E3), and E2 optimizers."""
    e1_opt = torch.optim.Adam(list(agent.e1.parameters()), lr=E1_LR)
    policy_opt = torch.optim.Adam(list(agent.e3.parameters()), lr=POLICY_LR)
    e2_opt = torch.optim.Adam(list(agent.e2.parameters()), lr=E2_LR)
    return e1_opt, policy_opt, e2_opt


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
    Run one training episode under a given attribution condition.

    The condition controls WHEN residue is accumulated:
      NAIVE       — always (on any harm<0)
      FORESEEABLE — only when E2 predicted harm > ATTRIBUTION_THRESHOLD
      ORACLE      — only when transition_type == "agent_caused_hazard"
    """
    obs = env.reset()
    agent.reset()
    total_harm = 0.0
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    prev_latent_z: Optional[torch.Tensor] = None
    prev_action_tensor: Optional[torch.Tensor] = None

    # Attribution tracking
    n_harm_steps = 0
    n_residue_accumulated = 0
    n_false_attribution = 0    # residue at env_caused step
    n_true_attribution = 0     # residue at agent_caused step
    n_missed_attribution = 0   # no residue at agent_caused step (FORESEEABLE/ORACLE only)

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        action, log_prob = agent.act_with_log_prob(obs_tensor)
        current_z = agent._current_latent.z_gamma.detach().clone()

        if prev_latent_z is not None and prev_action_tensor is not None:
            agent.record_transition(prev_latent_z, prev_action_tensor, current_z)

        # Predict harm BEFORE the step (while we still have z_t and the chosen action)
        with torch.no_grad():
            predicted_harm = agent.e2.predict_harm(current_z, action).squeeze().item()

        action_idx = action.argmax(dim=-1).item()
        next_obs, harm, done, info = env.step(action_idx)
        transition_type = info.get("transition_type", "none")

        if harm < 0:
            n_harm_steps += 1
            total_harm += abs(harm)

            # Determine whether to accumulate residue based on condition
            accumulate = False
            if condition == "NAIVE":
                accumulate = True
            elif condition == "FORESEEABLE":
                accumulate = predicted_harm > ATTRIBUTION_THRESHOLD
            elif condition == "ORACLE":
                accumulate = (transition_type == "agent_caused_hazard")

            if accumulate:
                agent.update_residue(harm)
                n_residue_accumulated += 1
                if transition_type == "env_caused_hazard":
                    n_false_attribution += 1
                elif transition_type == "agent_caused_hazard":
                    n_true_attribution += 1
            else:
                # Don't accumulate — but still count missed agent_caused attributions
                if transition_type == "agent_caused_hazard":
                    n_missed_attribution += 1
        else:
            agent.update_residue(harm)  # non-harm residue update (no-op if harm=0)

        log_probs.append(log_prob)
        rewards.append(-abs(harm) if harm < 0 else harm * 0.1)

        prev_latent_z = current_z
        prev_action_tensor = action.detach().clone()
        obs = next_obs

        if done:
            break

    # Policy (E3) update via REINFORCE — must come before E1 update because
    # log_probs reference E1 parameters (via e1_prior in generate_trajectories).
    if log_probs and any(r != 0 for r in rewards):
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        if returns_tensor.std() > 1e-8:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        policy_loss = -torch.stack(log_probs) * returns_tensor
        policy_opt.zero_grad()
        policy_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(list(agent.e3.parameters()), MAX_GRAD_NORM)
        policy_opt.step()

    # E1 update (after policy, so E1 step doesn't corrupt log_probs graph)
    e1_loss = agent.compute_prediction_loss()
    if e1_loss.requires_grad:
        e1_opt.zero_grad()
        e1_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(agent.e1.parameters()), MAX_GRAD_NORM)
        e1_opt.step()

    # E2 update
    e2_loss = agent.compute_e2_loss()
    if e2_loss.requires_grad:
        e2_opt.zero_grad()
        e2_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(agent.e2.parameters()), MAX_GRAD_NORM)
        e2_opt.step()

    return {
        "harm": total_harm,
        "n_harm_steps": n_harm_steps,
        "n_residue_accumulated": n_residue_accumulated,
        "n_false_attribution": n_false_attribution,
        "n_true_attribution": n_true_attribution,
        "n_missed_attribution": n_missed_attribution,
    }


def run_condition(
    condition: str,
    num_episodes: int,
    max_steps: int,
    seed: int,
    grid_size: int,
    num_hazards: int,
    num_resources: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Run one (condition, seed) trial."""
    torch.manual_seed(seed)

    env = CausalGridWorld(
        size=grid_size,
        num_hazards=num_hazards,
        num_resources=num_resources,
        seed=seed,
    )
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config)
    e1_opt, policy_opt, e2_opt = make_optimizers(agent)

    episode_harms: List[float] = []
    total_harm_steps = 0
    total_residue = 0
    total_false = 0
    total_true = 0
    total_missed = 0

    for ep in range(num_episodes):
        metrics = run_episode(
            agent, env, e1_opt, policy_opt, e2_opt, condition, max_steps
        )
        episode_harms.append(metrics["harm"])
        total_harm_steps += metrics["n_harm_steps"]
        total_residue += metrics["n_residue_accumulated"]
        total_false += metrics["n_false_attribution"]
        total_true += metrics["n_true_attribution"]
        total_missed += metrics["n_missed_attribution"]

        if verbose and (ep + 1) % 50 == 0:
            q = max(1, int(num_episodes * (1 - FINAL_QUARTILE_FRACTION)))
            recent = statistics.mean(episode_harms[max(0, ep-9):ep+1])
            final_harm_so_far = statistics.mean(episode_harms[q:ep+1]) if ep >= q else recent
            false_rate = total_false / total_residue if total_residue > 0 else 0.0
            print(f"    ep {ep+1:3d}/{num_episodes}  harm={recent:.3f}  "
                  f"false_attr_rate={false_rate:.3f}  residue={total_residue}")

    # Final harm = last FINAL_QUARTILE_FRACTION episodes
    q_start = int(num_episodes * (1 - FINAL_QUARTILE_FRACTION))
    final_harm = statistics.mean(episode_harms[q_start:]) if episode_harms[q_start:] else float("nan")

    false_attribution_rate = total_false / total_residue if total_residue > 0 else 0.0
    true_attribution_rate = total_true / (total_harm_steps) if total_harm_steps > 0 else 0.0

    return {
        "seed": seed,
        "condition": condition,
        "final_harm": round(final_harm, 4),
        "false_attribution_rate": round(false_attribution_rate, 4),
        "true_attribution_rate": round(true_attribution_rate, 4),
        "total_harm_steps": total_harm_steps,
        "total_residue_accumulated": total_residue,
        "total_false_attribution": total_false,
        "total_true_attribution": total_true,
        "total_missed_attribution": total_missed,
    }


def run_experiment(
    num_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seeds: List[int] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_hazards: int = DEFAULT_NUM_HAZARDS,
    num_resources: int = DEFAULT_NUM_RESOURCES,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    seeds = seeds or DEFAULT_SEEDS
    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print()
        print("[Selective Residue Attribution — MECH-072 / EVB-0047] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print(f"  Attribution threshold (FORESEEABLE): {ATTRIBUTION_THRESHOLD}")
        print()
        print("  Conditions:")
        print("    NAIVE       — accumulate residue on all harm (current behaviour)")
        print(f"    FORESEEABLE — accumulate when E2.predict_harm > {ATTRIBUTION_THRESHOLD}")
        print("    ORACLE      — accumulate only on agent_caused_hazard (ground truth)")
        print()

    condition_results: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        if verbose:
            print(f"  --- Condition: {condition} ---")
        for seed in seeds:
            if verbose:
                print(f"  Seed {seed}  Condition {condition}")
            result = run_condition(
                condition, num_episodes, max_steps, seed,
                grid_size, num_hazards, num_resources, verbose,
            )
            condition_results[condition].append(result)
            if verbose:
                print(f"    final_harm: {result['final_harm']:.4f}  "
                      f"false_attr_rate: {result['false_attribution_rate']:.4f}  "
                      f"true_attr_rate: {result['true_attribution_rate']:.4f}")
        if verbose:
            print()

    # Aggregate per condition
    def mean_metric(results: List[Dict], key: str) -> Optional[float]:
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    agg = {}
    for cond in CONDITIONS:
        agg[cond] = {
            "mean_final_harm": mean_metric(condition_results[cond], "final_harm"),
            "mean_false_attribution_rate": mean_metric(condition_results[cond], "false_attribution_rate"),
            "mean_true_attribution_rate": mean_metric(condition_results[cond], "true_attribution_rate"),
        }

    naive_harm = agg["NAIVE"]["mean_final_harm"]
    fore_harm = agg["FORESEEABLE"]["mean_final_harm"]
    naive_false = agg["NAIVE"]["mean_false_attribution_rate"]
    fore_false = agg["FORESEEABLE"]["mean_false_attribution_rate"]

    crit1 = (fore_false is not None and naive_false is not None and fore_false < naive_false)
    crit2 = (fore_harm is not None and naive_harm is not None
             and fore_harm <= naive_harm * HARM_REGRESSION_TOLERANCE)

    verdict = "PASS" if (crit1 and crit2) else "FAIL"

    if verbose:
        print("=" * 60)
        print("[Summary]")
        for cond in CONDITIONS:
            print(f"  {cond:<12}  "
                  f"final_harm: {agg[cond]['mean_final_harm']:.4f}  "
                  f"false_attr_rate: {agg[cond]['mean_false_attribution_rate']:.4f}  "
                  f"true_attr_rate: {agg[cond]['mean_true_attribution_rate']:.4f}")
        print()
        print(
            f"  Criterion 1 (FORE false_rate < NAIVE false_rate)?  "
            f"{'YES' if crit1 else 'NO'}  "
            f"({fore_false:.4f} < {naive_false:.4f})"
            if (fore_false is not None and naive_false is not None)
            else f"  Criterion 1 (FORE false_rate < NAIVE false_rate)?  NO  (None)"
        )
        print(
            f"  Criterion 2 (FORE harm <= NAIVE*{HARM_REGRESSION_TOLERANCE})?  "
            f"{'YES' if crit2 else 'NO'}  "
            f"({fore_harm:.4f} <= {naive_harm * HARM_REGRESSION_TOLERANCE:.4f})"
            if (fore_harm is not None and naive_harm is not None)
            else f"  Criterion 2 (FORE harm <= NAIVE*{HARM_REGRESSION_TOLERANCE})?  NO  (None)"
        )
        print()
        oracle_false = agg["ORACLE"]["mean_false_attribution_rate"]
        oracle_harm = agg["ORACLE"]["mean_final_harm"]
        print(f"  ORACLE interpretation: false_rate={oracle_false:.4f}  harm={oracle_harm:.4f}")
        print(f"    (theoretical upper bound — how much better perfect attribution could be)")
        print()
        print(f"  MECH-072 / EXQ-028 verdict: {verdict}")

    result_summary = (
        f"NAIVE false_attr: {naive_false:.4f} | "
        f"FORE false_attr: {fore_false:.4f} | "
        f"NAIVE harm: {naive_harm:.4f} | "
        f"FORE harm: {fore_harm:.4f} | "
        f"MECH-072 verdict: {verdict}"
    )

    result_doc = {
        "experiment": "selective_residue_attribution",
        "claim": "MECH-072",
        "evb_id": "EVB-0047",
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
            "attribution_threshold": ATTRIBUTION_THRESHOLD,
            "harm_regression_tolerance": HARM_REGRESSION_TOLERANCE,
            "final_quartile_fraction": FINAL_QUARTILE_FRACTION,
            "e1_lr": E1_LR,
            "e2_lr": E2_LR,
            "policy_lr": POLICY_LR,
        },
        "verdict": verdict,
        "result_summary": result_summary,
        "criteria_met": sum([crit1, crit2]),
        "aggregate": {
            **{f"{c.lower()}_mean_final_harm": agg[c]["mean_final_harm"] for c in CONDITIONS},
            **{f"{c.lower()}_mean_false_attribution_rate": agg[c]["mean_false_attribution_rate"] for c in CONDITIONS},
            **{f"{c.lower()}_mean_true_attribution_rate": agg[c]["mean_true_attribution_rate"] for c in CONDITIONS},
            "criterion_1_foreseeable_reduces_false_attribution": crit1,
            "criterion_2_foreseeable_no_harm_regression": crit2,
        },
        "per_condition": {c: condition_results[c] for c in CONDITIONS},
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "selective_residue_attribution"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"selective_residue_attribution_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"\n  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Selective Residue Attribution: does E2 foreseeability gating "
            "reduce false attribution?"
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
