"""
E2 Attribution Calibration (SD-003 / MECH-071 / EVB-0046) — REE-v2

Tests whether trained E2 predicts harm more reliably before agent-caused
transitions than before environment-caused transitions.

Background:
  In CausalGridWorld both agent_caused_hazard and env_caused_hazard require
  the agent to move into a hazardous cell — so the raw action counterfactual
  ||E2(z,a_actual) - E2(z,a_null)|| gives a large delta for both. The
  discrimination signal lives in z_t itself: the contamination_view encodes
  where the agent has been, letting a trained E2 anticipate contamination-based
  harm. Environment hazards are not visible in the contamination channel.

  Hypothesis (MECH-071): after training,
    mean(E2.predict_harm(z_t, a_actual) | agent_caused)
    > mean(E2.predict_harm(z_t, a_actual) | env_caused)
  because contamination is visible in z_gamma; env hazard placement is not.

Conditions:
  TRAINED  — E2 trained for WARMUP_EPISODES, then evaluated
  RANDOM   — no warmup, random E2 weights, evaluated (control)

Pass criteria (ALL must hold):
  1. TRAINED calibration_gap > 0.05
     (E2 foresees agent-caused harm measurably better than env-caused)
  2. |RANDOM calibration_gap| < 0.10
     (untrained E2 shows no spurious discrimination — control check)
  3. Both transition types observed during evaluation (data validity check)

Usage:
    python experiments/causal_attribution_calibration.py
    python experiments/causal_attribution_calibration.py --warmup 200 --eval 50 --seeds 7 42 99

Claims:
    MECH-071: e2.harm_prediction_calibration_asymmetry
    EVB-0046
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


DEFAULT_WARMUP_EPISODES = 200
DEFAULT_EVAL_EPISODES = 50
DEFAULT_MAX_STEPS = 100
DEFAULT_SEEDS = [7, 42, 99]
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 4
DEFAULT_NUM_RESOURCES = 5

CALIBRATION_GAP_THRESHOLD = 0.05   # TRAINED must exceed this
RANDOM_GAP_ABS_MAX = 0.10          # RANDOM gap must stay below this

E1_LR = 1e-4
E2_LR = 1e-3
POLICY_LR = 1e-3
MAX_GRAD_NORM = 1.0


def make_optimizers(agent: REEAgent) -> Tuple:
    """Create E1, policy (E3), and E2 optimizers."""
    e1_opt = torch.optim.Adam(list(agent.e1.parameters()), lr=E1_LR)
    policy_opt = torch.optim.Adam(list(agent.e3.parameters()), lr=POLICY_LR)
    e2_opt = torch.optim.Adam(list(agent.e2.parameters()), lr=E2_LR)
    return e1_opt, policy_opt, e2_opt


def run_warmup_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    e2_opt: torch.optim.Optimizer,
    max_steps: int,
) -> float:
    """Run one training episode. Returns total harm."""
    obs = env.reset()
    agent.reset()
    total_harm = 0.0
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    prev_latent_z: Optional[torch.Tensor] = None
    prev_action_tensor: Optional[torch.Tensor] = None

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        action, log_prob = agent.act_with_log_prob(obs_tensor)
        current_z = agent._current_latent.z_gamma.detach().clone()

        if prev_latent_z is not None and prev_action_tensor is not None:
            agent.record_transition(prev_latent_z, prev_action_tensor, current_z)

        action_idx = action.argmax(dim=-1).item()
        next_obs, harm, done, _info = env.step(action_idx)

        if harm < 0:
            total_harm += abs(harm)

        log_probs.append(log_prob)
        rewards.append(-abs(harm) if harm < 0 else harm * 0.1)

        prev_latent_z = current_z
        prev_action_tensor = action.detach().clone()
        obs = next_obs

        if done:
            break

    # Policy (E3) update via REINFORCE — must come before E1 update because
    # log_probs reference E1 parameters (via e1_prior in generate_trajectories).
    # Updating E1 first would invalidate the computation graph for policy_loss.
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

    return total_harm


def run_eval_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one evaluation episode (no training).

    At each step where harm occurs, record:
      - E2 predicted_harm before the step
      - Ground truth transition_type from environment
    """
    obs = env.reset()
    agent.reset()
    total_harm = 0.0
    records: List[Dict[str, Any]] = []

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)

        with torch.no_grad():
            action, _ = agent.act_with_log_prob(obs_tensor)
            z_t = agent._current_latent.z_gamma.detach().clone()
            # Predict harm for the action about to be taken
            predicted_harm = agent.e2.predict_harm(z_t, action).squeeze().item()

        action_idx = action.argmax(dim=-1).item()
        next_obs, harm, done, info = env.step(action_idx)

        if harm < 0:
            total_harm += abs(harm)
            records.append({
                "predicted_harm": predicted_harm,
                "actual_harm": abs(harm),
                "transition_type": info.get("transition_type", "unknown"),
            })

        obs = next_obs
        if done:
            break

    return {"total_harm": total_harm, "harm_records": records}


def compute_calibration_gap(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute calibration_gap = mean_predicted_harm[agent_caused] - mean_predicted_harm[env_caused].
    """
    agent_ph = [r["predicted_harm"] for r in records if r["transition_type"] == "agent_caused_hazard"]
    env_ph = [r["predicted_harm"] for r in records if r["transition_type"] == "env_caused_hazard"]

    mean_agent = statistics.mean(agent_ph) if agent_ph else None
    mean_env = statistics.mean(env_ph) if env_ph else None

    if mean_agent is not None and mean_env is not None:
        gap = mean_agent - mean_env
    else:
        gap = None

    return {
        "mean_predicted_harm_agent_caused": round(mean_agent, 4) if mean_agent is not None else None,
        "mean_predicted_harm_env_caused": round(mean_env, 4) if mean_env is not None else None,
        "calibration_gap": round(gap, 4) if gap is not None else None,
        "n_agent_caused_events": len(agent_ph),
        "n_env_caused_events": len(env_ph),
        "n_total_harm_events": len(records),
    }


def run_condition(
    condition: str,
    num_warmup: int,
    num_eval: int,
    max_steps: int,
    seed: int,
    grid_size: int,
    num_hazards: int,
    num_resources: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Run one (condition, seed) trial. Returns calibration stats."""
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

    warmup_to_run = num_warmup if condition == "TRAINED" else 0

    if verbose and warmup_to_run > 0:
        print(f"    Warmup ({warmup_to_run} eps)...", end="", flush=True)

    for ep in range(warmup_to_run):
        run_warmup_episode(agent, env, e1_opt, policy_opt, e2_opt, max_steps)
        if verbose and (ep + 1) % 50 == 0:
            print(f" {ep+1}", end="", flush=True)

    if verbose and warmup_to_run > 0:
        print(" done")

    # Evaluation
    all_records: List[Dict[str, Any]] = []
    for _ in range(num_eval):
        result = run_eval_episode(agent, env, max_steps)
        all_records.extend(result["harm_records"])

    return compute_calibration_gap(all_records)


def run_experiment(
    num_warmup: int = DEFAULT_WARMUP_EPISODES,
    num_eval: int = DEFAULT_EVAL_EPISODES,
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
        print("[E2 Attribution Calibration — MECH-071 / EVB-0046] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Warmup: {num_warmup}  Eval: {num_eval}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print("    TRAINED — E2 trained for warmup episodes, then evaluated")
        print("    RANDOM  — no warmup, random E2 weights (control)")
        print()
        print(f"  Pass criteria:")
        print(f"    1. TRAINED calibration_gap > {CALIBRATION_GAP_THRESHOLD}")
        print(f"    2. |RANDOM calibration_gap| < {RANDOM_GAP_ABS_MAX}")
        print()

    all_results: List[Dict[str, Any]] = []

    for condition in ["TRAINED", "RANDOM"]:
        if verbose:
            print(f"  --- Condition: {condition} ---")

        seed_gaps: List[float] = []
        seed_results: List[Dict[str, Any]] = []

        for seed in seeds:
            if verbose:
                print(f"  Seed {seed}", end="  ", flush=True)
            stats = run_condition(
                condition, num_warmup, num_eval, max_steps,
                seed, grid_size, num_hazards, num_resources, verbose,
            )
            if verbose:
                gap_str = f"{stats['calibration_gap']:+.4f}" if stats['calibration_gap'] is not None else "N/A"
                print(f"    calibration_gap: {gap_str}  "
                      f"(agent_caused n={stats['n_agent_caused_events']}, "
                      f"env_caused n={stats['n_env_caused_events']})")
            if stats["calibration_gap"] is not None:
                seed_gaps.append(stats["calibration_gap"])
            seed_results.append({"seed": seed, **stats})

        mean_gap = statistics.mean(seed_gaps) if seed_gaps else None
        seed_results_entry = {
            "condition": condition,
            "mean_calibration_gap": round(mean_gap, 4) if mean_gap is not None else None,
            "per_seed": seed_results,
        }
        all_results.append(seed_results_entry)

        if verbose:
            gap_str = f"{mean_gap:+.4f}" if mean_gap is not None else "N/A"
            print(f"  {condition} mean calibration_gap: {gap_str}")
            print()

    # Determine verdict
    trained_result = next(r for r in all_results if r["condition"] == "TRAINED")
    random_result = next(r for r in all_results if r["condition"] == "RANDOM")

    trained_gap = trained_result["mean_calibration_gap"]
    random_gap = random_result["mean_calibration_gap"]

    crit1 = trained_gap is not None and trained_gap > CALIBRATION_GAP_THRESHOLD
    crit2 = random_gap is None or abs(random_gap) < RANDOM_GAP_ABS_MAX

    # Data validity: at least one seed had both transition types
    trained_per_seed = trained_result["per_seed"]
    has_both_types = any(
        s["n_agent_caused_events"] > 0 and s["n_env_caused_events"] > 0
        for s in trained_per_seed
    )
    crit3 = has_both_types

    verdict = "PASS" if (crit1 and crit2 and crit3) else "FAIL"

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  TRAINED mean calibration_gap: "
              f"{trained_gap:+.4f}" if trained_gap is not None else "  TRAINED mean calibration_gap: N/A")
        print(f"  RANDOM  mean calibration_gap: "
              f"{random_gap:+.4f}" if random_gap is not None else "  RANDOM  mean calibration_gap: N/A")
        print()
        print(
            f"  Criterion 1 (TRAINED gap > {CALIBRATION_GAP_THRESHOLD})?  "
            f"{'YES' if crit1 else 'NO'}  ({trained_gap:+.4f})" if trained_gap is not None
            else f"  Criterion 1 (TRAINED gap > {CALIBRATION_GAP_THRESHOLD})?  NO  (None)"
        )
        print(
            f"  Criterion 2 (|RANDOM gap| < {RANDOM_GAP_ABS_MAX})?   "
            f"{'YES' if crit2 else 'NO'}  ({abs(random_gap):.4f})" if random_gap is not None
            else f"  Criterion 2 (|RANDOM gap| < {RANDOM_GAP_ABS_MAX})?   YES  (None)"
        )
        print(f"  Criterion 3 (both types observed)?  {'YES' if crit3 else 'NO'}")
        print()
        print(f"  MECH-071 / EXQ-027 verdict: {verdict}")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    Trained E2 anticipates agent-caused harm better than env-caused harm.")
            print("    Contamination encoding in z_gamma provides discriminating signal.")
            print("    SD-003 substrate calibrated — EXQ-028 (selective residue) can proceed.")
        else:
            print("  Interpretation:")
            print("    E2 cannot reliably distinguish agent-caused from env-caused harm.")
            print("    Consider: longer warmup, environment adjustment, or latent intervention approach.")

    result_doc = {
        "experiment": "causal_attribution_calibration",
        "claim": "MECH-071",
        "evb_id": "EVB-0046",
        "substrate": "ree-v2",
        "run_timestamp": run_timestamp,
        "config": {
            "num_warmup_episodes": num_warmup,
            "num_eval_episodes": num_eval,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "num_resources": num_resources,
            "environment": "CausalGridWorld",
            "calibration_gap_threshold": CALIBRATION_GAP_THRESHOLD,
            "random_gap_abs_max": RANDOM_GAP_ABS_MAX,
            "e1_lr": E1_LR,
            "e2_lr": E2_LR,
            "policy_lr": POLICY_LR,
        },
        "verdict": verdict,
        "criteria_met": sum([crit1, crit2, crit3]),
        "aggregate": {
            "trained_mean_calibration_gap": trained_gap,
            "random_mean_calibration_gap": random_gap,
            "criterion_1_trained_gap_above_threshold": crit1,
            "criterion_2_random_gap_below_max": crit2,
            "criterion_3_both_types_observed": crit3,
        },
        "per_condition": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "causal_attribution_calibration"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"causal_attribution_calibration_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description="E2 Attribution Calibration: does trained E2 discriminate agent-caused harm?"
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_EPISODES,
                        help="Training episodes before evaluation")
    parser.add_argument("--eval", type=int, default=DEFAULT_EVAL_EPISODES,
                        help="Evaluation episodes after warmup")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--num-hazards", type=int, default=DEFAULT_NUM_HAZARDS)
    parser.add_argument("--num-resources", type=int, default=DEFAULT_NUM_RESOURCES)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        num_warmup=args.warmup,
        num_eval=args.eval,
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
