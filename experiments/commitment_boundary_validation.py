"""
Commitment Boundary Validation Experiment (MECH-061 / EVB-0041) — REE-v2

V2 port: CausalGridWorld replaces GridWorld. All logic identical.

Tests whether the commit boundary correctly separates pre-commit simulation errors
(E2 harm predictions) from post-commit realised errors (actual env harm), and whether
keeping them separate is at least as good as blending them into a single signal.

Conditions:
  A (WITH-BOUNDARY): REINFORCE uses only realised env harm (post-commit).
  B (BLENDED):       Return signal mixes E2 prediction (50%) with env harm (50%).

Key diagnostics:
  1. |pre_post_corr WITH-BOUNDARY| < 0.7
  2. WITH-BOUNDARY last-quarter harm <= BLENDED * 1.05

Both must hold for MECH-061 PASS.

Usage:
    python experiments/commitment_boundary_validation.py
    python experiments/commitment_boundary_validation.py --episodes 5 --seeds 7

Claims:
    MECH-061: commitment.boundary_token_error_reclassification
    EVB-0041
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


DEFAULT_EPISODES = 200
DEFAULT_MAX_STEPS = 100
DEFAULT_SEEDS = [7, 42, 99]
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 4
MAX_GRAD_NORM = 1.0
BLEND_ALPHA = 0.5
E1_LR = 1e-4
POLICY_LR = 1e-3
E2_LR = 1e-3


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


def make_optimizers(
    agent: REEAgent,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
    """Build three optimizers: E1 world model, E2 motor model, policy."""
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


def get_e2_predicted_harm(trajectory) -> float:
    """Extract E2's harm prediction for the selected trajectory as a scalar."""
    if trajectory.harm_predictions is None:
        return 0.0
    return trajectory.harm_predictions.mean().item()


def run_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    e2_opt: torch.optim.Optimizer,
    condition: str,
    max_steps: int,
) -> Dict[str, Any]:
    """Run one episode, tracking pre-commit and post-commit harm signals separately."""
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    step_actual_harms: List[float] = []
    step_e2_preds: List[float] = []
    step_pred_errors: List[float] = []
    committed_count = 0
    total_actual_harm = 0.0
    total_blended_harm = 0.0
    steps = 0
    prev_latent_z = None
    prev_action_tensor = None

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

        current_z = agent._current_latent.z_gamma.detach()
        if prev_latent_z is not None and prev_action_tensor is not None:
            agent.record_transition(prev_latent_z, prev_action_tensor, current_z)

        result = agent.e3.select(candidates)
        if result.log_prob is not None:
            log_probs.append(result.log_prob)
        if result.committed:
            committed_count += 1

        e2_pred = get_e2_predicted_harm(result.selected_trajectory)
        step_e2_preds.append(e2_pred)

        action_idx = result.selected_action.argmax(dim=-1).item()
        prev_latent_z = current_z
        prev_action_tensor = result.selected_action.detach()

        next_obs, harm, done, _info = env.step(action_idx)

        actual_harm = abs(harm) if harm < 0 else 0.0
        step_actual_harms.append(actual_harm)
        step_pred_errors.append(abs(actual_harm - e2_pred))

        agent.update_residue(harm)

        total_actual_harm += actual_harm
        blended_step = BLEND_ALPHA * actual_harm + (1.0 - BLEND_ALPHA) * e2_pred
        total_blended_harm += blended_step

        obs = next_obs
        steps += 1
        if done:
            break

    policy_loss_val = 0.0
    if log_probs:
        if condition == "WITH-BOUNDARY":
            G = float(-total_actual_harm)
        else:  # BLENDED
            G = float(-total_blended_harm)

        policy_loss = -(torch.stack(log_probs) * G).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in policy_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        policy_opt.step()
        policy_loss_val = policy_loss.item()

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

    e2_loss = agent.compute_e2_loss()
    if e2_loss.requires_grad:
        e2_opt.zero_grad()
        e2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(agent.e2.parameters()), MAX_GRAD_NORM,
        )
        e2_opt.step()

    mean_actual = statistics.mean(step_actual_harms) if step_actual_harms else 0.0
    mean_e2_pred = statistics.mean(step_e2_preds) if step_e2_preds else 0.0
    mean_pred_error = statistics.mean(step_pred_errors) if step_pred_errors else 0.0
    commit_rate = committed_count / steps if steps > 0 else 0.0

    return {
        "total_harm": total_actual_harm,
        "steps": steps,
        "mean_actual_harm": mean_actual,
        "mean_e2_predicted_harm": mean_e2_pred,
        "mean_prediction_error": mean_pred_error,
        "commit_rate": commit_rate,
        "e1_loss": e1_loss_val,
        "policy_loss": policy_loss_val,
        "_ep_actual": mean_actual,
        "_ep_e2_pred": mean_e2_pred,
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
    e1_opt, policy_opt, e2_opt = make_optimizers(agent)

    ep_harms: List[float] = []
    ep_pred_errors: List[float] = []
    ep_actuals: List[float] = []
    ep_e2_preds: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, e2_opt, condition, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_pred_errors.append(metrics["mean_prediction_error"])
        ep_actuals.append(metrics["_ep_actual"])
        ep_e2_preds.append(metrics["_ep_e2_pred"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            recent_err = statistics.mean(ep_pred_errors[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  "
                f"seed={seed}  cond={condition}  "
                f"harm={recent_harm:.3f}  pred_err={recent_err:.3f}"
            )

    quarter = max(1, num_episodes // 4)
    pre_post_corr = pearson_corr(ep_e2_preds, ep_actuals)

    return {
        "condition": condition,
        "seed": seed,
        "first_quarter_harm": round(statistics.mean(ep_harms[:quarter]), 4),
        "last_quarter_harm": round(statistics.mean(ep_harms[-quarter:]), 4),
        "mean_prediction_error": round(statistics.mean(ep_pred_errors), 6),
        "pre_post_corr": round(pre_post_corr, 4),
        "abs_pre_post_corr": round(abs(pre_post_corr), 4),
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
        print("[Commitment Boundary Validation — MECH-061 / EVB-0041] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print("    A (WITH-BOUNDARY): REINFORCE uses realized env harm only (post-commit)")
        print("    B (BLENDED):       Return = 50% actual harm + 50% E2 prediction")
        print()
        print("  Diagnostic 1: |pre_post_corr WITH-BOUNDARY| < 0.7")
        print("  Diagnostic 2: WITH-BOUNDARY last-Q harm <= BLENDED * 1.05")
        print()

    all_results = []

    for seed in seeds:
        for condition in ["WITH-BOUNDARY", "BLENDED"]:
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
                    f"|corr|={result['abs_pre_post_corr']:.3f}"
                )
                print()

    with_boundary = [r for r in all_results if r["condition"] == "WITH-BOUNDARY"]
    blended = [r for r in all_results if r["condition"] == "BLENDED"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    wb_harm_last = _agg(with_boundary, "last_quarter_harm")
    bl_harm_last = _agg(blended, "last_quarter_harm")
    mean_abs_corr = round(
        statistics.mean(r["abs_pre_post_corr"] for r in with_boundary), 4
    )

    distinct_ok = mean_abs_corr < 0.7
    boundary_helps = wb_harm_last <= bl_harm_last * 1.05
    verdict = "PASS" if (distinct_ok and boundary_helps) else "FAIL"
    partial = (distinct_ok or boundary_helps) and not (distinct_ok and boundary_helps)

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  WITH-BOUNDARY  last-Q harm: {wb_harm_last:.3f}")
        print(f"  BLENDED        last-Q harm: {bl_harm_last:.3f}")
        print(f"  |pre_post_corr| WITH-BOUNDARY (mean): {mean_abs_corr:.3f}")
        print()
        print(f"  Distinct signals (|corr| < 0.7)?   {'YES' if distinct_ok else 'NO'}  ({mean_abs_corr:.3f})")
        print(f"  Boundary neutral or better?         {'YES' if boundary_helps else 'NO'}")
        print()
        print(f"  MECH-061 verdict: {verdict}")
        if partial:
            print("  (partial — one of two criteria met)")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    Pre/post-commit signals carry distinct information on V2 substrate.")
            print("    MECH-061 structural requirement confirmed.")

    result_doc = {
        "experiment": "commitment_boundary_validation",
        "claim": "MECH-061",
        "evb_id": "EVB-0041",
        "substrate": "ree-v2",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "environment": "CausalGridWorld",
            "blend_alpha": BLEND_ALPHA,
            "e1_lr": E1_LR,
            "policy_lr": POLICY_LR,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "with_boundary_harm_last_quarter": wb_harm_last,
            "blended_harm_last_quarter": bl_harm_last,
            "mean_abs_pre_post_corr_with_boundary": mean_abs_corr,
            "corr_threshold": 0.7,
            "harm_tolerance_factor": 1.05,
            "distinct_signals_criterion_met": distinct_ok,
            "boundary_helps_criterion_met": boundary_helps,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "commitment_boundary_validation"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"commitment_boundary_validation_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description="MECH-061: Commitment Boundary Validation (REE-v2)"
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
