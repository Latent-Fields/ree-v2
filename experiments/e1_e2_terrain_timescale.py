"""
E1/E2 Terrain-Timescale Separation Experiment (MECH-058 / EVB-0046) — REE-v2

Ground-up redesign exploiting CausalGridWorld's two natural timescales.

MECH-058 predicts that E1 (world-model) and the action-policy module must
operate on separated timescales to correctly model each signal class.
CausalGridWorld exposes this requirement directly:

  SLOW timescale — contamination terrain
    Contamination accumulates across many agent visits.  A world-model that
    learns too fast overwrites stable terrain knowledge with recent noise.
    Optimal E1 lr: slow (5e-5).

  FAST timescale — background env drift
    Active hazards relocate every env_drift_interval steps (default 5).
    A policy that learns too slowly cannot track where hazards moved.
    Optimal policy lr: fast (1e-3).

Conditions (3):
  SEPARATED      : e1_lr=5e-5, policy_lr=1e-3  (each matched to its timescale)
  COLLAPSED_FAST : e1_lr=1e-3, policy_lr=1e-3  (both fast — E1 terrain model unstable)
  COLLAPSED_SLOW : e1_lr=5e-5, policy_lr=5e-5  (both slow — policy sluggish on drift)

New metrics (exploiting CausalGridWorld info dict):

  contamination_avoidance_rate
    = 1 − (agent_caused_hazard steps / total steps)
    Measures how well the E1 terrain model guides path planning.
    Slow E1 → stable terrain knowledge → fewer revisits of contaminated cells.
    Expected ranking: SEPARATED > COLLAPSED_FAST + 0.03

  post_drift_harm_rate
    = env_caused_hazard steps within POST_DRIFT_WINDOW of each drift event
      / total drift events
    Measures how quickly the policy adapts after hazards relocate.
    Fast policy → rapid drift response → fewer hazard collisions post-drift.
    Expected ranking: SEPARATED < COLLAPSED_SLOW − 0.02

Pass criterion: ≥ 2 of 3 must hold:
  1. SEPARATED last-Q contamination_avoidance_rate
       > COLLAPSED_FAST last-Q contamination_avoidance_rate + 0.03
  2. SEPARATED last-Q post_drift_harm_rate
       < COLLAPSED_SLOW last-Q post_drift_harm_rate − 0.02
  3. SEPARATED last-Q harm ≤ min(COLLAPSED_FAST, COLLAPSED_SLOW) last-Q harm × 1.10

Usage:
    python experiments/e1_e2_terrain_timescale.py
    python experiments/e1_e2_terrain_timescale.py --episodes 5 --seeds 7

Claims:
    MECH-058: e1_e2.timescale_separation
    EVB-0046
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

# Steps after a drift event counted toward post_drift_harm_rate
POST_DRIFT_WINDOW = 3

# ── Condition learning rates ──────────────────────────────────────────────────

E1_LR_SLOW: float = 5e-5      # slow — matches contamination terrain timescale
POLICY_LR_FAST: float = 1e-3  # fast — matches env drift timescale

E2_LR: float = 1e-3  # E2 is a fast motor-sensory model; fixed across all conditions

CONDITION_LRS: Dict[str, Tuple[float, float]] = {
    "SEPARATED":      (E1_LR_SLOW, POLICY_LR_FAST),
    "COLLAPSED_FAST": (1e-3, 1e-3),
    "COLLAPSED_SLOW": (5e-5, 5e-5),
}

# ── Pass thresholds ───────────────────────────────────────────────────────────

AVOIDANCE_MARGIN: float = 0.03   # criterion 1: SEPARATED avoidance > CF + margin
DRIFT_HARM_MARGIN: float = 0.02  # criterion 2: SEPARATED drift harm < CS − margin
HARM_TOLERANCE: float = 1.10     # criterion 3: SEPARATED harm ≤ min_collapsed × tol


# ── Optimizer factory ─────────────────────────────────────────────────────────

def make_optimizers(
    agent: REEAgent,
    e1_lr: float,
    policy_lr: float,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
    """Separate optimizers for E1 world-model, E3 policy, and E2 transition model."""
    e1_params = (
        list(agent.e1.parameters())
        + list(agent.latent_stack.parameters())
        + list(agent.obs_encoder.parameters())
    )
    policy_params = list(agent.e3.parameters())
    e1_opt = torch.optim.Adam(e1_params, lr=e1_lr)
    policy_opt = torch.optim.Adam(policy_params, lr=policy_lr)
    e2_opt = torch.optim.Adam(list(agent.e2.parameters()), lr=E2_LR)
    return e1_opt, policy_opt, e2_opt


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    e2_opt: torch.optim.Optimizer,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode.

    Tracks two CausalGridWorld-specific metrics:
      contamination_avoidance_rate : 1 − (agent_caused_hazard steps / total steps)
      post_drift_harm_rate         : env_caused_hazard steps in post-drift window
                                     / total drift events
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    agent_caused_harm_steps = 0
    steps = 0
    drift_events = 0
    post_drift_countdown = 0
    post_drift_env_harm_steps = 0

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
        next_obs, harm, done, info = env.step(action_idx)

        transition_type = info.get("transition_type", "none")
        actual_harm = abs(harm) if harm < 0 else 0.0
        total_harm += actual_harm

        if transition_type == "agent_caused_hazard":
            agent_caused_harm_steps += 1

        # Track drift events and env-caused harm in the post-drift window
        if info.get("env_drift_occurred", False):
            drift_events += 1
            post_drift_countdown = POST_DRIFT_WINDOW

        if post_drift_countdown > 0:
            if transition_type == "env_caused_hazard":
                post_drift_env_harm_steps += 1
            post_drift_countdown -= 1

        agent.update_residue(harm)
        prev_latent_z = z_t
        prev_action_tensor = action_tensor
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

    # ── E1 world-model update ──
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

    # ── E2 motor-sensory update ──
    e2_loss = agent.compute_e2_loss()
    if e2_loss.requires_grad:
        e2_opt.zero_grad()
        e2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in e2_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        e2_opt.step()

    contamination_avoidance_rate = 1.0 - agent_caused_harm_steps / max(1, steps)
    post_drift_harm_rate = post_drift_env_harm_steps / max(1, drift_events)

    return {
        "total_harm": total_harm,
        "agent_caused_harm_steps": agent_caused_harm_steps,
        "steps": steps,
        "drift_events": drift_events,
        "contamination_avoidance_rate": contamination_avoidance_rate,
        "post_drift_harm_rate": post_drift_harm_rate,
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
    verbose: bool = True,
) -> Dict[str, Any]:
    e1_lr, policy_lr = CONDITION_LRS[condition]
    torch.manual_seed(seed)
    env = CausalGridWorld(size=grid_size, num_hazards=num_hazards)
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)
    e1_opt, policy_opt, e2_opt = make_optimizers(agent, e1_lr, policy_lr)

    ep_harms: List[float] = []
    ep_avoidances: List[float] = []
    ep_drift_rates: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, e2_opt, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_avoidances.append(metrics["contamination_avoidance_rate"])
        ep_drift_rates.append(metrics["post_drift_harm_rate"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            recent_avoid = statistics.mean(ep_avoidances[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  "
                f"seed={seed}  cond={condition}  "
                f"harm={recent_harm:.3f}  avoid={recent_avoid:.3f}"
            )

    quarter = max(1, num_episodes // 4)
    return {
        "condition": condition,
        "seed": seed,
        "e1_lr": e1_lr,
        "policy_lr": policy_lr,
        "first_quarter_harm": round(statistics.mean(ep_harms[:quarter]), 4),
        "last_quarter_harm": round(statistics.mean(ep_harms[-quarter:]), 4),
        "mean_contamination_avoidance_rate": round(
            statistics.mean(ep_avoidances), 4
        ),
        "last_quarter_contamination_avoidance_rate": round(
            statistics.mean(ep_avoidances[-quarter:]), 4
        ),
        "mean_post_drift_harm_rate": round(statistics.mean(ep_drift_rates), 4),
        "last_quarter_post_drift_harm_rate": round(
            statistics.mean(ep_drift_rates[-quarter:]), 4
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
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print("[E1/E2 Terrain-Timescale Separation — MECH-058 / EVB-0046] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print(f"    SEPARATED      : e1_lr={E1_LR_SLOW:.1e}, policy_lr={POLICY_LR_FAST:.1e}")
        print( "    COLLAPSED_FAST : e1_lr=1e-3,  policy_lr=1e-3")
        print( "    COLLAPSED_SLOW : e1_lr=5e-5,  policy_lr=5e-5")
        print()
        print("  Diagnostics:")
        print(f"    1. SEPARATED last-Q avoidance > COLLAPSED_FAST + {AVOIDANCE_MARGIN:.2f}")
        print(f"    2. SEPARATED last-Q drift harm < COLLAPSED_SLOW - {DRIFT_HARM_MARGIN:.2f}")
        print(f"    3. SEPARATED last-Q harm <= min_collapsed * {HARM_TOLERANCE:.2f}")
        print()

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for condition in ["SEPARATED", "COLLAPSED_FAST", "COLLAPSED_SLOW"]:
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
                    f"avoid(last-Q)={result['last_quarter_contamination_avoidance_rate']:.3f}  "
                    f"drift_harm(last-Q)={result['last_quarter_post_drift_harm_rate']:.3f}"
                )
                print()

    separated = [r for r in all_results if r["condition"] == "SEPARATED"]
    collapsed_fast = [r for r in all_results if r["condition"] == "COLLAPSED_FAST"]
    collapsed_slow = [r for r in all_results if r["condition"] == "COLLAPSED_SLOW"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    sep_harm_last = _agg(separated, "last_quarter_harm")
    cf_harm_last = _agg(collapsed_fast, "last_quarter_harm")
    cs_harm_last = _agg(collapsed_slow, "last_quarter_harm")
    min_collapsed_harm = min(cf_harm_last, cs_harm_last)

    sep_avoid_last = _agg(separated, "last_quarter_contamination_avoidance_rate")
    cf_avoid_last = _agg(collapsed_fast, "last_quarter_contamination_avoidance_rate")

    sep_drift_last = _agg(separated, "last_quarter_post_drift_harm_rate")
    cs_drift_last = _agg(collapsed_slow, "last_quarter_post_drift_harm_rate")

    # Criterion 1: slow E1 builds better terrain model than fast E1
    crit_1 = sep_avoid_last > cf_avoid_last + AVOIDANCE_MARGIN
    # Criterion 2: fast policy adapts to drift better than slow policy
    crit_2 = sep_drift_last < cs_drift_last - DRIFT_HARM_MARGIN
    # Criterion 3: timescale separation maintains overall performance
    crit_3 = sep_harm_last <= min_collapsed_harm * HARM_TOLERANCE

    num_met = sum([crit_1, crit_2, crit_3])
    verdict = "PASS" if num_met >= 2 else "FAIL"
    partial = num_met == 1

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  SEPARATED      harm (last-Q): {sep_harm_last:.3f}")
        print(f"  COLLAPSED_FAST harm (last-Q): {cf_harm_last:.3f}")
        print(f"  COLLAPSED_SLOW harm (last-Q): {cs_harm_last:.3f}")
        print()
        print(
            f"  Crit 1  avoidance  SEP={sep_avoid_last:.3f}  CF={cf_avoid_last:.3f}  "
            f"delta={sep_avoid_last - cf_avoid_last:+.3f}  "
            f"(need >+{AVOIDANCE_MARGIN:.2f})  {'MET' if crit_1 else 'MISSED'}"
        )
        print(
            f"  Crit 2  drift harm SEP={sep_drift_last:.3f}  CS={cs_drift_last:.3f}  "
            f"delta={sep_drift_last - cs_drift_last:+.3f}  "
            f"(need <-{DRIFT_HARM_MARGIN:.2f})  {'MET' if crit_2 else 'MISSED'}"
        )
        print(
            f"  Crit 3  harm       SEP={sep_harm_last:.3f}  "
            f"<= {min_collapsed_harm:.3f}*{HARM_TOLERANCE}={min_collapsed_harm * HARM_TOLERANCE:.3f}  "
            f"{'MET' if crit_3 else 'MISSED'}"
        )
        print()
        print(f"  Criteria met: {num_met}/3  ->  MECH-058 verdict: {verdict}")
        if partial:
            print("  (partial -- exactly 1 of 3 criteria met)")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    E1 timescale (slow) and policy timescale (fast) are functionally")
            print("    distinct: each benefits from a learning rate matched to its signal.")
            print("    MECH-058 timescale separation confirmed on V2 substrate.")

    result_doc: Dict[str, Any] = {
        "experiment": "e1_e2_terrain_timescale",
        "claim": "MECH-058",
        "evb_id": "EVB-0046",
        "substrate": "ree-v2",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "environment": "CausalGridWorld",
            "e1_lr_slow": E1_LR_SLOW,
            "policy_lr_fast": POLICY_LR_FAST,
            "post_drift_window": POST_DRIFT_WINDOW,
            "avoidance_margin": AVOIDANCE_MARGIN,
            "drift_harm_margin": DRIFT_HARM_MARGIN,
            "harm_tolerance": HARM_TOLERANCE,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "separated_harm_last_quarter": sep_harm_last,
            "collapsed_fast_harm_last_quarter": cf_harm_last,
            "collapsed_slow_harm_last_quarter": cs_harm_last,
            "separated_avoidance_rate_last_quarter": sep_avoid_last,
            "collapsed_fast_avoidance_rate_last_quarter": cf_avoid_last,
            "separated_post_drift_harm_rate_last_quarter": sep_drift_last,
            "collapsed_slow_post_drift_harm_rate_last_quarter": cs_drift_last,
            "criterion_1_terrain_timescale_met": crit_1,
            "criterion_2_drift_timescale_met": crit_2,
            "criterion_3_overall_performance_met": crit_3,
            "criteria_met": num_met,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "e1_e2_terrain_timescale"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"e1_e2_terrain_timescale_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MECH-058: E1/E2 Terrain-Timescale Separation (REE-v2)"
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
