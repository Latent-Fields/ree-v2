"""
Precision Regime Probe Experiment (ARC-016 / EVB-0024) — REE-v2

Tests ARC-016: cognitive_modes.control_plane_regimes

ARC-016 asserts that distinct cognitive modes emerge from distinct
control-plane regimes applied to the SAME underlying predictive machinery.
Modes are not separate modules — they are patterns of tuning: changes in
gain, horizon, suppression, learning eligibility, and hippocampal gating.
Modes are labels over stable regions of control-channel space.

The most directly observable control-channel parameter in REE is E3 precision
(agent.e3.current_precision), which controls how selectively the agent commits
to trajectories.  ARC-016 predicts that distinct precision regimes produce
measurably distinct behavioral signatures even though the underlying agent
architecture is identical.

This is a targeted probe (mode=targeted_probe): no pre-registered thresholds
required.  We probe two precision regimes, observe their behavioral profiles,
and check that the regimes are distinguishable.

Conditions (2):
  HIGH_REGIME : precision_max=2.0, precision_min=1.0
                Agent operates in high-selectivity mode.
                Predicts: lower commit_rate, lower harm when committed.

  LOW_REGIME  : precision_max=0.5, precision_min=0.1
                Agent operates in low-selectivity mode.
                Predicts: higher commit_rate, higher harm (less selective).

Key diagnostics:
  1. |mean_precision(HIGH) - mean_precision(LOW)| > 0.30
     (regimes are operationally distinct in the control channel)

  2. commit_rate(HIGH) < commit_rate(LOW) - 0.05
     OR harm(HIGH) < harm(LOW) * 0.95
     (regime distinction produces at least one measurable behavioral difference)

Pass: both criteria met.

Usage:
    python experiments/precision_regime_probe.py
    python experiments/precision_regime_probe.py --episodes 5 --seeds 7

Claims:
    ARC-016: cognitive_modes.control_plane_regimes
    EVB-0024
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

CONDITION_PRECISION: Dict[str, Tuple[float, float]] = {
    "HIGH_REGIME": (2.0, 1.0),   # (precision_max, precision_min)
    "LOW_REGIME":  (0.5, 0.1),
}

# ── Pass thresholds ───────────────────────────────────────────────────────────

PRECISION_GAP: float = 0.30      # criterion 1: |HIGH - LOW| mean precision
COMMIT_RATE_MARGIN: float = 0.05 # criterion 2a: HIGH commit_rate < LOW - margin
HARM_FACTOR: float = 0.95        # criterion 2b: HIGH harm < LOW * factor


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
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    committed_count = 0
    precisions: List[float] = []
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
        if getattr(result, "committed", False):
            committed_count += 1

        try:
            precisions.append(float(agent.e3.current_precision))
        except AttributeError:
            precisions.append(1.0)

        action_idx = result.selected_action.argmax(dim=-1).item()
        next_obs, harm, done, _info = env.step(action_idx)

        actual_harm = abs(harm) if harm < 0 else 0.0
        total_harm += actual_harm

        agent.update_residue(harm)
        obs = next_obs
        steps += 1
        if done:
            break

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

    commit_rate = committed_count / max(1, steps)
    mean_precision = statistics.mean(precisions) if precisions else 1.0

    return {
        "total_harm": total_harm,
        "commit_rate": commit_rate,
        "mean_precision": mean_precision,
        "steps": steps,
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
    precision_max, precision_min = CONDITION_PRECISION[condition]
    torch.manual_seed(seed)
    env = CausalGridWorld(size=grid_size, num_hazards=num_hazards)
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)

    # Apply precision regime
    try:
        agent.e3.config.precision_max = precision_max
        agent.e3.config.precision_min = precision_min
        agent.e3.current_precision = (precision_max + precision_min) / 2.0
    except AttributeError:
        pass  # Config not directly settable; proceed with defaults

    e1_opt, policy_opt = make_optimizers(agent)

    ep_harms: List[float] = []
    ep_commit_rates: List[float] = []
    ep_precisions: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_commit_rates.append(metrics["commit_rate"])
        ep_precisions.append(metrics["mean_precision"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            recent_prec = statistics.mean(ep_precisions[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  seed={seed}  cond={condition}  "
                f"harm={recent_harm:.3f}  precision={recent_prec:.3f}"
            )

    quarter = max(1, num_episodes // 4)
    return {
        "condition": condition,
        "seed": seed,
        "precision_max": precision_max,
        "precision_min": precision_min,
        "mean_precision": round(statistics.mean(ep_precisions), 4),
        "last_quarter_precision": round(statistics.mean(ep_precisions[-quarter:]), 4),
        "mean_commit_rate": round(statistics.mean(ep_commit_rates), 4),
        "last_quarter_commit_rate": round(statistics.mean(ep_commit_rates[-quarter:]), 4),
        "last_quarter_harm": round(statistics.mean(ep_harms[-quarter:]), 4),
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
        print("[Precision Regime Probe — ARC-016 / EVB-0024] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions (targeted probe):")
        print(f"    HIGH_REGIME: precision_max={CONDITION_PRECISION['HIGH_REGIME'][0]}  "
              f"precision_min={CONDITION_PRECISION['HIGH_REGIME'][1]}")
        print(f"    LOW_REGIME : precision_max={CONDITION_PRECISION['LOW_REGIME'][0]}  "
              f"precision_min={CONDITION_PRECISION['LOW_REGIME'][1]}")
        print()
        print("  Diagnostics:")
        print(f"    1. |HIGH - LOW| mean_precision > {PRECISION_GAP}")
        print(f"    2. HIGH commit_rate < LOW - {COMMIT_RATE_MARGIN}  OR  HIGH harm < LOW * {HARM_FACTOR}")
        print()

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for condition in ["HIGH_REGIME", "LOW_REGIME"]:
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
                    f"    precision={result['mean_precision']:.3f}  "
                    f"commit_rate={result['mean_commit_rate']:.3f}  "
                    f"harm(last-Q)={result['last_quarter_harm']:.3f}"
                )
                print()

    high = [r for r in all_results if r["condition"] == "HIGH_REGIME"]
    low = [r for r in all_results if r["condition"] == "LOW_REGIME"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    high_prec = _agg(high, "mean_precision")
    low_prec = _agg(low, "mean_precision")
    high_commit = _agg(high, "last_quarter_commit_rate")
    low_commit = _agg(low, "last_quarter_commit_rate")
    high_harm = _agg(high, "last_quarter_harm")
    low_harm = _agg(low, "last_quarter_harm")

    crit_1 = abs(high_prec - low_prec) > PRECISION_GAP
    crit_2 = (high_commit < low_commit - COMMIT_RATE_MARGIN) or (high_harm < low_harm * HARM_FACTOR)

    num_met = sum([crit_1, crit_2])
    verdict = "PASS" if num_met >= 2 else "FAIL"
    partial = num_met == 1

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  HIGH_REGIME  precision={high_prec:.3f}  commit={high_commit:.3f}  harm={high_harm:.3f}")
        print(f"  LOW_REGIME   precision={low_prec:.3f}  commit={low_commit:.3f}  harm={low_harm:.3f}")
        print()
        print(
            f"  Crit 1  |{high_prec:.3f} - {low_prec:.3f}| = {abs(high_prec-low_prec):.3f} > {PRECISION_GAP}  "
            f"{'MET' if crit_1 else 'MISSED'}"
        )
        print(
            f"  Crit 2  commit {high_commit:.3f}<{low_commit-COMMIT_RATE_MARGIN:.3f}? "
            f"{'Y' if high_commit < low_commit - COMMIT_RATE_MARGIN else 'N'}  OR  "
            f"harm {high_harm:.3f}<{low_harm*HARM_FACTOR:.3f}? "
            f"{'Y' if high_harm < low_harm * HARM_FACTOR else 'N'}  "
            f"{'MET' if crit_2 else 'MISSED'}"
        )
        print()
        print(f"  Criteria met: {num_met}/2  ->  ARC-016 verdict: {verdict}")
        if partial:
            print("  (partial -- 1 of 2 criteria met)")

    result_doc: Dict[str, Any] = {
        "experiment": "precision_regime_probe",
        "claim": "ARC-016",
        "evb_id": "EVB-0024",
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
            "high_precision_max": CONDITION_PRECISION["HIGH_REGIME"][0],
            "high_precision_min": CONDITION_PRECISION["HIGH_REGIME"][1],
            "low_precision_max": CONDITION_PRECISION["LOW_REGIME"][0],
            "low_precision_min": CONDITION_PRECISION["LOW_REGIME"][1],
            "precision_gap": PRECISION_GAP,
            "commit_rate_margin": COMMIT_RATE_MARGIN,
            "harm_factor": HARM_FACTOR,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "high_mean_precision": high_prec,
            "low_mean_precision": low_prec,
            "precision_gap_observed": round(abs(high_prec - low_prec), 4),
            "high_commit_rate_last_quarter": high_commit,
            "low_commit_rate_last_quarter": low_commit,
            "high_harm_last_quarter": high_harm,
            "low_harm_last_quarter": low_harm,
            "criterion_1_precision_gap_met": crit_1,
            "criterion_2_behavioral_distinction_met": crit_2,
            "criteria_met": num_met,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "precision_regime_probe"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(evidence_dir / f"precision_regime_probe_{ts}.json")
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARC-016: Precision Regime Probe (REE-v2)"
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
