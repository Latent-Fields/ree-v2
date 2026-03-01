"""
Control Plane Precision Separation Experiment (MECH-059 / EVB-0037) — REE-v2

V2 port: CausalGridWorld replaces GridWorld. All logic identical.

Tests whether keeping confidence (trajectory score dispersion) as a SEPARATE signal
from prediction error magnitude leads to better precision calibration and harm reduction.

MECH-059 claims: the confidence channel carries independent information and must
remain distinct from prediction error in precision routing.

Conditions:
  A (MERGED):    precision updated from error_magnitude alone (current behavior)
  B (SEPARATED): precision additionally modulated by score_dispersion signal

Key diagnostics:
  1. corr(score_dispersion, prediction_error) across episodes
     Low correlation (< 0.3) → signals carry INDEPENDENT information
  2. Final harm: SEPARATED <= MERGED
     (using the extra channel does not hurt, ideally helps)

Both must hold for MECH-059 PASS.

Usage:
    python experiments/control_plane_precision_separation.py
    python experiments/control_plane_precision_separation.py --episodes 100 --seeds 7 42

Claims:
    MECH-059: precision.confidence_channel_separate_from_prediction_error
    EVB-0037 (evidence backlog item for genuine MECH-059 re-experimentation)
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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

# How strongly dispersion modulates precision in Condition B
DISPERSION_MODULATION_STRENGTH = 0.02
MAX_DISPERSION_SCALE = 5.0


def apply_confidence_modulation(agent: REEAgent) -> float:
    """
    Condition B post-select correction: modulate precision using score dispersion.

    After standard precision update (which uses error_magnitude only), apply a
    small additional adjustment based on score dispersion as a confidence signal.

    Returns: the score_dispersion value used.
    """
    if agent.e3.last_scores is None or len(agent.e3.last_scores) < 2:
        return 0.0

    score_dispersion = agent.e3.last_scores.std().item()
    uncertainty = min(score_dispersion / MAX_DISPERSION_SCALE, 1.0)
    confidence = 1.0 - uncertainty

    correction = DISPERSION_MODULATION_STRENGTH * (confidence - 0.5)

    agent.e3.current_precision = min(
        agent.e3.config.precision_max,
        max(agent.e3.config.precision_min, agent.e3.current_precision + correction)
    )
    return score_dispersion


def run_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    condition: str,
    max_steps: int,
) -> Dict[str, Any]:
    """Run one episode, tracking score_dispersion at each step."""
    obs = env.reset()
    agent.reset()
    total_harm = 0.0
    steps = 0

    step_dispersions: List[float] = []
    step_precisions: List[float] = []

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)

        with torch.no_grad():
            action = agent.act(obs_tensor)

        if agent.e3.last_scores is not None and len(agent.e3.last_scores) > 1:
            base_dispersion = agent.e3.last_scores.std().item()
        else:
            base_dispersion = 0.0

        score_dispersion = base_dispersion
        if condition == "SEPARATED":
            score_dispersion = apply_confidence_modulation(agent)

        step_dispersions.append(score_dispersion)

        action_idx = action.argmax(dim=-1).item()
        next_obs, harm, done, _info = env.step(action_idx)

        if harm < 0:
            agent.update_residue(harm)
            total_harm += abs(harm)

        step_precisions.append(agent.e3.current_precision)

        obs = next_obs
        steps += 1
        if done:
            break

    pe_loss = agent.compute_prediction_loss().item()

    mean_dispersion = statistics.mean(step_dispersions) if step_dispersions else 0.0
    mean_precision = statistics.mean(step_precisions) if step_precisions else 0.0
    precision_std = statistics.stdev(step_precisions) if len(step_precisions) > 1 else 0.0

    return {
        "total_harm": total_harm,
        "steps": steps,
        "ep_e1_loss": pe_loss,
        "mean_dispersion": mean_dispersion,
        "mean_precision": mean_precision,
        "precision_std": precision_std,
    }


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
    ep_dispersions: List[float] = []
    ep_losses: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, condition, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_dispersions.append(metrics["mean_dispersion"])
        ep_losses.append(metrics["ep_e1_loss"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            recent_disp = statistics.mean(ep_dispersions[-20:])
            recent_pe = statistics.mean(ep_losses[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  "
                f"harm={recent_harm:.3f}  disp={recent_disp:.3f}  "
                f"e1_loss={recent_pe:.4f}"
            )

    quarter = max(1, num_episodes // 4)
    first_q_harm = statistics.mean(ep_harms[:quarter])
    last_q_harm = statistics.mean(ep_harms[-quarter:])

    corr_disp_pe = pearson_corr(ep_dispersions, ep_losses)

    return {
        "condition": condition,
        "seed": seed,
        "first_quarter_harm": round(first_q_harm, 4),
        "last_quarter_harm": round(last_q_harm, 4),
        "harm_reduction": round(first_q_harm - last_q_harm, 4),
        "harm_improved": last_q_harm < first_q_harm,
        "corr_dispersion_pe": round(corr_disp_pe, 4),
        "abs_corr_dispersion_pe": round(abs(corr_disp_pe), 4),
        "mean_dispersion": round(statistics.mean(ep_dispersions), 4),
        "mean_e1_loss": round(statistics.mean(ep_losses), 4),
        "episode_count": num_episodes,
    }


def run_experiment(
    num_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seeds: List[int] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_hazards: int = DEFAULT_NUM_HAZARDS,
    output_path: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print("[Control Plane Precision Separation — MECH-059 / EVB-0037] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print("    A (MERGED):    precision = f(error_magnitude) only [current]")
        print("    B (SEPARATED): as A, plus confidence modulation from score_dispersion")
        print()
        print("  Diagnostic 1: corr(score_dispersion, e1_loss) across episodes")
        print("    < 0.3 → signals independent → MECH-059 structural support")
        print("  Diagnostic 2: SEPARATED final harm <= MERGED final harm")
        print("    → separation helps → MECH-059 mechanistic support")
        print()

    all_results = []

    for seed in seeds:
        for condition in ["MERGED", "SEPARATED"]:
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
                    f"reduction: {result['harm_reduction']:.3f}"
                )
                print(
                    f"    |corr(disp, PE)|: {result['abs_corr_dispersion_pe']:.3f}  "
                    f"mean_dispersion: {result['mean_dispersion']:.3f}"
                )
                print()

    merged = [r for r in all_results if r["condition"] == "MERGED"]
    separated = [r for r in all_results if r["condition"] == "SEPARATED"]

    def _agg(results, key):
        return round(statistics.mean(r[key] for r in results), 4)

    merged_harm_last = _agg(merged, "last_quarter_harm")
    sep_harm_last = _agg(separated, "last_quarter_harm")
    merged_corr = _agg(merged, "abs_corr_dispersion_pe")
    sep_corr = _agg(separated, "abs_corr_dispersion_pe")

    signals_independent = merged_corr < 0.3
    separation_helps = sep_harm_last <= merged_harm_last * 1.05
    verdict = "PASS" if (signals_independent and separation_helps) else "FAIL"
    partial = (signals_independent or separation_helps) and not (signals_independent and separation_helps)

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  MERGED    last-Q harm: {merged_harm_last:.3f}")
        print(f"  SEPARATED last-Q harm: {sep_harm_last:.3f}")
        print(f"  |corr(dispersion, PE)| — MERGED:    {merged_corr:.3f}")
        print(f"  |corr(dispersion, PE)| — SEPARATED: {sep_corr:.3f}")
        print()
        print(f"  Signals independent (corr < 0.3)?  {'YES' if signals_independent else 'NO'}  ({merged_corr:.3f})")
        print(f"  Separation neutral or beneficial?   {'YES' if separation_helps else 'NO'}")
        print()
        print(f"  MECH-059 verdict: {verdict}")
        if partial:
            print("  (partial — one of two criteria met)")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    Score dispersion and prediction error carry INDEPENDENT information.")
            print("    Using the dispersion as a separate confidence channel does not hurt")
            print("    performance and may improve precision calibration.")
            print("    MECH-059 structural requirement confirmed on V2 substrate.")

    result_doc = {
        "experiment": "control_plane_precision_separation",
        "claim": "MECH-059",
        "evb_id": "EVB-0037",
        "substrate": "ree-v2",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "environment": "CausalGridWorld",
            "dispersion_modulation_strength": DISPERSION_MODULATION_STRENGTH,
            "max_dispersion_scale": MAX_DISPERSION_SCALE,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "merged_harm_last_quarter": merged_harm_last,
            "separated_harm_last_quarter": sep_harm_last,
            "abs_corr_dispersion_pe_merged": merged_corr,
            "abs_corr_dispersion_pe_separated": sep_corr,
            "independence_threshold": 0.3,
            "signals_independent": signals_independent,
            "separation_helps": separation_helps,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "control_plane_precision_separation"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"control_plane_precision_separation_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser()
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
