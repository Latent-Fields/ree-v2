"""
Valence-Regime Correlation Experiment (Q-007 / EVB-0006) — REE-v2

Tests Q-007: emotion.universal_expression_channel_mapping

Q-007 is an open question: do universal-looking expressions (e.g., pride,
victory) correspond to STABLE multi-channel control regimes in REE — and if
so, which combinations of arousal, readiness, precision, and valence align
with observed universals?

CausalGridWorld gives us a tractable proxy: the homeostatic state (health,
energy) serves as the agent's internal valence signal — high health/energy
approximates positive valence; depleted states approximate negative valence.
The E3 precision is the most directly observable control-plane parameter.

If Q-007's hypothesis is correct, there should be a stable, positive
correlation between internal valence (health) and E3 precision across
episodes — i.e., healthier agents should operate with higher precision,
reflecting a more selective, "confident" control-plane regime.

Conditions (2 — discriminative pair):
  RESOURCE_RICH   : num_resources=10  — agent stays healthy most episodes.
                    High valence is achievable and sustained.
  RESOURCE_SPARSE : num_resources=1   — resources rarely available.
                    Valence tends negative; health degrades.

Key diagnostics:
  Per episode: track mean_health (valence proxy) and mean_precision (control
  channel).  Aggregate across episodes to compute Pearson correlation.

  1. valence_precision_corr (RESOURCE_RICH) > 0.10
     (positive valence state corresponds to higher precision regime)

  2. mean_precision(RESOURCE_RICH) > mean_precision(RESOURCE_SPARSE) + 0.10
     (richer valence environment produces a distinctly higher precision regime)

Pass: >= 2 of 2 criteria met.
Partial: 1 of 2 criteria met.

Note: Q-007 is an open question, not an architectural commitment.  A PASS
here is evidence FOR the mapping hypothesis; a FAIL is evidence that the
mapping may not hold or may require additional channels beyond precision.

Usage:
    python experiments/valence_regime_correlation.py
    python experiments/valence_regime_correlation.py --episodes 5 --seeds 7

Claims:
    Q-007: emotion.universal_expression_channel_mapping
    EVB-0006
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

CONDITION_RESOURCES: Dict[str, int] = {
    "RESOURCE_RICH":   10,
    "RESOURCE_SPARSE": 1,
}

# ── Pass thresholds ───────────────────────────────────────────────────────────

CORR_THRESHOLD: float = 0.10         # criterion 1: RICH valence-precision corr > threshold
PRECISION_DELTA: float = 0.10        # criterion 2: RICH mean precision > SPARSE + delta


# ── Pearson correlation ───────────────────────────────────────────────────────

def pearson_corr(xs: List[float], ys: List[float]) -> float:
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
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode, tracking per-step health (valence) and E3 precision.
    Returns episode-level mean of both for correlation analysis.
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    step_healths: List[float] = []
    step_precisions: List[float] = []
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

        # Record precision before step
        try:
            precision = float(agent.e3.current_precision)
        except AttributeError:
            precision = 1.0
        step_precisions.append(precision)

        action_tensor = result.selected_action.detach().clone()
        action_idx = action_tensor.argmax(dim=-1).item()
        next_obs, harm, done, info = env.step(action_idx)

        # Record health (valence proxy) after step
        step_healths.append(float(info.get("health", 1.0)))

        actual_harm = abs(harm) if harm < 0 else 0.0
        total_harm += actual_harm

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

    # ── E1 update ──
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

    mean_health = statistics.mean(step_healths) if step_healths else 0.0
    mean_precision = statistics.mean(step_precisions) if step_precisions else 1.0

    return {
        "total_harm": total_harm,
        "mean_health": mean_health,
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
    num_resources = CONDITION_RESOURCES[condition]
    torch.manual_seed(seed)
    env = CausalGridWorld(
        size=grid_size,
        num_hazards=num_hazards,
        num_resources=num_resources,
    )
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)
    e1_opt, policy_opt, e2_opt = make_optimizers(agent)

    ep_harms: List[float] = []
    ep_healths: List[float] = []
    ep_precisions: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, e2_opt, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_healths.append(metrics["mean_health"])
        ep_precisions.append(metrics["mean_precision"])

        if verbose and (ep + 1) % 50 == 0:
            recent_health = statistics.mean(ep_healths[-20:])
            recent_prec = statistics.mean(ep_precisions[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  seed={seed}  cond={condition}  "
                f"health={recent_health:.3f}  precision={recent_prec:.3f}"
            )

    quarter = max(1, num_episodes // 4)
    corr = pearson_corr(ep_healths, ep_precisions)

    return {
        "condition": condition,
        "seed": seed,
        "num_resources": num_resources,
        "mean_health": round(statistics.mean(ep_healths), 4),
        "last_quarter_health": round(statistics.mean(ep_healths[-quarter:]), 4),
        "mean_precision": round(statistics.mean(ep_precisions), 4),
        "last_quarter_precision": round(statistics.mean(ep_precisions[-quarter:]), 4),
        "valence_precision_corr": round(corr, 4),
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
        print("[Valence-Regime Correlation — Q-007 / EVB-0006] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions (resource density = valence proxy):")
        print(f"    RESOURCE_RICH   : {CONDITION_RESOURCES['RESOURCE_RICH']} resources (high valence achievable)")
        print(f"    RESOURCE_SPARSE : {CONDITION_RESOURCES['RESOURCE_SPARSE']} resource  (low/negative valence)")
        print()
        print("  Diagnostics:")
        print(f"    1. RICH valence-precision corr > {CORR_THRESHOLD}")
        print(f"    2. RICH mean_precision > SPARSE mean_precision + {PRECISION_DELTA}")
        print()

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for condition in ["RESOURCE_RICH", "RESOURCE_SPARSE"]:
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
                    f"    health={result['mean_health']:.3f}  "
                    f"precision={result['mean_precision']:.3f}  "
                    f"corr={result['valence_precision_corr']:+.3f}"
                )
                print()

    rich = [r for r in all_results if r["condition"] == "RESOURCE_RICH"]
    sparse = [r for r in all_results if r["condition"] == "RESOURCE_SPARSE"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    rich_corr = _agg(rich, "valence_precision_corr")
    rich_prec = _agg(rich, "mean_precision")
    sparse_prec = _agg(sparse, "mean_precision")
    rich_health = _agg(rich, "mean_health")
    sparse_health = _agg(sparse, "mean_health")

    crit_1 = rich_corr > CORR_THRESHOLD
    crit_2 = rich_prec > sparse_prec + PRECISION_DELTA

    num_met = sum([crit_1, crit_2])
    verdict = "PASS" if num_met >= 2 else "FAIL"
    partial = num_met == 1

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  RESOURCE_RICH   health={rich_health:.3f}  precision={rich_prec:.3f}  corr={rich_corr:+.3f}")
        print(f"  RESOURCE_SPARSE health={sparse_health:.3f}  precision={sparse_prec:.3f}")
        print()
        print(
            f"  Crit 1  RICH corr={rich_corr:+.3f} > {CORR_THRESHOLD}  "
            f"{'MET' if crit_1 else 'MISSED'}"
        )
        print(
            f"  Crit 2  RICH prec={rich_prec:.3f} > SPARSE+{PRECISION_DELTA}={sparse_prec+PRECISION_DELTA:.3f}  "
            f"{'MET' if crit_2 else 'MISSED'}"
        )
        print()
        print(f"  Criteria met: {num_met}/2  ->  Q-007 verdict: {verdict}")
        if partial:
            print("  (partial -- 1 of 2 criteria met)")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    Positive valence state (high health) correlates with higher E3 precision.")
            print("    Evidence FOR Q-007: valence maps to at least one control-plane regime parameter.")
        else:
            print("  Interpretation:")
            print("    Weak or absent valence-precision mapping on this substrate.")
            print("    Evidence AGAINST Q-007 simple mapping, or additional channels needed.")

    result_doc: Dict[str, Any] = {
        "experiment": "valence_regime_correlation",
        "claim": "Q-007",
        "evb_id": "EVB-0006",
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
            "resource_rich_count": CONDITION_RESOURCES["RESOURCE_RICH"],
            "resource_sparse_count": CONDITION_RESOURCES["RESOURCE_SPARSE"],
            "corr_threshold": CORR_THRESHOLD,
            "precision_delta": PRECISION_DELTA,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "rich_mean_health": rich_health,
            "sparse_mean_health": sparse_health,
            "rich_mean_precision": rich_prec,
            "sparse_mean_precision": sparse_prec,
            "rich_valence_precision_corr": rich_corr,
            "criterion_1_corr_met": crit_1,
            "criterion_2_precision_delta_met": crit_2,
            "criteria_met": num_met,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "valence_regime_correlation"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(evidence_dir / f"valence_regime_correlation_{ts}.json")
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q-007: Valence-Regime Correlation (REE-v2)"
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
