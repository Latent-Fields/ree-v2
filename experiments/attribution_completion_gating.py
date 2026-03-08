"""
Attribution Completion Gating Experiment (MECH-057 / EVB-0047) — REE-v2

Ground-up redesign using CausalGridWorld's self-attribution signal.

MECH-057 predicts that the agent must gate its policy learning signal to steps
where it has completed self-attribution of harm — i.e., steps where it can
distinguish "this harm is mine" from "this harm came from the environment."
Without this gating, the contamination-avoidance signal is diluted by
unattributable env-caused events, and the policy cannot cleanly learn to
avoid self-caused harm.

CausalGridWorld makes self-attribution concrete via info["transition_type"]:

  "agent_caused_hazard"
    The agent stepped onto a cell it previously contaminated.
    This is a completed self-attribution event: harm source is unambiguous.

  "env_caused_hazard"
    A background hazard drifted into the agent's path.
    Harm source is external; attributing this to the agent is incorrect.

Conditions (3):
  COMPLETION_GATED  : G = −agent_caused_harm only.
                      Policy learns only from steps it can attribute to itself.
                      Tests: does completed self-attribution yield better
                      contamination avoidance?

  FREE_REPLAN       : G = −total_harm (agent-caused + env-caused).
                      Standard mixed signal.  Attribution is not completed
                      before updating; the policy must disentangle both sources.

  ATTRIBUTION_BLIND : G = −env_caused_harm only.
                      Self-attribution signal is absent.  The agent cannot
                      learn that its own contamination is harmful.  Serves
                      as a lower bound: if BLIND converges similarly to GATED,
                      self-attribution signal is not necessary.

New metrics:

  agent_harm_last_quarter
    Total harm from agent_caused_hazard events in the final episode quarter.
    Primary measure of contamination avoidance quality.

  total_harm_last_quarter
    Overall harm — performance sanity check.

Pass criterion (>= 2 of 3):
  1. COMPLETION_GATED last-Q agent_harm
       < FREE_REPLAN last-Q agent_harm * 0.95
     (attribution-gated signal reduces self-caused harm vs mixed signal)

  2. COMPLETION_GATED last-Q agent_harm
       < ATTRIBUTION_BLIND last-Q agent_harm * 0.95
     (self-attribution is specifically necessary: blind agent cannot learn
      contamination avoidance)

  3. COMPLETION_GATED last-Q total_harm
       <= FREE_REPLAN last-Q total_harm * 1.10
     (gating on self-attribution does not cause major overall regression)

Usage:
    python experiments/attribution_completion_gating.py
    python experiments/attribution_completion_gating.py --episodes 5 --seeds 7

Claims:
    MECH-057: e1.completion_gating_required_for_self_attribution
    EVB-0047
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

# ── Pass thresholds ───────────────────────────────────────────────────────────

# Criteria 1 & 2: GATED agent harm must be strictly lower than comparison × factor
AGENT_HARM_REDUCTION_FACTOR: float = 0.95
# Criterion 3: GATED total harm must not exceed FREE total harm × tolerance
TOTAL_HARM_TOLERANCE: float = 1.10


# ── Optimizer factory ─────────────────────────────────────────────────────────

def make_optimizers(
    agent: REEAgent,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Build E1 world-model and E3 policy optimizers (shared across conditions)."""
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
    condition: str,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode.

    The three conditions differ only in which harm signal drives REINFORCE:
      COMPLETION_GATED  : G = -agent_caused_harm  (completed self-attribution)
      FREE_REPLAN       : G = -total_harm          (mixed signal)
      ATTRIBUTION_BLIND : G = -env_caused_harm     (env signal only; no self-attribution)

    E1 world-model update is identical across all conditions: it always receives
    the full prediction loss, so the world model quality is held constant.
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    agent_caused_harm = 0.0
    env_caused_harm = 0.0
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

        transition_type = info.get("transition_type", "none")
        actual_harm = abs(harm) if harm < 0 else 0.0
        total_harm += actual_harm

        if transition_type == "agent_caused_hazard":
            agent_caused_harm += actual_harm
        elif transition_type == "env_caused_hazard":
            env_caused_harm += actual_harm

        agent.update_residue(harm)
        obs = next_obs
        steps += 1
        if done:
            break

    # ── Condition-specific return signal ──────────────────────────────────────
    if condition == "COMPLETION_GATED":
        G = float(-agent_caused_harm)   # only self-attributed harm
    elif condition == "FREE_REPLAN":
        G = float(-total_harm)           # all harm (mixed signal)
    else:  # ATTRIBUTION_BLIND
        G = float(-env_caused_harm)      # env harm only; self-attribution absent

    # ── REINFORCE policy update ──
    policy_loss_val = 0.0
    if log_probs:
        policy_loss = -(torch.stack(log_probs) * G).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in policy_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        policy_opt.step()
        policy_loss_val = policy_loss.item()

    # ── E1 world-model update (same across all conditions) ──
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

    agent_harm_frac = (
        agent_caused_harm / (total_harm + 1e-8) if total_harm > 1e-8 else 0.0
    )

    return {
        "total_harm": total_harm,
        "agent_caused_harm": agent_caused_harm,
        "env_caused_harm": env_caused_harm,
        "agent_harm_fraction": agent_harm_frac,
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
    torch.manual_seed(seed)
    env = CausalGridWorld(size=grid_size, num_hazards=num_hazards)
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)
    e1_opt, policy_opt = make_optimizers(agent)

    ep_total_harms: List[float] = []
    ep_agent_harms: List[float] = []
    ep_env_harms: List[float] = []
    ep_agent_fracs: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, condition, max_steps)
        ep_total_harms.append(metrics["total_harm"])
        ep_agent_harms.append(metrics["agent_caused_harm"])
        ep_env_harms.append(metrics["env_caused_harm"])
        ep_agent_fracs.append(metrics["agent_harm_fraction"])

        if verbose and (ep + 1) % 50 == 0:
            recent_agent = statistics.mean(ep_agent_harms[-20:])
            recent_total = statistics.mean(ep_total_harms[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  "
                f"seed={seed}  cond={condition}  "
                f"agent_harm={recent_agent:.3f}  total_harm={recent_total:.3f}"
            )

    quarter = max(1, num_episodes // 4)
    return {
        "condition": condition,
        "seed": seed,
        "first_quarter_total_harm": round(statistics.mean(ep_total_harms[:quarter]), 4),
        "last_quarter_total_harm": round(statistics.mean(ep_total_harms[-quarter:]), 4),
        "first_quarter_agent_harm": round(statistics.mean(ep_agent_harms[:quarter]), 4),
        "last_quarter_agent_harm": round(statistics.mean(ep_agent_harms[-quarter:]), 4),
        "last_quarter_env_harm": round(statistics.mean(ep_env_harms[-quarter:]), 4),
        "mean_agent_harm_fraction": round(statistics.mean(ep_agent_fracs), 4),
        "last_quarter_agent_harm_fraction": round(
            statistics.mean(ep_agent_fracs[-quarter:]), 4
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
        print("[Attribution Completion Gating — MECH-057 / EVB-0047] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions (return signal gating):")
        print("    COMPLETION_GATED  : G = -agent_caused_harm  (completed self-attribution)")
        print("    FREE_REPLAN       : G = -total_harm          (mixed signal)")
        print("    ATTRIBUTION_BLIND : G = -env_caused_harm     (self-attribution absent)")
        print()
        print("  Diagnostics:")
        print(f"    1. GATED last-Q agent_harm < FREE  * {AGENT_HARM_REDUCTION_FACTOR}")
        print(f"    2. GATED last-Q agent_harm < BLIND * {AGENT_HARM_REDUCTION_FACTOR}")
        print(f"    3. GATED last-Q total_harm <= FREE * {TOTAL_HARM_TOLERANCE}")
        print()

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for condition in ["COMPLETION_GATED", "FREE_REPLAN", "ATTRIBUTION_BLIND"]:
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
                    f"    agent_harm {result['first_quarter_agent_harm']:.3f} -> "
                    f"{result['last_quarter_agent_harm']:.3f}  "
                    f"total {result['first_quarter_total_harm']:.3f} -> "
                    f"{result['last_quarter_total_harm']:.3f}"
                )
                print()

    gated = [r for r in all_results if r["condition"] == "COMPLETION_GATED"]
    free = [r for r in all_results if r["condition"] == "FREE_REPLAN"]
    blind = [r for r in all_results if r["condition"] == "ATTRIBUTION_BLIND"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    gated_agent_last = _agg(gated, "last_quarter_agent_harm")
    free_agent_last = _agg(free, "last_quarter_agent_harm")
    blind_agent_last = _agg(blind, "last_quarter_agent_harm")
    gated_total_last = _agg(gated, "last_quarter_total_harm")
    free_total_last = _agg(free, "last_quarter_total_harm")

    # Criterion 1: gated learning signal outperforms mixed signal on self-caused harm
    crit_1 = gated_agent_last < free_agent_last * AGENT_HARM_REDUCTION_FACTOR
    # Criterion 2: self-attribution necessary — blind agent cannot match gated
    crit_2 = gated_agent_last < blind_agent_last * AGENT_HARM_REDUCTION_FACTOR
    # Criterion 3: gating on attribution does not cause major overall regression
    crit_3 = gated_total_last <= free_total_last * TOTAL_HARM_TOLERANCE

    num_met = sum([crit_1, crit_2, crit_3])
    verdict = "PASS" if num_met >= 2 else "FAIL"
    partial = num_met == 1

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  COMPLETION_GATED  agent_harm (last-Q): {gated_agent_last:.3f}")
        print(f"  FREE_REPLAN       agent_harm (last-Q): {free_agent_last:.3f}")
        print(f"  ATTRIBUTION_BLIND agent_harm (last-Q): {blind_agent_last:.3f}")
        print()
        print(
            f"  Crit 1  GATED={gated_agent_last:.3f} < "
            f"FREE*{AGENT_HARM_REDUCTION_FACTOR}={free_agent_last * AGENT_HARM_REDUCTION_FACTOR:.3f}  "
            f"{'MET' if crit_1 else 'MISSED'}"
        )
        print(
            f"  Crit 2  GATED={gated_agent_last:.3f} < "
            f"BLIND*{AGENT_HARM_REDUCTION_FACTOR}={blind_agent_last * AGENT_HARM_REDUCTION_FACTOR:.3f}  "
            f"{'MET' if crit_2 else 'MISSED'}"
        )
        print(
            f"  Crit 3  GATED_total={gated_total_last:.3f} <= "
            f"FREE*{TOTAL_HARM_TOLERANCE}={free_total_last * TOTAL_HARM_TOLERANCE:.3f}  "
            f"{'MET' if crit_3 else 'MISSED'}"
        )
        print()
        print(f"  Criteria met: {num_met}/3  ->  MECH-057 verdict: {verdict}")
        if partial:
            print("  (partial -- exactly 1 of 3 criteria met)")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    Gating the policy update on completed self-attribution reduces")
            print("    agent-caused harm vs both mixed signal and attribution-blind.")
            print("    MECH-057 completion gating requirement confirmed on V2 substrate.")

    result_doc: Dict[str, Any] = {
        "experiment": "attribution_completion_gating",
        "claim": "MECH-057",
        "evb_id": "EVB-0047",
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
            "agent_harm_reduction_factor": AGENT_HARM_REDUCTION_FACTOR,
            "total_harm_tolerance": TOTAL_HARM_TOLERANCE,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "gated_agent_harm_last_quarter": gated_agent_last,
            "free_agent_harm_last_quarter": free_agent_last,
            "blind_agent_harm_last_quarter": blind_agent_last,
            "gated_total_harm_last_quarter": gated_total_last,
            "free_total_harm_last_quarter": free_total_last,
            "criterion_1_gated_vs_free_met": crit_1,
            "criterion_2_gated_vs_blind_met": crit_2,
            "criterion_3_total_harm_parity_met": crit_3,
            "criteria_met": num_met,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "attribution_completion_gating"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"attribution_completion_gating_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MECH-057: Attribution Completion Gating (REE-v2)"
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
