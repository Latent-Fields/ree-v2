"""
Path Memory Ablation Experiment (ARC-007 / EVB-0009) — REE-v2

Tests ARC-007: hippocampus.path_memory

ARC-007 asserts that the hippocampal system indexes, stores, and replays
experienced trajectories through latent space, and that this path memory is
orthogonal to valuation and control — existing specifically to preserve
identity, continuity, and reflectability.

CausalGridWorld exposes path memory directly in the observation: the
contamination_view (5x5 float) shows the agent's causal footprint in its
neighbourhood, and footprint_density shows its own visit count at the current
cell.  Together these constitute an externally-visible path memory signal.

PRIMARY (PATH_MEMORY): agent receives full observation including contamination_view
  and footprint_density.  It can see where it has been and how heavily it has
  contaminated nearby cells.

ABLATED (PATH_ABLATED): contamination_view and footprint_density are zeroed out
  before being passed to the agent.  The agent has no access to its own prior
  trajectory — it operates as if path memory were absent.

If ARC-007 is correct — if path memory is a load-bearing component — then
PATH_ABLATED should show meaningfully higher agent-caused harm (more revisits
to already-contaminated cells, because the agent cannot see its footprint).

Key diagnostics:
  1. agent_harm_last_quarter: self-caused harm in the final quarter.
     PATH_MEMORY < PATH_ABLATED * 0.95  (path memory reduces contamination harm)

  2. total_harm_last_quarter: overall harm sanity check.
     PATH_MEMORY <= PATH_ABLATED * 1.05  (path memory does not worsen overall harm)

Pass: both criteria met.

Observation structure (used to compute ablation slice):
  position          : grid_size * grid_size
  local_view        : 5 * 5 * 6 = 150
  homeostatic       : 2
  contamination_view: 5 * 5 = 25   <-- zeroed in PATH_ABLATED
  footprint_density : 1             <-- zeroed in PATH_ABLATED

Usage:
    python experiments/path_memory_ablation.py
    python experiments/path_memory_ablation.py --episodes 5 --seeds 7

Claims:
    ARC-007: hippocampus.path_memory
    EVB-0009
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

AGENT_HARM_FACTOR: float = 0.95   # PATH_MEMORY agent_harm < PATH_ABLATED * factor
TOTAL_HARM_CEILING: float = 1.05  # PATH_MEMORY total_harm <= PATH_ABLATED * ceiling


# ── Observation path-memory slice ─────────────────────────────────────────────

def path_memory_slice(grid_size: int) -> Tuple[int, int]:
    """Return (start, end) index of contamination_view + footprint_density in obs."""
    position_dim = grid_size * grid_size
    local_view_dim = 5 * 5 * 6
    homeostatic_dim = 2
    start = position_dim + local_view_dim + homeostatic_dim
    end = start + 5 * 5 + 1  # contamination_view (25) + footprint_density (1)
    return start, end


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
    condition: str,
    max_steps: int,
    pm_start: int,
    pm_end: int,
) -> Dict[str, Any]:
    """
    Run one episode.

    PATH_MEMORY: full observation.
    PATH_ABLATED: contamination_view + footprint_density zeroed before agent sees it.
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    agent_caused_harm = 0.0
    steps = 0

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # ── Condition-specific observation masking ──
        if condition == "PATH_ABLATED":
            obs_tensor = obs_tensor.clone()
            obs_tensor[:, pm_start:pm_end] = 0.0

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

        result = agent.e3.select(candidates)
        if result.log_prob is not None:
            log_probs.append(result.log_prob)

        action_idx = result.selected_action.argmax(dim=-1).item()
        next_obs, harm, done, info = env.step(action_idx)

        actual_harm = abs(harm) if harm < 0 else 0.0
        total_harm += actual_harm
        if info.get("transition_type") == "agent_caused_hazard":
            agent_caused_harm += actual_harm

        agent.update_residue(harm)
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

    return {
        "total_harm": total_harm,
        "agent_caused_harm": agent_caused_harm,
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
    pm_start, pm_end = path_memory_slice(grid_size)

    ep_total_harms: List[float] = []
    ep_agent_harms: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(
            agent, env, e1_opt, policy_opt, condition, max_steps, pm_start, pm_end
        )
        ep_total_harms.append(metrics["total_harm"])
        ep_agent_harms.append(metrics["agent_caused_harm"])

        if verbose and (ep + 1) % 50 == 0:
            recent_agent = statistics.mean(ep_agent_harms[-20:])
            recent_total = statistics.mean(ep_total_harms[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  seed={seed}  cond={condition}  "
                f"agent_harm={recent_agent:.3f}  total={recent_total:.3f}"
            )

    quarter = max(1, num_episodes // 4)
    return {
        "condition": condition,
        "seed": seed,
        "first_quarter_total_harm": round(statistics.mean(ep_total_harms[:quarter]), 4),
        "last_quarter_total_harm": round(statistics.mean(ep_total_harms[-quarter:]), 4),
        "first_quarter_agent_harm": round(statistics.mean(ep_agent_harms[:quarter]), 4),
        "last_quarter_agent_harm": round(statistics.mean(ep_agent_harms[-quarter:]), 4),
        "path_memory_slice": [pm_start, pm_end],
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
    pm_start, pm_end = path_memory_slice(grid_size)

    if verbose:
        print("[Path Memory Ablation — ARC-007 / EVB-0009] (REE-v2)")
        print(f"  CausalGridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print(f"  Path memory obs slice: [{pm_start}:{pm_end}] "
              f"(contamination_view 25 + footprint_density 1)")
        print()
        print("  Conditions:")
        print("    PATH_MEMORY : full observation (path history visible)")
        print("    PATH_ABLATED: contamination_view + footprint zeroed (no path memory)")
        print()
        print("  Diagnostics:")
        print(f"    1. MEMORY agent_harm (last-Q) < ABLATED * {AGENT_HARM_FACTOR}")
        print(f"    2. MEMORY total_harm (last-Q) <= ABLATED * {TOTAL_HARM_CEILING}")
        print()

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for condition in ["PATH_MEMORY", "PATH_ABLATED"]:
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

    memory = [r for r in all_results if r["condition"] == "PATH_MEMORY"]
    ablated = [r for r in all_results if r["condition"] == "PATH_ABLATED"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    mem_agent_last = _agg(memory, "last_quarter_agent_harm")
    abl_agent_last = _agg(ablated, "last_quarter_agent_harm")
    mem_total_last = _agg(memory, "last_quarter_total_harm")
    abl_total_last = _agg(ablated, "last_quarter_total_harm")

    crit_1 = mem_agent_last < abl_agent_last * AGENT_HARM_FACTOR
    crit_2 = mem_total_last <= abl_total_last * TOTAL_HARM_CEILING

    num_met = sum([crit_1, crit_2])
    verdict = "PASS" if num_met >= 2 else "FAIL"
    partial = num_met == 1

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  PATH_MEMORY  agent_harm (last-Q): {mem_agent_last:.3f}  total: {mem_total_last:.3f}")
        print(f"  PATH_ABLATED agent_harm (last-Q): {abl_agent_last:.3f}  total: {abl_total_last:.3f}")
        print()
        print(
            f"  Crit 1  MEMORY={mem_agent_last:.3f} < "
            f"ABLATED*{AGENT_HARM_FACTOR}={abl_agent_last*AGENT_HARM_FACTOR:.3f}  "
            f"{'MET' if crit_1 else 'MISSED'}"
        )
        print(
            f"  Crit 2  MEMORY_total={mem_total_last:.3f} <= "
            f"ABLATED*{TOTAL_HARM_CEILING}={abl_total_last*TOTAL_HARM_CEILING:.3f}  "
            f"{'MET' if crit_2 else 'MISSED'}"
        )
        print()
        print(f"  Criteria met: {num_met}/2  ->  ARC-007 verdict: {verdict}")
        if partial:
            print("  (partial -- 1 of 2 criteria met)")

    result_doc: Dict[str, Any] = {
        "experiment": "path_memory_ablation",
        "claim": "ARC-007",
        "evb_id": "EVB-0009",
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
            "path_memory_obs_slice": [pm_start, pm_end],
            "agent_harm_factor": AGENT_HARM_FACTOR,
            "total_harm_ceiling": TOTAL_HARM_CEILING,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "memory_agent_harm_last_quarter": mem_agent_last,
            "ablated_agent_harm_last_quarter": abl_agent_last,
            "memory_total_harm_last_quarter": mem_total_last,
            "ablated_total_harm_last_quarter": abl_total_last,
            "criterion_1_agent_harm_met": crit_1,
            "criterion_2_total_harm_met": crit_2,
            "criteria_met": num_met,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "path_memory_ablation"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(evidence_dir / f"path_memory_ablation_{ts}.json")
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARC-007: Path Memory Ablation (REE-v2)"
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
