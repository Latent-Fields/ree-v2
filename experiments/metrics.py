"""Metrics helpers for V2 experiment harness runs.

V2 additions over V1:
- agent_caused_harm_count: steps where transition_type == 'agent_caused_hazard'
- env_caused_harm_count: steps where transition_type == 'env_caused_hazard'
- contamination_events: steps where contamination_delta > 0

These require info_list (list of info dicts from CausalGridWorld.step()) to be passed in.
"""

from typing import List, Optional


def compute_metrics_values(result, info_list: Optional[List[dict]] = None):
    """Build stable numeric metrics for Experiment Pack v1.

    Args:
        result: Dict returned by run_experiment_episode().
        info_list: Optional list of info dicts from each env.step() call.
                   Required for V2 CausalGridWorld metrics; defaults to zeros if absent.
    """
    # V2 CausalGridWorld-specific metrics
    agent_caused_harm_count = 0
    env_caused_harm_count = 0
    contamination_events = 0

    if info_list:
        for info in info_list:
            tt = info.get("transition_type", "none")
            if tt == "agent_caused_hazard":
                agent_caused_harm_count += 1
            elif tt == "env_caused_hazard":
                env_caused_harm_count += 1
            if info.get("contamination_delta", 0.0) > 0:
                contamination_events += 1

    return {
        # Core homeostatic
        "total_harm": float(result.get("total_harm", 0.0)),
        "total_reward": float(result.get("total_reward", 0.0)),
        "final_residue": float(result.get("final_residue", 0.0)),
        "steps_survived": int(result.get("steps", 0)),
        "max_steps": int(result.get("max_steps", 0)),
        "done": int(result.get("done", 0)),
        "final_health": float(result.get("final_health", 0.0)),
        "final_energy": float(result.get("final_energy", 0.0)),
        # Event counts
        "harm_event_count": int(result.get("harm_event_count", 0)),
        "hazard_event_count": int(result.get("hazard_event_count", 0)),
        "collision_event_count": int(result.get("collision_event_count", 0)),
        "resource_event_count": int(result.get("resource_event_count", 0)),
        "fatal_error_count": int(result.get("fatal_error_count", 0)),
        # V2 CausalGridWorld: agent-caused vs env-caused attribution
        "agent_caused_harm_count": agent_caused_harm_count,
        "env_caused_harm_count": env_caused_harm_count,
        "contamination_events": contamination_events,
    }


def compute_summary(result):
    """Backward-compatible summary values for quick terminal output."""
    return {
        "total_harm": float(result.get("total_harm", 0.0)),
        "final_residue": float(result.get("final_residue", 0.0)),
        "steps_survived": int(result.get("steps", 0)),
    }
