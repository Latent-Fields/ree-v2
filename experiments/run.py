"""
REE-v2 experiment harness.

Port of ree-v1-minimal/experiments/run.py with:
- CausalGridWorld replacing GridWorld
- JEPA adapter signals removed
- info_list collected per step for V2 metrics (agent_caused_harm_count, etc.)
- runner_name updated to ree-v2-harness
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
import traceback
from typing import Optional

# Allow direct invocation: python experiments/run.py --help
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.metrics import compute_metrics_values, compute_summary
from experiments.pack_writer import (
    EVIDENCE_DIRECTIONS,
    ExperimentPackWriter,
    deterministic_run_id,
    resolve_output_root,
    stable_config_hash,
    normalize_timestamp_utc,
)
from ree_core import __version__ as REE_VERSION
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency in some local envs
    np = None

MECH056_CLAIM_ID = "MECH-056"


def _clean_claim_ids(claim_ids: object) -> list[str]:
    if not isinstance(claim_ids, list):
        return []
    cleaned: list[str] = []
    for claim_id in claim_ids:
        value = str(claim_id).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        item_value = value.item()
        if isinstance(item_value, (int, float)) and not isinstance(item_value, bool):
            return float(item_value)
    return 0.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, p)) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _resolve_causal_env_runtime_config(suite: dict) -> dict:
    """Resolve CausalGridWorld runtime config from suite spec."""
    env_spec = suite.get("environment", {})
    runtime = env_spec.get("runtime", {}) if isinstance(env_spec, dict) else {}
    return {
        "size": int(runtime.get("size", 10)),
        "num_resources": int(runtime.get("num_resources", 5)),
        "num_hazards": int(runtime.get("num_hazards", 3)),
        "contamination_spread": float(runtime.get("contamination_spread", 0.5)),
        "contamination_threshold": float(runtime.get("contamination_threshold", 2.0)),
        "env_drift_interval": int(runtime.get("env_drift_interval", 5)),
        "env_drift_prob": float(runtime.get("env_drift_prob", 0.3)),
        "resource_benefit": float(runtime.get("resource_benefit", 0.3)),
        "hazard_harm": float(runtime.get("hazard_harm", 0.5)),
        "contaminated_harm": float(runtime.get("contaminated_harm", 0.3)),
        "energy_decay": float(runtime.get("energy_decay", 0.02)),
    }


def _build_environment_metadata(suite: dict, runtime_config: dict) -> dict:
    env_spec = suite.get("environment", {})
    if not isinstance(env_spec, dict):
        env_spec = {}

    dynamics_spec = env_spec.get(
        "dynamics",
        {
            "movement_model": "cardinal_plus_stay",
            "contamination_spread": runtime_config["contamination_spread"],
            "contamination_threshold": runtime_config["contamination_threshold"],
            "env_drift_interval": runtime_config["env_drift_interval"],
            "env_drift_prob": runtime_config["env_drift_prob"],
            "episode_end_conditions": ["health_depleted", "energy_depleted", "max_steps"],
        },
    )
    reward_spec = env_spec.get(
        "reward",
        {
            "resource_benefit": runtime_config["resource_benefit"],
            "hazard_harm": runtime_config["hazard_harm"],
            "contaminated_harm": runtime_config["contaminated_harm"],
        },
    )
    observation_spec = env_spec.get(
        "observation",
        {
            "position_encoding": "one_hot",
            "local_view": "5x5_entity_one_hot",
            "contamination_view": "5x5_float",
            "footprint_density": "scalar",
            "homeostatic_signals": ["health", "energy"],
        },
    )

    return {
        "env_id": str(env_spec.get("env_id", "ree.causal_grid_world")),
        "env_version": str(env_spec.get("env_version", "causal_grid_world/v2")),
        "dynamics_hash": stable_config_hash(dynamics_spec),
        "reward_hash": stable_config_hash(reward_spec),
        "observation_hash": stable_config_hash(observation_spec),
        "config_hash": stable_config_hash(runtime_config),
        "tier": str(env_spec.get("tier", "toy")),
    }


def _build_producer_capabilities() -> dict[str, bool]:
    return {
        "trajectory_integrity_channelized_bias": True,
        "mech056_dispatch_metric_set": True,
        "mech056_summary_escalation_trace": True,
        "sd003_causal_attribution": True,
    }


def _compute_mech056_metrics(result: dict) -> dict:
    steps = max(int(result.get("steps", 0)), 0)
    max_steps = max(int(result.get("max_steps", 0)), 1)
    harm_event_count = max(int(result.get("harm_event_count", 0)), 0)
    hazard_event_count = max(int(result.get("hazard_event_count", 0)), 0)
    collision_event_count = max(int(result.get("collision_event_count", 0)), 0)
    resource_event_count = max(int(result.get("resource_event_count", 0)), 0)
    final_residue = max(float(result.get("final_residue", 0.0)), 0.0)

    trajectory_commit_usage = steps
    perceptual_sampling_usage = hazard_event_count + collision_event_count
    if perceptual_sampling_usage == 0 and harm_event_count > 0:
        perceptual_sampling_usage = harm_event_count

    structural_consolidation_usage = 0
    if final_residue > 0 or harm_event_count > 0:
        structural_consolidation_usage = 1 + (1 if harm_event_count >= 3 else 0)

    structural_bias_magnitude = final_residue + (0.1 * harm_event_count)
    structural_bias_rate = structural_bias_magnitude / max(steps, 1)
    shortcut_leakage_events = max(0, harm_event_count - perceptual_sampling_usage)
    unobservable_critical_state_rate = shortcut_leakage_events / max(steps, 1)
    controllability_score = (resource_event_count + 1.0) / (
        resource_event_count + harm_event_count + 1.0
    )
    transition_consistency_rate = max(0.0, 1.0 - (collision_event_count / max_steps))

    return {
        "trajectory_commit_channel_usage_count": trajectory_commit_usage,
        "perceptual_sampling_channel_usage_count": perceptual_sampling_usage,
        "structural_consolidation_channel_usage_count": structural_consolidation_usage,
        "precommit_semantic_overwrite_events": 0,
        "structural_bias_magnitude": structural_bias_magnitude,
        "structural_bias_rate": structural_bias_rate,
        "environment_shortcut_leakage_events": shortcut_leakage_events,
        "environment_unobservable_critical_state_rate": unobservable_critical_state_rate,
        "environment_controllability_score": controllability_score,
        "environment_transition_consistency_rate": transition_consistency_rate,
    }


def _build_mech056_summary_lines(result: dict, metrics_values: dict) -> list[str]:
    order = ["trajectory_commit"]
    if int(metrics_values.get("perceptual_sampling_channel_usage_count", 0)) > 0:
        order.append("perceptual_sampling")
    if int(metrics_values.get("structural_consolidation_channel_usage_count", 0)) > 0:
        order.append("structural_consolidation")

    lines = [
        "",
        "## MECH-056 Escalation Trace",
        (
            "- channel_escalation_order_observed: `"
            + " -> ".join(order)
            + "`"
        ),
    ]

    perceptual_count = int(metrics_values.get("perceptual_sampling_channel_usage_count", 0))
    structural_count = int(metrics_values.get("structural_consolidation_channel_usage_count", 0))
    harm_count = int(result.get("harm_event_count", 0))
    hazard_count = int(result.get("hazard_event_count", 0))
    collision_count = int(result.get("collision_event_count", 0))
    structural_bias_rate = float(metrics_values.get("structural_bias_rate", 0.0))

    if perceptual_count > 0:
        lines.append(
            "- trigger_rationale_perceptual_sampling: activated after harm/collision cues "
            f"(harm_events={harm_count}, hazard_events={hazard_count}, collision_events={collision_count})."
        )
    if structural_count > 0:
        lines.append(
            "- trigger_rationale_structural_consolidation: activated to consolidate persistent bias "
            f"(final_residue={result.get('final_residue', 0.0):.6f}, structural_bias_rate={structural_bias_rate:.6f})."
        )
    return lines


def _resolve_claim_ids(
    suite_name: str, suite: dict, claim_ids_override: Optional[list[str]]
) -> list[str]:
    raw_claim_ids = (
        claim_ids_override
        if claim_ids_override is not None
        else suite.get("claim_ids_tested", [])
    )
    claim_ids = _clean_claim_ids(raw_claim_ids)
    if claim_ids:
        return claim_ids
    raise ValueError(
        f"suite '{suite_name}' is missing claim_ids_tested; configure it in "
        f"experiments/suites.json or pass --claim-id."
    )


def _resolve_evidence_class(suite: dict, evidence_class_override: Optional[str]) -> str:
    candidate = evidence_class_override
    if candidate is None:
        candidate = suite.get("evidence_class")
    if candidate is None:
        return "simulation"
    cleaned = str(candidate).strip()
    return cleaned or "simulation"


def _resolve_evidence_direction(
    suite: dict,
    status: str,
    evidence_direction_override: Optional[str],
) -> str:
    candidate = evidence_direction_override
    if candidate is None:
        suite_candidate = suite.get("evidence_direction")
        if suite_candidate is not None:
            candidate = str(suite_candidate)

    if candidate is None or not str(candidate).strip():
        return "supports" if status == "PASS" else "weakens"

    cleaned = str(candidate).strip().lower()
    if cleaned not in EVIDENCE_DIRECTIONS:
        expected = ", ".join(sorted(EVIDENCE_DIRECTIONS))
        raise ValueError(
            f"invalid evidence_direction '{candidate}' (expected one of: {expected})"
        )
    return cleaned


def load_suites() -> dict:
    suites_path = Path(__file__).resolve().parent / "suites.json"
    with suites_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_overrides(agent: REEAgent, suite: dict) -> None:
    overrides = suite.get("overrides", {})
    for component_name, config_overrides in overrides.items():
        if not isinstance(config_overrides, dict):
            continue
        component = getattr(agent, component_name, None)
        if component is None:
            continue
        target = getattr(component, "config", component)
        for field_name, value in config_overrides.items():
            if hasattr(target, field_name):
                setattr(target, field_name, value)


def run_experiment_episode(
    agent: REEAgent, env: CausalGridWorld, max_steps: int
) -> tuple[dict, list[dict]]:
    """Run one episode, returning (result_dict, info_list).

    V2: collects info_list from each step for CausalGridWorld attribution metrics.
    Uses transition_type from info for hazard/resource event counting.
    """
    agent.reset()
    observation = env.reset()

    total_harm = 0.0
    total_reward = 0.0
    harm_event_count = 0
    hazard_event_count = 0
    collision_event_count = 0  # CausalGridWorld has no collisions; kept for schema compat
    resource_event_count = 0
    info_list: list[dict] = []

    done = False
    final_info: dict = {"health": 1.0, "energy": 1.0}
    steps = 0

    for _ in range(max_steps):
        action = agent.act(observation)
        observation, harm_signal, done, info = env.step(action)
        info_list.append(info)
        residue_update_metrics = agent.update_residue(harm_signal)  # noqa: F841
        if agent.should_integrate():
            agent.offline_integration()

        if harm_signal < 0:
            total_harm += abs(float(harm_signal))
            harm_event_count += 1
        else:
            total_reward += float(harm_signal)

        # V2: use transition_type (CausalGridWorld) instead of info["event"] (GridWorld)
        tt = info.get("transition_type", "none")
        if tt in ("agent_caused_hazard", "env_caused_hazard"):
            hazard_event_count += 1
        elif tt == "resource":
            resource_event_count += 1

        final_info = info
        steps += 1
        if done:
            break

    residue_stats = agent.get_residue_statistics()
    result = {
        "steps": steps,
        "max_steps": max_steps,
        "done": int(done),
        "total_harm": total_harm,
        "total_reward": total_reward,
        "final_residue": float(residue_stats["total_residue"].item()),
        "final_health": float(final_info.get("health", 0.0)),
        "final_energy": float(final_info.get("energy", 0.0)),
        "harm_event_count": harm_event_count,
        "hazard_event_count": hazard_event_count,
        "collision_event_count": collision_event_count,
        "resource_event_count": resource_event_count,
        "fatal_error_count": 0,
    }
    return result, info_list


def known_failure_signatures(result: dict) -> list[str]:
    signatures: list[str] = []
    if int(result.get("fatal_error_count", 0)) > 0:
        return ["fatal_error"]
    if float(result.get("final_health", 1.0)) <= 0.0:
        signatures.append("agent_health_depleted")
    if float(result.get("final_energy", 1.0)) <= 0.0:
        signatures.append("agent_energy_depleted")
    return signatures


def build_summary_markdown(
    suite_name: str,
    suite: dict,
    seed: int,
    run_id: str,
    timestamp_utc: str,
    status: str,
    claim_ids_tested: list[str],
    evidence_class: str,
    evidence_direction: str,
    result: dict,
    metrics_values: dict,
    failure_signatures: list[str],
) -> str:
    summary = compute_summary(result)
    lines = [
        "# Experiment Run Summary",
        "",
        "## Scenario",
        f"- suite: `{suite_name}`",
        f"- run_id: `{run_id}`",
        f"- seed: `{seed}`",
        f"- timestamp_utc: `{timestamp_utc}`",
        f"- description: {suite.get('description', 'n/a')}",
        "- claim_ids_tested: " + ", ".join(f"`{claim_id}`" for claim_id in claim_ids_tested),
        f"- evidence_class: `{evidence_class}`",
        f"- evidence_direction: `{evidence_direction}`",
        "",
        "## Outcome",
        f"- status: **{status}**",
        f"- steps_survived: {summary['steps_survived']}",
        f"- total_harm: {summary['total_harm']:.6f}",
        f"- final_residue: {summary['final_residue']:.6f}",
        f"- final_health: {result.get('final_health', 0.0):.6f}",
        f"- final_energy: {result.get('final_energy', 0.0):.6f}",
        "",
        "## Interpretation",
    ]
    if failure_signatures:
        lines.append(
            "- run failed due to known failure signatures: "
            + ", ".join(f"`{sig}`" for sig in failure_signatures)
        )
    else:
        lines.append("- run passed known stop checks and did not trigger known signatures.")

    if MECH056_CLAIM_ID in claim_ids_tested:
        lines.extend(_build_mech056_summary_lines(result, metrics_values))

    return "\n".join(lines)


def execute_experiment(
    suite_name: str,
    seed: int = 0,
    max_steps: int = 200,
    output_root: Optional[str] = None,
    run_id: Optional[str] = None,
    timestamp_utc: Optional[str] = None,
    claim_ids_tested: Optional[list[str]] = None,
    evidence_class: Optional[str] = None,
    evidence_direction: Optional[str] = None,
    runner_name: str = "ree-v2-harness",
    runner_version: str = REE_VERSION,
) -> Path:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)

    suites = load_suites()
    if suite_name not in suites:
        known = ", ".join(sorted(suites.keys()))
        raise ValueError(f"unknown suite '{suite_name}'. Available suites: {known}")
    suite = suites[suite_name]
    resolved_claim_ids = _resolve_claim_ids(suite_name, suite, claim_ids_tested)
    resolved_evidence_class = _resolve_evidence_class(suite, evidence_class)
    runtime_config = _resolve_causal_env_runtime_config(suite)
    environment_metadata = _build_environment_metadata(suite, runtime_config)
    producer_capabilities = _build_producer_capabilities()

    normalized_timestamp = normalize_timestamp_utc(timestamp_utc)
    resolved_run_id = run_id or deterministic_run_id(suite_name, seed, normalized_timestamp)

    scenario = {
        "name": suite_name,
        "seed": seed,
        "config_hash": stable_config_hash(suite),
        "max_steps": max_steps,
    }

    traces_dir = None
    trace_text = None
    info_list: list[dict] = []
    try:
        env = CausalGridWorld(
            size=runtime_config["size"],
            num_hazards=runtime_config["num_hazards"],
            num_resources=runtime_config["num_resources"],
            contamination_spread=runtime_config["contamination_spread"],
            contamination_threshold=runtime_config["contamination_threshold"],
            env_drift_interval=runtime_config["env_drift_interval"],
            env_drift_prob=runtime_config["env_drift_prob"],
            hazard_harm=runtime_config["hazard_harm"],
            contaminated_harm=runtime_config["contaminated_harm"],
            resource_benefit=runtime_config["resource_benefit"],
            energy_decay=runtime_config["energy_decay"],
        )
        agent = REEAgent.from_config(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            latent_dim=64,
        )
        apply_overrides(agent, suite)
        result, info_list = run_experiment_episode(agent, env, max_steps=max_steps)
    except Exception:
        traces_dir = "traces"
        trace_text = traceback.format_exc()
        result = {
            "steps": 0,
            "max_steps": max_steps,
            "done": 1,
            "total_harm": 0.0,
            "total_reward": 0.0,
            "final_residue": 0.0,
            "final_health": 0.0,
            "final_energy": 0.0,
            "harm_event_count": 0,
            "hazard_event_count": 0,
            "collision_event_count": 0,
            "resource_event_count": 0,
            "fatal_error_count": 1,
        }

    failure_signatures = known_failure_signatures(result)
    status = "FAIL" if failure_signatures else "PASS"
    resolved_evidence_direction = _resolve_evidence_direction(suite, status, evidence_direction)
    metrics_values = compute_metrics_values(result, info_list)
    if MECH056_CLAIM_ID in resolved_claim_ids:
        metrics_values.update(_compute_mech056_metrics(result))

    summary_markdown = build_summary_markdown(
        suite_name=suite_name,
        suite=suite,
        seed=seed,
        run_id=resolved_run_id,
        timestamp_utc=normalized_timestamp,
        status=status,
        claim_ids_tested=resolved_claim_ids,
        evidence_class=resolved_evidence_class,
        evidence_direction=resolved_evidence_direction,
        result=result,
        metrics_values=metrics_values,
        failure_signatures=failure_signatures,
    )

    repo_root = Path(__file__).resolve().parents[1]
    writer = ExperimentPackWriter(
        output_root=resolve_output_root(output_root),
        repo_root=repo_root,
        runner_name=runner_name,
        runner_version=runner_version,
    )
    emitted = writer.write_pack(
        experiment_type=suite_name,
        run_id=resolved_run_id,
        timestamp_utc=normalized_timestamp,
        status=status,
        metrics_values=metrics_values,
        summary_markdown=summary_markdown,
        scenario=scenario,
        failure_signatures=failure_signatures,
        claim_ids_tested=resolved_claim_ids,
        evidence_class=resolved_evidence_class,
        evidence_direction=resolved_evidence_direction,
        producer_capabilities=producer_capabilities,
        environment=environment_metadata,
        traces_dir=traces_dir,
    )

    if trace_text and traces_dir:
        trace_path = emitted.run_dir / traces_dir / "fatal_error.txt"
        trace_path.write_text(trace_text, encoding="utf-8")

    return emitted.run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run REE-v2 experiment suite and emit Experiment Pack v1."
    )
    parser.add_argument(
        "--suite", required=True, help="Experiment suite name from experiments/suites.json."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum episode steps.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output path. Defaults to REE_EXPERIMENT_OUTPUT_ROOT or runs.",
    )
    parser.add_argument("--run-id", default=None, help="Optional explicit run_id.")
    parser.add_argument(
        "--timestamp-utc",
        default=None,
        help="Optional RFC3339 timestamp used for manifest timestamp_utc and deterministic run_id.",
    )
    parser.add_argument(
        "--claim-id",
        action="append",
        dest="claim_ids",
        default=None,
        help="Claim ID tested by this run. Pass multiple times for multiple claim IDs.",
    )
    parser.add_argument(
        "--evidence-class",
        default=None,
        help="Evidence class for claim linkage (e.g., simulation, ablation, regression).",
    )
    parser.add_argument(
        "--evidence-direction",
        choices=sorted(EVIDENCE_DIRECTIONS),
        default=None,
        help=(
            "Evidence direction for claim linkage. "
            "Defaults to supports/weakens inferred from PASS/FAIL."
        ),
    )
    parser.add_argument(
        "--runner-name",
        default="ree-v2-harness",
        help="Runner name written to manifest.runner.name.",
    )
    parser.add_argument(
        "--runner-version",
        default=REE_VERSION,
        help="Runner version written to manifest.runner.version.",
    )
    args = parser.parse_args()

    run_dir = execute_experiment(
        suite_name=args.suite,
        seed=args.seed,
        max_steps=args.max_steps,
        output_root=args.output_root,
        run_id=args.run_id,
        timestamp_utc=args.timestamp_utc,
        claim_ids_tested=args.claim_ids,
        evidence_class=args.evidence_class,
        evidence_direction=args.evidence_direction,
        runner_name=args.runner_name,
        runner_version=args.runner_version,
    )
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
