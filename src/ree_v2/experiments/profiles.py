"""Profile definitions for REE-v2 qualification bootstrap."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ResourceEstimate:
    runtime_minutes: float
    memory_gb: float


@dataclass(frozen=True)
class ProfileCondition:
    name: str
    base_metrics: dict[str, float]
    evidence_direction: str
    include_uncertainty: bool
    include_action_token: bool
    resources: ResourceEstimate


@dataclass(frozen=True)
class FailureRule:
    signature_id: str
    metric_key: str
    op: str
    threshold: float


@dataclass(frozen=True)
class ClaimProfile:
    experiment_type: str
    claim_id: str
    evidence_class: str
    required_metric_keys: tuple[str, ...]
    default_seeds: tuple[int, ...]
    conditions: tuple[ProfileCondition, ...]
    failure_rules: tuple[FailureRule, ...]


PROFILE_CATALOG: dict[str, ClaimProfile] = {
    "trajectory_integrity": ClaimProfile(
        experiment_type="trajectory_integrity",
        claim_id="MECH-056",
        evidence_class="simulation",
        required_metric_keys=(
            "ledger_edit_detected_count",
            "explanation_policy_divergence_rate",
            "domination_lock_in_events",
            "commitment_reversal_rate",
        ),
        default_seeds=(11, 29, 47),
        conditions=(
            ProfileCondition(
                name="trajectory_first_enabled",
                evidence_direction="supports",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=32.0, memory_gb=5.4),
                base_metrics={
                    "ledger_edit_detected_count": 0.0,
                    "explanation_policy_divergence_rate": 0.022,
                    "domination_lock_in_events": 0.0,
                    "commitment_reversal_rate": 0.041,
                    "latent_prediction_error_mean": 0.044,
                    "latent_prediction_error_p95": 0.082,
                    "latent_residual_coverage_rate": 0.983,
                    "precision_input_completeness_rate": 0.978,
                    "latent_uncertainty_calibration_error": 0.108,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="trajectory_first_ablated",
                evidence_direction="weakens",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=41.0, memory_gb=6.2),
                base_metrics={
                    "ledger_edit_detected_count": 2.0,
                    "explanation_policy_divergence_rate": 0.081,
                    "domination_lock_in_events": 1.0,
                    "commitment_reversal_rate": 0.123,
                    "latent_prediction_error_mean": 0.091,
                    "latent_prediction_error_p95": 0.154,
                    "latent_residual_coverage_rate": 0.901,
                    "precision_input_completeness_rate": 0.903,
                    "latent_uncertainty_calibration_error": 0.198,
                    "fatal_error_count": 0.0,
                },
            ),
        ),
        failure_rules=(
            FailureRule("ledger_editing", "ledger_edit_detected_count", ">", 0.0),
            FailureRule("explanation_policy_divergence", "explanation_policy_divergence_rate", ">", 0.05),
            FailureRule("domination_lock_in", "domination_lock_in_events", ">", 0.0),
        ),
    ),
    "jepa_anchor_ablation": ClaimProfile(
        experiment_type="jepa_anchor_ablation",
        claim_id="MECH-058",
        evidence_class="ablation",
        required_metric_keys=(
            "latent_prediction_error_mean",
            "latent_prediction_error_p95",
            "latent_rollout_consistency_rate",
            "e1_e2_timescale_separation_ratio",
            "representation_drift_rate",
        ),
        default_seeds=(11, 29, 47),
        conditions=(
            ProfileCondition(
                name="ema_anchor_on",
                evidence_direction="supports",
                include_uncertainty=False,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=68.0, memory_gb=8.7),
                base_metrics={
                    "latent_prediction_error_mean": 0.128,
                    "latent_prediction_error_p95": 0.206,
                    "latent_rollout_consistency_rate": 0.921,
                    "e1_e2_timescale_separation_ratio": 1.84,
                    "representation_drift_rate": 0.039,
                    "latent_residual_coverage_rate": 0.984,
                    "precision_input_completeness_rate": 0.978,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="ema_anchor_off",
                evidence_direction="weakens",
                include_uncertainty=False,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=64.0, memory_gb=8.2),
                base_metrics={
                    "latent_prediction_error_mean": 0.209,
                    "latent_prediction_error_p95": 0.341,
                    "latent_rollout_consistency_rate": 0.646,
                    "e1_e2_timescale_separation_ratio": 1.21,
                    "representation_drift_rate": 0.138,
                    "latent_residual_coverage_rate": 0.931,
                    "precision_input_completeness_rate": 0.938,
                    "fatal_error_count": 0.0,
                },
            ),
        ),
        failure_rules=(
            FailureRule("mech058:ema_drift_under_shift", "representation_drift_rate", ">", 0.12),
            FailureRule("mech058:latent_cluster_collapse", "latent_rollout_consistency_rate", "<", 0.70),
            FailureRule("mech058:anchor_separation_collapse", "e1_e2_timescale_separation_ratio", "<", 1.50),
        ),
    ),
    "jepa_uncertainty_channels": ClaimProfile(
        experiment_type="jepa_uncertainty_channels",
        claim_id="MECH-059",
        evidence_class="simulation",
        required_metric_keys=(
            "latent_prediction_error_mean",
            "latent_uncertainty_calibration_error",
            "precision_input_completeness_rate",
            "uncertainty_coverage_rate",
        ),
        default_seeds=(11, 29, 47),
        conditions=(
            ProfileCondition(
                name="deterministic_plus_dispersion",
                evidence_direction="mixed",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=74.0, memory_gb=9.1),
                base_metrics={
                    "latent_prediction_error_mean": 0.164,
                    "latent_prediction_error_p95": 0.242,
                    "latent_uncertainty_calibration_error": 0.156,
                    "precision_input_completeness_rate": 0.944,
                    "uncertainty_coverage_rate": 0.882,
                    "latent_residual_coverage_rate": 0.976,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="explicit_uncertainty_head",
                evidence_direction="supports",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=88.0, memory_gb=10.6),
                base_metrics={
                    "latent_prediction_error_mean": 0.147,
                    "latent_prediction_error_p95": 0.224,
                    "latent_uncertainty_calibration_error": 0.118,
                    "precision_input_completeness_rate": 0.971,
                    "uncertainty_coverage_rate": 0.927,
                    "latent_residual_coverage_rate": 0.982,
                    "fatal_error_count": 0.0,
                },
            ),
        ),
        failure_rules=(
            FailureRule("mech059:calibration_slope_break", "latent_uncertainty_calibration_error", ">", 0.20),
            FailureRule(
                "mech059:uncertainty_metric_gaming_detected",
                "precision_minus_coverage",
                ">",
                0.20,
            ),
            FailureRule("mech059:abstention_reliability_collapse", "uncertainty_coverage_rate", "<", 0.70),
        ),
    ),
    "commit_dual_error_channels": ClaimProfile(
        experiment_type="commit_dual_error_channels",
        claim_id="MECH-060",
        evidence_class="ablation",
        required_metric_keys=(
            "pre_commit_error_signal_to_noise",
            "post_commit_error_attribution_gain",
            "cross_channel_leakage_rate",
            "commitment_reversal_rate",
        ),
        default_seeds=(11, 29, 47),
        conditions=(
            ProfileCondition(
                name="single_error_stream",
                evidence_direction="weakens",
                include_uncertainty=False,
                include_action_token=True,
                resources=ResourceEstimate(runtime_minutes=56.0, memory_gb=7.9),
                base_metrics={
                    "pre_commit_error_signal_to_noise": 0.92,
                    "post_commit_error_attribution_gain": 0.191,
                    "cross_channel_leakage_rate": 0.248,
                    "commitment_reversal_rate": 0.121,
                    "latent_prediction_error_mean": 0.179,
                    "latent_prediction_error_p95": 0.262,
                    "latent_residual_coverage_rate": 0.951,
                    "precision_input_completeness_rate": 0.931,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="pre_post_split_streams",
                evidence_direction="supports",
                include_uncertainty=False,
                include_action_token=True,
                resources=ResourceEstimate(runtime_minutes=63.0, memory_gb=8.4),
                base_metrics={
                    "pre_commit_error_signal_to_noise": 2.18,
                    "post_commit_error_attribution_gain": 0.431,
                    "cross_channel_leakage_rate": 0.086,
                    "commitment_reversal_rate": 0.053,
                    "latent_prediction_error_mean": 0.142,
                    "latent_prediction_error_p95": 0.221,
                    "latent_residual_coverage_rate": 0.976,
                    "precision_input_completeness_rate": 0.966,
                    "fatal_error_count": 0.0,
                },
            ),
        ),
        failure_rules=(
            FailureRule("mech060:precommit_channel_contamination", "pre_commit_error_signal_to_noise", "<", 1.20),
            FailureRule("mech060:postcommit_channel_contamination", "cross_channel_leakage_rate", ">", 0.20),
            FailureRule("mech060:attribution_reliability_break", "post_commit_error_attribution_gain", "<", 0.25),
            FailureRule("mech060:commitment_reversal_spike", "commitment_reversal_rate", ">", 0.10),
        ),
    ),
}


def get_profiles(profile: str = "all") -> list[ClaimProfile]:
    if profile == "all":
        return [PROFILE_CATALOG[name] for name in sorted(PROFILE_CATALOG)]
    return [get_profile(profile)]


def get_profile(experiment_type: str) -> ClaimProfile:
    if experiment_type not in PROFILE_CATALOG:
        known = ", ".join(sorted(PROFILE_CATALOG))
        raise KeyError(f"Unknown profile '{experiment_type}'. Known profiles: {known}")
    return PROFILE_CATALOG[experiment_type]


def _seeded_jitter(
    experiment_type: str,
    condition_name: str,
    seed: int,
    metric_key: str,
    scale: float,
) -> float:
    material = f"{experiment_type}|{condition_name}|{seed}|{metric_key}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    centered = (raw * 2.0) - 1.0
    return centered * scale


def _apply_operation(lhs: float, op: str, rhs: float) -> bool:
    operations: dict[str, Callable[[float, float], bool]] = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
    }
    return operations[op](lhs, rhs)


def _resolve_metric_key(metrics: dict[str, float], metric_key: str) -> float:
    if metric_key == "precision_minus_coverage":
        return metrics.get("precision_input_completeness_rate", 0.0) - metrics.get("uncertainty_coverage_rate", 0.0)
    return metrics.get(metric_key, 0.0)


def simulate_metrics(experiment_type: str, condition_name: str, seed: int) -> dict[str, float]:
    # Local import avoids circular dependency with the runner module importing profile metadata.
    from ree_v2.experiments.runner import execute_profile_condition

    result = execute_profile_condition(
        experiment_type=experiment_type,
        condition_name=condition_name,
        seed=seed,
        write=False,
    )
    return dict(result.metrics_values)


def evaluate_failure_signatures(experiment_type: str, metrics: dict[str, float]) -> list[str]:
    profile = get_profile(experiment_type)
    signatures: list[str] = []
    for rule in profile.failure_rules:
        lhs = _resolve_metric_key(metrics, rule.metric_key)
        if _apply_operation(lhs, rule.op, rule.threshold):
            signatures.append(rule.signature_id)
    return signatures
